"""Hybrid retrieval engine for the Anchorium RAG Pipeline.

Combines BM25 sparse search with Pinecone dense vector search via
alpha-blended score fusion, applies temporal recency boosts, and
optionally reranks results using Cohere's rerank-v3.5 model.

Typical usage::

    result = retrieve(
        query="What are the RBI guidelines for OBU SBLC-backed lending?",
        settings=settings,
        embedding_service=embedding_svc,
        pinecone_indexer=pinecone_idx,
        bm25_manager=bm25_mgr,
        filters=MetadataFilter(issuing_body=IssuingBody.RBI, active_only=True),
    )
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import cohere
import structlog
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_pipeline.config import AppSettings, RetrievalConfig
from rag_pipeline.indexing import BM25IndexManager, EmbeddingService, PineconeIndexer
from rag_pipeline.ingestion import DocumentChunk, DocumentMetadata, IssuingBody

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScoredChunk(BaseModel):
    """A document chunk augmented with retrieval and reranking scores.

    Attributes:
        chunk: The underlying document chunk retrieved from the index.
        sparse_score: Raw BM25 score (0.0 if not found in sparse results).
        dense_score: Raw Pinecone cosine-similarity score (0.0 if absent).
        blended_score: Alpha-weighted fusion of normalised sparse/dense scores.
        rerank_score: Cohere rerank relevance score, or *None* if reranking
            was not applied.
        final_score: The definitive score used for result ordering — equals
            *rerank_score* when reranking succeeds, otherwise *blended_score*.
    """

    chunk: DocumentChunk
    sparse_score: float = 0.0
    dense_score: float = 0.0
    blended_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0


class MetadataFilter(BaseModel):
    """Typed filter specification for constraining retrieval results.

    Attributes:
        issuing_body: Restrict to a single issuing authority (e.g. RBI, SEBI).
        min_effective_date: Inclusive lower bound on the circular effective date.
        max_effective_date: Inclusive upper bound on the circular effective date.
        circular_number: Exact-match filter on the circular/notification number.
        active_only: When *True* (default), only retrieve currently-active
            circulars whose ``is_active`` flag is set.
    """

    issuing_body: IssuingBody | None = None
    min_effective_date: date | None = None
    max_effective_date: date | None = None
    circular_number: str | None = None
    active_only: bool = True


class RetrievalResult(BaseModel):
    """Complete retrieval result with scoring diagnostics.

    Attributes:
        query: The original natural-language query string.
        scored_chunks: Ordered list of chunks with all score components.
        total_sparse_hits: Number of BM25 matches before blending.
        total_dense_hits: Number of Pinecone matches before blending.
        reranking_applied: Whether Cohere reranking completed successfully.
        filters_applied: Dictionary snapshot of the metadata filters used.
        retrieval_duration_seconds: Wall-clock time for the full pipeline.
    """

    query: str
    scored_chunks: list[ScoredChunk]
    total_sparse_hits: int
    total_dense_hits: int
    reranking_applied: bool
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    retrieval_duration_seconds: float


# ---------------------------------------------------------------------------
# Hybrid search engine
# ---------------------------------------------------------------------------


class HybridSearchEngine:
    """Alpha-blended hybrid search combining BM25 sparse and Pinecone dense retrieval.

    The engine executes both retrieval modalities in sequence, normalises scores
    to a common [0, 1] range via min-max scaling, then fuses them with a
    configurable alpha weight::

        blended = α · sparse_norm + (1 − α) · dense_norm

    A temporal recency boost is applied after blending so that more recent
    circulars receive a mild advantage.

    Args:
        embedding_service: Service for encoding queries into dense vectors.
        pinecone_indexer: Pinecone vector-store client with a ``search`` method.
        bm25_manager: BM25 sparse index manager with a ``search`` method.
        config: Retrieval hyper-parameters (alpha, top_k, boost factor, etc.).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        pinecone_indexer: PineconeIndexer,
        bm25_manager: BM25IndexManager,
        config: RetrievalConfig,
    ) -> None:
        self._embedding_service = embedding_service
        self._pinecone_indexer = pinecone_indexer
        self._bm25_manager = bm25_manager
        self._config = config
        self._log = logger.bind(component="HybridSearchEngine")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: MetadataFilter | None = None,
    ) -> RetrievalResult:
        """Run full hybrid search pipeline and return scored chunks.

        Steps:
            1. BM25 sparse search.
            2. Pinecone dense search with optional metadata filtering.
            3. Min-max normalisation of both score distributions.
            4. Alpha-blended fusion over the union of result sets.
            5. Temporal recency boost.
            6. Sort descending by blended score and truncate to ``top_k_raw``.

        Args:
            query: Natural-language user query.
            filters: Optional metadata constraints.

        Returns:
            A ``RetrievalResult`` containing scored chunks and diagnostics.
        """
        t_start = time.perf_counter()
        active_filters = filters or MetadataFilter()

        self._log.info(
            "hybrid_search_started",
            query_preview=query[:120],
            alpha=self._config.alpha,
            top_k_raw=self._config.top_k_raw,
        )

        # --- Step 1: BM25 sparse search ---
        sparse_results: list[tuple[DocumentChunk, float]] = self._bm25_manager.search(
            query, top_k=self._config.top_k_raw
        )
        total_sparse_hits = len(sparse_results)
        self._log.debug("sparse_search_complete", hits=total_sparse_hits)

        # --- Step 2: Dense (Pinecone) search ---
        query_embedding: list[float] = self._embedding_service.embed_query(query)
        pinecone_filter = self._build_pinecone_filter(active_filters)

        dense_raw_results: list[dict[str, Any]] = self._pinecone_indexer.search(
            query_embedding,
            top_k=self._config.top_k_raw,
            metadata_filter=pinecone_filter,
        )
        total_dense_hits = len(dense_raw_results)
        self._log.debug("dense_search_complete", hits=total_dense_hits)

        # --- Step 3: Build lookup maps keyed by chunk_id ---
        sparse_map: dict[str, tuple[DocumentChunk, float]] = {
            chunk.chunk_id: (chunk, score) for chunk, score in sparse_results
        }

        dense_map: dict[str, tuple[DocumentChunk, float]] = {}
        for match in dense_raw_results:
            reconstructed = self._reconstruct_chunk_from_pinecone(match)
            score = float(match.get("score", 0.0))
            dense_map[reconstructed.chunk_id] = (reconstructed, score)

        # --- Step 4: Min-max normalise ---
        sparse_scores_raw = [s for _, s in sparse_map.values()]
        dense_scores_raw = [s for _, s in dense_map.values()]

        sparse_norm_list = self._normalize_scores(sparse_scores_raw)
        dense_norm_list = self._normalize_scores(dense_scores_raw)

        sparse_norm_map: dict[str, float] = dict(zip(sparse_map.keys(), sparse_norm_list, strict=True))
        dense_norm_map: dict[str, float] = dict(zip(dense_map.keys(), dense_norm_list, strict=True))

        # --- Step 5: Alpha-blend over the union ---
        all_chunk_ids = set(sparse_map.keys()) | set(dense_map.keys())
        alpha = self._config.alpha

        scored_chunks: list[ScoredChunk] = []
        for cid in all_chunk_ids:
            s_norm = sparse_norm_map.get(cid, 0.0)
            d_norm = dense_norm_map.get(cid, 0.0)
            blended = alpha * s_norm + (1.0 - alpha) * d_norm

            # Prefer the chunk object from whichever source has it
            chunk_obj: DocumentChunk
            raw_sparse = 0.0
            raw_dense = 0.0

            if cid in sparse_map:
                chunk_obj, raw_sparse = sparse_map[cid]
            else:
                chunk_obj, _ = dense_map[cid]

            if cid in dense_map:
                _, raw_dense = dense_map[cid]

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk_obj,
                    sparse_score=raw_sparse,
                    dense_score=raw_dense,
                    blended_score=blended,
                    final_score=blended,
                )
            )

        # --- Step 6: Temporal boost ---
        scored_chunks = self._apply_temporal_boost(scored_chunks)

        # --- Step 7: Sort & truncate ---
        scored_chunks.sort(key=lambda sc: sc.blended_score, reverse=True)
        scored_chunks = scored_chunks[: self._config.top_k_raw]

        # Update final_score to match blended after boost (pre-rerank)
        for sc in scored_chunks:
            sc.final_score = sc.blended_score

        duration = time.perf_counter() - t_start
        self._log.info(
            "hybrid_search_complete",
            result_count=len(scored_chunks),
            duration_s=round(duration, 4),
        )

        filters_snapshot: dict[str, Any] = active_filters.model_dump(exclude_none=True)

        return RetrievalResult(
            query=query,
            scored_chunks=scored_chunks,
            total_sparse_hits=total_sparse_hits,
            total_dense_hits=total_dense_hits,
            reranking_applied=False,
            filters_applied=filters_snapshot,
            retrieval_duration_seconds=round(duration, 6),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        """Min-max normalise a list of scores to the [0, 1] interval.

        When all scores are identical (including the degenerate single-element
        case) every entry is mapped to 1.0 so that no information is lost
        during the subsequent blending step.

        Args:
            scores: Raw score values from a single retrieval modality.

        Returns:
            A list of normalised floats in the same order as *scores*.
        """
        if not scores:
            return []

        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s

        if span == 0.0:
            return [1.0] * len(scores)

        return [(s - min_s) / span for s in scores]

    def _apply_temporal_boost(
        self,
        chunks: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Boost blended scores proportionally to document recency.

        The oldest effective date in the candidate set serves as the baseline.
        Each chunk receives a multiplicative boost of::

            blended_score *= 1 + temporal_boost_factor × years_newer

        where ``years_newer = (effective_date − oldest_date).days / 365.25``.

        Chunks whose metadata lacks an ``effective_date`` are left untouched.

        Args:
            chunks: Scored chunks after alpha-blending.

        Returns:
            The same list with ``blended_score`` values adjusted in-place.
        """
        dated_chunks = [sc for sc in chunks if sc.chunk.metadata.effective_date is not None]

        if not dated_chunks:
            self._log.debug("temporal_boost_skipped", reason="no_effective_dates")
            return chunks

        oldest_date = min(
            sc.chunk.metadata.effective_date  # type: ignore[type-var]
            for sc in dated_chunks
        )

        factor = self._config.temporal_boost_factor

        for sc in chunks:
            eff = sc.chunk.metadata.effective_date
            if eff is None:
                continue
            days_newer = (eff - oldest_date).days
            years_newer = days_newer / 365.25
            multiplier = 1.0 + factor * years_newer
            sc.blended_score *= multiplier

        self._log.debug(
            "temporal_boost_applied",
            oldest_date=oldest_date.isoformat(),
            boosted_count=len(dated_chunks),
            factor=factor,
        )

        return chunks

    @staticmethod
    def _build_pinecone_filter(filters: MetadataFilter) -> dict[str, Any]:
        """Convert a ``MetadataFilter`` into Pinecone's native filter dict.

        Pinecone expects filters in the form::

            {"$and": [{"field": {"$eq": value}}, ...]}

        Date-range filtering is expressed as ``$gte`` / ``$lte`` comparisons
        on the ISO-8601 string representation of the ``effective_date`` field.

        Args:
            filters: Typed metadata filter constraints.

        Returns:
            A Pinecone-compatible filter dictionary.  Returns an empty dict
            when no constraints are specified and ``active_only`` is *False*.
        """
        conditions: list[dict[str, Any]] = []

        if filters.active_only:
            conditions.append({"is_active": {"$eq": True}})

        if filters.issuing_body is not None:
            conditions.append({"issuing_body": {"$eq": filters.issuing_body.value}})

        if filters.min_effective_date is not None:
            conditions.append(
                {
                    "effective_date": {
                        "$gte": filters.min_effective_date.isoformat(),
                    }
                }
            )

        if filters.max_effective_date is not None:
            conditions.append(
                {
                    "effective_date": {
                        "$lte": filters.max_effective_date.isoformat(),
                    }
                }
            )

        if filters.circular_number is not None:
            conditions.append({"circular_number": {"$eq": filters.circular_number}})

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    @staticmethod
    def _reconstruct_chunk_from_pinecone(match: dict[str, Any]) -> DocumentChunk:
        """Reconstruct a ``DocumentChunk`` from flattened Pinecone match metadata.

        Pinecone stores metadata as a flat key-value map alongside each vector.
        This method re-hydrates the nested ``DocumentMetadata`` and wraps it
        in a ``DocumentChunk``.

        Args:
            match: A single match dict returned by Pinecone, expected to contain
                an ``id`` key, a ``metadata`` sub-dict, and a ``score`` key.

        Returns:
            A fully-populated ``DocumentChunk`` instance.

        Raises:
            KeyError: If essential metadata keys are missing from the match.
            ValueError: If metadata values cannot be coerced to expected types.
        """
        meta: dict[str, Any] = match.get("metadata", {})

        effective_date_raw = meta.get("effective_date")
        effective_date: date | None = None
        if effective_date_raw is not None:
            if isinstance(effective_date_raw, str):
                effective_date = date.fromisoformat(effective_date_raw)
            elif isinstance(effective_date_raw, date):
                effective_date = effective_date_raw

        issuing_body_raw = meta.get("issuing_body", "OTHER")
        try:
            issuing_body = IssuingBody(issuing_body_raw)
        except ValueError:
            issuing_body = IssuingBody.OTHER

        page_numbers_raw = meta.get("page_numbers", [])
        if isinstance(page_numbers_raw, str):
            page_numbers = [int(p.strip()) for p in page_numbers_raw.split(",") if p.strip()]
        elif isinstance(page_numbers_raw, list):
            page_numbers = [int(p) for p in page_numbers_raw]
        else:
            page_numbers = []

        doc_metadata = DocumentMetadata(
            document_title=meta.get("document_title", ""),
            issuing_body=issuing_body,
            circular_number=meta.get("circular_number"),
            effective_date=effective_date,
            is_active=bool(meta.get("is_active", True)),
            version=meta.get("version"),
            source_file=meta.get("source_file", ""),
            total_pages=meta.get("total_pages"),
        )

        return DocumentChunk(
            chunk_id=str(match.get("id", meta.get("chunk_id", ""))),
            text=meta.get("text", ""),
            token_count=int(meta.get("token_count", 0)),
            metadata=doc_metadata,
            hierarchical_path=meta.get("hierarchical_path", ""),
            page_numbers=page_numbers,
            contains_table=bool(meta.get("contains_table", False)),
            chunk_index=int(meta.get("chunk_index", 0)),
        )


# ---------------------------------------------------------------------------
# Cohere reranker
# ---------------------------------------------------------------------------


class CohereReranker:
    """Reranks hybrid search results using Cohere's neural reranking API.

    On transient failures the reranker degrades gracefully by falling back
    to the pre-existing blended scores rather than propagating exceptions.

    Args:
        api_key: Cohere API key wrapped in ``SecretStr`` for safe handling.
        config: Retrieval configuration holding model name and top-k settings.
    """

    def __init__(self, api_key: SecretStr, config: RetrievalConfig) -> None:
        self._client: cohere.ClientV2 = cohere.ClientV2(
            api_key=api_key.get_secret_value(),
        )
        self._config = config
        self._log = logger.bind(component="CohereReranker")

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Rerank scored chunks via Cohere and return the top-k results.

        On success, each returned ``ScoredChunk`` has its ``rerank_score``
        and ``final_score`` set to the Cohere relevance score.  On failure,
        the original chunks are returned sorted by ``blended_score`` so the
        caller always receives a usable result set.

        Args:
            query: The user's natural-language query.
            chunks: Candidate chunks from the hybrid search engine.

        Returns:
            A list of at most ``top_k_reranked`` chunks ordered by relevance.
        """
        if not chunks:
            self._log.debug("rerank_skipped", reason="empty_input")
            return []

        try:
            return self._call_cohere(query, chunks)
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            self._log.warning(
                "rerank_fallback_to_blended",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            fallback = sorted(chunks, key=lambda sc: sc.blended_score, reverse=True)
            for sc in fallback:
                sc.final_score = sc.blended_score
            return fallback[: self._config.top_k_reranked]

    @retry(
        retry=retry_if_exception_type((cohere.TooManyRequestsError, ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_cohere(
        self,
        query: str,
        chunks: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Execute the Cohere rerank API call with tenacity retry.

        Args:
            query: The user query string.
            chunks: Candidate scored chunks.

        Returns:
            Reranked and truncated list of ``ScoredChunk`` instances.

        Raises:
            cohere.TooManyRequestsError: Propagated after exhausting retries
                so the outer ``rerank`` method can catch it.
            ConnectionError: On network-level failures.
            TimeoutError: When the API call exceeds its deadline.
        """
        documents = [sc.chunk.text for sc in chunks]

        self._log.info(
            "cohere_rerank_started",
            model=self._config.cohere_rerank_model,
            doc_count=len(documents),
            top_n=self._config.top_k_reranked,
        )

        response = self._client.rerank(
            model=self._config.cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=self._config.top_k_reranked,
        )

        reranked: list[ScoredChunk] = []
        for result in response.results:
            idx = result.index
            relevance = float(result.relevance_score)

            sc = chunks[idx].model_copy(deep=True)
            sc.rerank_score = relevance
            sc.final_score = relevance
            reranked.append(sc)

        reranked.sort(key=lambda sc: sc.final_score, reverse=True)

        self._log.info(
            "cohere_rerank_complete",
            reranked_count=len(reranked),
            top_score=round(reranked[0].final_score, 4) if reranked else 0.0,
        )

        return reranked


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    settings: AppSettings,
    embedding_service: EmbeddingService,
    pinecone_indexer: PineconeIndexer,
    bm25_manager: BM25IndexManager,
    filters: MetadataFilter | None = None,
) -> RetrievalResult:
    """Execute the full hybrid-search-plus-reranking retrieval pipeline.

    This is the primary entry point for downstream consumers (e.g. the
    generation module).  It:

    1. Runs ``HybridSearchEngine.search`` to obtain alpha-blended,
       temporally-boosted candidate chunks.
    2. Applies ``CohereReranker.rerank`` for neural relevance re-scoring.
    3. Returns a ``RetrievalResult`` with timing diagnostics.

    Args:
        query: Natural-language question or search string.
        settings: Fully-loaded application settings (provides API keys,
            retrieval config, etc.).
        embedding_service: Initialised embedding service instance.
        pinecone_indexer: Initialised Pinecone indexer instance.
        bm25_manager: Initialised BM25 index manager instance.
        filters: Optional metadata constraints for narrowing results.

    Returns:
        A ``RetrievalResult`` whose ``scored_chunks`` are ordered by
        final relevance score (reranked when possible, blended otherwise).
    """
    log = logger.bind(function="retrieve")
    t_start = time.perf_counter()

    log.info("retrieval_pipeline_started", query_preview=query[:120])

    # --- Hybrid search ---
    engine = HybridSearchEngine(
        embedding_service=embedding_service,
        pinecone_indexer=pinecone_indexer,
        bm25_manager=bm25_manager,
        config=settings.retrieval,
    )

    result = engine.search(query=query, filters=filters)

    # --- Cohere reranking ---
    reranker = CohereReranker(
        api_key=settings.api_keys.cohere_api_key,
        config=settings.retrieval,
    )

    reranked_chunks = reranker.rerank(query=query, chunks=result.scored_chunks)
    reranking_applied = any(sc.rerank_score is not None for sc in reranked_chunks)

    total_duration = time.perf_counter() - t_start

    final_result = RetrievalResult(
        query=result.query,
        scored_chunks=reranked_chunks,
        total_sparse_hits=result.total_sparse_hits,
        total_dense_hits=result.total_dense_hits,
        reranking_applied=reranking_applied,
        filters_applied=result.filters_applied,
        retrieval_duration_seconds=round(total_duration, 6),
    )

    log.info(
        "retrieval_pipeline_complete",
        final_count=len(final_result.scored_chunks),
        reranking_applied=reranking_applied,
        duration_s=round(total_duration, 4),
    )

    return final_result
