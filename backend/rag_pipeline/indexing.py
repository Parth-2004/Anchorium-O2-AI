"""Vector database operations, BM25 sparse index management, and embedding generation.

Provides:
- ``EmbeddingService`` — batch / single-query embedding via OpenAI or VoyageAI.
- ``PineconeIndexer`` — serverless Pinecone index lifecycle, upsert, query, and deletion.
- ``BM25IndexManager`` — rank-bm25 sparse index with persistence (pickle).
- ``index_document`` — top-level convenience function that runs the full indexing pipeline.
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any

import openai
import structlog
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field, SecretStr
from rank_bm25 import BM25Okapi
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_pipeline.config import AppSettings, EmbeddingConfig, PineconeConfig
from rag_pipeline.ingestion import DocumentChunk

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IndexedChunkRecord(BaseModel):
    """A single chunk ready for vector database insertion.

    Attributes:
        chunk_id: Globally unique identifier for the chunk.
        text: Raw text content of the chunk (stored as Pinecone metadata).
        embedding: Dense embedding vector produced by the configured provider.
        pinecone_metadata: Flat dict of metadata fields stored alongside the
            vector in Pinecone (must contain only str / int / float / bool /
            list[str] values).
    """

    chunk_id: str
    text: str
    embedding: list[float]
    pinecone_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexingResult(BaseModel):
    """Summary statistics returned after an indexing operation.

    Attributes:
        total_chunks: Number of chunks submitted for indexing.
        successfully_indexed: Number of chunks that were persisted.
        failed_chunks: Number of chunks that failed during upsert.
        index_name: Name of the target Pinecone index.
        duration_seconds: Wall-clock elapsed time in seconds.
    """

    total_chunks: int
    successfully_indexed: int
    failed_chunks: int
    index_name: str
    duration_seconds: float


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Generates dense embeddings via OpenAI or VoyageAI.

    The service automatically batches input texts according to
    ``EmbeddingConfig.batch_size`` and applies exponential-backoff retries
    on transient API failures (rate limits, timeouts).

    Args:
        config: Embedding configuration (provider, model names, batch size).
        openai_api_key: OpenAI API key (used when provider is ``"openai"``).
        voyage_api_key: VoyageAI API key (used when provider is ``"voyage"``).
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        openai_api_key: SecretStr,
        voyage_api_key: SecretStr | None = None,
    ) -> None:
        self._config = config
        self._log = logger.bind(component="EmbeddingService", provider=config.provider)

        # Always initialise the OpenAI client — it is the default provider.
        self._openai_client = openai.OpenAI(api_key=openai_api_key.get_secret_value())

        self._voyage_client: Any | None = None
        if config.provider == "voyage":
            if voyage_api_key is None:
                raise ValueError(
                    "voyage_api_key is required when embedding provider is 'voyage'. "
                    "Set the ANCHORIUM_VOYAGE_API_KEY environment variable or pass the key explicitly."
                )
            try:
                import voyageai  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "The 'voyageai' package is required for the Voyage embedding provider. "
                    "Install it with: pip install 'anchorium-rag-pipeline[voyage]'"
                ) from exc
            self._voyage_client = voyageai.Client(api_key=voyage_api_key.get_secret_value())

        self._log.info(
            "embedding_service_initialised",
            model=config.openai_model if config.provider == "openai" else config.voyage_model,
            batch_size=config.batch_size,
        )

    # -- public API ----------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of document texts.

        Args:
            texts: Texts to embed (each typically a document chunk).

        Returns:
            List of embedding vectors in the same order as *texts*.

        Raises:
            openai.APIError: On unrecoverable OpenAI errors.
            RuntimeError: On unrecoverable VoyageAI errors.
            tenacity.RetryError: When all retry attempts are exhausted.
        """
        all_embeddings: list[list[float]] = []
        batch_size = self._config.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size

        self._log.info("embed_texts_start", total_texts=len(texts), total_batches=total_batches)

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch = texts[start:end]

            batch_embeddings = self._embed_batch(batch, input_type="document")
            all_embeddings.extend(batch_embeddings)

            self._log.debug(
                "embed_batch_complete",
                batch=batch_idx + 1,
                total_batches=total_batches,
                batch_size=len(batch),
            )

        self._log.info("embed_texts_complete", total_embeddings=len(all_embeddings))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        For VoyageAI the ``input_type`` is set to ``"query"`` to leverage
        asymmetric embedding optimisations.

        Args:
            query: The user query text.

        Returns:
            Dense embedding vector for the query.
        """
        self._log.debug("embed_query_start", query_length=len(query))
        result = self._embed_batch([query], input_type="query")
        return result[0]

    # -- internal helpers ----------------------------------------------------

    def _embed_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        """Embed a single batch with retry logic.

        Args:
            batch: A list of texts no larger than ``config.batch_size``.
            input_type: ``"document"`` or ``"query"``.

        Returns:
            Embedding vectors for the batch.
        """
        if self._config.provider == "voyage":
            return self._embed_batch_voyage(batch, input_type)
        return self._embed_batch_openai(batch)

    def _embed_batch_openai(self, batch: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API with tenacity retry."""

        @retry(
            retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self._config.max_retries),
            reraise=True,
        )
        def _call() -> list[list[float]]:
            response = self._openai_client.embeddings.create(
                input=batch,
                model=self._config.openai_model,
            )
            # Sort by index to guarantee ordering matches input ordering.
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [item.embedding for item in sorted_data]

        try:
            return _call()
        except RetryError:
            self._log.error(
                "openai_embedding_retries_exhausted",
                model=self._config.openai_model,
                batch_size=len(batch),
            )
            raise
        except openai.AuthenticationError:
            self._log.error("openai_authentication_failed")
            raise
        except openai.BadRequestError as exc:
            self._log.error("openai_bad_request", detail=str(exc))
            raise

    def _embed_batch_voyage(self, batch: list[str], input_type: str) -> list[list[float]]:
        """Call VoyageAI embeddings API with tenacity retry."""
        if self._voyage_client is None:
            raise RuntimeError("VoyageAI client is not initialised.")

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self._config.max_retries),
            reraise=True,
        )
        def _call() -> list[list[float]]:
            result = self._voyage_client.embed(
                texts=batch,
                model=self._config.voyage_model,
                input_type=input_type,
            )
            return result.embeddings  # type: ignore[no-any-return]

        try:
            return _call()
        except RetryError:
            self._log.error(
                "voyage_embedding_retries_exhausted",
                model=self._config.voyage_model,
                batch_size=len(batch),
            )
            raise
        except Exception:
            self._log.exception("voyage_embedding_unhandled_error")
            raise


# ---------------------------------------------------------------------------
# PineconeIndexer
# ---------------------------------------------------------------------------


class PineconeIndexer:
    """Manages a Pinecone serverless vector index lifecycle and operations.

    Handles index creation, vector upsert / deletion, metadata updates,
    and similarity search.  All network calls are wrapped with tenacity
    exponential-backoff retries.

    Args:
        config: Pinecone-specific settings (index name, cloud, region, etc.).
        api_key: Pinecone API key.
    """

    def __init__(self, config: PineconeConfig, api_key: SecretStr) -> None:
        self._config = config
        self._log = logger.bind(component="PineconeIndexer", index=config.index_name)
        self._pc = Pinecone(api_key=api_key.get_secret_value())
        self._log.info("pinecone_client_initialised")

    # -- index lifecycle -----------------------------------------------------

    def ensure_index_exists(self) -> None:
        """Create the Pinecone index if it does not already exist.

        Uses serverless spec with the cloud / region from the config.  Blocks
        until the index reaches a ``ready`` state.

        Raises:
            pinecone.exceptions.PineconeApiException: On API-level failures.
        """
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        if self._config.index_name in existing_indexes:
            self._log.info("pinecone_index_already_exists")
            return

        self._log.info(
            "pinecone_creating_index",
            dimension=self._config.dimension,
            metric=self._config.metric,
            cloud=self._config.cloud,
            region=self._config.region,
        )

        self._pc.create_index(
            name=self._config.index_name,
            dimension=self._config.dimension,
            metric=self._config.metric,
            spec=ServerlessSpec(cloud=self._config.cloud, region=self._config.region),
        )

        # Poll until the index reports ready.
        while not self._pc.describe_index(self._config.index_name).status.get("ready", False):
            self._log.debug("pinecone_waiting_for_index_ready")
            time.sleep(2)

        self._log.info("pinecone_index_ready")

    # -- upsert --------------------------------------------------------------

    def upsert_chunks(self, records: list[IndexedChunkRecord]) -> IndexingResult:
        """Batch-upsert ``IndexedChunkRecord`` objects into Pinecone.

        Each record is formatted as ``(id, values, metadata)`` tuple
        required by the Pinecone gRPC/REST upsert API.

        Args:
            records: Chunk records with embeddings and metadata.

        Returns:
            An ``IndexingResult`` summarising the operation.
        """
        start = time.monotonic()
        index = self._pc.Index(self._config.index_name)
        batch_size = self._config.batch_size
        total_batches = (len(records) + batch_size - 1) // batch_size
        success_count = 0
        fail_count = 0

        self._log.info(
            "upsert_start",
            total_records=len(records),
            total_batches=total_batches,
            batch_size=batch_size,
        )

        for batch_idx in range(total_batches):
            start_pos = batch_idx * batch_size
            end_pos = start_pos + batch_size
            batch = records[start_pos:end_pos]
            vectors = [(rec.chunk_id, rec.embedding, rec.pinecone_metadata) for rec in batch]
            try:
                self._upsert_batch_with_retry(index, vectors)
                success_count += len(batch)
                self._log.debug(
                    "upsert_batch_complete",
                    batch=batch_idx + 1,
                    total_batches=total_batches,
                )
            except (RetryError, Exception) as exc:
                fail_count += len(batch)
                self._log.error(
                    "upsert_batch_failed",
                    batch=batch_idx + 1,
                    error=str(exc),
                )

        elapsed = time.monotonic() - start
        result = IndexingResult(
            total_chunks=len(records),
            successfully_indexed=success_count,
            failed_chunks=fail_count,
            index_name=self._config.index_name,
            duration_seconds=round(elapsed, 3),
        )
        self._log.info(
            "upsert_complete",
            successfully_indexed=success_count,
            failed_chunks=fail_count,
            duration_seconds=result.duration_seconds,
        )
        return result

    @staticmethod
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _upsert_batch_with_retry(
        index: Any,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Upsert a single batch of vectors with exponential-backoff retry."""
        index.upsert(vectors=vectors)

    # -- supersede -----------------------------------------------------------

    def supersede_document(self, circular_number: str) -> int:
        """Mark all vectors for *circular_number* as inactive (``is_active=False``).

        Pinecone does not support metadata-only updates via filter.  The
        procedure is:

        1. Query to find all matching vector IDs.
        2. Fetch those vectors (with their existing embeddings).
        3. Re-upsert with updated ``is_active=False`` metadata.

        Args:
            circular_number: The regulatory circular number to supersede.

        Returns:
            Count of vectors that were updated.
        """
        self._log.info("supersede_document_start", circular_number=circular_number)
        index = self._pc.Index(self._config.index_name)

        # Step 1: Discover matching IDs via a high-top_k query with filter.
        # We use a zero-vector query to avoid relevance bias; only the filter matters.
        all_matching_ids: list[str] = []
        dummy_vector = [0.0] * self._config.dimension

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _query_ids() -> list[str]:
            resp = index.query(
                vector=dummy_vector,
                top_k=10_000,
                filter={"circular_number": {"$eq": circular_number}},
                include_metadata=False,
                include_values=False,
            )
            return [match.id for match in resp.matches]

        try:
            all_matching_ids = _query_ids()
        except (RetryError, Exception):
            self._log.exception("supersede_query_failed", circular_number=circular_number)
            return 0

        if not all_matching_ids:
            self._log.info("supersede_no_matches", circular_number=circular_number)
            return 0

        # Step 2: Fetch vectors in batches (Pinecone fetch limit is 1000 IDs).
        fetch_batch_size = 1000
        updated_count = 0

        for i in range(0, len(all_matching_ids), fetch_batch_size):
            id_batch = all_matching_ids[i : i + fetch_batch_size]
            try:
                fetched = self._fetch_with_retry(index, id_batch)
            except (RetryError, Exception):
                self._log.exception(
                    "supersede_fetch_failed",
                    batch_start=i,
                    circular_number=circular_number,
                )
                continue

            # Step 3: Re-upsert with updated metadata.
            upsert_vectors: list[tuple[str, list[float], dict[str, Any]]] = []
            for vec_id, vec_data in fetched.vectors.items():
                metadata = dict(vec_data.metadata) if vec_data.metadata else {}
                metadata["is_active"] = False
                upsert_vectors.append((vec_id, vec_data.values, metadata))

            if upsert_vectors:
                try:
                    self._upsert_batch_with_retry(index, upsert_vectors)
                    updated_count += len(upsert_vectors)
                except (RetryError, Exception):
                    self._log.exception(
                        "supersede_upsert_failed",
                        batch_start=i,
                        circular_number=circular_number,
                    )

        self._log.info(
            "supersede_document_complete",
            circular_number=circular_number,
            updated_count=updated_count,
        )
        return updated_count

    @staticmethod
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _fetch_with_retry(index: Any, ids: list[str]) -> Any:
        """Fetch vectors by ID with exponential-backoff retry."""
        return index.fetch(ids=ids)

    # -- delete --------------------------------------------------------------

    def delete_document(self, circular_number: str) -> int:
        """Delete all vectors matching *circular_number* from the index.

        Uses Pinecone's filter-based delete.

        Args:
            circular_number: The regulatory circular number to purge.

        Returns:
            Estimated count of vectors deleted (based on a pre-delete query).
        """
        self._log.info("delete_document_start", circular_number=circular_number)
        index = self._pc.Index(self._config.index_name)

        # First count matching vectors so we can report an accurate number.
        dummy_vector = [0.0] * self._config.dimension

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _count_matches() -> int:
            resp = index.query(
                vector=dummy_vector,
                top_k=10_000,
                filter={"circular_number": {"$eq": circular_number}},
                include_metadata=False,
                include_values=False,
            )
            return len(resp.matches)

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _delete() -> None:
            index.delete(filter={"circular_number": {"$eq": circular_number}})

        try:
            count = _count_matches()
        except (RetryError, Exception):
            self._log.exception("delete_count_failed", circular_number=circular_number)
            count = 0

        try:
            _delete()
        except (RetryError, Exception):
            self._log.exception("delete_failed", circular_number=circular_number)
            return 0

        self._log.info(
            "delete_document_complete",
            circular_number=circular_number,
            deleted_count=count,
        )
        return count

    # -- search --------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a similarity search against the Pinecone index.

        Args:
            query_embedding: Dense query vector.
            top_k: Number of nearest neighbours to return.
            metadata_filter: Optional Pinecone metadata filter dict.

        Returns:
            List of match dicts, each containing ``id``, ``score``, and
            ``metadata`` keys.
        """
        self._log.debug("pinecone_search_start", top_k=top_k, has_filter=metadata_filter is not None)
        index = self._pc.Index(self._config.index_name)

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _query() -> list[dict[str, Any]]:
            kwargs: dict[str, Any] = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
            }
            if metadata_filter is not None:
                kwargs["filter"] = metadata_filter

            resp = index.query(**kwargs)
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": dict(match.metadata) if match.metadata else {},
                }
                for match in resp.matches
            ]

        try:
            results = _query()
        except (RetryError, Exception):
            self._log.exception("pinecone_search_failed")
            return []

        self._log.debug("pinecone_search_complete", result_count=len(results))
        return results


# ---------------------------------------------------------------------------
# BM25IndexManager
# ---------------------------------------------------------------------------

# Pre-compiled regex for tokenisation.
_TOKENIZE_SPLIT_RE = re.compile(r"[^a-z0-9§/]+")


class BM25IndexManager:
    """Manages a BM25Okapi sparse index over ``DocumentChunk`` objects.

    The index is held in-memory and can be persisted to / loaded from disk
    via pickle serialisation.  Because ``rank_bm25.BM25Okapi`` does not
    support incremental updates, ``add_chunks`` rebuilds the full index.

    Typical workflow::

        mgr = BM25IndexManager()
        mgr.build_index(chunks)
        mgr.save_index(Path("bm25.pkl"), Path("chunks.pkl"))
        # later …
        mgr.load_index(Path("bm25.pkl"), Path("chunks.pkl"))
        results = mgr.search("interest rate")
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._log = logger.bind(component="BM25IndexManager")

    # -- build ---------------------------------------------------------------

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build a BM25Okapi index from a list of document chunks.

        Args:
            chunks: Document chunks whose ``text`` fields will be tokenised.
        """
        self._log.info("bm25_build_start", chunk_count=len(chunks))
        self._chunks = list(chunks)
        self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._log.info("bm25_build_complete", corpus_size=len(self._chunks))

    # -- persistence ---------------------------------------------------------

    def save_index(self, index_path: Path, chunks_path: Path) -> None:
        """Persist the BM25 index and chunk list to disk.

        Parent directories are created automatically if they do not exist.

        Args:
            index_path: Destination for the serialised ``BM25Okapi`` object.
            chunks_path: Destination for the serialised chunk list.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError("Cannot save: BM25 index has not been built. Call build_index() first.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as fh:
            pickle.dump(self._bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)

        with open(chunks_path, "wb") as fh:
            pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)

        self._log.info(
            "bm25_index_saved",
            index_path=str(index_path),
            chunks_path=str(chunks_path),
        )

    def load_index(self, index_path: Path, chunks_path: Path) -> None:
        """Load a previously-persisted BM25 index and chunk list.

        Args:
            index_path: Path to the serialised ``BM25Okapi`` object.
            chunks_path: Path to the serialised chunk list.

        Raises:
            FileNotFoundError: If either file is missing from disk.
        """
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index file not found at '{index_path}'. "
                "Run the indexing pipeline first to build and save the index."
            )
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"BM25 chunks file not found at '{chunks_path}'. "
                "Run the indexing pipeline first to build and save the chunk metadata."
            )

        with open(index_path, "rb") as fh:
            self._bm25 = pickle.load(fh)  # noqa: S301 — trusted internal data

        with open(chunks_path, "rb") as fh:
            self._chunks = pickle.load(fh)  # noqa: S301 — trusted internal data

        # Rebuild tokenized corpus so add_chunks() works correctly later.
        self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]

        self._log.info(
            "bm25_index_loaded",
            index_path=str(index_path),
            chunks_path=str(chunks_path),
            corpus_size=len(self._chunks),
        )

    # -- search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[tuple[DocumentChunk, float]]:
        """Run a BM25 keyword search.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(DocumentChunk, score)`` tuples sorted by descending
            BM25 score.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self._bm25 is None or not self._chunks:
            raise RuntimeError("BM25 index is empty. Call build_index() or load_index() first.")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair each chunk with its score, sort descending, return top_k.
        scored_chunks: list[tuple[DocumentChunk, float]] = [
            (chunk, float(score)) for chunk, score in zip(self._chunks, scores, strict=False)
        ]
        scored_chunks.sort(key=lambda pair: pair[1], reverse=True)
        return scored_chunks[:top_k]

    # -- incremental update --------------------------------------------------

    def add_chunks(self, new_chunks: list[DocumentChunk]) -> None:
        """Add new chunks and rebuild the BM25 index.

        Because ``BM25Okapi`` does not support incremental updates the entire
        index is rebuilt from scratch using the merged corpus.

        Args:
            new_chunks: Additional document chunks to index.
        """
        self._log.info(
            "bm25_add_chunks_start",
            existing=len(self._chunks),
            new=len(new_chunks),
        )
        merged = list(self._chunks) + list(new_chunks)
        self.build_index(merged)

    # -- tokenisation --------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise text for BM25 scoring.

        Lowercases the text, preserves legal citation markers (``§``, ``/``),
        strips all other punctuation, and splits on whitespace.

        Args:
            text: Raw text to tokenise.

        Returns:
            List of lowercase tokens.
        """
        lowered = text.lower()
        tokens = _TOKENIZE_SPLIT_RE.split(lowered)
        return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Helper — flatten DocumentChunk metadata for Pinecone
# ---------------------------------------------------------------------------


def _flatten_chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """Convert a ``DocumentChunk`` into a flat metadata dict for Pinecone.

    Pinecone metadata values must be one of str, int, float, bool, or
    list[str].  This function extracts relevant fields from both the chunk
    and its nested ``DocumentMetadata``.

    Args:
        chunk: The document chunk to flatten.

    Returns:
        Flat metadata dictionary suitable for Pinecone upsert.
    """
    meta: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "chunk_index": chunk.chunk_index,
        "contains_table": chunk.contains_table,
        "hierarchical_path": chunk.hierarchical_path,
        "page_numbers": chunk.page_numbers,
    }

    if chunk.metadata is not None:
        doc_meta = chunk.metadata
        meta["document_title"] = doc_meta.document_title
        issuing_body = doc_meta.issuing_body
        meta["issuing_body"] = issuing_body.value if hasattr(issuing_body, "value") else str(issuing_body)
        meta["circular_number"] = doc_meta.circular_number
        meta["is_active"] = doc_meta.is_active
        meta["version"] = doc_meta.version
        meta["source_file"] = doc_meta.source_file

        if doc_meta.effective_date is not None:
            meta["effective_date"] = doc_meta.effective_date.isoformat()
        if doc_meta.total_pages is not None:
            meta["total_pages"] = doc_meta.total_pages

    return meta


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def index_document(
    chunks: list[DocumentChunk],
    settings: AppSettings,
) -> IndexingResult:
    """Run the full indexing pipeline for a list of document chunks.

    Steps:

    1. Generate dense embeddings for all chunks via ``EmbeddingService``.
    2. Upsert embedded chunks into the Pinecone vector index.
    3. Update the local BM25 sparse index and persist it to disk.

    Args:
        chunks: Parsed and chunked document segments.
        settings: Fully-loaded application settings.

    Returns:
        ``IndexingResult`` with upsert statistics and timing information.

    Raises:
        ValueError: If *chunks* is empty.
    """
    log = logger.bind(operation="index_document")

    if not chunks:
        raise ValueError("Cannot index an empty chunk list.")

    log.info("index_document_start", chunk_count=len(chunks))
    pipeline_start = time.monotonic()

    # 1. Embedding ----------------------------------------------------------
    voyage_key: SecretStr | None = None
    if hasattr(settings.api_keys, "voyage_api_key"):
        voyage_key = settings.api_keys.voyage_api_key  # type: ignore[attr-defined]

    embedding_svc = EmbeddingService(
        config=settings.embedding,
        openai_api_key=settings.api_keys.openai_api_key,
        voyage_api_key=voyage_key,
    )

    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_svc.embed_texts(texts)

    # 2. Build IndexedChunkRecords -----------------------------------------
    records: list[IndexedChunkRecord] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        flat_meta = _flatten_chunk_metadata(chunk)
        records.append(
            IndexedChunkRecord(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                pinecone_metadata=flat_meta,
            )
        )

    # 3. Pinecone upsert ---------------------------------------------------
    pinecone_indexer = PineconeIndexer(
        config=settings.pinecone,
        api_key=settings.api_keys.pinecone_api_key,
    )
    pinecone_indexer.ensure_index_exists()
    result = pinecone_indexer.upsert_chunks(records)

    # 4. BM25 update -------------------------------------------------------
    bm25_mgr = BM25IndexManager()
    bm25_index_path = settings.bm25_index_path
    bm25_chunks_path = settings.bm25_chunks_path

    try:
        bm25_mgr.load_index(bm25_index_path, bm25_chunks_path)
        bm25_mgr.add_chunks(chunks)
    except FileNotFoundError:
        log.info("bm25_no_existing_index_building_fresh")
        bm25_mgr.build_index(chunks)

    bm25_mgr.save_index(bm25_index_path, bm25_chunks_path)

    # 5. Final result -------------------------------------------------------
    total_elapsed = time.monotonic() - pipeline_start
    result = IndexingResult(
        total_chunks=result.total_chunks,
        successfully_indexed=result.successfully_indexed,
        failed_chunks=result.failed_chunks,
        index_name=result.index_name,
        duration_seconds=round(total_elapsed, 3),
    )

    log.info(
        "index_document_complete",
        total_chunks=result.total_chunks,
        successfully_indexed=result.successfully_indexed,
        failed_chunks=result.failed_chunks,
        duration_seconds=result.duration_seconds,
    )
    return result
