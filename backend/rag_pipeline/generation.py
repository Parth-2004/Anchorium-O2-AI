"""LLM-based compliance answer generation with hallucination validation.

Implements a two-stage pipeline:
1. **Generation** — a strictly-prompted OpenAI completion that produces
   structured JSON answers with per-claim citations grounded in retrieved
   regulatory chunks.
2. **Validation** — a deterministic self-reflection pass that extracts
   every factual claim from the generated answer, traces each claim back to
   a source chunk, and flags unsupported or extra-contextual assertions.

The loop retries generation up to ``validation_max_retries`` times when the
validator detects hallucinations, and ultimately returns either a verified
response or a clearly-flagged unverified response with a refusal reason.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog
import tiktoken
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_pipeline.config import AppSettings, GenerationConfig

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

from rag_pipeline.retrieval import ScoredChunk

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A source citation linking a claim to a regulatory chunk.

    Each citation maps a specific factual assertion in the generated answer
    back to an identified regulatory document chunk, including a verbatim
    excerpt from that chunk that substantiates the claim.
    """

    chunk_id: str = Field(
        ...,
        description="Unique identifier of the source chunk.",
    )
    document_title: str = Field(
        ...,
        description="Title of the regulatory document.",
    )
    circular_number: str = Field(
        ...,
        description="RBI/FEMA circular or master direction number.",
    )
    issuing_body: str = Field(
        ...,
        description="Regulatory body that issued the document (e.g. RBI, SEBI).",
    )
    effective_date: str = Field(
        ...,
        description="Effective date in ISO-8601 format.",
    )
    hierarchical_path: str = Field(
        ...,
        description="Structural path within the document (e.g. Part > Chapter > Section).",
    )
    relevant_excerpt: str = Field(
        ...,
        description="Verbatim excerpt from the chunk that supports the claim.",
    )


class ValidationResult(BaseModel):
    """Result of the hallucination validation check.

    Captures a structured assessment of whether every factual claim in the
    generated answer can be traced to a source chunk.
    """

    is_valid: bool = Field(
        ...,
        description="True when all factual claims are verifiably supported.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Validator confidence in the assessment (0.0–1.0).",
    )
    failed_claims: list[str] = Field(
        default_factory=list,
        description="Factual claims that could not be verified against sources.",
    )
    reasoning: str = Field(
        ...,
        description="Chain-of-thought reasoning produced by the validator.",
    )


class ComplianceResponse(BaseModel):
    """The final verified (or flagged-unverified) compliance response.

    Wraps the generated answer, its citations, the validation result, and
    operational metadata about the generation run.
    """

    query: str = Field(
        ...,
        description="Original user query.",
    )
    answer: str = Field(
        ...,
        description="Generated compliance answer text.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations for factual claims in the answer.",
    )
    validation_result: ValidationResult = Field(
        ...,
        description="Outcome of the hallucination validation loop.",
    )
    is_verified: bool = Field(
        ...,
        description="True only when the validation loop confirmed all claims.",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Reason for refusal or unverified flag, if applicable.",
    )
    generation_attempts: int = Field(
        ...,
        ge=1,
        description="Total number of generation attempts (including retries).",
    )
    model_used: str = Field(
        ...,
        description="OpenAI model identifier used for generation.",
    )
    context_chunks_used: int = Field(
        ...,
        ge=0,
        description="Number of retrieval chunks included in the context window.",
    )


# ---------------------------------------------------------------------------
# Retry decorator for OpenAI calls
# ---------------------------------------------------------------------------

_OPENAI_RETRY = retry(
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError),
    ),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)


# ---------------------------------------------------------------------------
# ComplianceGenerator
# ---------------------------------------------------------------------------


class ComplianceGenerator:
    """Generates grounded compliance answers from retrieved regulatory chunks.

    Uses a heavily-constrained system prompt to force the LLM into producing
    structured JSON with per-claim citations, then runs a separate
    ``HallucinationValidator`` to verify every claim against source material
    before returning the response.

    Args:
        config: Generation hyper-parameters (model, temperature, token limits).
        api_key: OpenAI API key wrapped in ``SecretStr``.
    """

    def __init__(self, config: GenerationConfig, api_key: SecretStr) -> None:
        self._config = config
        self._client = OpenAI(api_key=api_key.get_secret_value())
        self._validator = HallucinationValidator(config=config, api_key=api_key)
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._log = logger.bind(component="ComplianceGenerator")

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
    ) -> ComplianceResponse:
        """Run the full generation + validation pipeline.

        Args:
            query: Natural-language compliance question.
            context_chunks: Scored and ranked regulatory chunks from retrieval.

        Returns:
            A ``ComplianceResponse`` — verified when possible, flagged
            otherwise.
        """
        self._log.info(
            "generation_started",
            query_length=len(query),
            context_chunks=len(context_chunks),
        )

        # 1. Empty context → immediate refusal
        if not context_chunks:
            self._log.warning("generation_refused_empty_context")
            return self._refusal_response(
                query=query,
                reason="Insufficient regulatory evidence found for this query.",
            )

        # 2. Build context block and optionally truncate
        working_chunks = list(context_chunks)
        system_prompt = self._build_system_prompt()
        context_block = self._build_context_block(working_chunks)
        user_message = self._build_user_message(query, context_block)

        total_tokens = (
            self._count_tokens(system_prompt)
            + self._count_tokens(user_message)
        )

        while total_tokens > self._config.context_window_limit and len(working_chunks) > 1:
            removed = working_chunks.pop()
            self._log.warning(
                "context_truncated",
                removed_chunk_id=removed.chunk.chunk_id,
                removed_final_score=removed.final_score,
                remaining_chunks=len(working_chunks),
            )
            context_block = self._build_context_block(working_chunks)
            user_message = self._build_user_message(query, context_block)
            total_tokens = (
                self._count_tokens(system_prompt)
                + self._count_tokens(user_message)
            )

        if total_tokens > self._config.context_window_limit:
            self._log.error(
                "context_still_exceeds_limit_after_truncation",
                total_tokens=total_tokens,
                limit=self._config.context_window_limit,
            )
            return self._refusal_response(
                query=query,
                reason=(
                    "The minimum context required exceeds the model's context "
                    "window limit even after truncation."
                ),
            )

        self._log.debug(
            "context_window_budget",
            total_tokens=total_tokens,
            limit=self._config.context_window_limit,
            chunks_used=len(working_chunks),
        )

        # 3. Generation + validation loop
        last_answer = ""
        last_citations: list[Citation] = []
        last_validation = ValidationResult(
            is_valid=False,
            confidence_score=0.0,
            failed_claims=["Generation not yet attempted."],
            reasoning="No generation attempt completed.",
        )

        max_attempts = 1 + self._config.validation_max_retries
        for attempt in range(1, max_attempts + 1):
            self._log.info("generation_attempt", attempt=attempt, max=max_attempts)

            raw_response = self._call_openai(system_prompt, user_message)
            last_answer, last_citations = self._parse_llm_response(raw_response)

            last_validation = self._validator.validate(
                answer=last_answer,
                citations=last_citations,
                source_chunks=working_chunks,
            )

            if last_validation.is_valid:
                self._log.info(
                    "generation_validated",
                    attempt=attempt,
                    confidence=last_validation.confidence_score,
                )
                return ComplianceResponse(
                    query=query,
                    answer=last_answer,
                    citations=last_citations,
                    validation_result=last_validation,
                    is_verified=True,
                    generation_attempts=attempt,
                    model_used=self._config.model_name,
                    context_chunks_used=len(working_chunks),
                )

            self._log.warning(
                "generation_validation_failed",
                attempt=attempt,
                failed_claims=last_validation.failed_claims,
                confidence=last_validation.confidence_score,
            )

        # All retries exhausted
        self._log.error(
            "generation_validation_exhausted",
            total_attempts=max_attempts,
            failed_claims=last_validation.failed_claims,
        )
        failed_summary = "; ".join(last_validation.failed_claims)
        return ComplianceResponse(
            query=query,
            answer=last_answer,
            citations=last_citations,
            validation_result=last_validation,
            is_verified=False,
            refusal_reason=(
                f"Hallucination validation failed after {max_attempts} attempts. "
                f"Unsupported claims: {failed_summary}"
            ),
            generation_attempts=max_attempts,
            model_used=self._config.model_name,
            context_chunks_used=len(working_chunks),
        )

    # -- prompt construction ------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Return the deterministic system prompt for compliance generation.

        The prompt constrains the LLM to act as a Senior Regulatory
        Compliance Analyst, mandates structured JSON output with per-claim
        citations, and explicitly forbids unsupported assertions.
        """
        return (
            "You are a Senior Regulatory Compliance Analyst specializing in "
            "Indian cross-border lending regulations, FEMA guidelines, RBI "
            "master directions, and GIFT City IFSC operational frameworks.\n\n"
            "YOUR MANDATE:\n"
            "You will be provided with a set of numbered regulatory source "
            "chunks retrieved from official circulars, master directions, and "
            "regulatory notifications. You MUST answer the user's query based "
            "SOLELY on the information contained in these source chunks.\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. EVERY factual claim in your answer MUST be supported by a "
            "specific citation referencing one of the provided source chunks. "
            "Do NOT make any statement that cannot be directly traced to a "
            "source chunk.\n"
            "2. You MUST respond with valid JSON containing exactly two keys:\n"
            '   - "answer": A comprehensive, professionally-worded regulatory '
            "advisory response addressing the user's query.\n"
            '   - "citations": An array of citation objects. Each citation '
            "object MUST contain:\n"
            '     - "chunk_id": The Chunk ID from the source header.\n'
            '     - "document_title": The Document title from the source header.\n'
            '     - "circular_number": The Circular number from the source header.\n'
            '     - "issuing_body": The Issuing Body from the source header.\n'
            '     - "effective_date": The Effective Date from the source header '
            "(ISO format).\n"
            '     - "hierarchical_path": The Path from the source header.\n'
            '     - "relevant_excerpt": A verbatim excerpt from the source '
            "chunk that directly supports the claim.\n"
            "3. You are EXPLICITLY FORBIDDEN from:\n"
            "   - Making assumptions or logical leaps beyond what the source "
            "material explicitly states.\n"
            "   - Synthesizing conclusions that are not directly supported by "
            "the provided chunks.\n"
            "   - Introducing information from your training data that is not "
            "present in the source chunks.\n"
            "   - Stating regulatory positions without a corresponding "
            "citation.\n"
            "4. If the retrieved source chunks do NOT contain sufficient "
            "information to answer the query definitively, you MUST respond "
            "with:\n"
            '   {"answer": "I cannot provide a definitive answer based on the '
            "available regulatory documents. The retrieved sources do not "
            "contain sufficient information to address this query with the "
            'required level of regulatory certainty.", "citations": []}\n'
            "5. Your tone must be professional regulatory advisory — precise, "
            "authoritative, and measured. Avoid hedging language unless the "
            "regulation itself contains qualifiers.\n"
            "6. When multiple regulations apply, present them in chronological "
            "order of effective date, noting any supersession or amendment "
            "relationships explicitly stated in the sources.\n\n"
            "OUTPUT FORMAT: Respond ONLY with the JSON object. Do not include "
            "any text before or after the JSON."
        )

    def _build_context_block(self, chunks: list[ScoredChunk]) -> str:
        """Format scored chunks into a numbered context block.

        Each chunk is rendered with its metadata header and full text so that
        the LLM can reference specific sources by number.
        """
        blocks: list[str] = []
        for idx, scored in enumerate(chunks, start=1):
            chunk = scored.chunk
            meta = chunk.metadata
            block = (
                f"[Source {idx}] (Chunk ID: {chunk.chunk_id})\n"
                f"Document: {meta.document_title}\n"
                f"Circular: {meta.circular_number} | "
                f"Issuing Body: {meta.issuing_body}\n"
                f"Effective Date: {meta.effective_date} | "
                f"Active: {meta.is_active}\n"
                f"Path: {chunk.hierarchical_path}\n"
                f"---\n"
                f"{chunk.text}\n"
                f"---"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def _build_user_message(self, query: str, context_block: str) -> str:
        """Combine the context block and user query into the user message."""
        return (
            "REGULATORY SOURCE CHUNKS:\n"
            "========================\n\n"
            f"{context_block}\n\n"
            "========================\n\n"
            f"COMPLIANCE QUERY:\n{query}\n\n"
            "Provide your answer as a JSON object with 'answer' and "
            "'citations' keys, following the system instructions exactly."
        )

    # -- LLM interaction ----------------------------------------------------

    @_OPENAI_RETRY
    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI chat completion with retry logic.

        Args:
            system_prompt: System-level instruction prompt.
            user_message: User-level message with context and query.

        Returns:
            Raw text content from the model's response.

        Raises:
            openai.APIConnectionError: After retry exhaustion.
            openai.APITimeoutError: After retry exhaustion.
            openai.RateLimitError: After retry exhaustion.
        """
        self._log.debug(
            "openai_call_started",
            model=self._config.model_name,
        )
        completion: ChatCompletion = self._client.chat.completions.create(
            model=self._config.model_name,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = completion.choices[0].message.content or ""
        self._log.debug(
            "openai_call_completed",
            response_length=len(content),
            usage_prompt=getattr(completion.usage, "prompt_tokens", None),
            usage_completion=getattr(completion.usage, "completion_tokens", None),
        )
        return content

    # -- response parsing ---------------------------------------------------

    def _parse_llm_response(
        self,
        raw_response: str,
    ) -> tuple[str, list[Citation]]:
        """Parse structured JSON from the LLM response.

        Handles common edge cases such as markdown-fenced JSON blocks and
        malformed output, gracefully degrading to an uncited answer when
        JSON parsing fails.

        Args:
            raw_response: Raw text returned by the LLM.

        Returns:
            A tuple of ``(answer_text, list_of_citations)``.
        """
        cleaned = raw_response.strip()

        # Strip markdown code fences if present
        fence_pattern = re.compile(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
            re.DOTALL,
        )
        fence_match = fence_pattern.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self._log.error(
                "llm_response_json_parse_failed",
                error=str(exc),
                raw_response_preview=raw_response[:500],
            )
            return raw_response, []

        if not isinstance(data, dict):
            self._log.error(
                "llm_response_unexpected_type",
                received_type=type(data).__name__,
            )
            return raw_response, []

        answer = data.get("answer", raw_response)
        raw_citations = data.get("citations", [])

        citations: list[Citation] = []
        for raw_cit in raw_citations:
            if not isinstance(raw_cit, dict):
                self._log.warning(
                    "citation_skipped_invalid_type",
                    received_type=type(raw_cit).__name__,
                )
                continue
            try:
                citations.append(Citation(**raw_cit))
            except (ValueError, TypeError) as exc:
                self._log.warning(
                    "citation_parse_failed",
                    error=str(exc),
                    raw_citation=raw_cit,
                )

        self._log.info(
            "llm_response_parsed",
            answer_length=len(answer),
            citations_count=len(citations),
        )
        return answer, citations

    # -- token counting -----------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding.

        Args:
            text: Input text to tokenize.

        Returns:
            Number of tokens.
        """
        return len(self._encoding.encode(text))

    # -- helpers ------------------------------------------------------------

    def _refusal_response(self, query: str, reason: str) -> ComplianceResponse:
        """Build a refusal ``ComplianceResponse`` when generation is skipped."""
        return ComplianceResponse(
            query=query,
            answer=(
                "I cannot provide a definitive answer based on the available "
                "regulatory documents."
            ),
            citations=[],
            validation_result=ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                failed_claims=[],
                reasoning=reason,
            ),
            is_verified=False,
            refusal_reason=reason,
            generation_attempts=1,
            model_used=self._config.model_name,
            context_chunks_used=0,
        )


# ---------------------------------------------------------------------------
# HallucinationValidator
# ---------------------------------------------------------------------------


class HallucinationValidator:
    """Post-generation hallucination detector.

    Runs a separate LLM pass that extracts every factual claim from the
    generated answer and attempts to trace each claim back to a specific
    source chunk.  Claims that cannot be verified are flagged as potential
    hallucinations.

    Args:
        config: Generation configuration (validation model, etc.).
        api_key: OpenAI API key wrapped in ``SecretStr``.
    """

    def __init__(self, config: GenerationConfig, api_key: SecretStr) -> None:
        self._config = config
        self._client = OpenAI(api_key=api_key.get_secret_value())
        self._log = logger.bind(component="HallucinationValidator")

    # -- public API ---------------------------------------------------------

    def validate(
        self,
        answer: str,
        citations: list[Citation],
        source_chunks: list[ScoredChunk],
    ) -> ValidationResult:
        """Validate an answer against source material for hallucinations.

        Args:
            answer: The generated compliance answer text.
            citations: Citations claimed by the generator.
            source_chunks: The original scored chunks (ground truth).

        Returns:
            A ``ValidationResult`` describing the validation outcome.
        """
        self._log.info(
            "validation_started",
            answer_length=len(answer),
            citations_count=len(citations),
            source_chunks_count=len(source_chunks),
        )

        if not citations:
            self._log.warning("validation_no_citations_provided")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                failed_claims=[
                    "No citations were provided to substantiate the answer."
                ],
                reasoning=(
                    "The generated answer did not include any citations. "
                    "All factual claims are therefore unverified."
                ),
            )

        prompt = self._build_validation_prompt(answer, citations, source_chunks)

        raw_result = self._call_validation_model(prompt)
        return self._parse_validation_response(raw_result)

    # -- prompt construction ------------------------------------------------

    def _build_validation_prompt(
        self,
        answer: str,
        citations: list[Citation],
        source_chunks: list[ScoredChunk],
    ) -> str:
        """Build a deterministic validation prompt.

        The prompt is designed to be as explicit and unambiguous as possible,
        leaving no room for subjective interpretation by the validation model.
        """
        # --- citations section ---
        citation_lines: list[str] = []
        for idx, cit in enumerate(citations, start=1):
            citation_lines.append(
                f"Citation {idx}:\n"
                f"  Chunk ID: {cit.chunk_id}\n"
                f"  Document: {cit.document_title}\n"
                f"  Circular: {cit.circular_number}\n"
                f"  Issuing Body: {cit.issuing_body}\n"
                f"  Effective Date: {cit.effective_date}\n"
                f"  Path: {cit.hierarchical_path}\n"
                f"  Relevant Excerpt: {cit.relevant_excerpt}"
            )
        citations_block = "\n\n".join(citation_lines)

        # --- source chunks section (ground truth) ---
        source_lines: list[str] = []
        for idx, scored in enumerate(source_chunks, start=1):
            chunk = scored.chunk
            meta = chunk.metadata
            source_lines.append(
                f"Source Chunk {idx} (Chunk ID: {chunk.chunk_id}):\n"
                f"  Document: {meta.document_title}\n"
                f"  Circular: {meta.circular_number}\n"
                f"  Issuing Body: {meta.issuing_body}\n"
                f"  Effective Date: {meta.effective_date}\n"
                f"  Active: {meta.is_active}\n"
                f"  Path: {chunk.hierarchical_path}\n"
                f"  Full Text:\n{chunk.text}"
            )
        sources_block = "\n\n".join(source_lines)

        return (
            "You are a Regulatory Compliance Validation Agent. Your SOLE "
            "purpose is to determine whether a generated compliance answer is "
            "faithfully grounded in the provided source material.\n\n"
            "TASK:\n"
            "1. Read the GENERATED ANSWER below.\n"
            "2. Read the CITATIONS CLAIMED by the generator.\n"
            "3. Read the SOURCE CHUNKS (ground truth).\n"
            "4. Extract EVERY distinct factual claim from the GENERATED "
            "ANSWER.\n"
            "5. For EACH factual claim, determine whether it can be DIRECTLY "
            "and UNAMBIGUOUSLY traced to text in one or more SOURCE CHUNKS.\n"
            "6. A claim FAILS verification if:\n"
            "   a. It states something not present in any source chunk.\n"
            "   b. It represents a logical leap or inference beyond what the "
            "source explicitly states.\n"
            "   c. It introduces information from outside the source chunks.\n"
            "   d. It misrepresents, exaggerates, or subtly alters the meaning "
            "of the source material.\n"
            "   e. Its cited source chunk does not actually contain the "
            "claimed information.\n\n"
            "GENERATED ANSWER:\n"
            "==================\n"
            f"{answer}\n"
            "==================\n\n"
            "CITATIONS CLAIMED:\n"
            "==================\n"
            f"{citations_block}\n"
            "==================\n\n"
            "SOURCE CHUNKS (GROUND TRUTH):\n"
            "==================\n"
            f"{sources_block}\n"
            "==================\n\n"
            "RESPOND with a JSON object containing EXACTLY these keys:\n"
            '- "is_valid": boolean — true ONLY if EVERY factual claim is '
            "directly supported by source chunks.\n"
            '- "confidence_score": float between 0.0 and 1.0 — your '
            "confidence in the overall assessment.\n"
            '- "failed_claims": array of strings — each string is a factual '
            "claim from the answer that FAILED verification. Empty array if "
            "all claims pass.\n"
            '- "reasoning": string — your step-by-step chain-of-thought '
            "explaining how you verified each claim.\n\n"
            "OUTPUT FORMAT: Respond ONLY with the JSON object. Do not include "
            "any text before or after the JSON."
        )

    # -- model interaction --------------------------------------------------

    @_OPENAI_RETRY
    def _call_validation_model(self, prompt: str) -> str:
        """Call the validation model with retry logic.

        Args:
            prompt: Full validation prompt.

        Returns:
            Raw text content from the validation model's response.

        Raises:
            openai.APIConnectionError: After retry exhaustion.
            openai.APITimeoutError: After retry exhaustion.
            openai.RateLimitError: After retry exhaustion.
        """
        self._log.debug(
            "validation_model_call_started",
            model=self._config.validation_model,
        )
        completion: ChatCompletion = self._client.chat.completions.create(
            model=self._config.validation_model,
            temperature=0.0,
            max_tokens=self._config.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous Regulatory Compliance "
                        "Validation Agent. Respond ONLY with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or ""
        self._log.debug(
            "validation_model_call_completed",
            response_length=len(content),
        )
        return content

    # -- response parsing ---------------------------------------------------

    def _parse_validation_response(self, raw_response: str) -> ValidationResult:
        """Parse the validator's JSON response into a ``ValidationResult``.

        Falls back to a conservative failure result when parsing fails.

        Args:
            raw_response: Raw text returned by the validation model.

        Returns:
            Parsed ``ValidationResult``.
        """
        cleaned = raw_response.strip()

        # Strip markdown code fences if present
        fence_pattern = re.compile(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
            re.DOTALL,
        )
        fence_match = fence_pattern.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self._log.error(
                "validation_response_json_parse_failed",
                error=str(exc),
                raw_response_preview=raw_response[:500],
            )
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                failed_claims=[
                    "Validation response could not be parsed as JSON."
                ],
                reasoning=f"JSON parse error: {exc}. Raw: {raw_response[:300]}",
            )

        if not isinstance(data, dict):
            self._log.error(
                "validation_response_unexpected_type",
                received_type=type(data).__name__,
            )
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                failed_claims=[
                    "Validation response was not a JSON object."
                ],
                reasoning=f"Expected dict, got {type(data).__name__}.",
            )

        # Extract with safe defaults
        is_valid = bool(data.get("is_valid", False))

        raw_confidence = data.get("confidence_score", 0.0)
        try:
            confidence_score = float(raw_confidence)
            confidence_score = max(0.0, min(1.0, confidence_score))
        except (ValueError, TypeError):
            self._log.warning(
                "validation_confidence_parse_failed",
                raw_value=raw_confidence,
            )
            confidence_score = 0.0

        raw_failed = data.get("failed_claims", [])
        if isinstance(raw_failed, list):
            failed_claims = [str(claim) for claim in raw_failed]
        else:
            failed_claims = [str(raw_failed)]

        reasoning = str(data.get("reasoning", "No reasoning provided."))

        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            failed_claims=failed_claims,
            reasoning=reasoning,
        )

        self._log.info(
            "validation_result_parsed",
            is_valid=result.is_valid,
            confidence=result.confidence_score,
            failed_claims_count=len(result.failed_claims),
        )
        return result


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def generate_compliance_response(
    query: str,
    context_chunks: list[ScoredChunk],
    settings: AppSettings,
) -> ComplianceResponse:
    """Run the full compliance generation + validation pipeline.

    Convenience wrapper that instantiates the ``ComplianceGenerator`` (which
    internally creates a ``HallucinationValidator``) and executes the
    end-to-end generation flow.

    Args:
        query: Natural-language compliance question.
        context_chunks: Scored and ranked regulatory chunks from retrieval.
        settings: Application settings containing generation config and API keys.

    Returns:
        A fully-populated ``ComplianceResponse``.
    """
    log = logger.bind(function="generate_compliance_response")
    log.info("pipeline_invoked", query_length=len(query), chunks=len(context_chunks))

    generator = ComplianceGenerator(
        config=settings.generation,
        api_key=settings.api_keys.openai_api_key,
    )
    response = generator.generate(query=query, context_chunks=context_chunks)

    log.info(
        "pipeline_completed",
        is_verified=response.is_verified,
        generation_attempts=response.generation_attempts,
        citations_count=len(response.citations),
    )
    return response
