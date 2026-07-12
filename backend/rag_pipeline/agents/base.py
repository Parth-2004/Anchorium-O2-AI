"""Base agent interface and output models for the Anchorium Omni-Engine.

Every specialist agent inherits from ``BaseAgent`` and implements
``process()``. The ``AgentOutput`` model standardizes what every agent
returns, ensuring the orchestrator can assemble reports from any
combination of agent outputs.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import structlog
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_pipeline.agents.confidence import ConfidenceTag, ConfidenceTier
from rag_pipeline.config import GenerationConfig
from rag_pipeline.prompts.registry import AgentPromptRegistry

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Retry decorator for OpenAI calls (shared across all agents)
_OPENAI_RETRY = retry(
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError),
    ),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)


class AgentOutput(BaseModel):
    """Standardized output from any Omni-Engine agent.

    Every agent produces this structure. The orchestrator uses it to
    assemble reports, determine confidence, and route for CA review.
    """

    agent_name: str = Field(
        ...,
        description="Name of the agent that produced this output.",
    )
    answer: str = Field(
        ...,
        description="The main narrative answer/analysis.",
    )
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific structured output (JSON-parsed).",
    )
    confidence_tier: ConfidenceTier = Field(
        default=ConfidenceTier.LOW,
        description="Overall confidence tier for this agent's output.",
    )
    confidence_tags: list[ConfidenceTag] = Field(
        default_factory=list,
        description="Per-claim confidence tags.",
    )
    requires_ca_review: bool = Field(
        default=True,
        description="Whether this output requires CA sign-off.",
    )
    disclaimer: str | None = Field(
        default=None,
        description="Agent-specific mandatory disclaimer text.",
    )
    cross_border_data_flag: bool = Field(
        default=False,
        description="True if cross-border data transfer was flagged.",
    )
    draft_banner: str = Field(
        default="DRAFT — PENDING HUMAN REVIEW. NOT LEGAL, TAX, OR CREDIT ADVICE.",
        description="The mandatory draft banner per Core Directives.",
    )
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source citations for claims made.",
    )
    model_used: str = Field(
        default="",
        description="LLM model identifier used for generation.",
    )
    raw_response: str = Field(
        default="",
        description="Raw LLM response (for audit trail).",
    )


class BaseAgent(ABC):
    """Abstract base class for all Omni-Engine specialist agents.

    Provides:
    - OpenAI-compatible client initialization (pointed at Ollama)
    - Prompt registry access
    - Shared LLM calling logic with retry
    - JSON response parsing

    Subclasses implement ``agent_name`` and ``process()``.

    Args:
        config: Generation configuration (model, temperature, etc.).
        api_key: API key (unused with Ollama but kept for interface compat).
    """

    # Ollama exposes an OpenAI-compatible API at /v1
    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, config: GenerationConfig, api_key: SecretStr) -> None:
        self._config = config
        # Connect to local Ollama instance via its OpenAI-compatible endpoint.
        # The api_key is required by the OpenAI SDK but Ollama ignores it.
        self._client = OpenAI(
            base_url=self.OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't need a real key
        )
        self._registry = AgentPromptRegistry()
        self._log = logger.bind(agent=self.agent_name)

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique identifier for this agent, matching the prompt registry."""
        ...

    @abstractmethod
    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Process a request and return standardized output.

        Args:
            query: The user's question or request.
            context_chunks: Optional RAG-retrieved context chunks.
            input_data: Optional structured input data from upstream agents.

        Returns:
            Standardized ``AgentOutput``.
        """
        ...

    def _get_system_prompt(self) -> str:
        """Get the full system prompt (Core Directives + agent-specific)."""
        return self._registry.get_system_prompt(self.agent_name)

    @_OPENAI_RETRY
    def _call_llm(
        self,
        system_prompt: str,
        user_message: str | list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_override: str | None = None,
    ) -> str:
        """Call LLM via Ollama's OpenAI-compatible API with retry logic.

        Args:
            system_prompt: Full system prompt.
            user_message: User message (string or list of content dicts for multimodal).
            temperature: Override temperature (uses config default if None).
            max_tokens: Override max tokens (uses config default if None).
            model_override: Override the model used for this specific call.

        Returns:
            Raw text content from the model's response.
        """
        model = model_override if model_override else self._config.model_name
        self._log.info(
            "llm_call_started",
            model=model,
            provider="ollama",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        completion = self._client.chat.completions.create(
            model=model,
            temperature=temperature if temperature is not None else self._config.temperature,
            messages=messages,
        )
        content = completion.choices[0].message.content or ""
        self._log.info(
            "llm_call_completed",
            response_length=len(content),
            model=model,
        )
        return content

    def _parse_json_response(self, raw_response: str) -> dict[str, Any]:
        """Parse JSON from an LLM response, handling markdown fences and extraneous text.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            Parsed dict. Returns ``{"answer": raw_response}`` on parse failure.
        """
        cleaned = raw_response.strip()

        # Extract the outermost JSON object if present, bypassing any leading/trailing text
        json_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1).strip()
        else:
            # Fallback to stripping markdown code fences if no braces found
            fence_pattern = re.compile(
                r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
                re.DOTALL,
            )
            fence_match = fence_pattern.match(cleaned)
            if fence_match:
                cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            self._log.error(
                "json_parse_failed",
                error=str(exc),
                raw_preview=raw_response[:500],
            )

        return {"answer": raw_response}

    def _build_context_block(self, context_chunks: list[Any]) -> str:
        """Format context chunks into a numbered block for the LLM.

        Handles both ScoredChunk objects (from RAG pipeline) and simple
        dicts with 'text' keys (from frontend).

        Args:
            context_chunks: List of context chunks.

        Returns:
            Formatted context string.
        """
        blocks: list[str] = []
        for idx, chunk in enumerate(context_chunks, start=1):
            if hasattr(chunk, "chunk"):
                # ScoredChunk from RAG pipeline
                c = chunk.chunk
                meta = c.metadata
                block = (
                    f"[Source {idx}] (Chunk ID: {c.chunk_id})\n"
                    f"Document: {meta.document_title}\n"
                    f"Circular: {meta.circular_number} | "
                    f"Issuing Body: {meta.issuing_body}\n"
                    f"Effective Date: {meta.effective_date} | "
                    f"Active: {meta.is_active}\n"
                    f"Path: {c.hierarchical_path}\n"
                    f"---\n"
                    f"{c.text}\n"
                    f"---"
                )
            elif isinstance(chunk, dict):
                text = chunk.get("text", "")
                doc_name = chunk.get("documentName", f"Source {idx}")
                score = chunk.get("score", 0.0)
                block = f"[Source {idx}] (Document: {doc_name}, Relevance: {score:.3f})\n---\n{text}\n---"
            else:
                block = f"[Source {idx}]\n---\n{chunk}\n---"
            blocks.append(block)
        return "\n\n".join(blocks)

    def _extract_confidence(self, data: dict[str, Any]) -> ConfidenceTier:
        """Extract confidence tier from parsed LLM response.

        Args:
            data: Parsed JSON response dict.

        Returns:
            Confidence tier (defaults to LOW if not parseable).
        """
        raw = data.get("confidence", "LOW")
        if isinstance(raw, str):
            raw = raw.upper().strip()
            if raw in ("HIGH", "MEDIUM", "LOW"):
                return ConfidenceTier(raw)
        return ConfidenceTier.LOW
