"""Agent 3 — Global Trust Score Agent.

Computes Anchorium's proprietary GTS (0–1000) — an internal risk signal for
partner banks. Enforces hard prohibitions against CIBIL/FICO conflation and
appends the mandatory disclaimer verbatim to every output.

THIS AGENT IS NOT A CREDIT BUREAU AND MUST NEVER BEHAVE LIKE ONE.
"""

from __future__ import annotations

from typing import Any

from rag_pipeline.agents.base import AgentOutput, BaseAgent
from rag_pipeline.agents.confidence import ConfidenceTier

# Mandatory disclaimer — appended verbatim to every output, no exceptions
GTS_MANDATORY_DISCLAIMER: str = (
    "Anchorium Global Trust Score (proprietary, internal use only). This is NOT a CIBIL "
    "score, FICO score, or output of any licensed Credit Information Company, and is not a "
    "substitute for one. It is one input Anchorium's partner banks may use alongside their "
    "own underwriting. The subject's actual CIBIL file, once established in India, will be "
    "independently determined by TransUnion CIBIL, Experian, Equifax, or CRIF High Mark "
    "based on their Indian credit history."
)


class TrustScoreAgent(BaseAgent):
    """Global Trust Score Agent — Agent 3."""

    @property
    def agent_name(self) -> str:
        return "trust_score"

    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Compute the Global Trust Score for a founder profile.

        Args:
            query: The trust score request.
            context_chunks: RAG-retrieved context (regulatory references).
            input_data: Structured founder data including financial metrics,
                collateral info, and credit history from upstream agents.

        Returns:
            ``AgentOutput`` with GTS breakdown, liquidity flag, and
            mandatory disclaimer.
        """
        system_prompt = self._get_system_prompt()

        parts: list[str] = []

        if context_chunks:
            context_block = self._build_context_block(context_chunks)
            parts.append(
                f"REGULATORY CONTEXT:\n========================\n\n{context_block}\n\n========================"
            )

        if input_data:
            parts.append(f"FOUNDER PROFILE DATA:\n{_format_profile_data(input_data)}")

        parts.append(f"TRUST SCORE REQUEST:\n{query}")
        parts.append(
            "Compute the Global Trust Score and provide your analysis as a "
            "JSON object following the output format in your system instructions. "
            "Remember: NEVER label any output as a CIBIL or FICO score."
        )

        user_message = "\n\n".join(parts)
        raw_response = self._call_llm(system_prompt, user_message)
        data = self._parse_json_response(raw_response)

        # Validate the disclaimer is present — if the LLM omitted it, force it
        llm_disclaimer = data.get("disclaimer", "")
        if "NOT a CIBIL" not in llm_disclaimer:
            data["disclaimer"] = GTS_MANDATORY_DISCLAIMER

        # Validate no prohibited terms leaked through
        answer = data.get("answer", raw_response)
        _validate_no_prohibited_terms(answer)

        confidence = self._extract_confidence(data)

        return AgentOutput(
            agent_name=self.agent_name,
            answer=answer,
            structured_data=data,
            confidence_tier=confidence,
            requires_ca_review=confidence != ConfidenceTier.HIGH,
            disclaimer=GTS_MANDATORY_DISCLAIMER,
            model_used=self._config.model_name,
            raw_response=raw_response,
        )


def _format_profile_data(data: dict[str, Any]) -> str:
    """Format founder profile data for the user message."""
    import json

    return json.dumps(data, indent=2, default=str)


def _validate_no_prohibited_terms(text: str) -> None:
    """Check that the output doesn't contain prohibited credit-score labels.

    This is a defense-in-depth check — the prompt already prohibits these,
    but this catches any LLM non-compliance.

    Raises:
        ValueError: If prohibited terms are found.
    """
    prohibited = [
        "CIBIL Score",
        "Estimated CIBIL",
        "CIBIL equivalent",
        "FICO equivalent",
        "converted from FICO",
        "FICO-to-CIBIL",
        "CIBIL-to-FICO",
    ]
    text_lower = text.lower()
    for term in prohibited:
        if term.lower() in text_lower:
            raise ValueError(
                f"PROHIBITED TERM DETECTED in GTS output: '{term}'. "
                f"This violates the Credit Information Companies (Regulation) "
                f"Act, 2005 guardrails. The output has been blocked."
            )
