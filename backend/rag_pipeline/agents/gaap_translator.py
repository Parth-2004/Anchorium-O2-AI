"""Agent 2 — GAAP ⇄ Ind AS Translation Agent.

Produces a first-draft mapping of US GAAP financials to Ind AS equivalents
for CA review. Never a final restatement. Always flags LIFO, leases, R&D,
FX translation, and ECL methodology as requiring CA judgment.
"""

from __future__ import annotations

from typing import Any

from rag_pipeline.agents.base import AgentOutput, BaseAgent
from rag_pipeline.agents.confidence import ConfidenceTier

# Categories that are ALWAYS flagged for CA judgment per the production spec
_ALWAYS_FLAG_CATEGORIES = frozenset(
    {
        "inventory_costing",
        "lifo",
        "leases",
        "lease",
        "r_and_d",
        "r&d",
        "development_costs",
        "research_development",
        "functional_currency",
        "hyperinflation",
        "fx_translation",
        "foreign_currency",
        "expected_credit_loss",
        "ecl",
        "cecl",
    }
)


class GaapTranslatorAgent(BaseAgent):
    """GAAP ⇄ Ind AS Translation Agent — Agent 2."""

    @property
    def agent_name(self) -> str:
        return "gaap_translator"

    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Translate US GAAP data to Ind AS equivalents.

        Args:
            query: The translation request or specific GAAP items to map.
            context_chunks: RAG-retrieved accounting standard references.
            input_data: Extracted US GAAP financial data from KYC agent.

        Returns:
            ``AgentOutput`` with side-by-side mapping table and CA-flagged items.
        """
        system_prompt = self._get_system_prompt()

        parts: list[str] = []

        if context_chunks:
            context_block = self._build_context_block(context_chunks)
            parts.append(
                "ACCOUNTING STANDARD REFERENCES:\n"
                "========================\n\n"
                f"{context_block}\n\n"
                "========================"
            )

        if input_data:
            parts.append(f"US GAAP FINANCIAL DATA (from extraction):\n{_format_financial_data(input_data)}")

        parts.append(f"TRANSLATION REQUEST:\n{query}")
        parts.append(
            "Provide your analysis as a JSON object following the "
            "output format specified in your system instructions. "
            "Remember: NEVER use 'certified', 'final', 'audited', or "
            "'GAAP-compliant' to describe your output."
        )

        user_message = "\n\n".join(parts)
        raw_response = self._call_llm(system_prompt, user_message)
        data = self._parse_json_response(raw_response)

        # Determine confidence — check mapping table entries
        mapping_table = data.get("mapping_table", [])
        ca_judgment_items = data.get("items_requiring_ca_judgment", [])
        has_ca_items = bool(ca_judgment_items)

        # Scan for auto-flagged categories
        confidence_tiers: list[ConfidenceTier] = []
        for item in mapping_table:
            raw_conf = item.get("confidence", "LOW")
            if isinstance(raw_conf, str) and raw_conf.upper() in ("HIGH", "MEDIUM", "LOW"):
                confidence_tiers.append(ConfidenceTier(raw_conf.upper()))
            else:
                confidence_tiers.append(ConfidenceTier.LOW)

        # Any CA-judgment item forces at least MEDIUM
        if has_ca_items:
            confidence_tiers.append(ConfidenceTier.MEDIUM)

        # Overall confidence
        if confidence_tiers:
            if ConfidenceTier.LOW in confidence_tiers:
                overall = ConfidenceTier.LOW
            elif ConfidenceTier.MEDIUM in confidence_tiers:
                overall = ConfidenceTier.MEDIUM
            else:
                overall = ConfidenceTier.HIGH
        else:
            overall = ConfidenceTier.LOW

        return AgentOutput(
            agent_name=self.agent_name,
            answer=data.get("answer", raw_response),
            structured_data=data,
            confidence_tier=overall,
            requires_ca_review=True,
            flaws=data.get('flaws', []),  # GAAP translation always needs CA review
            model_used=self._config.hf_model_name,
            raw_response=raw_response,
        )


def _format_financial_data(data: dict[str, Any]) -> str:
    """Format financial data for the user message."""
    import json

    return json.dumps(data, indent=2, default=str)
