"""Agent 4 — Arbitrage & Cost Calculator Agent.

Calculates cost comparisons between borrowing INR against pledged foreign
collateral vs. liquidating foreign assets. Every "borrow beats sell"
result MUST include a margin-call downside scenario — no exceptions.
"""

from __future__ import annotations

from typing import Any

from rag_pipeline.agents.base import AgentOutput, BaseAgent
from rag_pipeline.agents.confidence import ConfidenceTier


class ArbitrageCalculatorAgent(BaseAgent):
    """Arbitrage & Cost Calculator Agent — Agent 4."""

    @property
    def agent_name(self) -> str:
        return "arbitrage_calculator"

    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Calculate arbitrage scenarios with mandatory downside analysis.

        Args:
            query: The arbitrage/cost comparison request.
            context_chunks: RAG-retrieved regulatory context on lending rules.
            input_data: GTS output, collateral data, and loan terms from
                upstream agents.

        Returns:
            ``AgentOutput`` with favorable + downside scenarios, both mandatory.
        """
        system_prompt = self._get_system_prompt()

        parts: list[str] = []

        if context_chunks:
            context_block = self._build_context_block(context_chunks)
            parts.append(
                "REGULATORY AND MARKET CONTEXT:\n"
                "========================\n\n"
                f"{context_block}\n\n"
                "========================"
            )

        if input_data:
            parts.append(f"INPUT DATA (from upstream agents):\n{_format_arbitrage_input(input_data)}")

        parts.append(f"ARBITRAGE ANALYSIS REQUEST:\n{query}")
        parts.append(
            "Provide your analysis as a JSON object following the output "
            "format in your system instructions. CRITICAL: You MUST include "
            "both a favorable_scenario AND a downside_scenario. All arithmetic "
            "must show calculation steps. Interest rates must be per annum "
            "with monthly equivalent shown."
        )

        user_message = "\n\n".join(parts)
        raw_response = self._call_llm(system_prompt, user_message)
        data = self._parse_json_response(raw_response)

        # Validate: downside_scenario MUST be present
        has_downside = bool(data.get("downside_scenario"))
        if not has_downside:
            self._log.error(
                "arbitrage_missing_downside_scenario",
                raw_preview=raw_response[:200],
            )
            # Force a warning into the answer
            data["answer"] = (
                "⚠️ WARNING: The model failed to produce a mandatory downside "
                "scenario. This output is incomplete and must not be used for "
                "decision-making without CA review.\n\n" + data.get("answer", raw_response)
            )

        # Validate: favorable_scenario present
        has_favorable = bool(data.get("favorable_scenario"))

        # Determine confidence
        if not has_downside or not has_favorable:
            overall = ConfidenceTier.LOW
        else:
            overall = self._extract_confidence(data)

        return AgentOutput(
            agent_name=self.agent_name,
            answer=data.get("answer", raw_response),
            structured_data=data,
            confidence_tier=overall,
            requires_ca_review=True,  # Arbitrage always needs review
            model_used=self._config.model_name,
            raw_response=raw_response,
        )


def _format_arbitrage_input(data: dict[str, Any]) -> str:
    """Format arbitrage input data for the user message."""
    import json

    return json.dumps(data, indent=2, default=str)
