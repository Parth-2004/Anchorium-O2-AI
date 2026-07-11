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

        # Fast-Path: Pre-calculate the arbitrage deterministically if inputs are clean
        pre_calc = None
        if input_data and "collateral_value_usd" in input_data and "existing_debt_usd" in input_data:
            pre_calc = self._deterministic_arbitrage_calc(input_data)

        parts.append(f"ARBITRAGE ANALYSIS REQUEST:\n{query}")

        if pre_calc is not None:
            parts.append(
                f"SYSTEM OVERRIDE: The financial math has been pre-calculated deterministically.\n"
                f"Favorable Scenario - Total Annual Cost: {pre_calc['favorable_annual_cost']} USD. "
                f"Net advantage vs liquidation: {pre_calc['net_advantage']}.\n"
                f"Downside Scenario - Forced liquidation loss: {pre_calc['downside_loss']} USD "
                f"at a margin call threshold of {pre_calc['margin_call_threshold']}% LTV.\n\n"
                f"You MUST use these exact numbers in your structured JSON output. Additionally, you MUST "
                f"include the following exact 'chart_data' object at the top level of your JSON response:\n"
                f"{_format_arbitrage_input(pre_calc['chart_data'])}\n\n"
                f"Focus your LLM capabilities entirely on formatting the qualitative narrative."
            )

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

        # Enforce deterministic override to prevent LLM formatting hallucinations
        if pre_calc is not None:
            data["chart_data"] = pre_calc["chart_data"]
            if "favorable_scenario" not in data:
                data["favorable_scenario"] = {}
            data["favorable_scenario"]["total_cost_annual"] = pre_calc["favorable_annual_cost"]
            data["favorable_scenario"]["net_advantage_vs_liquidation"] = pre_calc["net_advantage"]

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


    def _deterministic_arbitrage_calc(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deterministically calculate the arbitrage costs to avoid LLM math hallucinations."""
        def safe_float(val: Any, default: float) -> float:
            if val is None: return default
            if isinstance(val, str):
                val = val.replace(",", "").strip()
            try: return float(val)
            except (ValueError, TypeError): return default

        collateral_value = safe_float(data.get("collateral_value_usd"), 1_000_000)
        loan_amount = safe_float(data.get("loan_amount_usd"), collateral_value * 0.5)

        # Assume static rates for the fast path if not provided
        borrow_rate_pa = safe_float(data.get("borrow_rate_pa"), 0.08)
        liquidation_tax_rate = safe_float(data.get("liquidation_tax_rate"), 0.20)

        # Favorable scenario (borrowing)
        annual_interest = loan_amount * borrow_rate_pa

        # Liquidation scenario
        tax_hit = collateral_value * liquidation_tax_rate
        net_advantage = tax_hit - annual_interest

        # Downside scenario (margin call at 80% LTV)
        margin_call_threshold = 80
        critical_collateral_value = loan_amount / (margin_call_threshold / 100)
        downside_loss = collateral_value - critical_collateral_value

        # Generate chart data for the UI
        chart_data = {
            "type": "bar",
            "title": "Cost Comparison: Borrowing vs. Liquidation (Year 1)",
            "data": [
                {"name": "Borrowing (Interest)", "Cost": annual_interest},
                {"name": "Liquidation (Tax Hit)", "Cost": tax_hit}
            ]
        }

        return {
            "favorable_annual_cost": annual_interest,
            "net_advantage": f"{net_advantage} USD saved in year 1",
            "downside_loss": downside_loss,
            "margin_call_threshold": margin_call_threshold,
            "chart_data": chart_data
        }

def _format_arbitrage_input(data: dict[str, Any]) -> str:
    """Format arbitrage input data for the user message."""
    import json

    return json.dumps(data, indent=2, default=str)
