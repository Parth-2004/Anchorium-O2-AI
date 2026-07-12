"""Agent 1 — RBI/FEMA Compliance Copilot.

Identifies corporate structuring options for foreign founders entering India,
grounded in current RBI/FEMA rules. Every recommendation cites its governing
provision and flags CA/CS sign-off requirements.
"""

from __future__ import annotations

from typing import Any

from rag_pipeline.agents.base import AgentOutput, BaseAgent
from rag_pipeline.agents.confidence import ConfidenceTier


class ComplianceCopilotAgent(BaseAgent):
    """RBI/FEMA Compliance Copilot — Agent 1."""

    @property
    def agent_name(self) -> str:
        return "compliance_copilot"

    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Analyze a founder's entry scenario against RBI/FEMA regulations.

        Args:
            query: The compliance question or founder scenario description.
            context_chunks: RAG-retrieved regulatory chunks.
            input_data: Optional structured founder data from KYC agent.

        Returns:
            ``AgentOutput`` with recommended structures and compliance steps.
        """
        system_prompt = self._get_system_prompt()

        # Build user message with context and input data
        parts: list[str] = []

        if context_chunks:
            context_block = self._build_context_block(context_chunks)
            parts.append(
                f"REGULATORY SOURCE CHUNKS:\n========================\n\n{context_block}\n\n========================"
            )

        if input_data:
            parts.append(f"FOUNDER PROFILE DATA (from KYC extraction):\n{_format_input_data(input_data)}")

        parts.append(f"COMPLIANCE QUERY:\n{query}")
        parts.append(
            "Provide your analysis as a JSON object following the output format specified in your system instructions."
        )

        user_message = "\n\n".join(parts)

        # Call LLM
        raw_response = self._call_llm(system_prompt, user_message)
        data = self._parse_json_response(raw_response)

        # Extract confidence
        structures = data.get("recommended_structures", [])
        confidence_tiers = []
        for s in structures:
            raw_conf = s.get("confidence", "LOW")
            if isinstance(raw_conf, str) and raw_conf.upper() in ("HIGH", "MEDIUM", "LOW"):
                confidence_tiers.append(ConfidenceTier(raw_conf.upper()))
            else:
                confidence_tiers.append(ConfidenceTier.LOW)

        # Overall confidence is the minimum of all structure confidences
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
            requires_ca_review=overall != ConfidenceTier.HIGH or bool(data.get("open_questions_for_ca_review")),
            flaws=data.get('flaws', []),
            model_used=self._config.hf_model_name,
            raw_response=raw_response,
        )


def _format_input_data(data: dict[str, Any]) -> str:
    """Format structured input data for the user message."""
    lines: list[str] = []
    for key, value in data.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value}")
    return "\n".join(lines)
