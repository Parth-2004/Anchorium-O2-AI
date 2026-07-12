from typing import Any

from rag_pipeline.agents.base import AgentOutput, BaseAgent
from rag_pipeline.agents.confidence import ConfidenceTier


class VerificationAgent(BaseAgent):
    """Fact-Checker / Verification Agent.

    Verifies the assembled draft report against the original context chunks.
    """

    @property
    def agent_name(self) -> str:
        return "verification_agent"

    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Verify the draft report.

        Args:
            query: The draft report to verify.
            context_chunks: The original context chunks.
            input_data: Optional.

        Returns:
            Verification results.
        """
        system_prompt = self._get_system_prompt()

        context_str = ""
        if context_chunks:
            context_str = "CONTEXT CHUNKS:\n" + self._build_context_block(context_chunks)

        user_message = (
            f"Please verify the following draft report against the context chunks.\n\n"
            f"DRAFT REPORT:\n{query}\n\n"
            f"{context_str}"
        )

        raw_response = self._call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
        )

        data = self._parse_json_response(raw_response)

        # Determine confidence override
        confidence_str = data.get("confidence_override", "LOW").upper()
        try:
            confidence_tier = ConfidenceTier(confidence_str)
        except ValueError:
            confidence_tier = ConfidenceTier.LOW

        unverified_claims = data.get("unverified_claims", [])
        requires_ca_review = len(unverified_claims) > 0 or confidence_tier == ConfidenceTier.LOW

        return AgentOutput(
            agent_name=self.agent_name,
            answer=data.get("answer", "Verification complete."),
            structured_data=data,
            confidence_tier=confidence_tier,
            requires_ca_review=requires_ca_review,
            flaws=data.get('flaws', []),
            model_used=self._config.hf_model_name,
            raw_response=raw_response,
        )
