"""Master Orchestrator — routes requests to specialist agents and assembles reports.

Implements the routing logic from the production spec:
  1. KYC/Extraction first
  2. Compliance + GAAP in parallel
  3. Global Trust Score (needs collateral findings)
  4. Arbitrage Calculator last
  5. One LOW-confidence item → entire report flagged for CA review

The orchestrator NEVER generates compliance, accounting, or credit content
itself — that is always delegated to the named specialist agent.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import SecretStr

from rag_pipeline.agents.arbitrage_calculator import ArbitrageCalculatorAgent
from rag_pipeline.agents.base import AgentOutput
from rag_pipeline.agents.compliance_copilot import ComplianceCopilotAgent
from rag_pipeline.agents.confidence import (
    ConfidenceTier,
    compute_report_confidence,
)
from rag_pipeline.agents.gaap_translator import GaapTranslatorAgent
from rag_pipeline.agents.kyc_extractor import KycExtractorAgent
from rag_pipeline.agents.trust_score import TrustScoreAgent
from rag_pipeline.agents.verification import VerificationAgent
from rag_pipeline.config import GenerationConfig

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Draft banner that opens every report
DRAFT_BANNER = "DRAFT — PENDING HUMAN REVIEW. NOT LEGAL, TAX, OR CREDIT ADVICE."

# CA review escalation banner
CA_REVIEW_BANNER = "REQUIRES CA REVIEW BEFORE CLIENT DELIVERY"


class OrchestratorResult:
    """Assembled output from the full orchestration pipeline."""

    def __init__(
        self,
        *,
        agent_outputs: dict[str, AgentOutput],
        overall_confidence: ConfidenceTier,
        requires_ca_review: bool,
        ca_review_reasons: list[str],
        assembled_report: str,
        audit_trail: dict[str, Any],
    ) -> None:
        self.agent_outputs = agent_outputs
        self.overall_confidence = overall_confidence
        self.requires_ca_review = requires_ca_review
        self.ca_review_reasons = ca_review_reasons
        self.assembled_report = assembled_report
        self.audit_trail = audit_trail

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API response."""
        return {
            "draft_banner": DRAFT_BANNER,
            "overall_confidence": self.overall_confidence.value,
            "requires_ca_review": self.requires_ca_review,
            "ca_review_reasons": self.ca_review_reasons,
            "agent_outputs": {
                name: {
                    "answer": output.answer,
                    "confidence_tier": output.confidence_tier.value,
                    "requires_ca_review": output.requires_ca_review,
                    "structured_data": output.structured_data,
                    "disclaimer": output.disclaimer,
                    "cross_border_data_flag": output.cross_border_data_flag,
                }
                for name, output in self.agent_outputs.items()
            },
            "assembled_report": self.assembled_report,
            "audit_trail": self.audit_trail,
        }


class MasterOrchestrator:
    """Orchestrates the multi-agent pipeline.

    Args:
        config: Generation configuration.
        api_key: OpenAI API key.
    """

    def __init__(self, config: GenerationConfig, api_key: SecretStr) -> None:
        self._config = config
        self._api_key = api_key
        self._log = logger.bind(component="MasterOrchestrator")

        # Initialize all specialist agents
        self._kyc = KycExtractorAgent(config=config, api_key=api_key)
        self._compliance = ComplianceCopilotAgent(config=config, api_key=api_key)
        self._gaap = GaapTranslatorAgent(config=config, api_key=api_key)
        self._trust = TrustScoreAgent(config=config, api_key=api_key)
        self._arbitrage = ArbitrageCalculatorAgent(config=config, api_key=api_key)
        self._verification = VerificationAgent(config=config, api_key=api_key)

    def run_single_agent(
        self,
        agent_name: str,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Run a single named agent directly (bypasses orchestration).

        Args:
            agent_name: Which agent to invoke.
            query: The query/request.
            context_chunks: Optional context.
            input_data: Optional structured input.

        Returns:
            The agent's output.
        """
        agents = {
            "compliance_copilot": self._compliance,
            "gaap_translator": self._gaap,
            "trust_score": self._trust,
            "arbitrage_calculator": self._arbitrage,
            "kyc_extractor": self._kyc,
            "verification_agent": self._verification,
        }

        agent = agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent '{agent_name}'. Valid agents: {sorted(agents.keys())}")

        self._log.info("single_agent_invoked", agent=agent_name)
        return agent.process(
            query=query,
            context_chunks=context_chunks,
            input_data=input_data,
        )

    def run_full_pipeline(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
    ) -> OrchestratorResult:
        """Run the full multi-agent pipeline in the correct order.

        Pipeline order (per production spec):
          1. KYC/Extraction
          2. Compliance + GAAP (parallel — run sequentially here for simplicity)
          3. Global Trust Score
          4. Arbitrage Calculator

        Args:
            query: The founder onboarding request.
            context_chunks: Uploaded document chunks.

        Returns:
            Assembled ``OrchestratorResult`` with all agent outputs.
        """
        self._log.info("full_pipeline_started")
        agent_outputs: dict[str, AgentOutput] = {}
        ca_review_reasons: list[str] = []

        # Step 1: KYC Extraction
        self._log.info("pipeline_step_1_kyc")
        kyc_output = self._kyc.process(
            query=(f"Extract all structured founder and financial data from the uploaded documents. Context: {query}"),
            context_chunks=context_chunks,
        )
        agent_outputs["kyc_extractor"] = kyc_output
        extracted_data = kyc_output.structured_data.get("extracted_fields", {})

        if kyc_output.cross_border_data_flag:
            ca_review_reasons.append("Cross-border data transfer flagged by KYC agent")

        # Step 2: Compliance + GAAP (sequential for simplicity)
        self._log.info("pipeline_step_2_compliance")
        compliance_output = self._compliance.process(
            query=query,
            context_chunks=context_chunks,
            input_data=extracted_data,
        )
        agent_outputs["compliance_copilot"] = compliance_output

        self._log.info("pipeline_step_2_gaap")
        gaap_output = self._gaap.process(
            query=query,
            context_chunks=context_chunks,
            input_data=extracted_data,
        )
        agent_outputs["gaap_translator"] = gaap_output

        # Step 3: Global Trust Score (uses compliance + GAAP findings)
        self._log.info("pipeline_step_3_trust_score")
        trust_input = {
            **extracted_data,
            "compliance_findings": compliance_output.structured_data,
            "gaap_findings": gaap_output.structured_data,
        }
        trust_output = self._trust.process(
            query=query,
            context_chunks=context_chunks,
            input_data=trust_input,
        )
        agent_outputs["trust_score"] = trust_output

        # Step 4: Arbitrage Calculator (uses GTS + all prior findings)
        self._log.info("pipeline_step_4_arbitrage")
        arbitrage_input = {
            **extracted_data,
            "global_trust_score": trust_output.structured_data,
            "compliance": compliance_output.structured_data,
        }
        arbitrage_output = self._arbitrage.process(
            query=query,
            context_chunks=context_chunks,
            input_data=arbitrage_input,
        )
        agent_outputs["arbitrage_calculator"] = arbitrage_output

        # Compute overall confidence (minimum of all agents)
        all_tiers = [o.confidence_tier for o in agent_outputs.values()]
        overall_confidence = compute_report_confidence(all_tiers)

        # Check for CA review triggers
        for name, output in agent_outputs.items():
            if output.requires_ca_review:
                ca_review_reasons.append(f"{name}: requires CA review (confidence: {output.confidence_tier.value})")
            if output.confidence_tier == ConfidenceTier.LOW:
                ca_review_reasons.append(f"{name}: LOW confidence detected")

        requires_ca = overall_confidence != ConfidenceTier.HIGH or bool(ca_review_reasons)

        # Assemble the draft report for verification
        draft_report = self._assemble_report(
            agent_outputs=agent_outputs,
            overall_confidence=overall_confidence,
            requires_ca=requires_ca,
            ca_review_reasons=ca_review_reasons,
        )

        # Step 5: Verification Agent
        self._log.info("pipeline_step_5_verification")
        verification_output = self._verification.process(
            query=draft_report,
            context_chunks=context_chunks,
        )
        agent_outputs["verification_agent"] = verification_output

        # Re-evaluate confidence and CA review requirements based on verification
        if verification_output.confidence_tier == ConfidenceTier.LOW:
            overall_confidence = ConfidenceTier.LOW
            requires_ca = True
            ca_review_reasons.append("verification_agent: Fact-check failed or low confidence")

        if verification_output.requires_ca_review:
            requires_ca = True
            ca_review_reasons.append("verification_agent: requires CA review")

        unverified_claims = verification_output.structured_data.get("unverified_claims", [])
        if unverified_claims:
            requires_ca = True
            for claim in unverified_claims:
                ca_review_reasons.append(f"UNVERIFIED CLAIM: {claim.get('claim')} ({claim.get('severity')}) - {claim.get('reason')}")

        # Final assembly of the report, possibly appending verification warnings
        assembled_report = self._assemble_report(
            agent_outputs=agent_outputs,
            overall_confidence=overall_confidence,
            requires_ca=requires_ca,
            ca_review_reasons=ca_review_reasons,
        )

        # Build audit trail
        audit_trail = {
            "agents_invoked": list(agent_outputs.keys()),
            "models_used": list({o.model_used for o in agent_outputs.values()}),
            "confidence_tiers": {name: o.confidence_tier.value for name, o in agent_outputs.items()},
            "overall_confidence": overall_confidence.value,
            "low_confidence_items": [
                name for name, o in agent_outputs.items() if o.confidence_tier == ConfidenceTier.LOW
            ],
            "cross_border_flags": [name for name, o in agent_outputs.items() if o.cross_border_data_flag],
            "unverified_claims_count": len(unverified_claims),
        }

        self._log.info(
            "full_pipeline_completed",
            overall_confidence=overall_confidence.value,
            requires_ca=requires_ca,
            agents_run=len(agent_outputs),
        )

        return OrchestratorResult(
            agent_outputs=agent_outputs,
            overall_confidence=overall_confidence,
            requires_ca_review=requires_ca,
            ca_review_reasons=ca_review_reasons,
            assembled_report=assembled_report,
            audit_trail=audit_trail,
        )

    def _assemble_report(
        self,
        *,
        agent_outputs: dict[str, AgentOutput],
        overall_confidence: ConfidenceTier,
        requires_ca: bool,
        ca_review_reasons: list[str],
    ) -> str:
        """Assemble the India Soft-Landing Playbook from agent outputs.

        Per the spec: "The final report opens with the DRAFT / NOT ADVICE
        banner, not the favorable numbers."
        """
        sections: list[str] = []

        # Banner first — always
        sections.append(f"# {DRAFT_BANNER}")

        if requires_ca:
            sections.append(f"\n## {CA_REVIEW_BANNER}")
            for reason in ca_review_reasons:
                sections.append(f"- {reason}")

        sections.append(f"\n**Overall Confidence:** {overall_confidence.value}")

        # KYC Section
        if "kyc_extractor" in agent_outputs:
            sections.append("\n---\n## 1. KYC / Data Extraction")
            sections.append(agent_outputs["kyc_extractor"].answer)

        # Compliance Section
        if "compliance_copilot" in agent_outputs:
            sections.append("\n---\n## 2. RBI/FEMA Compliance Analysis")
            sections.append(agent_outputs["compliance_copilot"].answer)

        # GAAP Section
        if "gaap_translator" in agent_outputs:
            sections.append("\n---\n## 3. US GAAP → Ind AS Translation")
            sections.append(agent_outputs["gaap_translator"].answer)

        # Trust Score Section
        if "trust_score" in agent_outputs:
            sections.append("\n---\n## 4. Global Trust Score")
            sections.append(agent_outputs["trust_score"].answer)
            if agent_outputs["trust_score"].disclaimer:
                sections.append(f"\n> **Disclaimer:** {agent_outputs['trust_score'].disclaimer}")

        # Arbitrage Section
        if "arbitrage_calculator" in agent_outputs:
            sections.append("\n---\n## 5. Arbitrage & Cost Analysis")
            sections.append(agent_outputs["arbitrage_calculator"].answer)

        # Verification Section
        if "verification_agent" in agent_outputs:
            sections.append("\n---\n## 6. Verification & Fact-Check")
            sections.append(agent_outputs["verification_agent"].answer)
            unverified_claims = agent_outputs["verification_agent"].structured_data.get("unverified_claims", [])
            if unverified_claims:
                sections.append("\n**UNVERIFIED CLAIMS DETECTED:**")
                for claim in unverified_claims:
                    sections.append(f"- **Claim:** {claim.get('claim')}\n  - **Severity:** {claim.get('severity')}\n  - **Reason:** {claim.get('reason')}")

        sections.append(f"\n---\n*Report generated by Anchorium Omni-Engine. {DRAFT_BANNER}*")

        return "\n".join(sections)
