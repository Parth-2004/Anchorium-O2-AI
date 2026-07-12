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

        # Fast-Path: Pre-calculate the Trust Score deterministically if fields are present
        pre_calculated_score = None
        if input_data and "foreign_bureau_score" in input_data and "annual_revenue_usd" in input_data:
            pre_calculated_score = self._deterministic_gts_calc(input_data)

        parts.append(f"TRUST SCORE REQUEST:\n{query}")

        if pre_calculated_score is not None:
            parts.append(
                f"SYSTEM OVERRIDE: The Global Trust Score has been pre-calculated deterministically "
                f"as {pre_calculated_score} / 1000. You MUST use this exact number in your output for the "
                f"'global_trust_score' key. Focus your LLM capabilities entirely on explaining the qualitative "
                f"breakdown based on the regulatory context."
            )

        parts.append(
            "Provide your analysis as a JSON object following the output format in your system instructions. "
            "Remember: NEVER label any output as a CIBIL or FICO score."
        )

        user_message = "\n\n".join(parts)
        raw_response = self._call_llm(system_prompt, user_message)
        data = self._parse_json_response(raw_response)

        # Enforce deterministic override if the LLM hallucinated the math
        if pre_calculated_score is not None:
            data["global_trust_score"] = pre_calculated_score

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
            flaws=data.get('flaws', []),
            disclaimer=GTS_MANDATORY_DISCLAIMER,
            model_used=self._config.hf_model_name,
            raw_response=raw_response,
        )

    def _deterministic_gts_calc(self, data: dict[str, Any]) -> int:
        """Deterministically calculate GTS based on the weights.

        - Foreign Bureau Data: 15%
        - Revenue/ARR Stability: 20%
        - Collateral Type & Liquidity: 25%
        - Credit History: 15%
        - Debt-to-Asset: 15%
        - Industry Risk: 10%
        """
        def safe_float(val: Any, default: float) -> float:
            if val is None:
                return default
            if isinstance(val, str):
                val = val.replace(",", "").strip()
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        score = 0.0

        # 1. Foreign Bureau (normalize FICO 300-850 to 0-150)
        fico = safe_float(data.get("foreign_bureau_score"), 600)
        fico_norm = max(0, min(150, ((fico - 300) / 550) * 150))
        score += fico_norm

        # 2. Revenue (0-200)
        rev = safe_float(data.get("annual_revenue_usd"), 0)
        if rev > 1_000_000:
            score += 200
        elif rev > 500_000:
            score += 150
        elif rev > 100_000:
            score += 100
        else:
            score += 50

        # 3. Collateral (0-250)
        col = str(data.get("collateral_type", "")).lower()
        if "public" in col or "liquid" in col or "cash" in col:
            score += 250
        elif "private" in col:
            score += 100
        else:
            score += 50

        # 4. History (0-150)
        hist = int(safe_float(data.get("credit_history_years"), 0))
        if hist >= 5:
            score += 150
        elif hist >= 2:
            score += 100
        else:
            score += 50

        # 5. Debt to Asset (0-150)
        debt = safe_float(data.get("existing_debt_usd"), 0)
        assets = safe_float(data.get("total_assets_usd"), 1) # avoid div zero
        if assets == 0:
            assets = 1
        dti = debt / assets
        if dti < 0.2:
            score += 150
        elif dti < 0.5:
            score += 100
        elif dti < 0.8:
            score += 50
        else:
            score += 0

        # 6. Industry Risk (0-100)
        ind = str(data.get("industry_vertical", "")).lower()
        if "saas" in ind or "tech" in ind:
            score += 100
        elif "crypto" in ind:
            score += 20
        else:
            score += 70

        return int(min(1000, max(0, score)))


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
