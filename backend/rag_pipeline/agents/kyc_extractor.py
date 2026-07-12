"""Agent 5 — KYC / Data Extraction Agent.

Extracts structured data from unstructured uploads for downstream agents.
This agent is a FORMATTING LAYER — it never performs identity/AML
verification itself. Every extracted field carries a confidence score
and low-confidence fields are flagged for human re-entry.
"""

from __future__ import annotations

from typing import Any

from rag_pipeline.agents.base import AgentOutput, BaseAgent
from rag_pipeline.agents.confidence import ConfidenceTier


class KycExtractorAgent(BaseAgent):
    """KYC / Data Extraction Agent — Agent 5."""

    @property
    def agent_name(self) -> str:
        return "kyc_extractor"

    def process(
        self,
        query: str,
        context_chunks: list[Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Extract structured data from unstructured document content.

        Args:
            query: The extraction request describing what to extract.
            context_chunks: Document chunks from uploaded PDFs/files.
            input_data: Not used for this agent (it's the first in the pipeline).

        Returns:
            ``AgentOutput`` with extracted fields, per-field confidence,
            and cross-border data flags.
        """
        system_prompt = self._get_system_prompt()

        text_parts: list[str] = []
        image_contents: list[dict[str, Any]] = []
        has_vision = False

        if context_chunks:
            # Separate text chunks from image chunks
            text_chunks = [c for c in context_chunks if isinstance(c, dict) and not c.get("is_image")]
            image_chunks = [c for c in context_chunks if isinstance(c, dict) and c.get("is_image")]

            if text_chunks:
                context_block = self._build_context_block(text_chunks)
                text_parts.append(
                    f"UPLOADED TEXT CONTENT:\n========================\n\n{context_block}\n\n========================"
                )

            if image_chunks:
                has_vision = True
                text_parts.append(f"UPLOADED IMAGES:\nI have attached {len(image_chunks)} images of documents. Please carefully analyze them using your vision capabilities.")
                for img in image_chunks:
                    b64 = img.get("base64_data", "")
                    mime = img.get("mime_type", "image/png")
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}"
                        }
                    })

        text_parts.append(f"EXTRACTION REQUEST:\n{query}")
        text_parts.append(
            "Extract structured data from the document content above. "
            "Provide your output as a JSON object following the format "
            "in your system instructions. Tag every extracted field with "
            "a confidence score. Flag any low-confidence fields for human "
            "re-entry. Do NOT perform identity or AML verification — "
            "prepare clean data only."
        )

        user_message_text = "\n\n".join(text_parts)

        # If there are images, format as a multimodal message array, else just a string
        if has_vision:
            user_message = [{"type": "text", "text": user_message_text}] + image_contents
            # Switch to llama3.2-vision if local, or whatever vision model is configured
            model_override = "llama3.2-vision"
        else:
            user_message = user_message_text
            model_override = None

        raw_response = self._call_llm(system_prompt, user_message, model_override=model_override)
        data = self._parse_json_response(raw_response)

        # Check cross-border data flag
        cross_border = bool(data.get("cross_border_data_flag", False))
        if cross_border:
            self._log.warning(
                "cross_border_data_flag_raised",
                details=data.get("cross_border_data_details", ""),
            )

        # Determine confidence from per-field confidence scores
        confidence_by_field = data.get("confidence_by_field", {})
        field_tiers: list[ConfidenceTier] = []
        for field_name, tier_str in confidence_by_field.items():
            if isinstance(tier_str, str) and tier_str.upper() in ("HIGH", "MEDIUM", "LOW"):
                field_tiers.append(ConfidenceTier(tier_str.upper()))
            else:
                field_tiers.append(ConfidenceTier.LOW)

        # Overall confidence
        if field_tiers:
            if ConfidenceTier.LOW in field_tiers:
                overall = ConfidenceTier.LOW
            elif ConfidenceTier.MEDIUM in field_tiers:
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
            requires_ca_review=overall != ConfidenceTier.HIGH or bool(data.get("flagged_for_human_review")),
            cross_border_data_flag=cross_border,
            model_used=self._config.model_name,
            raw_response=raw_response,
        )
