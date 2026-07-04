"""Confidence tier system for the Anchorium Omni-Engine.

Implements the 3-tier confidence tagging mandated by Core Directives:
  - HIGH: direct match to a current, named source
  - MEDIUM: reasonable inference, or a source >12 months old
  - LOW: no clean source match, conflicting sources, or a genuine judgment call

The orchestrator uses these tiers to determine report-level status:
one LOW item → entire report is "REQUIRES CA REVIEW BEFORE CLIENT DELIVERY".
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceTier(StrEnum):
    """Three-tier confidence classification per Core Directives Rule 3."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ConfidenceTag(BaseModel):
    """A confidence tag attached to a specific claim or data point."""

    tier: ConfidenceTier = Field(
        ...,
        description="The confidence classification.",
    )
    claim: str = Field(
        ...,
        description="The specific claim or data point this tag applies to.",
    )
    reason: str = Field(
        ...,
        description="Why this tier was assigned.",
    )
    source_name: str | None = Field(
        default=None,
        description="Name of the source document, if any.",
    )
    source_date: date | None = Field(
        default=None,
        description="Effective date of the source, if known.",
    )
    requires_human_review: bool = Field(
        default=False,
        description="True if this item must be reviewed by a human.",
    )


# Threshold for considering a source "stale" (Core Directives say 12 months)
STALE_SOURCE_MONTHS: int = 12


def classify_confidence(
    *,
    has_direct_source: bool,
    source_date: date | None = None,
    sources_conflict: bool = False,
    is_judgment_call: bool = False,
    stale_threshold_months: int = STALE_SOURCE_MONTHS,
) -> ConfidenceTier:
    """Classify a claim's confidence tier based on source quality.

    Args:
        has_direct_source: Whether a current, named source directly supports
            the claim.
        source_date: Effective date of the supporting source. Used to check
            staleness.
        sources_conflict: Whether multiple sources provide contradictory
            information for this claim.
        is_judgment_call: Whether the claim requires subjective expert
            judgment rather than mechanical rule-application.
        stale_threshold_months: Number of months after which a source is
            considered stale (default: 12).

    Returns:
        The appropriate ``ConfidenceTier``.
    """
    # LOW triggers — any of these forces LOW
    if sources_conflict or is_judgment_call or not has_direct_source:
        return ConfidenceTier.LOW

    # Check source staleness
    if source_date is not None:
        cutoff = date.today() - timedelta(days=stale_threshold_months * 30)
        if source_date < cutoff:
            return ConfidenceTier.MEDIUM

    # All checks pass — direct, current, unambiguous source
    return ConfidenceTier.HIGH


def compute_report_confidence(tiers: list[ConfidenceTier]) -> ConfidenceTier:
    """Compute the overall report confidence from component tiers.

    Per Core Directives: "overall report status is the minimum, not the
    average, of its components." One LOW → entire report is LOW.

    Args:
        tiers: List of confidence tiers from all agents/claims.

    Returns:
        The minimum (most conservative) tier.
    """
    if not tiers:
        return ConfidenceTier.LOW

    if ConfidenceTier.LOW in tiers:
        return ConfidenceTier.LOW
    if ConfidenceTier.MEDIUM in tiers:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.HIGH


def requires_ca_review(tier: ConfidenceTier) -> bool:
    """Check if a confidence tier requires mandatory CA review.

    Per Core Directives: "Any LOW item routes automatically to mandatory
    human review."

    Args:
        tier: The confidence tier to check.

    Returns:
        True if mandatory human review is required.
    """
    return tier == ConfidenceTier.LOW
