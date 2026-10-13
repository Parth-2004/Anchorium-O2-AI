import pytest
from datetime import date, timedelta
from unittest.mock import patch

from rag_pipeline.agents.confidence import (
    classify_confidence,
    ConfidenceTier,
    STALE_SOURCE_MONTHS,
)


@pytest.fixture
def mock_today():
    """Mock today's date for deterministic staleness testing."""
    mocked_date = date(2024, 1, 1)
    with patch("rag_pipeline.agents.confidence.date") as mock_date:
        mock_date.today.return_value = mocked_date
        yield mocked_date


class TestClassifyConfidence:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            # LOW tier due to triggers
            (
                {"has_direct_source": False},
                ConfidenceTier.LOW,
            ),
            (
                {"has_direct_source": True, "sources_conflict": True},
                ConfidenceTier.LOW,
            ),
            (
                {"has_direct_source": True, "is_judgment_call": True},
                ConfidenceTier.LOW,
            ),
            # HIGH tier (perfect conditions)
            (
                {"has_direct_source": True},
                ConfidenceTier.HIGH,
            ),
        ],
    )
    def test_low_and_high_triggers(self, kwargs, expected):
        """Test the explicit LOW and HIGH triggers that do not rely on date."""
        assert classify_confidence(**kwargs) == expected

    def test_stale_source_medium(self, mock_today):
        """Test that a stale source returns MEDIUM tier."""
        # STALE_SOURCE_MONTHS is 12 (12 * 30 = 360 days)
        # cutoff = today - 360 days
        # If source_date < cutoff, it's MEDIUM.

        # Exact cutoff is today - 360 days
        cutoff = mock_today - timedelta(days=STALE_SOURCE_MONTHS * 30)

        # 1 day older than cutoff -> MEDIUM
        stale_date = cutoff - timedelta(days=1)
        assert classify_confidence(has_direct_source=True, source_date=stale_date) == ConfidenceTier.MEDIUM

    def test_fresh_source_high(self, mock_today):
        """Test that a fresh source (newer than cutoff) returns HIGH tier."""
        cutoff = mock_today - timedelta(days=STALE_SOURCE_MONTHS * 30)

        # Exact cutoff day -> not less than cutoff -> HIGH
        assert classify_confidence(has_direct_source=True, source_date=cutoff) == ConfidenceTier.HIGH

        # 1 day newer than cutoff -> HIGH
        fresh_date = cutoff + timedelta(days=1)
        assert classify_confidence(has_direct_source=True, source_date=fresh_date) == ConfidenceTier.HIGH
