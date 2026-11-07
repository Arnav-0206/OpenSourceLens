"""Unit tests for EntityRateBudgeter.

Verifies:
1. Fast Snapshot (10 requests) and Standard Diagnostic (50 requests) configuration.
2. Rejection of unknown budget modes.
3. Pre-flight checks with can_consume() for entity and overall ceilings.
4. Consumption accounting and counter increments.
5. Enforcement of overall budget limit via BudgetExceededException.
6. Enforcement of entity-level allocation caps via EntityBudgetExceededException.
7. Telemetry reporting accuracy.
"""

import pytest

from backend.app.core.exceptions import (
    BudgetExceededException,
    EntityBudgetExceededException,
)
from backend.app.services.rate_budgeter import EntityRateBudgeter


class TestEntityRateBudgeter:
    """Test suite for EntityRateBudgeter."""

    def test_fast_snapshot_initialization(self):
        """Fast Snapshot mode must be initialized with total_limit = 10 and exact entity limits."""
        budgeter = EntityRateBudgeter(mode="fast_snapshot")
        assert budgeter.mode == "fast_snapshot"
        assert budgeter.total_limit == 10
        assert budgeter.remaining_total == 10
        assert budgeter.is_exhausted is False
        assert budgeter.total_used == 0

        # Check key entity quotas from Section 5.1 of plan
        assert budgeter.entity_limits["repository_metadata"] == 1
        assert budgeter.entity_limits["commits"] == 1
        assert budgeter.entity_limits["commit_details"] == 0
        assert budgeter.entity_limits["contributors"] == 1
        assert budgeter.entity_limits["pull_requests"] == 1
        assert budgeter.entity_limits["pr_reviews"] == 2
        assert budgeter.entity_limits["pr_inline_comments"] == 0
        assert budgeter.entity_limits["issues"] == 1
        assert budgeter.entity_limits["issue_comments"] == 1
        assert budgeter.entity_limits["releases"] == 1

    def test_standard_diagnostic_initialization(self):
        """Standard Diagnostic mode must be initialized with total_limit = 50 and exact entity limits."""
        budgeter = EntityRateBudgeter(mode="standard_diagnostic")
        assert budgeter.mode == "standard_diagnostic"
        assert budgeter.total_limit == 50
        assert budgeter.remaining_total == 50
        assert budgeter.is_exhausted is False
        assert budgeter.total_used == 0

        # Check key entity quotas from Section 5.1 of plan
        assert budgeter.entity_limits["repository_metadata"] == 1
        assert budgeter.entity_limits["commits"] == 3
        assert budgeter.entity_limits["commit_details"] == 5
        assert budgeter.entity_limits["contributors"] == 2
        assert budgeter.entity_limits["pull_requests"] == 3
        assert budgeter.entity_limits["pr_reviews"] == 20
        assert budgeter.entity_limits["pr_inline_comments"] == 2
        assert budgeter.entity_limits["issues"] == 5
        assert budgeter.entity_limits["issue_comments"] == 3
        assert budgeter.entity_limits["releases"] == 1

    def test_unknown_budget_mode_raises_value_error(self):
        """Unrecognized budget mode must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            EntityRateBudgeter(mode="ultra_deep_mode")
        assert "Unknown budget mode" in str(exc_info.value)

    def test_can_consume_and_consume_lifecycle(self):
        """can_consume reflects remaining allowance, and consume increments counters."""
        budgeter = EntityRateBudgeter(mode="fast_snapshot")

        # Can consume 1 commit
        assert budgeter.can_consume("commits", count=1) is True
        budgeter.consume("commits", count=1)
        assert budgeter.total_used == 1
        assert budgeter.entity_used["commits"] == 1

        # Commits limit in fast_snapshot is 1 -> cannot consume another commit
        assert budgeter.can_consume("commits", count=1) is False
        assert budgeter.remaining_for_entity("commits") == 0

        # Attempting to consume beyond entity limit raises EntityBudgetExceededException
        with pytest.raises(EntityBudgetExceededException) as exc_info:
            budgeter.consume("commits", count=1)
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "ENTITY_BUDGET_EXCEEDED"

    def test_zero_allocation_entity_rejection(self):
        """Entities with 0 allocation (e.g. commit_details in fast_snapshot) cannot be consumed."""
        budgeter = EntityRateBudgeter(mode="fast_snapshot")
        assert budgeter.can_consume("commit_details", count=1) is False
        assert budgeter.remaining_for_entity("commit_details") == 0

        with pytest.raises(EntityBudgetExceededException):
            budgeter.consume("commit_details", count=1)

    def test_overall_budget_ceiling_enforcement(self):
        """Exhausting total budget ceiling raises BudgetExceededException."""
        budgeter = EntityRateBudgeter(mode="fast_snapshot")

        # Consume allowable quotas across entities up to 10
        budgeter.consume("repository_metadata", 1)  # 1
        budgeter.consume("commits", 1)               # 2
        budgeter.consume("contributors", 1)          # 3
        budgeter.consume("pull_requests", 1)         # 4
        budgeter.consume("pr_reviews", 2)            # 6
        budgeter.consume("issues", 1)                # 7
        budgeter.consume("issue_comments", 1)        # 8
        budgeter.consume("releases", 1)              # 9
        budgeter.consume("headroom", 1)              # 10

        assert budgeter.total_used == 10
        assert budgeter.remaining_total == 0
        assert budgeter.is_exhausted is True

        # Total is exhausted -> can_consume is False for any entity
        assert budgeter.can_consume("commits", 1) is False

        # Attempting to consume when overall budget is exhausted raises BudgetExceededException
        with pytest.raises(BudgetExceededException) as exc_info:
            budgeter.consume("headroom", 1)
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "BUDGET_EXCEEDED"

    def test_telemetry_reporting(self):
        """get_telemetry returns complete dictionary matching current budgeter state."""
        budgeter = EntityRateBudgeter(mode="standard_diagnostic")
        budgeter.consume("repository_metadata", 1)
        budgeter.consume("commits", 2)

        telemetry = budgeter.get_telemetry()
        assert telemetry["budget_mode"] == "standard_diagnostic"
        assert telemetry["total_limit"] == 50
        assert telemetry["total_used"] == 3
        assert telemetry["total_remaining"] == 47
        assert telemetry["is_exhausted"] is False
        assert telemetry["entity_used"]["commits"] == 2
        assert telemetry["entity_remaining"]["commits"] == 1  # 3 limit - 2 used = 1
