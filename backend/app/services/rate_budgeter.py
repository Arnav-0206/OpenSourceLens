"""Entity-level GitHub API request budgeter and quota manager.

Enforces strict total request ceilings (10 for Fast Snapshot, 50 for Standard Diagnostic)
and entity-level allocations across all GitHub REST endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.app.core.exceptions import (
    BudgetExceededException,
    EntityBudgetExceededException,
)

# Canonical budget limits defined in Section 5.1 of Phase B Plan
BUDGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fast_snapshot": {
        "total_limit": 10,
        "entity_limits": {
            "repository_metadata": 1,
            "commits": 1,
            "commit_details": 0,
            "contributors": 1,
            "pull_requests": 1,
            "pr_reviews": 2,
            "pr_inline_comments": 0,
            "issues": 1,
            "issue_comments": 1,
            "releases": 1,
            "headroom": 1,
        },
    },
    "standard_diagnostic": {
        "total_limit": 50,
        "entity_limits": {
            "repository_metadata": 1,
            "commits": 3,
            "commit_details": 5,
            "contributors": 2,
            "pull_requests": 3,
            "pr_reviews": 20,
            "pr_inline_comments": 2,
            "issues": 5,
            "issue_comments": 3,
            "releases": 1,
            "headroom": 5,
        },
    },
}


class EntityRateBudgeter:
    """Manages and enforces entity-level and overall GitHub API request quotas."""

    def __init__(self, mode: str = "standard_diagnostic") -> None:
        """Initialize budgeter for the specified collection mode.

        Args:
            mode: 'fast_snapshot' (10 requests) or 'standard_diagnostic' (50 requests).

        Raises:
            ValueError: If mode is unrecognized.
        """
        normalized_mode = mode.lower().strip()
        if normalized_mode not in BUDGET_CONFIGS:
            raise ValueError(
                f"Unknown budget mode '{mode}'. Expected one of: {list(BUDGET_CONFIGS.keys())}"
            )

        self.mode = normalized_mode
        config = BUDGET_CONFIGS[normalized_mode]
        self.total_limit: int = config["total_limit"]
        self.entity_limits: Dict[str, int] = dict(config["entity_limits"])
        
        # Tracking counters
        self.total_used: int = 0
        self.entity_used: Dict[str, int] = {k: 0 for k in self.entity_limits}

    @property
    def remaining_total(self) -> int:
        """Total requests remaining before overall budget cap is reached."""
        return max(0, self.total_limit - self.total_used)

    @property
    def is_exhausted(self) -> bool:
        """Whether the overall request budget has been completely exhausted."""
        return self.total_used >= self.total_limit

    @property
    def budget_mode(self) -> str:
        """Alias for collection budget mode name."""
        return self.mode

    def remaining_for_entity(self, entity: str) -> int:
        """Remaining requests allowable for a specific entity type."""
        entity_key = entity.lower().strip()
        limit = self.entity_limits.get(entity_key, 0)
        used = self.entity_used.get(entity_key, 0)
        entity_remaining = max(0, limit - used)
        return min(self.remaining_total, entity_remaining)

    def can_consume(self, entity: str, count: int = 1) -> bool:
        """Check if request(s) can be made without breaching entity or total caps.

        Args:
            entity: Entity category name.
            count: Number of requests to check (default 1).

        Returns:
            bool: True if both overall and entity budgets have sufficient quota.
        """
        if count <= 0:
            return True

        if self.total_used + count > self.total_limit:
            return False

        entity_key = entity.lower().strip()
        limit = self.entity_limits.get(entity_key, 0)
        used = self.entity_used.get(entity_key, 0)

        return used + count <= limit

    def consume(self, entity: str, count: int = 1) -> None:
        """Consume request quota for an entity.

        Args:
            entity: Entity category name.
            count: Number of requests consumed (default 1).

        Raises:
            BudgetExceededException: If overall budget cap is breached.
            EntityBudgetExceededException: If entity-specific quota is breached.
        """
        if count <= 0:
            return

        entity_key = entity.lower().strip()

        # Enforce overall total ceiling first
        if self.total_used + count > self.total_limit:
            raise BudgetExceededException(
                f"Overall request budget of {self.total_limit} exceeded for mode '{self.mode}'. "
                f"Attempted to consume {count} requests with {self.total_used} already used.",
                budget_mode=self.mode,
                total_used=self.total_used,
                max_allowed=self.total_limit,
            )

        # Enforce entity-level allocation ceiling
        limit = self.entity_limits.get(entity_key, 0)
        used = self.entity_used.get(entity_key, 0)

        if used + count > limit:
            raise EntityBudgetExceededException(
                entity=entity_key,
                message=(
                    f"Entity quota of {limit} requests for '{entity_key}' exceeded in mode '{self.mode}'. "
                    f"Attempted to consume {count} requests with {used} already used."
                ),
                entity_used=used,
                entity_max=limit,
            )

        # Update counters
        self.total_used += count
        self.entity_used[entity_key] = used + count

    def get_telemetry(self) -> Dict[str, Any]:
        """Produce structured telemetry report of budget consumption."""
        return {
            "budget_mode": self.mode,
            "total_limit": self.total_limit,
            "total_used": self.total_used,
            "total_remaining": self.remaining_total,
            "is_exhausted": self.is_exhausted,
            "entity_limits": dict(self.entity_limits),
            "entity_used": dict(self.entity_used),
            "entity_remaining": {
                k: self.remaining_for_entity(k) for k in self.entity_limits
            },
        }
