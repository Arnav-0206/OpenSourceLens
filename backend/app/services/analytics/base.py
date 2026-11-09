"""Abstract contract and result container for analytical metric calculators.

Defines:
- CalculationResult: Container for domain metrics, coverage evaluations, confidence penalties, and findings.
- BaseMetricCalculator: Abstract base class requiring domain_name and frozen analysis_reference_time anchor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.schemas.analytics_v2 import MetricCoverageItem, RuleInterpretation


@dataclass
class CalculationResult:
    """Unified container for an analytical calculator domain output."""

    domain: str
    metrics: Dict[str, Any]
    coverage: MetricCoverageItem
    confidence_penalty: float = 0.0
    rules: List[RuleInterpretation] = field(default_factory=list)
    details: Optional[Dict[str, Any]] = None


class BaseMetricCalculator(ABC):
    """Abstract base class for all Phase C domain metric calculators.

    Guarantees:
    1. Every calculator receives the same frozen analysis_reference_time anchor.
    2. Reads are isolated to the worker's snapshot revision.
    3. Returns standardized CalculationResult for synthesis and persistence.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Domain identifier string (e.g. 'development_activity', 'pull_request_intelligence')."""
        raise NotImplementedError

    @abstractmethod
    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute analytical metrics within the current snapshot transaction.

        Args:
            repository_id: Foreign key ID of the target Repository.
            db: SQLAlchemy session inside the pinned read snapshot.
            analysis_reference_time: Frozen UTC reference timestamp for aging calculations.
            snapshot_revision: Repository revision pinned by the snapshot.

        Returns:
            CalculationResult containing metrics, coverage evaluation, and rule interpretations.
        """
        raise NotImplementedError
