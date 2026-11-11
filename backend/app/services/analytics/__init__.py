"""Phase C Multi-Domain Analytics Engine Package."""

from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.contributors import ContributorIntelligenceCalculator
from backend.app.services.analytics.cross_domain import CrossDomainSynthesisCalculator
from backend.app.services.analytics.development_activity import DevelopmentActivityCalculator
from backend.app.services.analytics.issues import IssueIntelligenceCalculator
from backend.app.services.analytics.orchestrator import (
    AnalyticsOrchestrator,
    AnalyticsService,
    OrchestratedAnalysisBundle,
    OrchestrationExecutionResult,
)
from backend.app.services.analytics.pull_requests import PullRequestIntelligenceCalculator
from backend.app.services.analytics.releases import ReleaseEvolutionCalculator

__all__ = [
    "AnalyticsOrchestrator",
    "AnalyticsService",
    "BaseMetricCalculator",
    "CalculationResult",
    "ContributorIntelligenceCalculator",
    "CrossDomainSynthesisCalculator",
    "DevelopmentActivityCalculator",
    "IssueIntelligenceCalculator",
    "OrchestratedAnalysisBundle",
    "OrchestrationExecutionResult",
    "PullRequestIntelligenceCalculator",
    "ReleaseEvolutionCalculator",
]

