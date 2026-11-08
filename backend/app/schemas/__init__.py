"""Pydantic schemas package for OpenSourceLens API."""

from backend.app.schemas.analytics import (
    GlobalAnalyticsOverview,
    RepositoryAnalyticsResponse,
    RepositoryComparisonItem,
    RepositoryComparisonResponse,
    RepositoryTrendsResponse,
    ResolutionMetrics,
    TrendPoint,
)
from backend.app.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.schemas.errors import (
    ErrorDetailItem,
    ErrorResponse,
    ValidationErrorResponse,
)
from backend.app.schemas.health import (
    DatabaseHealth,
    HealthResponse,
    RootResponse,
)
from backend.app.schemas.ingestion import (
    ConfidenceLevel,
    DomainConfidence,
    IngestionSummary,
    PaginatedResult,
    ParsedRepository,
    RepositoryConfidenceReport,
    RepositoryEligibilityResult,
)
from backend.app.schemas.issue import (
    IssueDetail,
    IssueLabelItem,
    IssueListResponse,
    IssueSummary,
)
from backend.app.schemas.models import (
    ModelInfoItem,
    ModelRegistryResponse,
)
from backend.app.schemas.related import (
    RelatedIssueMatch,
    RelatedIssuesResponse,
    RelatedSearchRequest,
)
from backend.app.schemas.repository import (
    RepositoryDetail,
    RepositorySummary,
)
from backend.app.schemas.resolution import (
    ResolutionPredictionRequest,
    ResolutionPredictionResponse,
    ResolutionStatsResponse,
)

__all__ = [
    "DatabaseHealth",
    "HealthResponse",
    "RootResponse",
    "RepositorySummary",
    "RepositoryDetail",
    "IssueSummary",
    "IssueDetail",
    "IssueLabelItem",
    "IssueListResponse",
    "ResolutionMetrics",
    "RepositoryAnalyticsResponse",
    "RepositoryComparisonItem",
    "RepositoryComparisonResponse",
    "GlobalAnalyticsOverview",
    "TrendPoint",
    "RepositoryTrendsResponse",
    "ClassificationRequest",
    "ClassificationResponse",
    "ConfidenceLevel",
    "DomainConfidence",
    "IngestionSummary",
    "PaginatedResult",
    "ParsedRepository",
    "RepositoryConfidenceReport",
    "RepositoryEligibilityResult",
    "RelatedIssueMatch",
    "RelatedIssuesResponse",
    "RelatedSearchRequest",
    "ResolutionPredictionRequest",
    "ResolutionPredictionResponse",
    "ResolutionStatsResponse",
    "ModelInfoItem",
    "ModelRegistryResponse",
    "ErrorDetailItem",
    "ErrorResponse",
    "ValidationErrorResponse",
]
