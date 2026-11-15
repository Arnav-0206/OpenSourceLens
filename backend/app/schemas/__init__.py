"""Pydantic schemas package for OpenSourceLens API."""

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
    IngestionPreviewRequest,
    IngestionPreviewResponse,
    IngestionStatusResponse,
    IngestionSubmitRequest,
    IngestionSubmitResponse,
    IngestionSummary,
    PaginatedResult,
    ParsedRepository,
    RepositoryConfidenceReport,
    RepositoryEligibilityResult,
)
from backend.app.schemas.models import (
    ModelInfoItem,
    ModelRegistryResponse,
)

__all__ = [
    "DatabaseHealth",
    "HealthResponse",
    "RootResponse",
    "ClassificationRequest",
    "ClassificationResponse",
    "ConfidenceLevel",
    "DomainConfidence",
    "IngestionPreviewRequest",
    "IngestionPreviewResponse",
    "IngestionStatusResponse",
    "IngestionSubmitRequest",
    "IngestionSubmitResponse",
    "IngestionSummary",
    "PaginatedResult",
    "ParsedRepository",
    "RepositoryConfidenceReport",
    "RepositoryEligibilityResult",
    "ModelInfoItem",
    "ModelRegistryResponse",
    "ErrorDetailItem",
    "ErrorResponse",
    "ValidationErrorResponse",
]
