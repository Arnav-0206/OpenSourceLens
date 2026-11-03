"""Domain-specific custom exceptions for OpenSourceLens.

Provides structured, hierarchical exception types with associated HTTP status codes
and machine-readable error codes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class OpenSourceLensException(Exception):
    """Base exception for all application-level domain errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_SERVER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class EntityNotFoundException(OpenSourceLensException):
    """Raised when an entity requested by identifier does not exist."""

    def __init__(
        self,
        message: str = "Requested resource was not found.",
        entity_type: str = "Resource",
        entity_id: Optional[Any] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details={"entity_type": entity_type, "entity_id": entity_id} if entity_id else {"entity_type": entity_type},
        )


class RepositoryNotFoundException(EntityNotFoundException):
    """Raised when a GitHub repository cannot be located in the database."""

    def __init__(self, repo_name: str) -> None:
        super().__init__(
            message=f"Repository '{repo_name}' not found.",
            entity_type="Repository",
            entity_id=repo_name,
        )


class IssueNotFoundException(EntityNotFoundException):
    """Raised when an issue cannot be located in the database."""

    def __init__(self, repo_name: str, issue_number: int) -> None:
        super().__init__(
            message=f"Issue #{issue_number} not found in repository '{repo_name}'.",
            entity_type="Issue",
            entity_id=f"{repo_name}#{issue_number}",
        )


class ModelNotFoundException(EntityNotFoundException):
    """Raised when a requested ML model artifact is unrecognized or missing."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"ML model '{model_id}' is not registered or supported.",
            entity_type="Model",
            entity_id=model_id,
        )


class InvalidParameterException(OpenSourceLensException):
    """Raised when a client-supplied parameter violates business logic constraints."""

    def __init__(self, message: str, parameter: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_code="INVALID_PARAMETER",
            details={"parameter": parameter} if parameter else {},
        )


class ModelInferenceError(OpenSourceLensException):
    """Raised when an ML model fails during feature extraction or inference."""

    def __init__(self, message: str, model_id: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="MODEL_INFERENCE_ERROR",
            details={"model_id": model_id} if model_id else {},
        )


class DatabaseServiceError(OpenSourceLensException):
    """Raised when database operations encounter unrecoverable connection or execution errors."""

    def __init__(self, message: str = "Database service encountered an internal error.") -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="DATABASE_ERROR",
        )
