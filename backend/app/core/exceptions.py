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


class InvalidRepositoryURLException(OpenSourceLensException):
    """Raised when a supplied GitHub URL or repository slug violates syntax, hostname, or naming rules."""

    def __init__(self, message: str, raw_input: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_code="INVALID_REPOSITORY_URL",
            details={"raw_input": raw_input} if raw_input else {},
        )


class RepositoryIneligibleException(OpenSourceLensException):
    """Raised when a repository is private, an empty stub, or otherwise ineligible for ingestion."""

    def __init__(self, message: str, repo_name: Optional[str] = None, reason: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_code="REPOSITORY_INELIGIBLE",
            details={"repo_name": repo_name, "reason": reason} if repo_name or reason else {},
        )


class GitHubAPIError(OpenSourceLensException):
    """Raised when GitHub API requests fail due to upstream server, transport, or protocol errors."""

    def __init__(
        self,
        message: str = "GitHub API request failed.",
        status_code: int = 502,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="GITHUB_API_ERROR",
            details=details,
        )


class GitHubRateLimitExceededException(GitHubAPIError):
    """Raised when GitHub API rate limits (primary or secondary) are breached."""

    def __init__(
        self,
        message: str = "GitHub API rate limit exceeded.",
        retry_after_seconds: Optional[int] = None,
        resumes_at: Optional[Any] = None,
    ) -> None:
        details = {}
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
        if resumes_at is not None:
            details["resumes_at"] = resumes_at.isoformat() if hasattr(resumes_at, "isoformat") else str(resumes_at)
        super().__init__(
            message=message,
            status_code=429,
            details=details,
        )
        self.error_code = "GITHUB_RATE_LIMIT_EXCEEDED"
        self.retry_after_seconds = retry_after_seconds
        self.resumes_at = resumes_at


class GitHubResourceNotFoundException(GitHubAPIError):
    """Raised when a repository, issue, or entity does not exist on GitHub."""

    def __init__(self, message: str = "Resource not found on GitHub.", repo_name: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            status_code=404,
            details={"repo_name": repo_name} if repo_name else {},
        )
        self.error_code = "GITHUB_RESOURCE_NOT_FOUND"


class BudgetExceededException(OpenSourceLensException):
    """Raised when the total request budget for a collection run is exhausted."""

    def __init__(
        self,
        message: str = "Total GitHub API request budget exhausted for this collection run.",
        budget_mode: Optional[str] = None,
        total_used: Optional[int] = None,
        max_allowed: Optional[int] = None,
    ) -> None:
        details = {
            "budget_mode": budget_mode,
            "total_used": total_used,
            "max_allowed": max_allowed,
        }
        super().__init__(
            message=message,
            status_code=429,
            error_code="BUDGET_EXCEEDED",
            details={k: v for k, v in details.items() if v is not None},
        )


class EntityBudgetExceededException(OpenSourceLensException):
    """Raised when a specific entity quota within a collection run is exhausted."""

    def __init__(
        self,
        entity: str,
        message: Optional[str] = None,
        entity_used: Optional[int] = None,
        entity_max: Optional[int] = None,
    ) -> None:
        msg = message or f"Request budget exhausted for entity '{entity}'."
        details = {
            "entity": entity,
            "entity_used": entity_used,
            "entity_max": entity_max,
        }
        super().__init__(
            message=msg,
            status_code=429,
            error_code="ENTITY_BUDGET_EXCEEDED",
            details={k: v for k, v in details.items() if v is not None},
        )
