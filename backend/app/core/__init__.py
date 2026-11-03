"""Core application settings, exceptions, and utilities."""

from backend.app.core.config import settings
from backend.app.core.error_handlers import register_error_handlers
from backend.app.core.exceptions import (
    DatabaseServiceError,
    EntityNotFoundException,
    InvalidParameterException,
    IssueNotFoundException,
    ModelInferenceError,
    ModelNotFoundException,
    OpenSourceLensException,
    RepositoryNotFoundException,
)

__all__ = [
    "settings",
    "register_error_handlers",
    "OpenSourceLensException",
    "EntityNotFoundException",
    "RepositoryNotFoundException",
    "IssueNotFoundException",
    "ModelNotFoundException",
    "InvalidParameterException",
    "ModelInferenceError",
    "DatabaseServiceError",
]

