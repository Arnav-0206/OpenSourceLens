"""Pydantic schemas for standardized API error responses.

Ensures consistent JSON error envelopes across all endpoints, providing
structured diagnostics while maintaining backwards compatibility with FastAPI's
standard 'detail' field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ErrorDetailItem(BaseModel):
    """Field-specific validation error details."""

    field: str = Field(..., description="Location path of the invalid parameter or body field.")
    message: str = Field(..., description="Human-readable explanation of why validation failed.")
    type: str = Field(..., description="Machine-readable validation error code (e.g. string_too_short).")


class ErrorResponse(BaseModel):
    """Uniform error response envelope for HTTP, domain, database, and internal server errors."""

    detail: str = Field(
        ...,
        description="Human-readable description of the error (backwards-compatible with standard FastAPI detail).",
    )
    error_code: str = Field(
        ...,
        description="Standardized machine-readable error classification code.",
        examples=["RESOURCE_NOT_FOUND", "INTERNAL_SERVER_ERROR", "DATABASE_ERROR"],
    )
    status_code: int = Field(..., description="HTTP response status code.")
    path: str = Field(..., description="Request URL path where the error occurred.")
    timestamp: datetime = Field(..., description="UTC ISO-8601 timestamp when the error occurred.")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional auxiliary dictionary containing contextual metadata.",
    )


class ValidationErrorResponse(BaseModel):
    """Specialized error response envelope for 422 Unprocessable Entity validation failures."""

    detail: str = Field(
        ...,
        description="Human-readable summary of validation violations.",
        examples=["Request validation failed for 1 field(s)."],
    )
    error_code: str = Field(
        default="VALIDATION_ERROR",
        description="Machine-readable validation error code.",
    )
    status_code: int = Field(default=422, description="HTTP 422 Unprocessable Entity status code.")
    path: str = Field(..., description="Request URL path where validation failed.")
    timestamp: datetime = Field(..., description="UTC ISO-8601 timestamp when the validation error occurred.")
    errors: List[ErrorDetailItem] = Field(
        default_factory=list,
        description="Granular list of parameter and field-level validation errors.",
    )
