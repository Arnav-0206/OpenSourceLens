"""Global exception handlers for OpenSourceLens FastAPI backend.

Intercepts HTTP errors, validation errors, database exceptions, domain exceptions,
and unhandled system exceptions to produce standardized, structured JSON responses.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.exceptions import OpenSourceLensException

logger = logging.getLogger(__name__)

STATUS_TO_ERROR_CODE = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "RESOURCE_NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    408: "REQUEST_TIMEOUT",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_SERVER_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


async def domain_exception_handler(
    request: Request, exc: OpenSourceLensException
) -> JSONResponse:
    """Handle custom application domain exceptions."""
    logger.warning(
        "Domain exception [%s] on %s: %s",
        exc.error_code,
        request.url.path,
        exc.message,
    )
    content: Dict[str, Any] = {
        "detail": exc.message,
        "error_code": exc.error_code,
        "status_code": exc.status_code,
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if exc.details:
        content["details"] = exc.details

    return JSONResponse(status_code=exc.status_code, content=content)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle standard Starlette / FastAPI HTTPExceptions with structured payload."""
    error_code = STATUS_TO_ERROR_CODE.get(exc.status_code, "HTTP_ERROR")
    detail_message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    logger.info(
        "HTTP %d [%s] on %s: %s",
        exc.status_code,
        error_code,
        request.url.path,
        detail_message,
    )
    content = {
        "detail": detail_message,
        "error_code": error_code,
        "status_code": exc.status_code,
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic RequestValidationError (HTTP 422) with granular field errors."""
    error_items: List[Dict[str, str]] = []
    for err in exc.errors():
        field_loc = ".".join(str(loc) for loc in err.get("loc", []))
        error_items.append(
            {
                "field": field_loc,
                "message": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            }
        )

    summary_msg = (
        f"Validation error: {error_items[0]['message']} at '{error_items[0]['field']}'."
        if error_items
        else "Request validation failed."
    )

    logger.warning(
        "Validation error (422) on %s: %d violations: %s",
        request.url.path,
        len(error_items),
        summary_msg,
    )

    content = {
        "detail": summary_msg,
        "error_code": "VALIDATION_ERROR",
        "status_code": 422,
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": error_items,
    }
    return JSONResponse(
        status_code=422,
        content=content,
    )


async def sqlalchemy_exception_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """Handle SQLAlchemy database exceptions safely without exposing internal details."""
    logger.exception(
        "Database exception during processing of %s: %s",
        request.url.path,
        str(exc),
    )
    content = {
        "detail": "Database operation failed. Please try again later.",
        "error_code": "DATABASE_ERROR",
        "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=content,
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all unhandled exception handler to prevent leaking stack traces."""
    logger.exception(
        "Unhandled server error on %s: %s",
        request.url.path,
        str(exc),
    )
    content = {
        "detail": "An internal server error occurred.",
        "error_code": "INTERNAL_SERVER_ERROR",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "path": request.url.path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all global exception handlers onto the FastAPI application instance."""
    app.add_exception_handler(OpenSourceLensException, domain_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
