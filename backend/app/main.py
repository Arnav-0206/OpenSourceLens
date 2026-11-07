"""FastAPI Main Application for OpenSourceLens.

Exposes REST API endpoints for repository metrics, issue classification,
semantic search, and resolution velocity analytics.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session
import uvicorn

from backend.app.api.v1.api import api_router
from backend.app.core.config import settings
from backend.app.core.error_handlers import register_error_handlers
from backend.app.database.init_db import check_connection, get_table_status
from backend.app.database.models import Repository
from backend.app.database.session import engine, get_db, init_db
from backend.app.schemas.health import (
    DatabaseHealth,
    HealthResponse,
    RootResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: verifies database readiness on startup."""
    logger.info("Initializing OpenSourceLens API v%s...", settings.VERSION)
    conn_status = check_connection(engine)
    logger.info(
        "Database status: %s (dialect: %s, latency: %.2f ms)",
        conn_status["status"],
        conn_status["dialect"],
        conn_status["latency_ms"],
    )
    # Ensure tables are present in database
    init_db(engine)

    # Optional model pre-warming on application boot
    if settings.PREWARM_ON_STARTUP:
        from backend.app.services.model_manager import model_manager

        logger.info("PREWARM_ON_STARTUP enabled: Pre-loading ML/DL models into host memory...")
        model_manager.prewarm_all()
        logger.info("All model artifacts pre-warmed successfully.")

    yield
    logger.info("Shutting down OpenSourceLens API...")
    engine.dispose()


def create_application() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url=settings.DOCS_URL,
        openapi_url=settings.OPENAPI_URL,
        redoc_url=settings.REDOC_URL,
        lifespan=lifespan,
    )

    # Configure CORS for Next.js frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 Router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Register centralized exception handlers
    register_error_handlers(app)

    return app


app = create_application()


@app.get(
    "/",
    response_model=RootResponse,
    tags=["system"],
    summary="Root Discovery & Project Info",
)
def read_root(db: Session = Depends(get_db)) -> RootResponse:
    """Return project overview metadata and supported target repositories."""
    repos = db.scalars(select(Repository.repo_name)).all()
    default_repos = [
        "microsoft/vscode",
        "pytorch/pytorch",
        "kubernetes/kubernetes",
        "huggingface/transformers",
    ]
    return RootResponse(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url=settings.DOCS_URL,
        api_v1_url=settings.API_V1_STR,
        supported_repositories=list(repos) if repos else default_repos,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health Check Probe",
    status_code=status.HTTP_200_OK,
)
def read_health(db: Session = Depends(get_db)) -> HealthResponse:
    """Evaluate live database connectivity and system operational status."""
    conn_info = check_connection(db.get_bind())
    status_info = get_table_status(db.get_bind(), include_details=False)

    is_healthy = conn_info["status"] == "connected"
    db_health = DatabaseHealth(
        status=conn_info["status"],
        dialect=conn_info["dialect"],
        latency_ms=conn_info["latency_ms"],
        tables_count=len(status_info["existing_tables"]),
        error=conn_info["error"],
    )

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc),
        database=db_health,
    )


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
