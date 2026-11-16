"""Core configuration settings for OpenSourceLens API."""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.app.database.session import get_database_url

# Automatically load environment variables from .env if present
load_dotenv()


class Settings(BaseModel):
    """Application configuration settings."""

    PROJECT_NAME: str = "OpenSourceLens API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Intelligent GitHub Repository Analytics, ML/NLP Issue Classification, "
        "Semantic Search, and Resolution Velocity Advisory API."
    )
    API_V1_STR: str = "/api/v1"

    GITHUB_TOKEN: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))

    # CORS configuration
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "BACKEND_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000",
            ).split(",")
            if origin.strip()
        ]
    )

    DATABASE_URL: str = Field(default_factory=get_database_url)

    DOCS_URL: str = "/docs"
    OPENAPI_URL: str = "/openapi.json"
    REDOC_URL: str = "/redoc"

    FASTAPI_HOST: str = Field(default_factory=lambda: os.getenv("FASTAPI_HOST", "0.0.0.0"))
    FASTAPI_PORT: int = Field(default_factory=lambda: int(os.getenv("FASTAPI_PORT", "8000")))

    PREWARM_ON_STARTUP: bool = Field(
        default_factory=lambda: os.getenv("PREWARM_ON_STARTUP", "false").lower()
        in ("true", "1", "yes")
    )
    ENABLE_ANALYTICS_WORKER: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_ANALYTICS_WORKER", "true").lower()
        in ("true", "1", "yes")
    )
    ANALYTICS_WORKER_POLL_INTERVAL_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("ANALYTICS_WORKER_POLL_INTERVAL_SECONDS", "1.0"))
    )
    ANALYTICS_WORKER_RECONCILE_INTERVAL_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("ANALYTICS_WORKER_RECONCILE_INTERVAL_SECONDS", "60.0"))
    )
    ENABLE_INGESTION_WORKER: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_INGESTION_WORKER", "true").lower()
        in ("true", "1", "yes")
    )
    INGESTION_WORKER_POLL_INTERVAL_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("INGESTION_WORKER_POLL_INTERVAL_SECONDS", "1.0"))
    )


settings = Settings()
