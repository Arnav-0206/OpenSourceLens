"""Core configuration settings for OpenSourceLens API."""

from __future__ import annotations

import os
from typing import List

from pydantic import BaseModel, Field

from backend.app.database.session import get_database_url


class Settings(BaseModel):
    """Application configuration settings."""

    PROJECT_NAME: str = "OpenSourceLens API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Intelligent GitHub Repository Analytics, ML/NLP Issue Classification, "
        "Semantic Search, and Resolution Velocity Advisory API."
    )
    API_V1_STR: str = "/api/v1"

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

    PREWARM_ON_STARTUP: bool = Field(
        default_factory=lambda: os.getenv("PREWARM_ON_STARTUP", "false").lower()
        in ("true", "1", "yes")
    )


settings = Settings()
