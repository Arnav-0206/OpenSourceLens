"""API v1 master router for OpenSourceLens.

Aggregates analytics, classification, ingestion, and model registry sub-routers.
"""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import (
    analytics,
    classify,
    ingest,
    models,
)

api_router = APIRouter()

# Mount API routers
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(classify.router, prefix="/classify", tags=["classification"])
api_router.include_router(models.router, prefix="/models", tags=["models"])


@api_router.get("/ping", tags=["system"])
def ping_api_v1():
    """Lightweight ping probe for API v1."""
    return {"message": "OpenSourceLens API v1 is active", "status": "ok"}
