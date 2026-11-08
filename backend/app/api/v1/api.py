"""API v1 master router for OpenSourceLens.

Aggregates repository, analytics, classification, related issue, resolution,
and model registry sub-routers.
"""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import (
    analytics,
    classify,
    ingest,
    models,
    related,
    repos,
    resolution,
)

api_router = APIRouter()

# Mount API routers
api_router.include_router(repos.router, prefix="/repos", tags=["repositories"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(classify.router, prefix="/classify", tags=["classification"])
api_router.include_router(related.router, prefix="/related", tags=["related-issues"])
api_router.include_router(resolution.router, prefix="/resolution", tags=["resolution"])
api_router.include_router(models.router, prefix="/models", tags=["models"])


@api_router.get("/ping", tags=["system"])
def ping_api_v1():
    """Lightweight ping probe for API v1."""
    return {"message": "OpenSourceLens API v1 is active", "status": "ok"}
