"""Resolution analysis and velocity prediction API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.resolution import (
    ResolutionPredictionRequest,
    ResolutionPredictionResponse,
    ResolutionStatsResponse,
)
from backend.app.services.resolution_service import resolution_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=ResolutionPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Advisory Resolution Velocity",
    tags=["resolution"],
)
def predict_resolution_velocity(
    request: ResolutionPredictionRequest,
    db: Session = Depends(get_db),
) -> ResolutionPredictionResponse:
    """Predict advisory issue resolution velocity tier and error bounds from filing-time metadata."""
    return resolution_service.predict_resolution(request=request, db=db)


@router.get(
    "/stats/{owner}/{name}",
    response_model=ResolutionStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Repository Resolution Statistics",
    tags=["resolution"],
)
def get_repository_resolution_stats(
    owner: str,
    name: str,
    db: Session = Depends(get_db),
) -> ResolutionStatsResponse:
    """Retrieve descriptive historical resolution percentiles and survival statistics for a repository."""
    return resolution_service.get_repository_stats(owner=owner, name=name, db=db)
