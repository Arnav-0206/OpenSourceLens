"""Model registry and introspection API endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, status

from backend.app.schemas.models import ModelInfoItem, ModelRegistryResponse
from backend.app.services.model_manager import model_manager

router = APIRouter()


@router.get(
    "",
    response_model=ModelRegistryResponse,
    status_code=status.HTTP_200_OK,
    summary="List Registered Models",
    tags=["models"],
)
def list_registered_models() -> ModelRegistryResponse:
    """List all machine-learning and NLP models registered across OpenSourceLens."""
    models = model_manager.get_all_models()
    return ModelRegistryResponse(
        total_models=len(models),
        system_device=model_manager.device,
        models=models,
    )


@router.get(
    "/{model_id}",
    response_model=ModelInfoItem,
    status_code=status.HTTP_200_OK,
    summary="Get Model Details & Status",
    tags=["models"],
)
def get_model_details(model_id: str) -> ModelInfoItem:
    """Retrieve operational status, architecture, and hyperparameter metadata for a model."""
    return model_manager.get_model_info(model_id=model_id)


@router.post(
    "/prewarm",
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary="Pre-warm All Models",
    tags=["models"],
)
def prewarm_models() -> Dict[str, str]:
    """Pre-load all model weights into host memory to eliminate cold-start latency."""
    return model_manager.prewarm_all()
