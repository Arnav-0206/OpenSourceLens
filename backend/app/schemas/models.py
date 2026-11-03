"""Pydantic schemas for model registry and health introspection endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelInfoItem(BaseModel):
    """Detailed metadata and operational state of a registered model."""

    model_config = ConfigDict(from_attributes=True)

    model_id: str = Field(..., description="Unique slug identifier (e.g. 'distilbert', 'svm')")
    name: str = Field(..., description="Human-readable model title")
    task_type: str = Field(..., description="Task category: classification, semantic_retrieval, resolution_prediction")
    version: str = Field(..., description="Semantic or training artifact version string")
    artifact_path: str = Field(..., description="Filesystem path where weights/checkpoints reside")
    is_loaded: bool = Field(..., description="True if model weights are currently resident in memory")
    device: str = Field("cpu", description="Computing device: 'cpu' or 'cuda'")
    details: Dict[str, Any] = Field(default_factory=dict, description="Model-specific hyperparams or class labels")
    status: str = Field("ready", description="Model health status: 'ready', 'loading', or 'error'")


class ModelRegistryResponse(BaseModel):
    """Container listing all active ML/NLP models registered in OpenSourceLens."""

    total_models: int = Field(..., description="Count of registered models")
    system_device: str = Field("cpu", description="Host computing device detected")
    models: List[ModelInfoItem] = Field(..., description="List of registered model descriptors")
