"""Pydantic schemas for healthcheck and root discovery endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatabaseHealth(BaseModel):
    """Database connectivity status model."""

    status: str = Field(..., description="connected or unreachable")
    dialect: str = Field(..., description="Target database dialect name")
    latency_ms: float = Field(..., description="Ping latency in milliseconds")
    tables_count: int = Field(..., description="Count of initialized tables")
    error: Optional[str] = Field(None, description="Error message if unreachable")


class HealthResponse(BaseModel):
    """System health check response model."""

    status: str = Field(..., description="Overall system status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="API semantic version")
    timestamp: datetime = Field(..., description="Health check evaluation timestamp")
    database: DatabaseHealth = Field(..., description="Database health status")
    queue_depths: Optional[Dict[str, Any]] = Field(None, description="Pending and running queue counts")
    active_workers: Optional[Dict[str, Any]] = Field(None, description="Worker heartbeat telemetry")
    model_status: Optional[Dict[str, Any]] = Field(None, description="Machine learning model readiness")


class RootResponse(BaseModel):
    """Root discovery and project metadata response."""

    title: str = Field(..., description="Project title")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="Project summary")
    docs_url: str = Field(..., description="Interactive OpenAPI documentation URL")
    api_v1_url: str = Field(..., description="Base API v1 endpoint prefix")
    supported_repositories: List[str] = Field(
        ..., description="Canonical open-source repositories analyzed"
    )
