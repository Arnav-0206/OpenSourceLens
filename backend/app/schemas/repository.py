"""Pydantic response models for repository endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class RepositorySummary(BaseModel):
    """High-level summary of an analyzed repository."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="GitHub repository ID")
    repo_name: str = Field(..., description="owner/repo identifier")
    owner: str = Field(..., description="Repository owner login")
    name: str = Field(..., description="Repository short name")
    description: Optional[str] = Field(None, description="Project summary description")
    primary_language: Optional[str] = Field(None, description="Primary programming language")
    stars_count: int = Field(..., description="GitHub stargazer count")
    forks_count: int = Field(..., description="GitHub forks count")
    open_issues_count: int = Field(..., description="Open issues count reported by GitHub")
    github_created_at: Optional[datetime] = Field(None, description="GitHub repository creation timestamp")
    github_updated_at: Optional[datetime] = Field(None, description="GitHub repository last update timestamp")
    collected_at: datetime = Field(..., description="Timestamp when repository data was collected")


class RepositoryDetail(RepositorySummary):
    """Detailed repository overview with analytical aggregates."""

    total_analyzed_issues: int = Field(0, description="Total issues analyzed in OpenSourceLens")
    closed_issues_count: int = Field(0, description="Count of resolved issues")
    median_resolution_hours: Optional[float] = Field(None, description="Median resolution time in hours")
    category_distribution: Optional[Dict[str, int]] = Field(None, description="Issue count per canonical category")
    speed_tier_distribution: Optional[Dict[str, int]] = Field(None, description="Issue count per velocity speed tier")
    snapshot_date: Optional[datetime] = Field(None, description="Date of latest analytical snapshot")
