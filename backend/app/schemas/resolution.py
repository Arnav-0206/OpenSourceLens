"""Pydantic schemas for resolution analysis and advisory velocity prediction."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ResolutionPredictionRequest(BaseModel):
    """Payload for predicting advisory issue resolution velocity tier."""

    model_config = ConfigDict(from_attributes=True)

    repo_name: str = Field(
        ...,
        description="Target repository identifier (e.g. 'microsoft/vscode')",
        examples=["microsoft/vscode"],
    )
    title: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Issue title text",
        examples=["[UI] Crash when opening debug console on Windows"],
    )
    body: Optional[str] = Field(
        None,
        description="Issue body text or reproduction steps",
        examples=["Debugger hangs immediately when launching on Windows 11:\n```python\nimport sys\n```"],
    )
    canonical_label: Optional[str] = Field(
        "other",
        description="Canonical classification category (bug, feature, documentation, question, other)",
    )
    author_association: str = Field(
        "NONE",
        description="GitHub author relationship (MEMBER, CONTRIBUTOR, NONE, etc.)",
    )
    author: Optional[str] = Field(
        None,
        description="Author login or bot identifier",
    )
    labels_count: int = Field(
        0,
        ge=0,
        description="Number of labels attached at creation time",
    )
    issue_id: Optional[int] = Field(
        None,
        description="Optional existing issue ID for prediction auditing linkage",
    )


class ResolutionPredictionResponse(BaseModel):
    """Response model with predicted velocity bracket and empirical error bounds."""

    repo_name: str = Field(..., description="Target repository evaluated")
    predicted_speed_tier: str = Field(
        ..., description="Predicted velocity tier: < 1 Day, 1-7 Days, 7-30 Days, 30+ Days"
    )
    confidence_score: float = Field(..., description="Model confidence score for the predicted tier [0.0, 1.0]")
    tier_probabilities: Dict[str, float] = Field(
        ..., description="Complete probability distribution across all 4 velocity tiers"
    )
    lower_bound_tier: str = Field(..., description="Conservative lower velocity boundary (adjacent accuracy)")
    upper_bound_tier: str = Field(..., description="Conservative upper velocity boundary (adjacent accuracy)")
    advisory_factors: List[str] = Field(
        default_factory=list, description="Contextual features influencing resolution duration"
    )
    reliability_notice: str = Field(
        ..., description="Mandatory scientific reliability notice regarding advisory nature of prediction"
    )
    model_used: str = Field("resolution-rf-classifier-champion", description="Model identifier executed")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")


class ResolutionStatsResponse(BaseModel):
    """Factual baseline resolution statistics for an open-source repository."""

    repo_name: str = Field(..., description="owner/repo identifier")
    n_closed_issues: int = Field(..., description="Total closed issues analyzed")
    median_resolution_hours: Optional[float] = Field(None, description="Median resolution duration in hours")
    p25_resolution_hours: Optional[float] = Field(None, description="25th percentile resolution duration")
    p75_resolution_hours: Optional[float] = Field(None, description="75th percentile resolution duration")
    pct_resolved_under_24h: Optional[float] = Field(None, description="Percentage resolved in < 24 hours")
    pct_resolved_under_7d: Optional[float] = Field(None, description="Percentage resolved in < 7 days")
    survival_rate_30d: Optional[float] = Field(None, description="30-day Kaplan-Meier survival probability")
    tier_distribution: Dict[str, float] = Field(
        default_factory=dict, description="Historical distribution percentage per velocity tier"
    )
