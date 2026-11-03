"""Pydantic schemas for issue classification inference requests and responses."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClassificationRequest(BaseModel):
    """Payload for issue text classification."""

    model_config = ConfigDict(from_attributes=True)

    title: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Issue title text to classify",
        examples=["Crash on startup in window manager"],
    )
    body: Optional[str] = Field(
        None,
        description="Optional issue body or reproduction steps",
        examples=["When launching the app on Windows 11, it immediately crashes with code 0xC0000005."],
    )
    repo_name: Optional[str] = Field(
        None,
        description="Target repository identifier (optional context)",
        examples=["microsoft/vscode"],
    )
    model_type: str = Field(
        "distilbert",
        description="Classification model to invoke: 'distilbert' (champion), 'svm', or 'logistic_regression'",
        pattern="^(distilbert|svm|logistic_regression)$",
    )
    issue_id: Optional[int] = Field(
        None,
        description="Optional existing GitHub issue ID for auditing linkage",
    )


class ClassificationResponse(BaseModel):
    """Response model containing predicted category and probability distribution."""

    predicted_label: str = Field(..., description="Predicted canonical category (e.g. bug, feature, other)")
    canonical_label_id: int = Field(..., description="Integer taxonomy ID for predicted label")
    confidence_score: float = Field(..., description="Softmax confidence score for predicted class [0.0, 1.0]")
    probabilities: Dict[str, float] = Field(..., description="Complete class probability distribution")
    model_used: str = Field(..., description="Model identifier executed for this inference")
    inference_time_ms: float = Field(..., description="Inference execution latency in milliseconds")
    clean_title: str = Field(..., description="Sanitized and normalized title used as input")
    clean_body: Optional[str] = Field(None, description="Sanitized and normalized body text")
    is_classified_category: bool = Field(True, description="Whether predicted class is in core taxonomy")
