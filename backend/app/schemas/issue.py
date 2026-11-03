"""Pydantic response models for issue queries and details."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class IssueLabelItem(BaseModel):
    """Normalized label attached to an issue."""

    model_config = ConfigDict(from_attributes=True)

    label_name: str = Field(..., description="Raw label text from GitHub")
    is_canonical: bool = Field(..., description="Whether label matches standard taxonomy")
    canonical_category: Optional[str] = Field(None, description="Mapped canonical category name")


class IssueSummary(BaseModel):
    """Compact summary of a GitHub issue for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="GitHub issue ID")
    repo_name: str = Field(..., description="owner/repo identifier")
    issue_number: int = Field(..., description="Issue number in repository")
    title: str = Field(..., description="Issue title")
    state: str = Field(..., description="open or closed")
    state_reason: Optional[str] = Field(None, description="completed, not_planned, etc.")
    author: str = Field(..., description="Author username")
    author_association: str = Field("NONE", description="MEMBER, CONTRIBUTOR, NONE, etc.")
    is_internal_author: bool = Field(False, description="True if author is collaborator or member")
    created_at: datetime = Field(..., description="Issue creation timestamp")
    closed_at: Optional[datetime] = Field(None, description="Issue closure timestamp")
    canonical_label: str = Field("other", description="Canonical classification category")
    canonical_label_id: int = Field(-1, description="Integer category ID")
    speed_tier: Optional[str] = Field(None, description="Resolution speed tier (< 1 Day, 1-7 Days, etc.)")
    comments_count: int = Field(0, description="Total comment count")
    reactions_count: int = Field(0, description="Total emoji reactions count")
    labels_count: int = Field(0, description="Attached labels count")


class IssueDetail(IssueSummary):
    """Detailed issue model with body text, NLP attributes, and resolution metrics."""

    body: Optional[str] = Field(None, description="Raw issue body description")
    clean_title: Optional[str] = Field(None, description="Cleaned title text")
    clean_body: Optional[str] = Field(None, description="Cleaned body text")
    combined_text: Optional[str] = Field(None, description="Cleaned title + body combined")
    clean_title_word_count: int = Field(0, description="Word count in cleaned title")
    clean_body_word_count: int = Field(0, description="Word count in cleaned body")
    has_code_block: bool = Field(False, description="Contains code blocks")
    code_block_count: int = Field(0, description="Number of code blocks detected")
    has_url: bool = Field(False, description="Contains URL links")
    has_stack_trace: bool = Field(False, description="Contains stack traces")
    resolution_time_hours: Optional[float] = Field(None, description="Observed resolution time in hours")
    resolution_time_days: Optional[float] = Field(None, description="Observed resolution time in days")
    resolution_type: Optional[str] = Field(None, description="Velocity classification")
    is_censored: bool = Field(False, description="True for unresolved open issues")
    is_immediate_closure: bool = Field(False, description="True for automated/bot closure")
    duration_at_risk_days: Optional[float] = Field(None, description="Observed or censored duration")
    labels: List[IssueLabelItem] = Field(default_factory=list, description="Attached labels list")


class IssueListResponse(BaseModel):
    """Paginated response container for issue queries."""

    total: int = Field(..., description="Total matching issues count")
    limit: int = Field(..., description="Requested page limit")
    offset: int = Field(..., description="Requested page offset")
    items: List[IssueSummary] = Field(..., description="Matching issue records")
