"""Pydantic schemas for related issues retrieval and semantic similarity."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RelatedIssueMatch(BaseModel):
    """Candidate issue matching the query issue by semantic similarity."""

    model_config = ConfigDict(from_attributes=True)

    issue_id: int = Field(..., description="GitHub issue ID")
    repo_name: str = Field(..., description="owner/repo identifier")
    issue_number: int = Field(..., description="GitHub issue number")
    title: str = Field(..., description="Candidate issue title")
    state: str = Field(..., description="open or closed")
    canonical_label: str = Field("other", description="Canonical category (bug, feature, etc.)")
    similarity_score: float = Field(..., description="Cosine similarity score [-1.0, 1.0]")
    rank: int = Field(..., description="Similarity rank (1 to K)")
    created_at: datetime = Field(..., description="Issue creation timestamp")
    closed_at: Optional[datetime] = Field(None, description="Issue closure timestamp")


class RelatedIssuesResponse(BaseModel):
    """Response container for related-issue retrieval queries."""

    query_issue_id: Optional[int] = Field(None, description="Source issue ID if cached lookup")
    query_text: Optional[str] = Field(None, description="Query text string if dynamic search")
    model_version: str = Field(..., description="Embedding model identifier (e.g. all-MiniLM-L6-v2)")
    total_matches: int = Field(..., description="Number of related matches returned")
    matches: List[RelatedIssueMatch] = Field(..., description="Ranked list of related issue candidates")
    retrieval_time_ms: float = Field(..., description="Retrieval latency in milliseconds")


class RelatedSearchRequest(BaseModel):
    """Payload for dynamic semantic similarity search."""

    title: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Issue title text to search for related issues",
        examples=["Crash on startup with SIGSEGV in libtorch"],
    )
    body: Optional[str] = Field(
        None,
        description="Optional issue body text for richer semantic context",
    )
    repo_name: Optional[str] = Field(
        None,
        description="Optional repository identifier to scope the search (e.g. 'pytorch/pytorch')",
    )
    state: Optional[str] = Field(
        None,
        description="Optional filter by issue state ('open' or 'closed')",
        pattern="^(open|closed)$",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of candidate matches to return",
    )
