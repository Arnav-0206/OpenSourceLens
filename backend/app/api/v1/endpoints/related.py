"""Semantic related-issue retrieval API endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.related import (
    RelatedIssuesResponse,
    RelatedSearchRequest,
)
from backend.app.services.related_service import related_service

router = APIRouter()


@router.get(
    "/{issue_id}",
    response_model=RelatedIssuesResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Cached Related Issues",
    tags=["related-issues"],
)
def get_cached_related_issues(
    issue_id: int,
    top_k: int = Query(5, ge=1, le=20, description="Max candidate matches to return"),
    state: Optional[str] = Query(None, description="Filter by issue state (open or closed)"),
    db: Session = Depends(get_db),
) -> RelatedIssuesResponse:
    """Retrieve precomputed top-K related issues from the relational cache for an existing issue."""
    return related_service.get_cached_related(
        issue_id=issue_id, top_k=top_k, state=state, db=db
    )


@router.post(
    "",
    response_model=RelatedIssuesResponse,
    status_code=status.HTTP_200_OK,
    summary="Dynamic Semantic Search",
    tags=["related-issues"],
)
def search_related_issues(
    request: RelatedSearchRequest,
    db: Session = Depends(get_db),
) -> RelatedIssuesResponse:
    """Search for semantically similar issues using dense sentence embeddings in real time."""
    return related_service.search_dynamic_related(request=request, db=db)
