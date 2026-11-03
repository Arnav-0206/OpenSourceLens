"""Repository and issue browsing API endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from backend.app.database.models import Issue, Repository, RepositoryAnalytics
from backend.app.database.session import get_db
from backend.app.schemas.issue import (
    IssueDetail,
    IssueLabelItem,
    IssueListResponse,
    IssueSummary,
)
from backend.app.schemas.repository import (
    RepositoryDetail,
    RepositorySummary,
)

router = APIRouter()


@router.get(
    "",
    response_model=List[RepositorySummary],
    summary="List Tracked Repositories",
    tags=["repositories"],
)
def list_repositories(db: Session = Depends(get_db)) -> List[RepositorySummary]:
    """Retrieve all open-source repositories analyzed by OpenSourceLens."""
    repos = db.scalars(
        select(Repository).order_by(Repository.stars_count.desc())
    ).all()
    return [RepositorySummary.model_validate(r) for r in repos]


@router.get(
    "/{owner}/{name}",
    response_model=RepositoryDetail,
    summary="Get Repository Overview",
    tags=["repositories"],
)
def get_repository(
    owner: str, name: str, db: Session = Depends(get_db)
) -> RepositoryDetail:
    """Retrieve detailed repository overview including latest resolution aggregates."""
    full_name = f"{owner}/{name}"
    repo = db.scalar(select(Repository).where(Repository.repo_name == full_name))
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{full_name}' not found.",
        )

    # Fetch latest repository analytics snapshot
    analytics = db.scalar(
        select(RepositoryAnalytics)
        .where(RepositoryAnalytics.repository_id == repo.id)
        .order_by(RepositoryAnalytics.snapshot_date.desc())
    )

    # Compute analyzed issue counts
    total_issues = db.scalar(
        select(func.count(Issue.id)).where(Issue.repository_id == repo.id)
    ) or 0
    closed_issues = db.scalar(
        select(func.count(Issue.id)).where(
            Issue.repository_id == repo.id, Issue.state == "closed"
        )
    ) or 0

    return RepositoryDetail(
        id=repo.id,
        repo_name=repo.repo_name,
        owner=repo.owner,
        name=repo.name,
        description=repo.description,
        primary_language=repo.primary_language,
        stars_count=repo.stars_count,
        forks_count=repo.forks_count,
        open_issues_count=repo.open_issues_count,
        github_created_at=repo.github_created_at,
        github_updated_at=repo.github_updated_at,
        collected_at=repo.collected_at,
        total_analyzed_issues=total_issues,
        closed_issues_count=closed_issues,
        median_resolution_hours=analytics.median_resolution_hours if analytics else None,
        category_distribution=analytics.category_distribution if analytics else None,
        speed_tier_distribution=analytics.speed_tier_distribution if analytics else None,
        snapshot_date=analytics.snapshot_date if analytics else None,
    )


@router.get(
    "/{owner}/{name}/issues",
    response_model=IssueListResponse,
    summary="List Repository Issues",
    tags=["issues"],
)
def list_repository_issues(
    owner: str,
    name: str,
    state: Optional[str] = Query(None, description="Filter by state (open or closed)"),
    category: Optional[str] = Query(None, description="Filter by canonical label (bug, enhancement, etc.)"),
    speed_tier: Optional[str] = Query(None, description="Filter by speed tier (< 1 Day, 1-7 Days, etc.)"),
    search: Optional[str] = Query(None, description="Case-insensitive title search"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: Session = Depends(get_db),
) -> IssueListResponse:
    """Query paginated issues for a repository with optional state, category, speed tier, and text filters."""
    full_name = f"{owner}/{name}"
    repo = db.scalar(select(Repository).where(Repository.repo_name == full_name))
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{full_name}' not found.",
        )

    stmt = select(Issue).where(Issue.repository_id == repo.id)

    if state:
        stmt = stmt.where(Issue.state == state.lower())
    if category:
        stmt = stmt.where(Issue.canonical_label == category.lower())
    if speed_tier:
        stmt = stmt.where(Issue.speed_tier == speed_tier)
    if search and search.strip():
        term = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                Issue.title.ilike(term),
                Issue.clean_title.ilike(term),
            )
        )

    # Get total matching count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_count = db.scalar(count_stmt) or 0

    # Paginate
    paged_stmt = stmt.order_by(Issue.created_at.desc()).offset(offset).limit(limit)
    issues = db.scalars(paged_stmt).all()

    items = [IssueSummary.model_validate(i) for i in issues]
    return IssueListResponse(
        total=total_count,
        limit=limit,
        offset=offset,
        items=items,
    )


@router.get(
    "/{owner}/{name}/issues/{number}",
    response_model=IssueDetail,
    summary="Get Single Issue Detail",
    tags=["issues"],
)
def get_issue_detail(
    owner: str,
    name: str,
    number: int,
    db: Session = Depends(get_db),
) -> IssueDetail:
    """Fetch complete issue details with text, author metadata, labels, and resolution metrics."""
    full_name = f"{owner}/{name}"
    issue = db.scalar(
        select(Issue)
        .options(selectinload(Issue.labels))
        .where(
            Issue.repo_name == full_name,
            Issue.issue_number == number,
        )
    )
    if not issue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Issue #{number} not found in repository '{full_name}'.",
        )

    label_items = [
        IssueLabelItem(
            label_name=lbl.label_name,
            is_canonical=lbl.is_canonical,
            canonical_category=lbl.canonical_category,
        )
        for lbl in issue.labels
    ]

    return IssueDetail(
        id=issue.id,
        repo_name=issue.repo_name,
        issue_number=issue.issue_number,
        title=issue.title,
        body=issue.body,
        clean_title=issue.clean_title,
        clean_body=issue.clean_body,
        combined_text=issue.combined_text,
        clean_title_word_count=issue.clean_title_word_count,
        clean_body_word_count=issue.clean_body_word_count,
        state=issue.state,
        state_reason=issue.state_reason,
        author=issue.author,
        author_association=issue.author_association,
        is_internal_author=issue.is_internal_author,
        created_at=issue.created_at,
        closed_at=issue.closed_at,
        canonical_label=issue.canonical_label,
        canonical_label_id=issue.canonical_label_id,
        speed_tier=issue.speed_tier,
        comments_count=issue.comments_count,
        reactions_count=issue.reactions_count,
        labels_count=issue.labels_count,
        has_code_block=issue.has_code_block,
        code_block_count=issue.code_block_count,
        has_url=issue.has_url,
        has_stack_trace=issue.has_stack_trace,
        resolution_time_hours=issue.resolution_time_hours,
        resolution_time_days=issue.resolution_time_days,
        resolution_type=issue.resolution_type,
        is_censored=issue.is_censored,
        is_immediate_closure=issue.is_immediate_closure,
        duration_at_risk_days=issue.duration_at_risk_days,
        labels=label_items,
    )
