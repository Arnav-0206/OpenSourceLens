"""Analytics and resolution-time SLA API endpoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Issue, Repository, RepositoryAnalytics
from backend.app.database.session import get_db
from backend.app.schemas.analytics import (
    GlobalAnalyticsOverview,
    RepositoryAnalyticsResponse,
    RepositoryComparisonItem,
    RepositoryComparisonResponse,
    RepositoryTrendsResponse,
    ResolutionMetrics,
    TrendPoint,
)

router = APIRouter()


@router.get(
    "/overview",
    response_model=GlobalAnalyticsOverview,
    summary="Global Cross-Repository Analytics Overview",
    tags=["analytics"],
)
def get_global_overview(db: Session = Depends(get_db)) -> GlobalAnalyticsOverview:
    """Retrieve global aggregated volume, resolution rates, and cross-repo category distributions."""
    repos = db.scalars(select(Repository)).all()
    analytics_list = db.scalars(select(RepositoryAnalytics)).all()

    total_issues = sum(a.total_issues for a in analytics_list)
    total_open = sum(a.open_issues for a in analytics_list)
    total_closed = sum(a.closed_issues for a in analytics_list)
    closure_rate = round(total_closed / total_issues, 4) if total_issues > 0 else 0.0

    global_categories: Dict[str, int] = defaultdict(int)
    global_tiers: Dict[str, int] = defaultdict(int)

    valid_medians = []
    for a in analytics_list:
        if a.category_distribution:
            for cat, count in a.category_distribution.items():
                global_categories[cat] += count
        if a.speed_tier_distribution:
            for tier, count in a.speed_tier_distribution.items():
                global_tiers[tier] += count
        if a.median_resolution_hours is not None:
            valid_medians.append((a.repo_name, a.median_resolution_hours))

    fastest_repo = min(valid_medians, key=lambda x: x[1])[0] if valid_medians else None
    slowest_repo = max(valid_medians, key=lambda x: x[1])[0] if valid_medians else None

    return GlobalAnalyticsOverview(
        total_repositories=len(repos),
        total_issues=total_issues,
        total_open_issues=total_open,
        total_closed_issues=total_closed,
        global_closure_rate=closure_rate,
        category_distribution=dict(global_categories),
        speed_tier_distribution=dict(global_tiers),
        fastest_repository=fastest_repo,
        slowest_repository=slowest_repo,
    )


@router.get(
    "/comparison",
    response_model=RepositoryComparisonResponse,
    summary="Multi-Repository Benchmark Comparison",
    tags=["analytics"],
)
def get_repository_comparison(db: Session = Depends(get_db)) -> RepositoryComparisonResponse:
    """Compare performance metrics, dominant categories, and velocity across all repositories."""
    repos = db.scalars(select(Repository).order_by(Repository.stars_count.desc())).all()
    comparison_items: List[RepositoryComparisonItem] = []

    for repo in repos:
        analytics = db.scalar(
            select(RepositoryAnalytics)
            .where(RepositoryAnalytics.repository_id == repo.id)
            .order_by(RepositoryAnalytics.snapshot_date.desc())
        )
        if not analytics:
            continue

        tot = analytics.total_issues
        closed = analytics.closed_issues
        rate = round(closed / tot, 4) if tot > 0 else 0.0

        dominant_cat = "other"
        if analytics.category_distribution:
            dominant_cat = max(
                analytics.category_distribution.items(), key=lambda kv: kv[1]
            )[0]

        rapid_pct = 0.0
        if analytics.speed_tier_distribution and tot > 0:
            lt_1d = analytics.speed_tier_distribution.get("< 1 Day", 0)
            rapid_pct = round((lt_1d / tot) * 100.0, 2)

        item = RepositoryComparisonItem(
            repo_name=repo.repo_name,
            owner=repo.owner,
            name=repo.name,
            stars_count=repo.stars_count,
            total_issues=tot,
            open_issues=analytics.open_issues,
            closed_issues=closed,
            closure_rate=rate,
            median_resolution_hours=analytics.median_resolution_hours,
            survival_rate_30d=analytics.survival_rate_30d,
            dominant_category=dominant_cat,
            rapid_closure_percentage=rapid_pct,
        )
        comparison_items.append(item)

    return RepositoryComparisonResponse(
        total_repositories=len(comparison_items),
        repositories=comparison_items,
    )


@router.get(
    "/{owner}/{name}",
    response_model=RepositoryAnalyticsResponse,
    summary="Single Repository Analytics Snapshot",
    tags=["analytics"],
)
def get_repository_analytics(
    owner: str, name: str, db: Session = Depends(get_db)
) -> RepositoryAnalyticsResponse:
    """Retrieve detailed resolution metrics percentiles, category breakdown, and survival probability."""
    full_name = f"{owner}/{name}"
    repo = db.scalar(select(Repository).where(Repository.repo_name == full_name))
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{full_name}' not found.",
        )

    analytics = db.scalar(
        select(RepositoryAnalytics)
        .where(RepositoryAnalytics.repository_id == repo.id)
        .order_by(RepositoryAnalytics.snapshot_date.desc())
    )
    if not analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analytics not yet computed for repository '{full_name}'.",
        )

    tot = analytics.total_issues
    closed = analytics.closed_issues
    closure_rate = round(closed / tot, 4) if tot > 0 else 0.0

    metrics = ResolutionMetrics(
        median_hours=analytics.median_resolution_hours,
        mean_hours=analytics.mean_resolution_hours,
        p25_hours=analytics.p25_resolution_hours,
        p75_hours=analytics.p75_resolution_hours,
        survival_rate_30d=analytics.survival_rate_30d,
    )

    return RepositoryAnalyticsResponse(
        repo_name=repo.repo_name,
        total_issues=tot,
        open_issues=analytics.open_issues,
        closed_issues=closed,
        closure_rate=closure_rate,
        category_distribution=analytics.category_distribution or {},
        speed_tier_distribution=analytics.speed_tier_distribution or {},
        resolution_metrics=metrics,
        snapshot_date=analytics.snapshot_date,
    )


@router.get(
    "/{owner}/{name}/trends",
    response_model=RepositoryTrendsResponse,
    summary="Repository Creation and Closure Time-Series Trends",
    tags=["analytics"],
)
def get_repository_trends(
    owner: str, name: str, db: Session = Depends(get_db)
) -> RepositoryTrendsResponse:
    """Retrieve monthly chronological submission and resolution time-series trends."""
    full_name = f"{owner}/{name}"
    repo = db.scalar(select(Repository).where(Repository.repo_name == full_name))
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{full_name}' not found.",
        )

    rows = db.execute(
        select(Issue.created_at, Issue.closed_at).where(
            Issue.repository_id == repo.id
        )
    ).all()

    created_by_period: Dict[str, int] = defaultdict(int)
    closed_by_period: Dict[str, int] = defaultdict(int)
    all_periods = set()

    for row in rows:
        created_at, closed_at = row[0], row[1]
        if created_at:
            cp = created_at.strftime("%Y-%m")
            created_by_period[cp] += 1
            all_periods.add(cp)
        if closed_at:
            clp = closed_at.strftime("%Y-%m")
            closed_by_period[clp] += 1
            all_periods.add(clp)

    sorted_periods = sorted(list(all_periods))
    trend_points = [
        TrendPoint(
            period=p,
            created_count=created_by_period.get(p, 0),
            closed_count=closed_by_period.get(p, 0),
        )
        for p in sorted_periods
    ]

    return RepositoryTrendsResponse(
        repo_name=repo.repo_name,
        total_periods=len(trend_points),
        trends=trend_points,
    )
