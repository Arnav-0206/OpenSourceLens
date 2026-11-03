"""Pydantic response models for analytical and SLA metrics endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ResolutionMetrics(BaseModel):
    """Statistical resolution time percentiles and survival indicators."""

    median_hours: Optional[float] = Field(None, description="Median resolution duration in hours")
    mean_hours: Optional[float] = Field(None, description="Arithmetic mean resolution duration in hours")
    p25_hours: Optional[float] = Field(None, description="25th percentile (fast resolution threshold) in hours")
    p75_hours: Optional[float] = Field(None, description="75th percentile (slow resolution threshold) in hours")
    survival_rate_30d: Optional[float] = Field(None, description="Probability an issue remains open after 30 days")


class RepositoryAnalyticsResponse(BaseModel):
    """Comprehensive analytical metrics snapshot for a specific repository."""

    repo_name: str = Field(..., description="owner/repo identifier")
    total_issues: int = Field(..., description="Total analyzed issues in repository")
    open_issues: int = Field(..., description="Count of currently open/unresolved issues")
    closed_issues: int = Field(..., description="Count of resolved issues")
    closure_rate: float = Field(..., description="Fraction of resolved issues [0.0, 1.0]")
    category_distribution: Dict[str, int] = Field(default_factory=dict, description="Issues count per category")
    speed_tier_distribution: Dict[str, int] = Field(default_factory=dict, description="Issues count per speed tier")
    resolution_metrics: ResolutionMetrics = Field(..., description="Resolution duration percentiles")
    snapshot_date: datetime = Field(..., description="Timestamp of analytics snapshot computation")


class RepositoryComparisonItem(BaseModel):
    """Comparative metric row for multi-repository benchmarks."""

    repo_name: str = Field(..., description="owner/repo identifier")
    owner: str = Field(..., description="Repository owner")
    name: str = Field(..., description="Repository short name")
    stars_count: int = Field(..., description="GitHub stars count")
    total_issues: int = Field(..., description="Total issues analyzed")
    open_issues: int = Field(..., description="Open issues count")
    closed_issues: int = Field(..., description="Closed issues count")
    closure_rate: float = Field(..., description="Resolution rate [0.0, 1.0]")
    median_resolution_hours: Optional[float] = Field(None, description="Median resolution hours")
    survival_rate_30d: Optional[float] = Field(None, description="30-day survival probability")
    dominant_category: str = Field("other", description="Most frequent canonical issue category")
    rapid_closure_percentage: float = Field(..., description="Percentage of issues resolved in < 1 day")


class RepositoryComparisonResponse(BaseModel):
    """Multi-repository comparative benchmark response."""

    total_repositories: int = Field(..., description="Count of compared repositories")
    repositories: List[RepositoryComparisonItem] = Field(..., description="Comparative benchmark metrics")


class GlobalAnalyticsOverview(BaseModel):
    """Cross-repository aggregate metrics overview."""

    total_repositories: int = Field(..., description="Count of monitored repositories")
    total_issues: int = Field(..., description="Global total issues analyzed")
    total_open_issues: int = Field(..., description="Global total open issues")
    total_closed_issues: int = Field(..., description="Global total closed issues")
    global_closure_rate: float = Field(..., description="Global resolution fraction")
    category_distribution: Dict[str, int] = Field(..., description="Global category breakdown")
    speed_tier_distribution: Dict[str, int] = Field(..., description="Global speed tier breakdown")
    fastest_repository: Optional[str] = Field(None, description="Repository with lowest median resolution hours")
    slowest_repository: Optional[str] = Field(None, description="Repository with highest median resolution hours")


class TrendPoint(BaseModel):
    """Monthly time-series data point for issue volume."""

    period: str = Field(..., description="Year-Month period (YYYY-MM)")
    created_count: int = Field(..., description="Count of issues submitted in this period")
    closed_count: int = Field(..., description="Count of issues resolved in this period")


class RepositoryTrendsResponse(BaseModel):
    """Time-series monthly creation and closure trends for a repository."""

    repo_name: str = Field(..., description="owner/repo identifier")
    total_periods: int = Field(..., description="Number of monthly time intervals")
    trends: List[TrendPoint] = Field(..., description="Chronological monthly issue trends")
