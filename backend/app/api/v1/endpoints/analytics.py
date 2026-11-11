"""Analytics and resolution-time SLA API endpoints (Legacy Phase A + Modern Phase C V2)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.core.exceptions import (
    EntityNotFoundException,
    InvalidParameterException,
    RepositoryNotFoundException,
)
from backend.app.database.models import AnalysisRun, Issue, Repository, RepositoryAnalytics
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
from backend.app.schemas.analytics_v2 import (
    AnalysisDomainSliceResponse,
    AnalysisHistoryResponse,
    AnalysisLatestResponse,
    AnalysisPayload,
    AnalysisRecomputeRequest,
    AnalysisRecomputeResponse,
    AnalysisRunAttemptRead,
    AnalysisRunDetailResponse,
    AnalysisRunRead,
    AnalysisStatusResponse,
    CURRENT_METHODOLOGY_VERSION,
    RepositoryFreshnessEnvelope,
)
from backend.app.services.analytics.orchestrator import ALL_ANALYTICS_DOMAINS
from backend.app.services.analytics_queue import AnalyticsJobQueue

router = APIRouter()
analytics_queue = AnalyticsJobQueue()

_VALID_RUN_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}
_URI_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_WIN_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s'\";]+")
_UNIX_PATH_PATTERN = re.compile(r"(?:(?<=^)|(?<=[\s(\"'=]))/(?:home|var|tmp|opt|srv|etc|Users|app|usr)/[^\s'\";)]+")


def _sanitize_public_error_message(raw_message: Optional[str]) -> Optional[str]:
    """Strip stack frames, local paths, and connection URIs from public API diagnostics."""
    if not raw_message:
        return raw_message
    text_val = raw_message.strip()
    if "Traceback (most recent call last):" in text_val:
        text_val = text_val.split("Traceback (most recent call last):", 1)[0].strip()
    first_line = text_val.splitlines()[0].strip() if text_val else ""
    redacted = _URI_PATTERN.sub("[REDACTED_URI]", first_line)
    redacted = _WIN_PATH_PATTERN.sub("[REDACTED_PATH]", redacted)
    redacted = _UNIX_PATH_PATTERN.sub("[REDACTED_PATH]", redacted)
    if len(redacted) > 300:
        return f"{redacted[:297]}..."
    return redacted or None


def _resolve_repository_ci(owner: str, name: str, db: Session) -> Repository:
    """Resolve repository by owner/name exact match first, falling back to case-insensitive match."""
    full_name = f"{owner.strip()}/{name.strip()}"
    repo = db.scalar(select(Repository).where(Repository.repo_name == full_name))
    if repo is None:
        repo = db.scalar(
            select(Repository)
            .where(func.lower(Repository.repo_name) == full_name.lower())
            .limit(1)
        )
    if repo is None:
        raise RepositoryNotFoundException(full_name)
    return repo


def _build_freshness_envelope(repo: Repository) -> RepositoryFreshnessEnvelope:
    """Build outer repository freshness metadata (keeping last_synced_at out of metrics_payload)."""
    return RepositoryFreshnessEnvelope(
        repository_id=repo.id,
        repo_name=repo.repo_name,
        analytics_source_revision=int(repo.analytics_source_revision or 1),
        last_synced_at=repo.last_synced_at,
        last_analyzed_at=repo.last_analyzed_at,
        primary_language=repo.primary_language,
        stars_count=int(repo.stars_count or 0),
        forks_count=int(repo.forks_count or 0),
        is_archived=bool(repo.is_archived),
    )


def _normalize_public_payload(raw_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Enforce the exact 5-key AnalysisPayload contract without fabricating missing metrics."""
    if raw_payload is None:
        return None
    validated = AnalysisPayload(
        development_activity=raw_payload.get("development_activity"),
        contributor_ecosystem=raw_payload.get("contributor_ecosystem"),
        pull_request_intelligence=raw_payload.get("pull_request_intelligence"),
        issue_intelligence=raw_payload.get("issue_intelligence"),
        repository_synthesis=raw_payload.get("repository_synthesis"),
    )
    return validated.model_dump()


def _serialize_analysis_run(run: AnalysisRun) -> AnalysisRunRead:
    """Serialize an AnalysisRun and sorted attempts safely without mutating ORM state."""
    sorted_attempts = sorted(
        list(run.attempts or []),
        key=lambda att: (att.attempt_number, att.id),
    )
    attempt_reads = [
        AnalysisRunAttemptRead(
            id=att.id,
            analysis_run_id=att.analysis_run_id,
            attempt_number=att.attempt_number,
            worker_id=att.worker_id,
            outcome=att.outcome,
            error_category=att.error_category,
            error_message=_sanitize_public_error_message(att.error_message),
            interruption_reason=att.interruption_reason,
            started_at=att.started_at,
            completed_at=att.completed_at,
        )
        for att in sorted_attempts
    ]

    return AnalysisRunRead(
        id=run.id,
        repository_id=run.repository_id,
        collection_run_id=run.collection_run_id,
        status=run.status,
        requested_source_revision=run.requested_source_revision,
        analytics_source_revision=run.analytics_source_revision,
        requested_methodology_version=run.requested_methodology_version,
        methodology_version=run.methodology_version,
        analysis_reference_time=run.analysis_reference_time,
        source_content_digest=run.source_content_digest,
        worker_id=run.worker_id,
        lease_version=run.lease_version,
        retry_count=run.retry_count,
        max_retries=run.max_retries,
        metrics_payload=_normalize_public_payload(run.metrics_payload),
        coverage_summary=run.coverage_summary,
        confidence_summary=run.confidence_summary,
        rule_interpretations=run.rule_interpretations,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=_sanitize_public_error_message(run.error_message),
        execution_time_ms=float(run.execution_time_ms or 0.0),
        attempts=attempt_reads,
    )


def _derive_failure_category(run: Optional[AnalysisRun], ordered_attempts: List[AnalysisRunAttemptRead]) -> Optional[str]:
    """Derive explicit run-level failure category from attempt telemetry and terminal status."""
    if run is None or run.status not in ("failed", "cancelled"):
        return None
    latest_attempt = ordered_attempts[-1] if ordered_attempts else None
    if run.status == "cancelled":
        return (latest_attempt.error_category if latest_attempt and latest_attempt.error_category else "stale_revision")
    if latest_attempt and latest_attempt.error_category:
        if latest_attempt.error_category == "retryable" and run.retry_count >= run.max_retries:
            return "exhausted_retries"
        return latest_attempt.error_category
    raw_msg = run.error_message or ""
    if "[NON_RETRYABLE]" in raw_msg:
        return "non_retryable"
    if "[RETRYABLE]" in raw_msg:
        return "exhausted_retries"
    if "[STALE_REVISION]" in raw_msg:
        return "stale_revision"
    return "failed"


def _validate_methodology_version(methodology_version: str) -> str:
    cleaned = (methodology_version or "").strip()
    if not cleaned:
        raise InvalidParameterException(
            "methodology_version must not be empty.",
            parameter="methodology_version",
        )
    if len(cleaned) > 50:
        raise InvalidParameterException(
            "methodology_version must not exceed 50 characters.",
            parameter="methodology_version",
        )
    return cleaned


def _get_latest_completed_run(
    repo_id: int,
    methodology_version: str,
    db: Session,
) -> Optional[AnalysisRun]:
    return db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(
            AnalysisRun.repository_id == repo_id,
            AnalysisRun.status == "completed",
            AnalysisRun.methodology_version == methodology_version,
        )
        .order_by(
            AnalysisRun.analytics_source_revision.desc(),
            AnalysisRun.completed_at.desc(),
            AnalysisRun.id.desc(),
        )
        .limit(1)
    )


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


# ---------------------------------------------------------------------------
# Phase C V2 Multi-Domain Analytics Endpoints (Static/Subpath Before Dynamic)
# ---------------------------------------------------------------------------


@router.get(
    "/runs/{run_id}",
    response_model=AnalysisRunDetailResponse,
    summary="Global Lookup for Analysis Run by ID",
    tags=["analytics-v2"],
)
def get_analysis_run_global(
    run_id: int,
    db: Session = Depends(get_db),
) -> AnalysisRunDetailResponse:
    """Retrieve any historical or active AnalysisRun by global ID with repository freshness metadata."""
    run = db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(AnalysisRun.id == run_id)
    )
    if run is None:
        raise EntityNotFoundException(
            message=f"Analysis run {run_id} not found.",
            entity_type="AnalysisRun",
            entity_id=run_id,
        )

    repo = db.scalar(select(Repository).where(Repository.id == run.repository_id))
    if repo is None:
        raise RepositoryNotFoundException(str(run.repository_id))

    current_rev = int(repo.analytics_source_revision or 1)
    is_stale = (
        run.status == "completed"
        and (run.analytics_source_revision or 0) < current_rev
    )
    return AnalysisRunDetailResponse(
        repository=_build_freshness_envelope(repo),
        is_stale=is_stale,
        run=_serialize_analysis_run(run),
    )


@router.post(
    "/{owner}/{name}/recompute",
    response_model=AnalysisRecomputeResponse,
    summary="Enqueue or Reuse Multi-Domain Repository Analytics",
    tags=["analytics-v2"],
)
def recompute_repository_analytics(
    owner: str,
    name: str,
    response: Response,
    payload: Optional[AnalysisRecomputeRequest] = Body(default=None),
    db: Session = Depends(get_db),
) -> AnalysisRecomputeResponse:
    """Enqueue an asynchronous analytics job or return an active/up-to-date cached run."""
    repo = _resolve_repository_ci(owner, name, db)
    req = payload or AnalysisRecomputeRequest()
    methodology = _validate_methodology_version(req.requested_methodology_version)

    run, is_new = analytics_queue.enqueue_analysis(
        repo_id=repo.id,
        db=db,
        requested_methodology=methodology,
        force=req.force,
    )
    loaded_run = db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(AnalysisRun.id == run.id)
    ) or run

    if is_new:
        response.status_code = status.HTTP_202_ACCEPTED
        enqueued = True
        already_active = False
        cached = False
        message = f"Analytics run {loaded_run.id} enqueued for '{repo.repo_name}' at revision {loaded_run.requested_source_revision}."
    elif loaded_run.status in ("pending", "running"):
        response.status_code = status.HTTP_202_ACCEPTED
        enqueued = False
        already_active = True
        cached = False
        message = f"An active analytics run (ID={loaded_run.id}) is already {loaded_run.status} for '{repo.repo_name}'."
    else:
        response.status_code = status.HTTP_200_OK
        enqueued = False
        already_active = False
        cached = True
        message = f"Returning cached completed analytics run (ID={loaded_run.id}) for '{repo.repo_name}' revision {loaded_run.analytics_source_revision}."

    return AnalysisRecomputeResponse(
        repository=_build_freshness_envelope(repo),
        run=_serialize_analysis_run(loaded_run),
        enqueued=enqueued,
        already_active=already_active,
        cached=cached,
        message=message,
    )


@router.get(
    "/{owner}/{name}/status",
    response_model=AnalysisStatusResponse,
    summary="Poll Repository Analysis Execution Status & Staleness",
    tags=["analytics-v2"],
)
def get_repository_analysis_status(
    owner: str,
    name: str,
    methodology_version: str = Query(
        CURRENT_METHODOLOGY_VERSION,
        description="Methodology version used to evaluate completed staleness.",
    ),
    db: Session = Depends(get_db),
) -> AnalysisStatusResponse:
    """Return repository source revision, active/latest run telemetry, retry state, and attempt audit history."""
    repo = _resolve_repository_ci(owner, name, db)
    methodology = _validate_methodology_version(methodology_version)

    active_run = db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(
            AnalysisRun.repository_id == repo.id,
            AnalysisRun.status.in_(["pending", "running"]),
        )
        .order_by(AnalysisRun.started_at.desc(), AnalysisRun.id.desc())
        .limit(1)
    )
    latest_any_run = active_run or db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(AnalysisRun.repository_id == repo.id)
        .order_by(AnalysisRun.started_at.desc(), AnalysisRun.id.desc())
        .limit(1)
    )
    latest_completed = _get_latest_completed_run(repo.id, methodology, db)

    current_rev = int(repo.analytics_source_revision or 1)
    is_stale = (
        latest_completed is not None
        and (latest_completed.analytics_source_revision or 0) < current_rev
    )

    if latest_any_run is None:
        return AnalysisStatusResponse(
            repository=_build_freshness_envelope(repo),
            has_analysis=False,
            status="missing",
            is_stale=False,
            retry_count=0,
            max_retries=analytics_queue.MAX_ALLOWED_RETRIES,
            failure_category=None,
            current_run=None,
            attempts=[],
            latest_completed_run_id=latest_completed.id if latest_completed else None,
        )

    serialized = _serialize_analysis_run(latest_any_run)
    failure_cat = _derive_failure_category(latest_any_run, serialized.attempts)
    return AnalysisStatusResponse(
        repository=_build_freshness_envelope(repo),
        has_analysis=True,
        status=serialized.status,
        is_stale=is_stale,
        retry_count=serialized.retry_count,
        max_retries=serialized.max_retries,
        failure_category=failure_cat,
        current_run=serialized,
        attempts=serialized.attempts,
        latest_completed_run_id=latest_completed.id if latest_completed else None,
    )


@router.get(
    "/{owner}/{name}/latest",
    response_model=AnalysisLatestResponse,
    summary="Get Latest Completed Multi-Domain Analytics Report",
    tags=["analytics-v2"],
)
def get_latest_repository_analytics(
    owner: str,
    name: str,
    methodology_version: str = Query(
        CURRENT_METHODOLOGY_VERSION,
        description="Target methodology version.",
    ),
    db: Session = Depends(get_db),
) -> AnalysisLatestResponse:
    """Return latest completed five-key AnalysisPayload with repository freshness and staleness metadata."""
    repo = _resolve_repository_ci(owner, name, db)
    methodology = _validate_methodology_version(methodology_version)

    completed_run = _get_latest_completed_run(repo.id, methodology, db)
    if completed_run is None:
        raise EntityNotFoundException(
            message=f"No completed analytics run found for repository '{repo.repo_name}' (methodology={methodology}).",
            entity_type="AnalysisRun",
            entity_id=repo.repo_name,
        )

    active_run = db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.repository_id == repo.id,
            AnalysisRun.status.in_(["pending", "running"]),
        )
        .order_by(AnalysisRun.started_at.desc(), AnalysisRun.id.desc())
        .limit(1)
    )
    current_rev = int(repo.analytics_source_revision or 1)
    is_stale = (completed_run.analytics_source_revision or 0) < current_rev

    return AnalysisLatestResponse(
        repository=_build_freshness_envelope(repo),
        is_stale=is_stale,
        active_run_id=active_run.id if active_run else None,
        active_run_status=active_run.status if active_run else None,
        run=_serialize_analysis_run(completed_run),
    )


@router.get(
    "/{owner}/{name}/history",
    response_model=AnalysisHistoryResponse,
    summary="List Historical Repository Analysis Runs",
    tags=["analytics-v2"],
)
def get_repository_analytics_history(
    owner: str,
    name: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum runs per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    run_status: Optional[str] = Query(
        None,
        alias="status",
        description="Optional status filter ('pending', 'running', 'completed', 'failed', 'cancelled')",
    ),
    db: Session = Depends(get_db),
) -> AnalysisHistoryResponse:
    """Return deterministically ordered historical AnalysisRuns for a repository."""
    repo = _resolve_repository_ci(owner, name, db)

    stmt = (
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(AnalysisRun.repository_id == repo.id)
    )
    count_stmt = select(func.count(AnalysisRun.id)).where(AnalysisRun.repository_id == repo.id)

    if run_status is not None:
        normalized_status = run_status.strip().lower()
        if normalized_status not in _VALID_RUN_STATUSES:
            raise InvalidParameterException(
                f"Invalid run status filter '{run_status}'. Expected one of {sorted(_VALID_RUN_STATUSES)}.",
                parameter="status",
            )
        stmt = stmt.where(AnalysisRun.status == normalized_status)
        count_stmt = count_stmt.where(AnalysisRun.status == normalized_status)

    total = int(db.scalar(count_stmt) or 0)
    runs = db.scalars(
        stmt.order_by(AnalysisRun.started_at.desc(), AnalysisRun.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    return AnalysisHistoryResponse(
        repository=_build_freshness_envelope(repo),
        total=total,
        limit=limit,
        offset=offset,
        items=[_serialize_analysis_run(r) for r in runs],
    )


@router.get(
    "/{owner}/{name}/runs/{run_id}",
    response_model=AnalysisRunDetailResponse,
    summary="Get Repository-Scoped Analysis Run by ID",
    tags=["analytics-v2"],
)
def get_repository_analysis_run_by_id(
    owner: str,
    name: str,
    run_id: int,
    db: Session = Depends(get_db),
) -> AnalysisRunDetailResponse:
    """Retrieve an AnalysisRun strictly scoped to the specified repository."""
    repo = _resolve_repository_ci(owner, name, db)
    run = db.scalar(
        select(AnalysisRun)
        .options(selectinload(AnalysisRun.attempts))
        .where(
            AnalysisRun.id == run_id,
            AnalysisRun.repository_id == repo.id,
        )
    )
    if run is None:
        raise EntityNotFoundException(
            message=f"Analysis run {run_id} not found for repository '{repo.repo_name}'.",
            entity_type="AnalysisRun",
            entity_id=run_id,
        )

    current_rev = int(repo.analytics_source_revision or 1)
    is_stale = (
        run.status == "completed"
        and (run.analytics_source_revision or 0) < current_rev
    )
    return AnalysisRunDetailResponse(
        repository=_build_freshness_envelope(repo),
        is_stale=is_stale,
        run=_serialize_analysis_run(run),
    )


@router.get(
    "/{owner}/{name}/domains/{domain_key}",
    response_model=AnalysisDomainSliceResponse,
    summary="Get Single-Domain Analytics Slice",
    tags=["analytics-v2"],
)
def get_repository_analysis_domain_slice(
    owner: str,
    name: str,
    domain_key: str,
    methodology_version: str = Query(
        CURRENT_METHODOLOGY_VERSION,
        description="Target methodology version.",
    ),
    db: Session = Depends(get_db),
) -> AnalysisDomainSliceResponse:
    """Return metrics, coverage, confidence breakdown, and rules for one analytics domain."""
    if domain_key not in ALL_ANALYTICS_DOMAINS:
        raise InvalidParameterException(
            f"Unknown analytics domain '{domain_key}'. Must be one of: {list(ALL_ANALYTICS_DOMAINS)}.",
            parameter="domain_key",
        )
    methodology = _validate_methodology_version(methodology_version)
    repo = _resolve_repository_ci(owner, name, db)
    completed_run = _get_latest_completed_run(repo.id, methodology, db)
    if completed_run is None:
        raise EntityNotFoundException(
            message=f"No completed analytics run found for repository '{repo.repo_name}' (methodology={methodology}).",
            entity_type="AnalysisRun",
            entity_id=repo.repo_name,
        )

    normalized_payload = _normalize_public_payload(completed_run.metrics_payload) or {}
    if domain_key == "release_evolution":
        syn_block = normalized_payload.get("repository_synthesis")
        domain_data = syn_block.get("release_evolution") if isinstance(syn_block, dict) else None
    else:
        domain_data = normalized_payload.get(domain_key)

    cov_summary = completed_run.coverage_summary if isinstance(completed_run.coverage_summary, dict) else {}
    conf_summary = completed_run.confidence_summary if isinstance(completed_run.confidence_summary, dict) else {}
    domain_breakdown = conf_summary.get("domain_breakdown") if isinstance(conf_summary.get("domain_breakdown"), dict) else {}
    domain_rules = [
        rule
        for rule in (completed_run.rule_interpretations or [])
        if isinstance(rule, dict) and rule.get("domain") == domain_key
    ]

    current_rev = int(repo.analytics_source_revision or 1)
    is_stale = (completed_run.analytics_source_revision or 0) < current_rev

    return AnalysisDomainSliceResponse(
        repository=_build_freshness_envelope(repo),
        run_id=completed_run.id,
        domain=domain_key,
        analytics_source_revision=completed_run.analytics_source_revision,
        methodology_version=completed_run.methodology_version,
        analysis_reference_time=completed_run.analysis_reference_time,
        is_stale=is_stale,
        data=domain_data,
        coverage=cov_summary.get(domain_key),
        confidence_breakdown=domain_breakdown.get(domain_key),
        rules=domain_rules,
    )


# ---------------------------------------------------------------------------
# Legacy Phase A Analytics Endpoints (Preserved for Backward Compatibility)
# ---------------------------------------------------------------------------


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
