"""Modern Phase C Analytics V2 API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from backend.app.core.exceptions import (
    EntityNotFoundException,
    InvalidParameterException,
    RepositoryNotFoundException,
)
from backend.app.database.models import AnalysisRun, Issue, Repository
from backend.app.database.session import get_db
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
    SimilarCandidateItem,
    SimilarQueryRequest,
    SimilarQueryResponse,
    TopicDiscoveryResponse,
)
from backend.app.services.nlp.semantic_index_service import semantic_index_service
from backend.app.services.nlp.topic_discovery_service import topic_discovery_service
from backend.app.services.analytics.report_generator import report_generator
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
    if domain_key in ("release_evolution", "discussions_intelligence"):
        syn_block = normalized_payload.get("repository_synthesis")
        domain_data = syn_block.get(domain_key) if isinstance(syn_block, dict) else None
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


@router.post(
    "/{owner}/{name}/similar",
    response_model=SimilarQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Issue & Duplicate Search",
    tags=["analytics-v2", "applied-intelligence"],
)
def find_similar_issues(
    owner: str,
    name: str,
    payload: SimilarQueryRequest,
    db: Session = Depends(get_db),
) -> SimilarQueryResponse:
    """Find semantically related issues and potential duplicate candidates within a repository.

    Supports querying by natural language text or by existing issue_number.
    Uses unit-normalized dense vectors (all-MiniLM-L6-v2) and dot product cosine similarity.
    """
    repo = _resolve_repository_ci(owner=owner, name=name, db=db)

    query_text = payload.query_text
    exclude_issue_id: Optional[int] = None

    if payload.issue_number is not None:
        target_issue = db.scalar(
            select(Issue).where(
                Issue.repository_id == repo.id,
                Issue.issue_number == payload.issue_number,
            )
        )
        if not target_issue:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Issue #{payload.issue_number} not found in {owner}/{name}.",
            )
        exclude_issue_id = target_issue.id
        if not query_text:
            query_text = f"Title: {target_issue.title} | Description: {target_issue.clean_body or target_issue.body or ''}"

    if not query_text or not query_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Either query_text or a valid issue_number must be provided.",
        )

    result = semantic_index_service.query_similar(
        repository_id=repo.id,
        query_text=query_text.strip(),
        db=db,
        top_k=payload.top_k,
        threshold=payload.threshold,
        include_discussions=payload.include_discussions,
    )

    # Filter out self-match if issue_number was queried
    if exclude_issue_id is not None:
        candidates = [
            c for c in result["candidates"]
            if not (c.get("entity_type") == "issue" and c.get("id") == exclude_issue_id)
        ]
        result["candidates"] = candidates
        result["total_candidates_found"] = len(candidates)
        result["has_potential_duplicate"] = (
            len(candidates) > 0
            and candidates[0]["similarity_score"] >= 0.70
        )

    return SimilarQueryResponse(
        query=result["query"],
        total_candidates_found=result["total_candidates_found"],
        has_potential_duplicate=result["has_potential_duplicate"],
        highest_similarity_score=result["highest_similarity_score"],
        primary_recommendation=result["primary_recommendation"],
        latency_ms=result["latency_ms"],
        candidates=result["candidates"],
    )


@router.get(
    "/{owner}/{name}/topics",
    response_model=TopicDiscoveryResponse,
    status_code=status.HTTP_200_OK,
    summary="Unsupervised Topic & Theme Discovery",
    tags=["analytics-v2", "applied-intelligence"],
)
def discover_topics(
    owner: str,
    name: str,
    max_clusters: int = Query(default=8, ge=2, le=15, description="Max topic clusters to extract"),
    db: Session = Depends(get_db),
) -> TopicDiscoveryResponse:
    """Discover recurring bug/feature themes and discussion topics across a repository."""
    repo = _resolve_repository_ci(owner=owner, name=name, db=db)
    return topic_discovery_service.discover_repository_topics(
        repository_id=repo.id,
        repo_name=repo.repo_name,
        db=db,
        max_clusters=max_clusters,
    )


@router.get(
    "/{owner}/{name}/report",
    summary="Download Executive Intelligence Report",
    tags=["analytics-v2", "reporting"],
)
def export_repository_report(
    owner: str,
    name: str,
    format: str = Query(default="json", pattern="^(json|markdown)$", description="Report format: 'markdown' or 'json'"),
    methodology_version: str = Query(
        CURRENT_METHODOLOGY_VERSION,
        description="Target methodology version.",
    ),
    db: Session = Depends(get_db),
):
    """Generate and download a comprehensive repository intelligence report in Markdown or JSON format."""
    repo = _resolve_repository_ci(owner=owner, name=name, db=db)
    methodology = _validate_methodology_version(methodology_version)
    completed_run = _get_latest_completed_run(repo.id, methodology, db)
    if not completed_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No completed analytics run available for {owner}/{name}. Run recompute first.",
        )

    payload = _normalize_public_payload(completed_run.metrics_payload) or {}

    # Discovered topics (if available)
    try:
        topics_res = topic_discovery_service.discover_repository_topics(
            repository_id=repo.id,
            repo_name=repo.repo_name,
            db=db,
        )
        topics_list = [t.model_dump() for t in topics_res.topics]
    except Exception:
        topics_list = []

    if format == "markdown":
        md_text = report_generator.generate_markdown(
            repo=repo,
            run=completed_run,
            payload=payload,
            topics=topics_list,
        )
        filename = f"{owner}_{name}_intelligence_report.md"
        return Response(
            content=md_text,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    else:
        json_data = report_generator.generate_json(
            repo=repo,
            run=completed_run,
            payload=payload,
            topics=topics_list,
        )
        filename = f"{owner}_{name}_intelligence_report.json"
        return JSONResponse(
            content=json_data,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )





