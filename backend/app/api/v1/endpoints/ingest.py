"""Repository ingestion API endpoints.

Exposes pre-flight eligibility preview, durable job queue submission, and
real-time collection telemetry endpoints for any eligible public GitHub repository.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    GitHubAPIError,
    GitHubRateLimitExceededException,
    GitHubResourceNotFoundException,
    InvalidRepositoryURLException,
    RepositoryIneligibleException,
)
from backend.app.database.models import CollectionRun, Repository
from backend.app.database.session import get_db
from backend.app.schemas.ingestion import (
    IngestionPreviewRequest,
    IngestionPreviewResponse,
    IngestionStatusResponse,
    IngestionSubmitRequest,
    IngestionSubmitResponse,
)
from backend.app.services.job_queue import DurableJobQueue
from backend.app.services.rate_budgeter import BUDGET_CONFIGS
from backend.app.services.repository_eligibility import RepositoryEligibilityChecker
from backend.app.services.validator import GitHubURLValidator

logger = logging.getLogger(__name__)

router = APIRouter()
job_queue = DurableJobQueue()


@router.post(
    "/preview",
    response_model=IngestionPreviewResponse,
    summary="Preview Repository Ingestion & Budget",
    tags=["ingestion"],
)
async def preview_repository(
    payload: IngestionPreviewRequest,
) -> IngestionPreviewResponse:
    """Pre-flight evaluation of repository eligibility, analytical confidence, and request budgets.

    Guarantees non-destructive, mutation-free execution: does not write to the database
    or create ingestion jobs.
    """
    try:
        checker = RepositoryEligibilityChecker()
        result = await checker.check_eligibility(payload.repo)

        metadata_envelope = {
            "stars_count": result.stars_count,
            "forks_count": result.forks_count,
            "open_issues_count": result.open_issues_count,
            "commits_count": result.commits_count,
            "default_branch": result.default_branch,
            "is_private": result.is_private,
            "is_archived": result.is_archived,
            "is_fork": result.is_fork,
            "upstream_repo": result.upstream_repo,
            "is_template": result.is_template,
            "is_empty": result.is_empty,
        }

        return IngestionPreviewResponse(
            repo_name=result.repo_name,
            owner=result.owner,
            name=result.name,
            canonical_url=result.canonical_url,
            is_eligible=result.is_eligible,
            rejection_reason=result.rejection_reason,
            metadata=metadata_envelope,
            estimated_budgets=BUDGET_CONFIGS,
            confidence_preview=result.confidence_report,
            badges=result.badges,
        )

    except InvalidRepositoryURLException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except GitHubResourceNotFoundException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except GitHubRateLimitExceededException as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        )
    except GitHubAPIError as exc:
        raise HTTPException(
            status_code=exc.status_code if 400 <= exc.status_code < 600 else 502,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error previewing repository '%s': %s", payload.repo, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error evaluating repository: {str(exc)}",
        )


@router.post(
    "/submit",
    response_model=IngestionSubmitResponse,
    summary="Submit Repository for Ingestion",
    tags=["ingestion"],
)
async def submit_ingestion_job(
    payload: IngestionSubmitRequest,
    response: Response,
    db: Session = Depends(get_db),
) -> IngestionSubmitResponse:
    """Validate repository and enqueue a durable background collection job."""
    # 1. Validate collection mode
    allowed_modes = {"fast_snapshot", "standard_diagnostic"}
    if payload.mode not in allowed_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid collection mode '{payload.mode}'. Must be one of: {sorted(allowed_modes)}",
        )

    # 2. Check repository eligibility
    try:
        checker = RepositoryEligibilityChecker()
        result = await checker.check_eligibility(payload.repo)
    except InvalidRepositoryURLException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except GitHubResourceNotFoundException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except GitHubRateLimitExceededException as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Validation failure submitting repository '%s': %s", payload.repo, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to validate repository: {str(exc)}",
        )

    if not result.is_eligible:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.rejection_reason or "Repository is not eligible for ingestion.",
        )

    # 3. Find or create repository anchor in the database
    repo = db.scalar(select(Repository).where(Repository.repo_name == result.repo_name))
    if not repo:
        # Determine ID: use GitHub repository ID if provided, otherwise a hash-derived integer
        repo_id = result.github_id or (abs(hash(result.repo_name)) % 1000000000)
        repo = Repository(
            id=repo_id,
            repo_name=result.repo_name,
            owner=result.owner,
            name=result.name,
            stars_count=result.stars_count,
            forks_count=result.forks_count,
            open_issues_count=result.open_issues_count,
            is_archived=result.is_archived,
            is_fork=result.is_fork,
            upstream_repo=result.upstream_repo,
            is_active=True,
        )
        db.add(repo)
        db.commit()
        db.refresh(repo)

    # 4. Enqueue durable job into collection_runs
    job, is_new = job_queue.submit_job(repo_id=repo.id, mode=payload.mode, db=db)

    if is_new:
        response.status_code = status.HTTP_201_CREATED
        message = "Ingestion job submitted successfully."
    else:
        response.status_code = status.HTTP_200_OK
        message = f"An active ingestion job (ID={job.id}) is already {job.status} for '{repo.repo_name}'."

    return IngestionSubmitResponse(
        job_id=job.id,
        repo_name=repo.repo_name,
        mode=job.mode,
        status=job.status,
        created_at=job.started_at,
        message=message,
    )


@router.get(
    "/status/{job_id}",
    response_model=IngestionStatusResponse,
    summary="Get Ingestion Job Status",
    tags=["ingestion"],
)
def get_job_status(
    job_id: int,
    db: Session = Depends(get_db),
) -> IngestionStatusResponse:
    """Retrieve detailed execution telemetry, entity coverage, and status for an ingestion job."""
    run = job_queue.get_job(job_id, db=db)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job with ID {job_id} not found.",
        )

    # Retrieve repository name
    repo_name = run.repository.repo_name if run.repository else "unknown/unknown"

    # Extract coverage summary elements
    coverage = run.coverage_summary or {}
    entity_status = coverage.get("entity_status", {})
    partial_reason = coverage.get("partial_reason")
    resumes_at_str = coverage.get("resumes_at")
    resumes_at_dt = None
    if resumes_at_str:
        try:
            resumes_at_dt = datetime.fromisoformat(resumes_at_str)
        except Exception:
            resumes_at_dt = None

    # Compute run duration if completed
    duration_seconds = None
    if run.completed_at and run.started_at:
        duration_seconds = max(0.0, (run.completed_at - run.started_at).total_seconds())
    elif coverage.get("execution_time_seconds") is not None:
        duration_seconds = float(coverage["execution_time_seconds"])

    return IngestionStatusResponse(
        job_id=run.id,
        repo_name=repo_name,
        mode=run.mode,
        status=run.status,
        worker_id=run.worker_id,
        leased_until=run.leased_until,
        lease_version=run.lease_version,
        retry_count=run.retry_count,
        max_retries=run.max_retries,
        is_partial=run.is_partial,
        partial_reason=partial_reason,
        resumes_at=resumes_at_dt,
        rate_limit_consumed=run.rate_limit_consumed,
        items_collected=run.items_collected or {},
        entity_status=entity_status,
        coverage_summary=run.coverage_summary,
        error_message=run.error_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_seconds=duration_seconds,
    )
