"""Durable database-backed analytics job queue, snapshot isolation, and lease manager.

Implements Phase C Step C.1.3:
1. Pydantic v2 analytics queue submission and lifecycle contracts.
2. Unified enqueue predicate with no-op suppression and race-safe deduplication.
3. Atomic worker claiming and optimistic lease fencing (PostgreSQL FOR UPDATE SKIP LOCKED / SQLite WAL).
4. Snapshot-consistent read transaction pinning (SQLite WAL read mark / PostgreSQL REPEATABLE READ).
5. Conditional rowcount-validated publication and attempt finalization (atomic with attempt).
6. Strict retry state machine (max_retries = 3, up to 4 attempts total).
7. Stale-result detection and race-safe active replacement-run reuse.
8. Durable reconciliation sweeper for expired leases and crash-safe missing job recovery.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import logging
import os
import socket
from typing import Any, Dict, Generator, List, Optional, Tuple
import uuid

from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    AttemptImmutableException,
    EntityNotFoundException,
    GhostWorkerRejectedException,
)
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    Repository,
)
from backend.app.schemas.analytics_v2 import CURRENT_METHODOLOGY_VERSION

logger = logging.getLogger(__name__)


class AnalyticsJobQueue:
    """Database-backed job queue for Phase C analytical pipelines."""

    DEFAULT_LEASE_DURATION_SECONDS: int = 600  # 10 minutes
    MAX_ALLOWED_RETRIES: int = 3  # 3 retries after initial attempt = 4 total attempts

    @staticmethod
    def generate_worker_id(prefix: Optional[str] = None) -> str:
        """Generate a universally unique worker identifier (prefix:hostname:pid:token)."""
        hostname = socket.gethostname()
        pid = os.getpid()
        token = uuid.uuid4().hex[:8]
        p = f"{prefix}:" if prefix else ""
        return f"{p}{hostname}:{pid}:{token}"

    def enqueue_analysis(
        self,
        repo_id: int,
        db: Session,
        collection_run_id: Optional[int] = None,
        requested_methodology: str = CURRENT_METHODOLOGY_VERSION,
        force: bool = False,
        max_retries: int = MAX_ALLOWED_RETRIES,
    ) -> Tuple[AnalysisRun, bool]:
        """Enqueue an analytical job using the unified predicate.

        Evaluates:
        1. If an active run ('pending' or 'running') exists -> returns existing active run, is_new=False.
        2. If force=False AND a completed run exists for (current_repo_revision, requested_methodology)
           -> returns existing completed run, is_new=False (no-op suppression).
        3. Else -> safely inserts a new pending AnalysisRun, is_new=True. Protected by
           uq_analysis_runs_active_repo against simultaneous submission races.

        Returns:
            Tuple[AnalysisRun, bool]: (run, is_new)
        """
        repo = db.scalar(select(Repository).where(Repository.id == repo_id))
        if repo is None:
            raise EntityNotFoundException(
                message=f"Repository ID {repo_id} not found.",
                entity_type="Repository",
                entity_id=repo_id,
            )

        # 1. Active Run Check (deduplication)
        active_run = db.scalar(
            select(AnalysisRun)
            .where(
                AnalysisRun.repository_id == repo_id,
                AnalysisRun.status.in_(["pending", "running"]),
            )
            .order_by(AnalysisRun.started_at.desc())
            .limit(1)
        )
        if active_run:
            logger.info(
                "Active analysis run ID=%d already exists for repository_id=%d (status=%s)",
                active_run.id,
                repo_id,
                active_run.status,
            )
            return active_run, False

        # 2. Completed Run Check (no-op suppression unless forced)
        if not force:
            completed_run = db.scalar(
                select(AnalysisRun)
                .where(
                    AnalysisRun.repository_id == repo_id,
                    AnalysisRun.status == "completed",
                    AnalysisRun.analytics_source_revision == repo.analytics_source_revision,
                    AnalysisRun.methodology_version == requested_methodology,
                )
                .order_by(AnalysisRun.completed_at.desc())
                .limit(1)
            )
            if completed_run:
                logger.info(
                    "Repository ID=%d already has completed analysis for revision=%d and methodology=%s",
                    repo_id,
                    repo.analytics_source_revision,
                    requested_methodology,
                )
                return completed_run, False

        # 3. Insert New Pending Run
        now = datetime.now(timezone.utc)
        new_run = AnalysisRun(
            repository_id=repo_id,
            collection_run_id=collection_run_id,
            status="pending",
            requested_source_revision=repo.analytics_source_revision,
            analytics_source_revision=None,
            requested_methodology_version=requested_methodology,
            methodology_version=None,
            started_at=now,
            lease_version=0,
            retry_count=0,
            max_retries=max_retries,
        )

        try:
            db.add(new_run)
            db.commit()
            db.refresh(new_run)
            logger.info(
                "Enqueued new analysis run ID=%d for repository_id=%d at revision=%d",
                new_run.id,
                repo_id,
                repo.analytics_source_revision,
            )
            return new_run, True
        except IntegrityError:
            # Race condition collision caught by uq_analysis_runs_active_repo
            db.rollback()
            active_run = db.scalar(
                select(AnalysisRun)
                .where(
                    AnalysisRun.repository_id == repo_id,
                    AnalysisRun.status.in_(["pending", "running"]),
                )
                .order_by(AnalysisRun.started_at.desc())
                .limit(1)
            )
            if active_run:
                logger.info(
                    "Race conflict resolved: returning active analysis run ID=%d for repository_id=%d",
                    active_run.id,
                    repo_id,
                )
                return active_run, False
            raise

    def claim_job(
        self,
        worker_id: str,
        db: Session,
        lease_duration_seconds: int = DEFAULT_LEASE_DURATION_SECONDS,
    ) -> Optional[Tuple[AnalysisRun, AnalysisRunAttempt]]:
        """Atomically claim the next pending or expired analysis run.

        Uses PostgreSQL FOR UPDATE SKIP LOCKED or SQLite WAL single-worker concurrency.
        On success, increments lease_version, marks status='running', and inserts a new
        running AnalysisRunAttempt.

        Returns:
            Optional[Tuple[AnalysisRun, AnalysisRunAttempt]]: Claimed run and attempt, or None if queue empty.
        """
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=lease_duration_seconds)

        bind = db.get_bind() if db else None
        dialect_name = bind.dialect.name if bind else "sqlite"

        if dialect_name == "postgresql":
            stmt = (
                select(AnalysisRun)
                .where(
                    or_(
                        AnalysisRun.status == "pending",
                        and_(
                            AnalysisRun.status == "running",
                            AnalysisRun.leased_until < now,
                            AnalysisRun.retry_count < AnalysisRun.max_retries,
                        ),
                    )
                )
                .order_by(AnalysisRun.started_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            run = db.scalar(stmt)
            if not run:
                return None

            return self._finalize_claim(
                run=run,
                worker_id=worker_id,
                expiry=expiry,
                now=now,
                db=db,
            )

        else:
            # SQLite WAL concurrency with BEGIN IMMEDIATE
            try:
                db.execute(text("BEGIN IMMEDIATE"))
            except Exception as e:
                logger.debug("SQLite database busy during claim: %s", e)
                return None

            try:
                # Single-worker check: Is another analysis run active with non-expired lease?
                active_running = db.scalar(
                    select(AnalysisRun)
                    .where(
                        AnalysisRun.status == "running",
                        AnalysisRun.leased_until > now,
                    )
                    .limit(1)
                )
                if active_running:
                    db.commit()
                    return None

                # Step 1: Check pending runs
                run = db.scalar(
                    select(AnalysisRun)
                    .where(AnalysisRun.status == "pending")
                    .order_by(AnalysisRun.started_at.asc())
                    .limit(1)
                )

                # Step 2: Check expired running runs
                if not run:
                    run = db.scalar(
                        select(AnalysisRun)
                        .where(
                            AnalysisRun.status == "running",
                            AnalysisRun.leased_until < now,
                            AnalysisRun.retry_count < AnalysisRun.max_retries,
                        )
                        .order_by(AnalysisRun.started_at.asc())
                        .limit(1)
                    )

                if not run:
                    db.commit()
                    return None

                return self._finalize_claim(
                    run=run,
                    worker_id=worker_id,
                    expiry=expiry,
                    now=now,
                    db=db,
                )
            except Exception:
                db.rollback()
                raise

    def _finalize_claim(
        self,
        run: AnalysisRun,
        worker_id: str,
        expiry: datetime,
        now: datetime,
        db: Session,
    ) -> Tuple[AnalysisRun, AnalysisRunAttempt]:
        """Finalize claim state updates and attempt insertion."""
        is_recovery = run.status == "running"
        if is_recovery:
            # Mark previous active attempt as lease_expired
            old_attempt = db.scalar(
                select(AnalysisRunAttempt)
                .where(
                    AnalysisRunAttempt.analysis_run_id == run.id,
                    AnalysisRunAttempt.outcome == "running",
                )
                .order_by(AnalysisRunAttempt.attempt_number.desc())
                .limit(1)
            )
            if old_attempt:
                old_attempt.outcome = "lease_expired"
                old_attempt.completed_at = now
                old_attempt.error_message = "Worker lease expired."

            run.retry_count += 1
            logger.info(
                "Worker %s recovering expired run ID=%d (retry %d/%d)",
                worker_id,
                run.id,
                run.retry_count,
                run.max_retries,
            )

        attempt_number = run.retry_count + 1

        run.status = "running"
        run.worker_id = worker_id
        run.lease_version += 1
        run.leased_until = expiry

        attempt = AnalysisRunAttempt(
            analysis_run_id=run.id,
            attempt_number=attempt_number,
            worker_id=worker_id,
            outcome="running",
            started_at=now,
        )
        db.add(attempt)
        db.commit()
        db.refresh(run)
        db.refresh(attempt)

        logger.info(
            "Worker %s claimed run ID=%d (attempt #%d, lease_version=%d, expires=%s)",
            worker_id,
            run.id,
            attempt.attempt_number,
            run.lease_version,
            expiry.isoformat(),
        )
        return run, attempt

    def renew_lease(
        self,
        run_id: int,
        worker_id: str,
        expected_lease_version: int,
        extension_seconds: int = DEFAULT_LEASE_DURATION_SECONDS,
        db: Session = None,
    ) -> bool:
        """Atomically extend the lease duration on a running analysis run.

        Validates expected lease_version and worker_id. Raises GhostWorkerRejectedException
        if lease was stolen, expired, or version mismatch.
        """
        now = datetime.now(timezone.utc)
        new_expiry = now + timedelta(seconds=extension_seconds)

        res = db.execute(
            update(AnalysisRun)
            .where(
                AnalysisRun.id == run_id,
                AnalysisRun.status == "running",
                AnalysisRun.worker_id == worker_id,
                AnalysisRun.lease_version == expected_lease_version,
            )
            .values(
                leased_until=new_expiry,
                lease_version=expected_lease_version + 1,
            )
        )
        if res.rowcount != 1:
            db.rollback()
            raise GhostWorkerRejectedException(
                f"Cannot renew lease for run {run_id}: worker mismatch or lease version expired."
            )

        db.commit()
        logger.debug(
            "Worker %s renewed lease for run %d to %s",
            worker_id,
            run_id,
            new_expiry.isoformat(),
        )
        return True

    @contextmanager
    def open_analysis_snapshot(
        self,
        db: Session,
        repository_id: int,
    ) -> Generator[Tuple[int, datetime], None, None]:
        """Establish a snapshot-consistent read transaction pinning source revision.

        Guarantees:
        - In SQLite WAL: Starts transaction with explicit BEGIN DEFERRED (if not already in transaction)
          and executes SELECT analytics_source_revision as the very first query to pin the WAL read mark.
        - In PostgreSQL: Sets REPEATABLE READ, READ ONLY isolation level before querying revision.
        - Freezes analysis_reference_time at transaction start to anchor all duration calculations.
        """
        bind = db.get_bind() if db else None
        dialect_name = bind.dialect.name if bind else "sqlite"

        reference_time = datetime.now(timezone.utc)

        if dialect_name == "postgresql":
            db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;"))
            stmt = select(Repository.analytics_source_revision).where(Repository.id == repository_id)
            snapshot_rev = db.scalar(stmt)
            if snapshot_rev is None:
                raise EntityNotFoundException(f"Repository {repository_id} not found in snapshot.", entity_type="Repository")
            try:
                yield snapshot_rev, reference_time
            finally:
                db.commit()
        else:
            # SQLite WAL: Start deferred transaction if not currently in transaction
            if not db.in_transaction():
                db.execute(text("BEGIN DEFERRED;"))

            # Pin SQLite WAL read mark with first revision query
            stmt = select(Repository.analytics_source_revision).where(Repository.id == repository_id)
            snapshot_rev = db.scalar(stmt)
            if snapshot_rev is None:
                raise EntityNotFoundException(f"Repository {repository_id} not found in snapshot.", entity_type="Repository")

            try:
                yield snapshot_rev, reference_time
            finally:
                if db.in_transaction():
                    db.commit()

    def publish_and_finalize(
        self,
        run_id: int,
        attempt_id: int,
        worker_id: str,
        expected_lease_version: int,
        attempt_number: int,
        snapshot_revision: int,
        actual_methodology: str,
        reference_time: datetime,
        metrics_payload: Dict[str, Any],
        coverage_summary: Dict[str, Any],
        confidence_summary: Dict[str, Any],
        rule_interpretations: List[Dict[str, Any]],
        db: Session,
    ) -> bool:
        """Atomic result publication and attempt finalization with race-safe active replacement run reuse.

        Protocol:
        1. Conditional UPDATE on AnalysisRun matching (id, status='running', worker_id, lease_version).
           Rowcount MUST == 1.
        2. Conditional UPDATE on AnalysisRunAttempt matching (id, run_id, attempt_number, worker_id, outcome='running').
           Rowcount MUST == 1.
        3. Freshness check:
           - If snapshot_revision == repo.analytics_source_revision -> update repo.last_analyzed_at = now.
           - If snapshot_revision < repo.analytics_source_revision -> result is stale.
             Old run is now 'completed' (freed from uq_analysis_runs_active_repo).
             Check if active replacement run exists. If not, safely enqueue replacement pending run.
        4. Commit transaction atomically.
        """
        now = datetime.now(timezone.utc)

        # 1. Conditional Atomic Update on Run
        res_run = db.execute(
            update(AnalysisRun)
            .where(
                AnalysisRun.id == run_id,
                AnalysisRun.status == "running",
                AnalysisRun.worker_id == worker_id,
                AnalysisRun.lease_version == expected_lease_version,
            )
            .values(
                status="completed",
                analytics_source_revision=snapshot_revision,
                methodology_version=actual_methodology,
                analysis_reference_time=reference_time,
                source_content_digest=None,
                metrics_payload=metrics_payload,
                coverage_summary=coverage_summary,
                confidence_summary=confidence_summary,
                rule_interpretations=rule_interpretations,
                completed_at=now,
            )
        )
        if res_run.rowcount != 1:
            db.rollback()
            raise GhostWorkerRejectedException(
                f"Cannot publish results for run {run_id}: worker lease expired or worker identity mismatch."
            )

        # 2. Conditional Atomic Update on Attempt
        res_att = db.execute(
            update(AnalysisRunAttempt)
            .where(
                AnalysisRunAttempt.id == attempt_id,
                AnalysisRunAttempt.analysis_run_id == run_id,
                AnalysisRunAttempt.attempt_number == attempt_number,
                AnalysisRunAttempt.worker_id == worker_id,
                AnalysisRunAttempt.outcome == "running",
            )
            .values(
                outcome="completed",
                completed_at=now,
            )
        )
        if res_att.rowcount != 1:
            db.rollback()
            raise AttemptImmutableException(
                f"Cannot finalize attempt {attempt_id}: attempt is not in running state or worker mismatch."
            )

        # 3. Check Stale State against Current Repository Revision
        run = db.scalar(select(AnalysisRun).where(AnalysisRun.id == run_id))
        repo = db.scalar(select(Repository).where(Repository.id == run.repository_id))

        if repo:
            if snapshot_revision == repo.analytics_source_revision:
                repo.last_analyzed_at = now
            else:
                # Stale publication: Check if an active replacement run already exists
                active_run = db.scalar(
                    select(AnalysisRun).where(
                        AnalysisRun.repository_id == repo.id,
                        AnalysisRun.status.in_(["pending", "running"]),
                    )
                )
                if active_run is None:
                    new_run = AnalysisRun(
                        repository_id=repo.id,
                        requested_source_revision=repo.analytics_source_revision,
                        requested_methodology_version=actual_methodology,
                        status="pending",
                        started_at=now,
                    )
                    db.add(new_run)
                    logger.info(
                        "Stale publication (rev %d < %d): enqueued replacement pending run for repo %d",
                        snapshot_revision,
                        repo.analytics_source_revision,
                        repo.id,
                    )
                else:
                    logger.info(
                        "Stale publication (rev %d < %d): active replacement Run %d already exists for repo %d",
                        snapshot_revision,
                        repo.analytics_source_revision,
                        active_run.id,
                        repo.id,
                    )

        db.commit()
        return True

    def record_attempt_failure(
        self,
        run_id: int,
        attempt_id: int,
        worker_id: str,
        expected_lease_version: int,
        attempt_number: int,
        error_message: str,
        db: Session,
        outcome: str = "failed",
    ) -> bool:
        """Finalize an attempt as failed or interrupted and evaluate retry transition.

        Transition Rules:
        - If retry_count < max_retries: retry_count increments, run resets to 'pending'
          with worker_id=None, leased_until=None.
        - If retry_count >= max_retries: retry_count increments, run transitions terminally
          to 'failed' with completed_at=now.
        """
        now = datetime.now(timezone.utc)

        # 1. Update Attempt
        res_att = db.execute(
            update(AnalysisRunAttempt)
            .where(
                AnalysisRunAttempt.id == attempt_id,
                AnalysisRunAttempt.analysis_run_id == run_id,
                AnalysisRunAttempt.attempt_number == attempt_number,
                AnalysisRunAttempt.worker_id == worker_id,
                AnalysisRunAttempt.outcome == "running",
            )
            .values(
                outcome=outcome,
                error_message=error_message,
                completed_at=now,
            )
        )
        if res_att.rowcount != 1:
            db.rollback()
            raise AttemptImmutableException(
                f"Cannot fail attempt {attempt_id}: attempt is not in running state or worker mismatch."
            )

        # 2. Check retry bounds on Run
        run = db.scalar(select(AnalysisRun).where(AnalysisRun.id == run_id))
        if not run:
            db.rollback()
            raise EntityNotFoundException(f"AnalysisRun {run_id} not found.", entity_type="AnalysisRun")

        new_retry_count = run.retry_count + 1
        if run.retry_count < run.max_retries:
            target_status = "pending"
            completed_at = None
            term_error = None
        else:
            target_status = "failed"
            completed_at = now
            term_error = error_message

        res_run = db.execute(
            update(AnalysisRun)
            .where(
                AnalysisRun.id == run_id,
                AnalysisRun.status == "running",
                AnalysisRun.worker_id == worker_id,
                AnalysisRun.lease_version == expected_lease_version,
            )
            .values(
                status=target_status,
                retry_count=new_retry_count,
                worker_id=None,
                leased_until=None,
                lease_version=expected_lease_version + 1,
                completed_at=completed_at,
                error_message=term_error,
            )
        )
        if res_run.rowcount != 1:
            db.rollback()
            raise GhostWorkerRejectedException(
                f"Cannot record failure on run {run_id}: worker lease expired or lease version mismatch."
            )

        db.commit()
        logger.info(
            "Recorded attempt #%d %s for run %d (new status=%s, retry_count=%d/%d)",
            attempt_number,
            outcome,
            run_id,
            target_status,
            new_retry_count,
            run.max_retries,
        )
        return True

    def finalize_attempt(
        self,
        attempt_id: int,
        outcome: str,
        worker_id: Optional[str] = None,
        error_message: Optional[str] = None,
        db: Session = None,
    ) -> bool:
        """Application-level gatekeeper enforcing attempt row immutability.

        Modifies an attempt ONLY if it is currently in 'running' state. Once terminal,
        raises AttemptImmutableException.
        """
        valid_outcomes = {"completed", "failed", "lease_expired", "interrupted"}
        if outcome not in valid_outcomes:
            raise ValueError(f"Invalid attempt outcome '{outcome}'. Expected one of {valid_outcomes}")

        stmt = update(AnalysisRunAttempt).where(
            AnalysisRunAttempt.id == attempt_id,
            AnalysisRunAttempt.outcome == "running",
        )
        if worker_id:
            stmt = stmt.where(AnalysisRunAttempt.worker_id == worker_id)

        now = datetime.now(timezone.utc)
        stmt = stmt.values(
            outcome=outcome,
            error_message=error_message,
            completed_at=now,
        )

        res = db.execute(stmt)
        if res.rowcount != 1:
            db.rollback()
            raise AttemptImmutableException(
                f"Attempt {attempt_id} is already in a terminal state or worker mismatch; cannot set outcome='{outcome}'."
            )

        db.commit()
        return True

    def reconcile_analytics_jobs(
        self,
        db: Session,
        max_lease_age_seconds: int = DEFAULT_LEASE_DURATION_SECONDS,
    ) -> Dict[str, int]:
        """Durable reconciliation sweeper for expired leases and missing analysis jobs.

        Guarantees:
        1. Expired Leases: Identifies running runs past leased_until, marks active attempt
           as 'lease_expired', and applies retry state transitions (to 'pending' or 'failed').
        2. Missing Jobs (Crash-Safe Ingestion Handoff): If ingestion commits data but worker crashes
           before enqueuing analysis, detects active repositories lacking active analysis and
           lacking completed analysis for current revision, safely enqueuing a pending job.

        Returns:
            Dict[str, int]: Count summary of recovered leases and missing jobs enqueued.
        """
        now = datetime.now(timezone.utc)
        summary = {"expired_leases_recovered": 0, "missing_jobs_enqueued": 0}

        # 1. Sweep Expired Leases
        expired_runs = db.scalars(
            select(AnalysisRun).where(
                AnalysisRun.status == "running",
                AnalysisRun.leased_until < now,
            )
        ).all()

        for run in expired_runs:
            # Finalize running attempt as lease_expired
            attempt = db.scalar(
                select(AnalysisRunAttempt).where(
                    AnalysisRunAttempt.analysis_run_id == run.id,
                    AnalysisRunAttempt.outcome == "running",
                ).order_by(AnalysisRunAttempt.attempt_number.desc()).limit(1)
            )
            if attempt:
                attempt.outcome = "lease_expired"
                attempt.completed_at = now
                attempt.error_message = f"Lease timed out after {max_lease_age_seconds}s."

            run.retry_count += 1
            if run.retry_count <= run.max_retries:
                run.status = "pending"
                run.worker_id = None
                run.leased_until = None
                run.lease_version += 1
                logger.info(
                    "Reconciliation recovered expired run ID=%d (reset to pending, retry %d/%d)",
                    run.id,
                    run.retry_count,
                    run.max_retries,
                )
            else:
                run.status = "failed"
                run.completed_at = now
                run.error_message = f"Terminal failure: lease expired after {run.retry_count} attempts."
                run.worker_id = None
                run.leased_until = None
                run.lease_version += 1
                logger.info(
                    "Reconciliation terminated expired run ID=%d (max retries %d exceeded)",
                    run.id,
                    run.max_retries,
                )
            summary["expired_leases_recovered"] += 1

        db.commit()

        # 2. Sweep Missing Analysis Jobs (Crash-Safe Ingestion Handoff)
        active_repos = db.scalars(select(Repository).where(Repository.is_active == True)).all()
        for repo in active_repos:
            # Check if active run exists
            has_active = db.scalar(
                select(AnalysisRun.id).where(
                    AnalysisRun.repository_id == repo.id,
                    AnalysisRun.status.in_(["pending", "running"]),
                ).limit(1)
            )
            if has_active:
                continue

            # Check if current revision is already analyzed with current methodology
            has_completed = db.scalar(
                select(AnalysisRun.id).where(
                    AnalysisRun.repository_id == repo.id,
                    AnalysisRun.status == "completed",
                    AnalysisRun.analytics_source_revision == repo.analytics_source_revision,
                    AnalysisRun.methodology_version == CURRENT_METHODOLOGY_VERSION,
                ).limit(1)
            )
            if has_completed:
                continue

            # Repo has unanalyzed revision and no active run: enqueue missing run!
            try:
                new_run = AnalysisRun(
                    repository_id=repo.id,
                    status="pending",
                    requested_source_revision=repo.analytics_source_revision,
                    requested_methodology_version=CURRENT_METHODOLOGY_VERSION,
                    started_at=now,
                    lease_version=0,
                    retry_count=0,
                    max_retries=self.MAX_ALLOWED_RETRIES,
                )
                db.add(new_run)
                db.commit()
                summary["missing_jobs_enqueued"] += 1
                logger.info(
                    "Crash-safe handoff: recovered missing analysis job for repo %d at revision %d",
                    repo.id,
                    repo.analytics_source_revision,
                )
            except IntegrityError:
                # Concurrent enqueue collision caught safely
                db.rollback()

        return summary
