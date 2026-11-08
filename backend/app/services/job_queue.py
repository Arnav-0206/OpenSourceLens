"""Durable database-backed ingestion job queue and lease manager.

Provides atomic job claiming, PostgreSQL row-level locking (FOR UPDATE SKIP LOCKED),
SQLite WAL concurrency with single-worker optimistic lease fencing, heartbeat renewals,
and automated recovery of abandoned or expired jobs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
import os
import socket
from typing import Any, Dict, Optional, Tuple, Union
import uuid

from sqlalchemy import and_, or_, select, update, text, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.models import CollectionRun, Repository
from backend.app.schemas.ingestion import IngestionSummary

logger = logging.getLogger(__name__)

# In-process lock to prevent concurrent claim contention against SQLite WAL
_sqlite_claim_lock = asyncio.Lock()


class JobLeaseLostException(Exception):
    """Raised when a worker attempts to write, renew, or finalize a job whose lease was revoked or expired."""

    pass


class DurableJobQueue:
    """Database-backed job queue for asynchronous repository collection and telemetry."""

    DEFAULT_LEASE_DURATION_SECONDS = 600  # 10 minutes

    @staticmethod
    def generate_worker_id(prefix: Optional[str] = None) -> str:
        """Generate a universally unique worker identifier (hostname:pid:uuid)."""
        hostname = socket.gethostname()
        pid = os.getpid()
        token = uuid.uuid4().hex[:8]
        p = f"{prefix}:" if prefix else ""
        return f"{p}{hostname}:{pid}:{token}"

    def submit_job(
        self,
        repo_id: int,
        mode: str = "standard_diagnostic",
        max_retries: int = 2,
        db: Session = None,
    ) -> Tuple[CollectionRun, bool]:
        """Enqueue a new persistent ingestion job or return an existing in-flight job.

        Applies application-level deduplication combined with a database-level partial
        unique index (uq_collection_runs_active_repo) to prevent race-condition duplicates.

        Args:
            repo_id: Foreign key ID of the target Repository.
            mode: Collection mode ('fast_snapshot' or 'standard_diagnostic').
            max_retries: Maximum number of times the job can be re-leased on worker death.
            db: SQLAlchemy session.

        Returns:
            Tuple[CollectionRun, bool]: (job, is_new) where is_new is True if a new job
            was created, or False if an existing active job (pending or in_progress) was returned.
        """
        # Step 1: Check for existing active job (pending or in_progress)
        existing_active = db.scalar(
            select(CollectionRun)
            .where(
                CollectionRun.repository_id == repo_id,
                CollectionRun.status.in_(["pending", "in_progress"]),
            )
            .order_by(CollectionRun.started_at.desc())
            .limit(1)
        )
        if existing_active:
            logger.info(
                "Active job ID=%d already exists for repository_id=%d (status=%s)",
                existing_active.id,
                repo_id,
                existing_active.status,
            )
            return existing_active, False

        # Step 2: Insert new job; partial unique index protects against simultaneous insert race
        now = datetime.now(timezone.utc)
        job = CollectionRun(
            repository_id=repo_id,
            mode=mode,
            status="pending",
            lease_version=0,
            retry_count=0,
            max_retries=max_retries,
            is_partial=False,
            started_at=now,
        )
        try:
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(
                "Submitted new ingestion job ID=%d for repository_id=%d (mode=%s)",
                job.id,
                repo_id,
                mode,
            )
            return job, True
        except IntegrityError:
            # Concurrent submission race caught by database-level unique index
            db.rollback()
            existing_active = db.scalar(
                select(CollectionRun)
                .where(
                    CollectionRun.repository_id == repo_id,
                    CollectionRun.status.in_(["pending", "in_progress"]),
                )
                .order_by(CollectionRun.started_at.desc())
                .limit(1)
            )
            if existing_active:
                logger.info(
                    "Race conflict resolved: returning active job ID=%d for repository_id=%d",
                    existing_active.id,
                    repo_id,
                )
                return existing_active, False
            raise

    def claim_next_job(
        self,
        worker_id: str,
        lease_duration_seconds: int = DEFAULT_LEASE_DURATION_SECONDS,
        db: Session = None,
    ) -> Optional[CollectionRun]:
        """Atomically claim the next eligible pending or expired job.

        In PostgreSQL, uses 'FOR UPDATE SKIP LOCKED' for multi-worker concurrency.
        In SQLite, uses optimistic concurrency control with lease-version fencing.

        Args:
            worker_id: Unique identifier for the claiming worker.
            lease_duration_seconds: Duration of the acquired lease before expiration.
            db: SQLAlchemy session.

        Returns:
            CollectionRun if a job was successfully claimed, or None if the queue is empty.
        """
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=lease_duration_seconds)

        # Detect database dialect
        bind = db.get_bind() if db else None
        dialect_name = bind.dialect.name if bind else "sqlite"

        if dialect_name == "postgresql":
            # PostgreSQL atomic row-locking query using FOR UPDATE SKIP LOCKED
            stmt = (
                select(CollectionRun)
                .where(
                    or_(
                        CollectionRun.status == "pending",
                        and_(
                            CollectionRun.status == "in_progress",
                            CollectionRun.leased_until < now,
                            CollectionRun.retry_count < CollectionRun.max_retries,
                        ),
                    )
                )
                .order_by(CollectionRun.started_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            job = db.scalar(stmt)
            if not job:
                return None

            if job.status == "in_progress":
                job.retry_count += 1
                logger.info(
                    "Worker %s recovering expired job ID=%d (retry %d/%d)",
                    worker_id,
                    job.id,
                    job.retry_count,
                    job.max_retries,
                )

            job.status = "in_progress"
            job.worker_id = worker_id
            job.leased_until = expiry
            job.lease_version += 1
            job.started_at = now
            db.commit()
            db.refresh(job)
            return job

        else:
            # SQLite WAL single-writer mode with atomic BEGIN IMMEDIATE
            try:
                db.execute(text("BEGIN IMMEDIATE"))
            except Exception as e:
                # If database write lock is temporarily held by another transaction, return None
                logger.debug("SQLite database busy during claim: %s", e)
                return None

            try:
                # Enforce SQLite single-worker restriction:
                # Verify no other job is currently active ('in_progress' with non-expired lease)
                active_job = db.scalar(
                    select(CollectionRun)
                    .where(
                        CollectionRun.status == "in_progress",
                        CollectionRun.leased_until > now,
                    )
                    .limit(1)
                )
                if active_job:
                    # Another worker process is currently active; release lock and exit
                    db.commit()
                    return None

                # Step 1: Look for pending job
                stmt_pending = (
                    select(CollectionRun)
                    .where(CollectionRun.status == "pending")
                    .order_by(CollectionRun.started_at.asc())
                    .limit(1)
                )
                job = db.scalar(stmt_pending)

                is_recovery = False
                if not job:
                    # Step 2: Check for expired in-progress jobs eligible for retry
                    stmt_expired = (
                        select(CollectionRun)
                        .where(
                            CollectionRun.status == "in_progress",
                            CollectionRun.leased_until < now,
                            CollectionRun.retry_count < CollectionRun.max_retries,
                        )
                        .order_by(CollectionRun.started_at.asc())
                        .limit(1)
                    )
                    job = db.scalar(stmt_expired)
                    if not job:
                        db.commit()
                        return None
                    is_recovery = True

                expected_version = job.lease_version
                new_retry_count = job.retry_count + (1 if is_recovery else 0)

                # Optimistic concurrency check: update only if lease_version has not changed
                update_stmt = (
                    update(CollectionRun)
                    .where(
                        CollectionRun.id == job.id,
                        CollectionRun.lease_version == expected_version,
                    )
                    .values(
                        status="in_progress",
                        worker_id=worker_id,
                        leased_until=expiry,
                        lease_version=expected_version + 1,
                        retry_count=new_retry_count,
                        started_at=now,
                    )
                )
                res = db.execute(update_stmt)
                db.commit()

                if res.rowcount == 0:
                    # Race condition lost to another worker
                    logger.debug(
                        "Worker %s lost race to claim job ID=%d (version mismatch)",
                        worker_id,
                        job.id,
                    )
                    return None

                db.refresh(job)
                logger.info(
                    "Worker %s claimed job ID=%d (mode=%s, lease_version=%d, recovery=%s)",
                    worker_id,
                    job.id,
                    job.mode,
                    job.lease_version,
                    is_recovery,
                )
                return job
            except Exception:
                db.rollback()
                raise

    def renew_lease(
        self,
        job_id: int,
        worker_id: str,
        lease_version: int,
        extension_seconds: int = DEFAULT_LEASE_DURATION_SECONDS,
        db: Session = None,
    ) -> bool:
        """Renew worker lease expiration for an active in-progress job under lease fencing.

        Returns True if the lease was extended, False if the worker's lease was lost.
        """
        now = datetime.now(timezone.utc)
        new_expiry = now + timedelta(seconds=extension_seconds)
        stmt = (
            update(CollectionRun)
            .where(
                CollectionRun.id == job_id,
                CollectionRun.worker_id == worker_id,
                CollectionRun.lease_version == lease_version,
                CollectionRun.status == "in_progress",
            )
            .values(leased_until=new_expiry)
        )
        res = db.execute(stmt)
        db.commit()
        if res.rowcount > 0:
            logger.debug(
                "Lease renewed for job ID=%d by worker %s until %s",
                job_id,
                worker_id,
                new_expiry.isoformat(),
            )
            return True
        logger.warning(
            "Lease renewal rejected for job ID=%d: worker %s lost lease version %d",
            job_id,
            worker_id,
            lease_version,
        )
        return False

    def complete_job(
        self,
        job_id: int,
        worker_id: str,
        lease_version: int,
        summary: IngestionSummary,
        db: Session = None,
    ) -> bool:
        """Finalize an in-progress job with harvested results under lease-version fencing.

        Raises JobLeaseLostException if the worker's lease was revoked or reassigned.
        """
        now = datetime.now(timezone.utc)
        if summary.resumes_at is not None:
            final_status = "rate_limited"
        elif summary.is_partial:
            final_status = "completed_partial"
        else:
            final_status = "completed"

        resumes_at_val = (
            summary.resumes_at.isoformat()
            if hasattr(summary.resumes_at, "isoformat")
            else summary.resumes_at
        )

        coverage_envelope = {
            "entity_status": summary.entity_status,
            "partial_reason": summary.partial_reason,
            "resumes_at": resumes_at_val,
            "total_requests": summary.total_requests_consumed,
            "errors": summary.errors,
            "execution_time_seconds": summary.execution_time_seconds,
        }

        stmt = (
            update(CollectionRun)
            .where(
                CollectionRun.id == job_id,
                CollectionRun.worker_id == worker_id,
                CollectionRun.lease_version == lease_version,
                CollectionRun.status == "in_progress",
            )
            .values(
                status=final_status,
                is_partial=summary.is_partial,
                completed_at=now,
                items_collected=summary.items_collected,
                coverage_summary=coverage_envelope,
                rate_limit_consumed=summary.total_requests_consumed,
                error_message="; ".join(summary.errors) if summary.errors else None,
            )
        )
        res = db.execute(stmt)
        db.commit()

        if res.rowcount == 0:
            raise JobLeaseLostException(
                f"Fencing token revoked: Worker '{worker_id}' lost lease version {lease_version} for job ID={job_id}."
            )

        logger.info(
            "Job ID=%d finalized by worker %s as '%s' (items: %s, partial: %s)",
            job_id,
            worker_id,
            final_status,
            summary.items_collected,
            summary.is_partial,
        )
        return True

    def fail_job(
        self,
        job_id: int,
        worker_id: str,
        lease_version: int,
        error_message: str,
        db: Session = None,
    ) -> bool:
        """Mark an in-progress job as failed under lease-version fencing.

        Raises JobLeaseLostException if the worker's lease was revoked or reassigned.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            update(CollectionRun)
            .where(
                CollectionRun.id == job_id,
                CollectionRun.worker_id == worker_id,
                CollectionRun.lease_version == lease_version,
                CollectionRun.status == "in_progress",
            )
            .values(
                status="failed",
                completed_at=now,
                error_message=error_message,
            )
        )
        res = db.execute(stmt)
        db.commit()

        if res.rowcount == 0:
            raise JobLeaseLostException(
                f"Fencing token revoked: Worker '{worker_id}' lost lease version {lease_version} for job ID={job_id}."
            )

        logger.warning(
            "Job ID=%d marked as failed by worker %s: %s",
            job_id,
            worker_id,
            error_message,
        )
        return True

    def recover_expired_jobs(
        self,
        max_retries: int = 2,
        db: Session = None,
    ) -> Dict[str, int]:
        """Scan and transition expired in-progress jobs that have exceeded max_retries to 'failed'.

        Returns a dictionary with the count of permanently failed expired jobs.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            update(CollectionRun)
            .where(
                CollectionRun.status == "in_progress",
                CollectionRun.leased_until < now,
                CollectionRun.retry_count >= max_retries,
            )
            .values(
                status="failed",
                completed_at=now,
                error_message=f"Job lease expired and max retry limit ({max_retries}) was exceeded.",
            )
        )
        res = db.execute(stmt)
        db.commit()
        if res.rowcount > 0:
            logger.warning(
                "Marked %d abandoned jobs as failed (exceeded max_retries=%d)",
                res.rowcount,
                max_retries,
            )
        return {"failed_count": res.rowcount}

    def get_job(self, job_id: int, db: Session) -> Optional[CollectionRun]:
        """Retrieve a CollectionRun by ID."""
        return db.get(CollectionRun, job_id)
