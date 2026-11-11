"""Asynchronous background ingestion worker service.

Processes durable collection jobs with periodic lease heartbeats, lease-version
fencing, graceful exception handling, and rate-limit resumption awareness.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import CollectionRun, Repository
from backend.app.schemas.ingestion import IngestionSummary
from backend.app.services.github_client import AsyncGitHubClient
from backend.app.services.job_queue import DurableJobQueue, JobLeaseLostException
from backend.app.services.rate_budgeter import EntityRateBudgeter
from backend.app.services.repository_collector import RepositoryCollector

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Worker instance that claims and executes repository ingestion jobs."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        job_queue: Optional[DurableJobQueue] = None,
        lease_duration_seconds: int = 600,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        """Initialize worker with unique identity and lease configuration.

        Args:
            worker_id: Optional custom identifier. If None, generated automatically.
            job_queue: DurableJobQueue instance for database operations.
            lease_duration_seconds: Duration of the lease in seconds (default: 600s).
            heartbeat_interval_seconds: Frequency of heartbeat lease renewals in seconds.
        """
        self.job_queue = job_queue or DurableJobQueue()
        self.worker_id = worker_id or self.job_queue.generate_worker_id("worker")
        self.lease_duration_seconds = lease_duration_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    async def process_one_job(
        self,
        db: Session,
        client: Optional[AsyncGitHubClient] = None,
        collector: Optional[RepositoryCollector] = None,
    ) -> Optional[int]:
        """Claim and execute the next eligible job from the durable queue.

        Runs with heartbeat lease renewal in the background and enforces lease-version
        fencing on finalization to prevent zombie worker writes.

        Args:
            db: SQLAlchemy session.
            client: Optional AsyncGitHubClient. If None, instantiates a new client.
            collector: Optional RepositoryCollector. If None, instantiates with client.

        Returns:
            The job ID processed, or None if no pending/expired jobs were available.

        Raises:
            JobLeaseLostException: If the worker's lease was expired or revoked during processing.
        """
        # 1. Recover expired jobs that exceeded max_retries before claiming
        self.job_queue.recover_expired_jobs(db=db)

        # 2. Atomically claim next job
        job = self.job_queue.claim_next_job(
            worker_id=self.worker_id,
            lease_duration_seconds=self.lease_duration_seconds,
            db=db,
        )
        if not job:
            return None

        job_id = job.id
        lease_version = job.lease_version
        mode = job.mode

        # Retrieve target repository name
        repo = db.get(Repository, job.repository_id)
        if not repo:
            error_msg = f"Target repository with ID={job.repository_id} does not exist."
            logger.error(error_msg)
            self.job_queue.fail_job(
                job_id=job_id,
                worker_id=self.worker_id,
                lease_version=lease_version,
                error_message=error_msg,
                db=db,
            )
            return job_id

        repo_name = repo.repo_name
        logger.info(
            "Worker %s starting execution of job ID=%d for '%s' (mode=%s, lease_version=%d)",
            self.worker_id,
            job_id,
            repo_name,
            mode,
            lease_version,
        )

        # 3. Setup background heartbeat task
        stop_heartbeat = asyncio.Event()
        lease_lost_event = asyncio.Event()

        async def _heartbeat_loop():
            while not stop_heartbeat.is_set():
                try:
                    await asyncio.sleep(self.heartbeat_interval_seconds)
                except asyncio.CancelledError:
                    break

                if stop_heartbeat.is_set():
                    break

                # Attempt lease renewal
                renewed = self.job_queue.renew_lease(
                    job_id=job_id,
                    worker_id=self.worker_id,
                    lease_version=lease_version,
                    extension_seconds=self.lease_duration_seconds,
                    db=db,
                )
                if not renewed:
                    logger.warning(
                        "Heartbeat failed for worker %s on job ID=%d: lease was revoked or version changed.",
                        self.worker_id,
                        job_id,
                    )
                    lease_lost_event.set()
                    break

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        # 4. Execute repository collection
        owns_client = client is None
        active_client = client or AsyncGitHubClient()
        active_collector = collector or RepositoryCollector(client=active_client)
        budgeter = EntityRateBudgeter(mode=mode)

        try:
            summary = await active_collector.collect_repository(
                repo_name=repo_name,
                mode=mode,
                db=db,
                budgeter=budgeter,
            )

            # Check if heartbeat flagged a revoked lease during collection
            if lease_lost_event.is_set():
                raise JobLeaseLostException(
                    f"Worker {self.worker_id} lease revoked during execution of job ID={job_id}."
                )

            # Complete job with lease-version fencing check
            self.job_queue.complete_job(
                job_id=job_id,
                worker_id=self.worker_id,
                lease_version=lease_version,
                summary=summary,
                db=db,
            )
            if summary.errors:
                logger.error(
                    "Worker %s finalized job ID=%d for '%s' as 'failed' due to collector errors: %s",
                    self.worker_id,
                    job_id,
                    repo_name,
                    "; ".join(summary.errors),
                )
            else:
                logger.info(
                    "Worker %s successfully finalized job ID=%d for '%s' (partial=%s, resumes_at=%s)",
                    self.worker_id,
                    job_id,
                    repo_name,
                    summary.is_partial,
                    summary.resumes_at,
                )
            return job_id

        except JobLeaseLostException:
            logger.critical(
                "Worker %s lost lease for job ID=%d. Aborting write to preserve state.",
                self.worker_id,
                job_id,
            )
            raise

        except Exception as exc:
            if lease_lost_event.is_set():
                raise JobLeaseLostException(
                    f"Worker {self.worker_id} lost lease during failed execution of job ID={job_id}."
                ) from exc

            logger.exception("Error executing job ID=%d for '%s': %s", job_id, repo_name, exc)
            try:
                self.job_queue.fail_job(
                    job_id=job_id,
                    worker_id=self.worker_id,
                    lease_version=lease_version,
                    error_message=str(exc),
                    db=db,
                )
            except JobLeaseLostException:
                logger.warning(
                    "Could not mark job ID=%d as failed: lease was already reclaimed by another worker.",
                    job_id,
                )
            raise

        finally:
            stop_heartbeat.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            if owns_client and hasattr(active_client, "close"):
                await active_client.close()

    async def run_worker_loop(
        self,
        db_factory: Callable[[], Session],
        client: Optional[AsyncGitHubClient] = None,
        poll_interval: float = 1.0,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Continuously poll and process jobs from the durable queue until stop_event is set."""
        logger.info("Worker %s entering execution loop...", self.worker_id)
        while stop_event is None or not stop_event.is_set():
            db = db_factory()
            try:
                processed_id = await self.process_one_job(db=db, client=client)
                if processed_id is None:
                    # Queue is empty, pause before next poll
                    await asyncio.sleep(poll_interval)
            except JobLeaseLostException:
                logger.warning("Worker %s encountered lease loss, continuing loop...", self.worker_id)
                await asyncio.sleep(poll_interval)
            except Exception as e:
                logger.error("Unexpected worker loop error: %s", e)
                await asyncio.sleep(poll_interval)
            finally:
                db.close()
