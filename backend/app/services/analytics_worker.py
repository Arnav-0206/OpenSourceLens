"""Durable background analytics worker service (Phase C Step C.8.1).

Coordinates periodic queue reconciliation and single-job / continuous-loop
execution using the existing C.1 AnalyticsJobQueue and C.7 AnalyticsOrchestrator
without creating a second job queue or duplicate state machine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    AttemptImmutableException,
    GhostWorkerRejectedException,
    InvalidParameterException,
)
from backend.app.services.analytics.orchestrator import (
    AnalyticsOrchestrator,
    OrchestrationExecutionResult,
)
from backend.app.services.analytics_queue import AnalyticsJobQueue

logger = logging.getLogger(__name__)


class AnalyticsWorker:
    """Worker that claims, reconciles, and processes queued Phase C analytics jobs."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        orchestrator: Optional[AnalyticsOrchestrator] = None,
        job_queue: Optional[AnalyticsJobQueue] = None,
        lease_duration_seconds: int = AnalyticsJobQueue.DEFAULT_LEASE_DURATION_SECONDS,
        reconcile_on_process: bool = False,
        reconcile_interval_seconds: float = 60.0,
    ) -> None:
        """Initialize the analytics worker reusing the canonical analytics queue and orchestrator."""
        if lease_duration_seconds <= 0:
            raise InvalidParameterException(
                "lease_duration_seconds must be positive.",
                parameter="lease_duration_seconds",
            )

        if orchestrator is not None:
            self.orchestrator = orchestrator
            self.job_queue = job_queue or orchestrator.queue
            self.orchestrator.queue = self.job_queue
        else:
            self.job_queue = job_queue or AnalyticsJobQueue()
            self.orchestrator = AnalyticsOrchestrator(job_queue=self.job_queue)

        self.worker_id: str = worker_id or self.job_queue.generate_worker_id("analytics-worker")
        self.lease_duration_seconds: int = int(lease_duration_seconds)
        self.reconcile_on_process: bool = bool(reconcile_on_process)
        self.reconcile_interval_seconds: float = float(reconcile_interval_seconds)

    def reconcile_jobs(self, db: Session) -> Dict[str, int]:
        """Recover expired analytics leases and missing active-repo analysis jobs."""
        summary = self.job_queue.reconcile_analytics_jobs(
            db=db,
            max_lease_age_seconds=self.lease_duration_seconds,
        )
        if summary["expired_leases_recovered"] > 0 or summary["missing_jobs_enqueued"] > 0:
            logger.info(
                "Analytics worker %s reconciliation summary: %s",
                self.worker_id,
                summary,
            )
        return summary

    def process_one_job(
        self,
        db: Session,
        *,
        reconcile: Optional[bool] = None,
    ) -> Optional[OrchestrationExecutionResult]:
        """Claim and execute the next eligible analytics job from AnalyticsJobQueue.

        Returns None when no pending or expired-lease run is available.
        Propagates GhostWorkerRejectedException / AttemptImmutableException on lease loss.
        """
        should_reconcile = self.reconcile_on_process if reconcile is None else bool(reconcile)
        if should_reconcile:
            self.reconcile_jobs(db=db)

        result = self.orchestrator.process_next_job(
            db=db,
            worker_id=self.worker_id,
            lease_duration_seconds=self.lease_duration_seconds,
        )
        if result is None:
            return None

        logger.info(
            "Analytics worker %s finished run ID=%s (repo=%s, status=%s, published=%s, failure_category=%s)",
            self.worker_id,
            result.run_id,
            result.repository_id,
            result.status,
            result.published,
            result.failure_category,
        )
        return result

    def drain_queue(
        self,
        db: Session,
        *,
        max_jobs: int = 100,
        reconcile_first: bool = False,
    ) -> List[OrchestrationExecutionResult]:
        """Synchronously process queued analytics jobs until empty or max_jobs is reached."""
        if max_jobs <= 0:
            raise InvalidParameterException("max_jobs must be positive.", parameter="max_jobs")

        if reconcile_first:
            self.reconcile_jobs(db=db)

        results: List[OrchestrationExecutionResult] = []
        for _ in range(max_jobs):
            outcome = self.process_one_job(db=db, reconcile=False)
            if outcome is None:
                break
            results.append(outcome)
        return results

    async def run_worker_loop(
        self,
        db_factory: Callable[[], Session],
        *,
        poll_interval: float = 1.0,
        reconcile_interval: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Continuously reconcile and process analytics jobs until stopped."""
        effective_reconcile_interval = (
            self.reconcile_interval_seconds
            if reconcile_interval is None
            else float(reconcile_interval)
        )
        last_reconcile_monotonic: Optional[float] = None
        iterations = 0

        logger.info("Analytics worker %s entering execution loop...", self.worker_id)
        while (stop_event is None or not stop_event.is_set()) and (
            max_iterations is None or iterations < max_iterations
        ):
            iterations += 1
            db = db_factory()
            should_sleep = False
            try:
                now_mono = time.monotonic()
                if effective_reconcile_interval >= 0.0 and (
                    last_reconcile_monotonic is None
                    or (now_mono - last_reconcile_monotonic) >= effective_reconcile_interval
                ):
                    self.reconcile_jobs(db=db)
                    last_reconcile_monotonic = now_mono

                result = self.process_one_job(db=db, reconcile=False)
                if result is None:
                    should_sleep = True
            except (GhostWorkerRejectedException, AttemptImmutableException) as exc:
                logger.warning(
                    "Analytics worker %s encountered lease or attempt conflict (%s); continuing loop.",
                    self.worker_id,
                    exc,
                )
                if db.in_transaction():
                    db.rollback()
                should_sleep = True
            except Exception as exc:
                logger.exception(
                    "Unexpected analytics worker %s loop error: %s",
                    self.worker_id,
                    exc,
                )
                if db.in_transaction():
                    db.rollback()
                should_sleep = True
            finally:
                db.close()

            if should_sleep and (stop_event is None or not stop_event.is_set()) and (
                max_iterations is None or iterations < max_iterations
            ):
                await asyncio.sleep(poll_interval)


async def run_standalone_worker(
    *,
    db_factory: Optional[Callable[[], Session]] = None,
    worker_id: Optional[str] = None,
    poll_interval: float = 1.0,
    reconcile_interval: float = 60.0,
    stop_event: Optional[asyncio.Event] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """Run an independent standalone AnalyticsWorker loop using SessionLocal by default."""
    if db_factory is None:
        from backend.app.database.session import SessionLocal

        factory = SessionLocal
    else:
        factory = db_factory

    worker = AnalyticsWorker(
        worker_id=worker_id,
        reconcile_interval_seconds=reconcile_interval,
    )
    await worker.run_worker_loop(
        db_factory=factory,
        poll_interval=poll_interval,
        reconcile_interval=reconcile_interval,
        stop_event=stop_event,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        asyncio.run(run_standalone_worker())
    except KeyboardInterrupt:
        logger.info("AnalyticsWorker stopped by interrupt.")
