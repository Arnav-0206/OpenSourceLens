"""Unit and Integration Tests for Phase C Step C.8.1:
- AnalyticsWorker single-job, drain, reconciliation, and async worker-loop behavior
- CollectionRun coverage_summary top-level + entity_status downstream compatibility
- C.8.1 API read/envelope schemas (error_category, interruption_reason, freshness envelope)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.exceptions import (
    GhostWorkerRejectedException,
    InvalidParameterException,
)
from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    CollectionRun,
    Commit,
    Contributor,
    Issue,
    PullRequest,
    Release,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    AnalysisDomainSliceResponse,
    AnalysisHistoryResponse,
    AnalysisLatestResponse,
    AnalysisPayload,
    AnalysisRecomputeRequest,
    AnalysisRecomputeResponse,
    AnalysisRunAttemptRead,
    AnalysisRunRead,
    AnalysisStatusResponse,
    AnalyticsAttemptRead,
    CURRENT_METHODOLOGY_VERSION,
    RepositoryFreshnessEnvelope,
)
from backend.app.schemas.ingestion import IngestionSummary
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.orchestrator import AnalyticsOrchestrator
from backend.app.services.analytics_queue import AnalyticsJobQueue
from backend.app.services.analytics_worker import AnalyticsWorker
from backend.app.services.ingestion_worker import IngestionWorker
from backend.app.services.job_queue import DurableJobQueue


@pytest.fixture
def in_memory_db():
    """Isolated in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionClass()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _seed_repo_with_entities(db: Session, repo_id: int = 8101) -> Repository:
    """Seed an unseen repository with minimal realistic entities across streams."""
    anchor = datetime(2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc)
    repo = Repository(
        id=repo_id,
        repo_name=f"worker-org/repo_{repo_id}",
        owner="worker-org",
        name=f"repo_{repo_id}",
        stars_count=420,
        forks_count=55,
        primary_language="Python",
        is_archived=False,
        is_fork=False,
        open_issues_count=10,
        last_synced_at=anchor,
        analytics_source_revision=2,
        is_active=True,
    )
    db.add(repo)
    db.add(
        CollectionRun(
            repository_id=repo_id,
            mode="standard_diagnostic",
            status="completed",
            is_partial=False,
            coverage_summary={
                "commits": "completed",
                "contributors": "completed",
                "pull_requests": "completed",
                "pr_reviews": "completed",
                "pr_inline_comments": "completed",
                "issues": "completed",
                "issue_comments": "completed",
                "releases": "completed",
                "entity_status": {
                    "commits": "completed",
                    "contributors": "completed",
                    "pull_requests": "completed",
                    "pr_reviews": "completed",
                    "pr_inline_comments": "completed",
                    "issues": "completed",
                    "issue_comments": "completed",
                    "releases": "completed",
                },
            },
            started_at=anchor - timedelta(hours=1),
            completed_at=anchor,
        )
    )
    db.add(
        Contributor(
            id=repo_id * 10 + 1,
            repository_id=repo_id,
            github_id=1001,
            login="maintainer_1",
            total_commits=12,
        )
    )
    for i in range(6):
        db.add(
            Commit(
                repository_id=repo_id,
                sha=f"sha_{repo_id}_{i}",
                author_login="maintainer_1",
                author_id=1001,
                message=f"Commit #{i}",
                author_date=anchor - timedelta(days=i * 3),
                committer_date=anchor - timedelta(days=i * 3),
                additions=10,
                deletions=2,
                total_changes=12,
            )
        )
    for i in range(4):
        db.add(
            PullRequest(
                id=repo_id * 100 + i,
                repository_id=repo_id,
                pr_number=i + 1,
                title=f"PR {i + 1}",
                state="closed",
                author="maintainer_1",
                author_id=1001,
                created_at=anchor - timedelta(days=20 - i * 3),
                closed_at=anchor - timedelta(days=19 - i * 3),
                merged_at=anchor - timedelta(days=19 - i * 3),
                additions=20,
                deletions=5,
                changed_files=2,
            )
        )
    for i in range(4):
        db.add(
            Issue(
                id=repo_id * 1000 + i,
                issue_number=i + 1,
                repository_id=repo_id,
                repo_name=repo.repo_name,
                title=f"Issue {i + 1}",
                author="user_x",
                state="closed" if i % 2 == 0 else "open",
                created_at=anchor - timedelta(days=25 - i * 4),
                closed_at=(anchor - timedelta(days=23 - i * 4)) if i % 2 == 0 else None,
            )
        )
    db.add(
        Release(
            id=repo_id * 10 + 1,
            repository_id=repo_id,
            release_id=9900 + repo_id,
            tag_name="v0.1.0",
            name="Release v0.1.0",
            is_prerelease=False,
            published_at=anchor - timedelta(days=10),
        )
    )
    db.commit()
    return repo


class TestAnalyticsV2SchemasC8:
    """Verify C.8.1 schema additions for attempt diagnostics and outer freshness envelopes."""

    def test_attempt_read_exposes_error_category_and_interruption_reason(self, in_memory_db):
        repo = _seed_repo_with_entities(in_memory_db, repo_id=8101)
        now = datetime.now(timezone.utc)
        run = AnalysisRun(
            repository_id=repo.id,
            status="failed",
            requested_source_revision=repo.analytics_source_revision,
            requested_methodology_version=CURRENT_METHODOLOGY_VERSION,
            started_at=now - timedelta(seconds=10),
            completed_at=now,
            error_message="[NON_RETRYABLE] InvalidParameterException: bad revision",
            lease_version=1,
            retry_count=0,
            max_retries=3,
        )
        in_memory_db.add(run)
        in_memory_db.flush()

        att_failed = AnalysisRunAttempt(
            analysis_run_id=run.id,
            attempt_number=1,
            worker_id="w_schema:1",
            outcome="failed",
            error_category="non_retryable",
            error_message="Bad contract",
            interruption_reason=None,
            started_at=now - timedelta(seconds=9),
            completed_at=now - timedelta(seconds=5),
        )
        att_interrupted = AnalysisRunAttempt(
            analysis_run_id=run.id,
            attempt_number=2,
            worker_id="w_schema:2",
            outcome="interrupted",
            error_category="stale_revision",
            error_message="Superseded",
            interruption_reason="stale_revision_superseded",
            started_at=now - timedelta(seconds=4),
            completed_at=now,
        )
        in_memory_db.add_all([att_failed, att_interrupted])
        in_memory_db.commit()
        in_memory_db.refresh(run)

        assert AnalyticsAttemptRead is AnalysisRunAttemptRead
        read_failed = AnalyticsAttemptRead.model_validate(att_failed)
        assert read_failed.error_category == "non_retryable"
        assert read_failed.interruption_reason is None

        read_run = AnalysisRunRead.model_validate(run)
        assert len(read_run.attempts) == 2
        assert read_run.attempts[0].error_category == "non_retryable"
        assert read_run.attempts[1].interruption_reason == "stale_revision_superseded"

    def test_api_envelope_schemas_preserve_freshness_separation(self, in_memory_db):
        repo = _seed_repo_with_entities(in_memory_db, repo_id=8102)
        orch = AnalyticsOrchestrator()
        exec_res = orch.enqueue_and_process(repository_id=repo.id, db=in_memory_db, worker_id="w_env")
        assert exec_res.published is True

        run = in_memory_db.get(AnalysisRun, exec_res.run_id)
        in_memory_db.refresh(repo)

        # Confirm metrics_payload remains strictly 5-key AnalysisPayload with zero last_synced_at
        validated_payload = AnalysisPayload(**run.metrics_payload)
        dumped_payload = validated_payload.model_dump()
        assert set(dumped_payload.keys()) == {
            "development_activity",
            "contributor_ecosystem",
            "pull_request_intelligence",
            "issue_intelligence",
            "repository_synthesis",
        }
        assert "last_synced_at" not in dumped_payload

        freshness = RepositoryFreshnessEnvelope(
            repository_id=repo.id,
            repo_name=repo.repo_name,
            analytics_source_revision=repo.analytics_source_revision,
            last_synced_at=repo.last_synced_at,
            last_analyzed_at=repo.last_analyzed_at,
            primary_language=repo.primary_language,
            stars_count=repo.stars_count,
            forks_count=repo.forks_count,
            is_archived=repo.is_archived,
        )
        run_read = AnalysisRunRead.model_validate(run)

        req = AnalysisRecomputeRequest()
        assert req.force is False
        assert req.requested_methodology_version == CURRENT_METHODOLOGY_VERSION

        recompute_resp = AnalysisRecomputeResponse(
            repository=freshness,
            run=run_read,
            enqueued=False,
            cached=True,
            message="Up-to-date completed analysis already exists.",
        )
        assert recompute_resp.repository.last_synced_at == repo.last_synced_at

        status_resp = AnalysisStatusResponse(
            repository=freshness,
            has_analysis=True,
            is_stale=False,
            current_run=run_read,
            latest_completed_run_id=run.id,
        )
        assert status_resp.latest_completed_run_id == run.id

        latest_resp = AnalysisLatestResponse(
            repository=freshness,
            is_stale=False,
            run=run_read,
        )
        assert latest_resp.run.metrics_payload is not None

        hist_resp = AnalysisHistoryResponse(
            repository=freshness,
            total=1,
            limit=20,
            offset=0,
            items=[run_read],
        )
        assert hist_resp.total == 1

        domain_resp = AnalysisDomainSliceResponse(
            repository=freshness,
            run_id=run.id,
            domain="release_evolution",
            analytics_source_revision=run.analytics_source_revision,
            methodology_version=run.methodology_version,
            analysis_reference_time=run.analysis_reference_time,
            is_stale=False,
            data=run.metrics_payload["repository_synthesis"]["release_evolution"],
            coverage=run.coverage_summary["release_evolution"],
            confidence_breakdown=run.confidence_summary["domain_breakdown"]["release_evolution"],
            rules=[],
        )
        assert domain_resp.domain == "release_evolution"


class TestAnalyticsWorkerBehavior:
    """Verify AnalyticsWorker reuses existing queue & orchestrator cleanly across lifecycle scenarios."""

    def test_worker_reuses_single_queue_and_processes_job(self, in_memory_db):
        repo = _seed_repo_with_entities(in_memory_db, repo_id=8201)
        shared_queue = AnalyticsJobQueue()
        worker = AnalyticsWorker(worker_id="aw_unit_1", job_queue=shared_queue)

        assert worker.job_queue is shared_queue
        assert worker.orchestrator.queue is shared_queue

        # Empty queue returns None
        assert worker.process_one_job(db=in_memory_db) is None

        run, is_new = shared_queue.enqueue_analysis(repo_id=repo.id, db=in_memory_db)
        assert is_new is True

        outcome = worker.process_one_job(db=in_memory_db)
        assert outcome is not None
        assert outcome.run_id == run.id
        assert outcome.status == "completed"
        assert outcome.published is True

        persisted = in_memory_db.get(AnalysisRun, run.id)
        assert persisted.status == "completed"
        assert persisted.analytics_source_revision == 2
        assert persisted.metrics_payload is not None

    def test_worker_validates_positive_lease_and_max_jobs(self, in_memory_db):
        with pytest.raises(InvalidParameterException):
            AnalyticsWorker(lease_duration_seconds=0)

        worker = AnalyticsWorker(worker_id="aw_param")
        with pytest.raises(InvalidParameterException):
            worker.drain_queue(db=in_memory_db, max_jobs=0)

    def test_worker_reconciliation_recovers_expired_lease_and_missing_jobs(self, in_memory_db):
        repo_expired = _seed_repo_with_entities(in_memory_db, repo_id=8202)
        repo_missing = _seed_repo_with_entities(in_memory_db, repo_id=8203)

        now = datetime.now(timezone.utc)
        # Seed an expired running lease on repo_expired
        expired_run = AnalysisRun(
            repository_id=repo_expired.id,
            status="running",
            worker_id="dead_analytics_worker",
            lease_version=1,
            leased_until=now - timedelta(minutes=2),
            requested_source_revision=repo_expired.analytics_source_revision,
            requested_methodology_version=CURRENT_METHODOLOGY_VERSION,
            started_at=now - timedelta(minutes=12),
            retry_count=0,
            max_retries=3,
        )
        in_memory_db.add(expired_run)
        in_memory_db.flush()
        expired_attempt = AnalysisRunAttempt(
            analysis_run_id=expired_run.id,
            attempt_number=1,
            worker_id="dead_analytics_worker",
            outcome="running",
            started_at=now - timedelta(minutes=12),
        )
        in_memory_db.add(expired_attempt)
        in_memory_db.commit()

        # repo_missing is active at rev=2 and has zero AnalysisRuns yet
        worker = AnalyticsWorker(worker_id="aw_reconciler")
        summary = worker.reconcile_jobs(db=in_memory_db)
        assert summary["expired_leases_recovered"] == 1
        assert summary["missing_jobs_enqueued"] == 1

        in_memory_db.refresh(expired_run)
        in_memory_db.refresh(expired_attempt)
        assert expired_run.status == "pending"
        assert expired_run.retry_count == 1
        assert expired_attempt.outcome == "lease_expired"

        # Drain queue processes both the recovered job and the newly enqueued missing job
        results = worker.drain_queue(db=in_memory_db, max_jobs=10)
        assert len(results) == 2
        assert all(r.status == "completed" and r.published is True for r in results)

    def test_worker_handles_non_retryable_failure_terminally(self, in_memory_db):
        repo = _seed_repo_with_entities(in_memory_db, repo_id=8204)

        class _NonRetryableCalc(BaseMetricCalculator):
            @property
            def domain_name(self) -> str:
                return "development_activity"

            def calculate(self, repository_id, db, analysis_reference_time, snapshot_revision):
                raise InvalidParameterException("Unsupported domain invariant", parameter="repo")

        orchestrator = AnalyticsOrchestrator(
            calculators={
                "development_activity": _NonRetryableCalc(),
                "contributor_ecosystem": None,
                "pull_request_intelligence": None,
                "issue_intelligence": None,
                "release_evolution": None,
                "repository_synthesis": None,
            }
        )
        worker = AnalyticsWorker(worker_id="aw_nonretry", orchestrator=orchestrator)
        run, _ = worker.job_queue.enqueue_analysis(repo_id=repo.id, db=in_memory_db, max_retries=3)

        res = worker.process_one_job(db=in_memory_db)
        assert res is not None
        assert res.status == "failed"
        assert res.failure_category == "non_retryable"
        assert res.published is False

        persisted = in_memory_db.get(AnalysisRun, run.id)
        assert persisted.status == "failed"
        assert persisted.retry_count == 0
        assert persisted.attempts[0].error_category == "non_retryable"
        assert worker.process_one_job(db=in_memory_db) is None

    def test_run_worker_loop_processes_jobs_and_survives_ghost_exception(self, tmp_path):
        db_file = tmp_path / "analytics_worker_loop.db"
        engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        SessionLocalTest = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        seed_db = SessionLocalTest()
        repo = _seed_repo_with_entities(seed_db, repo_id=8205)
        repo_id = repo.id
        seed_db.close()

        class _FlakyThenSuccessWorker(AnalyticsWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._injected_ghost = False

            def process_one_job(self, db: Session, *, reconcile=None):
                if not self._injected_ghost:
                    self._injected_ghost = True
                    raise GhostWorkerRejectedException("Simulated lease contention in loop")
                return super().process_one_job(db=db, reconcile=reconcile)

        async def _run():
            stop_event = asyncio.Event()
            worker = _FlakyThenSuccessWorker(
                worker_id="aw_loop",
                reconcile_interval_seconds=0.0,
            )

            async def _stop_when_done():
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    check_db = SessionLocalTest()
                    done = check_db.scalar(
                        select(AnalysisRun).where(
                            AnalysisRun.repository_id == repo_id,
                            AnalysisRun.status == "completed",
                        )
                    )
                    check_db.close()
                    if done is not None:
                        stop_event.set()
                        return
                stop_event.set()

            await asyncio.gather(
                worker.run_worker_loop(
                    db_factory=SessionLocalTest,
                    poll_interval=0.01,
                    reconcile_interval=30.0,
                    stop_event=stop_event,
                    max_iterations=20,
                ),
                _stop_when_done(),
            )

        asyncio.run(_run())

        verify_db = SessionLocalTest()
        completed_runs = verify_db.scalars(
            select(AnalysisRun).where(
                AnalysisRun.repository_id == repo_id,
                AnalysisRun.status == "completed",
            )
        ).all()
        assert len(completed_runs) == 1
        assert completed_runs[0].metrics_payload is not None
        verify_db.close()
        engine.dispose()


class TestCoverageCompatibilityWithIngestionAndCalculators:
    """Verify IngestionWorker -> CollectionRun coverage_summary feeds C.4/C.6/C.7 accurately."""

    def test_ingestion_coverage_summary_propagates_skipped_and_truncated_streams_to_analytics(
        self, in_memory_db
    ):
        repo = _seed_repo_with_entities(in_memory_db, repo_id=8301)
        # Remove seeded CollectionRun so only IngestionWorker-finalized CollectionRun exists
        in_memory_db.query(CollectionRun).filter(CollectionRun.repository_id == repo.id).delete()
        in_memory_db.commit()

        async def _run():
            ing_queue = DurableJobQueue()
            job, _ = ing_queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=in_memory_db)
            ing_worker = IngestionWorker(
                worker_id="ing_cov_test:1",
                job_queue=ing_queue,
                heartbeat_interval_seconds=0.2,
            )
            mock_collector = MagicMock()
            mock_collector.collect_repository = AsyncMock(
                return_value=IngestionSummary(
                    repo_name=repo.repo_name,
                    mode="fast_snapshot",
                    is_partial=True,
                    partial_reason="fast_snapshot capped inline comments and reviews",
                    total_requests_consumed=9,
                    items_collected={
                        "repository": 1,
                        "commits": 6,
                        "contributors": 1,
                        "pull_requests": 4,
                        "pr_reviews": 0,
                        "pr_inline_comments": 0,
                        "issues": 4,
                        "issue_comments": 0,
                        "releases": 1,
                    },
                    entity_status={
                        "repository": "completed",
                        "commits": "completed",
                        "contributors": "completed",
                        "pull_requests": "completed",
                        "pr_reviews": "truncated",
                        "pr_inline_comments": "skipped",
                        "issues": "completed",
                        "issue_comments": "skipped",
                        "releases": "completed",
                    },
                    execution_time_seconds=0.8,
                    errors=[],
                )
            )
            await ing_worker.process_one_job(db=in_memory_db, collector=mock_collector)
            return job

        job = asyncio.run(_run())
        in_memory_db.refresh(job)
        assert job.status == "completed_partial"
        # Both nested entity_status and top-level streams are present cleanly
        assert job.coverage_summary["pr_inline_comments"] == "skipped"
        assert job.coverage_summary["entity_status"]["pr_inline_comments"] == "skipped"

        # Now run analytics worker on this repository; C.4 & C.6 must read skipped/truncated streams
        worker = AnalyticsWorker(worker_id="aw_cov_compat")
        run, _ = worker.job_queue.enqueue_analysis(
            repo_id=repo.id,
            db=in_memory_db,
            collection_run_id=job.id,
        )
        res = worker.process_one_job(db=in_memory_db)
        assert res is not None
        assert res.status == "completed"
        assert res.published is True

        persisted = in_memory_db.get(AnalysisRun, run.id)
        pr_scope = persisted.metrics_payload["pull_request_intelligence"]["scope"]
        assert pr_scope["pr_reviews_stream_status"] == "truncated"
        assert pr_scope["pr_inline_comments_stream_status"] == "skipped"
        assert pr_scope["inline_comments_skipped_by_mode"] is True

        pair02_cov = persisted.coverage_summary["summary"]["cross_domain_correlations"]["pair02_collection_coverage"]
        assert pair02_cov["stream_statuses"]["pr_reviews"] == "truncated"
        assert pair02_cov["stream_statuses"]["pr_inline_comments"] == "skipped"
        assert pair02_cov["is_representative"] is False
        assert pair02_cov["suppress_inferential"] is True
        assert "pr_reviews" in pair02_cov["truncated_streams"]
        assert "pr_inline_comments" in pair02_cov["skipped_streams"]
