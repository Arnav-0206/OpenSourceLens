"""Comprehensive Integration & Reliability Tests for Phase C Step C.8.2:
Post-Ingestion Analytics Enqueue Wiring in IngestionWorker.

Covers all 12 required scenarios on isolated databases (zero offline benchmark DB mutation):
1. Successful ingestion automatically enqueues analytics linked to collection_run_id.
2. Clean completed_partial ingestion enqueues analytics with partial/truncated coverage preserved.
3. Failed ingestion (collector exception) does not enqueue analytics.
4. Non-empty summary.errors marks collection failed and does not enqueue analytics.
5. Rate-limited ingestion (resumes_at set) does not enqueue analytics.
6. Lease-revoked ingestion worker does not enqueue analytics.
7. Analytics enqueue failure preserves committed ingestion data, keeps CollectionRun completed,
   logs diagnostics, records failure state cleanly, and allows later reconciliation recovery.
8. Zero-mutation incremental ingestion suppresses redundant analytics recomputation.
9. Concurrent duplicate ingestion/enqueue reuses single active AnalysisRun without errors.
10. New source revisions from mutating re-ingestion trigger new analytics runs.
11. Stale analysis replacement triggers cleanly when ingestion commits newer data during analysis.
12. Auto-enqueue opt-out and missing-repository handling preserve Phase B parity.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    CollectionRun,
    Commit,
    Contributor,
    Issue,
    PullRequest,
    Release,
    Repository,
)
from backend.app.schemas.ingestion import IngestionSummary, PaginatedResult
from backend.app.services.analytics_queue import AnalyticsJobQueue
from backend.app.services.analytics_worker import AnalyticsWorker
from backend.app.services.ingestion_worker import IngestionWorker
from backend.app.services.job_queue import DurableJobQueue, JobLeaseLostException
from backend.app.services.repository_collector import RepositoryCollector


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


def _seed_repo(
    db: Session,
    *,
    repo_id: int = 9401,
    source_revision: int = 2,
) -> Repository:
    repo = Repository(
        id=repo_id,
        repo_name=f"wiring-org/repo_{repo_id}",
        owner="wiring-org",
        name=f"repo_{repo_id}",
        stars_count=250,
        forks_count=30,
        open_issues_count=5,
        primary_language="Python",
        analytics_source_revision=source_revision,
        is_active=True,
    )
    db.add(repo)
    db.commit()
    return repo


def _seed_entities_for_repo(db: Session, repo: Repository, *, anchor: datetime) -> None:
    db.add(
        Contributor(
            repository_id=repo.id,
            github_id=7701,
            login="author_1",
            total_commits=5,
        )
    )
    for i in range(5):
        db.add(
            Commit(
                repository_id=repo.id,
                sha=f"c_{repo.id}_{i}",
                author_login="author_1",
                author_id=7701,
                message=f"feat #{i}",
                author_date=anchor - timedelta(days=i * 2),
                committer_date=anchor - timedelta(days=i * 2),
                additions=15,
                deletions=3,
                total_changes=18,
            )
        )
    for i in range(3):
        db.add(
            PullRequest(
                repository_id=repo.id,
                pr_number=i + 1,
                title=f"PR #{i + 1}",
                state="closed",
                author="author_1",
                author_id=7701,
                created_at=anchor - timedelta(days=12 - i * 2),
                closed_at=anchor - timedelta(days=11 - i * 2),
                merged_at=anchor - timedelta(days=11 - i * 2),
                additions=20,
                deletions=4,
                changed_files=2,
            )
        )
    for i in range(3):
        db.add(
            Issue(
                id=repo.id * 100 + i,
                issue_number=i + 1,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                title=f"Issue #{i + 1}",
                author="reporter",
                state="closed" if i == 0 else "open",
                created_at=anchor - timedelta(days=15 - i * 3),
                closed_at=(anchor - timedelta(days=14 - i * 3)) if i == 0 else None,
            )
        )
    db.add(
        Release(
            repository_id=repo.id,
            release_id=repo.id * 1000 + 1,
            tag_name="v1.0.0",
            name="v1.0.0",
            is_prerelease=False,
            published_at=anchor - timedelta(days=5),
        )
    )
    db.commit()


def _clean_summary(repo_name: str, *, mode: str = "standard_diagnostic") -> IngestionSummary:
    return IngestionSummary(
        repo_name=repo_name,
        mode=mode,
        is_partial=False,
        partial_reason=None,
        resumes_at=None,
        total_requests_consumed=12,
        items_collected={
            "repository": 1,
            "commits": 5,
            "contributors": 1,
            "pull_requests": 3,
            "pr_reviews": 0,
            "pr_inline_comments": 0,
            "issues": 3,
            "issue_comments": 0,
            "releases": 1,
        },
        entity_status={
            "repository": "completed",
            "commits": "completed",
            "contributors": "completed",
            "pull_requests": "completed",
            "pr_reviews": "completed",
            "pr_inline_comments": "completed" if mode == "standard_diagnostic" else "skipped",
            "issues": "completed",
            "issue_comments": "completed",
            "releases": "completed",
        },
        records_mutated=13,
        execution_time_seconds=0.5,
        errors=[],
    )


def _mock_collector_returning(summary: IngestionSummary) -> MagicMock:
    collector = MagicMock(spec=RepositoryCollector)
    collector.collect_repository = AsyncMock(return_value=summary)
    return collector


def _make_mock_github_client(
    *,
    repo_id: int,
    repo_name: str,
    commits_payload: List[Dict[str, Any]],
) -> MagicMock:
    owner, short_name = repo_name.split("/")
    client = MagicMock()
    client.is_primary_rate_limited = False
    client.resumes_at = None

    meta_json = {
        "id": repo_id,
        "name": short_name,
        "full_name": repo_name,
        "description": "Integration wiring test repo",
        "language": "Python",
        "stargazers_count": 250,
        "forks_count": 30,
        "open_issues_count": 5,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2026-04-01T00:00:00Z",
        "archived": False,
        "fork": False,
    }

    async def _request(method: str, endpoint_or_url: str, **kwargs):
        resp = MagicMock()
        if endpoint_or_url == f"/repos/{repo_name}":
            resp.json.return_value = meta_json
        elif "/commits/" in endpoint_or_url:
            sha = endpoint_or_url.rsplit("/", 1)[-1]
            resp.json.return_value = {
                "sha": sha,
                "stats": {"additions": 10, "deletions": 2, "total": 12},
            }
        else:
            resp.json.return_value = {}
        return resp

    async def _paginate(endpoint_or_url: str = "", entity: str = "", **kwargs):
        if entity == "commits":
            return PaginatedResult(
                entity="commits",
                items=list(commits_payload),
                total_requests=1,
                is_partial=False,
            )
        return PaginatedResult(
            entity=entity,
            items=[],
            total_requests=1,
            is_partial=False,
        )

    client.request = AsyncMock(side_effect=_request)
    client.paginate = AsyncMock(side_effect=_paginate)
    client.close = AsyncMock()
    return client


class TestPostIngestionAnalyticsTriggers:
    """Tests 1–6: Trigger eligibility, partial preservation, and negative suppression gates."""

    def test_01_successful_ingestion_enqueues_analytics_and_links_collection_run(
        self, in_memory_db
    ):
        anchor = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        repo = _seed_repo(in_memory_db, repo_id=9401, source_revision=2)
        _seed_entities_for_repo(in_memory_db, repo, anchor=anchor)

        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job, _ = ing_queue.submit_job(repo_id=repo.id, mode="standard_diagnostic", db=in_memory_db)

        ing_worker = IngestionWorker(
            worker_id="w_ing_01",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=0.2,
        )
        collector = _mock_collector_returning(_clean_summary(repo.repo_name))

        processed_id = asyncio.run(ing_worker.process_one_job(db=in_memory_db, collector=collector))
        assert processed_id == job.id
        in_memory_db.refresh(job)
        assert job.status == "completed"

        assert ing_worker.last_analytics_enqueue_result is not None
        assert ing_worker.last_analytics_enqueue_result["status"] == "enqueued"
        assert ing_worker.last_analytics_enqueue_result["enqueued"] is True

        runs = in_memory_db.scalars(
            select(AnalysisRun).where(AnalysisRun.repository_id == repo.id)
        ).all()
        assert len(runs) == 1
        assert runs[0].status == "pending"
        assert runs[0].collection_run_id == job.id
        assert runs[0].requested_source_revision == 2

        # AnalyticsWorker executes the queued run cleanly
        ana_worker = AnalyticsWorker(worker_id="w_ana_01", job_queue=ana_queue)
        exec_res = ana_worker.process_one_job(db=in_memory_db)
        assert exec_res is not None
        assert exec_res.status == "completed"
        assert exec_res.published is True

        in_memory_db.refresh(runs[0])
        assert runs[0].status == "completed"
        assert runs[0].analytics_source_revision == 2
        assert set(runs[0].metrics_payload.keys()) == {
            "development_activity",
            "contributor_ecosystem",
            "pull_request_intelligence",
            "issue_intelligence",
            "repository_synthesis",
        }

    def test_02_completed_partial_ingestion_enqueues_analytics_with_partial_coverage_preserved(
        self, in_memory_db
    ):
        anchor = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        repo = _seed_repo(in_memory_db, repo_id=9402, source_revision=2)
        _seed_entities_for_repo(in_memory_db, repo, anchor=anchor)

        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job, _ = ing_queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=in_memory_db)

        partial_summary = IngestionSummary(
            repo_name=repo.repo_name,
            mode="fast_snapshot",
            is_partial=True,
            partial_reason="Entity budget ceiling reached for pr_reviews; inline comments skipped",
            resumes_at=None,
            total_requests_consumed=10,
            items_collected={
                "repository": 1,
                "commits": 5,
                "contributors": 1,
                "pull_requests": 3,
                "pr_reviews": 0,
                "pr_inline_comments": 0,
                "issues": 3,
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
            records_mutated=13,
            execution_time_seconds=0.7,
            errors=[],
        )

        ing_worker = IngestionWorker(
            worker_id="w_ing_02",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=0.2,
        )
        asyncio.run(
            ing_worker.process_one_job(
                db=in_memory_db,
                collector=_mock_collector_returning(partial_summary),
            )
        )

        in_memory_db.refresh(job)
        assert job.status == "completed_partial"
        assert job.is_partial is True

        ana_worker = AnalyticsWorker(worker_id="w_ana_02", job_queue=ana_queue)
        res = ana_worker.process_one_job(db=in_memory_db)
        assert res is not None and res.status == "completed" and res.published is True

        run = in_memory_db.get(AnalysisRun, res.run_id)
        assert run.collection_run_id == job.id
        pr_scope = run.metrics_payload["pull_request_intelligence"]["scope"]
        assert pr_scope["pr_reviews_stream_status"] == "truncated"
        assert pr_scope["pr_inline_comments_stream_status"] == "skipped"
        assert pr_scope["inline_comments_skipped_by_mode"] is True

        pair02 = run.coverage_summary["summary"]["cross_domain_correlations"]["pair02_collection_coverage"]
        assert pair02["is_representative"] is False
        assert pair02["suppress_inferential"] is True
        assert run.coverage_summary["summary"]["is_bounded_or_partial"] is True
        assert run.coverage_summary["summary"]["representativeness"] == "fast_snapshot_bounded"

    def test_03_failed_ingestion_exception_does_not_enqueue_analytics(self, in_memory_db):
        repo = _seed_repo(in_memory_db, repo_id=9403, source_revision=2)
        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job, _ = ing_queue.submit_job(repo_id=repo.id, mode="standard_diagnostic", db=in_memory_db)

        collector = MagicMock(spec=RepositoryCollector)
        collector.collect_repository = AsyncMock(side_effect=RuntimeError("GitHub 500 boom"))

        ing_worker = IngestionWorker(
            worker_id="w_ing_03",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=0.2,
        )
        with pytest.raises(RuntimeError, match="GitHub 500 boom"):
            asyncio.run(ing_worker.process_one_job(db=in_memory_db, collector=collector))

        in_memory_db.refresh(job)
        assert job.status == "failed"
        assert in_memory_db.scalars(select(AnalysisRun)).all() == []

    def test_04_non_empty_summary_errors_does_not_enqueue_analytics(self, in_memory_db):
        repo = _seed_repo(in_memory_db, repo_id=9404, source_revision=2)
        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job, _ = ing_queue.submit_job(repo_id=repo.id, mode="standard_diagnostic", db=in_memory_db)

        err_summary = IngestionSummary(
            repo_name=repo.repo_name,
            mode="standard_diagnostic",
            is_partial=False,
            resumes_at=None,
            total_requests_consumed=2,
            items_collected={"commits": 0},
            entity_status={"repository": "completed", "commits": "failed"},
            execution_time_seconds=0.2,
            errors=["Unexpected SQL integrity crash during collect_commits"],
        )

        ing_worker = IngestionWorker(
            worker_id="w_ing_04",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=0.2,
        )
        processed_id = asyncio.run(
            ing_worker.process_one_job(
                db=in_memory_db,
                collector=_mock_collector_returning(err_summary),
            )
        )
        assert processed_id == job.id
        in_memory_db.refresh(job)
        assert job.status == "failed"
        assert ing_worker.last_analytics_enqueue_result["status"] == "skipped_collector_errors"
        assert in_memory_db.scalars(select(AnalysisRun)).all() == []

    def test_05_rate_limited_ingestion_does_not_enqueue_analytics(self, in_memory_db):
        repo = _seed_repo(in_memory_db, repo_id=9405, source_revision=2)
        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job, _ = ing_queue.submit_job(repo_id=repo.id, mode="standard_diagnostic", db=in_memory_db)

        rate_summary = IngestionSummary(
            repo_name=repo.repo_name,
            mode="standard_diagnostic",
            is_partial=True,
            partial_reason="Primary rate limit reached",
            resumes_at=datetime.now(timezone.utc) + timedelta(minutes=20),
            total_requests_consumed=9,
            items_collected={"commits": 2},
            entity_status={"repository": "completed", "commits": "truncated"},
            execution_time_seconds=0.3,
            errors=[],
        )

        ing_worker = IngestionWorker(
            worker_id="w_ing_05",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=0.2,
        )
        asyncio.run(
            ing_worker.process_one_job(
                db=in_memory_db,
                collector=_mock_collector_returning(rate_summary),
            )
        )

        in_memory_db.refresh(job)
        assert job.status == "rate_limited"
        assert ing_worker.last_analytics_enqueue_result["status"] == "skipped_rate_limited"
        assert in_memory_db.scalars(select(AnalysisRun)).all() == []

    def test_06_lease_revoked_ingestion_does_not_enqueue_analytics(self, in_memory_db):
        repo = _seed_repo(in_memory_db, repo_id=9406, source_revision=2)
        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job, _ = ing_queue.submit_job(repo_id=repo.id, mode="standard_diagnostic", db=in_memory_db)

        async def _steal_lease(*args, **kwargs):
            job.lease_version = 77
            job.worker_id = "other_ing_worker"
            in_memory_db.commit()
            return _clean_summary(repo.repo_name)

        collector = MagicMock(spec=RepositoryCollector)
        collector.collect_repository = AsyncMock(side_effect=_steal_lease)

        ing_worker = IngestionWorker(
            worker_id="w_ing_06",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=5.0,
        )

        with pytest.raises(JobLeaseLostException):
            asyncio.run(ing_worker.process_one_job(db=in_memory_db, collector=collector))

        in_memory_db.refresh(job)
        assert job.lease_version == 77
        assert job.status == "in_progress"
        assert in_memory_db.scalars(select(AnalysisRun)).all() == []


class TestPostIngestionErrorIsolationAndRevisions:
    """Tests 7–12: Failure isolation, no-op suppression, concurrent dedup, revisions & stale replacement."""

    def test_07_analytics_enqueue_failure_preserves_completed_ingestion_and_allows_reconciliation(
        self, tmp_path, caplog
    ):
        db_file = tmp_path / "enqueue_isolation.db"
        engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionClass()

        try:
            repo = _seed_repo(db, repo_id=9407, source_revision=1)
            ing_queue = DurableJobQueue()
            job, _ = ing_queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=db)

            mock_client = _make_mock_github_client(
                repo_id=repo.id,
                repo_name=repo.repo_name,
                commits_payload=[
                    {
                        "sha": "iso_sha_1",
                        "commit": {
                            "author": {"name": "Dev", "email": "dev@wiring.org", "date": "2026-03-10T10:00:00Z"},
                            "committer": {"date": "2026-03-10T10:00:00Z"},
                            "message": "Initial commit",
                        },
                        "author": {"id": 8801, "login": "dev_wiring"},
                    }
                ],
            )
            real_collector = RepositoryCollector(client=mock_client)

            failing_ana_queue = MagicMock(spec=AnalyticsJobQueue)
            failing_ana_queue.enqueue_analysis.side_effect = RuntimeError("Analytics queue lock contention")

            ing_worker = IngestionWorker(
                worker_id="w_ing_iso",
                job_queue=ing_queue,
                analytics_queue=failing_ana_queue,
                heartbeat_interval_seconds=5.0,
            )

            # Guard against earlier Alembic fileConfig(..., disable_existing_loggers=True) in full suites
            iw_logger = logging.getLogger("backend.app.services.ingestion_worker")
            prev_disabled = iw_logger.disabled
            iw_logger.disabled = False
            try:
                with caplog.at_level(logging.ERROR, logger="backend.app.services.ingestion_worker"):
                    processed_id = asyncio.run(
                        ing_worker.process_one_job(db=db, client=mock_client, collector=real_collector)
                    )
            finally:
                iw_logger.disabled = prev_disabled

            assert processed_id == job.id
            db.refresh(job)
            db.refresh(repo)

            # CollectionRun and harvested entity mutations stayed durably committed
            assert job.status == "completed"
            assert repo.analytics_source_revision == 2
            assert db.scalar(select(Commit).where(Commit.sha == "iso_sha_1")) is not None

            # Failure was logged and exposed cleanly on worker state
            assert ing_worker.last_analytics_enqueue_result["status"] == "enqueue_failed"
            assert "Analytics queue lock contention" in ing_worker.last_analytics_enqueue_result["error"]
            assert "Analytics enqueue failed after completed ingestion" in caplog.text
            assert db.scalars(select(AnalysisRun)).all() == []

            # Later reconciliation recovers missing analysis cleanly
            ana_worker = AnalyticsWorker(worker_id="w_ana_rec")
            rec_summary = ana_worker.reconcile_jobs(db=db)
            assert rec_summary["missing_jobs_enqueued"] == 1
            recovered = db.scalars(select(AnalysisRun).where(AnalysisRun.repository_id == repo.id)).all()
            assert len(recovered) == 1
            assert recovered[0].status == "pending"
            assert recovered[0].requested_source_revision == 2
        finally:
            db.close()
            engine.dispose()

    def test_08_and_10_zero_mutation_noop_vs_new_source_revision_reingestion(self, in_memory_db):
        repo = _seed_repo(in_memory_db, repo_id=9408, source_revision=1)
        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        ing_worker = IngestionWorker(
            worker_id="w_ing_rev",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=5.0,
        )
        ana_worker = AnalyticsWorker(worker_id="w_ana_rev", job_queue=ana_queue)

        commit_1 = {
            "sha": "rev_sha_1",
            "commit": {
                "author": {"name": "Dev", "email": "dev@wiring.org", "date": "2026-03-10T10:00:00Z"},
                "committer": {"date": "2026-03-10T10:00:00Z"},
                "message": "Commit 1",
            },
            "author": {"id": 8802, "login": "dev_rev"},
        }

        # Ingestion 1: Inserts commit_1 -> rev advances 1 -> 2 -> auto-enqueues Run #1
        client_1 = _make_mock_github_client(repo_id=repo.id, repo_name=repo.repo_name, commits_payload=[commit_1])
        job1, _ = ing_queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=in_memory_db)
        asyncio.run(
            ing_worker.process_one_job(
                db=in_memory_db,
                client=client_1,
                collector=RepositoryCollector(client=client_1),
            )
        )
        in_memory_db.refresh(repo)
        assert repo.analytics_source_revision == 2
        assert ing_worker.last_analytics_enqueue_result["enqueued"] is True

        exec_1 = ana_worker.process_one_job(db=in_memory_db)
        assert exec_1 is not None and exec_1.status == "completed"
        assert exec_1.snapshot_revision == 2

        # Test 8: Incremental ingestion with identical payload -> 0 mutations -> rev stays 2 -> NO new AnalysisRun
        client_2 = _make_mock_github_client(repo_id=repo.id, repo_name=repo.repo_name, commits_payload=[commit_1])
        job2, _ = ing_queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=in_memory_db)
        asyncio.run(
            ing_worker.process_one_job(
                db=in_memory_db,
                client=client_2,
                collector=RepositoryCollector(client=client_2),
            )
        )
        in_memory_db.refresh(repo)
        assert repo.analytics_source_revision == 2
        assert ing_worker.last_analytics_enqueue_result["enqueued"] is False
        assert ing_worker.last_analytics_enqueue_result["status"] == "reused_completed"
        assert ing_worker.last_analytics_enqueue_result["analysis_run_id"] == exec_1.run_id
        assert len(in_memory_db.scalars(select(AnalysisRun)).all()) == 1

        # Test 10: New commit added -> rev advances 2 -> 3 -> auto-enqueues Run #2 at rev 3!
        commit_2 = {
            "sha": "rev_sha_2",
            "commit": {
                "author": {"name": "Dev", "email": "dev@wiring.org", "date": "2026-03-12T10:00:00Z"},
                "committer": {"date": "2026-03-12T10:00:00Z"},
                "message": "Commit 2",
            },
            "author": {"id": 8802, "login": "dev_rev"},
        }
        client_3 = _make_mock_github_client(
            repo_id=repo.id,
            repo_name=repo.repo_name,
            commits_payload=[commit_1, commit_2],
        )
        job3, _ = ing_queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=in_memory_db)
        asyncio.run(
            ing_worker.process_one_job(
                db=in_memory_db,
                client=client_3,
                collector=RepositoryCollector(client=client_3),
            )
        )
        in_memory_db.refresh(repo)
        assert repo.analytics_source_revision == 3
        assert ing_worker.last_analytics_enqueue_result["enqueued"] is True

        exec_2 = ana_worker.process_one_job(db=in_memory_db)
        assert exec_2 is not None and exec_2.status == "completed"
        assert exec_2.run_id != exec_1.run_id
        assert exec_2.snapshot_revision == 3

        all_runs = in_memory_db.scalars(
            select(AnalysisRun).where(AnalysisRun.repository_id == repo.id).order_by(AnalysisRun.id.asc())
        ).all()
        assert len(all_runs) == 2
        assert [r.analytics_source_revision for r in all_runs] == [2, 3]
        assert [r.collection_run_id for r in all_runs] == [job1.id, job3.id]

    def test_09_concurrent_duplicate_enqueue_reuses_single_active_analysis_run(self, tmp_path):
        db_file = tmp_path / "concurrent_enqueue_dedup.db"
        engine = create_engine(
            f"sqlite:///{db_file}",
            connect_args={"check_same_thread": False, "timeout": 15},
        )
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        seed_db = SessionClass()
        repo = _seed_repo(seed_db, repo_id=9409, source_revision=2)
        crun1 = CollectionRun(repository_id=repo.id, mode="fast_snapshot", status="completed")
        crun2 = CollectionRun(repository_id=repo.id, mode="fast_snapshot", status="completed")
        seed_db.add_all([crun1, crun2])
        seed_db.commit()
        crun1_id, crun2_id = crun1.id, crun2.id
        seed_db.close()

        def _enqueue_worker(c_id: int):
            s = SessionClass()
            try:
                q = AnalyticsJobQueue()
                run, is_new = q.enqueue_analysis(repo_id=9409, db=s, collection_run_id=c_id)
                return run.id, is_new
            finally:
                s.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_enqueue_worker, crun1_id)
            f2 = pool.submit(_enqueue_worker, crun2_id)
            id1, new1 = f1.result()
            id2, new2 = f2.result()

        assert id1 == id2
        assert {new1, new2} == {True, False}

        verify_db = SessionClass()
        runs = verify_db.scalars(select(AnalysisRun).where(AnalysisRun.repository_id == 9409)).all()
        assert len(runs) == 1
        assert runs[0].status == "pending"
        verify_db.close()
        engine.dispose()

    def test_11_stale_analysis_replacement_when_ingestion_completes_during_analysis(self, tmp_path):
        db_file = tmp_path / "stale_replacement_wiring.db"
        engine = create_engine(
            f"sqlite:///{db_file}",
            connect_args={"check_same_thread": False, "timeout": 15},
        )
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.commit()
        Base.metadata.create_all(bind=engine)
        SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        anchor = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        seed_db = SessionClass()
        repo = _seed_repo(seed_db, repo_id=9410, source_revision=2)
        _seed_entities_for_repo(seed_db, repo, anchor=anchor)
        ana_queue = AnalyticsJobQueue()
        run1, _ = ana_queue.enqueue_analysis(repo_id=repo.id, db=seed_db)
        run1_id = run1.id
        seed_db.close()

        # Worker opens snapshot at rev 2 and pauses inside domain calculation while ingestion #2 commits rev 3!
        worker_db = SessionClass()
        ing_db = SessionClass()

        def _simulate_mid_analysis_ingestion():
            ing_q = DurableJobQueue()
            job, _ = ing_q.submit_job(repo_id=9410, mode="fast_snapshot", db=ing_db)
            ing_worker = IngestionWorker(
                worker_id="w_ing_mid",
                job_queue=ing_q,
                analytics_queue=ana_queue,
                heartbeat_interval_seconds=5.0,
            )

            async def _collect_rev3(*args, **kwargs):
                r = ing_db.get(Repository, 9410)
                r.analytics_source_revision = 3
                ing_db.commit()
                return _clean_summary(r.repo_name, mode="fast_snapshot")

            coll = MagicMock(spec=RepositoryCollector)
            coll.collect_repository = AsyncMock(side_effect=_collect_rev3)
            asyncio.run(ing_worker.process_one_job(db=ing_db, collector=coll))
            # While run1 was active, post-ingestion enqueue reused run1 without duplicating
            assert ing_worker.last_analytics_enqueue_result["analysis_run_id"] == run1_id
            assert ing_worker.last_analytics_enqueue_result["enqueued"] is False

        from backend.app.services.analytics.development_activity import DevelopmentActivityCalculator
        from backend.app.services.analytics.orchestrator import AnalyticsOrchestrator

        orig_calc = DevelopmentActivityCalculator.calculate
        triggered = {"done": False}

        def _hooked_calc(self_calc, repository_id, db, analysis_reference_time, snapshot_revision):
            if not triggered["done"]:
                triggered["done"] = True
                _simulate_mid_analysis_ingestion()
            return orig_calc(self_calc, repository_id, db, analysis_reference_time, snapshot_revision)

        with patch.object(DevelopmentActivityCalculator, "calculate", _hooked_calc):
            ana_worker = AnalyticsWorker(
                worker_id="w_ana_mid",
                orchestrator=AnalyticsOrchestrator(job_queue=ana_queue),
            )
            res1 = ana_worker.process_one_job(db=worker_db)

        assert res1 is not None
        assert res1.run_id == run1_id
        assert res1.status == "completed_stale_replacement_enqueued"
        assert res1.stale_revision_detected is True
        assert res1.replacement_run_enqueued is True

        worker_db.expire_all()
        repo_after_r1 = worker_db.get(Repository, 9410)
        assert repo_after_r1.last_analyzed_at is None  # Stale rev 2 must not mark rev 3 analyzed!

        # Next AnalyticsWorker cycle drains replacement Run #2 at rev 3!
        res2 = ana_worker.process_one_job(db=worker_db)
        assert res2 is not None
        assert res2.run_id != run1_id
        assert res2.status == "completed"
        assert res2.snapshot_revision == 3

        worker_db.refresh(repo_after_r1)
        assert repo_after_r1.last_analyzed_at is not None

        worker_db.close()
        ing_db.close()
        engine.dispose()

    def test_12_opt_out_and_missing_repository_preserve_phase_b_parity(self, in_memory_db):
        repo = _seed_repo(in_memory_db, repo_id=9411, source_revision=2)
        ing_queue = DurableJobQueue()
        ana_queue = AnalyticsJobQueue()
        job1, _ = ing_queue.submit_job(repo_id=repo.id, mode="standard_diagnostic", db=in_memory_db)

        worker_no_auto = IngestionWorker(
            worker_id="w_ing_optout",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            auto_enqueue_analytics=False,
            heartbeat_interval_seconds=0.2,
        )
        asyncio.run(
            worker_no_auto.process_one_job(
                db=in_memory_db,
                collector=_mock_collector_returning(_clean_summary(repo.repo_name)),
            )
        )
        in_memory_db.refresh(job1)
        assert job1.status == "completed"
        assert worker_no_auto.last_analytics_enqueue_result["status"] == "disabled"
        assert in_memory_db.scalars(select(AnalysisRun)).all() == []

        # Idle queue returns None cleanly
        worker_idle = IngestionWorker(
            worker_id="w_ing_idle",
            job_queue=ing_queue,
            analytics_queue=ana_queue,
            heartbeat_interval_seconds=0.2,
        )
        assert asyncio.run(worker_idle.process_one_job(db=in_memory_db)) is None

        # Orphaned CollectionRun with missing Repository fails cleanly without enqueueing analytics
        orphan_job = CollectionRun(
            repository_id=999999,
            mode="standard_diagnostic",
            status="pending",
            lease_version=0,
            retry_count=0,
            max_retries=2,
            is_partial=False,
            started_at=datetime.now(timezone.utc),
        )
        in_memory_db.add(orphan_job)
        in_memory_db.commit()
        processed_orphan = asyncio.run(worker_idle.process_one_job(db=in_memory_db))
        assert processed_orphan == orphan_job.id
        in_memory_db.refresh(orphan_job)
        assert orphan_job.status == "failed"
        assert in_memory_db.scalars(select(AnalysisRun)).all() == []
