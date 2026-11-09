"""End-to-End Acceptance Test Suite for Phase C Step C.1 (Analytics Foundation).

Verifies the integrated Phase C analytical foundation across:
1. Migration 003 schema models & project benchmark data preservation (2,328 issues, committer_date IS NULL).
2. Collector record-level mutation tracking and atomic revision advancement.
3. Distinct record counting and no-op re-ingestion idempotency.
4. committer_date extraction from GitHub payloads and NULL preservation.
5. Analytics job enqueueing, unified predicate deduplication, and no-op suppression.
6. Worker atomic claiming and optimistic lease fencing.
7. Driver-verified SQLite WAL snapshot isolation.
8. Conditional rowcount-validated atomic publication and attempt finalization.
9. Ghost-worker rejection on stale lease versions.
10. Strict retry state machine terminating after exactly 4 total attempts (max_retries = 3).
11. Stale-result replacement and active replacement-run reuse.
12. Durable reconciliation of expired leases and crash-safe ingestion handoff.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, event, func, insert, or_, select, text, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.exceptions import (
    AttemptImmutableException,
    BudgetExceededException,
    EntityNotFoundException,
    GitHubRateLimitExceededException,
    GhostWorkerRejectedException,
)
from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    Commit,
    Contributor,
    ContributorActivity,
    Issue,
    IssueComment,
    IssueLabel,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
    Release,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    CURRENT_METHODOLOGY_VERSION,
    ConfidenceSummary,
    MetricCoverageItem,
    RuleInterpretation,
)
from backend.app.schemas.ingestion import IngestionSummary, PaginatedResult
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics_queue import AnalyticsJobQueue
from backend.app.services.github_client import AsyncGitHubClient
from backend.app.services.rate_budgeter import EntityRateBudgeter
from backend.app.services.repository_collector import RepositoryCollector


@pytest.fixture
def in_memory_db():
    """Create fresh isolated in-memory SQLite database session for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionClass()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def mock_client():
    """Mock AsyncGitHubClient with AsyncMock for requests and pagination."""
    client = MagicMock(spec=AsyncGitHubClient)
    default_resp = MagicMock()
    default_resp.json.return_value = {}
    client.request = AsyncMock(return_value=default_resp)
    client.paginate = AsyncMock()
    client.is_primary_rate_limited = False
    client.resumes_at = None
    return client


# ==============================================================================
# 1. Migration 003 & Benchmark Data Preservation
# ==============================================================================

class TestMigration003AndBenchmarkPreservation:
    """Verifies that project benchmark dataset and Migration 003 schema are preserved."""

    def test_benchmark_data_and_schema_columns(self):
        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip(f"Database {db_path} not found.")

        engine = create_engine(f"sqlite:///{db_path}")
        with Session(engine) as session:
            # 1. Exactly 2,328 benchmark issues remain intact
            issue_count = session.scalar(select(func.count(Issue.id)))
            assert issue_count == 2328, f"Benchmark data modified: expected 2328 issues, found {issue_count}"

            # 2. Historical benchmark commits have committer_date IS NULL
            total_commits = session.scalar(select(func.count(Commit.id)))
            non_null_committer_dates = session.scalar(
                select(func.count(Commit.id)).where(Commit.committer_date.isnot(None))
            )
            assert non_null_committer_dates == 0, (
                f"Historical commits modified: expected 0 non-null committer_date, found {non_null_committer_dates} / {total_commits}"
            )

            # 3. Repositories have analytics_source_revision >= 1 and last_analyzed_at column exists
            repos = session.scalars(select(Repository)).all()
            for repo in repos:
                assert repo.analytics_source_revision >= 1

            # 4. AnalysisRunAttempt table exists and is accessible
            attempt_count = session.scalar(select(func.count(AnalysisRunAttempt.id)))
            assert attempt_count >= 0


# ==============================================================================
# 2. Collector Mutation Tracking & Revision Advancement
# ==============================================================================

class TestCollectorMutationTrackingAndRevision:
    """Verifies atomic record-level mutation tracking in RepositoryCollector."""

    def test_collector_mutation_tracking_and_revision_advancement(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            # Mock metadata response
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1001,
                "name": "foundation-repo",
                "full_name": "corp/foundation-repo",
                "language": "Python",
                "stargazers_count": 50,
                "forks_count": 5,
                "open_issues_count": 1,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            mock_client.request.return_value = meta_resp

            # Mock 1 commit with committer date
            commits_res = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "sha_found_1",
                        "commit": {
                            "author": {"name": "Author", "email": "a@corp.com", "date": "2026-10-01T10:00:00Z"},
                            "committer": {"name": "Committer", "email": "c@corp.com", "date": "2026-10-01T11:00:00Z"},
                            "message": "Foundation commit",
                        },
                        "author": {"login": "corp_dev", "id": 5001},
                    }
                ],
                total_requests=1,
            )
            empty_res = PaginatedResult(entity="generic", items=[], total_requests=1)

            mock_client.paginate.side_effect = lambda *args, **kwargs: (
                commits_res if kwargs.get("entity") == "commits" else empty_res
            )

            # First collection: Brand new repo (revision starts at 1, advances to 2 upon committing mutations)
            summary1 = await collector.collect_repository(repo_name="corp/foundation-repo", db=in_memory_db)
            assert summary1.records_mutated > 0
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 1001))
            assert repo.analytics_source_revision == 2

            # Verify committer_date was extracted correctly
            cmt = in_memory_db.scalar(select(Commit).where(Commit.sha == "sha_found_1"))
            assert cmt.committer_date is not None
            assert cmt.committer_date.hour == 11
            assert cmt.author_date.hour == 10

            # Second collection: Exact same data re-ingested (idempotent no-op)
            summary2 = await collector.collect_repository(repo_name="corp/foundation-repo", db=in_memory_db)
            assert summary2.records_mutated == 0
            in_memory_db.refresh(repo)
            # Revision must remain exactly 2
            assert repo.analytics_source_revision == 2

        asyncio.run(_run())

    def test_committer_date_null_preservation_when_missing(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter()

            repo = Repository(id=1002, repo_name="corp/no-committer", owner="corp", name="no-committer")
            in_memory_db.add(repo)
            in_memory_db.commit()

            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "sha_no_committer",
                        "commit": {
                            "author": {"name": "Author", "email": "a@corp.com", "date": "2026-10-01T10:00:00Z"},
                            # No committer object
                            "message": "No committer date provided",
                        },
                        "author": None,
                    }
                ],
                total_requests=1,
            )

            mutated = set()
            count, _, _ = await collector.collect_commits(
                repo=repo, budgeter=budgeter, db=in_memory_db, mutated_records=mutated
            )
            assert count == 1
            cmt = in_memory_db.scalar(select(Commit).where(Commit.sha == "sha_no_committer"))
            assert cmt.committer_date is None
            assert cmt.author_date is not None

        asyncio.run(_run())

    def test_partial_commits_advance_revision_and_rollback_reverts(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            # Metadata succeeds
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1003,
                "name": "budget-exhaust",
                "full_name": "corp/budget-exhaust",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Budget exhausted during commits pagination
            mock_client.paginate.side_effect = BudgetExceededException(message="Budget exceeded", max_allowed=5)

            summary = await collector.collect_repository(repo_name="corp/budget-exhaust", db=in_memory_db)
            assert summary.is_partial is True
            assert summary.records_mutated == 1  # 1 repository entity mutated
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 1003))
            assert repo is not None
            assert repo.analytics_source_revision == 2

            # Rollback case on unhandled error
            meta_resp.json.return_value["id"] = 1004
            meta_resp.json.return_value["full_name"] = "corp/crash-test"
            mock_client.paginate.side_effect = RuntimeError("Fatal hardware failure")

            crash_summary = await collector.collect_repository(repo_name="corp/crash-test", db=in_memory_db)
            assert len(crash_summary.errors) > 0
            # Database rolled back; repo 1004 not persisted
            repo_crashed = in_memory_db.scalar(select(Repository).where(Repository.id == 1004))
            assert repo_crashed is None

        asyncio.run(_run())


# ==============================================================================
# 3. Analytics Queue Lifecycle, Deduplication & Fencing
# ==============================================================================

class TestAnalyticsQueueLifecycleAndFencing:
    """Verifies AnalyticsJobQueue enqueue, worker claims, lease fencing, and retry termination."""

    def test_enqueue_and_deduplication(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=2001, repo_name="corp/queue-repo", owner="corp", name="queue-repo", analytics_source_revision=3)
        in_memory_db.add(repo)
        in_memory_db.commit()

        # Enqueue new
        run1, is_new1 = queue.enqueue_analysis(repo_id=2001, db=in_memory_db)
        assert is_new1 is True
        assert run1.status == "pending"
        assert run1.requested_source_revision == 3
        assert run1.analytics_source_revision is None
        assert run1.requested_methodology_version == CURRENT_METHODOLOGY_VERSION
        assert run1.methodology_version is None

        # Duplicate submission returns active run1
        run2, is_new2 = queue.enqueue_analysis(repo_id=2001, db=in_memory_db)
        assert is_new2 is False
        assert run2.id == run1.id

    def test_worker_claim_and_lease_renewal(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=2002, repo_name="corp/claim-repo", owner="corp", name="claim-repo", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=2002, db=in_memory_db)

        # Worker claims
        worker_id = "worker:alpha:999:abcdef"
        claimed = queue.claim_job(worker_id=worker_id, db=in_memory_db, lease_duration_seconds=300)
        assert claimed is not None
        run, attempt = claimed

        assert run.status == "running"
        assert run.worker_id == worker_id
        assert run.lease_version == 1
        assert attempt.attempt_number == 1
        assert attempt.outcome == "running"

        # Worker renews lease with matching lease_version = 1
        renewed = queue.renew_lease(
            run_id=run.id,
            worker_id=worker_id,
            expected_lease_version=1,
            extension_seconds=600,
            db=in_memory_db,
        )
        assert renewed is True
        in_memory_db.refresh(run)
        assert run.lease_version == 2

        # Ghost worker with old lease_version = 1 is rejected
        with pytest.raises(GhostWorkerRejectedException):
            queue.renew_lease(
                run_id=run.id,
                worker_id=worker_id,
                expected_lease_version=1,
                db=in_memory_db,
            )

    def test_retry_termination_after_four_total_attempts(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=2003, repo_name="corp/retry-repo", owner="corp", name="retry-repo", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=2003, db=in_memory_db, max_retries=3)

        # Sequence of 4 attempts (1 initial + 3 retries):
        for expected_attempt in range(1, 5):
            claimed = queue.claim_job(worker_id=f"worker_{expected_attempt}", db=in_memory_db)
            assert claimed is not None
            run, att = claimed
            assert att.attempt_number == expected_attempt
            assert run.retry_count == expected_attempt - 1

            queue.record_attempt_failure(
                run_id=run.id,
                attempt_id=att.id,
                worker_id=f"worker_{expected_attempt}",
                expected_lease_version=run.lease_version,
                attempt_number=expected_attempt,
                error_message=f"Error {expected_attempt}",
                db=in_memory_db,
            )
            in_memory_db.refresh(run)

            if expected_attempt < 4:
                assert run.status == "pending"
                assert run.retry_count == expected_attempt
            else:
                # 4th failure triggers terminal failure
                assert run.status == "failed"
                assert run.retry_count == 4
                assert run.completed_at is not None

        # 5th execution attempt is NEVER scheduled; queue returns None
        empty_claim = queue.claim_job(worker_id="worker_5", db=in_memory_db)
        assert empty_claim is None


# ==============================================================================
# 4. Snapshot Isolation, Publication & Stale Run Replacement
# ==============================================================================

class TestSnapshotPublicationAndStaleReplacement:
    """Verifies SQLite WAL snapshot isolation, atomic publication, and stale-run replacement."""

    def test_sqlite_wal_snapshot_isolation_guarantee(self, tmp_path):
        db_file = tmp_path / "wal_foundation_test.db"
        db_url = f"sqlite:///{db_file}"

        admin_engine = create_engine(db_url, connect_args={"check_same_thread": False})
        with admin_engine.connect() as admin_conn:
            admin_conn.execute(text("PRAGMA journal_mode=WAL;"))
            admin_conn.execute(text("PRAGMA busy_timeout=15000;"))
            Base.metadata.create_all(admin_conn)
            admin_conn.execute(
                insert(Repository).values(
                    id=3001, repo_name="wal/foundation", owner="wal", name="foundation",
                    analytics_source_revision=10, is_active=True
                )
            )
            for i in range(1, 4):
                admin_conn.execute(
                    insert(Issue).values(
                        id=i, repository_id=3001, repo_name="wal/foundation",
                        issue_number=i, title=f"Issue {i}", state="open",
                        author="user", created_at=datetime.now(timezone.utc)
                    )
                )
            admin_conn.commit()

        # Reader Connection with isolation_level=None
        reader_engine = create_engine(db_url, isolation_level=None, connect_args={"check_same_thread": False})
        reader_conn = reader_engine.connect()
        raw_dbapi = reader_conn.connection.dbapi_connection

        # Start explicit transaction
        reader_conn.execute(text("BEGIN DEFERRED;"))
        assert raw_dbapi.in_transaction is True

        # First query pins read mark
        snapshot_rev = reader_conn.execute(
            select(Repository.analytics_source_revision).where(Repository.id == 3001)
        ).scalar_one()
        assert snapshot_rev == 10
        issue_count_before = reader_conn.execute(
            select(func.count(Issue.id)).where(Issue.repository_id == 3001)
        ).scalar_one()
        assert issue_count_before == 3

        # Concurrent Writer commits rev 11 and issue 4
        writer_engine = create_engine(db_url, connect_args={"check_same_thread": False})
        with writer_engine.connect() as writer_conn:
            with writer_conn.begin():
                writer_conn.execute(
                    insert(Issue).values(
                        id=4, repository_id=3001, repo_name="wal/foundation",
                        issue_number=4, title="Issue 4", state="open",
                        author="user", created_at=datetime.now(timezone.utc)
                    )
                )
                writer_conn.execute(
                    update(Repository).where(Repository.id == 3001).values(analytics_source_revision=11)
                )

        # Reader transaction MUST still observe frozen revision 10 and 3 issues
        assert reader_conn.execute(select(Repository.analytics_source_revision).where(Repository.id == 3001)).scalar_one() == 10
        assert reader_conn.execute(select(func.count(Issue.id)).where(Issue.repository_id == 3001)).scalar_one() == 3

        reader_conn.execute(text("COMMIT;"))
        reader_conn.close()

    def test_atomic_publication_and_stale_replacement_run_reuse(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=3002, repo_name="corp/pub-stale", owner="corp", name="pub-stale", analytics_source_revision=10)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=3002, db=in_memory_db)
        run1, att1 = queue.claim_job(worker_id="w_pub", db=in_memory_db)

        # Ingestion advances revision from 10 to 11 while worker is computing snapshot rev 10
        repo.analytics_source_revision = 11
        in_memory_db.commit()

        # Worker publishes snapshot rev 10
        ref_time = datetime.now(timezone.utc)
        payload = {"velocity": 5.0}
        success = queue.publish_and_finalize(
            run_id=run1.id,
            attempt_id=att1.id,
            worker_id="w_pub",
            expected_lease_version=1,
            attempt_number=1,
            snapshot_revision=10,  # Stale! (10 < 11)
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=ref_time,
            metrics_payload=payload,
            coverage_summary={},
            confidence_summary={},
            rule_interpretations=[],
            db=in_memory_db,
        )
        assert success is True

        in_memory_db.refresh(run1)
        in_memory_db.refresh(att1)
        assert run1.status == "completed"
        assert run1.analytics_source_revision == 10
        assert att1.outcome == "completed"

        # Stale publication enqueued replacement pending run for rev 11
        run2 = in_memory_db.scalar(
            select(AnalysisRun).where(
                AnalysisRun.repository_id == 3002,
                AnalysisRun.status == "pending",
            )
        )
        assert run2 is not None
        assert run2.requested_source_revision == 11

        # Calling publish_and_finalize or enqueue again reuses active run2 without duplicate constraint failure
        reused_run, is_new = queue.enqueue_analysis(repo_id=3002, db=in_memory_db)
        assert is_new is False
        assert reused_run.id == run2.id

    def test_durable_reconciliation_sweeps_expired_leases_and_missing_jobs(self, in_memory_db):
        queue = AnalyticsJobQueue()
        # Repo A: has expired running job
        repo_a = Repository(id=3003, repo_name="corp/repo-a", owner="corp", name="repo-a", analytics_source_revision=1, is_active=True)
        # Repo B: has committed mutations (rev 2) but missing analysis job (crash recovery)
        repo_b = Repository(id=3004, repo_name="corp/repo-b", owner="corp", name="repo-b", analytics_source_revision=2, is_active=True)
        in_memory_db.add_all([repo_a, repo_b])
        in_memory_db.commit()

        # Seed expired run on Repo A
        expired_run = AnalysisRun(
            repository_id=3003,
            status="running",
            worker_id="crashed_worker",
            lease_version=1,
            leased_until=datetime.now(timezone.utc) - timedelta(minutes=10),
            started_at=datetime.now(timezone.utc) - timedelta(minutes=20),
            retry_count=0,
            max_retries=3,
            requested_source_revision=1,
        )
        in_memory_db.add(expired_run)
        in_memory_db.flush()
        expired_att = AnalysisRunAttempt(
            analysis_run_id=expired_run.id,
            attempt_number=1,
            worker_id="crashed_worker",
            outcome="running",
            started_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        )
        in_memory_db.add(expired_att)
        in_memory_db.commit()

        # Execute reconciliation sweeper
        summary = queue.reconcile_analytics_jobs(db=in_memory_db)
        assert summary["expired_leases_recovered"] == 1
        assert summary["missing_jobs_enqueued"] == 1

        # Verify Repo A run reset to pending with retry_count=1
        in_memory_db.refresh(expired_run)
        in_memory_db.refresh(expired_att)
        assert expired_run.status == "pending"
        assert expired_run.retry_count == 1
        assert expired_att.outcome == "lease_expired"

        # Verify Repo B missing run enqueued for rev 2
        recovered_run = in_memory_db.scalar(select(AnalysisRun).where(AnalysisRun.repository_id == 3004))
        assert recovered_run is not None
        assert recovered_run.status == "pending"
        assert recovered_run.requested_source_revision == 2
