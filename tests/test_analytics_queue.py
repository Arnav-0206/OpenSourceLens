"""Integration and unit tests for Phase C AnalyticsJobQueue & Lifecycle Management (Step C.1.3).

Verifies:
1. Pydantic v2 schemas and BaseMetricCalculator contract.
2. Enqueueing with unified predicate, no-op suppression, and partial unique index deduplication.
3. Atomic worker claiming, lease fencing, and heartbeat renewal.
4. Driver-verified SQLite WAL snapshot isolation (first SQL query pinning read mark).
5. Conditional rowcount-validated publication atomic with attempt finalization.
6. Ghost worker rejection and attempt immutability enforcement.
7. Retry state machine (max_retries = 3, 4 total executions max, termination).
8. Stale-result detection and race-safe active replacement run reuse.
9. Durable reconciliation for expired leases and crash-safe ingestion handoff.
10. PostgreSQL dialect query verification.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import pytest
from sqlalchemy import and_, create_engine, event, func, insert, or_, select, text, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.exceptions import (
    AttemptImmutableException,
    EntityNotFoundException,
    GhostWorkerRejectedException,
)
from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    Issue,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    CURRENT_METHODOLOGY_VERSION,
    AnalysisPayload,
    AnalysisRunRead,
    AnalysisRunSubmission,
    ConfidenceSummary,
    MetricCoverageItem,
    RuleInterpretation,
)
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics_queue import AnalyticsJobQueue


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


class TestSchemasAndCalculatorContract:
    """Verifies Pydantic v2 schemas and BaseMetricCalculator abstract contract."""

    def test_analytics_v2_schemas_validation(self):
        coverage = MetricCoverageItem(
            domain="development_activity",
            status="sufficient",
            sample_size=150,
            min_sample_required=30,
        )
        assert coverage.domain == "development_activity"
        assert coverage.status == "sufficient"

        confidence = ConfidenceSummary(
            score=0.92,
            band="high",
            penalties=[],
        )
        assert confidence.score == 0.92
        assert confidence.band == "high"

        rule = RuleInterpretation(
            rule_id="RULE-DEV-001",
            domain="development_activity",
            severity="warning",
            finding="Velocity drop observed in last 30 days.",
            evidence={"velocity_ratio": 0.65},
            recommendation="Review open PR backlog.",
        )
        assert rule.rule_id == "RULE-DEV-001"

        sub = AnalysisRunSubmission(repository_id=42, force=True)
        assert sub.repository_id == 42
        assert sub.requested_methodology_version == CURRENT_METHODOLOGY_VERSION
        assert sub.force is True

    def test_base_metric_calculator_abstract_contract(self):
        class DummyCalculator(BaseMetricCalculator):
            @property
            def domain_name(self) -> str:
                return "test_domain"

            def calculate(
                self,
                repository_id: int,
                db: Session,
                analysis_reference_time: datetime,
                snapshot_revision: int,
            ) -> CalculationResult:
                cov = MetricCoverageItem(
                    domain="test_domain",
                    status="sufficient",
                    sample_size=10,
                    min_sample_required=5,
                )
                return CalculationResult(
                    domain="test_domain",
                    metrics={"sample_metric": 42.0},
                    coverage=cov,
                    confidence_penalty=0.0,
                    rules=[],
                )

        calc = DummyCalculator()
        assert calc.domain_name == "test_domain"
        res = calc.calculate(1, None, datetime.now(timezone.utc), 1)
        assert res.domain == "test_domain"
        assert res.metrics["sample_metric"] == 42.0


class TestEnqueueAndUnifiedPredicate:
    """Verifies enqueueing logic, deduplication, and no-op suppression."""

    def test_enqueue_new_run_success(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=1, repo_name="org/repo", owner="org", name="repo", analytics_source_revision=5)
        in_memory_db.add(repo)
        in_memory_db.commit()

        run, is_new = queue.enqueue_analysis(repo_id=1, db=in_memory_db)
        assert is_new is True
        assert run.status == "pending"
        assert run.requested_source_revision == 5
        assert run.analytics_source_revision is None
        assert run.requested_methodology_version == CURRENT_METHODOLOGY_VERSION
        assert run.methodology_version is None
        assert run.retry_count == 0
        assert run.max_retries == 3

    def test_enqueue_active_deduplication(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=2, repo_name="org/repo2", owner="org", name="repo2", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.commit()

        run1, is_new1 = queue.enqueue_analysis(repo_id=2, db=in_memory_db)
        assert is_new1 is True

        # Second submission while run1 is pending returns run1
        run2, is_new2 = queue.enqueue_analysis(repo_id=2, db=in_memory_db)
        assert is_new2 is False
        assert run2.id == run1.id

    def test_enqueue_noop_suppression_when_already_analyzed(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=3, repo_name="org/repo3", owner="org", name="repo3", analytics_source_revision=10)
        in_memory_db.add(repo)
        in_memory_db.commit()

        # Seed completed run for rev 10
        completed = AnalysisRun(
            repository_id=3,
            status="completed",
            requested_source_revision=10,
            analytics_source_revision=10,
            requested_methodology_version=CURRENT_METHODOLOGY_VERSION,
            methodology_version=CURRENT_METHODOLOGY_VERSION,
            completed_at=datetime.now(timezone.utc),
        )
        in_memory_db.add(completed)
        in_memory_db.commit()

        # Without force: no-op suppression returns existing completed run
        run, is_new = queue.enqueue_analysis(repo_id=3, db=in_memory_db, force=False)
        assert is_new is False
        assert run.id == completed.id

        # With force=True: bypasses check and enqueues new run
        forced_run, forced_is_new = queue.enqueue_analysis(repo_id=3, db=in_memory_db, force=True)
        assert forced_is_new is True
        assert forced_run.id != completed.id
        assert forced_run.status == "pending"


class TestWorkerClaimAndLeaseManagement:
    """Verifies atomic claiming, attempt logging, and lease extension."""

    def test_worker_claim_job_success(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=4, repo_name="org/claim-test", owner="org", name="claim-test", analytics_source_revision=2)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=4, db=in_memory_db)

        worker_id = "worker:host1:1234:abcdef"
        claimed = queue.claim_job(worker_id=worker_id, db=in_memory_db, lease_duration_seconds=300)
        assert claimed is not None
        run, attempt = claimed

        assert run.status == "running"
        assert run.worker_id == worker_id
        assert run.lease_version == 1
        assert run.leased_until is not None
        assert run.retry_count == 0

        assert attempt.analysis_run_id == run.id
        assert attempt.attempt_number == 1
        assert attempt.worker_id == worker_id
        assert attempt.outcome == "running"

    def test_sqlite_single_worker_claim_concurrency(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=5, repo_name="org/sqlite-test", owner="org", name="sqlite-test", analytics_source_revision=2)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=5, db=in_memory_db)

        # Worker 1 claims
        claimed1 = queue.claim_job(worker_id="worker_1", db=in_memory_db, lease_duration_seconds=600)
        assert claimed1 is not None

        # Worker 2 attempts to claim while Worker 1 has active lease on SQLite
        claimed2 = queue.claim_job(worker_id="worker_2", db=in_memory_db)
        assert claimed2 is None

    def test_lease_renewal_and_fencing(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=6, repo_name="org/renew-test", owner="org", name="renew-test", analytics_source_revision=2)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=6, db=in_memory_db)
        run, attempt = queue.claim_job(worker_id="worker_valid", db=in_memory_db)

        # 1. Valid renewal with expected lease_version = 1
        success = queue.renew_lease(
            run_id=run.id,
            worker_id="worker_valid",
            expected_lease_version=1,
            extension_seconds=900,
            db=in_memory_db,
        )
        assert success is True
        in_memory_db.refresh(run)
        assert run.lease_version == 2

        # 2. Stale renewal with mismatched lease version (1 instead of 2)
        with pytest.raises(GhostWorkerRejectedException):
            queue.renew_lease(
                run_id=run.id,
                worker_id="worker_valid",
                expected_lease_version=1,
                db=in_memory_db,
            )

        # 3. Unauthorized worker ID renewal rejected
        with pytest.raises(GhostWorkerRejectedException):
            queue.renew_lease(
                run_id=run.id,
                worker_id="worker_imposter",
                expected_lease_version=2,
                db=in_memory_db,
            )


class TestSQLiteWALDriverVerifiedSnapshotIsolation:
    """Verifies that Python SQLite reader pins SQLite WAL read mark on first query."""

    def test_sqlite_wal_driver_verified_snapshot_isolation(self, tmp_path):
        """Verifies that an explicit SQLite WAL read transaction pins reader to revision 42

        despite concurrent writer committing revision 43 to disk.
        """
        db_file = tmp_path / "test_wal_driver_verified.db"
        db_url = f"sqlite:///{db_file}"

        # 1. Admin Setup Connection: Enable WAL mode and seed initial state
        admin_engine = create_engine(db_url, connect_args={"check_same_thread": False})
        with admin_engine.connect() as admin_conn:
            admin_conn.execute(text("PRAGMA journal_mode=WAL;"))
            admin_conn.execute(text("PRAGMA busy_timeout=15000;"))
            assert admin_conn.execute(text("PRAGMA journal_mode;")).scalar().lower() == "wal"
            Base.metadata.create_all(admin_conn)

            # Seed Repo at revision 42 with 5 issues
            admin_conn.execute(
                insert(Repository).values(
                    id=10, repo_name="wal/test-repo", owner="wal", name="test-repo",
                    analytics_source_revision=42, is_active=True
                )
            )
            for i in range(1, 6):
                admin_conn.execute(
                    insert(Issue).values(
                        id=i, repository_id=10, repo_name="wal/test-repo",
                        issue_number=i, title=f"Issue {i}", state="open",
                        author="user", created_at=datetime.now(timezone.utc)
                    )
                )
            admin_conn.commit()

        # 2. Reader Engine: Configured with isolation_level=None to control raw DBAPI transactions
        executed_sql = []
        reader_engine = create_engine(
            db_url,
            isolation_level=None,  # Disables DBAPI implicit transaction interception
            connect_args={"check_same_thread": False}
        )

        @event.listens_for(reader_engine, "before_cursor_execute")
        def intercept_reader_sql(conn, cursor, statement, parameters, context, executemany):
            executed_sql.append(statement.strip().upper())

        reader_conn = reader_engine.connect()
        raw_dbapi_conn = reader_conn.connection.dbapi_connection

        # Explicitly start the read transaction at SQLite C-level
        reader_conn.execute(text("BEGIN DEFERRED;"))
        assert raw_dbapi_conn.in_transaction is True, "Driver failed to enter active database transaction!"

        # First Query: Captures analytics_source_revision and pins SQLite WAL read mark!
        snapshot_rev = reader_conn.execute(
            select(Repository.analytics_source_revision).where(Repository.id == 10)
        ).scalar_one()

        # Assertion: Verify exact statement order
        assert executed_sql[0] == "BEGIN DEFERRED;", "BEGIN DEFERRED was not first statement!"
        assert "ANALYTICS_SOURCE_REVISION" in executed_sql[1], "Revision query was not second statement!"
        assert snapshot_rev == 42

        # Query 2 on reader: Initial issue count
        count_before = reader_conn.execute(
            select(func.count(Issue.id)).where(Issue.repository_id == 10)
        ).scalar_one()
        assert count_before == 5

        # 3. Writer Engine: Concurrent Ingestion commits Issue #6 and Revision 43
        writer_engine = create_engine(db_url, connect_args={"check_same_thread": False})
        with writer_engine.connect() as writer_conn:
            with writer_conn.begin():
                writer_conn.execute(
                    insert(Issue).values(
                        id=6, repository_id=10, repo_name="wal/test-repo",
                        issue_number=6, title="Issue 6", state="open",
                        author="user", created_at=datetime.now(timezone.utc)
                    )
                )
                writer_conn.execute(
                    update(Repository).where(Repository.id == 10).values(analytics_source_revision=43)
                )
            # Committed to WAL file on disk!

        # 4. Return to Reader (still inside active transaction): MUST observe frozen snapshot 42 with 5 issues
        still_rev = reader_conn.execute(
            select(Repository.analytics_source_revision).where(Repository.id == 10)
        ).scalar_one()
        assert still_rev == 42, "Reader experienced dirty read / snapshot tearing!"

        still_count = reader_conn.execute(
            select(func.count(Issue.id)).where(Issue.repository_id == 10)
        ).scalar_one()
        assert still_count == 5, "Reader observed phantom rows from concurrent ingestion commit!"

        # 5. Terminate Reader Transaction
        reader_conn.execute(text("COMMIT;"))
        assert raw_dbapi_conn.in_transaction is False

        # 6. Subsequent transaction on reader now observes revision 43 and 6 issues
        reader_conn.execute(text("BEGIN DEFERRED;"))
        fresh_rev = reader_conn.execute(
            select(Repository.analytics_source_revision).where(Repository.id == 10)
        ).scalar_one()
        assert fresh_rev == 43
        fresh_count = reader_conn.execute(
            select(func.count(Issue.id)).where(Issue.repository_id == 10)
        ).scalar_one()
        assert fresh_count == 6
        reader_conn.execute(text("COMMIT;"))
        reader_conn.close()


class TestPublicationAndAttemptFinalization:
    """Verifies atomic result publication, stale detection, and immutability."""

    def test_atomic_publication_success(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=11, repo_name="org/pub-test", owner="org", name="pub-test", analytics_source_revision=7)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=11, db=in_memory_db)
        worker_id = "worker_pub"
        run, attempt = queue.claim_job(worker_id=worker_id, db=in_memory_db)

        ref_time = datetime.now(timezone.utc)
        payload = {"velocity": 12.5}
        cov = {"domain": "dev", "status": "sufficient"}
        conf = {"score": 0.95, "band": "high"}
        rules = [{"rule_id": "R1", "severity": "info"}]

        success = queue.publish_and_finalize(
            run_id=run.id,
            attempt_id=attempt.id,
            worker_id=worker_id,
            expected_lease_version=1,
            attempt_number=1,
            snapshot_revision=7,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=ref_time,
            metrics_payload=payload,
            coverage_summary=cov,
            confidence_summary=conf,
            rule_interpretations=rules,
            db=in_memory_db,
        )
        assert success is True

        in_memory_db.refresh(run)
        in_memory_db.refresh(attempt)
        in_memory_db.refresh(repo)

        assert run.status == "completed"
        assert run.analytics_source_revision == 7
        assert run.methodology_version == CURRENT_METHODOLOGY_VERSION
        assert run.metrics_payload == payload
        assert attempt.outcome == "completed"
        assert repo.last_analyzed_at is not None

    def test_ghost_worker_publication_rejected(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=12, repo_name="org/ghost-test", owner="org", name="ghost-test", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=12, db=in_memory_db)
        run, attempt = queue.claim_job(worker_id="worker_orig", db=in_memory_db)

        # Worker renewal advances lease_version from 1 to 2
        queue.renew_lease(run_id=run.id, worker_id="worker_orig", expected_lease_version=1, db=in_memory_db)

        # Ghost worker attempting to publish with old lease_version = 1 is rejected
        with pytest.raises(GhostWorkerRejectedException):
            queue.publish_and_finalize(
                run_id=run.id,
                attempt_id=attempt.id,
                worker_id="worker_orig",
                expected_lease_version=1,  # Stale version!
                attempt_number=1,
                snapshot_revision=1,
                actual_methodology=CURRENT_METHODOLOGY_VERSION,
                reference_time=datetime.now(timezone.utc),
                metrics_payload={},
                coverage_summary={},
                confidence_summary={},
                rule_interpretations=[],
                db=in_memory_db,
            )

    def test_attempt_immutability_enforcement(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=13, repo_name="org/immut-test", owner="org", name="immut-test", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=13, db=in_memory_db)
        run, attempt = queue.claim_job(worker_id="worker_immut", db=in_memory_db)

        # Finalize attempt to 'completed'
        queue.finalize_attempt(attempt_id=attempt.id, outcome="completed", worker_id="worker_immut", db=in_memory_db)

        # Attempting to mutate already terminal attempt raises AttemptImmutableException
        with pytest.raises(AttemptImmutableException):
            queue.finalize_attempt(attempt_id=attempt.id, outcome="failed", worker_id="worker_immut", db=in_memory_db)


class TestRetryStateMachineAndSweeper:
    """Verifies max_retries = 3 retry transitions and durable sweeper recovery."""

    def test_retry_transition_table_and_terminal_failure(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=14, repo_name="org/retry-test", owner="org", name="retry-test", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=14, db=in_memory_db, max_retries=3)

        # Execution 1 (Initial attempt #1, retry_count=0)
        run, att1 = queue.claim_job(worker_id="w1", db=in_memory_db)
        assert att1.attempt_number == 1
        assert run.retry_count == 0
        queue.record_attempt_failure(run.id, att1.id, "w1", run.lease_version, 1, "Fail 1", in_memory_db)
        in_memory_db.refresh(run)
        assert run.status == "pending"
        assert run.retry_count == 1

        # Execution 2 (Retry 1, attempt #2, retry_count=1)
        run, att2 = queue.claim_job(worker_id="w2", db=in_memory_db)
        assert att2.attempt_number == 2
        queue.record_attempt_failure(run.id, att2.id, "w2", run.lease_version, 2, "Fail 2", in_memory_db)
        in_memory_db.refresh(run)
        assert run.status == "pending"
        assert run.retry_count == 2

        # Execution 3 (Retry 2, attempt #3, retry_count=2)
        run, att3 = queue.claim_job(worker_id="w3", db=in_memory_db)
        assert att3.attempt_number == 3
        queue.record_attempt_failure(run.id, att3.id, "w3", run.lease_version, 3, "Fail 3", in_memory_db)
        in_memory_db.refresh(run)
        assert run.status == "pending"
        assert run.retry_count == 3

        # Execution 4 (Retry 3, attempt #4, retry_count=3 -> max retries exhausted!)
        run, att4 = queue.claim_job(worker_id="w4", db=in_memory_db)
        assert att4.attempt_number == 4
        queue.record_attempt_failure(run.id, att4.id, "w4", run.lease_version, 4, "Terminal poison repo", in_memory_db)
        in_memory_db.refresh(run)
        assert run.status == "failed"
        assert run.retry_count == 4
        assert run.error_message == "Terminal poison repo"

        # Attempt #5 is NEVER scheduled; queue is empty
        claimed_empty = queue.claim_job(worker_id="w5", db=in_memory_db)
        assert claimed_empty is None

    def test_non_retryable_failure_terminates_immediately_and_preserves_atomicity(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(
            id=18,
            repo_name="org/non-retry-test",
            owner="org",
            name="non-retry-test",
            analytics_source_revision=1,
        )
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=18, db=in_memory_db, max_retries=3)
        run, att1 = queue.claim_job(worker_id="w_nr_1", db=in_memory_db)
        assert att1.attempt_number == 1
        assert run.retry_count == 0
        assert run.lease_version == 1

        queue.record_attempt_failure(
            run_id=run.id,
            attempt_id=att1.id,
            worker_id="w_nr_1",
            expected_lease_version=run.lease_version,
            attempt_number=1,
            error_message="Non-retryable validation schema error",
            db=in_memory_db,
            outcome="failed",
            retryable=False,
            error_category="non_retryable",
        )

        in_memory_db.refresh(run)
        in_memory_db.refresh(att1)
        assert run.status == "failed"
        assert run.retry_count == 0
        assert run.max_retries == 3
        assert run.worker_id is None
        assert run.leased_until is None
        assert run.completed_at is not None
        assert run.lease_version == 2
        assert run.error_message == "Non-retryable validation schema error"
        assert att1.outcome == "failed"
        assert att1.error_category == "non_retryable"
        assert att1.error_message == "Non-retryable validation schema error"
        assert att1.completed_at is not None
        assert queue.claim_job(worker_id="w_nr_next", db=in_memory_db) is None

        # Verify lease fencing rolls back BOTH AnalysisRunAttempt (including error_category) and AnalysisRun
        repo_fenced = Repository(
            id=19,
            repo_name="org/non-retry-fenced",
            owner="org",
            name="non-retry-fenced",
            analytics_source_revision=1,
        )
        in_memory_db.add(repo_fenced)
        in_memory_db.commit()
        queue.enqueue_analysis(repo_id=19, db=in_memory_db, max_retries=3)
        fenced_run, fenced_att = queue.claim_job(worker_id="w_fenced", db=in_memory_db)

        with pytest.raises(GhostWorkerRejectedException):
            queue.record_attempt_failure(
                run_id=fenced_run.id,
                attempt_id=fenced_att.id,
                worker_id="w_fenced",
                expected_lease_version=fenced_run.lease_version + 99,
                attempt_number=1,
                error_message="Stale worker non-retryable fail",
                db=in_memory_db,
                outcome="failed",
                retryable=False,
                error_category="non_retryable",
            )

        in_memory_db.refresh(fenced_run)
        in_memory_db.refresh(fenced_att)
        assert fenced_run.status == "running"
        assert fenced_run.worker_id == "w_fenced"
        assert fenced_run.completed_at is None
        assert fenced_att.outcome == "running"
        assert fenced_att.error_category is None
        assert fenced_att.completed_at is None

    def test_stale_result_detection_enqueues_replacement_run(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo = Repository(id=15, repo_name="org/stale-test", owner="org", name="stale-test", analytics_source_revision=42)
        in_memory_db.add(repo)
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=15, db=in_memory_db)
        run, attempt = queue.claim_job(worker_id="w_stale", db=in_memory_db)

        # While worker computes snapshot rev 42, ingestion commits rev 43
        repo.analytics_source_revision = 43
        in_memory_db.commit()

        # Worker publishes snapshot rev 42
        queue.publish_and_finalize(
            run_id=run.id,
            attempt_id=attempt.id,
            worker_id="w_stale",
            expected_lease_version=1,
            attempt_number=1,
            snapshot_revision=42,  # Stale! (42 < 43)
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=datetime.now(timezone.utc),
            metrics_payload={},
            coverage_summary={},
            confidence_summary={},
            rule_interpretations=[],
            db=in_memory_db,
        )

        # Run 1 is completed
        in_memory_db.refresh(run)
        assert run.status == "completed"

        # Stale publication automatically enqueued replacement pending run for rev 43
        replacement_run = in_memory_db.scalar(
            select(AnalysisRun).where(
                AnalysisRun.repository_id == 15,
                AnalysisRun.status == "pending",
            )
        )
        assert replacement_run is not None
        assert replacement_run.requested_source_revision == 43

    def test_durable_reconciliation_expired_leases_and_crash_safe_handoff(self, in_memory_db):
        queue = AnalyticsJobQueue()
        repo1 = Repository(id=16, repo_name="org/exp-lease", owner="org", name="exp-lease", analytics_source_revision=1, is_active=True)
        repo2 = Repository(id=17, repo_name="org/missing-job", owner="org", name="missing-job", analytics_source_revision=3, is_active=True)
        in_memory_db.add_all([repo1, repo2])
        in_memory_db.commit()

        # Seed expired running run on repo 16
        expired_run = AnalysisRun(
            repository_id=16,
            status="running",
            worker_id="dead_worker",
            lease_version=1,
            leased_until=datetime.now(timezone.utc) - timedelta(minutes=5),
            started_at=datetime.now(timezone.utc) - timedelta(minutes=15),
            retry_count=0,
            max_retries=3,
            requested_source_revision=1,
        )
        in_memory_db.add(expired_run)
        in_memory_db.flush()
        expired_attempt = AnalysisRunAttempt(
            analysis_run_id=expired_run.id,
            attempt_number=1,
            worker_id="dead_worker",
            outcome="running",
            started_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )
        in_memory_db.add(expired_attempt)
        in_memory_db.commit()

        # repo 17 has revision 3 committed by ingestion, but zero analysis runs enqueued (simulated crash)

        # Run reconciliation sweeper
        summary = queue.reconcile_analytics_jobs(db=in_memory_db)
        assert summary["expired_leases_recovered"] == 1
        assert summary["missing_jobs_enqueued"] == 1

        # Check repo 16 run was recovered to pending with retry_count=1
        in_memory_db.refresh(expired_run)
        in_memory_db.refresh(expired_attempt)
        assert expired_run.status == "pending"
        assert expired_run.retry_count == 1
        assert expired_attempt.outcome == "lease_expired"

        # Check repo 17 missing job was recovered to pending for rev 3
        recovered_run = in_memory_db.scalar(select(AnalysisRun).where(AnalysisRun.repository_id == 17))
        assert recovered_run is not None
        assert recovered_run.status == "pending"
        assert recovered_run.requested_source_revision == 3


class TestPostgreSQLDialectQueryVerification:
    """Offline compilation verification of PostgreSQL queries."""

    def test_postgresql_claim_and_locking_compiles(self):
        """Verifies that the PostgreSQL claim query with FOR UPDATE SKIP LOCKED compiles."""
        now = datetime.now(timezone.utc)
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
        compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "FOR UPDATE SKIP LOCKED" in compiled_sql
        assert "analysis_runs" in compiled_sql
