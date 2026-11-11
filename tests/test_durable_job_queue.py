"""Unit and reliability tests for DurableJobQueue and IngestionWorker.

Tests verify:
1. Atomic job claiming and mutual exclusion between multiple workers.
2. Lease expiration and job recovery by successor workers.
3. Heartbeat renewals and lease extensions.
4. Stale worker write rejection via lease-version fencing.
5. Recovery of abandoned jobs exceeding max retry limits.
6. SQLite WAL single-writer concurrency safeguards.
7. PostgreSQL row locking (FOR UPDATE SKIP LOCKED) simulation.
8. Full worker lifecycle: success, rate-limiting, and crash recovery.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import multiprocessing
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import CollectionRun, Repository
from backend.app.schemas.ingestion import IngestionSummary
from backend.app.services.job_queue import DurableJobQueue, JobLeaseLostException
from backend.app.services.ingestion_worker import IngestionWorker


def _multiprocess_worker_claim_task(
    db_path: str,
    worker_id: str,
    lease_duration_seconds: int,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue,
):
    """Top-level task function executed in a separate operating system process."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.app.services.job_queue import DurableJobQueue

        engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 15})
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        queue = DurableJobQueue()
        # Synchronize process start to maximize race conditions
        barrier.wait(timeout=10)
        claimed = queue.claim_next_job(
            worker_id=worker_id,
            lease_duration_seconds=lease_duration_seconds,
            db=session,
        )
        claimed_id = claimed.id if claimed else None
        result_queue.put({"worker_id": worker_id, "claimed_id": claimed_id, "error": None})
        session.close()
        engine.dispose()
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "claimed_id": None, "error": str(e)})


def ensure_utc(dt: datetime) -> datetime:
    """Helper to convert naive datetimes from SQLite to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


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
def sample_repo(in_memory_db):
    """Seed a sample repository for job associations."""
    repo = Repository(
        id=9999,
        repo_name="org/sample-repo",
        owner="org",
        name="sample-repo",
        stars_count=1200,
        forks_count=350,
        open_issues_count=45,
        is_active=True,
    )
    in_memory_db.add(repo)
    in_memory_db.commit()
    in_memory_db.refresh(repo)
    return repo


class TestDurableJobQueueClaimingAndFencing:
    """Core reliability tests for atomic job queue claiming and lease fencing."""

    def test_submit_job_persists_in_pending_state(self, in_memory_db, sample_repo):
        """Verify submitting a job persists it with pending status and initial zero counters."""
        queue = DurableJobQueue()
        job, is_new = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

        assert is_new is True
        assert job.id is not None
        assert job.repository_id == sample_repo.id
        assert job.mode == "standard_diagnostic"
        assert job.status == "pending"
        assert job.lease_version == 0
        assert job.retry_count == 0
        assert job.worker_id is None
        assert job.leased_until is None

    def test_two_workers_attempting_to_claim_same_job_only_one_succeeds(self, in_memory_db, sample_repo):
        """Verify that when two workers compete for a pending job, only one claims it and the other gets None."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="fast_snapshot", db=in_memory_db)

        worker1 = "worker-alpha:1001:abcd"
        worker2 = "worker-beta:1002:ef01"

        claimed_1 = queue.claim_next_job(worker_id=worker1, lease_duration_seconds=300, db=in_memory_db)
        assert claimed_1 is not None
        assert claimed_1.id == job.id
        assert claimed_1.worker_id == worker1
        assert claimed_1.status == "in_progress"
        assert claimed_1.lease_version == 1

        # Second worker attempts to claim; queue is now empty
        claimed_2 = queue.claim_next_job(worker_id=worker2, lease_duration_seconds=300, db=in_memory_db)
        assert claimed_2 is None

    def test_two_sqlite_workers_competing_for_different_pending_jobs(self, in_memory_db, sample_repo):
        """Verify that in SQLite, only one worker can have an active job at any time."""
        repo2 = Repository(id=8888, repo_name="org/other-repo", owner="org", name="other-repo", stars_count=100)
        in_memory_db.add(repo2)
        in_memory_db.commit()

        queue = DurableJobQueue()
        job1, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
        job2, _ = queue.submit_job(repo_id=repo2.id, mode="standard_diagnostic", db=in_memory_db)

        worker1 = "worker-1:1001:one"
        worker2 = "worker-2:1002:two"

        # Worker 1 claims Job 1
        claimed_1 = queue.claim_next_job(worker_id=worker1, lease_duration_seconds=600, db=in_memory_db)
        assert claimed_1 is not None
        assert claimed_1.id == job1.id
        assert claimed_1.status == "in_progress"

        # Worker 2 attempts to claim Job 2 while Job 1 is still active
        claimed_2 = queue.claim_next_job(worker_id=worker2, lease_duration_seconds=600, db=in_memory_db)
        # SQLite single-worker restriction MUST block Worker 2 from claiming Job 2
        assert claimed_2 is None

        # Worker 1 completes Job 1
        summary = IngestionSummary(
            repo_name=sample_repo.repo_name,
            mode="standard_diagnostic",
            is_partial=False,
            total_requests_consumed=10,
            items_collected={"commits": 50},
            entity_status={"commits": "completed"},
            execution_time_seconds=1.0,
            errors=[],
        )
        queue.complete_job(job_id=job1.id, worker_id=worker1, lease_version=claimed_1.lease_version, summary=summary, db=in_memory_db)

        # Now that Job 1 is completed, Worker 2 can successfully claim Job 2
        claimed_2_after = queue.claim_next_job(worker_id=worker2, lease_duration_seconds=600, db=in_memory_db)
        assert claimed_2_after is not None
        assert claimed_2_after.id == job2.id
        assert claimed_2_after.status == "in_progress"

    def test_two_sqlite_workers_separate_processes_single_worker_enforcement(self, tmp_path):
        """Verify across separate OS processes that SQLite BEGIN IMMEDIATE single-worker restriction is enforced."""
        db_path = str(tmp_path / "sqlite_proc_test.db")
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 15})
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.commit()
        Base.metadata.create_all(bind=engine)

        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        r1 = Repository(id=101, repo_name="org/proc1", owner="org", name="proc1")
        r2 = Repository(id=102, repo_name="org/proc2", owner="org", name="proc2")
        db.add_all([r1, r2])
        db.commit()

        queue = DurableJobQueue()
        queue.submit_job(repo_id=101, mode="fast_snapshot", db=db)
        queue.submit_job(repo_id=102, mode="standard_diagnostic", db=db)
        db.close()
        engine.dispose()

        q = multiprocessing.Queue()
        barrier = multiprocessing.Barrier(2)

        p1 = multiprocessing.Process(
            target=_multiprocess_worker_claim_task,
            args=(db_path, "os-proc-worker-1", 60, barrier, q),
        )
        p2 = multiprocessing.Process(
            target=_multiprocess_worker_claim_task,
            args=(db_path, "os-proc-worker-2", 60, barrier, q),
        )

        p1.start()
        p2.start()

        res1 = q.get(timeout=10)
        res2 = q.get(timeout=10)
        p1.join(timeout=5)
        p2.join(timeout=5)

        assert res1["error"] is None
        assert res2["error"] is None

        claimed = [res1["claimed_id"], res2["claimed_id"]]
        # Exactly one process must claim a job; the other must receive None
        assert sum(1 for cid in claimed if cid is not None) == 1
        assert None in claimed

    def test_duplicate_pending_and_in_progress_submission_deduplication(self, in_memory_db, sample_repo):
        """Verify duplicate submissions of the same repository return the existing active job."""
        queue = DurableJobQueue()

        # 1. First submission -> new job
        job1, is_new1 = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
        assert is_new1 is True

        # 2. Second submission while job1 is pending -> returns job1
        job2, is_new2 = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
        assert is_new2 is False
        assert job2.id == job1.id
        assert job2.status == "pending"

        # 3. Worker claims job1 -> in_progress
        claimed = queue.claim_next_job(worker_id="worker-dedup:1", lease_duration_seconds=600, db=in_memory_db)
        assert claimed.id == job1.id

        # 4. Third submission while job1 is in_progress -> returns job1
        job3, is_new3 = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
        assert is_new3 is False
        assert job3.id == job1.id
        assert job3.status == "in_progress"

        # 5. Worker completes job1
        summary = IngestionSummary(
            repo_name=sample_repo.repo_name,
            mode="standard_diagnostic",
            is_partial=False,
            total_requests_consumed=5,
            items_collected={"commits": 20},
            entity_status={"commits": "completed"},
            execution_time_seconds=1.0,
            errors=[],
        )
        queue.complete_job(job_id=job1.id, worker_id="worker-dedup:1", lease_version=claimed.lease_version, summary=summary, db=in_memory_db)

        # 6. Fourth submission after completion -> new job created for refresh
        job4, is_new4 = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
        assert is_new4 is True
        assert job4.id != job1.id
        assert job4.status == "pending"

    def test_partial_unique_index_database_safeguard(self, in_memory_db, sample_repo):
        """Verify the database partial unique index strictly prevents two active jobs for the same repo."""
        from sqlalchemy.exc import IntegrityError

        # Direct insertion bypassing application check
        job1 = CollectionRun(repository_id=sample_repo.id, mode="standard_diagnostic", status="pending")
        in_memory_db.add(job1)
        in_memory_db.commit()

        # Second direct insertion of active status must raise IntegrityError
        job2 = CollectionRun(repository_id=sample_repo.id, mode="standard_diagnostic", status="in_progress")
        in_memory_db.add(job2)
        with pytest.raises(IntegrityError):
            in_memory_db.commit()
        in_memory_db.rollback()

    def test_lease_expiration_and_job_recovery_by_second_worker(self, in_memory_db, sample_repo):
        """Verify that an expired in-progress job is claimed and recovered by a successor worker."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

        worker1 = "worker-primary:100:1111"
        worker2 = "worker-successor:200:2222"

        # Worker 1 claims job
        claimed_1 = queue.claim_next_job(worker_id=worker1, lease_duration_seconds=60, db=in_memory_db)
        assert claimed_1.lease_version == 1

        # Simulate lease expiration: set leased_until into the past
        past_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        claimed_1.leased_until = past_time
        in_memory_db.commit()

        # Worker 2 claims next job: should detect expired job and recover it
        claimed_2 = queue.claim_next_job(worker_id=worker2, lease_duration_seconds=600, db=in_memory_db)
        assert claimed_2 is not None
        assert claimed_2.id == job.id
        assert claimed_2.worker_id == worker2
        assert claimed_2.status == "in_progress"
        assert claimed_2.retry_count == 1
        assert claimed_2.lease_version == 2
        assert ensure_utc(claimed_2.leased_until) > datetime.now(timezone.utc)

    def test_heartbeat_and_lease_renewal(self, in_memory_db, sample_repo):
        """Verify that heartbeat renewal successfully extends the lease expiration."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
        worker_id = "worker-heartbeat:300:3333"

        claimed = queue.claim_next_job(worker_id=worker_id, lease_duration_seconds=30, db=in_memory_db)
        original_expiry = claimed.leased_until

        # Renew lease with 600s extension
        renewed = queue.renew_lease(
            job_id=claimed.id,
            worker_id=worker_id,
            lease_version=claimed.lease_version,
            extension_seconds=600,
            db=in_memory_db,
        )
        assert renewed is True

        in_memory_db.refresh(claimed)
        assert ensure_utc(claimed.leased_until) > ensure_utc(original_expiry)
        assert ensure_utc(claimed.leased_until) > datetime.now(timezone.utc) + timedelta(seconds=550)

    def test_stale_worker_rejected_by_lease_version_fencing_on_complete(self, in_memory_db, sample_repo):
        """Verify that a worker whose lease was superseded cannot write results and raises JobLeaseLostException."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

        worker1 = "worker-stale:401:aaaa"
        worker2 = "worker-active:402:bbbb"

        # Worker 1 claims job (lease_version = 1)
        claimed_1 = queue.claim_next_job(worker_id=worker1, lease_duration_seconds=10, db=in_memory_db)
        worker1_version = claimed_1.lease_version
        assert worker1_version == 1

        # Simulate lease expiration and recovery by Worker 2 (lease_version becomes 2)
        claimed_1.leased_until = datetime.now(timezone.utc) - timedelta(seconds=30)
        in_memory_db.commit()

        claimed_2 = queue.claim_next_job(worker_id=worker2, lease_duration_seconds=600, db=in_memory_db)
        assert claimed_2.lease_version == 2
        assert claimed_2.worker_id == worker2

        # Stale Worker 1 attempts to finalize job with lease_version=1
        summary = IngestionSummary(
            repo_name=sample_repo.repo_name,
            mode="standard_diagnostic",
            is_partial=False,
            total_requests_consumed=25,
            items_collected={"commits": 100},
            entity_status={"commits": "completed"},
            execution_time_seconds=12.0,
            errors=[],
        )

        with pytest.raises(JobLeaseLostException) as exc_info:
            queue.complete_job(
                job_id=job.id,
                worker_id=worker1,
                lease_version=worker1_version,
                summary=summary,
                db=in_memory_db,
            )

        assert "Fencing token revoked" in str(exc_info.value)

        # Confirm job state in DB was untouched by Worker 1
        in_memory_db.refresh(job)
        assert job.status == "in_progress"
        assert job.worker_id == worker2
        assert job.lease_version == 2
        assert job.completed_at is None

    def test_stale_worker_rejected_by_lease_version_fencing_on_fail(self, in_memory_db, sample_repo):
        """Verify that a stale worker cannot mark a recovered job as failed."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

        worker1 = "worker-stale-fail:501:cccc"
        claimed_1 = queue.claim_next_job(worker_id=worker1, lease_duration_seconds=10, db=in_memory_db)

        # Reassign lease manually simulating recovery
        job.lease_version = 5
        job.worker_id = "worker-new:502:dddd"
        in_memory_db.commit()

        # Worker 1 attempts fail_job with old lease_version=1
        with pytest.raises(JobLeaseLostException):
            queue.fail_job(
                job_id=job.id,
                worker_id=worker1,
                lease_version=1,
                error_message="Something failed",
                db=in_memory_db,
            )

    def test_recover_expired_jobs_exceeding_max_retries(self, in_memory_db, sample_repo):
        """Verify that abandoned jobs whose retry count has reached max_retries are marked as permanently failed."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", max_retries=2, db=in_memory_db)

        # Set job to expired and retries exhausted
        job.status = "in_progress"
        job.worker_id = "worker-crashed:600:dead"
        job.leased_until = datetime.now(timezone.utc) - timedelta(minutes=10)
        job.retry_count = 2  # equals max_retries
        in_memory_db.commit()

        # Claiming should not return it
        claimed = queue.claim_next_job(worker_id="worker-alive:601:live", db=in_memory_db)
        assert claimed is None

        # Recovery pass marks it as permanently failed
        result = queue.recover_expired_jobs(max_retries=2, db=in_memory_db)
        assert result["failed_count"] == 1

        in_memory_db.refresh(job)
        assert job.status == "failed"
        assert "max retry limit (2) was exceeded" in job.error_message


class TestPostgreSQLClaimSimulation:
    """Test PostgreSQL dialect claim logic using mocked execution."""

    def test_simulated_postgresql_claim_behavior(self, sample_repo):
        """Verify that when dialect is detected as 'postgresql', with_for_update(skip_locked=True) is applied."""
        from sqlalchemy.dialects import postgresql

        queue = DurableJobQueue()
        mock_db = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_db.get_bind.return_value = mock_bind

        pending_job = CollectionRun(
            id=42,
            repository_id=sample_repo.id,
            mode="standard_diagnostic",
            status="pending",
            lease_version=0,
            retry_count=0,
            max_retries=2,
            is_partial=False,
        )
        mock_db.scalar.return_value = pending_job

        claimed = queue.claim_next_job(worker_id="pg-worker:901:abcd", db=mock_db)
        assert claimed is not None
        assert claimed.worker_id == "pg-worker:901:abcd"
        assert claimed.status == "in_progress"
        assert claimed.lease_version == 1

        # Verify that scalar was called with a query compiled with FOR UPDATE SKIP LOCKED
        call_args = mock_db.scalar.call_args[0][0]
        compiled = str(call_args.compile(dialect=postgresql.dialect()))
        assert "FOR UPDATE SKIP LOCKED" in compiled


class TestIngestionWorkerExecution:
    """Integration tests for IngestionWorker lifecycle execution."""

    def test_worker_process_one_job_success(self, in_memory_db, sample_repo):
        """Verify worker successfully claims, executes mock collection, and finalizes job."""
        async def _run():
            queue = DurableJobQueue()
            job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

            worker = IngestionWorker(worker_id="test-worker:1", job_queue=queue, heartbeat_interval_seconds=0.1)

            # Mock RepositoryCollector
            mock_collector = MagicMock()
            mock_collector.collect_repository = AsyncMock(
                return_value=IngestionSummary(
                    repo_name=sample_repo.repo_name,
                    mode="standard_diagnostic",
                    is_partial=False,
                    total_requests_consumed=15,
                    items_collected={"commits": 100, "issues": 50},
                    entity_status={"commits": "completed", "issues": "completed"},
                    execution_time_seconds=2.5,
                    errors=[],
                )
            )

            processed_id = await worker.process_one_job(
                db=in_memory_db,
                collector=mock_collector,
            )

            assert processed_id == job.id

            in_memory_db.refresh(job)
            assert job.status == "completed"
            assert job.is_partial is False
            assert job.rate_limit_consumed == 15
            assert job.items_collected["commits"] == 100
            assert job.coverage_summary["entity_status"]["commits"] == "completed"

        asyncio.run(_run())

    def test_worker_process_one_job_rate_limited(self, in_memory_db, sample_repo):
        """Verify worker finalizes job with status 'rate_limited' when resumes_at is returned."""
        async def _run():
            queue = DurableJobQueue()
            job, _ = queue.submit_job(repo_id=sample_repo.id, mode="fast_snapshot", db=in_memory_db)

            worker = IngestionWorker(worker_id="test-worker:2", job_queue=queue, heartbeat_interval_seconds=0.1)

            resumes_dt = datetime.now(timezone.utc) + timedelta(minutes=15)
            mock_collector = MagicMock()
            mock_collector.collect_repository = AsyncMock(
                return_value=IngestionSummary(
                    repo_name=sample_repo.repo_name,
                    mode="fast_snapshot",
                    is_partial=True,
                    partial_reason="Primary GitHub rate limit threshold reached",
                    resumes_at=resumes_dt,
                    total_requests_consumed=8,
                    items_collected={"commits": 50},
                    entity_status={"commits": "truncated"},
                    execution_time_seconds=1.8,
                    errors=[],
                )
            )

            processed_id = await worker.process_one_job(
                db=in_memory_db,
                collector=mock_collector,
            )

            assert processed_id == job.id

            in_memory_db.refresh(job)
            assert job.status == "rate_limited"
            assert job.is_partial is True
            assert job.coverage_summary["resumes_at"] == resumes_dt.isoformat()

        asyncio.run(_run())

    def test_worker_process_one_job_crash_marks_failed(self, in_memory_db, sample_repo):
        """Verify worker marks job as 'failed' with error_message when collector raises exception."""
        async def _run():
            queue = DurableJobQueue()
            job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

            worker = IngestionWorker(worker_id="test-worker:3", job_queue=queue, heartbeat_interval_seconds=0.1)

            mock_collector = MagicMock()
            mock_collector.collect_repository = AsyncMock(side_effect=RuntimeError("Simulated upstream network crash"))

            with pytest.raises(RuntimeError) as exc_info:
                await worker.process_one_job(
                    db=in_memory_db,
                    collector=mock_collector,
                )

            assert "Simulated upstream network crash" in str(exc_info.value)

            in_memory_db.refresh(job)
            assert job.status == "failed"
            assert "Simulated upstream network crash" in job.error_message

        asyncio.run(_run())

    def test_worker_fenced_out_during_execution_raises_lease_lost(self, in_memory_db, sample_repo):
        """Verify worker whose lease was superseded during collection pass aborts with JobLeaseLostException."""
        async def _run():
            queue = DurableJobQueue()
            job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)

            worker = IngestionWorker(worker_id="test-worker:fenced", job_queue=queue, heartbeat_interval_seconds=0.05)

            async def _stealing_collector(*args, **kwargs):
                await asyncio.sleep(0.1)
                # Reclaim job in DB under another worker to invalidate original worker's fencing token
                job.lease_version = 99
                job.worker_id = "worker-thief:999"
                in_memory_db.commit()
                return IngestionSummary(
                    repo_name=sample_repo.repo_name,
                    mode="standard_diagnostic",
                    is_partial=False,
                    total_requests_consumed=5,
                    items_collected={"commits": 10},
                    entity_status={"commits": "completed"},
                    execution_time_seconds=0.2,
                    errors=[],
                )

            mock_collector = MagicMock()
            mock_collector.collect_repository = AsyncMock(side_effect=_stealing_collector)

            with pytest.raises(JobLeaseLostException):
                await worker.process_one_job(
                    db=in_memory_db,
                    collector=mock_collector,
                )

            # Verify that the DB state remained under the thief (version 99), not overwritten
            in_memory_db.refresh(job)
            assert job.lease_version == 99
            assert job.worker_id == "worker-thief:999"
            assert job.status == "in_progress"
            assert job.completed_at is None

        asyncio.run(_run())

    def test_complete_job_exposes_top_level_stream_statuses_preserving_entity_status(
        self, in_memory_db, sample_repo
    ):
        """Verify coverage_summary promotes stream keys cleanly without losing entity_status or reserved keys."""
        queue = DurableJobQueue()
        job, _ = queue.submit_job(repo_id=sample_repo.id, mode="fast_snapshot", db=in_memory_db)
        claimed = queue.claim_next_job(worker_id="worker-cov:1", db=in_memory_db)
        assert claimed is not None

        summary = IngestionSummary(
            repo_name=sample_repo.repo_name,
            mode="fast_snapshot",
            is_partial=True,
            partial_reason="Budget ceiling reached",
            total_requests_consumed=10,
            items_collected={"commits": 50, "pull_requests": 10},
            entity_status={
                "commits": "completed",
                "pull_requests": "truncated",
                "pr_reviews": "truncated",
                "pr_inline_comments": "skipped",
                "issues": "completed",
                "errors": "ignored_reserved_key_collision",
            },
            execution_time_seconds=1.25,
            errors=[],
        )
        queue.complete_job(
            job_id=job.id,
            worker_id="worker-cov:1",
            lease_version=claimed.lease_version,
            summary=summary,
            db=in_memory_db,
        )

        in_memory_db.refresh(job)
        assert job.status == "completed_partial"
        assert job.coverage_summary["commits"] == "completed"
        assert job.coverage_summary["pull_requests"] == "truncated"
        assert job.coverage_summary["pr_reviews"] == "truncated"
        assert job.coverage_summary["pr_inline_comments"] == "skipped"
        assert job.coverage_summary["entity_status"]["commits"] == "completed"
        assert job.coverage_summary["entity_status"]["pr_inline_comments"] == "skipped"
        # Reserved envelope keys cannot be overwritten by an entity_status entry
        assert job.coverage_summary["errors"] == []
        assert job.coverage_summary["partial_reason"] == "Budget ceiling reached"

    def test_worker_collector_summary_errors_marks_job_failed_not_completed(
        self, in_memory_db, sample_repo
    ):
        """Verify collector returning non-empty summary.errors transitions CollectionRun to 'failed'."""
        async def _run():
            queue = DurableJobQueue()
            job, _ = queue.submit_job(repo_id=sample_repo.id, mode="standard_diagnostic", db=in_memory_db)
            worker = IngestionWorker(
                worker_id="worker-err:1",
                job_queue=queue,
                heartbeat_interval_seconds=0.1,
            )

            mock_collector = MagicMock()
            mock_collector.collect_repository = AsyncMock(
                return_value=IngestionSummary(
                    repo_name=sample_repo.repo_name,
                    mode="standard_diagnostic",
                    is_partial=False,
                    resumes_at=None,
                    total_requests_consumed=3,
                    items_collected={"commits": 0},
                    entity_status={"repository": "completed", "commits": "failed"},
                    execution_time_seconds=0.4,
                    errors=["Unexpected SQL integrity failure during commit batch"],
                )
            )

            processed_id = await worker.process_one_job(db=in_memory_db, collector=mock_collector)
            assert processed_id == job.id

            in_memory_db.refresh(job)
            assert job.status == "failed"
            assert job.error_message == "Unexpected SQL integrity failure during commit batch"
            assert job.coverage_summary["errors"] == [
                "Unexpected SQL integrity failure during commit batch"
            ]
            assert job.coverage_summary["commits"] == "failed"
            assert job.coverage_summary["entity_status"]["commits"] == "failed"

        asyncio.run(_run())

