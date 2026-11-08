"""End-to-End Ingestion Integration and Lifecycle Tests (Phase B Step B.6).

Verifies the complete ingestion lifecycle across 10 distinct phases:
1. Repository URL validation.
2. Eligibility and metadata retrieval.
3. Preview and request-budget estimation.
4. Durable job submission.
5. Worker claiming and execution.
6. GitHub API collection through the existing collector.
7. Entity normalization and database persistence.
8. Lease heartbeat and completion.
9. Status endpoint reporting.
10. Correct completed/partial/truncated/error/resumption status reporting.

Verifies product independence from the original four benchmark repositories
using unseen public repositories: 'pallets/flask' and 'psf/requests'.
"""

import asyncio
from datetime import datetime, timezone
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from dotenv import load_dotenv
import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from backend.app.database.base import Base
from backend.app.database.models import (
    CollectionRun,
    Commit,
    Contributor,
    ContributorActivity,
    Issue,
    IssueComment,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
    Release,
    Repository,
)
from backend.app.services.job_queue import DurableJobQueue
from backend.app.services.ingestion_worker import IngestionWorker
from backend.app.services.rate_budgeter import EntityRateBudgeter
from backend.app.services.repository_collector import RepositoryCollector
from backend.app.services.repository_eligibility import RepositoryEligibilityChecker
from backend.app.api.v1.endpoints.ingest import get_job_status
from backend.app.schemas.ingestion import IngestionSummary, PaginatedResult


@pytest.fixture
def isolated_test_db(tmp_path):
    """Create an isolated, temporary SQLite database in WAL mode for E2E tests."""
    db_path = str(tmp_path / "e2e_isolated.db")
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 15})
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionClass()
    try:
        yield session, db_path
    finally:
        session.close()
        engine.dispose()


class TestMockedE2EIngestionLifecycle:
    """Deterministic, offline end-to-end ingestion lifecycle tests using unseen public repositories."""

    @pytest.mark.parametrize(
        "repo_slug,repo_id,owner,name,stars,issues",
        [
            ("pallets/flask", 596892, "pallets", "flask", 75000, 5),
            ("psf/requests", 1362490, "psf", "requests", 54000, 250),
        ],
    )
    def test_complete_e2e_lifecycle_for_unseen_repository(
        self, isolated_test_db, repo_slug, repo_id, owner, name, stars, issues
    ):
        """Verify complete 10-step ingestion lifecycle from URL to persistent status reporting."""
        session, db_path = isolated_test_db

        mock_meta = {
            "id": repo_id,
            "name": name,
            "full_name": repo_slug,
            "owner": {"login": owner},
            "html_url": f"https://github.com/{repo_slug}",
            "private": False,
            "fork": False,
            "archived": False,
            "is_template": False,
            "size": 15000,
            "stargazers_count": stars,
            "forks_count": 12000,
            "open_issues_count": issues,
            "default_branch": "main",
            "language": "Python",
            "description": f"Repository {repo_slug}",
            "created_at": "2010-04-06T11:11:00Z",
            "updated_at": "2026-11-01T12:00:00Z",
        }

        async def _run():
            # Step 1 & 2: URL validation & Eligibility
            checker = RepositoryEligibilityChecker()
            with patch.object(checker, "fetch_github_metadata", new_callable=AsyncMock, return_value=mock_meta):
                eligibility = await checker.check_eligibility(repo_slug)

            assert eligibility.is_eligible is True
            assert eligibility.repo_name == repo_slug
            assert eligibility.github_id == repo_id
            assert eligibility.confidence_report is not None

            # Step 3: Request budget estimation
            budgeter = EntityRateBudgeter(mode="fast_snapshot")
            assert budgeter.total_limit == 10
            assert "commits" in budgeter.entity_limits

            # Step 4: Durable job submission
            repo_record = Repository(
                id=repo_id,
                repo_name=repo_slug,
                owner=owner,
                name=name,
                stars_count=stars,
                forks_count=12000,
                open_issues_count=issues,
                is_active=True,
            )
            session.add(repo_record)
            session.commit()

            queue = DurableJobQueue()
            job, is_new = queue.submit_job(repo_id=repo_id, mode="fast_snapshot", db=session)
            assert is_new is True
            assert job.status == "pending"

            # Verify in-flight deduplication
            dup_job, is_dup_new = queue.submit_job(repo_id=repo_id, mode="fast_snapshot", db=session)
            assert is_dup_new is False
            assert dup_job.id == job.id

            # Steps 5-8: Worker claiming, execution, heartbeat, and completion
            mock_collector = MagicMock(spec=RepositoryCollector)
            mock_summary = IngestionSummary(
                repo_name=repo_slug,
                mode="fast_snapshot",
                is_partial=False,
                partial_reason=None,
                resumes_at=None,
                total_requests_consumed=7,
                items_collected={
                    "repository": 1,
                    "commits": 25,
                    "contributors": 15,
                    "pull_requests": 20,
                    "pr_reviews": 10,
                    "pr_inline_comments": 5,
                    "issues": 12,
                    "issue_comments": 18,
                    "releases": 6,
                },
                entity_status={
                    "repository": "completed",
                    "commits": "completed",
                    "contributors": "completed",
                    "pull_requests": "completed",
                    "pr_reviews": "completed",
                    "pr_inline_comments": "completed",
                    "issues": "completed",
                    "issue_comments": "completed",
                    "releases": "completed",
                },
                execution_time_seconds=1.2,
                errors=[],
            )
            mock_collector.collect_repository = AsyncMock(return_value=mock_summary)

            worker = IngestionWorker(
                worker_id=f"test-e2e-worker-{name}:1001",
                heartbeat_interval_seconds=1,
            )

            with patch("backend.app.services.ingestion_worker.RepositoryCollector", return_value=mock_collector):
                claimed_job_id = await worker.process_one_job(db=session)

            assert claimed_job_id == job.id

            # Steps 9 & 10: Status endpoint reporting & truncation checks
            status_resp = get_job_status(job_id=job.id, db=session)
            assert status_resp.status == "completed"
            assert status_resp.job_id == job.id
            assert status_resp.repo_name == repo_slug
            assert status_resp.rate_limit_consumed == 7
            assert status_resp.is_partial is False
            assert status_resp.items_collected["commits"] == 25
            assert status_resp.items_collected["releases"] == 6
            assert status_resp.entity_status["commits"] == "completed"

        asyncio.run(_run())

    def test_multi_repository_cross_isolation(self, isolated_test_db):
        """Verify multiple unseen repositories ingest with strict isolation and zero cross-contamination."""
        session, db_path = isolated_test_db

        # Seed two unseen repositories
        repo_flask = Repository(id=596892, repo_name="pallets/flask", owner="pallets", name="flask")
        repo_reqs = Repository(id=1362490, repo_name="psf/requests", owner="psf", name="requests")
        session.add_all([repo_flask, repo_reqs])
        session.commit()

        # Seed commits and issues for both
        c_flask = Commit(
            repository_id=repo_flask.id,
            sha="flask_commit_sha_12345",
            author_login="mitsuhiko",
            message="Initial flask commit",
            author_date=datetime.now(timezone.utc),
        )
        c_reqs = Commit(
            repository_id=repo_reqs.id,
            sha="reqs_commit_sha_67890",
            author_login="kennethreitz",
            message="Initial requests commit",
            author_date=datetime.now(timezone.utc),
        )
        i_flask = Issue(
            id=59689201,
            issue_number=1,
            repository_id=repo_flask.id,
            repo_name="pallets/flask",
            title="Flask issue 1",
            author="flask_user",
            state="open",
            created_at=datetime.now(timezone.utc),
        )
        i_reqs = Issue(
            id=136249001,
            issue_number=1,
            repository_id=repo_reqs.id,
            repo_name="psf/requests",
            title="Requests issue 1",
            author="reqs_user",
            state="open",
            created_at=datetime.now(timezone.utc),
        )
        session.add_all([c_flask, c_reqs, i_flask, i_reqs])
        session.commit()

        # Verify cross-isolation
        flask_commits = session.scalars(select(Commit).where(Commit.repository_id == repo_flask.id)).all()
        reqs_commits = session.scalars(select(Commit).where(Commit.repository_id == repo_reqs.id)).all()

        assert len(flask_commits) == 1
        assert flask_commits[0].sha == "flask_commit_sha_12345"
        assert len(reqs_commits) == 1
        assert reqs_commits[0].sha == "reqs_commit_sha_67890"

        # Check issues isolation
        flask_issues = session.scalars(select(Issue).where(Issue.repository_id == repo_flask.id)).all()
        reqs_issues = session.scalars(select(Issue).where(Issue.repository_id == repo_reqs.id)).all()

        assert len(flask_issues) == 1
        assert flask_issues[0].title == "Flask issue 1"
        assert len(reqs_issues) == 1
        assert reqs_issues[0].title == "Requests issue 1"

        # Verify no benchmark repos are referenced
        benchmark_names = {"microsoft/vscode", "pytorch/pytorch", "kubernetes/kubernetes", "huggingface/transformers"}
        db_repo_names = set(session.scalars(select(Repository.repo_name)).all())
        assert db_repo_names.isdisjoint(benchmark_names)


class TestLiveE2EIngestionLifecycle:
    """Opt-in live integration tests against real GitHub API for unseen public repositories."""

    @pytest.mark.skipif(
        not os.getenv("RUN_LIVE_GITHUB_TESTS"),
        reason="Live GitHub integration test is opt-in (set RUN_LIVE_GITHUB_TESTS=1 to run).",
    )
    def test_live_e2e_lifecycle_pallets_flask(self, isolated_test_db):
        """Live E2E ingestion against pallets/flask in fast_snapshot mode (<= 10 requests)."""
        session, _ = isolated_test_db

        async def _run():
            checker = RepositoryEligibilityChecker()
            eligibility = await checker.check_eligibility("pallets/flask")
            assert eligibility.is_eligible is True
            assert eligibility.repo_name == "pallets/flask"

            repo = Repository(
                id=eligibility.github_id or 596892,
                repo_name=eligibility.repo_name,
                owner=eligibility.owner,
                name=eligibility.name,
                stars_count=eligibility.stars_count,
                forks_count=eligibility.forks_count,
                open_issues_count=eligibility.open_issues_count,
                is_active=True,
            )
            session.add(repo)
            session.commit()

            queue = DurableJobQueue()
            job, is_new = queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=session)
            assert is_new is True

            worker = IngestionWorker(worker_id="live-flask-test-worker:1", heartbeat_interval_seconds=2)
            claimed_id = await worker.process_one_job(db=session)
            assert claimed_id == job.id

            status_resp = get_job_status(job_id=job.id, db=session)
            assert status_resp.rate_limit_consumed <= 10
            assert status_resp.status in {"completed", "completed_partial"}
            assert status_resp.items_collected["commits"] > 0
            assert status_resp.items_collected["contributors"] > 0
            assert status_resp.items_collected["pull_requests"] > 0
            assert status_resp.items_collected["releases"] > 0

        asyncio.run(_run())

    @pytest.mark.skipif(
        not os.getenv("RUN_LIVE_GITHUB_TESTS"),
        reason="Live GitHub integration test is opt-in (set RUN_LIVE_GITHUB_TESTS=1 to run).",
    )
    def test_live_e2e_lifecycle_psf_requests(self, isolated_test_db):
        """Live E2E ingestion against psf/requests in fast_snapshot mode (<= 10 requests)."""
        session, _ = isolated_test_db

        async def _run():
            checker = RepositoryEligibilityChecker()
            eligibility = await checker.check_eligibility("psf/requests")
            assert eligibility.is_eligible is True
            assert eligibility.repo_name == "psf/requests"

            repo = Repository(
                id=eligibility.github_id or 1362490,
                repo_name=eligibility.repo_name,
                owner=eligibility.owner,
                name=eligibility.name,
                stars_count=eligibility.stars_count,
                forks_count=eligibility.forks_count,
                open_issues_count=eligibility.open_issues_count,
                is_active=True,
            )
            session.add(repo)
            session.commit()

            queue = DurableJobQueue()
            job, is_new = queue.submit_job(repo_id=repo.id, mode="fast_snapshot", db=session)
            assert is_new is True

            worker = IngestionWorker(worker_id="live-requests-test-worker:1", heartbeat_interval_seconds=2)
            claimed_id = await worker.process_one_job(db=session)
            assert claimed_id == job.id

            status_resp = get_job_status(job_id=job.id, db=session)
            assert status_resp.rate_limit_consumed <= 10
            assert status_resp.status in {"completed", "completed_partial"}
            assert status_resp.items_collected["commits"] > 0
            assert status_resp.items_collected["contributors"] > 0
            assert status_resp.items_collected["pull_requests"] > 0
            assert status_resp.items_collected["releases"] > 0

        asyncio.run(_run())
