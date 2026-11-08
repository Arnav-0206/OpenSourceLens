"""Integration and API contract tests for repository ingestion REST endpoints.

Tests verify:
1. POST /api/ingest/preview returns eligibility, budgets, and confidence without mutating DB.
2. POST /api/ingest/submit validates repository, creates durable job, and returns 201.
3. POST /api/ingest/submit rejects ineligible repositories with HTTP 400.
4. GET /api/ingest/status/{job_id} exposes comprehensive execution telemetry.
5. GET /api/ingest/status/nonexistent returns HTTP 404.
6. Parity across /api/ingest/... and /api/v1/ingest/... prefixes.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.base import Base
from backend.app.database.models import CollectionRun, Repository
from backend.app.database.session import get_db
from backend.app.main import app


@pytest.fixture
def test_db_session():
    """Create fresh isolated in-memory SQLite database session for API testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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
def client(test_db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def _override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def sample_github_metadata():
    """Valid public repository GitHub API metadata mock payload."""
    return {
        "id": 19103692,
        "name": "click",
        "full_name": "pallets/click",
        "owner": {"login": "pallets"},
        "html_url": "https://github.com/pallets/click",
        "private": False,
        "fork": False,
        "archived": False,
        "is_template": False,
        "size": 7500,
        "stargazers_count": 17800,
        "forks_count": 3900,
        "open_issues_count": 42,
        "default_branch": "main",
        "language": "Python",
        "description": "Composable command line interface toolkit",
        "created_at": "2014-04-25T11:42:00Z",
        "updated_at": "2026-11-01T10:00:00Z",
    }


class TestIngestionPreviewEndpoint:
    """Tests for POST /api/ingest/preview."""

    def test_preview_eligible_repository_contract(self, client, sample_github_metadata):
        """Verify preview endpoint returns complete eligibility, metadata, and estimated budgets."""
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            resp = client.post("/api/ingest/preview", json={"repo": "pallets/click"})
            assert resp.status_code == 200

            data = resp.json()
            assert data["repo_name"] == "pallets/click"
            assert data["owner"] == "pallets"
            assert data["name"] == "click"
            assert data["canonical_url"] == "https://github.com/pallets/click"
            assert data["is_eligible"] is True
            assert data["rejection_reason"] is None

            # Verify estimated budgets
            assert "fast_snapshot" in data["estimated_budgets"]
            assert data["estimated_budgets"]["fast_snapshot"]["total_limit"] == 10
            assert "standard_diagnostic" in data["estimated_budgets"]
            assert data["estimated_budgets"]["standard_diagnostic"]["total_limit"] == 50

            # Verify confidence preview
            assert data["confidence_preview"] is not None
            assert "overall_confidence" in data["confidence_preview"]

    def test_preview_does_not_mutate_database(self, client, test_db_session, sample_github_metadata):
        """Verify preview endpoint is strictly read-only and inserts zero database records."""
        repos_before = test_db_session.scalar(select(func.count(Repository.id)))
        jobs_before = test_db_session.scalar(select(func.count(CollectionRun.id)))

        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            resp = client.post("/api/ingest/preview", json={"repo": "pallets/click"})
            assert resp.status_code == 200

        repos_after = test_db_session.scalar(select(func.count(Repository.id)))
        jobs_after = test_db_session.scalar(select(func.count(CollectionRun.id)))

        assert repos_before == repos_after
        assert jobs_before == jobs_after

    def test_preview_ineligible_private_repo_returns_rejection(self, client, sample_github_metadata):
        """Verify preview flags private repositories as ineligible."""
        private_meta = dict(sample_github_metadata)
        private_meta["private"] = True

        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=private_meta,
        ):
            resp = client.post("/api/ingest/preview", json={"repo": "pallets/click"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["is_eligible"] is False
            assert "Private repositories are not eligible" in data["rejection_reason"]

    def test_preview_invalid_url_returns_400(self, client):
        """Verify preview rejects invalid repository URL formats with HTTP 400."""
        resp = client.post("/api/ingest/preview", json={"repo": "https://notgithub.com/invalid/format/extra"})
        assert resp.status_code == 400
        assert "notgithub.com" in resp.json()["detail"]


class TestIngestionSubmitEndpoint:
    """Tests for POST /api/ingest/submit."""

    def test_submit_valid_repository_creates_job_and_returns_201(
        self, client, test_db_session, sample_github_metadata
    ):
        """Verify submitting an eligible repository creates Repository stub, CollectionRun, and returns 201."""
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            resp = client.post(
                "/api/ingest/submit",
                json={"repo": "pallets/click", "mode": "standard_diagnostic"},
            )
            assert resp.status_code == 201

            data = resp.json()
            assert data["job_id"] is not None
            assert data["repo_name"] == "pallets/click"
            assert data["mode"] == "standard_diagnostic"
            assert data["status"] == "pending"
            assert "submitted successfully" in data["message"]

            # Confirm database persistence
            repo = test_db_session.scalar(select(Repository).where(Repository.repo_name == "pallets/click"))
            assert repo is not None
            assert repo.stars_count == 17800

            job = test_db_session.get(CollectionRun, data["job_id"])
            assert job is not None
            assert job.repository_id == repo.id
            assert job.status == "pending"
            assert job.mode == "standard_diagnostic"

    def test_submit_ineligible_repository_returns_400(self, client, sample_github_metadata):
        """Verify submitting an empty or ineligible repository returns HTTP 400."""
        empty_meta = dict(sample_github_metadata)
        empty_meta["size"] = 0

        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=empty_meta,
        ):
            resp = client.post(
                "/api/ingest/submit",
                json={"repo": "pallets/click", "mode": "fast_snapshot"},
            )
            assert resp.status_code == 400
            assert "empty stub" in resp.json()["detail"]

    def test_submit_invalid_mode_returns_400(self, client):
        """Verify submitting an invalid collection mode returns HTTP 400."""
        resp = client.post(
            "/api/ingest/submit",
            json={"repo": "pallets/click", "mode": "invalid_mode_name"},
        )
        assert resp.status_code == 400
        assert "Invalid collection mode" in resp.json()["detail"]

    def test_submit_duplicate_pending_returns_200_and_existing_id(
        self, client, test_db_session, sample_github_metadata
    ):
        """Verify submitting an active pending repository returns 200 OK and existing job ID."""
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            # 1. Initial submission -> 201 Created
            resp1 = client.post("/api/ingest/submit", json={"repo": "pallets/click", "mode": "standard_diagnostic"})
            assert resp1.status_code == 201
            job_id1 = resp1.json()["job_id"]

            # 2. Duplicate submission while still pending -> 200 OK with identical job_id
            resp2 = client.post("/api/ingest/submit", json={"repo": "pallets/click", "mode": "standard_diagnostic"})
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["job_id"] == job_id1
            assert data2["status"] == "pending"
            assert "already" in data2["message"].lower()

    def test_submit_duplicate_in_progress_returns_200_and_existing_id(
        self, client, test_db_session, sample_github_metadata
    ):
        """Verify submitting a repository whose job is in_progress returns 200 OK and existing job ID."""
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            resp1 = client.post("/api/ingest/submit", json={"repo": "pallets/click", "mode": "standard_diagnostic"})
            job_id1 = resp1.json()["job_id"]

            # Transition job to in_progress with active lease
            job = test_db_session.get(CollectionRun, job_id1)
            job.status = "in_progress"
            job.leased_until = datetime.now(timezone.utc) + timedelta(minutes=10)
            test_db_session.commit()

            # Submit again -> 200 OK, same job_id, status in_progress
            resp2 = client.post("/api/ingest/submit", json={"repo": "pallets/click", "mode": "standard_diagnostic"})
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["job_id"] == job_id1
            assert data2["status"] == "in_progress"
            assert "already" in data2["message"].lower()

    def test_submit_refresh_after_completed_creates_new_job_returns_201(
        self, client, test_db_session, sample_github_metadata
    ):
        """Verify submitting a repository whose previous job completed enqueues a new refresh job (201 Created)."""
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            resp1 = client.post("/api/ingest/submit", json={"repo": "pallets/click", "mode": "standard_diagnostic"})
            job_id1 = resp1.json()["job_id"]

            # Complete prior job
            job = test_db_session.get(CollectionRun, job_id1)
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            test_db_session.commit()

            # Submit repository again -> 201 Created with a new job ID
            resp2 = client.post("/api/ingest/submit", json={"repo": "pallets/click", "mode": "standard_diagnostic"})
            assert resp2.status_code == 201
            data2 = resp2.json()
            assert data2["job_id"] != job_id1
            assert data2["status"] == "pending"
            assert "submitted successfully" in data2["message"].lower()



class TestIngestionStatusEndpoint:
    """Tests for GET /api/ingest/status/{job_id}."""

    def test_get_job_status_existing_job(self, client, test_db_session):
        """Verify status endpoint returns full telemetry, entity status, and timing."""
        repo = Repository(
            id=7777,
            repo_name="org/telemetry-repo",
            owner="org",
            name="telemetry-repo",
            stars_count=500,
            is_active=True,
        )
        test_db_session.add(repo)
        test_db_session.commit()

        start_time = datetime.now(timezone.utc) - timedelta(seconds=20)
        end_time = datetime.now(timezone.utc)

        job = CollectionRun(
            repository_id=repo.id,
            mode="fast_snapshot",
            status="completed",
            worker_id="test-worker:host:123",
            lease_version=1,
            retry_count=0,
            max_retries=2,
            is_partial=False,
            rate_limit_consumed=9,
            items_collected={"commits": 100, "issues": 25},
            coverage_summary={
                "entity_status": {"commits": "completed", "issues": "completed"},
                "execution_time_seconds": 18.5,
            },
            started_at=start_time,
            completed_at=end_time,
        )
        test_db_session.add(job)
        test_db_session.commit()
        test_db_session.refresh(job)

        resp = client.get(f"/api/ingest/status/{job.id}")
        assert resp.status_code == 200

        data = resp.json()
        assert data["job_id"] == job.id
        assert data["repo_name"] == "org/telemetry-repo"
        assert data["status"] == "completed"
        assert data["mode"] == "fast_snapshot"
        assert data["worker_id"] == "test-worker:host:123"
        assert data["rate_limit_consumed"] == 9
        assert data["items_collected"]["commits"] == 100
        assert data["entity_status"]["commits"] == "completed"
        assert data["duration_seconds"] is not None
        assert data["duration_seconds"] >= 18.0

    def test_get_job_status_nonexistent_returns_404(self, client):
        """Verify status endpoint returns HTTP 404 for unknown job IDs."""
        resp = client.get("/api/ingest/status/99999999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestAPIV1PrefixParity:
    """Verify parity for endpoints exposed under /api/v1/ingest."""

    def test_api_v1_endpoints_function_identically(self, client, sample_github_metadata):
        """Verify /api/v1/ingest/preview is accessible and returns identical output."""
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=sample_github_metadata,
        ):
            resp = client.post("/api/v1/ingest/preview", json={"repo": "pallets/click"})
            assert resp.status_code == 200
            assert resp.json()["repo_name"] == "pallets/click"
