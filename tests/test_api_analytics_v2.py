"""Comprehensive API v2 & End-to-End Integration Tests for Phase C Step C.8.3.

Covers all 18 C.8.3 API, worker-lifecycle, concurrency, error-contract,
and full ingestion-to-analytics-API requirements using isolated databases
and dynamically seeded unseen repositories (zero benchmark DB mutation).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.core.exceptions import InvalidParameterException
from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    CollectionRun,
    Commit,
    Contributor,
    Issue,
    IssueComment,
    IssueLabel,
    PullRequest,
    PullRequestReview,
    Release,
    Repository,
    RepositoryAnalytics,
)
from backend.app.database.session import get_db
from backend.app.main import create_application
from backend.app.schemas.analytics_v2 import CURRENT_METHODOLOGY_VERSION
from backend.app.schemas.ingestion import IngestionSummary, PaginatedResult
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.orchestrator import (
    ALL_ANALYTICS_DOMAINS,
    AnalyticsOrchestrator,
)
from backend.app.services.analytics_queue import AnalyticsJobQueue
from backend.app.services.analytics_worker import AnalyticsWorker, run_standalone_worker
from backend.app.services.ingestion_worker import IngestionWorker
from backend.app.services.job_queue import DurableJobQueue
from backend.app.services.repository_collector import RepositoryCollector


@pytest.fixture
def isolated_api_db():
    """Isolated thread-safe in-memory SQLite session for FastAPI TestClient integration tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionClass()
    try:
        yield session, SessionClass
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def v2_client(isolated_api_db):
    """FastAPI TestClient backed cleanly by isolated_api_db."""
    session, _ = isolated_api_db
    test_app = create_application()

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    test_app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(test_app)
    try:
        yield client, session
    finally:
        test_app.dependency_overrides.clear()


def _seed_unseen_rich_repo(
    db: Session,
    *,
    repo_id: int = 9501,
    owner: str = "unseen-org",
    name: str = "lens-alpha",
    source_revision: int = 2,
    collection_mode: str = "full_history",
    is_partial: bool = False,
) -> Repository:
    """Seed an unseen public repository across all 6 analytical domains."""
    anchor = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
    repo = Repository(
        id=repo_id,
        repo_name=f"{owner}/{name}",
        owner=owner,
        name=name,
        description="Dynamic unseen repository for API v2 tests",
        primary_language="Python",
        stars_count=980,
        forks_count=140,
        open_issues_count=14,
        last_synced_at=anchor - timedelta(minutes=10),
        analytics_source_revision=source_revision,
        is_active=True,
        is_archived=False,
        is_fork=False,
    )
    db.add(repo)
    db.add(
        CollectionRun(
            repository_id=repo_id,
            mode=collection_mode,
            status="completed_partial" if is_partial else "completed",
            is_partial=is_partial,
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
            completed_at=anchor - timedelta(minutes=15),
        )
    )

    for c_i in range(4):
        db.add(
            Contributor(
                id=repo_id * 10 + c_i,
                repository_id=repo_id,
                github_id=7100 + c_i,
                login=f"contrib_{c_i}",
                git_email=f"contrib_{c_i}@example.org",
                is_bot=False,
                total_commits=10 - c_i * 2,
            )
        )

    for m_i in range(14):
        dt = anchor - timedelta(days=(13 - m_i) * 28 + 3)
        db.add(
            Commit(
                repository_id=repo_id,
                sha=f"sha_{repo_id}_{m_i}",
                author_login=f"contrib_{m_i % 4}",
                author_email=f"contrib_{m_i % 4}@example.org",
                author_id=7100 + (m_i % 4),
                message=f"feat: change #{m_i}",
                author_date=dt,
                committer_date=dt,
                additions=40 + m_i * 3,
                deletions=10 + m_i,
                total_changes=50 + m_i * 4,
            )
        )

    for p_i in range(6):
        created = anchor - timedelta(days=90 - p_i * 12)
        merged = created + timedelta(hours=16)
        pr = PullRequest(
            id=repo_id * 100 + p_i,
            repository_id=repo_id,
            pr_number=p_i + 1,
            title=f"PR #{p_i + 1}",
            state="closed",
            author="contrib_0",
            author_id=7100,
            created_at=created,
            closed_at=merged,
            merged_at=merged,
            additions=45,
            deletions=12,
            changed_files=3,
        )
        db.add(pr)
        db.flush()
        db.add(
            PullRequestReview(
                pull_request_id=pr.id,
                review_id=repo_id * 1000 + p_i,
                author="contrib_1",
                author_id=7101,
                state="APPROVED",
                submitted_at=created + timedelta(hours=3),
            )
        )

    for i_i in range(6):
        c_dt = anchor - timedelta(days=80 - i_i * 10)
        is_closed = i_i < 4
        cl_dt = (c_dt + timedelta(hours=24)) if is_closed else None
        iss = Issue(
            id=repo_id * 1000 + i_i,
            issue_number=i_i + 1,
            repository_id=repo_id,
            repo_name=repo.repo_name,
            title=f"Issue #{i_i + 1}",
            author="community_dev",
            state="closed" if is_closed else "open",
            canonical_label="bug" if i_i % 2 == 0 else "feature",
            speed_tier="1-7 Days" if is_closed else None,
            resolution_time_hours=24.0 if is_closed else None,
            comments_count=2,
            created_at=c_dt,
            closed_at=cl_dt,
        )
        db.add(iss)
        db.flush()
        db.add(
            IssueLabel(
                issue_id=iss.id,
                label_name="bug" if i_i % 2 == 0 else "enhancement",
                is_canonical=True,
                canonical_category="bug" if i_i % 2 == 0 else "feature",
            )
        )
        db.add(
            IssueComment(
                repository_id=repo_id,
                comment_id=repo_id * 10000 + i_i,
                issue_number=i_i + 1,
                issue_id=iss.id,
                author="contrib_2",
                body="Investigating root cause.",
                created_at=c_dt + timedelta(hours=2),
            )
        )

    for r_i, tag in enumerate(("v0.9.0", "v1.0.0", "v1.1.0")):
        pub = anchor - timedelta(days=150 - r_i * 45)
        db.add(
            Release(
                repository_id=repo_id,
                release_id=repo_id * 500 + r_i,
                tag_name=tag,
                name=f"Release {tag}",
                is_prerelease=False,
                published_at=pub,
            )
        )

    db.commit()
    return repo


class TestAnalyticsV2RecomputeAndStatusEndpoints:
    """Covers tests 1–6: recompute enqueue/dedup/cache/force and status telemetry."""

    def test_01_and_02_and_05_status_missing_then_recompute_new_then_active_dedup(self, v2_client):
        client, db = v2_client
        repo = _seed_unseen_rich_repo(db, repo_id=9501, source_revision=2)

        # Test 5: Status before any analysis exists
        status_missing = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/status")
        assert status_missing.status_code == 200
        s_data = status_missing.json()
        assert s_data["has_analysis"] is False
        assert s_data["status"] == "missing"
        assert s_data["is_stale"] is False
        assert s_data["latest_completed_run_id"] is None
        assert s_data["attempts"] == []
        assert s_data["repository"]["analytics_source_revision"] == 2
        assert s_data["repository"]["last_synced_at"] is not None

        # Test 1: Recompute enqueues new non-blocking job (HTTP 202) without synchronous execution
        with patch.object(
            AnalyticsOrchestrator,
            "run_snapshot_analysis",
            side_effect=AssertionError("POST /recompute must not execute analytics synchronously"),
        ):
            rec_new = client.post(f"/api/v1/analytics/{repo.owner}/{repo.name}/recompute")
        assert rec_new.status_code == 202
        r1 = rec_new.json()
        assert r1["enqueued"] is True
        assert r1["already_active"] is False
        assert r1["cached"] is False
        assert r1["run"]["status"] == "pending"
        assert r1["run"]["requested_source_revision"] == 2
        run_id_1 = r1["run"]["id"]

        # Test 2: Duplicate recompute while active pending returns HTTP 202 and reuses exact run ID
        rec_dup = client.post(f"/api/v1/analytics/{repo.owner}/{repo.name}/recompute")
        assert rec_dup.status_code == 202
        r2 = rec_dup.json()
        assert r2["enqueued"] is False
        assert r2["already_active"] is True
        assert r2["cached"] is False
        assert r2["run"]["id"] == run_id_1

        # Verify status shows pending before completion
        s_pending = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/status").json()
        assert s_pending["has_analysis"] is True
        assert s_pending["status"] == "pending"
        assert s_pending["is_stale"] is False
        assert s_pending["latest_completed_run_id"] is None

    def test_03_and_04_cached_completed_recompute_and_force_recomputation(self, v2_client):
        client, db = v2_client
        repo = _seed_unseen_rich_repo(db, repo_id=9502, source_revision=2)

        # Enqueue & process via AnalyticsWorker
        rec_init = client.post(f"/api/v1/analytics/{repo.owner}/{repo.name}/recompute")
        assert rec_init.status_code == 202
        run_id_1 = rec_init.json()["run"]["id"]

        worker = AnalyticsWorker(worker_id="w_c83_cache")
        exec_res = worker.process_one_job(db=db)
        assert exec_res is not None and exec_res.status == "completed"

        # Test 3: Calling recompute with force=False on current revision reuses cached completed run (HTTP 200)
        rec_cached = client.post(
            f"/api/v1/analytics/{repo.owner}/{repo.name}/recompute",
            json={"force": False, "requested_methodology_version": CURRENT_METHODOLOGY_VERSION},
        )
        assert rec_cached.status_code == 200
        c_data = rec_cached.json()
        assert c_data["enqueued"] is False
        assert c_data["already_active"] is False
        assert c_data["cached"] is True
        assert c_data["run"]["id"] == run_id_1
        assert c_data["run"]["status"] == "completed"

        # Test 4: Calling recompute with force=True creates a fresh run (HTTP 202)
        rec_force = client.post(
            f"/api/v1/analytics/{repo.owner}/{repo.name}/recompute",
            json={"force": True},
        )
        assert rec_force.status_code == 202
        f_data = rec_force.json()
        assert f_data["enqueued"] is True
        assert f_data["already_active"] is False
        assert f_data["cached"] is False
        assert f_data["run"]["id"] != run_id_1
        assert f_data["run"]["status"] == "pending"

    def test_06_and_15_status_after_completion_and_failure_with_attempt_telemetry(self, v2_client):
        client, db = v2_client
        repo = _seed_unseen_rich_repo(db, repo_id=9503, source_revision=2)

        # Fail with non-retryable error after one transient retry + interrupted telemetry check
        queue = AnalyticsJobQueue()
        run, _ = queue.enqueue_analysis(repo_id=repo.id, db=db, max_retries=3)

        class _TransientThenNonRetryable(BaseMetricCalculator):
            def __init__(self) -> None:
                self.calls = 0

            @property
            def domain_name(self) -> str:
                return "issue_intelligence"

            def calculate(self, repository_id, db, analysis_reference_time, snapshot_revision):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Transient timeout")
                raise InvalidParameterException("Non-retryable unsupported repo contract", parameter="repo")

        from backend.app.services.analytics import (
            ContributorIntelligenceCalculator,
            CrossDomainSynthesisCalculator,
            DevelopmentActivityCalculator,
            PullRequestIntelligenceCalculator,
            ReleaseEvolutionCalculator,
        )

        orch = AnalyticsOrchestrator(
            job_queue=queue,
            calculators={
                "development_activity": DevelopmentActivityCalculator(),
                "contributor_ecosystem": ContributorIntelligenceCalculator(),
                "pull_request_intelligence": PullRequestIntelligenceCalculator(),
                "issue_intelligence": _TransientThenNonRetryable(),
                "release_evolution": ReleaseEvolutionCalculator(),
                "repository_synthesis": CrossDomainSynthesisCalculator(),
            },
        )
        worker = AnalyticsWorker(worker_id="w_c83_fail", orchestrator=orch, job_queue=queue)
        r1 = worker.process_one_job(db=db)
        assert r1 is not None and r1.status == "retry_scheduled"
        r2 = worker.process_one_job(db=db)
        assert r2 is not None and r2.status == "failed"

        # Seed interrupted reason on attempt 1 to confirm interruption_reason exposure cleanly
        att1 = db.scalar(
            select(AnalysisRunAttempt).where(
                AnalysisRunAttempt.analysis_run_id == run.id,
                AnalysisRunAttempt.attempt_number == 1,
            )
        )
        att1.interruption_reason = "transient_retry_handoff"
        db.commit()

        # Test 6 & Test 15: /status returns retry_count, max_retries, failure_category, attempt error_category & interruption_reason
        st_resp = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/status")
        assert st_resp.status_code == 200
        st = st_resp.json()
        assert st["has_analysis"] is True
        assert st["status"] == "failed"
        assert st["retry_count"] == 1
        assert st["max_retries"] == 3
        assert st["failure_category"] == "non_retryable"
        assert st["latest_completed_run_id"] is None
        assert len(st["attempts"]) == 2
        assert st["attempts"][0]["error_category"] == "retryable"
        assert st["attempts"][0]["interruption_reason"] == "transient_retry_handoff"
        assert st["attempts"][1]["error_category"] == "non_retryable"


class TestAnalyticsV2LatestHistoryRunsAndDomains:
    """Covers tests 7–15: /latest, staleness, /history, /runs, all 6 /domains, case-insensitivity & errors."""

    def test_07_08_11_12_13_latest_stale_and_all_six_domains_including_embedded_release(self, v2_client):
        client, db = v2_client
        repo = _seed_unseen_rich_repo(
            db,
            repo_id=9504,
            owner="Dynamic-Org",
            name="Target-Repo",
            source_revision=2,
        )

        # /latest returns 404 before any completed analysis exists
        missing_latest = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/latest")
        assert missing_latest.status_code == 404
        assert missing_latest.json()["error_code"] == "RESOURCE_NOT_FOUND"

        queue = AnalyticsJobQueue()
        queue.enqueue_analysis(repo_id=repo.id, db=db)
        worker = AnalyticsWorker(worker_id="w_c83_latest", job_queue=queue)
        exec_res = worker.process_one_job(db=db)
        assert exec_res is not None and exec_res.status == "completed"

        # Test 7 & Test 13: Case-insensitive lookup on /latest
        latest_resp = client.get("/api/v1/analytics/dynamic-org/target-repo/latest")
        assert latest_resp.status_code == 200
        latest = latest_resp.json()
        assert latest["is_stale"] is False
        assert latest["active_run_id"] is None
        assert latest["repository"]["repo_name"] == "Dynamic-Org/Target-Repo"
        assert latest["repository"]["last_synced_at"] is not None
        assert latest["run"]["status"] == "completed"
        assert latest["run"]["analytics_source_revision"] == 2

        payload = latest["run"]["metrics_payload"]
        assert set(payload.keys()) == {
            "development_activity",
            "contributor_ecosystem",
            "pull_request_intelligence",
            "issue_intelligence",
            "repository_synthesis",
        }
        assert "last_synced_at" not in payload
        assert "release_evolution" not in payload
        assert "release_evolution" in payload["repository_synthesis"]

        # Test 11 & Test 12 & Test 13: All six domains + embedded release_evolution projection
        for domain_key in ALL_ANALYTICS_DOMAINS:
            d_resp = client.get(f"/api/v1/analytics/DYNAMIC-ORG/target-repo/domains/{domain_key}")
            assert d_resp.status_code == 200, f"Domain {domain_key} failed: {d_resp.text}"
            d_json = d_resp.json()
            assert d_json["domain"] == domain_key
            assert d_json["is_stale"] is False
            assert d_json["data"] is not None
            assert d_json["coverage"]["status"] in {"sufficient", "degraded", "insufficient", "empty"}
            assert d_json["confidence_breakdown"] is not None
            assert "score" in d_json["confidence_breakdown"]

        rel_slice = client.get("/api/v1/analytics/dynamic-org/target-repo/domains/release_evolution").json()
        assert rel_slice["data"] == payload["repository_synthesis"]["release_evolution"]
        assert "cadence" in rel_slice["data"]
        assert "semver_adoption" in rel_slice["data"]
        assert len(rel_slice["rules"]) >= 1

        # Test 8: Advance source revision -> /latest marks is_stale=True and exposes pending replacement run
        repo.analytics_source_revision = 3
        repo.last_synced_at = datetime.now(timezone.utc)
        db.commit()
        rep_run, is_new = queue.enqueue_analysis(repo_id=repo.id, db=db)
        assert is_new is True

        stale_latest = client.get("/api/v1/analytics/dynamic-org/target-repo/latest")
        assert stale_latest.status_code == 200
        s_lat = stale_latest.json()
        assert s_lat["is_stale"] is True
        assert s_lat["active_run_id"] == rep_run.id
        assert s_lat["active_run_status"] == "pending"
        assert s_lat["repository"]["analytics_source_revision"] == 3
        assert s_lat["run"]["analytics_source_revision"] == 2

    def test_09_and_10_history_pagination_filtering_and_scoped_plus_global_run_lookup(self, v2_client):
        client, db = v2_client
        repo_a = _seed_unseen_rich_repo(db, repo_id=9505, owner="hist-org", name="repo-a", source_revision=2)
        repo_b = _seed_unseen_rich_repo(db, repo_id=9506, owner="hist-org", name="repo-b", source_revision=2)

        queue = AnalyticsJobQueue()
        worker = AnalyticsWorker(worker_id="w_c83_hist", job_queue=queue)

        # Create 3 runs on repo_a: 2 completed, 1 failed
        r_c1, _ = queue.enqueue_analysis(repo_id=repo_a.id, db=db)
        worker.process_one_job(db=db)

        r_c2, _ = queue.enqueue_analysis(repo_id=repo_a.id, db=db, force=True)
        worker.process_one_job(db=db)

        failed_run = AnalysisRun(
            repository_id=repo_a.id,
            status="failed",
            requested_source_revision=2,
            requested_methodology_version=CURRENT_METHODOLOGY_VERSION,
            started_at=datetime.now(timezone.utc) + timedelta(seconds=5),
            completed_at=datetime.now(timezone.utc) + timedelta(seconds=6),
            error_message="[NON_RETRYABLE] postgresql://user:secret@dbhost/prod C:\\secret\\file.py\nTraceback (most recent call last):\n  File ...",
            lease_version=1,
            retry_count=0,
            max_retries=3,
        )
        db.add(failed_run)
        db.commit()

        # Run on repo_b must remain isolated from repo_a
        r_b, _ = queue.enqueue_analysis(repo_id=repo_b.id, db=db)
        worker.process_one_job(db=db)

        # Test 9: History pagination & deterministic ordering
        hist_all = client.get(f"/api/v1/analytics/{repo_a.owner}/{repo_a.name}/history?limit=2&offset=0")
        assert hist_all.status_code == 200
        ha = hist_all.json()
        assert ha["total"] == 3
        assert ha["limit"] == 2
        assert ha["offset"] == 0
        assert [item["id"] for item in ha["items"]] == [failed_run.id, r_c2.id]
        # Verify internal DSN/path/traceback redaction on failed run diagnostics!
        failed_msg = ha["items"][0]["error_message"]
        assert "secret" not in failed_msg
        assert "C:\\secret" not in failed_msg
        assert "Traceback" not in failed_msg
        assert "[REDACTED_URI]" in failed_msg

        hist_comp = client.get(
            f"/api/v1/analytics/{repo_a.owner}/{repo_a.name}/history?status=completed&limit=1&offset=1"
        )
        assert hist_comp.status_code == 200
        hc = hist_comp.json()
        assert hc["total"] == 2
        assert len(hc["items"]) == 1
        assert hc["items"][0]["id"] == r_c1.id

        # Test 10: Repository-scoped lookup allows own run, rejects other repo's run
        ok_scoped = client.get(f"/api/v1/analytics/{repo_a.owner}/{repo_a.name}/runs/{r_c1.id}")
        assert ok_scoped.status_code == 200
        assert ok_scoped.json()["run"]["id"] == r_c1.id

        cross_scoped = client.get(f"/api/v1/analytics/{repo_a.owner}/{repo_a.name}/runs/{r_b.id}")
        assert cross_scoped.status_code == 404
        assert cross_scoped.json()["error_code"] == "RESOURCE_NOT_FOUND"

        # Global lookup /runs/{run_id} works without colliding with /{owner}/{name}
        global_b = client.get(f"/api/v1/analytics/runs/{r_b.id}")
        assert global_b.status_code == 200
        assert global_b.json()["repository"]["repo_name"] == "hist-org/repo-b"
        assert global_b.json()["run"]["id"] == r_b.id

    def test_14_and_unavailable_domain_preservation_and_validation_contracts(self, v2_client):
        client, db = v2_client
        repo = _seed_unseen_rich_repo(db, repo_id=9507, owner="err-org", name="edge-repo", source_revision=2)

        # Missing repository -> 404 across endpoints
        assert client.get("/api/v1/analytics/nope/missing/status").status_code == 404
        assert client.get("/api/v1/analytics/nope/missing/latest").status_code == 404
        assert client.post("/api/v1/analytics/nope/missing/recompute").status_code == 404
        assert client.get("/api/v1/analytics/nope/missing/history").status_code == 404
        assert client.get("/api/v1/analytics/nope/missing/domains/development_activity").status_code == 404
        assert client.get("/api/v1/analytics/runs/9999999").status_code == 404

        # Invalid domain key -> 400
        bad_dom = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/domains/nonexistent_domain")
        assert bad_dom.status_code == 400
        assert bad_dom.json()["error_code"] == "INVALID_PARAMETER"

        # Invalid pagination & status filter
        assert client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/history?limit=0").status_code == 422
        assert client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/history?offset=-1").status_code == 422
        assert client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/history?status=bogus").status_code == 400

        # Unavailable domain preservation without fabricating zero metrics
        orch = AnalyticsOrchestrator(
            calculators={
                "development_activity": None,
                "contributor_ecosystem": None,
                "pull_request_intelligence": None,
                "issue_intelligence": None,
                "release_evolution": None,
                "repository_synthesis": None,
            }
        )
        res = orch.enqueue_and_process(repository_id=repo.id, db=db, worker_id="w_unavail_api")
        assert res.published is True

        rel_unavail = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/domains/release_evolution")
        assert rel_unavail.status_code == 200
        ru = rel_unavail.json()
        assert ru["coverage"]["status"] == "unavailable"
        assert ru["data"]["available"] is False
        assert ru["data"]["status"] == "unavailable"

    def test_16_legacy_analytics_routes_preserved_on_dynamically_seeded_repo(self, v2_client):
        client, db = v2_client
        repo = _seed_unseen_rich_repo(db, repo_id=9508, owner="legacy-org", name="compat-repo", source_revision=2)
        db.add(
            RepositoryAnalytics(
                repository_id=repo.id,
                repo_name=repo.repo_name,
                total_issues=6,
                open_issues=2,
                closed_issues=4,
                median_resolution_hours=24.0,
                mean_resolution_hours=24.0,
                p25_resolution_hours=18.0,
                p75_resolution_hours=30.0,
                survival_rate_30d=0.33,
                category_distribution={"bug": 3, "feature": 3},
                speed_tier_distribution={"1-7 Days": 4},
                snapshot_date=datetime.now(timezone.utc),
            )
        )
        db.commit()

        ov = client.get("/api/v1/analytics/overview")
        assert ov.status_code == 200
        assert ov.json()["total_repositories"] == 1
        assert ov.json()["total_issues"] == 6

        comp = client.get("/api/v1/analytics/comparison")
        assert comp.status_code == 200
        assert comp.json()["repositories"][0]["repo_name"] == repo.repo_name

        snap = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}")
        assert snap.status_code == 200
        assert snap.json()["resolution_metrics"]["median_hours"] == 24.0

        trends = client.get(f"/api/v1/analytics/{repo.owner}/{repo.name}/trends")
        assert trends.status_code == 200
        assert trends.json()["total_periods"] > 0


class TestFullIngestionToAnalyticsAPILifecycleAndConcurrency:
    """Covers tests 17–18 plus standalone/lifespan worker startup-shutdown verification."""

    def test_17_full_ingestion_submit_to_worker_to_analytics_worker_to_latest_api(self, v2_client):
        client, db = v2_client
        github_id = 960101
        repo_name = "e2e-org/universal-repo"

        meta_json = {
            "id": github_id,
            "name": "universal-repo",
            "full_name": repo_name,
            "owner": {"login": "e2e-org"},
            "html_url": f"https://github.com/{repo_name}",
            "private": False,
            "fork": False,
            "archived": False,
            "is_template": False,
            "size": 5000,
            "stargazers_count": 640,
            "forks_count": 90,
            "open_issues_count": 8,
            "default_branch": "main",
            "language": "Python",
            "description": "End-to-end dynamically ingested repo",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2026-04-20T00:00:00Z",
        }

        # 1. Submit repository URL via Phase B API
        with patch(
            "backend.app.services.repository_eligibility.RepositoryEligibilityChecker.fetch_github_metadata",
            new_callable=AsyncMock,
            return_value=meta_json,
        ):
            sub_resp = client.post(
                "/api/ingest/submit",
                json={"repo": f"https://github.com/{repo_name}", "mode": "fast_snapshot"},
            )
        assert sub_resp.status_code == 201
        job_id = sub_resp.json()["job_id"]

        # Check analytics /status immediately -> missing before ingestion completes
        st_pre = client.get(f"/api/v1/analytics/{repo_name}/status")
        assert st_pre.status_code == 200
        assert st_pre.json()["status"] == "missing"

        # 2. Run IngestionWorker with mocked AsyncGitHubClient executing real RepositoryCollector writes
        mock_client = MagicMock()
        mock_client.is_primary_rate_limited = False
        mock_client.resumes_at = None
        mock_client.close = AsyncMock()

        async def _req(method: str, endpoint_or_url: str, **kwargs):
            resp = MagicMock()
            if endpoint_or_url == f"/repos/{repo_name}":
                resp.json.return_value = meta_json
            else:
                resp.json.return_value = {}
            return resp

        async def _pag(endpoint_or_url: str = "", entity: str = "", **kwargs):
            if entity == "commits":
                return PaginatedResult(
                    entity="commits",
                    items=[
                        {
                            "sha": "e2e_sha_1",
                            "commit": {
                                "author": {"name": "Alice", "email": "alice@e2e.org", "date": "2026-04-01T10:00:00Z"},
                                "committer": {"date": "2026-04-01T10:00:00Z"},
                                "message": "feat: initial",
                            },
                            "author": {"id": 5001, "login": "alice"},
                        }
                    ],
                    total_requests=1,
                    is_partial=False,
                )
            if entity == "releases":
                return PaginatedResult(
                    entity="releases",
                    items=[
                        {
                            "id": 99001,
                            "tag_name": "v0.1.0",
                            "name": "Initial release",
                            "prerelease": False,
                            "published_at": "2026-04-05T10:00:00Z",
                        }
                    ],
                    total_requests=1,
                    is_partial=False,
                )
            return PaginatedResult(entity=entity, items=[], total_requests=1, is_partial=False)

        mock_client.request = AsyncMock(side_effect=_req)
        mock_client.paginate = AsyncMock(side_effect=_pag)

        ing_worker = IngestionWorker(
            worker_id="w_e2e_ing",
            heartbeat_interval_seconds=5.0,
        )
        processed_job = asyncio.run(
            ing_worker.process_one_job(
                db=db,
                client=mock_client,
                collector=RepositoryCollector(client=mock_client),
            )
        )
        assert processed_job == job_id

        # Analytics run should now be auto-enqueued (status='pending')
        st_pending = client.get(f"/api/v1/analytics/{repo_name}/status").json()
        assert st_pending["status"] == "pending"
        assert st_pending["repository"]["analytics_source_revision"] == 2

        # 3. Process queued run via standalone AnalyticsWorker
        ana_worker = AnalyticsWorker(worker_id="w_e2e_ana")
        exec_res = ana_worker.process_one_job(db=db)
        assert exec_res is not None and exec_res.status == "completed" and exec_res.published is True

        # 4. Retrieve via /latest and /domains/release_evolution
        lat_resp = client.get(f"/api/v1/analytics/{repo_name}/latest")
        assert lat_resp.status_code == 200
        lat = lat_resp.json()
        assert lat["is_stale"] is False
        assert lat["run"]["collection_run_id"] == job_id
        assert lat["run"]["analytics_source_revision"] == 2
        assert lat["run"]["metrics_payload"]["development_activity"]["summary"]["total_commits"] == 1
        assert "last_synced_at" in lat["repository"]

        rel_resp = client.get(f"/api/v1/analytics/{repo_name}/domains/release_evolution")
        assert rel_resp.status_code == 200
        rel_data = rel_resp.json()["data"]
        assert rel_data["cadence"]["total_releases"] == 1
        assert rel_data["cadence"]["latest_release_tag"] == "v0.1.0"
        assert rel_data["scope"]["mode"] == "fast_snapshot"

    def test_18_concurrent_recompute_requests_deduplicate_without_errors(self, tmp_path):
        db_file = tmp_path / "concurrent_v2_api.db"
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
        repo = _seed_unseen_rich_repo(seed_db, repo_id=9602, owner="conc-org", name="conc-repo", source_revision=2)
        repo_id = repo.id
        repo_owner = repo.owner
        repo_name = repo.name
        seed_db.close()

        test_app = create_application()

        def _override_get_db():
            s = SessionClass()
            try:
                yield s
            finally:
                s.close()

        test_app.dependency_overrides[get_db] = _override_get_db
        client = TestClient(test_app)

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(client.post, f"/api/v1/analytics/{repo_owner}/{repo_name}/recompute")
                f2 = pool.submit(client.post, f"/api/v1/analytics/{repo_owner}/{repo_name}/recompute")
                resp1 = f1.result()
                resp2 = f2.result()

            assert resp1.status_code == 202
            assert resp2.status_code == 202
            d1, d2 = resp1.json(), resp2.json()
            assert d1["run"]["id"] == d2["run"]["id"]
            assert {d1["enqueued"], d2["enqueued"]} == {True, False}

            verify_db = SessionClass()
            runs = verify_db.scalars(select(AnalysisRun).where(AnalysisRun.repository_id == repo_id)).all()
            assert len(runs) == 1
            verify_db.close()
        finally:
            test_app.dependency_overrides.clear()
            engine.dispose()

    def test_worker_startup_shutdown_and_session_cleanup(self, tmp_path):
        db_file = tmp_path / "worker_lifespan.db"
        engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=engine)
        RawSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        tracked_sessions: List[Dict[str, Any]] = []

        class _TrackedSessionProxy:
            def __init__(self, inner: Session):
                self._inner = inner
                self.closed = False
                tracked_sessions.append({"session": self, "closed": False})

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

            def close(self):
                self.closed = True
                for item in tracked_sessions:
                    if item["session"] is self:
                        item["closed"] = True
                self._inner.close()

        def _tracked_factory():
            return _TrackedSessionProxy(RawSession())

        seed_db = RawSession()
        repo = _seed_unseen_rich_repo(seed_db, repo_id=9603, owner="life-org", name="life-repo", source_revision=2)
        repo_id = repo.id
        seed_db.close()

        async def _run():
            stop_ev = asyncio.Event()

            async def _wait_and_stop():
                for _ in range(60):
                    await asyncio.sleep(0.02)
                    chk = RawSession()
                    done = chk.scalar(
                        select(AnalysisRun).where(
                            AnalysisRun.repository_id == repo_id,
                            AnalysisRun.status == "completed",
                        )
                    )
                    chk.close()
                    if done is not None:
                        stop_ev.set()
                        return
                stop_ev.set()

            await asyncio.gather(
                run_standalone_worker(
                    db_factory=_tracked_factory,
                    worker_id="w_standalone_life",
                    poll_interval=0.01,
                    reconcile_interval=30.0,
                    stop_event=stop_ev,
                    max_iterations=25,
                ),
                _wait_and_stop(),
            )

        asyncio.run(_run())

        assert len(tracked_sessions) >= 1
        assert all(entry["closed"] is True for entry in tracked_sessions)

        # Verify FastAPI lifespan also starts and cleanly stops AnalyticsWorker when opted in
        orig_enable = settings.ENABLE_ANALYTICS_WORKER
        orig_poll = settings.ANALYTICS_WORKER_POLL_INTERVAL_SECONDS
        settings.ENABLE_ANALYTICS_WORKER = True
        settings.ANALYTICS_WORKER_POLL_INTERVAL_SECONDS = 0.05
        try:
            with patch("backend.app.main.check_connection", return_value={"status": "connected", "dialect": "sqlite", "latency_ms": 0.1}), \
                 patch("backend.app.main.init_db"), \
                 patch("backend.app.main.SessionLocal", side_effect=_tracked_factory), \
                 patch.object(AnalyticsWorker, "reconcile_jobs", return_value={"expired_leases_recovered": 0, "missing_jobs_enqueued": 0}), \
                 patch.object(AnalyticsWorker, "process_one_job", return_value=None):
                life_app = create_application()
                with TestClient(life_app) as tc:
                    assert tc.get("/api/v1/ping").status_code == 200
        finally:
            settings.ENABLE_ANALYTICS_WORKER = orig_enable
            settings.ANALYTICS_WORKER_POLL_INTERVAL_SECONDS = orig_poll
            engine.dispose()
