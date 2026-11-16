"""Production hardening and system resilience tests for Stage 6.

Validates:
1. Comprehensive health telemetry (/health and /api/v1/health) reporting DB latency, queue depths, workers, models.
2. Worker lease claiming, atomic version increments, and stale lease recovery.
3. Executive intelligence report generation and download endpoint in Markdown and JSON formats.
4. Multi-container Docker and Docker Compose configuration integrity.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

from fastapi.testclient import TestClient
import yaml

from backend.app.database.models import AnalysisRun, Repository
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services.analytics_queue import AnalyticsJobQueue


class TestProductionHardening(unittest.TestCase):
    """Test suite for production health telemetry, queue resilience, and deployment configurations."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.session = SessionLocal()
        cls.queue = AnalyticsJobQueue()

        cls.owner = "hardening-org"
        cls.name = "resilience-repo"
        cls.repo_name = f"{cls.owner}/{cls.name}"

        # Clean up any preexisting repository
        existing = cls.session.query(Repository).filter(Repository.repo_name == cls.repo_name).first()
        if existing:
            cls.session.delete(existing)
            cls.session.commit()

        cls.repo = Repository(
            id=777888,
            repo_name=cls.repo_name,
            owner=cls.owner,
            name=cls.name,
            stars_count=2500,
            forks_count=600,
            primary_language="Python",
            analytics_source_revision=1,
        )
        cls.session.add(cls.repo)
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        repo = cls.session.query(Repository).filter(Repository.id == 777888).first()
        if repo:
            cls.session.delete(repo)
            cls.session.commit()
        cls.session.close()

    def test_health_check_telemetry(self):
        """Verify /health and /api/v1/health report complete operational diagnostics."""
        for path in ["/health", "/api/v1/health"]:
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, f"Endpoint {path} failed")
            data = resp.json()

            self.assertIn(data["status"], ["healthy", "degraded"])
            self.assertIn("version", data)
            self.assertIn("database", data)
            self.assertIn("dialect", data["database"])
            self.assertIn("latency_ms", data["database"])

            # Queue depths
            self.assertIn("queue_depths", data)
            self.assertIn("ingestion", data["queue_depths"])
            self.assertIn("analytics", data["queue_depths"])

            # Active workers
            self.assertIn("active_workers", data)
            self.assertIn("active_workers_count", data["active_workers"])

            # Model status
            self.assertIn("model_status", data)
            self.assertIn("classifier_generalizable", data["model_status"])

    def test_queue_atomic_lease_claiming_and_concurrency(self):
        """Verify atomic lease claiming ensures exactly one worker claims a job."""
        # Enqueue a new pending analysis run
        run_obj, created = self.queue.enqueue_analysis(
            repo_id=self.repo.id,
            db=self.session,
            force=True,
        )
        self.assertIsNotNone(run_obj)
        run_id = run_obj.id

        # Worker 1 claims the pending run
        claim1 = self.queue.claim_job(worker_id="worker-alpha", db=self.session)
        self.assertIsNotNone(claim1)
        run1, attempt1 = claim1
        self.assertEqual(run1.id, run_id)
        self.assertEqual(run1.status, "running")
        self.assertEqual(run1.worker_id, "worker-alpha")
        self.assertEqual(run1.lease_version, 1)

        # Worker 2 attempts to claim while Worker 1 holds active lease
        claim2 = self.queue.claim_job(worker_id="worker-beta", db=self.session)
        # Should be None since the only pending job is leased to worker-alpha
        if claim2 is not None:
            # If another pending job existed in DB from earlier tests, verify it is not run_id
            self.assertNotEqual(claim2[0].id, run_id)

        # Clean up this run
        self.session.delete(run1)
        self.session.commit()

    def test_stale_lease_recovery(self):
        """Verify an expired worker lease is reclaimed and lease_version incremented."""
        # Create an abandoned run with expired lease
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        abandoned_run = AnalysisRun(
            repository_id=self.repo.id,
            status="running",
            requested_source_revision=1,
            analytics_source_revision=1,
            methodology_version="2026.2",
            worker_id="crashed-worker-99",
            leased_until=expired_time,
            lease_version=1,
            retry_count=0,
            max_retries=3,
        )
        self.session.add(abandoned_run)
        self.session.commit()
        run_id = abandoned_run.id

        # Reclaim by a new healthy worker
        claim = self.queue.claim_job(worker_id="recovering-worker-1", db=self.session)
        self.assertIsNotNone(claim)
        reclaimed_run, attempt = claim

        # Verify new worker ownership and incremented lease version
        self.assertEqual(reclaimed_run.id, run_id)
        self.assertEqual(reclaimed_run.worker_id, "recovering-worker-1")
        self.assertEqual(reclaimed_run.lease_version, 2)
        self.assertEqual(reclaimed_run.status, "running")

        # Clean up
        self.session.delete(reclaimed_run)
        self.session.commit()

    def test_executive_report_export_api(self):
        """Verify GET /api/v1/analytics/{owner}/{name}/report supports Markdown and JSON."""
        # Add completed analysis run
        completed_run = AnalysisRun(
            repository_id=self.repo.id,
            status="completed",
            requested_source_revision=1,
            analytics_source_revision=1,
            methodology_version="2026.2",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            confidence_summary={"score": 0.90, "band": "high", "penalties": []},
            rule_interpretations=[],
            metrics_payload={
                "development_activity": {"summary": {"total_commits": 100}},
                "contributor_ecosystem": {"total_contributors": 10},
                "pull_request_intelligence": {"total_prs": 20},
                "issue_intelligence": {"total_issues": 30},
                "repository_synthesis": {
                    "release_evolution": {"total_releases": 5},
                    "cross_domain_correlations": [],
                },
            },
        )
        self.session.add(completed_run)
        self.session.commit()

        # Test Markdown download
        md_resp = self.client.get(f"/api/v1/analytics/{self.owner}/{self.name}/report?format=markdown")
        self.assertEqual(md_resp.status_code, 200)
        self.assertIn("text/markdown", md_resp.headers["content-type"])
        self.assertIn("attachment; filename=", md_resp.headers["content-disposition"])
        self.assertIn("# OpenSourceLens — Repository Intelligence Audit", md_resp.text)

        # Test JSON download
        json_resp = self.client.get(f"/api/v1/analytics/{self.owner}/{self.name}/report?format=json")
        self.assertEqual(json_resp.status_code, 200)
        self.assertIn("application/json", json_resp.headers["content-type"])
        json_data = json_resp.json()
        self.assertEqual(json_data["metadata"]["repository"], self.repo_name)

        # Test invalid format returns 422
        inv_resp = self.client.get(f"/api/v1/analytics/{self.owner}/{self.name}/report?format=xml")
        self.assertEqual(inv_resp.status_code, 422)

        # Clean up
        self.session.delete(completed_run)
        self.session.commit()

    def test_docker_and_compose_configurations(self):
        """Verify presence and validity of production deployment Docker configurations."""
        root = Path(__file__).resolve().parent.parent

        dockerfile_backend = root / "Dockerfile.backend"
        dockerfile_worker = root / "Dockerfile.worker"
        dockerfile_frontend = root / "Dockerfile.frontend"
        docker_compose = root / "docker-compose.yml"
        dockerignore = root / ".dockerignore"

        self.assertTrue(dockerfile_backend.exists(), "Dockerfile.backend missing")
        self.assertTrue(dockerfile_worker.exists(), "Dockerfile.worker missing")
        self.assertTrue(dockerfile_frontend.exists(), "Dockerfile.frontend missing")
        self.assertTrue(docker_compose.exists(), "docker-compose.yml missing")
        self.assertTrue(dockerignore.exists(), ".dockerignore missing")

        # Verify backend Dockerfile directives
        backend_content = dockerfile_backend.read_text(encoding="utf-8")
        self.assertIn("FROM python:", backend_content)
        self.assertIn("EXPOSE 8000", backend_content)
        self.assertIn("HEALTHCHECK", backend_content)

        # Verify worker Dockerfile directives
        worker_content = dockerfile_worker.read_text(encoding="utf-8")
        self.assertIn("analytics_worker", worker_content)

        # Verify frontend Dockerfile directives
        frontend_content = dockerfile_frontend.read_text(encoding="utf-8")
        self.assertIn("FROM node:20-alpine", frontend_content)
        self.assertIn("EXPOSE 3000", frontend_content)

        # Verify compose YAML parsing
        compose_content = docker_compose.read_text(encoding="utf-8")
        parsed = yaml.safe_load(compose_content)
        self.assertIn("services", parsed)
        self.assertIn("db", parsed["services"])
        self.assertIn("backend", parsed["services"])
        self.assertIn("analytics_worker", parsed["services"])
        self.assertIn("frontend", parsed["services"])


if __name__ == "__main__":
    unittest.main()
