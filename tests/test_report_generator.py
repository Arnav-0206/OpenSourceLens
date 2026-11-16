"""Unit and integration tests for executive report generator and export endpoint."""

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from backend.app.database.models import (
    AnalysisRun,
    Repository,
)
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services.analytics.report_generator import report_generator


class TestReportGenerator(unittest.TestCase):
    """Test suite for ReportGeneratorService and GET /{owner}/{name}/report."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.session = SessionLocal()

        cls.owner = "report-org"
        cls.name = "audit-repo"
        cls.repo_name = f"{cls.owner}/{cls.name}"

        existing = cls.session.query(Repository).filter(Repository.repo_name == cls.repo_name).first()
        if existing:
            cls.session.delete(existing)
            cls.session.commit()

        cls.repo = Repository(
            id=555666,
            repo_name=cls.repo_name,
            owner=cls.owner,
            name=cls.name,
            stars_count=1200,
            forks_count=350,
            primary_language="TypeScript",
            analytics_source_revision=1,
        )
        cls.session.add(cls.repo)
        cls.session.commit()

        # Add a completed AnalysisRun
        cls.analysis_run = AnalysisRun(
            id=8881,
            repository_id=cls.repo.id,
            status="completed",
            requested_source_revision=1,
            analytics_source_revision=1,
            methodology_version="2026.2",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            confidence_summary={
                "score": 0.95,
                "band": "high",
                "penalties": [],
            },
            rule_interpretations=[
                {
                    "rule_id": "RULE-PR-01",
                    "severity": "warning",
                    "domain": "pull_requests",
                    "message": "PR review turnaround median exceeds 48 hours.",
                }
            ],
            metrics_payload={
                "development_activity": {
                    "summary": {
                        "total_commits": 450,
                        "velocity_trend": "increasing",
                    }
                },
                "contributor_ecosystem": {
                    "total_contributors": 35,
                    "gini_coefficient": 0.42,
                    "bus_factors": {"bf50": 3, "bf80": 8},
                },
                "pull_request_intelligence": {
                    "total_prs": 85,
                    "outcomes": {"merge_rate": 0.78},
                    "turnaround_hours": {"p50": 18.5},
                },
                "issue_intelligence": {
                    "total_issues": 120,
                    "resolution_sla_hours": {"p50": 36.0},
                    "aging": {"open_stale_count": 5},
                },
                "repository_synthesis": {
                    "release_evolution": {
                        "total_releases": 12,
                        "semver_adoption_rate": 1.0,
                    },
                    "discussions_intelligence": {
                        "status": "active",
                        "total_discussions": 24,
                        "qna": {"qna_answered_rate": 0.85},
                    },
                    "cross_domain_correlations": [
                        {
                            "pair_id": "PAIR-01",
                            "label_x": "Code Churn",
                            "label_y": "Issue Inflow",
                            "spearman_rho": 0.52,
                            "permutation_p_value": 0.03,
                            "q_value": 0.04,
                            "max_abs_ar1": 0.15,
                            "significant_fdr_05": True,
                        }
                    ],
                },
            },
        )
        cls.session.add(cls.analysis_run)
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        repo = cls.session.query(Repository).filter(Repository.id == 555666).first()
        if repo:
            cls.session.delete(repo)
            cls.session.commit()
        cls.session.close()

    def test_generate_markdown_content(self):
        """Verify markdown report generator compiles complete markdown document."""
        md = report_generator.generate_markdown(
            repo=self.repo,
            run=self.analysis_run,
            payload=self.analysis_run.metrics_payload,
            topics=[{"cluster_id": 0, "title": "Theme A", "keywords": ["auth", "token"], "total_count": 5, "open_count": 3, "closed_count": 2}],
        )
        self.assertIn("# OpenSourceLens — Repository Intelligence Audit", md)
        self.assertIn("report-org/audit-repo", md)
        self.assertIn("1. Multi-Domain Diagnostic Scorecard", md)
        self.assertIn("2. Data Confidence & Penalty Audit", md)
        self.assertIn("3. Active Rule Interpretations", md)
        self.assertIn("4. Cross-Domain Correlation Explorer", md)
        self.assertIn("Scientific Disclaimer", md)
        self.assertIn("PAIR-01", md)
        self.assertIn("5. Discovered Backlog Themes", md)

    def test_generate_json_content(self):
        """Verify JSON report generator compiles structured dictionary."""
        js = report_generator.generate_json(
            repo=self.repo,
            run=self.analysis_run,
            payload=self.analysis_run.metrics_payload,
        )
        self.assertEqual(js["metadata"]["repository"], "report-org/audit-repo")
        self.assertIn("confidence_summary", js)
        self.assertIn("analytics_payload", js)

    def test_api_export_markdown_report(self):
        """Verify GET /api/v1/analytics/{owner}/{name}/report?format=markdown."""
        resp = self.client.get(
            f"/api/v1/analytics/{self.owner}/{self.name}/report?format=markdown"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/markdown", resp.headers["content-type"])
        self.assertIn("attachment; filename=", resp.headers["content-disposition"])
        self.assertIn("# OpenSourceLens", resp.text)

    def test_api_export_json_report(self):
        """Verify GET /api/v1/analytics/{owner}/{name}/report?format=json."""
        resp = self.client.get(
            f"/api/v1/analytics/{self.owner}/{self.name}/report?format=json"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("application/json", resp.headers["content-type"])
        data = resp.json()
        self.assertEqual(data["metadata"]["repository"], "report-org/audit-repo")


if __name__ == "__main__":
    unittest.main()
