"""Integration Test Suite for Database Pipeline, ORM Queries, and Frontend Endpoints (Phase 11, Step 11.2).

Verifies the complete data pipeline from the SQLite / PostgreSQL database through SQLAlchemy 2.0
ORM queries and FastAPI endpoints into the data structures expected by the Next.js frontend:
1. Database engine readiness, connection pooling, and session lifecycle.
2. Repository queries, star ordering, and analytical snapshots.
3. Multi-dimensional issue querying (pagination, state, canonical label, velocity speed tier, text search).
4. Analytical aggregation verification (closure rates, SLA percentiles, trend time series).
5. Error handling and 404 boundaries for non-existent entities.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.init_db import check_connection
from backend.app.database.models import Issue, Repository, RepositoryAnalytics
from backend.app.database.session import SessionLocal, engine, get_db
from backend.app.main import app


class TestDatabasePipelineIntegration(unittest.TestCase):
    """Integration test suite verifying database querying and frontend-facing data contracts."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.target_repos = [
            "microsoft/vscode",
            "pytorch/pytorch",
            "kubernetes/kubernetes",
            "huggingface/transformers",
        ]

    def setUp(self):
        self.db: Session = SessionLocal()

    def tearDown(self):
        self.db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Database Connection, Engine & Session Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def test_database_engine_connection_and_dialect(self):
        """Verify engine connection ping reports connected status and valid dialect."""
        status = check_connection(engine)
        self.assertEqual(status["status"], "connected")
        self.assertIn(status["dialect"], ["sqlite", "postgresql"])
        self.assertGreater(status["latency_ms"], 0.0)
        self.assertIsNone(status["error"])

    def test_database_session_dependency_injection(self):
        """Verify get_db generator yields an active session and closes it cleanly."""
        session_gen = get_db()
        session = next(session_gen)
        try:
            self.assertIsInstance(session, Session)
            # Simple scalar query to ensure connection is live
            result = session.scalar(select(func.count(Repository.id)))
            self.assertGreaterEqual(result, 4)
        finally:
            try:
                next(session_gen)
            except StopIteration:
                pass  # Clean generator completion

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Repository Queries & Metadata Integrity
    # ──────────────────────────────────────────────────────────────────────────

    def test_all_target_repositories_present_in_database(self):
        """Verify all 4 target repositories are stored in the database with complete metadata."""
        repos = self.db.scalars(
            select(Repository).order_by(Repository.stars_count.desc())
        ).all()
        repo_names = [r.repo_name for r in repos]

        for expected in self.target_repos:
            self.assertIn(expected, repo_names)

        # Verify VS Code is among top starred repositories
        self.assertGreater(repos[0].stars_count, 50000)

        for r in repos:
            self.assertIsNotNone(r.id)
            self.assertTrue(len(r.owner) > 0)
            self.assertTrue(len(r.name) > 0)
            self.assertGreater(r.stars_count, 0)
            self.assertGreater(r.forks_count, 0)
            self.assertIsNotNone(r.collected_at)

    def test_endpoint_list_repositories_returns_db_records(self):
        """Verify GET /api/v1/repos returns formatted repository summaries from the database."""
        response = self.client.get("/api/v1/repos")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 4)

        names = [item["repo_name"] for item in data]
        for target in self.target_repos:
            self.assertIn(target, names)

    def test_endpoint_single_repository_detail_with_analytics_join(self):
        """Verify GET /api/v1/repos/{owner}/{name} joins repository with analytical aggregates."""
        for repo_slug in self.target_repos:
            owner, name = repo_slug.split("/")
            response = self.client.get(f"/api/v1/repos/{owner}/{name}")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["repo_name"], repo_slug)
            self.assertEqual(data["owner"], owner)
            self.assertEqual(data["name"], name)
            self.assertGreater(data["total_analyzed_issues"], 0)
            self.assertGreater(data["closed_issues_count"], 0)
            self.assertIsNotNone(data["median_resolution_hours"])
            self.assertIsInstance(data["category_distribution"], dict)
            self.assertIsInstance(data["speed_tier_distribution"], dict)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Issue Queries, Multi-Condition Filtering & Pagination
    # ──────────────────────────────────────────────────────────────────────────

    def test_issue_pagination_limits_and_offsets(self):
        """Verify pagination query parameters slice results without data duplication."""
        owner, name = "microsoft", "vscode"

        # Fetch page 1 (items 0-9)
        p1_res = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?limit=10&offset=0")
        self.assertEqual(p1_res.status_code, 200)
        p1 = p1_res.json()
        self.assertEqual(len(p1["items"]), 10)
        p1_ids = [item["id"] for item in p1["items"]]

        # Fetch page 2 (items 10-19)
        p2_res = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?limit=10&offset=10")
        self.assertEqual(p2_res.status_code, 200)
        p2 = p2_res.json()
        self.assertEqual(len(p2["items"]), 10)
        p2_ids = [item["id"] for item in p2["items"]]

        # Sets should not overlap
        self.assertEqual(len(set(p1_ids).intersection(set(p2_ids))), 0)
        self.assertEqual(p1["total"], p2["total"])
        self.assertGreaterEqual(p1["total"], 500)

    def test_issue_state_filter(self):
        """Verify filtering issues by open and closed states against database records."""
        owner, name = "microsoft", "vscode"

        # Filter open
        open_res = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?state=open&limit=20")
        self.assertEqual(open_res.status_code, 200)
        open_data = open_res.json()
        for item in open_data["items"]:
            self.assertEqual(item["state"], "open")

        # Filter closed
        closed_res = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?state=closed&limit=20")
        self.assertEqual(closed_res.status_code, 200)
        closed_data = closed_res.json()
        for item in closed_data["items"]:
            self.assertEqual(item["state"], "closed")

        # Sum of open + closed should equal total
        self.assertEqual(open_data["total"] + closed_data["total"], 536)

    def test_issue_canonical_label_filter(self):
        """Verify filtering by canonical category returns only matching issues."""
        owner, name = "pytorch", "pytorch"
        response = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?category=bug&limit=15")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["total"], 0)
        for item in data["items"]:
            self.assertEqual(item["canonical_label"], "bug")

    def test_issue_speed_tier_filter(self):
        """Verify filtering by resolution speed tier returns only matching issues."""
        owner, name = "kubernetes", "kubernetes"
        response = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?speed_tier=%3C%201%20Day&limit=15")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["total"], 0)
        for item in data["items"]:
            self.assertEqual(item["speed_tier"], "< 1 Day")

    def test_issue_text_search_filter(self):
        """Verify title substring search returns matching issues and handles non-matching queries."""
        owner, name = "microsoft", "vscode"

        # Search matching term
        res = self.client.get(f"/api/v1/repos/{owner}/{name}/issues?search=window&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["total"], 0)
        for item in data["items"]:
            self.assertIn("window", (item["title"] + " " + (item.get("clean_title") or "")).lower())

        # Search non-matching term should cleanly return empty list
        empty_res = self.client.get(
            f"/api/v1/repos/{owner}/{name}/issues?search=nonexistent_xyz_phrase_12345"
        )
        self.assertEqual(empty_res.status_code, 200)
        empty_data = empty_res.json()
        self.assertEqual(empty_data["total"], 0)
        self.assertEqual(len(empty_data["items"]), 0)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Analytics Aggregation & Cross-Repository Verification
    # ──────────────────────────────────────────────────────────────────────────

    def test_global_analytics_overview_db_aggregation(self):
        """Verify /api/v1/analytics/overview accurately aggregates counts across all database issues."""
        response = self.client.get("/api/v1/analytics/overview")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Database contains exactly 2,328 standardized issues
        self.assertEqual(data["total_issues"], 2328)
        self.assertEqual(data["total_repositories"], 4)
        self.assertEqual(
            data["total_open_issues"] + data["total_closed_issues"],
            data["total_issues"],
        )
        self.assertAlmostEqual(
            data["global_closure_rate"],
            data["total_closed_issues"] / data["total_issues"],
            places=3,
        )

        cat_sum = sum(data["category_distribution"].values())
        self.assertEqual(cat_sum, data["total_issues"])

    def test_comparative_matrix_consistency(self):
        """Verify /api/v1/analytics/comparison matches per-repository issue counts."""
        response = self.client.get("/api/v1/analytics/comparison")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_repositories"], 4)
        total_matrix_issues = sum(r["total_issues"] for r in data["repositories"])
        self.assertEqual(total_matrix_issues, 2328)

        for r in data["repositories"]:
            self.assertGreater(r["total_issues"], 500)
            self.assertGreater(r["closed_issues"], 200)
            self.assertAlmostEqual(
                r["closure_rate"],
                r["closed_issues"] / r["total_issues"],
                places=3,
            )

    def test_repository_monthly_trends_grouping(self):
        """Verify /api/v1/analytics/{owner}/{name}/trends groups issues by year-month."""
        response = self.client.get("/api/v1/analytics/huggingface/transformers/trends")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["repo_name"], "huggingface/transformers")
        self.assertGreaterEqual(data["total_periods"], 1)
        self.assertIsInstance(data["trends"], list)
        if data["trends"]:
            first_point = data["trends"][0]
            self.assertIn("period", first_point)
            self.assertIn("created_count", first_point)
            self.assertIn("closed_count", first_point)
            self.assertGreater(first_point["created_count"], 0)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Error Boundaries for Non-Existent Entities
    # ──────────────────────────────────────────────────────────────────────────

    def test_nonexistent_repository_returns_404(self):
        """Verify querying non-existent repository returns 404 with structured error."""
        response = self.client.get("/api/v1/repos/fake_owner_123/nonexistent_repo")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)

    def test_nonexistent_issue_returns_404(self):
        """Verify querying non-existent issue number returns 404."""
        response = self.client.get("/api/v1/repos/microsoft/vscode/issues/999999999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)


if __name__ == "__main__":
    unittest.main()
