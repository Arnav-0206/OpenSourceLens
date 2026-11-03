"""Integration tests for analytics and SLA metrics API endpoints."""

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class TestAPIAnalytics(unittest.TestCase):
    """Test suite for /api/v1/analytics endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient."""
        cls.client = TestClient(app)

    def test_get_global_overview(self):
        """Verify GET /api/v1/analytics/overview returns cross-repo aggregates."""
        response = self.client.get("/api/v1/analytics/overview")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_repositories"], 4)
        self.assertEqual(data["total_issues"], 2328)
        self.assertEqual(data["total_open_issues"], 1193)
        self.assertEqual(data["total_closed_issues"], 1135)
        self.assertAlmostEqual(data["global_closure_rate"], 0.4875, delta=0.01)

        self.assertEqual(data["fastest_repository"], "microsoft/vscode")
        self.assertEqual(data["slowest_repository"], "kubernetes/kubernetes")

        self.assertIn("bug", data["category_distribution"])
        self.assertIn("< 1 Day", data["speed_tier_distribution"])

    def test_get_repository_comparison(self):
        """Verify GET /api/v1/analytics/comparison returns comparative benchmark matrix."""
        response = self.client.get("/api/v1/analytics/comparison")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_repositories"], 4)
        repos = data["repositories"]
        self.assertEqual(len(repos), 4)

        repo_names = [r["repo_name"] for r in repos]
        self.assertIn("microsoft/vscode", repo_names)
        self.assertIn("pytorch/pytorch", repo_names)

        # Verify VS Code rapid closure percentage is highest
        vscode = next(r for r in repos if r["repo_name"] == "microsoft/vscode")
        self.assertGreater(vscode["rapid_closure_percentage"], 30.0)
        self.assertLess(vscode["median_resolution_hours"], 10.0)

    def test_get_single_repository_analytics(self):
        """Verify GET /api/v1/analytics/{owner}/{name} returns percentile distributions."""
        response = self.client.get("/api/v1/analytics/microsoft/vscode")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["repo_name"], "microsoft/vscode")
        self.assertEqual(data["total_issues"], 536)
        self.assertEqual(data["closed_issues"], 242)

        metrics = data["resolution_metrics"]
        self.assertIsNotNone(metrics["median_hours"])
        self.assertIsNotNone(metrics["p25_hours"])
        self.assertIsNotNone(metrics["p75_hours"])

        # Check percentile consistency: p25 <= median <= p75
        self.assertLessEqual(metrics["p25_hours"], metrics["median_hours"])
        self.assertLessEqual(metrics["median_hours"], metrics["p75_hours"])

        self.assertGreater(metrics["survival_rate_30d"], 0.0)
        self.assertLess(metrics["survival_rate_30d"], 1.0)

    def test_get_repository_analytics_not_found(self):
        """Verify 404 response for nonexistent repository."""
        response = self.client.get("/api/v1/analytics/nonexistent_org/nonexistent_repo")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_get_repository_trends(self):
        """Verify GET /api/v1/analytics/{owner}/{name}/trends returns time-series points."""
        response = self.client.get("/api/v1/analytics/microsoft/vscode/trends")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["repo_name"], "microsoft/vscode")
        self.assertGreater(data["total_periods"], 0)

        trends = data["trends"]
        self.assertGreater(len(trends), 0)

        first_pt = trends[0]
        self.assertIn("period", first_pt)
        self.assertEqual(len(first_pt["period"]), 7)  # "YYYY-MM"
        self.assertGreaterEqual(first_pt["created_count"], 0)
        self.assertGreaterEqual(first_pt["closed_count"], 0)

    def test_get_repository_trends_not_found(self):
        """Verify 404 for trends on nonexistent repository."""
        response = self.client.get("/api/v1/analytics/nonexistent_org/nonexistent_repo/trends")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
