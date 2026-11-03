"""Integration tests for repository and issue browsing API endpoints."""

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class TestAPIRepositories(unittest.TestCase):
    """Test suite for /api/v1/repos endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient."""
        cls.client = TestClient(app)

    def test_list_repositories(self):
        """Verify GET /api/v1/repos returns all 4 tracked repositories ordered by stars."""
        response = self.client.get("/api/v1/repos")
        self.assertEqual(response.status_code, 200)

        repos = response.json()
        self.assertIsInstance(repos, list)
        self.assertEqual(len(repos), 4)

        repo_names = [r["repo_name"] for r in repos]
        self.assertIn("microsoft/vscode", repo_names)
        self.assertIn("pytorch/pytorch", repo_names)
        self.assertIn("kubernetes/kubernetes", repo_names)
        self.assertIn("huggingface/transformers", repo_names)

        # Check descending order by stars
        stars = [r["stars_count"] for r in repos]
        self.assertEqual(stars, sorted(stars, reverse=True))

    def test_get_repository_detail(self):
        """Verify GET /api/v1/repos/{owner}/{name} returns detailed stats and analytics."""
        response = self.client.get("/api/v1/repos/microsoft/vscode")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["repo_name"], "microsoft/vscode")
        self.assertEqual(data["owner"], "microsoft")
        self.assertEqual(data["name"], "vscode")
        self.assertEqual(data["primary_language"], "TypeScript")
        self.assertEqual(data["total_analyzed_issues"], 536)
        self.assertEqual(data["closed_issues_count"], 242)
        self.assertIsNotNone(data["median_resolution_hours"])
        self.assertAlmostEqual(data["median_resolution_hours"], 7.71, delta=0.5)
        self.assertIsInstance(data["category_distribution"], dict)
        self.assertIn("bug", data["category_distribution"])
        self.assertIsInstance(data["speed_tier_distribution"], dict)

    def test_get_repository_not_found(self):
        """Verify 404 response for nonexistent repository."""
        response = self.client.get("/api/v1/repos/nonexistent_org/nonexistent_repo")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_list_repository_issues_pagination(self):
        """Verify paginated issue retrieval for a repository."""
        limit = 15
        offset = 5
        response = self.client.get(
            "/api/v1/repos/microsoft/vscode/issues",
            params={"limit": limit, "offset": offset},
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total"], 536)
        self.assertEqual(data["limit"], limit)
        self.assertEqual(data["offset"], offset)
        self.assertEqual(len(data["items"]), limit)

    def test_list_issues_filters(self):
        """Verify state, category, speed_tier, and search query filters."""
        # 1. State filter: closed
        res_state = self.client.get(
            "/api/v1/repos/microsoft/vscode/issues",
            params={"state": "closed", "limit": 20},
        )
        self.assertEqual(res_state.status_code, 200)
        items_state = res_state.json()["items"]
        for it in items_state:
            self.assertEqual(it["state"], "closed")

        # 2. Category filter: bug
        res_cat = self.client.get(
            "/api/v1/repos/microsoft/vscode/issues",
            params={"category": "bug", "limit": 20},
        )
        self.assertEqual(res_cat.status_code, 200)
        items_cat = res_cat.json()["items"]
        for it in items_cat:
            self.assertEqual(it["canonical_label"], "bug")

        # 3. Speed tier filter: < 1 Day
        res_speed = self.client.get(
            "/api/v1/repos/microsoft/vscode/issues",
            params={"speed_tier": "< 1 Day", "limit": 20},
        )
        self.assertEqual(res_speed.status_code, 200)
        items_speed = res_speed.json()["items"]
        for it in items_speed:
            self.assertEqual(it["speed_tier"], "< 1 Day")

    def test_get_issue_detail_success(self):
        """Verify retrieval of detailed single issue with labels."""
        # First retrieve an issue number from the repository list
        list_res = self.client.get("/api/v1/repos/microsoft/vscode/issues", params={"limit": 1})
        self.assertEqual(list_res.status_code, 200)
        first_issue = list_res.json()["items"][0]
        issue_number = first_issue["issue_number"]

        # Fetch detail
        detail_res = self.client.get(f"/api/v1/repos/microsoft/vscode/issues/{issue_number}")
        self.assertEqual(detail_res.status_code, 200)

        detail = detail_res.json()
        self.assertEqual(detail["repo_name"], "microsoft/vscode")
        self.assertEqual(detail["issue_number"], issue_number)
        self.assertEqual(detail["title"], first_issue["title"])
        self.assertIn("body", detail)
        self.assertIn("clean_title", detail)
        self.assertIn("clean_body", detail)
        self.assertIn("labels", detail)
        self.assertIsInstance(detail["labels"], list)

    def test_get_issue_detail_not_found(self):
        """Verify 404 response for nonexistent issue number."""
        response = self.client.get("/api/v1/repos/microsoft/vscode/issues/999999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
