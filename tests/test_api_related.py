"""Integration tests for semantic related-issue retrieval API endpoints."""

import unittest

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.models import Issue, RelatedIssuesCache
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestAPIRelated(unittest.TestCase):
    """Test suite for /api/v1/related endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient and locate a sample issue with cached relationships."""
        cls.client = TestClient(app)
        session = SessionLocal()
        try:
            # Find an issue that has cached related items
            cached_edge = session.scalar(select(RelatedIssuesCache))
            cls.sample_issue_id = cached_edge.source_issue_id if cached_edge else None
        finally:
            session.close()

    def test_get_cached_related_issues_success(self):
        """Verify GET /api/v1/related/{issue_id} returns cached top-K candidates."""
        self.assertIsNotNone(self.sample_issue_id)
        top_k = 3
        response = self.client.get(
            f"/api/v1/related/{self.sample_issue_id}",
            params={"top_k": top_k},
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query_issue_id"], self.sample_issue_id)
        self.assertIn("all-MiniLM-L6-v2", data["model_version"])
        self.assertEqual(data["total_matches"], top_k)
        self.assertGreater(data["retrieval_time_ms"], 0.0)

        matches = data["matches"]
        self.assertEqual(len(matches), top_k)
        ranks = [m["rank"] for m in matches]
        self.assertEqual(ranks, [1, 2, 3])

        for match in matches:
            self.assertNotEqual(match["issue_id"], self.sample_issue_id)
            self.assertGreaterEqual(match["similarity_score"], -1.0)
            self.assertLessEqual(match["similarity_score"], 1.0)
            self.assertIsNotNone(match["title"])
            self.assertIn(match["state"], ["open", "closed"])

    def test_get_cached_related_with_state_filter(self):
        """Verify GET /api/v1/related/{issue_id}?state=closed filters target state."""
        response = self.client.get(
            f"/api/v1/related/{self.sample_issue_id}",
            params={"top_k": 5, "state": "closed"},
        )
        self.assertEqual(response.status_code, 200)

        matches = response.json()["matches"]
        for m in matches:
            self.assertEqual(m["state"], "closed")

    def test_get_cached_related_not_found(self):
        """Verify 404 response for nonexistent issue ID."""
        response = self.client.get("/api/v1/related/99999999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    def test_post_dynamic_search_with_repo_scope(self):
        """Verify POST /api/v1/related embeds query text and filters by repository."""
        payload = {
            "title": "CUDA out of memory error when executing matrix multiplication",
            "body": "torch.cuda.OutOfMemoryError: CUDA out of memory in backward pass",
            "repo_name": "pytorch/pytorch",
            "top_k": 3,
        }
        response = self.client.post("/api/v1/related", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query_text"], payload["title"])
        self.assertIn("all-MiniLM-L6-v2", data["model_version"])
        self.assertEqual(data["total_matches"], 3)
        self.assertGreater(data["retrieval_time_ms"], 0.0)

        matches = data["matches"]
        self.assertEqual(len(matches), 3)
        for m in matches:
            self.assertEqual(m["repo_name"], "pytorch/pytorch")
            self.assertGreater(m["similarity_score"], 0.0)

    def test_post_dynamic_search_cross_repo(self):
        """Verify POST /api/v1/related cross-repo semantic retrieval."""
        payload = {
            "title": "Fix broken markdown links in documentation tutorial",
            "top_k": 4,
        }
        response = self.client.post("/api/v1/related", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_matches"], 4)
        matches = data["matches"]
        self.assertEqual(len(matches), 4)

        # Ranks must be 1..4 in descending score order
        scores = [m["similarity_score"] for m in matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_validation_errors(self):
        """Verify payload validations on query text length and state filter."""
        # Title too short
        res_short = self.client.post("/api/v1/related", json={"title": "hi"})
        self.assertEqual(res_short.status_code, 422)

        # Invalid state
        res_bad_state = self.client.post(
            "/api/v1/related",
            json={"title": "Valid title text", "state": "pending_review"},
        )
        self.assertEqual(res_bad_state.status_code, 422)


if __name__ == "__main__":
    unittest.main()
