"""Integration tests for POST /api/v1/analytics/{owner}/{name}/similar endpoint."""

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from backend.app.database.models import Issue, Repository
from backend.app.database.session import SessionLocal
from backend.app.main import app


class TestAPISemanticSimilarity(unittest.TestCase):
    """Test suite for semantic similarity and duplicate retrieval endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.session = SessionLocal()

        cls.owner = "sim-org"
        cls.name = "search-repo"
        cls.repo_name = f"{cls.owner}/{cls.name}"

        # Clean up any existing records
        existing = cls.session.query(Repository).filter(Repository.repo_name == cls.repo_name).first()
        if existing:
            cls.session.delete(existing)
            cls.session.commit()

        cls.repo = Repository(
            id=987654,
            repo_name=cls.repo_name,
            owner=cls.owner,
            name=cls.name,
            stars_count=50,
            forks_count=10,
            open_issues_count=3,
        )
        cls.session.add(cls.repo)
        cls.session.commit()

        now = datetime.now(timezone.utc)
        cls.issue1 = Issue(
            id=98701,
            repository_id=cls.repo.id,
            repo_name=cls.repo_name,
            issue_number=101,
            title="Fatal crash when opening large markdown file in editor",
            body="Memory leak causes OOM and application terminates unexpectedly with SIGSEGV",
            state="open",
            canonical_label="bug",
            author="tester1",
            created_at=now,
        )
        cls.issue2 = Issue(
            id=98702,
            repository_id=cls.repo.id,
            repo_name=cls.repo_name,
            issue_number=102,
            title="Add dark high-contrast theme for accessibility",
            body="Feature proposal for enhanced accessibility color palette in preferences",
            state="closed",
            canonical_label="feature",
            author="designer1",
            created_at=now,
        )
        cls.session.add_all([cls.issue1, cls.issue2])
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        # Cleanup test data
        repo = cls.session.query(Repository).filter(Repository.id == 987654).first()
        if repo:
            cls.session.delete(repo)
            cls.session.commit()
        cls.session.close()

    def test_similar_query_text_duplicate_match(self):
        """Verify semantic query by text detects duplicate of issue 101."""
        payload = {
            "query_text": "Application crash and memory leak on large markdown files",
            "top_k": 3,
            "threshold": 0.40,
        }
        resp = self.client.post(
            f"/api/v1/analytics/{self.owner}/{self.name}/similar",
            json=payload,
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertGreaterEqual(data["total_candidates_found"], 1)
        top = data["candidates"][0]
        self.assertEqual(top["number"], 101)
        self.assertEqual(top["entity_type"], "issue")
        self.assertGreaterEqual(top["similarity_score"], 0.65)
        self.assertIn("latency_ms", data)
        self.assertGreater(data["latency_ms"], 0.0)

    def test_similar_query_by_issue_number(self):
        """Verify querying by issue_number finds related issues while excluding itself."""
        payload = {
            "issue_number": 101,
            "top_k": 3,
            "threshold": 0.0,
        }
        resp = self.client.post(
            f"/api/v1/analytics/{self.owner}/{self.name}/similar",
            json=payload,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Target issue 101 must NOT appear in its own results
        for candidate in data["candidates"]:
            self.assertNotEqual(candidate["number"], 101)

    def test_similar_404_repository(self):
        """Verify 404 response for unknown repository."""
        resp = self.client.post(
            "/api/v1/analytics/nonexistent-org/unknown-repo/similar",
            json={"query_text": "sample query text"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_similar_validation_error_missing_query(self):
        """Verify 422 error when neither query_text nor issue_number is provided."""
        resp = self.client.post(
            f"/api/v1/analytics/{self.owner}/{self.name}/similar",
            json={},
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
