"""Unit and API integration tests for TopicDiscoveryService."""

from datetime import datetime, timezone
import unittest

from fastapi.testclient import TestClient

from backend.app.database.models import Issue, Repository
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services.nlp.topic_discovery_service import topic_discovery_service


class TestTopicDiscovery(unittest.TestCase):
    """Test suite for unsupervised topic discovery clustering and API endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.session = SessionLocal()

        cls.owner = "topic-org"
        cls.name = "clusters-repo"
        cls.repo_name = f"{cls.owner}/{cls.name}"

        existing = cls.session.query(Repository).filter(Repository.repo_name == cls.repo_name).first()
        if existing:
            cls.session.delete(existing)
            cls.session.commit()

        cls.repo = Repository(
            id=777888,
            repo_name=cls.repo_name,
            owner=cls.owner,
            name=cls.name,
            stars_count=80,
            forks_count=15,
            open_issues_count=6,
        )
        cls.session.add(cls.repo)
        cls.session.commit()

        now = datetime.now(timezone.utc)
        # Create 6 issues with distinct thematic groups (Group A: CUDA memory, Group B: Documentation)
        issues = [
            Issue(
                id=77701,
                repository_id=cls.repo.id,
                repo_name=cls.repo_name,
                issue_number=1,
                title="CUDA out of memory error during backward pass",
                body="Memory allocation fails in GPU buffer",
                state="open",
                author="dev1",
                created_at=now,
            ),
            Issue(
                id=77702,
                repository_id=cls.repo.id,
                repo_name=cls.repo_name,
                issue_number=2,
                title="GPU memory allocation leak in libtorch",
                body="CUDA allocator does not release memory on tensor deletion",
                state="open",
                author="dev2",
                created_at=now,
            ),
            Issue(
                id=77703,
                repository_id=cls.repo.id,
                repo_name=cls.repo_name,
                issue_number=3,
                title="PyTorch CUDA OOM when running large batch sizes",
                body="Out of memory error in multi-GPU training",
                state="closed",
                author="dev3",
                created_at=now,
            ),
            Issue(
                id=77704,
                repository_id=cls.repo.id,
                repo_name=cls.repo_name,
                issue_number=4,
                title="Fix broken docstring tutorial links in README",
                body="Documentation link points to 404 page in installation guide",
                state="open",
                author="writer1",
                created_at=now,
            ),
            Issue(
                id=77705,
                repository_id=cls.repo.id,
                repo_name=cls.repo_name,
                issue_number=5,
                title="Update API documentation for tensor operations",
                body="Add documentation examples and tutorial guide for new APIs",
                state="closed",
                author="writer2",
                created_at=now,
            ),
            Issue(
                id=77706,
                repository_id=cls.repo.id,
                repo_name=cls.repo_name,
                issue_number=6,
                title="Documentation spelling typo in quickstart doc",
                body="Fix misspelling on lines 24-28 of getting started docs",
                state="closed",
                author="writer3",
                created_at=now,
            ),
        ]
        cls.session.add_all(issues)
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        repo = cls.session.query(Repository).filter(Repository.id == 777888).first()
        if repo:
            cls.session.delete(repo)
            cls.session.commit()
        cls.session.close()

    def test_discover_topics_service(self):
        """Verify TopicDiscoveryService runs clustering and returns valid topics."""
        result = topic_discovery_service.discover_repository_topics(
            repository_id=self.repo.id,
            repo_name=self.repo_name,
            db=self.session,
            max_clusters=4,
        )
        self.assertEqual(result.repo_name, self.repo_name)
        self.assertEqual(result.total_items_clustered, 6)
        self.assertGreaterEqual(result.total_topics_discovered, 2)

        # Check topic properties
        for topic in result.topics:
            self.assertGreater(topic.total_count, 0)
            self.assertEqual(topic.open_count + topic.closed_count, topic.total_count)
            self.assertIsInstance(topic.title, str)
            self.assertGreater(len(topic.title), 0)
            self.assertIsInstance(topic.keywords, list)
            self.assertIsInstance(topic.representative_items, list)

    def test_api_topics_endpoint(self):
        """Verify GET /api/v1/analytics/{owner}/{name}/topics returns 200 and schema."""
        resp = self.client.get(f"/api/v1/analytics/{self.owner}/{self.name}/topics")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["repo_name"], self.repo_name)
        self.assertIn("topics", data)
        self.assertGreaterEqual(len(data["topics"]), 2)
        self.assertIn("execution_time_ms", data)


if __name__ == "__main__":
    unittest.main()
