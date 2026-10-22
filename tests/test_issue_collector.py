"""Unit and integration tests for Issue Collector."""

import unittest
from ml.data_collection.github_client import NotFoundError
from ml.data_collection.issue_collector import (
    IssueCollector,
    is_pull_request,
    extract_issue_metadata,
)


class TestIssueCollector(unittest.TestCase):
    """Test suite for issue collection, PR filtering, schema validation, and error handling."""

    def setUp(self):
        self.collector = IssueCollector()

    def test_is_pull_request(self):
        """Test identification of Pull Requests vs Pure Issues."""
        pure_issue = {"id": 1, "number": 100, "title": "Bug"}
        pr_item = {"id": 2, "number": 101, "title": "Fix", "pull_request": {"url": "https://..."}}
        pr_null = {"id": 3, "number": 102, "title": "Task", "pull_request": None}

        self.assertFalse(is_pull_request(pure_issue))
        self.assertTrue(is_pull_request(pr_item))
        self.assertFalse(is_pull_request(pr_null))

    def test_extract_issue_metadata_schema_fields(self):
        """Verify that all 12 schema fields from docs/0.3-data-definition.md are extracted."""
        raw_mock = {
            "id": 12345678,
            "number": 42,
            "title": "Application crashes on startup",
            "body": "Here is the stack trace and reproduction steps.",
            "state": "closed",
            "user": {"login": "testuser"},
            "labels": [
                {"name": "bug", "color": "d73a4a"},
                {"name": "critical", "color": "b60205"},
            ],
            "created_at": "2026-01-01T10:00:00Z",
            "updated_at": "2026-01-02T12:00:00Z",
            "closed_at": "2026-01-03T15:00:00Z",
            "comments": 4,
        }

        extracted = extract_issue_metadata(raw_mock, "microsoft/vscode")

        expected_fields = [
            "issue_id",
            "issue_number",
            "repository",
            "title",
            "body",
            "state",
            "author",
            "labels",
            "created_at",
            "updated_at",
            "closed_at",
            "comments_count",
            "is_pull_request",
            "collected_at",
        ]
        for field in expected_fields:
            self.assertIn(field, extracted)

        self.assertEqual(extracted["issue_id"], 12345678)
        self.assertEqual(extracted["issue_number"], 42)
        self.assertEqual(extracted["repository"], "microsoft/vscode")
        self.assertEqual(extracted["title"], "Application crashes on startup")
        self.assertEqual(extracted["state"], "closed")
        self.assertEqual(extracted["author"], "testuser")
        self.assertEqual(extracted["labels"], ["bug", "critical"])
        self.assertEqual(extracted["comments_count"], 4)
        self.assertEqual(extracted["closed_at"], "2026-01-03T15:00:00Z")
        self.assertFalse(extracted["is_pull_request"])

    def test_extract_issue_metadata_null_and_missing_values(self):
        """Test handling of null descriptions, deleted authors, and null timestamps."""
        minimal_raw = {
            "id": 999,
            "number": 1,
            "title": "Empty issue",
            "body": None,
            "state": "open",
            "user": None,  # Deleted user / ghost
            "labels": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "closed_at": None,
            "comments": 0,
        }

        extracted = extract_issue_metadata(minimal_raw, "pytorch/pytorch")

        self.assertEqual(extracted["body"], "")
        self.assertEqual(extracted["author"], "ghost")
        self.assertIsNone(extracted["closed_at"])
        self.assertEqual(extracted["labels"], [])
        self.assertEqual(extracted["comments_count"], 0)

    def test_live_collect_single_closed_issue(self):
        """Integration test: fetch known closed issue #1 from microsoft/vscode."""
        result = self.collector.collect_issue("microsoft/vscode", 1, allow_pull_request=True)

        self.assertIn("issue", result)
        self.assertIn("raw", result)

        issue = result["issue"]
        self.assertEqual(issue["issue_number"], 1)
        self.assertEqual(issue["repository"], "microsoft/vscode")
        self.assertEqual(issue["state"], "closed")
        self.assertIsNotNone(issue["closed_at"])
        self.assertIsNotNone(issue["created_at"])
        self.assertIsInstance(issue["labels"], list)

    def test_live_collect_page_with_pr_filtering(self):
        """Integration test: collect page of issues and verify PRs are excluded."""
        results = self.collector.collect_issues_page(
            "microsoft/vscode",
            state="all",
            page=1,
            per_page=20,
            filter_pull_requests=True,
        )

        self.assertGreater(len(results), 0)
        for item in results:
            issue = item["issue"]
            raw = item["raw"]
            # Verify PR filtering holds
            self.assertFalse(issue["is_pull_request"])
            self.assertNotIn("pull_request", raw)
            # Verify required schema fields
            self.assertIsNotNone(issue["issue_id"])
            self.assertIsNotNone(issue["issue_number"])
            self.assertIn(issue["state"], ["open", "closed"])

    def test_live_collect_nonexistent_issue(self):
        """Integration test: querying a non-existent issue number must raise NotFoundError."""
        with self.assertRaises(NotFoundError):
            self.collector.collect_issue("microsoft/vscode", 999999999)


if __name__ == "__main__":
    unittest.main()
