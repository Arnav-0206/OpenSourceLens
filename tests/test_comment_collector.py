"""Unit and integration tests for Comment Collector."""

import unittest
from ml.data_collection.github_client import NotFoundError
from ml.data_collection.comment_collector import (
    CommentCollector,
    extract_comment_metadata,
)


class TestCommentCollector(unittest.TestCase):
    """Test suite for comment collection, schema validation, and error handling."""

    def setUp(self):
        self.collector = CommentCollector()

    def test_extract_comment_metadata_schema_fields(self):
        """Verify that all comment fields from docs/0.3-data-definition.md are extracted."""
        raw_mock = {
            "id": 888888,
            "body": "This issue is fixed in commit abc1234.",
            "user": {"login": "dev1"},
            "created_at": "2026-02-01T10:00:00Z",
            "updated_at": "2026-02-01T11:00:00Z",
        }

        extracted = extract_comment_metadata(raw_mock, "microsoft/vscode", issue_number=10)

        expected_fields = [
            "comment_id",
            "issue_number",
            "repository",
            "author",
            "body",
            "created_at",
            "updated_at",
            "collected_at",
        ]
        for field in expected_fields:
            self.assertIn(field, extracted)

        self.assertEqual(extracted["comment_id"], 888888)
        self.assertEqual(extracted["issue_number"], 10)
        self.assertEqual(extracted["repository"], "microsoft/vscode")
        self.assertEqual(extracted["author"], "dev1")
        self.assertEqual(extracted["body"], "This issue is fixed in commit abc1234.")
        self.assertEqual(extracted["created_at"], "2026-02-01T10:00:00Z")
        self.assertEqual(extracted["updated_at"], "2026-02-01T11:00:00Z")
        self.assertIsNotNone(extracted["collected_at"])

    def test_extract_comment_metadata_null_and_missing_values(self):
        """Test handling of null body and deleted/ghost author."""
        minimal_raw = {
            "id": 999999,
            "body": None,
            "user": None,
            "created_at": "2026-02-01T00:00:00Z",
            "updated_at": "2026-02-01T00:00:00Z",
        }

        extracted = extract_comment_metadata(minimal_raw, "pytorch/pytorch", issue_number=5)

        self.assertEqual(extracted["body"], "")
        self.assertEqual(extracted["author"], "ghost")

    def test_live_collect_comments_for_issue(self):
        """Integration test: fetch live comments for microsoft/vscode issue #1."""
        results = self.collector.collect_comments_page("microsoft/vscode", issue_number=1, per_page=5)

        self.assertGreater(len(results), 0)
        first_comment = results[0]["comment"]
        raw = results[0]["raw"]

        self.assertEqual(first_comment["issue_number"], 1)
        self.assertEqual(first_comment["repository"], "microsoft/vscode")
        self.assertIsNotNone(first_comment["comment_id"])
        self.assertIsNotNone(first_comment["author"])
        self.assertIsNotNone(first_comment["created_at"])
        self.assertIn("id", raw)

    def test_live_collect_all_comments_with_cap(self):
        """Integration test: test collect_all_comments_for_issue with a max_comments cap."""
        results = self.collector.collect_all_comments_for_issue("microsoft/vscode", issue_number=1, max_comments=3)

        self.assertEqual(len(results), 3)

    def test_live_collect_nonexistent_issue_comments(self):
        """Integration test: querying comments for a non-existent repo or issue must handle properly."""
        with self.assertRaises(NotFoundError):
            self.collector.collect_comments_page("microsoft/vscode-definitely-nonexistent-repo-9999", issue_number=1)


if __name__ == "__main__":
    unittest.main()
