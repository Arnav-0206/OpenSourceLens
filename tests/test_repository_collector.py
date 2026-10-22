"""Unit and integration tests for the Repository Collector."""

import unittest
from unittest.mock import MagicMock
from ml.data_collection.github_client import GitHubClient, NotFoundError
from ml.data_collection.repository_collector import RepositoryCollector, collect_repository


class TestRepositoryCollector(unittest.TestCase):
    """Test suite for repository identifier parsing, metadata extraction, and error handling."""

    def setUp(self):
        self.collector = RepositoryCollector()

    def test_parse_repo_identifier_valid(self):
        """Test valid owner/repo formats."""
        cases = [
            ("microsoft/vscode", ("microsoft", "vscode")),
            ("pytorch/pytorch/", ("pytorch", "pytorch")),
            ("https://github.com/kubernetes/kubernetes", ("kubernetes", "kubernetes")),
            ("https://github.com/huggingface/transformers/", ("huggingface", "transformers")),
        ]
        for input_str, expected in cases:
            with self.subTest(input_str=input_str):
                owner, repo = RepositoryCollector.parse_repo_identifier(input_str)
                self.assertEqual((owner, repo), expected)

    def test_parse_repo_identifier_invalid(self):
        """Test invalid formats raise ValueError."""
        invalid_cases = [
            "",
            "   ",
            "just-a-repo",
            "too/many/nested/slashes",
            "https://invalid-url.com/a/b/c",
        ]
        for input_str in invalid_cases:
            with self.subTest(input_str=input_str):
                with self.assertRaises(ValueError):
                    RepositoryCollector.parse_repo_identifier(input_str)

    def test_extract_metadata_schema_fields(self):
        """Test that all required schema fields from docs/0.3-data-definition.md are extracted."""
        mock_raw = {
            "name": "vscode",
            "full_name": "microsoft/vscode",
            "owner": {"login": "microsoft"},
            "description": "Visual Studio Code",
            "language": "TypeScript",
            "stargazers_count": 150000,
            "forks_count": 35000,
            "open_issues_count": 5000,
            "created_at": "2015-09-03T20:23:38Z",
            "updated_at": "2026-10-19T00:00:00Z",
            "pushed_at": "2026-10-19T01:00:00Z",
            "has_issues": True,
            "license": {"spdx_id": "MIT", "name": "MIT License"},
            "default_branch": "main",
            "archived": False,
        }

        extracted = self.collector._extract_metadata(mock_raw)

        # Verify all core fields exist
        required_fields = [
            "repository_name",
            "full_name",
            "owner",
            "description",
            "language",
            "stars",
            "forks",
            "open_issues_count",
            "created_at",
            "updated_at",
            "pushed_at",
            "has_issues",
            "license",
            "default_branch",
            "archived",
            "collected_at",
        ]
        for field in required_fields:
            self.assertIn(field, extracted)

        self.assertEqual(extracted["repository_name"], "vscode")
        self.assertEqual(extracted["owner"], "microsoft")
        self.assertEqual(extracted["stars"], 150000)
        self.assertEqual(extracted["license"], "MIT")

    def test_extract_metadata_missing_optional_fields(self):
        """Test that missing optional fields (description, license, language) fall back gracefully."""
        minimal_raw = {
            "name": "minimal-repo",
            "full_name": "user/minimal-repo",
            "owner": {"login": "user"},
        }
        extracted = self.collector._extract_metadata(minimal_raw)

        self.assertEqual(extracted["repository_name"], "minimal-repo")
        self.assertIsNone(extracted["description"])
        self.assertIsNone(extracted["license"])
        self.assertIsNone(extracted["language"])
        self.assertEqual(extracted["stars"], 0)
        self.assertEqual(extracted["forks"], 0)

    def test_live_collect_candidate_repo(self):
        """Integration test: collect real metadata for candidate repo microsoft/vscode."""
        result = self.collector.collect("microsoft/vscode")

        self.assertIn("metadata", result)
        self.assertIn("raw", result)

        meta = result["metadata"]
        self.assertEqual(meta["repository_name"], "vscode")
        self.assertEqual(meta["owner"], "microsoft")
        self.assertEqual(meta["full_name"], "microsoft/vscode")
        self.assertGreater(meta["stars"], 100000)
        self.assertGreater(meta["forks"], 10000)
        self.assertEqual(meta["language"], "TypeScript")
        self.assertTrue(meta["has_issues"])
        self.assertIsNotNone(meta["created_at"])

    def test_live_collect_nonexistent_repo(self):
        """Integration test: collecting a non-existent repo must raise NotFoundError."""
        with self.assertRaises(NotFoundError):
            self.collector.collect("microsoft/nonexistent-repo-for-opensourcelens-test-999")


if __name__ == "__main__":
    unittest.main()
