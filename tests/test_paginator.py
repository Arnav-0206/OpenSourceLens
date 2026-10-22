"""Unit and integration tests for GitHubPaginator."""

import unittest
from ml.data_collection.paginator import GitHubPaginator
from ml.data_collection.issue_collector import IssueCollector


class TestGitHubPaginator(unittest.TestCase):
    """Test suite for Link header parsing, generator streaming, and multi-page pagination."""

    def setUp(self):
        self.paginator = GitHubPaginator(polite_pause=0.01)

    def test_parse_link_header_valid(self):
        """Test parsing of valid RFC 5988 Link headers."""
        raw_header = (
            '<https://api.github.com/repositories/1/issues?page=2>; rel="next", '
            '<https://api.github.com/repositories/1/issues?page=100>; rel="last", '
            '<https://api.github.com/repositories/1/issues?page=1>; rel="first"'
        )
        parsed = GitHubPaginator.parse_link_header(raw_header)

        self.assertEqual(parsed.get("next"), "https://api.github.com/repositories/1/issues?page=2")
        self.assertEqual(parsed.get("last"), "https://api.github.com/repositories/1/issues?page=100")
        self.assertEqual(parsed.get("first"), "https://api.github.com/repositories/1/issues?page=1")
        self.assertNotIn("prev", parsed)

    def test_parse_link_header_empty_and_none(self):
        """Test that empty or None Link headers return an empty dictionary."""
        self.assertEqual(GitHubPaginator.parse_link_header(None), {})
        self.assertEqual(GitHubPaginator.parse_link_header(""), {})
        self.assertEqual(GitHubPaginator.parse_link_header("   "), {})

    def test_live_pagination_crossing_pages(self):
        """Integration test: stream 25 items with per_page=10, crossing at least 3 pages."""
        endpoint = "/repos/microsoft/vscode/issues"
        params = {"state": "all", "per_page": 10}

        items = list(self.paginator.paginate_items(endpoint, params=params, max_items=25))

        self.assertEqual(len(items), 25)
        # Ensure all items have standard issue keys
        for item in items:
            self.assertIn("id", item)
            self.assertIn("number", item)

    def test_live_paginate_pages_metadata(self):
        """Integration test: verify page-level generator and metadata."""
        endpoint = "/repos/microsoft/vscode/issues"
        params = {"state": "all", "per_page": 5}

        pages_collected = []
        for items, meta in self.paginator.paginate_pages(endpoint, params=params, max_pages=2):
            pages_collected.append((items, meta))

        self.assertEqual(len(pages_collected), 2)
        # Verify first page
        page1_items, page1_meta = pages_collected[0]
        self.assertEqual(page1_meta["page"], 1)
        self.assertEqual(len(page1_items), 5)
        self.assertTrue(page1_meta["has_next"])
        self.assertIsNotNone(page1_meta["next_url"])

        # Verify second page
        page2_items, page2_meta = pages_collected[1]
        self.assertEqual(page2_meta["page"], 2)
        self.assertEqual(len(page2_items), 5)

    def test_live_collect_issues_stream_pr_filtering(self):
        """Integration test: stream pure issues across multiple pages using IssueCollector."""
        collector = IssueCollector()
        stream = collector.collect_issues_stream(
            "microsoft/vscode",
            state="all",
            max_issues=15,
            per_page=10,
            filter_pull_requests=True,
        )

        issues = list(stream)
        self.assertEqual(len(issues), 15)

        for record in issues:
            issue = record["issue"]
            raw = record["raw"]
            self.assertFalse(issue["is_pull_request"])
            self.assertNotIn("pull_request", raw)
            self.assertEqual(issue["repository"], "microsoft/vscode")


if __name__ == "__main__":
    unittest.main()
