"""Unit and integration tests for Event Collector."""

import unittest
from ml.data_collection.github_client import NotFoundError
from ml.data_collection.event_collector import (
    EventCollector,
    extract_event_metadata,
)


class TestEventCollector(unittest.TestCase):
    """Test suite for issue timeline events collection, schema validation, and error handling."""

    def setUp(self):
        self.collector = EventCollector()

    def test_extract_event_metadata_closed_event(self):
        """Verify schema extraction for a 'closed' event."""
        raw_mock = {
            "id": 1001,
            "event": "closed",
            "actor": {"login": "maintainer1"},
            "commit_id": "7b3a37f0a6bdc0d834b47eb537ae7ab29d5a7770",
            "created_at": "2026-03-01T12:00:00Z",
        }

        extracted = extract_event_metadata(raw_mock, "microsoft/vscode", issue_number=42)

        expected_fields = [
            "event_id",
            "issue_number",
            "repository",
            "event",
            "actor",
            "created_at",
            "commit_id",
            "label_name",
            "assignee",
            "collected_at",
        ]
        for field in expected_fields:
            self.assertIn(field, extracted)

        self.assertEqual(extracted["event_id"], 1001)
        self.assertEqual(extracted["issue_number"], 42)
        self.assertEqual(extracted["repository"], "microsoft/vscode")
        self.assertEqual(extracted["event"], "closed")
        self.assertEqual(extracted["actor"], "maintainer1")
        self.assertEqual(extracted["commit_id"], "7b3a37f0a6bdc0d834b47eb537ae7ab29d5a7770")
        self.assertIsNone(extracted["label_name"])
        self.assertIsNone(extracted["assignee"])

    def test_extract_event_metadata_labeled_and_assigned(self):
        """Verify schema extraction for 'labeled' and 'assigned' event types."""
        labeled_mock = {
            "id": 1002,
            "event": "labeled",
            "actor": {"login": "bot"},
            "label": {"name": "bug", "color": "d73a4a"},
            "created_at": "2026-03-01T10:00:00Z",
        }
        assigned_mock = {
            "id": 1003,
            "event": "assigned",
            "actor": {"login": "lead_dev"},
            "assignee": {"login": "junior_dev"},
            "created_at": "2026-03-01T10:30:00Z",
        }

        extracted_labeled = extract_event_metadata(labeled_mock, "pytorch/pytorch", issue_number=10)
        self.assertEqual(extracted_labeled["event"], "labeled")
        self.assertEqual(extracted_labeled["label_name"], "bug")

        extracted_assigned = extract_event_metadata(assigned_mock, "pytorch/pytorch", issue_number=10)
        self.assertEqual(extracted_assigned["event"], "assigned")
        self.assertEqual(extracted_assigned["assignee"], "junior_dev")

    def test_extract_event_metadata_null_and_missing_values(self):
        """Test fallback handling for missing actor (system bots) and empty commit_id."""
        minimal_raw = {
            "id": 1004,
            "event": "locked",
            "actor": None,
            "created_at": "2026-03-01T00:00:00Z",
        }

        extracted = extract_event_metadata(minimal_raw, "kubernetes/kubernetes", issue_number=99)
        self.assertEqual(extracted["actor"], "system")
        self.assertIsNone(extracted["commit_id"])
        self.assertIsNone(extracted["label_name"])
        self.assertIsNone(extracted["assignee"])

    def test_live_collect_events_for_issue(self):
        """Integration test: fetch live timeline events for microsoft/vscode issue #342495."""
        results = self.collector.collect_events_page("microsoft/vscode", issue_number=342495, per_page=5)

        self.assertGreater(len(results), 0)
        first_event = results[0]["event"]
        raw = results[0]["raw"]

        self.assertEqual(first_event["issue_number"], 342495)
        self.assertEqual(first_event["repository"], "microsoft/vscode")
        self.assertIsNotNone(first_event["event_id"])
        self.assertIsNotNone(first_event["event"])
        self.assertIsNotNone(first_event["actor"])
        self.assertIsNotNone(first_event["created_at"])
        self.assertIn("id", raw)

    def test_live_collect_all_events_with_cap(self):
        """Integration test: verify collect_all_events_for_issue respects max_events cap."""
        results = self.collector.collect_all_events_for_issue("microsoft/vscode", issue_number=342495, max_events=2)
        self.assertEqual(len(results), 2)

    def test_live_collect_nonexistent_issue_events(self):
        """Integration test: querying events for a non-existent repo must raise NotFoundError."""
        with self.assertRaises(NotFoundError):
            self.collector.collect_events_page("microsoft/vscode-nonexistent-repo-test-9999", issue_number=1)


if __name__ == "__main__":
    unittest.main()
