"""Unit tests for DataInventory audit engine."""

import json
import tempfile
import unittest
from pathlib import Path
from ml.data_collection.inventory import DataInventory


class TestDataInventory(unittest.TestCase):
    """Test suite for DataInventory."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_path = Path(self.temp_dir.name)

        # Create subdirectories
        (self.raw_path / "repositories").mkdir(parents=True)
        (self.raw_path / "issues").mkdir(parents=True)
        (self.raw_path / "comments").mkdir(parents=True)
        (self.raw_path / "events").mkdir(parents=True)

        # Write dummy repository file
        repo_data = {
            "name": "mockrepo",
            "full_name": "testowner/mockrepo",
            "stargazers_count": 1200,
            "forks_count": 300,
            "subscribers_count": 50,
            "open_issues_count": 42,
            "language": "Python",
            "license": {"name": "MIT License"},
            "created_at": "2020-01-01T00:00:00Z",
            "updated_at": "2026-10-19T00:00:00Z",
        }
        (self.raw_path / "repositories" / "testowner__mockrepo.json").write_text(
            json.dumps(repo_data), encoding="utf-8"
        )

        # Write dummy issues files
        issues_closed = [
            {
                "id": 1,
                "number": 101,
                "state": "closed",
                "title": "Bug in model",
                "body": "Fix this crash please.",
                "user": {"login": "alice"},
                "labels": [{"name": "bug"}, {"name": "priority-high"}],
                "created_at": "2026-01-01T10:00:00Z",
                "closed_at": "2026-01-02T10:00:00Z",
                "comments": 2,
            },
            {
                "id": 2,
                "number": 102,
                "state": "closed",
                "title": "Documentation typo",
                "body": "",  # Empty body
                "user": {"login": "bob"},
                "labels": [],  # Unlabeled
                "created_at": "2026-01-03T10:00:00Z",
                "closed_at": None,  # Closed but missing closed_at
                "comments": 0,
            },
        ]
        issues_open = [
            {
                "id": 3,
                "number": 103,
                "state": "open",
                "title": "Feature request",
                "body": "Add support for GPU acceleration.",
                "user": None,  # Missing author
                "labels": [{"name": "feature"}, {"name": "bug"}],
                "created_at": "2026-01-05T10:00:00Z",
                "closed_at": None,
                "comments": 1,
            }
        ]

        (self.raw_path / "issues" / "testowner__mockrepo__issues_closed_2.json").write_text(
            json.dumps(issues_closed), encoding="utf-8"
        )
        (self.raw_path / "issues" / "testowner__mockrepo__issues_open_1.json").write_text(
            json.dumps(issues_open), encoding="utf-8"
        )

        # Write dummy comments & events
        comments_data = [{"id": 11, "body": "Fixed in PR #10"}]
        (self.raw_path / "comments" / "testowner__mockrepo__issue_101__comments.json").write_text(
            json.dumps(comments_data), encoding="utf-8"
        )

        events_data = [{"id": 21, "event": "closed"}, {"id": 22, "event": "labeled"}]
        (self.raw_path / "events" / "testowner__mockrepo__issue_101__events.json").write_text(
            json.dumps(events_data), encoding="utf-8"
        )

        self.inventory = DataInventory(raw_dir=str(self.raw_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_repositories(self):
        repos = self.inventory.scan_repositories()
        self.assertIn("testowner/mockrepo", repos)
        data = repos["testowner/mockrepo"]
        self.assertEqual(data["stars"], 1200)
        self.assertEqual(data["language"], "Python")
        self.assertEqual(data["license"], "MIT License")

    def test_scan_issues(self):
        issues_stats = self.inventory.scan_issues()
        self.assertIn("testowner/mockrepo", issues_stats)
        s = issues_stats["testowner/mockrepo"]
        self.assertEqual(s["total_issues"], 3)
        self.assertEqual(s["closed_issues"], 2)
        self.assertEqual(s["open_issues"], 1)
        self.assertEqual(s["empty_body_count"], 1)
        self.assertEqual(s["missing_author_count"], 1)
        self.assertEqual(s["closed_missing_closed_at"], 1)
        self.assertEqual(s["unlabeled_count"], 1)
        self.assertEqual(s["labels_counter"]["bug"], 2)
        self.assertEqual(s["labels_counter"]["feature"], 1)

    def test_scan_comments_and_events(self):
        res = self.inventory.scan_comments_and_events()
        comments = res["comments_by_repo"].get("testowner/mockrepo")
        self.assertIsNotNone(comments)
        self.assertEqual(comments["files"], 1)
        self.assertEqual(comments["total_comments"], 1)

        events = res["events_by_repo"].get("testowner/mockrepo")
        self.assertIsNotNone(events)
        self.assertEqual(events["files"], 1)
        self.assertEqual(events["total_events"], 2)
        self.assertEqual(events["event_types"]["closed"], 1)

    def test_generate_inventory_and_markdown(self):
        inv = self.inventory.generate_inventory()
        g = inv["global_summary"]
        self.assertEqual(g["total_repositories"], 1)
        self.assertEqual(g["total_issues"], 3)
        self.assertEqual(g["total_open_issues"], 1)
        self.assertEqual(g["total_closed_issues"], 2)
        self.assertEqual(g["total_empty_bodies"], 1)
        self.assertEqual(g["total_missing_authors"], 1)
        self.assertEqual(g["total_unlabeled_issues"], 1)
        self.assertEqual(g["total_distinct_labels"], 3)  # bug, priority-high, feature

        md = self.inventory.format_markdown_report(inv)
        self.assertIn("# OpenSourceLens — Raw Data Inventory", md)
        self.assertIn("testowner/mockrepo", md)
        self.assertIn("MIT License", md)

    def test_save_report_and_json(self):
        out_md = self.raw_path / "report.md"
        out_json = self.raw_path / "report.json"

        p_md = self.inventory.save_report(str(out_md))
        p_json = self.inventory.save_json(str(out_json))

        self.assertTrue(p_md.exists())
        self.assertTrue(p_json.exists())
        loaded = json.loads(p_json.read_text(encoding="utf-8"))
        self.assertEqual(loaded["global_summary"]["total_issues"], 3)


if __name__ == "__main__":
    unittest.main()
