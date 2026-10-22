"""Unit tests for RawDataLoader in ml.preprocessing.loader."""

import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from ml.preprocessing.loader import RawDataLoader


class TestRawDataLoader(unittest.TestCase):
    """Test suite for RawDataLoader."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_path = Path(self.temp_dir.name)

        # Setup standard subdirectories
        (self.raw_path / "repositories").mkdir(parents=True)
        (self.raw_path / "issues").mkdir(parents=True)
        (self.raw_path / "comments").mkdir(parents=True)
        (self.raw_path / "events").mkdir(parents=True)

        # Write mock repository file
        repo_data = {
            "name": "vscode",
            "full_name": "microsoft/vscode",
            "owner": {"login": "microsoft"},
            "description": "Visual Studio Code",
            "stargazers_count": 160000,
            "forks_count": 30000,
            "subscribers_count": 3500,
            "open_issues_count": 5000,
            "language": "TypeScript",
            "license": {"name": "MIT License"},
            "created_at": "2015-09-03T20:23:38Z",
            "updated_at": "2026-10-19T00:00:00Z",
        }
        (self.raw_path / "repositories" / "microsoft__vscode.json").write_text(
            json.dumps(repo_data), encoding="utf-8"
        )

        # Write mock issues file
        issues_closed = [
            {
                "id": 1001,
                "number": 10,
                "title": "Bug in editor scrolling",
                "body": "Scroll bar flickers on reload.",
                "state": "closed",
                "state_reason": "completed",
                "user": {"login": "dev1", "type": "User"},
                "author_association": "CONTRIBUTOR",
                "created_at": "2026-01-01T12:00:00Z",
                "updated_at": "2026-01-02T12:00:00Z",
                "closed_at": "2026-01-02T12:00:00Z",
                "comments": 3,
                "reactions": {"total_count": 5},
                "labels": [{"name": "bug"}, {"name": "editor"}],
            }
        ]
        issues_open = [
            {
                "id": 1002,
                "number": 11,
                "title": "Feature: add vim mode toggle",
                "body": None,  # null body test
                "state": "open",
                "state_reason": None,
                "user": {"login": "dev2", "type": "User"},
                "author_association": "NONE",
                "created_at": "2026-01-03T12:00:00Z",
                "updated_at": "2026-01-03T12:00:00Z",
                "closed_at": None,
                "comments": 0,
                "reactions": None,  # null reactions test
                "labels": [],  # no labels test
            }
        ]
        (self.raw_path / "issues" / "microsoft__vscode__issues_closed_1.json").write_text(
            json.dumps(issues_closed), encoding="utf-8"
        )
        (self.raw_path / "issues" / "microsoft__vscode__issues_open_1.json").write_text(
            json.dumps(issues_open), encoding="utf-8"
        )

        # Write mock comments file
        comments_data = [
            {
                "id": 5001,
                "user": {"login": "reviewer1"},
                "author_association": "MEMBER",
                "created_at": "2026-01-01T14:00:00Z",
                "updated_at": "2026-01-01T14:00:00Z",
                "body": "Looking into this repro.",
                "reactions": {"total_count": 2},
            }
        ]
        (self.raw_path / "comments" / "microsoft__vscode__issue_10__comments.json").write_text(
            json.dumps(comments_data), encoding="utf-8"
        )

        # Write mock events file
        events_data = [
            {
                "id": 7001,
                "event": "labeled",
                "actor": {"login": "triage-bot"},
                "created_at": "2026-01-01T12:05:00Z",
                "label": {"name": "bug"},
                "assignee": None,
            },
            {
                "id": 7002,
                "event": "renamed",
                "actor": {"login": "dev1"},
                "created_at": "2026-01-01T12:10:00Z",
                "rename": {"from": "Old title", "to": "Bug in editor scrolling"},
            },
        ]
        (self.raw_path / "events" / "microsoft__vscode__issue_10__events.json").write_text(
            json.dumps(events_data), encoding="utf-8"
        )

        self.loader = RawDataLoader(raw_dir=self.raw_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_directory_handling(self):
        empty_dir = tempfile.TemporaryDirectory()
        loader = RawDataLoader(raw_dir=empty_dir.name)

        df_repos = loader.load_repositories()
        self.assertEqual(len(df_repos), 0)
        self.assertEqual(list(df_repos.columns), RawDataLoader.REPOSITORY_COLUMNS)

        df_issues = loader.load_issues()
        self.assertEqual(len(df_issues), 0)
        self.assertEqual(list(df_issues.columns), RawDataLoader.ISSUE_COLUMNS)

        df_comments = loader.load_comments()
        self.assertEqual(len(df_comments), 0)
        self.assertEqual(list(df_comments.columns), RawDataLoader.COMMENT_COLUMNS)

        df_events = loader.load_events()
        self.assertEqual(len(df_events), 0)
        self.assertEqual(list(df_events.columns), RawDataLoader.EVENT_COLUMNS)

        empty_dir.cleanup()

    def test_load_repositories(self):
        df = self.loader.load_repositories()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["repo_name"], "microsoft/vscode")
        self.assertEqual(row["owner"], "microsoft")
        self.assertEqual(row["language"], "TypeScript")
        self.assertEqual(row["license"], "MIT License")
        self.assertEqual(row["stars"], 160000)

        # Filtering test
        df_filtered = self.loader.load_repositories(repo_filter=["other/repo"])
        self.assertEqual(len(df_filtered), 0)

    def test_load_issues(self):
        df = self.loader.load_issues()
        self.assertEqual(len(df), 2)

        # Closed issue assertions
        closed_row = df[df["state"] == "closed"].iloc[0]
        self.assertEqual(closed_row["repo_name"], "microsoft/vscode")
        self.assertEqual(closed_row["issue_number"], 10)
        self.assertEqual(closed_row["title"], "Bug in editor scrolling")
        self.assertEqual(closed_row["body"], "Scroll bar flickers on reload.")
        self.assertEqual(closed_row["raw_labels"], ["bug", "editor"])
        self.assertEqual(closed_row["labels_count"], 2)
        self.assertEqual(closed_row["reactions_count"], 5)
        self.assertEqual(closed_row["author"], "dev1")

        # Open issue with nulls assertions
        open_row = df[df["state"] == "open"].iloc[0]
        self.assertEqual(open_row["issue_number"], 11)
        self.assertTrue(pd.isna(open_row["body"]))
        self.assertEqual(open_row["raw_labels"], [])
        self.assertEqual(open_row["reactions_count"], 0)

        # State filter test
        df_closed = self.loader.load_issues(state_filter="closed")
        self.assertEqual(len(df_closed), 1)
        self.assertEqual(df_closed.iloc[0]["state"], "closed")

        # Repo filter test
        df_empty = self.loader.load_issues(repo_filter=["pytorch/pytorch"])
        self.assertEqual(len(df_empty), 0)

    def test_load_comments(self):
        df = self.loader.load_comments()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["repo_name"], "microsoft/vscode")
        self.assertEqual(row["issue_number"], 10)
        self.assertEqual(row["comment_id"], 5001)
        self.assertEqual(row["author"], "reviewer1")
        self.assertEqual(row["body"], "Looking into this repro.")
        self.assertEqual(row["reactions_count"], 2)

    def test_load_events(self):
        df = self.loader.load_events()
        self.assertEqual(len(df), 2)

        label_ev = df[df["event_type"] == "labeled"].iloc[0]
        self.assertEqual(label_ev["issue_number"], 10)
        self.assertEqual(label_ev["actor"], "triage-bot")
        self.assertEqual(label_ev["label_name"], "bug")

        rename_ev = df[df["event_type"] == "renamed"].iloc[0]
        self.assertEqual(rename_ev["rename_from"], "Old title")
        self.assertEqual(rename_ev["rename_to"], "Bug in editor scrolling")

    def test_load_all_and_summarize(self):
        data = self.loader.load_all()
        self.assertIn("repositories", data)
        self.assertIn("issues", data)
        self.assertIn("comments", data)
        self.assertIn("events", data)

        summary = self.loader.summarize(data)
        self.assertEqual(summary["repositories"]["rows"], 1)
        self.assertEqual(summary["issues"]["rows"], 2)
        self.assertEqual(summary["comments"]["rows"], 1)
        self.assertEqual(summary["events"]["rows"], 2)
        self.assertIn("microsoft/vscode", summary["issues"]["repositories"])


if __name__ == "__main__":
    unittest.main()
