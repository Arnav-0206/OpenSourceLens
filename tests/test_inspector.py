"""Unit tests for DatasetInspector in ml.preprocessing.inspector."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.preprocessing.inspector import DatasetInspector


class TestDatasetInspector(unittest.TestCase):
    """Test suite for DatasetInspector."""

    def setUp(self):
        self.sample_issues = pd.DataFrame([
            {
                "repo_name": "microsoft/vscode",
                "issue_id": 101,
                "issue_number": 1,
                "title": "Fix scrolling bug",
                "body": "Here is code:\n```python\nprint('hello')\n```\n<!-- template comment -->",
                "state": "closed",
                "state_reason": "completed",
                "author": "alice",
                "author_type": "User",
                "author_association": "MEMBER",
                "created_at": "2026-01-01T10:00:00Z",
                "updated_at": "2026-01-02T10:00:00Z",
                "closed_at": "2026-01-02T10:00:00Z",
                "comments_count": 2,
                "reactions_count": 3,
                "raw_labels": ["bug", "ui"],
                "labels_count": 2,
            },
            {
                "repo_name": "microsoft/vscode",
                "issue_id": 102,
                "issue_number": 2,
                "title": "Add new dark theme",
                "body": "Check this link: https://example.com/theme <p>custom tag</p>",
                "state": "open",
                "state_reason": None,
                "author": "bob",
                "author_type": "User",
                "author_association": "CONTRIBUTOR",
                "created_at": "2026-01-03T10:00:00Z",
                "updated_at": "2026-01-03T10:00:00Z",
                "closed_at": None,
                "comments_count": 0,
                "reactions_count": 0,
                "raw_labels": [],
                "labels_count": 0,
            },
            {
                "repo_name": "pytorch/pytorch",
                "issue_id": 103,
                "issue_number": 1,
                "title": "Tensor shape mismatch",
                "body": "",  # Empty body
                "state": "open",
                "state_reason": None,
                "author": "charlie",
                "author_type": "User",
                "author_association": "NONE",
                "created_at": "2026-01-04T10:00:00Z",
                "updated_at": "2026-01-04T10:00:00Z",
                "closed_at": None,
                "comments_count": 1,
                "reactions_count": 1,
                "raw_labels": ["bug", "triage", "core", "gpu"],
                "labels_count": 4,
            },
        ])
        self.inspector = DatasetInspector(df_issues=self.sample_issues)

    def test_empty_dataframe(self):
        empty_inspector = DatasetInspector(df_issues=pd.DataFrame())
        schema = empty_inspector.audit_schema()
        self.assertEqual(schema["total_rows"], 0)

        dups = empty_inspector.audit_duplicates()
        self.assertEqual(dups["duplicate_ids"], 0)

        missing = empty_inspector.audit_missingness()
        self.assertEqual(missing["empty_bodies_count"], 0)

        rep_dict = empty_inspector.generate_report_dict()
        self.assertIn("schema", rep_dict)

    def test_audit_schema(self):
        schema = self.inspector.audit_schema()
        self.assertEqual(schema["total_rows"], 3)
        self.assertEqual(schema["total_columns"], len(self.sample_issues.columns))
        self.assertIn("title", schema["columns"])
        self.assertEqual(schema["columns"]["title"]["null_count"], 0)
        self.assertEqual(schema["columns"]["closed_at"]["null_count"], 2)

    def test_audit_duplicates(self):
        # Base data has no duplicates
        dups = self.inspector.audit_duplicates()
        self.assertEqual(dups["duplicate_ids"], 0)
        self.assertEqual(dups["duplicate_composite_keys"], 0)

        # Inject duplicate
        dup_df = pd.concat([self.sample_issues, self.sample_issues.iloc[[0]]], ignore_index=True)
        dup_inspector = DatasetInspector(df_issues=dup_df)
        dups_detected = dup_inspector.audit_duplicates()
        self.assertEqual(dups_detected["duplicate_ids"], 1)
        self.assertEqual(dups_detected["duplicate_composite_keys"], 1)
        self.assertEqual(dups_detected["duplicate_titles"], 1)

    def test_audit_missingness(self):
        missing = self.inspector.audit_missingness()
        self.assertEqual(missing["empty_bodies_count"], 1)  # charlie's empty body
        ct = missing["cross_tab_state_closed_at"]
        self.assertIn("open", ct)
        self.assertIn("closed", ct)
        self.assertEqual(ct["open"]["closed_at_missing"], 2)
        self.assertEqual(ct["closed"]["closed_at_present"], 1)

    def test_audit_categoricals(self):
        cats = self.inspector.audit_categoricals()
        self.assertIn("state", cats)
        self.assertIn("state_reason", cats)
        self.assertIn("author_association", cats)

        # 2 open, 1 closed
        states = {item["value"]: item["count"] for item in cats["state"]}
        self.assertEqual(states["open"], 2)
        self.assertEqual(states["closed"], 1)

    def test_audit_text(self):
        text_stats = self.inspector.audit_text()
        self.assertIn("title_char_stats", text_stats)
        self.assertIn("body_char_stats", text_stats)

        marts = text_stats["markdown_artifacts"]
        self.assertEqual(marts["issues_with_code_blocks"], 1)
        self.assertEqual(marts["issues_with_urls"], 1)
        self.assertEqual(marts["issues_with_html_comments"], 1)
        self.assertEqual(marts["issues_with_html_tags"], 1)

    def test_audit_labels(self):
        labels_stats = self.inspector.audit_labels()
        self.assertEqual(labels_stats["unlabeled_count"], 1)
        dist = labels_stats["distribution"]
        self.assertEqual(dist["0_labels"]["count"], 1)
        self.assertEqual(dist["1_to_2_labels"]["count"], 1)
        self.assertEqual(dist["3_to_5_labels"]["count"], 1)

        # Most common label is "bug" (2 occurrences)
        top1 = labels_stats["top_global_labels"][0]
        self.assertEqual(top1["name"], "bug")
        self.assertEqual(top1["count"], 2)

    def test_save_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "inspection.md"
            saved_path = self.inspector.save_report(out_file)
            self.assertTrue(saved_path.exists())
            content = saved_path.read_text(encoding="utf-8")
            self.assertIn("# OpenSourceLens", content)
            self.assertIn("microsoft/vscode", content)
            self.assertIn("## 1. Column-Level Schema & Completeness Audit", content)


if __name__ == "__main__":
    unittest.main()
