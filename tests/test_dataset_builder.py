"""Unit tests for IssueDatasetBuilder in ml.preprocessing.dataset_builder."""

import unittest
import numpy as np
import pandas as pd

from ml.preprocessing.dataset_builder import IssueDatasetBuilder


class TestIssueDatasetBuilder(unittest.TestCase):
    """Test suite for IssueDatasetBuilder."""

    def setUp(self):
        self.builder = IssueDatasetBuilder()
        self.mock_raw_df = pd.DataFrame([
            {
                "repo_name": "microsoft/vscode",
                "issue_id": 1001,
                "issue_number": 1,
                "title": "Fix scrolling bug",
                "body": "Here is repro: ```x = 1``` in <URL> reported by @alice in #10",
                "state": "closed",
                "state_reason": "completed",
                "author": "alice",
                "author_type": "User",
                "author_association": "MEMBER",
                "created_at": "2026-10-10T10:00:00Z",
                "updated_at": "2026-10-12T10:00:00Z",
                "closed_at": "2026-10-12T10:00:00Z",
                "comments_count": 2,
                "reactions_count": 1,
                "raw_labels": ["bug", "ui"],
                "labels_count": 2,
            },
            {
                "repo_name": "pytorch/pytorch",
                "issue_id": 1002,
                "issue_number": 2,
                "title": "Support complex32 CPU",
                "body": "",  # Empty body -> should be imputed from title
                "state": "open",
                "state_reason": None,
                "author": "bob",
                "author_type": "User",
                "author_association": "CONTRIBUTOR",
                "created_at": "2026-10-14T10:00:00Z",
                "updated_at": "2026-10-14T12:00:00Z",
                "closed_at": None,
                "comments_count": 0,
                "reactions_count": 0,
                "raw_labels": ["feature request"],
                "labels_count": 1,
            },
        ])

    def test_build_from_dataframe(self):
        master_df, audit = self.builder.build_from_dataframe(self.mock_raw_df)

        self.assertEqual(len(master_df), 2)
        self.assertEqual(list(master_df.columns), IssueDatasetBuilder.CANONICAL_COLUMNS)
        self.assertTrue(audit["validation"]["is_valid"])
        self.assertEqual(audit["validation"]["violations_count"], 0)

        # Row 0 assertions
        row0 = master_df.iloc[0]
        self.assertEqual(row0["repo_name"], "microsoft/vscode")
        self.assertEqual(row0["canonical_label"], "bug")
        self.assertEqual(row0["canonical_label_id"], 0)
        self.assertTrue(row0["is_classified_category"])
        self.assertFalse(row0["is_body_imputed"])
        self.assertEqual(row0["resolution_time_days"], 2.0)

        # Row 1 assertions
        row1 = master_df.iloc[1]
        self.assertEqual(row1["canonical_label"], "feature")
        self.assertEqual(row1["canonical_label_id"], 1)
        self.assertTrue(row1["is_body_imputed"])
        self.assertEqual(row1["clean_body"], "Support complex32 CPU")
        self.assertTrue(np.isnan(row1["resolution_time_days"]))

    def test_validate_master_dataset_valid(self):
        master_df, _ = self.builder.build_from_dataframe(self.mock_raw_df)
        val = IssueDatasetBuilder.validate_master_dataset(master_df)
        self.assertTrue(val["is_valid"])
        self.assertEqual(val["violations_count"], 0)

    def test_validate_master_dataset_violations(self):
        master_df, _ = self.builder.build_from_dataframe(self.mock_raw_df)
        bad_df = master_df.copy()

        # Inject violations:
        # 1. Null in mandatory column (clean_title)
        bad_df.loc[0, "clean_title"] = np.nan
        # 2. Target leakage: open issue has resolution_time_days
        bad_df.loc[1, "resolution_time_days"] = 5.0
        # 3. Empty clean_body
        bad_df.loc[0, "clean_body"] = "   "

        val = IssueDatasetBuilder.validate_master_dataset(bad_df)
        self.assertFalse(val["is_valid"])
        self.assertGreaterEqual(val["violations_count"], 3)

    def test_generate_data_dictionary(self):
        doc = IssueDatasetBuilder.generate_data_dictionary()
        self.assertIn("# OpenSourceLens — Canonical Master Issues Data Dictionary", doc)
        self.assertIn("resolution_time_days", doc)
        self.assertIn("canonical_label_id", doc)
        self.assertIn("is_body_imputed", doc)


if __name__ == "__main__":
    unittest.main()
