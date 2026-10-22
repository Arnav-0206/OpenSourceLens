"""Unit tests for MissingValueHandler in ml.preprocessing.missing_handler."""

import unittest
import numpy as np
import pandas as pd

from ml.preprocessing.missing_handler import MissingValueHandler


class TestMissingValueHandler(unittest.TestCase):
    """Test suite for MissingValueHandler."""

    def setUp(self):
        self.handler = MissingValueHandler()
        self.sample_df = pd.DataFrame([
            {
                "issue_id": 1,
                "clean_title": "Fix scrolling bug",
                "clean_body": "Detailed steps to reproduce bug.",
                "combined_text": "Fix scrolling bug Detailed steps to reproduce bug.",
                "clean_body_word_count": 5,
                "state": "closed",
                "state_reason": "completed",
                "raw_labels": ["bug"],
                "labels_count": 1,
                "resolution_time_days": 2.5,
            },
            {
                "issue_id": 2,
                "clean_title": "Empty body feature request",
                "clean_body": "",  # Empty body
                "combined_text": "Empty body feature request",
                "clean_body_word_count": 0,
                "state": "open",
                "state_reason": None,  # Open without reason
                "raw_labels": [],
                "labels_count": 0,
                "resolution_time_days": np.nan,  # Open issue target must stay NaN
            },
            {
                "issue_id": 3,
                "clean_title": "Legacy closed issue",
                "clean_body": None,  # None body
                "combined_text": "Legacy closed issue",
                "clean_body_word_count": 0,
                "state": "closed",
                "state_reason": None,  # Closed without reason
                "raw_labels": None,  # None labels
                "labels_count": 0,
                "resolution_time_days": 1.2,
            },
        ])

    def test_handle_body_missingness(self):
        res = self.handler.handle_body_missingness(self.sample_df)

        # Row 0: was not empty
        self.assertFalse(res.loc[0, "is_body_imputed"])
        self.assertEqual(res.loc[0, "clean_body"], "Detailed steps to reproduce bug.")

        # Row 1: was empty string -> imputed with title
        self.assertTrue(res.loc[1, "is_body_imputed"])
        self.assertEqual(res.loc[1, "clean_body"], "Empty body feature request")
        self.assertEqual(res.loc[1, "clean_body_word_count"], 4)
        self.assertIn("Empty body feature request", res.loc[1, "combined_text"])

        # Row 2: was None -> imputed with title
        self.assertTrue(res.loc[2, "is_body_imputed"])
        self.assertEqual(res.loc[2, "clean_body"], "Legacy closed issue")
        self.assertEqual(res.loc[2, "clean_body_word_count"], 3)

    def test_handle_state_reason(self):
        res = self.handler.handle_state_reason(self.sample_df)

        # Row 0: was completed -> preserved
        self.assertEqual(res.loc[0, "state_reason"], "completed")

        # Row 1: open with None -> unresolved
        self.assertEqual(res.loc[1, "state_reason"], "unresolved")

        # Row 2: closed with None -> closed_unspecified
        self.assertEqual(res.loc[2, "state_reason"], "closed_unspecified")

    def test_handle_labels(self):
        res = self.handler.handle_labels(self.sample_df)

        # Row 0: has labels
        self.assertEqual(res.loc[0, "raw_labels"], ["bug"])
        self.assertFalse(res.loc[0, "is_unlabeled"])

        # Row 1: empty list
        self.assertEqual(res.loc[1, "raw_labels"], [])
        self.assertTrue(res.loc[1, "is_unlabeled"])

        # Row 2: was None -> converted to []
        self.assertEqual(res.loc[2, "raw_labels"], [])
        self.assertTrue(res.loc[2, "is_unlabeled"])

    def test_clean_dataframe_integration(self):
        clean_df = self.handler.clean_dataframe(self.sample_df)

        # Zero empty bodies remain
        self.assertEqual((clean_df["clean_body"].str.strip() == "").sum(), 0)
        # Zero null state reasons remain
        self.assertEqual(clean_df["state_reason"].isna().sum(), 0)
        # Target protection: open issue resolution_time_days MUST remain NaN
        self.assertTrue(np.isnan(clean_df.loc[1, "resolution_time_days"]))

    def test_audit_missingness(self):
        audit = MissingValueHandler.audit_missingness(self.sample_df)
        self.assertEqual(audit["total_records"], 3)
        self.assertIn("clean_body", audit["columns"])
        self.assertEqual(audit["columns"]["clean_body"]["null_count"], 1)

    def test_summarize_imputation(self):
        clean_df = self.handler.clean_dataframe(self.sample_df)
        summary = MissingValueHandler.summarize_imputation(self.sample_df, clean_df)

        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["imputed_bodies_count"], 2)
        self.assertEqual(summary["after_empty_bodies"], 0)
        self.assertEqual(summary["after_null_state_reasons"], 0)
        self.assertEqual(summary["unlabeled_issues_count"], 2)


if __name__ == "__main__":
    unittest.main()
