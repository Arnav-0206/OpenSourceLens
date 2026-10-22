"""Unit tests for DataFilter in ml.preprocessing.filter."""

import unittest
import pandas as pd

from ml.preprocessing.filter import DataFilter


class TestDataFilter(unittest.TestCase):
    """Test suite for DataFilter."""

    def setUp(self):
        self.filter = DataFilter(min_words=2, min_chars=5)
        self.sample_df = pd.DataFrame([
            {
                "issue_id": 1,
                "clean_title": "Fix scrolling bug in editor",
                "clean_body": "Steps to reproduce scrolling issue.",
                "combined_text": "Fix scrolling bug in editor Steps to reproduce scrolling issue.",
                "raw_labels": ["bug"],
                "repo_name": "microsoft/vscode",
            },
            {
                "issue_id": 2,  # Duplicate of issue 1
                "clean_title": "Fix scrolling bug in editor",
                "clean_body": "Steps to reproduce scrolling issue.",
                "combined_text": "Fix scrolling bug in editor Steps to reproduce scrolling issue.",
                "raw_labels": ["bug"],
                "repo_name": "microsoft/vscode",
            },
            {
                "issue_id": 3,  # Spam issue
                "clean_title": "Free crypto giveaway visit link",
                "clean_body": "Click here for free coins <URL>",
                "combined_text": "Free crypto giveaway visit link Click here for free coins <URL>",
                "raw_labels": ["SPAM", "triage"],
                "repo_name": "microsoft/vscode",
            },
            {
                "issue_id": 4,  # Degenerate title
                "clean_title": "hi",
                "clean_body": "hello",
                "combined_text": "hi hello",
                "raw_labels": ["question"],
                "repo_name": "pytorch/pytorch",
            },
            {
                "issue_id": 5,  # Valid issue
                "clean_title": "Support torch.div complex32 on CPU",
                "clean_body": "Feature request for complex numbers.",
                "combined_text": "Support torch.div complex32 on CPU Feature request for complex numbers.",
                "raw_labels": ["feature"],
                "repo_name": "pytorch/pytorch",
            },
        ])

    def test_filter_duplicates(self):
        retained, dropped = self.filter.filter_duplicates(self.sample_df)

        self.assertEqual(len(retained), 4)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped.iloc[0]["issue_id"], 2)
        self.assertEqual(dropped.iloc[0]["drop_reason"], "exact_duplicate")

    def test_filter_spam_labels(self):
        retained, dropped = self.filter.filter_spam_labels(self.sample_df)

        self.assertEqual(len(retained), 4)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped.iloc[0]["issue_id"], 3)
        self.assertEqual(dropped.iloc[0]["drop_reason"], "spam_label")

    def test_filter_degenerate_text(self):
        retained, dropped = self.filter.filter_degenerate_text(self.sample_df)

        self.assertEqual(len(retained), 4)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped.iloc[0]["issue_id"], 4)
        self.assertEqual(dropped.iloc[0]["drop_reason"], "degenerate_text")

    def test_filter_dataframe_pipeline(self):
        clean_df, audit = self.filter.filter_dataframe(self.sample_df)

        # 5 initial -> dropped: #3 (spam), #2 (dup), #4 (degenerate) -> 2 retained (#1, #5)
        self.assertEqual(audit["initial_records"], 5)
        self.assertEqual(audit["retained_records"], 2)
        self.assertEqual(audit["total_dropped"], 3)
        self.assertEqual(audit["retention_rate_percent"], 40.0)

        reasons = audit["dropped_by_reason"]
        self.assertEqual(reasons.get("spam_label"), 1)
        self.assertEqual(reasons.get("exact_duplicate"), 1)
        self.assertEqual(reasons.get("degenerate_text"), 1)

        retained_ids = set(clean_df["issue_id"])
        self.assertEqual(retained_ids, {1, 5})

    def test_format_audit_report(self):
        _, audit = self.filter.filter_dataframe(self.sample_df)
        report = DataFilter.format_audit_report(audit)

        self.assertIn("### Data Filtering & Noise Reduction Audit", report)
        self.assertIn("Initial Issue Count", report)
        self.assertIn("exact_duplicate", report)
        self.assertIn("spam_label", report)
        self.assertIn("degenerate_text", report)


if __name__ == "__main__":
    unittest.main()
