"""Unit tests for TimestampCleaner in ml.preprocessing.timestamp_cleaner."""

import unittest
import numpy as np
import pandas as pd

from ml.preprocessing.timestamp_cleaner import TimestampCleaner


class TestTimestampCleaner(unittest.TestCase):
    """Test suite for TimestampCleaner."""

    def setUp(self):
        self.cleaner = TimestampCleaner(reference_date="2026-10-19T12:00:00Z")
        self.sample_df = pd.DataFrame([
            {
                "issue_id": 1,
                "state": "closed",
                "created_at": "2026-10-10T10:00:00Z",  # Saturday
                "updated_at": "2026-10-12T10:00:00Z",
                "closed_at": "2026-10-12T10:00:00Z",  # Exactly 2.0 days later (48 hours)
            },
            {
                "issue_id": 2,
                "state": "open",
                "created_at": "2026-10-14T14:30:00Z",  # Wednesday
                "updated_at": "2026-10-15T09:00:00Z",
                "closed_at": None,
            },
        ])

    def test_parse_timestamps(self):
        parsed = self.cleaner.parse_timestamps(self.sample_df)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(parsed["created_at"]))
        self.assertEqual(str(parsed["created_at"].dt.tz), "UTC")
        self.assertTrue(pd.isna(parsed.loc[1, "closed_at"]))
        self.assertTrue(parsed.loc[0, "closed_at"].tzinfo is not None)

    def test_validate_chronological_invariants_valid(self):
        parsed = self.cleaner.parse_timestamps(self.sample_df)
        invariants = self.cleaner.validate_chronological_invariants(parsed)
        self.assertTrue(invariants["is_valid"])
        self.assertEqual(invariants["created_gt_updated_count"], 0)
        self.assertEqual(invariants["created_gt_closed_count"], 0)
        self.assertEqual(invariants["open_with_closed_at_count"], 0)
        self.assertEqual(invariants["closed_without_closed_at_count"], 0)

    def test_validate_chronological_invariants_violations(self):
        bad_df = pd.DataFrame([
            {
                "issue_id": 10,
                "state": "closed",
                "created_at": "2026-10-15T10:00:00Z",
                "updated_at": "2026-10-10T10:00:00Z",  # created > updated
                "closed_at": "2026-10-12T10:00:00Z",   # created > closed
            },
            {
                "issue_id": 11,
                "state": "open",
                "created_at": "2026-10-01T10:00:00Z",
                "updated_at": "2026-10-02T10:00:00Z",
                "closed_at": "2026-10-03T10:00:00Z",   # open with closed_at
            },
            {
                "issue_id": 12,
                "state": "closed",
                "created_at": "2026-10-01T10:00:00Z",
                "updated_at": "2026-10-02T10:00:00Z",
                "closed_at": None,                     # closed without closed_at
            },
        ])
        parsed_bad = self.cleaner.parse_timestamps(bad_df)
        invariants = self.cleaner.validate_chronological_invariants(parsed_bad)

        self.assertFalse(invariants["is_valid"])
        self.assertEqual(invariants["created_gt_updated_count"], 1)
        self.assertEqual(invariants["created_gt_closed_count"], 1)
        self.assertEqual(invariants["open_with_closed_at_count"], 1)
        self.assertEqual(invariants["closed_without_closed_at_count"], 1)

    def test_compute_resolution_time(self):
        parsed = self.cleaner.parse_timestamps(self.sample_df)
        res_df = self.cleaner.compute_resolution_time(parsed)

        # Row 0: closed issue, exactly 2.0 days (48.0 hours)
        self.assertEqual(res_df.loc[0, "resolution_time_days"], 2.0)
        self.assertEqual(res_df.loc[0, "resolution_time_hours"], 48.0)

        # Row 1: open issue, must be NaN
        self.assertTrue(np.isnan(res_df.loc[1, "resolution_time_days"]))
        self.assertTrue(np.isnan(res_df.loc[1, "resolution_time_hours"]))

    def test_compute_issue_age(self):
        parsed = self.cleaner.parse_timestamps(self.sample_df)
        # reference: 2026-10-19T12:00:00Z
        # Row 0 created: 2026-10-10T10:00:00Z -> 9.0833 days (9 days 2 hours)
        age_df = self.cleaner.compute_issue_age(parsed)
        self.assertAlmostEqual(age_df.loc[0, "issue_age_days"], 9.08, places=1)
        # Row 1 created: 2026-10-14T14:30:00Z -> ~4.90 days
        self.assertAlmostEqual(age_df.loc[1, "issue_age_days"], 4.90, places=1)

    def test_extract_temporal_features(self):
        parsed = self.cleaner.parse_timestamps(self.sample_df)
        feat_df = self.cleaner.extract_temporal_features(parsed)

        # Row 0: 2026-10-10T10:00:00Z (Saturday)
        self.assertEqual(feat_df.loc[0, "created_year"], 2026)
        self.assertEqual(feat_df.loc[0, "created_month"], 10)
        self.assertEqual(feat_df.loc[0, "created_day"], 10)
        self.assertEqual(feat_df.loc[0, "created_day_of_week"], 5)  # Saturday
        self.assertEqual(feat_df.loc[0, "created_hour"], 10)
        self.assertTrue(feat_df.loc[0, "is_weekend"])

        # Row 1: 2026-10-14T14:30:00Z (Wednesday)
        self.assertEqual(feat_df.loc[1, "created_day_of_week"], 2)  # Wednesday
        self.assertEqual(feat_df.loc[1, "created_hour"], 14)
        self.assertFalse(feat_df.loc[1, "is_weekend"])

    def test_clean_dataframe_end_to_end(self):
        clean_df = self.cleaner.clean_dataframe(self.sample_df)
        expected_cols = [
            "created_at",
            "updated_at",
            "closed_at",
            "resolution_time_days",
            "resolution_time_hours",
            "issue_age_days",
            "created_year",
            "created_month",
            "created_day",
            "created_day_of_week",
            "created_hour",
            "is_weekend",
        ]
        for col in expected_cols:
            self.assertIn(col, clean_df.columns)

    def test_summarize_timestamps(self):
        clean_df = self.cleaner.clean_dataframe(self.sample_df)
        summary = TimestampCleaner.summarize_timestamps(clean_df)

        self.assertEqual(summary["total_issues"], 2)
        self.assertEqual(summary["closed_issues_count"], 1)
        self.assertEqual(summary["open_issues_count"], 1)
        self.assertEqual(summary["resolution_days"]["median"], 2.0)
        self.assertEqual(summary["weekend_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
