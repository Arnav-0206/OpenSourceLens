"""
Unit tests for OpenSourceLens Phase 8, Step 8.1: Resolution Time Definition.

Validates:
- Exact continuous mathematical duration calculations (seconds, hours, days)
- Log-transformed hour normalization
- Survival-analysis right-censoring for open issues
- Sub-minute and immediate closure detection (< 300 seconds)
- Operational speed tier boundary binning
- Resolution type disaggregation (completed vs rejected vs duplicate vs unresolved)
- Strict chronological invariant verification
- Dataset enrichment and metadata manifest serialization
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.resolution_analysis.definition import (
    DEFAULT_IMMEDIATE_THRESHOLD_SECONDS,
    DEFAULT_REFERENCE_DATE,
    ResolutionTimeDefiner,
    SPEED_TIER_BINS,
    SPEED_TIER_LABELS,
)


class TestResolutionTimeDefinition(unittest.TestCase):
    """Test suite for ResolutionTimeDefiner module and temporal formalization."""

    def setUp(self):
        """Set up synthetic test fixtures representing standard, open, immediate, and edge cases."""
        self.ref_date = "2026-10-19T12:00:00Z"
        self.definer = ResolutionTimeDefiner(reference_date=self.ref_date, immediate_threshold_seconds=300)

        # Synthetic DataFrame covering all edge cases
        self.sample_df = pd.DataFrame([
            {
                # Standard closed bug: exactly 2 days (48.0 hours = 172,800 seconds)
                "issue_id": 1,
                "issue_number": 101,
                "repo_name": "microsoft/vscode",
                "canonical_label": "bug",
                "state": "closed",
                "state_reason": "completed",
                "created_at": "2026-10-10T10:00:00Z",
                "closed_at": "2026-10-12T10:00:00Z",
            },
            {
                # Immediate bot/canary closure: 30 seconds
                "issue_id": 2,
                "issue_number": 102,
                "repo_name": "microsoft/vscode",
                "canonical_label": "other",
                "state": "closed",
                "state_reason": "not_planned",
                "created_at": "2026-10-14T08:00:00Z",
                "closed_at": "2026-10-14T08:00:30Z",
            },
            {
                # Fast closure (< 1 day): 12 hours = 43,200 seconds
                "issue_id": 3,
                "issue_number": 103,
                "repo_name": "pytorch/pytorch",
                "canonical_label": "feature",
                "state": "closed",
                "state_reason": "completed",
                "created_at": "2026-10-15T00:00:00Z",
                "closed_at": "2026-10-15T12:00:00Z",
            },
            {
                # Extended closure (10 days = 240 hours)
                "issue_id": 4,
                "issue_number": 104,
                "repo_name": "kubernetes/kubernetes",
                "canonical_label": "bug",
                "state": "closed",
                "state_reason": "completed",
                "created_at": "2026-10-01T00:00:00Z",
                "closed_at": "2026-10-11T00:00:00Z",
            },
            {
                # Backlog closure (40 days = 960 hours)
                "issue_id": 5,
                "issue_number": 105,
                "repo_name": "huggingface/transformers",
                "canonical_label": "feature",
                "state": "closed",
                "state_reason": "completed",
                "created_at": "2026-08-01T00:00:00Z",
                "closed_at": "2026-09-10T00:00:00Z",
            },
            {
                # Duplicate closure
                "issue_id": 6,
                "issue_number": 106,
                "repo_name": "pytorch/pytorch",
                "canonical_label": "bug",
                "state": "closed",
                "state_reason": "duplicate",
                "created_at": "2026-10-10T00:00:00Z",
                "closed_at": "2026-10-11T00:00:00Z",
            },
            {
                # Open issue: created 5 days before snapshot (no closed_at)
                "issue_id": 7,
                "issue_number": 107,
                "repo_name": "microsoft/vscode",
                "canonical_label": "feature",
                "state": "open",
                "state_reason": None,
                "created_at": "2026-10-14T12:00:00Z",
                "closed_at": None,
            },
        ])

    def test_initialization_and_reference_date(self):
        """Verify initialization with default and custom reference dates."""
        default_def = ResolutionTimeDefiner()
        self.assertEqual(default_def.reference_date, pd.Timestamp(DEFAULT_REFERENCE_DATE, tz="UTC"))
        self.assertEqual(default_def.immediate_threshold_seconds, DEFAULT_IMMEDIATE_THRESHOLD_SECONDS)

        custom_ts = pd.Timestamp("2026-11-01T00:00:00Z")
        custom_def = ResolutionTimeDefiner(reference_date=custom_ts, immediate_threshold_seconds=120)
        self.assertEqual(custom_def.reference_date, custom_ts)
        self.assertEqual(custom_def.immediate_threshold_seconds, 120.0)

        with self.assertRaises(TypeError):
            ResolutionTimeDefiner(reference_date=12345)  # Invalid type

    def test_compute_resolution_metrics_exact_math(self):
        """Verify mathematical correctness of duration formulas (seconds, hours, days)."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)

        # Issue 1: exactly 2 days (48.0 hours = 172,800 seconds)
        row1 = enriched[enriched["issue_id"] == 1].iloc[0]
        self.assertEqual(row1["resolution_time_seconds"], 172800.0)
        self.assertEqual(row1["resolution_time_hours"], 48.0)
        self.assertEqual(row1["resolution_time_days"], 2.0)
        self.assertAlmostEqual(row1["resolution_time_log_hours"], np.log(49.0), places=4)
        self.assertFalse(row1["is_immediate_closure"])
        self.assertEqual(row1["speed_tier"], "1-7 Days")
        self.assertEqual(row1["resolution_type"], "completed")

    def test_open_issue_right_censoring(self):
        """Verify right-censorship invariants on open issues (no leakage)."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)

        row_open = enriched[enriched["issue_id"] == 7].iloc[0]
        self.assertTrue(row_open["is_censored"])
        self.assertEqual(row_open["event_observed"], 0)
        self.assertTrue(np.isnan(row_open["resolution_time_seconds"]))
        self.assertTrue(np.isnan(row_open["resolution_time_hours"]))
        self.assertTrue(np.isnan(row_open["resolution_time_days"]))
        self.assertTrue(np.isnan(row_open["resolution_time_log_hours"]))
        self.assertEqual(row_open["speed_tier"], "Unresolved")
        self.assertEqual(row_open["resolution_type"], "unresolved")

        # Duration at risk should be exactly 5.0 days (from 2026-10-14T12:00:00Z to 2026-10-19T12:00:00Z)
        self.assertAlmostEqual(row_open["duration_at_risk_days"], 5.0, places=4)

    def test_immediate_closures_flagging(self):
        """Verify sub-minute closures (< 300 seconds) are flagged as immediate."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)

        # Issue 2 closed in 30 seconds
        row2 = enriched[enriched["issue_id"] == 2].iloc[0]
        self.assertEqual(row2["resolution_time_seconds"], 30.0)
        self.assertTrue(row2["is_immediate_closure"])
        self.assertEqual(row2["resolution_type"], "rejected")
        self.assertEqual(row2["speed_tier"], "< 1 Day")

    def test_speed_tier_boundaries(self):
        """Verify operational speed tiers assign correct categories across boundaries."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)

        # Issue 3: 0.5 days -> "< 1 Day"
        row3 = enriched[enriched["issue_id"] == 3].iloc[0]
        self.assertEqual(row3["speed_tier"], "< 1 Day")

        # Issue 1: 2.0 days -> "1-7 Days"
        row1 = enriched[enriched["issue_id"] == 1].iloc[0]
        self.assertEqual(row1["speed_tier"], "1-7 Days")

        # Issue 4: 10.0 days -> "7-30 Days"
        row4 = enriched[enriched["issue_id"] == 4].iloc[0]
        self.assertEqual(row4["speed_tier"], "7-30 Days")

        # Issue 5: 40.0 days -> "30+ Days"
        row5 = enriched[enriched["issue_id"] == 5].iloc[0]
        self.assertEqual(row5["speed_tier"], "30+ Days")

    def test_resolution_type_disaggregation(self):
        """Verify state_reason maps cleanly to completed, rejected, duplicate, and unresolved."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)

        res_types = enriched.set_index("issue_id")["resolution_type"].to_dict()
        self.assertEqual(res_types[1], "completed")
        self.assertEqual(res_types[2], "rejected")
        self.assertEqual(res_types[6], "duplicate")
        self.assertEqual(res_types[7], "unresolved")

    def test_validate_invariants_clean_dataset(self):
        """Verify invariant check returns all_invariants_pass=True on clean data."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)
        invariants = self.definer.validate_invariants(enriched)

        self.assertTrue(invariants["all_invariants_pass"])
        self.assertEqual(invariants["negative_durations_count"], 0)
        self.assertEqual(invariants["open_with_closed_at_count"], 0)
        self.assertEqual(invariants["closed_without_closed_at_count"], 0)
        self.assertEqual(invariants["censored_mismatch_count"], 0)
        self.assertEqual(invariants["duration_censored_mismatch_count"], 0)

    def test_validate_invariants_anomalous_dataset(self):
        """Verify invariant validator detects chronological contradictions and anomalies."""
        anomalous_df = pd.DataFrame([
            {
                # Negative duration (closed before created)
                "issue_id": 99,
                "state": "closed",
                "created_at": "2026-10-15T12:00:00Z",
                "closed_at": "2026-10-10T12:00:00Z",
            },
            {
                # Open issue with non-null closed_at
                "issue_id": 98,
                "state": "open",
                "created_at": "2026-10-10T12:00:00Z",
                "closed_at": "2026-10-11T12:00:00Z",
            },
            {
                # Closed issue with null closed_at
                "issue_id": 97,
                "state": "closed",
                "created_at": "2026-10-10T12:00:00Z",
                "closed_at": None,
            },
        ])

        invariants = self.definer.validate_invariants(anomalous_df)
        self.assertFalse(invariants["all_invariants_pass"])
        self.assertEqual(invariants["negative_durations_count"], 1)
        self.assertEqual(invariants["open_with_closed_at_count"], 1)
        self.assertEqual(invariants["closed_without_closed_at_count"], 1)

    def test_summarize_metrics(self):
        """Verify summarize dictionary produces valid statistics and segment breakdowns."""
        enriched = self.definer.compute_resolution_metrics(self.sample_df)
        summary = self.definer.summarize(enriched)

        overview = summary["dataset_overview"]
        self.assertEqual(overview["total_issues"], 7)
        self.assertEqual(overview["closed_issues"], 6)
        self.assertEqual(overview["open_issues"], 1)

        self.assertIn("mean", summary["resolution_statistics_hours"])
        self.assertIn("median", summary["resolution_statistics_hours"])
        self.assertIn("quantiles", summary["resolution_statistics_hours"])
        self.assertIn("speed_tier_distribution", summary)
        self.assertIn("resolution_type_distribution", summary)
        self.assertIn("immediate_closures", summary)
        self.assertEqual(summary["immediate_closures"]["count"], 1)

    def test_save_enriched_dataset(self):
        """Verify serialization exports valid Parquet, CSV, and metadata manifest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_parquet = Path(tmp_dir) / "test_resolution.parquet"
            out_manifest = Path(tmp_dir) / "test_manifest.json"

            p_out, p_man = self.definer.save_enriched_dataset(
                self.sample_df,
                output_parquet=out_parquet,
                manifest_path=out_manifest,
            )

            self.assertTrue(p_out.exists())
            self.assertTrue(p_out.with_suffix(".csv").exists())
            self.assertTrue(p_man.exists())

            # Verify manifest JSON load
            with open(p_man, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            self.assertEqual(manifest_data["phase"], "Phase 8: Resolution Analysis & Prediction")
            self.assertEqual(manifest_data["step"], "Step 8.1: Define Resolution Time")
            self.assertTrue(manifest_data["invariant_checks"]["all_invariants_pass"])
            self.assertEqual(manifest_data["summary_metrics"]["dataset_overview"]["total_issues"], 7)


if __name__ == "__main__":
    unittest.main()
