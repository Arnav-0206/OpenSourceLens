"""Unit tests for ProcessedDataStorage in ml.preprocessing.storage_processed."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.preprocessing.storage_processed import ProcessedDataStorage


class TestProcessedDataStorage(unittest.TestCase):
    """Test suite for ProcessedDataStorage."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = ProcessedDataStorage(processed_dir=self.temp_dir.name)

        self.mock_df = pd.DataFrame([
            {
                "issue_id": 1,
                "repo_name": "microsoft/vscode",
                "clean_title": "Fix scrolling bug",
                "clean_body": "Detailed steps <CODE_BLOCK>",
                "combined_text": "Fix scrolling bug Detailed steps <CODE_BLOCK>",
                "created_at": pd.Timestamp("2026-10-10 10:00:00", tz="UTC"),
                "closed_at": pd.Timestamp("2026-10-12 10:00:00", tz="UTC"),
                "resolution_time_days": 2.0,
                "raw_labels": ["bug", "editor"],
                "canonical_label": "bug",
                "canonical_label_id": 0,
                "is_classified_category": True,
                "is_unlabeled": False,
                "has_code_block": True,
            },
            {
                "issue_id": 2,
                "repo_name": "pytorch/pytorch",
                "clean_title": "Support complex32 on CPU",
                "clean_body": "Feature description <URL>",
                "combined_text": "Support complex32 on CPU Feature description <URL>",
                "created_at": pd.Timestamp("2026-10-14 10:00:00", tz="UTC"),
                "closed_at": pd.NaT,
                "resolution_time_days": float("nan"),
                "raw_labels": ["feature request"],
                "canonical_label": "feature",
                "canonical_label_id": 1,
                "is_classified_category": True,
                "is_unlabeled": False,
                "has_code_block": False,
            },
        ])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_master_dataset(self):
        files_info = self.storage.save_master_dataset(self.mock_df, base_name="test_issues")

        # Parquet assertions (if supported on platform)
        if "parquet" in files_info:
            p_path = Path(files_info["parquet"]["path"])
            self.assertTrue(p_path.exists())
            self.assertEqual(len(files_info["parquet"]["sha256"]), 64)
            self.assertGreater(files_info["parquet"]["size_bytes"], 0)

        # CSV assertions
        self.assertIn("csv", files_info)
        c_path = Path(files_info["csv"]["path"])
        self.assertTrue(c_path.exists())
        self.assertEqual(len(files_info["csv"]["sha256"]), 64)
        self.assertGreater(files_info["csv"]["size_bytes"], 0)

    def test_round_trip_parquet(self):
        files_info = self.storage.save_master_dataset(self.mock_df, base_name="issues_master")
        if "parquet" not in files_info:
            self.skipTest("Parquet engine unavailable or blocked by OS policy.")
        loaded = self.storage.load_master_dataset(format="parquet")

        self.assertEqual(len(loaded), len(self.mock_df))
        self.assertEqual(list(loaded.columns), list(self.mock_df.columns))

        # Check list preservation
        self.assertEqual(list(loaded.loc[0, "raw_labels"]), ["bug", "editor"])
        # Check UTC timestamp preservation
        self.assertEqual(str(loaded["created_at"].dt.tz), "UTC")
        # Check NaT preservation
        self.assertTrue(pd.isna(loaded.loc[1, "closed_at"]))

    def test_round_trip_csv(self):
        self.storage.save_master_dataset(self.mock_df, base_name="issues_master")
        loaded = self.storage.load_master_dataset(format="csv")

        self.assertEqual(len(loaded), len(self.mock_df))
        self.assertEqual(list(loaded.columns), list(self.mock_df.columns))

        # Check list string parsed back to python list
        self.assertEqual(loaded.loc[0, "raw_labels"], ["bug", "editor"])
        self.assertEqual(loaded.loc[1, "raw_labels"], ["feature request"])

    def test_save_label_mappings(self):
        path = self.storage.save_label_mappings()
        self.assertTrue(path.exists())

        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("label_to_id", data)
        self.assertEqual(data["label_to_id"]["bug"], 0)
        self.assertIn("core_categories", data)

    def test_save_manifest(self):
        files_info = self.storage.save_master_dataset(self.mock_df, base_name="issues_master")
        manifest_path = self.storage.save_manifest(self.mock_df, files_info)
        self.assertTrue(manifest_path.exists())

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["total_records"], 2)
        self.assertEqual(data["total_features"], len(self.mock_df.columns))
        self.assertIn("csv", data["files"])
        if "parquet" in files_info:
            self.assertIn("parquet", data["files"])
        self.assertIn("distributions", data)

    def test_save_all(self):
        summary = self.storage.save_all(self.mock_df, base_name="issues_master")
        self.assertEqual(summary["total_records"], 2)
        self.assertTrue(Path(summary["label_mappings_path"]).exists())
        self.assertTrue(Path(summary["manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
