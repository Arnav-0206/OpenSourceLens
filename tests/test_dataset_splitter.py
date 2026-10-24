"""Unit tests for ml.classification.split (DatasetSplitter)."""

import unittest
import tempfile
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ml.classification.split import DatasetSplitter
from ml.classification.target_definition import ClassificationTargetDefiner


class TestDatasetSplitter(unittest.TestCase):
    """Test suite for DatasetSplitter class."""

    def setUp(self):
        """Construct synthetic dataset with known class distribution and timestamps."""
        # 40 issues: 20 bug (50%), 12 other (30%), 8 feature (20%)
        # across two repositories and timestamps spanning 40 days
        np.random.seed(42)
        n = 40
        labels = ["bug"] * 20 + ["other"] * 12 + ["feature"] * 8
        repos = ["repo_a"] * 20 + ["repo_b"] * 20
        dates = pd.date_range("2025-01-01", periods=n, freq="D").astype(str)

        self.mock_df = pd.DataFrame({
            "issue_id": list(range(100, 100 + n)),
            "repo_name": repos,
            "created_at": dates,
            "clean_title": [f"Issue title {i}" for i in range(n)],
            "clean_body": [f"Issue body description {i}" for i in range(n)],
            "combined_text": [f"Issue title {i} Issue body description {i}" for i in range(n)],
            "clean_title_word_count": [3] * n,
            "clean_body_word_count": [4] * n,
            "canonical_label": labels,
            "target_label": labels,
            "target_id": [{"bug": 0, "feature": 1, "other": 2}[l] for l in labels],
        })

    def test_init_with_dataframe(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label")
        self.assertEqual(len(splitter.df), 40)
        self.assertEqual(splitter.target_col, "target_label")
        self.assertEqual(splitter.source_path, "DataFrame in-memory")

    def test_init_with_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "split_test.csv"
            self.mock_df.to_csv(csv_path, index=False)
            splitter = DatasetSplitter(csv_path)
            self.assertEqual(len(splitter.df), 40)
            self.assertEqual(splitter.source_path, str(csv_path))

    def test_init_with_target_definer(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        splitter = DatasetSplitter(definer)
        self.assertEqual(len(splitter.df), 40)
        self.assertIn("target_mode", splitter.metadata)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DatasetSplitter("non_existent_dataset.csv")

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            DatasetSplitter(pd.DataFrame())

    def test_init_missing_target_col(self):
        bad_df = pd.DataFrame({"issue_id": [1], "clean_title": ["hello"]})
        with self.assertRaises(ValueError):
            DatasetSplitter(bad_df, target_col="target_label")

    def test_split_train_test_proportions(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label", random_state=42)
        train_df, test_df = splitter.split_train_test(test_size=0.20, stratify=True)

        self.assertEqual(len(train_df), 32)
        self.assertEqual(len(test_df), 8)
        self.assertEqual(len(train_df) + len(test_df), 40)

        # Check stratification
        # Population: bug=50%, other=30%, feature=20%
        # Test (8 issues): bug=4 (50%), other=2 or 3 (~30%), feature=1 or 2 (~20%)
        test_counts = test_df["target_label"].value_counts().to_dict()
        self.assertEqual(test_counts["bug"], 4)
        self.assertEqual(test_counts["other"], 2)
        self.assertEqual(test_counts["feature"], 2)

    def test_split_train_val_test(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label", random_state=42)
        train_df, val_df, test_df = splitter.split_train_val_test(val_size=0.15, test_size=0.15, stratify=True)

        # Total 40: 15% is 6 issues each
        self.assertEqual(len(test_df), 6)
        self.assertEqual(len(val_df), 6)
        self.assertEqual(len(train_df), 28)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), 40)

        # Zero overlap between all 3 sets
        train_ids = set(train_df["issue_id"])
        val_ids = set(val_df["issue_id"])
        test_ids = set(test_df["issue_id"])
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_split_temporal(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label")
        train_df, test_df = splitter.split_temporal(date_col="created_at", test_size=0.20)

        self.assertEqual(len(train_df), 32)
        self.assertEqual(len(test_df), 8)

        # Temporal constraint: max train timestamp <= min test timestamp
        max_train_ts = pd.to_datetime(train_df["created_at"], utc=True).max()
        min_test_ts = pd.to_datetime(test_df["created_at"], utc=True).min()
        self.assertLessEqual(max_train_ts, min_test_ts)

    def test_split_temporal_missing_col(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label")
        with self.assertRaises(KeyError):
            splitter.split_temporal(date_col="non_existent_date")

    def test_invalid_split_sizes(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label")
        with self.assertRaises(ValueError):
            splitter.split_train_test(test_size=1.5)
        with self.assertRaises(ValueError):
            splitter.split_train_test(test_size=-0.1)
        with self.assertRaises(ValueError):
            splitter.split_train_val_test(val_size=0.6, test_size=0.5)

    def test_stratified_kfold(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label", random_state=42)
        folds = list(splitter.get_stratified_kfold(n_splits=4))
        self.assertEqual(len(folds), 4)

        seen_val_indices = set()
        for train_idx, val_idx in folds:
            # 40 issues / 4 folds = 10 validation per fold, 30 train per fold
            self.assertEqual(len(val_idx), 10)
            self.assertEqual(len(train_idx), 30)
            # Train and val in each fold must be disjoint
            self.assertTrue(set(train_idx).isdisjoint(set(val_idx)))
            seen_val_indices.update(val_idx)

        # All 40 samples must be validated exactly once
        self.assertEqual(len(seen_val_indices), 40)

    def test_stratified_kfold_minority_error(self):
        # Create dataset where minority class has only 2 samples
        tiny_df = pd.DataFrame({
            "target_label": ["bug"] * 10 + ["rare"] * 2,
            "clean_title": ["t"] * 12,
        })
        splitter = DatasetSplitter(tiny_df, target_col="target_label")
        with self.assertRaises(ValueError):
            # 5 splits is invalid when minority has 2 samples
            list(splitter.get_stratified_kfold(n_splits=5))

    def test_diagnostics(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label", random_state=42)
        train_df, test_df = splitter.split_train_test(test_size=0.20, stratify=True)
        diag = splitter.get_split_diagnostics(train_df, test_df)

        self.assertEqual(diag["total_samples"], 40)
        self.assertTrue(diag["zero_overlap_verified"])
        self.assertEqual(diag["overlap_count"], 0)
        self.assertIn("bug", diag["class_distribution"])
        self.assertIn("repo_a", diag["repository_distribution"])
        self.assertIn("train", diag["temporal_bounds"])
        self.assertIn("train", diag["text_length_stats"])

    def test_save_and_load_splits(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label", random_state=42)
        train_df, test_df = splitter.split_train_test(test_size=0.20, stratify=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = splitter.save_splits(train_df, test_df, output_dir=tmpdir)
            self.assertTrue(Path(saved["train"]).exists())
            self.assertTrue(Path(saved["test"]).exists())
            self.assertTrue(Path(saved["manifest"]).exists())

            # Load back
            loaded = DatasetSplitter.load_splits(tmpdir)
            self.assertEqual(len(loaded["train"]), 32)
            self.assertEqual(len(loaded["test"]), 8)
            # Column and content match
            pd.testing.assert_frame_equal(loaded["train"], train_df)
            pd.testing.assert_frame_equal(loaded["test"], test_df)

    def test_load_splits_tamper_detection(self):
        splitter = DatasetSplitter(self.mock_df, target_col="target_label", random_state=42)
        train_df, test_df = splitter.split_train_test(test_size=0.20, stratify=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            splitter.save_splits(train_df, test_df, output_dir=tmpdir)
            # Tamper with train.csv
            train_file = Path(tmpdir) / "train.csv"
            with open(train_file, "a", encoding="utf-8") as f:
                f.write("corrupted,row,tampered\n")

            with self.assertRaises(ValueError):
                DatasetSplitter.load_splits(tmpdir, verify_integrity=True)

    def test_live_master_dataset_integration(self):
        csv_path = Path("data/processed/issues_master.csv")
        if not csv_path.exists():
            self.skipTest("issues_master.csv not found.")

        target_definer = ClassificationTargetDefiner(csv_path, target_mode="core_3class")
        splitter = DatasetSplitter(target_definer, target_col="target_label", random_state=42)

        # 1. Test 80/20 Stratified Split
        train_df, test_df = splitter.split_train_test(test_size=0.20, stratify=True)
        self.assertEqual(len(train_df), 1537)
        self.assertEqual(len(test_df), 385)
        self.assertEqual(len(train_df) + len(test_df), 1922)

        # Class counts in train: 640 bug (41.6%), 625 other (40.7%), 272 feature (17.7%)
        train_counts = train_df["target_label"].value_counts().to_dict()
        self.assertEqual(train_counts["bug"], 640)
        self.assertEqual(train_counts["other"], 625)
        self.assertEqual(train_counts["feature"], 272)

        # Class counts in test: 160 bug (41.6%), 157 other (40.8%), 68 feature (17.7%)
        test_counts = test_df["target_label"].value_counts().to_dict()
        self.assertEqual(test_counts["bug"], 160)
        self.assertEqual(test_counts["other"], 157)
        self.assertEqual(test_counts["feature"], 68)

        # 2. Check Zero Leakage / Zero Overlap
        train_ids = set(train_df["issue_id"])
        test_ids = set(test_df["issue_id"])
        self.assertTrue(train_ids.isdisjoint(test_ids))

        for leak in ClassificationTargetDefiner.LEAKAGE_COLUMNS:
            self.assertNotIn(leak, train_df.columns)
            self.assertNotIn(leak, test_df.columns)

        # 3. Stratified 5-Fold Cross Validation on Train Set
        folds = list(splitter.get_stratified_kfold(train_df, n_splits=5))
        self.assertEqual(len(folds), 5)
        for t_idx, v_idx in folds:
            self.assertEqual(len(t_idx) + len(v_idx), 1537)

        # 4. Diagnostics Generation
        diag = splitter.get_split_diagnostics(train_df, test_df)
        self.assertTrue(diag["zero_overlap_verified"])
        self.assertEqual(diag["overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
