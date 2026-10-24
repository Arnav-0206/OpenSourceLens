"""Unit Tests for Transformer Dataset Preparation (Step 6.2)."""

from pathlib import Path
import tempfile
import unittest
import json

import pandas as pd
import numpy as np
import torch

from ml.deep_learning.dataset_preparation import TransformerDatasetBuilder, IssueDataset


class TestTransformerDatasetPreparation(unittest.TestCase):
    """Test suite for IssueDataset and TransformerDatasetBuilder."""

    def setUp(self):
        """Set up synthetic datasets for testing."""
        # Create a mock training set with 30 items
        labels = ["bug"] * 12 + ["other"] * 12 + ["feature"] * 6
        self.mock_train_df = pd.DataFrame(
            {
                "issue_id": [f"train_{i}" for i in range(30)],
                "repo_name": ["repo/test"] * 30,
                "title": [f"Issue title {i}" for i in range(30)],
                "body": [f"Issue body description for issue {i} with error details." for i in range(30)],
                "canonical_label": labels,
            }
        )

        # Create a mock test set with 9 items
        test_labels = ["bug"] * 4 + ["other"] * 3 + ["feature"] * 2
        self.mock_test_df = pd.DataFrame(
            {
                "issue_id": [f"test_{i}" for i in range(9)],
                "repo_name": ["repo/test"] * 9,
                "title": [f"Test title {i}" for i in range(9)],
                "body": [f"Test description {i} with stack trace." for i in range(9)],
                "canonical_label": test_labels,
            }
        )

    def test_pytorch_issue_dataset(self):
        """Verify IssueDataset behavior, length, and indexing."""
        texts = ["Text one", "Text two", "Text three"]
        labels = [0, 1, 2]
        ids = ["id_1", "id_2", "id_3"]

        dataset = IssueDataset(texts, labels, ids)
        self.assertEqual(len(dataset), 3)

        sample = dataset[1]
        self.assertEqual(sample["text"], "Text two")
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["issue_id"], "id_2")
        self.assertIsInstance(sample["label"], int)

        # Mismatch length raises ValueError
        with self.assertRaises(ValueError):
            IssueDataset(["Text"], [0, 1])

    def test_format_text(self):
        """Verify text formatting with delimiters and preprocessing."""
        builder = TransformerDatasetBuilder(use_enhanced_preprocessing=True)
        res = builder.format_text(
            title="Crash in auth.py with #123",
            body="Fatal error at line 50.",
        )
        self.assertTrue(res.startswith("Title:"))
        self.assertIn("| Description:", res)
        self.assertIn("file_path", res)  # Normalized domain token

        # Only title
        only_t = builder.format_text("Simple title", "")
        self.assertEqual(only_t, "Title: simple title")

    def test_prepare_dataframe(self):
        """Verify label mapping and schema transformation."""
        builder = TransformerDatasetBuilder()
        prepared = builder.prepare_dataframe(self.mock_train_df)

        self.assertEqual(len(prepared), 30)
        self.assertIn("text", prepared.columns)
        self.assertIn("label", prepared.columns)
        self.assertIn("target_label", prepared.columns)
        self.assertIn("word_count", prepared.columns)

        # Check label IDs
        self.assertEqual(set(prepared["label"]), {0, 1, 2})

    def test_build_splits_and_leakage_check(self):
        """Verify stratified splitting and disjointness verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_csv = Path(tmp_dir) / "train.csv"
            test_csv = Path(tmp_dir) / "test.csv"

            self.mock_train_df.to_csv(train_csv, index=False)
            self.mock_test_df.to_csv(test_csv, index=False)

            builder = TransformerDatasetBuilder(val_ratio=0.20, random_state=42)
            train_df, val_df, test_df = builder.build_splits(train_csv, test_csv)

            # Check sizes: 30 * 0.20 = 6 validation, 24 train
            self.assertEqual(len(val_df), 6)
            self.assertEqual(len(train_df), 24)
            self.assertEqual(len(test_df), 9)

            # Check zero data leakage
            train_ids = set(train_df["issue_id"])
            val_ids = set(val_df["issue_id"])
            test_ids = set(test_df["issue_id"])

            self.assertEqual(len(train_ids.intersection(val_ids)), 0)
            self.assertEqual(len(train_ids.intersection(test_ids)), 0)
            self.assertEqual(len(val_ids.intersection(test_ids)), 0)

    def test_data_leakage_detection_raises(self):
        """Verify that overlapping issue IDs trigger RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_csv = Path(tmp_dir) / "train.csv"
            test_csv = Path(tmp_dir) / "test.csv"

            # Intentionally inject overlapping issue_id
            leaky_test = self.mock_test_df.copy()
            leaky_test.loc[0, "issue_id"] = self.mock_train_df.loc[0, "issue_id"]

            self.mock_train_df.to_csv(train_csv, index=False)
            leaky_test.to_csv(test_csv, index=False)

            builder = TransformerDatasetBuilder()
            with self.assertRaises(RuntimeError):
                builder.build_splits(train_csv, test_csv)

    def test_save_manifest_and_plot(self):
        """Verify manifest serialization and plot generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_csv = Path(tmp_dir) / "train.csv"
            test_csv = Path(tmp_dir) / "test.csv"
            out_dir = Path(tmp_dir) / "output"
            fig_path = Path(tmp_dir) / "fig17.png"

            self.mock_train_df.to_csv(train_csv, index=False)
            self.mock_test_df.to_csv(test_csv, index=False)

            builder = TransformerDatasetBuilder(val_ratio=0.20, random_state=42)
            res = builder.run_all(train_csv, test_csv, out_dir, fig_path)

            self.assertTrue(fig_path.exists())
            self.assertGreater(fig_path.stat().st_size, 1000)

            manifest_path = out_dir / "dataset_manifest.json"
            self.assertTrue(manifest_path.exists())

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["metadata"]["step"], "6.2")
            self.assertTrue(data["metadata"]["leakage_verified"])
            self.assertIn("splits", data)
            self.assertIn("train", data["splits"])
            self.assertIn("validation", data["splits"])
            self.assertIn("test", data["splits"])


if __name__ == "__main__":
    unittest.main()
