"""Unit tests for ml.classification.target_definition (ClassificationTargetDefiner)."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from ml.classification.target_definition import ClassificationTargetDefiner


class TestClassificationTargetDefiner(unittest.TestCase):
    """Test suite for ClassificationTargetDefiner."""

    def setUp(self):
        """Construct synthetic test dataset with varied labels, unlabeled records, and text."""
        self.mock_df = pd.DataFrame({
            "issue_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "repo_name": ["repo_a"] * 5 + ["repo_b"] * 5,
            "title": [
                "Crash on startup", "Add new export button", "Docs typo fix", "How to install?",
                "Refactor codebase", "Unlabeled item 1", "Unlabeled item 2", "Another crash bug",
                "Dark theme request", "Performance issue"
            ],
            "body": ["App crashes with error", "Please add button", "Typo in readme", "Install guidance",
                     "Clean up internal logic", "No label on this", "Also unlabeled", "Memory leak crash",
                     "Support dark mode", "Slow startup time"],
            "canonical_label": [
                "bug", "feature", "documentation", "question", "other",
                "unlabeled", "unlabeled", "bug", "feature", "other"
            ],
            "raw_labels": [
                ["kind/bug"], ["kind/feature"], ["kind/doc"], ["kind/question"], ["kind/refactor"],
                [], [], ["kind/bug"], ["kind/feature"], ["kind/perf"]
            ],
            "canonical_label_id": [0, 1, 2, 3, 4, -1, -1, 0, 1, 4],
            "clean_title_word_count": [3, 4, 3, 3, 2, 3, 2, 3, 3, 2],
            "clean_body_word_count": [4, 3, 3, 2, 4, 4, 2, 3, 3, 3],
            "has_code_block": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "has_stack_trace": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "has_url": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "title_has_bracket_tag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "is_weekend": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "labels_count": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            "is_unlabeled": [False, False, False, False, False, True, True, False, False, False],
            "is_classified_category": [True, True, True, True, True, False, False, True, True, True],
        })

    def test_init_from_dataframe(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        self.assertEqual(len(definer.labeled_df), 8)
        self.assertEqual(len(definer.unlabeled_df), 2)
        self.assertEqual(definer.source_path, "DataFrame in-memory")

    def test_init_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            self.mock_df.to_csv(csv_path, index=False)
            definer = ClassificationTargetDefiner(csv_path, target_mode="core_3class")
            self.assertEqual(len(definer.labeled_df), 8)
            self.assertEqual(definer.source_path, str(csv_path))

    def test_init_from_parquet_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            parquet_path = Path(tmpdir) / "test_data.parquet"
            self.mock_df.to_csv(csv_path, index=False)
            parquet_path.write_bytes(b"MOCK_PARQUET_FALLBACK")
            definer = ClassificationTargetDefiner(parquet_path, target_mode="core_3class")
            self.assertEqual(len(definer.labeled_df), 8)

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ClassificationTargetDefiner("non_existent_file.csv")

    def test_init_invalid_type(self):
        with self.assertRaises(TypeError):
            ClassificationTargetDefiner(12345)  # type: ignore

    def test_init_empty_dataframe(self):
        with self.assertRaises(ValueError):
            ClassificationTargetDefiner(pd.DataFrame())

    def test_init_invalid_target_mode(self):
        with self.assertRaises(ValueError):
            ClassificationTargetDefiner(self.mock_df, target_mode="invalid_mode")

    def test_init_missing_label_column(self):
        bad_df = pd.DataFrame({"issue_id": [1], "title": ["crash"]})
        with self.assertRaises(KeyError):
            ClassificationTargetDefiner(bad_df)

    def test_core_3class_mapping(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        counts = definer.labeled_df["target_label"].value_counts().to_dict()

        # In mock_df: 2 bug, 2 feature, 1 doc + 1 question + 2 other = 4 other
        self.assertEqual(counts["bug"], 2)
        self.assertEqual(counts["feature"], 2)
        self.assertEqual(counts["other"], 4)
        self.assertEqual(set(definer.classes), {"bug", "feature", "other"})

    def test_binary_bug_mapping(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="binary_bug")
        counts = definer.labeled_df["target_label"].value_counts().to_dict()

        # 2 bug, 6 non_bug
        self.assertEqual(counts["bug"], 2)
        self.assertEqual(counts["non_bug"], 6)
        self.assertEqual(set(definer.classes), {"bug", "non_bug"})

    def test_full_multiclass_mapping(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="full_multiclass")
        counts = definer.labeled_df["target_label"].value_counts().to_dict()

        self.assertEqual(counts["bug"], 2)
        self.assertEqual(counts["feature"], 2)
        self.assertEqual(counts["documentation"], 1)
        self.assertEqual(counts["question"], 1)
        self.assertEqual(counts["other"], 2)
        self.assertEqual(len(definer.classes), 5)

    def test_class_weights_computation(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        weights = definer.class_weights
        weights_named = definer.class_weights_named

        # Formula: N / (K * n_j) -> N = 8, K = 3
        # for 'other': 8 / (3 * 4) = 8 / 12 = 0.6667
        # for 'bug': 8 / (3 * 2) = 8 / 6 = 1.3333
        # for 'feature': 8 / (3 * 2) = 8 / 6 = 1.3333
        self.assertAlmostEqual(weights_named["other"], 0.6667, places=3)
        self.assertAlmostEqual(weights_named["bug"], 1.3333, places=3)
        self.assertAlmostEqual(weights_named["feature"], 1.3333, places=3)

    def test_get_target_distribution(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        dist = definer.get_target_distribution()

        self.assertEqual(len(dist), 3)
        self.assertEqual(dist["count"].sum(), 8)
        self.assertAlmostEqual(dist["percentage"].sum(), 100.0, places=1)
        self.assertIn("class_name", dist.columns)
        self.assertIn("class_id", dist.columns)
        self.assertIn("balanced_weight", dist.columns)

    def test_get_class_mappings(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        mappings = definer.get_class_mappings()

        self.assertIn("label_to_id", mappings)
        self.assertIn("id_to_label", mappings)
        self.assertEqual(len(mappings["label_to_id"]), 3)
        for name, idx in mappings["label_to_id"].items():
            self.assertEqual(mappings["id_to_label"][idx], name)

    def test_prepare_labeled_dataset_no_leakage(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        prepared = definer.prepare_labeled_dataset(include_auxiliary_features=True)

        self.assertEqual(len(prepared), 8)
        # Verify target is present
        self.assertIn("target_label", prepared.columns)
        self.assertIn("target_id", prepared.columns)
        self.assertIn("clean_title", prepared.columns)
        self.assertIn("clean_body", prepared.columns)
        self.assertIn("combined_text", prepared.columns)

        # Verify STRICT data leakage prevention
        for leak_col in ClassificationTargetDefiner.LEAKAGE_COLUMNS:
            self.assertNotIn(leak_col, prepared.columns, f"Data leakage detected! '{leak_col}' is in prepared dataset.")

    def test_prepare_labeled_dataset_without_auxiliary(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        prepared = definer.prepare_labeled_dataset(include_auxiliary_features=False)

        self.assertNotIn("has_code_block", prepared.columns)
        self.assertIn("combined_text", prepared.columns)
        self.assertIn("target_id", prepared.columns)

    def test_unlabeled_pool_extraction(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        unlabeled = definer.get_unlabeled_pool()

        self.assertEqual(len(unlabeled), 2)
        # Zero leakage in unlabeled pool
        for leak_col in ClassificationTargetDefiner.LEAKAGE_COLUMNS:
            self.assertNotIn(leak_col, unlabeled.columns)

    def test_get_summary(self):
        definer = ClassificationTargetDefiner(self.mock_df, target_mode="core_3class")
        summary = definer.get_summary()

        self.assertEqual(summary["total_raw_records"], 10)
        self.assertEqual(summary["labeled_modeling_records"], 8)
        self.assertEqual(summary["unlabeled_inference_records"], 2)
        self.assertEqual(summary["classes_count"], 3)
        self.assertTrue(summary["data_leakage_prevented"])

    def test_live_master_dataset_integration(self):
        csv_path = Path("data/processed/issues_master.csv")
        if not csv_path.exists():
            self.skipTest("Master dataset CSV not found.")

        definer = ClassificationTargetDefiner(csv_path, target_mode="core_3class")
        summary = definer.get_summary()

        # In real master dataset: 2,328 total records
        self.assertEqual(summary["total_raw_records"], 2328)
        self.assertEqual(summary["labeled_modeling_records"], 1922)
        self.assertEqual(summary["unlabeled_inference_records"], 406)

        # In core_3class mode:
        # bug: 800
        # feature: 340
        # other: 761 (other) + 17 (doc) + 4 (question) = 782
        dist = definer.get_target_distribution().set_index("class_name")
        self.assertEqual(dist.loc["bug", "count"], 800)
        self.assertEqual(dist.loc["feature", "count"], 340)
        self.assertEqual(dist.loc["other", "count"], 782)

        prepared = definer.prepare_labeled_dataset()
        self.assertEqual(len(prepared), 1922)
        for leak in ClassificationTargetDefiner.LEAKAGE_COLUMNS:
            self.assertNotIn(leak, prepared.columns)


if __name__ == "__main__":
    unittest.main()
