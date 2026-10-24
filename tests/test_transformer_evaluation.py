"""Unit Tests for Transformer Evaluation Pipeline (Step 6.5)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ml.deep_learning.evaluator import TransformerEvaluator


class TestTransformerEvaluation(unittest.TestCase):
    """Test suite for TransformerEvaluator, metrics computation, and artifact generation."""

    @classmethod
    def setUpClass(cls):
        """Set up small dataset and evaluator instance using checkpoint."""
        # Use existing fine-tuned checkpoint if available, or fallback to models/distilbert_finetuned
        cls.evaluator = TransformerEvaluator(
            checkpoint_dir="models/distilbert_finetuned",
            max_length=64,
        )

        cls.mock_test_df = pd.DataFrame({
            "issue_id": ["iss_101", "iss_102", "iss_103", "iss_104", "iss_105", "iss_106"],
            "repo_name": ["vscode", "pytorch", "kubernetes", "vscode", "transformers", "pytorch"],
            "text": [
                "Title: Critical crash on window open | Description: Null reference exception thrown",
                "Title: Add support for AVX-512 instructions | Description: Performance feature request",
                "Title: Fix broken documentation links | Description: Updated tutorial URLs in markdown",
                "Title: Out of memory in language worker | Description: Memory leak during syntax parse",
                "Title: Add batch inference option | Description: New functionality to speed up pipeline",
                "Title: Bump version to 1.5.0 | Description: Release chores and badge update",
            ],
            "target_label": ["bug", "feature", "other", "bug", "feature", "other"],
            "label": [0, 1, 2, 0, 1, 2],
        })

    def test_evaluator_initialization_and_mappings(self):
        """Verify label mappings, tokenizer, and model properties."""
        self.assertEqual(self.evaluator.num_labels, 3)
        self.assertEqual(self.evaluator.id2label[0], "bug")
        self.assertEqual(self.evaluator.id2label[1], "feature")
        self.assertEqual(self.evaluator.id2label[2], "other")
        self.assertEqual(self.evaluator.label2id["bug"], 0)
        self.assertEqual(self.evaluator.label2id["feature"], 1)
        self.assertEqual(self.evaluator.label2id["other"], 2)
        self.assertIsNotNone(self.evaluator.model)
        self.assertIsNotNone(self.evaluator.tokenizer)

    def test_predict_proba_and_predict(self):
        """Verify probability shapes, simplex constraint (sums to 1.0), and predicted labels."""
        texts = self.mock_test_df["text"].tolist()
        probs = self.evaluator.predict_proba(texts, batch_size=4)

        self.assertEqual(probs.shape, (len(texts), 3))
        # Ensure all probabilities are valid within [0, 1]
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        # Ensure rows sum to 1.0 (softmax simplex)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

        preds, probs_ret = self.evaluator.predict(texts, batch_size=4)
        self.assertEqual(len(preds), len(texts))
        self.assertTrue(np.all(np.isin(preds, [0, 1, 2])))
        self.assertTrue(np.array_equal(preds, np.argmax(probs, axis=1)))

    def test_evaluate_dataset_metrics(self):
        """Verify full evaluation metric calculation and predictions DataFrame."""
        metrics, preds_df = self.evaluator.evaluate_dataset(self.mock_test_df, batch_size=4)

        # Core metric presence and range checks
        self.assertEqual(metrics["sample_count"], len(self.mock_test_df))
        self.assertGreaterEqual(metrics["overall_accuracy"], 0.0)
        self.assertLessEqual(metrics["overall_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)
        self.assertGreaterEqual(metrics["log_loss"], 0.0)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertGreaterEqual(metrics["macro_roc_auc"], 0.0)
        self.assertLessEqual(metrics["macro_roc_auc"], 1.0)

        # Per-class metrics
        self.assertIn("bug", metrics["per_class"])
        self.assertIn("feature", metrics["per_class"])
        self.assertIn("other", metrics["per_class"])
        for c_name in ["bug", "feature", "other"]:
            c_info = metrics["per_class"][c_name]
            self.assertIn("precision", c_info)
            self.assertIn("recall", c_info)
            self.assertIn("f1_score", c_info)
            self.assertIn("support", c_info)
            self.assertIn("roc_auc", c_info)
            self.assertIn("average_precision", c_info)

        # Confusion Matrix
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (3, 3))
        self.assertEqual(cm.sum(), len(self.mock_test_df))

        # Benchmarks
        bench = metrics["inference_benchmarks"]
        self.assertIn("total_inference_seconds", bench)
        self.assertIn("latency_ms_per_sample", bench)
        self.assertIn("throughput_samples_per_sec", bench)
        self.assertGreater(bench["throughput_samples_per_sec"], 0.0)

        # Predictions DataFrame checks
        self.assertEqual(len(preds_df), len(self.mock_test_df))
        for col in ["pred_label", "pred_target", "confidence", "is_correct", "prob_bug", "prob_feature", "prob_other"]:
            self.assertIn(col, preds_df.columns)

    def test_visualizations_and_exports(self):
        """Verify confusion matrix plot, ROC/PR curves plot, and file serialization."""
        metrics, preds_df = self.evaluator.evaluate_dataset(self.mock_test_df, batch_size=4)
        probs_matrix = preds_df[["prob_bug", "prob_feature", "prob_other"]].values

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Confusion matrix plot
            cm_fig = tmp_path / "test_cm.png"
            self.evaluator.plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_fig)
            self.assertTrue(cm_fig.exists())
            self.assertGreater(cm_fig.stat().st_size, 1000)

            # ROC & PR curves plot
            curves_fig = tmp_path / "test_curves.png"
            self.evaluator.plot_roc_pr_curves(self.mock_test_df["label"].tolist(), probs_matrix, save_path=curves_fig)
            self.assertTrue(curves_fig.exists())
            self.assertGreater(curves_fig.stat().st_size, 1000)

            # JSON summary export
            json_file = tmp_path / "test_eval.json"
            self.evaluator.save_evaluation_summary(metrics, save_path=json_file)
            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["sample_count"], len(self.mock_test_df))

            # CSV predictions export
            csv_file = tmp_path / "test_preds.csv"
            self.evaluator.save_predictions_csv(preds_df, save_path=csv_file)
            self.assertTrue(csv_file.exists())
            loaded_df = pd.read_csv(csv_file)
            self.assertEqual(len(loaded_df), len(self.mock_test_df))


if __name__ == "__main__":
    unittest.main()
