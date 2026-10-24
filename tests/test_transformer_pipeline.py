"""Unit Tests for Transformer Production Inference Pipeline (Step 6.8)."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from ml.deep_learning.pipeline import (
    TransformerIssueClassifier,
    export_production_bundle,
)


class TestTransformerPipeline(unittest.TestCase):
    """Test suite for TransformerIssueClassifier production wrapper and export bundle."""

    @classmethod
    def setUpClass(cls):
        """Set up classifier pipeline from saved model checkpoint."""
        cls.classifier = TransformerIssueClassifier(
            model_dir="models/distilbert_finetuned",
            max_length=64,
            uncertainty_threshold=0.55,
        )

    def test_pipeline_properties_and_labels(self):
        """Verify label mappings and underlying model initialization."""
        self.assertEqual(len(self.classifier.class_names), 3)
        self.assertIn("bug", self.classifier.class_names)
        self.assertIn("feature", self.classifier.class_names)
        self.assertIn("other", self.classifier.class_names)
        self.assertEqual(self.classifier.label2id["bug"], 0)
        self.assertEqual(self.classifier.label2id["feature"], 1)
        self.assertEqual(self.classifier.label2id["other"], 2)

    def test_predict_one_output_structure(self):
        """Verify predict_one output keys, types, and probability simplex."""
        title = "Fix crash when opening large project files"
        body = "Null reference encountered in workspace loader during file scan."
        result = self.classifier.predict_one(title, body)

        self.assertIn("label", result)
        self.assertIn("label_id", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("is_uncertain", result)
        self.assertIn("recommendation", result)

        self.assertIn(result["label"], ["bug", "feature", "other"])
        self.assertIn(result["label_id"], [0, 1, 2])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Check probabilities sum to 1.0
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        self.assertEqual(result["confidence"], result["probabilities"][result["label"]])

    def test_predict_batch(self):
        """Verify batched prediction on lists of issue dictionaries."""
        issues = [
            {"title": "Segmentation fault in tensor indexing", "body": "Core dumped on cuda:0"},
            {"title": "Add support for float8 dtype", "body": "Feature request for fp8 quantization"},
            {"title": "Update README docs", "body": "Fixed broken anchor link"},
        ]
        results = self.classifier.predict_batch(issues, batch_size=2)

        self.assertEqual(len(results), 3)
        for res in results:
            self.assertIn("label", res)
            self.assertIn("confidence", res)
            self.assertIn("probabilities", res)

    def test_predict_df(self):
        """Verify DataFrame enrichment with prediction columns."""
        df = pd.DataFrame({
            "clean_title": ["Crash on start", "New UI toggle"],
            "clean_body": ["Crash traceback", "Add dark mode"],
        })
        out_df = self.classifier.predict_df(df, batch_size=2)

        self.assertEqual(len(out_df), 2)
        expected_cols = [
            "pred_label",
            "pred_target",
            "confidence",
            "is_uncertain",
            "prob_bug",
            "prob_feature",
            "prob_other",
        ]
        for col in expected_cols:
            self.assertIn(col, out_df.columns)

    def test_uncertainty_threshold_flag(self):
        """Verify is_uncertain flag accurately reflects confidence threshold."""
        # High threshold: should force uncertainty flag
        strict_classifier = TransformerIssueClassifier(
            model_dir="models/distilbert_finetuned",
            max_length=64,
            uncertainty_threshold=0.99,
        )
        res_strict = strict_classifier.predict_one("Ambiguous issue title")
        self.assertTrue(res_strict["is_uncertain"])
        self.assertEqual(res_strict["recommendation"], "human_triage_required")

        # Zero threshold: should never flag uncertainty
        lax_classifier = TransformerIssueClassifier(
            model_dir="models/distilbert_finetuned",
            max_length=64,
            uncertainty_threshold=0.0,
        )
        res_lax = lax_classifier.predict_one("Ambiguous issue title")
        self.assertFalse(res_lax["is_uncertain"])
        self.assertEqual(res_lax["recommendation"], "automated_classification")

    def test_health_check(self):
        """Verify self-test health check returns healthy status and latency."""
        health = self.classifier.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertGreater(health["latency_ms"], 0.0)
        self.assertIn("test_prediction", health)

    def test_export_production_bundle(self):
        """Verify manifest.json and README.md generation with file inventory and hashes."""
        bundle = export_production_bundle(model_dir="models/distilbert_finetuned")

        self.assertIn("model_identity", bundle)
        self.assertIn("file_inventory", bundle)
        self.assertIn("verified_test_metrics", bundle)
        self.assertIn("operational_profile", bundle)

        # Verify files exist on disk
        manifest_path = Path("models/distilbert_finetuned/manifest.json")
        readme_path = Path("models/distilbert_finetuned/README.md")
        self.assertTrue(manifest_path.exists())
        self.assertTrue(readme_path.exists())

        # Verify manifest JSON can be parsed and contains checksums
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_loaded = json.load(f)
        self.assertIn("model.safetensors", manifest_loaded["file_inventory"])
        self.assertIn("sha256", manifest_loaded["file_inventory"]["model.safetensors"])


if __name__ == "__main__":
    unittest.main()
