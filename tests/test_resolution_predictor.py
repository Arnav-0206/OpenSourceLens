"""
OpenSourceLens — Unit Tests for Resolution Predictor & Reliability Decision (Phase 8, Step 8.7).

Validates feature extraction from raw issue text, velocity classification inference,
calibrated probability distributions, repository historical profiles, disclaimer compliance,
and Phase 8 milestone decision artifacts.
"""

import json
from pathlib import Path
import unittest

import numpy as np

from ml.resolution_analysis.predictor import (
    ADVISORY_DISCLAIMER,
    SPEED_TIER_ORDER,
    ResolutionPredictor,
)


class TestResolutionPredictor(unittest.TestCase):
    """Test suite for ResolutionPredictor."""

    @classmethod
    def setUpClass(cls):
        """Initializes the predictor."""
        cls.predictor = ResolutionPredictor()

    def test_predictor_initialization(self):
        """Verifies model loading, feature names, classes, and scaler parameters."""
        self.assertIsNotNone(self.predictor.classifier_model)
        self.assertEqual(len(self.predictor.classes_), len(SPEED_TIER_ORDER))
        self.assertEqual(len(self.predictor.feature_names), 28)
        self.assertEqual(len(self.predictor.scaler_params), 20)

    def test_feature_extraction_from_raw_text(self):
        """Validates feature extraction from raw title, body, and repo metadata."""
        X = self.predictor.extract_features(
            title="[Bug] Segmentation fault in tensor allocation",
            body="Code snippet:\n```python\nx = torch.randn(10, 10)\n```\nhttp://github.com",
            repo_name="pytorch/pytorch",
            canonical_label="bug",
        )
        self.assertEqual(X.shape, (1, 28))
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isinf(X).any())

    def test_prediction_output_schema(self):
        """Validates return structure and required fields."""
        result = self.predictor.predict_velocity(
            title="Update README documentation for installation",
            body="Fix typo in installation guide.",
            repo_name="huggingface/transformers",
            canonical_label="documentation",
        )
        self.assertIn("predicted_tier", result)
        self.assertIn("confidence", result)
        self.assertIn("tier_probabilities", result)
        self.assertIn("adjacent_window", result)
        self.assertIn("advisory_factors", result)
        self.assertIn("disclaimer", result)
        self.assertIn(result["predicted_tier"], SPEED_TIER_ORDER)

    def test_confidence_probabilities_sum_to_one(self):
        """Ensures probabilities sum to ~1.0 and confidence is bounded."""
        result = self.predictor.predict_velocity(
            title="[Feature] Add support for quantized LoRA",
            body="Detailed RFC proposal for quantized LoRA weights.",
            repo_name="pytorch/pytorch",
            canonical_label="feature",
        )
        probs = result["tier_probabilities"]
        self.assertEqual(len(probs), 4)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_adjacent_tier_bounds(self):
        """Validates upper and lower bracket definitions."""
        result = self.predictor.predict_velocity(
            title="Quick fix for CSS styling",
            body="Small fix.",
            repo_name="microsoft/vscode",
            canonical_label="other",
        )
        window = result["adjacent_window"]
        self.assertIn("lower_bound", window)
        self.assertIn("upper_bound", window)
        self.assertIn(window["lower_bound"], SPEED_TIER_ORDER)
        self.assertIn(window["upper_bound"], SPEED_TIER_ORDER)

    def test_repository_profile_retrieval(self):
        """Verifies historical descriptive profile for known repository."""
        profile = self.predictor.get_repository_resolution_profile("microsoft/vscode")
        self.assertIn("median_resolution_hours", profile)
        self.assertIn("pct_resolved_under_24h", profile)
        self.assertGreater(profile["n_closed_issues_analyzed"], 100)
        self.assertGreater(profile["pct_resolved_under_24h"], 0.0)

    def test_disclaimer_presence(self):
        """Enforces mandatory advisory disclaimer presence."""
        result = self.predictor.predict_velocity(
            title="Memory leak in worker thread",
            body="Investigation details.",
            repo_name="kubernetes/kubernetes",
        )
        self.assertEqual(result["disclaimer"], ADVISORY_DISCLAIMER)
        self.assertIn("Does not constitute an SLA guarantee", result["disclaimer"])

    def test_decision_manifest_and_figures(self):
        """Validates that Phase 8 decision manifest and Figure 34 exist and are valid."""
        base_dir = Path(__file__).resolve().parent.parent
        manifest_file = base_dir / "data" / "processed" / "evaluation" / "resolution_reliability_decision.json"
        fig34_file = base_dir / "ml" / "resolution_analysis" / "figures" / "fig34_resolution_decision_and_architecture.png"

        self.assertTrue(manifest_file.exists())
        self.assertTrue(fig34_file.exists())
        self.assertGreater(fig34_file.stat().st_size, 50000)

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("scientific_decision_summary", data)
        self.assertEqual(data["metadata"]["status"], "APPROVED_ADVISORY_TIERS_ONLY")


if __name__ == "__main__":
    unittest.main()
