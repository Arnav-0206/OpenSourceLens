"""Unit tests for ml.classification.error_analysis (ClassificationErrorAnalyzer)."""

import unittest
import tempfile
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ml.classification.error_analysis import ClassificationErrorAnalyzer


class TestClassificationErrorAnalyzer(unittest.TestCase):
    """Test suite for ClassificationErrorAnalyzer."""

    def setUp(self):
        """Construct synthetic test DataFrame, predictions, and probabilities."""
        self.classes = ["bug", "feature", "other"]
        self.analyzer = ClassificationErrorAnalyzer(classes=self.classes)

        # 6 samples across 2 repos
        self.mock_df = pd.DataFrame({
            "issue_id": [101, 102, 103, 104, 105, 106],
            "repo_name": ["repo/a", "repo/a", "repo/a", "repo/b", "repo/b", "repo/b"],
            "target_label": ["bug", "bug", "feature", "feature", "other", "other"],
            "clean_title": ["Bug 1", "Bug 2", "Feat 1", "Feat 2", "Other 1", "Other 2"],
            "clean_body": ["Body 1", "Body 2", "Body 3", "Body 4", "Body 5", "Body 6"],
            "clean_title_word_count": [2, 2, 2, 2, 2, 2],
            "clean_body_word_count": [20, 50, 150, 400, 30, 80],
        })

        # Predictions: 1, 3, 4 are correct; 2, 5, 6 are errors
        self.y_pred = np.array(["bug", "other", "feature", "feature", "bug", "feature"])

        self.y_probas = np.array([
            [0.85, 0.05, 0.10],  # Correct bug (p=0.85)
            [0.20, 0.05, 0.75],  # Error: true=bug, pred=other (conf=0.75)
            [0.05, 0.90, 0.05],  # Correct feature
            [0.10, 0.80, 0.10],  # Correct feature
            [0.80, 0.05, 0.15],  # Error: true=other, pred=bug (conf=0.80)
            [0.10, 0.65, 0.25],  # Error: true=other, pred=feature (conf=0.65)
        ])

    def test_extract_errors(self):
        errors_df = self.analyzer.extract_errors(self.mock_df, self.y_pred, self.y_probas)

        # 3 errors total
        self.assertEqual(len(errors_df), 3)
        self.assertIn("predicted_label", errors_df.columns)
        self.assertIn("confidence", errors_df.columns)
        self.assertIn("margin", errors_df.columns)
        self.assertIn("total_word_count", errors_df.columns)

        # Check correct error IDs
        self.assertEqual(errors_df["issue_id"].tolist(), [102, 105, 106])

    def test_get_confusion_pair_breakdown(self):
        errors_df = self.analyzer.extract_errors(self.mock_df, self.y_pred, self.y_probas)
        breakdown = self.analyzer.get_confusion_pair_breakdown(errors_df)

        self.assertEqual(breakdown.get("bug -> other"), 1)
        self.assertEqual(breakdown.get("other -> bug"), 1)
        self.assertEqual(breakdown.get("other -> feature"), 1)
        self.assertEqual(sum(breakdown.values()), 3)

    def test_get_high_confidence_errors(self):
        errors_df = self.analyzer.extract_errors(self.mock_df, self.y_pred, self.y_probas)
        high_conf = self.analyzer.get_high_confidence_errors(errors_df, threshold=0.70)

        # Two errors >= 0.70 confidence: 105 (0.80) and 102 (0.75)
        self.assertEqual(len(high_conf), 2)
        self.assertEqual(high_conf[0]["issue_id"], 105)
        self.assertAlmostEqual(high_conf[0]["confidence"], 0.80, places=2)
        self.assertEqual(high_conf[1]["issue_id"], 102)

    def test_get_repository_error_rates(self):
        errors_df = self.analyzer.extract_errors(self.mock_df, self.y_pred, self.y_probas)
        repo_rates = self.analyzer.get_repository_error_rates(self.mock_df, errors_df)

        self.assertIn("repo/a", repo_rates)
        self.assertIn("repo/b", repo_rates)

        # repo/a: 1 error out of 3 = 33.33%
        self.assertEqual(repo_rates["repo/a"]["total_samples"], 3)
        self.assertEqual(repo_rates["repo/a"]["error_count"], 1)
        self.assertAlmostEqual(repo_rates["repo/a"]["error_rate_pct"], 33.33, places=1)

        # repo/b: 2 errors out of 3 = 66.67%
        self.assertEqual(repo_rates["repo/b"]["total_samples"], 3)
        self.assertEqual(repo_rates["repo/b"]["error_count"], 2)

    def test_get_length_correlation(self):
        errors_df = self.analyzer.extract_errors(self.mock_df, self.y_pred, self.y_probas)
        len_stats = self.analyzer.get_length_correlation(self.mock_df, errors_df)

        self.assertIn("Short (<25)", len_stats)
        self.assertIn("Medium (25-100)", len_stats)
        self.assertIn("Long (100-300)", len_stats)
        self.assertIn("Very Long (>300)", len_stats)

        total_binned = sum(s["total_samples"] for s in len_stats.values())
        self.assertEqual(total_binned, len(self.mock_df))

    def test_plot_error_breakdown(self):
        errors_df = self.analyzer.extract_errors(self.mock_df, self.y_pred, self.y_probas)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_img = Path(tmpdir) / "test_error_plot.png"
            saved_path = self.analyzer.plot_error_breakdown(
                self.mock_df, errors_df, self.y_probas, self.y_pred, output_path=out_img
            )
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)

    def test_generate_full_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "test_error_report.json"
            report = self.analyzer.generate_full_report(
                self.mock_df, self.y_pred, self.y_probas, output_json=out_json
            )

            self.assertTrue(out_json.exists())
            self.assertEqual(report["summary"]["total_errors"], 3)
            self.assertIn("failure_pair_taxonomy", report)
            self.assertIn("insights_and_recommendations", report)

    def test_live_test_set_error_analysis(self):
        test_path = Path("data/processed/splits/test.csv")
        tfidf_path = Path("models/tfidf_vectorizer.joblib")
        svm_path = Path("models/baseline_svm.joblib")

        if not (test_path.exists() and tfidf_path.exists() and svm_path.exists()):
            self.skipTest("Required artifacts for live error analysis not found.")

        test_df = pd.read_csv(test_path)
        from ml.classification.features import TextFeaturePipeline
        from ml.classification.svm import SVMBaseline

        pipeline = TextFeaturePipeline.load(tfidf_path)
        X_test = pipeline.transform(test_df)

        model = SVMBaseline.load(svm_path)
        y_pred = model.predict(X_test)
        y_probas = model.predict_proba(X_test)

        errors_df = self.analyzer.extract_errors(test_df, y_pred, y_probas)

        # In real test set: 385 issues, 72 errors (81.3% accuracy)
        self.assertEqual(len(errors_df), 72)

        # Check breakdown: zero bug -> feature confusion
        breakdown = self.analyzer.get_confusion_pair_breakdown(errors_df)
        self.assertNotIn("bug -> feature", breakdown)

        # Check repository breakdown
        repo_rates = self.analyzer.get_repository_error_rates(test_df, errors_df)
        self.assertEqual(len(repo_rates), 4)
        for r in ["microsoft/vscode", "pytorch/pytorch", "kubernetes/kubernetes", "huggingface/transformers"]:
            self.assertIn(r, repo_rates)


if __name__ == "__main__":
    unittest.main()
