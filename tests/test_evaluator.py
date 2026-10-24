"""Unit tests for ml.classification.evaluator (ClassifierEvaluator)."""

import unittest
import tempfile
import json
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.sparse

from ml.classification.evaluator import ClassifierEvaluator


class DummyModel:
    """Mock classifier for testing evaluator metrics."""

    def __init__(self, predictions, probas=None):
        self.predictions = np.asarray(predictions)
        self.probas = np.asarray(probas) if probas is not None else None

    def predict(self, X):
        return self.predictions[: X.shape[0]]

    def predict_proba(self, X):
        if self.probas is not None:
            return self.probas[: X.shape[0]]
        raise AttributeError("No predict_proba")


class TestClassifierEvaluator(unittest.TestCase):
    """Test suite for ClassifierEvaluator."""

    def setUp(self):
        """Construct synthetic ground truth and predictions across 3 classes."""
        self.classes = ["bug", "feature", "other"]
        self.evaluator = ClassifierEvaluator(classes=self.classes)

        # 6 samples: 2 bug, 2 feature, 2 other
        self.y_true = np.array(["bug", "bug", "feature", "feature", "other", "other"])
        self.y_pred = np.array(["bug", "other", "feature", "feature", "other", "bug"])
        self.X_dummy = scipy.sparse.csr_matrix(np.zeros((6, 10)))

        self.mock_probas = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.1, 0.7],
            [0.05, 0.9, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.1, 0.3],
        ])

    def test_evaluate_model_metrics(self):
        model = DummyModel(self.y_pred, self.mock_probas)
        res = self.evaluator.evaluate_model(model, self.X_dummy, self.y_true, model_name="MockModel")

        self.assertEqual(res["model_name"], "MockModel")
        self.assertEqual(res["test_samples"], 6)
        # 4 correct out of 6 = 0.6667
        self.assertAlmostEqual(res["accuracy"], 0.6667, places=3)
        self.assertIn("macro_f1", res)
        self.assertIn("weighted_f1", res)
        self.assertIn("per_class", res)
        self.assertIn("bug", res["per_class"])
        self.assertIn("feature", res["per_class"])
        self.assertIn("other", res["per_class"])
        self.assertIsNotNone(res["log_loss"])
        self.assertIsNotNone(res["brier_score"])

    def test_confusion_matrix_properties(self):
        model = DummyModel(self.y_pred)
        res = self.evaluator.evaluate_model(model, self.X_dummy, self.y_true)

        cm = np.array(res["confusion_matrix"]["raw_matrix"])
        norm_cm = np.array(res["confusion_matrix"]["normalized_matrix"])

        self.assertEqual(cm.shape, (3, 3))
        # Total samples in confusion matrix must equal test sample count
        self.assertEqual(cm.sum(), 6)
        # Row sums of normalized matrix must equal 1.0
        np.testing.assert_allclose(norm_cm.sum(axis=1), np.ones(3), atol=1e-5)

    def test_compare_models_ranking(self):
        # Perfect model
        perfect_model = DummyModel(self.y_true)
        # Imperfect model
        imperfect_model = DummyModel(self.y_pred)

        models = {"Imperfect": imperfect_model, "Perfect": perfect_model}
        comparison = self.evaluator.compare_models(models, self.X_dummy, self.y_true)

        self.assertEqual(comparison["champion_model"], "Perfect")
        self.assertEqual(len(comparison["comparison_table"]), 2)
        self.assertIn("Perfect", comparison["detailed_evaluations"])
        self.assertIn("Imperfect", comparison["detailed_evaluations"])

    def test_plot_confusion_matrices(self):
        model1 = DummyModel(self.y_true)
        model2 = DummyModel(self.y_pred)
        comparison = self.evaluator.compare_models({"M1": model1, "M2": model2}, self.X_dummy, self.y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_img = Path(tmpdir) / "test_cm.png"
            saved_path = self.evaluator.plot_confusion_matrices(comparison, output_path=out_img)
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 1000)

    def test_save_report(self):
        model = DummyModel(self.y_pred)
        comparison = self.evaluator.compare_models({"ModelA": model}, self.X_dummy, self.y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "report.json"
            saved_path = self.evaluator.save_report(comparison, output_path=out_json)
            self.assertTrue(saved_path.exists())

            with open(saved_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["champion_model"], "ModelA")

    def test_live_held_out_evaluation(self):
        test_path = Path("data/processed/splits/test.csv")
        tfidf_path = Path("models/tfidf_vectorizer.joblib")
        lr_path = Path("models/baseline_logistic_regression.joblib")
        svm_path = Path("models/baseline_svm.joblib")

        if not (test_path.exists() and tfidf_path.exists() and lr_path.exists() and svm_path.exists()):
            self.skipTest("Required artifacts for live test evaluation not found.")

        test_df = pd.read_csv(test_path)
        from ml.classification.features import TextFeaturePipeline
        from ml.classification.logistic_regression import LogisticRegressionBaseline
        from ml.classification.svm import SVMBaseline

        pipeline = TextFeaturePipeline.load(tfidf_path)
        X_test = pipeline.transform(test_df)
        y_test = test_df["target_label"].to_numpy()

        lr_model = LogisticRegressionBaseline.load(lr_path)
        svm_model = SVMBaseline.load(svm_path)

        comparison = self.evaluator.compare_models(
            {"Logistic Regression": lr_model, "Support Vector Machine": svm_model},
            X_test,
            y_test,
        )

        self.assertEqual(comparison["test_sample_count"], 385)
        self.assertIn(comparison["champion_model"], ["Logistic Regression", "Support Vector Machine"])

        # Check that both classical models score >= 80% on held-out test data
        for model_eval in comparison["comparison_table"]:
            self.assertGreaterEqual(model_eval["accuracy"], 0.80)
            self.assertGreaterEqual(model_eval["macro_f1"], 0.80)


if __name__ == "__main__":
    unittest.main()
