"""Unit tests for ml.classification.svm (SVMBaseline)."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.classification.svm import SVMBaseline


class TestSVMBaseline(unittest.TestCase):
    """Test suite for SVMBaseline."""

    def setUp(self):
        """Construct synthetic dataset with known distinguishing vocabulary."""
        self.texts = [
            "Critical crash error segfault",
            "Bug report fatal error memory crash",
            "Kernel panic segmentation crash",
            "Add feature support dark mode",
            "Feature request export markdown button",
            "New feature enhance usability support",
            "Documentation typo in readme guide",
            "Refactor internal code cleanup",
            "Update license and contribution policy",
        ]
        self.labels = ["bug", "bug", "bug", "feature", "feature", "feature", "other", "other", "other"]

        self.vectorizer = TfidfVectorizer(min_df=1)
        self.X = self.vectorizer.fit_transform(self.texts)
        self.y = np.array(self.labels)

    def test_fit_and_predict_basic(self):
        baseline = SVMBaseline(C=1.0)
        baseline.fit(self.X, self.y)

        self.assertTrue(baseline.is_fitted)
        self.assertEqual(len(baseline.classes_), 3)

        preds = baseline.predict(self.X)
        self.assertEqual(len(preds), len(self.texts))
        for p in preds:
            self.assertIn(p, ["bug", "feature", "other"])

    def test_decision_function(self):
        baseline = SVMBaseline(C=1.0)
        baseline.fit(self.X, self.y)

        margins = baseline.decision_function(self.X)
        self.assertEqual(margins.shape, (len(self.texts), 3))

    def test_calibrated_probabilities(self):
        baseline = SVMBaseline(C=1.0, calibrate_probabilities=True)
        baseline.fit(self.X, self.y)

        probas = baseline.predict_proba(self.X)
        self.assertEqual(probas.shape, (len(self.texts), 3))

        # Check probability bounds and row simplex condition: sum(p_i) == 1
        self.assertTrue(np.all(probas >= 0.0))
        self.assertTrue(np.all(probas <= 1.0))
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(len(self.texts)), atol=1e-5)

    def test_predict_proba_uncalibrated_raises(self):
        baseline = SVMBaseline(C=1.0, calibrate_probabilities=False)
        baseline.fit(self.X, self.y)

        with self.assertRaises(RuntimeError):
            baseline.predict_proba(self.X)

    def test_fit_empty_raises(self):
        baseline = SVMBaseline()
        with self.assertRaises(ValueError):
            baseline.fit(scipy.sparse.csr_matrix((0, 10)), np.array([]))

    def test_fit_mismatched_dimensions_raises(self):
        baseline = SVMBaseline()
        with self.assertRaises(ValueError):
            baseline.fit(self.X, np.array(["bug", "feature"]))

    def test_predict_unfitted_raises(self):
        baseline = SVMBaseline()
        with self.assertRaises(RuntimeError):
            baseline.predict(self.X)

    def test_tune_and_fit_cv(self):
        baseline = SVMBaseline(random_state=42)
        cv_res = baseline.tune_and_fit(
            self.X,
            self.y,
            param_grid={"C": [0.1, 1.0]},
            cv=3,
            scoring="f1_macro",
        )

        self.assertTrue(baseline.is_fitted)
        self.assertIn("best_params", cv_res)
        self.assertIn("best_cv_macro_f1", cv_res)
        self.assertIn(baseline.C, [0.1, 1.0])
        self.assertGreater(cv_res["best_cv_macro_f1"], 0.0)

    def test_feature_coefficients(self):
        baseline = SVMBaseline(C=1.0)
        baseline.fit(self.X, self.y)

        feature_names = self.vectorizer.get_feature_names_out().tolist()
        coefs = baseline.get_feature_coefficients(feature_names, top_k=5)

        self.assertIn("bug", coefs)
        self.assertIn("feature", coefs)
        self.assertIn("other", coefs)

        bug_pos_terms = [t for t, _ in coefs["bug"]["positive"]]
        self.assertTrue("crash" in bug_pos_terms or "error" in bug_pos_terms)

        feature_pos_terms = [t for t, _ in coefs["feature"]["positive"]]
        self.assertTrue("feature" in feature_pos_terms or "support" in feature_pos_terms)

    def test_save_and_load_model(self):
        baseline = SVMBaseline(C=1.0, calibrate_probabilities=True)
        baseline.fit(self.X, self.y)
        expected_preds = baseline.predict(self.X)
        expected_probas = baseline.predict_proba(self.X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_svm.joblib"
            saved_file = baseline.save(save_path)
            self.assertTrue(saved_file.exists())

            loaded = SVMBaseline.load(save_path)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.classes_, baseline.classes_)

            loaded_preds = loaded.predict(self.X)
            loaded_probas = loaded.predict_proba(self.X)

            np.testing.assert_array_equal(expected_preds, loaded_preds)
            np.testing.assert_allclose(expected_probas, loaded_probas, atol=1e-5)

    def test_save_unfitted_raises(self):
        baseline = SVMBaseline()
        with self.assertRaises(RuntimeError):
            baseline.save("dummy.joblib")

    def test_load_non_existent_raises(self):
        with self.assertRaises(FileNotFoundError):
            SVMBaseline.load("missing_model_file.joblib")

    def test_live_training_and_cv(self):
        train_path = Path("data/processed/splits/train.csv")
        tfidf_path = Path("models/tfidf_vectorizer.joblib")
        if not train_path.exists() or not tfidf_path.exists():
            self.skipTest("train.csv or tfidf_vectorizer.joblib not found.")

        train_df = pd.read_csv(train_path)
        from ml.classification.features import TextFeaturePipeline

        pipeline = TextFeaturePipeline.load(tfidf_path)
        X_train = pipeline.transform(train_df)
        y_train = train_df["target_label"].to_numpy()

        baseline = SVMBaseline(random_state=42, calibrate_probabilities=True)
        cv_summary = baseline.tune_and_fit(
            X_train,
            y_train,
            param_grid={"C": [0.1, 0.5, 1.0]},
            cv=5,
            scoring="f1_macro",
        )

        self.assertTrue(baseline.is_fitted)
        self.assertGreaterEqual(cv_summary["best_cv_macro_f1"], 0.80)

        # Check calibrated probabilities on training set
        probas = baseline.predict_proba(X_train[:10])
        self.assertEqual(probas.shape, (10, 3))
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(10), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
