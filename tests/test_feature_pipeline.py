"""Unit tests for ml.classification.features (TextFeaturePipeline)."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.sparse

from ml.classification.features import TextFeaturePipeline


class TestTextFeaturePipeline(unittest.TestCase):
    """Test suite for TextFeaturePipeline class."""

    def setUp(self):
        """Construct synthetic dataset with known bug, feature, and other text."""
        self.mock_train_df = pd.DataFrame({
            "issue_id": [1, 2, 3, 4, 5, 6],
            "combined_text": [
                "Application crash segmentation fault memory error crash",
                "Fatal crash null pointer exception error",
                "Add new feature support export markdown",
                "Feature request support dark mode theme",
                "Refactor internal code cleanup",
                "Update documentation readme guide",
            ],
            "clean_title_word_count": [6, 6, 6, 6, 4, 4],
            "clean_body_word_count": [10, 8, 12, 14, 5, 7],
            "has_code_block": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "has_stack_trace": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "has_url": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "code_block_count": [2, 1, 0, 0, 0, 0],
            "url_count": [0, 0, 1, 0, 0, 1],
            "target_label": ["bug", "bug", "feature", "feature", "other", "other"],
        })

        self.mock_test_df = pd.DataFrame({
            "issue_id": [7, 8],
            "combined_text": [
                "Another crash memory leak error unknown_rare_word",
                "Please add support for json export xyz_unseen_term",
            ],
            "clean_title_word_count": [7, 7],
            "clean_body_word_count": [11, 9],
            "has_code_block": [1.0, 0.0],
            "has_stack_trace": [1.0, 0.0],
            "has_url": [0.0, 1.0],
            "code_block_count": [1, 0],
            "url_count": [0, 1],
            "target_label": ["bug", "feature"],
        })

    def test_fit_and_transform_text(self):
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        pipeline.fit(self.mock_train_df)

        self.assertTrue(pipeline.is_fitted)
        self.assertGreater(pipeline.vocabulary_size_, 0)

        X_train = pipeline.transform(self.mock_train_df)
        self.assertTrue(scipy.sparse.issparse(X_train))
        self.assertEqual(X_train.shape[0], len(self.mock_train_df))
        self.assertEqual(X_train.shape[1], pipeline.vocabulary_size_)

    def test_fit_transform_convenience(self):
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        X_train = pipeline.fit_transform(self.mock_train_df)

        self.assertTrue(pipeline.is_fitted)
        self.assertEqual(X_train.shape[0], len(self.mock_train_df))

    def test_transform_unseen_words(self):
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        pipeline.fit(self.mock_train_df)

        # Unseen words should be gracefully ignored by transform
        X_test = pipeline.transform(self.mock_test_df)
        self.assertEqual(X_test.shape[0], len(self.mock_test_df))
        self.assertEqual(X_test.shape[1], pipeline.vocabulary_size_)

    def test_leakage_free_vocabulary(self):
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        pipeline.fit(self.mock_train_df)

        # Terms occurring only in test set must NOT exist in the fitted vocabulary
        self.assertNotIn("unknown_rare_word", pipeline.feature_names_)
        self.assertNotIn("xyz_unseen_term", pipeline.feature_names_)

    def test_fallback_text_concatenation(self):
        df_no_combined = pd.DataFrame({
            "clean_title": ["System crash"],
            "clean_body": ["Crash on boot with memory fault"],
        })
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        pipeline.fit(self.mock_train_df)

        X = pipeline.transform(df_no_combined)
        self.assertEqual(X.shape[0], 1)
        self.assertEqual(X.shape[1], pipeline.vocabulary_size_)

    def test_fit_empty_df_raises(self):
        pipeline = TextFeaturePipeline()
        with self.assertRaises(ValueError):
            pipeline.fit(pd.DataFrame())

    def test_transform_unfitted_raises(self):
        pipeline = TextFeaturePipeline()
        with self.assertRaises(RuntimeError):
            pipeline.transform(self.mock_train_df)

    def test_transform_empty_df_raises(self):
        pipeline = TextFeaturePipeline(min_df=1)
        pipeline.fit(self.mock_train_df)
        with self.assertRaises(ValueError):
            pipeline.transform(pd.DataFrame())

    def test_ngram_extraction(self):
        pipeline = TextFeaturePipeline(ngram_range=(1, 2), min_df=1)
        pipeline.fit(self.mock_train_df)

        # Should extract bigrams like "segmentation fault", "memory error", etc.
        bigrams = [f for f in pipeline.feature_names_ if " " in f]
        self.assertGreater(len(bigrams), 0)
        self.assertIn("segmentation fault", bigrams)

    def test_use_auxiliary_features(self):
        pipeline = TextFeaturePipeline(
            max_features=100,
            min_df=1,
            use_auxiliary_features=True,
            auxiliary_features=["clean_title_word_count", "has_code_block", "has_stack_trace"],
        )
        pipeline.fit(self.mock_train_df)

        self.assertEqual(len(pipeline.fitted_auxiliary_cols_), 3)
        self.assertEqual(pipeline.total_features, pipeline.vocabulary_size_ + 3)

        X_train = pipeline.transform(self.mock_train_df)
        self.assertEqual(X_train.shape[1], pipeline.vocabulary_size_ + 3)

        X_test = pipeline.transform(self.mock_test_df)
        self.assertEqual(X_test.shape[1], pipeline.vocabulary_size_ + 3)

    def test_top_features_per_class(self):
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        X = pipeline.fit_transform(self.mock_train_df)

        top_terms = pipeline.get_top_features_per_class(
            X, self.mock_train_df["target_label"], top_k=5
        )

        self.assertIn("bug", top_terms)
        self.assertIn("feature", top_terms)
        self.assertIn("other", top_terms)

        bug_terms = [t for t, _ in top_terms["bug"]]
        self.assertTrue("crash" in bug_terms or "error" in bug_terms)

        feature_terms = [t for t, _ in top_terms["feature"]]
        self.assertTrue("feature" in feature_terms or "support" in feature_terms)

    def test_save_and_load_pipeline(self):
        pipeline = TextFeaturePipeline(max_features=100, min_df=1)
        pipeline.fit(self.mock_train_df)
        expected_matrix = pipeline.transform(self.mock_test_df).toarray()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tfidf_model.joblib"
            saved_file = pipeline.save(save_path)
            self.assertTrue(saved_file.exists())

            loaded_pipeline = TextFeaturePipeline.load(save_path)
            self.assertTrue(loaded_pipeline.is_fitted)
            self.assertEqual(loaded_pipeline.vocabulary_size_, pipeline.vocabulary_size_)

            loaded_matrix = loaded_pipeline.transform(self.mock_test_df).toarray()
            np.testing.assert_allclose(expected_matrix, loaded_matrix, atol=1e-6)

    def test_save_unfitted_raises(self):
        pipeline = TextFeaturePipeline()
        with self.assertRaises(RuntimeError):
            pipeline.save("dummy_path.joblib")

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            TextFeaturePipeline.load("non_existent_path.joblib")

    def test_live_split_integration(self):
        train_path = Path("data/processed/splits/train.csv")
        test_path = Path("data/processed/splits/test.csv")
        if not train_path.exists() or not test_path.exists():
            self.skipTest("Splits train.csv and test.csv not found.")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        pipeline = TextFeaturePipeline(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.85)
        pipeline.fit(train_df)

        self.assertEqual(pipeline.vocabulary_size_, 5000)

        X_train = pipeline.transform(train_df)
        X_test = pipeline.transform(test_df)

        self.assertEqual(X_train.shape, (1537, 5000))
        self.assertEqual(X_test.shape, (385, 5000))

        # Check top features for bug vs feature
        top_terms = pipeline.get_top_features_per_class(X_train, train_df["target_label"], top_k=10)
        bug_words = [t for t, _ in top_terms["bug"]]
        feature_words = [t for t, _ in top_terms["feature"]]

        # Meaningful domain terms should dominate
        self.assertTrue(any(w in bug_words for w in ["code_block", "test", "tests", "version", "ci", "flaking", "code"]))
        self.assertTrue(any(w in feature_words for w in ["feature", "feature request", "support", "motivation", "contribution", "add", "request"]))


if __name__ == "__main__":
    unittest.main()
