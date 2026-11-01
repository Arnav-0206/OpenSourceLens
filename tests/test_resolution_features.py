"""
Unit tests for OpenSourceLens Phase 8, Step 8.3: Feature Engineering and Leakage Audit.

Validates:
- Strict data leakage boundary enforcement (exclusion of post-creation metadata)
- Numeric correlation rankings (Pearson r, Spearman rho)
- Non-linear Mutual Information and ANOVA F-statistic ranking
- Feature matrix standardization (mean=0, std=1) and one-hot encoding
- Multicollinearity screening matrix properties (symmetry, unit diagonal)
- Publication Figure 30 generation and feature matrix Parquet serialization
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.resolution_analysis.features import (
    ELIGIBLE_CATEGORICAL_FEATURES,
    ELIGIBLE_NUMERIC_FEATURES,
    LEAKED_FEATURES,
    ResolutionFeatureEngineer,
)


class TestResolutionFeatures(unittest.TestCase):
    """Test suite for ResolutionFeatureEngineer."""

    def setUp(self):
        """Set up synthetic test dataset with eligible and leaked features."""
        np.random.seed(42)
        n_samples = 60

        records = []
        for i in range(n_samples):
            # Target duration (hours)
            h = float(np.random.exponential(50.0)) + 0.1
            log_h = float(np.log(h + 1.0))

            records.append({
                "issue_id": 100 + i,
                "issue_number": 1000 + i,
                # Eligible creation-time features
                "repo_name": ["microsoft/vscode", "pytorch/pytorch", "kubernetes/kubernetes"][i % 3],
                "canonical_label": ["bug", "feature", "other"][i % 3],
                "author_association": ["MEMBER", "CONTRIBUTOR", "NONE"][i % 3],
                "clean_title_word_count": int(np.random.randint(5, 20)),
                "clean_body_word_count": int(np.random.randint(20, 500)),
                "body_to_title_word_ratio": float(np.random.uniform(2.0, 30.0)),
                "avg_word_length_title": float(np.random.uniform(4.0, 8.0)),
                "avg_word_length_body": float(np.random.uniform(4.0, 8.0)),
                "has_code_block": bool(i % 2 == 0),
                "code_block_count": int(i % 4),
                "has_stack_trace": bool(i % 3 == 0),
                "has_url": bool(i % 2 == 1),
                "url_count": int(i % 3),
                "title_has_bracket_tag": bool(i % 4 == 0),
                "uppercase_ratio_title": float(np.random.uniform(0.0, 0.2)),
                "question_mark_count": int(i % 3),
                "exclamation_mark_count": int(i % 2),
                "is_internal_author": bool(i % 4 == 0),
                "is_bot_author": False,
                "created_hour": int(i % 24),
                "created_day_of_week": int(i % 7),
                "is_weekend": bool((i % 7) >= 5),
                "labels_count": int(np.random.randint(1, 5)),
                # Post-creation / Target LEAKED features (to test audit)
                "comments_count": int(np.random.poisson(5)),
                "reactions_count": int(np.random.poisson(2)),
                "closed_at": "2026-10-15T12:00:00Z",
                "state": "closed",
                "state_reason": "completed",
                "is_censored": False,
                "resolution_time_hours": h,
                "resolution_time_log_hours": log_h,
            })

        self.sample_df = pd.DataFrame(records)
        self.engineer = ResolutionFeatureEngineer(self.sample_df)

    def test_leakage_audit_isolation(self):
        """Verify leakage audit strictly separates eligible creation features from leaked features."""
        audit = ResolutionFeatureEngineer.audit_leakage(self.sample_df)

        self.assertTrue(audit["leakage_boundary_enforced"])
        self.assertGreater(audit["total_eligible_features"], 10)

        # Check that post-creation metadata is classified as leaked
        leaked = audit["excluded_leaked_features"]
        self.assertIn("comments_count", leaked)
        self.assertIn("reactions_count", leaked)
        self.assertIn("closed_at", leaked)
        self.assertIn("state_reason", leaked)

        # Check that no leaked feature is present in eligible sets
        for f in leaked:
            self.assertNotIn(f, audit["eligible_numeric_features"])
            self.assertNotIn(f, audit["eligible_categorical_features"])

    def test_compute_numeric_rankings(self):
        """Verify numeric correlations are within valid bounds [-1, 1] and properly sorted."""
        ranks = self.engineer.compute_numeric_rankings()

        self.assertGreater(len(ranks), 10)
        self.assertIn("feature", ranks.columns)
        self.assertIn("pearson_r", ranks.columns)
        self.assertIn("spearman_rho", ranks.columns)
        self.assertIn("abs_spearman", ranks.columns)

        # Bounds check
        self.assertTrue(np.all((ranks["pearson_r"] >= -1.0) & (ranks["pearson_r"] <= 1.0)))
        self.assertTrue(np.all((ranks["spearman_rho"] >= -1.0) & (ranks["spearman_rho"] <= 1.0)))

        # Sorted descending by absolute Spearman correlation
        abs_vals = ranks["abs_spearman"].values
        self.assertTrue(np.all(np.diff(abs_vals) <= 1e-7))

    def test_compute_all_feature_rankings(self):
        """Verify unified feature rankings with ANOVA F-statistic and Mutual Information."""
        all_ranks = self.engineer.compute_all_feature_rankings(random_state=42)

        self.assertGreater(len(all_ranks), 15)
        self.assertIn("mutual_info", all_ranks.columns)
        self.assertIn("f_statistic", all_ranks.columns)

        # Mutual information must be non-negative
        mi_scores = all_ranks["mutual_info"].values
        self.assertTrue(np.all(mi_scores >= 0.0))

        # Sorted descending by Mutual Information
        self.assertTrue(np.all(np.diff(mi_scores) <= 1e-7))

    def test_build_feature_matrix_scaling_and_encoding(self):
        """Verify feature matrix scaling (mean ~0, std ~1) and one-hot encoding."""
        X, y = self.engineer.build_feature_matrix()

        # Dimension checks
        self.assertEqual(len(X), len(self.sample_df))
        self.assertEqual(len(y), len(self.sample_df))
        self.assertFalse(X.isnull().any().any())

        # Check numeric scaling
        for col in ["clean_title_word_count", "clean_body_word_count"]:
            if col in X.columns:
                vals = X[col].values
                self.assertAlmostEqual(float(np.mean(vals)), 0.0, places=1)
                self.assertAlmostEqual(float(np.std(vals)), 1.0, places=1)

        # Check that one-hot categorical columns are present
        cat_dummies = [c for c in X.columns if "repo_name_" in c or "canonical_label_" in c]
        self.assertGreater(len(cat_dummies), 0)

        # Target verification
        self.assertEqual(y.name, "resolution_time_log_hours")

    def test_check_multicollinearity_matrix(self):
        """Verify collinearity matrix is symmetric with 1.0 on diagonal."""
        corr_matrix = self.engineer.check_multicollinearity(top_n=6)

        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        # Diagonal elements must be 1.0
        diag = np.diag(corr_matrix.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-3)
        # Symmetry check
        np.testing.assert_allclose(corr_matrix.values, corr_matrix.values.T, atol=1e-3)

    def test_generate_fig30_visualization(self):
        """Verify Figure 30 PNG is created and non-empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = Path(tmp_dir) / "test_fig30.png"
            res_path = self.engineer.generate_fig30_visualization(out_png)

            self.assertTrue(res_path.exists())
            self.assertGreater(res_path.stat().st_size, 10000)

    def test_export_artifacts(self):
        """Verify export produces valid Parquet matrix and JSON manifest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_parquet = Path(tmp_dir) / "test_features.parquet"
            out_manifest = Path(tmp_dir) / "test_manifest.json"

            p_feat, p_man = self.engineer.export_artifacts(out_parquet, out_manifest)

            self.assertTrue(p_feat.exists())
            self.assertTrue(p_man.exists())

            # Read parquet
            read_df = pd.read_parquet(p_feat)
            self.assertEqual(len(read_df), len(self.sample_df))
            self.assertIn("resolution_time_log_hours", read_df.columns)

            # Read manifest
            with open(p_man, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["phase"], "Phase 8: Resolution Analysis & Prediction")
            self.assertEqual(data["step"], "Step 8.3: Identify Useful Features")
            self.assertTrue(data["leakage_audit"]["leakage_boundary_enforced"])
            self.assertIn("top_features_by_mutual_info", data)


if __name__ == "__main__":
    unittest.main()
