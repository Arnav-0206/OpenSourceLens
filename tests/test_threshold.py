"""
Unit tests for SimilarityThresholdCalibrator (Phase 7, Step 7.5).
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.related_issues.store import IssueEmbeddingStore
from ml.related_issues.threshold import (
    DEFAULT_THRESHOLDS,
    SimilarityThresholdCalibrator,
)


class TestSimilarityThresholdCalibrator(unittest.TestCase):
    """Test suite for similarity threshold calibration, confidence tiers, and filtering."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.calibrator = SimilarityThresholdCalibrator()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classify_score_default_bands(self):
        """Test classifying scores across High, Medium, Low, and Unrelated bands."""
        # High tier (>= 0.70)
        res_high = self.calibrator.classify_score(0.75)
        self.assertEqual(res_high["tier"], "HIGH")
        self.assertEqual(res_high["confidence"], "High")
        self.assertTrue(res_high["is_related"])

        # Boundary at 0.70
        self.assertEqual(self.calibrator.classify_score(0.70)["tier"], "HIGH")

        # Medium tier (0.55 <= score < 0.70)
        res_med = self.calibrator.classify_score(0.62)
        self.assertEqual(res_med["tier"], "MEDIUM")
        self.assertEqual(res_med["confidence"], "Medium")
        self.assertTrue(res_med["is_related"])

        # Low tier (0.45 <= score < 0.55)
        res_low = self.calibrator.classify_score(0.48)
        self.assertEqual(res_low["tier"], "LOW")
        self.assertEqual(res_low["confidence"], "Low")
        self.assertTrue(res_low["is_related"])

        # Unrelated (< 0.45)
        res_unrel = self.calibrator.classify_score(0.35)
        self.assertEqual(res_unrel["tier"], "UNRELATED")
        self.assertEqual(res_unrel["confidence"], "None")
        self.assertFalse(res_unrel["is_related"])

    def test_custom_threshold_boundaries(self):
        """Test overriding default thresholds with custom values."""
        custom_th = {"high": 0.80, "medium": 0.60, "low": 0.40}
        res = self.calibrator.classify_score(0.75, thresholds=custom_th)
        self.assertEqual(res["tier"], "MEDIUM")

        calib_custom = SimilarityThresholdCalibrator(thresholds=custom_th)
        self.assertEqual(calib_custom.classify_score(0.35)["tier"], "UNRELATED")
        self.assertEqual(calib_custom.classify_score(0.42)["tier"], "LOW")

    def test_filter_and_enrich_candidates(self):
        """Test candidate list enrichment and low-score candidate filtration."""
        raw_candidates = [
            {"rank": 1, "issue_id": 101, "similarity_score": 0.82},
            {"rank": 2, "issue_id": 102, "similarity_score": 0.61},
            {"rank": 3, "issue_id": 103, "similarity_score": 0.49},
            {"rank": 4, "issue_id": 104, "similarity_score": 0.32},
        ]

        # Default min_threshold is 0.45 -> candidate 4 should be dropped
        enriched = self.calibrator.filter_and_enrich_candidates(raw_candidates)
        self.assertEqual(len(enriched), 3)

        # Check fields attached
        self.assertEqual(enriched[0]["confidence_tier"], "HIGH")
        self.assertEqual(enriched[1]["confidence_tier"], "MEDIUM")
        self.assertEqual(enriched[2]["confidence_tier"], "LOW")
        self.assertIn("action_recommendation", enriched[0])

        # Test stricter cutoff (e.g. min_threshold = 0.55)
        stricter = self.calibrator.filter_and_enrich_candidates(raw_candidates, min_threshold=0.55)
        self.assertEqual(len(stricter), 2)
        self.assertEqual(stricter[0]["issue_id"], 101)
        self.assertEqual(stricter[1]["issue_id"], 102)

    def test_sweep_threshold_tradeoffs(self):
        """Test threshold trade-off sweep on mock multi-repo dataset."""
        store_dir = Path(self.temp_dir) / "test_store_tradeoff"
        store = IssueEmbeddingStore(store_dir)

        # Synthetic embeddings: 20 issues, 2 repos
        np.random.seed(42)
        embs_a = np.zeros((10, 384), dtype=np.float32)
        embs_a[:, 0] = 1.0
        embs_a += np.random.normal(0, 0.05, (10, 384))
        embs_a /= np.linalg.norm(embs_a, axis=1, keepdims=True)

        embs_b = np.zeros((10, 384), dtype=np.float32)
        embs_b[:, 1] = 1.0
        embs_b += np.random.normal(0, 0.05, (10, 384))
        embs_b /= np.linalg.norm(embs_b, axis=1, keepdims=True)

        all_embs = np.vstack([embs_a, embs_b])
        all_df = pd.DataFrame({
            "issue_id": list(range(20)),
            "repo_name": ["repo_1"] * 10 + ["repo_2"] * 10,
            "title": [f"Title {i}" for i in range(20)],
            "canonical_label": ["bug"] * 5 + ["feature"] * 5 + ["bug"] * 5 + ["feature"] * 5,
        })
        store.save(all_embs, all_df)

        tradeoffs = self.calibrator.sweep_threshold_tradeoffs(
            store, sample_size=20, threshold_steps=np.array([0.40, 0.55, 0.70])
        )

        self.assertEqual(len(tradeoffs), 3)
        self.assertIn("noise_rejection_rate", tradeoffs.columns)
        self.assertIn("intra_retention_rate", tradeoffs.columns)
        self.assertIn("label_concordance", tradeoffs.columns)

        # Noise rejection should increase or stay constant with threshold
        rejections = tradeoffs["noise_rejection_rate"].tolist()
        self.assertTrue(all(rejections[i] <= rejections[i+1] for i in range(len(rejections)-1)))

        # Retention should decrease or stay constant with threshold
        retentions = tradeoffs["intra_retention_rate"].tolist()
        self.assertTrue(all(retentions[i] >= retentions[i+1] for i in range(len(retentions)-1)))

    def test_plot_threshold_calibration(self):
        """Test diagnostic figure generation."""
        fig_path = Path(self.temp_dir) / "test_fig27.png"
        dummy_df = pd.DataFrame({
            "threshold": [0.40, 0.55, 0.70, 0.80],
            "noise_rejection_rate": [0.95, 0.99, 1.0, 1.0],
            "intra_retention_rate": [0.30, 0.15, 0.05, 0.01],
            "label_concordance": [0.65, 0.75, 0.90, 1.0],
        })
        dummy_counts = {"HIGH": 30, "MEDIUM": 45, "LOW": 20, "UNRELATED": 5}

        out = self.calibrator.plot_threshold_calibration(
            tradeoff_df=dummy_df,
            tier_counts=dummy_counts,
            save_path=fig_path,
        )
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
