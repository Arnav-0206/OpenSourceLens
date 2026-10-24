"""Unit Tests for Issue Embedding Pipeline (Phase 7, Step 7.1)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.related_issues.embedder import IssueEmbedder


class TestIssueEmbedder(unittest.TestCase):
    """Test suite for IssueEmbedder vector generation, normalization, and diagnostics."""

    @classmethod
    def setUpClass(cls):
        """Set up embedder instance shared across tests."""
        cls.embedder = IssueEmbedder(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)

    def test_embedder_initialization_and_dimensions(self):
        """Verify model dimension and properties."""
        self.assertEqual(self.embedder.dimension, 384)
        self.assertEqual(self.embedder.model_name, "all-MiniLM-L6-v2")
        self.assertTrue(self.embedder.normalize_embeddings)

    def test_format_issue_text(self):
        """Verify canonical text formatting for title and description."""
        formatted = self.embedder.format_issue_text("Crash on launch", "Segmentation fault in core")
        self.assertEqual(formatted, "Title: Crash on launch | Description: Segmentation fault in core")

        # Empty body
        formatted_no_body = self.embedder.format_issue_text("Short title", "")
        self.assertEqual(formatted_no_body, "Title: Short title")

    def test_encode_text_single_and_normalization(self):
        """Verify 1D vector shape and strict unit L2 normalization."""
        vec = self.embedder.encode_text("Title: Fix memory leak | Description: Buffer overflow in parser")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (384,))
        self.assertEqual(vec.dtype, np.float32)

        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_encode_texts_batch(self):
        """Verify batch encoding dimensions and pairwise distinctions."""
        texts = [
            "Title: Memory leak in worker | Description: Memory steadily grows over time",
            "Title: Add support for ARM64 | Description: Feature request for Apple Silicon builds",
            "Title: Typo in documentation | Description: Fixed grammar in quickstart guide",
        ]
        embeddings = self.embedder.encode_texts(texts, batch_size=2)
        self.assertEqual(embeddings.shape, (3, 384))

        # Check all rows are unit normalized
        norms = np.linalg.norm(embeddings, axis=1)
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=4)

        # Semantically distinct issues should have cosine similarity < 1.0
        dot_product_sim = np.dot(embeddings[0], embeddings[1])
        self.assertLess(dot_product_sim, 0.95)

    def test_encode_issues_df(self):
        """Verify DataFrame encoding and benchmark statistics."""
        df = pd.DataFrame({
            "clean_title": ["Null reference in parser", "Add CSV export button"],
            "clean_body": ["Crash traceback on line 42", "Allow users to download reports"],
        })
        embeddings, benchmark = self.embedder.encode_issues(df, batch_size=2, show_progress_bar=False)

        self.assertEqual(embeddings.shape, (2, 384))
        self.assertIn("throughput_samples_per_sec", benchmark)
        self.assertIn("total_encoding_seconds", benchmark)
        self.assertIn("memory_footprint_mb", benchmark)
        self.assertEqual(benchmark["sample_count"], 2)
        self.assertEqual(benchmark["embedding_dimension"], 384)

    def test_diagnostics_plot_and_json_export(self):
        """Verify diagnostic figure and benchmark JSON serialization."""
        np.random.seed(42)
        # Generate 10 mock normalized vectors
        mock_vecs = np.random.randn(10, 384).astype(np.float32)
        mock_vecs /= np.linalg.norm(mock_vecs, axis=1, keepdims=True)

        mock_benchmark = {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "sample_count": 10,
            "throughput_samples_per_sec": 120.5,
            "memory_footprint_mb": 0.015,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fig_path = tmp_path / "test_fig24.png"
            json_path = tmp_path / "test_benchmark.json"

            # Test plot export
            self.embedder.plot_embedding_diagnostics(mock_vecs, save_path=fig_path)
            self.assertTrue(fig_path.exists())
            self.assertGreater(fig_path.stat().st_size, 1000)

            # Test JSON export
            self.embedder.save_benchmark_summary(mock_benchmark, save_path=json_path)
            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["sample_count"], 10)
            self.assertEqual(loaded["embedding_dimension"], 384)


if __name__ == "__main__":
    unittest.main()
