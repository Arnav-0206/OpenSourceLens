"""
Unit tests for IssueEmbeddingStore (Phase 7, Step 7.2).
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml.related_issues.store import IssueEmbeddingStore, compute_file_sha256


class TestIssueEmbeddingStore(unittest.TestCase):
    """Test suite for embedding persistence, checksum verification, and fast lookups."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir) / "test_store"

        # Create dummy data: 20 issues, 384 dimensions
        np.random.seed(42)
        raw_embs = np.random.randn(20, 384).astype(np.float32)
        self.dummy_embeddings = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

        self.dummy_df = pd.DataFrame({
            "issue_id": [1000 + i for i in range(20)],
            "issue_number": [i + 1 for i in range(20)],
            "repo_name": ["repo_a"] * 12 + ["repo_b"] * 8,
            "title": [f"Test issue title {i}" for i in range(20)],
            "clean_title": [f"test issue title {i}" for i in range(20)],
            "clean_body": [f"sample body description {i}" for i in range(20)],
            "canonical_label": ["bug"] * 10 + ["feature"] * 6 + ["other"] * 4,
            "state": ["closed"] * 15 + ["open"] * 5,
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_initialization(self):
        """Test initial unpopulated state of store."""
        store = IssueEmbeddingStore(self.store_dir)
        self.assertFalse(store.is_loaded)
        self.assertIsNone(store.embeddings)
        self.assertIsNone(store.metadata)
        self.assertEqual(store.store_dir, self.store_dir)

    def test_save_and_load_roundtrip(self):
        """Test complete serialization, manifest generation, and round-trip fidelity."""
        store = IssueEmbeddingStore(self.store_dir)
        bench = store.save(
            embeddings=self.dummy_embeddings,
            metadata_df=self.dummy_df,
            model_name="test-model-384",
        )

        self.assertEqual(bench["sample_count"], 20)
        self.assertEqual(bench["embedding_dimension"], 384)
        self.assertTrue((self.store_dir / "embeddings.npz").exists())
        self.assertTrue((self.store_dir / "metadata.parquet").exists())
        self.assertTrue((self.store_dir / "metadata.csv").exists())
        self.assertTrue((self.store_dir / "manifest.json").exists())
        self.assertTrue((self.store_dir / "README.md").exists())

        # Load into fresh store
        loaded_store = IssueEmbeddingStore(self.store_dir)
        loaded_store.load(verify_checksums=True)

        self.assertTrue(loaded_store.is_loaded)
        self.assertEqual(len(loaded_store.embeddings), 20)
        self.assertEqual(loaded_store.embeddings.shape[1], 384)
        np.testing.assert_allclose(self.dummy_embeddings, loaded_store.embeddings, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(loaded_store.metadata), 20)
        self.assertEqual(list(loaded_store.metadata["issue_id"]), list(self.dummy_df["issue_id"]))
        self.assertTrue(loaded_store.manifest["is_l2_normalized"])

    def test_tampered_file_detection(self):
        """Test that modifying a file triggers a cryptographic hash mismatch error."""
        store = IssueEmbeddingStore(self.store_dir)
        store.save(embeddings=self.dummy_embeddings, metadata_df=self.dummy_df)

        # Corrupt metadata.csv by appending a rogue byte
        csv_path = self.store_dir / "metadata.csv"
        with open(csv_path, "a") as f:
            f.write("\ncorrupted_line_entry")

        loader = IssueEmbeddingStore(self.store_dir)
        with self.assertRaises(ValueError) as ctx:
            loader.load(verify_checksums=True)
        self.assertIn("Cryptographic hash mismatch", str(ctx.exception))

    def test_get_issue_by_index_and_id(self):
        """Test fast O(1) ID lookup and positional indexing."""
        store = IssueEmbeddingStore(self.store_dir)
        store.save(embeddings=self.dummy_embeddings, metadata_df=self.dummy_df)

        # Index lookup
        emb_0, meta_0 = store.get_issue_by_index(0)
        np.testing.assert_allclose(emb_0, self.dummy_embeddings[0])
        self.assertEqual(meta_0["issue_id"], 1000)

        # ID lookup (int)
        emb_id, meta_id = store.get_issue_by_id(1005)
        np.testing.assert_allclose(emb_id, self.dummy_embeddings[5])
        self.assertEqual(meta_id["title"], "Test issue title 5")

        # ID lookup (string representation)
        emb_str, meta_str = store.get_issue_by_id("1005")
        np.testing.assert_allclose(emb_str, self.dummy_embeddings[5])

        # Out of bounds and missing key
        with self.assertRaises(IndexError):
            store.get_issue_by_index(999)
        with self.assertRaises(KeyError):
            store.get_issue_by_id(99999)

    def test_filter_by_repo(self):
        """Test repository-scoped subset filtering."""
        store = IssueEmbeddingStore(self.store_dir)
        store.save(embeddings=self.dummy_embeddings, metadata_df=self.dummy_df)

        sub_emb_a, sub_df_a, mask_a = store.filter_by_repo("repo_a")
        self.assertEqual(len(sub_emb_a), 12)
        self.assertEqual(len(sub_df_a), 12)
        self.assertEqual(mask_a.sum(), 12)
        self.assertTrue((sub_df_a["repo_name"] == "repo_a").all())

        sub_emb_b, sub_df_b, mask_b = store.filter_by_repo("repo_b")
        self.assertEqual(len(sub_emb_b), 8)
        self.assertEqual(len(sub_df_b), 8)
        self.assertEqual(mask_b.sum(), 8)

    def test_dimension_mismatch_exception(self):
        """Test error raised when embeddings count differs from metadata rows."""
        store = IssueEmbeddingStore(self.store_dir)
        with self.assertRaises(ValueError) as ctx:
            store.save(
                embeddings=self.dummy_embeddings[:10],
                metadata_df=self.dummy_df,
            )
        self.assertIn("does not match metadata rows", str(ctx.exception))

    def test_benchmark_io(self):
        """Test IO latency and throughput benchmarking helper."""
        store = IssueEmbeddingStore(self.store_dir)
        bench = store.benchmark_io(
            embeddings=self.dummy_embeddings,
            metadata_df=self.dummy_df,
            temp_dir=Path(self.temp_dir) / "io_bench",
        )
        self.assertIn("save_latency_seconds", bench)
        self.assertIn("load_latency_seconds", bench)
        self.assertIn("storage_size_mb", bench)
        self.assertEqual(bench["sample_count"], 20)
        self.assertGreater(bench["save_throughput_samples_per_sec"], 0)


if __name__ == "__main__":
    unittest.main()
