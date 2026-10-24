"""
OpenSourceLens — Issue Embedding Store (Phase 7, Step 7.2)
Handles persistence, serialization, cryptographic integrity verification,
fast O(1) issue ID lookup, and repository filtering for issue dense embeddings.
"""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 cryptographic hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class IssueEmbeddingStore:
    """High-performance storage, verification, and retrieval interface for issue vector embeddings."""

    def __init__(self, store_dir: Union[str, Path] = "models/related_issues"):
        self.store_dir = Path(store_dir)
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None
        self.issue_ids: Optional[np.ndarray] = None
        self.manifest: Optional[Dict[str, Any]] = None
        self._id_to_idx: Dict[Any, int] = {}

    @property
    def is_loaded(self) -> bool:
        """Check if store has loaded embeddings and metadata."""
        return self.embeddings is not None and self.metadata is not None

    def save(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
        model_name: str = "all-MiniLM-L6-v2",
        store_dir: Optional[Union[str, Path]] = None,
        id_column: str = "issue_id",
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save embeddings array and aligned metadata DataFrame to disk.

        Args:
            embeddings: 2D numpy array of shape (N, D).
            metadata_df: DataFrame of N rows matching embeddings order.
            model_name: Name of embedding model used.
            store_dir: Optional override for storage directory.
            id_column: Name of column representing issue ID.
            extra_info: Additional metadata to store in manifest.

        Returns:
            Dictionary containing save performance benchmarks and manifest summary.
        """
        target_dir = Path(store_dir) if store_dir is not None else self.store_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        n_samples, dim = embeddings.shape
        if len(metadata_df) != n_samples:
            raise ValueError(
                f"Embedding count ({n_samples}) does not match metadata rows ({len(metadata_df)})."
            )

        if id_column in metadata_df.columns:
            issue_ids = metadata_df[id_column].to_numpy()
        else:
            issue_ids = np.arange(n_samples)

        t_save_start = time.perf_counter()

        # 1. Save compressed embeddings and IDs
        embeddings_path = target_dir / "embeddings.npz"
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings.astype(np.float32),
            issue_ids=issue_ids,
        )

        # 2. Save metadata in columnar Parquet and CSV
        parquet_path = target_dir / "metadata.parquet"
        csv_path = target_dir / "metadata.csv"

        metadata_df.to_parquet(parquet_path, index=False)
        metadata_df.to_csv(csv_path, index=False)

        save_time = time.perf_counter() - t_save_start

        # 3. Compute checksums
        checksums = {
            "embeddings.npz": {
                "sha256": compute_file_sha256(embeddings_path),
                "size_bytes": embeddings_path.stat().st_size,
            },
            "metadata.parquet": {
                "sha256": compute_file_sha256(parquet_path),
                "size_bytes": parquet_path.stat().st_size,
            },
            "metadata.csv": {
                "sha256": compute_file_sha256(csv_path),
                "size_bytes": csv_path.stat().st_size,
            },
        }

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = bool(np.allclose(norms, 1.0, atol=1e-3))

        # 4. Generate manifest
        manifest: Dict[str, Any] = {
            "title": "OpenSourceLens Related Issue Embedding Store Manifest",
            "model_name": model_name,
            "sample_count": n_samples,
            "embedding_dimension": dim,
            "is_l2_normalized": is_normalized,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "id_column": id_column,
            "files": checksums,
            "extra_info": extra_info or {},
        }

        manifest_path = target_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 5. Write documentation README.md
        readme_path = target_dir / "README.md"
        readme_content = f"""# OpenSourceLens Issue Embedding Store

## Overview
- **Model Architecture:** `{model_name}` (Sentence-Transformers)
- **Vector Dimension:** `{dim}`
- **Total Corpus Size:** `{n_samples:,}` issues
- **L2 Normalized:** `{is_normalized}` ($||\\vec{{v}}||_2 = 1.0$)
- **Generated At:** `{manifest['created_at']}`

## File Structure
- `embeddings.npz`: Compressed NumPy archive containing float32 `embeddings` matrix of shape `({n_samples}, {dim})` and aligned `issue_ids`.
- `metadata.parquet`: High-performance columnar issue metadata (titles, bodies, repos, labels, states).
- `metadata.csv`: Tabular mirror for inspection.
- `manifest.json`: Cryptographic SHA-256 verification hashes and dimensions.

## Checksums
- `embeddings.npz`: `{checksums['embeddings.npz']['sha256']}` ({checksums['embeddings.npz']['size_bytes'] / (1024*1024):.2f} MB)
- `metadata.parquet`: `{checksums['metadata.parquet']['sha256']}` ({checksums['metadata.parquet']['size_bytes'] / (1024*1024):.2f} MB)
- `metadata.csv`: `{checksums['metadata.csv']['sha256']}` ({checksums['metadata.csv']['size_bytes'] / (1024*1024):.2f} MB)
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        # Cache in memory
        self.store_dir = target_dir
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata_df.copy()
        self.issue_ids = issue_ids
        self.manifest = manifest
        self._rebuild_index()

        total_size_mb = sum(info["size_bytes"] for info in checksums.values()) / (1024 * 1024)

        return {
            "sample_count": n_samples,
            "embedding_dimension": dim,
            "save_time_seconds": round(save_time, 4),
            "throughput_samples_per_sec": round(n_samples / save_time, 2) if save_time > 0 else 0.0,
            "total_storage_mb": round(total_size_mb, 3),
            "manifest_path": str(manifest_path),
        }

    def load(
        self,
        store_dir: Optional[Union[str, Path]] = None,
        verify_checksums: bool = True,
    ) -> "IssueEmbeddingStore":
        """
        Load embeddings, metadata, and manifest from disk.

        Args:
            store_dir: Optional override for storage directory.
            verify_checksums: Whether to verify SHA-256 checksums before loading.

        Returns:
            Loaded IssueEmbeddingStore instance.
        """
        target_dir = Path(store_dir) if store_dir is not None else self.store_dir

        manifest_path = target_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found in {target_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if verify_checksums:
            for fname, meta in manifest.get("files", {}).items():
                fpath = target_dir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Required artifact '{fname}' missing from {target_dir}")
                actual_hash = compute_file_sha256(fpath)
                expected_hash = meta["sha256"]
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Cryptographic hash mismatch for '{fname}'! "
                        f"Expected {expected_hash}, got {actual_hash}."
                    )

        # Load embeddings
        embeddings_path = target_dir / "embeddings.npz"
        npz_data = np.load(embeddings_path)
        embeddings = npz_data["embeddings"]
        issue_ids = npz_data["issue_ids"]

        # Load metadata
        parquet_path = target_dir / "metadata.parquet"
        csv_path = target_dir / "metadata.csv"
        if parquet_path.exists():
            metadata = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            metadata = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No metadata file found in {target_dir}")

        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Loaded embeddings count ({len(embeddings)}) differs from metadata rows ({len(metadata)})."
            )

        self.store_dir = target_dir
        self.embeddings = embeddings
        self.issue_ids = issue_ids
        self.metadata = metadata
        self.manifest = manifest
        self._rebuild_index()

        return self

    def _rebuild_index(self) -> None:
        """Build O(1) hash map from issue_id to positional row index."""
        self._id_to_idx = {}
        if self.issue_ids is not None:
            for idx, iid in enumerate(self.issue_ids):
                # Map both string and native types for robust matching
                self._id_to_idx[iid] = idx
                self._id_to_idx[str(iid)] = idx

    def get_issue_by_index(self, idx: int) -> Tuple[np.ndarray, pd.Series]:
        """Retrieve embedding vector and metadata Series for a positional index."""
        if not self.is_loaded or self.embeddings is None or self.metadata is None:
            raise RuntimeError("Store has not loaded embeddings. Call load() first.")
        if idx < 0 or idx >= len(self.embeddings):
            raise IndexError(f"Index {idx} out of range (0 to {len(self.embeddings) - 1}).")

        return self.embeddings[idx], self.metadata.iloc[idx]

    def get_issue_by_id(self, issue_id: Union[int, str]) -> Tuple[np.ndarray, pd.Series]:
        """Retrieve embedding vector and metadata Series by issue ID in O(1) time."""
        if not self.is_loaded or self.embeddings is None or self.metadata is None:
            raise RuntimeError("Store has not loaded embeddings. Call load() first.")

        if issue_id not in self._id_to_idx:
            raise KeyError(f"Issue ID '{issue_id}' not found in embedding store.")

        idx = self._id_to_idx[issue_id]
        return self.embeddings[idx], self.metadata.iloc[idx]

    def filter_by_repo(
        self, repo_name: str
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Extract repository-specific subset of embeddings and metadata.

        Args:
            repo_name: Repository name (e.g., 'microsoft/vscode').

        Returns:
            Tuple of (subset_embeddings, subset_metadata_df, boolean_mask).
        """
        if not self.is_loaded or self.embeddings is None or self.metadata is None:
            raise RuntimeError("Store has not loaded embeddings. Call load() first.")

        if "repo_name" not in self.metadata.columns:
            raise KeyError("'repo_name' column not present in metadata.")

        mask = (self.metadata["repo_name"] == repo_name).to_numpy()
        subset_embeddings = self.embeddings[mask]
        subset_metadata = self.metadata[mask].reset_index(drop=True)

        return subset_embeddings, subset_metadata, mask

    def benchmark_io(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Profile save and load latency, throughput, and memory consumption."""
        benchmark_dir = Path(temp_dir) if temp_dir else self.store_dir / "_benchmark_tmp"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Profile Save
            t0 = time.perf_counter()
            save_result = self.save(
                embeddings=embeddings,
                metadata_df=metadata_df,
                store_dir=benchmark_dir,
            )
            save_sec = time.perf_counter() - t0

            # 2. Profile Load
            loader = IssueEmbeddingStore(store_dir=benchmark_dir)
            t1 = time.perf_counter()
            loader.load(verify_checksums=True)
            load_sec = time.perf_counter() - t1

            n = len(embeddings)
            total_bytes = sum(
                (benchmark_dir / f).stat().st_size
                for f in ["embeddings.npz", "metadata.parquet", "metadata.csv", "manifest.json"]
                if (benchmark_dir / f).exists()
            )

            benchmark = {
                "sample_count": n,
                "embedding_dimension": int(embeddings.shape[1]),
                "save_latency_seconds": round(save_sec, 4),
                "save_throughput_samples_per_sec": round(n / save_sec, 2) if save_sec > 0 else 0.0,
                "load_latency_seconds": round(load_sec, 4),
                "load_throughput_samples_per_sec": round(n / load_sec, 2) if load_sec > 0 else 0.0,
                "storage_size_mb": round(total_bytes / (1024 * 1024), 3),
                "embeddings_npz_size_mb": round(
                    (benchmark_dir / "embeddings.npz").stat().st_size / (1024 * 1024), 3
                ),
                "metadata_parquet_size_mb": round(
                    (benchmark_dir / "metadata.parquet").stat().st_size / (1024 * 1024), 3
                ),
            }
            return benchmark
        finally:
            if temp_dir is None and benchmark_dir.exists():
                import shutil
                shutil.rmtree(benchmark_dir, ignore_errors=True)


def run_embedding_storage_pipeline(
    issues_csv_path: Union[str, Path] = "data/processed/issues_master.csv",
    store_dir: Union[str, Path] = "models/related_issues",
    benchmark_out_path: Union[str, Path] = "data/processed/evaluation/embedding_storage_benchmark.json",
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[IssueEmbeddingStore, Dict[str, Any]]:
    """
    Execute end-to-end storage pipeline for Phase 7, Step 7.2.
    Encodes full corpus or loads existing embedder, persists artifacts to models/related_issues/,
    and verifies checksums and IO benchmarks.
    """
    from ml.related_issues.embedder import IssueEmbedder

    print(f"\n=======================================================")
    print(f"Starting Embedding Storage Pipeline (Phase 7, Step 7.2)")
    print(f"Source: {issues_csv_path} | Target: {store_dir}")
    print(f"=======================================================\n")

    df = pd.read_csv(issues_csv_path)
    n_samples = len(df)
    print(f"Loaded {n_samples} historical issues.")

    print(f"Encoding issues using {model_name}...")
    embedder = IssueEmbedder(model_name=model_name, normalize_embeddings=True)
    embeddings, embed_bench = embedder.encode_issues(
        df=df,
        title_col="clean_title",
        body_col="clean_body",
        batch_size=64,
        show_progress_bar=True,
    )

    store = IssueEmbeddingStore(store_dir=store_dir)
    print(f"\nPersisting embeddings and metadata to {store_dir}...")
    save_result = store.save(
        embeddings=embeddings,
        metadata_df=df,
        model_name=model_name,
        extra_info={"embed_benchmark": embed_bench},
    )

    print("Verifying storage artifacts and cryptographic checksums...")
    verifier = IssueEmbeddingStore(store_dir=store_dir)
    t0_load = time.perf_counter()
    verifier.load(verify_checksums=True)
    load_time = time.perf_counter() - t0_load

    np.testing.assert_allclose(embeddings, verifier.embeddings, rtol=1e-5, atol=1e-5)
    assert len(verifier.metadata) == n_samples
    print(f"Verification successful! Round-trip embeddings and metadata are identical.")

    # IO Benchmark
    bench_data = {
        "title": "OpenSourceLens Issue Embedding Storage Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "sample_count": n_samples,
        "embedding_dimension": int(embeddings.shape[1]),
        "save_time_seconds": save_result["save_time_seconds"],
        "save_throughput_samples_per_sec": save_result["throughput_samples_per_sec"],
        "load_time_seconds": round(load_time, 4),
        "load_throughput_samples_per_sec": round(n_samples / load_time, 2) if load_time > 0 else 0.0,
        "storage_size_mb": save_result["total_storage_mb"],
        "files": verifier.manifest["files"],
    }

    bench_path = Path(benchmark_out_path)
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    print(f"Storage benchmark saved to: {bench_path}")

    print("\n--- Embedding Storage Summary ---")
    print(f"Corpus Size:               {bench_data['sample_count']} issues")
    print(f"Vector Dimensions:         {bench_data['embedding_dimension']}")
    print(f"Save Time:                 {bench_data['save_time_seconds']:.4f} s ({bench_data['save_throughput_samples_per_sec']:.1f} issues/s)")
    print(f"Load Time:                 {bench_data['load_time_seconds']:.4f} s ({bench_data['load_throughput_samples_per_sec']:.1f} issues/s)")
    print(f"Total Disk Footprint:      {bench_data['storage_size_mb']:.2f} MB")
    print("---------------------------------\n")

    return store, bench_data


if __name__ == "__main__":
    run_embedding_storage_pipeline()
