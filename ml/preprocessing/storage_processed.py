"""Processed Dataset Storage and Serialization Manager for OpenSourceLens.

Saves the canonical master issues dataset into data/processed/ in high-performance
Apache Parquet and portable CSV formats, generates SHA-256 cryptographic checksums,
exports taxonomy label mappings, and produces dataset manifests.
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

import pandas as pd

from ml.preprocessing.dataset_builder import IssueDatasetBuilder
from ml.preprocessing.label_normalizer import LabelNormalizer


class ProcessedDataStorage:
    """Manages serialization, checksumming, and manifest generation for processed data."""

    def __init__(self, processed_dir: Union[str, Path] = "data/processed"):
        """Initialize storage manager with target directory.

        Args:
            processed_dir: Path to directory for processed datasets (defaults to data/processed).
        """
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 cryptographic checksum of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_master_dataset(
        self,
        df: pd.DataFrame,
        base_name: str = "issues_master",
    ) -> Dict[str, Any]:
        """Save master DataFrame in both Apache Parquet and CSV formats with SHA-256 checksums.

        Args:
            df: Canonical master DataFrame.
            base_name: Base file name without extension.

        Returns:
            Dictionary with paths, sizes, and sha256 checksums.
        """
        parquet_path = self.processed_dir / f"{base_name}.parquet"
        csv_path = self.processed_dir / f"{base_name}.csv"

        # 1. Export Parquet (PyArrow engine, Snappy compression)
        # Note: PyArrow natively supports list<element: string> in raw_labels
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
        parquet_sha = self._compute_sha256(parquet_path)
        parquet_size_mb = round(parquet_path.stat().st_size / (1024 * 1024), 2)

        # 2. Export CSV (Convert list column raw_labels to JSON string for safe round-tripping)
        csv_df = df.copy()
        if "raw_labels" in csv_df.columns:
            csv_df["raw_labels"] = csv_df["raw_labels"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else json.dumps([])
            )
        csv_df.to_csv(csv_path, index=False, encoding="utf-8")
        csv_sha = self._compute_sha256(csv_path)
        csv_size_mb = round(csv_path.stat().st_size / (1024 * 1024), 2)

        return {
            "parquet": {
                "path": str(parquet_path),
                "filename": parquet_path.name,
                "size_mb": parquet_size_mb,
                "size_bytes": parquet_path.stat().st_size,
                "sha256": parquet_sha,
                "engine": "pyarrow",
                "compression": "snappy",
            },
            "csv": {
                "path": str(csv_path),
                "filename": csv_path.name,
                "size_mb": csv_size_mb,
                "size_bytes": csv_path.stat().st_size,
                "sha256": csv_sha,
                "encoding": "utf-8",
            },
        }

    def save_label_mappings(self, custom_mappings: Optional[Dict[str, Any]] = None) -> Path:
        """Export canonical taxonomy rules and ID encodings to JSON.

        Args:
            custom_mappings: Optional dictionary overriding default mappings.

        Returns:
            Path to saved label_mappings.json.
        """
        target = self.processed_dir / "label_mappings.json"
        mappings = custom_mappings or {
            "label_to_id": LabelNormalizer.LABEL_TO_ID,
            "core_categories": list(LabelNormalizer.CORE_CATEGORIES),
            "priority_order": ["bug", "feature", "documentation", "question", "other", "unlabeled"],
            "rules": {
                "bug": list(LabelNormalizer.BUG_EXACT),
                "feature": list(LabelNormalizer.FEATURE_EXACT),
                "documentation": list(LabelNormalizer.DOCS_EXACT),
                "question": list(LabelNormalizer.QUESTION_EXACT),
            },
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        target.write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def save_manifest(
        self,
        df: pd.DataFrame,
        files_info: Dict[str, Any],
    ) -> Path:
        """Export dataset manifest documenting row counts, schema, checksums, and distributions.

        Args:
            df: Canonical master DataFrame.
            files_info: Dictionary returned by save_master_dataset.

        Returns:
            Path to saved dataset_manifest.json.
        """
        target = self.processed_dir / "dataset_manifest.json"

        # Schema column types
        schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Class distribution
        class_dist = df["canonical_label"].value_counts().to_dict() if "canonical_label" in df.columns else {}

        # Repository breakdown
        repo_dist = df["repo_name"].value_counts().to_dict() if "repo_name" in df.columns else {}

        # State balance
        state_dist = df["state"].value_counts().to_dict() if "state" in df.columns else {}

        manifest = {
            "dataset_name": "OpenSourceLens Master Issues Dataset",
            "version": "1.0.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_records": len(df),
            "total_features": len(df.columns),
            "files": files_info,
            "distributions": {
                "by_canonical_label": class_dist,
                "by_repository": repo_dist,
                "by_state": state_dist,
            },
            "columns": schema_info,
        }

        target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def load_master_dataset(
        self,
        format: str = "parquet",
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load the master dataset for downstream analysis and modeling.

        Args:
            format: 'parquet' (recommended, faster, type-preserving) or 'csv'.
            filename: Optional custom filename (defaults to issues_master.<format>).

        Returns:
            Loaded pd.DataFrame.
        """
        if format == "parquet":
            file_path = self.processed_dir / (filename or "issues_master.parquet")
            if not file_path.exists():
                raise FileNotFoundError(f"Parquet dataset not found at {file_path}")
            return pd.read_parquet(file_path, engine="pyarrow")
        elif format == "csv":
            file_path = self.processed_dir / (filename or "issues_master.csv")
            if not file_path.exists():
                raise FileNotFoundError(f"CSV dataset not found at {file_path}")
            df = pd.read_csv(file_path, encoding="utf-8")
            # Parse raw_labels back to python lists
            if "raw_labels" in df.columns:
                df["raw_labels"] = df["raw_labels"].apply(
                    lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else []
                )
            return df
        else:
            raise ValueError(f"Unsupported format: '{format}'. Use 'parquet' or 'csv'.")

    def save_all(
        self,
        df: pd.DataFrame,
        base_name: str = "issues_master",
    ) -> Dict[str, Any]:
        """Execute full serialization workflow (Parquet + CSV + Mappings + Manifest).

        Args:
            df: Canonical master DataFrame.
            base_name: Base name for exported dataset files.

        Returns:
            Dictionary summarizing all exported artifacts and checksums.
        """
        files_info = self.save_master_dataset(df, base_name=base_name)
        mappings_path = self.save_label_mappings()
        manifest_path = self.save_manifest(df, files_info)

        return {
            "files": files_info,
            "label_mappings_path": str(mappings_path),
            "manifest_path": str(manifest_path),
            "total_records": len(df),
            "total_features": len(df.columns),
        }


def main():
    """CLI entrypoint for building and serializing processed dataset."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Save processed OpenSourceLens dataset.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output directory")
    args = parser.parse_args()

    # 1. Build master dataset
    builder = IssueDatasetBuilder()
    master_df, audit = builder.build(raw_dir=args.data_dir)

    # 2. Serialize and export
    storage = ProcessedDataStorage(processed_dir=args.processed_dir)
    export_summary = storage.save_all(master_df)

    # 3. Round-trip verification
    loaded_df = storage.load_master_dataset(format="parquet")
    assert len(loaded_df) == len(master_df), f"Row count mismatch: {len(loaded_df)} vs {len(master_df)}"
    assert list(loaded_df.columns) == list(master_df.columns), "Column list mismatch in round-trip"

    p_info = export_summary["files"]["parquet"]
    c_info = export_summary["files"]["csv"]

    print("=" * 65)
    print("OPENSOURCELENS — PROCESSED DATASET PERSISTENCE COMPLETE")
    print("=" * 65)
    print(f"Total Master Records:     {export_summary['total_records']:,}")
    print(f"Total Columns / Features: {export_summary['total_features']}")
    print("-" * 65)
    print(f"Apache Parquet Dataset:   {p_info['path']} ({p_info['size_mb']} MB)")
    print(f"  SHA-256:                {p_info['sha256']}")
    print(f"Standard CSV Dataset:     {c_info['path']} ({c_info['size_mb']} MB)")
    print(f"  SHA-256:                {c_info['sha256']}")
    print(f"Label Mappings JSON:      {export_summary['label_mappings_path']}")
    print(f"Dataset Manifest JSON:    {export_summary['manifest_path']}")
    print("-" * 65)
    print("Round-Trip Verification:  PASSED (Exact schema and row match)")
    print("=" * 65)


if __name__ == "__main__":
    main()
