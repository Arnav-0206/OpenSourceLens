"""Dataset Splitting and Training/Test Strategy Module for OpenSourceLens.

Provides deterministic, reproducible stratified and temporal dataset partitioning,
Stratified K-Fold cross-validation generators, comprehensive split diagnostics,
and serialization with cryptographic checksum integrity verification to prevent
data leakage across classical ML and transformer models.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Iterator
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.classification.target_definition import ClassificationTargetDefiner


class DatasetSplitter:
    """Manages reproducible dataset partitioning, stratification, cross-validation, and split diagnostics."""

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path, ClassificationTargetDefiner],
        target_col: str = "target_label",
        random_state: int = 42,
    ):
        """Initialize the DatasetSplitter.

        Args:
            data: A prepared DataFrame, path to dataset file (.csv or .parquet), or ClassificationTargetDefiner instance.
            target_col: Target column to stratify on (default: 'target_label').
            random_state: Random seed for deterministic reproducibility (default: 42).

        Raises:
            FileNotFoundError: If input file path does not exist.
            ValueError: If dataset is empty or target_col is missing.
        """
        self.random_state = random_state
        self.target_col = target_col
        self.metadata: Dict[str, Any] = {}

        if isinstance(data, ClassificationTargetDefiner):
            self.df = data.prepare_labeled_dataset(include_auxiliary_features=True)
            self.source_path = data.source_path
            self.metadata["target_mode"] = data.target_mode
            self.metadata["classes"] = data.classes
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.source_path = "DataFrame in-memory"
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found at: {path}")

            if path.suffix == ".parquet":
                try:
                    self.df = pd.read_parquet(path)
                except Exception:
                    csv_alt = path.with_suffix(".csv")
                    if csv_alt.exists():
                        self.df = pd.read_csv(csv_alt)
                    else:
                        raise
            elif path.suffix == ".csv":
                self.df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Expected .parquet or .csv")
            self.source_path = str(path)
        else:
            raise TypeError(
                f"Expected DataFrame, str, Path, or ClassificationTargetDefiner, got {type(data).__name__}"
            )

        if self.df.empty:
            raise ValueError("Input dataset is empty (0 rows).")

        if self.target_col not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in dataset. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Drop any leakage columns if present in raw inputs
        for leak_col in ClassificationTargetDefiner.LEAKAGE_COLUMNS:
            if leak_col in self.df.columns and leak_col != self.target_col:
                self.df.drop(columns=[leak_col], inplace=True)

    def split_train_test(
        self,
        test_size: float = 0.20,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into deterministic Train and Test subsets.

        Args:
            test_size: Proportion of dataset allocated to held-out test set (e.g. 0.20 for 80/20).
            stratify: Whether to preserve class proportions across splits (default: True).

        Returns:
            Tuple of (train_df, test_df) with reset indices.

        Raises:
            ValueError: If test_size is not strictly between 0 and 1.
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        strat_target = self.df[self.target_col] if stratify else None

        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=strat_target,
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def split_train_val_test(
        self,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into 3-way Train, Validation, and Test subsets.

        Args:
            val_size: Proportion of overall dataset allocated to validation set (default: 0.15).
            test_size: Proportion of overall dataset allocated to test set (default: 0.15).
            stratify: Whether to preserve class proportions across all splits (default: True).

        Returns:
            Tuple of (train_df, val_df, test_df) with reset indices.

        Raises:
            ValueError: If val_size or test_size are invalid or their sum is >= 1.0.
        """
        if not (0.0 < val_size < 1.0) or not (0.0 < test_size < 1.0):
            raise ValueError("val_size and test_size must each be between 0 and 1.")

        if val_size + test_size >= 1.0:
            raise ValueError(
                f"Sum of val_size ({val_size}) and test_size ({test_size}) must be strictly < 1.0."
            )

        strat_target = self.df[self.target_col] if stratify else None

        # First partition held-out test set
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=strat_target,
        )

        # Then partition train and val sets from remainder
        rel_val_size = val_size / (1.0 - test_size)
        strat_sub_target = train_val_df[self.target_col] if stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=rel_val_size,
            random_state=self.random_state,
            stratify=strat_sub_target,
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def split_temporal(
        self,
        date_col: str = "created_at",
        test_size: float = 0.20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset chronologically into Train (earlier issues) and Test (later issues).

        Used to evaluate temporal distribution drift and model performance over time.

        Args:
            date_col: Column containing timestamp information (default: 'created_at').
            test_size: Proportion of latest issues reserved for test (default: 0.20).

        Returns:
            Tuple of (train_df, test_df) chronologically ordered with reset indices.

        Raises:
            KeyError: If date_col is missing from dataset.
            ValueError: If test_size is not strictly between 0 and 1.
        """
        if date_col not in self.df.columns:
            raise KeyError(f"Date column '{date_col}' not found in dataset columns: {list(self.df.columns)}")

        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        df_sorted = self.df.copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], utc=True)
        df_sorted = df_sorted.sort_values(by=date_col, ascending=True).reset_index(drop=True)

        n_train = int(len(df_sorted) * (1.0 - test_size))
        train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[n_train:].copy().reset_index(drop=True)

        return train_df, test_df

    def get_stratified_kfold(
        self,
        train_df: Optional[pd.DataFrame] = None,
        n_splits: int = 5,
        shuffle: bool = True,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_idx, val_idx) fold index pairs for stratified cross-validation.

        Args:
            train_df: DataFrame to split. If None, uses the initialized dataset.
            n_splits: Number of cross-validation folds (default: 5).
            shuffle: Whether to shuffle before splitting (default: True).

        Yields:
            Tuples of (train_indices, val_indices) numpy integer arrays.

        Raises:
            ValueError: If any class in the dataset has fewer samples than n_splits.
        """
        target_df = train_df if train_df is not None else self.df
        class_counts = target_df[self.target_col].value_counts()
        min_count = class_counts.min()

        if min_count < n_splits:
            raise ValueError(
                f"Cannot create {n_splits}-fold StratifiedKFold because minority class "
                f"'{class_counts.idxmin()}' only has {min_count} samples (minimum required: {n_splits})."
            )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        y = target_df[self.target_col].to_numpy()
        X = np.zeros(len(target_df))  # Dummy feature matrix for generator

        yield from skf.split(X, y)

    def get_split_diagnostics(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Compute detailed statistical diagnostics across dataset splits.

        Evaluates class distributions, repository proportions, word counts, and temporal boundaries.

        Args:
            train_df: Training partition.
            test_df: Test partition.
            val_df: Optional validation partition.

        Returns:
            Dictionary of diagnostic statistics and drift assessments.
        """
        splits: Dict[str, pd.DataFrame] = {"train": train_df, "test": test_df}
        if val_df is not None:
            splits["val"] = val_df

        total_samples = sum(len(df) for df in splits.values())
        diagnostics: Dict[str, Any] = {
            "total_samples": total_samples,
            "split_counts": {name: len(df) for name, df in splits.items()},
            "split_percentages": {
                name: round(len(df) / total_samples * 100, 2) for name, df in splits.items()
            },
            "class_distribution": {},
            "repository_distribution": {},
            "temporal_bounds": {},
            "text_length_stats": {},
        }

        # Check for zero overlap across splits if issue_id is present
        id_col = "issue_id" if "issue_id" in train_df.columns else None
        if id_col:
            train_ids = set(train_df[id_col])
            test_ids = set(test_df[id_col])
            overlap_train_test = len(train_ids.intersection(test_ids))
            overlap_val = 0
            if val_df is not None:
                val_ids = set(val_df[id_col])
                overlap_val = len(train_ids.intersection(val_ids)) + len(test_ids.intersection(val_ids))

            diagnostics["zero_overlap_verified"] = (overlap_train_test == 0) and (overlap_val == 0)
            diagnostics["overlap_count"] = overlap_train_test + overlap_val

        # Class breakdown per split
        classes = sorted(self.df[self.target_col].dropna().unique().tolist())
        for cls in classes:
            diagnostics["class_distribution"][cls] = {}
            for name, df in splits.items():
                cnt = int((df[self.target_col] == cls).sum())
                pct = round(cnt / len(df) * 100, 2) if len(df) > 0 else 0.0
                diagnostics["class_distribution"][cls][name] = {
                    "count": cnt,
                    "percentage": pct,
                }

        # Repository breakdown per split
        if "repo_name" in train_df.columns:
            repos = sorted(self.df["repo_name"].dropna().unique().tolist())
            for repo in repos:
                diagnostics["repository_distribution"][repo] = {}
                for name, df in splits.items():
                    cnt = int((df["repo_name"] == repo).sum())
                    pct = round(cnt / len(df) * 100, 2) if len(df) > 0 else 0.0
                    diagnostics["repository_distribution"][repo][name] = {
                        "count": cnt,
                        "percentage": pct,
                    }

        # Temporal bounds per split
        if "created_at" in train_df.columns:
            for name, df in splits.items():
                if not df.empty and df["created_at"].notna().any():
                    ts = pd.to_datetime(df["created_at"], utc=True)
                    diagnostics["temporal_bounds"][name] = {
                        "min": ts.min().isoformat(),
                        "max": ts.max().isoformat(),
                    }

        # Text length summary
        if "clean_title_word_count" in train_df.columns:
            for name, df in splits.items():
                diagnostics["text_length_stats"][name] = {
                    "title_word_count_mean": round(float(df["clean_title_word_count"].mean()), 2),
                    "body_word_count_mean": (
                        round(float(df["clean_body_word_count"].mean()), 2)
                        if "clean_body_word_count" in df.columns
                        else None
                    ),
                }

        return diagnostics

    def save_splits(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        output_dir: Union[str, Path] = "data/processed/splits",
    ) -> Dict[str, Path]:
        """Save train, test, and optional val partitions to disk alongside an integrity manifest.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            val_df: Optional Validation DataFrame.
            output_dir: Target directory path (default: 'data/processed/splits').

        Returns:
            Dictionary mapping artifact names to their saved file paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}
        file_hashes: Dict[str, str] = {}

        # Save train
        train_path = out_path / "train.csv"
        train_df.to_csv(train_path, index=False)
        saved_files["train"] = train_path
        file_hashes["train"] = self._compute_file_sha256(train_path)

        # Save test
        test_path = out_path / "test.csv"
        test_df.to_csv(test_path, index=False)
        saved_files["test"] = test_path
        file_hashes["test"] = self._compute_file_sha256(test_path)

        # Save val if present
        if val_df is not None:
            val_path = out_path / "val.csv"
            val_df.to_csv(val_path, index=False)
            saved_files["val"] = val_path
            file_hashes["val"] = self._compute_file_sha256(val_path)

        # Compute diagnostics for manifest
        diag = self.get_split_diagnostics(train_df, test_df, val_df)

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_col": self.target_col,
            "random_state": self.random_state,
            "source_path": self.source_path,
            "total_records": len(train_df) + len(test_df) + (len(val_df) if val_df is not None else 0),
            "files": {name: str(p.name) for name, p in saved_files.items()},
            "sha256_checksums": file_hashes,
            "diagnostics": diag,
        }

        # Store issue_ids for exact entity verification if available
        if "issue_id" in train_df.columns:
            manifest["issue_ids"] = {
                "train": train_df["issue_id"].tolist(),
                "test": test_df["issue_id"].tolist(),
            }
            if val_df is not None:
                manifest["issue_ids"]["val"] = val_df["issue_id"].tolist()

        manifest_path = out_path / "split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        saved_files["manifest"] = manifest_path
        return saved_files

    @classmethod
    def load_splits(
        cls,
        split_dir: Union[str, Path] = "data/processed/splits",
        verify_integrity: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load persisted train, test, and optional validation datasets from disk.

        Args:
            split_dir: Directory containing persisted splits and manifest.
            verify_integrity: If True, verifies SHA256 checksums match the manifest.

        Returns:
            Dictionary containing 'train', 'test', and optionally 'val' DataFrames.

        Raises:
            FileNotFoundError: If required files (train.csv, test.csv) do not exist.
            ValueError: If SHA256 checksum verification fails.
        """
        s_path = Path(split_dir)
        train_path = s_path / "train.csv"
        test_path = s_path / "test.csv"
        val_path = s_path / "val.csv"
        manifest_path = s_path / "split_manifest.json"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing train split at: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test split at: {test_path}")

        if verify_integrity and manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            checksums = manifest.get("sha256_checksums", {})
            if "train" in checksums:
                current_train_hash = cls._compute_file_sha256(train_path)
                if current_train_hash != checksums["train"]:
                    raise ValueError(
                        f"Train set SHA256 mismatch! Expected {checksums['train']}, got {current_train_hash}."
                    )
            if "test" in checksums:
                current_test_hash = cls._compute_file_sha256(test_path)
                if current_test_hash != checksums["test"]:
                    raise ValueError(
                        f"Test set SHA256 mismatch! Expected {checksums['test']}, got {current_test_hash}."
                    )
            if val_path.exists() and "val" in checksums:
                current_val_hash = cls._compute_file_sha256(val_path)
                if current_val_hash != checksums["val"]:
                    raise ValueError(
                        f"Val set SHA256 mismatch! Expected {checksums['val']}, got {current_val_hash}."
                    )

        result: Dict[str, pd.DataFrame] = {
            "train": pd.read_csv(train_path),
            "test": pd.read_csv(test_path),
        }
        if val_path.exists():
            result["val"] = pd.read_csv(val_path)

        return result

    @staticmethod
    def _compute_file_sha256(file_path: Union[str, Path]) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
