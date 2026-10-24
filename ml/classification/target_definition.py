"""Classification Target Definition and Dataset Extraction Module for OpenSourceLens.

Establishes ground-truth target formulations for issue classification, resolves severe class
imbalance across canonical categories, enforces strict data leakage prevention by stripping
label-derived features from input matrices, and computes balanced class weights for classical
ML baselines (Logistic Regression and Support Vector Machines).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import pandas as pd
import numpy as np


class ClassificationTargetDefiner:
    """Defines supervised classification targets, isolates unlabeled pools, and prevents data leakage."""

    TARGET_MODES = ["core_3class", "binary_bug", "full_multiclass"]

    # Explicitly prohibited feature columns that would cause label leakage
    LEAKAGE_COLUMNS = [
        "raw_labels",
        "raw_labels_clean",
        "canonical_label",
        "canonical_label_id",
        "labels_count",
        "label_count",
        "is_classified_category",
        "is_unlabeled",
        "primary_label",
    ]

    # Safe auxiliary structural/tabular features permitted for feature fusion
    SAFE_AUXILIARY_FEATURES = [
        "clean_title_word_count",
        "clean_body_word_count",
        "has_code_block",
        "has_stack_trace",
        "has_url",
        "title_has_bracket_tag",
        "is_weekend",
        "code_block_count",
        "url_count",
        "uppercase_ratio_title",
        "avg_word_length_title",
        "avg_word_length_body",
    ]

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path] = "data/processed/issues_master.csv",
        target_mode: str = "core_3class",
        exclude_unlabeled: bool = True,
    ):
        """Initialize the target definer.

        Args:
            data: Loaded pd.DataFrame or path to processed dataset file (.csv or .parquet).
            target_mode: Strategy for target mapping ('core_3class', 'binary_bug', 'full_multiclass').
            exclude_unlabeled: Whether to separate unlabeled issues into a reserved inference pool.

        Raises:
            FileNotFoundError: If input file path does not exist.
            ValueError: If dataset is empty or target_mode is unrecognized.
        """
        mode_lower = target_mode.lower()
        if mode_lower not in self.TARGET_MODES:
            raise ValueError(f"Invalid target_mode '{target_mode}'. Expected one of {self.TARGET_MODES}.")

        self.target_mode = mode_lower
        self.exclude_unlabeled = exclude_unlabeled

        # Load dataset with resilient fallback
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found at: {path}")

            if path.suffix == ".parquet":
                try:
                    self.raw_df = pd.read_parquet(path)
                except Exception:
                    csv_alt = path.with_suffix(".csv")
                    if csv_alt.exists():
                        self.raw_df = pd.read_csv(csv_alt)
                    else:
                        raise
            elif path.suffix == ".csv":
                self.raw_df = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Expected .parquet or .csv")
            self.source_path = str(path)
        elif isinstance(data, pd.DataFrame):
            self.raw_df = data.copy()
            self.source_path = "DataFrame in-memory"
        else:
            raise TypeError(f"Expected DataFrame, str, or Path, got {type(data).__name__}")

        if self.raw_df.empty:
            raise ValueError("Provided dataset is empty (0 rows).")

        label_col = "canonical_label" if "canonical_label" in self.raw_df.columns else "primary_label"
        if label_col not in self.raw_df.columns:
            raise KeyError("Dataset must contain 'canonical_label' or 'primary_label'.")

        self.label_col = label_col

        # Ensure text columns and combined_text exist
        self._ensure_text_columns()

        # Partition unlabeled vs. labeled records
        unlabeled_mask = (self.raw_df[self.label_col] == "unlabeled") | (self.raw_df[self.label_col].isna())
        self.unlabeled_df = self.raw_df[unlabeled_mask].copy()

        if self.exclude_unlabeled:
            self.labeled_df = self.raw_df[~unlabeled_mask].copy()
        else:
            self.labeled_df = self.raw_df.copy()

        if self.labeled_df.empty:
            raise ValueError("No labeled records available after filtering unlabeled issues.")

        # Assign mapped target string labels
        self.labeled_df["target_label"] = self._map_target_labels(self.labeled_df[self.label_col], self.target_mode)

        # Build consistent label <-> ID mappings
        self._build_mappings()

        # Assign integer target IDs
        self.labeled_df["target_id"] = self.labeled_df["target_label"].map(self.label_to_id)

    def _ensure_text_columns(self):
        """Verify presence of text columns and construct combined_text if missing."""
        if "clean_title" not in self.raw_df.columns and "title" in self.raw_df.columns:
            self.raw_df["clean_title"] = self.raw_df["title"]
        elif "clean_title" not in self.raw_df.columns:
            self.raw_df["clean_title"] = ""

        if "clean_body" not in self.raw_df.columns and "body" in self.raw_df.columns:
            self.raw_df["clean_body"] = self.raw_df["body"]
        elif "clean_body" not in self.raw_df.columns:
            self.raw_df["clean_body"] = ""

        if "combined_text" not in self.raw_df.columns:
            t = self.raw_df["clean_title"].fillna("").astype(str)
            b = self.raw_df["clean_body"].fillna("").astype(str)
            self.raw_df["combined_text"] = (t + " " + b).str.strip()

    @staticmethod
    def _map_target_labels(series: pd.Series, mode: str) -> pd.Series:
        """Map raw canonical labels into configured modeling target classes."""
        clean_s = series.astype(str).str.lower().str.strip()

        if mode == "core_3class":
            # Map bug -> bug, feature -> feature, all other labeled categories (other, documentation, question) -> other
            mapping = {
                "bug": "bug",
                "feature": "feature",
                "other": "other",
                "documentation": "other",
                "question": "other",
            }
            return clean_s.map(lambda x: mapping.get(x, "other"))

        elif mode == "binary_bug":
            # Map bug -> bug, anything else -> non_bug
            return clean_s.apply(lambda x: "bug" if x == "bug" else "non_bug")

        elif mode == "full_multiclass":
            # Retain all original categories
            return clean_s

        raise ValueError(f"Unrecognized target mode: {mode}")

    def _build_mappings(self):
        """Construct deterministic class mappings and calculate balanced inverse-frequency weights."""
        classes = sorted(self.labeled_df["target_label"].unique().tolist())
        self.classes = classes
        self.label_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.id_to_label = {idx: cls_name for cls_name, idx in self.label_to_id.items()}

        # Compute balanced class weights: w_j = N / (K * n_j)
        n_samples = len(self.labeled_df)
        n_classes = len(classes)
        class_counts = self.labeled_df["target_label"].value_counts().to_dict()

        self.class_weights: Dict[int, float] = {}
        self.class_weights_named: Dict[str, float] = {}

        for cls_name, count in class_counts.items():
            cls_id = self.label_to_id[cls_name]
            weight = round(n_samples / (n_classes * count), 4) if count > 0 else 1.0
            self.class_weights[cls_id] = weight
            self.class_weights_named[cls_name] = weight

    def get_target_distribution(self) -> pd.DataFrame:
        """Return comprehensive breakdown of target classes, counts, percentages, and weights."""
        counts = self.labeled_df["target_label"].value_counts()
        total = len(self.labeled_df)

        rows = []
        for cls_name in self.classes:
            cnt = int(counts.get(cls_name, 0))
            pct = round(cnt / total * 100, 2)
            cls_id = self.label_to_id[cls_name]
            weight = self.class_weights[cls_id]

            rows.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "count": cnt,
                "percentage": pct,
                "balanced_weight": weight,
            })

        return pd.DataFrame(rows).sort_values(by="count", ascending=False).reset_index(drop=True)

    def get_class_mappings(self) -> Dict[str, Any]:
        """Return metadata dictionary of class encodings and weights."""
        return {
            "target_mode": self.target_mode,
            "classes": self.classes,
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
            "class_weights": self.class_weights,
            "class_weights_named": self.class_weights_named,
            "total_labeled_samples": len(self.labeled_df),
            "total_unlabeled_samples": len(self.unlabeled_df),
        }

    def prepare_labeled_dataset(self, include_auxiliary_features: bool = True) -> pd.DataFrame:
        """Extract clean modeling dataset with input features and target, stripping all label leakage.

        Args:
            include_auxiliary_features: Whether to retain safe structural/tabular features in DataFrame.

        Returns:
            Clean pd.DataFrame guaranteed free of data leakage columns.
        """
        # Base identification and text features
        base_cols = ["issue_id", "repo_name", "created_at", "clean_title", "clean_body", "combined_text"]
        available_base = [c for c in base_cols if c in self.labeled_df.columns]

        # Target columns
        target_cols = ["target_label", "target_id"]

        # Auxiliary features
        aux_cols: List[str] = []
        if include_auxiliary_features:
            for col in self.SAFE_AUXILIARY_FEATURES:
                if col in self.labeled_df.columns and col not in self.LEAKAGE_COLUMNS:
                    aux_cols.append(col)

        selected_cols = available_base + aux_cols + target_cols
        prepared = self.labeled_df[selected_cols].copy()

        # Assert zero data leakage
        for col in self.LEAKAGE_COLUMNS:
            if col in prepared.columns:
                prepared.drop(columns=[col], inplace=True)

        return prepared

    def get_unlabeled_pool(self) -> pd.DataFrame:
        """Return the reserved pool of unlabeled issues with all label leakage columns dropped."""
        base_cols = ["issue_id", "repo_name", "created_at", "clean_title", "clean_body", "combined_text"]
        available_base = [c for c in base_cols if c in self.unlabeled_df.columns]

        aux_cols = [c for c in self.SAFE_AUXILIARY_FEATURES if c in self.unlabeled_df.columns and c not in self.LEAKAGE_COLUMNS]
        selected_cols = available_base + aux_cols

        clean_unlabeled = self.unlabeled_df[selected_cols].copy()
        return clean_unlabeled

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level summary of the dataset splits and target configuration."""
        dist_df = self.get_target_distribution()
        return {
            "target_mode": self.target_mode,
            "total_raw_records": len(self.raw_df),
            "labeled_modeling_records": len(self.labeled_df),
            "unlabeled_inference_records": len(self.unlabeled_df),
            "classes_count": len(self.classes),
            "classes": self.classes,
            "class_distribution": dist_df.to_dict(orient="records"),
            "data_leakage_prevented": True,
        }
