"""Transformer Dataset Preparation for OpenSourceLens (Phase 6 Deep Learning).

Constructs stratified, leakage-free train, validation, and test datasets for transformer fine-tuning.
Formats structured natural language issue text, encodes integer class labels, implements a reusable
PyTorch IssueDataset class, and generates publication-quality distribution visualizations and manifests.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor


class IssueDataset(Dataset):
    """PyTorch Dataset wrapper for GitHub issue text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        issue_ids: Optional[List[str]] = None,
    ):
        """Initialize the dataset.

        Args:
            texts: List of formatted natural language issue text strings.
            labels: List of integer target labels (0: bug, 1: feature, 2: other).
            issue_ids: Optional list of issue identifiers.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Mismatch between texts length ({len(texts)}) and labels length ({len(labels)})"
            )

        self.texts = texts
        self.labels = labels
        self.issue_ids = issue_ids if issue_ids is not None else [str(i) for i in range(len(texts))]

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample at given index."""
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx]),
            "issue_id": str(self.issue_ids[idx]),
        }


class TransformerDatasetBuilder:
    """Orchestrates stratified dataset splitting and formatting for pretrained transformers."""

    LABEL2ID = {"bug": 0, "feature": 1, "other": 2}
    ID2LABEL = {0: "bug", 1: "feature", 2: "other"}
    TARGET_CLASSES = ["bug", "feature", "other"]

    def __init__(
        self,
        val_ratio: float = 0.15,
        random_state: int = 42,
        use_enhanced_preprocessing: bool = True,
    ):
        """Initialize the builder.

        Args:
            val_ratio: Fraction of training data to allocate for validation (default: 0.15).
            random_state: Random state for deterministic stratified splitting (default: 42).
            use_enhanced_preprocessing: Apply software domain text normalization (default: True).
        """
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.preprocessor = EnhancedIssueTextPreprocessor() if use_enhanced_preprocessing else None

    def format_text(self, title: str, body: str) -> str:
        """Format issue title and description with structured natural language delimiters."""
        if self.use_enhanced_preprocessing and self.preprocessor is not None:
            t = self.preprocessor.clean_text(title or "", is_title=True)
            b = self.preprocessor.clean_text(body or "", is_title=False)
        else:
            t = (title or "").strip()
            b = (body or "").strip()

        # Structured natural language format with explicit semantic boundaries
        if t and b:
            return f"Title: {t} | Description: {b}"
        elif t:
            return f"Title: {t}"
        else:
            return f"Description: {b}"

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Transform raw issue columns into standardized transformer input format."""
        if label_col is None:
            if "target_label" in df.columns:
                label_col = "target_label"
            elif "canonical_label" in df.columns:
                label_col = "canonical_label"
            elif "primary_label" in df.columns:
                label_col = "primary_label"
            else:
                raise ValueError("DataFrame must contain 'target_label', 'canonical_label', or 'primary_label'")

        t_col = "clean_title" if "clean_title" in df.columns else ("title" if "title" in df.columns else df.columns[0])
        b_col = "clean_body" if "clean_body" in df.columns else ("body" if "body" in df.columns else t_col)
        id_col = "issue_id" if "issue_id" in df.columns else None

        records = []
        for idx, row in df.iterrows():
            raw_title = str(row.get(t_col, ""))
            raw_body = str(row.get(b_col, ""))
            formatted_text = self.format_text(raw_title, raw_body)
            lbl_str = str(row.get(label_col, ""))

            if lbl_str not in self.LABEL2ID:
                raise ValueError(f"Unknown label '{lbl_str}'. Expected one of {list(self.LABEL2ID.keys())}")

            lbl_id = self.LABEL2ID[lbl_str]
            issue_id = str(row.get(id_col, f"issue_{idx}")) if id_col else f"issue_{idx}"
            repo_name = str(row.get("repo_name", "unknown"))

            records.append(
                {
                    "issue_id": issue_id,
                    "repo_name": repo_name,
                    "text": formatted_text,
                    "target_label": lbl_str,
                    "label": lbl_id,
                    "word_count": len(formatted_text.split()),
                }
            )

        return pd.DataFrame(records)

    def create_pytorch_dataset(self, df: pd.DataFrame) -> IssueDataset:
        """Construct an IssueDataset instance directly from a prepared DataFrame."""
        return IssueDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            issue_ids=df["issue_id"].tolist(),
        )

    def build_splits(
        self,
        train_path: Union[str, Path],
        test_path: Union[str, Path],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split training set into train/val and prepare held-out test split.

        Args:
            train_path: Path to canonical training CSV.
            test_path: Path to canonical test CSV.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        raw_train_df = pd.read_csv(train_path)
        raw_test_df = pd.read_csv(test_path)

        # 1. Stratified 85/15 train/val split from training data
        label_col = "target_label" if "target_label" in raw_train_df.columns else "canonical_label"
        y_train = raw_train_df[label_col].values

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.val_ratio,
            random_state=self.random_state,
        )
        train_idx, val_idx = next(sss.split(raw_train_df, y_train))

        split_train_raw = raw_train_df.iloc[train_idx].copy().reset_index(drop=True)
        split_val_raw = raw_train_df.iloc[val_idx].copy().reset_index(drop=True)

        # 2. Format texts and integer label IDs
        train_df = self.prepare_dataframe(split_train_raw)
        val_df = self.prepare_dataframe(split_val_raw)
        test_df = self.prepare_dataframe(raw_test_df)

        # 3. Data Leakage Verification
        train_ids = set(train_df["issue_id"])
        val_ids = set(val_df["issue_id"])
        test_ids = set(test_df["issue_id"])

        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)

        if train_val_overlap:
            raise RuntimeError(f"Data leakage detected! {len(train_val_overlap)} issues overlap between train and val")
        if train_test_overlap:
            raise RuntimeError(f"Data leakage detected! {len(train_test_overlap)} issues overlap between train and test")
        if val_test_overlap:
            raise RuntimeError(f"Data leakage detected! {len(val_test_overlap)} issues overlap between val and test")

        return train_df, val_df, test_df

    def plot_distributions(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: Union[str, Path],
    ) -> Path:
        """Plot dataset class balance and text length distributions (fig17)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        # Panel 1: Class Distribution by Split
        classes = self.TARGET_CLASSES
        x = np.arange(len(classes))
        width = 0.25

        train_counts = [int(np.sum(train_df["target_label"] == cls)) for cls in classes]
        val_counts = [int(np.sum(val_df["target_label"] == cls)) for cls in classes]
        test_counts = [int(np.sum(test_df["target_label"] == cls)) for cls in classes]

        train_pcts = [c / len(train_df) * 100 for c in train_counts]
        val_pcts = [c / len(val_df) * 100 for c in val_counts]
        test_pcts = [c / len(test_df) * 100 for c in test_counts]

        axes[0].bar(x - width, train_pcts, width, label=f"Train ({len(train_df)})", color="#1d4ed8", edgecolor="#1f2937", alpha=0.9)
        axes[0].bar(x, val_pcts, width, label=f"Validation ({len(val_df)})", color="#059669", edgecolor="#1f2937", alpha=0.9)
        axes[0].bar(x + width, test_pcts, width, label=f"Held-Out Test ({len(test_df)})", color="#d97706", edgecolor="#1f2937", alpha=0.9)

        axes[0].set_title("Class Proportions Across Splits (Stratification Verification)", fontsize=12, fontweight="bold", pad=10)
        axes[0].set_ylabel("Percentage of Split (%)", fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([cls.upper() for cls in classes], fontsize=11, fontweight="semibold")
        axes[0].set_ylim(0, 50)
        axes[0].legend(loc="upper right", frameon=True, fontsize=10)

        for i in range(len(classes)):
            axes[0].text(x[i] - width, train_pcts[i] + 0.8, f"{train_pcts[i]:.1f}%\n({train_counts[i]})", ha="center", fontsize=8.5)
            axes[0].text(x[i], val_pcts[i] + 0.8, f"{val_pcts[i]:.1f}%\n({val_counts[i]})", ha="center", fontsize=8.5)
            axes[0].text(x[i] + width, test_pcts[i] + 0.8, f"{test_pcts[i]:.1f}%\n({test_counts[i]})", ha="center", fontsize=8.5)

        # Panel 2: Sequence Length (Word Count) Distribution
        axes[1].hist(
            train_df["word_count"],
            bins=40,
            range=(0, 400),
            alpha=0.6,
            color="#1d4ed8",
            label="Train",
            density=True,
            edgecolor="#1f2937",
        )
        axes[1].hist(
            val_df["word_count"],
            bins=40,
            range=(0, 400),
            alpha=0.6,
            color="#059669",
            label="Validation",
            density=True,
            edgecolor="#1f2937",
        )
        axes[1].hist(
            test_df["word_count"],
            bins=40,
            range=(0, 400),
            alpha=0.6,
            color="#d97706",
            label="Test",
            density=True,
            edgecolor="#1f2937",
        )

        axes[1].axvline(256, color="#dc2626", linestyle="--", linewidth=1.8, label="Max Sequence Cutoff (256)")
        axes[1].set_title("Issue Word Count Density Distribution", fontsize=12, fontweight="bold", pad=10)
        axes[1].set_xlabel("Word Count", fontsize=11)
        axes[1].set_ylabel("Density", fontsize=11)
        axes[1].legend(loc="upper right", frameon=True, fontsize=10)

        plt.suptitle("Phase 6 Deep Learning — Transformer Dataset Distribution & Stratification (Step 6.2)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def save_manifest(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: Union[str, Path],
    ) -> Path:
        """Save dataset manifest with split statistics and integrity hashes."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def get_split_stats(df: pd.DataFrame) -> Dict[str, Any]:
            counts = {cls: int(np.sum(df["target_label"] == cls)) for cls in self.TARGET_CLASSES}
            ratios = {cls: round(c / len(df), 4) for cls, c in counts.items()}
            words = df["word_count"]
            return {
                "total_samples": len(df),
                "class_counts": counts,
                "class_ratios": ratios,
                "word_count_stats": {
                    "mean": round(float(words.mean()), 1),
                    "median": round(float(words.median()), 1),
                    "p75": round(float(words.quantile(0.75)), 1),
                    "p95": round(float(words.quantile(0.95)), 1),
                    "max": int(words.max()),
                },
                "sha256": hashlib.sha256(df["text"].str.cat().encode("utf-8")).hexdigest()[:16],
            }

        manifest = {
            "metadata": {
                "phase": "6",
                "step": "6.2",
                "title": "Prepare Transformer Dataset",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "label_mapping": self.LABEL2ID,
                "val_ratio": self.val_ratio,
                "random_state": self.random_state,
                "enhanced_preprocessing": self.use_enhanced_preprocessing,
                "total_samples": len(train_df) + len(val_df) + len(test_df),
                "leakage_verified": True,
            },
            "splits": {
                "train": get_split_stats(train_df),
                "validation": get_split_stats(val_df),
                "test": get_split_stats(test_df),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return output_path

    def run_all(
        self,
        train_source_path: Union[str, Path],
        test_source_path: Union[str, Path],
        output_dir: Union[str, Path],
        figure_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Execute complete transformer dataset preparation workflow."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[Step 6.2] Splitting canonical datasets into stratified train, val, and test...")
        train_df, val_df, test_df = self.build_splits(train_source_path, test_source_path)

        train_dest = output_dir / "train.csv"
        val_dest = output_dir / "val.csv"
        test_dest = output_dir / "test.csv"
        manifest_dest = output_dir / "dataset_manifest.json"

        train_df.to_csv(train_dest, index=False)
        val_df.to_csv(val_dest, index=False)
        test_df.to_csv(test_dest, index=False)

        print(f"  -> Saved Train:      {len(train_df)} rows to {train_dest}")
        print(f"  -> Saved Validation: {len(val_df)} rows to {val_dest}")
        print(f"  -> Saved Test:       {len(test_df)} rows to {test_dest}")

        fig_file = self.plot_distributions(train_df, val_df, test_df, figure_path)
        manifest_file = self.save_manifest(train_df, val_df, test_df, manifest_dest)

        print(f"[Step 6.2] Saved distribution plot to: {fig_file}")
        print(f"[Step 6.2] Saved dataset manifest to: {manifest_file}")

        # PyTorch dataset sanity check
        torch_ds = self.create_pytorch_dataset(train_df)
        sample = torch_ds[0]
        print(f"[Step 6.2] PyTorch IssueDataset verified (Sample 0 label: {sample['label']} [{self.ID2LABEL[sample['label']]}])")

        return {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "figure_path": str(fig_file),
            "manifest_path": str(manifest_file),
        }


def main():
    """CLI execution entry point."""
    train_source = Path("data/processed/splits/train.csv")
    test_source = Path("data/processed/splits/test.csv")
    out_dir = Path("data/processed/transformer")
    fig_dest = Path("ml/deep_learning/figures/fig17_transformer_dataset_distribution.png")

    builder = TransformerDatasetBuilder(
        val_ratio=0.15,
        random_state=42,
        use_enhanced_preprocessing=True,
    )
    builder.run_all(train_source, test_source, out_dir, fig_dest)


if __name__ == "__main__":
    main()
