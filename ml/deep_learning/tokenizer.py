"""Transformer Tokenization and Subword Analysis for OpenSourceLens (Phase 6 Deep Learning).

Implements the subword tokenization pipeline for DistilBERT / BERT, provides PyTorch-compatible
TokenizedIssueDataset wrappers for DataLoader batching, profiles subword token length distributions
and truncation rates, and saves local tokenizer artifacts for offline training and inference.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import warnings
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class IssueTokenizer:
    """Production tokenizer wrapper for transformer-based GitHub issue classification."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        local_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the tokenizer.

        Args:
            model_name: Hugging Face model identifier (default: 'distilbert-base-uncased').
            max_length: Maximum sequence length budget (default: 256).
            local_dir: Optional local path to load/save tokenizer offline.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.local_dir = Path(local_dir) if local_dir else None

        # Load from local cache if available, otherwise fetch from Hub
        if self.local_dir and self.local_dir.exists() and (
            (self.local_dir / "tokenizer.json").exists() or (self.local_dir / "vocab.txt").exists()
        ):
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(str(self.local_dir))
        else:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return self.tokenizer.vocab_size

    def tokenize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single text string into PyTorch tensors with padding and truncation.

        Args:
            text: Input text string.
            max_length: Optional override for sequence length.

        Returns:
            Dictionary containing 'input_ids' and 'attention_mask' tensors of shape (1, max_length).
        """
        seq_len = max_length or self.max_length
        encoded = self.tokenizer(
            text,
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def batch_tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of text strings into batched PyTorch tensors.

        Args:
            texts: List of text strings.
            max_length: Optional override for sequence length.

        Returns:
            Dictionary containing batched 'input_ids' and 'attention_mask' tensors of shape (N, max_length).
        """
        seq_len = max_length or self.max_length
        encoded = self.tokenizer(
            texts,
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def count_tokens(self, text: str) -> int:
        """Measure raw subword token count without length truncation."""
        return len(self.tokenizer.tokenize(text))

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Save tokenizer configuration and vocabulary files locally for offline use."""
        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(dest))
        self.local_dir = dest
        return dest


class TokenizedIssueDataset(Dataset):
    """PyTorch Dataset yielding pre-tokenized tensors for transformer training and evaluation."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: IssueTokenizer,
        max_length: int = 256,
        issue_ids: Optional[List[str]] = None,
        pre_tokenize: bool = False,
    ):
        """Initialize the tokenized dataset.

        Args:
            texts: List of natural language issue strings.
            labels: List of integer target labels.
            tokenizer: Configured IssueTokenizer instance.
            max_length: Maximum sequence length (default: 256).
            issue_ids: Optional list of issue identifiers.
            pre_tokenize: Whether to tokenize all texts upfront in memory (default: False).
        """
        if len(texts) != len(labels):
            raise ValueError(f"Texts length ({len(texts)}) != labels length ({len(labels)})")

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.issue_ids = issue_ids if issue_ids is not None else [str(i) for i in range(len(texts))]
        self.pre_tokenize = pre_tokenize

        self._cached_encodings: Optional[Dict[str, torch.Tensor]] = None
        if pre_tokenize:
            self._cached_encodings = self.tokenizer.batch_tokenize(texts, max_length=max_length)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample at given index."""
        if self._cached_encodings is not None:
            input_ids = self._cached_encodings["input_ids"][idx]
            attention_mask = self._cached_encodings["attention_mask"][idx]
        else:
            enc = self.tokenizer.tokenize_text(self.texts[idx], max_length=self.max_length)
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

        return {
            "input_ids": input_ids.clone().detach(),
            "attention_mask": attention_mask.clone().detach(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "issue_id": str(self.issue_ids[idx]),
        }


class SubwordTokenAnalyzer:
    """Profiles token length distributions, truncation rates, and domain token segmentations."""

    def __init__(self, tokenizer: IssueTokenizer):
        """Initialize the analyzer."""
        self.tokenizer = tokenizer

    def analyze_split(self, df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """Compute subword token lengths and truncation statistics for a DataFrame."""
        texts = df["text"].tolist()
        token_lengths = [self.tokenizer.count_tokens(t) for t in texts]
        token_lengths_arr = np.array(token_lengths)

        cutoff = self.tokenizer.max_length
        truncated_count = int(np.sum(token_lengths_arr > cutoff))
        truncation_rate = float(truncated_count / len(token_lengths_arr)) if len(token_lengths_arr) > 0 else 0.0

        total_raw_tokens = int(np.sum(token_lengths_arr))
        retained_tokens = int(np.sum(np.minimum(token_lengths_arr, cutoff)))
        retention_ratio = float(retained_tokens / total_raw_tokens) if total_raw_tokens > 0 else 1.0

        return {
            "split_name": split_name,
            "sample_count": len(df),
            "cutoff_max_length": cutoff,
            "truncated_samples": truncated_count,
            "truncation_rate_pct": round(truncation_rate * 100, 2),
            "token_retention_pct": round(retention_ratio * 100, 2),
            "token_length_stats": {
                "mean": round(float(np.mean(token_lengths_arr)), 1),
                "std": round(float(np.std(token_lengths_arr)), 1),
                "median_p50": round(float(np.median(token_lengths_arr)), 1),
                "p75": round(float(np.percentile(token_lengths_arr, 75)), 1),
                "p90": round(float(np.percentile(token_lengths_arr, 90)), 1),
                "p95": round(float(np.percentile(token_lengths_arr, 95)), 1),
                "p99": round(float(np.percentile(token_lengths_arr, 99)), 1),
                "max": int(np.max(token_lengths_arr)),
            },
            "token_lengths": token_lengths,
        }

    def audit_domain_entities(self) -> List[Dict[str, Any]]:
        """Inspect how software domain technical terms segment into WordPiece tokens."""
        sample_terms = [
            "hex_addr",
            "commit_hash",
            "version_num",
            "file_path",
            "issue_ref",
            "NullPointerException",
            "IndexOutOfBoundsException",
            "SIGSEGV",
            "torch.cuda.is_available()",
            "kubernetes/kubelet",
            "vscode/workbench",
        ]
        results = []
        for term in sample_terms:
            subwords = self.tokenizer.tokenizer.tokenize(term)
            subword_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(subwords)
            results.append(
                {
                    "term": term,
                    "subword_count": len(subwords),
                    "subwords": subwords,
                    "subword_ids": subword_ids,
                }
            )
        return results

    def plot_token_distributions(
        self,
        analysis_data: Dict[str, Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        """Plot subword token length histograms and cumulative retention curves (fig18)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

        colors = {"train": "#1d4ed8", "val": "#059669", "test": "#d97706"}
        cutoff = self.tokenizer.max_length

        # Panel 1: Token Length Histograms
        for split, data in analysis_data.items():
            lengths = data["token_lengths"]
            c = colors.get(split, "#4f46e5")
            axes[0].hist(
                lengths,
                bins=40,
                range=(0, 500),
                alpha=0.45,
                color=c,
                label=f"{split.capitalize()} (Median: {data['token_length_stats']['median_p50']})",
                density=True,
                edgecolor="#1f2937",
            )

        axes[0].axvline(cutoff, color="#dc2626", linestyle="--", linewidth=2.0, label=f"Max Length Cutoff ({cutoff})")
        axes[0].set_title("Subword Token Length Density Distribution", fontsize=12, fontweight="bold", pad=10)
        axes[0].set_xlabel("WordPiece Subword Token Count", fontsize=11)
        axes[0].set_ylabel("Density", fontsize=11)
        axes[0].legend(loc="upper right", frameon=True, fontsize=10)

        # Panel 2: Cumulative Sequence Length Retention Curves
        seq_thresholds = np.arange(16, 513, 8)
        for split, data in analysis_data.items():
            lengths = np.array(data["token_lengths"])
            c = colors.get(split, "#4f46e5")
            cum_pct = [np.mean(lengths <= th) * 100 for th in seq_thresholds]
            axes[1].plot(seq_thresholds, cum_pct, label=f"{split.capitalize()}", color=c, linewidth=2.0)

        # Draw cutoff annotation
        ref_key = "train" if "train" in analysis_data else list(analysis_data.keys())[0]
        retained_pct = np.mean(np.array(analysis_data[ref_key]["token_lengths"]) <= cutoff) * 100
        axes[1].axvline(cutoff, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Cutoff = {cutoff} ({retained_pct:.1f}% preserved)")
        axes[1].axhline(retained_pct, color="#dc2626", linestyle=":", linewidth=1.2)
        axes[1].set_title("Cumulative Issue Retention vs. Sequence Length Budget", fontsize=12, fontweight="bold", pad=10)
        axes[1].set_xlabel("Sequence Length Budget (Tokens)", fontsize=11)
        axes[1].set_ylabel("% Issues Completely Contained Without Truncation", fontsize=11)
        axes[1].set_ylim(0, 105)
        axes[1].legend(loc="lower right", frameon=True, fontsize=10)

        plt.suptitle("Phase 6 Deep Learning — DistilBERT Subword Tokenization Profiling (Step 6.3)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def save_analysis(
        self,
        analysis_data: Dict[str, Dict[str, Any]],
        domain_audit: List[Dict[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        """Persist structured tokenization evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip raw token_lengths array before serialization for compact JSON
        clean_splits = {}
        for split, data in analysis_data.items():
            clean_splits[split] = {k: v for k, v in data.items() if k != "token_lengths"}

        payload = {
            "metadata": {
                "phase": "6",
                "step": "6.3",
                "title": "Tokenization",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": self.tokenizer.model_name,
                "vocab_size": self.tokenizer.vocab_size,
                "max_sequence_length": self.tokenizer.max_length,
                "special_tokens": {
                    "cls_token": self.tokenizer.tokenizer.cls_token,
                    "sep_token": self.tokenizer.tokenizer.sep_token,
                    "pad_token": self.tokenizer.tokenizer.pad_token,
                    "unk_token": self.tokenizer.tokenizer.unk_token,
                },
            },
            "split_token_statistics": clean_splits,
            "domain_token_segmentation_audit": domain_audit,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path


def main():
    """CLI execution entry point."""
    train_file = Path("data/processed/transformer/train.csv")
    val_file = Path("data/processed/transformer/val.csv")
    test_file = Path("data/processed/transformer/test.csv")
    model_dir = Path("models/tokenizer")
    fig_dest = Path("ml/deep_learning/figures/fig18_token_length_distribution.png")
    json_dest = Path("data/processed/evaluation/tokenization_analysis.json")

    print("[Step 6.3] Initializing IssueTokenizer for 'distilbert-base-uncased'...")
    tokenizer = IssueTokenizer(model_name="distilbert-base-uncased", max_length=256)
    print(f"  -> Tokenizer Vocab Size: {tokenizer.vocab_size} WordPiece tokens")
    print(f"  -> Max Sequence Length:  {tokenizer.max_length} tokens")

    saved_path = tokenizer.save_pretrained(model_dir)
    print(f"[Step 6.3] Saved offline tokenizer configuration to: {saved_path}")

    print("[Step 6.3] Analyzing subword token length distributions across splits...")
    analyzer = SubwordTokenAnalyzer(tokenizer)

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    split_analysis = {
        "train": analyzer.analyze_split(train_df, "train"),
        "val": analyzer.analyze_split(val_df, "val"),
        "test": analyzer.analyze_split(test_df, "test"),
    }

    for split, data in split_analysis.items():
        stats = data["token_length_stats"]
        print(f"  -> {split.upper():<5}: Mean={stats['mean']} tokens, Median={stats['median_p50']}, P95={stats['p95']} | Truncation Rate: {data['truncation_rate_pct']}% (Retention: {data['token_retention_pct']}%)")

    print("[Step 6.3] Auditing domain token subword segmentations...")
    domain_audit = analyzer.audit_domain_entities()
    for item in domain_audit[:5]:
        print(f"  -> '{item['term']}': {item['subword_count']} subwords -> {item['subwords']}")

    fig_file = analyzer.plot_token_distributions(split_analysis, fig_dest)
    json_file = analyzer.save_analysis(split_analysis, domain_audit, json_dest)

    print(f"\n[Step 6.3] Saved distribution plot to: {fig_file}")
    print(f"[Step 6.3] Saved tokenization report to: {json_file}")

    # Verify PyTorch TokenizedIssueDataset
    tokenized_ds = TokenizedIssueDataset(
        texts=train_df["text"].tolist()[:10],
        labels=train_df["label"].tolist()[:10],
        tokenizer=tokenizer,
        max_length=256,
    )
    sample_batch = tokenized_ds[0]
    print(f"[Step 6.3] PyTorch TokenizedIssueDataset verified (Tensor input_ids shape: {sample_batch['input_ids'].shape}, label: {sample_batch['label']})")


if __name__ == "__main__":
    main()
