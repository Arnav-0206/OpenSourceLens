"""Transformer Model Evaluation Pipeline for Issue Classification (Phase 6, Step 6.5).

Evaluates the fine-tuned DistilBERT classifier on held-out test data:
- Out-of-sample metrics (Accuracy, Macro/Weighted F1, Precision, Recall)
- Multi-class Log Loss (Cross-Entropy) and Brier Score
- One-vs-Rest ROC-AUC and Precision-Recall Average Precision (AP) curves
- Confusion matrix heatmap with raw and row-normalized percentages
- Inference latency profiling (ms/sample and throughput)
- Detailed per-sample predictions export for error analysis
"""

import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from ml.deep_learning.tokenizer import IssueTokenizer, TokenizedIssueDataset

DEFAULT_LABEL_TO_ID: Dict[str, int] = {"bug": 0, "feature": 1, "other": 2}
DEFAULT_ID_TO_LABEL: Dict[int, str] = {0: "bug", 1: "feature", 2: "other"}
CLASS_NAMES: List[str] = ["bug", "feature", "other"]
CLASS_COLORS: Dict[str, str] = {
    "bug": "#1f77b4",       # Blue
    "feature": "#ff7f0e",   # Orange
    "other": "#2ca02c",     # Green
}


class TransformerEvaluator:
    """Evaluates fine-tuned transformer models on held-out datasets with comprehensive metrics."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "models/distilbert_finetuned",
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 256,
    ):
        """Initialize evaluator and load fine-tuned model checkpoint."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_length = max_length

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata if present
        self.metadata: Dict[str, Any] = {}
        meta_file = self.checkpoint_dir / "training_metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

        # Resolve mappings
        raw_id2label = self.metadata.get("id2label", DEFAULT_ID_TO_LABEL)
        self.id2label: Dict[int, str] = {int(k): str(v) for k, v in raw_id2label.items()}
        self.label2id: Dict[str, int] = {str(k): int(v) for k, v in self.metadata.get("label2id", DEFAULT_LABEL_TO_ID).items()}
        self.num_labels = len(self.id2label)

        # Load tokenizer
        if (self.checkpoint_dir / "tokenizer.json").exists() or (self.checkpoint_dir / "vocab.txt").exists():
            self.tokenizer = IssueTokenizer(
                model_name=str(self.checkpoint_dir),
                max_length=self.max_length,
                local_dir=self.checkpoint_dir,
            )
        else:
            self.tokenizer = IssueTokenizer(
                model_name="distilbert-base-uncased",
                max_length=self.max_length,
                local_dir=Path("models/tokenizer") if Path("models/tokenizer").exists() else None,
            )

        # Load sequence classification model
        if self.checkpoint_dir.exists() and (
            (self.checkpoint_dir / "model.safetensors").exists() or (self.checkpoint_dir / "pytorch_model.bin").exists()
        ):
            self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint_dir))
        else:
            self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )

        self.model.to(self.device)
        self.model.eval()

    def predict_proba(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Compute softmax class probability distributions for a list of input texts.

        Args:
            texts: List of natural language issue strings.
            batch_size: Evaluation batch size (default: 32).

        Returns:
            np.ndarray of shape (N, num_labels) containing normalized class probabilities.
        """
        dummy_labels = [0] * len(texts)
        dataset = TokenizedIssueDataset(
            texts=texts,
            labels=dummy_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pre_tokenize=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        if all_probs:
            return np.vstack(all_probs)
        return np.empty((0, self.num_labels), dtype=np.float32)

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute discrete predicted classes and associated class probabilities.

        Returns:
            Tuple of (pred_labels, prob_matrix).
        """
        probs = self.predict_proba(texts, batch_size=batch_size)
        preds = np.argmax(probs, axis=1) if len(probs) > 0 else np.array([], dtype=int)
        return preds, probs

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Execute complete evaluation lifecycle on a test dataset.

        Args:
            df: DataFrame containing at minimum 'text' and 'label' columns.
            batch_size: Batch size for model inference.

        Returns:
            Tuple of (evaluation_metrics_dict, detailed_predictions_dataframe).
        """
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Input DataFrame must contain 'text' and 'label' columns.")

        texts = df["text"].tolist()
        y_true = np.array(df["label"].tolist(), dtype=int)
        n_samples = len(df)

        # Profile inference latency
        t0 = time.perf_counter()
        probs = self.predict_proba(texts, batch_size=batch_size)
        total_eval_time = time.perf_counter() - t0

        y_pred = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        latency_ms_per_sample = (total_eval_time / n_samples) * 1000.0 if n_samples > 0 else 0.0
        throughput_samples_per_sec = n_samples / total_eval_time if total_eval_time > 0 else 0.0

        # Core classification metrics
        acc = float(accuracy_score(y_true, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # One-hot encoding for multiclass loss and ROC/PR
        y_true_onehot = np.zeros((n_samples, self.num_labels), dtype=np.float32)
        for i, val in enumerate(y_true):
            if 0 <= val < self.num_labels:
                y_true_onehot[i, val] = 1.0

        # Log Loss (Cross-Entropy) & Multi-class Brier score
        loss_val = float(log_loss(y_true, probs, labels=list(range(self.num_labels))))
        brier_score = float(np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1)))

        # Per-class ROC-AUC and Average Precision
        class_metrics: Dict[str, Any] = {}
        ovr_roc_auc_scores: List[float] = []
        ovr_ap_scores: List[float] = []

        for c in range(self.num_labels):
            class_name = self.id2label.get(c, str(c))
            support = int(per_sup[c]) if c < len(per_sup) else 0

            # ROC-AUC (OvR)
            if np.sum(y_true_onehot[:, c]) > 0 and np.sum(y_true_onehot[:, c]) < n_samples:
                roc_auc = float(roc_auc_score(y_true_onehot[:, c], probs[:, c]))
                ap_score = float(average_precision_score(y_true_onehot[:, c], probs[:, c]))
            else:
                roc_auc = 0.5
                ap_score = 0.0

            ovr_roc_auc_scores.append(roc_auc)
            ovr_ap_scores.append(ap_score)

            class_metrics[class_name] = {
                "class_id": c,
                "precision": float(per_p[c]) if c < len(per_p) else 0.0,
                "recall": float(per_r[c]) if c < len(per_r) else 0.0,
                "f1_score": float(per_f1[c]) if c < len(per_f1) else 0.0,
                "support": support,
                "roc_auc": roc_auc,
                "average_precision": ap_score,
            }

        macro_roc_auc = float(np.mean(ovr_roc_auc_scores))
        macro_ap = float(np.mean(ovr_ap_scores))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_labels)))
        cm_norm = (
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            if np.all(cm.sum(axis=1) > 0)
            else np.zeros_like(cm, dtype=float)
        )

        metrics = {
            "model_name": self.metadata.get("model_name", "distilbert-base-uncased"),
            "evaluated_checkpoint": str(self.checkpoint_dir),
            "sample_count": n_samples,
            "overall_accuracy": acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "log_loss": loss_val,
            "brier_score": brier_score,
            "macro_roc_auc": macro_roc_auc,
            "macro_average_precision": macro_ap,
            "per_class": class_metrics,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "class_names": [self.id2label.get(c, str(c)) for c in range(self.num_labels)],
            "inference_benchmarks": {
                "total_inference_seconds": round(total_eval_time, 3),
                "latency_ms_per_sample": round(latency_ms_per_sample, 2),
                "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
                "device": str(self.device),
                "batch_size": batch_size,
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Build detailed predictions DataFrame
        preds_df = df.copy()
        preds_df["pred_label"] = y_pred
        preds_df["pred_target"] = [self.id2label.get(p, str(p)) for p in y_pred]
        preds_df["confidence"] = confidences.round(4)
        preds_df["is_correct"] = y_true == y_pred

        for c in range(self.num_labels):
            c_name = self.id2label.get(c, str(c))
            preds_df[f"prob_{c_name}"] = probs[:, c].round(4)

        return metrics, preds_df

    def plot_confusion_matrix(
        self,
        cm: Union[np.ndarray, List[List[int]]],
        save_path: Union[str, Path] = "ml/deep_learning/figures/fig20_transformer_confusion_matrix.png",
    ) -> Path:
        """Plot a dual-panel confusion matrix (raw counts and row-normalized percentages)."""
        cm_arr = np.array(cm)
        cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
        class_labels = [self.id2label.get(c, str(c)) for c in range(self.num_labels)]

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel 1: Raw Counts
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar=False,
            ax=ax1,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax1.set_title("Test Confusion Matrix (Sample Counts)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax1.set_ylabel("True Class", fontsize=11, fontweight="bold")

        # Panel 2: Row-Normalized Recall Percentages
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Greens",
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar=False,
            ax=ax2,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax2.set_title("Normalized Confusion Matrix (Recall / Class)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax2.set_ylabel("True Class", fontsize=11, fontweight="bold")

        plt.suptitle(
            f"OpenSourceLens: DistilBERT Test Performance (N={int(cm_arr.sum())})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix plot saved to: {out_path}")
        return out_path

    def plot_roc_pr_curves(
        self,
        y_true: Sequence[int],
        y_probs: np.ndarray,
        save_path: Union[str, Path] = "ml/deep_learning/figures/fig21_transformer_roc_pr_curves.png",
    ) -> Path:
        """Plot dual-panel multiclass One-vs-Rest ROC and Precision-Recall curves."""
        y_true_arr = np.array(y_true, dtype=int)
        n_samples = len(y_true_arr)
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel 1: ROC Curves
        for c in range(self.num_labels):
            class_name = self.id2label.get(c, str(c))
            color = CLASS_COLORS.get(class_name, "#333333")
            binary_true = (y_true_arr == c).astype(int)

            if np.sum(binary_true) > 0 and np.sum(binary_true) < n_samples:
                fpr, tpr, _ = roc_curve(binary_true, y_probs[:, c])
                auc_val = roc_auc_score(binary_true, y_probs[:, c])
                ax1.plot(
                    fpr,
                    tpr,
                    color=color,
                    linewidth=2.2,
                    label=f"{class_name.capitalize()} (AUC = {auc_val:.3f})",
                )

        ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.500)")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
        ax1.set_title("One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right", frameon=True, fontsize=10)

        # Panel 2: Precision-Recall Curves
        for c in range(self.num_labels):
            class_name = self.id2label.get(c, str(c))
            color = CLASS_COLORS.get(class_name, "#333333")
            binary_true = (y_true_arr == c).astype(int)

            if np.sum(binary_true) > 0:
                prec, rec, _ = precision_recall_curve(binary_true, y_probs[:, c])
                ap_val = average_precision_score(binary_true, y_probs[:, c])
                prevalence = np.mean(binary_true)
                ax2.plot(
                    rec,
                    prec,
                    color=color,
                    linewidth=2.2,
                    label=f"{class_name.capitalize()} (AP = {ap_val:.3f}, Prev = {prevalence:.1%})",
                )

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
        ax2.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", frameon=True, fontsize=9)

        plt.suptitle(
            "OpenSourceLens: DistilBERT Discrimination & Calibration Curves",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"ROC and PR curves saved to: {out_path}")
        return out_path

    def save_evaluation_summary(
        self,
        metrics: Dict[str, Any],
        save_path: Union[str, Path] = "data/processed/evaluation/transformer_test_evaluation.json",
    ) -> Path:
        """Serialize evaluation metrics dictionary to structured JSON."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Evaluation summary JSON saved to: {out_path}")
        return out_path

    def save_predictions_csv(
        self,
        predictions_df: pd.DataFrame,
        save_path: Union[str, Path] = "data/processed/transformer/test_predictions.csv",
    ) -> Path:
        """Save detailed per-sample predictions DataFrame to CSV."""
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(out_path, index=False)
        print(f"Per-sample test predictions saved to: {out_path}")
        return out_path


def run_transformer_evaluation_pipeline(
    test_path: Union[str, Path] = "data/processed/transformer/test.csv",
    checkpoint_dir: Union[str, Path] = "models/distilbert_finetuned",
    output_json_path: Union[str, Path] = "data/processed/evaluation/transformer_test_evaluation.json",
    predictions_csv_path: Union[str, Path] = "data/processed/transformer/test_predictions.csv",
    cm_figure_path: Union[str, Path] = "ml/deep_learning/figures/fig20_transformer_confusion_matrix.png",
    curves_figure_path: Union[str, Path] = "ml/deep_learning/figures/fig21_transformer_roc_pr_curves.png",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation pipeline on held-out test data for Step 6.5."""
    test_df = pd.read_csv(test_path)
    evaluator = TransformerEvaluator(checkpoint_dir=checkpoint_dir, max_length=256)

    print(f"\n=======================================================")
    print(f"Evaluating Fine-Tuned Transformer on Held-Out Test Set")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Test Samples: {len(test_df)} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    metrics, preds_df = evaluator.evaluate_dataset(test_df, batch_size=batch_size)

    # Generate figures
    evaluator.plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_figure_path)
    probs_matrix = preds_df[[f"prob_{c}" for c in metrics["class_names"]]].values
    evaluator.plot_roc_pr_curves(test_df["label"].tolist(), probs_matrix, save_path=curves_figure_path)

    # Save data artifacts
    evaluator.save_evaluation_summary(metrics, save_path=output_json_path)
    evaluator.save_predictions_csv(preds_df, save_path=predictions_csv_path)

    print("\n--- Evaluation Summary ---")
    print(f"Test Accuracy:         {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"Macro-F1 Score:        {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1 Score:     {metrics['weighted_f1']:.4f}")
    print(f"Log Loss:              {metrics['log_loss']:.4f}")
    print(f"Brier Score:           {metrics['brier_score']:.4f}")
    print(f"Macro ROC-AUC:         {metrics['macro_roc_auc']:.4f}")
    print(f"Inference Latency:     {metrics['inference_benchmarks']['latency_ms_per_sample']:.1f} ms/sample")
    print(f"Throughput:            {metrics['inference_benchmarks']['throughput_samples_per_sec']:.1f} samples/sec")
    print("--------------------------\n")

    return metrics


if __name__ == "__main__":
    run_transformer_evaluation_pipeline()
