"""Transformer Fine-Tuning Pipeline for Issue Classification (Phase 6, Step 6.4).

Implements end-to-end PyTorch fine-tuning for DistilBERT/BERT on GitHub issues:
- Stratified batching and PyTorch DataLoader integration
- Class-weighted CrossEntropyLoss to address class imbalance
- AdamW optimizer with decoupled weight decay (excluding biases & LayerNorm)
- Linear learning rate schedule with warmup
- Gradient clipping for numerical stability
- Validation tracking (Loss, Accuracy, Macro/Weighted/Per-class F1)
- Model checkpointing based on best validation Macro-F1
- Training curve visualization and JSON history logging
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

from ml.deep_learning.tokenizer import IssueTokenizer, TokenizedIssueDataset

# Default class label mappings
LABEL_TO_ID: Dict[str, int] = {"bug": 0, "feature": 1, "other": 2}
ID_TO_LABEL: Dict[int, str] = {0: "bug", 1: "feature", 2: "other"}


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int = 3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights: w_c = N / (C * N_c).

    Args:
        labels: Sequence of integer class labels.
        num_classes: Total number of target classes (default: 3).
        device: Target torch device for the output tensor.

    Returns:
        Tensor of float weights with shape (num_classes,).
    """
    total = len(labels)
    if total == 0:
        raise ValueError("Cannot compute class weights on empty labels list.")

    counts = np.bincount(labels, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        cnt = counts[c]
        if cnt > 0:
            weights[c] = total / (num_classes * cnt)
        else:
            weights[c] = 1.0

    tensor_weights = torch.tensor(weights, dtype=torch.float)
    if device is not None:
        tensor_weights = tensor_weights.to(device)
    return tensor_weights


class TransformerFineTuner:
    """Manages fine-tuning, validation, checkpointing, and evaluation of transformer models."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        batch_size: int = 16,
        epochs: int = 3,
        max_length: int = 256,
        use_class_weights: bool = True,
        tokenizer_dir: Optional[Union[str, Path]] = "models/tokenizer",
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        """Initialize fine-tuner configuration and underlying transformer architecture."""
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label or ID_TO_LABEL
        self.label2id = label2id or LABEL_TO_ID
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.use_class_weights = use_class_weights
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None
        self.seed = seed

        set_seed(self.seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        self.tokenizer = IssueTokenizer(
            model_name=self.model_name,
            max_length=self.max_length,
            local_dir=self.tokenizer_dir,
        )

        # Initialize sequence classification model
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

        self.class_weights: Optional[torch.Tensor] = None
        self.criterion: Optional[nn.Module] = None
        self.training_history: Dict[str, Any] = {}

    def get_optimizer_and_scheduler(
        self,
        total_steps: int,
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """Construct AdamW optimizer with weight decay exclusions and linear warmup scheduler."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8,
        )

        num_warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        return optimizer, scheduler

    def build_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        pre_tokenize: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training and validation splits.

        Args:
            train_df: DataFrame containing 'text' and 'label' columns for training.
            val_df: DataFrame containing 'text' and 'label' columns for validation.
            pre_tokenize: Whether to cache tokenization tensors in RAM upfront.

        Returns:
            Tuple of (train_dataloader, val_dataloader).
        """
        train_dataset = TokenizedIssueDataset(
            texts=train_df["text"].tolist(),
            labels=train_df["label"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            issue_ids=train_df["issue_id"].astype(str).tolist() if "issue_id" in train_df.columns else None,
            pre_tokenize=pre_tokenize,
        )

        val_dataset = TokenizedIssueDataset(
            texts=val_df["text"].tolist(),
            labels=val_df["label"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            issue_ids=val_df["issue_id"].astype(str).tolist() if "issue_id" in val_df.columns else None,
            pre_tokenize=pre_tokenize,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        epoch_idx: int = 1,
        log_interval: int = 20,
    ) -> float:
        """Execute one training epoch with gradient clipping and scheduler stepping.

        Returns:
            Average training loss across all batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            if step % log_interval == 0 or step == num_batches:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  [Epoch {epoch_idx}/{self.epochs}] Step {step}/{num_batches} "
                    f"| Loss: {batch_loss:.4f} | Avg Loss: {total_loss / step:.4f} | LR: {current_lr:.2e}"
                )

        return total_loss / num_batches

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance on a dataset.

        Returns:
            Dictionary containing loss, accuracy, macro/weighted metrics, per-class F1,
            and confusion matrix.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[List[float]] = []

        if criterion is None:
            criterion = self.criterion or nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        acc = float(accuracy_score(all_labels, all_preds))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        class_metrics = {}
        for c in range(self.num_labels):
            name = self.id2label.get(c, str(c))
            class_metrics[name] = {
                "precision": float(per_p[c]) if c < len(per_p) else 0.0,
                "recall": float(per_r[c]) if c < len(per_r) else 0.0,
                "f1_score": float(per_f1[c]) if c < len(per_f1) else 0.0,
                "support": int(per_sup[c]) if c < len(per_sup) else 0,
            }

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_labels))).tolist()

        return {
            "loss": float(avg_loss),
            "accuracy": acc,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
            "per_class": class_metrics,
            "confusion_matrix": cm,
            "sample_count": len(all_labels),
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: Union[str, Path] = "models/distilbert_finetuned",
        save_best: bool = True,
        log_interval: int = 20,
    ) -> Dict[str, Any]:
        """Execute full training and validation lifecycle over all epochs.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
            output_dir: Destination path to checkpoint the best model weights.
            save_best: Whether to save checkpoint of epoch with highest validation Macro-F1.
            log_interval: Number of steps between console logs.

        Returns:
            Dictionary containing epoch-level histories, best epoch metadata, and hyperparameters.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set up class weights and loss criterion
        if self.use_class_weights:
            self.class_weights = compute_class_weights(
                train_df["label"].tolist(),
                num_classes=self.num_labels,
                device=self.device,
            )
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        train_loader, val_loader = self.build_dataloaders(train_df, val_df, pre_tokenize=True)
        total_steps = len(train_loader) * self.epochs
        optimizer, scheduler = self.get_optimizer_and_scheduler(total_steps=total_steps)

        print(f"\n=======================================================")
        print(f"Starting Fine-Tuning: {self.model_name}")
        print(f"Device: {self.device} | Epochs: {self.epochs} | Batch Size: {self.batch_size}")
        print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
        print(f"Total training steps: {total_steps} (Warmup: {int(total_steps * self.warmup_ratio)} steps)")
        if self.class_weights is not None:
            weights_str = ", ".join(
                f"{self.id2label.get(i, i)}: {w:.4f}"
                for i, w in enumerate(self.class_weights.cpu().numpy())
            )
            print(f"Class Weights: [{weights_str}]")
        print(f"=======================================================\n")

        epoch_records: List[Dict[str, Any]] = []
        best_val_macro_f1 = -1.0
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            start_time = datetime.datetime.now()
            train_loss = self.train_epoch(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=self.criterion,
                epoch_idx=epoch,
                log_interval=log_interval,
            )

            val_metrics = self.evaluate(val_loader, criterion=self.criterion)
            duration_sec = (datetime.datetime.now() - start_time).total_seconds()

            is_best = val_metrics["macro_f1"] > best_val_macro_f1
            if is_best:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                if save_best:
                    self.save_checkpoint(output_path, metadata={"epoch": epoch, "val_metrics": val_metrics})
                    print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Macro-F1: {best_val_macro_f1:.4f})")

            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_per_class": val_metrics["per_class"],
                "duration_seconds": round(duration_sec, 2),
                "is_best": is_best,
            }
            epoch_records.append(epoch_record)

            print(
                f"Epoch {epoch}/{self.epochs} Complete ({duration_sec:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Best Val Macro-F1: {best_val_macro_f1:.4f}"
            )
            print("-" * 55)

        self.training_history = {
            "model_name": self.model_name,
            "architecture": "DistilBertForSequenceClassification",
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
                "max_grad_norm": self.max_grad_norm,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "max_length": self.max_length,
                "use_class_weights": self.use_class_weights,
                "seed": self.seed,
            },
            "class_weights": (
                {self.id2label.get(i, str(i)): float(w) for i, w in enumerate(self.class_weights.cpu().numpy())}
                if self.class_weights is not None
                else None
            ),
            "label_mapping": self.id2label,
            "dataset_sizes": {"train": len(train_df), "val": len(val_df)},
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_macro_f1),
            "epochs": epoch_records,
            "device": str(self.device),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        return self.training_history

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save fine-tuned transformer weights, config, and tokenizer to disk."""
        target_dir = Path(checkpoint_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(str(target_dir))
        self.tokenizer.tokenizer.save_pretrained(str(target_dir))

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        meta_file = target_dir / "training_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return target_dir

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> None:
        """Load fine-tuned model checkpoint from directory."""
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

        self.model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        self.model.to(self.device)
        self.tokenizer = IssueTokenizer(
            model_name=str(ckpt_path),
            max_length=self.max_length,
            local_dir=ckpt_path,
        )

    def plot_training_curves(
        self,
        history: Optional[Dict[str, Any]] = None,
        save_path: Union[str, Path] = "ml/deep_learning/figures/fig19_transformer_training_curves.png",
    ) -> Path:
        """Plot loss curves and validation F1 trajectories across epochs."""
        hist = history or self.training_history
        if not hist or "epochs" not in hist:
            raise ValueError("No training history available to plot.")

        records = hist["epochs"]
        epochs = [r["epoch"] for r in records]
        train_losses = [r["train_loss"] for r in records]
        val_losses = [r["val_loss"] for r in records]
        val_macro_f1s = [r["val_macro_f1"] for r in records]
        val_accs = [r["val_accuracy"] for r in records]

        bug_f1s = [r["val_per_class"].get("bug", {}).get("f1_score", 0.0) for r in records]
        feature_f1s = [r["val_per_class"].get("feature", {}).get("f1_score", 0.0) for r in records]
        other_f1s = [r["val_per_class"].get("other", {}).get("f1_score", 0.0) for r in records]

        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Subplot 1: Loss curves
        ax1.plot(epochs, train_losses, "o-", color="#1f77b4", linewidth=2.2, label="Train Loss")
        ax1.plot(epochs, val_losses, "s--", color="#d62728", linewidth=2.2, label="Val Loss")
        ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax1.set_ylabel("CrossEntropy Loss", fontsize=11, fontweight="bold")
        ax1.set_title("Training & Validation Loss Convergence", fontsize=12, fontweight="bold")
        ax1.set_xticks(epochs)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)

        # Highlight best epoch on loss
        best_ep = hist.get("best_epoch", epochs[-1])
        ax1.axvline(x=best_ep, color="green", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Ep {best_ep})")
        ax1.legend(loc="upper right", frameon=True)

        # Subplot 2: Metric trajectories
        ax2.plot(epochs, val_macro_f1s, "o-", color="#2ca02c", linewidth=2.5, label="Val Macro-F1 (Champion)")
        ax2.plot(epochs, val_accs, "^:", color="#7f7f7f", linewidth=1.8, label="Val Accuracy")
        ax2.plot(epochs, bug_f1s, ".-", color="#1f77b4", alpha=0.7, label="Bug F1")
        ax2.plot(epochs, feature_f1s, ".-", color="#ff7f0e", alpha=0.7, label="Feature F1")
        ax2.plot(epochs, other_f1s, ".-", color="#9467bd", alpha=0.7, label="Other F1")

        ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax2.set_title("Validation Metric Progression Across Epochs", fontsize=12, fontweight="bold")
        ax2.set_xticks(epochs)
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", frameon=True, fontsize=9)

        plt.suptitle(
            f"OpenSourceLens: Fine-Tuning Dynamics ({hist.get('model_name', 'distilbert-base-uncased')})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        print(f"Training curves saved to: {out_path}")
        return out_path

    def save_training_history(
        self,
        history: Optional[Dict[str, Any]] = None,
        save_path: Union[str, Path] = "data/processed/evaluation/transformer_training_history.json",
    ) -> Path:
        """Save full training history dictionary to JSON."""
        hist = history or self.training_history
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)

        print(f"Training history saved to: {out_path}")
        return out_path


def run_transformer_training_pipeline(
    train_path: Union[str, Path] = "data/processed/transformer/train.csv",
    val_path: Union[str, Path] = "data/processed/transformer/val.csv",
    checkpoint_dir: Union[str, Path] = "models/distilbert_finetuned",
    figure_path: Union[str, Path] = "ml/deep_learning/figures/fig19_transformer_training_curves.png",
    history_path: Union[str, Path] = "data/processed/evaluation/transformer_training_history.json",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2.5e-5,
    max_length: int = 256,
) -> Dict[str, Any]:
    """Execute end-to-end training pipeline for Step 6.4."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    trainer = TransformerFineTuner(
        model_name="distilbert-base-uncased",
        num_labels=3,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        use_class_weights=True,
    )

    history = trainer.fit(
        train_df=train_df,
        val_df=val_df,
        output_dir=checkpoint_dir,
        save_best=True,
    )

    trainer.plot_training_curves(history, save_path=figure_path)
    trainer.save_training_history(history, save_path=history_path)

    return history


if __name__ == "__main__":
    run_transformer_training_pipeline()
