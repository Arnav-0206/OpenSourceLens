"""Model Evaluation and Comparative Analysis Module for OpenSourceLens.

Computes comprehensive classification metrics (accuracy, macro/weighted precision,
recall, F1-scores, per-class metrics, confusion matrices, log loss, and Brier scores)
on held-out test splits, generates dual publication-quality confusion matrix heatmaps,
and produces structured baseline comparative evaluation reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)


class ClassifierEvaluator:
    """Evaluates classical and deep learning classifiers on held-out test splits."""

    def __init__(self, classes: Optional[List[str]] = None):
        """Initialize the evaluator.

        Args:
            classes: List of class labels in deterministic order (default: ['bug', 'feature', 'other']).
        """
        self.classes = list(classes) if classes is not None else ["bug", "feature", "other"]

    def evaluate_model(
        self,
        model: Any,
        X_test: scipy.sparse.csr_matrix,
        y_test: Union[np.ndarray, pd.Series, List[Any]],
        model_name: str = "Classifier",
    ) -> Dict[str, Any]:
        """Compute full classification metrics and confusion matrices for a given model.

        Args:
            model: Trained classifier exposing predict() and optionally predict_proba().
            X_test: Held-out test feature matrix.
            y_test: True class labels.
            model_name: Descriptive name for reporting (default: 'Classifier').

        Returns:
            Dictionary of comprehensive evaluation metrics.
        """
        y_arr = np.asarray(y_test)
        y_pred = np.asarray(model.predict(X_test))

        # Overall summary metrics
        acc = float(accuracy_score(y_arr, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_arr, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_arr, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        class_rep = classification_report(
            y_arr,
            y_pred,
            labels=self.classes,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )

        per_class: Dict[str, Dict[str, Any]] = {}
        for cls in self.classes:
            if cls in class_rep:
                per_class[cls] = {
                    "precision": round(float(class_rep[cls]["precision"]), 4),
                    "recall": round(float(class_rep[cls]["recall"]), 4),
                    "f1_score": round(float(class_rep[cls]["f1-score"]), 4),
                    "support": int(class_rep[cls]["support"]),
                }

        # Confusion matrix (raw and row-normalized)
        cm_raw = confusion_matrix(y_arr, y_pred, labels=self.classes)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm_raw.astype(float),
            row_sums,
            out=np.zeros_like(cm_raw, dtype=float),
            where=row_sums != 0,
        )

        results: Dict[str, Any] = {
            "model_name": model_name,
            "test_samples": len(y_arr),
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "confusion_matrix": {
                "labels": self.classes,
                "raw_matrix": cm_raw.tolist(),
                "normalized_matrix": [[round(float(v), 4) for v in row] for row in cm_norm],
            },
        }

        # Probabilistic calibration metrics if model supports predict_proba
        if hasattr(model, "predict_proba"):
            try:
                y_probas = model.predict_proba(X_test)
                loss = float(log_loss(y_arr, y_probas, labels=self.classes))
                results["log_loss"] = round(loss, 4)

                # Multi-class Brier score (mean squared error between one-hot true and predicted probabilities)
                y_onehot = np.zeros_like(y_probas)
                for idx, cls in enumerate(self.classes):
                    y_onehot[y_arr == cls, idx] = 1.0

                brier = float(np.mean(np.sum((y_probas - y_onehot) ** 2, axis=1)))
                results["brier_score"] = round(brier, 4)
            except Exception:
                results["log_loss"] = None
                results["brier_score"] = None
        else:
            results["log_loss"] = None
            results["brier_score"] = None

        return results

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: scipy.sparse.csr_matrix,
        y_test: Union[np.ndarray, pd.Series, List[Any]],
    ) -> Dict[str, Any]:
        """Evaluate multiple models side-by-side and determine comparative ranking.

        Args:
            models: Dictionary mapping model names to model instances.
            X_test: Held-out test feature matrix.
            y_test: Held-out test labels.

        Returns:
            Dictionary containing individual evaluations, comparison summary, and champion model.
        """
        evaluations: Dict[str, Any] = {}
        comparison_rows: List[Dict[str, Any]] = []

        for name, model in models.items():
            res = self.evaluate_model(model, X_test, y_test, model_name=name)
            evaluations[name] = res

            comparison_rows.append({
                "model_name": name,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
                "bug_f1": res["per_class"].get("bug", {}).get("f1_score"),
                "feature_f1": res["per_class"].get("feature", {}).get("f1_score"),
                "other_f1": res["per_class"].get("other", {}).get("f1_score"),
                "log_loss": res.get("log_loss"),
                "brier_score": res.get("brier_score"),
            })

        # Rank models by macro_f1 descending (tie-break on accuracy)
        ranked = sorted(
            comparison_rows,
            key=lambda x: (x["macro_f1"], x["accuracy"]),
            reverse=True,
        )

        champion = ranked[0]["model_name"] if ranked else None

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_sample_count": len(y_test),
            "champion_model": champion,
            "comparison_table": comparison_rows,
            "detailed_evaluations": evaluations,
        }

    def plot_confusion_matrices(
        self,
        comparison_results: Dict[str, Any],
        output_path: Union[str, Path] = "ml/classification/figures/fig10_baseline_confusion_matrices.png",
    ) -> Path:
        """Generate and save dual confusion matrix heatmaps for candidate models.

        Args:
            comparison_results: Output from compare_models().
            output_path: Target image file path.

        Returns:
            Path of saved figure.
        """
        evals = comparison_results.get("detailed_evaluations", {})
        model_names = list(evals.keys())
        if not model_names:
            raise ValueError("No models found in comparison results to plot.")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        n_models = min(len(model_names), 2)
        fig, axes = plt.subplots(1, n_models, figsize=(14, 6))
        if n_models == 1:
            axes = [axes]

        for idx in range(n_models):
            name = model_names[idx]
            ax = axes[idx]
            model_eval = evals[name]

            raw_cm = np.array(model_eval["confusion_matrix"]["raw_matrix"])
            norm_cm = np.array(model_eval["confusion_matrix"]["normalized_matrix"])
            labels = model_eval["confusion_matrix"]["labels"]

            # Annotation strings combining raw count and percentage
            annot = np.empty_like(raw_cm, dtype=object)
            for r in range(raw_cm.shape[0]):
                for c in range(raw_cm.shape[1]):
                    annot[r, c] = f"{raw_cm[r, c]}\n({norm_cm[r, c]*100:.1f}%)"

            sns.heatmap(
                norm_cm,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                cbar=True,
                vmin=0.0,
                vmax=1.0,
                ax=ax,
            )

            ax.set_title(
                f"{name}\nAcc: {model_eval['accuracy']:.4f} | Macro-F1: {model_eval['macro_f1']:.4f}",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
            ax.set_ylabel("True Ground Truth Label", fontsize=11, fontweight="bold")

        plt.suptitle(
            "OpenSourceLens — Baseline Issue Classification Confusion Matrices (Held-out Test Set, N=385)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return out

    def save_report(
        self,
        comparison_results: Dict[str, Any],
        output_path: Union[str, Path] = "data/processed/evaluation/baseline_comparison.json",
    ) -> Path:
        """Persist structured evaluation metrics to JSON.

        Args:
            comparison_results: Output dictionary from compare_models().
            output_path: File destination.

        Returns:
            Path of saved report.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2)

        return path
