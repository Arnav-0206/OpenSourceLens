"""Final Classical NLP Pipeline for OpenSourceLens (Phase 5 NLP Enhancement).

Encapsulates domain-specific issue preprocessing, 3x title weighting, sublinear (1, 2) TF-IDF
vectorization, and Platt-calibrated LinearSVC classification into an end-to-end production pipeline.
Provides class probability estimation, local feature explainability, model serialization, and
head-to-head empirical benchmarking against the Phase 4 baseline.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    classification_report,
)

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor


class FinalClassicalNLPPipeline:
    """Production-grade classical NLP pipeline for GitHub issue classification."""

    def __init__(
        self,
        max_features: int = 5000,
        random_state: int = 42,
        title_multiplier: int = 3,
        use_enhanced_preprocessing: bool = True,
        classifier_c: float = 0.5,
        cv_calibration: int = 5,
    ):
        """Initialize the pipeline configuration.

        Args:
            max_features: Maximum vocabulary size budget (default: 5000).
            random_state: Deterministic random seed (default: 42).
            title_multiplier: Repetition multiplier for title weighting (default: 3).
            use_enhanced_preprocessing: Apply software domain text normalization (default: True).
            classifier_c: Regularization parameter C for LinearSVC (default: 0.5).
            cv_calibration: Cross-validation folds for Platt probability scaling (default: 5).
        """
        self.max_features = max_features
        self.random_state = random_state
        self.title_multiplier = title_multiplier
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.classifier_c = classifier_c
        self.cv_calibration = cv_calibration

        self.preprocessor = EnhancedIssueTextPreprocessor() if use_enhanced_preprocessing else None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.base_classifier: Optional[LinearSVC] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.classes_: List[str] = []
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def prepare_text(self, title: str, body: str) -> str:
        """Apply domain cleaning and compose weighted title and body."""
        if self.use_enhanced_preprocessing and self.preprocessor is not None:
            t = self.preprocessor.clean_text(title or "", is_title=True)
            b = self.preprocessor.clean_text(body or "", is_title=False)
        else:
            t = (title or "").strip()
            b = (body or "").strip()

        if self.title_multiplier > 1:
            repeated_title = " ".join([t] * self.title_multiplier)
            return f"{repeated_title} {b}".strip()
        elif self.title_multiplier == 1:
            return f"{t} {b}".strip()
        else:
            return b.strip()

    def _format_inputs(
        self,
        inputs: Union[pd.DataFrame, List[Tuple[str, str]], List[Dict[str, str]], List[str]],
    ) -> List[str]:
        """Convert diverse input structures into a list of normalized composed strings."""
        formatted_texts: List[str] = []

        if isinstance(inputs, pd.DataFrame):
            t_col = "clean_title" if "clean_title" in inputs.columns else ("title" if "title" in inputs.columns else inputs.columns[0])
            b_col = "clean_body" if "clean_body" in inputs.columns else ("body" if "body" in inputs.columns else t_col)
            for _, row in inputs.iterrows():
                formatted_texts.append(self.prepare_text(row.get(t_col, ""), row.get(b_col, "")))

        elif isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, tuple) and len(item) == 2:
                    formatted_texts.append(self.prepare_text(item[0], item[1]))
                elif isinstance(item, dict):
                    t = item.get("clean_title", item.get("title", ""))
                    b = item.get("clean_body", item.get("body", ""))
                    formatted_texts.append(self.prepare_text(t, b))
                elif isinstance(item, str):
                    formatted_texts.append(self.prepare_text(item, ""))
                else:
                    raise ValueError(f"Unsupported input item format: {type(item)}")
        else:
            raise ValueError(f"Unsupported inputs container type: {type(inputs)}")

        return formatted_texts

    def fit(
        self,
        train_data: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> "FinalClassicalNLPPipeline":
        """Fit the vectorizer, base LinearSVC, and calibrated classifier on training data.

        Args:
            train_data: Training DataFrame.
            label_col: Column containing ground-truth targets.

        Returns:
            Fitted pipeline instance.
        """
        if label_col is None:
            if "target_label" in train_data.columns:
                label_col = "target_label"
            elif "canonical_label" in train_data.columns:
                label_col = "canonical_label"
            elif "primary_label" in train_data.columns:
                label_col = "primary_label"
            else:
                raise ValueError("DataFrame must contain 'target_label', 'canonical_label', or 'primary_label'")

        texts = self._format_inputs(train_data)
        labels = np.array(train_data[label_col].astype(str).tolist(), dtype=str)
        self.classes_ = sorted(list(np.unique(labels)))

        min_df = 1 if len(texts) < 50 else 2
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=min_df,
        )
        X = self.vectorizer.fit_transform(texts)
        self.feature_names_ = self.vectorizer.get_feature_names_out().tolist()

        # Fit base LinearSVC for coefficient extraction & explainability
        self.base_classifier = LinearSVC(
            C=self.classifier_c,
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=2000,
        )
        self.base_classifier.fit(X, labels)

        # Fit CalibratedClassifierCV with Platt scaling
        # Adapt calibration folds for tiny mock tests
        cal_cv = min(self.cv_calibration, len(texts) // len(self.classes_))
        if cal_cv < 2:
            cal_cv = 2

        self.calibrated_model = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=self.classifier_c,
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=2000,
            ),
            method="sigmoid",
            cv=cal_cv,
        )
        self.calibrated_model.fit(X, labels)

        self.is_fitted = True
        return self

    def predict(
        self,
        inputs: Union[pd.DataFrame, List[Tuple[str, str]], List[Dict[str, str]], List[str]],
    ) -> np.ndarray:
        """Predict categorical class labels ('bug', 'feature', 'other')."""
        if not self.is_fitted or self.calibrated_model is None or self.vectorizer is None:
            raise RuntimeError("Pipeline is not fitted. Call fit() or load() first.")

        texts = self._format_inputs(inputs)
        X = self.vectorizer.transform(texts)
        return self.calibrated_model.predict(X)

    def predict_proba(
        self,
        inputs: Union[pd.DataFrame, List[Tuple[str, str]], List[Dict[str, str]], List[str]],
    ) -> np.ndarray:
        """Predict calibrated posterior class probabilities summing to 1.0."""
        if not self.is_fitted or self.calibrated_model is None or self.vectorizer is None:
            raise RuntimeError("Pipeline is not fitted. Call fit() or load() first.")

        texts = self._format_inputs(inputs)
        X = self.vectorizer.transform(texts)
        return self.calibrated_model.predict_proba(X)

    def explain_prediction(
        self,
        title: str,
        body: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Explain an individual prediction by attributing feature contributions.

        Args:
            title: Issue title string.
            body: Issue body description string.
            top_k: Number of top contributing features to return (default: 5).

        Returns:
            Dictionary containing predicted label, probabilities, and top features with weights.
        """
        if not self.is_fitted or self.vectorizer is None or self.base_classifier is None:
            raise RuntimeError("Pipeline is not fitted. Call fit() or load() first.")

        composed = self.prepare_text(title, body)
        x_sparse = self.vectorizer.transform([composed])
        proba = self.predict_proba([(title, body)])[0]

        pred_idx = int(np.argmax(proba))
        predicted_class = self.classes_[pred_idx]
        predicted_confidence = float(proba[pred_idx])

        # Feature contribution = non-zero feature tfidf value * model coefficient for predicted class
        coef = self.base_classifier.coef_[pred_idx]
        x_dense = x_sparse.toarray()[0]
        nonzero_indices = np.where(x_dense > 0)[0]

        contributions = []
        for idx in nonzero_indices:
            term = self.feature_names_[idx]
            weight = float(coef[idx])
            tfidf_val = float(x_dense[idx])
            score = weight * tfidf_val
            contributions.append(
                {
                    "feature": term,
                    "contribution_score": round(score, 4),
                    "coefficient": round(weight, 4),
                    "tfidf_value": round(tfidf_val, 4),
                }
            )

        # Sort by contribution score descending
        contributions.sort(key=lambda item: item["contribution_score"], reverse=True)

        return {
            "title": title,
            "predicted_class": predicted_class,
            "confidence": round(predicted_confidence, 4),
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self.classes_, proba)
            },
            "top_contributing_features": contributions[:top_k],
        }

    def save(
        self,
        model_path: Union[str, Path] = "models/final_classical_nlp_pipeline.joblib",
        manifest_path: Union[str, Path] = "models/final_classical_nlp_manifest.json",
    ) -> Tuple[Path, Path]:
        """Serialize pipeline and export JSON metadata manifest."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline.")

        m_path = Path(model_path)
        mani_path = Path(manifest_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        mani_path.parent.mkdir(parents=True, exist_ok=True)

        # Save binary artifact
        joblib.dump(self, m_path)

        # Export metadata manifest
        manifest = {
            "model_name": "FinalClassicalNLPPipeline",
            "phase": "5",
            "step": "5.5",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "classes": self.classes_,
            "vocabulary_size": len(self.feature_names_),
            "max_features_budget": self.max_features,
            "title_multiplier": self.title_multiplier,
            "classifier_c": self.classifier_c,
            "calibration_method": "Platt Scaling (Sigmoid)",
            "preprocessing": "EnhancedIssueTextPreprocessor",
            "file_size_bytes": m_path.stat().st_size if m_path.exists() else 0,
        }

        with open(mani_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return m_path, mani_path

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = "models/final_classical_nlp_pipeline.joblib",
    ) -> "FinalClassicalNLPPipeline":
        """Load a serialized FinalClassicalNLPPipeline from disk."""
        m_path = Path(model_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found: {m_path}")
        loaded = joblib.load(m_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is {type(loaded)}, expected {cls}")
        return loaded


class FinalPipelineEvaluator:
    """Evaluates the final classical pipeline and produces head-to-head benchmarking against Phase 4."""

    def __init__(
        self,
        pipeline: FinalClassicalNLPPipeline,
        baseline_comparison_path: Union[str, Path] = "data/processed/evaluation/baseline_comparison.json",
    ):
        """Initialize the evaluator."""
        self.pipeline = pipeline
        self.baseline_comparison_path = Path(baseline_comparison_path)

    def evaluate_on_test(
        self,
        test_df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute full test metrics for the Phase 5 pipeline."""
        if label_col is None:
            if "target_label" in test_df.columns:
                label_col = "target_label"
            elif "canonical_label" in test_df.columns:
                label_col = "canonical_label"
            else:
                label_col = "primary_label"

        y_true = np.array(test_df[label_col].astype(str).tolist(), dtype=str)
        classes = self.pipeline.classes_

        start_time = time.perf_counter()
        y_pred = self.pipeline.predict(test_df)
        y_proba = self.pipeline.predict_proba(test_df)
        infer_time = time.perf_counter() - start_time

        acc = float(accuracy_score(y_true, y_pred))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )

        loss = float(log_loss(y_true, y_proba, labels=classes))

        # Multi-class Brier score
        brier_scores = []
        for i, cls in enumerate(classes):
            y_binary = (y_true == cls).astype(int)
            brier_scores.append(brier_score_loss(y_binary, y_proba[:, i]))
        mean_brier = float(np.mean(brier_scores))

        cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
        report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

        per_class_metrics = {}
        for idx, cls in enumerate(classes):
            per_class_metrics[cls] = {
                "precision": round(float(p_class[idx]), 4),
                "recall": round(float(r_class[idx]), 4),
                "f1_score": round(float(f1_class[idx]), 4),
                "support": int(np.sum(y_true == cls)),
            }

        return {
            "test_samples": len(y_true),
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(p_wt), 4),
            "weighted_recall": round(float(r_wt), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "per_class": per_class_metrics,
            "log_loss": round(loss, 4),
            "brier_score": round(mean_brier, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "inference_time_sec": round(infer_time, 4),
        }

    def compute_head_to_head(
        self,
        phase5_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute exact deltas comparing Phase 5 vs. Phase 4 baseline champion."""
        phase4_data = {}
        if self.baseline_comparison_path.exists():
            with open(self.baseline_comparison_path, "r", encoding="utf-8") as f:
                raw_baseline = json.load(f)
                # Extract SVM champion from Phase 4
                for row in raw_baseline.get("comparison_table", []):
                    if "Support Vector Machine" in row.get("model_name", ""):
                        phase4_data = row
                        break

        # Defaults if baseline file was missing
        p4_acc = phase4_data.get("accuracy", 0.8130)
        p4_macro_f1 = phase4_data.get("macro_f1", 0.8114)
        p4_bug_f1 = phase4_data.get("bug_f1", 0.8457)
        p4_feature_f1 = phase4_data.get("feature_f1", 0.8062)
        p4_other_f1 = phase4_data.get("other_f1", 0.7823)

        p5_acc = phase5_results["accuracy"]
        p5_macro_f1 = phase5_results["macro_f1"]
        p5_bug_f1 = phase5_results["per_class"]["bug"]["f1_score"]
        p5_feature_f1 = phase5_results["per_class"]["feature"]["f1_score"]
        p5_other_f1 = phase5_results["per_class"]["other"]["f1_score"]

        deltas = {
            "accuracy_delta": round(p5_acc - p4_acc, 4),
            "macro_f1_delta": round(p5_macro_f1 - p4_macro_f1, 4),
            "bug_f1_delta": round(p5_bug_f1 - p4_bug_f1, 4),
            "feature_f1_delta": round(p5_feature_f1 - p4_feature_f1, 4),
            "other_f1_delta": round(p5_other_f1 - p4_other_f1, 4),
        }

        return {
            "phase4_baseline": {
                "model_name": "Phase 4 Champion (SVM Baseline)",
                "accuracy": p4_acc,
                "macro_f1": p4_macro_f1,
                "bug_f1": p4_bug_f1,
                "feature_f1": p4_feature_f1,
                "other_f1": p4_other_f1,
            },
            "phase5_pipeline": {
                "model_name": "Phase 5 Final Classical NLP Pipeline",
                "accuracy": p5_acc,
                "macro_f1": p5_macro_f1,
                "bug_f1": p5_bug_f1,
                "feature_f1": p5_feature_f1,
                "other_f1": p5_other_f1,
            },
            "deltas": deltas,
        }

    def plot_comparison(
        self,
        head_to_head: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> Path:
        """Plot publication-quality grouped bar chart comparing Phase 4 vs Phase 5 (fig16)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = ["Accuracy", "Macro-F1", "Bug F1", "Feature F1", "Other F1"]
        p4_vals = [
            head_to_head["phase4_baseline"]["accuracy"],
            head_to_head["phase4_baseline"]["macro_f1"],
            head_to_head["phase4_baseline"]["bug_f1"],
            head_to_head["phase4_baseline"]["feature_f1"],
            head_to_head["phase4_baseline"]["other_f1"],
        ]
        p5_vals = [
            head_to_head["phase5_pipeline"]["accuracy"],
            head_to_head["phase5_pipeline"]["macro_f1"],
            head_to_head["phase5_pipeline"]["bug_f1"],
            head_to_head["phase5_pipeline"]["feature_f1"],
            head_to_head["phase5_pipeline"]["other_f1"],
        ]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(11, 6))

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width / 2, p4_vals, width, label="Phase 4 Baseline (Step 4.6)", color="#64748b", edgecolor="#1f2937", alpha=0.9)
        bars2 = ax.bar(x + width / 2, p5_vals, width, label="Phase 5 Final Classical Pipeline (Step 5.5)", color="#1d4ed8", edgecolor="#1f2937", alpha=0.9)

        ax.set_ylabel("Metric Score", fontsize=11, fontweight="medium")
        ax.set_title("Performance Comparison: Phase 4 Baseline vs. Phase 5 Final Classical NLP Pipeline\n(Held-Out Test Set, N=385)", fontsize=13, fontweight="bold", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10.5, fontweight="semibold")
        ax.set_ylim(0.70, 0.90)
        ax.legend(loc="lower right", frameon=True, fontsize=10)

        # Annotations on bars
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

        for bar, p4, p5 in zip(bars2, p4_vals, p5_vals):
            h = bar.get_height()
            diff = (p5 - p4) * 100
            diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            ax.annotate(f"{h:.4f}\n({diff_str})", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def save_results(
        self,
        phase5_results: Dict[str, Any],
        head_to_head: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> Path:
        """Persist structured evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "phase": "5",
                "step": "5.5",
                "title": "Establish Final Classical NLP Pipeline",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "test_samples": phase5_results["test_samples"],
                "champion_pipeline": "FinalClassicalNLPPipeline (Title 3x + Domain Preprocessing + Platt LinearSVC)",
            },
            "phase5_detailed_metrics": phase5_results,
            "head_to_head_comparison": head_to_head,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path


def main():
    """CLI execution entry point."""
    train_file = Path("data/processed/splits/train.csv")
    test_file = Path("data/processed/splits/test.csv")
    model_dest = Path("models/final_classical_nlp_pipeline.joblib")
    manifest_dest = Path("models/final_classical_nlp_manifest.json")
    fig_dest = Path("ml/nlp/figures/fig16_phase4_vs_phase5_performance.png")
    json_dest = Path("data/processed/evaluation/final_classical_nlp_evaluation.json")

    print("[Step 5.5] Initializing FinalClassicalNLPPipeline...")
    pipeline = FinalClassicalNLPPipeline(
        max_features=5000,
        random_state=42,
        title_multiplier=3,
        use_enhanced_preprocessing=True,
        classifier_c=0.5,
        cv_calibration=5,
    )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"[Step 5.5] Fitting pipeline on full train split ({len(train_df)} rows)...")
    start = time.perf_counter()
    pipeline.fit(train_df)
    fit_time = time.perf_counter() - start
    print(f"[Step 5.5] Fitted in {fit_time:.2f} seconds. Vocabulary dimension: {len(pipeline.feature_names_)} terms.")

    # Save model and manifest
    saved_model, saved_mani = pipeline.save(model_dest, manifest_dest)
    print(f"[Step 5.5] Saved model to: {saved_model} ({saved_model.stat().st_size / 1024:.1f} KB)")
    print(f"[Step 5.5] Saved manifest to: {saved_mani}")

    # Evaluate on held-out test data
    print(f"[Step 5.5] Evaluating on held-out test split ({len(test_df)} rows)...")
    evaluator = FinalPipelineEvaluator(pipeline)
    test_metrics = evaluator.evaluate_on_test(test_df)
    print(f"  -> Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  -> Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    for cls in pipeline.classes_:
        print(f"  -> {cls.upper()} F1: {test_metrics['per_class'][cls]['f1_score']:.4f}")

    # Head-to-Head Comparison against Phase 4
    head_to_head = evaluator.compute_head_to_head(test_metrics)
    print("\n[Step 5.5] Head-to-Head Comparison against Phase 4 Champion:")
    print(f"  Accuracy:  {head_to_head['phase4_baseline']['accuracy']:.4f} -> {head_to_head['phase5_pipeline']['accuracy']:.4f} ({head_to_head['deltas']['accuracy_delta']:+.4f})")
    print(f"  Macro-F1:  {head_to_head['phase4_baseline']['macro_f1']:.4f} -> {head_to_head['phase5_pipeline']['macro_f1']:.4f} ({head_to_head['deltas']['macro_f1_delta']:+.4f})")
    print(f"  Bug F1:    {head_to_head['phase4_baseline']['bug_f1']:.4f} -> {head_to_head['phase5_pipeline']['bug_f1']:.4f} ({head_to_head['deltas']['bug_f1_delta']:+.4f})")
    print(f"  Feature F1:{head_to_head['phase4_baseline']['feature_f1']:.4f} -> {head_to_head['phase5_pipeline']['feature_f1']:.4f} ({head_to_head['deltas']['feature_f1_delta']:+.4f})")
    print(f"  Other F1:  {head_to_head['phase4_baseline']['other_f1']:.4f} -> {head_to_head['phase5_pipeline']['other_f1']:.4f} ({head_to_head['deltas']['other_f1_delta']:+.4f})")

    # Sample explanation verification
    sample_expl = pipeline.explain_prediction(
        title="NullPointerException in authentication module",
        body="Unexpected fatal crash with exit code 139 when token is empty.",
    )
    print("\n[Step 5.5] Sample Prediction & Explanation:")
    print(f"  Predicted Class: {sample_expl['predicted_class']} (Confidence: {sample_expl['confidence']:.4f})")
    print("  Top Contributing Terms:", [item["feature"] for item in sample_expl["top_contributing_features"]])

    fig_path = evaluator.plot_comparison(head_to_head, fig_dest)
    json_path = evaluator.save_results(test_metrics, head_to_head, json_dest)

    print(f"\n[Step 5.5] Saved comparison plot to: {fig_path}")
    print(f"[Step 5.5] Saved evaluation report to: {json_path}")


if __name__ == "__main__":
    main()
