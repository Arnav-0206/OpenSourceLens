"""Vocabulary and Feature Salience Analysis for OpenSourceLens (Phase 5 NLP Enhancement).

Extracts and analyzes class-specific linear coefficients, Chi-squared (chi2) feature relevance,
domain-specific token contributions (hex_addr, commit_hash, etc.), and assesses vocabulary
size sensitivity (pruning curves from 500 to 10,000 features).
Generates publication-quality visualizations and structured evaluation JSON artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor


class VocabularyFeatureAnalyzer:
    """Performs vocabulary extraction, coefficient analysis, and feature dimension audits."""

    DOMAIN_TOKENS = [
        "hex_addr",
        "commit_hash",
        "version_num",
        "issue_ref",
        "file_path",
        "exception_type",
    ]

    DEFAULT_VOCAB_SIZES = [500, 1000, 2000, 3500, 5000, 7500, 10000]

    def __init__(
        self,
        max_features: int = 5000,
        random_state: int = 42,
        title_multiplier: int = 3,
        use_enhanced_preprocessing: bool = True,
        classifier_c: float = 0.5,
    ):
        """Initialize the analyzer.

        Args:
            max_features: Default vocabulary capacity (default: 5000).
            random_state: Random state for reproducibility (default: 42).
            title_multiplier: Weighting for issue titles (default: 3).
            use_enhanced_preprocessing: Apply EnhancedIssueTextPreprocessor (default: True).
            classifier_c: Regularization parameter C for LinearSVC (default: 0.5).
        """
        self.max_features = max_features
        self.random_state = random_state
        self.title_multiplier = title_multiplier
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.classifier_c = classifier_c

        self.preprocessor = EnhancedIssueTextPreprocessor() if use_enhanced_preprocessing else None
        self.classes_: List[str] = []
        self.feature_names_: List[str] = []
        self.coefficients_by_class_: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.chi2_rankings_: List[Dict[str, Any]] = []
        self.domain_token_audit_: List[Dict[str, Any]] = []
        self.vocab_curve_results_: Dict[str, Any] = {}

    def prepare_text(self, title: str, body: str) -> str:
        """Clean and compose title and body into unified text string."""
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

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Resolve columns and prepare texts and target array."""
        if label_col is None:
            if "target_label" in df.columns:
                label_col = "target_label"
            elif "canonical_label" in df.columns:
                label_col = "canonical_label"
            elif "primary_label" in df.columns:
                label_col = "primary_label"
            else:
                raise ValueError("DataFrame must contain 'target_label', 'canonical_label', or 'primary_label'")

        if "clean_title" in df.columns:
            title_col = "clean_title"
        elif "title" in df.columns:
            title_col = "title"
        else:
            title_col = df.columns[0]

        if "clean_body" in df.columns:
            body_col = "clean_body"
        elif "body" in df.columns:
            body_col = "body"
        else:
            body_col = title_col

        texts = [
            self.prepare_text(row.get(title_col, ""), row.get(body_col, ""))
            for _, row in df.iterrows()
        ]
        labels = df[label_col].astype(str).values

        if not self.classes_:
            self.classes_ = sorted(list(np.unique(labels)))

        return texts, labels

    def analyze_features_and_salience(
        self,
        train_df: pd.DataFrame,
        top_k_positive: int = 20,
        top_k_negative: int = 10,
    ) -> Dict[str, Any]:
        """Fit vectorizer and LinearSVC on train data, extracting coefficients, chi2, and domain tokens.

        Args:
            train_df: Training DataFrame.
            top_k_positive: Number of top positive features per class to retain (default: 20).
            top_k_negative: Number of top negative features per class to retain (default: 10).

        Returns:
            Dictionary containing class coefficients, chi2 rankings, and domain token audit.
        """
        texts, y = self.prepare_dataset(train_df)
        self.classes_ = sorted(list(np.unique(y)))

        min_df = 1 if len(texts) < 50 else 2
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=min_df,
        )
        X = vectorizer.fit_transform(texts)
        self.feature_names_ = vectorizer.get_feature_names_out().tolist()

        # 1. Fit LinearSVC
        clf = LinearSVC(
            C=self.classifier_c,
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=2000,
        )
        clf.fit(X, y)

        # 2. Extract Coefficients per class
        coefs = clf.coef_  # Shape: (n_classes, n_features)
        coefficients_by_class: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for idx, cls in enumerate(self.classes_):
            cls_coefs = coefs[idx]
            sorted_indices_desc = np.argsort(cls_coefs)[::-1]
            sorted_indices_asc = np.argsort(cls_coefs)

            pos_features = [
                {
                    "feature": self.feature_names_[i],
                    "weight": round(float(cls_coefs[i]), 4),
                    "rank": rank + 1,
                }
                for rank, i in enumerate(sorted_indices_desc[:top_k_positive])
            ]

            neg_features = [
                {
                    "feature": self.feature_names_[i],
                    "weight": round(float(cls_coefs[i]), 4),
                    "rank": rank + 1,
                }
                for rank, i in enumerate(sorted_indices_asc[:top_k_negative])
            ]

            coefficients_by_class[cls] = {
                "positive": pos_features,
                "negative": neg_features,
            }

        self.coefficients_by_class_ = coefficients_by_class

        # 3. Statistical Chi-squared (chi2) ranking
        chi2_scores, p_values = chi2(X, y)
        sorted_chi2_indices = np.argsort(chi2_scores)[::-1]

        chi2_rankings = [
            {
                "feature": self.feature_names_[i],
                "chi2_score": round(float(chi2_scores[i]), 4),
                "p_value": float(p_values[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(sorted_chi2_indices[:50])
        ]
        self.chi2_rankings_ = chi2_rankings

        # 4. Domain Token Audit
        feat_to_idx = {feat: idx for idx, feat in enumerate(self.feature_names_)}
        chi2_rank_map = {self.feature_names_[i]: rank + 1 for rank, i in enumerate(sorted_chi2_indices)}

        domain_audit = []
        for d_token in self.DOMAIN_TOKENS:
            matching_feats = [f for f in self.feature_names_ if d_token in f]
            for feat in matching_feats:
                f_idx = feat_to_idx[feat]
                class_weights = {
                    cls: round(float(coefs[c_idx, f_idx]), 4)
                    for c_idx, cls in enumerate(self.classes_)
                }
                domain_audit.append(
                    {
                        "feature": feat,
                        "base_domain_type": d_token,
                        "chi2_score": round(float(chi2_scores[f_idx]), 4),
                        "chi2_rank": int(chi2_rank_map.get(feat, len(self.feature_names_))),
                        "class_weights": class_weights,
                    }
                )

        self.domain_token_audit_ = domain_audit

        return {
            "coefficients_by_class": coefficients_by_class,
            "top_chi2_features": chi2_rankings[:25],
            "domain_token_audit": domain_audit,
        }

    def evaluate_vocab_size_sensitivity(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        vocab_sizes: Optional[List[int]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Measure cross-validation and test Macro-F1 across vocabulary capacities.

        Args:
            train_df: Training DataFrame.
            test_df: Optional held-out test DataFrame.
            vocab_sizes: List of vocabulary size budgets to evaluate.
            n_splits: Number of cross-validation folds (default: 5).

        Returns:
            Dictionary of vocabulary size sensitivity results.
        """
        train_texts, y_train = self.prepare_dataset(train_df)
        self.classes_ = sorted(list(np.unique(y_train)))

        has_test = test_df is not None
        test_texts, y_test = ([], np.array([]))
        if has_test:
            test_texts, y_test = self.prepare_dataset(test_df)

        if vocab_sizes is None:
            # If tiny dataset in unit test, scale budgets down
            if len(train_texts) < 50:
                vocab_sizes = [50, 100, 200, 500]
            else:
                vocab_sizes = self.DEFAULT_VOCAB_SIZES

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        min_df = 1 if len(train_texts) < 50 else 2
        curve_data: List[Dict[str, Any]] = []

        for v_size in vocab_sizes:
            fold_macro_f1s = []
            fold_accs = []

            for train_idx, val_idx in skf.split(train_texts, y_train):
                f_train_texts = [train_texts[i] for i in train_idx]
                f_val_texts = [train_texts[i] for i in val_idx]
                f_y_train, f_y_val = y_train[train_idx], y_train[val_idx]

                vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=v_size,
                    sublinear_tf=True,
                    min_df=min_df,
                )
                X_tr = vec.fit_transform(f_train_texts)
                X_va = vec.transform(f_val_texts)

                clf = LinearSVC(
                    C=self.classifier_c,
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=2000,
                )
                clf.fit(X_tr, f_y_train)
                y_pred = clf.predict(X_va)

                _, _, f1, _ = precision_recall_fscore_support(
                    f_y_val, y_pred, average="macro", zero_division=0
                )
                acc = accuracy_score(f_y_val, y_pred)
                fold_macro_f1s.append(float(f1))
                fold_accs.append(float(acc))

            # Test evaluation if test_df is supplied
            test_macro_f1 = None
            test_acc = None
            if has_test:
                vec_full = TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=v_size,
                    sublinear_tf=True,
                    min_df=min_df,
                )
                X_full = vec_full.fit_transform(train_texts)
                X_te = vec_full.transform(test_texts)

                clf_full = LinearSVC(
                    C=self.classifier_c,
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=2000,
                )
                clf_full.fit(X_full, y_train)
                y_te_pred = clf_full.predict(X_te)

                _, _, test_f1, _ = precision_recall_fscore_support(
                    y_test, y_te_pred, average="macro", zero_division=0
                )
                test_macro_f1 = float(test_f1)
                test_acc = float(accuracy_score(y_test, y_te_pred))

            curve_data.append(
                {
                    "vocab_size": int(v_size),
                    "cv_macro_f1_mean": round(float(np.mean(fold_macro_f1s)), 4),
                    "cv_macro_f1_std": round(float(np.std(fold_macro_f1s)), 4),
                    "cv_accuracy_mean": round(float(np.mean(fold_accs)), 4),
                    "test_macro_f1": round(test_macro_f1, 4) if test_macro_f1 is not None else None,
                    "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
                }
            )

        self.vocab_curve_results_ = {
            "vocab_sizes": vocab_sizes,
            "curve_data": curve_data,
        }
        return self.vocab_curve_results_

    def plot_top_features_per_class(self, output_path: Union[str, Path]) -> Path:
        """Plot horizontal bar charts of top positive and negative features per class (fig14).

        Args:
            output_path: Path to target image file.

        Returns:
            Resolved Path of the saved image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.coefficients_by_class_:
            raise ValueError("Feature analysis not run. Call analyze_features_and_salience first.")

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=False)

        class_palette = {
            "bug": {"pos": "#dc2626", "neg": "#f87171"},       # Red theme for bugs
            "feature": {"pos": "#2563eb", "neg": "#60a5fa"},   # Blue theme for features
            "other": {"pos": "#059669", "neg": "#34d399"},     # Green theme for chores/other
        }

        for idx, cls in enumerate(self.classes_):
            ax = axes[idx]
            class_data = self.coefficients_by_class_[cls]
            pos = class_data["positive"][:15][::-1]  # Top 15 positive, ascending for barh
            neg = class_data["negative"][:5][::-1]   # Top 5 negative, ascending for barh

            features = [item["feature"] for item in neg] + [item["feature"] for item in pos]
            weights = [item["weight"] for item in neg] + [item["weight"] for item in pos]

            colors = [
                class_palette.get(cls, {}).get("neg", "#9ca3af") if w < 0 else class_palette.get(cls, {}).get("pos", "#3b82f6")
                for w in weights
            ]

            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, weights, color=colors, edgecolor="#1f2937", alpha=0.9)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9.5, fontweight="medium")
            ax.axvline(0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.set_title(f"Class: '{cls.upper()}' Salient Features", fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("LinearSVC Coefficient Weight", fontsize=10.5)

            # Value labels on bars
            for bar, val in zip(bars, weights):
                offset = 0.05 if val >= 0 else -0.05
                ha = "left" if val >= 0 else "right"
                ax.text(
                    val + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}",
                    va="center",
                    ha=ha,
                    fontsize=8.5,
                    fontweight="semibold",
                )

        plt.suptitle(
            "Phase 5 NLP Enhancement — Top Informative Vocabulary Features per Class (Step 5.4)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def plot_vocab_size_sensitivity(self, output_path: Union[str, Path]) -> Path:
        """Plot performance curve vs. vocabulary size budget (fig15).

        Args:
            output_path: Path to target image file.

        Returns:
            Resolved Path of the saved image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.vocab_curve_results_:
            raise ValueError("Vocab curve not evaluated. Call evaluate_vocab_size_sensitivity first.")

        curve = self.vocab_curve_results_["curve_data"]
        sizes = [item["vocab_size"] for item in curve]
        cv_f1s = [item["cv_macro_f1_mean"] for item in curve]
        cv_stds = [item["cv_macro_f1_std"] for item in curve]
        test_f1s = [item.get("test_macro_f1") for item in curve]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(10, 5.5))

        # CV Curve with std error band
        ax.plot(sizes, cv_f1s, marker="o", color="#1d4ed8", linewidth=2.2, label="5-Fold CV Macro-F1")
        ax.fill_between(
            sizes,
            np.array(cv_f1s) - np.array(cv_stds),
            np.array(cv_f1s) + np.array(cv_stds),
            color="#3b82f6",
            alpha=0.18,
            label="CV ±1 Std Dev Band",
        )

        # Held-out test curve if available
        if any(t is not None for t in test_f1s):
            ax.plot(sizes, test_f1s, marker="s", color="#059669", linewidth=2.0, linestyle="--", label="Held-Out Test Macro-F1 (N=385)")

        # Value annotations
        for s, f1 in zip(sizes, cv_f1s):
            ax.annotate(
                f"{f1:.4f}",
                (s, f1),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                fontweight="semibold",
                color="#1e3a8a",
            )

        ax.set_title(
            "Vocabulary Dimension Pruning Sensitivity Curve (500 to 10,000 Features)",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Vocabulary Size Budget (Max Features)", fontsize=11)
        ax.set_ylabel("Macro-F1 Score", fontsize=11)
        ax.set_ylim(0.75, 0.88)
        ax.set_xticks(sizes)
        ax.legend(loc="lower right", frameon=True, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def save_results(self, output_path: Union[str, Path]) -> Path:
        """Persist structured vocabulary and feature analysis results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "step": "5.4",
                "title": "Analyze Vocabulary and Features",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classifier": "LinearSVC",
                "classifier_c": self.classifier_c,
                "max_features_budget": self.max_features,
                "classes": self.classes_,
                "total_vocabulary_extracted": len(self.feature_names_),
                "analyzed_domain_tokens": self.DOMAIN_TOKENS,
            },
            "coefficients_by_class": self.coefficients_by_class_,
            "chi2_top_discriminative_features": self.chi2_rankings_[:30],
            "domain_token_audit": self.domain_token_audit_,
            "vocab_size_sensitivity_curve": self.vocab_curve_results_.get("curve_data", []),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path

    def run_all(
        self,
        train_path: Union[str, Path],
        test_path: Union[str, Path],
        top_features_fig_path: Union[str, Path],
        vocab_curve_fig_path: Union[str, Path],
        json_output_path: Union[str, Path],
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Orchestrate the complete Step 5.4 vocabulary analysis pipeline."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"[Step 5.4] Loaded train ({len(train_df)} rows) and test ({len(test_df)} rows).")
        print("[Step 5.4] Extracting class-specific linear coefficients and Chi-squared features...")
        salience_data = self.analyze_features_and_salience(train_df)

        for cls in self.classes_:
            top_pos = [item["feature"] for item in self.coefficients_by_class_[cls]["positive"][:5]]
            print(f"  -> '{cls.upper()}' top positive indicators: {', '.join(top_pos)}")

        print(f"[Step 5.4] Audited {len(self.domain_token_audit_)} domain-specific feature occurrences.")
        for d in self.domain_token_audit_[:5]:
            print(f"  -> Domain Token '{d['feature']}': chi2_rank={d['chi2_rank']}, class_weights={d['class_weights']}")

        print("[Step 5.4] Evaluating vocabulary size sensitivity curve across budgets...")
        curve_results = self.evaluate_vocab_size_sensitivity(
            train_df=train_df,
            test_df=test_df,
            n_splits=n_splits,
        )

        for pt in curve_results["curve_data"]:
            test_str = f", Test-F1 = {pt['test_macro_f1']:.4f}" if pt.get("test_macro_f1") is not None else ""
            print(f"  -> Vocab Size {pt['vocab_size']:>5}: CV Macro-F1 = {pt['cv_macro_f1_mean']:.4f} (±{pt['cv_macro_f1_std']:.4f}){test_str}")

        fig14_path = self.plot_top_features_per_class(top_features_fig_path)
        fig15_path = self.plot_vocab_size_sensitivity(vocab_curve_fig_path)
        json_path = self.save_results(json_output_path)

        print(f"[Step 5.4] Top features figure saved to: {fig14_path}")
        print(f"[Step 5.4] Vocab curve figure saved to: {fig15_path}")
        print(f"[Step 5.4] Structured results saved to: {json_path}")

        return {
            "salience_data": salience_data,
            "curve_results": curve_results,
            "fig14_path": str(fig14_path),
            "fig15_path": str(fig15_path),
            "json_path": str(json_path),
        }


def main():
    """CLI execution entry point."""
    train_file = Path("data/processed/splits/train.csv")
    test_file = Path("data/processed/splits/test.csv")
    fig14_dest = Path("ml/nlp/figures/fig14_top_features_per_class.png")
    fig15_dest = Path("ml/nlp/figures/fig15_vocab_size_vs_performance.png")
    json_dest = Path("data/processed/evaluation/vocab_and_features_analysis.json")

    analyzer = VocabularyFeatureAnalyzer(
        max_features=5000,
        random_state=42,
        title_multiplier=3,
        use_enhanced_preprocessing=True,
        classifier_c=0.5,
    )

    analyzer.run_all(
        train_path=train_file,
        test_path=test_file,
        top_features_fig_path=fig14_dest,
        vocab_curve_fig_path=fig15_dest,
        json_output_path=json_dest,
        n_splits=5,
    )


if __name__ == "__main__":
    main()
