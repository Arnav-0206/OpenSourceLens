"""Title and Body Text Composition Experiments for OpenSourceLens (Phase 5 NLP Enhancement).

Evaluates text composition strategies (title-only, body-only, unweighted concatenation,
title-frequency weighting, and dual-channel TF-IDF feature unions) via 5-fold Stratified
Cross-Validation, tests generalization on held-out test splits, and generates comparative
visualizations and structured evaluation artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor


class TitleBodyExperimentRunner:
    """Orchestrates title vs. body weighting and composition experiments."""

    STRATEGIES = [
        "title_only",
        "body_only",
        "title_body_1x",
        "title_2x_body",
        "title_3x_body",
        "dual_channel_union",
    ]

    STRATEGY_DISPLAY_NAMES = {
        "title_only": "Title Only",
        "body_only": "Body Only",
        "title_body_1x": "Title + Body (1x)",
        "title_2x_body": "Title (2x) + Body",
        "title_3x_body": "Title (3x) + Body",
        "dual_channel_union": "Dual-Channel Union",
    }

    def __init__(
        self,
        max_features: int = 5000,
        random_state: int = 42,
        use_enhanced_preprocessing: bool = True,
    ):
        """Initialize the experiment runner.

        Args:
            max_features: Total vocabulary dimension budget (default: 5000).
            random_state: Random state for deterministic folds (default: 42).
            use_enhanced_preprocessing: Apply EnhancedIssueTextPreprocessor first (default: True).
        """
        self.max_features = max_features
        self.random_state = random_state
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.preprocessor = EnhancedIssueTextPreprocessor() if use_enhanced_preprocessing else None

    def _ensure_text_columns(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Extract or enhance title and body Series from DataFrame."""
        if self.use_enhanced_preprocessing and self.preprocessor is not None:
            t_col = "enhanced_title" if "enhanced_title" in df.columns else ("clean_title" if "clean_title" in df.columns else "title")
            b_col = "enhanced_body" if "enhanced_body" in df.columns else ("clean_body" if "clean_body" in df.columns else "body")

            if "enhanced_title" not in df.columns or "enhanced_body" not in df.columns:
                enhanced_df = self.preprocessor.transform_dataframe(df)
                return enhanced_df["enhanced_title"], enhanced_df["enhanced_body"]
            return df[t_col].fillna("").astype(str), df[b_col].fillna("").astype(str)

        # Raw / basic fallback
        t_col = "clean_title" if "clean_title" in df.columns else ("title" if "title" in df.columns else None)
        b_col = "clean_body" if "clean_body" in df.columns else ("body" if "body" in df.columns else None)

        titles = df[t_col].fillna("").astype(str) if t_col else pd.Series([""] * len(df))
        bodies = df[b_col].fillna("").astype(str) if b_col else pd.Series([""] * len(df))
        return titles, bodies

    def prepare_text_series(
        self,
        df: pd.DataFrame,
        strategy: str,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """Construct input text representations for a given strategy.

        Args:
            df: Input DataFrame.
            strategy: One of STRATEGIES.

        Returns:
            pd.Series for single-text strategies, or (title_series, body_series) tuple for dual_channel_union.

        Raises:
            ValueError: If strategy is unrecognized.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Expected one of {self.STRATEGIES}")

        titles, bodies = self._ensure_text_columns(df)

        if strategy == "title_only":
            return titles
        elif strategy == "body_only":
            return bodies
        elif strategy == "title_body_1x":
            return (titles + " " + bodies).str.strip()
        elif strategy == "title_2x_body":
            return (titles + " " + titles + " " + bodies).str.strip()
        elif strategy == "title_3x_body":
            return (titles + " " + titles + " " + titles + " " + bodies).str.strip()
        elif strategy == "dual_channel_union":
            return titles, bodies

        raise ValueError(f"Unhandled strategy: {strategy}")

    def _fit_transform_strategy(
        self,
        train_df: pd.DataFrame,
        strategy: str,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[scipy.sparse.csr_matrix, Optional[scipy.sparse.csr_matrix]]:
        """Fit vectorizer(s) on train_df and transform train_df (and test_df if provided)."""
        train_data = self.prepare_text_series(train_df, strategy)
        test_data = self.prepare_text_series(test_df, strategy) if test_df is not None else None

        # Dynamically set min_df for small datasets (e.g. unit tests with <50 samples)
        effective_min_df = 1 if len(train_df) < 50 else (1 if strategy == "title_only" else 2)

        if strategy != "dual_channel_union":
            vec = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=effective_min_df,
                max_df=0.85,
                sublinear_tf=True,
                stop_words="english",
            )
            X_train = vec.fit_transform(train_data)
            X_test = vec.transform(test_data) if test_data is not None else None
            return X_train.tocsr(), X_test.tocsr() if X_test is not None else None

        # Dual-Channel Union: 1500 title features + 3500 body features
        train_titles, train_bodies = train_data  # type: ignore
        test_titles, test_bodies = test_data if test_data is not None else (None, None)  # type: ignore

        vec_title = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.85,
            sublinear_tf=True,
            stop_words="english",
        )
        vec_body = TfidfVectorizer(
            max_features=3500,
            ngram_range=(1, 2),
            min_df=effective_min_df,
            max_df=0.85,
            sublinear_tf=True,
            stop_words="english",
        )

        X_tr_title = vec_title.fit_transform(train_titles)
        X_tr_body = vec_body.fit_transform(train_bodies)
        X_train = scipy.sparse.hstack([X_tr_title, X_tr_body], format="csr")

        X_test = None
        if test_titles is not None and test_bodies is not None:
            X_te_title = vec_title.transform(test_titles)
            X_te_body = vec_body.transform(test_bodies)
            X_test = scipy.sparse.hstack([X_te_title, X_te_body], format="csr")

        return X_train, X_test

    def run_cross_validation_experiments(
        self,
        train_df: pd.DataFrame,
        cv: int = 5,
        target_col: str = "target_label",
    ) -> Dict[str, Any]:
        """Execute 5-fold Stratified CV across all 6 strategies and rank by Macro-F1.

        Args:
            train_df: Training DataFrame.
            cv: Number of cross-validation folds (default: 5).
            target_col: Ground truth label column name (default: 'target_label').

        Returns:
            Dictionary containing benchmark metrics, rankings, and fold statistics.
        """
        y_train = np.asarray(train_df[target_col])
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        strategy_results: List[Dict[str, Any]] = []

        for strat in self.STRATEGIES:
            fold_macro_f1s: List[float] = []
            fold_accuracies: List[float] = []
            fold_weighted_f1s: List[float] = []

            for train_idx, val_idx in skf.split(train_df, y_train):
                fold_train = train_df.iloc[train_idx].reset_index(drop=True)
                fold_val = train_df.iloc[val_idx].reset_index(drop=True)

                y_tr_fold = y_train[train_idx]
                y_val_fold = y_train[val_idx]

                X_tr, X_val = self._fit_transform_strategy(fold_train, strat, fold_val)

                clf = LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=2000,
                    dual="auto",
                )
                clf.fit(X_tr, y_tr_fold)
                y_pred = clf.predict(X_val)

                acc = float(accuracy_score(y_val_fold, y_pred))
                macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                    y_val_fold, y_pred, average="macro", zero_division=0
                )
                _, _, weighted_f1, _ = precision_recall_fscore_support(
                    y_val_fold, y_pred, average="weighted", zero_division=0
                )

                fold_accuracies.append(acc)
                fold_macro_f1s.append(float(macro_f1))
                fold_weighted_f1s.append(float(weighted_f1))

            strategy_results.append({
                "strategy": strat,
                "display_name": self.STRATEGY_DISPLAY_NAMES[strat],
                "mean_macro_f1": round(float(np.mean(fold_macro_f1s)), 4),
                "std_macro_f1": round(float(np.std(fold_macro_f1s)), 4),
                "mean_accuracy": round(float(np.mean(fold_accuracies)), 4),
                "std_accuracy": round(float(np.std(fold_accuracies)), 4),
                "mean_weighted_f1": round(float(np.mean(fold_weighted_f1s)), 4),
                "fold_macro_f1s": [round(f, 4) for f in fold_macro_f1s],
            })

        # Rank strategies descending by mean Macro-F1
        ranked = sorted(strategy_results, key=lambda x: (x["mean_macro_f1"], x["mean_accuracy"]), reverse=True)
        best_strategy = ranked[0]["strategy"]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cv_folds": cv,
            "train_sample_count": len(train_df),
            "best_strategy": best_strategy,
            "best_strategy_display": self.STRATEGY_DISPLAY_NAMES[best_strategy],
            "ranked_strategies": ranked,
        }

    def evaluate_best_on_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        best_strategy: str,
        target_col: str = "target_label",
    ) -> Dict[str, Any]:
        """Train the winning strategy on the entire training set and evaluate on held-out test data.

        Args:
            train_df: Training DataFrame (1,537 samples).
            test_df: Held-out test DataFrame (385 samples).
            best_strategy: Strategy selected from CV experiments.
            target_col: Target column name.

        Returns:
            Dictionary of held-out test metrics.
        """
        y_train = np.asarray(train_df[target_col])
        y_test = np.asarray(test_df[target_col])

        X_train, X_test = self._fit_transform_strategy(train_df, best_strategy, test_df)

        clf = LinearSVC(
            C=0.5,
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=2000,
            dual="auto",
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        classes = sorted(np.unique(y_train).tolist())
        per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, zero_division=0
        )

        per_class = {}
        for idx, cls in enumerate(classes):
            per_class[cls] = {
                "precision": round(float(per_class_p[idx]), 4),
                "recall": round(float(per_class_r[idx]), 4),
                "f1_score": round(float(per_class_f1[idx]), 4),
                "support": int(per_class_sup[idx]),
            }

        return {
            "strategy": best_strategy,
            "display_name": self.STRATEGY_DISPLAY_NAMES[best_strategy],
            "test_sample_count": len(test_df),
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
        }

    def plot_comparison(
        self,
        cv_results: Dict[str, Any],
        output_path: Union[str, Path] = "ml/nlp/figures/fig12_title_body_weighting_comparison.png",
    ) -> Path:
        """Plot publication-quality comparative bar chart showing CV performance with error bars.

        Args:
            cv_results: Output from run_cross_validation_experiments().
            output_path: Target image file path.

        Returns:
            Path of saved figure.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        ranked = cv_results["ranked_strategies"]
        names = [r["display_name"] for r in ranked]
        macro_f1s = [r["mean_macro_f1"] for r in ranked]
        f1_stds = [r["std_macro_f1"] for r in ranked]
        accuracies = [r["mean_accuracy"] for r in ranked]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        x = np.arange(len(names))
        width = 0.35

        rects1 = ax.bar(x - width / 2, macro_f1s, width, yerr=f1_stds, label="Mean Macro-F1", capsize=4, color="#2b5c8f", edgecolor="black")
        rects2 = ax.bar(x + width / 2, accuracies, width, label="Mean Accuracy", color="#78c679", edgecolor="black")

        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(
            f"OpenSourceLens — Title vs. Body Text Composition Benchmark (5-Fold CV, N={cv_results['train_sample_count']})",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10, fontweight="bold")
        ax.set_ylim(0.5, 0.92)
        ax.legend(loc="lower right", frameon=True)

        # Annotate highest performing bar
        best_idx = 0
        best_val = macro_f1s[0]
        ax.annotate(
            f"Best: {best_val:.4f}",
            xy=(best_idx - width / 2, best_val),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

        plt.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return out

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path] = "data/processed/evaluation/title_body_experiments.json",
    ) -> Path:
        """Persist structured experimental results to JSON.

        Args:
            results: Results dictionary.
            output_path: File path.

        Returns:
            Path of saved JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return path
