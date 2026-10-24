"""Text Representation Experiments for OpenSourceLens (Phase 5 NLP Enhancement).

Evaluates diverse textual feature representations (word n-grams, character-wb n-grams,
hybrid word+character feature unions, binary term occurrence, and sublinear scaling variations)
via 5-fold Stratified Cross-Validation on training data and tests generalization on held-out test data.
Generates comparative visualizations and structured evaluation JSON artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor


class TextRepresentationExperimentRunner:
    """Orchestrates textual feature representation experiments on GitHub issues."""

    STRATEGIES = [
        "word_unigrams",
        "word_unigrams_bigrams",
        "word_trigrams",
        "char_wb_ngrams",
        "hybrid_word_char",
        "binary_occurrence",
        "standard_tfidf",
        "term_frequency_only",
    ]

    STRATEGY_DISPLAY_NAMES = {
        "word_unigrams": "Word (1, 1) TF-IDF",
        "word_unigrams_bigrams": "Word (1, 2) TF-IDF (Baseline)",
        "word_trigrams": "Word (1, 3) TF-IDF",
        "char_wb_ngrams": "Char-WB (3, 5) TF-IDF",
        "hybrid_word_char": "Hybrid Word + Char Union",
        "binary_occurrence": "Binary Word (1, 2)",
        "standard_tfidf": "Standard Word (1, 2) TF-IDF",
        "term_frequency_only": "Term Frequency Only (No IDF)",
    }

    def __init__(
        self,
        max_features: int = 5000,
        random_state: int = 42,
        title_multiplier: int = 3,
        use_enhanced_preprocessing: bool = True,
        classifier_c: float = 0.5,
    ):
        """Initialize the experiment runner.

        Args:
            max_features: Total vocabulary dimension budget (default: 5000).
            random_state: Random state for deterministic cross-validation (default: 42).
            title_multiplier: Number of times to repeat title in composition (default: 3).
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
        self.cv_results_: Dict[str, Dict[str, Any]] = {}
        self.test_results_: Dict[str, Dict[str, Any]] = {}

    def build_vectorizer(
        self,
        strategy_name: str,
        corpus_size: Optional[int] = None,
    ) -> Union[TfidfVectorizer, FeatureUnion]:
        """Build a scikit-learn vectorizer or FeatureUnion corresponding to a strategy.

        Args:
            strategy_name: Name of the representation strategy.
            corpus_size: Size of the training corpus (used to adapt min_df for small test sets).

        Returns:
            Configured TfidfVectorizer or FeatureUnion pipeline component.
        """
        if strategy_name not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Must be one of {self.STRATEGIES}"
            )

        # Dynamic min_df adaptation for small unit test fixtures
        min_df = 1 if (corpus_size is not None and corpus_size < 50) else 2

        if strategy_name == "word_unigrams":
            return TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=min_df,
            )

        elif strategy_name == "word_unigrams_bigrams":
            return TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=min_df,
            )

        elif strategy_name == "word_trigrams":
            return TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=min_df,
            )

        elif strategy_name == "char_wb_ngrams":
            return TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=self.max_features,
                sublinear_tf=True,
                min_df=min_df,
            )

        elif strategy_name == "hybrid_word_char":
            # 70% word n-grams (1, 2), 30% char_wb n-grams (3, 5)
            word_features = int(self.max_features * 0.70)
            char_features = self.max_features - word_features

            return FeatureUnion(
                [
                    (
                        "word_tfidf",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            max_features=word_features,
                            sublinear_tf=True,
                            min_df=min_df,
                        ),
                    ),
                    (
                        "char_wb_tfidf",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            max_features=char_features,
                            sublinear_tf=True,
                            min_df=min_df,
                        ),
                    ),
                ]
            )

        elif strategy_name == "binary_occurrence":
            return TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                binary=True,
                sublinear_tf=False,
                use_idf=True,
                min_df=min_df,
            )

        elif strategy_name == "standard_tfidf":
            return TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                binary=False,
                sublinear_tf=False,
                use_idf=True,
                min_df=min_df,
            )

        elif strategy_name == "term_frequency_only":
            return TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=self.max_features,
                binary=False,
                sublinear_tf=True,
                use_idf=False,
                min_df=min_df,
            )

        raise ValueError(f"Unhandled strategy: {strategy_name}")

    def prepare_text(self, title: str, body: str) -> str:
        """Compose title and body into normalized text according to title_multiplier.

        Args:
            title: Raw or processed issue title.
            body: Raw or processed issue body.

        Returns:
            Composed, normalized single text string.
        """
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
        """Transform a DataFrame into prepared text representations and label array.

        Args:
            df: DataFrame containing issue data.
            label_col: Optional column name for labels.

        Returns:
            Tuple of (texts, labels).
        """
        if label_col is None:
            if "target_label" in df.columns:
                label_col = "target_label"
            elif "canonical_label" in df.columns:
                label_col = "canonical_label"
            elif "primary_label" in df.columns:
                label_col = "primary_label"
            else:
                raise ValueError(
                    "DataFrame must contain 'target_label', 'canonical_label', or 'primary_label'"
                )

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

    def run_cross_validation(
        self,
        train_df: pd.DataFrame,
        n_splits: int = 5,
        label_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run Stratified K-Fold cross-validation across all representation strategies.

        Args:
            train_df: Training DataFrame.
            n_splits: Number of cross-validation folds (default: 5).
            label_col: Target label column name.

        Returns:
            Dictionary mapping strategy name to performance statistics.
        """
        texts, y = self.prepare_dataset(train_df, label_col=label_col)
        self.classes_ = sorted(list(np.unique(y)))
        n_samples = len(texts)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        results: Dict[str, Dict[str, Any]] = {}

        for strategy in self.STRATEGIES:
            fold_macro_f1s = []
            fold_accs = []
            fold_weighted_f1s = []
            fold_per_class: Dict[str, List[float]] = {cls: [] for cls in self.classes_}

            start_time = time.perf_counter()

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
                fold_train_texts = [texts[i] for i in train_idx]
                fold_val_texts = [texts[i] for i in val_idx]
                fold_y_train, fold_y_val = y[train_idx], y[val_idx]

                vectorizer = self.build_vectorizer(strategy, corpus_size=len(fold_train_texts))
                X_train = vectorizer.fit_transform(fold_train_texts)
                X_val = vectorizer.transform(fold_val_texts)

                clf = LinearSVC(
                    C=self.classifier_c,
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=2000,
                )
                clf.fit(X_train, fold_y_train)
                y_pred = clf.predict(X_val)

                acc = accuracy_score(fold_y_val, y_pred)
                p, r, f1, _ = precision_recall_fscore_support(
                    fold_y_val, y_pred, average="macro", zero_division=0
                )
                _, _, wf1, _ = precision_recall_fscore_support(
                    fold_y_val, y_pred, average="weighted", zero_division=0
                )
                _, _, per_class_f1, _ = precision_recall_fscore_support(
                    fold_y_val, y_pred, labels=self.classes_, average=None, zero_division=0
                )

                fold_macro_f1s.append(float(f1))
                fold_accs.append(float(acc))
                fold_weighted_f1s.append(float(wf1))
                for cls, cls_f1 in zip(self.classes_, per_class_f1):
                    fold_per_class[cls].append(float(cls_f1))

            elapsed = time.perf_counter() - start_time

            results[strategy] = {
                "display_name": self.STRATEGY_DISPLAY_NAMES[strategy],
                "macro_f1_mean": float(np.mean(fold_macro_f1s)),
                "macro_f1_std": float(np.std(fold_macro_f1s)),
                "accuracy_mean": float(np.mean(fold_accs)),
                "accuracy_std": float(np.std(fold_accs)),
                "weighted_f1_mean": float(np.mean(fold_weighted_f1s)),
                "weighted_f1_std": float(np.std(fold_weighted_f1s)),
                "per_class_f1_mean": {
                    cls: float(np.mean(fold_per_class[cls])) for cls in self.classes_
                },
                "per_class_f1_std": {
                    cls: float(np.std(fold_per_class[cls])) for cls in self.classes_
                },
                "fold_macro_f1s": fold_macro_f1s,
                "elapsed_seconds": round(elapsed, 2),
            }

        self.cv_results_ = results
        return results

    def evaluate_on_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fit each representation on full train set and evaluate on test set.

        Args:
            train_df: Full training DataFrame.
            test_df: Held-out test DataFrame.
            label_col: Target label column name.

        Returns:
            Dictionary mapping strategy name to held-out test performance.
        """
        train_texts, y_train = self.prepare_dataset(train_df, label_col=label_col)
        test_texts, y_test = self.prepare_dataset(test_df, label_col=label_col)
        self.classes_ = sorted(list(np.unique(y_train)))

        results: Dict[str, Dict[str, Any]] = {}

        for strategy in self.STRATEGIES:
            vectorizer = self.build_vectorizer(strategy, corpus_size=len(train_texts))
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)

            clf = LinearSVC(
                C=self.classifier_c,
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=2000,
            )
            start_time = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time = time.perf_counter() - start_time

            infer_start = time.perf_counter()
            y_pred = clf.predict(X_test)
            infer_time = time.perf_counter() - infer_start

            acc = accuracy_score(y_test, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )
            _, _, wf1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            _, _, per_class_f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=self.classes_, average=None, zero_division=0
            )

            # Actual feature dimension extracted
            if hasattr(vectorizer, "vocabulary_"):
                dim = len(vectorizer.vocabulary_)
            elif hasattr(vectorizer, "transformer_list"):
                dim = sum(len(trans.vocabulary_) for _, trans in vectorizer.transformer_list)
            else:
                dim = X_train.shape[1]

            results[strategy] = {
                "display_name": self.STRATEGY_DISPLAY_NAMES[strategy],
                "test_accuracy": float(acc),
                "test_macro_f1": float(f1),
                "test_weighted_f1": float(wf1),
                "test_per_class_f1": {
                    cls: float(cls_f1) for cls, cls_f1 in zip(self.classes_, per_class_f1)
                },
                "feature_dimension": int(dim),
                "train_time_sec": round(train_time, 3),
                "infer_time_sec": round(infer_time, 4),
            }

        self.test_results_ = results
        return results

    def plot_comparison(self, output_path: Union[str, Path]) -> Path:
        """Generate high-clarity publication-quality comparison figure (fig13).

        Args:
            output_path: Destination image path.

        Returns:
            Resolved Path of the saved visualization.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.cv_results_:
            raise ValueError("Cross-validation results are empty. Run run_cross_validation first.")

        strategies = self.STRATEGIES
        display_names = [self.STRATEGY_DISPLAY_NAMES[s] for s in strategies]
        cv_macro_f1s = [self.cv_results_[s]["macro_f1_mean"] for s in strategies]
        cv_stds = [self.cv_results_[s]["macro_f1_std"] for s in strategies]

        has_test = bool(self.test_results_)
        test_macro_f1s = [
            self.test_results_[s]["test_macro_f1"] if has_test else 0.0 for s in strategies
        ]

        # Determine champion strategy based on CV Macro-F1
        best_idx = int(np.argmax(cv_macro_f1s))
        best_strategy = strategies[best_idx]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        # Plot 1: 5-Fold Cross Validation Macro-F1
        colors = ["#2b5c8f" if i != best_idx else "#107c41" for i in range(len(strategies))]
        bars1 = axes[0].barh(
            display_names,
            cv_macro_f1s,
            xerr=cv_stds,
            capsize=4,
            color=colors,
            edgecolor="#1f2937",
            alpha=0.88,
        )
        axes[0].set_title(
            f"5-Fold Cross-Validation Macro-F1\n(Champion: {self.STRATEGY_DISPLAY_NAMES[best_strategy]} — {cv_macro_f1s[best_idx]:.4f})",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        axes[0].set_xlabel("Macro-F1 Score", fontsize=11)
        axes[0].set_xlim(0.65, 0.90)

        for bar, val in zip(bars1, cv_macro_f1s):
            axes[0].text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="semibold",
            )

        # Plot 2: Held-out Test Macro-F1 Comparison
        if has_test:
            best_test_idx = int(np.argmax(test_macro_f1s))
            test_colors = ["#4f46e5" if i != best_test_idx else "#059669" for i in range(len(strategies))]
            bars2 = axes[1].barh(
                display_names,
                test_macro_f1s,
                color=test_colors,
                edgecolor="#1f2937",
                alpha=0.88,
            )
            axes[1].set_title(
                f"Held-Out Test Set Macro-F1 (N=385)\n(Best Test: {self.STRATEGY_DISPLAY_NAMES[strategies[best_test_idx]]} — {test_macro_f1s[best_test_idx]:.4f})",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            axes[1].set_xlabel("Macro-F1 Score", fontsize=11)
            axes[1].set_xlim(0.65, 0.90)

            for bar, val in zip(bars2, test_macro_f1s):
                axes[1].text(
                    val + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}",
                    va="center",
                    ha="left",
                    fontsize=9.5,
                    fontweight="semibold",
                )
        else:
            axes[1].text(0.5, 0.5, "Test Evaluation Not Run", ha="center", va="center")

        plt.suptitle(
            "Phase 5 NLP Enhancement — Text Representation Comparison (Step 5.3)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def save_results(self, output_path: Union[str, Path]) -> Path:
        """Save complete structured evaluation dictionary to JSON.

        Args:
            output_path: Path to target JSON file.

        Returns:
            Resolved Path of the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Identify champion representations
        cv_macro_scores = {s: self.cv_results_[s]["macro_f1_mean"] for s in self.STRATEGIES if s in self.cv_results_}
        champion_cv = max(cv_macro_scores, key=cv_macro_scores.get) if cv_macro_scores else None

        test_macro_scores = {
            s: self.test_results_[s]["test_macro_f1"] for s in self.STRATEGIES if s in self.test_results_
        }
        champion_test = max(test_macro_scores, key=test_macro_scores.get) if test_macro_scores else None

        payload = {
            "metadata": {
                "step": "5.3",
                "title": "Experiment with Different Text Representations",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "classifier": "LinearSVC",
                "classifier_c": self.classifier_c,
                "max_features_budget": self.max_features,
                "title_multiplier": self.title_multiplier,
                "enhanced_preprocessing": self.use_enhanced_preprocessing,
                "classes": self.classes_,
                "champion_cv_strategy": champion_cv,
                "champion_cv_macro_f1": cv_macro_scores.get(champion_cv) if champion_cv else None,
                "champion_test_strategy": champion_test,
                "champion_test_macro_f1": test_macro_scores.get(champion_test) if champion_test else None,
            },
            "cross_validation_results": self.cv_results_,
            "test_evaluation_results": self.test_results_,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path

    def run_all(
        self,
        train_path: Union[str, Path],
        test_path: Union[str, Path],
        figure_output_path: Union[str, Path],
        json_output_path: Union[str, Path],
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Orchestrate the full Step 5.3 experiment pipeline."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"[Step 5.3] Loaded train ({len(train_df)} rows) and test ({len(test_df)} rows).")
        print(f"[Step 5.3] Running 5-fold CV across {len(self.STRATEGIES)} text representations...")

        cv_results = self.run_cross_validation(train_df, n_splits=n_splits)
        for s in self.STRATEGIES:
            res = cv_results[s]
            print(f"  -> {res['display_name']:<30}: Macro-F1 = {res['macro_f1_mean']:.4f} (±{res['macro_f1_std']:.4f})")

        print(f"[Step 5.3] Evaluating generalization on held-out test split (N={len(test_df)})...")
        test_results = self.evaluate_on_test(train_df, test_df)
        for s in self.STRATEGIES:
            res = test_results[s]
            print(f"  -> {res['display_name']:<30}: Test Macro-F1 = {res['test_macro_f1']:.4f}, Accuracy = {res['test_accuracy']:.4f}")

        fig_path = self.plot_comparison(figure_output_path)
        json_path = self.save_results(json_output_path)

        print(f"[Step 5.3] Comparison figure saved to: {fig_path}")
        print(f"[Step 5.3] Evaluation results saved to: {json_path}")

        return {
            "cv_results": cv_results,
            "test_results": test_results,
            "figure_path": str(fig_path),
            "json_path": str(json_path),
        }


def main():
    """CLI execution entry point."""
    train_file = Path("data/processed/splits/train.csv")
    test_file = Path("data/processed/splits/test.csv")
    fig_dest = Path("ml/nlp/figures/fig13_text_representations_comparison.png")
    json_dest = Path("data/processed/evaluation/text_representations_experiments.json")

    runner = TextRepresentationExperimentRunner(
        max_features=5000,
        random_state=42,
        title_multiplier=3,
        use_enhanced_preprocessing=True,
        classifier_c=0.5,
    )

    runner.run_all(
        train_path=train_file,
        test_path=test_file,
        figure_output_path=fig_dest,
        json_output_path=json_dest,
        n_splits=5,
    )


if __name__ == "__main__":
    main()
