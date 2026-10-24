"""TF-IDF Feature Extraction and Text Representation Pipeline for OpenSourceLens.

Builds leakage-free numerical representations of GitHub issue titles and descriptions,
supports sublinear term-frequency scaling, n-gram extraction, vocabulary pruning,
optional multi-modal tabular feature fusion, and disk serialization.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import joblib

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ml.classification.target_definition import ClassificationTargetDefiner


class TextFeaturePipeline:
    """Configures, fits, and applies TF-IDF and auxiliary feature transformations."""

    DEFAULT_AUXILIARY_COLS = [
        "clean_title_word_count",
        "clean_body_word_count",
        "has_code_block",
        "has_stack_trace",
        "has_url",
        "code_block_count",
        "url_count",
    ]

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.85,
        sublinear_tf: bool = True,
        use_auxiliary_features: bool = False,
        auxiliary_features: Optional[List[str]] = None,
    ):
        """Initialize the feature extraction pipeline.

        Args:
            max_features: Maximum size of the TF-IDF vocabulary (default: 5000).
            ngram_range: Lower and upper boundary of range of n-values (default: (1, 2)).
            min_df: Minimum document frequency threshold (default: 2).
            max_df: Maximum document frequency proportion threshold (default: 0.85).
            sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf) (default: True).
            use_auxiliary_features: Whether to horizontally stack scaled tabular features (default: False).
            auxiliary_features: Specific list of tabular feature column names (default: DEFAULT_AUXILIARY_COLS).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.use_auxiliary_features = use_auxiliary_features
        self.auxiliary_features = (
            auxiliary_features if auxiliary_features is not None else list(self.DEFAULT_AUXILIARY_COLS)
        )

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
        )
        self.scaler: Optional[StandardScaler] = StandardScaler() if self.use_auxiliary_features else None

        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.vocabulary_size_: int = 0
        self.fitted_auxiliary_cols_: List[str] = []

    @property
    def total_features(self) -> int:
        """Return total number of features in fitted matrix."""
        return len(self.feature_names_)

    def _extract_text_series(self, df: pd.DataFrame, text_col: str = "combined_text") -> pd.Series:
        """Safely extract or construct text Series from DataFrame."""
        if text_col in df.columns:
            return df[text_col].fillna("").astype(str)

        # Fallback to concatenating clean_title and clean_body
        title = df["clean_title"].fillna("").astype(str) if "clean_title" in df.columns else pd.Series([""] * len(df))
        body = df["clean_body"].fillna("").astype(str) if "clean_body" in df.columns else pd.Series([""] * len(df))
        return (title + " " + body).str.strip()

    def fit(self, train_df: pd.DataFrame, text_col: str = "combined_text") -> "TextFeaturePipeline":
        """Fit the TF-IDF vocabulary (and optional auxiliary scaler) exclusively on training data.

        Args:
            train_df: Training DataFrame.
            text_col: Text column name (default: 'combined_text').

        Returns:
            Self (fitted pipeline).

        Raises:
            ValueError: If train_df is empty.
        """
        if train_df.empty:
            raise ValueError("Cannot fit TextFeaturePipeline on empty training DataFrame.")

        # Fit TF-IDF on training text
        texts = self._extract_text_series(train_df, text_col)
        self.vectorizer.fit(texts)

        text_features = self.vectorizer.get_feature_names_out().tolist()
        self.vocabulary_size_ = len(text_features)
        self.feature_names_ = list(text_features)

        # Fit scaler on auxiliary features if enabled
        if self.use_auxiliary_features and self.scaler is not None:
            available_aux = [
                col for col in self.auxiliary_features
                if col in train_df.columns and col not in ClassificationTargetDefiner.LEAKAGE_COLUMNS
            ]
            if available_aux:
                aux_matrix = train_df[available_aux].fillna(0.0).to_numpy(dtype=np.float64)
                self.scaler.fit(aux_matrix)
                self.fitted_auxiliary_cols_ = available_aux
                self.feature_names_.extend([f"aux_{col}" for col in available_aux])
            else:
                self.fitted_auxiliary_cols_ = []

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, text_col: str = "combined_text") -> scipy.sparse.csr_matrix:
        """Transform text (and optional auxiliary features) into a sparse numerical feature matrix.

        Args:
            df: DataFrame to transform (train, val, test, or inference).
            text_col: Text column name (default: 'combined_text').

        Returns:
            scipy.sparse.csr_matrix representing transformed features.

        Raises:
            RuntimeError: If pipeline has not been fitted prior to transform.
            ValueError: If input DataFrame is empty.
        """
        if not self.is_fitted:
            raise RuntimeError("TextFeaturePipeline must be fitted before calling transform().")

        if df.empty:
            raise ValueError("Cannot transform empty DataFrame.")

        texts = self._extract_text_series(df, text_col)
        X_text = self.vectorizer.transform(texts)

        if not self.use_auxiliary_features or not self.fitted_auxiliary_cols_ or self.scaler is None:
            return X_text.tocsr()

        # Transform auxiliary features
        aux_data: List[np.ndarray] = []
        for col in self.fitted_auxiliary_cols_:
            if col in df.columns:
                aux_data.append(df[col].fillna(0.0).to_numpy(dtype=np.float64))
            else:
                aux_data.append(np.zeros(len(df), dtype=np.float64))

        aux_matrix = np.column_stack(aux_data)
        aux_scaled = self.scaler.transform(aux_matrix)
        X_aux = scipy.sparse.csr_matrix(aux_scaled)

        # Horizontally stack sparse matrices
        X_combined = scipy.sparse.hstack([X_text, X_aux], format="csr")
        return X_combined

    def fit_transform(self, train_df: pd.DataFrame, text_col: str = "combined_text") -> scipy.sparse.csr_matrix:
        """Fit vocabulary on training set and return transformed feature matrix."""
        return self.fit(train_df, text_col=text_col).transform(train_df, text_col=text_col)

    def get_top_features_per_class(
        self,
        X: scipy.sparse.csr_matrix,
        y: Union[np.ndarray, pd.Series, List[Any]],
        classes: Optional[List[str]] = None,
        top_k: int = 15,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Compute top discriminating terms per target class based on mean TF-IDF weights.

        Args:
            X: Transformed feature matrix (samples x features).
            y: Target class array or Series.
            classes: Optional explicit list of class names. If None, inferred from y.
            top_k: Number of highest-weight terms to return per class (default: 15).

        Returns:
            Dictionary mapping class name to list of (term, mean_tfidf_weight) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before inspecting top features.")

        y_arr = np.asarray(y)
        unique_classes = classes if classes is not None else sorted(np.unique(y_arr).tolist())
        top_features: Dict[str, List[Tuple[str, float]]] = {}

        feature_names = np.array(self.feature_names_)

        for cls in unique_classes:
            mask = y_arr == cls
            if not np.any(mask):
                top_features[str(cls)] = []
                continue

            # Compute mean weight per feature for this class
            class_mean = np.asarray(X[mask].mean(axis=0)).flatten()
            top_indices = class_mean.argsort()[::-1][:top_k]

            terms = [
                (str(feature_names[idx]), round(float(class_mean[idx]), 4))
                for idx in top_indices
                if class_mean[idx] > 0
            ]
            top_features[str(cls)] = terms

        return top_features

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic summary of the feature pipeline configuration and fitted state."""
        return {
            "is_fitted": self.is_fitted,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "use_auxiliary_features": self.use_auxiliary_features,
            "vocabulary_size": self.vocabulary_size_,
            "total_features": len(self.feature_names_),
            "fitted_auxiliary_cols": self.fitted_auxiliary_cols_,
        }

    def save(self, filepath: Union[str, Path] = "models/tfidf_vectorizer.joblib") -> Path:
        """Serialize the fitted pipeline to disk.

        Args:
            filepath: Destination file path (default: 'models/tfidf_vectorizer.joblib').

        Returns:
            Path of saved artifact.

        Raises:
            RuntimeError: If attempting to save an unfitted pipeline.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted TextFeaturePipeline.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/tfidf_vectorizer.joblib") -> "TextFeaturePipeline":
        """Load serialized feature pipeline from disk.

        Args:
            filepath: Path to saved .joblib file.

        Returns:
            Loaded TextFeaturePipeline instance.

        Raises:
            FileNotFoundError: If filepath does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Serialized pipeline file not found at: {path}")

        loaded = joblib.load(path)
        if not isinstance(loaded, TextFeaturePipeline):
            raise TypeError(f"Loaded object is of type {type(loaded).__name__}, expected TextFeaturePipeline.")

        return loaded
