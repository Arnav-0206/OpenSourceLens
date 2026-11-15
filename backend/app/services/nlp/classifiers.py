"""Classical NLP classification pipelines and baseline models for OpenSourceLens.

Provides production inference classes for TF-IDF feature extraction,
Logistic Regression, and Support Vector Machine baseline classifiers.
Includes module-level unpickler compatibility mappings for legacy serialized models.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
        return len(self.feature_names_)

    def _extract_text_series(self, df: pd.DataFrame, text_col: str = "combined_text") -> pd.Series:
        if text_col in df.columns:
            return df[text_col].fillna("").astype(str)
        title = df["clean_title"].fillna("").astype(str) if "clean_title" in df.columns else pd.Series([""] * len(df))
        body = df["clean_body"].fillna("").astype(str) if "clean_body" in df.columns else pd.Series([""] * len(df))
        return (title + " " + body).str.strip()

    def fit(self, train_df: pd.DataFrame, text_col: str = "combined_text") -> "TextFeaturePipeline":
        if train_df.empty:
            raise ValueError("Cannot fit TextFeaturePipeline on empty training DataFrame.")

        texts = self._extract_text_series(train_df, text_col)
        self.vectorizer.fit(texts)

        text_features = self.vectorizer.get_feature_names_out().tolist()
        self.vocabulary_size_ = len(text_features)
        self.feature_names_ = list(text_features)

        if self.use_auxiliary_features and self.scaler is not None:
            available_aux = [
                col for col in self.auxiliary_features
                if col in train_df.columns
            ]
            self.fitted_auxiliary_cols_ = available_aux

            if available_aux:
                aux_data = train_df[available_aux].fillna(0.0).to_numpy(dtype=np.float64)
                self.scaler.fit(aux_data)
                self.feature_names_.extend([f"aux_{col}" for col in available_aux])

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, text_col: str = "combined_text") -> scipy.sparse.csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        texts = self._extract_text_series(df, text_col)
        X_text = self.vectorizer.transform(texts)

        if not self.use_auxiliary_features or not self.fitted_auxiliary_cols_ or self.scaler is None:
            return X_text.tocsr()

        aux_data: List[np.ndarray] = []
        for col in self.fitted_auxiliary_cols_:
            if col in df.columns:
                aux_data.append(df[col].fillna(0.0).to_numpy(dtype=np.float64))
            else:
                aux_data.append(np.zeros(len(df), dtype=np.float64))

        aux_matrix = np.column_stack(aux_data)
        aux_scaled = self.scaler.transform(aux_matrix)
        X_aux = scipy.sparse.csr_matrix(aux_scaled)
        return scipy.sparse.hstack([X_text, X_aux], format="csr")

    def fit_transform(self, train_df: pd.DataFrame, text_col: str = "combined_text") -> scipy.sparse.csr_matrix:
        return self.fit(train_df, text_col=text_col).transform(train_df, text_col=text_col)


class LogisticRegressionBaseline:
    """Trains, tunes, and evaluates a multinomial Logistic Regression baseline classifier."""

    DEFAULT_PARAM_GRID = {
        "C": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    }

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Union[str, Dict[Any, float]] = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver

        self.model: Optional[LogisticRegression] = None
        self.is_fitted: bool = False
        self.classes_: List[str] = []
        self.best_params_: Optional[Dict[str, Any]] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.best_cv_score_: Optional[float] = None

    def fit(
        self,
        X_train: scipy.sparse.csr_matrix,
        y_train: Union[np.ndarray, pd.Series, List[Any]],
    ) -> "LogisticRegressionBaseline":
        if X_train.shape[0] == 0:
            raise ValueError("Cannot fit model on empty training feature matrix.")

        y_arr = np.asarray(y_train)
        if len(y_arr) != X_train.shape[0]:
            raise ValueError(f"X_train rows ({X_train.shape[0]}) do not match y_train length ({len(y_arr)}).")

        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.model.fit(X_train, y_arr)

        self.is_fitted = True
        self.classes_ = [str(c) for c in self.model.classes_]
        return self

    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self.model.predict(X)

    def predict_proba(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        return self.model.predict_proba(X)


class SVMBaseline:
    """Trains, tunes, and evaluates a Linear Support Vector Machine baseline classifier."""

    DEFAULT_PARAM_GRID = {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    }

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Union[str, Dict[Any, float]] = "balanced",
        max_iter: int = 2000,
        random_state: int = 42,
        dual: Union[str, bool] = "auto",
        calibrate_probabilities: bool = True,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.dual = dual
        self.calibrate_probabilities = calibrate_probabilities

        self.base_model: Optional[LinearSVC] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.is_fitted: bool = False
        self.classes_: List[str] = []
        self.best_params_: Optional[Dict[str, Any]] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.best_cv_score_: Optional[float] = None

    def fit(
        self,
        X_train: scipy.sparse.csr_matrix,
        y_train: Union[np.ndarray, pd.Series, List[Any]],
    ) -> "SVMBaseline":
        if X_train.shape[0] == 0:
            raise ValueError("Cannot fit SVM model on empty training feature matrix.")

        y_arr = np.asarray(y_train)
        if len(y_arr) != X_train.shape[0]:
            raise ValueError(f"X_train rows ({X_train.shape[0]}) do not match y_train length ({len(y_arr)}).")

        self.base_model = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual=self.dual,
        )
        self.base_model.fit(X_train, y_arr)

        if self.calibrate_probabilities:
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.base_model,
                cv=3,
            )
            self.calibrated_model.fit(X_train, y_arr)
        else:
            self.calibrated_model = None

        self.is_fitted = True
        self.classes_ = [str(c) for c in self.base_model.classes_]
        return self

    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SVM model must be fitted before calling predict().")
        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        if self.base_model is not None:
            return self.base_model.predict(X)
        raise RuntimeError("No model available for prediction.")

    def predict_proba(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SVM model must be fitted before calling predict_proba().")
        if self.calibrated_model is None:
            raise RuntimeError(
                "predict_proba() is only available when calibrate_probabilities=True was used during initialization."
            )
        return self.calibrated_model.predict_proba(X)

    def decision_function(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        if not self.is_fitted or self.base_model is None:
            raise RuntimeError("SVM model must be fitted before calling decision_function().")
        return self.base_model.decision_function(X)


# ---------------------------------------------------------------------------
# Compatibility Registration for Legacy Serialized Model Artifacts
# ---------------------------------------------------------------------------
for _legacy_mod in [
    "ml",
    "ml.classification",
    "ml.classification.features",
    "ml.classification.logistic_regression",
    "ml.classification.svm",
]:
    if _legacy_mod not in sys.modules:
        sys.modules[_legacy_mod] = types.ModuleType(_legacy_mod)

sys.modules["ml.classification.features"].TextFeaturePipeline = TextFeaturePipeline
sys.modules["ml.classification.logistic_regression"].LogisticRegressionBaseline = LogisticRegressionBaseline
sys.modules["ml.classification.svm"].SVMBaseline = SVMBaseline
