"""Support Vector Machine (SVM) Baseline Classifier for OpenSourceLens.

Implements a Linear Support Vector Classifier (LinearSVC) with balanced class penalties,
hyperparameter optimization via 5-fold Stratified Cross-Validation on the training split,
calibrated probability estimation via Platt scaling, feature margin interpretability,
and disk serialization.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import joblib

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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
        """Initialize the SVM baseline model.

        Args:
            C: Regularization parameter (default: 1.0).
            class_weight: Class penalty balancing strategy ('balanced' or dict, default: 'balanced').
            max_iter: Maximum iterations for solver convergence (default: 2000).
            random_state: Random seed for deterministic reproducibility (default: 42).
            dual: Formulation selection ('auto', True, or False, default: 'auto').
            calibrate_probabilities: Whether to fit CalibratedClassifierCV for predict_proba (default: True).
        """
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
        """Fit LinearSVC model directly using configured hyperparameters.

        Args:
            X_train: Sparse feature matrix (samples x features).
            y_train: Target label array.

        Returns:
            Self (fitted model).

        Raises:
            ValueError: If X_train or y_train are empty or mismatched.
        """
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
            # Calibrate probabilities via Platt scaling using 3-fold CV on train
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

    def tune_and_fit(
        self,
        X_train: scipy.sparse.csr_matrix,
        y_train: Union[np.ndarray, pd.Series, List[Any]],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        scoring: str = "f1_macro",
    ) -> Dict[str, Any]:
        """Tune regularization parameter C via Stratified K-Fold CV and fit best estimator.

        Args:
            X_train: Training feature matrix.
            y_train: Training target array.
            param_grid: Dictionary of hyperparameter lists to search (default: DEFAULT_PARAM_GRID).
            cv: Number of cross-validation folds (default: 5).
            scoring: Evaluation metric to optimize (default: 'f1_macro').

        Returns:
            Dictionary of tuning results, optimal parameters, and fold scores.
        """
        grid = param_grid if param_grid is not None else self.DEFAULT_PARAM_GRID
        y_arr = np.asarray(y_train)

        base_estimator = LinearSVC(
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual=self.dual,
        )

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=grid,
            cv=skf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_arr)

        self.base_model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_cv_score_ = float(grid_search.best_score_)
        self.C = self.best_params_.get("C", self.C)

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

        # Structure detailed CV fold summary
        cv_summary = {
            "best_params": self.best_params_,
            "best_cv_macro_f1": round(float(grid_search.best_score_), 4),
            "scoring": scoring,
            "cv_folds": cv,
            "param_scores": [],
        }

        means = grid_search.cv_results_["mean_test_score"]
        stds = grid_search.cv_results_["std_test_score"]
        params = grid_search.cv_results_["params"]
        for mean, std, param in zip(means, stds, params):
            cv_summary["param_scores"].append({
                "params": param,
                "mean_cv_macro_f1": round(float(mean), 4),
                "std_cv_macro_f1": round(float(std), 4),
            })

        self.cv_results_ = cv_summary
        return cv_summary

    def predict(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Predict class labels for given samples.

        Args:
            X: Sparse feature matrix.

        Returns:
            Numpy array of predicted class strings.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted or self.base_model is None:
            raise RuntimeError("SVM model must be fitted before calling predict().")

        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        return self.base_model.predict(X)

    def predict_proba(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Predict calibrated class probability distribution.

        Args:
            X: Sparse feature matrix.

        Returns:
            Numpy array of probabilities of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If model is not fitted or calibrate_probabilities was set to False.
        """
        if not self.is_fitted:
            raise RuntimeError("SVM model must be fitted before calling predict_proba().")

        if self.calibrated_model is None:
            raise RuntimeError(
                "predict_proba() is only available when calibrate_probabilities=True was used during initialization."
            )

        return self.calibrated_model.predict_proba(X)

    def decision_function(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """Return raw decision margin values from the separating hyperplanes.

        Args:
            X: Sparse feature matrix.

        Returns:
            Numpy array of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self.is_fitted or self.base_model is None:
            raise RuntimeError("SVM model must be fitted before calling decision_function().")

        return self.base_model.decision_function(X)

    def get_feature_coefficients(
        self,
        feature_names: List[str],
        top_k: int = 15,
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """Extract top positive and negative margin hyperplane coefficients per class.

        Args:
            feature_names: Ordered list of feature names corresponding to matrix columns.
            top_k: Number of highest magnitude terms to return per class (default: 15).

        Returns:
            Dictionary mapping class name to {'positive': [...], 'negative': [...]}.
        """
        if not self.is_fitted or self.base_model is None:
            raise RuntimeError("Model must be fitted before inspecting feature coefficients.")

        names_arr = np.asarray(feature_names)
        coef_dict: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

        coef_matrix = self.base_model.coef_
        is_binary = coef_matrix.shape[0] == 1

        for idx, cls_name in enumerate(self.classes_):
            if is_binary and idx == 0:
                coefs = -coef_matrix[0]
            elif is_binary and idx == 1:
                coefs = coef_matrix[0]
            else:
                coefs = coef_matrix[idx]

            # Top positive margin terms (pull towards this class)
            pos_indices = np.argsort(coefs)[::-1][:top_k]
            pos_terms = [
                (str(names_arr[i]), round(float(coefs[i]), 4))
                for i in pos_indices
                if coefs[i] > 0
            ]

            # Top negative margin terms (push away from this class)
            neg_indices = np.argsort(coefs)[:top_k]
            neg_terms = [
                (str(names_arr[i]), round(float(coefs[i]), 4))
                for i in neg_indices
                if coefs[i] < 0
            ]

            coef_dict[cls_name] = {
                "positive": pos_terms,
                "negative": neg_terms,
            }

        return coef_dict

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic summary of model hyperparameters and fitted state."""
        return {
            "is_fitted": self.is_fitted,
            "C": self.C,
            "class_weight": self.class_weight,
            "dual": self.dual,
            "max_iter": self.max_iter,
            "calibrate_probabilities": self.calibrate_probabilities,
            "classes": self.classes_,
            "best_params": self.best_params_,
            "best_cv_macro_f1": self.best_cv_score_,
        }

    def save(self, filepath: Union[str, Path] = "models/baseline_svm.joblib") -> Path:
        """Serialize trained model and calibrator to disk.

        Args:
            filepath: Target file path (default: 'models/baseline_svm.joblib').

        Returns:
            Path of saved artifact.

        Raises:
            RuntimeError: If attempting to save unfitted model.
        """
        if not self.is_fitted or self.base_model is None:
            raise RuntimeError("Cannot save unfitted SVMBaseline.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path] = "models/baseline_svm.joblib") -> "SVMBaseline":
        """Load serialized model from disk.

        Args:
            filepath: Path to saved .joblib file.

        Returns:
            Loaded SVMBaseline instance.

        Raises:
            FileNotFoundError: If filepath does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        loaded = joblib.load(path)
        if not isinstance(loaded, SVMBaseline):
            raise TypeError(f"Loaded object is of type {type(loaded).__name__}, expected SVMBaseline.")

        return loaded
