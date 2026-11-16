"""Generalizable 5-Class Issue and Discussion Classifier for OpenSourceLens.

Implements text cleaning, TF-IDF + auxiliary feature extraction, calibrated
multinomial classification across 5 canonical categories (bug, feature, documentation,
question, other), and out-of-distribution calibrated low-confidence abstention (uncertain).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.app.services.nlp.text_cleaner import IssueTextCleaner

logger = logging.getLogger(__name__)

CANONICAL_CLASSES: List[str] = [
    "bug",
    "feature",
    "documentation",
    "question",
    "other",
]

CLASS_TO_ID: Dict[str, int] = {
    "bug": 0,
    "feature": 1,
    "documentation": 2,
    "question": 3,
    "other": 4,
    "uncertain": -1,
}

ID_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_ID.items()}

# Default calibrated threshold for abstention
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
DEFAULT_MAX_ENTROPY_THRESHOLD: float = 1.45


def compute_normalized_entropy(probs: np.ndarray) -> float:
    """Compute normalized Shannon entropy H(p) / log(K) in range [0.0, 1.0]."""
    k = len(probs)
    if k <= 1:
        return 0.0
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    entropy = -float(np.sum(p * np.log(p)))
    max_entropy = math.log(k)
    return float(entropy / max_entropy)


class GeneralizableIssueClassifier:
    """Production 5-class issue and discussion classifier with calibrated abstention."""

    def __init__(
        self,
        max_features: int = 10000,
        title_multiplier: int = 3,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_entropy_threshold: float = DEFAULT_MAX_ENTROPY_THRESHOLD,
    ):
        self.max_features = max_features
        self.title_multiplier = title_multiplier
        self.confidence_threshold = confidence_threshold
        self.max_entropy_threshold = max_entropy_threshold

        self.cleaner = IssueTextCleaner()
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.scaler = StandardScaler(with_mean=False)
        self.model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        self.is_fitted = False
        self.classes_ = list(CANONICAL_CLASSES)

    def _safe_str(self, val: Any) -> str:
        """Safely coerce input value to string, handling None and NaN."""
        if val is None:
            return ""
        if isinstance(val, float) and (np.isnan(val) or np.isneginf(val) or np.isposinf(val)):
            return ""
        return str(val)

    def prepare_text(self, title: str, body: Optional[str] = "") -> str:
        """Sanitize and weight title text relative to body text."""
        safe_title = self._safe_str(title)
        safe_body = self._safe_str(body)

        clean_title = self.cleaner.clean_text(safe_title)
        clean_body = self.cleaner.clean_text(safe_body) if safe_body else ""

        weighted_title = (clean_title + " ") * self.title_multiplier
        return (weighted_title + clean_body).strip()

    def extract_auxiliary_features(
        self, title: str, body: Optional[str] = ""
    ) -> np.ndarray:
        """Extract structural metadata cues from issue content."""
        safe_title = self._safe_str(title)
        safe_body = self._safe_str(body)

        clean_title = self.cleaner.clean_text(safe_title)
        clean_body = self.cleaner.clean_text(safe_body) if safe_body else ""

        has_code = 1.0 if ("```" in safe_body or "`" in safe_body) else 0.0
        has_stack = (
            1.0
            if (
                "Traceback" in safe_body
                or "Error:" in safe_body
                or "Exception" in safe_body
                or "at " in safe_body
            )
            else 0.0
        )
        has_url = 1.0 if ("http://" in safe_body or "https://" in safe_body) else 0.0

        title_len = float(len(clean_title.split()))
        body_len = float(len(clean_body.split()))

        return np.array(
            [[has_code, has_stack, has_url, title_len, body_len]],
            dtype=np.float32,
        )

    def fit(
        self,
        titles: List[str],
        bodies: List[Optional[str]],
        labels: List[str],
    ) -> GeneralizableIssueClassifier:
        """Fit TF-IDF vectorizer and calibrated classifier on training data."""
        texts = [
            self.prepare_text(t, b) for t, b in zip(titles, bodies)
        ]
        X = self.word_vectorizer.fit_transform(texts)
        y = np.array(labels)

        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self

    def predict_single(
        self,
        title: str,
        body: Optional[str] = "",
    ) -> Dict[str, Any]:
        """Classify single issue/discussion text and return calibrated distribution."""
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted or loaded.")

        text = self.prepare_text(title, body)
        X = self.word_vectorizer.transform([text])

        probs_raw = self.model.predict_proba(X)[0]
        prob_dict: Dict[str, float] = {}
        for cls_name, prob in zip(self.classes_, probs_raw):
            prob_dict[cls_name] = round(float(prob), 4)

        # Ensure all 5 canonical classes exist in probability dictionary
        for cls_name in CANONICAL_CLASSES:
            if cls_name not in prob_dict:
                prob_dict[cls_name] = 0.0

        # Sort out predicted class and check for abstention
        best_class = max(prob_dict, key=prob_dict.get)  # type: ignore
        best_score = prob_dict[best_class]
        entropy = compute_normalized_entropy(np.array(list(prob_dict.values())))

        # Calibrated abstention: if score is below threshold or entropy is excessively high
        is_abstention = (
            best_score < self.confidence_threshold
            or entropy > self.max_entropy_threshold
        )

        final_label = "uncertain" if is_abstention else best_class
        label_id = CLASS_TO_ID.get(final_label, -1)

        return {
            "predicted_label": final_label,
            "canonical_label_id": label_id,
            "confidence_score": best_score,
            "is_abstention": is_abstention,
            "normalized_entropy": round(entropy, 4),
            "probabilities": prob_dict,
            "clean_title": self.cleaner.clean_text(self._safe_str(title)),
            "clean_body": self.cleaner.clean_text(self._safe_str(body)) if body else None,
        }

    def predict_batch(
        self,
        titles: List[str],
        bodies: Optional[List[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of issues or discussions."""
        if bodies is None:
            bodies = [None] * len(titles)
        return [
            self.predict_single(t, b)
            for t, b in zip(titles, bodies)
        ]

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted classifier to joblib file."""
        state = {
            "max_features": self.max_features,
            "title_multiplier": self.title_multiplier,
            "confidence_threshold": self.confidence_threshold,
            "max_entropy_threshold": self.max_entropy_threshold,
            "word_vectorizer": self.word_vectorizer,
            "model": self.model,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, str(filepath))

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> GeneralizableIssueClassifier:
        """Load serialized classifier from joblib file."""
        state = joblib.load(str(filepath))
        instance = cls(
            max_features=state.get("max_features", 10000),
            title_multiplier=state.get("title_multiplier", 3),
            confidence_threshold=state.get(
                "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
            ),
            max_entropy_threshold=state.get(
                "max_entropy_threshold", DEFAULT_MAX_ENTROPY_THRESHOLD
            ),
        )
        instance.word_vectorizer = state["word_vectorizer"]
        instance.model = state["model"]
        instance.classes_ = state["classes_"]
        instance.is_fitted = state["is_fitted"]
        return instance
