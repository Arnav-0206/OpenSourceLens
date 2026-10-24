"""OpenSourceLens Issue Classification Package (Phase 4).

Provides modules for classification target definition, train/test splitting,
feature representation (TF-IDF), classical baseline models (Logistic Regression, SVM),
evaluation metrics, and error analysis.
"""

from ml.classification.target_definition import ClassificationTargetDefiner

__all__ = [
    "ClassificationTargetDefiner",
]
