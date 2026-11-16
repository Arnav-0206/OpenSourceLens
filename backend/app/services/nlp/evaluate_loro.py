"""Leave-One-Repo-Out (LORO) Cross-Validation and Model Artifact Exporter.

Performs strict out-of-distribution evaluation of GeneralizableIssueClassifier across
the four benchmark repositories (vscode, pytorch, kubernetes, transformers), then
trains the production champion artifact on all labeled data with calibrated abstention.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from backend.app.services.nlp.generalizable_classifier import (
    CANONICAL_CLASSES,
    GeneralizableIssueClassifier,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "issues_master.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_PATH = MODELS_DIR / "generalizable_issue_classifier.joblib"
MANIFEST_OUTPUT_PATH = MODELS_DIR / "generalizable_issue_classifier_manifest.json"


def run_loro_evaluation_and_export() -> Dict[str, Any]:
    """Execute LORO cross-validation and export production classifier artifact."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df_labeled = df[df["canonical_label"] != "unlabeled"].copy()

    repos = sorted(df_labeled["repo_name"].unique())
    logger.info("Executing LORO evaluation across %d repositories: %s", len(repos), repos)

    loro_results: Dict[str, Any] = {}

    for held_out_repo in repos:
        train_df = df_labeled[df_labeled["repo_name"] != held_out_repo]
        test_df = df_labeled[df_labeled["repo_name"] == held_out_repo]

        classifier = GeneralizableIssueClassifier(
            max_features=8000,
            title_multiplier=3,
            confidence_threshold=0.40,
        )
        classifier.fit(
            titles=train_df["title"].tolist(),
            bodies=train_df["body"].tolist(),
            labels=train_df["canonical_label"].tolist(),
        )

        test_preds = classifier.predict_batch(
            titles=test_df["title"].tolist(),
            bodies=test_df["body"].tolist(),
        )

        pred_labels = [p["predicted_label"] for p in test_preds]
        true_labels = test_df["canonical_label"].tolist()
        abstentions = [p["is_abstention"] for p in test_preds]

        raw_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        raw_acc = accuracy_score(true_labels, pred_labels)

        # Metrics on confident (non-abstained) predictions
        confident_indices = [i for i, a in enumerate(abstentions) if not a]
        if confident_indices:
            conf_pred = [pred_labels[i] for i in confident_indices]
            conf_true = [true_labels[i] for i in confident_indices]
            conf_acc = accuracy_score(conf_true, conf_pred)
            conf_f1 = f1_score(conf_true, conf_pred, average="macro", zero_division=0)
        else:
            conf_acc = 0.0
            conf_f1 = 0.0

        loro_results[held_out_repo] = {
            "test_sample_count": len(test_df),
            "macro_f1": round(float(raw_f1), 4),
            "accuracy": round(float(raw_acc), 4),
            "abstention_rate": round(float(np.mean(abstentions)), 4),
            "confident_coverage": round(float(len(confident_indices) / len(test_df)), 4),
            "confident_accuracy": round(float(conf_acc), 4),
            "confident_macro_f1": round(float(conf_f1), 4),
        }

    # Now train the production artifact on ALL labeled data
    logger.info("Fitting production artifact on all %d labeled issues...", len(df_labeled))
    production_classifier = GeneralizableIssueClassifier(
        max_features=10000,
        title_multiplier=3,
        confidence_threshold=0.40,
    )
    production_classifier.fit(
        titles=df_labeled["title"].tolist(),
        bodies=df_labeled["body"].tolist(),
        labels=df_labeled["canonical_label"].tolist(),
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    production_classifier.save(MODEL_OUTPUT_PATH)

    # Compile manifest
    manifest: Dict[str, Any] = {
        "model_name": "GeneralizableIssueClassifier",
        "task": "5-Class Issue and Discussion Categorization with Calibrated Abstention",
        "version": "2026.5.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "classes": CANONICAL_CLASSES,
        "abstention_label": "uncertain",
        "calibrated_confidence_threshold": 0.40,
        "max_entropy_threshold": 1.45,
        "vocabulary_size": len(production_classifier.word_vectorizer.vocabulary_),
        "total_training_samples": len(df_labeled),
        "class_distribution": df_labeled["canonical_label"].value_counts().to_dict(),
        "loro_validation_summary": loro_results,
        "average_loro_macro_f1": round(
            float(np.mean([r["macro_f1"] for r in loro_results.values()])), 4
        ),
        "average_loro_confident_acc": round(
            float(np.mean([r["confident_accuracy"] for r in loro_results.values()])), 4
        ),
    }

    with open(MANIFEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved production model to %s", MODEL_OUTPUT_PATH)
    logger.info("Saved manifest to %s", MANIFEST_OUTPUT_PATH)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_loro_evaluation_and_export()
    print(json.dumps(res, indent=2))
