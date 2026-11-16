"""Issue Classification Inference Service for OpenSourceLens.

Loads and caches pre-trained model artifacts (Generalizable 5-class classifier,
fine-tuned DistilBERT, and classical TF-IDF baselines) and runs inference with
database audit logging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
import torch

from backend.app.database.models import Prediction
from backend.app.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.services.nlp.text_cleaner import IssueTextCleaner
from backend.app.services.nlp.generalizable_classifier import (
    CLASS_TO_ID,
    GeneralizableIssueClassifier,
)
import backend.app.services.nlp.classifiers  # noqa: F401

logger = logging.getLogger(__name__)

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
DISTILBERT_DIR = MODELS_DIR / "distilbert_finetuned"
LR_MODEL_PATH = MODELS_DIR / "baseline_logistic_regression.joblib"
SVM_MODEL_PATH = MODELS_DIR / "baseline_svm.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
GENERALIZABLE_MODEL_PATH = MODELS_DIR / "generalizable_issue_classifier.joblib"
LABEL_MAPPINGS_PATH = PROJECT_ROOT / "data" / "processed" / "label_mappings.json"


class ClassificationService:
    """Singleton service providing cached ML and transformer model inference."""

    _instance: Optional[ClassificationService] = None

    def __new__(cls) -> ClassificationService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._cleaner = IssueTextCleaner()
        self._generalizable_classifier: Optional[GeneralizableIssueClassifier] = None
        self._distilbert_model = None
        self._distilbert_tokenizer = None
        self._lr_model = None
        self._svm_model = None
        self._tfidf_pipeline = None
        self._label_mappings: Dict[str, Any] = {}
        self._load_label_mappings()
        self._initialized = True

    def _load_label_mappings(self) -> None:
        """Load category name-to-id mapping dictionary."""
        if LABEL_MAPPINGS_PATH.exists():
            with open(LABEL_MAPPINGS_PATH, "r", encoding="utf-8") as f:
                self._label_mappings = json.load(f)
        else:
            self._label_mappings = {
                "label_to_id": {
                    "bug": 0,
                    "feature": 1,
                    "documentation": 2,
                    "question": 3,
                    "other": 4,
                    "uncertain": -1,
                }
            }

    def _get_generalizable_classifier(self) -> GeneralizableIssueClassifier:
        """Lazy load generalizable 5-class issue classifier champion."""
        if self._generalizable_classifier is None:
            if not GENERALIZABLE_MODEL_PATH.exists():
                logger.warning(
                    "Generalizable classifier artifact not found at %s. Attempting training...",
                    GENERALIZABLE_MODEL_PATH,
                )
                from backend.app.services.nlp.evaluate_loro import run_loro_evaluation_and_export
                run_loro_evaluation_and_export()

            logger.info("Loading Generalizable Classifier from %s...", GENERALIZABLE_MODEL_PATH)
            self._generalizable_classifier = GeneralizableIssueClassifier.load(GENERALIZABLE_MODEL_PATH)
        return self._generalizable_classifier

    def _get_distilbert(self) -> Tuple[Any, Any]:
        """Lazy load DistilBERT model and tokenizer on CPU."""
        if self._distilbert_model is None or self._distilbert_tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading DistilBERT model from %s...", DISTILBERT_DIR)
            self._distilbert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
            self._distilbert_model = AutoModelForSequenceClassification.from_pretrained(
                str(DISTILBERT_DIR)
            )
            self._distilbert_model.eval()
        return self._distilbert_model, self._distilbert_tokenizer

    def _get_lr_model(self) -> Any:
        """Lazy load Logistic Regression champion artifact."""
        if self._lr_model is None:
            logger.info("Loading Logistic Regression artifact from %s...", LR_MODEL_PATH)
            self._lr_model = joblib.load(str(LR_MODEL_PATH))
        return self._lr_model

    def _get_svm_model(self) -> Any:
        """Lazy load Support Vector Machine artifact."""
        if self._svm_model is None:
            logger.info("Loading SVM artifact from %s...", SVM_MODEL_PATH)
            self._svm_model = joblib.load(str(SVM_MODEL_PATH))
        return self._svm_model

    def _get_tfidf_pipeline(self) -> Any:
        """Lazy load fitted TextFeaturePipeline vectorizer."""
        if self._tfidf_pipeline is None:
            logger.info("Loading TF-IDF feature pipeline from %s...", TFIDF_PATH)
            self._tfidf_pipeline = joblib.load(str(TFIDF_PATH))
        return self._tfidf_pipeline

    def classify(
        self,
        request: ClassificationRequest,
        db: Optional[Session] = None,
    ) -> ClassificationResponse:
        """Execute text sanitization, run model prediction, and log audit record.

        Args:
            request: Validated ClassificationRequest payload.
            db: Optional active SQLAlchemy session for persistence.

        Returns:
            ClassificationResponse with predicted label, confidence, and calibrated abstention.
        """
        t0 = time.perf_counter()

        # 1. Text Sanitization & Cleaning
        clean_title = self._cleaner.clean_text(request.title)
        clean_body = self._cleaner.clean_text(request.body) if request.body else ""
        combined_text = f"{clean_title} {clean_body}".strip()

        model_type = request.model_type.lower()
        probabilities: Dict[str, float] = {}
        is_abstention = False
        normalized_entropy: Optional[float] = None

        # 2. Model Inference
        if model_type in ("generalizable", "generalizable_classifier"):
            classifier = self._get_generalizable_classifier()
            pred = classifier.predict_single(request.title, request.body)
            predicted_label = pred["predicted_label"]
            confidence = pred["confidence_score"]
            probabilities = pred["probabilities"]
            is_abstention = pred["is_abstention"]
            normalized_entropy = pred["normalized_entropy"]
            canon_id = pred["canonical_label_id"]
            model_identifier = "generalizable-5class-v2026.5"

        elif model_type == "distilbert":
            model, tokenizer = self._get_distilbert()
            inputs = tokenizer(
                combined_text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

            id2label = model.config.id2label
            for idx, p in enumerate(probs):
                label_name = id2label.get(idx, str(idx))
                probabilities[label_name] = round(float(p), 4)

            model_identifier = "distilbert-finetuned-v1"
            predicted_label = max(probabilities.items(), key=lambda kv: kv[1])[0]
            confidence = probabilities[predicted_label]
            canon_id = int(CLASS_TO_ID.get(predicted_label, -1))

        elif model_type in ("logistic_regression", "svm"):
            vec = self._get_tfidf_pipeline()
            df = pd.DataFrame(
                [
                    {
                        "combined_text": combined_text,
                        "clean_title": clean_title,
                        "clean_body": clean_body,
                        "title": request.title,
                        "body": request.body or "",
                    }
                ]
            )
            X = vec.transform(df)

            if model_type == "logistic_regression":
                lr = self._get_lr_model()
                probs = lr.predict_proba(X)[0]
                classes = lr.classes_
                for cls_name, p in zip(classes, probs):
                    probabilities[str(cls_name)] = round(float(p), 4)
                model_identifier = "tfidf-logistic-regression-baseline"
            else:
                svm = self._get_svm_model()
                decision = svm.decision_function(X)[0]
                exp_scores = np.exp(decision - np.max(decision))
                probs = exp_scores / np.sum(exp_scores)
                classes = svm.classes_
                for cls_name, p in zip(classes, probs):
                    probabilities[str(cls_name)] = round(float(p), 4)
                model_identifier = "tfidf-svm-baseline"

            predicted_label = max(probabilities.items(), key=lambda kv: kv[1])[0]
            confidence = probabilities[predicted_label]
            canon_id = int(CLASS_TO_ID.get(predicted_label, -1))
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        is_classified = predicted_label in [
            "bug",
            "feature",
            "documentation",
            "question",
            "other",
        ]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # 3. Optional Database Audit Logging
        if db is not None:
            audit = Prediction(
                issue_id=request.issue_id,
                repo_name=request.repo_name,
                input_title=request.title,
                input_body=request.body,
                model_name=model_identifier,
                task_type="classification",
                predicted_label=predicted_label,
                confidence_score=confidence,
                probabilities=probabilities,
                inference_time_ms=latency_ms,
            )
            db.add(audit)
            db.commit()

        return ClassificationResponse(
            predicted_label=predicted_label,
            canonical_label_id=canon_id,
            confidence_score=confidence,
            probabilities=probabilities,
            model_used=model_identifier,
            inference_time_ms=latency_ms,
            clean_title=clean_title,
            clean_body=clean_body if clean_body else None,
            is_classified_category=is_classified,
            is_abstention=is_abstention,
            normalized_entropy=normalized_entropy,
        )


classification_service = ClassificationService()
