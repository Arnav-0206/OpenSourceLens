"""Unified Model Registry and Manager Service for OpenSourceLens.

Tracks lifecycle, status, memory footprint, and device placement for all
machine learning, deep learning, and NLP models across Phases 4 through 8.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
import torch

from backend.app.schemas.models import ModelInfoItem, ModelRegistryResponse
from backend.app.services.classification_service import classification_service
from backend.app.services.related_service import related_service
from backend.app.services.resolution_service import resolution_service

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"


class ModelManager:
    """Central registry and lifecycle manager for all model artifacts."""

    _instance: Optional[ModelManager] = None

    def __new__(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True

    def _get_registry_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Static metadata definitions for all project model subsystems."""
        return {
            "distilbert": {
                "name": "Fine-Tuned DistilBERT Issue Classifier",
                "task_type": "classification",
                "version": "1.0.0 (Phase 6 Fine-Tuned Champion)",
                "artifact_path": str(MODELS_DIR / "distilbert_finetuned"),
                "check_loaded": lambda: classification_service._distilbert_model is not None,
                "check_exists": (MODELS_DIR / "distilbert_finetuned" / "model.safetensors").exists(),
                "details": {
                    "architecture": "DistilBertForSequenceClassification",
                    "base_model": "distilbert-base-uncased",
                    "num_parameters": 66955011,
                    "classes": ["bug", "feature", "other"],
                    "evaluation_macro_f1": 0.584,
                },
            },
            "classical_baselines": {
                "name": "TF-IDF Logistic Regression & Linear SVM Baselines",
                "task_type": "classification",
                "version": "1.0.0 (Phase 4/5 Baselines)",
                "artifact_path": str(MODELS_DIR / "baseline_logistic_regression.joblib"),
                "check_loaded": lambda: classification_service._lr_model is not None,
                "check_exists": (MODELS_DIR / "baseline_logistic_regression.joblib").exists(),
                "details": {
                    "vectorizer": "TF-IDF (10,000 max features, sublinear tf)",
                    "models": ["LogisticRegression(C=1.0)", "LinearSVC(C=1.0)"],
                    "classes": ["bug", "feature", "other"],
                },
            },
            "sentence_transformer": {
                "name": "Sentence-Transformers Dense Embedding Retriever",
                "task_type": "semantic_retrieval",
                "version": "all-MiniLM-L6-v2 (Phase 7)",
                "artifact_path": str(MODELS_DIR / "related_issues" / "embeddings.npz"),
                "check_loaded": lambda: related_service._embedder is not None,
                "check_exists": (MODELS_DIR / "related_issues" / "embeddings.npz").exists(),
                "details": {
                    "embedding_dimension": 384,
                    "metric": "cosine_similarity",
                    "cached_embeddings_count": 2328,
                    "normalized": True,
                },
            },
            "resolution_velocity": {
                "name": "Resolution Velocity 4-Tier Champion Classifier",
                "task_type": "resolution_prediction",
                "version": "1.0.0 (Phase 8 Random Forest Champion)",
                "artifact_path": str(MODELS_DIR / "resolution_classifier_champion.joblib"),
                "check_loaded": lambda: resolution_service._predictor.classifier_model is not None,
                "check_exists": (MODELS_DIR / "resolution_classifier_champion.joblib").exists(),
                "details": {
                    "classifier": "RandomForestClassifier(n_estimators=100, max_depth=12)",
                    "regressor": "Ridge(alpha=100.0)",
                    "classes": ["< 1 Day", "1-7 Days", "7-30 Days", "30+ Days"],
                    "adjacent_accuracy": "88.11% (±1 tier)",
                },
            },
        }

    def get_all_models(self) -> List[ModelInfoItem]:
        """Return status and metadata for all registered models."""
        definitions = self._get_registry_definitions()
        items = []

        for model_id, meta in definitions.items():
            is_loaded = meta["check_loaded"]()
            exists = meta["check_exists"]
            status_val = "ready" if exists else "error"

            items.append(
                ModelInfoItem(
                    model_id=model_id,
                    name=meta["name"],
                    task_type=meta["task_type"],
                    version=meta["version"],
                    artifact_path=meta["artifact_path"],
                    is_loaded=is_loaded,
                    device=self.device,
                    details=meta["details"],
                    status=status_val,
                )
            )

        return items

    def get_model_info(self, model_id: str) -> ModelInfoItem:
        """Lookup detailed metadata for a specific registered model identifier."""
        definitions = self._get_registry_definitions()
        clean_id = model_id.strip().lower()

        # Support common aliases
        alias_map = {
            "distilbert": "distilbert",
            "transformer": "distilbert",
            "classical": "classical_baselines",
            "svm": "classical_baselines",
            "logistic_regression": "classical_baselines",
            "related": "sentence_transformer",
            "embedder": "sentence_transformer",
            "sentence_transformers": "sentence_transformer",
            "resolution": "resolution_velocity",
            "velocity": "resolution_velocity",
        }
        target_id = alias_map.get(clean_id, clean_id)

        if target_id not in definitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model identifier '{model_id}' not found in registry. "
                f"Valid choices: {list(definitions.keys())}",
            )

        meta = definitions[target_id]
        is_loaded = meta["check_loaded"]()
        exists = meta["check_exists"]

        return ModelInfoItem(
            model_id=target_id,
            name=meta["name"],
            task_type=meta["task_type"],
            version=meta["version"],
            artifact_path=meta["artifact_path"],
            is_loaded=is_loaded,
            device=self.device,
            details=meta["details"],
            status="ready" if exists else "error",
        )

    def prewarm_all(self) -> Dict[str, str]:
        """Pre-load all model weights into memory to minimize first-request latency."""
        results = {}

        # 1. DistilBERT
        try:
            classification_service._get_distilbert()
            results["distilbert"] = "prewarmed"
        except Exception as exc:
            results["distilbert"] = f"error: {exc}"

        # 2. Classical baselines
        try:
            classification_service._get_lr_model()
            classification_service._get_svm_model()
            classification_service._get_tfidf_pipeline()
            results["classical_baselines"] = "prewarmed"
        except Exception as exc:
            results["classical_baselines"] = f"error: {exc}"

        # 3. Sentence Transformer
        try:
            related_service._get_embedder()
            related_service._get_embeddings_store()
            results["sentence_transformer"] = "prewarmed"
        except Exception as exc:
            results["sentence_transformer"] = f"error: {exc}"

        # 4. Resolution Velocity Predictor
        try:
            _ = resolution_service._predictor
            results["resolution_velocity"] = "prewarmed"
        except Exception as exc:
            results["resolution_velocity"] = f"error: {exc}"

        return results


model_manager = ModelManager()
