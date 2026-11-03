"""Resolution Analysis & Advisory Prediction Service for OpenSourceLens.

Integrates the Phase 8 Random Forest classifier champion and historical SLA
percentiles with FastAPI endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Prediction, Repository, RepositoryAnalytics
from backend.app.schemas.resolution import (
    ResolutionPredictionRequest,
    ResolutionPredictionResponse,
    ResolutionStatsResponse,
)
from ml.resolution_analysis.predictor import ResolutionPredictor

logger = logging.getLogger(__name__)


class ResolutionService:
    """Singleton service for advisory resolution velocity prediction and statistics."""

    _instance: Optional[ResolutionService] = None

    def __new__(cls) -> ResolutionService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        logger.info("Initializing ResolutionPredictor for API service...")
        self._predictor = ResolutionPredictor()
        self._initialized = True

    def predict_resolution(
        self,
        request: ResolutionPredictionRequest,
        db: Optional[Session] = None,
    ) -> ResolutionPredictionResponse:
        """Predict advisory resolution velocity tier and error bounds.

        Args:
            request: Validated ResolutionPredictionRequest.
            db: Optional active SQLAlchemy session for prediction auditing.

        Returns:
            ResolutionPredictionResponse with tier probabilities and advisory disclaimer.
        """
        t0 = time.perf_counter()

        res = self._predictor.predict_velocity(
            title=request.title,
            body=request.body,
            repo_name=request.repo_name,
            canonical_label=request.canonical_label,
            author=request.author,
            author_association=request.author_association,
            labels_count=request.labels_count,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        model_name = "resolution-rf-classifier-champion"

        # Log prediction audit if session is active
        if db is not None:
            audit = Prediction(
                issue_id=request.issue_id,
                repo_name=request.repo_name,
                input_title=request.title,
                input_body=request.body,
                model_name=model_name,
                task_type="resolution_prediction",
                predicted_label=res["predicted_tier"],
                confidence_score=res["confidence"],
                probabilities=res["tier_probabilities"],
                inference_time_ms=latency_ms,
            )
            db.add(audit)
            db.commit()

        return ResolutionPredictionResponse(
            repo_name=request.repo_name,
            predicted_speed_tier=res["predicted_tier"],
            confidence_score=res["confidence"],
            tier_probabilities=res["tier_probabilities"],
            lower_bound_tier=res["adjacent_window"]["lower_bound"],
            upper_bound_tier=res["adjacent_window"]["upper_bound"],
            advisory_factors=res.get("advisory_factors", []),
            reliability_notice=res.get("disclaimer", ""),
            model_used=model_name,
            inference_time_ms=latency_ms,
        )

    def get_repository_stats(
        self,
        owner: str,
        name: str,
        db: Session,
    ) -> ResolutionStatsResponse:
        """Fetch descriptive resolution percentiles and survival statistics for a repository."""
        full_name = f"{owner}/{name}"
        repo = db.scalar(select(Repository).where(Repository.repo_name == full_name))
        if not repo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository '{full_name}' not found.",
            )

        analytics = db.scalar(
            select(RepositoryAnalytics)
            .where(RepositoryAnalytics.repository_id == repo.id)
            .order_by(RepositoryAnalytics.snapshot_date.desc())
        )
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analytics computed for repository '{full_name}'.",
            )

        tier_dist = {}
        tot = analytics.total_issues
        if analytics.speed_tier_distribution and tot > 0:
            tier_dist = {
                k: round((v / tot) * 100.0, 1)
                for k, v in analytics.speed_tier_distribution.items()
            }

        pct_24h = tier_dist.get("< 1 Day", 0.0)
        pct_7d = pct_24h + tier_dist.get("1-7 Days", 0.0)

        return ResolutionStatsResponse(
            repo_name=repo.repo_name,
            n_closed_issues=analytics.closed_issues,
            median_resolution_hours=analytics.median_resolution_hours,
            p25_resolution_hours=analytics.p25_resolution_hours,
            p75_resolution_hours=analytics.p75_resolution_hours,
            pct_resolved_under_24h=round(pct_24h, 1),
            pct_resolved_under_7d=round(pct_7d, 1),
            survival_rate_30d=analytics.survival_rate_30d,
            tier_distribution=tier_dist,
        )


resolution_service = ResolutionService()
