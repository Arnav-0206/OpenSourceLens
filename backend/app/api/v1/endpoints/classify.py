"""Issue classification inference endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.services.classification_service import classification_service

router = APIRouter()


@router.post(
    "",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Issue Text",
    tags=["classification"],
)
def classify_issue_text(
    request: ClassificationRequest,
    db: Session = Depends(get_db),
) -> ClassificationResponse:
    """Predict issue canonical category (bug, feature, other) from title and body text.

    Supports fine-tuned DistilBERT (champion transformer) and classical TF-IDF baselines.
    """
    return classification_service.classify(request=request, db=db)
