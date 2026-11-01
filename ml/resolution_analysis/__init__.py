"""OpenSourceLens — Resolution Analysis & Prediction Package (Phase 8).

Provides modules for:
- Resolution time definition and edge-case handling (Step 8.1)
- Distribution and survival analysis (Step 8.2)
- Feature engineering, leakage auditing, and ranking (Step 8.3)
- Resolution factor investigation (Step 8.4)
- Baseline prediction modeling (Step 8.5)
- Model evaluation and error diagnostics (Step 8.6)
- Reliability decision and pipeline packaging (Step 8.7)
"""

from ml.resolution_analysis.definition import (
    ResolutionTimeDefiner,
    SPEED_TIER_BINS,
    SPEED_TIER_LABELS,
)
from ml.resolution_analysis.distribution import ResolutionDistributionAnalyzer
from ml.resolution_analysis.features import (
    ELIGIBLE_CATEGORICAL_FEATURES,
    ELIGIBLE_NUMERIC_FEATURES,
    LEAKED_FEATURES,
    ResolutionFeatureEngineer,
)
from ml.resolution_analysis.factors import (
    FACTOR_FAMILIES,
    ResolutionFactorAnalyzer,
)
from ml.resolution_analysis.baseline_models import (
    SPEED_TIER_ORDER,
    ResolutionBaselineTrainer,
)
from ml.resolution_analysis.evaluation import ResolutionEvaluator
from ml.resolution_analysis.predictor import (
    ADVISORY_DISCLAIMER,
    ResolutionPredictor,
)

__all__ = [
    "ResolutionTimeDefiner",
    "ResolutionDistributionAnalyzer",
    "ResolutionFeatureEngineer",
    "ResolutionFactorAnalyzer",
    "ResolutionBaselineTrainer",
    "ResolutionEvaluator",
    "ResolutionPredictor",
    "SPEED_TIER_BINS",
    "SPEED_TIER_LABELS",
    "SPEED_TIER_ORDER",
    "ADVISORY_DISCLAIMER",
    "ELIGIBLE_NUMERIC_FEATURES",
    "ELIGIBLE_CATEGORICAL_FEATURES",
    "LEAKED_FEATURES",
    "FACTOR_FAMILIES",
]
