"""OpenSourceLens NLP Enhancement Package (Phase 5).

Provides advanced text preprocessing, title/body weighting, alternative text representations,
vocabulary analysis, and feature refinement pipelines.
"""

from ml.nlp.enhanced_preprocessor import EnhancedIssueTextPreprocessor
from ml.nlp.title_body_experiment import TitleBodyExperimentRunner
from ml.nlp.representations import TextRepresentationExperimentRunner
from ml.nlp.vocab_analysis import VocabularyFeatureAnalyzer
from ml.nlp.final_pipeline import FinalClassicalNLPPipeline, FinalPipelineEvaluator

__all__ = [
    "EnhancedIssueTextPreprocessor",
    "TitleBodyExperimentRunner",
    "TextRepresentationExperimentRunner",
    "VocabularyFeatureAnalyzer",
    "FinalClassicalNLPPipeline",
    "FinalPipelineEvaluator",
]
