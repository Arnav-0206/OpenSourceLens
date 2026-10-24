"""OpenSourceLens Deep Learning & Transformer Package (Phase 6).

Provides pretrained transformer selection, dataset preparation, tokenization,
fine-tuning with PyTorch / Hugging Face Transformers, evaluation, comparative benchmarking,
failure error diagnostics, and production inference packaging.
"""

from ml.deep_learning.model_selection import TransformerModelSelector
from ml.deep_learning.dataset_preparation import TransformerDatasetBuilder, IssueDataset
from ml.deep_learning.tokenizer import IssueTokenizer, TokenizedIssueDataset, SubwordTokenAnalyzer
from ml.deep_learning.train import (
    TransformerFineTuner,
    compute_class_weights,
    run_transformer_training_pipeline,
)
from ml.deep_learning.evaluator import (
    TransformerEvaluator,
    run_transformer_evaluation_pipeline,
)
from ml.deep_learning.comparator import (
    ModelComparator,
    run_model_comparison_pipeline,
)
from ml.deep_learning.error_analyzer import (
    TransformerErrorAnalyzer,
    run_error_analysis_pipeline,
)
from ml.deep_learning.pipeline import (
    TransformerIssueClassifier,
    export_production_bundle,
)

__all__ = [
    "TransformerModelSelector",
    "TransformerDatasetBuilder",
    "IssueDataset",
    "IssueTokenizer",
    "TokenizedIssueDataset",
    "SubwordTokenAnalyzer",
    "TransformerFineTuner",
    "compute_class_weights",
    "run_transformer_training_pipeline",
    "TransformerEvaluator",
    "run_transformer_evaluation_pipeline",
    "ModelComparator",
    "run_model_comparison_pipeline",
    "TransformerErrorAnalyzer",
    "run_error_analysis_pipeline",
    "TransformerIssueClassifier",
    "export_production_bundle",
]
