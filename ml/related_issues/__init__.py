"""OpenSourceLens Related Issue Detection Package (Phase 7).

Provides sentence transformer embedding generation, semantic vector indexing,
cosine similarity computation, and top-K related issue retrieval for GitHub repositories.
"""

from ml.related_issues.embedder import (
    IssueEmbedder,
    run_embedding_generation_pipeline,
)
from ml.related_issues.store import (
    IssueEmbeddingStore,
    run_embedding_storage_pipeline,
)
from ml.related_issues.similarity import (
    SimilarityEngine,
    run_similarity_pipeline,
)
from ml.related_issues.retriever import (
    RelatedIssueRetriever,
    run_retrieval_pipeline,
)
from ml.related_issues.threshold import (
    DEFAULT_THRESHOLDS,
    SimilarityThresholdCalibrator,
    run_threshold_calibration_pipeline,
)
from ml.related_issues.evaluator import (
    LexicalRetrievalBaseline,
    RetrievalEvaluator,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    run_retrieval_evaluation_pipeline,
)
from ml.related_issues.pipeline import (
    RelatedIssuePipeline,
    build_github_issue_url,
    run_pipeline_demo,
)

__all__ = [
    "IssueEmbedder",
    "run_embedding_generation_pipeline",
    "IssueEmbeddingStore",
    "run_embedding_storage_pipeline",
    "SimilarityEngine",
    "run_similarity_pipeline",
    "RelatedIssueRetriever",
    "run_retrieval_pipeline",
    "DEFAULT_THRESHOLDS",
    "SimilarityThresholdCalibrator",
    "run_threshold_calibration_pipeline",
    "LexicalRetrievalBaseline",
    "RetrievalEvaluator",
    "compute_mrr",
    "compute_hit_rate_at_k",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "run_retrieval_evaluation_pipeline",
    "RelatedIssuePipeline",
    "build_github_issue_url",
    "run_pipeline_demo",
]
