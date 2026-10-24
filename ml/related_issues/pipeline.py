"""
OpenSourceLens — Related Issue Inference Pipeline (Phase 7, Step 7.7)
Unified production service for semantic related-issue detection and duplicate alerting.
Provides API-ready standardized JSON responses with GitHub URLs, confidence tiers,
and actionable triage guidance.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from ml.related_issues.embedder import IssueEmbedder
from ml.related_issues.retriever import RelatedIssueRetriever
from ml.related_issues.store import IssueEmbeddingStore
from ml.related_issues.threshold import (
    DEFAULT_THRESHOLDS,
    SimilarityThresholdCalibrator,
)


CONFIDENCE_TIER_CUTOFFS: Dict[str, float] = {
    "HIGH": 0.70,
    "MEDIUM": 0.55,
    "LOW": 0.45,
    "ALL": -1.0,
}


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder converting NumPy types to Python primitives."""
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


def sanitize_val(val: Any) -> Any:
    """Sanitize NumPy and Pandas types to standard JSON-compatible Python primitives."""
    if hasattr(val, "item"):
        return val.item()
    if pd_not_na(val):
        if isinstance(val, (int, float, str, bool)):
            return val
        if str(val).isdigit():
            return int(val)
        return str(val)
    return None


def build_github_issue_url(repo_name: str, issue_number: Optional[int]) -> str:
    """Construct direct GitHub URL for an issue."""
    clean_repo = repo_name.strip()
    clean_num = sanitize_val(issue_number)
    if clean_num is not None and isinstance(clean_num, int) and clean_num > 0:
        return f"https://github.com/{clean_repo}/issues/{clean_num}"
    return f"https://github.com/{clean_repo}/issues"


def pd_not_na(val: Any) -> bool:
    """Helper to check if a value is not NaN without requiring pandas import."""
    return val is not None and val == val and str(val).lower() not in ("nan", "none", "<na>")


class RelatedIssuePipeline:
    """Unified production pipeline for semantic related-issue retrieval and duplicate detection."""

    def __init__(
        self,
        store_dir: Union[str, Path] = "models/related_issues",
        embedder: Optional[IssueEmbedder] = None,
        store: Optional[IssueEmbeddingStore] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.store_dir = Path(store_dir)
        self.store = store if store is not None else IssueEmbeddingStore(self.store_dir)
        self.embedder = embedder
        self.retriever = RelatedIssueRetriever(
            store=self.store,
            embedder=self.embedder,
            store_dir=self.store_dir,
        )
        self.calibrator = SimilarityThresholdCalibrator(
            thresholds=thresholds or DEFAULT_THRESHOLDS
        )
        self.model_name = "all-MiniLM-L6-v2"

    def _ensure_initialized(self) -> None:
        """Ensure store and embedder are loaded."""
        if not self.store.is_loaded:
            self.store.load(verify_checksums=True)
        if self.embedder is None:
            self.embedder = IssueEmbedder(model_name=self.model_name, normalize_embeddings=True)
            self.retriever.embedder = self.embedder

    def _format_payload(
        self,
        query_info: Dict[str, Any],
        raw_candidates: List[Dict[str, Any]],
        latency_ms: float,
        min_confidence: str = "LOW",
    ) -> Dict[str, Any]:
        """Format candidate results into the standardized production schema."""
        cutoff = CONFIDENCE_TIER_CUTOFFS.get(min_confidence.upper(), 0.45)

        # Enrich and filter through threshold calibrator
        enriched_candidates = self.calibrator.filter_and_enrich_candidates(
            candidates=raw_candidates,
            min_threshold=cutoff,
        )

        formatted_issues: List[Dict[str, Any]] = []
        has_duplicate = False

        for rank, cand in enumerate(enriched_candidates, 1):
            is_dup = (cand.get("confidence_tier") == "HIGH")
            if is_dup:
                has_duplicate = True

            issue_num = cand.get("issue_number")
            repo = cand.get("repo_name", "")

            formatted_issues.append({
                "rank": rank,
                "issue_id": sanitize_val(cand.get("issue_id")),
                "issue_number": sanitize_val(issue_num),
                "repo_name": repo,
                "title": cand.get("title", ""),
                "clean_title": cand.get("clean_title", ""),
                "canonical_label": cand.get("canonical_label", "other"),
                "state": cand.get("state", "open"),
                "created_at": cand.get("created_at", ""),
                "html_url": build_github_issue_url(repo, issue_num),
                "similarity_score": cand.get("similarity_score", 0.0),
                "confidence_tier": cand.get("confidence_tier", "LOW"),
                "tier_label": cand.get("tier_label", "Weak Match"),
                "confidence": cand.get("confidence", "Low"),
                "action_recommendation": cand.get("action_recommendation", ""),
                "is_potential_duplicate": is_dup,
            })

        # Generate summary
        total_found = len(formatted_issues)
        highest_score = formatted_issues[0]["similarity_score"] if total_found > 0 else 0.0

        if has_duplicate:
            top_dup = formatted_issues[0]
            primary_rec = (
                f"Potential duplicate detected: #{top_dup['issue_number']} "
                f"in {top_dup['repo_name']} ({top_dup['similarity_score']*100:.1f}% similarity). "
                f"Review existing issue before filing."
            )
        elif total_found > 0 and formatted_issues[0]["confidence_tier"] == "MEDIUM":
            top_rel = formatted_issues[0]
            primary_rec = (
                f"Found {total_found} semantically related issue(s) sharing component context. "
                f"Top match: #{top_rel['issue_number']} ({top_rel['similarity_score']*100:.1f}% similarity)."
            )
        elif total_found > 0:
            primary_rec = (
                f"Found {total_found} weakly related issue(s) with partial topical overlap. "
                f"No direct duplicates identified."
            )
        else:
            primary_rec = "No related issues found meeting the confidence threshold. Unique issue report."

        return {
            "query": {
                **query_info,
                "min_confidence_filter": min_confidence.upper(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "summary": {
                "total_candidates_found": total_found,
                "has_potential_duplicate": has_duplicate,
                "highest_similarity_score": highest_score,
                "primary_recommendation": primary_rec,
            },
            "related_issues": formatted_issues,
            "execution_metadata": {
                "latency_ms": round(latency_ms, 2),
                "embedding_model": self.model_name,
                "corpus_size": len(self.store.embeddings) if self.store.is_loaded else 0,
                "pipeline_version": "v1.0.0",
            },
        }

    def find_related_by_text(
        self,
        title: str,
        body: str = "",
        repo_name: Optional[str] = None,
        top_k: int = 5,
        min_confidence: str = "LOW",
        state_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze arbitrary issue title and description to retrieve related issues.

        Args:
            title: Title of query issue.
            body: Description body of query issue.
            repo_name: Optional repository to constrain search.
            top_k: Maximum number of candidate issues to return.
            min_confidence: Minimum confidence tier ('HIGH', 'MEDIUM', 'LOW', 'ALL').
            state_filter: Optional issue state filter ('open', 'closed').

        Returns:
            Standardized production response dictionary.
        """
        self._ensure_initialized()
        t0 = time.perf_counter()

        raw_candidates = self.retriever.retrieve_by_text(
            title=title,
            body=body,
            top_k=top_k,
            repo_filter=repo_name,
            state_filter=state_filter,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        query_info = {
            "query_type": "text",
            "title": title,
            "has_body": bool(body.strip()),
            "repo_scope": repo_name or "all",
            "top_k_requested": top_k,
        }

        return self._format_payload(
            query_info=query_info,
            raw_candidates=raw_candidates,
            latency_ms=elapsed_ms,
            min_confidence=min_confidence,
        )

    def find_related_by_id(
        self,
        issue_id: Union[int, str],
        top_k: int = 5,
        min_confidence: str = "LOW",
        repo_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve related issues for an existing historical issue, strictly excluding self.

        Args:
            issue_id: Database issue ID.
            top_k: Maximum number of candidates to return.
            min_confidence: Minimum confidence tier.
            repo_filter: Optional repository constraint.
            state_filter: Optional state filter.

        Returns:
            Standardized production response dictionary.
        """
        self._ensure_initialized()
        t0 = time.perf_counter()

        # Fetch original issue metadata for query reporting
        _, meta = self.store.get_issue_by_id(issue_id)
        raw_candidates = self.retriever.retrieve_by_id(
            issue_id=issue_id,
            top_k=top_k,
            repo_filter=repo_filter,
            state_filter=state_filter,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        query_info = {
            "query_type": "issue_id",
            "issue_id": sanitize_val(issue_id),
            "issue_number": sanitize_val(meta.get("issue_number")),
            "repo_name": str(meta.get("repo_name", "")),
            "title": str(meta.get("title", "")),
            "top_k_requested": top_k,
        }

        return self._format_payload(
            query_info=query_info,
            raw_candidates=raw_candidates,
            latency_ms=elapsed_ms,
            min_confidence=min_confidence,
        )


def run_pipeline_demo(
    store_dir: Union[str, Path] = "models/related_issues",
    output_sample_path: Union[str, Path] = "data/processed/evaluation/sample_related_issues_output.json",
) -> Tuple[RelatedIssuePipeline, Dict[str, Any]]:
    """Demonstrate end-to-end related-issue pipeline and export sample API payload for Step 7.7."""
    print(f"\n=======================================================")
    print(f"Starting Related Issue Inference Pipeline Demo (Phase 7, Step 7.7)")
    print(f"Store: {store_dir}")
    print(f"=======================================================\n")

    pipeline = RelatedIssuePipeline(store_dir=store_dir)

    # 1. Demo Natural Language Query
    demo_title = "CUDA memory out of bounds during backpropagation with multi-GPU training"
    demo_body = "Encountered CUDA runtime error: device-side assert triggered when calling loss.backward() with torch.distributed."
    print("Executing Natural Language Query Analysis...")
    text_payload = pipeline.find_related_by_text(
        title=demo_title,
        body=demo_body,
        repo_name="pytorch/pytorch",
        top_k=3,
        min_confidence="LOW",
    )

    # 2. Demo Query by Existing Issue ID
    sample_issue_id = pipeline.store.metadata.iloc[10]["issue_id"]
    print(f"Executing Historical Issue ID Query Analysis (ID: {sample_issue_id})...")
    id_payload = pipeline.find_related_by_id(
        issue_id=sample_issue_id,
        top_k=3,
        min_confidence="LOW",
    )

    sample_output = {
        "text_query_response": text_payload,
        "id_query_response": id_payload,
    }

    out_path = Path(output_sample_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2, cls=NumpyJSONEncoder)

    print(f"Sample related-issue output JSON saved to: {out_path}")

    print("\n--- Pipeline Demo Summary ---")
    print(f"Text Query Latency:        {text_payload['execution_metadata']['latency_ms']} ms")
    print(f"Text Candidates Found:     {text_payload['summary']['total_candidates_found']}")
    print(f"Has Duplicate Alert:       {text_payload['summary']['has_potential_duplicate']}")
    print(f"Primary Recommendation:    {text_payload['summary']['primary_recommendation']}")
    print(f"\nID Query Latency:          {id_payload['execution_metadata']['latency_ms']} ms")
    print(f"ID Candidates Found:       {id_payload['summary']['total_candidates_found']}")
    print("-----------------------------\n")

    return pipeline, sample_output


if __name__ == "__main__":
    run_pipeline_demo()
