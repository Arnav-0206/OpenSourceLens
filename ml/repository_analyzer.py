"""
OpenSourceLens — Repository Analyzer Service
Milestone Enhancement: Live GitHub Repository Ingestion, Batch Classification,
and Intra-Repository Semantic Duplicate Detection.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml.data_collection.github_client import GitHubAPIError, GitHubClient, NotFoundError
from ml.data_collection.issue_collector import IssueCollector, is_pull_request
from ml.data_collection.repository_collector import RepositoryCollector
from ml.deep_learning.pipeline import TransformerIssueClassifier


def normalize_repo_input(raw_input: str) -> Tuple[str, str]:
    """Validate and normalize a GitHub repository string into (owner, repo).

    Accepts:
        - 'owner/repo'
        - 'https://github.com/owner/repo'
        - 'http://github.com/owner/repo/'
        - 'github.com/owner/repo'

    Returns:
        Tuple of (owner, repo).

    Raises:
        ValueError: If repository identifier is malformed.
    """
    cleaned = (raw_input or "").strip().rstrip("/")
    if "github.com/" in cleaned:
        cleaned = cleaned.split("github.com/")[-1]

    parts = [p.strip() for p in cleaned.split("/") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid repository identifier: '{raw_input}'. "
            f"Expected format: 'owner/repo' (e.g., 'psf/requests' or 'https://github.com/psf/requests')."
        )
    return parts[0], parts[1]


def build_github_issue_url(owner: str, repo: str, issue_number: Union[int, str]) -> str:
    """Build direct URL to a GitHub issue."""
    return f"https://github.com/{owner}/{repo}/issues/{issue_number}"


def fetch_repo_metadata_and_issues(
    repo_str: str,
    target_count: int = 25,
    client: Optional[GitHubClient] = None,
) -> Dict[str, Any]:
    """Ingest repository metadata and active issues from GitHub REST API.

    Args:
        repo_str: Repository slug or URL (e.g. 'psf/requests').
        target_count: Desired number of pure issues to ingest (default: 25).
        client: Optional shared GitHubClient.

    Returns:
        Dictionary containing standardized repository metadata and list of issues.
    """
    owner, repo = normalize_repo_input(repo_str)
    full_slug = f"{owner}/{repo}"
    active_client = client or GitHubClient()

    # 1. Fetch Repository Metadata
    repo_collector = RepositoryCollector(client=active_client)
    repo_info = repo_collector.collect(full_slug)
    meta = repo_info["metadata"]

    standard_meta = {
        "owner": owner,
        "repo": repo,
        "full_name": meta.get("full_name") or full_slug,
        "description": meta.get("description") or "No description provided.",
        "stars": meta.get("stars", 0) or 0,
        "forks": meta.get("forks", 0) or 0,
        "open_issues_count": meta.get("open_issues_count", 0) or 0,
        "language": meta.get("language") or "General",
        "html_url": f"https://github.com/{full_slug}",
    }

    # 2. Fetch Recent Pure Issues (filtering out pull requests)
    issue_collector = IssueCollector(client=active_client)
    pure_issues: List[Dict[str, Any]] = []

    # Attempt 1: Fetch recent open issues
    raw_open_items = issue_collector.collect_issues_page(
        repo_identifier=full_slug,
        state="open",
        per_page=min(max(target_count * 2, 40), 100),
        sort="created",
        direction="desc",
        filter_pull_requests=True,
    )

    for item in raw_open_items:
        iss = item["issue"]
        pure_issues.append({
            "issue_id": iss.get("issue_id"),
            "issue_number": iss.get("issue_number"),
            "title": iss.get("title", ""),
            "body": iss.get("body", ""),
            "state": iss.get("state", "open"),
            "author": iss.get("author", "unknown"),
            "labels": iss.get("labels", []),
            "comments_count": iss.get("comments_count", 0),
            "created_at": iss.get("created_at"),
            "html_url": build_github_issue_url(owner, repo, iss.get("issue_number", 0)),
        })
        if len(pure_issues) >= target_count:
            break

    # Graceful fallback: If repo has very few open issues, fetch recent closed issues
    if len(pure_issues) < 10:
        raw_all_items = issue_collector.collect_issues_page(
            repo_identifier=full_slug,
            state="all",
            per_page=min(max(target_count * 2, 40), 100),
            sort="created",
            direction="desc",
            filter_pull_requests=True,
        )
        seen_numbers = {i["issue_number"] for i in pure_issues}
        for item in raw_all_items:
            iss = item["issue"]
            if iss.get("issue_number") not in seen_numbers:
                pure_issues.append({
                    "issue_id": iss.get("issue_id"),
                    "issue_number": iss.get("issue_number"),
                    "title": iss.get("title", ""),
                    "body": iss.get("body", ""),
                    "state": iss.get("state", "open"),
                    "author": iss.get("author", "unknown"),
                    "labels": iss.get("labels", []),
                    "comments_count": iss.get("comments_count", 0),
                    "created_at": iss.get("created_at"),
                    "html_url": build_github_issue_url(owner, repo, iss.get("issue_number", 0)),
                })
                seen_numbers.add(iss.get("issue_number"))
                if len(pure_issues) >= target_count:
                    break

    return {
        "metadata": standard_meta,
        "issues": pure_issues[:target_count],
    }


def classify_issues_batch(
    classifier: TransformerIssueClassifier,
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Classify a list of issues using the fine-tuned DistilBERT transformer.

    Args:
        classifier: Production TransformerIssueClassifier instance.
        issues: List of issue dictionaries containing 'title' and 'body'.

    Returns:
        List of enriched issue dictionaries with prediction results.
    """
    if not issues:
        return []

    # Prepare inputs for batch prediction
    batch_inputs = [{"title": it.get("title", ""), "body": it.get("body", "")} for it in issues]
    predictions = classifier.predict_batch(batch_inputs, batch_size=32)

    enriched_issues: List[Dict[str, Any]] = []
    for issue, pred in zip(issues, predictions):
        enriched = dict(issue)
        enriched["predicted_category"] = pred.get("label", "other")
        enriched["confidence"] = pred.get("confidence", 0.0)
        enriched["probabilities"] = pred.get("probabilities", {})
        enriched["is_uncertain"] = pred.get("is_uncertain", False)
        enriched_issues.append(enriched)

    return enriched_issues


def calculate_category_breakdown(classified_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate counts and percentage distribution of predicted categories."""
    total = len(classified_issues)
    if total == 0:
        return {
            "total": 0,
            "counts": {"bug": 0, "feature": 0, "other": 0},
            "percentages": {"bug": 0.0, "feature": 0.0, "other": 0.0},
        }

    counts = {"bug": 0, "feature": 0, "other": 0}
    for item in classified_issues:
        cat = item.get("predicted_category", "other").lower()
        if cat in counts:
            counts[cat] += 1
        else:
            counts["other"] += 1

    percentages = {
        cat: round((cnt / total) * 100.0, 1) for cat, cnt in counts.items()
    }

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
    }


def find_intra_repo_duplicates(
    embedder: Any,
    issues: List[Dict[str, Any]],
    duplicate_threshold: float = 0.80,
) -> Dict[str, Any]:
    """Compute pairwise cosine similarities across issues to identify intra-repo duplicates.

    Args:
        embedder: IssueEmbedder or SentenceTransformer instance.
        issues: List of issue dictionaries.
        duplicate_threshold: Similarity score cutoff to flag as duplicate (default: 0.80).

    Returns:
        Dictionary with duplicate alerts and top similarity pairs.
    """
    if len(issues) < 2:
        return {
            "has_duplicates": False,
            "duplicate_pairs": [],
            "highest_similarity_pair": None,
        }

    # Format texts for embedding
    texts = []
    for it in issues:
        clean_title = (it.get("title") or "").strip()
        body_snippet = (it.get("body") or "").strip()[:200]
        if body_snippet:
            texts.append(f"Title: {clean_title} | Description: {body_snippet}")
        else:
            texts.append(f"Title: {clean_title}")

    # Generate unit-normalized embeddings
    if hasattr(embedder, "encode_texts"):
        embeddings = embedder.encode_texts(texts)
    elif hasattr(embedder, "encode"):
        embeddings = embedder.encode(texts, normalize_embeddings=True)
    elif hasattr(embedder, "embed_batch"):
        embeddings = embedder.embed_batch(texts)
    elif hasattr(embedder, "embed"):
        embeddings = embedder.embed(texts)
    else:
        raise ValueError("Unsupported embedder instance.")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Compute pairwise cosine similarity matrix
    sim_matrix = np.dot(embeddings, embeddings.T)

    n = len(issues)
    pairs: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            pair_info = {
                "issue_1": issues[i],
                "issue_2": issues[j],
                "similarity_score": round(score, 4),
                "similarity_pct": round(score * 100.0, 1),
            }
            pairs.append(pair_info)

    # Sort descending by similarity score
    pairs.sort(key=lambda p: p["similarity_score"], reverse=True)

    highest_pair = pairs[0] if pairs else None
    duplicate_pairs = [p for p in pairs if p["similarity_score"] >= duplicate_threshold]

    # Build recommendations for duplicate pairs
    for p in duplicate_pairs:
        num1 = p["issue_1"].get("issue_number")
        num2 = p["issue_2"].get("issue_number")
        p["recommendation"] = (
            f"Issues #{num1} and #{num2} share {p['similarity_pct']}% semantic similarity. "
            f"Review for duplicate root cause and link or merge discussion."
        )

    return {
        "has_duplicates": len(duplicate_pairs) > 0,
        "duplicate_pairs": duplicate_pairs,
        "highest_similarity_pair": highest_pair,
        "total_pairs_evaluated": len(pairs),
    }


def analyze_repository(
    repo_str: str,
    classifier: TransformerIssueClassifier,
    embedder: Any,
    target_count: int = 25,
    client: Optional[GitHubClient] = None,
) -> Dict[str, Any]:
    """Execute end-to-end repository analysis pipeline.

    1. Ingest metadata and issues.
    2. Batch classify into Bug/Feature/Other.
    3. Calculate category distribution breakdown.
    4. Compute pairwise similarities for duplicate detection.
    """
    t0 = time.perf_counter()

    # Step 1: Ingestion
    ingest_result = fetch_repo_metadata_and_issues(
        repo_str=repo_str,
        target_count=target_count,
        client=client,
    )
    metadata = ingest_result["metadata"]
    raw_issues = ingest_result["issues"]

    # Step 2: Classification
    classified_issues = classify_issues_batch(classifier=classifier, issues=raw_issues)

    # Step 3: Breakdown
    breakdown = calculate_category_breakdown(classified_issues)

    # Step 4: Duplicate Detection
    duplicates = find_intra_repo_duplicates(embedder=embedder, issues=classified_issues)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "repository": metadata,
        "issues": classified_issues,
        "breakdown": breakdown,
        "duplicates": duplicates,
        "execution_metadata": {
            "latency_ms": round(elapsed_ms, 1),
            "analyzed_count": len(classified_issues),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
