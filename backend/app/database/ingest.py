"""Data ingestion pipeline for OpenSourceLens.

Loads repository metadata, processed issues with resolution attributes,
issue labels, analytical aggregate snapshots, and precomputed semantic similarity
pairings into the relational database.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.app.database.base import Base
from backend.app.database.init_db import create_database_tables, drop_database_tables
from backend.app.database.models import (
    Issue,
    IssueLabel,
    RelatedIssuesCache,
    Repository,
    RepositoryAnalytics,
)
from backend.app.database.session import SessionLocal, build_engine, engine as default_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default asset paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPOS_DIR = PROJECT_ROOT / "data" / "raw" / "repositories"
DEFAULT_ISSUES_PARQUET = PROJECT_ROOT / "data" / "processed" / "issues_resolution.parquet"
DEFAULT_LABEL_MAPPINGS = PROJECT_ROOT / "data" / "processed" / "label_mappings.json"
DEFAULT_EMBEDDINGS_NPZ = PROJECT_ROOT / "models" / "related_issues" / "embeddings.npz"


def _parse_iso_datetime(val: Any) -> Optional[datetime]:
    """Safely convert strings or timestamps to timezone-aware UTC datetime."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        if val.tzinfo is None:
            return val.tz_localize(timezone.utc).to_pydatetime()
        return val.tz_convert(timezone.utc).to_pydatetime()
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def ingest_repositories(
    session: Session, repos_dir: Optional[Path] = None
) -> List[Repository]:
    """Ingest repositories from raw JSON files into the repositories table.

    Args:
        session: Active SQLAlchemy session.
        repos_dir: Directory containing raw repository JSON files.

    Returns:
        List of created or updated Repository instances.
    """
    directory = Path(repos_dir or DEFAULT_REPOS_DIR)
    if not directory.exists():
        raise FileNotFoundError(f"Repositories directory not found: {directory}")

    repo_files = list(directory.glob("*.json"))
    if not repo_files:
        raise ValueError(f"No JSON repository files found in: {directory}")

    ingested = []
    for filepath in repo_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        repo_id = int(data["id"])
        full_name = data.get("full_name") or f"{data['owner']['login']}/{data['name']}"
        owner_name = data["owner"]["login"] if isinstance(data.get("owner"), dict) else data.get("owner", "")

        existing = session.get(Repository, repo_id)
        if existing:
            # Update attributes
            existing.repo_name = full_name
            existing.owner = owner_name
            existing.name = data.get("name", "")
            existing.description = data.get("description")
            existing.primary_language = data.get("language")
            existing.stars_count = int(data.get("stargazers_count", 0))
            existing.forks_count = int(data.get("forks_count", 0))
            existing.open_issues_count = int(data.get("open_issues_count", 0))
            existing.github_created_at = _parse_iso_datetime(data.get("created_at"))
            existing.github_updated_at = _parse_iso_datetime(data.get("updated_at"))
            ingested.append(existing)
        else:
            repo = Repository(
                id=repo_id,
                repo_name=full_name,
                owner=owner_name,
                name=data.get("name", ""),
                description=data.get("description"),
                primary_language=data.get("language"),
                stars_count=int(data.get("stargazers_count", 0)),
                forks_count=int(data.get("forks_count", 0)),
                open_issues_count=int(data.get("open_issues_count", 0)),
                github_created_at=_parse_iso_datetime(data.get("created_at")),
                github_updated_at=_parse_iso_datetime(data.get("updated_at")),
            )
            session.add(repo)
            ingested.append(repo)

    session.commit()
    logger.info("Successfully ingested %d repositories.", len(ingested))
    return ingested


def ingest_issues(
    session: Session,
    parquet_path: Optional[Path] = None,
    batch_size: int = 500,
) -> int:
    """Ingest cleaned issues with resolution metrics into the issues table.

    Args:
        session: Active SQLAlchemy session.
        parquet_path: Path to issues_resolution.parquet.
        batch_size: Number of records per commit batch.

    Returns:
        Count of ingested issues.
    """
    path = Path(parquet_path or DEFAULT_ISSUES_PARQUET)
    if not path.exists():
        raise FileNotFoundError(f"Processed issues parquet not found: {path}")

    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s for issues ingestion.", len(df), path.name)

    # Build repository lookup cache
    repo_map = {r.repo_name: r.id for r in session.scalars(select(Repository)).all()}
    if not repo_map:
        raise ValueError("Repositories table is empty. Ingest repositories before issues.")

    # Remove existing issues if any to ensure clean idempotent insert
    existing_count = session.scalar(select(func.count()).select_from(Issue))
    if existing_count > 0:
        logger.info("Purging %d existing issues for fresh ingestion.", existing_count)
        session.execute(delete(Issue))
        session.commit()

    issues_to_add: List[Issue] = []
    total_added = 0

    for _, row in df.iterrows():
        repo_name = row["repo_name"]
        repo_id = repo_map.get(repo_name)
        if repo_id is None:
            logger.warning("Issue #%s has unmapped repo %s; skipping.", row["issue_number"], repo_name)
            continue

        issue = Issue(
            id=int(row["issue_id"]),
            repository_id=repo_id,
            repo_name=repo_name,
            issue_number=int(row["issue_number"]),
            title=str(row["title"]),
            body=str(row["body"]) if pd.notna(row.get("body")) else None,
            clean_title=str(row["clean_title"]) if pd.notna(row.get("clean_title")) else None,
            clean_body=str(row["clean_body"]) if pd.notna(row.get("clean_body")) else None,
            combined_text=str(row["combined_text"]) if pd.notna(row.get("combined_text")) else None,
            clean_title_word_count=int(row.get("clean_title_word_count", 0)),
            clean_body_word_count=int(row.get("clean_body_word_count", 0)),
            state=str(row["state"]),
            state_reason=str(row["state_reason"]) if pd.notna(row.get("state_reason")) else None,
            author=str(row["author"]),
            author_type=str(row.get("author_type", "User")),
            author_association=str(row.get("author_association", "NONE")),
            is_internal_author=bool(row.get("is_internal_author", False)),
            is_bot_author=bool(row.get("is_bot_author", False)),
            created_at=_parse_iso_datetime(row["created_at"]),
            updated_at=_parse_iso_datetime(row.get("updated_at")),
            closed_at=_parse_iso_datetime(row.get("closed_at")),
            comments_count=int(row.get("comments_count", 0)),
            reactions_count=int(row.get("reactions_count", 0)),
            labels_count=int(row.get("labels_count", 0)),
            has_code_block=bool(row.get("has_code_block", False)),
            code_block_count=int(row.get("code_block_count", 0)),
            has_url=bool(row.get("has_url", False)),
            has_stack_trace=bool(row.get("has_stack_trace", False)),
            canonical_label=str(row.get("canonical_label", "other")),
            canonical_label_id=int(row.get("canonical_label_id", -1)),
            is_classified_category=bool(row.get("is_classified_category", False)),
            is_unlabeled=bool(row.get("is_unlabeled", False)),
            resolution_time_hours=float(row["resolution_time_hours"]) if pd.notna(row.get("resolution_time_hours")) else None,
            resolution_time_days=float(row["resolution_time_days"]) if pd.notna(row.get("resolution_time_days")) else None,
            speed_tier=str(row["speed_tier"]) if pd.notna(row.get("speed_tier")) else None,
            resolution_type=str(row["resolution_type"]) if pd.notna(row.get("resolution_type")) else None,
            is_censored=bool(row.get("is_censored", False)),
            is_immediate_closure=bool(row.get("is_immediate_closure", False)),
            duration_at_risk_days=float(row["duration_at_risk_days"]) if pd.notna(row.get("duration_at_risk_days")) else None,
        )
        issues_to_add.append(issue)

        if len(issues_to_add) >= batch_size:
            session.add_all(issues_to_add)
            session.commit()
            total_added += len(issues_to_add)
            issues_to_add = []

    if issues_to_add:
        session.add_all(issues_to_add)
        session.commit()
        total_added += len(issues_to_add)

    logger.info("Successfully ingested %d issues.", total_added)
    return total_added


def ingest_issue_labels(
    session: Session,
    parquet_path: Optional[Path] = None,
    label_mappings_path: Optional[Path] = None,
    batch_size: int = 1000,
) -> int:
    """Ingest normalized labels from issues_resolution.parquet into issue_labels table.

    Args:
        session: Active SQLAlchemy session.
        parquet_path: Path to issues_resolution.parquet.
        label_mappings_path: Path to label_mappings.json.
        batch_size: Batch size for insertions.

    Returns:
        Count of ingested labels.
    """
    path = Path(parquet_path or DEFAULT_ISSUES_PARQUET)
    mappings_file = Path(label_mappings_path or DEFAULT_LABEL_MAPPINGS)

    rules: Dict[str, List[str]] = {}
    if mappings_file.exists():
        with open(mappings_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            rules = mapping_data.get("rules", {})

    # Invert mapping rules for O(1) canonical categorization
    raw_to_canonical: Dict[str, str] = {}
    for cat, aliases in rules.items():
        for alias in aliases:
            raw_to_canonical[alias.strip().lower()] = cat

    df = pd.read_parquet(path)

    # Clean existing labels
    session.execute(delete(IssueLabel))
    session.commit()

    labels_to_add: List[IssueLabel] = []
    total_added = 0

    for _, row in df.iterrows():
        issue_id = int(row["issue_id"])
        raw_labels = row.get("raw_labels")

        if raw_labels is None:
            continue

        if isinstance(raw_labels, np.ndarray):
            label_list = raw_labels.tolist()
        elif isinstance(raw_labels, list):
            label_list = raw_labels
        else:
            continue

        for lbl in label_list:
            lbl_str = str(lbl).strip()
            if not lbl_str:
                continue

            canonical_cat = raw_to_canonical.get(lbl_str.lower())
            is_canon = canonical_cat is not None

            label_record = IssueLabel(
                issue_id=issue_id,
                label_name=lbl_str,
                is_canonical=is_canon,
                canonical_category=canonical_cat,
            )
            labels_to_add.append(label_record)

            if len(labels_to_add) >= batch_size:
                session.add_all(labels_to_add)
                session.commit()
                total_added += len(labels_to_add)
                labels_to_add = []

    if labels_to_add:
        session.add_all(labels_to_add)
        session.commit()
        total_added += len(labels_to_add)

    logger.info("Successfully ingested %d issue labels.", total_added)
    return total_added


def ingest_repository_analytics(session: Session) -> List[RepositoryAnalytics]:
    """Calculate and store repository analytical aggregates into repository_analytics.

    Args:
        session: Active SQLAlchemy session.

    Returns:
        List of created RepositoryAnalytics snapshot records.
    """
    repos = session.scalars(select(Repository)).all()
    if not repos:
        raise ValueError("Repositories table is empty. Ingest repositories first.")

    # Clean existing analytics snapshots
    session.execute(delete(RepositoryAnalytics))
    session.commit()

    now = datetime.now(timezone.utc)
    analytics_records = []

    for repo in repos:
        issues = session.scalars(select(Issue).where(Issue.repository_id == repo.id)).all()
        total = len(issues)
        if total == 0:
            continue

        open_count = sum(1 for i in issues if i.state == "open")
        closed_count = sum(1 for i in issues if i.state == "closed")

        # Category distribution
        cat_counts: Dict[str, int] = {}
        for i in issues:
            cat = i.canonical_label or "other"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # Speed tier distribution
        tier_counts: Dict[str, int] = {}
        for i in issues:
            tier = i.speed_tier or "Unresolved"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Resolution statistics for closed issues
        closed_hours = [
            i.resolution_time_hours
            for i in issues
            if i.state == "closed" and i.resolution_time_hours is not None
        ]

        if closed_hours:
            arr = np.array(closed_hours)
            med_h = float(np.median(arr))
            mean_h = float(np.mean(arr))
            p25_h = float(np.percentile(arr, 25))
            p75_h = float(np.percentile(arr, 75))
            # Fraction of issues taking > 30 days (720 hours) or unresolved
            gt_30d = sum(1 for h in closed_hours if h > 720.0) + open_count
            survival_30d = float(gt_30d / total)
        else:
            med_h = mean_h = p25_h = p75_h = survival_30d = None

        record = RepositoryAnalytics(
            repository_id=repo.id,
            repo_name=repo.repo_name,
            snapshot_date=now,
            total_issues=total,
            open_issues=open_count,
            closed_issues=closed_count,
            category_distribution=cat_counts,
            median_resolution_hours=round(med_h, 2) if med_h is not None else None,
            mean_resolution_hours=round(mean_h, 2) if mean_h is not None else None,
            p25_resolution_hours=round(p25_h, 2) if p25_h is not None else None,
            p75_resolution_hours=round(p75_h, 2) if p75_h is not None else None,
            survival_rate_30d=round(survival_30d, 4) if survival_30d is not None else None,
            speed_tier_distribution=tier_counts,
            created_at=now,
        )
        session.add(record)
        analytics_records.append(record)

    session.commit()
    logger.info("Successfully computed and ingested analytics for %d repositories.", len(analytics_records))
    return analytics_records


def ingest_related_issues_cache(
    session: Session,
    embeddings_path: Optional[Path] = None,
    issues_parquet_path: Optional[Path] = None,
    top_k: int = 5,
    batch_size: int = 1000,
) -> int:
    """Precompute and store top-K semantic similarity pairings per repository.

    Args:
        session: Active SQLAlchemy session.
        embeddings_path: Path to embeddings.npz.
        issues_parquet_path: Path to issues_resolution.parquet.
        top_k: Top K nearest neighbors to retain per issue.
        batch_size: Batch size for insertions.

    Returns:
        Count of cached similarity edges.
    """
    emb_path = Path(embeddings_path or DEFAULT_EMBEDDINGS_NPZ)
    parquet_path = Path(issues_parquet_path or DEFAULT_ISSUES_PARQUET)

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Issues file not found: {parquet_path}")

    # Load embeddings and issue ID mapping
    npz_data = np.load(emb_path)
    embeddings = npz_data["embeddings"]  # Shape: (N, 384)
    emb_issue_ids = npz_data["issue_ids"]  # Shape: (N,)

    # Load metadata to group by repo
    df = pd.read_parquet(parquet_path, columns=["issue_id", "repo_name"])
    id_to_repo = dict(zip(df["issue_id"], df["repo_name"]))

    # Clean existing related cache
    session.execute(delete(RelatedIssuesCache))
    session.commit()

    # Pre-index embeddings by repository
    repo_indices: Dict[str, List[int]] = {}
    for idx, issue_id in enumerate(emb_issue_ids):
        repo = id_to_repo.get(issue_id)
        if repo:
            repo_indices.setdefault(repo, []).append(idx)

    edges_to_add: List[RelatedIssuesCache] = []
    total_added = 0
    now = datetime.now(timezone.utc)

    for repo_name, idx_list in repo_indices.items():
        sub_indices = np.array(idx_list)
        sub_embeddings = embeddings[sub_indices]  # (M, 384)
        sub_issue_ids = emb_issue_ids[sub_indices]  # (M,)
        m_count = len(sub_indices)

        if m_count < 2:
            continue

        effective_k = min(top_k, m_count - 1)

        # Pairwise dot product (embeddings are already L2-normalized)
        sim_matrix = np.dot(sub_embeddings, sub_embeddings.T)  # (M, M)

        # Mask diagonal (self-match)
        np.fill_diagonal(sim_matrix, -1.0)

        # For each issue, get top_k candidates
        for i in range(m_count):
            src_id = int(sub_issue_ids[i])
            row_sims = sim_matrix[i]

            # Fast top_k selection
            top_k_indices = np.argpartition(row_sims, -effective_k)[-effective_k:]
            sorted_top_k = top_k_indices[np.argsort(-row_sims[top_k_indices])]

            for rank_0, target_idx in enumerate(sorted_top_k):
                target_id = int(sub_issue_ids[target_idx])
                score = float(row_sims[target_idx])

                edge = RelatedIssuesCache(
                    source_issue_id=src_id,
                    target_issue_id=target_id,
                    similarity_score=round(score, 4),
                    rank=rank_0 + 1,
                    model_version="all-MiniLM-L6-v2",
                    calculated_at=now,
                )
                edges_to_add.append(edge)

                if len(edges_to_add) >= batch_size:
                    session.add_all(edges_to_add)
                    session.commit()
                    total_added += len(edges_to_add)
                    edges_to_add = []

    if edges_to_add:
        session.add_all(edges_to_add)
        session.commit()
        total_added += len(edges_to_add)

    logger.info("Successfully ingested %d related issue pairings.", total_added)
    return total_added


def run_ingest_pipeline(
    session: Session,
    clean: bool = False,
    top_k: int = 5,
    target_engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute the full end-to-end data ingestion pipeline.

    Args:
        session: Active SQLAlchemy session.
        clean: If True, drops and recreates schema before ingestion.
        top_k: Number of related neighbors to compute per issue.
        target_engine: Optional SQLAlchemy engine.

    Returns:
        Summary dict of pipeline results.
    """
    eng = target_engine or default_engine
    t0 = time.perf_counter()

    if clean:
        logger.info("Clean ingestion requested. Dropping and recreating tables.")
        drop_database_tables(eng, confirm=True)
        create_database_tables(eng)
    else:
        create_database_tables(eng)

    repos = ingest_repositories(session)
    issues_count = ingest_issues(session)
    labels_count = ingest_issue_labels(session)
    analytics = ingest_repository_analytics(session)
    related_count = ingest_related_issues_cache(session, top_k=top_k)

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "status": "success",
        "repositories_count": len(repos),
        "issues_count": issues_count,
        "labels_count": labels_count,
        "analytics_snapshots": len(analytics),
        "related_edges_count": related_count,
        "elapsed_seconds": elapsed,
    }
    logger.info("Ingestion pipeline completed in %.2f seconds: %s", elapsed, summary)
    return summary


def main() -> int:
    """CLI entrypoint for data ingestion."""
    parser = argparse.ArgumentParser(description="OpenSourceLens Data Ingestion Pipeline")
    parser.add_argument("--url", type=str, default=None, help="Database URL")
    parser.add_argument("--clean", action="store_true", help="Drop and recreate tables before loading")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (repos, issues, labels, analytics, related)")
    parser.add_argument("--repos", action="store_true", help="Ingest repositories only")
    parser.add_argument("--issues", action="store_true", help="Ingest issues only")
    parser.add_argument("--labels", action="store_true", help="Ingest labels only")
    parser.add_argument("--analytics", action="store_true", help="Compute analytics snapshots only")
    parser.add_argument("--related", action="store_true", help="Compute related issue pairings only")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K related issues (default 5)")

    args = parser.parse_args()
    target_eng = build_engine(args.url) if args.url else default_engine
    local_session = SessionLocal() if not args.url else Session(bind=target_eng)

    try:
        if args.clean or args.all or not any([args.repos, args.issues, args.labels, args.analytics, args.related]):
            summary = run_ingest_pipeline(
                local_session, clean=args.clean, top_k=args.top_k, target_engine=target_eng
            )
            print("\n--- Ingestion Pipeline Summary ---")
            print(json.dumps(summary, indent=2))
            return 0

        if args.repos:
            ingest_repositories(local_session)
        if args.issues:
            ingest_issues(local_session)
        if args.labels:
            ingest_issue_labels(local_session)
        if args.analytics:
            ingest_repository_analytics(local_session)
        if args.related:
            ingest_related_issues_cache(local_session, top_k=args.top_k)

        print("\nTarget ingestion stage completed successfully.")
        return 0
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    finally:
        local_session.close()
        target_eng.dispose()


if __name__ == "__main__":
    sys.exit(main())
