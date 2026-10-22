"""Label Taxonomy Normalization and Target Encoding Engine for OpenSourceLens.

Consolidates heterogeneous repository-specific GitHub labels into canonical classes:
- bug
- feature
- documentation
- question
- other (workflow / SIG / triage metadata only)
- unlabeled (0 labels)

Encodes canonical targets with integer IDs and flags supervised learning candidates.
"""

import sys
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd


class LabelNormalizer:
    """Normalizes disparate GitHub label strings into a unified classification taxonomy."""

    LABEL_TO_ID: Dict[str, int] = {
        "bug": 0,
        "feature": 1,
        "documentation": 2,
        "question": 3,
        "other": 4,
        "unlabeled": -1,
    }

    CORE_CATEGORIES = {"bug", "feature", "documentation", "question"}

    # Keyword matching definitions (case-folded)
    BUG_EXACT = {
        "bug",
        "kind/bug",
        "module: correctness (silent)",
        "kind/flake",
        "kind/failing-test",
        "module: flaky-tests",
        "module: crash",
        "defect",
        "crash",
        "regression",
        "type: bug",
    }
    BUG_SUBSTRINGS = ["bug", "crash", "defect"]

    FEATURE_EXACT = {
        "feature request",
        "kind/feature",
        "enhancement",
        "new model",
        "feature",
        "kind/enhancement",
        "type: feature",
        "type: enhancement",
    }
    FEATURE_SUBSTRINGS = ["feature", "enhancement"]

    DOCS_EXACT = {
        "documentation",
        "kind/documentation",
        "docs",
        "module: docs",
        "type: doc",
        "type: documentation",
    }
    DOCS_SUBSTRINGS = ["doc", "documentation"]

    QUESTION_EXACT = {
        "question",
        "kind/support",
        "support",
        "needs-information",
        "help wanted",
    }
    QUESTION_SUBSTRINGS = ["question"]

    def __init__(self, deterministic_priority: Optional[List[str]] = None):
        """Initialize label normalizer.

        Args:
            deterministic_priority: Priority order when an issue matches multiple categories.
                                    Defaults to ['bug', 'feature', 'documentation', 'question'].
        """
        self.priority = deterministic_priority or ["bug", "feature", "documentation", "question"]

    def map_labels(self, raw_labels: Any) -> str:
        """Map raw labels list to a canonical taxonomy class.

        Priority order:
            1. bug (highest priority for software engineering triage)
            2. feature
            3. documentation
            4. question
            5. other (has labels, but only workflow/triage/SIG tags)
            6. unlabeled (empty list or null)

        Args:
            raw_labels: List of label strings.

        Returns:
            Canonical label string.
        """
        if not isinstance(raw_labels, list) or len(raw_labels) == 0:
            return "unlabeled"

        normalized = [str(lbl).lower().strip() for lbl in raw_labels if lbl]
        if not normalized:
            return "unlabeled"

        # Check for matches
        has_bug = any(l in self.BUG_EXACT for l in normalized) or any(
            any(sub in l for sub in self.BUG_SUBSTRINGS) for l in normalized
        )
        has_feature = any(l in self.FEATURE_EXACT for l in normalized) or any(
            any(sub in l for sub in self.FEATURE_SUBSTRINGS) for l in normalized
        )
        has_docs = any(l in self.DOCS_EXACT for l in normalized) or any(
            any(sub in l for sub in self.DOCS_SUBSTRINGS) for l in normalized
        )
        has_question = any(l in self.QUESTION_EXACT for l in normalized) or any(
            any(sub in l for sub in self.QUESTION_SUBSTRINGS) for l in normalized
        )

        matches = {
            "bug": has_bug,
            "feature": has_feature,
            "documentation": has_docs,
            "question": has_question,
        }

        # Resolve via deterministic priority
        for cat in self.priority:
            if matches.get(cat, False):
                return cat

        # Labels exist, but belong to workflow/triage/SIG/module categories
        return "other"

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add canonical taxonomy columns and target encodings to DataFrame.

        Adds:
            - canonical_label: String category name
            - canonical_label_id: Integer target ID (0 to 4, -1 for unlabeled)
            - is_classified_category: Boolean flag (True for bug, feature, docs, question)

        Args:
            df: Ingested issues DataFrame with raw_labels column.

        Returns:
            DataFrame enriched with canonical taxonomy columns.
        """
        out_df = df.copy()

        if "raw_labels" in out_df.columns:
            out_df["canonical_label"] = out_df["raw_labels"].apply(self.map_labels)
        else:
            out_df["canonical_label"] = "unlabeled"

        out_df["canonical_label_id"] = out_df["canonical_label"].map(self.LABEL_TO_ID).astype("int64")
        out_df["is_classified_category"] = out_df["canonical_label"].isin(self.CORE_CATEGORIES).astype(bool)

        return out_df

    @classmethod
    def summarize_distribution(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute label distribution, class counts, and repository breakdown.

        Args:
            df: DataFrame containing canonical_label and repo_name.

        Returns:
            Dictionary with global and per-repository class statistics.
        """
        if df.empty or "canonical_label" not in df.columns:
            return {}

        total = len(df)
        counts = df["canonical_label"].value_counts().to_dict()

        distribution = []
        for cat, cid in cls.LABEL_TO_ID.items():
            cnt = counts.get(cat, 0)
            pct = round(100.0 * cnt / total, 2) if total > 0 else 0.0
            distribution.append({
                "label": cat,
                "label_id": cid,
                "count": cnt,
                "percent": pct,
                "is_core": cat in cls.CORE_CATEGORIES,
            })

        # Supervised candidate count
        core_count = int(df["is_classified_category"].sum()) if "is_classified_category" in df.columns else sum(
            counts.get(c, 0) for c in cls.CORE_CATEGORIES
        )

        # Per-repository breakdown
        per_repo = {}
        if "repo_name" in df.columns:
            for repo, sub in df.groupby("repo_name"):
                sub_counts = sub["canonical_label"].value_counts().to_dict()
                per_repo[str(repo)] = {
                    cat: sub_counts.get(cat, 0) for cat in cls.LABEL_TO_ID.keys()
                }

        return {
            "total_records": total,
            "core_candidates_count": core_count,
            "core_candidates_percent": round(100.0 * core_count / total, 2) if total else 0.0,
            "distribution": distribution,
            "per_repository": per_repo,
        }

    @classmethod
    def format_distribution_report(cls, summary: Dict[str, Any]) -> str:
        """Format distribution summary as Markdown."""
        lines = [
            "### Label Normalization & Taxonomy Summary",
            "",
            f"**Total Issues Audited:** {summary['total_records']:,}  ",
            f"**Core Supervised Candidates (`is_classified_category`):** {summary['core_candidates_count']:,} ({summary['core_candidates_percent']}%)  ",
            "",
            "#### Canonical Class Distribution",
            "",
            "| Class Name | Target ID | Category Type | Count | Percentage |",
            "|---|---|---|---|---|",
        ]

        for item in summary.get("distribution", []):
            ctype = "Core ML Target" if item["is_core"] else "Workflow / Auxiliary"
            lines.append(f"| `{item['label']}` | `{item['label_id']}` | {ctype} | {item['count']:,} | {item['percent']}% |")

        lines.extend([
            "",
            "#### Class Distribution Per Repository",
            "",
            "| Repository | Bug | Feature | Docs | Question | Other | Unlabeled |",
            "|---|---|---|---|---|---|---|",
        ])

        for repo, counts in summary.get("per_repository", {}).items():
            lines.append(
                f"| `{repo}` | {counts.get('bug', 0):,} | {counts.get('feature', 0):,} | "
                f"{counts.get('documentation', 0):,} | {counts.get('question', 0):,} | "
                f"{counts.get('other', 0):,} | {counts.get('unlabeled', 0):,} |"
            )

        return "\n".join(lines)


def main():
    """CLI entrypoint for label normalization."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Normalize GitHub issue labels into canonical taxonomy.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    from ml.preprocessing.loader import RawDataLoader
    from ml.preprocessing.text_cleaner import IssueTextCleaner
    from ml.preprocessing.timestamp_cleaner import TimestampCleaner
    from ml.preprocessing.missing_handler import MissingValueHandler
    from ml.preprocessing.filter import DataFilter

    # Load and preprocess
    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all()
    df_raw = data["issues"]

    text_cleaner = IssueTextCleaner()
    df_text = text_cleaner.clean_dataframe(df_raw)

    ts_cleaner = TimestampCleaner()
    df_ts = ts_cleaner.clean_dataframe(df_text)

    missing_handler = MissingValueHandler()
    df_missing = missing_handler.clean_dataframe(df_ts)

    data_filter = DataFilter()
    df_filtered, _ = data_filter.filter_dataframe(df_missing)

    # Label Normalizer
    normalizer = LabelNormalizer()
    df_norm = normalizer.normalize_dataframe(df_filtered)
    summary = LabelNormalizer.summarize_distribution(df_norm)

    print("=" * 65)
    print("OPENSOURCELENS — LABEL NORMALIZATION SUMMARY")
    print("=" * 65)
    print(f"Total Retained Issues:     {summary['total_records']:,}")
    print(f"Core Supervised Candidates:{summary['core_candidates_count']:,} ({summary['core_candidates_percent']}%)")
    print("-" * 65)
    for item in summary["distribution"]:
        print(f"  - {item['label']:14} (ID: {item['label_id']:2}): {item['count']:4} ({item['percent']:5.2f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
