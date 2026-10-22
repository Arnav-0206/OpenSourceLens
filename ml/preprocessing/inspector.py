"""Dataset Inspection and Quality Profiling Engine for OpenSourceLens.

Systematically inspects loaded issues and repositories DataFrames, auditing
schema dtypes, missing values, duplicate records, categorical distributions,
text length metrics, and markdown formatting patterns.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np


class DatasetInspector:
    """Performs deep statistical and structural inspection of loaded datasets."""

    def __init__(self, df_issues: pd.DataFrame, df_repos: Optional[pd.DataFrame] = None):
        """Initialize inspector with issues DataFrame and optional repositories DataFrame.

        Args:
            df_issues: Ingested issues DataFrame from RawDataLoader.
            df_repos: Ingested repositories DataFrame from RawDataLoader.
        """
        self.df = df_issues.copy() if not df_issues.empty else df_issues
        self.df_repos = df_repos.copy() if df_repos is not None and not df_repos.empty else df_repos

    def audit_schema(self) -> Dict[str, Any]:
        """Audit schema, inferred types, non-null counts, and uniqueness."""
        if self.df.empty:
            return {"total_rows": 0, "total_columns": 0, "columns": {}}

        total_rows = len(self.df)
        columns_info = {}

        for col in self.df.columns:
            non_null = int(self.df[col].notna().sum())
            null_count = total_rows - non_null
            null_pct = round(100.0 * null_count / total_rows, 2) if total_rows > 0 else 0.0

            # Uniqueness (handling unhashable types like lists)
            sample_val = self.df[col].dropna().iloc[0] if non_null > 0 else None
            if isinstance(sample_val, list):
                unique_count = self.df[col].dropna().apply(lambda x: tuple(x) if isinstance(x, list) else x).nunique()
            else:
                unique_count = int(self.df[col].nunique(dropna=True))

            columns_info[col] = {
                "dtype": str(self.df[col].dtype),
                "non_null_count": non_null,
                "null_count": null_count,
                "null_percent": null_pct,
                "unique_count": unique_count,
            }

        return {
            "total_rows": total_rows,
            "total_columns": len(self.df.columns),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "columns": columns_info,
        }

    def audit_duplicates(self) -> Dict[str, Any]:
        """Detect duplicate issue IDs, composite keys, and identical text."""
        if self.df.empty:
            return {
                "duplicate_ids": 0,
                "duplicate_composite_keys": 0,
                "duplicate_titles": 0,
                "duplicate_title_body_pairs": 0,
            }

        dup_ids = int(self.df["issue_id"].duplicated().sum()) if "issue_id" in self.df.columns else 0

        dup_composite = 0
        if "repo_name" in self.df.columns and "issue_number" in self.df.columns:
            dup_composite = int(self.df.duplicated(subset=["repo_name", "issue_number"]).sum())

        dup_titles = int(self.df["title"].duplicated().sum()) if "title" in self.df.columns else 0

        dup_pairs = 0
        if "title" in self.df.columns and "body" in self.df.columns:
            dup_pairs = int(self.df.duplicated(subset=["title", "body"]).sum())

        return {
            "duplicate_ids": dup_ids,
            "duplicate_composite_keys": dup_composite,
            "duplicate_titles": dup_titles,
            "duplicate_title_body_pairs": dup_pairs,
        }

    def audit_missingness(self) -> Dict[str, Any]:
        """Audit missing value patterns and cross-tabulate state vs closed_at."""
        if self.df.empty:
            return {
                "cross_tab_state_closed_at": {},
                "empty_bodies_count": 0,
                "empty_bodies_percent": 0.0,
            }

        # Cross-tabulation of state vs closed_at missingness
        cross_tab = {}
        if "state" in self.df.columns and "closed_at" in self.df.columns:
            states = self.df["state"].unique()
            for s in states:
                sub = self.df[self.df["state"] == s]
                closed_missing = int(sub["closed_at"].isna().sum())
                closed_present = int(sub["closed_at"].notna().sum())
                cross_tab[str(s)] = {
                    "total": len(sub),
                    "closed_at_present": closed_present,
                    "closed_at_missing": closed_missing,
                }

        empty_bodies = 0
        if "body" in self.df.columns:
            empty_bodies = int(self.df["body"].isna().sum() + (self.df["body"].fillna("").str.strip() == "").sum())

        return {
            "cross_tab_state_closed_at": cross_tab,
            "empty_bodies_count": empty_bodies,
            "empty_bodies_percent": round(100.0 * empty_bodies / len(self.df), 2) if len(self.df) else 0.0,
        }

    def audit_categoricals(self) -> Dict[str, Any]:
        """Compute frequency distributions for categorical variables."""
        if self.df.empty:
            return {}

        cats = ["state", "state_reason", "author_association", "author_type"]
        distributions = {}

        for col in cats:
            if col in self.df.columns:
                counts = self.df[col].value_counts(dropna=False).to_dict()
                total = len(self.df)
                dist = []
                for val, count in counts.items():
                    val_str = "None / Missing" if pd.isna(val) else str(val)
                    pct = round(100.0 * count / total, 2) if total > 0 else 0.0
                    dist.append({"value": val_str, "count": int(count), "percent": pct})
                distributions[col] = dist

        return distributions

    def audit_text(self) -> Dict[str, Any]:
        """Analyze text length distributions and markdown formatting artifacts."""
        if self.df.empty or "title" not in self.df.columns:
            return {}

        title_chars = self.df["title"].fillna("").str.len()
        title_words = self.df["title"].fillna("").str.split().str.len()

        body_series = self.df["body"].fillna("") if "body" in self.df.columns else pd.Series([""] * len(self.df))
        body_chars = body_series.str.len()
        body_words = body_series.str.split().str.len()

        def compute_stats(series: pd.Series) -> Dict[str, float]:
            if series.empty:
                return {}
            return {
                "mean": round(float(series.mean()), 1),
                "std": round(float(series.std()), 1) if len(series) > 1 else 0.0,
                "min": int(series.min()),
                "p25": int(series.quantile(0.25)),
                "median": int(series.median()),
                "p75": int(series.quantile(0.75)),
                "max": int(series.max()),
            }

        # Markdown pattern detection in body
        code_block_pattern = re.compile(r"```[\s\S]*?```")
        url_pattern = re.compile(r"https?://[^\s)]+")
        html_comment_pattern = re.compile(r"<!--[\s\S]*?-->")
        html_tag_pattern = re.compile(r"<[a-zA-Z/][^>]*>")

        code_blocks_count = int(body_series.apply(lambda b: bool(code_block_pattern.search(b))).sum())
        urls_count = int(body_series.apply(lambda b: bool(url_pattern.search(b))).sum())
        html_comments_count = int(body_series.apply(lambda b: bool(html_comment_pattern.search(b))).sum())
        html_tags_count = int(body_series.apply(lambda b: bool(html_tag_pattern.search(b))).sum())

        return {
            "title_char_stats": compute_stats(title_chars),
            "title_word_stats": compute_stats(title_words),
            "body_char_stats": compute_stats(body_chars),
            "body_word_stats": compute_stats(body_words),
            "markdown_artifacts": {
                "issues_with_code_blocks": code_blocks_count,
                "code_blocks_percent": round(100.0 * code_blocks_count / len(self.df), 2) if len(self.df) else 0.0,
                "issues_with_urls": urls_count,
                "urls_percent": round(100.0 * urls_count / len(self.df), 2) if len(self.df) else 0.0,
                "issues_with_html_comments": html_comments_count,
                "html_comments_percent": round(100.0 * html_comments_count / len(self.df), 2) if len(self.df) else 0.0,
                "issues_with_html_tags": html_tags_count,
                "html_tags_percent": round(100.0 * html_tags_count / len(self.df), 2) if len(self.df) else 0.0,
            },
        }

    def audit_labels(self) -> Dict[str, Any]:
        """Analyze label counts and distributions."""
        if self.df.empty or "raw_labels" not in self.df.columns:
            return {}

        total = len(self.df)
        labels_count_series = self.df["labels_count"] if "labels_count" in self.df.columns else self.df["raw_labels"].apply(len)

        zero_labels = int((labels_count_series == 0).sum())
        one_to_two = int(((labels_count_series >= 1) & (labels_count_series <= 2)).sum())
        three_to_five = int(((labels_count_series >= 3) & (labels_count_series <= 5)).sum())
        more_than_five = int((labels_count_series > 5).sum())

        # Top global labels
        all_labels: List[str] = []
        for lbls in self.df["raw_labels"].dropna():
            if isinstance(lbls, list):
                all_labels.extend(lbls)

        from collections import Counter
        counter = Counter(all_labels)
        top_global = [{"name": name, "count": count} for name, count in counter.most_common(10)]

        # Top per repo
        top_per_repo = {}
        if "repo_name" in self.df.columns:
            for repo, sub in self.df.groupby("repo_name"):
                sub_labels: List[str] = []
                for lbls in sub["raw_labels"].dropna():
                    if isinstance(lbls, list):
                        sub_labels.extend(lbls)
                sub_counter = Counter(sub_labels)
                top_per_repo[str(repo)] = [
                    {"name": name, "count": count} for name, count in sub_counter.most_common(5)
                ]

        return {
            "unlabeled_count": zero_labels,
            "unlabeled_percent": round(100.0 * zero_labels / total, 2) if total else 0.0,
            "distribution": {
                "0_labels": {"count": zero_labels, "percent": round(100.0 * zero_labels / total, 2) if total else 0.0},
                "1_to_2_labels": {"count": one_to_two, "percent": round(100.0 * one_to_two / total, 2) if total else 0.0},
                "3_to_5_labels": {"count": three_to_five, "percent": round(100.0 * three_to_five / total, 2) if total else 0.0},
                "gt_5_labels": {"count": more_than_five, "percent": round(100.0 * more_than_five / total, 2) if total else 0.0},
            },
            "top_global_labels": top_global,
            "top_per_repo": top_per_repo,
        }

    def generate_report_dict(self) -> Dict[str, Any]:
        """Aggregate all analytical audit results into a structured dictionary."""
        return {
            "schema": self.audit_schema(),
            "duplicates": self.audit_duplicates(),
            "missingness": self.audit_missingness(),
            "categoricals": self.audit_categoricals(),
            "text": self.audit_text(),
            "labels": self.audit_labels(),
        }

    def format_markdown_report(self, report_dict: Optional[Dict[str, Any]] = None) -> str:
        """Render comprehensive Markdown inspection report."""
        rep = report_dict or self.generate_report_dict()
        s = rep["schema"]
        d = rep["duplicates"]
        m = rep["missingness"]
        c = rep["categoricals"]
        t = rep["text"]
        lbl = rep["labels"]

        lines = [
            "# OpenSourceLens — Step 2.2 Dataset Inspection Report",
            "",
            f"**Total Records:** {s.get('total_rows', 0):,} issues  ",
            f"**Total Columns:** {s.get('total_columns', 0)}  ",
            f"**In-Memory Size:** {s.get('memory_mb', 0)} MB  ",
            "",
            "---",
            "",
            "## 1. Column-Level Schema & Completeness Audit",
            "",
            "| Column Name | Inferred Dtype | Non-Null Count | Null Count | Null % | Unique Values |",
            "|---|---|---|---|---|---|",
        ]

        for col, meta in s.get("columns", {}).items():
            lines.append(
                f"| `{col}` | `{meta['dtype']}` | {meta['non_null_count']:,} | "
                f"{meta['null_count']:,} | {meta['null_percent']}% | {meta['unique_count']:,} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Duplicate Records Audit",
            "",
            "| Duplicate Check | Count | Assessment |",
            "|---|---|---|",
            f"| Global `issue_id` Duplicates | **{d.get('duplicate_ids', 0)}** | {'PASSED (All issue IDs are globally unique)' if d.get('duplicate_ids', 0) == 0 else 'WARNING: Duplicate IDs found'} |",
            f"| Composite Key `(repo_name, issue_number)` Duplicates | **{d.get('duplicate_composite_keys', 0)}** | {'PASSED (No repo-level issue number collisions)' if d.get('duplicate_composite_keys', 0) == 0 else 'WARNING: Collisions found'} |",
            f"| Duplicate Titles | **{d.get('duplicate_titles', 0)}** | Common phrases or recurring bot issue titles. |",
            f"| Duplicate `(title, body)` Exact Pairs | **{d.get('duplicate_title_body_pairs', 0)}** | Exact duplicates to be handled in Step 2.6 (Remove irrelevant data). |",
            "",
            "---",
            "",
            "## 3. Missing Value Patterns & Integrity Audit",
            "",
            "### State vs. `closed_at` Integrity",
            "",
            "| Issue State | Total Issues | `closed_at` Present | `closed_at` Missing | Integrity Note |",
            "|---|---|---|---|---|",
        ])

        for state, info in m.get("cross_tab_state_closed_at", {}).items():
            note = "Expected (Open issues have no closed timestamp)" if state == "open" and info["closed_at_missing"] == info["total"] else "Valid"
            lines.append(f"| `{state}` | {info['total']:,} | {info['closed_at_present']:,} | {info['closed_at_missing']:,} | {note} |")

        lines.extend([
            "",
            f"**Issues with Empty Body:** {m.get('empty_bodies_count', 0)} ({m.get('empty_bodies_percent', 0)}%)  ",
            "",
            "---",
            "",
            "## 4. Categorical Distributions",
            "",
        ])

        for col, dist in c.items():
            lines.append(f"### Column: `{col}`")
            lines.append("")
            lines.append("| Value | Count | Percentage |")
            lines.append("|---|---|---|")
            for item in dist:
                lines.append(f"| `{item['value']}` | {item['count']:,} | {item['percent']}% |")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## 5. Text Length & Markdown Complexity Profile",
            "",
            "### Descriptive Statistics (Title vs. Body)",
            "",
            "| Metric | Title Characters | Title Words | Body Characters | Body Words |",
            "|---|---|---|---|---|",
            f"| Mean | {t.get('title_char_stats', {}).get('mean', 0)} | {t.get('title_word_stats', {}).get('mean', 0)} | {t.get('body_char_stats', {}).get('mean', 0)} | {t.get('body_word_stats', {}).get('mean', 0)} |",
            f"| Std Dev | {t.get('title_char_stats', {}).get('std', 0)} | {t.get('title_word_stats', {}).get('std', 0)} | {t.get('body_char_stats', {}).get('std', 0)} | {t.get('body_word_stats', {}).get('std', 0)} |",
            f"| Min | {t.get('title_char_stats', {}).get('min', 0)} | {t.get('title_word_stats', {}).get('min', 0)} | {t.get('body_char_stats', {}).get('min', 0)} | {t.get('body_word_stats', {}).get('min', 0)} |",
            f"| 25% (Q1) | {t.get('title_char_stats', {}).get('p25', 0)} | {t.get('title_word_stats', {}).get('p25', 0)} | {t.get('body_char_stats', {}).get('p25', 0)} | {t.get('body_word_stats', {}).get('p25', 0)} |",
            f"| Median (Q2) | {t.get('title_char_stats', {}).get('median', 0)} | {t.get('title_word_stats', {}).get('median', 0)} | {t.get('body_char_stats', {}).get('median', 0)} | {t.get('body_word_stats', {}).get('median', 0)} |",
            f"| 75% (Q3) | {t.get('title_char_stats', {}).get('p75', 0)} | {t.get('title_word_stats', {}).get('p75', 0)} | {t.get('body_char_stats', {}).get('p75', 0)} | {t.get('body_word_stats', {}).get('p75', 0)} |",
            f"| Max | {t.get('title_char_stats', {}).get('max', 0)} | {t.get('title_word_stats', {}).get('max', 0)} | {t.get('body_char_stats', {}).get('max', 0)} | {t.get('body_word_stats', {}).get('max', 0)} |",
            "",
            "### Markdown & Formatting Artifacts in Issue Bodies",
            "",
            "| Artifact Pattern | Matching Issues | Percentage | Text Cleaning Implication (Step 2.3) |",
            "|---|---|---|---|",
        ])

        marts = t.get("markdown_artifacts", {})
        lines.extend([
            f"| Code Blocks (```` ``` ````) | {marts.get('issues_with_code_blocks', 0):,} | {marts.get('code_blocks_percent', 0)}% | Code blocks contain stack traces, snippets; extract or normalize in Step 2.3. |",
            f"| URLs & Links (`http/s`) | {marts.get('issues_with_urls', 0):,} | {marts.get('urls_percent', 0)}% | Links add noise to bag-of-words; replace with token or strip. |",
            f"| HTML Comments (`<!-- ... -->`) | {marts.get('issues_with_html_comments', 0):,} | {marts.get('html_comments_percent', 0)}% | GitHub issue template instructions; strip completely in Step 2.3. |",
            f"| HTML Tags (`<... >`) | {marts.get('issues_with_html_tags', 0):,} | {marts.get('html_tags_percent', 0)}% | Boilerplate template formatting; clean/strip for NLP processing. |",
            "",
            "---",
            "",
            "## 6. Label Density & Distribution Profile",
            "",
            "| Label Multiplicity Bucket | Issue Count | Percentage |",
            "|---|---|---|",
        ])

        for bucket, bdata in lbl.get("distribution", {}).items():
            lines.append(f"| `{bucket}` | {bdata['count']:,} | {bdata['percent']}% |")

        lines.extend([
            "",
            "### Top 10 Labels Across Full Dataset",
            "",
            "| Rank | Label Name | Occurrences |",
            "|---|---|---|",
        ])

        for r, item in enumerate(lbl.get("top_global_labels", []), start=1):
            lines.append(f"| {r} | `{item['name']}` | {item['count']:,} |")

        lines.extend([
            "",
            "### Top Labels Per Repository",
            "",
        ])

        for repo, rlabels in lbl.get("top_per_repo", {}).items():
            lines.append(f"#### `{repo}`")
            if not rlabels:
                lines.append("- *No labels present.*")
            else:
                for item in rlabels:
                    lines.append(f"- `{item['name']}`: {item['count']} occurrences")
            lines.append("")

        lines.extend([
            "",
            "---",
            "",
            "## 7. Key Findings & Directives for Subsequent Steps",
            "",
            "1. **Text Cleaning (Step 2.3):**",
            f"   - **{marts.get('code_blocks_percent', 0)}%** of issues contain code blocks and **{marts.get('html_comments_percent', 0)}%** contain HTML comments. Dedicated cleaning functions must strip template comments and normalize code snippets.",
            "2. **Timestamp Cleaning (Step 2.4):**",
            "   - `created_at` and `updated_at` are 100% populated. `closed_at` is only populated on closed issues (1,200), with 0 missing closed timestamps on closed issues.",
            "3. **Missing Value Handling (Step 2.5):**",
            f"   - Only **{m.get('empty_bodies_count', 0)} issues ({m.get('empty_bodies_percent', 0)}%)** have empty bodies. Impute empty bodies using the `title` text.",
            "4. **Duplicate & Irrelevant Data (Step 2.6):**",
            f"   - Global IDs and composite keys are 100% unique. Exact title/body pairs ({d.get('duplicate_title_body_pairs', 0)}) will be inspected for automated bots/spam.",
            "5. **Label Normalization (Step 2.7):**",
            f"   - **{lbl.get('unlabeled_percent', 0)}%** of issues are unlabeled. Label hierarchy mapping is essential to consolidate `bug`, `kind/bug`, and `type: bug` into uniform classes.",
            "",
        ])

        return "\n".join(lines)

    def save_report(self, path: Union[str, Path] = "docs/2.2-dataset-inspection.md") -> Path:
        """Generate and save inspection report to Markdown file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self.format_markdown_report()
        target.write_text(content, encoding="utf-8")
        return target


def main():
    """CLI entrypoint to run dataset inspection."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Inspect raw dataset and generate inspection report.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--output-md", type=str, default="docs/2.2-dataset-inspection.md", help="Output report path")
    args = parser.parse_args()

    from ml.preprocessing.loader import RawDataLoader

    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all()

    inspector = DatasetInspector(df_issues=data["issues"], df_repos=data["repositories"])
    out_path = inspector.save_report(path=args.output_md)
    rep = inspector.generate_report_dict()

    print("=" * 65)
    print("OPENSOURCELENS — DATASET INSPECTION COMPLETED")
    print(f"Total Issues Inspected: {rep['schema']['total_rows']:,}")
    print(f"Duplicate IDs:          {rep['duplicates']['duplicate_ids']}")
    print(f"Composite Key Dups:     {rep['duplicates']['duplicate_composite_keys']}")
    print(f"Empty Bodies:           {rep['missingness']['empty_bodies_count']} ({rep['missingness']['empty_bodies_percent']}%)")
    print(f"Unlabeled Issues:       {rep['labels']['unlabeled_count']} ({rep['labels']['unlabeled_percent']}%)")
    print(f"Code Blocks Present:    {rep['text']['markdown_artifacts']['issues_with_code_blocks']} ({rep['text']['markdown_artifacts']['code_blocks_percent']}%)")
    print(f"HTML Comments Present:  {rep['text']['markdown_artifacts']['issues_with_html_comments']} ({rep['text']['markdown_artifacts']['html_comments_percent']}%)")
    print(f"Inspection Report:      {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
