"""Raw Data Ingestion and Normalization Module for OpenSourceLens.

Reads harvested JSON files from data/raw/ and parses them into unified,
well-typed Pandas DataFrames with schema normalization and flattening.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd


class RawDataLoader:
    """Ingests raw GitHub API JSON payloads into structured Pandas DataFrames."""

    REPOSITORY_COLUMNS = [
        "repo_name",
        "name",
        "owner",
        "description",
        "stars",
        "forks",
        "watchers",
        "open_issues_github",
        "language",
        "license",
        "created_at",
        "updated_at",
    ]

    ISSUE_COLUMNS = [
        "repo_name",
        "issue_id",
        "issue_number",
        "title",
        "body",
        "state",
        "state_reason",
        "author",
        "author_type",
        "author_association",
        "created_at",
        "updated_at",
        "closed_at",
        "comments_count",
        "reactions_count",
        "raw_labels",
        "labels_count",
    ]

    COMMENT_COLUMNS = [
        "repo_name",
        "issue_number",
        "comment_id",
        "author",
        "author_association",
        "created_at",
        "updated_at",
        "body",
        "reactions_count",
    ]

    EVENT_COLUMNS = [
        "repo_name",
        "issue_number",
        "event_id",
        "event_type",
        "actor",
        "created_at",
        "label_name",
        "assignee",
        "rename_from",
        "rename_to",
    ]

    def __init__(self, raw_dir: Union[str, Path] = "data/raw"):
        """Initialize loader with raw data base directory.

        Args:
            raw_dir: Path to raw data folder (defaults to data/raw).
        """
        self.raw_dir = Path(raw_dir)
        self.repos_dir = self.raw_dir / "repositories"
        self.issues_dir = self.raw_dir / "issues"
        self.comments_dir = self.raw_dir / "comments"
        self.events_dir = self.raw_dir / "events"

    @staticmethod
    def _safe_user_field(user_obj: Any, field: str) -> Optional[str]:
        """Safely extract field from user dictionary."""
        if isinstance(user_obj, dict):
            return user_obj.get(field)
        return None

    def load_repositories(self, repo_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load repository metadata into a DataFrame.

        Args:
            repo_filter: Optional list of repository identifiers (e.g. ['microsoft/vscode']).

        Returns:
            pd.DataFrame containing repository metadata.
        """
        records: List[Dict[str, Any]] = []

        if self.repos_dir.exists():
            for file_path in sorted(self.repos_dir.glob("*.json")):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    full_name = data.get("full_name") or file_path.stem.replace("__", "/")

                    if repo_filter and full_name not in repo_filter:
                        continue

                    lic = data.get("license")
                    license_name = lic.get("name") if isinstance(lic, dict) else None

                    records.append({
                        "repo_name": full_name,
                        "name": data.get("name"),
                        "owner": self._safe_user_field(data.get("owner"), "login"),
                        "description": data.get("description"),
                        "stars": data.get("stargazers_count", 0),
                        "forks": data.get("forks_count", 0),
                        "watchers": data.get("subscribers_count", 0),
                        "open_issues_github": data.get("open_issues_count", 0),
                        "language": data.get("language"),
                        "license": license_name,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    })
                except Exception:
                    continue

        df = pd.DataFrame(records, columns=self.REPOSITORY_COLUMNS)
        return df

    def load_issues(
        self,
        repo_filter: Optional[List[str]] = None,
        state_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load pure GitHub issues into a DataFrame.

        Args:
            repo_filter: Optional list of repository identifiers to include.
            state_filter: Optional state filter ('open' or 'closed').

        Returns:
            pd.DataFrame of flattened issues.
        """
        records: List[Dict[str, Any]] = []

        if self.issues_dir.exists():
            for file_path in sorted(self.issues_dir.glob("*.json")):
                stem = file_path.stem
                parts = stem.split("__issues_")
                repo_name = parts[0].replace("__", "/") if parts else "unknown"

                if repo_filter and repo_name not in repo_filter:
                    continue

                try:
                    issues_list = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                if not isinstance(issues_list, list):
                    continue

                for issue in issues_list:
                    state = issue.get("state")
                    if state_filter and state != state_filter:
                        continue

                    user = issue.get("user")
                    reactions = issue.get("reactions")
                    reactions_count = reactions.get("total_count", 0) if isinstance(reactions, dict) else 0

                    raw_labels: List[str] = []
                    for lbl in issue.get("labels", []):
                        if isinstance(lbl, dict) and lbl.get("name"):
                            raw_labels.append(lbl["name"])
                        elif isinstance(lbl, str):
                            raw_labels.append(lbl)

                    records.append({
                        "repo_name": repo_name,
                        "issue_id": issue.get("id"),
                        "issue_number": issue.get("number"),
                        "title": issue.get("title") or "",
                        "body": issue.get("body"),
                        "state": state,
                        "state_reason": issue.get("state_reason"),
                        "author": self._safe_user_field(user, "login"),
                        "author_type": self._safe_user_field(user, "type"),
                        "author_association": issue.get("author_association"),
                        "created_at": issue.get("created_at"),
                        "updated_at": issue.get("updated_at"),
                        "closed_at": issue.get("closed_at"),
                        "comments_count": issue.get("comments", 0),
                        "reactions_count": reactions_count,
                        "raw_labels": raw_labels,
                        "labels_count": len(raw_labels),
                    })

        df = pd.DataFrame(records, columns=self.ISSUE_COLUMNS)
        return df

    def load_comments(self, repo_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load discussion comments into a DataFrame.

        Args:
            repo_filter: Optional list of repository identifiers.

        Returns:
            pd.DataFrame of comments.
        """
        records: List[Dict[str, Any]] = []

        if self.comments_dir.exists():
            for file_path in sorted(self.comments_dir.glob("*.json")):
                stem = file_path.stem
                # Format: <owner>__<repo>__issue_<num>__comments.json
                parts = stem.split("__issue_")
                if len(parts) != 2:
                    continue

                repo_name = parts[0].replace("__", "/")
                if repo_filter and repo_name not in repo_filter:
                    continue

                issue_num_str = parts[1].split("__comments")[0]
                try:
                    issue_num = int(issue_num_str)
                except ValueError:
                    issue_num = None

                try:
                    comments_list = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                if not isinstance(comments_list, list):
                    continue

                for c in comments_list:
                    reactions = c.get("reactions")
                    reactions_count = reactions.get("total_count", 0) if isinstance(reactions, dict) else 0

                    records.append({
                        "repo_name": repo_name,
                        "issue_number": issue_num,
                        "comment_id": c.get("id"),
                        "author": self._safe_user_field(c.get("user"), "login"),
                        "author_association": c.get("author_association"),
                        "created_at": c.get("created_at"),
                        "updated_at": c.get("updated_at"),
                        "body": c.get("body"),
                        "reactions_count": reactions_count,
                    })

        df = pd.DataFrame(records, columns=self.COMMENT_COLUMNS)
        return df

    def load_events(self, repo_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """Load timeline events into a DataFrame.

        Args:
            repo_filter: Optional list of repository identifiers.

        Returns:
            pd.DataFrame of events.
        """
        records: List[Dict[str, Any]] = []

        if self.events_dir.exists():
            for file_path in sorted(self.events_dir.glob("*.json")):
                stem = file_path.stem
                # Format: <owner>__<repo>__issue_<num>__events.json
                parts = stem.split("__issue_")
                if len(parts) != 2:
                    continue

                repo_name = parts[0].replace("__", "/")
                if repo_filter and repo_name not in repo_filter:
                    continue

                issue_num_str = parts[1].split("__events")[0]
                try:
                    issue_num = int(issue_num_str)
                except ValueError:
                    issue_num = None

                try:
                    events_list = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                if not isinstance(events_list, list):
                    continue

                for ev in events_list:
                    lbl = ev.get("label")
                    label_name = lbl.get("name") if isinstance(lbl, dict) else None

                    rename = ev.get("rename")
                    rename_from = rename.get("from") if isinstance(rename, dict) else None
                    rename_to = rename.get("to") if isinstance(rename, dict) else None

                    records.append({
                        "repo_name": repo_name,
                        "issue_number": issue_num,
                        "event_id": ev.get("id"),
                        "event_type": ev.get("event"),
                        "actor": self._safe_user_field(ev.get("actor"), "login"),
                        "created_at": ev.get("created_at"),
                        "label_name": label_name,
                        "assignee": self._safe_user_field(ev.get("assignee"), "login"),
                        "rename_from": rename_from,
                        "rename_to": rename_to,
                    })

        df = pd.DataFrame(records, columns=self.EVENT_COLUMNS)
        return df

    def load_all(self, repo_filter: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load all four data categories into a dictionary of DataFrames.

        Args:
            repo_filter: Optional list of repository identifiers.

        Returns:
            Dictionary with keys 'repositories', 'issues', 'comments', 'events'.
        """
        return {
            "repositories": self.load_repositories(repo_filter=repo_filter),
            "issues": self.load_issues(repo_filter=repo_filter),
            "comments": self.load_comments(repo_filter=repo_filter),
            "events": self.load_events(repo_filter=repo_filter),
        }

    def summarize(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """Generate high-level summary of loaded DataFrames.

        Args:
            data: Optional dictionary of DataFrames (if omitted, load_all is called).

        Returns:
            Dictionary with shapes, memory usage in MB, and repo lists.
        """
        dfs = data or self.load_all()
        summary = {}

        for name, df in dfs.items():
            mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            repos_present = sorted(df["repo_name"].dropna().unique().tolist()) if "repo_name" in df.columns else []

            summary[name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "memory_mb": mem_mb,
                "repositories": repos_present,
            }

        return summary


def main():
    """CLI entrypoint for testing raw data loading."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Load and inspect raw GitHub datasets.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--repo", type=str, default=None, help="Filter by specific repository")
    args = parser.parse_args()

    filter_repos = [args.repo] if args.repo else None
    loader = RawDataLoader(raw_dir=args.data_dir)
    data = loader.load_all(repo_filter=filter_repos)
    summary = loader.summarize(data)

    print("=" * 65)
    print("OPENSOURCELENS — RAW DATA INGESTION SUMMARY")
    print("=" * 65)
    for table, s in summary.items():
        print(f"Table: {table.upper()}")
        print(f"  Rows:         {s['rows']:,}")
        print(f"  Columns:      {s['columns']} {s['column_names'][:5]}...")
        print(f"  Memory:       {s['memory_mb']} MB")
        print(f"  Repositories: {s['repositories']}")
        print("-" * 65)


if __name__ == "__main__":
    main()
