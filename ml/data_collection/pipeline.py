"""Data Collection Pipeline Orchestrator for OpenSourceLens.

Orchestrates the harvesting of repository metadata, open/closed pure issues,
discussion comments, and timeline events across the selected candidate repositories
and saves raw API payloads into data/raw/.
"""

import sys
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from ml.data_collection.github_client import GitHubClient, GitHubAPIError, NotFoundError
from ml.data_collection.repository_collector import RepositoryCollector
from ml.data_collection.issue_collector import IssueCollector
from ml.data_collection.comment_collector import CommentCollector
from ml.data_collection.event_collector import EventCollector
from ml.data_collection.storage import RawDataStorage

DEFAULT_CANDIDATE_REPOSITORIES = [
    "microsoft/vscode",
    "pytorch/pytorch",
    "kubernetes/kubernetes",
    "huggingface/transformers",
]


class CollectionPipeline:
    """Orchestrates end-to-end data collection across candidate repositories."""

    def __init__(
        self,
        client: Optional[GitHubClient] = None,
        storage: Optional[RawDataStorage] = None,
        issues_per_repo: int = 600,
        sample_details_count: int = 10,
    ):
        """Initialize collection pipeline.

        Args:
            client: Shared GitHubClient instance.
            storage: Shared RawDataStorage instance.
            issues_per_repo: Total issues to collect per repository (split between open and closed).
            sample_details_count: Number of issues per repo to fetch comments and events for.
        """
        self.client = client or GitHubClient()
        self.storage = storage or RawDataStorage()
        self.issues_per_repo = issues_per_repo
        self.sample_details_count = sample_details_count

        self.repo_collector = RepositoryCollector(client=self.client)
        self.issue_collector = IssueCollector(client=self.client)
        self.comment_collector = CommentCollector(client=self.client)
        self.event_collector = EventCollector(client=self.client)

    def log(self, message: str) -> None:
        """Log pipeline status with UTC timestamp."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts} UTC] {message}")

    def collect_repository_metadata(self, repo_identifier: str) -> Dict[str, Any]:
        """Harvest and persist repository metadata."""
        self.log(f"Fetching repository metadata for '{repo_identifier}'...")
        result = self.repo_collector.collect(repo_identifier)
        saved_path = self.storage.save_repository(repo_identifier, result["raw"])
        self.log(f"  -> Saved metadata to: {saved_path}")
        return result["metadata"]

    def collect_issues(
        self,
        repo_identifier: str,
        target_count: int,
    ) -> Dict[str, List[int]]:
        """Harvest and persist balanced open and closed pure issues.

        Args:
            repo_identifier: 'owner/repo' string.
            target_count: Total issues to harvest for this repo.

        Returns:
            Dictionary with 'open_numbers' and 'closed_numbers' collected.
        """
        half = target_count // 2
        states = [("closed", half), ("open", half)]
        issue_numbers: Dict[str, List[int]] = {"closed": [], "open": []}

        for state, limit in states:
            self.log(f"Streaming {limit} '{state}' pure issues for '{repo_identifier}'...")
            raw_items = []
            numbers = []

            stream = self.issue_collector.collect_issues_stream(
                repo_identifier,
                state=state,
                max_issues=limit,
                per_page=100,
                sort="created",
                direction="desc",
                filter_pull_requests=True,
            )

            for item in stream:
                raw_items.append(item["raw"])
                numbers.append(item["issue"]["issue_number"])

            saved_path = self.storage.save_issues(repo_identifier, raw_items, batch_id=f"{state}_{len(raw_items)}")
            issue_numbers[state] = numbers
            self.log(f"  -> Saved {len(raw_items)} '{state}' issues to: {saved_path}")

        return issue_numbers

    def collect_discussions_and_events(
        self,
        repo_identifier: str,
        sample_numbers: List[int],
    ) -> Dict[str, int]:
        """Harvest comments and events for a sample of issues."""
        counts = {"comments": 0, "events": 0}
        if not sample_numbers:
            return counts

        self.log(f"Harvesting comments and events for {len(sample_numbers)} sample issues in '{repo_identifier}'...")

        for num in sample_numbers:
            # Comments
            comments_batch = self.comment_collector.collect_comments_page(
                repo_identifier,
                issue_number=num,
                per_page=10,
            )
            if comments_batch:
                raw_comments = [c["raw"] for c in comments_batch]
                self.storage.save_comments(repo_identifier, num, raw_comments)
                counts["comments"] += len(raw_comments)

            # Events
            events_batch = self.event_collector.collect_events_page(
                repo_identifier,
                issue_number=num,
                per_page=10,
            )
            if events_batch:
                raw_events = [e["raw"] for e in events_batch]
                self.storage.save_events(repo_identifier, num, raw_events)
                counts["events"] += len(raw_events)

        self.log(f"  -> Saved {counts['comments']} comments and {counts['events']} events for '{repo_identifier}'.")
        return counts

    def run(self, repositories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute full collection pipeline across target repositories.

        Args:
            repositories: Optional list of repositories (defaults to candidate repos).

        Returns:
            Execution summary dictionary.
        """
        targets = repositories or DEFAULT_CANDIDATE_REPOSITORIES
        start_time = time.time()
        initial_quota = self.client.get_quota_status()

        self.log("=" * 65)
        self.log("STARTING OPENSOURCELENS DATA COLLECTION PIPELINE")
        self.log(f"Target Repositories: {targets}")
        self.log(f"Target Issues Per Repo: {self.issues_per_repo} ({self.issues_per_repo // 2} closed, {self.issues_per_repo // 2} open)")
        self.log(f"Initial Rate Limit: {initial_quota['remaining']} / {initial_quota['limit']} requests remaining")
        self.log("=" * 65)

        summary: Dict[str, Any] = {
            "repositories": {},
            "total_issues_collected": 0,
            "total_comments_collected": 0,
            "total_events_collected": 0,
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
        }

        for repo in targets:
            self.log("-" * 65)
            self.log(f"PROCESSING REPOSITORY: {repo}")
            self.log("-" * 65)

            try:
                # 1. Metadata
                meta = self.collect_repository_metadata(repo)

                # 2. Issues
                numbers = self.collect_issues(repo, target_count=self.issues_per_repo)
                closed_count = len(numbers.get("closed", []))
                open_count = len(numbers.get("open", []))
                total_repo_issues = closed_count + open_count

                # 3. Comments and Events on sample closed issues
                sample_candidates = numbers.get("closed", [])[:self.sample_details_count]
                details_counts = self.collect_discussions_and_events(repo, sample_candidates)

                summary["repositories"][repo] = {
                    "stars": meta.get("stars"),
                    "language": meta.get("language"),
                    "closed_issues": closed_count,
                    "open_issues": open_count,
                    "total_issues": total_repo_issues,
                    "comments_sampled": details_counts["comments"],
                    "events_sampled": details_counts["events"],
                }

                summary["total_issues_collected"] += total_repo_issues
                summary["total_comments_collected"] += details_counts["comments"]
                summary["total_events_collected"] += details_counts["events"]

            except (ValueError, NotFoundError, GitHubAPIError) as err:
                self.log(f"ERROR processing repository '{repo}': {err}")
                summary["repositories"][repo] = {"error": str(err)}

        elapsed = round(time.time() - start_time, 2)
        final_quota = self.client.get_quota_status()
        quota_used = initial_quota["remaining"] - final_quota["remaining"]
        storage_summary = self.storage.get_storage_summary()

        summary["end_time_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = elapsed
        summary["quota_used"] = quota_used
        summary["quota_remaining"] = final_quota["remaining"]
        summary["storage"] = storage_summary

        self.log("=" * 65)
        self.log("COLLECTION PIPELINE COMPLETED")
        self.log(f"Total Issues Harvested:   {summary['total_issues_collected']}")
        self.log(f"Total Raw Files Stored:   {storage_summary['total_files']} files ({storage_summary['total_size_mb']} MB)")
        self.log(f"API Quota Consumed:       {quota_used} requests (Remaining: {final_quota['remaining']})")
        self.log(f"Elapsed Time:             {elapsed} seconds")
        self.log("=" * 65)

        return summary


def main():
    """CLI entrypoint for running the collection pipeline."""
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Run OpenSourceLens data collection pipeline.")
    parser.add_argument("--issues-per-repo", type=int, default=600, help="Total issues to collect per repo (default: 600)")
    parser.add_argument("--sample", action="store_true", help="Run fast verification sample (10 issues per repo)")
    parser.add_argument("--repo", type=str, default=None, help="Target specific repository only (e.g. 'microsoft/vscode')")
    parser.add_argument("--sample-details", type=int, default=10, help="Number of issues per repo to fetch comments/events for (default: 10)")
    args = parser.parse_args()

    count = 10 if args.sample else args.issues_per_repo
    details = 2 if args.sample else args.sample_details
    targets = [args.repo] if args.repo else None

    pipeline = CollectionPipeline(issues_per_repo=count, sample_details_count=details)
    pipeline.run(repositories=targets)


if __name__ == "__main__":
    main()
