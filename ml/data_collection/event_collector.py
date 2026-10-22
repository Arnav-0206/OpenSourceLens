"""Event Collector for OpenSourceLens.

Collects, validates, and standardizes GitHub issue timeline events according to
docs/0.3-data-definition.md and Step 1.1 API exploration findings.
"""

import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from ml.data_collection.github_client import GitHubClient, GitHubAPIError, NotFoundError
from ml.data_collection.repository_collector import RepositoryCollector


def extract_event_metadata(raw: Dict[str, Any], repository: str, issue_number: int) -> Dict[str, Any]:
    """Extract and standardize issue timeline event fields.

    Args:
        raw: Raw JSON dictionary from GitHub API issue events endpoint.
        repository: Repository identifier string (e.g. 'microsoft/vscode').
        issue_number: Parent issue number.

    Returns:
        Standardized event dictionary.
    """
    actor_info = raw.get("actor") or {}
    actor = actor_info.get("login") or "system"

    label_info = raw.get("label") or {}
    label_name = label_info.get("name") if isinstance(label_info, dict) else None

    assignee_info = raw.get("assignee") or {}
    assignee = assignee_info.get("login") if isinstance(assignee_info, dict) else None

    return {
        "event_id": raw.get("id"),
        "issue_number": issue_number,
        "repository": repository,
        "event": raw.get("event"),
        "actor": actor,
        "created_at": raw.get("created_at"),
        "commit_id": raw.get("commit_id"),
        "label_name": label_name,
        "assignee": assignee,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


class EventCollector:
    """Collector responsible for fetching and standardizing GitHub issue timeline events."""

    def __init__(self, client: Optional[GitHubClient] = None):
        """Initialize with an optional shared GitHubClient."""
        self.client = client or GitHubClient()

    def collect_events_page(
        self,
        repo_identifier: str,
        issue_number: int,
        page: int = 1,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch a single page of timeline events for a specific issue.

        Args:
            repo_identifier: 'owner/repo' string (e.g. 'microsoft/vscode').
            issue_number: Issue number within the repository.
            page: Page number (1-indexed).
            per_page: Results per page (up to 100).

        Returns:
            List of dictionaries, each containing 'event' and 'raw'.

        Raises:
            ValueError: On invalid input.
            NotFoundError: If the repository or issue is not found.
            GitHubAPIError: On API or network errors.
        """
        owner, repo = RepositoryCollector.parse_repo_identifier(repo_identifier)
        repo_full_name = f"{owner}/{repo}"
        endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}/events"

        params = {
            "page": max(1, page),
            "per_page": min(max(1, per_page), 100),
        }

        response = self.client.get(endpoint, params=params)
        raw_items = response.json()

        collected = []
        for item in raw_items:
            structured = extract_event_metadata(item, repo_full_name, issue_number)
            collected.append({
                "event": structured,
                "raw": item,
            })
        return collected

    def collect_all_events_for_issue(
        self,
        repo_identifier: str,
        issue_number: int,
        max_events: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all events for an issue across all pages.

        Args:
            repo_identifier: 'owner/repo' string (e.g. 'microsoft/vscode').
            issue_number: Issue number within the repository.
            max_events: Optional cap on the maximum number of events to collect.

        Returns:
            List of all collected event items.
        """
        all_events: List[Dict[str, Any]] = []
        page = 1
        per_page = 100

        while True:
            batch = self.collect_events_page(
                repo_identifier=repo_identifier,
                issue_number=issue_number,
                page=page,
                per_page=per_page,
            )

            if not batch:
                break

            all_events.extend(batch)

            if max_events and len(all_events) >= max_events:
                all_events = all_events[:max_events]
                break

            if len(batch) < per_page:
                break

            page += 1

        return all_events


def collect_events(
    repo_identifier: str,
    issue_number: int,
    page: int = 1,
    per_page: int = 30,
    client: Optional[GitHubClient] = None,
) -> List[Dict[str, Any]]:
    """Convenience helper to collect events for an issue."""
    collector = EventCollector(client=client)
    return collector.collect_events_page(repo_identifier, issue_number, page=page, per_page=per_page)


def main():
    """CLI entrypoint for standalone event collection."""
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Collect GitHub issue timeline events for OpenSourceLens.")
    parser.add_argument("repository", type=str, help="Repository in 'owner/repo' format (e.g., 'microsoft/vscode')")
    parser.add_argument("--issue", type=int, required=True, help="Issue number to collect events for")
    parser.add_argument("--all", action="store_true", help="Collect all events across all pages")
    parser.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    parser.add_argument("--per-page", type=int, default=10, help="Results per page (default: 10, max: 100)")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of events to retrieve")
    parser.add_argument("--raw", action="store_true", help="Output raw API JSON instead of structured metadata")
    args = parser.parse_args()

    collector = EventCollector()

    try:
        if args.all:
            results = collector.collect_all_events_for_issue(args.repository, args.issue, max_events=args.max)
        else:
            results = collector.collect_events_page(
                args.repository,
                args.issue,
                page=args.page,
                per_page=args.per_page,
            )

        output = [r["raw"] if args.raw else r["event"] for r in results]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n[Summary] Retrieved {len(results)} event(s) for {args.repository}#{args.issue}.", file=sys.stderr)
    except (ValueError, NotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except GitHubAPIError as err:
        print(f"GitHub API Error: {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
