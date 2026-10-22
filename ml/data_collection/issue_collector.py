"""Issue Collector for OpenSourceLens.

Collects, validates, and standardizes GitHub issue data according to
docs/0.3-data-definition.md and filters out Pull Requests based on Step 1.1 findings.
"""

import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Generator

from ml.data_collection.github_client import GitHubClient, GitHubAPIError, NotFoundError
from ml.data_collection.repository_collector import RepositoryCollector
from ml.data_collection.paginator import GitHubPaginator


def is_pull_request(item: Dict[str, Any]) -> bool:
    """Determine if a GitHub API issue payload represents a Pull Request.

    In the GitHub REST API, the /issues endpoint returns both pure issues
    and pull requests. Pull requests contain a top-level 'pull_request' dictionary.
    """
    return "pull_request" in item and item["pull_request"] is not None


def extract_issue_metadata(raw: Dict[str, Any], repository: str) -> Dict[str, Any]:
    """Extract and standardize issue fields according to docs/0.3-data-definition.md.

    Args:
        raw: Raw JSON dictionary returned from GitHub API issue endpoint.
        repository: Repository identifier string (e.g. 'microsoft/vscode').

    Returns:
        Standardized issue metadata dictionary.
    """
    # Extract author login safely (handle deleted accounts where user is null)
    user_info = raw.get("user") or {}
    author = user_info.get("login") or "ghost"

    # Extract label names
    raw_labels = raw.get("labels", [])
    labels = []
    for label in raw_labels:
        if isinstance(label, dict) and "name" in label:
            labels.append(label["name"])
        elif isinstance(label, str):
            labels.append(label)

    return {
        "issue_id": raw.get("id"),
        "issue_number": raw.get("number"),
        "repository": repository,
        "title": raw.get("title") or "",
        "body": raw.get("body") or "",
        "state": raw.get("state"),
        "author": author,
        "labels": labels,
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "closed_at": raw.get("closed_at"),
        "comments_count": raw.get("comments", 0),
        "is_pull_request": is_pull_request(raw),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


class IssueCollector:
    """Collector responsible for fetching and standardizing GitHub issues."""

    def __init__(self, client: Optional[GitHubClient] = None):
        """Initialize with an optional shared GitHubClient."""
        self.client = client or GitHubClient()

    def collect_issue(
        self,
        repo_identifier: str,
        issue_number: int,
        allow_pull_request: bool = False,
    ) -> Dict[str, Any]:
        """Fetch and standardize a single GitHub issue by its number.

        Args:
            repo_identifier: 'owner/repo' string (e.g. 'microsoft/vscode').
            issue_number: Issue number within the repository.
            allow_pull_request: If False, raises ValueError if the number corresponds to a PR.

        Returns:
            Dictionary containing:
              - 'issue': Standardized metadata dictionary
              - 'raw': Complete raw JSON response payload from GitHub API

        Raises:
            ValueError: On invalid input or if the item is a PR when allow_pull_request is False.
            NotFoundError: If the issue does not exist.
            GitHubAPIError: On rate limit, authentication, or network failures.
        """
        owner, repo = RepositoryCollector.parse_repo_identifier(repo_identifier)
        repo_full_name = f"{owner}/{repo}"
        endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}"

        response = self.client.get(endpoint)
        raw_data = response.json()

        if not allow_pull_request and is_pull_request(raw_data):
            raise ValueError(
                f"Item #{issue_number} in '{repo_full_name}' is a Pull Request, not a pure issue. "
                f"Set allow_pull_request=True if you wish to retrieve it."
            )

        structured_issue = extract_issue_metadata(raw_data, repo_full_name)
        return {
            "issue": structured_issue,
            "raw": raw_data,
        }

    def collect_issues_page(
        self,
        repo_identifier: str,
        state: str = "all",
        page: int = 1,
        per_page: int = 100,
        sort: str = "created",
        direction: str = "asc",
        filter_pull_requests: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fetch a single page of issues from a repository.

        Args:
            repo_identifier: 'owner/repo' string (e.g. 'microsoft/vscode').
            state: Filter by state: 'open', 'closed', or 'all'.
            page: Page number (1-indexed).
            per_page: Results per page (up to 100).
            sort: Sort field ('created', 'updated', 'comments').
            direction: Sort direction ('asc' or 'desc').
            filter_pull_requests: If True, excludes any items that are pull requests.

        Returns:
            List of dictionaries, each containing 'issue' and 'raw'.

        Raises:
            ValueError: On invalid input.
            NotFoundError: If repository is not found.
            GitHubAPIError: On API or network errors.
        """
        owner, repo = RepositoryCollector.parse_repo_identifier(repo_identifier)
        repo_full_name = f"{owner}/{repo}"
        endpoint = f"/repos/{owner}/{repo}/issues"

        params = {
            "state": state,
            "page": page,
            "per_page": min(max(1, per_page), 100),
            "sort": sort,
            "direction": direction,
        }

        response = self.client.get(endpoint, params=params)
        raw_items = response.json()

        collected = []
        for item in raw_items:
            if filter_pull_requests and is_pull_request(item):
                continue
            structured_issue = extract_issue_metadata(item, repo_full_name)
            collected.append({
                "issue": structured_issue,
                "raw": item,
            })

        return collected

    def collect_issues_stream(
        self,
        repo_identifier: str,
        state: str = "all",
        max_issues: Optional[int] = None,
        per_page: int = 100,
        sort: str = "created",
        direction: str = "asc",
        filter_pull_requests: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream pure issues across multiple paginated pages.

        Args:
            repo_identifier: 'owner/repo' string (e.g. 'microsoft/vscode').
            state: Filter by state: 'open', 'closed', or 'all'.
            max_issues: Maximum number of pure issues to yield.
            per_page: Results per page (up to 100).
            sort: Sort field ('created', 'updated', 'comments').
            direction: Sort direction ('asc' or 'desc').
            filter_pull_requests: If True, excludes any items that are pull requests.

        Yields:
            Dictionaries containing 'issue' (standardized) and 'raw' (original JSON).
        """
        owner, repo = RepositoryCollector.parse_repo_identifier(repo_identifier)
        repo_full_name = f"{owner}/{repo}"
        endpoint = f"/repos/{owner}/{repo}/issues"

        params = {
            "state": state,
            "per_page": min(max(1, per_page), 100),
            "sort": sort,
            "direction": direction,
        }

        paginator = GitHubPaginator(client=self.client)
        yielded = 0

        for raw_item in paginator.paginate_items(endpoint, params=params):
            if filter_pull_requests and is_pull_request(raw_item):
                continue

            structured_issue = extract_issue_metadata(raw_item, repo_full_name)
            yield {
                "issue": structured_issue,
                "raw": raw_item,
            }
            yielded += 1
            if max_issues and yielded >= max_issues:
                break


def collect_issue(repo_identifier: str, issue_number: int, client: Optional[GitHubClient] = None) -> Dict[str, Any]:
    """Convenience helper to collect a single issue."""
    collector = IssueCollector(client=client)
    return collector.collect_issue(repo_identifier, issue_number)


def collect_issues_page(
    repo_identifier: str,
    page: int = 1,
    per_page: int = 30,
    state: str = "all",
    client: Optional[GitHubClient] = None,
) -> List[Dict[str, Any]]:
    """Convenience helper to collect a page of issues."""
    collector = IssueCollector(client=client)
    return collector.collect_issues_page(repo_identifier, state=state, page=page, per_page=per_page)


def collect_issues_stream(
    repo_identifier: str,
    state: str = "all",
    max_issues: Optional[int] = None,
    per_page: int = 100,
    client: Optional[GitHubClient] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Convenience helper to stream issues across pages."""
    collector = IssueCollector(client=client)
    return collector.collect_issues_stream(repo_identifier, state=state, max_issues=max_issues, per_page=per_page)


def main():
    """CLI entrypoint for standalone issue collection."""
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Collect GitHub issues for OpenSourceLens.")
    parser.add_argument("repository", type=str, help="Repository in 'owner/repo' format (e.g., 'microsoft/vscode')")
    parser.add_argument("--issue", type=int, help="Specific issue number to collect")
    parser.add_argument("--max-issues", type=int, default=None, help="Stream up to N issues across multiple pages")
    parser.add_argument("--page", type=int, default=1, help="Page number for single page query (default: 1)")
    parser.add_argument("--per-page", type=int, default=10, help="Results per page (default: 10, max: 100)")
    parser.add_argument("--state", type=str, default="all", choices=["open", "closed", "all"], help="Issue state filter (default: all)")
    parser.add_argument("--raw", action="store_true", help="Output raw API JSON instead of structured metadata")
    args = parser.parse_args()

    collector = IssueCollector()

    try:
        if args.issue:
            result = collector.collect_issue(args.repository, args.issue)
            output = result["raw"] if args.raw else result["issue"]
            print(json.dumps(output, indent=2, ensure_ascii=False))
        elif args.max_issues:
            count = 0
            for item in collector.collect_issues_stream(
                args.repository,
                state=args.state,
                max_issues=args.max_issues,
                per_page=args.per_page,
            ):
                output = item["raw"] if args.raw else item["issue"]
                print(json.dumps(output, ensure_ascii=False))
                count += 1
            print(f"\n[Summary] Streamed {count} pure issues across pages.", file=sys.stderr)
        else:
            results = collector.collect_issues_page(
                args.repository,
                state=args.state,
                page=args.page,
                per_page=args.per_page,
            )
            output = [r["raw"] if args.raw else r["issue"] for r in results]
            print(json.dumps(output, indent=2, ensure_ascii=False))
            print(f"\n[Summary] Retrieved {len(results)} pure issues (PRs filtered out).", file=sys.stderr)
    except (ValueError, NotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except GitHubAPIError as err:
        print(f"GitHub API Error: {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
