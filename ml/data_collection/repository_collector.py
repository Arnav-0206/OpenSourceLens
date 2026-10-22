"""Repository Collector for OpenSourceLens.

Collects, validates, and standardizes repository-level metadata from the
GitHub REST API according to docs/0.3-data-definition.md.
"""

import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from ml.data_collection.github_client import GitHubClient, GitHubAPIError, NotFoundError


class RepositoryCollector:
    """Collector responsible for fetching and standardizing repository metadata."""

    def __init__(self, client: Optional[GitHubClient] = None):
        """Initialize with an optional shared GitHubClient."""
        self.client = client or GitHubClient()

    @staticmethod
    def parse_repo_identifier(repo_identifier: str) -> Tuple[str, str]:
        """Validate and parse a repository identifier string.

        Args:
            repo_identifier: Formatted as 'owner/repo' (e.g. 'microsoft/vscode')
                            or 'https://github.c    om/owner/repo'.

        Returns:
            Tuple of (owner, repo).

        Raises:
            ValueError: If the identifier format is invalid.
        """
        cleaned = repo_identifier.strip().rstrip("/")
        if "github.com/" in cleaned:
            cleaned = cleaned.split("github.com/")[-1]

        parts = [p.strip() for p in cleaned.split("/") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid repository identifier: '{repo_identifier}'. "
                f"Expected format: 'owner/repo' (e.g., 'microsoft/vscode')."
            )
        return parts[0], parts[1]

    def collect(self, repo_identifier: str) -> Dict[str, Any]:
        """Collect and format repository metadata for a given repository.

        Args:
            repo_identifier: 'owner/repo' string (e.g. 'microsoft/vscode').

        Returns:
            A dictionary containing:
              - 'metadata': Standardized dictionary of repository fields
              - 'raw': Complete raw JSON response payload from GitHub API

        Raises:
            ValueError: On invalid input.
            NotFoundError: If the repository does not exist on GitHub.
            GitHubAPIError: On rate limit, authentication, or network failures.
        """
        owner, repo = self.parse_repo_identifier(repo_identifier)
        endpoint = f"/repos/{owner}/{repo}"

        response = self.client.get(endpoint)
        raw_data = response.json()

        structured_metadata = self._extract_metadata(raw_data)
        return {
            "metadata": structured_metadata,
            "raw": raw_data,
        }

    def _extract_metadata(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standardized repository fields from raw API response."""
        license_info = raw.get("license") or {}
        license_spdx = license_info.get("spdx_id") or license_info.get("name")
        if license_spdx == "NOASSERTION":
            license_spdx = license_info.get("name")

        return {
            "repository_name": raw.get("name"),
            "full_name": raw.get("full_name"),
            "owner": (raw.get("owner") or {}).get("login"),
            "description": raw.get("description"),
            "language": raw.get("language"),
            "stars": raw.get("stargazers_count", 0),
            "forks": raw.get("forks_count", 0),
            "open_issues_count": raw.get("open_issues_count", 0),
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
            "pushed_at": raw.get("pushed_at"),
            "has_issues": raw.get("has_issues", True),
            "license": license_spdx,
            "default_branch": raw.get("default_branch"),
            "archived": raw.get("archived", False),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }


def collect_repository(repo_identifier: str, client: Optional[GitHubClient] = None) -> Dict[str, Any]:
    """Convenience helper to collect repository metadata."""
    collector = RepositoryCollector(client=client)
    return collector.collect(repo_identifier)


def main():
    """CLI entrypoint for standalone repository collection."""
    # Ensure UTF-8 output encoding on Windows consoles
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Collect repository metadata from GitHub.")
    parser.add_argument("repository", type=str, help="Repository identifier in 'owner/repo' format (e.g., 'microsoft/vscode')")
    parser.add_argument("--raw", action="store_true", help="Print raw API response payload instead of structured metadata")
    args = parser.parse_args()

    try:
        collector = RepositoryCollector()
        result = collector.collect(args.repository)
        output = result["raw"] if args.raw else result["metadata"]
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except (ValueError, NotFoundError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except GitHubAPIError as err:
        print(f"GitHub API Error: {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
