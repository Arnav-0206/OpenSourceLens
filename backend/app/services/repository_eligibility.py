"""Repository eligibility checker and pre-flight validation service.

Decouples ingestion eligibility (public, accessible, non-empty code gate) from
per-domain analytical confidence assessment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from backend.app.core.exceptions import (
    GitHubAPIError,
    GitHubRateLimitExceededException,
    GitHubResourceNotFoundException,
    RepositoryIneligibleException,
)
from backend.app.schemas.ingestion import (
    ParsedRepository,
    RepositoryEligibilityResult,
)
from backend.app.services.confidence_matrix import ConfidenceMatrixEvaluator
from backend.app.services.validator import GitHubURLValidator

logger = logging.getLogger(__name__)


class RepositoryEligibilityChecker:
    """Evaluates repository ingestion eligibility and statistical analytical readiness."""

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(self, github_token: Optional[str] = None) -> None:
        """Initialize with optional GitHub API token for authenticated pre-flight requests."""
        self.github_token = github_token

    def _get_headers(self) -> Dict[str, str]:
        """Construct standard GitHub API headers."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OpenSourceLens-EligibilityChecker/1.0",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    async def fetch_github_metadata(
        self, owner: str, repo: str, client: Optional[httpx.AsyncClient] = None
    ) -> Dict[str, Any]:
        """Fetch repository metadata envelope directly from GitHub REST API v3.

        Args:
            owner: Repository owner login.
            repo: Repository short name.
            client: Optional httpx.AsyncClient for connection pooling.

        Returns:
            Dict containing raw GitHub repository JSON metadata.

        Raises:
            GitHubResourceNotFoundException: If repository does not exist (404).
            GitHubRateLimitExceededException: If rate limit exceeded (403/429).
            GitHubAPIError: For other HTTP or connection errors.
        """
        url = f"{self.GITHUB_API_BASE}/repos/{owner}/{repo}"
        headers = self._get_headers()

        async def _do_request(c: httpx.AsyncClient) -> httpx.Response:
            return await c.get(url, headers=headers, timeout=15.0)

        try:
            if client:
                resp = await _do_request(client)
            else:
                async with httpx.AsyncClient() as new_client:
                    resp = await _do_request(new_client)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                raise GitHubResourceNotFoundException(
                    f"Repository '{owner}/{repo}' was not found on GitHub.",
                    repo_name=f"{owner}/{repo}",
                )
            elif resp.status_code in (403, 429):
                retry_after_str = resp.headers.get("Retry-After")
                retry_after = int(retry_after_str) if retry_after_str and retry_after_str.isdigit() else None
                message = resp.json().get("message", "GitHub API rate limit exceeded.") if resp.content else "Rate limited."
                raise GitHubRateLimitExceededException(
                    message=f"GitHub API rate limit exceeded: {message}",
                    retry_after_seconds=retry_after,
                )
            elif resp.status_code == 401:
                raise GitHubAPIError(
                    "GitHub API authentication failed. Verify configured token.",
                    status_code=401,
                )
            else:
                raise GitHubAPIError(
                    f"GitHub API returned unexpected status {resp.status_code} for '{owner}/{repo}'.",
                    status_code=resp.status_code,
                )
        except httpx.RequestError as exc:
            raise GitHubAPIError(
                f"Transport failure connecting to GitHub API: {str(exc)}",
                status_code=502,
            ) from exc

    def evaluate_eligibility_from_metadata(
        self,
        parsed_repo: ParsedRepository,
        metadata: Dict[str, Any],
        activity_metrics: Optional[Dict[str, int]] = None,
    ) -> RepositoryEligibilityResult:
        """Evaluate ingestion eligibility and analytical confidence from GitHub metadata envelope.

        Eligibility Rules:
        1. Public accessibility: Repository must not be private.
        2. Non-empty code presence: Repository size > 0 or has observed commits.
        3. Archived and forked repositories are eligible (with informational badges).

        Args:
            parsed_repo: Normalized repository slug.
            metadata: GitHub repository API response dict.
            activity_metrics: Optional pre-harvested activity counts for confidence scoring.

        Returns:
            RepositoryEligibilityResult: Detailed eligibility evaluation and confidence report.
        """
        is_private = bool(metadata.get("private", False))
        is_archived = bool(metadata.get("archived", False))
        is_fork = bool(metadata.get("fork", False))
        is_template = bool(metadata.get("is_template", False))
        size_kb = metadata.get("size", 0)
        open_issues = metadata.get("open_issues_count", 0)
        stars = metadata.get("stargazers_count", metadata.get("stars_count", 0))
        forks = metadata.get("forks_count", 0)
        default_branch = metadata.get("default_branch")
        
        # Parent repository if this is a fork
        parent_meta = metadata.get("parent") or metadata.get("source") or {}
        upstream_repo = parent_meta.get("full_name") if is_fork else None

        badges: List[str] = []
        if is_archived:
            badges.append("Archived repository: read-only historical analysis")
        if is_fork:
            badges.append(f"Fork of {upstream_repo}" if upstream_repo else "Fork repository")
        if is_template:
            badges.append("Template repository")

        # Ingestion Gate 1: Public accessibility
        if is_private:
            return RepositoryEligibilityResult(
                is_eligible=False,
                repo_name=parsed_repo.full_name,
                owner=parsed_repo.owner,
                name=parsed_repo.repo,
                canonical_url=parsed_repo.canonical_url,
                rejection_reason=(
                    "OpenSourceLens only analyzes public open-source software. "
                    "Private repositories are not eligible for ingestion."
                ),
                is_private=True,
                is_archived=is_archived,
                is_fork=is_fork,
                upstream_repo=upstream_repo,
                is_template=is_template,
                stars_count=stars,
                forks_count=forks,
                open_issues_count=open_issues,
                default_branch=default_branch,
                badges=badges,
            )

        # Ingestion Gate 2: Minimum code presence gate (Requires at least 1 commit)
        # A repository is ineligible if:
        # 1. commits_count is explicitly provided/known and < 1 (even if issues exist).
        # 2. OR repository size is 0 KB (empty repository on GitHub with no pushed commits).
        commits_metric = (activity_metrics or {}).get("commits_count")
        has_zero_commits = commits_metric is not None and commits_metric < 1
        is_empty_stub = size_kb == 0
        is_empty = has_zero_commits or (commits_metric is None and is_empty_stub)

        if is_empty:
            return RepositoryEligibilityResult(
                is_eligible=False,
                repo_name=parsed_repo.full_name,
                owner=parsed_repo.owner,
                name=parsed_repo.repo,
                canonical_url=parsed_repo.canonical_url,
                rejection_reason="Repository is an empty stub with no commits or code (Requires ≥ 1 commit).",
                is_private=False,
                is_archived=is_archived,
                is_fork=is_fork,
                upstream_repo=upstream_repo,
                is_template=is_template,
                is_empty=True,
                stars_count=stars,
                forks_count=forks,
                open_issues_count=open_issues,
                commits_count=commits_metric,
                default_branch=default_branch,
                badges=badges,
            )

        # Decoupled Analytical Confidence Evaluation
        # If activity metrics were provided, evaluate confidence matrix
        metrics = activity_metrics or {}
        confidence_report = ConfidenceMatrixEvaluator.evaluate(
            commits_count=metrics.get("commits_count", 0),
            contributors_count=metrics.get("contributors_count", 0),
            prs_count=metrics.get("prs_count", 0),
            issues_count=metrics.get("issues_count", open_issues),
            closed_issues_count=metrics.get("closed_issues_count", 0),
        )

        return RepositoryEligibilityResult(
            is_eligible=True,
            repo_name=parsed_repo.full_name,
            owner=parsed_repo.owner,
            name=parsed_repo.repo,
            canonical_url=parsed_repo.canonical_url,
            rejection_reason=None,
            is_private=False,
            is_archived=is_archived,
            is_fork=is_fork,
            upstream_repo=upstream_repo,
            is_template=is_template,
            is_empty=False,
            stars_count=stars,
            forks_count=forks,
            open_issues_count=open_issues,
            commits_count=commits_metric,
            default_branch=default_branch,
            confidence_report=confidence_report,
            badges=badges,
        )

    async def check_eligibility(
        self,
        raw_input: str,
        cached_metadata: Optional[Dict[str, Any]] = None,
        activity_metrics: Optional[Dict[str, int]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> RepositoryEligibilityResult:
        """Complete workflow: validate URL, retrieve metadata, evaluate eligibility and confidence.

        Args:
            raw_input: Raw GitHub URL or slug string.
            cached_metadata: Optional pre-loaded GitHub metadata (avoids API calls during testing).
            activity_metrics: Optional counts for analytical confidence evaluation.
            client: Optional httpx.AsyncClient.

        Returns:
            RepositoryEligibilityResult: Eligibility verdict, metadata flags, and confidence report.
        """
        parsed_repo = GitHubURLValidator.validate_and_canonicalize(raw_input)

        if cached_metadata is not None:
            metadata = cached_metadata
        else:
            metadata = await self.fetch_github_metadata(
                parsed_repo.owner, parsed_repo.repo, client=client
            )

        return self.evaluate_eligibility_from_metadata(
            parsed_repo=parsed_repo,
            metadata=metadata,
            activity_metrics=activity_metrics,
        )
