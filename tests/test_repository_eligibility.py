"""Unit test suite for repository eligibility evaluation and decoupled confidence matrix.

Verifies:
1. Public code gate: non-empty public repositories are eligible for ingestion.
2. Private repository rejection: explicit ineligibility reason.
3. Empty repository rejection: stubs with 0 commits and 0 size are rejected.
4. Non-blocking metadata flags: archived, forks, and templates are ingested with badges.
5. Decoupled Confidence Matrix: per-domain statistical sufficiency evaluations across all 5 domains.
6. Asynchronous GitHub API client error handling (404, 403, 429 with Retry-After, 401, network transport errors).
"""

import asyncio
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from backend.app.core.exceptions import (
    GitHubAPIError,
    GitHubRateLimitExceededException,
    GitHubResourceNotFoundException,
)
from backend.app.schemas.ingestion import (
    ConfidenceLevel,
    ParsedRepository,
)
from backend.app.services.confidence_matrix import ConfidenceMatrixEvaluator
from backend.app.services.repository_eligibility import RepositoryEligibilityChecker


class TestRepositoryEligibility:
    """Test suite for repository eligibility rules and status flags."""

    @pytest.fixture
    def checker(self):
        return RepositoryEligibilityChecker()

    @pytest.fixture
    def sample_parsed(self):
        return ParsedRepository(
            owner="pallets",
            repo="flask",
            full_name="pallets/flask",
            canonical_url="https://github.com/pallets/flask",
        )

    def test_eligible_public_repository(self, checker, sample_parsed):
        """Standard public repository with code is eligible for ingestion."""
        metadata = {
            "private": False,
            "archived": False,
            "fork": False,
            "size": 4200,
            "stargazers_count": 65000,
            "forks_count": 15000,
            "open_issues_count": 50,
            "default_branch": "main",
        }
        activity = {
            "commits_count": 4500,
            "contributors_count": 320,
            "prs_count": 1200,
            "issues_count": 800,
            "closed_issues_count": 750,
        }

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
            activity_metrics=activity,
        )

        assert res.is_eligible is True
        assert res.rejection_reason is None
        assert res.is_private is False
        assert res.is_empty is False
        assert res.confidence_report is not None
        assert res.confidence_report.overall_confidence == ConfidenceLevel.CONFIDENT
        assert res.confidence_report.overall_score == 1.0

    def test_ineligible_private_repository(self, checker, sample_parsed):
        """Private repositories must be strictly rejected with clear explanation."""
        metadata = {
            "private": True,
            "archived": False,
            "size": 1500,
        }

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
        )

        assert res.is_eligible is False
        assert res.is_private is True
        assert "Private repositories are not eligible" in res.rejection_reason

    def test_ineligible_empty_stub_repository(self, checker, sample_parsed):
        """Empty repository stubs with 0 size and 0 commits are rejected."""
        metadata = {
            "private": False,
            "size": 0,
            "open_issues_count": 0,
            "default_branch": "main",
        }
        activity = {"commits_count": 0}

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
            activity_metrics=activity,
        )

        assert res.is_eligible is False
        assert res.is_empty is True
        assert "empty stub" in res.rejection_reason.lower()
        assert "≥ 1 commit" in res.rejection_reason

    def test_ineligible_zero_commits_even_with_open_issues(self, checker, sample_parsed):
        """Repository with 0 commits is rejected even if issues or descriptions exist."""
        metadata = {
            "private": False,
            "size": 50,
            "open_issues_count": 5,
            "default_branch": "main",
        }
        activity = {"commits_count": 0}

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
            activity_metrics=activity,
        )

        assert res.is_eligible is False
        assert res.is_empty is True
        assert "empty stub" in res.rejection_reason.lower()
        assert "≥ 1 commit" in res.rejection_reason

    def test_ineligible_zero_size_stub_with_issues(self, checker, sample_parsed):
        """Repository with size=0 KB is rejected as empty stub even if open issues exist."""
        metadata = {
            "private": False,
            "size": 0,
            "open_issues_count": 10,
            "default_branch": "main",
        }

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
        )

        assert res.is_eligible is False
        assert res.is_empty is True
        assert "empty stub" in res.rejection_reason.lower()

    def test_ineligible_missing_default_branch(self, checker, sample_parsed):
        """Uninitialized repository with no default branch is rejected."""
        metadata = {
            "private": False,
            "size": 0,
            "default_branch": None,
        }

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
        )

        assert res.is_eligible is False
        assert res.is_empty is True

    def test_eligible_single_commit_repository(self, checker, sample_parsed):
        """Repository with at least 1 commit is eligible for ingestion."""
        metadata = {
            "private": False,
            "size": 15,
            "open_issues_count": 1,
            "default_branch": "main",
        }
        activity = {"commits_count": 1}

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
            activity_metrics=activity,
        )

        assert res.is_eligible is True
        assert res.is_empty is False
        assert res.rejection_reason is None

    def test_eligible_archived_repository_with_badge(self, checker, sample_parsed):
        """Archived repositories are eligible for historical analysis and badged accordingly."""
        metadata = {
            "private": False,
            "archived": True,
            "size": 3000,
            "open_issues_count": 10,
        }

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
        )

        assert res.is_eligible is True
        assert res.is_archived is True
        assert any("Archived repository" in badge for badge in res.badges)

    def test_eligible_fork_repository_with_badge(self, checker, sample_parsed):
        """Forked repositories are eligible and display upstream parent linkage badge."""
        metadata = {
            "private": False,
            "fork": True,
            "size": 2500,
            "parent": {"full_name": "upstream-org/original-flask"},
        }

        res = checker.evaluate_eligibility_from_metadata(
            parsed_repo=sample_parsed,
            metadata=metadata,
        )

        assert res.is_eligible is True
        assert res.is_fork is True
        assert res.upstream_repo == "upstream-org/original-flask"
        assert any("Fork of upstream-org/original-flask" in badge for badge in res.badges)


class TestConfidenceMatrix:
    """Test suite for decoupled per-domain statistical confidence matrix."""

    def test_domain_1_development_activity_thresholds(self):
        """Verifies commits thresholds: < 3 (insufficient), 3-9 (limited), >= 10 (confident)."""
        # Insufficient (< 3)
        for count in (0, 1, 2):
            res = ConfidenceMatrixEvaluator.evaluate_development_activity(count)
            assert res.level == ConfidenceLevel.INSUFFICIENT_DATA
            assert res.is_sufficient is False
            assert "Requires at least 3 commits" in res.message

        # Limited sample (3 <= commits < 10)
        for count in (3, 5, 9):
            res = ConfidenceMatrixEvaluator.evaluate_development_activity(count)
            assert res.level == ConfidenceLevel.LIMITED_SAMPLE
            assert res.is_sufficient is True
            assert res.warning is not None

        # Confident (>= 10)
        for count in (10, 25, 1000):
            res = ConfidenceMatrixEvaluator.evaluate_development_activity(count)
            assert res.level == ConfidenceLevel.CONFIDENT
            assert res.is_sufficient is True
            assert res.warning is None

    def test_domain_2_contributor_ecosystem_thresholds(self):
        """Verifies contributors thresholds: < 2 (insufficient), 2-4 (limited), >= 5 (confident)."""
        # Insufficient (< 2)
        for count in (0, 1):
            res = ConfidenceMatrixEvaluator.evaluate_contributor_ecosystem(count)
            assert res.level == ConfidenceLevel.INSUFFICIENT_DATA
            assert res.is_sufficient is False
            assert "mathematically degenerate" in res.message

        # Limited sample (2 <= contributors < 5)
        for count in (2, 3, 4):
            res = ConfidenceMatrixEvaluator.evaluate_contributor_ecosystem(count)
            assert res.level == ConfidenceLevel.LIMITED_SAMPLE
            assert res.is_sufficient is True
            assert "Low contributor count" in res.warning

        # Confident (>= 5)
        for count in (5, 12, 500):
            res = ConfidenceMatrixEvaluator.evaluate_contributor_ecosystem(count)
            assert res.level == ConfidenceLevel.CONFIDENT
            assert res.is_sufficient is True
            assert res.warning is None

    def test_domain_3_pr_lifecycle_thresholds(self):
        """Verifies PR thresholds: 0 (insufficient), 1-9 (limited), >= 10 (confident)."""
        # Insufficient (0)
        res = ConfidenceMatrixEvaluator.evaluate_pr_lifecycle(0)
        assert res.level == ConfidenceLevel.INSUFFICIENT_DATA
        assert res.is_sufficient is False
        assert "No pull requests" in res.message

        # Limited sample (1 <= prs < 10)
        for count in (1, 4, 9):
            res = ConfidenceMatrixEvaluator.evaluate_pr_lifecycle(count)
            assert res.level == ConfidenceLevel.LIMITED_SAMPLE
            assert res.is_sufficient is True

        # Confident (>= 10)
        for count in (10, 50, 500):
            res = ConfidenceMatrixEvaluator.evaluate_pr_lifecycle(count)
            assert res.level == ConfidenceLevel.CONFIDENT
            assert res.is_sufficient is True
            assert res.warning is None

    def test_domain_4_issue_intelligence_thresholds(self):
        """Verifies issue thresholds: 0 (insufficient), 1-9 (limited), >= 10 (confident)."""
        # Insufficient (0)
        res = ConfidenceMatrixEvaluator.evaluate_issue_intelligence(0)
        assert res.level == ConfidenceLevel.INSUFFICIENT_DATA
        assert res.is_sufficient is False

        # Limited sample (1 <= issues < 10)
        for count in (1, 5, 9):
            res = ConfidenceMatrixEvaluator.evaluate_issue_intelligence(count)
            assert res.level == ConfidenceLevel.LIMITED_SAMPLE
            assert res.is_sufficient is True

        # Confident (>= 10)
        for count in (10, 80, 2000):
            res = ConfidenceMatrixEvaluator.evaluate_issue_intelligence(count)
            assert res.level == ConfidenceLevel.CONFIDENT
            assert res.is_sufficient is True
            assert res.warning is None

    def test_domain_5_resolution_velocity_thresholds(self):
        """Verifies closed issues thresholds: < 5 (insufficient), 5-19 (limited), >= 20 (confident)."""
        # Insufficient (< 5)
        for count in (0, 2, 4):
            res = ConfidenceMatrixEvaluator.evaluate_resolution_velocity(count)
            assert res.level == ConfidenceLevel.INSUFFICIENT_DATA
            assert res.is_sufficient is False
            assert "Requires at least 5 closed issues" in res.message

        # Limited sample (5 <= closed < 20)
        for count in (5, 12, 19):
            res = ConfidenceMatrixEvaluator.evaluate_resolution_velocity(count)
            assert res.level == ConfidenceLevel.LIMITED_SAMPLE
            assert res.is_sufficient is True

        # Confident (>= 20)
        for count in (20, 50, 2500):
            res = ConfidenceMatrixEvaluator.evaluate_resolution_velocity(count)
            assert res.level == ConfidenceLevel.CONFIDENT
            assert res.is_sufficient is True
            assert res.warning is None

    def test_composite_confidence_evaluation(self):
        """Composite evaluation aggregates all 5 domains and weights overall readiness."""
        # Scenario 1: All confident
        report_conf = ConfidenceMatrixEvaluator.evaluate(
            commits_count=50,
            contributors_count=10,
            prs_count=25,
            issues_count=40,
            closed_issues_count=30,
        )
        assert report_conf.overall_confidence == ConfidenceLevel.CONFIDENT
        assert report_conf.overall_score == 1.0
        assert report_conf.sufficient_domains_count == 5

        # Scenario 2: Moderate / Limited sample
        report_lim = ConfidenceMatrixEvaluator.evaluate(
            commits_count=5,
            contributors_count=3,
            prs_count=2,
            issues_count=5,
            closed_issues_count=8,
        )
        assert report_lim.overall_confidence == ConfidenceLevel.LIMITED_SAMPLE
        assert report_lim.overall_score == 0.5
        assert report_lim.sufficient_domains_count == 5

        # Scenario 3: Sparse / Insufficient
        report_sparse = ConfidenceMatrixEvaluator.evaluate(
            commits_count=1,
            contributors_count=1,
            prs_count=0,
            issues_count=0,
            closed_issues_count=0,
        )
        assert report_sparse.overall_confidence == ConfidenceLevel.INSUFFICIENT_DATA
        assert report_sparse.overall_score == 0.0
        assert report_sparse.sufficient_domains_count == 0


class TestGitHubAPIFetcher:
    """Test suite for async GitHub API accessibility check and error mapping."""

    def test_fetch_github_metadata_success(self):
        """200 OK returns parsed JSON metadata."""
        mock_response = httpx.Response(
            status_code=200,
            json={"name": "flask", "owner": {"login": "pallets"}, "stargazers_count": 65000},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )
        checker = RepositoryEligibilityChecker()
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response)):
            data = asyncio.run(checker.fetch_github_metadata("pallets", "flask"))
            assert data["name"] == "flask"

    def test_fetch_github_metadata_404_not_found(self):
        """404 Not Found raises GitHubResourceNotFoundException."""
        mock_response = httpx.Response(
            status_code=404,
            json={"message": "Not Found"},
            request=httpx.Request("GET", "https://api.github.com/repos/nonexistent/repo"),
        )
        checker = RepositoryEligibilityChecker()
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response)):
            with pytest.raises(GitHubResourceNotFoundException) as exc_info:
                asyncio.run(checker.fetch_github_metadata("nonexistent", "repo"))
            assert exc_info.value.status_code == 404

    def test_fetch_github_metadata_rate_limited(self):
        """403 / 429 raises GitHubRateLimitExceededException with Retry-After parsed."""
        mock_response = httpx.Response(
            status_code=403,
            headers={"Retry-After": "120"},
            json={"message": "API rate limit exceeded for user"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )
        checker = RepositoryEligibilityChecker()
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response)):
            with pytest.raises(GitHubRateLimitExceededException) as exc_info:
                asyncio.run(checker.fetch_github_metadata("pallets", "flask"))
            assert exc_info.value.status_code == 429
            assert exc_info.value.retry_after_seconds == 120

    def test_fetch_github_metadata_401_bad_credentials(self):
        """401 Unauthorized raises GitHubAPIError."""
        mock_response = httpx.Response(
            status_code=401,
            json={"message": "Bad credentials"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )
        checker = RepositoryEligibilityChecker(github_token="invalid_token")
        with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response)):
            with pytest.raises(GitHubAPIError) as exc_info:
                asyncio.run(checker.fetch_github_metadata("pallets", "flask"))
            assert exc_info.value.status_code == 401

    def test_fetch_github_metadata_transport_error(self):
        """Network transport errors raise GitHubAPIError with status 502."""
        checker = RepositoryEligibilityChecker()
        with patch.object(
            httpx.AsyncClient,
            "get",
            AsyncMock(side_effect=httpx.ConnectTimeout("Connection timed out")),
        ):
            with pytest.raises(GitHubAPIError) as exc_info:
                asyncio.run(checker.fetch_github_metadata("pallets", "flask"))
            assert exc_info.value.status_code == 502
