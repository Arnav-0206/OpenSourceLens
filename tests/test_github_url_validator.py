"""Unit test suite for strict GitHub URL validation and normalization.

Verifies:
1. Valid canonical slugs and full URLs across schemas and domains.
2. Trailing slash and .git normalization.
3. Explicit rejection of non-GitHub hostnames (GitLab, Bitbucket, npm, etc.).
4. Explicit rejection of npm-style scoped packages (@scope/name).
5. Rejection of GitHub reserved segment paths (pulls, issues, explore, etc.).
6. Rejection of deep sub-paths (branches, trees, PR links).
7. Syntactic constraint enforcement on owner and repository tokens.
"""

import pytest

from backend.app.core.exceptions import InvalidRepositoryURLException
from backend.app.services.validator import GitHubURLValidator


class TestGitHubURLValidator:
    """Test suite for GitHubURLValidator."""

    # --------------------------------------------------------------------------
    # Valid Inputs
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "raw_input,expected_owner,expected_repo",
        [
            ("pallets/flask", "pallets", "flask"),
            ("facebook/react", "facebook", "react"),
            ("psf/requests", "psf", "requests"),
            ("tiangolo/fastapi", "tiangolo", "fastapi"),
            ("https://github.com/pallets/flask", "pallets", "flask"),
            ("http://github.com/pallets/flask", "pallets", "flask"),
            ("https://www.github.com/facebook/react", "facebook", "react"),
            ("http://www.github.com/facebook/react", "facebook", "react"),
            ("github.com/pallets/flask", "pallets", "flask"),
            ("www.github.com/facebook/react", "facebook", "react"),
            ("https://github.com/pallets/flask.git", "pallets", "flask"),
            ("http://github.com/pallets/flask.git", "pallets", "flask"),
            ("git://github.com/pallets/flask.git", "pallets", "flask"),
            ("https://github.com/pallets/flask/", "pallets", "flask"),
            ("https://github.com/pallets/flask///", "pallets", "flask"),
            ("  pallets/flask  ", "pallets", "flask"),
            ("\n\t https://github.com/pallets/flask \t\n", "pallets", "flask"),
            ("microsoft/vscode", "microsoft", "vscode"),
            ("pytorch/pytorch", "pytorch", "pytorch"),
            ("kubernetes/kubernetes", "kubernetes", "kubernetes"),
            ("huggingface/transformers", "huggingface", "transformers"),
            ("foo-bar/my-cool-project", "foo-bar", "my-cool-project"),
            ("project-dots/lib.v2.core", "project-dots", "lib.v2.core"),
            ("user123/repo_123", "user123", "repo_123"),
        ],
    )
    def test_valid_repository_inputs(self, raw_input, expected_owner, expected_repo):
        """Valid inputs must normalize cleanly to canonical ParsedRepository."""
        parsed = GitHubURLValidator.validate_and_canonicalize(raw_input)
        assert parsed.owner == expected_owner
        assert parsed.repo == expected_repo
        assert parsed.full_name == f"{expected_owner}/{expected_repo}"
        assert parsed.canonical_url == f"https://github.com/{expected_owner}/{expected_repo}"

    # --------------------------------------------------------------------------
    # Scoped Package Rejections
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "scoped_input",
        [
            "@vuejs/core",
            "@angular/cli",
            "@babel/core",
            "  @org/package  ",
            "@fastify/jwt",
        ],
    )
    def test_rejection_of_scoped_packages(self, scoped_input):
        """Scoped npm-style packages must be explicitly rejected with guidance message."""
        with pytest.raises(InvalidRepositoryURLException) as exc_info:
            GitHubURLValidator.validate_and_canonicalize(scoped_input)
        assert "Scoped package names" in exc_info.value.message
        assert "@owner/repo" in exc_info.value.message
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "INVALID_REPOSITORY_URL"

    # --------------------------------------------------------------------------
    # Non-GitHub Hostname Rejections
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "external_host_url",
        [
            "https://gitlab.com/owner/project",
            "http://gitlab.com/owner/project.git",
            "https://bitbucket.org/owner/project",
            "https://npmjs.com/package/express",
            "https://pypi.org/project/fastapi",
            "gitlab.com/owner/repo",
            "bitbucket.org/owner/repo",
            "//gitlab.com/owner/repo",
        ],
    )
    def test_rejection_of_external_hosts(self, external_host_url):
        """Non-GitHub hostnames must be strictly rejected."""
        with pytest.raises(InvalidRepositoryURLException) as exc_info:
            GitHubURLValidator.validate_and_canonicalize(external_host_url)
        assert "Only GitHub repositories are supported" in exc_info.value.message
        assert exc_info.value.status_code == 400

    # --------------------------------------------------------------------------
    # Reserved Path Blacklist Rejections
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "reserved_slug",
        [
            "pulls/123",
            "issues/456",
            "settings/profile",
            "explore/trending",
            "marketplace/actions",
            "login/oauth",
            "https://github.com/about/company",
            "https://github.com/features/actions",
            "https://github.com/orgs/community",
            "search/repositories",
            "security/advisories",
            "sponsors/dashboard",
        ],
    )
    def test_rejection_of_reserved_github_paths(self, reserved_slug):
        """Reserved GitHub path segments cannot be treated as repository owners."""
        with pytest.raises(InvalidRepositoryURLException) as exc_info:
            GitHubURLValidator.validate_and_canonicalize(reserved_slug)
        assert "reserved GitHub path" in exc_info.value.message
        assert exc_info.value.status_code == 400

    # --------------------------------------------------------------------------
    # Invalid Path Structures & Deep Paths
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "invalid_path",
        [
            "https://github.com/pallets/flask/tree/main",
            "https://github.com/pallets/flask/pull/42",
            "https://github.com/pallets/flask/issues/100",
            "https://github.com/pallets/flask/blob/main/README.md",
            "pallets/flask/subfolder",
            "single_token_without_slash",
            "/only_slash_start",
            "only_slash_end/",
            "",
            "   ",
            "//",
            "https://github.com/",
            "https://github.com/onlyowner",
        ],
    )
    def test_rejection_of_invalid_path_structures(self, invalid_path):
        """Deep paths or single-token inputs must be rejected."""
        with pytest.raises(InvalidRepositoryURLException) as exc_info:
            GitHubURLValidator.validate_and_canonicalize(invalid_path)
        assert exc_info.value.status_code == 400

    # --------------------------------------------------------------------------
    # Syntactic & Character Rules
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "malformed_name",
        [
            "-owner/repo",  # owner cannot start with hyphen
            "owner-/repo",  # owner cannot end with hyphen
            "ow--ner/repo",  # consecutive hyphens
            "owner$/repo",  # illegal character $
            "owner/repo*",  # illegal character *
            "owner/repo?query=1",  # query parameter attached to slug
            "owner/.",  # reserved token .
            "owner/..",  # reserved token ..
            "a" * 40 + "/repo",  # owner length > 39 chars
            "owner/" + "b" * 101,  # repo length > 100 chars
        ],
    )
    def test_rejection_of_malformed_tokens(self, malformed_name):
        """Tokens violating GitHub naming constraints must be rejected."""
        with pytest.raises(InvalidRepositoryURLException) as exc_info:
            GitHubURLValidator.validate_and_canonicalize(malformed_name)
        assert exc_info.value.status_code == 400
