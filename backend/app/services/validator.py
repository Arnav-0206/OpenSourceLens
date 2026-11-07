"""Strict GitHub URL parsing, normalization, and validation service.

Conforms to RFC 3986 URL validation, GitHub naming specifications, and strict
OpenSourceLens product boundaries.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from backend.app.core.exceptions import InvalidRepositoryURLException
from backend.app.schemas.ingestion import ParsedRepository


# GitHub reserved system paths that cannot serve as valid user/organization owners
RESERVED_GITHUB_PATHS = frozenset(
    {
        "about",
        "apps",
        "contact",
        "dashboard",
        "discussions",
        "explore",
        "features",
        "issues",
        "join",
        "login",
        "logout",
        "marketplace",
        "notifications",
        "orgs",
        "pricing",
        "pulls",
        "search",
        "security",
        "settings",
        "site",
        "sponsors",
        "stars",
        "trending",
        "users",
    }
)

# Valid GitHub owner: alphanumeric with single hyphens, 1-39 chars, cannot start or end with hyphen
GITHUB_OWNER_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?$")

# Valid GitHub repo name: alphanumeric, hyphens, underscores, dots, 1-100 chars
GITHUB_REPO_REGEX = re.compile(r"^[a-zA-Z0-9_.-]+$")

# Permitted GitHub domains (case-insensitive)
ALLOWED_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})


class GitHubURLValidator:
    """Strict parser and validator for GitHub repository URLs and slugs."""

    @classmethod
    def validate_and_canonicalize(cls, raw_input: str) -> ParsedRepository:
        """Parse, validate, and normalize a user-supplied repository string.

        Supports:
        - Full URLs: 'https://github.com/owner/repo', 'http://github.com/owner/repo.git'
        - Scheme-less URLs: 'github.com/owner/repo', 'www.github.com/owner/repo/'
        - Canonical slugs: 'owner/repo'

        Rejects:
        - Non-GitHub hostnames (GitLab, Bitbucket, etc.)
        - Scoped package names (e.g., '@vuejs/core')
        - Reserved GitHub path segments
        - Deep paths (e.g., repository tree/pull/issues sub-paths)
        - Malformed owner or repository names violating GitHub constraints

        Args:
            raw_input: Raw input string provided by user or API caller.

        Returns:
            ParsedRepository: Canonical owner, repo, full_name, and https URL.

        Raises:
            InvalidRepositoryURLException: If the input violates any rule.
        """
        if not raw_input or not isinstance(raw_input, str):
            raise InvalidRepositoryURLException("Repository input cannot be empty.")

        trimmed = raw_input.strip()
        if not trimmed:
            raise InvalidRepositoryURLException("Repository input cannot be empty.")

        # Check 1: Explicit rejection of npm-style scoped packages
        if trimmed.startswith("@"):
            raise InvalidRepositoryURLException(
                "Invalid GitHub repository slug. Scoped package names (e.g. '@owner/repo') "
                "are not supported; use 'owner/repo' or 'https://github.com/owner/repo'.",
                raw_input=trimmed,
            )

        # Check 2: Parse URL or extract slug
        path_candidate: str

        if "://" in trimmed or trimmed.startswith("//"):
            # Formal URL with protocol scheme or protocol-relative
            parsed = urlsplit(trimmed)
            raw_netloc = parsed.netloc.split("@")[-1].split(":")[0].lower()
            if not raw_netloc:
                raise InvalidRepositoryURLException(
                    f"Malformed repository URL: missing hostname in '{trimmed}'.",
                    raw_input=trimmed,
                )

            if raw_netloc not in ALLOWED_GITHUB_HOSTS:
                raise InvalidRepositoryURLException(
                    f"Only GitHub repositories are supported. Received host: '{raw_netloc}'.",
                    raw_input=trimmed,
                )

            path_candidate = parsed.path
        elif trimmed.lower().startswith("github.com/") or trimmed.lower().startswith("www.github.com/"):
            # Domain prefix without scheme (e.g. github.com/pallets/flask)
            _, _, path_candidate = trimmed.partition("/")
        else:
            # Assumed slug format (e.g. pallets/flask)
            # If the user included any other domain-like pattern (e.g. gitlab.com/owner/repo)
            if "/" in trimmed:
                first_segment = trimmed.split("/")[0].lower()
                if "." in first_segment and first_segment not in ALLOWED_GITHUB_HOSTS:
                    raise InvalidRepositoryURLException(
                        f"Only GitHub repositories are supported. Received host: '{first_segment}'.",
                        raw_input=trimmed,
                    )
            path_candidate = trimmed

        # Check 3: Extract and validate path segments
        clean_path = path_candidate.strip("/")
        segments = [seg for seg in clean_path.split("/") if seg]

        if len(segments) != 2:
            raise InvalidRepositoryURLException(
                f"Invalid GitHub repository path: expected 'owner/repo', got '{trimmed}'.",
                raw_input=trimmed,
            )

        owner = segments[0].strip()
        repo = segments[1].strip()

        # Check 4: Strip trailing .git suffix if present
        if repo.lower().endswith(".git"):
            repo = repo[:-4]

        if not repo:
            raise InvalidRepositoryURLException(
                f"Invalid repository name after removing .git suffix from '{segments[1]}'.",
                raw_input=trimmed,
            )

        # Check 5: Reserved GitHub path blacklist
        if owner.lower() in RESERVED_GITHUB_PATHS:
            raise InvalidRepositoryURLException(
                f"'{owner}' is a reserved GitHub path and cannot be a repository owner.",
                raw_input=trimmed,
            )

        # Check 6: Owner syntactic constraints
        if len(owner) > 39:
            raise InvalidRepositoryURLException(
                f"Repository owner '{owner}' exceeds the maximum allowed length of 39 characters.",
                raw_input=trimmed,
            )

        if not GITHUB_OWNER_REGEX.match(owner) or "--" in owner:
            raise InvalidRepositoryURLException(
                f"Invalid repository owner '{owner}'. GitHub usernames may only contain "
                "alphanumeric characters or single hyphens, and cannot begin or end with a hyphen.",
                raw_input=trimmed,
            )

        # Check 7: Repository name syntactic constraints
        if len(repo) > 100:
            raise InvalidRepositoryURLException(
                f"Repository name '{repo}' exceeds the maximum allowed length of 100 characters.",
                raw_input=trimmed,
            )

        if repo in {".", ".."}:
            raise InvalidRepositoryURLException(
                f"Invalid repository name '{repo}': reserved filesystem token.",
                raw_input=trimmed,
            )

        if not GITHUB_REPO_REGEX.match(repo):
            raise InvalidRepositoryURLException(
                f"Invalid repository name '{repo}'. Repository names may only contain "
                "alphanumeric characters, hyphens, underscores, and periods.",
                raw_input=trimmed,
            )

        full_name = f"{owner}/{repo}"
        canonical_url = f"https://github.com/{owner}/{repo}"

        return ParsedRepository(
            owner=owner,
            repo=repo,
            full_name=full_name,
            canonical_url=canonical_url,
        )
