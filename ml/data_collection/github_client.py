"""GitHub API Client for OpenSourceLens.

Centralizes authentication, request headers, error handling,
automatic retry loops, exponential backoff, and rate-limit awareness.
"""

import os
import time
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class NotFoundError(GitHubAPIError):
    """Raised when a requested resource (repository, issue, etc.) is not found (HTTP 404)."""
    pass


class RateLimitError(GitHubAPIError):
    """Raised when the GitHub API rate limit is exceeded (HTTP 403 or 429)."""
    def __init__(self, message: str, reset_epoch: Optional[int] = None, status_code: int = 403, response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=status_code, response_data=response_data)
        self.reset_epoch = reset_epoch


class AuthenticationError(GitHubAPIError):
    """Raised when authentication fails or token is invalid (HTTP 401)."""
    pass


class GitHubClient:
    """Resilient HTTP Client for interacting with the GitHub REST API (v2022-11-28)."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        auto_wait_on_rate_limit: bool = True,
        max_rate_limit_wait: int = 3600,
    ):
        """Initialize resilient GitHub client.

        Args:
            token: Optional GitHub personal access token (defaults to GITHUB_TOKEN in env).
            timeout: Request timeout in seconds.
            max_retries: Number of retries for transient errors (5xx, timeouts).
            backoff_factor: Multiplier for exponential backoff delays.
            auto_wait_on_rate_limit: If True, pauses when hourly quota is exhausted until reset time.
            max_rate_limit_wait: Maximum seconds to sleep on rate limit before raising RateLimitError.
        """
        self.token = (token or os.getenv("GITHUB_TOKEN", "")).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.auto_wait_on_rate_limit = auto_wait_on_rate_limit
        self.max_rate_limit_wait = max_rate_limit_wait
        self.session = requests.Session()
        self.last_rate_limit: Dict[str, Any] = {}

    @property
    def is_authenticated(self) -> bool:
        """Return True if an API token is configured."""
        return bool(self.token and not self.token.startswith("your_github_personal_access_token"))

    def _get_headers(self) -> Dict[str, str]:
        """Generate standard GitHub API headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "OpenSourceLens-Collector/1.0",
        }
        if self.is_authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff duration with jitter."""
        jitter = random.uniform(0.05, 0.2)
        return (self.backoff_factor * (2 ** attempt)) + jitter

    def _extract_rate_limit_headers(self, response: requests.Response) -> None:
        """Extract and cache rate-limit headers from response."""
        limit = response.headers.get("x-ratelimit-limit")
        remaining = response.headers.get("x-ratelimit-remaining")
        reset_ts = response.headers.get("x-ratelimit-reset")

        if limit and remaining and reset_ts:
            reset_epoch = int(reset_ts) if reset_ts.isdigit() else 0
            self.last_rate_limit = {
                "limit": int(limit) if limit.isdigit() else 0,
                "remaining": int(remaining) if remaining.isdigit() else 0,
                "reset_epoch": reset_epoch,
                "reset_time_utc": datetime.fromtimestamp(reset_epoch, tz=timezone.utc).isoformat() if reset_epoch else None,
            }

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """Execute a GET request against the GitHub REST API with retries and rate limit management.

        Args:
            endpoint: API endpoint path (e.g. '/repos/microsoft/vscode') or full URL.
            params: Optional query parameters dictionary.

        Returns:
            requests.Response object on success.

        Raises:
            NotFoundError: On HTTP 404 (non-retryable).
            AuthenticationError: On HTTP 401 (non-retryable).
            RateLimitError: When rate limit is exceeded and auto-wait is disabled/exceeded.
            GitHubAPIError: On unrecoverable HTTP or network errors.
        """
        url = endpoint if endpoint.startswith("http") else f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt < self.max_retries:
                    sleep_sec = self._calculate_backoff(attempt)
                    time.sleep(sleep_sec)
                    continue
                raise GitHubAPIError(f"Network error querying GitHub API after {self.max_retries} retries: {exc}") from exc
            except requests.exceptions.RequestException as exc:
                raise GitHubAPIError(f"Fatal network error querying GitHub API: {exc}") from exc

            self._extract_rate_limit_headers(response)

            # Fast-fail for non-retryable errors
            if response.status_code == 404:
                raise NotFoundError(
                    f"Resource not found at {url}: {response.text}",
                    status_code=404,
                    response_data=self._safe_json(response),
                )

            if response.status_code == 401:
                raise AuthenticationError(
                    f"GitHub API authentication failed. Verify your GITHUB_TOKEN: {response.text}",
                    status_code=401,
                    response_data=self._safe_json(response),
                )

            # Transient server errors (500, 502, 503, 504) -> retry with backoff
            if response.status_code in (500, 502, 503, 504):
                if attempt < self.max_retries:
                    sleep_sec = self._calculate_backoff(attempt)
                    time.sleep(sleep_sec)
                    continue
                raise GitHubAPIError(
                    f"GitHub server error (HTTP {response.status_code}) after {self.max_retries} retries: {response.text}",
                    status_code=response.status_code,
                    response_data=self._safe_json(response),
                )

            # Rate limiting & Secondary Abuse limits (403, 429)
            if response.status_code in (403, 429):
                retry_after = response.headers.get("Retry-After")
                remaining = response.headers.get("x-ratelimit-remaining")
                reset_ts = response.headers.get("x-ratelimit-reset")
                reset_epoch = int(reset_ts) if reset_ts and reset_ts.isdigit() else None

                # Secondary abuse limit (Retry-After header present)
                if retry_after and retry_after.isdigit():
                    wait_sec = int(retry_after) + 1
                    if attempt < self.max_retries and wait_sec <= self.max_rate_limit_wait:
                        time.sleep(wait_sec)
                        continue
                    raise RateLimitError(
                        f"Secondary rate limit triggered (Retry-After: {wait_sec}s).",
                        reset_epoch=reset_epoch,
                        status_code=response.status_code,
                        response_data=self._safe_json(response),
                    )

                # Primary hourly rate limit
                if remaining == "0" or "rate limit" in response.text.lower():
                    current_epoch = int(time.time())
                    wait_sec = (reset_epoch - current_epoch + 1) if reset_epoch else 60

                    if self.auto_wait_on_rate_limit and 0 < wait_sec <= self.max_rate_limit_wait and attempt < self.max_retries:
                        time.sleep(wait_sec)
                        continue

                    raise RateLimitError(
                        f"GitHub API hourly rate limit exceeded. Reset time epoch: {reset_epoch}",
                        reset_epoch=reset_epoch,
                        status_code=403,
                        response_data=self._safe_json(response),
                    )

                raise GitHubAPIError(
                    f"Access forbidden (HTTP 403): {response.text}",
                    status_code=403,
                    response_data=self._safe_json(response),
                )

            if not response.ok:
                raise GitHubAPIError(
                    f"GitHub API returned error HTTP {response.status_code} for {url}: {response.text}",
                    status_code=response.status_code,
                    response_data=self._safe_json(response),
                )

            return response

        raise GitHubAPIError(f"Request failed after {self.max_retries} attempts.")

    def get_rate_limit(self) -> Dict[str, Any]:
        """Fetch current rate limit status via the /rate_limit endpoint."""
        resp = self.get("/rate_limit")
        return resp.json().get("resources", {}).get("core", {})

    def get_quota_status(self) -> Dict[str, Any]:
        """Retrieve formatted quota overview including limit, remaining, and reset time."""
        core = self.get_rate_limit()
        limit = core.get("limit", 0)
        remaining = core.get("remaining", 0)
        reset_epoch = core.get("reset", 0)
        used = limit - remaining
        percent_used = round((used / limit * 100), 2) if limit > 0 else 0.0

        return {
            "is_authenticated": self.is_authenticated,
            "limit": limit,
            "remaining": remaining,
            "used": used,
            "percent_used": percent_used,
            "reset_epoch": reset_epoch,
            "reset_time_utc": datetime.fromtimestamp(reset_epoch, tz=timezone.utc).isoformat() if reset_epoch else None,
        }

    @staticmethod
    def _safe_json(response: requests.Response) -> Dict[str, Any]:
        """Safely parse JSON response body without throwing exceptions."""
        try:
            return response.json()
        except ValueError:
            return {}
