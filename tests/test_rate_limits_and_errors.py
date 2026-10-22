"""Unit and integration tests for GitHubClient error handling, retries, and rate limits."""

import unittest
from unittest.mock import MagicMock, patch
import requests

from ml.data_collection.github_client import (
    GitHubClient,
    GitHubAPIError,
    NotFoundError,
    RateLimitError,
    AuthenticationError,
)


class TestRateLimitsAndErrors(unittest.TestCase):
    """Test suite for retry loops, exponential backoff, rate limits, and fail-fast behavior."""

    def setUp(self):
        self.client = GitHubClient(max_retries=2, backoff_factor=0.01)

    def test_retry_on_503_transient_error(self):
        """Test that a transient 503 error triggers a retry and succeeds on next attempt."""
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.ok = False
        resp_503.text = "Service Unavailable"
        resp_503.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.ok = True
        resp_200.json.return_value = {"message": "success"}
        resp_200.headers = {}

        self.client.session.get = MagicMock(side_effect=[resp_503, resp_200])

        response = self.client.get("/test-endpoint")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session.get.call_count, 2)

    def test_fail_fast_on_404_not_found(self):
        """Test that 404 raises NotFoundError immediately without retrying."""
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.ok = False
        resp_404.text = "Not Found"
        resp_404.headers = {}

        self.client.session.get = MagicMock(return_value=resp_404)

        with self.assertRaises(NotFoundError):
            self.client.get("/nonexistent")

        self.assertEqual(self.client.session.get.call_count, 1)

    def test_fail_fast_on_401_unauthorized(self):
        """Test that 401 raises AuthenticationError immediately without retrying."""
        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.ok = False
        resp_401.text = "Bad credentials"
        resp_401.headers = {}

        self.client.session.get = MagicMock(return_value=resp_401)

        with self.assertRaises(AuthenticationError):
            self.client.get("/user")

        self.assertEqual(self.client.session.get.call_count, 1)

    def test_retry_after_secondary_rate_limit(self):
        """Test that HTTP 429 with Retry-After header triggers backoff and retries."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.ok = False
        resp_429.text = "Too Many Requests"
        resp_429.headers = {"Retry-After": "0"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.ok = True
        resp_200.json.return_value = {"status": "ok"}
        resp_200.headers = {}

        self.client.session.get = MagicMock(side_effect=[resp_429, resp_200])

        response = self.client.get("/test-endpoint")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.session.get.call_count, 2)

    def test_primary_rate_limit_exhausted_raises_when_wait_disabled(self):
        """Test that rate limit exhaustion raises RateLimitError when auto_wait is False."""
        client_no_wait = GitHubClient(auto_wait_on_rate_limit=False)

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.ok = False
        resp_403.text = "API rate limit exceeded"
        resp_403.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1800000000",
        }

        client_no_wait.session.get = MagicMock(return_value=resp_403)

        with self.assertRaises(RateLimitError) as ctx:
            client_no_wait.get("/rate-limited-endpoint")

        self.assertEqual(ctx.exception.reset_epoch, 1800000000)

    def test_live_get_quota_status(self):
        """Integration test: query real rate limit quota on GitHub API."""
        client = GitHubClient()
        status = client.get_quota_status()

        self.assertIn("limit", status)
        self.assertIn("remaining", status)
        self.assertIn("used", status)
        self.assertIn("percent_used", status)
        self.assertIn("reset_epoch", status)
        self.assertIn("reset_time_utc", status)
        self.assertGreater(status["limit"], 0)
        self.assertTrue(status["is_authenticated"])


if __name__ == "__main__":
    unittest.main()
