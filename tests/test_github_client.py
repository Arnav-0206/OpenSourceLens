"""Unit tests for AsyncGitHubClient.

Verifies:
1. Link header RFC 5988 parsing and malformed header resilience.
2. Retry-After header parsing (integer seconds and HTTP-date formats).
3. Concurrency throttling (max 2 concurrent requests).
4. Inter-request pacing (minimum 100 ms between requests).
5. Primary rate limit monitoring (X-RateLimit-Remaining < 10).
6. Retry-After handling:
   - Waits <= 300s are respected with full sleep and retry.
   - Waits > 300s halt run immediately and record scheduled resumption time.
7. Paginated traversal with budget tracking and truncation reporting.
8. HTTP error mapping (404, 401, transport timeouts).
"""

import asyncio
from datetime import datetime, timezone, timedelta
import time
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from backend.app.core.exceptions import (
    BudgetExceededException,
    EntityBudgetExceededException,
    GitHubAPIError,
    GitHubRateLimitExceededException,
    GitHubResourceNotFoundException,
)
from backend.app.services.github_client import AsyncGitHubClient
from backend.app.services.rate_budgeter import EntityRateBudgeter


class TestLinkAndHeaderParsing:
    """Test suite for Link and Retry-After header parsing."""

    def test_parse_link_header_standard(self):
        """Standard GitHub Link header with next, last, first, prev."""
        raw_header = (
            '<https://api.github.com/repos/pallets/flask/issues?page=2>; rel="next", '
            '<https://api.github.com/repos/pallets/flask/issues?page=5>; rel="last", '
            '<https://api.github.com/repos/pallets/flask/issues?page=1>; rel="first"'
        )
        links = AsyncGitHubClient.parse_link_header(raw_header)
        assert links["next"] == "https://api.github.com/repos/pallets/flask/issues?page=2"
        assert links["last"] == "https://api.github.com/repos/pallets/flask/issues?page=5"
        assert links["first"] == "https://api.github.com/repos/pallets/flask/issues?page=1"

    def test_parse_link_header_resilience(self):
        """Empty, None, or malformed Link headers return empty dict safely without crashing."""
        assert AsyncGitHubClient.parse_link_header(None) == {}
        assert AsyncGitHubClient.parse_link_header("") == {}
        assert AsyncGitHubClient.parse_link_header("   ") == {}
        assert AsyncGitHubClient.parse_link_header("invalid_link_format_without_rel") == {}
        assert AsyncGitHubClient.parse_link_header('<https://url.com>; something="else"') == {}

    def test_parse_retry_after_integer_seconds(self):
        """Integer seconds are parsed directly."""
        assert AsyncGitHubClient.parse_retry_after("60") == 60
        assert AsyncGitHubClient.parse_retry_after("  120  ") == 120
        assert AsyncGitHubClient.parse_retry_after("0") == 1

    def test_parse_retry_after_http_date(self):
        """HTTP-date format calculates seconds delta from current UTC time."""
        future_dt = datetime.now(timezone.utc) + timedelta(seconds=90)
        # Format as RFC 2822 / 7231 string: 'Wed, 21 Oct 2026 07:28:00 GMT'
        date_str = future_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
        delta = AsyncGitHubClient.parse_retry_after(date_str)
        assert delta is not None
        assert 80 <= delta <= 100

    def test_parse_retry_after_invalid_formats(self):
        """None, empty, or unparseable formats return None."""
        assert AsyncGitHubClient.parse_retry_after(None) is None
        assert AsyncGitHubClient.parse_retry_after("") is None
        assert AsyncGitHubClient.parse_retry_after("invalid_text") is None


class TestConcurrencyAndPacing:
    """Test suite for concurrency limiting (max 2) and pacing (>= 100 ms)."""

    def test_concurrency_throttling_max_2(self):
        """Simultaneous requests must not exceed concurrency ceiling of 2."""
        client = AsyncGitHubClient(max_concurrency=2, pacing_delay=0.0)

        active_concurrency = 0
        max_observed_concurrency = 0
        lock = asyncio.Lock()

        async def _mock_slow_request(*args, **kwargs):
            nonlocal active_concurrency, max_observed_concurrency
            async with lock:
                active_concurrency += 1
                if active_concurrency > max_observed_concurrency:
                    max_observed_concurrency = active_concurrency

            await asyncio.sleep(0.05)

            async with lock:
                active_concurrency -= 1

            return httpx.Response(
                status_code=200,
                json={"status": "ok"},
                request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
            )

        async def _run():
            with patch.object(httpx.AsyncClient, "request", side_effect=_mock_slow_request):
                # Launch 6 concurrent requests
                tasks = [
                    client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                    for _ in range(6)
                ]
                await asyncio.gather(*tasks)

        asyncio.run(_run())
        assert max_observed_concurrency <= 2

    def test_inter_request_pacing(self):
        """Pacing enforces at least 100 ms (0.1s) delay between consecutive requests."""
        client = AsyncGitHubClient(pacing_delay=0.1)

        mock_resp = httpx.Response(
            status_code=200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )

        async def _run():
            with patch.object(httpx.AsyncClient, "request", AsyncMock(return_value=mock_resp)):
                t0 = time.perf_counter()
                await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                elapsed = time.perf_counter() - t0
                return elapsed

        elapsed_time = asyncio.run(_run())
        assert elapsed_time >= 0.095  # Within timing tolerance of 100 ms


class TestRateLimitingAndRetries:
    """Test suite for primary and secondary rate limit handling."""

    def test_primary_rate_limit_monitoring(self):
        """X-RateLimit-Remaining < 10 trips rate limited flag and blocks subsequent calls."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        reset_ts = int(time.time()) + 1800

        mock_resp = httpx.Response(
            status_code=200,
            headers={
                "X-RateLimit-Remaining": "5",
                "X-RateLimit-Reset": str(reset_ts),
            },
            json={"data": "sample"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )

        async def _run():
            with patch.object(httpx.AsyncClient, "request", AsyncMock(return_value=mock_resp)):
                resp = await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                assert resp.status_code == 200
                assert client.is_primary_rate_limited is True
                assert client.resumes_at is not None

                # Next request should raise GitHubRateLimitExceededException immediately
                with pytest.raises(GitHubRateLimitExceededException) as exc_info:
                    await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                assert "Primary GitHub API rate limit" in exc_info.value.message

        asyncio.run(_run())

    def test_retry_after_under_300s_sleeps_and_retries(self):
        """Retry-After <= 300s sleeps full requested duration and retries successfully."""
        client = AsyncGitHubClient(pacing_delay=0.0, max_retries=2)

        rate_limit_resp = httpx.Response(
            status_code=429,
            headers={"Retry-After": "2"},
            json={"message": "Too Many Requests"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )
        success_resp = httpx.Response(
            status_code=200,
            json={"data": "success"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )

        async def _run():
            # Mock asyncio.sleep to record wait duration without delaying test run
            sleep_calls = []

            async def _fake_sleep(seconds):
                sleep_calls.append(seconds)

            with patch.object(
                httpx.AsyncClient,
                "request",
                AsyncMock(side_effect=[rate_limit_resp, success_resp]),
            ), patch("asyncio.sleep", side_effect=_fake_sleep):
                resp = await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                assert resp.status_code == 200
                assert len(sleep_calls) >= 1
                assert 2 in sleep_calls  # Slept full 2 seconds without arbitrary cap

        asyncio.run(_run())

    def test_retry_after_over_300s_halts_and_persists_resumes_at(self):
        """Retry-After > 300s raises GitHubRateLimitExceededException immediately without sleeping."""
        client = AsyncGitHubClient(pacing_delay=0.0)

        rate_limit_resp = httpx.Response(
            status_code=429,
            headers={"Retry-After": "450"},
            json={"message": "Abuse rate limit triggered"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask"),
        )

        async def _run():
            with patch.object(httpx.AsyncClient, "request", AsyncMock(return_value=rate_limit_resp)):
                with pytest.raises(GitHubRateLimitExceededException) as exc_info:
                    await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                assert exc_info.value.retry_after_seconds == 450
                assert client.resumes_at is not None
                expected_min_resume = datetime.now(timezone.utc) + timedelta(seconds=440)
                assert client.resumes_at >= expected_min_resume

        asyncio.run(_run())


class TestPaginationAndBudgeting:
    """Test suite for paginated collection and budget tracking."""

    def test_pagination_natural_completion(self):
        """Pagination traverses Link headers until no next relation exists."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="standard_diagnostic")

        page_1_resp = httpx.Response(
            status_code=200,
            headers={"Link": '<https://api.github.com/repos/pallets/flask/commits?page=2>; rel="next"'},
            json=[{"sha": "c1"}, {"sha": "c2"}],
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )
        page_2_resp = httpx.Response(
            status_code=200,
            headers={},  # No Link header -> remote exhaustion
            json=[{"sha": "c3"}],
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits?page=2"),
        )

        async def _run():
            with patch.object(
                httpx.AsyncClient, "request", AsyncMock(side_effect=[page_1_resp, page_2_resp])
            ):
                result = await client.paginate(
                    endpoint_or_url="/repos/pallets/flask/commits",
                    entity="commits",
                    budgeter=budgeter,
                )
                assert len(result.items) == 3
                assert result.total_requests == 2
                assert result.is_partial is False
                assert result.partial_reason is None
                assert budgeter.entity_used["commits"] == 2
                assert budgeter.total_used == 2

        asyncio.run(_run())

    def test_pagination_truncated_by_entity_budget(self):
        """Pagination stops gracefully when entity budget limit is reached, marking is_partial = True."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        # Fast snapshot allocates exactly 1 request for 'commits'
        budgeter = EntityRateBudgeter(mode="fast_snapshot")

        page_1_resp = httpx.Response(
            status_code=200,
            headers={"Link": '<https://api.github.com/repos/pallets/flask/commits?page=2>; rel="next"'},
            json=[{"sha": f"c_{i}"} for i in range(100)],
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )

        async def _run():
            with patch.object(httpx.AsyncClient, "request", AsyncMock(return_value=page_1_resp)):
                result = await client.paginate(
                    endpoint_or_url="/repos/pallets/flask/commits",
                    entity="commits",
                    budgeter=budgeter,
                )
                assert len(result.items) == 100
                assert result.total_requests == 1
                assert result.is_partial is True
                assert "budget exhausted" in result.partial_reason.lower()
                assert budgeter.entity_used["commits"] == 1

        asyncio.run(_run())

    def test_http_error_mappings(self):
        """Standard HTTP errors map to domain exception hierarchy."""
        client = AsyncGitHubClient(pacing_delay=0.0)

        async def _run():
            # 404
            resp_404 = httpx.Response(
                status_code=404,
                json={"message": "Not Found"},
                request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/releases/9999"),
            )
            with patch.object(httpx.AsyncClient, "request", AsyncMock(return_value=resp_404)):
                with pytest.raises(GitHubResourceNotFoundException) as exc_404:
                    await client.request("GET", "/repos/pallets/flask/releases/9999", entity="releases")
                assert exc_404.value.status_code == 404

            # 401
            resp_401 = httpx.Response(
                status_code=401,
                json={"message": "Bad credentials"},
                request=httpx.Request("GET", "https://api.github.com/user"),
            )
            with patch.object(httpx.AsyncClient, "request", AsyncMock(return_value=resp_401)):
                with pytest.raises(GitHubAPIError) as exc_401:
                    await client.request("GET", "/user", entity="repository_metadata")
                assert exc_401.value.status_code == 401

            # Transport timeout
            with patch.object(
                httpx.AsyncClient,
                "request",
                AsyncMock(side_effect=httpx.ConnectTimeout("Connection timed out")),
            ):
                with pytest.raises(GitHubAPIError) as exc_timeout:
                    await client.request("GET", "/repos/pallets/flask", entity="repository_metadata")
                assert exc_timeout.value.status_code == 502

        asyncio.run(_run())


class TestCompleteRequestBudgetAccounting:
    """Test suite ensuring all actual dispatched HTTP requests (initial + retries + pagination)

    strictly consume the collection budget and can NEVER exceed the hard budget ceiling.
    """

    def test_retry_after_429_consumes_budget_for_each_dispatched_attempt(self):
        """Retrying after a 429 response dispatches a 2nd request, consuming 2 budget units."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="standard_diagnostic")  # 'commits' limit = 3

        resp_429 = httpx.Response(
            status_code=429,
            headers={"Retry-After": "1"},
            json={"message": "Secondary rate limit"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )
        resp_200 = httpx.Response(
            status_code=200,
            json=[{"sha": "c1"}],
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )

        async def _run():
            mock_request = AsyncMock(side_effect=[resp_429, resp_200])
            with patch.object(httpx.AsyncClient, "request", mock_request), \
                 patch("asyncio.sleep", AsyncMock()):
                resp = await client.request(
                    "GET",
                    "/repos/pallets/flask/commits",
                    entity="commits",
                    budgeter=budgeter,
                )
                assert resp.status_code == 200
                assert mock_request.call_count == 2
                # Both attempts consumed budget
                assert budgeter.entity_used["commits"] == 2
                assert budgeter.total_used == 2

        asyncio.run(_run())

    def test_retry_blocked_when_budget_exhausted_enforcing_hard_ceiling(self):
        """If an initial request fails with 429 and quota has reached its ceiling, retry is blocked."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="fast_snapshot")  # 'commits' limit = 1

        resp_429 = httpx.Response(
            status_code=429,
            headers={"Retry-After": "1"},
            json={"message": "Rate limited"},
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )

        async def _run():
            mock_request = AsyncMock(return_value=resp_429)
            with patch.object(httpx.AsyncClient, "request", mock_request), \
                 patch("asyncio.sleep", AsyncMock()):
                with pytest.raises((BudgetExceededException, EntityBudgetExceededException)):
                    await client.request(
                        "GET",
                        "/repos/pallets/flask/commits",
                        entity="commits",
                        budgeter=budgeter,
                    )
                # Exactly 1 request dispatched; attempt 2 was BLOCKED by budget ceiling
                assert mock_request.call_count == 1
                assert budgeter.entity_used["commits"] == 1
                assert budgeter.total_used == 1

        asyncio.run(_run())

    def test_5xx_retry_consumes_budget_and_blocks_excess_dispatches(self):
        """Transient 5xx retries consume budget; dispatches cease immediately when ceiling is hit."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="fast_snapshot")  # 'commits' limit = 1

        resp_502 = httpx.Response(
            status_code=502,
            text="Bad Gateway",
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )

        async def _run():
            mock_request = AsyncMock(return_value=resp_502)
            with patch.object(httpx.AsyncClient, "request", mock_request), \
                 patch("asyncio.sleep", AsyncMock()):
                with pytest.raises((BudgetExceededException, EntityBudgetExceededException)):
                    await client.request(
                        "GET",
                        "/repos/pallets/flask/commits",
                        entity="commits",
                        budgeter=budgeter,
                    )
                # Exactly 1 request dispatched; retry was blocked
                assert mock_request.call_count == 1
                assert budgeter.entity_used["commits"] == 1
                assert budgeter.total_used == 1

        asyncio.run(_run())

    def test_transport_failure_retry_consumes_budget_and_blocks_excess_dispatches(self):
        """Transport error retries consume budget; dispatches cease immediately when ceiling is hit."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="fast_snapshot")  # 'commits' limit = 1

        async def _run():
            mock_request = AsyncMock(side_effect=httpx.ConnectTimeout("Timed out"))
            with patch.object(httpx.AsyncClient, "request", mock_request), \
                 patch("asyncio.sleep", AsyncMock()):
                with pytest.raises((BudgetExceededException, EntityBudgetExceededException)):
                    await client.request(
                        "GET",
                        "/repos/pallets/flask/commits",
                        entity="commits",
                        budgeter=budgeter,
                    )
                # Exactly 1 request dispatched
                assert mock_request.call_count == 1
                assert budgeter.entity_used["commits"] == 1
                assert budgeter.total_used == 1

        asyncio.run(_run())

    def test_pagination_with_retries_never_exceeds_budget(self):
        """Pagination combined with 5xx retries strictly respects entity and total budget ceilings."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="standard_diagnostic")  # 'commits' limit = 3

        # Page 1 attempt 0: 503 error (consumes request #1)
        resp_503 = httpx.Response(
            status_code=503,
            text="Service Unavailable",
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )
        # Page 1 attempt 1: 200 OK (consumes request #2)
        resp_page_1 = httpx.Response(
            status_code=200,
            headers={"Link": '<https://api.github.com/repos/pallets/flask/commits?page=2>; rel="next"'},
            json=[{"sha": "c1"}, {"sha": "c2"}],
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )
        # Page 2 attempt 0: 200 OK (consumes request #3)
        resp_page_2 = httpx.Response(
            status_code=200,
            headers={"Link": '<https://api.github.com/repos/pallets/flask/commits?page=3>; rel="next"'},
            json=[{"sha": "c3"}],
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits?page=2"),
        )
        # Page 3: MUST NOT BE DISPATCHED because all 3 requests are used!

        async def _run():
            mock_request = AsyncMock(side_effect=[resp_503, resp_page_1, resp_page_2])
            with patch.object(httpx.AsyncClient, "request", mock_request), \
                 patch("asyncio.sleep", AsyncMock()):
                result = await client.paginate(
                    endpoint_or_url="/repos/pallets/flask/commits",
                    entity="commits",
                    budgeter=budgeter,
                )
                # Verify exactly 3 HTTP requests were dispatched
                assert mock_request.call_count == 3
                assert result.total_requests == 3
                assert budgeter.entity_used["commits"] == 3
                assert budgeter.total_used == 3
                # Verify truncation was flagged gracefully
                assert result.is_partial is True
                assert "budget exhausted" in result.partial_reason.lower()
                # Verify items from successful pages were preserved
                assert len(result.items) == 3

        asyncio.run(_run())

    def test_fast_snapshot_hard_ceiling_10_requests_with_mixed_retries(self):
        """Simulate a multi-entity collection run with retries to prove total dispatches <= 10."""
        client = AsyncGitHubClient(pacing_delay=0.0)
        budgeter = EntityRateBudgeter(mode="fast_snapshot")  # Total ceiling = 10

        # Pre-consume 8 requests across valid fast_snapshot entities
        for ent in [
            "repository_metadata",
            "contributors",
            "pull_requests",
            "pr_reviews",
            "pr_reviews",
            "issues",
            "issue_comments",
            "releases",
        ]:
            budgeter.consume(ent, count=1)

        assert budgeter.total_used == 8

        # 2 requests remaining in total budget.
        # Now try to fetch 'commits' (limit=1 in fast_snapshot).
        # Commits attempt 0 fails with 502 (consumes request #9).
        # Attempt 1 should be blocked because commits limit is 1.
        resp_502 = httpx.Response(
            status_code=502,
            text="Bad Gateway",
            request=httpx.Request("GET", "https://api.github.com/repos/pallets/flask/commits"),
        )

        async def _run():
            mock_request = AsyncMock(return_value=resp_502)
            with patch.object(httpx.AsyncClient, "request", mock_request), \
                 patch("asyncio.sleep", AsyncMock()):
                with pytest.raises((BudgetExceededException, EntityBudgetExceededException)):
                    await client.request(
                        "GET",
                        "/repos/pallets/flask/commits",
                        entity="commits",
                        budgeter=budgeter,
                    )
                assert mock_request.call_count == 1
                assert budgeter.total_used == 9
                assert budgeter.total_used <= budgeter.total_limit

        asyncio.run(_run())
