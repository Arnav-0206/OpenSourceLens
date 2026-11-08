"""Asynchronous GitHub REST API v3 Client.

Implements:
1. Strict concurrency throttling (max 2 concurrent requests per repository).
2. Inter-request pacing (minimum 100 ms between requests).
3. Complete Retry-After handling:
   - Waits <= 300s (5 min) are respected with full async sleep.
   - Waits > 300s halt collection gracefully and record resumption time.
   - No arbitrary 60-second caps.
4. Primary rate limit monitoring (X-RateLimit-Remaining < 10).
5. RFC 5988 Link header pagination with malformed link resilience.
6. Seamless integration with EntityRateBudgeter for partial collection tracking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional
import httpx

from backend.app.core.exceptions import (
    BudgetExceededException,
    EntityBudgetExceededException,
    GitHubAPIError,
    GitHubRateLimitExceededException,
    GitHubResourceNotFoundException,
)
from backend.app.schemas.ingestion import PaginatedResult
from backend.app.services.rate_budgeter import EntityRateBudgeter

logger = logging.getLogger(__name__)

# Regex pattern for RFC 5988 Link header parsing: <url>; rel="relation"
LINK_HEADER_PATTERN = re.compile(r'<([^>]+)>;\s*rel="([^"]+)"')


class AsyncGitHubClient:
    """Production asynchronous client for GitHub REST API v3."""

    def __init__(
        self,
        github_token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        max_concurrency: int = 2,
        pacing_delay: float = 0.1,  # 100 ms minimum inter-request delay
        max_retries: int = 3,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize GitHub client with rate-limiting, pacing, and concurrency controls.

        Args:
            github_token: Optional GitHub personal access token.
            base_url: GitHub API base URL.
            max_concurrency: Maximum concurrent requests per client instance (default 2).
            pacing_delay: Minimum seconds between consecutive requests (default 0.1s / 100ms).
            max_retries: Maximum automatic retries for transient errors or brief rate limits.
            client: Optional externally-managed httpx.AsyncClient.
        """
        self.github_token = github_token
        self.base_url = base_url.rstrip("/")
        self.max_concurrency = max_concurrency
        self.pacing_delay = pacing_delay
        self.max_retries = max_retries

        # Internal concurrency & pacing primitives
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._pacing_lock = asyncio.Lock()
        self._last_request_time: float = 0.0

        # Rate limit state
        self.is_primary_rate_limited: bool = False
        self.primary_rate_limit_reset: Optional[datetime] = None
        self.resumes_at: Optional[datetime] = None

        # Underlying HTTP client
        self._external_client = client

    def _get_default_headers(self) -> Dict[str, str]:
        """Construct standard GitHub REST API v3 headers."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OpenSourceLens-Collector/1.0",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    @staticmethod
    def parse_link_header(link_header: Optional[str]) -> Dict[str, str]:
        """Parse RFC 5988 Link header into a mapping of relation to URL.

        Resilient against missing, empty, or malformed Link headers.

        Args:
            link_header: Raw string from response 'Link' header.

        Returns:
            Dict mapping relations (e.g., 'next', 'last', 'prev') to target URLs.
        """
        if not link_header or not isinstance(link_header, str):
            return {}

        links: Dict[str, str] = {}
        # Each link component is separated by commas
        parts = link_header.split(",")
        for part in parts:
            match = LINK_HEADER_PATTERN.search(part.strip())
            if match:
                url, rel = match.groups()
                links[rel.strip().lower()] = url.strip()

        return links

    @staticmethod
    def parse_retry_after(header_val: Optional[str]) -> Optional[int]:
        """Parse Retry-After header into integer seconds (supports seconds or HTTP-date).

        Args:
            header_val: Raw Retry-After header value.

        Returns:
            Integer seconds to wait, or None if unparseable.
        """
        if not header_val or not header_val.strip():
            return None

        clean_val = header_val.strip()

        # Case 1: Simple integer seconds
        if clean_val.isdigit():
            return max(1, int(clean_val))

        # Case 2: RFC 7231 / RFC 2822 HTTP date
        try:
            target_dt = parsedate_to_datetime(clean_val)
            now_dt = datetime.now(timezone.utc)
            delta_seconds = int((target_dt - now_dt).total_seconds())
            return max(1, delta_seconds)
        except Exception:
            return None

    async def _apply_pacing(self) -> None:
        """Enforce minimum inter-request pacing delay."""
        async with self._pacing_lock:
            now = time.perf_counter()
            elapsed = now - self._last_request_time
            if elapsed < self.pacing_delay:
                await asyncio.sleep(self.pacing_delay - elapsed)
            self._last_request_time = time.perf_counter()

    async def request(
        self,
        method: str,
        endpoint_or_url: str,
        entity: str,
        budgeter: Optional[EntityRateBudgeter] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Any] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> httpx.Response:
        """Execute a paced, concurrency-bounded, budget-tracked GitHub API request.

        Args:
            method: HTTP method ('GET', 'POST', etc.).
            endpoint_or_url: Relative path ('/repos/owner/repo') or absolute URL.
            entity: Target entity for rate budget tracking ('commits', 'pull_requests', etc.).
            budgeter: Optional EntityRateBudgeter to consume quota from.
            params: Optional query string parameters.
            headers: Optional additional request headers.
            json_data: Optional JSON payload.
            client: Optional httpx.AsyncClient instance.

        Returns:
            httpx.Response: Successful HTTP response.

        Raises:
            GitHubRateLimitExceededException: If rate limit exceeded (> 300s or primary limit breached).
            GitHubResourceNotFoundException: If resource does not exist (404).
            GitHubAPIError: For authentication or other upstream errors.
            BudgetExceededException: If request budget is exhausted.
        """
        # Halt immediately if primary rate limits have been breached
        if self.is_primary_rate_limited:
            raise GitHubRateLimitExceededException(
                "Primary GitHub API rate limit threshold reached (remaining < 10). "
                "Halting further requests until quota resets.",
                retry_after_seconds=(
                    int((self.primary_rate_limit_reset - datetime.now(timezone.utc)).total_seconds())
                    if self.primary_rate_limit_reset
                    else 3600
                ),
                resumes_at=self.primary_rate_limit_reset,
            )

        # Build target URL
        if endpoint_or_url.startswith("http://") or endpoint_or_url.startswith("https://"):
            url = endpoint_or_url
        else:
            url = f"{self.base_url}/{endpoint_or_url.lstrip('/')}"

        # Merge headers
        req_headers = self._get_default_headers()
        if headers:
            req_headers.update(headers)

        active_client = client or self._external_client

        async def _execute_with_client(c: httpx.AsyncClient) -> httpx.Response:
            for attempt in range(self.max_retries):
                # Hard request-budget accounting: each actual outgoing HTTP request
                # (initial request or retry) consumes 1 request from the collection budget.
                if budgeter is not None:
                    # Raises BudgetExceededException or EntityBudgetExceededException if exhausted
                    budgeter.consume(entity, count=1)

                # Enforce concurrency limit and inter-request pacing
                async with self._semaphore:
                    await self._apply_pacing()
                    try:
                        resp = await c.request(
                            method=method,
                            url=url,
                            params=params,
                            headers=req_headers,
                            json=json_data,
                            timeout=20.0,
                        )
                    except httpx.RequestError as exc:
                        if attempt < self.max_retries - 1:
                            # Pre-check budget before sleeping and attempting retry
                            if budgeter is not None and not budgeter.can_consume(entity, count=1):
                                raise BudgetExceededException(
                                    f"Cannot retry transport failure on '{url}': collection request budget exhausted.",
                                    budget_mode=budgeter.budget_mode,
                                    total_used=budgeter.total_used,
                                    max_allowed=budgeter.total_limit,
                                )
                            backoff = 1.0 * (2 ** attempt) + random.uniform(0.1, 0.5)
                            logger.warning(
                                "Transient transport failure on %s (%s). Retrying in %.2fs (attempt %d/%d).",
                                url, exc, backoff, attempt + 1, self.max_retries
                            )
                            await asyncio.sleep(backoff)
                            continue
                        raise GitHubAPIError(
                            f"Transport failure connecting to GitHub API at '{url}': {str(exc)}",
                            status_code=502,
                        ) from exc

                # Inspect Primary Rate Limit headers
                remaining_header = resp.headers.get("X-RateLimit-Remaining")
                if remaining_header and remaining_header.isdigit():
                    remaining = int(remaining_header)
                    if remaining < 10:
                        self.is_primary_rate_limited = True
                        reset_header = resp.headers.get("X-RateLimit-Reset")
                        if reset_header and reset_header.isdigit():
                            self.primary_rate_limit_reset = datetime.fromtimestamp(
                                int(reset_header), tz=timezone.utc
                            )
                            self.resumes_at = self.primary_rate_limit_reset
                            logger.warning(
                                "Primary rate limit threshold reached: %d remaining. Resets at %s.",
                                remaining, self.primary_rate_limit_reset.isoformat()
                            )

                # Successful HTTP Response
                if resp.status_code in (200, 201, 204):
                    return resp

                # HTTP 404: Not Found
                if resp.status_code == 404:
                    raise GitHubResourceNotFoundException(
                        f"GitHub resource at '{url}' was not found.",
                        repo_name=entity,
                    )

                # HTTP 401: Unauthorized
                if resp.status_code == 401:
                    raise GitHubAPIError(
                        "GitHub API authentication failed. Verify configured token.",
                        status_code=401,
                    )

                # HTTP 403 or 429: Rate Limiting / Abuse Protection
                if resp.status_code in (403, 429):
                    raw_retry_after = resp.headers.get("Retry-After")
                    retry_seconds = self.parse_retry_after(raw_retry_after)

                    if retry_seconds is None:
                        # Fallback backoff starting at 60s with jitter if Retry-After is absent
                        retry_seconds = int(60.0 * (2 ** attempt) + random.uniform(1.0, 3.0))

                    logger.warning(
                        "GitHub rate limit encountered on %s (status %d, retry-after %ds).",
                        url, resp.status_code, retry_seconds
                    )

                    # Protocol requirement: if wait > 300s, calculate resumption and abort run
                    # (B.5 durable queue will persist resumes_at into checkpoint/job tables)
                    if retry_seconds > 300:
                        self.resumes_at = datetime.now(timezone.utc) + timedelta(seconds=retry_seconds)
                        raise GitHubRateLimitExceededException(
                            f"GitHub rate limit requires a wait of {retry_seconds}s (> 300s threshold). "
                            f"Halting collection run. Scheduled resumption at {self.resumes_at.isoformat()}.",
                            retry_after_seconds=retry_seconds,
                            resumes_at=self.resumes_at,
                        )

                    # If wait <= 300s, verify budget before sleeping and retrying
                    if attempt < self.max_retries - 1:
                        if budgeter is not None and not budgeter.can_consume(entity, count=1):
                            raise BudgetExceededException(
                                f"Cannot retry rate-limited request on '{url}': collection request budget exhausted.",
                                budget_mode=budgeter.budget_mode,
                                total_used=budgeter.total_used,
                                max_allowed=budgeter.total_limit,
                            )
                        logger.info("Sleeping full Retry-After duration of %d seconds.", retry_seconds)
                        await asyncio.sleep(retry_seconds)
                        continue

                    # Exhausted retry attempts
                    raise GitHubRateLimitExceededException(
                        f"Exhausted {self.max_retries} retries under GitHub rate limits.",
                        retry_after_seconds=retry_seconds,
                        resumes_at=self.resumes_at,
                    )

                # Transient 5xx Upstream Server Errors
                if resp.status_code in (500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        if budgeter is not None and not budgeter.can_consume(entity, count=1):
                            raise BudgetExceededException(
                                f"Cannot retry upstream {resp.status_code} on '{url}': collection request budget exhausted.",
                                budget_mode=budgeter.budget_mode,
                                total_used=budgeter.total_used,
                                max_allowed=budgeter.total_limit,
                            )
                        backoff = 1.5 * (2 ** attempt) + random.uniform(0.1, 0.5)
                        logger.warning(
                            "Upstream GitHub %d on %s. Retrying in %.2fs (attempt %d/%d).",
                            resp.status_code, url, backoff, attempt + 1, self.max_retries
                        )
                        await asyncio.sleep(backoff)
                        continue

                # Unhandled error response
                raise GitHubAPIError(
                    f"GitHub API returned error status {resp.status_code} for '{url}'.",
                    status_code=resp.status_code,
                    details={"url": url, "body": resp.text[:500]},
                )

            raise GitHubAPIError(f"Failed to execute request to '{url}' after {self.max_retries} retries.")

        if active_client is not None:
            return await _execute_with_client(active_client)
        else:
            async with httpx.AsyncClient() as new_client:
                return await _execute_with_client(new_client)

    async def paginate(
        self,
        endpoint_or_url: str,
        entity: str,
        budgeter: EntityRateBudgeter,
        params: Optional[Dict[str, Any]] = None,
        max_pages: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> PaginatedResult:
        """Traverse paginated GitHub endpoints following Link headers within budget limits.

        Args:
            endpoint_or_url: Relative endpoint path or initial URL.
            entity: Entity category name for budget accounting.
            budgeter: EntityRateBudgeter governing total and entity limits.
            params: Query parameters for initial request (e.g., {'per_page': 100}).
            max_pages: Optional ceiling on pages retrieved.
            client: Optional httpx.AsyncClient instance.

        Returns:
            PaginatedResult: Harvested items, total requests consumed, and truncation flags.
        """
        all_items: List[Dict[str, Any]] = []
        current_url: Optional[str] = endpoint_or_url
        current_params: Optional[Dict[str, Any]] = params
        is_partial: bool = False
        partial_reason: Optional[str] = None
        pages_retrieved: int = 0
        start_entity_used: int = budgeter.entity_used.get(entity, 0)

        while current_url:
            # Check 1: Stop if entity or overall budget would be breached
            if not budgeter.can_consume(entity, count=1):
                is_partial = True
                partial_reason = (
                    f"Request budget exhausted for '{entity}' "
                    f"(used: {budgeter.entity_used.get(entity, 0)}/{budgeter.entity_limits.get(entity, 0)}, "
                    f"total: {budgeter.total_used}/{budgeter.total_limit})."
                )
                break

            # Check 2: Stop if primary rate limits were breached
            if self.is_primary_rate_limited:
                is_partial = True
                partial_reason = "Primary GitHub API rate limit threshold reached (remaining < 10)."
                break

            # Execute page request
            try:
                resp = await self.request(
                    method="GET",
                    endpoint_or_url=current_url,
                    entity=entity,
                    budgeter=budgeter,
                    params=current_params,
                    client=client,
                )
            except (BudgetExceededException, EntityBudgetExceededException) as exc:
                is_partial = True
                partial_reason = str(exc.message)
                break

            pages_retrieved += 1

            # Parse page items
            data = resp.json()
            if isinstance(data, list):
                all_items.extend(data)
            elif isinstance(data, dict):
                # Some endpoints envelope items (e.g. search) or return single dict
                items = data.get("items")
                if isinstance(items, list):
                    all_items.extend(items)
                else:
                    all_items.append(data)

            # Check max_pages ceiling if configured
            if max_pages and pages_retrieved >= max_pages:
                link_header = resp.headers.get("Link")
                links = self.parse_link_header(link_header)
                if links.get("next"):
                    is_partial = True
                    partial_reason = f"Configured max_pages ceiling ({max_pages}) reached."
                break

            # Extract next page URL from Link header
            link_header = resp.headers.get("Link")
            links = self.parse_link_header(link_header)
            next_url = links.get("next")

            if next_url:
                current_url = next_url
                # Next URL already includes pagination query parameters
                current_params = None
            else:
                # No 'next' relation exists -> all remote data exhausted
                current_url = None

        requests_consumed = budgeter.entity_used.get(entity, 0) - start_entity_used
        return PaginatedResult(
            entity=entity,
            items=all_items,
            total_requests=requests_consumed,
            is_partial=is_partial,
            partial_reason=partial_reason,
            resumes_at=self.resumes_at,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client session if created."""
        if self._external_client and not self._external_client.is_closed:
            await self._external_client.aclose()

    async def __aenter__(self) -> AsyncGitHubClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

