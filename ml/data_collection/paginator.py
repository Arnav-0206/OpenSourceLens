"""GitHub API Pagination Engine for OpenSourceLens.

Handles RFC 5988 Link header parsing and page parameter traversal
to reliably stream thousands of items across any GitHub REST endpoint.
"""

import time
from typing import Dict, Any, Optional, Generator, List, Tuple
import requests

from ml.data_collection.github_client import GitHubClient, GitHubAPIError


class GitHubPaginator:
    """Pagination engine supporting Link-header traversal and generator streaming."""

    def __init__(self, client: Optional[GitHubClient] = None, polite_pause: float = 0.05):
        """Initialize with an optional GitHubClient and inter-request pause in seconds."""
        self.client = client or GitHubClient()
        self.polite_pause = polite_pause

    @staticmethod
    def parse_link_header(link_header: Optional[str]) -> Dict[str, str]:
        """Parse RFC 5988 Link header into a rel -> url dictionary.

        Example:
            '<https://...?page=2>; rel="next", <https://...?page=10>; rel="last"'
            -> {'next': 'https://...?page=2', 'last': 'https://...?page=10'}
        """
        if not link_header or not link_header.strip():
            return {}

        parsed_links = requests.utils.parse_header_links(link_header)
        result: Dict[str, str] = {}
        for link in parsed_links:
            rel = link.get("rel")
            url = link.get("url")
            if rel and url:
                result[rel] = url
        return result

    def paginate_pages(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: Optional[int] = None,
    ) -> Generator[Tuple[List[Dict[str, Any]], Dict[str, Any]], None, None]:
        """Stream pages of items from a GitHub REST endpoint.

        Args:
            endpoint: API endpoint path (e.g. '/repos/microsoft/vscode/issues') or full URL.
            params: Initial query parameters.
            max_pages: Optional maximum number of pages to retrieve.

        Yields:
            Tuples of (items_list, pagination_metadata_dict).
        """
        current_url: Optional[str] = endpoint
        current_params: Optional[Dict[str, Any]] = dict(params) if params else {}
        page_number = 1

        while current_url:
            if max_pages and page_number > max_pages:
                break

            response = self.client.get(current_url, params=current_params)
            raw_data = response.json()

            # Ensure response is a list of items (paginated collection)
            if not isinstance(raw_data, list):
                # If endpoint returned a single object or error payload, yield and stop
                yield [raw_data], {"page": page_number, "has_next": False}
                break

            items: List[Dict[str, Any]] = raw_data
            link_header = response.headers.get("Link")
            links = self.parse_link_header(link_header)

            next_url = links.get("next")
            has_next = bool(next_url)

            meta = {
                "page": page_number,
                "items_in_page": len(items),
                "has_next": has_next,
                "next_url": next_url,
                "last_url": links.get("last"),
            }

            yield items, meta

            if not has_next or len(items) == 0:
                break

            # For subsequent requests, the 'next' URL from Link header already
            # contains all necessary query parameters (page, per_page, state, etc.)
            current_url = next_url
            current_params = None  # Clear params since next_url has query string
            page_number += 1

            if self.polite_pause > 0:
                time.sleep(self.polite_pause)

    def paginate_items(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_items: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream individual items one by one across paginated pages.

        Args:
            endpoint: API endpoint path or URL.
            params: Initial query parameters.
            max_items: Maximum total items to yield.
            max_pages: Maximum pages to traverse.

        Yields:
            Individual raw item dictionaries.
        """
        count = 0
        for page_items, _ in self.paginate_pages(endpoint, params=params, max_pages=max_pages):
            for item in page_items:
                yield item
                count += 1
                if max_items and count >= max_items:
                    return
