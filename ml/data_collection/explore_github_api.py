"""GitHub REST API Exploration Script for OpenSourceLens (Step 1.1).

This script explores the GitHub REST API endpoints required for Phase 1:
- Rate limit checking and quota management
- Repository metadata schema (/repos/{owner}/{repo})
- Issues endpoint and Pull Request distinction (/repos/{owner}/{repo}/issues)
- Issue comments schema (/repos/{owner}/{repo}/issues/{number}/comments)
- Issue events schema (/repos/{owner}/{repo}/issues/{number}/events)
- Pagination headers (Link rel="next", rel="last")
- Label taxonomy comparisons across the 4 candidate repositories
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Ensure utf-8 output encoding for consoles (e.g. Windows cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "https://api.github.com"
CANDIDATE_REPOSITORIES = [
    ("microsoft", "vscode"),
    ("pytorch", "pytorch"),
    ("kubernetes", "kubernetes"),
    ("huggingface", "transformers"),
]


def get_headers() -> Dict[str, str]:
    """Build request headers with optional GitHub token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "OpenSourceLens-Collector/1.0",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token and token.strip() and not token.startswith("your_github_personal_access_token"):
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def format_timestamp(ts_epoch: int) -> str:
    """Format an epoch timestamp to ISO UTC string."""
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def check_rate_limit(session: requests.Session) -> Dict[str, Any]:
    """Query and print current GitHub API rate limit status."""
    print("=" * 70)
    print("1. RATE LIMIT STATUS (/rate_limit)")
    print("=" * 70)
    
    url = f"{BASE_URL}/rate_limit"
    response = session.get(url, headers=get_headers())
    
    if response.status_code != 200:
        print(f"Failed to check rate limit: HTTP {response.status_code}")
        print(response.text)
        return {}

    data = response.json()
    core = data.get("resources", {}).get("core", {})
    limit = core.get("limit", 0)
    remaining = core.get("remaining", 0)
    reset_epoch = core.get("reset", 0)
    reset_time = format_timestamp(reset_epoch)

    token_configured = "Authorization" in get_headers()
    auth_status = "AUTHENTICATED (Token active)" if token_configured else "UNAUTHENTICATED (IP-based limit)"

    print(f"Status:             {auth_status}")
    print(f"Hourly Limit:       {limit} requests")
    print(f"Remaining:          {remaining} requests")
    print(f"Resets At:          {reset_time}")

    if not token_configured:
        print("\n[NOTE] Running unauthenticated. Rate limit is 60 req/hr.")
        print("To increase to 5,000 req/hr, add GITHUB_TOKEN to your .env file.")
    print()
    return core


def inspect_repository(session: requests.Session, owner: str, repo: str) -> Optional[Dict[str, Any]]:
    """Inspect repository metadata endpoint."""
    print("=" * 70)
    print(f"2. REPOSITORY METADATA (/repos/{owner}/{repo})")
    print("=" * 70)

    url = f"{BASE_URL}/repos/{owner}/{repo}"
    response = session.get(url, headers=get_headers())

    if response.status_code != 200:
        print(f"Error fetching repo: HTTP {response.status_code}")
        return None

    repo_data = response.json()
    key_fields = {
        "repository_name": repo_data.get("name"),
        "full_name": repo_data.get("full_name"),
        "owner": repo_data.get("owner", {}).get("login"),
        "description": (repo_data.get("description") or "")[:80] + ("..." if len(repo_data.get("description") or "") > 80 else ""),
        "language": repo_data.get("language"),
        "stars": repo_data.get("stargazers_count"),
        "forks": repo_data.get("forks_count"),
        "open_issues_count": repo_data.get("open_issues_count"),
        "created_at": repo_data.get("created_at"),
        "updated_at": repo_data.get("updated_at"),
        "has_issues": repo_data.get("has_issues"),
    }

    print("Observed Key Metadata Fields:")
    for k, v in key_fields.items():
        print(f"  - {k}: {v}")
    print()
    return repo_data


def inspect_issues_and_pr_distinction(session: requests.Session, owner: str, repo: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Inspect issues endpoint and verify Pull Request vs Pure Issue distinction."""
    print("=" * 70)
    print(f"3. ISSUES ENDPOINT & PR FILTERING (/repos/{owner}/{repo}/issues)")
    print("=" * 70)

    # Note: By default, GitHub's /issues endpoint returns both Issues and Pull Requests!
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    params = {"state": "all", "per_page": 10, "sort": "created", "direction": "desc"}
    response = session.get(url, headers=get_headers(), params=params)

    if response.status_code != 200:
        print(f"Error fetching issues: HTTP {response.status_code}")
        return None, None

    items = response.json()
    print(f"Sampled {len(items)} items from /repos/{owner}/{repo}/issues (state=all):")

    sample_issue = None
    sample_pr = None

    for item in items:
        is_pr = "pull_request" in item
        item_type = "PULL REQUEST" if is_pr else "PURE ISSUE"
        number = item.get("number")
        title = item.get("title", "")[:50]
        state = item.get("state")
        labels = [l.get("name") for l in item.get("labels", [])]
        print(f"  [#{number}] [{item_type:<12}] [{state:<6}] Labels: {labels[:3]} | Title: {title}")

        if not is_pr and sample_issue is None:
            sample_issue = item
        if is_pr and sample_pr is None:
            sample_pr = item

    print("\n--- Key Finding: Issue vs Pull Request Schema ---")
    if sample_pr:
        print("  * Pull Request item contains key 'pull_request':")
        print(f"      'pull_request': {list(sample_pr['pull_request'].keys())}")
    if sample_issue:
        print("  * Pure Issue item does NOT contain 'pull_request' key.")
        print(f"  * Sample Issue #{sample_issue['number']} Schema Fields:")
        print(f"      - id: {sample_issue.get('id')}")
        print(f"      - number: {sample_issue.get('number')}")
        print(f"      - title: {sample_issue.get('title')}")
        print(f"      - state: {sample_issue.get('state')}")
        print(f"      - author: {sample_issue.get('user', {}).get('login')}")
        print(f"      - created_at: {sample_issue.get('created_at')}")
        print(f"      - closed_at: {sample_issue.get('closed_at')}")
        print(f"      - comments: {sample_issue.get('comments')}")
        print(f"      - body_length: {len(sample_issue.get('body') or '')} chars")
        print(f"      - labels count: {len(sample_issue.get('labels', []))}")
    print()
    return sample_issue, sample_pr


def inspect_comments(session: requests.Session, owner: str, repo: str, issue_number: int) -> None:
    """Inspect comments endpoint for a specific issue."""
    print("=" * 70)
    print(f"4. COMMENTS ENDPOINT (/repos/{owner}/{repo}/issues/{issue_number}/comments)")
    print("=" * 70)

    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    params = {"per_page": 2}
    response = session.get(url, headers=get_headers(), params=params)

    if response.status_code != 200:
        print(f"Error fetching comments: HTTP {response.status_code}")
        return

    comments = response.json()
    print(f"Retrieved {len(comments)} comment sample(s):")
    for c in comments:
        print(f"  - Comment ID: {c.get('id')}")
        print(f"    Author:     {c.get('user', {}).get('login')}")
        print(f"    Created:    {c.get('created_at')}")
        body = (c.get("body") or "").strip().replace("\n", " ")
        print(f"    Body snippet: {body[:80]}...")
    print()


def inspect_events(session: requests.Session, owner: str, repo: str, issue_number: int) -> None:
    """Inspect events endpoint for an issue."""
    print("=" * 70)
    print(f"5. EVENTS ENDPOINT (/repos/{owner}/{repo}/issues/{issue_number}/events)")
    print("=" * 70)

    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/events"
    params = {"per_page": 5}
    response = session.get(url, headers=get_headers(), params=params)

    if response.status_code != 200:
        print(f"Error fetching events: HTTP {response.status_code}")
        return

    events = response.json()
    if not events:
        # Try a closed issue which typically has close/label/mention events
        closed_resp = session.get(f"{BASE_URL}/repos/{owner}/{repo}/issues", headers=get_headers(), params={"state": "closed", "per_page": 1})
        if closed_resp.status_code == 200 and closed_resp.json():
            closed_num = closed_resp.json()[0].get("number")
            print(f"  (Issue #{issue_number} has 0 events; checking closed issue #{closed_num}...)")
            events_resp = session.get(f"{BASE_URL}/repos/{owner}/{repo}/issues/{closed_num}/events", headers=get_headers(), params={"per_page": 5})
            if events_resp.status_code == 200:
                events = events_resp.json()

    print(f"Retrieved {len(events)} event sample(s):")
    for ev in events:
        actor = ev.get("actor", {}).get("login") if ev.get("actor") else "system/none"
        event_type = ev.get("event")
        created_at = ev.get("created_at")
        commit_id = ev.get("commit_id")
        print(f"  - Event: '{event_type:<15}' by @{actor:<20} at {created_at} (commit: {commit_id})")
    print()


def inspect_pagination(session: requests.Session, owner: str, repo: str) -> None:
    """Inspect pagination mechanics using Link headers."""
    print("=" * 70)
    print(f"6. PAGINATION MECHANICS (/repos/{owner}/{repo}/issues)")
    print("=" * 70)

    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    params = {"state": "all", "per_page": 5, "page": 1}
    response = session.get(url, headers=get_headers(), params=params)

    link_header = response.headers.get("Link")
    print(f"HTTP Status: {response.status_code}")
    print(f"Raw 'Link' Header: {link_header}")

    if link_header:
        links = requests.utils.parse_header_links(link_header)
        print("\nParsed Pagination Links:")
        for l in links:
            print(f"  - rel='{l.get('rel')}': URL={l.get('url')}")
    else:
        print("No Link header returned (results fit in a single page or not provided).")
    print()


def inspect_label_taxonomies(session: requests.Session) -> None:
    """Inspect and compare label conventions across candidate repositories."""
    print("=" * 70)
    print("7. LABEL TAXONOMY COMPARISON ACROSS CANDIDATE REPOSITORIES")
    print("=" * 70)

    for owner, repo in CANDIDATE_REPOSITORIES:
        url = f"{BASE_URL}/repos/{owner}/{repo}/labels"
        params = {"per_page": 8}
        response = session.get(url, headers=get_headers(), params=params)
        if response.status_code == 200:
            labels = response.json()
            label_names = [l.get("name") for l in labels]
            print(f"[{owner}/{repo}]")
            print(f"  Sample Labels: {label_names}")
        else:
            print(f"[{owner}/{repo}] HTTP {response.status_code}")
    print()


def main():
    session = requests.Session()
    print("Starting OpenSourceLens GitHub API Exploration (Step 1.1)...")
    
    # 1. Rate Limit
    check_rate_limit(session)
    
    # 2. Repository Metadata on primary repo (vscode)
    inspect_repository(session, "microsoft", "vscode")
    
    # 3. Issues and PR distinction
    sample_issue, _ = inspect_issues_and_pr_distinction(session, "microsoft", "vscode")
    
    # 4. Comments (using sample issue if available)
    issue_number = sample_issue.get("number") if sample_issue else 1
    inspect_comments(session, "microsoft", "vscode", issue_number)
    
    # 5. Events
    inspect_events(session, "microsoft", "vscode", issue_number)
    
    # 6. Pagination
    inspect_pagination(session, "microsoft", "vscode")
    
    # 7. Label Taxonomies
    inspect_label_taxonomies(session)
    
    # Final rate limit check to report quota used
    print("=" * 70)
    print("FINAL QUOTA SUMMARY")
    print("=" * 70)
    final_limit = check_rate_limit(session)
    print("Step 1.1 API exploration script completed.")


if __name__ == "__main__":
    main()
