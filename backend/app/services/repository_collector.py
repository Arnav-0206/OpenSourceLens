"""Universal Multi-Entity GitHub Repository Collector & Normalizer.

Implements Phase B Step B.4:
1. Multi-entity asynchronous harvesting (commits, contributors, PRs, reviews,
   inline comments, issues, conversation comments, releases) for ANY eligible public repo.
2. Complete rate budgeting through AsyncGitHubClient and EntityRateBudgeter.
3. Idempotent natural-key upserts across all entities.
4. Contributor identity resolution anchored on immutable GitHub user IDs without speculative merging.
5. Dual-parent linking with CHECK constraint for issue/PR comments.
6. SQL distinct count aggregation reconciliation for contributor metrics.
7. Multi-layered PR synchronization with lookback overlap and open/closed sweeps.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    BudgetExceededException,
    EntityBudgetExceededException,
    GitHubAPIError,
    GitHubRateLimitExceededException,
    GitHubResourceNotFoundException,
)
from backend.app.database.models import (
    CollectionRun,
    Commit,
    Contributor,
    ContributorActivity,
    Issue,
    IssueComment,
    IssueLabel,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
    Release,
    Repository,
)
from backend.app.schemas.ingestion import IngestionSummary, PaginatedResult
from backend.app.services.github_client import AsyncGitHubClient
from backend.app.services.rate_budgeter import EntityRateBudgeter
from ml.preprocessing.text_cleaner import IssueTextCleaner

logger = logging.getLogger(__name__)


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 / RFC 3339 datetime string into UTC timezone-aware datetime."""
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        clean_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean_str)
    except Exception:
        return None


def _datetimes_equal(d1: Optional[datetime], d2: Optional[datetime]) -> bool:
    """Compare two datetimes for equality, normalizing timezone and precision across dialects."""
    if d1 is None and d2 is None:
        return True
    if d1 is None or d2 is None:
        return False
    n1 = d1.astimezone(timezone.utc).replace(tzinfo=None) if d1.tzinfo is not None else d1
    n2 = d2.astimezone(timezone.utc).replace(tzinfo=None) if d2.tzinfo is not None else d2
    return abs((n1 - n2).total_seconds()) < 1e-3


def _floats_equal(f1: Optional[float], f2: Optional[float]) -> bool:
    """Compare two optional floats for equality within floating-point tolerance."""
    if f1 is None and f2 is None:
        return True
    if f1 is None or f2 is None:
        return False
    return abs(f1 - f2) < 1e-4


def extract_canonical_category(labels: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Detect canonical category and numeric ID from issue label list.

    Returns:
        Tuple of (canonical_category_name, canonical_id).
    """
    category_map = {
        "bug": 0,
        "feature": 1,
        "documentation": 2,
        "question": 3,
        "other": 4,
    }

    if not labels:
        return "other", 4

    label_names = [l.get("name", "").lower() for l in labels if isinstance(l, dict)]

    for name in label_names:
        if any(kw in name for kw in ("bug", "defect", "error", "crash", "fault")):
            return "bug", 0
        if any(kw in name for kw in ("feature", "enhancement", "feat", "request")):
            return "feature", 1
        if any(kw in name for kw in ("doc", "documentation")):
            return "documentation", 2
        if any(kw in name for kw in ("question", "support", "help")):
            return "question", 3

    return "other", 4


class RepositoryCollector:
    """Asynchronous multi-entity collector and normalizer for universal GitHub repos."""

    def __init__(
        self,
        github_token: Optional[str] = None,
        client: Optional[AsyncGitHubClient] = None,
        lookback_overlap_minutes: int = 60,
    ) -> None:
        self.github_token = github_token
        self.client = client or AsyncGitHubClient(github_token=github_token)
        self.lookback_overlap_minutes = lookback_overlap_minutes
        self._text_cleaner = IssueTextCleaner()

    async def collect_repository(
        self,
        repo_name: str,
        mode: str = "standard_diagnostic",
        db: Optional[Session] = None,
        budgeter: Optional[EntityRateBudgeter] = None,
        collection_run_id: Optional[int] = None,
    ) -> IngestionSummary:
        """Execute full multi-entity data collection for any eligible public GitHub repo.

        Args:
            repo_name: Canonical slug 'owner/repo'.
            mode: Collection mode ('fast_snapshot' or 'standard_diagnostic').
            db: Active SQLAlchemy database session.
            budgeter: Optional EntityRateBudgeter (instantiated if not provided).
            collection_run_id: Optional CollectionRun ID to update status and audit telemetry.

        Returns:
            IngestionSummary detailing harvested entity counts, execution duration, and partial flags.
        """
        t0 = time.perf_counter()
        clean_repo = repo_name.strip()
        active_budgeter = budgeter or EntityRateBudgeter(mode=mode)
        items_collected: Dict[str, int] = {
            "repository": 0,
            "commits": 0,
            "contributors": 0,
            "pull_requests": 0,
            "pr_reviews": 0,
            "pr_inline_comments": 0,
            "issues": 0,
            "issue_comments": 0,
            "releases": 0,
        }
        entity_status: Dict[str, str] = {
            "repository": "pending",
            "commits": "pending",
            "contributors": "pending",
            "pull_requests": "pending",
            "pr_reviews": "pending",
            "pr_inline_comments": "pending" if mode == "standard_diagnostic" else "skipped",
            "issues": "pending",
            "issue_comments": "pending",
            "releases": "pending",
        }
        errors: List[str] = []
        is_partial = False
        partial_reason: Optional[str] = None
        resumes_at: Optional[datetime] = None
        mutated_records: Set[Tuple[str, Any]] = set()
        repo: Optional[Repository] = None

        if db is None:
            raise ValueError("Active SQLAlchemy database session (db) is required for collection.")

        try:
            # 1. Repository Metadata
            repo = await self.collect_repository_metadata(
                repo_name=clean_repo,
                budgeter=active_budgeter,
                db=db,
                mutated_records=mutated_records,
            )
            items_collected["repository"] = 1
            entity_status["repository"] = "completed"
            last_synced = repo.last_synced_at

            # 2. Commits
            c_count, c_part, c_reason = await self.collect_commits(
                repo=repo,
                budgeter=active_budgeter,
                db=db,
                since=last_synced,
                mode=mode,
                mutated_records=mutated_records,
            )
            items_collected["commits"] = c_count
            entity_status["commits"] = "truncated" if c_part else "completed"
            if c_part:
                is_partial = True
                partial_reason = partial_reason or c_reason

            # 3. Contributors
            contrib_count, contrib_part, contrib_reason = await self.collect_contributors(
                repo=repo,
                budgeter=active_budgeter,
                db=db,
                mutated_records=mutated_records,
            )
            items_collected["contributors"] = contrib_count
            entity_status["contributors"] = "truncated" if contrib_part else "completed"
            if contrib_part:
                is_partial = True
                partial_reason = partial_reason or contrib_reason

            # 4. Pull Requests
            pr_list, pr_count, pr_part, pr_reason = await self.collect_pull_requests(
                repo=repo,
                budgeter=active_budgeter,
                db=db,
                last_synced_at=last_synced,
                mutated_records=mutated_records,
            )
            items_collected["pull_requests"] = pr_count
            entity_status["pull_requests"] = "truncated" if pr_part else "completed"
            if pr_part:
                is_partial = True
                partial_reason = partial_reason or pr_reason

            # 5. PR Reviews
            review_count, r_part, r_reason = await self.collect_pr_reviews(
                repo=repo,
                pull_requests=pr_list,
                budgeter=active_budgeter,
                db=db,
                mode=mode,
                mutated_records=mutated_records,
            )
            items_collected["pr_reviews"] = review_count
            entity_status["pr_reviews"] = "truncated" if r_part else "completed"
            if r_part:
                is_partial = True
                partial_reason = partial_reason or r_reason

            # 6. PR Inline Review Comments (Repo-wide diff comments)
            if mode == "standard_diagnostic" and active_budgeter.can_consume("pr_inline_comments", 1):
                inline_count, in_part, in_reason = await self.collect_pr_inline_comments(
                    repo=repo,
                    budgeter=active_budgeter,
                    db=db,
                    mutated_records=mutated_records,
                )
                items_collected["pr_inline_comments"] = inline_count
                entity_status["pr_inline_comments"] = "truncated" if in_part else "completed"
                if in_part:
                    is_partial = True
                    partial_reason = partial_reason or in_reason
            elif mode == "standard_diagnostic":
                entity_status["pr_inline_comments"] = "skipped"

            # 7. Issues (filtering out PRs)
            issue_count, iss_part, iss_reason = await self.collect_issues(
                repo=repo,
                budgeter=active_budgeter,
                db=db,
                since=last_synced,
                mutated_records=mutated_records,
            )
            items_collected["issues"] = issue_count
            entity_status["issues"] = "truncated" if iss_part else "completed"
            if iss_part:
                is_partial = True
                partial_reason = partial_reason or iss_reason

            # 8. Issue & PR Conversation Comments
            comm_count, comm_part, comm_reason = await self.collect_issue_comments(
                repo=repo,
                budgeter=active_budgeter,
                db=db,
                since=last_synced,
                mutated_records=mutated_records,
            )
            items_collected["issue_comments"] = comm_count
            entity_status["issue_comments"] = "truncated" if comm_part else "completed"
            if comm_part:
                is_partial = True
                partial_reason = partial_reason or comm_reason

            # 9. Releases
            rel_count, rel_part, rel_reason = await self.collect_releases(
                repo=repo,
                budgeter=active_budgeter,
                db=db,
                mutated_records=mutated_records,
            )
            items_collected["releases"] = rel_count
            entity_status["releases"] = "truncated" if rel_part else "completed"
            if rel_part:
                is_partial = True
                partial_reason = partial_reason or rel_reason

            # 10. Contributor SQL Metric Reconciliation (eliminates counter inflation)
            self.reconcile_contributor_metrics(
                repository_id=repo.id,
                db=db,
                mutated_records=mutated_records,
            )

            # Check if client tripped primary rate limit threshold during collection
            if getattr(self.client, "is_primary_rate_limited", False):
                is_partial = True
                partial_reason = partial_reason or "Primary GitHub API rate limit threshold reached (remaining < 10)."
                resumes_at = resumes_at or getattr(self.client, "resumes_at", None)

            # Update repository sync cursor
            repo.last_synced_at = datetime.now(timezone.utc)
            if repo is not None and len(mutated_records) > 0:
                repo.analytics_source_revision += 1
            db.commit()

        except (BudgetExceededException, EntityBudgetExceededException) as exc:
            is_partial = True
            partial_reason = partial_reason or str(exc.message)
            logger.info("Collection run reached request budget limit: %s", exc)
            for k, v in entity_status.items():
                if v == "pending":
                    entity_status[k] = "skipped"
            if repo is not None and len(mutated_records) > 0:
                repo.analytics_source_revision += 1
            db.commit()
        except GitHubRateLimitExceededException as exc:
            is_partial = True
            partial_reason = partial_reason or str(exc.message)
            resumes_at = exc.resumes_at or self.client.resumes_at
            logger.warning("Collection run rate limited by GitHub: %s. Resumes at %s", exc, resumes_at)
            for k, v in entity_status.items():
                if v == "pending":
                    entity_status[k] = "skipped"
            if repo is not None and len(mutated_records) > 0:
                repo.analytics_source_revision += 1
            db.commit()
        except Exception as exc:
            logger.exception("Unexpected error during repository collection: %s", exc)
            errors.append(str(exc))
            for k, v in entity_status.items():
                if v == "pending":
                    entity_status[k] = "failed"
            db.rollback()

        duration = round(time.perf_counter() - t0, 3)

        # Audit collection run record if requested
        if collection_run_id and db is not None:
            c_run = db.scalar(select(CollectionRun).where(CollectionRun.id == collection_run_id))
            if c_run:
                if resumes_at is not None:
                    c_run.status = "rate_limited"
                elif is_partial:
                    c_run.status = "completed_partial"
                elif errors:
                    c_run.status = "failed"
                else:
                    c_run.status = "completed"
                c_run.is_partial = is_partial
                c_run.coverage_summary = entity_status
                c_run.items_collected = items_collected
                c_run.rate_limit_consumed = active_budgeter.total_used
                c_run.completed_at = datetime.now(timezone.utc)
                if errors:
                    c_run.error_message = "; ".join(errors)
                db.commit()

        return IngestionSummary(
            repo_name=clean_repo,
            mode=mode,
            is_partial=is_partial,
            partial_reason=partial_reason,
            resumes_at=resumes_at,
            total_requests_consumed=active_budgeter.total_used,
            items_collected=items_collected,
            entity_status=entity_status,
            records_mutated=len(mutated_records),
            execution_time_seconds=duration,
            errors=errors,
        )

    async def collect_repository_metadata(
        self,
        repo_name: str,
        budgeter: EntityRateBudgeter,
        db: Session,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Repository:
        """Fetch and idempotently upsert repository root metadata."""
        owner, name = repo_name.split("/")
        endpoint = f"/repos/{owner}/{name}"

        resp = await self.client.request(
            method="GET",
            endpoint_or_url=endpoint,
            entity="repository_metadata",
            budgeter=budgeter,
        )
        data = resp.json()
        repo_id = int(data["id"])

        # Idempotent match by repo_name or GitHub ID
        repo = db.scalar(
            select(Repository).where(
                or_(Repository.id == repo_id, Repository.repo_name == repo_name)
            )
        )

        created_dt = parse_datetime(data.get("created_at"))
        updated_dt = parse_datetime(data.get("updated_at"))

        new_lang = data.get("language")
        new_stars = data.get("stargazers_count", 0)
        new_forks = data.get("forks_count", 0)
        new_open_issues = data.get("open_issues_count", 0)
        new_is_archived = bool(data.get("archived", False))
        new_is_fork = bool(data.get("fork", False))

        if repo is None:
            repo = Repository(
                id=repo_id,
                repo_name=repo_name,
                owner=owner,
                name=name,
                description=data.get("description"),
                primary_language=new_lang,
                stars_count=new_stars,
                forks_count=new_forks,
                open_issues_count=new_open_issues,
                github_created_at=created_dt,
                github_updated_at=updated_dt,
                is_active=True,
                is_archived=new_is_archived,
                is_fork=new_is_fork,
                upstream_repo=(data.get("parent") or {}).get("full_name") if new_is_fork else None,
            )
            db.add(repo)
            db.flush()
            if mutated_records is not None:
                mutated_records.add(("repository", repo.id))
        else:
            analytical_changed = (
                repo.primary_language != new_lang
                or repo.stars_count != new_stars
                or repo.forks_count != new_forks
                or repo.open_issues_count != new_open_issues
                or repo.is_archived != new_is_archived
                or repo.is_fork != new_is_fork
            )
            if analytical_changed and mutated_records is not None:
                mutated_records.add(("repository", repo.id))

            repo.id = repo_id
            repo.repo_name = repo_name
            repo.owner = owner
            repo.name = name
            repo.description = data.get("description", repo.description)
            repo.primary_language = new_lang
            repo.stars_count = new_stars
            repo.forks_count = new_forks
            repo.open_issues_count = new_open_issues
            repo.github_updated_at = updated_dt or repo.github_updated_at
            repo.is_archived = new_is_archived
            repo.is_fork = new_is_fork
            repo.upstream_repo = (data.get("parent") or {}).get("full_name") if new_is_fork else None
            db.flush()

        return repo

    async def collect_commits(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        since: Optional[datetime] = None,
        mode: str = "standard_diagnostic",
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest commits and sample commit detail stats within budget."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/commits"
        params: Dict[str, Any] = {"per_page": 100}
        if since:
            params["since"] = since.isoformat()

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="commits",
            budgeter=budgeter,
            params=params,
        )

        inserted_count = 0
        sample_commits_for_details: List[str] = []

        for item in res.items:
            sha = item.get("sha")
            if not sha:
                continue

            commit_data = item.get("commit") or {}
            author_data = commit_data.get("author") or {}
            author_name = author_data.get("name")
            author_email = author_data.get("email")
            author_date = parse_datetime(author_data.get("date")) or datetime.now(timezone.utc)
            committer_data = commit_data.get("committer") or {}
            committer_date = parse_datetime(committer_data.get("date"))
            message = commit_data.get("message")

            user_data = item.get("author") or {}
            author_login = user_data.get("login")
            author_id = user_data.get("id")

            existing = db.scalar(
                select(Commit).where(
                    Commit.repository_id == repo.id,
                    Commit.sha == sha,
                )
            )

            if existing is None:
                record = Commit(
                    repository_id=repo.id,
                    sha=sha,
                    author_name=author_name,
                    author_email=author_email,
                    author_login=author_login,
                    author_id=author_id,
                    author_date=author_date,
                    committer_date=committer_date,
                    message=message,
                )
                db.add(record)
                inserted_count += 1
                if mutated_records is not None:
                    mutated_records.add(("commit", sha))
                if len(sample_commits_for_details) < 5:
                    sample_commits_for_details.append(sha)
            else:
                changed = (
                    existing.author_name != author_name
                    or existing.author_email != author_email
                    or existing.author_login != author_login
                    or existing.author_id != author_id
                    or not _datetimes_equal(existing.author_date, author_date)
                    or not _datetimes_equal(existing.committer_date, committer_date)
                    or existing.message != message
                )
                if changed:
                    existing.author_name = author_name
                    existing.author_email = author_email
                    existing.author_login = author_login
                    existing.author_id = author_id
                    existing.author_date = author_date
                    existing.committer_date = committer_date
                    existing.message = message
                    if mutated_records is not None:
                        mutated_records.add(("commit", sha))

            # Link activity if contributor exists or can be anchored
            self._record_activity_if_contributor_exists(
                repo_id=repo.id,
                author_id=author_id,
                author_login=author_login,
                author_email=author_email,
                activity_type="commit",
                activity_date=author_date,
                entity_id=sha,
                db=db,
                mutated_records=mutated_records,
            )

        db.flush()

        # Sample commit details for up to 5 commits if in standard_diagnostic
        if mode == "standard_diagnostic" and sample_commits_for_details:
            for sha in sample_commits_for_details:
                if not budgeter.can_consume("commit_details", count=1):
                    break
                try:
                    c_resp = await self.client.request(
                        method="GET",
                        endpoint_or_url=f"/repos/{repo.owner}/{repo.name}/commits/{sha}",
                        entity="commit_details",
                        budgeter=budgeter,
                    )
                    c_stats = c_resp.json().get("stats") or {}
                    cmt = db.scalar(
                        select(Commit).where(Commit.repository_id == repo.id, Commit.sha == sha)
                    )
                    if cmt:
                        new_add = c_stats.get("additions")
                        new_del = c_stats.get("deletions")
                        new_tot = c_stats.get("total")
                        if (
                            cmt.additions != new_add
                            or cmt.deletions != new_del
                            or cmt.total_changes != new_tot
                        ):
                            cmt.additions = new_add
                            cmt.deletions = new_del
                            cmt.total_changes = new_tot
                            if mutated_records is not None:
                                mutated_records.add(("commit", sha))
                except Exception as exc:
                    logger.debug("Could not fetch detailed stats for commit %s: %s", sha, exc)

        db.flush()
        return inserted_count, res.is_partial, res.partial_reason

    async def collect_contributors(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest contributors anchoring on immutable GitHub user IDs without speculative merging."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/contributors"
        params = {"per_page": 100}

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="contributors",
            budgeter=budgeter,
            params=params,
        )

        count = 0
        for item in res.items:
            github_id = item.get("id")
            login = item.get("login")
            if not login and not github_id:
                continue

            avatar_url = item.get("avatar_url")
            user_type = item.get("type", "User")
            is_bot = (user_type == "Bot") or (login and login.endswith("[bot]"))

            contributor: Optional[Contributor] = None
            if github_id is not None:
                contributor = db.scalar(
                    select(Contributor).where(
                        Contributor.repository_id == repo.id,
                        Contributor.github_id == github_id,
                    )
                )

            # Check if there is an unlinked contributor with the same login needing upgrade
            if contributor is None and login:
                unlinked = db.scalar(
                    select(Contributor).where(
                        Contributor.repository_id == repo.id,
                        Contributor.login == login,
                        Contributor.github_id.is_(None),
                    )
                )
                if unlinked is not None:
                    changed = (
                        unlinked.github_id != github_id
                        or (avatar_url and unlinked.avatar_url != avatar_url)
                        or unlinked.is_bot != is_bot
                        or unlinked.is_unlinked is not False
                        or not _floats_equal(unlinked.identity_confidence, 1.0)
                    )
                    unlinked.github_id = github_id
                    unlinked.avatar_url = avatar_url or unlinked.avatar_url
                    unlinked.is_bot = is_bot
                    unlinked.is_unlinked = False
                    unlinked.identity_confidence = 1.0
                    contributor = unlinked
                    if changed and mutated_records is not None:
                        mutated_records.add(("contributor", unlinked.id))

            if contributor is None:
                # Insert verified contributor anchored on GitHub ID
                contributor = Contributor(
                    repository_id=repo.id,
                    github_id=github_id,
                    login=login,
                    avatar_url=avatar_url,
                    is_bot=is_bot,
                    is_unlinked=False,
                    identity_confidence=1.0,
                )
                db.add(contributor)
                db.flush()
                count += 1
                if mutated_records is not None:
                    mutated_records.add(("contributor", contributor.id))
            else:
                # Update username if changed on GitHub while preserving immutable GitHub ID
                changed = (
                    contributor.login != login
                    or contributor.avatar_url != avatar_url
                    or contributor.is_bot != is_bot
                )
                if changed:
                    contributor.login = login
                    contributor.avatar_url = avatar_url
                    contributor.is_bot = is_bot
                    if mutated_records is not None:
                        mutated_records.add(("contributor", contributor.id))

        db.flush()
        return count, res.is_partial, res.partial_reason

    async def collect_pull_requests(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        last_synced_at: Optional[datetime] = None,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[List[PullRequest], int, bool, Optional[str]]:
        """Harvest pull requests using lookback overlap and open/closed reconciliation sweeps."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/pulls"
        params = {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100}

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="pull_requests",
            budgeter=budgeter,
            params=params,
        )

        pr_records: List[PullRequest] = []
        count = 0

        for item in res.items:
            pr_num = item.get("number")
            if not pr_num:
                continue

            title = item.get("title", "")
            state = item.get("state", "open").lower()
            user_data = item.get("user") or {}
            author = user_data.get("login")
            author_id = user_data.get("id")

            created_at = parse_datetime(item.get("created_at")) or datetime.now(timezone.utc)
            closed_at = parse_datetime(item.get("closed_at"))
            merged_at = parse_datetime(item.get("merged_at"))

            # Calculate turnaround duration
            time_to_merge_hours: Optional[float] = None
            if merged_at and created_at:
                diff_sec = (merged_at - created_at).total_seconds()
                time_to_merge_hours = round(max(0.0, diff_sec / 3600.0), 2)

            comments_count = item.get("comments", 0)
            review_comments_count = item.get("review_comments", 0)
            additions = item.get("additions")
            deletions = item.get("deletions")
            changed_files = item.get("changed_files")

            pr = db.scalar(
                select(PullRequest).where(
                    PullRequest.repository_id == repo.id,
                    PullRequest.pr_number == pr_num,
                )
            )

            if pr is None:
                pr = PullRequest(
                    repository_id=repo.id,
                    pr_number=pr_num,
                    title=title,
                    state=state,
                    author=author,
                    author_id=author_id,
                    created_at=created_at,
                    closed_at=closed_at,
                    merged_at=merged_at,
                    time_to_merge_hours=time_to_merge_hours,
                    comments_count=comments_count,
                    review_comments_count=review_comments_count,
                    additions=additions,
                    deletions=deletions,
                    changed_files=changed_files,
                )
                db.add(pr)
                count += 1
                if mutated_records is not None:
                    mutated_records.add(("pull_request", pr_num))
            else:
                changed = (
                    pr.title != title
                    or pr.state != state
                    or pr.author != author
                    or pr.author_id != author_id
                    or not _datetimes_equal(pr.closed_at, closed_at)
                    or not _datetimes_equal(pr.merged_at, merged_at)
                    or not _floats_equal(pr.time_to_merge_hours, time_to_merge_hours)
                    or pr.comments_count != comments_count
                    or pr.review_comments_count != review_comments_count
                    or (additions is not None and pr.additions != additions)
                    or (deletions is not None and pr.deletions != deletions)
                    or (changed_files is not None and pr.changed_files != changed_files)
                )
                if changed:
                    pr.title = title
                    pr.state = state
                    pr.author = author
                    pr.author_id = author_id
                    pr.closed_at = closed_at
                    pr.merged_at = merged_at
                    pr.time_to_merge_hours = time_to_merge_hours
                    pr.comments_count = comments_count
                    pr.review_comments_count = review_comments_count
                    if additions is not None:
                        pr.additions = additions
                    if deletions is not None:
                        pr.deletions = deletions
                    if changed_files is not None:
                        pr.changed_files = changed_files
                    if mutated_records is not None:
                        mutated_records.add(("pull_request", pr_num))

            pr_records.append(pr)

            # Record activities
            self._record_activity_if_contributor_exists(
                repo_id=repo.id,
                author_id=author_id,
                author_login=author,
                author_email=None,
                activity_type="pr_opened",
                activity_date=created_at,
                entity_id=str(pr_num),
                db=db,
                mutated_records=mutated_records,
            )
            if merged_at:
                self._record_activity_if_contributor_exists(
                    repo_id=repo.id,
                    author_id=author_id,
                    author_login=author,
                    author_email=None,
                    activity_type="pr_merged",
                    activity_date=merged_at,
                    entity_id=str(pr_num),
                    db=db,
                    mutated_records=mutated_records,
                )

        db.flush()
        return pr_records, count, res.is_partial, res.partial_reason

    async def collect_pr_reviews(
        self,
        repo: Repository,
        pull_requests: List[PullRequest],
        budgeter: EntityRateBudgeter,
        db: Session,
        mode: str = "standard_diagnostic",
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest review decision envelopes on sampled PRs within quota."""
        max_pr_samples = 2 if mode == "fast_snapshot" else 20
        # Target recent PRs with review activity
        sampled_prs = [p for p in pull_requests if p.review_comments_count > 0 or p.closed_at][:max_pr_samples]
        if not sampled_prs:
            sampled_prs = pull_requests[:max_pr_samples]

        review_count = 0
        is_partial = False
        partial_reason = None

        for pr in sampled_prs:
            if not budgeter.can_consume("pr_reviews", count=1):
                is_partial = True
                partial_reason = "PR review entity budget exhausted."
                break

            endpoint = f"/repos/{repo.owner}/{repo.name}/pulls/{pr.pr_number}/reviews"
            try:
                res = await self.client.request(
                    method="GET",
                    endpoint_or_url=endpoint,
                    entity="pr_reviews",
                    budgeter=budgeter,
                )
                reviews_data = res.json()
                if not isinstance(reviews_data, list):
                    continue

                for r_item in reviews_data:
                    review_id = r_item.get("id")
                    if not review_id:
                        continue

                    user_data = r_item.get("user") or {}
                    author = user_data.get("login")
                    author_id = user_data.get("id")
                    state = r_item.get("state", "COMMENTED")
                    submitted_at = parse_datetime(r_item.get("submitted_at"))
                    body = r_item.get("body")

                    rev = db.scalar(
                        select(PullRequestReview).where(
                            PullRequestReview.review_id == review_id
                        )
                    )

                    if rev is None:
                        rev = PullRequestReview(
                            pull_request_id=pr.id,
                            review_id=review_id,
                            author=author,
                            author_id=author_id,
                            state=state,
                            submitted_at=submitted_at,
                            body=body,
                        )
                        db.add(rev)
                        review_count += 1
                        if mutated_records is not None:
                            mutated_records.add(("pr_review", review_id))
                    else:
                        changed = (
                            rev.state != state
                            or not _datetimes_equal(rev.submitted_at, submitted_at)
                            or rev.body != body
                        )
                        if changed:
                            rev.state = state
                            rev.submitted_at = submitted_at
                            rev.body = body
                            if mutated_records is not None:
                                mutated_records.add(("pr_review", review_id))

                    # Update PR time_to_first_review_hours
                    if submitted_at and pr.created_at:
                        diff_h = (submitted_at - pr.created_at).total_seconds() / 3600.0
                        if diff_h >= 0:
                            calc_hours = round(diff_h, 2)
                            target_hours = (
                                calc_hours
                                if pr.time_to_first_review_hours is None
                                else min(pr.time_to_first_review_hours, calc_hours)
                            )
                            if pr.time_to_first_review_hours != target_hours:
                                pr.time_to_first_review_hours = target_hours
                                if mutated_records is not None:
                                    mutated_records.add(("pull_request", pr.pr_number))

                    if submitted_at:
                        self._record_activity_if_contributor_exists(
                            repo_id=repo.id,
                            author_id=author_id,
                            author_login=author,
                            author_email=None,
                            activity_type="review_submitted",
                            activity_date=submitted_at,
                            entity_id=str(review_id),
                            db=db,
                            mutated_records=mutated_records,
                        )

            except (BudgetExceededException, EntityBudgetExceededException) as exc:
                is_partial = True
                partial_reason = str(exc.message)
                break
            except Exception as exc:
                logger.debug("Failed to harvest reviews for PR #%d: %s", pr.pr_number, exc)

        db.flush()
        return review_count, is_partial, partial_reason

    async def collect_pr_inline_comments(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest repo-wide diff inline comments tied to specific code lines."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/pulls/comments"
        params = {"sort": "updated", "direction": "desc", "per_page": 100}

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="pr_inline_comments",
            budgeter=budgeter,
            params=params,
        )

        count = 0
        for item in res.items:
            comment_id = item.get("id")
            if not comment_id:
                continue

            pr_url = item.get("pull_request_url", "")
            pr_num_str = pr_url.rstrip("/").split("/")[-1] if pr_url else None
            if not pr_num_str or not pr_num_str.isdigit():
                continue
            pr_num = int(pr_num_str)

            pr = db.scalar(
                select(PullRequest).where(
                    PullRequest.repository_id == repo.id,
                    PullRequest.pr_number == pr_num,
                )
            )
            if not pr:
                continue

            user_data = item.get("user") or {}
            author = user_data.get("login")
            author_id = user_data.get("id")
            path = item.get("path")
            line = item.get("line") or item.get("original_line")
            commit_id = item.get("commit_id")
            body = item.get("body")
            created_at = parse_datetime(item.get("created_at")) or datetime.now(timezone.utc)
            updated_at = parse_datetime(item.get("updated_at"))

            inline = db.scalar(
                select(PullRequestReviewComment).where(
                    PullRequestReviewComment.comment_id == comment_id
                )
            )

            if inline is None:
                inline = PullRequestReviewComment(
                    pull_request_id=pr.id,
                    comment_id=comment_id,
                    author=author,
                    author_id=author_id,
                    path=path,
                    line=line,
                    commit_id=commit_id,
                    body=body,
                    created_at=created_at,
                    updated_at=updated_at,
                )
                db.add(inline)
                count += 1
                if mutated_records is not None:
                    mutated_records.add(("pr_inline_comment", comment_id))
            else:
                changed = (
                    inline.body != body
                    or inline.line != line
                    or inline.path != path
                    or not _datetimes_equal(inline.updated_at, updated_at)
                )
                if changed:
                    inline.body = body
                    inline.line = line
                    inline.path = path
                    inline.updated_at = updated_at
                    if mutated_records is not None:
                        mutated_records.add(("pr_inline_comment", comment_id))

        db.flush()
        return count, res.is_partial, res.partial_reason

    async def collect_issues(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        since: Optional[datetime] = None,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest pure issues (strictly filtering out PRs), text cleaning, and canonical labels."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/issues"
        params: Dict[str, Any] = {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100}
        if since:
            params["since"] = since.isoformat()

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="issues",
            budgeter=budgeter,
            params=params,
        )

        count = 0
        for item in res.items:
            # STRICT FILTER: Discard pull requests returned by GitHub issues endpoint
            if "pull_request" in item:
                continue

            issue_id = item.get("id")
            issue_number = item.get("number")
            if not issue_id or not issue_number:
                continue

            title = item.get("title", "")
            body = item.get("body") or ""
            state = item.get("state", "open").lower()
            state_reason = item.get("state_reason")

            user_data = item.get("user") or {}
            author = user_data.get("login") or "ghost"
            author_type = user_data.get("type", "User")
            author_association = item.get("author_association", "NONE")
            is_internal_author = author_association.upper() in {"MEMBER", "COLLABORATOR", "OWNER"}
            is_bot_author = (author_type == "Bot") or author.endswith("[bot]") or (author == "ghost")

            created_at = parse_datetime(item.get("created_at")) or datetime.now(timezone.utc)
            updated_at = parse_datetime(item.get("updated_at"))
            closed_at = parse_datetime(item.get("closed_at"))

            comments_count = item.get("comments", 0)

            # Clean and sanitize issue text
            clean_title = self._text_cleaner.clean_text(title)
            clean_body = self._text_cleaner.clean_text(body)
            combined_text = f"{clean_title} {clean_body}".strip()
            clean_title_wc = len(clean_title.split())
            clean_body_wc = len(clean_body.split())

            # Technical evidence indicators
            has_code = bool("```" in body or "<code>" in body)
            code_count = body.count("```") // 2 + body.count("<code>")
            has_url = bool("http://" in body or "https://" in body)
            has_stack = bool(re.search(r"Traceback \(most recent call last\)|Exception in thread|panic:", body))

            # Canonical label extraction
            labels = item.get("labels", [])
            canon_category, canon_id = extract_canonical_category(labels)

            # Resolution turnaround metrics
            res_hours: Optional[float] = None
            res_days: Optional[float] = None
            speed_tier: Optional[str] = None
            if closed_at and created_at:
                diff_sec = (closed_at - created_at).total_seconds()
                res_hours = round(max(0.0, diff_sec / 3600.0), 2)
                res_days = round(res_hours / 24.0, 2)
                if res_days < 1.0:
                    speed_tier = "< 1 Day"
                elif res_days <= 7.0:
                    speed_tier = "1-7 Days"
                elif res_days <= 30.0:
                    speed_tier = "7-30 Days"
                else:
                    speed_tier = "30+ Days"

            existing = db.scalar(
                select(Issue).where(
                    Issue.repository_id == repo.id,
                    Issue.issue_number == issue_number,
                )
            )

            if existing is None:
                iss = Issue(
                    id=issue_id,
                    repository_id=repo.id,
                    repo_name=repo.repo_name,
                    issue_number=issue_number,
                    title=title,
                    body=body,
                    clean_title=clean_title,
                    clean_body=clean_body,
                    combined_text=combined_text,
                    clean_title_word_count=clean_title_wc,
                    clean_body_word_count=clean_body_wc,
                    state=state,
                    state_reason=state_reason,
                    author=author,
                    author_type=author_type,
                    author_association=author_association,
                    is_internal_author=is_internal_author,
                    is_bot_author=is_bot_author,
                    created_at=created_at,
                    updated_at=updated_at,
                    closed_at=closed_at,
                    comments_count=comments_count,
                    labels_count=len(labels),
                    has_code_block=has_code,
                    code_block_count=code_count,
                    has_url=has_url,
                    has_stack_trace=has_stack,
                    canonical_label=canon_category,
                    canonical_label_id=canon_id,
                    is_classified_category=canon_category != "other",
                    is_unlabeled=len(labels) == 0,
                    resolution_time_hours=res_hours,
                    resolution_time_days=res_days,
                    speed_tier=speed_tier,
                )
                db.add(iss)
                count += 1
                if mutated_records is not None:
                    mutated_records.add(("issue", issue_number))
            else:
                iss = existing
                changed = (
                    iss.title != title
                    or iss.body != body
                    or iss.clean_title != clean_title
                    or iss.clean_body != clean_body
                    or iss.state != state
                    or iss.state_reason != state_reason
                    or not _datetimes_equal(iss.closed_at, closed_at)
                    or not _datetimes_equal(iss.updated_at, updated_at)
                    or iss.comments_count != comments_count
                    or iss.labels_count != len(labels)
                    or iss.has_code_block != has_code
                    or iss.code_block_count != code_count
                    or iss.has_url != has_url
                    or iss.has_stack_trace != has_stack
                    or iss.canonical_label != canon_category
                    or iss.canonical_label_id != canon_id
                    or not _floats_equal(iss.resolution_time_hours, res_hours)
                    or not _floats_equal(iss.resolution_time_days, res_days)
                    or iss.speed_tier != speed_tier
                )
                if changed:
                    iss.title = title
                    iss.body = body
                    iss.clean_title = clean_title
                    iss.clean_body = clean_body
                    iss.combined_text = combined_text
                    iss.clean_title_word_count = clean_title_wc
                    iss.clean_body_word_count = clean_body_wc
                    iss.state = state
                    iss.state_reason = state_reason
                    iss.closed_at = closed_at
                    iss.updated_at = updated_at
                    iss.comments_count = comments_count
                    iss.labels_count = len(labels)
                    iss.has_code_block = has_code
                    iss.code_block_count = code_count
                    iss.has_url = has_url
                    iss.has_stack_trace = has_stack
                    iss.canonical_label = canon_category
                    iss.canonical_label_id = canon_id
                    iss.resolution_time_hours = res_hours
                    iss.resolution_time_days = res_days
                    iss.speed_tier = speed_tier
                    if mutated_records is not None:
                        mutated_records.add(("issue", issue_number))

            # Insert raw issue labels
            for lbl in labels:
                lbl_name = lbl.get("name")
                if lbl_name:
                    existing_lbl = db.scalar(
                        select(IssueLabel).where(
                            IssueLabel.issue_id == issue_id,
                            IssueLabel.label_name == lbl_name,
                        )
                    )
                    if existing_lbl is None:
                        db.add(
                            IssueLabel(
                                issue_id=issue_id,
                                label_name=lbl_name,
                                is_canonical=(lbl_name.lower() in {"bug", "feature", "documentation", "question"}),
                                canonical_category=canon_category,
                            )
                        )
                        if mutated_records is not None:
                            mutated_records.add(("issue_label", f"{issue_id}:{lbl_name}"))

            # Contributor activity
            self._record_activity_if_contributor_exists(
                repo_id=repo.id,
                author_id=user_data.get("id"),
                author_login=author,
                author_email=None,
                activity_type="issue_opened",
                activity_date=created_at,
                entity_id=str(issue_number),
                db=db,
                mutated_records=mutated_records,
            )

        db.flush()
        return count, res.is_partial, res.partial_reason

    async def collect_issue_comments(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        since: Optional[datetime] = None,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest issue & PR conversation comments with mutual-exclusion parent CHECK constraint."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/issues/comments"
        params: Dict[str, Any] = {"sort": "updated", "direction": "desc", "per_page": 100}
        if since:
            params["since"] = since.isoformat()

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="issue_comments",
            budgeter=budgeter,
            params=params,
        )

        count = 0
        for item in res.items:
            comment_id = item.get("id")
            if not comment_id:
                continue

            issue_url = item.get("issue_url", "")
            issue_num_str = issue_url.rstrip("/").split("/")[-1] if issue_url else None
            if not issue_num_str or not issue_num_str.isdigit():
                continue
            issue_num = int(issue_num_str)

            user_data = item.get("user") or {}
            author = user_data.get("login") or "ghost"
            author_id = user_data.get("id")
            author_assoc = item.get("author_association")
            body = item.get("body")
            created_at = parse_datetime(item.get("created_at")) or datetime.now(timezone.utc)
            updated_at = parse_datetime(item.get("updated_at"))

            # DUAL-PARENT LINKING WITH STRICT CHECK CONSTRAINT:
            # Check if parent is an Issue or a PullRequest. NEVER set both simultaneously.
            matched_issue = db.scalar(
                select(Issue).where(
                    Issue.repository_id == repo.id,
                    Issue.issue_number == issue_num,
                )
            )
            matched_pr = None
            if matched_issue is None:
                matched_pr = db.scalar(
                    select(PullRequest).where(
                        PullRequest.repository_id == repo.id,
                        PullRequest.pr_number == issue_num,
                    )
                )

            fk_issue_id: Optional[int] = matched_issue.id if matched_issue else None
            fk_pr_id: Optional[int] = matched_pr.id if matched_pr else None

            # Assert mutual exclusion invariant in code before DB insert
            assert not (fk_issue_id is not None and fk_pr_id is not None), (
                f"Invariant breached: comment_id {comment_id} linked to both issue and PR"
            )

            comm = db.scalar(
                select(IssueComment).where(IssueComment.comment_id == comment_id)
            )

            if comm is None:
                comm = IssueComment(
                    repository_id=repo.id,
                    comment_id=comment_id,
                    issue_number=issue_num,
                    issue_id=fk_issue_id,
                    pull_request_id=fk_pr_id,
                    author=author,
                    author_id=author_id,
                    author_association=author_assoc,
                    created_at=created_at,
                    updated_at=updated_at,
                    body=body,
                )
                db.add(comm)
                count += 1
                if mutated_records is not None:
                    mutated_records.add(("issue_comment", comment_id))
            else:
                changed = (
                    comm.body != body
                    or not _datetimes_equal(comm.updated_at, updated_at)
                    or comm.issue_id != fk_issue_id
                    or comm.pull_request_id != fk_pr_id
                )
                if changed:
                    comm.body = body
                    comm.updated_at = updated_at
                    comm.issue_id = fk_issue_id
                    comm.pull_request_id = fk_pr_id
                    if mutated_records is not None:
                        mutated_records.add(("issue_comment", comment_id))

            self._record_activity_if_contributor_exists(
                repo_id=repo.id,
                author_id=author_id,
                author_login=author,
                author_email=None,
                activity_type="comment_created",
                activity_date=created_at,
                entity_id=str(comment_id),
                db=db,
                mutated_records=mutated_records,
            )

        db.flush()
        return count, res.is_partial, res.partial_reason

    async def collect_releases(
        self,
        repo: Repository,
        budgeter: EntityRateBudgeter,
        db: Session,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> Tuple[int, bool, Optional[str]]:
        """Harvest releases and release tags."""
        endpoint = f"/repos/{repo.owner}/{repo.name}/releases"
        params = {"per_page": 100}

        res: PaginatedResult = await self.client.paginate(
            endpoint_or_url=endpoint,
            entity="releases",
            budgeter=budgeter,
            params=params,
        )

        count = 0
        for item in res.items:
            rel_id = item.get("id")
            tag_name = item.get("tag_name")
            if not rel_id or not tag_name:
                continue

            name = item.get("name")
            published_at = parse_datetime(item.get("published_at"))
            is_prerelease = bool(item.get("prerelease", False))
            body = item.get("body")

            rel = db.scalar(
                select(Release).where(Release.release_id == rel_id)
            )

            if rel is None:
                rel = Release(
                    repository_id=repo.id,
                    release_id=rel_id,
                    tag_name=tag_name,
                    name=name,
                    published_at=published_at,
                    is_prerelease=is_prerelease,
                    body=body,
                )
                db.add(rel)
                count += 1
                if mutated_records is not None:
                    mutated_records.add(("release", rel_id))
            else:
                changed = (
                    rel.tag_name != tag_name
                    or rel.name != name
                    or not _datetimes_equal(rel.published_at, published_at)
                    or rel.is_prerelease != is_prerelease
                    or rel.body != body
                )
                if changed:
                    rel.tag_name = tag_name
                    rel.name = name
                    rel.published_at = published_at
                    rel.is_prerelease = is_prerelease
                    rel.body = body
                    if mutated_records is not None:
                        mutated_records.add(("release", rel_id))

        db.flush()
        return count, res.is_partial, res.partial_reason

    def reconcile_contributor_metrics(
        self,
        repository_id: int,
        db: Session,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> None:
        """Idempotently recalculate contributor activity metrics via distinct SQL aggregations."""
        contributors = db.scalars(
            select(Contributor).where(Contributor.repository_id == repository_id)
        ).all()

        for c in contributors:
            # 1. Total commits
            c_filter = [Commit.repository_id == repository_id]
            if c.login and c.git_email:
                c_filter.append(or_(Commit.author_login == c.login, Commit.author_email == c.git_email))
            elif c.login:
                c_filter.append(Commit.author_login == c.login)
            elif c.git_email:
                c_filter.append(Commit.author_email == c.git_email)
            else:
                c_filter.append(Commit.author_id == c.github_id)

            new_total_commits = db.scalar(
                select(func.count(func.distinct(Commit.sha))).where(and_(*c_filter))
            ) or 0

            # 2. PRs authored
            new_prs_authored = 0
            new_prs_reviewed = 0
            new_issues_opened = 0
            if c.login:
                new_prs_authored = db.scalar(
                    select(func.count(func.distinct(PullRequest.pr_number))).where(
                        PullRequest.repository_id == repository_id,
                        PullRequest.author == c.login,
                    )
                ) or 0

                # 3. PRs reviewed
                new_prs_reviewed = db.scalar(
                    select(func.count(func.distinct(PullRequestReview.review_id))).join(
                        PullRequest, PullRequestReview.pull_request_id == PullRequest.id
                    ).where(
                        PullRequest.repository_id == repository_id,
                        PullRequestReview.author == c.login,
                    )
                ) or 0

                # 4. Issues opened
                new_issues_opened = db.scalar(
                    select(func.count(func.distinct(Issue.issue_number))).where(
                        Issue.repository_id == repository_id,
                        Issue.author == c.login,
                    )
                ) or 0

            # 5. Contribution tenure and activity bounds
            act_stmt = select(
                func.min(ContributorActivity.activity_date),
                func.max(ContributorActivity.activity_date),
            ).where(
                ContributorActivity.repository_id == repository_id,
                ContributorActivity.contributor_id == c.id,
            )
            min_date, max_date = db.execute(act_stmt).first() or (None, None)

            new_tenure = c.tenure_days
            new_is_newcomer = c.is_newcomer
            if min_date and max_date:
                new_tenure = max(0.0, round((max_date - min_date).total_seconds() / 86400.0, 2))
                new_is_newcomer = new_tenure <= 30.0

            # Core contributor classification
            new_is_core = (new_total_commits >= 10) or (new_prs_authored >= 5)

            changed = (
                c.total_commits != new_total_commits
                or c.total_prs_authored != new_prs_authored
                or c.total_prs_reviewed != new_prs_reviewed
                or c.total_issues_opened != new_issues_opened
                or not _datetimes_equal(c.first_contribution_date, min_date)
                or not _datetimes_equal(c.last_contribution_date, max_date)
                or not _floats_equal(c.tenure_days, new_tenure)
                or c.is_newcomer != new_is_newcomer
                or c.is_core != new_is_core
            )

            if changed:
                c.total_commits = new_total_commits
                c.total_prs_authored = new_prs_authored
                c.total_prs_reviewed = new_prs_reviewed
                c.total_issues_opened = new_issues_opened
                c.first_contribution_date = min_date
                c.last_contribution_date = max_date
                c.tenure_days = new_tenure
                c.is_newcomer = new_is_newcomer
                c.is_core = new_is_core
                if mutated_records is not None:
                    mutated_records.add(("contributor", c.id))

        db.flush()

    def _record_activity_if_contributor_exists(
        self,
        repo_id: int,
        author_id: Optional[int],
        author_login: Optional[str],
        author_email: Optional[str],
        activity_type: str,
        activity_date: datetime,
        entity_id: str,
        db: Session,
        mutated_records: Optional[Set[Tuple[str, Any]]] = None,
    ) -> None:
        """Helper to find or create contributor and record a timestamped activity event."""
        contributor: Optional[Contributor] = None

        if author_id is not None:
            contributor = db.scalar(
                select(Contributor).where(
                    Contributor.repository_id == repo_id,
                    Contributor.github_id == author_id,
                )
            )

        # If not found by github_id and author_id is verified, check if an unlinked record with this email exists
        if contributor is None and author_id is not None and author_email is not None:
            unlinked_record = db.scalar(
                select(Contributor).where(
                    Contributor.repository_id == repo_id,
                    Contributor.git_email == author_email,
                    Contributor.github_id.is_(None),
                )
            )
            if unlinked_record is not None:
                # Upgrade previously unlinked author with verified GitHub identity
                unlinked_record.github_id = author_id
                if author_login:
                    unlinked_record.login = author_login
                unlinked_record.is_unlinked = False
                unlinked_record.identity_confidence = 1.0
                contributor = unlinked_record
                if mutated_records is not None:
                    mutated_records.add(("contributor", unlinked_record.id))

        # If contributor was found by author_id, check if there was a separate unlinked row with author_email
        if contributor is not None and author_id is not None and author_email is not None:
            if contributor.git_email is None:
                contributor.git_email = author_email
            old_unlinked = db.scalar(
                select(Contributor).where(
                    Contributor.repository_id == repo_id,
                    Contributor.git_email == author_email,
                    Contributor.github_id.is_(None),
                )
            )
            if old_unlinked is not None and old_unlinked.id != contributor.id:
                # Reassign activities to verified contributor and remove redundant unlinked stub
                if mutated_records is not None:
                    mutated_records.add(("contributor", old_unlinked.id))
                    mutated_records.add(("contributor", contributor.id))
                db.execute(
                    ContributorActivity.__table__.update()
                    .where(ContributorActivity.contributor_id == old_unlinked.id)
                    .values(contributor_id=contributor.id)
                )
                db.delete(old_unlinked)
                db.flush()

        # If still not found and no author_id, look up by git_email for unlinked author
        if contributor is None and author_email is not None and author_id is None:
            contributor = db.scalar(
                select(Contributor).where(
                    Contributor.repository_id == repo_id,
                    Contributor.git_email == author_email,
                )
            )

        if contributor is None and author_login is not None and author_id is not None:
            contributor = db.scalar(
                select(Contributor).where(
                    Contributor.repository_id == repo_id,
                    Contributor.login == author_login,
                )
            )

        # If still not found and we have an author identifier, create an unlinked or newly observed contributor
        if contributor is None and (author_id is not None or author_email is not None or author_login is not None):
            is_unlinked = author_id is None
            contributor = Contributor(
                repository_id=repo_id,
                github_id=author_id,
                login=author_login,
                git_email=author_email,
                is_unlinked=is_unlinked,
                identity_confidence=1.0 if author_id else 0.7,
            )
            db.add(contributor)
            db.flush()
            if mutated_records is not None:
                mutated_records.add(("contributor", contributor.id))

        if contributor is not None:
            # Check unique activity natural key: (repository_id, activity_type, entity_id)
            existing_act = db.scalar(
                select(ContributorActivity).where(
                    ContributorActivity.repository_id == repo_id,
                    ContributorActivity.activity_type == activity_type,
                    ContributorActivity.entity_id == entity_id,
                )
            )
            if existing_act is None:
                act = ContributorActivity(
                    contributor_id=contributor.id,
                    repository_id=repo_id,
                    activity_type=activity_type,
                    activity_date=activity_date,
                    entity_id=entity_id,
                )
                db.add(act)
                if mutated_records is not None:
                    mutated_records.add(("activity", f"{repo_id}:{activity_type}:{entity_id}"))
