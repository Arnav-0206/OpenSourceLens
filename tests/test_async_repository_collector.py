"""Unit and integration tests for Phase B Universal Multi-Entity Repository Collector.

Verifies:
1. Universal repository collection for arbitrary eligible public repositories.
2. All 8 entities: commits, contributors, pull requests, PR reviews, inline PR comments,
   issues, issue comments, and releases.
3. Strict PR filtering from the issues endpoint.
4. Issue comments dual-parent linking with CHECK constraint validation (3 valid states + invalid state).
5. Contributor immutable GitHub ID anchoring and unlinked author handling.
6. Contributor metric reconciliation via distinct SQL aggregations (anti-inflation).
7. Idempotent upserts across all entities.
8. Request budget enforcement and partial run reporting.
9. Extended rate-limit handling with resumes_at propagation.
"""

import asyncio
from datetime import datetime, timezone, timedelta
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.app.core.exceptions import (
    BudgetExceededException,
    EntityBudgetExceededException,
    GitHubRateLimitExceededException,
)
from backend.app.database.base import Base
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
from backend.app.services.repository_collector import (
    RepositoryCollector,
    extract_canonical_category,
    parse_datetime,
)


@pytest.fixture
def in_memory_db():
    """Create fresh isolated in-memory SQLite database session for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionClass()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def mock_client():
    """Mock AsyncGitHubClient with AsyncMock for requests and pagination."""
    client = MagicMock(spec=AsyncGitHubClient)
    default_resp = MagicMock()
    default_resp.json.return_value = {}
    client.request = AsyncMock(return_value=default_resp)
    client.paginate = AsyncMock()
    client.is_primary_rate_limited = False
    client.resumes_at = None
    return client


class TestHelperFunctions:
    """Tests for collector helper functions."""

    def test_parse_datetime_iso(self):
        dt = parse_datetime("2026-10-19T14:30:00Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 10
        assert dt.day == 19
        assert dt.tzinfo is not None

    def test_parse_datetime_none_and_invalid(self):
        assert parse_datetime(None) is None
        assert parse_datetime("") is None
        assert parse_datetime("not-a-datetime") is None

    def test_extract_canonical_category(self):
        assert extract_canonical_category([{"name": "bug"}]) == ("bug", 0)
        assert extract_canonical_category([{"name": "critical-defect"}]) == ("bug", 0)
        assert extract_canonical_category([{"name": "feature-request"}]) == ("feature", 1)
        assert extract_canonical_category([{"name": "docs"}]) == ("documentation", 2)
        assert extract_canonical_category([{"name": "question"}]) == ("question", 3)
        assert extract_canonical_category([{"name": "misc-label"}]) == ("other", 4)
        assert extract_canonical_category([]) == ("other", 4)


class TestRepositoryMetadataCollection:
    """Tests for repository root metadata harvesting and idempotent upsert."""

    def test_collect_metadata_insert_and_update(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            # Mock GitHub /repos/sample-org/sample-repo response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": 999001,
                "name": "sample-repo",
                "full_name": "sample-org/sample-repo",
                "description": "An arbitrary open source project",
                "language": "Python",
                "stargazers_count": 420,
                "forks_count": 55,
                "open_issues_count": 12,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            mock_client.request.return_value = mock_response

            # 1. Initial Insert
            repo = await collector.collect_repository_metadata(
                repo_name="sample-org/sample-repo",
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert repo.id == 999001
            assert repo.repo_name == "sample-org/sample-repo"
            assert repo.stars_count == 420
            assert repo.primary_language == "Python"
            assert repo.is_archived is False

            # Verify DB persistence
            db_repo = in_memory_db.scalar(select(Repository).where(Repository.id == 999001))
            assert db_repo is not None
            assert db_repo.stars_count == 420

            # 2. Idempotent Update with changed stars and description
            mock_response.json.return_value["stargazers_count"] = 500
            mock_response.json.return_value["description"] = "Updated description"

            updated_repo = await collector.collect_repository_metadata(
                repo_name="sample-org/sample-repo",
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert updated_repo.id == 999001
            assert updated_repo.stars_count == 500
            assert updated_repo.description == "Updated description"

            # Verify no duplicate repository was created
            all_repos = in_memory_db.scalars(select(Repository)).all()
            assert len(all_repos) == 1

        asyncio.run(_run())


class TestCommitsCollection:
    """Tests for commit harvesting, detail sampling, and unlinked author handling."""

    def test_collect_commits_and_sample_details(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            # Seed repository
            repo = Repository(
                id=1001,
                repo_name="octocat/hello-world",
                owner="octocat",
                name="hello-world",
                stars_count=10,
            )
            in_memory_db.add(repo)
            in_memory_db.commit()

            # Mock paginated commits list
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "aaa111",
                        "commit": {
                            "author": {
                                "name": "Octo Cat",
                                "email": "octo@github.com",
                                "date": "2026-10-01T10:00:00Z",
                            },
                            "message": "feat: initial commit",
                        },
                        "author": {
                            "login": "octocat",
                            "id": 583231,
                        },
                    },
                    {
                        "sha": "bbb222",
                        "commit": {
                            "author": {
                                "name": "Anonymous Dev",
                                "email": "anon@noreply.git",
                                "date": "2026-10-02T12:00:00Z",
                            },
                            "message": "fix: bugfix without github profile",
                        },
                        "author": None,  # Unlinked Git author
                    },
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            # Mock commit detail response for additions/deletions
            mock_detail = MagicMock()
            mock_detail.json.return_value = {
                "stats": {"additions": 45, "deletions": 12, "total": 57}
            }
            mock_client.request.return_value = mock_detail

            count, is_part, reason = await collector.collect_commits(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
                mode="standard_diagnostic",
            )

            assert count == 2
            assert is_part is False

            # Verify commits in DB
            cmt1 = in_memory_db.scalar(select(Commit).where(Commit.sha == "aaa111"))
            assert cmt1 is not None
            assert cmt1.author_login == "octocat"
            assert cmt1.author_id == 583231
            assert cmt1.additions == 45
            assert cmt1.deletions == 12

            cmt2 = in_memory_db.scalar(select(Commit).where(Commit.sha == "bbb222"))
            assert cmt2 is not None
            assert cmt2.author_login is None
            assert cmt2.author_email == "anon@noreply.git"

            # Verify unlinked contributor was tracked
            unlinked_contrib = in_memory_db.scalar(
                select(Contributor).where(Contributor.git_email == "anon@noreply.git")
            )
            assert unlinked_contrib is not None
            assert unlinked_contrib.is_unlinked is True
            assert unlinked_contrib.identity_confidence == 0.7

        asyncio.run(_run())


class TestContributorsCollection:
    """Tests for contributor identity anchoring on immutable GitHub ID."""

    def test_collect_contributors_anchoring(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1002, repo_name="org/proj", owner="org", name="proj")
            in_memory_db.add(repo)
            in_memory_db.commit()

            mock_client.paginate.return_value = PaginatedResult(
                entity="contributors",
                items=[
                    {
                        "id": 12345,
                        "login": "devone",
                        "avatar_url": "https://avatars.githubusercontent.com/u/12345",
                        "type": "User",
                    },
                    {
                        "id": 67890,
                        "login": "dependabot[bot]",
                        "avatar_url": "https://avatars.githubusercontent.com/u/67890",
                        "type": "Bot",
                    },
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count, is_part, reason = await collector.collect_contributors(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert count == 2
            dev1 = in_memory_db.scalar(select(Contributor).where(Contributor.github_id == 12345))
            assert dev1 is not None
            assert dev1.login == "devone"
            assert dev1.is_bot is False
            assert dev1.identity_confidence == 1.0

            bot = in_memory_db.scalar(select(Contributor).where(Contributor.github_id == 67890))
            assert bot is not None
            assert bot.is_bot is True

            # Test idempotent update when username changes
            mock_client.paginate.return_value = PaginatedResult(
                entity="contributors",
                items=[
                    {
                        "id": 12345,
                        "login": "devone-renamed",
                        "avatar_url": "https://avatars.githubusercontent.com/u/12345-new",
                        "type": "User",
                    }
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count2, _, _ = await collector.collect_contributors(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )
            assert count2 == 0  # Updated existing, not inserted new
            dev1_updated = in_memory_db.scalar(select(Contributor).where(Contributor.github_id == 12345))
            assert dev1_updated.login == "devone-renamed"

        asyncio.run(_run())

    def test_unlinked_author_upgraded_when_github_id_verified_later(self, in_memory_db, mock_client):
        """Test unlinked commit author is upgraded when verified GitHub ID arrives on a subsequent commit."""
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1020, repo_name="org/upgrade", owner="org", name="upgrade")
            in_memory_db.add(repo)
            in_memory_db.commit()

            # Pass 1: Commit without GitHub user ID (unlinked author)
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c_early",
                        "commit": {
                            "author": {"name": "Unlinked Dev", "email": "dev@unlinked.com", "date": "2026-09-01T00:00:00Z"},
                            "message": "early unlinked commit",
                        },
                        "author": None,
                    }
                ],
                total_requests=1,
            )

            await collector.collect_commits(repo=repo, budgeter=budgeter, db=in_memory_db, mode="standard_diagnostic")

            contrib = in_memory_db.scalar(select(Contributor).where(Contributor.git_email == "dev@unlinked.com"))
            assert contrib is not None
            assert contrib.github_id is None
            assert contrib.is_unlinked is True
            assert contrib.identity_confidence == 0.7

            # Pass 2: Later commit where GitHub attaches verified profile to this email
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c_verified",
                        "commit": {
                            "author": {"name": "Unlinked Dev", "email": "dev@unlinked.com", "date": "2026-10-01T00:00:00Z"},
                            "message": "verified commit with github account",
                        },
                        "author": {"id": 777888, "login": "verified_dev"},
                    }
                ],
                total_requests=1,
            )

            await collector.collect_commits(repo=repo, budgeter=budgeter, db=in_memory_db, mode="standard_diagnostic")

            # Verify the unlinked contributor record was upgraded in place
            in_memory_db.refresh(contrib)
            assert contrib.github_id == 777888
            assert contrib.login == "verified_dev"
            assert contrib.is_unlinked is False
            assert contrib.identity_confidence == 1.0

            # Verify only 1 contributor row exists
            all_contribs = in_memory_db.scalars(select(Contributor).where(Contributor.repository_id == 1020)).all()
            assert len(all_contribs) == 1

            # Verify both commit activities belong to this single contributor
            acts = in_memory_db.scalars(select(ContributorActivity).where(ContributorActivity.contributor_id == contrib.id)).all()
            assert len(acts) == 2

        asyncio.run(_run())

    def test_two_contributors_sharing_display_name_are_never_merged(self, in_memory_db, mock_client):
        """Test contributors sharing a display name ('Alex') are never merged without identical GitHub IDs."""
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1021, repo_name="org/display", owner="org", name="display")
            in_memory_db.add(repo)
            in_memory_db.commit()

            # Two commits by two different people, both named "Alex Developer" in git config:
            # - First Alex has email "alex.one@corp.com" (unlinked)
            # - Second Alex has GitHub ID 999111, login "alexdev2", email "alex.two@corp.com"
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c_alex1",
                        "commit": {
                            "author": {"name": "Alex Developer", "email": "alex.one@corp.com", "date": "2026-09-01T00:00:00Z"},
                            "message": "commit from Alex 1",
                        },
                        "author": None,
                    },
                    {
                        "sha": "c_alex2",
                        "commit": {
                            "author": {"name": "Alex Developer", "email": "alex.two@corp.com", "date": "2026-09-02T00:00:00Z"},
                            "message": "commit from Alex 2",
                        },
                        "author": {"id": 999111, "login": "alexdev2"},
                    },
                ],
                total_requests=1,
            )

            await collector.collect_commits(repo=repo, budgeter=budgeter, db=in_memory_db, mode="standard_diagnostic")

            # Verify two distinct contributors are preserved
            all_contribs = in_memory_db.scalars(select(Contributor).where(Contributor.repository_id == 1021)).all()
            assert len(all_contribs) == 2

            c_unlinked = in_memory_db.scalar(select(Contributor).where(Contributor.git_email == "alex.one@corp.com"))
            assert c_unlinked.github_id is None
            assert c_unlinked.is_unlinked is True

            c_verified = in_memory_db.scalar(select(Contributor).where(Contributor.github_id == 999111))
            assert c_verified.login == "alexdev2"
            assert c_verified.is_unlinked is False

        asyncio.run(_run())

    def test_existing_verified_contributor_adopts_unlinked_activities(self, in_memory_db, mock_client):
        """Test existing verified contributor reassigns unlinked activities when GitHub attests email link."""
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1022, repo_name="org/adopt", owner="org", name="adopt")
            in_memory_db.add(repo)
            in_memory_db.flush()

            # Seed existing verified contributor from /contributors
            verified = Contributor(
                repository_id=repo.id,
                github_id=444555,
                login="sarah",
                git_email=None,
                is_unlinked=False,
                identity_confidence=1.0,
            )
            in_memory_db.add(verified)
            in_memory_db.commit()

            # Seed early commit without GitHub ID creating unlinked stub
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c_sarah_early",
                        "commit": {
                            "author": {"name": "Sarah", "email": "sarah@corp.com", "date": "2026-08-01T00:00:00Z"},
                            "message": "early commit",
                        },
                        "author": None,
                    }
                ],
                total_requests=1,
            )
            await collector.collect_commits(repo=repo, budgeter=budgeter, db=in_memory_db, mode="standard_diagnostic")

            # Verify unlinked stub exists
            unlinked_stub = in_memory_db.scalar(select(Contributor).where(Contributor.git_email == "sarah@corp.com", Contributor.github_id.is_(None)))
            assert unlinked_stub is not None

            # Now commit arrives where GitHub verifies author_id=444555 for email sarah@corp.com
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c_sarah_now",
                        "commit": {
                            "author": {"name": "Sarah", "email": "sarah@corp.com", "date": "2026-09-01T00:00:00Z"},
                            "message": "verified commit",
                        },
                        "author": {"id": 444555, "login": "sarah"},
                    }
                ],
                total_requests=1,
            )
            await collector.collect_commits(repo=repo, budgeter=budgeter, db=in_memory_db, mode="standard_diagnostic")

            # Unlinked stub should be eliminated, verified contributor has git_email, and owns both activities
            in_memory_db.refresh(verified)
            assert verified.git_email == "sarah@corp.com"

            remaining_unlinked = in_memory_db.scalar(select(Contributor).where(Contributor.git_email == "sarah@corp.com", Contributor.github_id.is_(None)))
            assert remaining_unlinked is None

            acts = in_memory_db.scalars(select(ContributorActivity).where(ContributorActivity.contributor_id == verified.id)).all()
            assert len(acts) == 2

        asyncio.run(_run())


class TestPullRequestsAndReviewsCollection:
    """Tests for PR harvesting, review envelopes, inline comments, and SLA metrics."""

    def test_collect_pull_requests_and_reviews(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1003, repo_name="org/prs", owner="org", name="prs")
            in_memory_db.add(repo)
            in_memory_db.commit()

            # Mock PR list
            mock_client.paginate.return_value = PaginatedResult(
                entity="pull_requests",
                items=[
                    {
                        "number": 101,
                        "title": "Add async support",
                        "state": "closed",
                        "user": {"login": "alice", "id": 11111},
                        "created_at": "2026-10-01T10:00:00Z",
                        "closed_at": "2026-10-01T15:00:00Z",
                        "merged_at": "2026-10-01T15:00:00Z",  # 5 hours turnaround
                        "comments": 2,
                        "review_comments": 3,
                        "additions": 150,
                        "deletions": 20,
                        "changed_files": 4,
                    }
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            prs, count, is_part, reason = await collector.collect_pull_requests(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert count == 1
            assert len(prs) == 1
            pr = prs[0]
            assert pr.pr_number == 101
            assert pr.time_to_merge_hours == 5.0

            # Mock reviews for PR #101
            mock_review_resp = MagicMock()
            mock_review_resp.json.return_value = [
                {
                    "id": 9001,
                    "user": {"login": "bob", "id": 22222},
                    "state": "APPROVED",
                    "submitted_at": "2026-10-01T12:00:00Z",  # 2 hours after creation
                    "body": "Looks great to me!",
                }
            ]
            mock_client.request.return_value = mock_review_resp

            r_count, r_part, r_reason = await collector.collect_pr_reviews(
                repo=repo,
                pull_requests=prs,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert r_count == 1
            rev = in_memory_db.scalar(select(PullRequestReview).where(PullRequestReview.review_id == 9001))
            assert rev is not None
            assert rev.author == "bob"
            assert rev.state == "APPROVED"
            assert pr.time_to_first_review_hours == 2.0

            # Mock repo-wide diff inline comments
            mock_client.paginate.return_value = PaginatedResult(
                entity="pr_inline_comments",
                items=[
                    {
                        "id": 8881,
                        "pull_request_url": "https://api.github.com/repos/org/prs/pulls/101",
                        "user": {"login": "bob", "id": 22222},
                        "path": "backend/app/main.py",
                        "line": 42,
                        "commit_id": "c123456",
                        "body": "Consider using asyncio.gather here",
                        "created_at": "2026-10-01T12:30:00Z",
                        "updated_at": "2026-10-01T12:30:00Z",
                    }
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            in_count, in_part, in_reason = await collector.collect_pr_inline_comments(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert in_count == 1
            inline_comment = in_memory_db.scalar(
                select(PullRequestReviewComment).where(PullRequestReviewComment.comment_id == 8881)
            )
            assert inline_comment is not None
            assert inline_comment.path == "backend/app/main.py"
            assert inline_comment.line == 42

        asyncio.run(_run())


class TestIssuesCollection:
    """Tests for pure issues harvesting, PR exclusion, text cleaning, and canonical labels."""

    def test_collect_issues_with_strict_pr_filter(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1004, repo_name="org/issues", owner="org", name="issues")
            in_memory_db.add(repo)
            in_memory_db.commit()

            # GitHub /issues endpoint returns BOTH pure issues and PRs!
            mock_client.paginate.return_value = PaginatedResult(
                entity="issues",
                items=[
                    {
                        # PURE ISSUE: should be harvested
                        "id": 5001,
                        "number": 1,
                        "title": "Crash on startup when config file is missing",
                        "body": "Here is the error traceback:\n```\nTraceback (most recent call last):\nException: missing config\n```",
                        "state": "closed",
                        "user": {"login": "reporter", "id": 33333, "type": "User"},
                        "labels": [{"name": "bug"}, {"name": "priority-high"}],
                        "created_at": "2026-10-01T08:00:00Z",
                        "closed_at": "2026-10-01T20:00:00Z",  # 12 hours (0.5 days)
                        "comments": 3,
                    },
                    {
                        # PULL REQUEST: must be strictly skipped
                        "id": 5002,
                        "number": 2,
                        "title": "Fix crash on startup",
                        "pull_request": {"url": "https://api.github.com/repos/org/issues/pulls/2"},
                        "user": {"login": "fixer", "id": 44444, "type": "User"},
                        "labels": [{"name": "bug"}],
                        "created_at": "2026-10-01T09:00:00Z",
                    },
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count, is_part, reason = await collector.collect_issues(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert count == 1  # Only 1 pure issue, PR skipped

            # Verify pure issue
            iss = in_memory_db.scalar(select(Issue).where(Issue.id == 5001))
            assert iss is not None
            assert iss.issue_number == 1
            assert iss.canonical_label == "bug"
            assert iss.canonical_label_id == 0
            assert iss.has_code_block is True
            assert iss.has_stack_trace is True
            assert iss.resolution_time_hours == 12.0
            assert iss.speed_tier == "< 1 Day"

            # Verify PR was NOT inserted into issues table
            pr_in_issues = in_memory_db.scalar(select(Issue).where(Issue.id == 5002))
            assert pr_in_issues is None

        asyncio.run(_run())

    def test_collect_issues_ghost_author_and_idempotency(self, in_memory_db, mock_client):
        """Test issues collection handles deleted users (ghost/None) and updates existing issues idempotently."""
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1004, repo_name="org/issues", owner="org", name="issues")
            in_memory_db.add(repo)
            in_memory_db.commit()

            # First run: deleted user (user=None)
            mock_client.paginate.return_value = PaginatedResult(
                entity="issues",
                items=[
                    {
                        "id": 5003,
                        "number": 3,
                        "title": "Issue by deleted user",
                        "body": None,  # None body
                        "state": "open",
                        "user": None,  # Deleted user
                        "labels": [],
                        "created_at": "2026-10-01T08:00:00Z",
                    }
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count1, _, _ = await collector.collect_issues(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )
            assert count1 == 1

            iss = in_memory_db.scalar(select(Issue).where(Issue.id == 5003))
            assert iss.author == "ghost"
            assert iss.is_bot_author is True
            assert iss.body == ""
            assert iss.canonical_label == "other"
            assert iss.is_unlabeled is True

            # Second run: Same issue closed later
            mock_client.paginate.return_value = PaginatedResult(
                entity="issues",
                items=[
                    {
                        "id": 5003,
                        "number": 3,
                        "title": "Issue by deleted user",
                        "body": None,
                        "state": "closed",
                        "user": None,
                        "labels": [],
                        "created_at": "2026-10-01T08:00:00Z",
                        "closed_at": "2026-10-02T08:00:00Z",
                    }
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count2, _, _ = await collector.collect_issues(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )
            assert count2 == 0  # Updated in place, no duplicate row
            in_memory_db.refresh(iss)
            assert iss.state == "closed"
            assert iss.resolution_time_days == 1.0

            all_issues = in_memory_db.scalars(select(Issue).where(Issue.repository_id == 1004)).all()
            assert len(all_issues) == 1

        asyncio.run(_run())


class TestIssueCommentsDualParentLinking:
    """Tests for issue/PR comments dual-parent linking and CHECK constraint."""

    def test_comments_linking_and_check_constraint(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1005, repo_name="org/comments", owner="org", name="comments")
            in_memory_db.add(repo)
            in_memory_db.flush()

            # Seed 1 pure issue (#10) and 1 PR (#20)
            issue = Issue(
                id=7010,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=10,
                title="Database slow query",
                body="Query takes 10s",
                state="open",
                author="tester",
                created_at=datetime.now(timezone.utc),
            )
            pr = PullRequest(
                id=8020,
                repository_id=repo.id,
                pr_number=20,
                title="Index optimization",
                state="open",
                created_at=datetime.now(timezone.utc),
            )
            in_memory_db.add_all([issue, pr])
            in_memory_db.commit()

            # Mock 3 comments:
            # 1. Comment on Issue #10
            # 2. Comment on PR #20
            # 3. Comment on unknown/uningested parent #30
            mock_client.paginate.return_value = PaginatedResult(
                entity="issue_comments",
                items=[
                    {
                        "id": 9101,
                        "issue_url": "https://api.github.com/repos/org/comments/issues/10",
                        "user": {"login": "charlie", "id": 55555},
                        "body": "Can you share the EXPLAIN query plan?",
                        "created_at": "2026-10-02T10:00:00Z",
                    },
                    {
                        "id": 9202,
                        "issue_url": "https://api.github.com/repos/org/comments/issues/20",
                        "user": {"login": "dana", "id": 66666},
                        "body": "LGTM, benchmarks show 10x improvement.",
                        "created_at": "2026-10-02T11:00:00Z",
                    },
                    {
                        "id": 9303,
                        "issue_url": "https://api.github.com/repos/org/comments/issues/30",
                        "user": {"login": "evan", "id": 77777},
                        "body": "Parent not yet in local database.",
                        "created_at": "2026-10-02T12:00:00Z",
                    },
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count, is_part, reason = await collector.collect_issue_comments(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert count == 3

            # State 1: Issue comment linked to Issue
            c1 = in_memory_db.scalar(select(IssueComment).where(IssueComment.comment_id == 9101))
            assert c1.issue_id == 7010
            assert c1.pull_request_id is None
            assert c1.issue_number == 10

            # State 2: PR conversation comment linked to PR
            c2 = in_memory_db.scalar(select(IssueComment).where(IssueComment.comment_id == 9202))
            assert c2.issue_id is None
            assert c2.pull_request_id == 8020
            assert c2.issue_number == 20

            # State 3: Both NULL (parent not yet ingested)
            c3 = in_memory_db.scalar(select(IssueComment).where(IssueComment.comment_id == 9303))
            assert c3.issue_id is None
            assert c3.pull_request_id is None
            assert c3.issue_number == 30

            # State 4 (INVALID): Direct violation of CHECK constraint chk_issue_comment_parent
            invalid_comment = IssueComment(
                repository_id=repo.id,
                comment_id=99999,
                issue_number=10,
                issue_id=7010,
                pull_request_id=8020,  # BOTH NON-NULL -> Violates CHECK constraint!
                created_at=datetime.now(timezone.utc),
                body="Illegal dual parent comment",
            )
            in_memory_db.add(invalid_comment)
            with pytest.raises(IntegrityError):
                in_memory_db.commit()
            in_memory_db.rollback()

        asyncio.run(_run())


class TestReleasesCollection:
    """Tests for software release tag collection."""

    def test_collect_releases_idempotency(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=1006, repo_name="org/releases", owner="org", name="releases")
            in_memory_db.add(repo)
            in_memory_db.commit()

            mock_client.paginate.return_value = PaginatedResult(
                entity="releases",
                items=[
                    {
                        "id": 1111,
                        "tag_name": "v1.0.0",
                        "name": "Version 1.0.0 Stable",
                        "published_at": "2026-09-01T00:00:00Z",
                        "prerelease": False,
                        "body": "First stable release notes",
                    },
                    {
                        "id": 2222,
                        "tag_name": "v2.0.0-rc1",
                        "name": "Version 2.0.0 Release Candidate",
                        "published_at": "2026-10-01T00:00:00Z",
                        "prerelease": True,
                        "body": "Testing candidate",
                    },
                ],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            count, is_part, reason = await collector.collect_releases(
                repo=repo,
                budgeter=budgeter,
                db=in_memory_db,
            )

            assert count == 2
            r1 = in_memory_db.scalar(select(Release).where(Release.release_id == 1111))
            assert r1.tag_name == "v1.0.0"
            assert r1.is_prerelease is False

            r2 = in_memory_db.scalar(select(Release).where(Release.release_id == 2222))
            assert r2.tag_name == "v2.0.0-rc1"
            assert r2.is_prerelease is True

        asyncio.run(_run())


class TestContributorMetricReconciliation:
    """Tests for SQL distinct count metric recalculation (anti-inflation)."""

    def test_reconcile_contributor_metrics_distinct_counts(self, in_memory_db):
        collector = RepositoryCollector()

        repo = Repository(id=1007, repo_name="org/stats", owner="org", name="stats")
        in_memory_db.add(repo)
        in_memory_db.flush()

        contributor = Contributor(
            repository_id=repo.id,
            github_id=88888,
            login="leaddev",
            total_commits=999,  # Inflated initial counter
            total_prs_authored=999,
            total_prs_reviewed=999,
            total_issues_opened=999,
        )
        in_memory_db.add(contributor)
        in_memory_db.flush()

        # Seed distinct entity records for leaddev
        # 3 Commits
        for sha in ["c1", "c2", "c3"]:
            in_memory_db.add(
                Commit(
                    repository_id=repo.id,
                    sha=sha,
                    author_login="leaddev",
                    author_id=88888,
                    author_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            )

        # 2 PRs authored
        for pr_num in [10, 20]:
            in_memory_db.add(
                PullRequest(
                    repository_id=repo.id,
                    pr_number=pr_num,
                    title=f"PR #{pr_num}",
                    state="closed",
                    author="leaddev",
                    created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                )
            )

        # 1 PR reviewed
        pr_other = PullRequest(
            repository_id=repo.id,
            pr_number=30,
            title="PR #30",
            state="closed",
            author="otherdev",
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        in_memory_db.add(pr_other)
        in_memory_db.flush()
        in_memory_db.add(
            PullRequestReview(
                pull_request_id=pr_other.id,
                review_id=7701,
                author="leaddev",
                state="APPROVED",
                submitted_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
            )
        )

        # 1 Issue opened
        in_memory_db.add(
            Issue(
                id=6601,
                repository_id=repo.id,
                repo_name=repo.repo_name,
                issue_number=1,
                title="Issue #1",
                state="open",
                author="leaddev",
                created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            )
        )

        # Activity events for tenure calculation (first: Jan 1, last: Mar 2)
        in_memory_db.add_all([
            ContributorActivity(
                contributor_id=contributor.id,
                repository_id=repo.id,
                activity_type="commit",
                activity_date=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                entity_id="c1",
            ),
            ContributorActivity(
                contributor_id=contributor.id,
                repository_id=repo.id,
                activity_type="review_submitted",
                activity_date=datetime(2026, 3, 2, 0, 0, 0, tzinfo=timezone.utc),
                entity_id="7701",
            ),
        ])
        in_memory_db.commit()

        # Run reconciliation
        collector.reconcile_contributor_metrics(repository_id=repo.id, db=in_memory_db)
        in_memory_db.commit()

        # Verify all inflated counters were reset to exact distinct counts
        assert contributor.total_commits == 3
        assert contributor.total_prs_authored == 2
        assert contributor.total_prs_reviewed == 1
        assert contributor.total_issues_opened == 1
        assert contributor.tenure_days == 60.0  # Jan 1 to Mar 2 = 60 days
        assert contributor.is_newcomer is False  # > 30 days


class TestFullCollectionOrchestrator:
    """Tests for full multi-entity pipeline, budget exhaustion, and rate limit propagation."""

    def test_full_collection_success_flow(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            # Mock metadata
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1008,
                "name": "full-test",
                "full_name": "org/full-test",
                "description": "Integration test repo",
                "language": "TypeScript",
                "stargazers_count": 100,
                "forks_count": 20,
                "open_issues_count": 5,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            mock_client.request.return_value = meta_resp
            mock_client.paginate.return_value = PaginatedResult(
                entity="generic",
                items=[],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            summary: IngestionSummary = await collector.collect_repository(
                repo_name="org/full-test",
                mode="fast_snapshot",
                db=in_memory_db,
            )

            assert summary.repo_name == "org/full-test"
            assert summary.mode == "fast_snapshot"
            assert summary.is_partial is False
            assert summary.items_collected["repository"] == 1
            assert summary.errors == []

        asyncio.run(_run())

    def test_full_collection_budget_exhaustion(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="fast_snapshot")

            # Mock metadata
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1009,
                "name": "budget-test",
                "full_name": "org/budget-test",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Simulate budget exhaustion during commits pagination
            mock_client.paginate.side_effect = BudgetExceededException(
                message="Total request budget of 10 requests exhausted.",
                max_allowed=10,
            )

            summary = await collector.collect_repository(
                repo_name="org/budget-test",
                mode="fast_snapshot",
                db=in_memory_db,
                budgeter=budgeter,
            )

            assert summary.is_partial is True
            assert "exhausted" in summary.partial_reason.lower()

        asyncio.run(_run())

    def test_full_collection_rate_limit_with_resumes_at(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1010,
                "name": "ratelimit-test",
                "full_name": "org/ratelimit-test",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Seed repo before CollectionRun (satisfying foreign key constraint)
            repo = Repository(
                id=1010,
                repo_name="org/ratelimit-test",
                owner="org",
                name="ratelimit-test",
            )
            in_memory_db.add(repo)
            in_memory_db.flush()

            # Create CollectionRun audit row
            c_run = CollectionRun(
                repository_id=1010,
                mode="standard_diagnostic",
                status="running",
            )
            in_memory_db.add(c_run)
            in_memory_db.commit()

            expected_resume = datetime(2026, 11, 10, 8, 30, 0, tzinfo=timezone.utc)
            mock_client.paginate.side_effect = GitHubRateLimitExceededException(
                message="GitHub primary rate limit exhausted. Resumes at 2026-11-10T08:30:00Z",
                retry_after_seconds=600,
                resumes_at=expected_resume,
            )

            summary = await collector.collect_repository(
                repo_name="org/ratelimit-test",
                mode="standard_diagnostic",
                db=in_memory_db,
                budgeter=budgeter,
                collection_run_id=c_run.id,
            )

            assert summary.is_partial is True
            assert summary.resumes_at == expected_resume

            # Verify CollectionRun audit record was updated
            in_memory_db.refresh(c_run)
            assert c_run.status == "rate_limited"
            assert c_run.is_partial is True

        asyncio.run(_run())

    def test_entity_status_granularity_reporting(self, in_memory_db, mock_client):
        """Test IngestionSummary reports granular status per entity stream (completed, truncated, skipped)."""
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="fast_snapshot")

            # Mock metadata
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1030,
                "name": "status-repo",
                "full_name": "org/status-repo",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Commits truncated by budget
            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[{"sha": "c1", "commit": {"author": {"email": "e@e.com", "date": "2026-01-01T00:00:00Z"}}, "author": None}],
                total_requests=1,
                is_partial=True,
                partial_reason="Page limit reached",
            )

            summary = await collector.collect_repository(
                repo_name="org/status-repo",
                mode="fast_snapshot",
                db=in_memory_db,
                budgeter=budgeter,
            )

            # Check granular entity statuses
            assert summary.entity_status["repository"] == "completed"
            assert summary.entity_status["commits"] == "truncated"
            # Fast snapshot mode skips inline comments
            assert summary.entity_status["pr_inline_comments"] == "skipped"

        asyncio.run(_run())

    def test_empty_repository_endpoint_marks_completed_not_truncated(self, in_memory_db, mock_client):
        """Test entity returning zero records when fully queried is marked 'completed', not 'truncated' or 'skipped'."""
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            # Mock metadata
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 1040,
                "name": "empty-entities-repo",
                "full_name": "org/empty-entities-repo",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # All paginated endpoints return 0 items and is_partial=False (fully exhausted empty endpoints)
            mock_client.paginate.return_value = PaginatedResult(
                entity="generic",
                items=[],
                total_requests=1,
                is_partial=False,
                partial_reason=None,
            )

            summary = await collector.collect_repository(
                repo_name="org/empty-entities-repo",
                mode="standard_diagnostic",
                db=in_memory_db,
                budgeter=budgeter,
            )

            # Every queried endpoint with 0 items returned must be 'completed', never 'truncated' or 'skipped'
            assert summary.entity_status["repository"] == "completed"
            assert summary.entity_status["commits"] == "completed"
            assert summary.entity_status["contributors"] == "completed"
            assert summary.entity_status["pull_requests"] == "completed"
            assert summary.entity_status["pr_reviews"] == "completed"
            assert summary.entity_status["pr_inline_comments"] == "completed"
            assert summary.entity_status["issues"] == "completed"
            assert summary.entity_status["issue_comments"] == "completed"
            assert summary.entity_status["releases"] == "completed"
            assert summary.is_partial is False

        asyncio.run(_run())


class TestRealWorldLiveIngestion:
    """Live integration test against an eligible public GitHub repository outside the 4 benchmarks."""

    @pytest.mark.skipif(
        not os.getenv("RUN_LIVE_GITHUB_TESTS"),
        reason="Live GitHub integration test is opt-in (set RUN_LIVE_GITHUB_TESTS=1 to run).",
    )
    def test_live_fast_snapshot_ingestion_pallets_click(self, in_memory_db):
        """Perform a controlled live ingestion against pallets/click in fast_snapshot mode (<= 10 requests)."""
        async def _run():
            collector = RepositoryCollector()
            budgeter = EntityRateBudgeter(mode="fast_snapshot")

            try:
                summary = await collector.collect_repository(
                    repo_name="pallets/click",
                    mode="fast_snapshot",
                    db=in_memory_db,
                    budgeter=budgeter,
                )
            except Exception as exc:
                pytest.skip(f"GitHub API unreachable or rate limit exceeded: {exc}")

            if summary.resumes_at is not None or (summary.is_partial and "rate limit" in str(summary.partial_reason).lower()):
                pytest.skip("Unauthenticated GitHub API rate limit reached, skipping live test.")

            assert summary.repo_name == "pallets/click"
            assert summary.total_requests_consumed <= 10
            assert summary.items_collected["repository"] == 1
            assert summary.items_collected["commits"] > 0
            assert summary.items_collected["contributors"] > 0
            assert summary.items_collected["pull_requests"] > 0
            assert summary.items_collected["releases"] > 0
            assert summary.entity_status["repository"] == "completed"
            assert summary.errors == []

            # Verify database normalization in isolated test DB
            repo = in_memory_db.scalar(select(Repository).where(Repository.repo_name == "pallets/click"))
            assert repo is not None
            assert repo.stars_count > 10000
            assert repo.primary_language == "Python"

            # Verify pure issues filtering
            issues_count = in_memory_db.scalar(select(func.count(Issue.id)).where(Issue.repository_id == repo.id))
            assert issues_count > 0

        asyncio.run(_run())
