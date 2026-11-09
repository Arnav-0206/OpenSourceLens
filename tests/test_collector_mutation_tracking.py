"""Unit & Integration Tests for Record-Level Mutation Tracking in RepositoryCollector (Phase C Step C.1.2).

Verifies:
1. Distinct entity mutation tracking across analytical fields.
2. committer_date extraction from GitHub commit payloads and NULL preservation.
3. Atomic revision advancement on successful ingestion commits (analytics_source_revision += 1).
4. Unchanged re-ingestion idempotency: records_mutated == 0 and revision is NOT advanced.
5. Operational metadata changes (e.g., last_synced_at, description) do NOT advance revision.
6. Multiple touches on the same entity within one transaction count as exactly 1 mutation.
7. Atomic revision advancement on partial commits (request budget exhaustion and rate limit).
8. Atomic rollback on unexpected exceptions reverts any uncommitted revision increment.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.exceptions import (
    BudgetExceededException,
    GitHubRateLimitExceededException,
)
from backend.app.database.base import Base
from backend.app.database.models import (
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
    _datetimes_equal,
    _floats_equal,
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


class TestDatetimeAndFloatHelpers:
    """Tests for dialect-invariant equality helpers."""

    def test_datetimes_equal_cross_timezone(self):
        dt_aware = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
        dt_naive = datetime(2026, 10, 15, 12, 0, 0)
        assert _datetimes_equal(dt_aware, dt_naive)
        assert _datetimes_equal(None, None)
        assert not _datetimes_equal(dt_aware, None)
        assert not _datetimes_equal(None, dt_naive)

    def test_floats_equal_tolerance(self):
        assert _floats_equal(1.00001, 1.00002)
        assert not _floats_equal(1.0, 1.1)
        assert _floats_equal(None, None)
        assert not _floats_equal(1.0, None)


class TestCommitterDateExtraction:
    """Tests for commit committer_date extraction and NULL preservation."""

    def test_committer_date_extracted_when_present(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=101, repo_name="owner/repo", owner="owner", name="repo")
            in_memory_db.add(repo)
            in_memory_db.commit()

            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "sha_committer_valid",
                        "commit": {
                            "author": {
                                "name": "Author",
                                "email": "a@example.com",
                                "date": "2026-10-01T10:00:00Z",
                            },
                            "committer": {
                                "name": "Committer",
                                "email": "c@example.com",
                                "date": "2026-10-02T15:30:00Z",
                            },
                            "message": "Commit with distinct committer date",
                        },
                        "author": {"login": "author_user", "id": 111},
                    }
                ],
                total_requests=1,
            )

            mutated = set()
            count, is_part, _ = await collector.collect_commits(
                repo=repo, budgeter=budgeter, db=in_memory_db, mutated_records=mutated
            )

            assert count == 1
            assert ("commit", "sha_committer_valid") in mutated

            cmt = in_memory_db.scalar(select(Commit).where(Commit.sha == "sha_committer_valid"))
            assert cmt is not None
            assert cmt.committer_date is not None
            assert cmt.committer_date.year == 2026
            assert cmt.committer_date.day == 2

        asyncio.run(_run())

    def test_committer_date_null_preservation_when_missing(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=102, repo_name="owner/repo2", owner="owner", name="repo2")
            in_memory_db.add(repo)
            in_memory_db.commit()

            mock_client.paginate.return_value = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "sha_committer_missing",
                        "commit": {
                            "author": {
                                "name": "Author",
                                "email": "a@example.com",
                                "date": "2026-10-01T10:00:00Z",
                            },
                            # "committer" is absent entirely
                            "message": "Commit with no committer object",
                        },
                        "author": {"login": "author_user", "id": 111},
                    },
                    {
                        "sha": "sha_committer_empty",
                        "commit": {
                            "author": {
                                "name": "Author",
                                "email": "a@example.com",
                                "date": "2026-10-01T10:00:00Z",
                            },
                            "committer": {},  # empty committer
                            "message": "Commit with empty committer",
                        },
                        "author": None,
                    },
                ],
                total_requests=1,
            )

            mutated = set()
            count, _, _ = await collector.collect_commits(
                repo=repo, budgeter=budgeter, db=in_memory_db, mutated_records=mutated
            )

            assert count == 2
            cmt1 = in_memory_db.scalar(select(Commit).where(Commit.sha == "sha_committer_missing"))
            assert cmt1.committer_date is None
            # Must not fall back to author_date
            assert cmt1.author_date is not None

            cmt2 = in_memory_db.scalar(select(Commit).where(Commit.sha == "sha_committer_empty"))
            assert cmt2.committer_date is None

        asyncio.run(_run())


class TestAtomicRevisionAdvancement:
    """Tests for record-level mutation tracking and atomic revision advancement."""

    def test_new_repository_and_entities_advance_revision(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            # Metadata response
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 201,
                "name": "project",
                "full_name": "corp/project",
                "description": "Initial description",
                "language": "Python",
                "stargazers_count": 100,
                "forks_count": 10,
                "open_issues_count": 2,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            mock_client.request.return_value = meta_resp

            # Commits pagination response
            commits_res = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c101",
                        "commit": {
                            "author": {"name": "Dev", "email": "d@corp.com", "date": "2026-10-01T10:00:00Z"},
                            "committer": {"name": "Dev", "email": "d@corp.com", "date": "2026-10-01T10:00:00Z"},
                            "message": "Initial commit",
                        },
                        "author": {"login": "dev1", "id": 501},
                    }
                ],
                total_requests=1,
            )

            # Generic empty paginations for others
            empty_res = PaginatedResult(entity="generic", items=[], total_requests=1)

            def mock_paginate(*args, **kwargs):
                if kwargs.get("entity") == "commits":
                    return commits_res
                return empty_res

            mock_client.paginate.side_effect = mock_paginate

            summary: IngestionSummary = await collector.collect_repository(
                repo_name="corp/project",
                mode="fast_snapshot",
                db=in_memory_db,
            )

            assert summary.is_partial is False
            assert summary.records_mutated > 0

            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 201))
            assert repo is not None
            # Default is 1, incremented by 1 on first successful transaction with mutations -> 2
            assert repo.analytics_source_revision == 2
            assert repo.last_synced_at is not None

        asyncio.run(_run())

    def test_identical_reingestion_is_noop_and_preserves_revision(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            # Metadata response
            meta_json = {
                "id": 202,
                "name": "idempotent-repo",
                "full_name": "corp/idempotent-repo",
                "description": "Stable repo",
                "language": "Go",
                "stargazers_count": 250,
                "forks_count": 30,
                "open_issues_count": 1,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            meta_resp = MagicMock()
            meta_resp.json.return_value = meta_json
            mock_client.request.return_value = meta_resp

            commits_res = PaginatedResult(
                entity="commits",
                items=[
                    {
                        "sha": "c201",
                        "commit": {
                            "author": {"name": "Gopher", "email": "g@corp.com", "date": "2026-10-01T10:00:00Z"},
                            "committer": {"name": "Gopher", "email": "g@corp.com", "date": "2026-10-01T10:00:00Z"},
                            "message": "Stable commit",
                        },
                        "author": {"login": "gopher1", "id": 601},
                    }
                ],
                total_requests=1,
            )
            empty_res = PaginatedResult(entity="generic", items=[], total_requests=1)

            mock_client.paginate.side_effect = lambda *args, **kwargs: (
                commits_res if kwargs.get("entity") == "commits" else empty_res
            )

            # 1. Initial Ingestion
            summary1 = await collector.collect_repository(
                repo_name="corp/idempotent-repo",
                mode="fast_snapshot",
                db=in_memory_db,
            )
            assert summary1.records_mutated > 0
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 202))
            assert repo.analytics_source_revision == 2
            initial_synced_at = repo.last_synced_at

            # 2. Identical Re-Ingestion (No data changes)
            summary2 = await collector.collect_repository(
                repo_name="corp/idempotent-repo",
                mode="fast_snapshot",
                db=in_memory_db,
            )

            # Distinct records mutated MUST be exactly 0
            assert summary2.records_mutated == 0

            # Revision MUST remain exactly 2
            in_memory_db.refresh(repo)
            assert repo.analytics_source_revision == 2

            # Operational field last_synced_at is updated
            assert repo.last_synced_at is not None

        asyncio.run(_run())

    def test_operational_metadata_change_only_preserves_revision(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            meta_json = {
                "id": 203,
                "name": "op-repo",
                "full_name": "corp/op-repo",
                "description": "Old description",
                "language": "Rust",
                "stargazers_count": 500,
                "forks_count": 50,
                "open_issues_count": 0,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            meta_resp = MagicMock()
            meta_resp.json.return_value = meta_json
            mock_client.request.return_value = meta_resp

            empty_res = PaginatedResult(entity="generic", items=[], total_requests=1)
            mock_client.paginate.return_value = empty_res

            # Initial Ingestion
            summary1 = await collector.collect_repository(
                repo_name="corp/op-repo",
                mode="fast_snapshot",
                db=in_memory_db,
            )
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 203))
            assert repo.analytics_source_revision == 2

            # Update ONLY operational metadata: description & github_updated_at
            meta_json["description"] = "New fancy description!"
            meta_json["updated_at"] = "2026-11-05T00:00:00Z"

            summary2 = await collector.collect_repository(
                repo_name="corp/op-repo",
                mode="fast_snapshot",
                db=in_memory_db,
            )

            # Operational-only changes must not register as analytical mutations
            assert summary2.records_mutated == 0
            in_memory_db.refresh(repo)
            assert repo.description == "New fancy description!"
            assert repo.analytics_source_revision == 2

        asyncio.run(_run())

    def test_analytical_field_change_advances_revision(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            meta_json = {
                "id": 204,
                "name": "analytical-repo",
                "full_name": "corp/analytical-repo",
                "description": "Same description",
                "language": "Rust",
                "stargazers_count": 100,
                "forks_count": 20,
                "open_issues_count": 0,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-11-01T00:00:00Z",
                "archived": False,
                "fork": False,
            }
            meta_resp = MagicMock()
            meta_resp.json.return_value = meta_json
            mock_client.request.return_value = meta_resp
            mock_client.paginate.return_value = PaginatedResult(entity="generic", items=[], total_requests=1)

            # Run 1
            await collector.collect_repository(repo_name="corp/analytical-repo", db=in_memory_db)
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 204))
            assert repo.analytics_source_revision == 2

            # Analytical change: star count jumps from 100 to 150
            meta_json["stargazers_count"] = 150

            # Run 2
            summary2 = await collector.collect_repository(repo_name="corp/analytical-repo", db=in_memory_db)
            assert summary2.records_mutated == 1  # 1 repository entity mutated
            in_memory_db.refresh(repo)
            assert repo.stars_count == 150
            assert repo.analytics_source_revision == 3

        asyncio.run(_run())

    def test_multiple_mutations_on_same_entity_count_once(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            repo = Repository(id=205, repo_name="corp/multi-touch", owner="corp", name="multi-touch")
            in_memory_db.add(repo)
            in_memory_db.commit()

            # Mock 1 PR
            mock_client.paginate.return_value = PaginatedResult(
                entity="pull_requests",
                items=[
                    {
                        "number": 10,
                        "title": "Feature PR",
                        "state": "open",
                        "user": {"login": "dev1", "id": 1001},
                        "created_at": "2026-10-01T10:00:00Z",
                        "comments": 0,
                        "review_comments": 1,
                    }
                ],
                total_requests=1,
            )

            # Mock review response for PR 10 that calculates turnaround
            review_resp = MagicMock()
            review_resp.json.return_value = [
                {
                    "id": 901,
                    "user": {"login": "reviewer", "id": 1002},
                    "state": "APPROVED",
                    "submitted_at": "2026-10-01T12:00:00Z",
                    "body": "Looks good!",
                }
            ]
            mock_client.request.return_value = review_resp

            mutated: set = set()
            # 1. Collect PR
            prs, count, _, _ = await collector.collect_pull_requests(
                repo=repo, budgeter=budgeter, db=in_memory_db, mutated_records=mutated
            )
            # PR #10 is in mutated
            assert ("pull_request", 10) in mutated

            # 2. Collect Reviews on that same PR (updates pr.time_to_first_review_hours)
            rev_count, _, _ = await collector.collect_pr_reviews(
                repo=repo, pull_requests=prs, budgeter=budgeter, db=in_memory_db, mutated_records=mutated
            )

            # PR #10 was updated with review turnaround, but must appear only once in mutated
            pr_entries = [m for m in mutated if m[0] == "pull_request" and m[1] == 10]
            assert len(pr_entries) == 1
            assert ("pr_review", 901) in mutated

        asyncio.run(_run())

    def test_partial_commit_on_budget_exhaustion_advances_revision(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="fast_snapshot")

            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 206,
                "name": "budget-repo",
                "full_name": "corp/budget-repo",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Simulate budget exhaustion during commits collection
            mock_client.paginate.side_effect = BudgetExceededException(
                message="Budget exhausted.", max_allowed=5
            )

            summary = await collector.collect_repository(
                repo_name="corp/budget-repo",
                mode="fast_snapshot",
                db=in_memory_db,
                budgeter=budgeter,
            )

            assert summary.is_partial is True
            # Metadata was collected and inserted before exception
            assert summary.records_mutated == 1  # 1 repo inserted
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 206))
            assert repo is not None
            # Atomic commit on budget exhaustion must have incremented revision
            assert repo.analytics_source_revision == 2

        asyncio.run(_run())

    def test_partial_commit_on_rate_limit_advances_revision(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)
            budgeter = EntityRateBudgeter(mode="standard_diagnostic")

            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 207,
                "name": "ratelimit-repo",
                "full_name": "corp/ratelimit-repo",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Simulate rate limit during commits collection
            resumes_dt = datetime(2026, 11, 10, 2, 0, 0, tzinfo=timezone.utc)
            mock_client.paginate.side_effect = GitHubRateLimitExceededException(
                message="Rate limited.", resumes_at=resumes_dt
            )

            summary = await collector.collect_repository(
                repo_name="corp/ratelimit-repo",
                mode="standard_diagnostic",
                db=in_memory_db,
                budgeter=budgeter,
            )

            assert summary.is_partial is True
            assert summary.records_mutated == 1
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 207))
            assert repo is not None
            assert repo.analytics_source_revision == 2

        asyncio.run(_run())

    def test_reconcile_contributor_metrics_records_mutations_only_on_diff(self, in_memory_db):
        collector = RepositoryCollector()
        repo = Repository(id=209, repo_name="corp/reconcile-test", owner="corp", name="reconcile-test")
        in_memory_db.add(repo)
        in_memory_db.flush()

        contributor = Contributor(
            id=301,
            repository_id=repo.id,
            github_id=9999,
            login="leaddev",
            total_commits=999,  # Inflated counter
            total_prs_authored=999,
            total_prs_reviewed=999,
            total_issues_opened=999,
        )
        in_memory_db.add(contributor)
        in_memory_db.flush()

        # Seed 2 real commits
        in_memory_db.add_all([
            Commit(repository_id=repo.id, sha="c1", author_login="leaddev", author_id=9999, author_date=datetime(2026, 1, 1, tzinfo=timezone.utc)),
            Commit(repository_id=repo.id, sha="c2", author_login="leaddev", author_id=9999, author_date=datetime(2026, 1, 2, tzinfo=timezone.utc)),
        ])
        in_memory_db.commit()

        # 1. First reconciliation: detects difference from 999 to 2
        mutated_pass1 = set()
        collector.reconcile_contributor_metrics(repository_id=repo.id, db=in_memory_db, mutated_records=mutated_pass1)
        in_memory_db.commit()

        assert ("contributor", 301) in mutated_pass1
        assert contributor.total_commits == 2

        # 2. Second reconciliation: data is already 2, no change
        mutated_pass2 = set()
        collector.reconcile_contributor_metrics(repository_id=repo.id, db=in_memory_db, mutated_records=mutated_pass2)
        in_memory_db.commit()

        assert len(mutated_pass2) == 0

    def test_rollback_on_unexpected_error_reverts_revision_increment(self, in_memory_db, mock_client):
        async def _run():
            collector = RepositoryCollector(client=mock_client)

            # Metadata succeeds
            meta_resp = MagicMock()
            meta_resp.json.return_value = {
                "id": 208,
                "name": "crash-repo",
                "full_name": "corp/crash-repo",
                "created_at": "2026-01-01T00:00:00Z",
            }
            mock_client.request.return_value = meta_resp

            # Commits crashes with an unhandled RuntimeError
            mock_client.paginate.side_effect = RuntimeError("Fatal hardware failure")

            summary = await collector.collect_repository(
                repo_name="corp/crash-repo",
                db=in_memory_db,
            )

            assert len(summary.errors) > 0
            assert "Fatal hardware failure" in summary.errors[0]

            # Since db.rollback() was called, the newly inserted repository was rolled back!
            repo = in_memory_db.scalar(select(Repository).where(Repository.id == 208))
            assert repo is None

        asyncio.run(_run())

    def test_benchmark_data_and_null_committer_dates_preserved(self):
        """Verifies that the 2,328 benchmark issues and all existing commits in the project DB remain intact with committer_date = NULL."""
        import os
        from sqlalchemy import create_engine, select, func
        from sqlalchemy.orm import Session
        from backend.app.database.models import Issue, Commit

        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip("Local database data/opensource_lens.db not present.")

        engine = create_engine(f"sqlite:///{db_path}")
        with Session(engine) as session:
            # 1. Verify 2,328 benchmark issues
            issue_count = session.scalar(select(func.count(Issue.id)))
            assert issue_count == 2328, f"Expected 2328 benchmark issues, found {issue_count}"

            # 2. Verify all existing benchmark commits have committer_date IS NULL
            total_commits = session.scalar(select(func.count(Commit.id)))
            non_null_committer_dates = session.scalar(
                select(func.count(Commit.id)).where(Commit.committer_date.isnot(None))
            )
            assert non_null_committer_dates == 0, (
                f"Expected 0 non-null committer_date commits in benchmark, found {non_null_committer_dates} / {total_commits}"
            )
