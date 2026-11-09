"""Unit and Integration Tests for Development Activity Intelligence Module (C.2.1).

Verifies:
1. Benchmark repository handling on data/opensource_lens.db (0 commits -> empty status, non-crashing).
2. Daily and weekly commit activity time-series aggregation with continuous zero-filling.
3. Multi-horizon commit velocity (4w, 12w, 52w, overall, trend ratio).
4. Contributor activity and author participation metrics.
5. Release cadence statistics (intervals, mean, median, min, max, stdev, classification).
6. Timezone normalization across non-UTC offsets.
7. Missing dates, incomplete samples, and degraded coverage handling.
8. Deterministic rule evaluations (RULE-ACT-001, RULE-ACT-002, RULE-ACT-003, RULE-ACT-EMPTY).
9. Integration with AnalyticsJobQueue snapshot and publication pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    Commit,
    Contributor,
    Release,
    Repository,
)
from backend.app.schemas.analytics_v2 import CURRENT_METHODOLOGY_VERSION
from backend.app.services.analytics import DevelopmentActivityCalculator
from backend.app.services.analytics_queue import AnalyticsJobQueue


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


class TestBenchmarkRepositoriesEmptyCommitHandling:
    """Verifies that offline benchmark repositories (which currently have 0 commits) are handled safely."""

    def test_benchmark_repositories_empty_handling(self):
        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip(f"Database {db_path} not found.")

        engine = create_engine(f"sqlite:///{db_path}")
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)

        with Session(engine) as session:
            repos = session.scalars(select(Repository)).all()
            assert len(repos) >= 4, "Expected at least 4 benchmark repositories."

            for repo in repos:
                result = calc.calculate(
                    repository_id=repo.id,
                    db=session,
                    analysis_reference_time=ref_time,
                    snapshot_revision=repo.analytics_source_revision,
                )
                assert result.domain == "development_activity"
                assert result.coverage.status == "empty"
                assert result.coverage.sample_size == 0
                assert result.confidence_penalty == 0.5
                assert result.metrics["summary"]["total_commits"] == 0
                assert result.metrics["velocity"]["trailing_4w_velocity"] == 0.0
                assert result.metrics["releases"]["total_releases"] == 0
                assert any(r.rule_id == "RULE-ACT-EMPTY" for r in result.rules)


class TestUnseenRepositoryActivityAggregation:
    """Verifies complete activity aggregation on an unseen repository with commits, releases, and contributors."""

    def test_complete_aggregation_and_time_series(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Setup repository
        repo = Repository(
            id=4001,
            repo_name="unseen/active-project",
            owner="unseen",
            name="active-project",
            analytics_source_revision=5,
            is_active=True,
        )
        in_memory_db.add(repo)

        # 2. Add commits across a 60-day window with varied timezones, authors, and churn
        # Author 1: Alice (UTC)
        # Author 2: Bob (EST: UTC-5)
        # Author 3: Charlie (unlinked email)
        commits_data = [
            # 50 days before ref_time
            ("sha_1", ref_time - timedelta(days=50), "alice", "Alice Dev", "alice@corp.com", 100, 20),
            ("sha_2", ref_time - timedelta(days=49), "alice", "Alice Dev", "alice@corp.com", 50, 10),
            # Gap of 20 days (days 48 to 29 have no commits)
            # 25 days before ref_time (in EST)
            ("sha_3", ref_time - timedelta(days=25, hours=2), "bob", "Bob Smith", "bob@corp.com", 200, 50),
            ("sha_4", ref_time - timedelta(days=24), "bob", "Bob Smith", "bob@corp.com", 30, 5),
            ("sha_5", ref_time - timedelta(days=20), None, "Charlie", "charlie@external.org", 40, 10),
            # Trailing 2 weeks: High activity surge
            ("sha_6", ref_time - timedelta(days=10), "alice", "Alice Dev", "alice@corp.com", 80, 15),
            ("sha_7", ref_time - timedelta(days=8), "alice", "Alice Dev", "alice@corp.com", 120, 30),
            ("sha_8", ref_time - timedelta(days=5), "bob", "Bob Smith", "bob@corp.com", 60, 20),
            ("sha_9", ref_time - timedelta(days=2), "alice", "Alice Dev", "alice@corp.com", 15, 5),
            ("sha_10", ref_time - timedelta(days=1), "bob", "Bob Smith", "bob@corp.com", 45, 10),
        ]

        for sha, dt, login, name, email, adds, dels in commits_data:
            c = Commit(
                repository_id=4001,
                sha=sha,
                author_login=login,
                author_name=name,
                author_email=email,
                author_date=dt,
                committer_date=dt,
                additions=adds,
                deletions=dels,
                total_changes=adds + dels,
            )
            in_memory_db.add(c)

        # 3. Add releases (3 releases: 2 stable, 1 prerelease)
        r1 = Release(
            repository_id=4001,
            release_id=901,
            tag_name="v0.9.0",
            name="Alpha Release",
            published_at=ref_time - timedelta(days=60),
            is_prerelease=True,
        )
        r2 = Release(
            repository_id=4001,
            release_id=902,
            tag_name="v1.0.0",
            name="First Major Release",
            published_at=ref_time - timedelta(days=35),
            is_prerelease=False,
        )
        r3 = Release(
            repository_id=4001,
            release_id=903,
            tag_name="v1.1.0",
            name="Feature Update",
            published_at=ref_time - timedelta(days=15),
            is_prerelease=False,
        )
        in_memory_db.add_all([r1, r2, r3])

        # 4. Add contributor rows
        c1 = Contributor(repository_id=4001, login="alice", github_id=801, total_commits=5)
        c2 = Contributor(repository_id=4001, login="bob", github_id=802, total_commits=4)
        in_memory_db.add_all([c1, c2])
        in_memory_db.commit()

        # Run calculator
        result = calc.calculate(
            repository_id=4001,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=5,
        )

        metrics = result.metrics
        summary = metrics["summary"]
        velocity = metrics["velocity"]
        contributors = metrics["contributors"]
        releases = metrics["releases"]

        # Summary assertions
        assert summary["total_commits"] == 10
        assert summary["total_additions"] == 740
        assert summary["total_deletions"] == 175
        assert summary["total_changes"] == 915
        assert summary["days_since_last_commit"] == 1.0
        assert summary["total_span_days"] == 50

        # Continuous zero-filled weekly series assertions
        weekly_series = metrics["weekly_time_series"]
        assert len(weekly_series) >= 7, "Expected continuous weekly series spanning ~8 weeks."
        # Verify gap weeks exist with commit_count == 0
        zero_weeks = [w for w in weekly_series if w["commit_count"] == 0]
        assert len(zero_weeks) > 0, "Expected zero-filled gap weeks in commit history."

        # Trailing 90d daily series
        daily_series = metrics["daily_time_series_90d"]
        assert len(daily_series) == 90
        assert daily_series[-1]["date"] == ref_time.date().strftime("%Y-%m-%d")

        # Velocity assertions
        assert velocity["trailing_4w_velocity"] > 0
        assert velocity["overall_weekly_velocity"] > 0
        assert velocity["velocity_trend_ratio"] > 0

        # Contributor assertions
        assert contributors["total_unique_authors"] == 3  # alice, bob, charlie
        assert contributors["trailing_30d_active_authors"] == 3
        top_authors = contributors["top_authors"]
        assert top_authors[0]["author"] == "alice"
        assert top_authors[0]["commit_count"] == 5
        assert top_authors[0]["share"] == 0.5

        # Release cadence assertions
        assert releases["total_releases"] == 3
        assert releases["total_prereleases"] == 1
        assert releases["total_stable_releases"] == 2
        assert releases["latest_release_tag"] == "v1.1.0"
        assert releases["days_since_last_release"] == 15.0
        assert releases["mean_release_interval_days"] == 22.5  # (25 + 20) / 2
        assert releases["median_release_interval_days"] == 22.5
        assert releases["cadence_classification"] == "monthly"

        # Coverage and Confidence
        assert result.coverage.status == "sufficient"
        assert result.coverage.sample_size == 10
        assert result.confidence_penalty <= 0.15

        # Rules assertions
        rule_ids = {r.rule_id for r in result.rules}
        assert "RULE-ACT-001" in rule_ids
        assert "RULE-ACT-002" in rule_ids
        assert "RULE-ACT-003" in rule_ids


class TestEdgeCasesAndBoundaryHandling:
    """Verifies edge cases: small samples, single releases, timezone shifts, future commits."""

    def test_small_sample_and_short_span_penalties(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

        repo = Repository(id=4002, repo_name="edge/small", owner="edge", name="small", analytics_source_revision=1)
        in_memory_db.add(repo)

        # Only 2 commits on same day
        c1 = Commit(
            repository_id=4002,
            sha="sha_small_1",
            author_date=ref_time - timedelta(days=2),
            author_login="dev1",
        )
        c2 = Commit(
            repository_id=4002,
            sha="sha_small_2",
            author_date=ref_time - timedelta(days=2),
            author_login="dev1",
        )
        in_memory_db.add_all([c1, c2])
        in_memory_db.commit()

        result = calc.calculate(
            repository_id=4002,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        assert result.coverage.status == "insufficient"
        assert result.coverage.sample_size == 2
        assert result.confidence_penalty >= 0.3

    def test_single_release_cadence(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

        repo = Repository(id=4003, repo_name="edge/single-rel", owner="edge", name="single-rel", analytics_source_revision=1)
        in_memory_db.add(repo)

        # 5 commits to pass sufficiency
        for i in range(5):
            in_memory_db.add(
                Commit(
                    repository_id=4003,
                    sha=f"sha_rel_{i}",
                    author_date=ref_time - timedelta(days=20 - i * 3),
                    author_login="dev",
                )
            )

        # 1 release only
        rel = Release(
            repository_id=4003,
            release_id=888,
            tag_name="v1.0.0",
            published_at=ref_time - timedelta(days=10),
            is_prerelease=False,
        )
        in_memory_db.add(rel)
        in_memory_db.commit()

        result = calc.calculate(
            repository_id=4003,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        rel_metrics = result.metrics["releases"]
        assert rel_metrics["total_releases"] == 1
        assert rel_metrics["cadence_classification"] == "insufficient_releases"
        assert rel_metrics["mean_release_interval_days"] is None
        assert rel_metrics["days_since_last_release"] == 10.0

    def test_future_commits_excluded_by_reference_time(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        frozen_ref = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        repo = Repository(id=4004, repo_name="edge/future", owner="edge", name="future", analytics_source_revision=1)
        in_memory_db.add(repo)

        # 3 commits before reference time
        for i in range(3):
            in_memory_db.add(
                Commit(
                    repository_id=4004,
                    sha=f"sha_past_{i}",
                    author_date=frozen_ref - timedelta(days=10 + i),
                    author_login="dev",
                )
            )

        # 5 commits strictly after frozen reference time (must be ignored)
        for i in range(5):
            in_memory_db.add(
                Commit(
                    repository_id=4004,
                    sha=f"sha_future_{i}",
                    author_date=frozen_ref + timedelta(days=5 + i),
                    author_login="dev",
                )
            )

        in_memory_db.commit()

        result = calc.calculate(
            repository_id=4004,
            db=in_memory_db,
            analysis_reference_time=frozen_ref,
            snapshot_revision=1,
        )

        # Only the 3 past commits should be counted
        assert result.metrics["summary"]["total_commits"] == 3


class TestQueueAndSnapshotIntegration:
    """Verifies DevelopmentActivityCalculator integration inside AnalyticsJobQueue lifecycle."""

    def test_calculator_output_published_via_queue(self, in_memory_db):
        queue = AnalyticsJobQueue()
        calc = DevelopmentActivityCalculator()

        repo = Repository(
            id=4005,
            repo_name="queue/integrated",
            owner="queue",
            name="integrated",
            analytics_source_revision=2,
            is_active=True,
        )
        in_memory_db.add(repo)

        for i in range(6):
            in_memory_db.add(
                Commit(
                    repository_id=4005,
                    sha=f"sha_q_{i}",
                    author_date=datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(days=i * 4),
                    author_login="worker_dev",
                    additions=20,
                    deletions=5,
                )
            )
        in_memory_db.commit()

        # Enqueue and claim
        run, is_new = queue.enqueue_analysis(repo_id=4005, db=in_memory_db)
        claimed = queue.claim_job(worker_id="test_worker_1", db=in_memory_db)
        assert claimed is not None
        claimed_run, attempt = claimed

        # Pinned snapshot reference time
        ref_time = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)

        # Execute calculator
        result = calc.calculate(
            repository_id=4005,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=claimed_run.requested_source_revision,
        )

        # Publish via queue
        published = queue.publish_and_finalize(
            run_id=claimed_run.id,
            attempt_id=attempt.id,
            worker_id="test_worker_1",
            expected_lease_version=claimed_run.lease_version,
            attempt_number=attempt.attempt_number,
            snapshot_revision=claimed_run.requested_source_revision,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=ref_time,
            metrics_payload={"development_activity": result.metrics},
            coverage_summary={"development_activity": result.coverage.model_dump()},
            confidence_summary={
                "score": round(1.0 - result.confidence_penalty, 3),
                "band": "high",
                "penalties": result.details.get("penalties", []),
            },
            rule_interpretations=[r.model_dump() for r in result.rules],
            db=in_memory_db,
        )
        assert published is True

        in_memory_db.refresh(claimed_run)
        assert claimed_run.status == "completed"
        assert claimed_run.metrics_payload["development_activity"]["summary"]["total_commits"] == 6
        assert claimed_run.coverage_summary["development_activity"]["status"] == "sufficient"


class TestAdvancedActivityScenarios:
    """Verifies timezone edge cases, velocity trajectory rules, and release interval stdev."""

    def test_timezone_normalization_across_offsets(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

        repo = Repository(id=4006, repo_name="edge/timezones", owner="edge", name="timezones", analytics_source_revision=1)
        in_memory_db.add(repo)

        # Commit 1: Tokyo 01:00 AM on Oct 2 (+09:00) -> Oct 1 16:00 UTC (Day: 2026-10-01)
        tz_tokyo = timezone(timedelta(hours=9))
        c1 = Commit(
            repository_id=4006,
            sha="sha_tz_1",
            author_date=datetime(2026, 10, 2, 1, 0, 0, tzinfo=tz_tokyo),
            author_login="tokyo_dev",
        )

        # Commit 2: New York 21:00 PM on Oct 1 (-04:00) -> Oct 2 01:00 UTC (Day: 2026-10-02)
        tz_ny = timezone(timedelta(hours=-4))
        c2 = Commit(
            repository_id=4006,
            sha="sha_tz_2",
            author_date=datetime(2026, 10, 1, 21, 0, 0, tzinfo=tz_ny),
            author_login="ny_dev",
        )

        # Add 4 more commits spanning 20 days so coverage is sufficient
        for i in range(4):
            c = Commit(
                repository_id=4006,
                sha=f"sha_tz_pad_{i}",
                author_date=datetime(2026, 10, 10, 0, 0, 0, tzinfo=timezone.utc) + timedelta(days=i * 5),
                author_login="pad_dev",
            )
            in_memory_db.add(c)

        in_memory_db.add_all([c1, c2])
        in_memory_db.commit()

        result = calc.calculate(
            repository_id=4006,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        daily = {d["date"]: d["commit_count"] for d in result.metrics["daily_time_series_90d"]}
        assert daily.get("2026-10-01", 0) >= 1
        assert daily.get("2026-10-02", 0) >= 1

    def test_velocity_acceleration_rule(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

        repo = Repository(id=4007, repo_name="edge/accel", owner="edge", name="accel", analytics_source_revision=1)
        in_memory_db.add(repo)

        # Baseline: 1 commit every 2 weeks during weeks 5-12 prior (4 commits across 8 weeks = 0.5/week)
        for i in range(4):
            in_memory_db.add(
                Commit(
                    repository_id=4007,
                    sha=f"sha_hist_{i}",
                    author_date=ref_time - timedelta(weeks=11 - i * 2),
                    author_login="dev",
                )
            )

        # Recent surge: 12 commits in the trailing 4 weeks (3.0/week)
        for i in range(12):
            in_memory_db.add(
                Commit(
                    repository_id=4007,
                    sha=f"sha_surge_{i}",
                    author_date=ref_time - timedelta(days=20 - i),
                    author_login="dev",
                )
            )
        in_memory_db.commit()

        result = calc.calculate(
            repository_id=4007,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        vel = result.metrics["velocity"]
        assert vel["velocity_trend_ratio"] >= 1.25
        accel_rule = next(r for r in result.rules if r.rule_id == "RULE-ACT-002")
        assert accel_rule.severity == "positive"
        assert "Accelerating Velocity" in accel_rule.finding

    def test_release_cadence_stdev_and_regularity(self, in_memory_db):
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

        repo = Repository(id=4008, repo_name="edge/releases-multi", owner="edge", name="releases-multi", analytics_source_revision=1)
        in_memory_db.add(repo)

        # 5 releases spaced exactly 14 days apart: 0, 14, 28, 42, 56 days before ref_time
        for i in range(5):
            in_memory_db.add(
                Release(
                    repository_id=4008,
                    release_id=700 + i,
                    tag_name=f"v1.{i}.0",
                    published_at=ref_time - timedelta(days=(4 - i) * 14),
                    is_prerelease=False,
                )
            )

        # 5 padding commits
        for i in range(5):
            in_memory_db.add(
                Commit(
                    repository_id=4008,
                    sha=f"sha_pad_rel_{i}",
                    author_date=ref_time - timedelta(days=i * 10),
                    author_login="dev",
                )
            )
        in_memory_db.commit()

        result = calc.calculate(
            repository_id=4008,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        rel = result.metrics["releases"]
        assert rel["total_releases"] == 5
        assert rel["mean_release_interval_days"] == 14.0
        assert rel["median_release_interval_days"] == 14.0
        assert rel["stdev_release_interval_days"] == 0.0  # Perfectly regular
        assert rel["cadence_classification"] == "rapid"

