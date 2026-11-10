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


class TestTrailingVelocityWindowRefinement:
    """Regression tests for exact rolling day-window velocities (28d, 84d, 364d)."""

    def test_monday_vs_friday_equivalent_activity_rates(self, in_memory_db):
        """Steady daily commit cadence must produce identical velocity on Monday vs Friday."""
        calc = DevelopmentActivityCalculator()
        repo = Repository(id=4009, repo_name="steady/mon-fri", owner="steady", name="mon-fri", analytics_source_revision=1)
        in_memory_db.add(repo)

        # 1 commit every day at 12:00 UTC from June 1, 2026 through Nov 15, 2026 (>160 days)
        start_day = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        for day_idx in range(168):
            in_memory_db.add(
                Commit(
                    repository_id=4009,
                    sha=f"sha_steady_{day_idx}",
                    author_date=start_day + timedelta(days=day_idx),
                    author_login="steady_dev",
                )
            )
        in_memory_db.commit()

        # Monday Nov 2, 2026 12:00 UTC (early in calendar week)
        ref_monday = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert ref_monday.weekday() == 0
        res_mon = calc.calculate(4009, in_memory_db, ref_monday, snapshot_revision=1)

        # Friday Nov 6, 2026 12:00 UTC (late in calendar week)
        ref_friday = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)
        assert ref_friday.weekday() == 4
        res_fri = calc.calculate(4009, in_memory_db, ref_friday, snapshot_revision=1)

        # Calendar-week series still reflects partial Monday (1 commit) vs Friday (5 commits)
        assert res_mon.metrics["weekly_time_series"][-1]["week_start"] == "2026-11-02"
        assert res_mon.metrics["weekly_time_series"][-1]["commit_count"] == 1
        assert res_fri.metrics["weekly_time_series"][-1]["week_start"] == "2026-11-02"
        assert res_fri.metrics["weekly_time_series"][-1]["commit_count"] == 5

        # Trailing velocities must be completely invariant to Monday vs Friday day-of-week
        assert res_mon.metrics["velocity"]["trailing_4w_velocity"] == 7.0
        assert res_fri.metrics["velocity"]["trailing_4w_velocity"] == 7.0
        assert res_mon.metrics["velocity"]["trailing_12w_velocity"] == 7.0
        assert res_fri.metrics["velocity"]["trailing_12w_velocity"] == 7.0
        assert res_mon.metrics["velocity"]["velocity_trend_ratio"] == 1.0
        assert res_fri.metrics["velocity"]["velocity_trend_ratio"] == 1.0

        mon_rule = next(r for r in res_mon.rules if r.rule_id == "RULE-ACT-002")
        fri_rule = next(r for r in res_fri.rules if r.rule_id == "RULE-ACT-002")
        assert mon_rule.severity == "info"
        assert fri_rule.severity == "info"

    def test_incomplete_current_week_preserves_steady_velocity(self, in_memory_db):
        """Monday 01:00 UTC with 0 commits in the current hour must not trigger false deceleration."""
        calc = DevelopmentActivityCalculator()
        repo = Repository(id=4010, repo_name="steady/early-mon", owner="steady", name="early-mon", analytics_source_revision=1)
        in_memory_db.add(repo)

        # Anchor: Monday Nov 9, 2026 01:00 UTC (brand new incomplete week with 0 commits)
        ref_mon_early = datetime(2026, 11, 9, 1, 0, 0, tzinfo=timezone.utc)
        assert ref_mon_early.weekday() == 0

        # Seed strictly 3 commits/week on Tue, Thu, Sat at 10:00 UTC for 16 prior weeks
        # First week Monday starts 16 weeks before Nov 9 -> Jul 20, 2026
        first_monday = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
        commit_idx = 0
        for week_idx in range(16):
            week_mon = first_monday + timedelta(weeks=week_idx)
            for offset_days in (1, 3, 5):  # Tue, Thu, Sat
                in_memory_db.add(
                    Commit(
                        repository_id=4010,
                        sha=f"sha_tue_thu_sat_{commit_idx}",
                        author_date=week_mon + timedelta(days=offset_days),
                        author_login="dev",
                    )
                )
                commit_idx += 1
        in_memory_db.commit()

        result = calc.calculate(4010, in_memory_db, ref_mon_early, snapshot_revision=1)

        # Current calendar week in series has 0 commits (Monday 01:00 UTC)
        assert result.metrics["weekly_time_series"][-1]["week_start"] == "2026-11-09"
        assert result.metrics["weekly_time_series"][-1]["commit_count"] == 0

        # Trailing 28d and 84d windows capture exact 4 weeks (12) and 12 weeks (36) of prior Tue/Thu/Sat
        vel = result.metrics["velocity"]
        assert vel["trailing_4w_velocity"] == 3.0
        assert vel["trailing_12w_velocity"] == 3.0
        assert vel["velocity_trend_ratio"] == 1.0
        trend_rule = next(r for r in result.rules if r.rule_id == "RULE-ACT-002")
        assert trend_rule.severity == "info"
        assert "Steady Velocity" in trend_rule.finding

    def test_exact_28d_and_84d_window_boundaries(self, in_memory_db):
        """Verify half-open boundaries (ref_time - N days, ref_time] across 28d, 84d, and 364d."""
        calc = DevelopmentActivityCalculator()
        repo = Repository(id=4011, repo_name="edge/boundaries", owner="edge", name="boundaries", analytics_source_revision=1)
        in_memory_db.add(repo)

        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        boundary_commits = [
            ("sha_future", ref_time + timedelta(seconds=1)),                          # excluded completely
            ("sha_at_ref", ref_time),                                                # in 28d, 84d, 364d
            ("sha_28d_in", ref_time - timedelta(days=28) + timedelta(seconds=1)),    # in 28d, 84d, 364d
            ("sha_28d_out", ref_time - timedelta(days=28)),                          # in 84d, 364d (excluded from 28d)
            ("sha_84d_in", ref_time - timedelta(days=84) + timedelta(seconds=1)),    # in 84d, 364d
            ("sha_84d_out", ref_time - timedelta(days=84)),                          # in 364d (excluded from 84d)
            ("sha_364d_in", ref_time - timedelta(days=364) + timedelta(seconds=1)),  # in 364d
            ("sha_364d_out", ref_time - timedelta(days=364)),                        # excluded from 364d
        ]

        for sha, dt in boundary_commits:
            in_memory_db.add(Commit(repository_id=4011, sha=sha, author_date=dt, author_login="dev"))
        in_memory_db.commit()

        result = calc.calculate(4011, in_memory_db, ref_time, snapshot_revision=1)

        assert result.metrics["summary"]["total_commits"] == 7
        vel = result.metrics["velocity"]
        # 2 commits in (T-28d, T] -> 2 / 4 = 0.50
        assert vel["trailing_4w_velocity"] == 0.5
        # 4 commits in (T-84d, T] -> 4 / 12 = 0.33
        assert vel["trailing_12w_velocity"] == 0.33
        # 6 commits in (T-364d, T] over full 52w observable history -> 6 / 52 = 0.12
        assert vel["trailing_52w_velocity"] == 0.12

    def test_less_than_52_weeks_history_normalization(self, in_memory_db):
        """Young repos normalize 52w velocity by observable history instead of dividing by 52."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        # Case A: 20-week-old repository (140 days), 20 commits at exact weekly rate of 1/week
        repo_20w = Repository(id=4012, repo_name="young/20w", owner="young", name="20w", analytics_source_revision=1)
        in_memory_db.add(repo_20w)
        for w in range(20):
            in_memory_db.add(
                Commit(
                    repository_id=4012,
                    sha=f"sha_20w_{w}",
                    author_date=ref_time - timedelta(days=140 - w * 7),
                    author_login="dev",
                )
            )

        # Case B: Brand-new 3-day-old repository with 5 commits (floors divisor at 1.0 week)
        repo_3d = Repository(id=4013, repo_name="young/3d", owner="young", name="3d", analytics_source_revision=1)
        in_memory_db.add(repo_3d)
        for i in range(5):
            in_memory_db.add(
                Commit(
                    repository_id=4013,
                    sha=f"sha_3d_{i}",
                    author_date=ref_time - timedelta(days=3) + timedelta(hours=i * 12),
                    author_login="dev",
                )
            )

        # Case C: Mature 450-day repository with 104 commits inside trailing 364d + older history (caps at 52.0 weeks)
        repo_old = Repository(id=4014, repo_name="mature/old", owner="mature", name="old", analytics_source_revision=1)
        in_memory_db.add(repo_old)
        in_memory_db.add(
            Commit(
                repository_id=4014,
                sha="sha_old_anchor",
                author_date=ref_time - timedelta(days=450),
                author_login="dev",
            )
        )
        for i in range(104):
            # 2 commits every 7 days within trailing 364d
            in_memory_db.add(
                Commit(
                    repository_id=4014,
                    sha=f"sha_mature_{i}",
                    author_date=ref_time - timedelta(days=int(i * 3.5)),
                    author_login="dev",
                )
            )
        in_memory_db.commit()

        res_20w = calc.calculate(4012, in_memory_db, ref_time, snapshot_revision=1)
        # 20 commits / 20.0 observed weeks = 1.00 (not 20/52 = 0.38)
        assert res_20w.metrics["velocity"]["trailing_52w_velocity"] == 1.0

        res_3d = calc.calculate(4013, in_memory_db, ref_time, snapshot_revision=1)
        # 5 commits / max(1.0, 3/7) = 5.00
        assert res_3d.metrics["velocity"]["trailing_52w_velocity"] == 5.0

        res_old = calc.calculate(4014, in_memory_db, ref_time, snapshot_revision=1)
        # 104 commits / min(52.0, 450/7) = 2.00
        assert res_old.metrics["velocity"]["trailing_52w_velocity"] == 2.0

    def test_future_commits_and_timezone_normalization_with_aware_reference(self, in_memory_db):
        """Timezone-aware reference time and non-UTC commit timestamps normalize consistently to UTC."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        calc = DevelopmentActivityCalculator()
        # Analysis reference time in JST (+09:00): 2026-11-10 11:00+09:00 == 2026-11-10 02:00:00 UTC
        tz_jst = timezone(timedelta(hours=9))
        tz_est = timezone(timedelta(hours=-5))
        ref_time_jst = datetime(2026, 11, 10, 11, 0, 0, tzinfo=tz_jst)

        # Simulate driver returning mixed timezone-aware datetimes (e.g. PostgreSQL or offset-aware driver)
        # Cutoff 28d in UTC: 2026-10-13 02:00:00 UTC == 2026-10-12 21:00:00 EST
        rows = [
            # Chronologically earliest in UTC (even though listed out-of-order / mixed offsets):
            # 2026-10-12 20:59:59 EST == 2026-10-13 01:59:59 UTC (1s outside 28d, inside 84d)
            SimpleNamespace(
                id=1, sha="s_out_28d", author_date=datetime(2026, 10, 12, 20, 59, 59, tzinfo=tz_est),
                committer_date=None, author_login="est_dev", author_name=None, author_email=None,
                additions=10, deletions=0, total_changes=10,
            ),
            # 2026-10-12 21:00:01 EST == 2026-10-13 02:00:01 UTC (1s inside 28d)
            SimpleNamespace(
                id=2, sha="s_in_28d", author_date=datetime(2026, 10, 12, 21, 0, 1, tzinfo=tz_est),
                committer_date=None, author_login="est_dev", author_name=None, author_email=None,
                additions=10, deletions=0, total_changes=10,
            ),
            # 2026-11-10 10:30:00 JST == 2026-11-10 01:30:00 UTC (30m before ref_time, inside 28d)
            SimpleNamespace(
                id=3, sha="s_past_jst", author_date=datetime(2026, 11, 10, 10, 30, 0, tzinfo=tz_jst),
                committer_date=None, author_login="jst_dev", author_name=None, author_email=None,
                additions=10, deletions=0, total_changes=10,
            ),
            # 2026-11-09 22:00:00 EST == 2026-11-10 03:00:00 UTC (1h AFTER ref_time despite Nov 9 local date!)
            SimpleNamespace(
                id=4, sha="s_future_est", author_date=datetime(2026, 11, 9, 22, 0, 0, tzinfo=tz_est),
                committer_date=None, author_login="est_dev", author_name=None, author_email=None,
                additions=999, deletions=999, total_changes=1998,
            ),
        ]

        mock_db = MagicMock()
        mock_execute_commits = MagicMock()
        mock_execute_commits.fetchall.return_value = rows
        mock_execute_releases = MagicMock()
        mock_execute_releases.fetchall.return_value = []
        mock_db.execute.side_effect = [mock_execute_commits, mock_execute_releases]
        mock_db.scalar.return_value = 2

        result = calc.calculate(9999, mock_db, ref_time_jst, snapshot_revision=1)

        # Future EST commit (Nov 9 22:00-05:00 = Nov 10 03:00Z > 02:00Z) must be excluded
        assert result.metrics["summary"]["total_commits"] == 3
        assert result.metrics["summary"]["total_changes"] == 30
        assert result.metrics["summary"]["earliest_commit_date"] == "2026-10-13T01:59:59+00:00"
        assert result.metrics["summary"]["latest_commit_date"] == "2026-11-10T01:30:00+00:00"

        # 2 commits inside 28d -> 2 / 4 = 0.50; 3 commits inside 84d -> 3 / 12 = 0.25
        assert result.metrics["velocity"]["trailing_4w_velocity"] == 0.5
        assert result.metrics["velocity"]["trailing_12w_velocity"] == 0.25

        # Verify UTC day binning from EST evening and JST morning offsets
        daily = {d["date"]: d["commit_count"] for d in result.metrics["daily_time_series_90d"]}
        assert daily["2026-10-13"] == 2  # Both 2026-10-12 20:59/21:00 EST rolled to Oct 13 UTC
        assert daily["2026-11-10"] == 1  # Nov 10 10:30 JST -> Nov 10 01:30 UTC (excluded Nov 9 EST future commit)

        # Also verify real SQLite DB session queries against non-UTC analysis_reference_time
        repo = Repository(id=4017, repo_name="tz/ref", owner="tz", name="ref", analytics_source_revision=1)
        in_memory_db.add(repo)
        in_memory_db.add_all([
            Commit(repository_id=4017, sha="db_past_1", author_date=datetime(2026, 11, 10, 1, 30, 0, tzinfo=timezone.utc), author_login="dev"),
            Commit(repository_id=4017, sha="db_past_2", author_date=datetime(2026, 10, 13, 2, 0, 1, tzinfo=timezone.utc), author_login="dev"),
            Commit(repository_id=4017, sha="db_future_1", author_date=datetime(2026, 11, 10, 2, 0, 1, tzinfo=timezone.utc), author_login="dev"),
        ])
        in_memory_db.commit()

        db_res = calc.calculate(4017, in_memory_db, ref_time_jst, snapshot_revision=1)
        assert db_res.metrics["summary"]["total_commits"] == 2
        assert db_res.metrics["velocity"]["trailing_4w_velocity"] == 0.5

    def test_deceleration_and_acceleration_rules_after_refinement(self, in_memory_db):
        """Verify RULE-ACT-002 produces expected accelerating and decelerating diagnostics."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Deceleration repository: 36 commits in prior 8 weeks (days -83..-29), only 2 commits in trailing 28d
        repo_decel = Repository(id=4015, repo_name="trend/decel", owner="trend", name="decel", analytics_source_revision=1)
        in_memory_db.add(repo_decel)
        for i in range(36):
            in_memory_db.add(
                Commit(
                    repository_id=4015,
                    sha=f"sha_decel_old_{i}",
                    author_date=ref_time - timedelta(days=30 + i),
                    author_login="dev",
                )
            )
        for i in range(2):
            in_memory_db.add(
                Commit(
                    repository_id=4015,
                    sha=f"sha_decel_recent_{i}",
                    author_date=ref_time - timedelta(days=5 + i * 7),
                    author_login="dev",
                )
            )

        # 2. Acceleration repository: 6 commits in prior 8 weeks, 20 commits in trailing 28d
        repo_accel = Repository(id=4016, repo_name="trend/accel", owner="trend", name="accel", analytics_source_revision=1)
        in_memory_db.add(repo_accel)
        for i in range(6):
            in_memory_db.add(
                Commit(
                    repository_id=4016,
                    sha=f"sha_accel_old_{i}",
                    author_date=ref_time - timedelta(days=35 + i * 8),
                    author_login="dev",
                )
            )
        for i in range(20):
            in_memory_db.add(
                Commit(
                    repository_id=4016,
                    sha=f"sha_accel_recent_{i}",
                    author_date=ref_time - timedelta(days=1 + i),
                    author_login="dev",
                )
            )
        in_memory_db.commit()

        res_decel = calc.calculate(4015, in_memory_db, ref_time, snapshot_revision=1)
        # 28d: 2/4 = 0.50; 84d: 38/12 = 3.17 -> ratio = 0.16 <= 0.75
        assert res_decel.metrics["velocity"]["trailing_4w_velocity"] == 0.5
        assert res_decel.metrics["velocity"]["trailing_12w_velocity"] == 3.17
        assert res_decel.metrics["velocity"]["velocity_trend_ratio"] <= 0.75
        decel_rule = next(r for r in res_decel.rules if r.rule_id == "RULE-ACT-002")
        assert decel_rule.severity == "warning"
        assert "Decelerating Velocity" in decel_rule.finding

        res_accel = calc.calculate(4016, in_memory_db, ref_time, snapshot_revision=1)
        # 28d: 20/4 = 5.00; 84d: 26/12 = 2.17 -> ratio = 2.30 >= 1.25
        assert res_accel.metrics["velocity"]["trailing_4w_velocity"] == 5.0
        assert res_accel.metrics["velocity"]["trailing_12w_velocity"] == 2.17
        assert res_accel.metrics["velocity"]["velocity_trend_ratio"] >= 1.25
        accel_rule = next(r for r in res_accel.rules if r.rule_id == "RULE-ACT-002")
        assert accel_rule.severity == "positive"
        assert "Accelerating Velocity" in accel_rule.finding


