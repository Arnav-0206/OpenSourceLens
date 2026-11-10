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
                assert result.metrics["consistency"]["consistency_classification"] == "insufficient_history"
                assert result.metrics["anomalies"]["status"] == "insufficient_history"
                assert result.metrics["change_points"]["justified"] is False
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
        dev_act = claimed_run.metrics_payload["development_activity"]
        assert dev_act["summary"]["total_commits"] == 6
        assert "consistency" in dev_act and "rolling_statistics" in dev_act
        assert "anomalies" in dev_act and "change_points" in dev_act
        assert claimed_run.coverage_summary["development_activity"]["status"] == "sufficient"
        assert "subdomains" in claimed_run.coverage_summary["development_activity"]["details"]


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


def _seed_completed_weekly_commits(
    db: Session,
    repo_id: int,
    ref_time_utc: datetime,
    weekly_counts: list[int],
    sha_prefix: str = "c22",
) -> None:
    """Seed exact commit counts per completed Monday-Sunday UTC week strictly prior to ref_time's week."""
    current_week_monday = datetime(
        ref_time_utc.year, ref_time_utc.month, ref_time_utc.day, 0, 0, 0, tzinfo=timezone.utc
    ) - timedelta(days=ref_time_utc.weekday())
    first_week_monday = current_week_monday - timedelta(weeks=len(weekly_counts))

    db.add(Repository(id=repo_id, repo_name=f"c22/{sha_prefix}", owner="c22", name=sha_prefix, analytics_source_revision=1))
    commit_seq = 0
    for week_idx, count in enumerate(weekly_counts):
        week_base = first_week_monday + timedelta(weeks=week_idx, days=1, hours=12)  # Tuesday 12:00 UTC
        for j in range(count):
            db.add(
                Commit(
                    repository_id=repo_id,
                    sha=f"sha_{sha_prefix}_{repo_id}_{commit_seq}",
                    author_date=week_base + timedelta(minutes=j * 10),
                    author_login="c22_dev",
                )
            )
            commit_seq += 1
    db.commit()


class TestActivityConsistencyRollingAnomaliesC22:
    """Focused tests for C.2.2 consistency, rolling statistics, anomaly detection, and change-point detection."""

    def test_normal_steady_activity_and_rolling_windows(self, in_memory_db):
        """Steady ~10 commits/week yields consistent_steady, accurate 4w/12w rolling stats, and 0 anomalies."""
        calc = DevelopmentActivityCalculator()
        # Monday 2026-11-09 02:00 UTC (incomplete current week with 0 commits)
        ref_time = datetime(2026, 11, 9, 2, 0, 0, tzinfo=timezone.utc)
        pattern = [9, 10, 11, 10, 10, 9, 11, 10, 10, 11, 9, 10, 10, 11]  # 14 completed weeks
        _seed_completed_weekly_commits(in_memory_db, 5001, ref_time, pattern, "steady")

        res = calc.calculate(5001, in_memory_db, ref_time, snapshot_revision=1)
        cons = res.metrics["consistency"]
        roll = res.metrics["rolling_statistics"]
        anom = res.metrics["anomalies"]
        cp = res.metrics["change_points"]

        assert cons["status"] == "sufficient"
        assert cons["completed_weeks_count"] == 14
        assert cons["consistency_classification"] == "consistent_steady"
        assert cons["active_week_ratio"] == 1.0
        assert cons["coefficient_of_variation"] < 0.15

        # Rolling stats must exclude incomplete Monday 02:00 week and flag warmup windows cleanly
        assert roll["status"] == "sufficient"
        assert roll["incomplete_current_week"]["week_start"] == "2026-11-09"
        assert roll["incomplete_current_week"]["commit_count"] == 0
        assert roll["incomplete_current_week"]["excluded_from_baseline"] is True
        assert roll["series"][0]["window_4w_complete"] is False
        assert roll["series"][0]["rolling_4w_mean"] is None
        assert roll["series"][3]["window_4w_complete"] is True
        assert roll["series"][3]["rolling_4w_mean"] == 10.0  # (9+10+11+10)/4
        assert roll["latest"]["window_12w_complete"] is True
        assert roll["latest"]["rolling_4w_mean"] == 10.0  # (9+10+10+11)/4
        assert roll["latest"]["rolling_12w_mean"] == 10.17  # 122/12

        # Zero false anomalies or false structural breaks
        assert anom["status"] == "analyzed"
        assert anom["total_anomalies_detected"] == 0
        assert anom["ongoing_partial_week_surge"] is None
        assert cp["status"] == "no_significant_shift"
        assert cp["detected_change_points"] == []

        assert any(r.rule_id == "RULE-ACT-004" and r.severity == "positive" for r in res.rules)
        assert not any(r.rule_id == "RULE-ACT-005" for r in res.rules)
        assert not any(r.rule_id == "RULE-ACT-006" for r in res.rules)

    def test_distinguish_consistently_low_from_sparse_and_zero_mean(self, in_memory_db):
        """Consistently low (1/wk) is separated from sparse zero-heavy series; neither triggers false anomalies."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Consistently low: 1 commit every week for 10 completed weeks
        _seed_completed_weekly_commits(in_memory_db, 5002, ref_time, [1] * 10, "low_steady")
        res_low = calc.calculate(5002, in_memory_db, ref_time, snapshot_revision=1)
        assert res_low.metrics["consistency"]["consistency_classification"] == "consistently_low"
        assert res_low.metrics["consistency"]["coefficient_of_variation"] == 0.0
        assert res_low.metrics["anomalies"]["total_anomalies_detected"] == 0
        low_rule = next(r for r in res_low.rules if r.rule_id == "RULE-ACT-004")
        assert low_rule.severity == "info"
        assert "Consistently Low Activity" in low_rule.finding

        # 2. Sparse zero-heavy series: occasional commits across 16 completed weeks (active ratio = 4/16 = 0.25)
        sparse_pattern = [1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]
        _seed_completed_weekly_commits(in_memory_db, 5003, ref_time, sparse_pattern, "sparse")
        res_sparse = calc.calculate(5003, in_memory_db, ref_time, snapshot_revision=1)
        assert res_sparse.metrics["consistency"]["consistency_classification"] == "sparse_intermittent"
        assert res_sparse.metrics["consistency"]["zero_week_ratio"] == 0.75
        assert res_sparse.metrics["anomalies"]["total_anomalies_detected"] == 0
        assert res_sparse.metrics["change_points"]["justified"] is False

        # 3. Zero-mean completed weeks (commits exist only in current in-progress week -> CV safely None)
        db_repo = Repository(id=5004, repo_name="c22/zero_completed", owner="c22", name="zero_completed", analytics_source_revision=1)
        in_memory_db.add(db_repo)
        in_memory_db.add(Commit(repository_id=5004, sha="sha_cur_week_only", author_date=ref_time - timedelta(hours=1), author_login="dev"))
        in_memory_db.commit()
        res_zero = calc.calculate(5004, in_memory_db, ref_time, snapshot_revision=1)
        assert res_zero.metrics["consistency"]["completed_weeks_count"] == 0
        assert res_zero.metrics["consistency"]["coefficient_of_variation"] is None

    def test_sudden_spike_detected_and_separated_from_change_point(self, in_memory_db):
        """A single-week surge (35 vs ~5.5) triggers MAD anomaly with evidence, without false change-point."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
        # Non-zero MAD baseline (4..8, median ~5.5, MAD >= 1.0) followed by isolated spike 35 at week index 10
        pattern = [4, 7, 5, 6, 4, 8, 5, 6, 4, 7, 35, 5, 6, 4, 7, 5]
        _seed_completed_weekly_commits(in_memory_db, 5005, ref_time, pattern, "spike")

        res = calc.calculate(5005, in_memory_db, ref_time, snapshot_revision=1)
        anom = res.metrics["anomalies"]
        cp = res.metrics["change_points"]

        assert anom["status"] == "analyzed"
        assert anom["total_anomalies_detected"] == 1
        assert anom["high_activity_anomalies_count"] == 1
        assert anom["low_activity_anomalies_count"] == 0

        item = anom["items"][0]
        assert item["direction"] == "high"
        assert item["observed_commits"] == 35
        assert item["period_status"] == "completed_week"
        assert item["method"] == "rolling_robust_mad"
        assert 5.0 <= item["baseline"]["median"] <= 6.0
        assert item["deviation"]["score"] >= 10.0
        assert "week_start" in item and "confidence" in item and "limitations" in item

        # Single burst week must NOT register as a permanent structural regime shift
        assert cp["status"] == "no_significant_shift"
        assert cp["detected_change_points"] == []
        assert any(r.rule_id == "RULE-ACT-005" for r in res.rules)
        assert not any(r.rule_id == "RULE-ACT-006" for r in res.rules)

        # Also verify zero-MAD with non-zero stdev selects rolling_zscore fallback cleanly
        z_pattern = [5, 6, 5, 5, 6, 5, 5, 6, 5, 6, 30, 5]
        _seed_completed_weekly_commits(in_memory_db, 5011, ref_time, z_pattern, "zscore_fb")
        res_z = calc.calculate(5011, in_memory_db, ref_time, snapshot_revision=1)
        assert res_z.metrics["anomalies"]["total_anomalies_detected"] == 1
        assert res_z.metrics["anomalies"]["items"][0]["method"] == "rolling_zscore"

    def test_sudden_drop_detected_and_incomplete_monday_not_misflagged(self, in_memory_db):
        """Drop to 0 inside an active baseline triggers drop anomaly; empty incomplete current week does not."""
        calc = DevelopmentActivityCalculator()
        # Monday 00:30 UTC (only 30 mins elapsed in current week -> 0 commits so far)
        ref_time = datetime(2026, 11, 9, 0, 30, 0, tzinfo=timezone.utc)
        # Drop happens at completed week index 9 (0 vs 8-10 baseline), recovering at week 10-11
        pattern = [9, 8, 10, 9, 8, 9, 10, 8, 9, 0, 9, 8]
        _seed_completed_weekly_commits(in_memory_db, 5006, ref_time, pattern, "drop")

        res = calc.calculate(5006, in_memory_db, ref_time, snapshot_revision=1)
        anom = res.metrics["anomalies"]

        assert anom["total_anomalies_detected"] == 1
        assert anom["low_activity_anomalies_count"] == 1
        item = anom["items"][0]
        assert item["direction"] == "low"
        assert item["observed_commits"] == 0
        assert item["deviation"]["score"] <= -2.5
        assert item["baseline"]["median"] >= 8.0
        assert any(r.rule_id == "RULE-ACT-005" and r.severity == "warning" for r in res.rules)

    def test_constant_baseline_poisson_fallback_and_ongoing_partial_week_surge(self, in_memory_db):
        """Zero-MAD constant series uses Poisson dispersion floor and avoids +1 false alarms while catching surges."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)  # Wednesday of current week
        # Strictly constant baseline of 6 commits/wk -> MAD=0, stdev=0; week 8 has routine +1 (7),
        # followed by 8 constant weeks [6]*8, and current partial week (Mon-Wed) has 25 commits!
        pattern = [6] * 8 + [7] + [6] * 8
        _seed_completed_weekly_commits(in_memory_db, 5007, ref_time, pattern, "poisson")

        # Add 25 commits inside ongoing Wednesday current week
        for i in range(25):
            in_memory_db.add(
                Commit(
                    repository_id=5007,
                    sha=f"sha_partial_surge_{i}",
                    author_date=ref_time - timedelta(hours=1 + i),
                    author_login="c22_dev",
                )
            )
        in_memory_db.commit()

        res = calc.calculate(5007, in_memory_db, ref_time, snapshot_revision=1)
        anom = res.metrics["anomalies"]

        # +1 jitter week (7 vs 6) is NOT flagged as an anomaly
        assert anom["total_anomalies_detected"] == 0
        # Ongoing 25-commit surge inside Wednesday partial week IS flagged transparently
        surge = anom["ongoing_partial_week_surge"]
        assert surge is not None
        assert surge["direction"] == "high"
        assert surge["period_status"] == "partial_current_week"
        assert surge["method"] == "rolling_poisson_floored"
        assert surge["observed_commits"] == 25

    def test_structural_change_points_step_increase_and_decrease(self, in_memory_db):
        """Max-SSB permutation test identifies sustained regime shifts accurately with corrected p-values."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Step Increase: 8 weeks at 2-3/wk -> 8 weeks at 18-22/wk
        inc_pattern = [2, 3, 2, 3, 2, 3, 2, 3, 18, 20, 19, 22, 20, 18, 21, 19]
        _seed_completed_weekly_commits(in_memory_db, 5008, ref_time, inc_pattern, "cp_inc")
        res_inc = calc.calculate(5008, in_memory_db, ref_time, snapshot_revision=1)
        cp_inc = res_inc.metrics["change_points"]
        assert cp_inc["status"] == "detected"
        assert cp_inc["method"] == "max_ssb_permutation_test"
        assert cp_inc["permutations"] == 999
        assert cp_inc["justified"] is True
        assert len(cp_inc["detected_change_points"]) == 1
        rec_inc = cp_inc["detected_change_points"][0]
        assert rec_inc["direction"] == "step_increase"
        assert rec_inc["before_segment"]["weeks"] == 8
        assert rec_inc["after_segment"]["weeks"] == 8
        assert rec_inc["statistics"]["permutation_pvalue"] == 0.001
        assert rec_inc["statistics"]["permutations"] == 999
        assert rec_inc["statistics"]["eligible_splits_count"] == 7  # 16 - 5 - 5 + 1
        assert "mann_whitney_u_diagnostic_pvalue" in rec_inc["statistics"]
        assert "exchangeable" in rec_inc["limitations"].lower()
        assert "autocorrelation" in rec_inc["limitations"].lower()
        assert any(r.rule_id == "RULE-ACT-006" and r.severity == "positive" for r in res_inc.rules)

        # 2. Step Decrease: 8 weeks at 18-22/wk -> 8 weeks at 1-2/wk
        dec_pattern = [20, 19, 21, 18, 22, 19, 20, 18, 2, 1, 2, 1, 2, 1, 2, 1]
        _seed_completed_weekly_commits(in_memory_db, 5009, ref_time, dec_pattern, "cp_dec")
        res_dec = calc.calculate(5009, in_memory_db, ref_time, snapshot_revision=1)
        cp_dec = res_dec.metrics["change_points"]
        assert cp_dec["status"] == "detected"
        rec_dec = cp_dec["detected_change_points"][0]
        assert rec_dec["direction"] == "step_decrease"
        assert rec_dec["statistics"]["permutation_pvalue"] == 0.001
        assert rec_dec["before_segment"]["median_weekly_commits"] >= 18.0
        assert rec_dec["after_segment"]["median_weekly_commits"] <= 2.0
        assert any(r.rule_id == "RULE-ACT-006" and r.severity == "warning" for r in res_dec.rules)

    def test_insufficient_samples_and_reference_time_future_exclusion(self, in_memory_db):
        """Short histories defer statistical claims cleanly; future commits beyond ref_time are excluded."""
        calc = DevelopmentActivityCalculator()
        tz_plus5 = timezone(timedelta(hours=5))
        # Frozen reference time in +05:00 -> converts to 2026-11-09 07:00 UTC (Monday)
        ref_time_tz = datetime(2026, 11, 9, 12, 0, 0, tzinfo=tz_plus5)
        ref_utc = ref_time_tz.astimezone(timezone.utc)

        # Only 3 completed weeks (insufficient for consistency, rolling 4w/12w, anomalies, change-points)
        _seed_completed_weekly_commits(in_memory_db, 5010, ref_utc, [4, 5, 4], "short_hist")
        # Add future commits after ref_time (must never enter completed weeks or anomaly stats)
        for i in range(40):
            in_memory_db.add(
                Commit(
                    repository_id=5010,
                    sha=f"sha_future_c22_{i}",
                    author_date=ref_utc + timedelta(days=7, hours=i),
                    author_login="c22_dev",
                )
            )
        in_memory_db.commit()

        res = calc.calculate(5010, in_memory_db, ref_time_tz, snapshot_revision=1)
        assert res.metrics["summary"]["total_commits"] == 13
        assert res.metrics["consistency"]["status"] == "insufficient"
        assert res.metrics["consistency"]["consistency_classification"] == "insufficient_history"
        assert res.metrics["rolling_statistics"]["status"] == "insufficient"
        assert res.metrics["rolling_statistics"]["latest"]["window_4w_complete"] is False
        assert res.metrics["rolling_statistics"]["latest"]["rolling_4w_mean"] is None
        assert res.metrics["anomalies"]["status"] == "insufficient_history"
        assert res.metrics["change_points"]["status"] == "insufficient_history"
        assert res.metrics["change_points"]["justified"] is False

    def test_highly_volatile_classification_and_queue_json_persistence(self, in_memory_db):
        """Verify highly_volatile regime detection and full C.2.2 JSON publication through AnalyticsJobQueue."""
        import json

        calc = DevelopmentActivityCalculator()
        queue = AnalyticsJobQueue()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # Highly volatile alternating bursts across 12 completed weeks
        volatile_pattern = [1, 2, 30, 1, 2, 35, 1, 1, 28, 2, 1, 25]
        _seed_completed_weekly_commits(in_memory_db, 5012, ref_time, volatile_pattern, "volatile")

        run, _ = queue.enqueue_analysis(repo_id=5012, db=in_memory_db)
        claimed_run, attempt = queue.claim_job(worker_id="c22_worker", db=in_memory_db)

        res = calc.calculate(5012, in_memory_db, ref_time, snapshot_revision=claimed_run.requested_source_revision)
        assert res.metrics["consistency"]["consistency_classification"] == "highly_volatile"
        assert res.metrics["consistency"]["coefficient_of_variation"] > 1.0
        assert any(r.rule_id == "RULE-ACT-004" and r.severity == "warning" for r in res.rules)

        # Ensure strict JSON serializability and full lifecycle persistence via AnalyticsJobQueue
        json.dumps(res.metrics)
        published = queue.publish_and_finalize(
            run_id=claimed_run.id,
            attempt_id=attempt.id,
            worker_id="c22_worker",
            expected_lease_version=claimed_run.lease_version,
            attempt_number=attempt.attempt_number,
            snapshot_revision=claimed_run.requested_source_revision,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=ref_time,
            metrics_payload={"development_activity": res.metrics},
            coverage_summary={"development_activity": res.coverage.model_dump()},
            confidence_summary={
                "score": round(1.0 - res.confidence_penalty, 3),
                "band": "high",
                "penalties": res.details.get("penalties", []),
            },
            rule_interpretations=[r.model_dump() for r in res.rules],
            db=in_memory_db,
        )
        assert published is True
        in_memory_db.refresh(claimed_run)
        assert claimed_run.status == "completed"
        assert claimed_run.metrics_payload["development_activity"]["consistency"]["consistency_classification"] == "highly_volatile"


class TestMaxSsbPermutationChangePointCorrection:
    """Tests for deterministic max-SSB permutation correction eliminating post-selection bias."""

    def test_stationary_series_and_post_selection_bias_suppression(self, in_memory_db):
        """Stationary sequences where unadjusted Mann-Whitney falls <=0.01 are rejected by p_perm > 0.01."""
        import random
        from scipy.stats import mannwhitneyu

        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # Concrete 26-week i.i.d. stationary vector where unadjusted MW yields p=0.00873<=0.01,
        # while full-search max-SSB permutation test gives p_perm > 0.01.
        post_selection_trap = [14, 12, 16, 13, 3, 16, 11, 10, 8, 3, 4, 10, 3, 9, 9, 5, 6, 9, 5, 6, 7, 7, 6, 7, 9, 17]
        best_split, _, p_perm, eligible = calc._run_max_ssb_permutation_test(post_selection_trap)
        mw_p = float(mannwhitneyu(post_selection_trap[:best_split], post_selection_trap[best_split:], alternative="two-sided").pvalue)
        assert mw_p <= 0.01
        assert p_perm > 0.01
        assert eligible == 17  # 26 - 5 - 5 + 1

        _seed_completed_weekly_commits(in_memory_db, 5020, ref_time, post_selection_trap, "post_sel_trap")
        res_trap = calc.calculate(5020, in_memory_db, ref_time, snapshot_revision=1)
        assert res_trap.metrics["change_points"]["status"] == "no_significant_shift"
        assert res_trap.metrics["change_points"]["detected_change_points"] == []

        # Also evaluate 20 seeded stationary sequences via full calculator
        rng = random.Random(202611)
        false_positives = 0
        for idx in range(20):
            stationary = [rng.randint(3, 17) for _ in range(20)]
            _seed_completed_weekly_commits(in_memory_db, 5100 + idx, ref_time, stationary, f"stat_{idx}")
            res = calc.calculate(5100 + idx, in_memory_db, ref_time, snapshot_revision=1)
            if res.metrics["change_points"]["status"] == "detected":
                false_positives += 1
        assert false_positives == 0

    def test_deterministic_reproducibility_and_minimum_resolution(self, in_memory_db):
        """Repeated runs produce bit-identical results; B=999 provides exact 0.001 minimum p_perm resolution."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # Bit-identical repeated runs
        pattern = [2, 3, 2, 3, 2, 3, 2, 3, 18, 20, 19, 22, 20, 18, 21, 19]
        _seed_completed_weekly_commits(in_memory_db, 5021, ref_time, pattern, "determinism")
        res1 = calc.calculate(5021, in_memory_db, ref_time, snapshot_revision=1)
        res2 = calc.calculate(5021, in_memory_db, ref_time, snapshot_revision=1)

        assert res1.metrics == res2.metrics
        assert res1.rules == res2.rules

        # Exact resolution checks: B=999 -> min p_perm = 1 / 1000 = 0.001; constant vector -> p_perm = 1.0
        _, _, p_min, _ = calc._run_max_ssb_permutation_test([2] * 8 + [20] * 8, min_segment=5, n_permutations=999)
        assert p_min == pytest.approx(0.001, abs=1e-9)

        _, _, p_const, _ = calc._run_max_ssb_permutation_test([5] * 12, min_segment=5, n_permutations=999)
        assert p_const == pytest.approx(1.0, abs=1e-9)

        # Single extreme outlier spike remains non-significant under max-SSB permutation
        _, _, p_spike, _ = calc._run_max_ssb_permutation_test(
            [4, 7, 5, 6, 4, 8, 5, 6, 4, 7, 35, 5, 6, 4, 7, 5], min_segment=5, n_permutations=999
        )
        assert p_spike > 0.01

    def test_minimum_history_and_minimum_segment_boundaries(self, in_memory_db):
        """Enforce min_weeks=12, min_commits=30, min_active=6, and min_segment=5 boundaries."""
        calc = DevelopmentActivityCalculator()
        ref_time = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)

        # 1. 11 completed weeks (< 12) even with large shift -> insufficient_history
        _seed_completed_weekly_commits(in_memory_db, 5022, ref_time, [2] * 5 + [20] * 6, "k11")
        assert calc.calculate(5022, in_memory_db, ref_time, snapshot_revision=1).metrics["change_points"]["status"] == "insufficient_history"

        # 2. 12 completed weeks with 29 commits (< 30) -> insufficient_history
        _seed_completed_weekly_commits(in_memory_db, 5023, ref_time, [1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 3], "c29")
        assert calc.calculate(5023, in_memory_db, ref_time, snapshot_revision=1).metrics["change_points"]["status"] == "insufficient_history"

        # 3. 12 completed weeks with >=30 commits across only 5 active weeks (< 6) -> insufficient_history
        _seed_completed_weekly_commits(in_memory_db, 5024, ref_time, [8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8], "a5")
        assert calc.calculate(5024, in_memory_db, ref_time, snapshot_revision=1).metrics["change_points"]["status"] == "insufficient_history"

        # 4. Surge strictly in trailing 3 weeks (< min_segment=5) -> excluded splits + permutation rejects
        _seed_completed_weekly_commits(in_memory_db, 5025, ref_time, [2] * 9 + [25] * 3, "seg3")
        res_seg3 = calc.calculate(5025, in_memory_db, ref_time, snapshot_revision=1)
        assert res_seg3.metrics["change_points"]["status"] == "no_significant_shift"

        # 5. Exact minimal valid shift boundary: K=12, segments 5 vs 7 -> detected with before=5, after=7
        _seed_completed_weekly_commits(in_memory_db, 5026, ref_time, [2] * 5 + [20] * 7, "k12_valid")
        res_valid = calc.calculate(5026, in_memory_db, ref_time, snapshot_revision=1)
        cp_valid = res_valid.metrics["change_points"]
        assert cp_valid["status"] == "detected"
        assert cp_valid["detected_change_points"][0]["before_segment"]["weeks"] == 5
        assert cp_valid["detected_change_points"][0]["after_segment"]["weeks"] == 7
        assert cp_valid["detected_change_points"][0]["statistics"]["eligible_splits_count"] == 3  # t in [5, 6, 7]
        assert cp_valid["detected_change_points"][0]["statistics"]["permutation_pvalue"] <= 0.01





