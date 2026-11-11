"""Unit and Integration Tests for Issue & Maintenance Analytics Engine (Phase C.5 / Module 4).

Verifies:
1. Empty & insufficient-data repositories (exact gates: >=10 flow, 5..19 limited / >=20 confident resolution, >=15 KM).
2. Monthly creation/closure flow, net & cumulative flow differentials, and flow ratio.
3. Backlog stock vs observed sample flow separation and combined_open_observed_to_reported_ratio_percentage.
4. Strict exclusion of Repository.last_synced_at from metrics_payload.
5. Resolution quantiles (P25/P50/P75/P90/IQR), un-truncated mean, exact 0.0h closures, negative-duration isolation,
   skewness (> 1.0), dynamic speed tiers, and non-reuse of stale pre-aggregated Issue columns.
6. Kaplan-Meier S(30d), S(60d), S(90d), and median survival time with ties & right-censored open issues.
7. Five continuous backlog-age bins (<7d, 7..30d, >30..90d, >90..365d, >365d), fractional boundaries,
   reference-time future closures, and non-UTC timezone normalization.
8. Canonical categories preserving 'unlabeled' strictly distinct from 'other' plus top raw IssueLabels.
9. Discussion intensity (primary Issue.comments_count, supplemental IssueComment rows) & Spearman zero-variance guards.
10. Live execution against all 4 benchmark repositories in data/opensource_lens.db (all 2,328 issues).
11. JSON serializability and AnalyticsJobQueue snapshot + atomic publication lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    CollectionRun,
    Issue,
    IssueComment,
    IssueLabel,
    PullRequest,
    Repository,
)
from backend.app.schemas.analytics_v2 import CURRENT_METHODOLOGY_VERSION
from backend.app.services.analytics import IssueIntelligenceCalculator
from backend.app.services.analytics_queue import AnalyticsJobQueue


@pytest.fixture
def in_memory_db():
    """Create isolated in-memory SQLite database session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionClass()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


class TestEmptyThresholdsAndBacklogStockC5:
    """Verifies empty/insufficient thresholds, monthly flow math, stock separation, and no last_synced_at."""

    def test_empty_repository_returns_safe_structure(self, in_memory_db):
        calc = IssueIntelligenceCalculator()
        assert calc.domain_name == "issue_intelligence"

        in_memory_db.add(
            Repository(
                id=9001,
                repo_name="iss/empty",
                owner="iss",
                name="empty",
                open_issues_count=0,
                last_synced_at=datetime(2026, 11, 11, 0, 0, tzinfo=timezone.utc),
                analytics_source_revision=1,
            )
        )
        in_memory_db.commit()

        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        res = calc.calculate(9001, in_memory_db, ref_time, snapshot_revision=1)

        assert res.domain == "issue_intelligence"
        assert res.coverage.status == "empty"
        assert res.coverage.sample_size == 0
        assert res.metrics["observed_flow"]["status"] == "empty"
        assert res.metrics["resolution_distribution"]["status"] == "empty"
        assert res.metrics["survival_km"]["status"] == "empty"
        assert res.metrics["backlog_aging"]["status"] == "empty"
        assert res.metrics["discussion_intensity"]["status"] == "empty"
        assert res.metrics["canonical_categories"]["status"] == "empty"
        assert any(r.rule_id == "RULE-ISS-EMPTY" for r in res.rules)

        serialized = json.dumps(res.metrics)
        assert "last_synced_at" not in serialized
        assert "metadata_synced_at" not in serialized

    def test_flow_threshold_10_km_threshold_15_and_backlog_stock_ratio(self, in_memory_db):
        """9 issues => flow insufficient (<10) & KM insufficient (<15); 10 issues => flow computed, KM still <15; 15 => KM computed."""
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(
            Repository(
                id=9002,
                repo_name="iss/flow",
                owner="iss",
                name="flow",
                open_issues_count=40,
                last_synced_at=datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc),
                analytics_source_revision=2,
            )
        )
        # Seed 2 open PRs + 1 closed PR to verify combined open ratio (open_issues + open_prs) / reported_open_issues_count
        in_memory_db.add(
            PullRequest(
                id=900201,
                repository_id=9002,
                pr_number=1,
                title="Open PR 1",
                state="open",
                created_at=ref_time - timedelta(days=5),
            )
        )
        in_memory_db.add(
            PullRequest(
                id=900202,
                repository_id=9002,
                pr_number=2,
                title="Open PR 2",
                state="open",
                created_at=ref_time - timedelta(days=3),
            )
        )
        in_memory_db.add(
            PullRequest(
                id=900203,
                repository_id=9002,
                pr_number=3,
                title="Merged PR 3",
                state="closed",
                created_at=ref_time - timedelta(days=10),
                closed_at=ref_time - timedelta(days=8),
                merged_at=ref_time - timedelta(days=8),
            )
        )

        # 9 issues across Sep..Nov (6 open, 3 closed): flow (<10) & KM (<15) both insufficient!
        for i in range(9):
            c_dt = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc) + timedelta(days=i)
            is_closed = i < 3
            in_memory_db.add(
                Issue(
                    id=900200 + i,
                    repository_id=9002,
                    repo_name="iss/flow",
                    issue_number=i + 1,
                    title=f"Issue {i + 1}",
                    state="closed" if is_closed else "open",
                    author="dev",
                    created_at=c_dt,
                    closed_at=datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc) if is_closed else None,
                    canonical_label="bug",
                )
            )
        in_memory_db.commit()

        res_9 = calc.calculate(9002, in_memory_db, ref_time, snapshot_revision=2)
        assert res_9.coverage.status == "degraded"  # 5 <= 9 < 10
        assert res_9.metrics["observed_flow"]["status"] == "insufficient"
        assert res_9.metrics["observed_flow"]["monthly_series"] is None
        assert res_9.metrics["resolution_distribution"]["status"] == "insufficient"  # 3 < 5
        assert res_9.metrics["survival_km"]["status"] == "insufficient"  # 9 < 15

        # Add 10th issue (created Nov 1, closed Nov 5 -> triggers continuous Sep, Oct, Nov flow with Oct zero-fill!)
        in_memory_db.add(
            Issue(
                id=900209,
                repository_id=9002,
                repo_name="iss/flow",
                issue_number=10,
                title="Issue 10",
                state="closed",
                author="dev",
                created_at=datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc),
                closed_at=datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc),
                canonical_label="feature",
            )
        )
        in_memory_db.commit()

        res_10 = calc.calculate(9002, in_memory_db, ref_time, snapshot_revision=2)
        flow = res_10.metrics["observed_flow"]
        stock = res_10.metrics["backlog_stock_summary"]

        assert flow["status"] == "sufficient"
        assert flow["total_created_observed"] == 10
        assert flow["total_closed_observed"] == 4
        assert flow["net_flow_differential"] == 6
        assert flow["flow_ratio"] == 0.4
        series = {row["month"]: row for row in flow["monthly_series"]}
        assert list(series.keys()) == ["2026-09", "2026-10", "2026-11"]
        assert series["2026-09"] == {"month": "2026-09", "created_count": 9, "closed_count": 0, "net_flow_differential": 9, "cumulative_flow_differential": 9}
        assert series["2026-10"] == {"month": "2026-10", "created_count": 0, "closed_count": 0, "net_flow_differential": 0, "cumulative_flow_differential": 9}
        assert series["2026-11"] == {"month": "2026-11", "created_count": 1, "closed_count": 4, "net_flow_differential": -3, "cumulative_flow_differential": 6}

        # Stock separation: 6 open pure issues + 2 open PRs = 8 out of reported 40 => 20.0%
        assert stock["github_metadata_snapshot"]["github_combined_open_issues_and_prs"] == 40
        assert stock["local_harvested_entities"]["locally_harvested_open_pure_issues"] == 6
        assert stock["local_harvested_entities"]["locally_harvested_closed_pure_issues"] == 4
        assert stock["local_harvested_entities"]["locally_harvested_open_prs"] == 2
        assert stock["local_harvested_entities"]["locally_harvested_closed_prs"] == 1
        assert stock["coverage_assessment"]["combined_open_observed_to_reported_ratio_percentage"] == 20.0
        assert stock["coverage_assessment"]["pure_issue_sample_coverage_percentage"] is None

        # Verify Repository.last_synced_at NEVER leaks into metrics_payload
        serialized = json.dumps(res_10.metrics)
        assert "last_synced_at" not in serialized
        assert "metadata_synced_at" not in serialized

        # Still only 10 issues -> KM stays insufficient (<15)! Add 5 more -> flips KM to sufficient!
        assert res_10.metrics["survival_km"]["status"] == "insufficient"
        for idx in range(5):
            in_memory_db.add(
                Issue(
                    id=900210 + idx,
                    repository_id=9002,
                    repo_name="iss/flow",
                    issue_number=11 + idx,
                    title=f"Issue {11 + idx}",
                    state="open",
                    author="dev",
                    created_at=datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc),
                )
            )
        in_memory_db.commit()

        res_15 = calc.calculate(9002, in_memory_db, ref_time, snapshot_revision=2)
        assert res_15.metrics["survival_km"]["status"] == "sufficient"


class TestResolutionSurvivalAgingAndRefTimeC5:
    """Verifies exact 0h closures, negative rejections, skewness, KM horizons, and 5 continuous age bins."""

    def test_resolution_quantiles_zero_hour_negative_skew_and_non_reuse(self, in_memory_db):
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=9003, repo_name="iss/res", owner="iss", name="res", open_issues_count=5, analytics_source_revision=1))

        # 6 valid closed issues with skewed durations [0.0, 2.0, 4.0, 8.0, 12.0, 2400.0]h
        # Plus poisoned resolution_time_hours=99999 and poisoned speed_tier="30+ Days" which MUST be ignored!
        durations_h = [0.0, 2.0, 4.0, 8.0, 12.0, 2400.0]
        base_dt = ref_time - timedelta(days=120)
        for idx, h in enumerate(durations_h):
            in_memory_db.add(
                Issue(
                    id=900300 + idx,
                    repository_id=9003,
                    repo_name="iss/res",
                    issue_number=idx + 1,
                    title=f"Res #{idx}",
                    state="closed",
                    author="dev",
                    created_at=base_dt,
                    closed_at=base_dt + timedelta(hours=h),
                    resolution_time_hours=99999.0,  # Poisoned!
                    speed_tier="30+ Days",          # Poisoned!
                )
            )

        # 1 corrupt issue with negative clock skew (closed_at < created_at) -> excluded!
        in_memory_db.add(
            Issue(
                id=900399,
                repository_id=9003,
                repo_name="iss/res",
                issue_number=99,
                title="Negative Clock Skew",
                state="closed",
                author="dev",
                created_at=ref_time - timedelta(days=5),
                closed_at=ref_time - timedelta(days=6),
                resolution_time_hours=0.0,  # Collector-clamped value must be ignored!
            )
        )
        in_memory_db.commit()

        res = calc.calculate(9003, in_memory_db, ref_time, snapshot_revision=1)
        rdist = res.metrics["resolution_distribution"]

        assert rdist["status"] == "limited"  # 5 <= 6 < 20
        assert rdist["valid_resolved_count"] == 6
        assert rdist["zero_hour_closures_count"] == 1  # Exact 0.0h closure preserved!
        assert rdist["negative_duration_count"] == 1
        assert rdist["invalid_records_count"] == 1
        # Un-truncated mean = (0 + 2 + 4 + 8 + 12 + 2400) / 6 = 404.33
        assert rdist["mean_hours"] == 404.33
        assert rdist["p50_hours"] == 6.0  # Median of [0, 2, 4, 8, 12, 2400]
        assert rdist["skewness"] > 1.0
        assert rdist["high_skew_warning"] is True
        # Speed tiers dynamically derived: 5 in "< 1 Day" (<24h), 1 in "30+ Days" (2400h = 100d)
        assert rdist["speed_tier_distribution"] == {"< 1 Day": 5, "1-7 Days": 0, "7-30 Days": 0, "30+ Days": 1}
        assert any(r.rule_id == "RULE-ISS-003" for r in res.rules)
        assert any(r.rule_id == "RULE-ISS-005" for r in res.rules)

    def test_five_continuous_aging_bins_boundaries_and_future_closures(self, in_memory_db):
        """Verifies continuous 5-bin boundaries, fractional days (30.5d, 90.5d), future closures open at ref_time, and IST offset."""
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=9004, repo_name="iss/age", owner="iss", name="age", open_issues_count=20, analytics_source_revision=1))

        # Test exact & fractional boundaries across all 5 bins:
        # 6.99d (<7 -> fresh), 7.0d & 30.0d (7<=age<=30 -> active),
        # 30.5d & 90.0d (30<age<=90 -> aging), 90.5d & 365.0d (90<age<=365 -> stale), 365.5d (>365 -> dormant)
        boundary_ages_days = [
            (6.99, "open", None),
            (7.0, "open", None),
            (30.0, "open", None),
            (30.5, "open", None),
            (90.0, "open", None),
            (90.5, "closed", ref_time + timedelta(days=5)),   # Closed AFTER ref_time => MUST be open/stale at ref_time!
            (365.0, "open", None),
            (365.5, "open", None),
        ]
        for idx, (age_d, state, closed_dt) in enumerate(boundary_ages_days):
            in_memory_db.add(
                Issue(
                    id=900400 + idx,
                    repository_id=9004,
                    repo_name="iss/age",
                    issue_number=idx + 1,
                    title=f"Age #{idx}",
                    state=state,
                    author="dev",
                    created_at=ref_time - timedelta(days=age_d),
                    closed_at=closed_dt,
                )
            )
        # Future created issue (> ref_time) -> ignored completely
        in_memory_db.add(
            Issue(
                id=900499,
                repository_id=9004,
                repo_name="iss/age",
                issue_number=99,
                title="Future issue",
                state="open",
                author="dev",
                created_at=ref_time + timedelta(hours=3),
            )
        )
        in_memory_db.commit()

        res = calc.calculate(9004, in_memory_db, ref_time, snapshot_revision=1)
        aging = res.metrics["backlog_aging"]

        assert aging["open_issues_count"] == 8  # Includes future-closed issue!
        assert res.metrics["observed_flow"]["future_issues_excluded_count"] == 1
        assert aging["bins"]["fresh"]["count"] == 1      # 6.99
        assert aging["bins"]["active"]["count"] == 2     # 7.0, 30.0
        assert aging["bins"]["aging"]["count"] == 2      # 30.5, 90.0
        assert aging["bins"]["stale"]["count"] == 2      # 90.5, 365.0
        assert aging["bins"]["dormant"]["count"] == 1    # 365.5
        assert aging["stale_gt_90d_count"] == 3          # Stale (2) + Dormant (1)
        assert aging["stale_fraction"] == round(3 / 8, 4)

        # Equivalent non-UTC offset instant matches UTC identically
        res_ist = calc.calculate(
            9004,
            in_memory_db,
            ref_time.astimezone(timezone(timedelta(hours=5, minutes=30))),
            snapshot_revision=1,
        )
        assert res_ist.metrics == res.metrics

    def test_issue_kaplan_meier_survival_30_60_90_with_ties_and_censoring(self, in_memory_db):
        """15 issues: 5 closed at d=20, 5 open (censored) at d=40, 5 closed at d=75 (ref_time anchor)."""
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=9005, repo_name="iss/km", owner="iss", name="km", open_issues_count=5, analytics_source_revision=1))

        # Group 1 (5 issues): created 100d ago, closed at duration 20d
        for i in range(5):
            c_dt = ref_time - timedelta(days=100)
            in_memory_db.add(
                Issue(
                    id=900500 + i,
                    repository_id=9005,
                    repo_name="iss/km",
                    issue_number=1 + i,
                    title=f"C20 #{i}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=20),
                )
            )
        # Group 2 (5 issues): open, created 40d ago (right-censored at 40d!)
        for i in range(5):
            in_memory_db.add(
                Issue(
                    id=900510 + i,
                    repository_id=9005,
                    repo_name="iss/km",
                    issue_number=10 + i,
                    title=f"Open40 #{i}",
                    state="open",
                    author="dev",
                    created_at=ref_time - timedelta(days=40),
                )
            )
        # Group 3 (5 issues): created 100d ago, closed at duration 75d
        for i in range(5):
            c_dt = ref_time - timedelta(days=100)
            in_memory_db.add(
                Issue(
                    id=900520 + i,
                    repository_id=9005,
                    repo_name="iss/km",
                    issue_number=20 + i,
                    title=f"C75 #{i}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=75),
                )
            )
        in_memory_db.commit()

        res = calc.calculate(9005, in_memory_db, ref_time, snapshot_revision=1)
        km = res.metrics["survival_km"]

        # Exact KM math:
        # t=20: n=15, d=5 => S(20)=10/15=0.6667 (S(30d)=0.6667)
        # t=40: c=5 (risk drops 10 -> 5), S(60d)=0.6667
        # t=75: n=5, d=5 => S(75)=0.6667 * (1 - 5/5) = 0.0 (S(90d)=0.0, median=75.0d)
        assert km["status"] == "sufficient"
        assert km["s30d"] == round(10 / 15, 4)
        assert km["s60d"] == round(10 / 15, 4)
        assert km["s90d"] == 0.0
        assert km["median_survival_days"] == 75.0
        assert km["s30d_within_followup"] is True
        assert km["s60d_within_followup"] is True
        assert km["s90d_within_followup"] is True  # terminal surv == 0.0 => known exact 0 at 90d

        # Verify simultaneous tied event + right-censoring at exact day 30.0 (n_risk=15: 6 events at 30d + 3 open at 30d + 6 open at 60d)
        # Standard event-before-censoring tie convention: S(30d) = 1 - 6/15 = 0.6000 (NOT 1 - 6/12 = 0.50!)
        in_memory_db.add(Repository(id=90051, repo_name="iss/km_tie", owner="iss", name="km_tie", open_issues_count=9, analytics_source_revision=1))
        for j in range(6):
            c_dt = ref_time - timedelta(days=50)
            in_memory_db.add(
                Issue(
                    id=905100 + j,
                    repository_id=90051,
                    repo_name="iss/km_tie",
                    issue_number=1 + j,
                    title=f"TieClose30 #{j}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=30),
                )
            )
        for j in range(3):
            in_memory_db.add(
                Issue(
                    id=905110 + j,
                    repository_id=90051,
                    repo_name="iss/km_tie",
                    issue_number=20 + j,
                    title=f"TieOpen30 #{j}",
                    state="open",
                    author="dev",
                    created_at=ref_time - timedelta(days=30),
                )
            )
        for j in range(6):
            in_memory_db.add(
                Issue(
                    id=905120 + j,
                    repository_id=90051,
                    repo_name="iss/km_tie",
                    issue_number=30 + j,
                    title=f"Open60 #{j}",
                    state="open",
                    author="dev",
                    created_at=ref_time - timedelta(days=60),
                )
            )
        in_memory_db.commit()

        res_tie_1 = calc.calculate(90051, in_memory_db, ref_time, snapshot_revision=1)
        res_tie_2 = calc.calculate(90051, in_memory_db, ref_time, snapshot_revision=1)
        assert res_tie_1.metrics["survival_km"]["s30d"] == 0.6
        assert res_tie_1.metrics["survival_km"]["s60d"] == 0.6
        assert res_tie_1.metrics["survival_km"]["s90d"] == 0.6
        assert res_tie_1.metrics == res_tie_2.metrics


class TestCategoriesDiscussionAndSpearmanGuardsC5:
    """Verifies canonical vs 'unlabeled' distinction, discussion comments_count, and Spearman zero-variance guards."""

    def test_unlabeled_distinct_from_other_and_spearman_zero_variance_guard(self, in_memory_db):
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=9006, repo_name="iss/cat", owner="iss", name="cat", open_issues_count=10, analytics_source_revision=1))

        # 10 closed issues ALL having IDENTICAL comments_count=3 (zero variance in comments -> Spearman MUST guard!)
        # 4 'bug', 3 'other' (with attached non-canonical IssueLabel!), 3 unlabeled (2 via canonical_label='unlabeled', 1 via is_unlabeled=True + canonical_label='other' + 0 labels)
        c_dt = ref_time - timedelta(days=20)
        for i in range(4):
            in_memory_db.add(
                Issue(
                    id=900600 + i,
                    repository_id=9006,
                    repo_name="iss/cat",
                    issue_number=1 + i,
                    title=f"Bug #{i}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(hours=10 + i),
                    canonical_label="bug",
                    is_unlabeled=False,
                    comments_count=3,
                )
            )
            in_memory_db.add(IssueLabel(issue_id=900600 + i, label_name="bug", is_canonical=True, canonical_category="bug"))

        for i in range(3):
            in_memory_db.add(
                Issue(
                    id=900610 + i,
                    repository_id=9006,
                    repo_name="iss/cat",
                    issue_number=10 + i,
                    title=f"Other #{i}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(hours=20 + i),
                    canonical_label="other",
                    is_unlabeled=False,
                    comments_count=3,
                )
            )
            in_memory_db.add(IssueLabel(issue_id=900610 + i, label_name="triage-needed", is_canonical=False, canonical_category="other"))

        # Unlabeled rows (2 explicit canonical_label='unlabeled', 1 collector-style is_unlabeled=True & canonical_label='other' & no labels)
        for i in range(3):
            in_memory_db.add(
                Issue(
                    id=900620 + i,
                    repository_id=9006,
                    repo_name="iss/cat",
                    issue_number=20 + i,
                    title=f"Unlabeled #{i}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(hours=30 + i),
                    canonical_label="other" if i == 2 else "unlabeled",
                    is_unlabeled=True,
                    comments_count=3,
                )
            )
        in_memory_db.commit()

        res = calc.calculate(9006, in_memory_db, ref_time, snapshot_revision=1)
        cats = res.metrics["canonical_categories"]
        disc = res.metrics["discussion_intensity"]

        # Strict 'unlabeled' vs 'other' separation
        assert cats["is_unlabeled_count"] == 3
        assert cats["labeled_issues_count"] == 7
        assert cats["categories"]["bug"]["total_count"] == 4
        assert cats["categories"]["other"]["total_count"] == 3
        assert cats["categories"]["unlabeled"]["total_count"] == 3
        assert any(r["label_name"] == "triage-needed" and r["count"] == 3 for r in cats["top_raw_labels"])
        assert any(r.rule_id == "RULE-ISS-004" for r in res.rules)  # 30% unlabeled >= 25%

        # Discussion volume comes from Issue.comments_count even with 0 IssueComment rows!
        assert disc["mean_comments_per_issue"] == 3.0
        assert disc["supplemental_issue_comments"]["detail_available"] is False
        assert disc["spearman_comments_vs_resolution"]["status"] == "undefined_zero_variance"
        assert disc["spearman_comments_vs_resolution"]["spearman_r"] is None

    def test_supplemental_issue_comments_and_finite_spearman(self, in_memory_db):
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=9007, repo_name="iss/corr", owner="iss", name="corr", open_issues_count=5, analytics_source_revision=1))

        # Monotonically increasing comment_count vs closure hours -> Spearman r_s = +1.0
        c_dt = ref_time - timedelta(days=20)
        for i in range(6):
            iss_id = 900700 + i
            in_memory_db.add(
                Issue(
                    id=iss_id,
                    repository_id=9007,
                    repo_name="iss/corr",
                    issue_number=i + 1,
                    title=f"Corr #{i}",
                    state="closed",
                    author="dev",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(hours=10 * (i + 1)),
                    comments_count=i + 1,
                )
            )
            in_memory_db.add(
                IssueComment(
                    repository_id=9007,
                    issue_id=iss_id,
                    issue_number=i + 1,
                    comment_id=9700 + i,
                    author=f"commenter_{i % 3}",
                    author_id=500 + (i % 3),
                    created_at=c_dt + timedelta(hours=2),
                )
            )
        in_memory_db.commit()

        res = calc.calculate(9007, in_memory_db, ref_time, snapshot_revision=1)
        disc = res.metrics["discussion_intensity"]
        assert disc["supplemental_issue_comments"]["detail_available"] is True
        assert disc["supplemental_issue_comments"]["distinct_commenters"] == 3
        sp = disc["spearman_comments_vs_resolution"]
        assert sp["status"] == "computed"
        assert sp["spearman_r"] == 1.0
        assert sp["p_value"] < 0.05
        assert "does not imply causal" in sp["note"].lower()


class TestBenchmarkAndQueueIntegrationC5:
    """Verifies real computation on all 2,328 benchmark issues and AnalyticsJobQueue publication."""

    def test_benchmark_repositories_full_issue_analytics(self):
        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip(f"Database {db_path} not found.")

        engine = create_engine(f"sqlite:///{db_path}")
        calc = IssueIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)

        total_issues_analyzed = 0
        with Session(engine) as session:
            repos = session.scalars(select(Repository)).all()
            assert len(repos) == 4
            for repo in repos:
                res = calc.calculate(repo.id, session, ref_time, repo.analytics_source_revision)
                assert res.domain == "issue_intelligence"
                assert res.coverage.status == "sufficient"
                total_issues_analyzed += res.coverage.sample_size

                flow = res.metrics["observed_flow"]
                stock = res.metrics["backlog_stock_summary"]
                rdist = res.metrics["resolution_distribution"]
                km = res.metrics["survival_km"]
                aging = res.metrics["backlog_aging"]
                disc = res.metrics["discussion_intensity"]
                cats = res.metrics["canonical_categories"]

                assert flow["status"] == "sufficient"
                assert stock["coverage_assessment"]["pure_issue_sample_coverage_percentage"] is None
                assert stock["coverage_assessment"]["combined_open_observed_to_reported_ratio_percentage"] is not None
                assert rdist["status"] == "sufficient"
                assert rdist["p50_hours"] >= 0.0
                assert km["status"] == "sufficient"
                assert 1.0 >= km["s30d"] >= km["s60d"] >= km["s90d"] >= 0.0
                assert aging["status"] == "sufficient"
                assert aging["stale_fraction"] >= 0.0
                assert disc["status"] == "sufficient"
                assert disc["mean_comments_per_issue"] > 0.0
                assert cats["status"] == "sufficient"
                assert "bug" in cats["categories"]

                dumped = json.dumps(res.metrics)
                assert "last_synced_at" not in dumped

        assert total_issues_analyzed == 2328

    def test_issue_calculator_published_via_queue(self, in_memory_db):
        calc = IssueIntelligenceCalculator()
        queue = AnalyticsJobQueue()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(
            Repository(
                id=9090,
                repo_name="queue/c5",
                owner="queue",
                name="c5",
                open_issues_count=50,
                last_synced_at=ref_time,
                analytics_source_revision=2,
                is_active=True,
            )
        )
        in_memory_db.add(
            CollectionRun(
                repository_id=9090,
                mode="standard_diagnostic",
                status="completed",
                is_partial=False,
                coverage_summary={"issues": "completed", "issue_comments": "completed"},
            )
        )
        for i in range(20):
            c_dt = ref_time - timedelta(days=50 - i)
            is_closed = i < 12
            in_memory_db.add(
                Issue(
                    id=909000 + i,
                    repository_id=9090,
                    repo_name="queue/c5",
                    issue_number=i + 1,
                    title=f"QIss #{i + 1}",
                    state="closed" if is_closed else "open",
                    author="qdev",
                    created_at=c_dt,
                    closed_at=(c_dt + timedelta(days=5)) if is_closed else None,
                    canonical_label="bug" if i % 2 == 0 else "feature",
                    comments_count=i % 4,
                )
            )
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=9090, db=in_memory_db)
        claimed_run, attempt = queue.claim_job(worker_id="c5_worker", db=in_memory_db)

        with queue.open_analysis_snapshot(db=in_memory_db, repository_id=9090) as (snap_rev, snap_ref_time):
            res = calc.calculate(9090, in_memory_db, snap_ref_time, snapshot_revision=snap_rev)
            serialized = json.dumps(res.metrics)
            assert "last_synced_at" not in serialized

        published = queue.publish_and_finalize(
            run_id=claimed_run.id,
            attempt_id=attempt.id,
            worker_id="c5_worker",
            expected_lease_version=claimed_run.lease_version,
            attempt_number=attempt.attempt_number,
            snapshot_revision=snap_rev,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=snap_ref_time,
            metrics_payload={"issue_intelligence": res.metrics},
            coverage_summary={"issue_intelligence": res.coverage.model_dump()},
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
        payload = claimed_run.metrics_payload["issue_intelligence"]
        assert payload["survival_km"]["status"] == "sufficient"
        assert payload["observed_flow"]["total_created_observed"] == 20
        assert set(claimed_run.coverage_summary["issue_intelligence"]["details"]["subdomains"].keys()) == {
            "observed_flow",
            "backlog_stock",
            "resolution_distribution",
            "survival_km",
            "backlog_aging",
            "discussion_intensity",
            "canonical_categories",
        }
