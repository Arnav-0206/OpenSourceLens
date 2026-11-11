"""Unit and Integration Tests for Pull Request Lifecycle & Turnaround Engine (Phase C.4 / Module 3).

Verifies:
1. Empty, sparse, and exact submetric thresholds (TTFResp 3-tier vs TTFR_formal >= 10,
   TTFR_inline >= 5, merge_latency >= 5, Kaplan-Meier >= 15 without invented intermediate bands).
2. Separated responsiveness channels (TTFResp, TTFR_formal, TTFR_inline) across reviews, inline
   diff comments, and PR conversation IssueComments; ignoring stale pre-aggregated PR fields.
3. Strict exclusion of PR-author self-interactions, bots ([bot] & Contributor.is_bot), and unattributed noise.
4. Collection-mode & stream-coverage awareness (fast_snapshot inline skipped vs uncollected reviews vs genuine 0).
5. Frozen analysis_reference_time anchoring, future closure handling, instant 0.0h merges, negative duration exclusion.
6. Deterministic Kaplan-Meier survival curve math (ties, censoring, S(7d), S(30d), median, no S(14d)).
7. PR size vs turnaround bins strictly distinguishing NULL diff metrics from genuine zero-line changes.
8. Review bottleneck diagnostics (latency share, human reviewer load, unreviewed open PR age).
9. Safe offline benchmark DB execution (0 PRs) and AnalyticsJobQueue atomic publication lifecycle.
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
    Contributor,
    Issue,
    IssueComment,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
    Repository,
)
from backend.app.schemas.analytics_v2 import CURRENT_METHODOLOGY_VERSION
from backend.app.services.analytics import PullRequestIntelligenceCalculator
from backend.app.services.analytics.pull_requests import compute_kaplan_meier_curve
from backend.app.services.analytics_queue import AnalyticsJobQueue


@pytest.fixture
def in_memory_db():
    """Create isolated in-memory SQLite session for each test."""
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


class TestEmptyThresholdsAndChannels:
    """Verifies empty repos, channel separation, and exact sample thresholds without invented intermediate bands."""

    def test_empty_repository_returns_safe_empty_structure(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        assert calc.domain_name == "pull_request_intelligence"

        in_memory_db.add(Repository(id=8001, repo_name="pr/empty", owner="pr", name="empty", analytics_source_revision=1))
        in_memory_db.commit()

        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        res = calc.calculate(8001, in_memory_db, ref_time, snapshot_revision=1)

        assert res.domain == "pull_request_intelligence"
        assert res.coverage.status == "empty"
        assert res.coverage.sample_size == 0
        assert res.metrics["outcome_distribution"]["status"] == "empty"
        assert res.metrics["responsiveness"]["ttf_resp"]["status"] == "empty"
        assert res.metrics["responsiveness"]["ttfr_formal"]["status"] == "empty"
        assert res.metrics["responsiveness"]["ttfr_inline"]["status"] == "empty"
        assert res.metrics["merge_latency"]["status"] == "empty"
        assert res.metrics["survival_km"]["status"] == "empty"
        assert res.metrics["size_vs_turnaround"]["status"] == "empty"
        assert res.metrics["review_bottlenecks"]["status"] == "empty"
        assert any(r.rule_id == "RULE-PR-EMPTY" for r in res.rules)
        json.dumps(res.metrics)

    def test_exact_channel_thresholds_without_invented_intermediate_tiers(self, in_memory_db):
        """TTFResp uses <3 / 3..9 / >=10; TTFR_formal (>=10), TTFR_inline (>=5), merge (>=5) stay insufficient below cutoff."""
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=8002, repo_name="pr/thresholds", owner="pr", name="thresholds", analytics_source_revision=1))

        # 9 PRs: 9 formal reviews, 4 inline comments, 4 merged -> TTFResp='limited' (3..9), all others 'insufficient'
        for idx in range(9):
            c_dt = ref_time - timedelta(days=20 - idx)
            pr = PullRequest(
                id=80020 + idx,
                repository_id=8002,
                pr_number=idx + 1,
                title=f"PR #{idx + 1}",
                state="closed" if idx < 4 else "open",
                author="alice",
                author_id=101,
                created_at=c_dt,
                closed_at=(c_dt + timedelta(hours=10)) if idx < 4 else None,
                merged_at=(c_dt + timedelta(hours=10)) if idx < 4 else None,
            )
            in_memory_db.add(pr)
            in_memory_db.add(
                PullRequestReview(
                    pull_request_id=80020 + idx,
                    review_id=9000 + idx,
                    author="bob",
                    author_id=102,
                    state="APPROVED",
                    submitted_at=c_dt + timedelta(hours=3),
                )
            )
            if idx < 4:
                in_memory_db.add(
                    PullRequestReviewComment(
                        pull_request_id=80020 + idx,
                        comment_id=9100 + idx,
                        author="carol",
                        author_id=103,
                        created_at=c_dt + timedelta(hours=1),
                    )
                )
        in_memory_db.commit()

        res = calc.calculate(8002, in_memory_db, ref_time, snapshot_revision=1)
        resp = res.metrics["responsiveness"]

        assert resp["ttf_resp"]["status"] == "limited"
        assert resp["ttf_resp"]["sample_count"] == 9
        assert resp["ttf_resp"]["p50_hours"] == 3.0  # 5 of 9 at 3h, 4 of 9 at 1h -> median 3.0h

        assert resp["ttfr_formal"]["status"] == "insufficient"
        assert resp["ttfr_formal"]["sample_count"] == 9
        assert resp["ttfr_formal"]["p50_hours"] is None

        assert resp["ttfr_inline"]["status"] == "insufficient"
        assert resp["ttfr_inline"]["sample_count"] == 4
        assert resp["ttfr_inline"]["p50_hours"] is None

        assert res.metrics["merge_latency"]["status"] == "insufficient"
        assert res.metrics["merge_latency"]["valid_merged_count"] == 4
        assert res.metrics["merge_latency"]["p50_hours"] is None

        # Add 10th PR with formal review, inline comment, and merge -> flips TTFR_formal(10), TTFR_inline(5), merge(5), TTFResp(10) to sufficient
        c_dt10 = ref_time - timedelta(days=5)
        in_memory_db.add(
            PullRequest(
                id=80029,
                repository_id=8002,
                pr_number=10,
                title="PR #10",
                state="closed",
                author="alice",
                author_id=101,
                created_at=c_dt10,
                closed_at=c_dt10 + timedelta(hours=20),
                merged_at=c_dt10 + timedelta(hours=20),
            )
        )
        in_memory_db.add(
            PullRequestReview(
                pull_request_id=80029,
                review_id=9099,
                author="bob",
                author_id=102,
                state="CHANGES_REQUESTED",
                submitted_at=c_dt10 + timedelta(hours=6),
            )
        )
        in_memory_db.add(
            PullRequestReviewComment(
                pull_request_id=80029,
                comment_id=9199,
                author="carol",
                author_id=103,
                created_at=c_dt10 + timedelta(hours=2),
            )
        )
        in_memory_db.commit()

        res2 = calc.calculate(8002, in_memory_db, ref_time, snapshot_revision=1)
        resp2 = res2.metrics["responsiveness"]
        assert resp2["ttf_resp"]["status"] == "sufficient"
        assert resp2["ttfr_formal"]["status"] == "sufficient"
        assert resp2["ttfr_inline"]["status"] == "sufficient"
        assert res2.metrics["merge_latency"]["status"] == "sufficient"
        assert res2.metrics["merge_latency"]["p50_hours"] == 10.0


class TestSelfBotAndCollectionModeGuards:
    """Verifies self-interaction/bot filtering, non-reuse of stale PR fields, and stream-skip guards."""

    def test_self_interaction_bot_exclusion_and_stale_column_ignored(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=8003, repo_name="pr/hygiene", owner="pr", name="hygiene", analytics_source_revision=1))
        in_memory_db.add(Contributor(repository_id=8003, login="ci-helper", github_id=9901, is_bot=True))

        # Build 10 PRs with poisoned time_to_first_review_hours = 0.01 (must be ignored!)
        for idx in range(10):
            c_dt = ref_time - timedelta(days=30 - idx)
            pr_id = 80030 + idx
            in_memory_db.add(
                PullRequest(
                    id=pr_id,
                    repository_id=8003,
                    pr_number=idx + 1,
                    title=f"PR #{idx + 1}",
                    state="open",
                    author="alice",
                    author_id=201,
                    created_at=c_dt,
                    time_to_first_review_hours=0.01,  # Poisoned pre-aggregated column!
                )
            )
            # Self-review by author at +0.5h (via matching ID or login) -> MUST BE EXCLUDED
            in_memory_db.add(
                PullRequestReview(
                    pull_request_id=pr_id,
                    review_id=9200 + idx,
                    author="alice",
                    author_id=None if idx % 2 == 0 else 201,
                    state="COMMENTED",
                    submitted_at=c_dt + timedelta(minutes=30),
                )
            )
            # Bot inline comment at +0.1h & bot issue comment via Contributor.is_bot -> MUST BE EXCLUDED
            in_memory_db.add(
                PullRequestReviewComment(
                    pull_request_id=pr_id,
                    comment_id=9300 + idx,
                    author="codecov[bot]",
                    author_id=9902,
                    created_at=c_dt + timedelta(minutes=6),
                )
            )
            in_memory_db.add(
                IssueComment(
                    repository_id=8003,
                    pull_request_id=pr_id,
                    comment_id=9400 + idx,
                    issue_number=idx + 1,
                    author="ci-helper",
                    author_id=9901,
                    created_at=c_dt + timedelta(minutes=12),
                )
            )
            # Genuine external human interactions:
            # First 5 PRs have IssueComment at +2h, inline at +4h, formal review at +6h
            # Next 5 PRs have inline at +5h, formal review at +8h (and a DISMISSED review at +1h -> counts for TTFResp, not TTFR_formal)
            if idx < 5:
                in_memory_db.add(
                    IssueComment(
                        repository_id=8003,
                        pull_request_id=pr_id,
                        comment_id=9500 + idx,
                        issue_number=idx + 1,
                        author="dave",
                        author_id=301,
                        created_at=c_dt + timedelta(hours=2),
                    )
                )
                in_memory_db.add(
                    PullRequestReviewComment(
                        pull_request_id=pr_id,
                        comment_id=9600 + idx,
                        author="eve",
                        author_id=302,
                        created_at=c_dt + timedelta(hours=4),
                    )
                )
                in_memory_db.add(
                    PullRequestReview(
                        pull_request_id=pr_id,
                        review_id=9700 + idx,
                        author="eve",
                        author_id=302,
                        state="APPROVED",
                        submitted_at=c_dt + timedelta(hours=6),
                    )
                )
            else:
                in_memory_db.add(
                    PullRequestReview(
                        pull_request_id=pr_id,
                        review_id=9800 + idx,
                        author="frank",
                        author_id=303,
                        state="DISMISSED",
                        submitted_at=c_dt + timedelta(hours=1),
                    )
                )
                in_memory_db.add(
                    PullRequestReviewComment(
                        pull_request_id=pr_id,
                        comment_id=9650 + idx,
                        author="eve",
                        author_id=302,
                        created_at=c_dt + timedelta(hours=5),
                    )
                )
                in_memory_db.add(
                    PullRequestReview(
                        pull_request_id=pr_id,
                        review_id=9750 + idx,
                        author="frank",
                        author_id=303,
                        state="APPROVED",
                        submitted_at=c_dt + timedelta(hours=8),
                    )
                )
            if idx == 0:
                # Future interaction > ref_time & negative interaction < created_at -> both excluded cleanly!
                in_memory_db.add(
                    PullRequestReview(
                        pull_request_id=pr_id,
                        review_id=9891,
                        author="grace",
                        author_id=304,
                        state="APPROVED",
                        submitted_at=ref_time + timedelta(hours=2),
                    )
                )
                in_memory_db.add(
                    PullRequestReviewComment(
                        pull_request_id=pr_id,
                        comment_id=9892,
                        author="grace",
                        author_id=304,
                        created_at=c_dt - timedelta(hours=1),
                    )
                )
        in_memory_db.commit()

        res = calc.calculate(8003, in_memory_db, ref_time, snapshot_revision=1)
        resp = res.metrics["responsiveness"]

        # TTFResp picks min(IssueComment=2h for 0..4, DISMISSED=1h for 5..9) -> sorted: [1]*5 + [2]*5 -> P50 = 1.5h
        assert resp["ttf_resp"]["status"] == "sufficient"
        assert resp["ttf_resp"]["p50_hours"] == 1.5
        # TTFR_formal ignores self-reviews, bots, and DISMISSED -> [6]*5 + [8]*5 -> P50 = 7.0h
        assert resp["ttfr_formal"]["status"] == "sufficient"
        assert resp["ttfr_formal"]["p50_hours"] == 7.0
        # TTFR_inline ignores bot codecov[bot] -> [4]*5 + [5]*5 -> P50 = 4.5h
        assert resp["ttfr_inline"]["status"] == "sufficient"
        assert resp["ttfr_inline"]["p50_hours"] == 4.5

        hygiene = resp["interaction_hygiene"]
        assert hygiene["self_interactions_excluded_count"] == 10
        assert hygiene["bot_interactions_excluded_count"] == 20
        assert hygiene["future_interactions_excluded_count"] == 1
        assert hygiene["negative_interaction_durations_excluded_count"] == 1

        # Maintainer review load excludes self & bots, includes human formal reviews (eve=5, frank=5)
        load = res.metrics["review_bottlenecks"]["maintainer_review_load"]
        assert load["total_human_formal_reviews"] == 10
        assert load["active_human_reviewers_count"] == 2

    def test_collection_mode_skipped_vs_uncollected_vs_empty_reviews(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=8004, repo_name="pr/fast", owner="pr", name="fast", analytics_source_revision=1))
        in_memory_db.add(
            CollectionRun(
                repository_id=8004,
                mode="fast_snapshot",
                status="completed",
                is_partial=False,
                coverage_summary={
                    "pull_requests": "completed",
                    "pr_reviews": "skipped",
                    "pr_inline_comments": "skipped",
                    "issue_comments": "completed",
                },
            )
        )
        for idx in range(4):
            in_memory_db.add(
                PullRequest(
                    id=80040 + idx,
                    repository_id=8004,
                    pr_number=idx + 1,
                    title=f"PR #{idx + 1}",
                    state="open",
                    author="alice",
                    author_id=101,
                    created_at=ref_time - timedelta(days=5),
                )
            )
        in_memory_db.commit()

        res = calc.calculate(8004, in_memory_db, ref_time, snapshot_revision=1)
        resp = res.metrics["responsiveness"]
        assert resp["ttfr_inline"]["status"] == "skipped_by_collection_mode"
        assert resp["ttfr_inline"]["p50_hours"] is None
        assert resp["ttfr_formal"]["status"] == "reviews_not_collected"
        assert resp["ttfr_formal"]["p50_hours"] is None
        assert any(r.rule_id == "RULE-PR-005" for r in res.rules)


class TestTemporalOutcomesInstantMergesAndNegativeDurations:
    """Verifies ref_time anchoring, future closure handling, 0h merges, negative durations, and outcomes."""

    def test_ref_time_anchoring_instant_merges_and_negative_durations(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=8005, repo_name="pr/time", owner="pr", name="time", analytics_source_revision=1))

        # 5 valid merged PRs (including one exact 0.0h instant merge!)
        merge_hours = [0.0, 4.0, 8.0, 12.0, 20.0]
        for idx, h in enumerate(merge_hours):
            c_dt = ref_time - timedelta(days=10)
            in_memory_db.add(
                PullRequest(
                    id=80050 + idx,
                    repository_id=8005,
                    pr_number=idx + 1,
                    title=f"Merged #{idx + 1}",
                    state="closed",
                    author="alice",
                    author_id=101,
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(hours=h),
                    merged_at=c_dt + timedelta(hours=h),
                )
            )

        # 1 corrupt merged PR with negative merge duration (merged_at < created_at) -> excluded from latency & KM!
        in_memory_db.add(
            PullRequest(
                id=80055,
                repository_id=8005,
                pr_number=6,
                title="Negative merge skew",
                state="closed",
                author="alice",
                author_id=101,
                created_at=ref_time - timedelta(days=5),
                closed_at=ref_time - timedelta(days=6),
                merged_at=ref_time - timedelta(days=6),
            )
        )

        # 5 closed-unmerged PRs (abandonment)
        for idx in range(5):
            c_dt = ref_time - timedelta(days=15)
            in_memory_db.add(
                PullRequest(
                    id=80060 + idx,
                    repository_id=8005,
                    pr_number=10 + idx,
                    title=f"Abandoned #{idx}",
                    state="closed",
                    author="bob",
                    author_id=102,
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=3),
                    merged_at=None,
                )
            )

        # 1 PR created BEFORE ref_time that merges AFTER ref_time -> MUST be treated as OPEN at ref_time!
        in_memory_db.add(
            PullRequest(
                id=80070,
                repository_id=8005,
                pr_number=20,
                title="Merged after ref_time",
                state="closed",
                author="bob",
                author_id=102,
                created_at=ref_time - timedelta(days=5),
                closed_at=ref_time + timedelta(days=2),
                merged_at=ref_time + timedelta(days=2),
            )
        )

        # 1 PR created AFTER ref_time -> MUST be completely excluded!
        in_memory_db.add(
            PullRequest(
                id=80071,
                repository_id=8005,
                pr_number=21,
                title="Future PR",
                state="open",
                author="carol",
                author_id=103,
                created_at=ref_time + timedelta(hours=1),
            )
        )
        in_memory_db.commit()

        res = calc.calculate(8005, in_memory_db, ref_time, snapshot_revision=1)
        out = res.metrics["outcome_distribution"]
        ml = res.metrics["merge_latency"]

        assert out["total_eligible_prs"] == 12  # 5 + 1 corrupt merged + 5 closed_unmerged + 1 open_at_ref_time
        assert out["future_prs_excluded_count"] == 1
        assert out["merged_count"] == 6
        assert out["closed_unmerged_count"] == 5
        assert out["open_count"] == 1  # Future-merged PR #20 is open as of ref_time!
        assert out["merge_rate"] == round(6 / 11, 4)
        assert out["abandonment_rate"] == round(5 / 11, 4)

        assert ml["status"] == "sufficient"
        assert ml["valid_merged_count"] == 5
        assert ml["zero_hour_merges_count"] == 1  # Instant 0.0h merge preserved!
        assert ml["invalid_records_count"] == 1  # Negative clock-skew PR excluded!
        assert ml["p50_hours"] == 8.0
        assert ml["iqr_hours"] == 8.0  # [0, 4, 8, 12, 20] -> P25=4, P75=12, IQR=8

        assert any(r.rule_id == "RULE-PR-003" for r in res.rules)  # abandonment 5/11 >= 35%
        assert any(r.rule_id == "RULE-PR-005" for r in res.rules)  # negative duration reported

        # Non-UTC offset representing identical instant must match identically
        res_ist = calc.calculate(
            8005,
            in_memory_db,
            ref_time.astimezone(timezone(timedelta(hours=5, minutes=30))),
            snapshot_revision=1,
        )
        assert res_ist.metrics == res.metrics


class TestKaplanMeierSurvivalAndBottlenecks:
    """Verifies KM math axioms, strict >= 15 PR gate, PR size bins with NULL handling, and bottlenecks."""

    def test_kaplan_meier_pure_mathematical_properties(self):
        # Plan benchmark check: 10 items, 5 closed at day 10, 5 right-censored at day 20 => S(10)=0.50, S(20)=0.50
        records = [(10.0, 1)] * 5 + [(20.0, 0)] * 5
        km = compute_kaplan_meier_curve(records, horizons_days=(7.0, 10.0, 20.0, 30.0))
        assert km["horizon_survival"][7.0] == 1.0
        assert km["horizon_survival"][10.0] == 0.5
        assert km["horizon_survival"][20.0] == 0.5
        assert km["median_days"] == 10.0

        # Tie convention: events at day 10 happen before simultaneous censorings at day 10
        # n=4: 2 events at 10.0, 2 censored at 10.0 => S(10) = 1 - 2/4 = 0.50
        km_ties = compute_kaplan_meier_curve([(10.0, 1), (10.0, 1), (10.0, 0), (10.0, 0)], horizons_days=(10.0,))
        assert km_ties["horizon_survival"][10.0] == 0.5

        # All-censored data: survival stays 1.0, median not reached, monotonic non-increasing check
        km_censored = compute_kaplan_meier_curve([(5.0, 0), (15.0, 0), (40.0, 0)], horizons_days=(7.0, 30.0))
        assert km_censored["horizon_survival"][7.0] == 1.0
        assert km_censored["horizon_survival"][30.0] == 1.0
        assert km_censored["median_days"] is None
        probs = [s["survival_prob"] for s in km["curve_steps"]]
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_km_15_pr_cutoff_survival_size_bins_and_bottlenecks(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=8006, repo_name="pr/km15", owner="pr", name="km15", analytics_source_revision=1))

        # First insert 14 valid PRs + 1 negative-duration PR (still 14 valid for KM -> MUST stay insufficient!)
        # Cohort A (5 PRs): small (0..20 lines, including genuine 0-line diff!), merge day 3, review 2h
        for i in range(5):
            c_dt = ref_time - timedelta(days=40)
            in_memory_db.add(
                PullRequest(
                    id=80100 + i,
                    repository_id=8006,
                    pr_number=1 + i,
                    title=f"Small #{i}",
                    state="closed",
                    author="alice",
                    author_id=101,
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=3),
                    merged_at=c_dt + timedelta(days=3),
                    additions=0 if i == 0 else 10,
                    deletions=0 if i == 0 else 5,
                    comments_count=2,
                    review_comments_count=1,
                )
            )
            in_memory_db.add(
                PullRequestReview(
                    pull_request_id=80100 + i,
                    review_id=81100 + i,
                    author="reviewer_a",
                    author_id=401,
                    state="APPROVED",
                    submitted_at=c_dt + timedelta(hours=2),
                )
            )

        # Cohort B (5 PRs): XL (>1000 lines), merge day 10 (240h), slow review 100h (> 72h)
        for i in range(5):
            c_dt = ref_time - timedelta(days=40)
            in_memory_db.add(
                PullRequest(
                    id=80110 + i,
                    repository_id=8006,
                    pr_number=10 + i,
                    title=f"XL #{i}",
                    state="closed",
                    author="alice",
                    author_id=101,
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=10),
                    merged_at=c_dt + timedelta(days=10),
                    additions=900,
                    deletions=300,
                    comments_count=4,
                    review_comments_count=6,
                )
            )
            in_memory_db.add(
                PullRequestReview(
                    pull_request_id=80110 + i,
                    review_id=81110 + i,
                    author="reviewer_a",
                    author_id=401,
                    state="CHANGES_REQUESTED",
                    submitted_at=c_dt + timedelta(hours=100),
                )
            )

        # Cohort C (4 PRs right now -> 14 valid total): Open PRs unreviewed created 40 days ago (NULL diffs!)
        for i in range(4):
            c_dt = ref_time - timedelta(days=40)
            in_memory_db.add(
                PullRequest(
                    id=80120 + i,
                    repository_id=8006,
                    pr_number=20 + i,
                    title=f"Open #{i}",
                    state="open",
                    author="bob",
                    author_id=102,
                    created_at=c_dt,
                    additions=None,
                    deletions=None,
                )
            )
        in_memory_db.commit()

        res_14 = calc.calculate(8006, in_memory_db, ref_time, snapshot_revision=1)
        assert res_14.metrics["survival_km"]["sample_count"] == 14
        assert res_14.metrics["survival_km"]["status"] == "insufficient"
        assert res_14.metrics["survival_km"]["s7d"] is None
        assert res_14.metrics["survival_km"]["s30d"] is None

        # Add 15th valid PR (open, unreviewed, created 40 days ago) -> crosses >= 15 threshold!
        in_memory_db.add(
            PullRequest(
                id=80124,
                repository_id=8006,
                pr_number=24,
                title="Open #4",
                state="open",
                author="bob",
                author_id=102,
                created_at=ref_time - timedelta(days=40),
                additions=None,
                deletions=None,
            )
        )
        in_memory_db.commit()

        res_15 = calc.calculate(8006, in_memory_db, ref_time, snapshot_revision=1)
        km = res_15.metrics["survival_km"]
        size = res_15.metrics["size_vs_turnaround"]
        bot = res_15.metrics["review_bottlenecks"]

        assert km["status"] == "sufficient"
        assert km["sample_count"] == 15
        assert "s14d" not in km  # Decision A.5: no S(14d)
        assert km["s7d"] == round(10.0 / 15.0, 4)  # 5 of 15 closed at day 3
        assert km["s30d"] == round((10.0 / 15.0) * (5.0 / 10.0), 4)  # another 5 closed at day 10
        assert km["median_survival_days"] == 10.0
        assert km["s7d_within_followup"] is True
        assert km["s30d_within_followup"] is True

        # PR Size vs Turnaround preserves 0-line diff vs NULL open diffs
        assert size["status"] == "sufficient"
        assert size["prs_with_diff_count"] == 10
        assert size["prs_zero_line_diff_count"] == 1
        assert size["prs_missing_diff_count"] == 5
        assert size["diff_coverage_ratio"] == round(10 / 15, 4)
        assert size["bins"]["small"]["median_merge_latency_hours"] == 72.0  # 3 days
        assert size["bins"]["xl"]["median_merge_latency_hours"] == 240.0  # 10 days
        assert any(r.rule_id == "RULE-PR-004" for r in res_15.rules)

        # Bottlenecks: latency share, reviewer concentration (reviewer_a = 100% of 10), unreviewed open age (40d)
        assert bot["review_latency_share"] is not None
        assert bot["maintainer_review_load"]["top1_reviewer_share"] == 1.0
        assert bot["unreviewed_open_prs"]["open_prs_count"] == 5
        assert bot["unreviewed_open_prs"]["unreviewed_open_prs_count"] == 5
        assert bot["unreviewed_open_prs"]["median_age_days"] == 40.0
        assert any(r.rule_id == "RULE-PR-002" for r in res_15.rules)

    def test_all_null_pr_diffs_return_unavailable_never_coerced_to_zero(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=8007, repo_name="pr/nulldiff", owner="pr", name="nulldiff", analytics_source_revision=1))
        for i in range(12):
            c_dt = ref_time - timedelta(days=10)
            in_memory_db.add(
                PullRequest(
                    id=80200 + i,
                    repository_id=8007,
                    pr_number=i + 1,
                    title=f"NoDiff #{i}",
                    state="closed",
                    author="alice",
                    created_at=c_dt,
                    closed_at=c_dt + timedelta(days=1),
                    merged_at=c_dt + timedelta(days=1),
                    additions=None,
                    deletions=None,
                )
            )
        in_memory_db.commit()

        res = calc.calculate(8007, in_memory_db, ref_time, snapshot_revision=1)
        size = res.metrics["size_vs_turnaround"]
        assert size["status"] == "unavailable_no_diff_metrics"
        assert size["prs_with_diff_count"] == 0
        assert size["prs_missing_diff_count"] == 12
        assert size["bins"] is None


class TestBenchmarkAndQueueIntegrationC4:
    """Verifies safe offline benchmark DB execution and atomic AnalyticsJobQueue publication."""

    def test_benchmark_repositories_safe_handling(self):
        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip(f"Database {db_path} not found.")

        engine = create_engine(f"sqlite:///{db_path}")
        calc = PullRequestIntelligenceCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)

        with Session(engine) as session:
            repos = session.scalars(select(Repository)).all()
            assert len(repos) >= 4
            for repo in repos:
                res = calc.calculate(repo.id, session, ref_time, repo.analytics_source_revision)
                assert res.domain == "pull_request_intelligence"
                assert res.coverage.status == "empty"
                assert res.metrics["outcome_distribution"]["status"] == "empty"
                assert res.metrics["survival_km"]["status"] == "empty"
                json.dumps(res.metrics)

    def test_pr_calculator_published_via_queue(self, in_memory_db):
        calc = PullRequestIntelligenceCalculator()
        queue = AnalyticsJobQueue()
        ref_time = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(Repository(id=8090, repo_name="queue/c4", owner="queue", name="c4", analytics_source_revision=3, is_active=True))
        in_memory_db.add(
            CollectionRun(
                repository_id=8090,
                mode="standard_diagnostic",
                status="completed",
                is_partial=False,
                coverage_summary={
                    "pull_requests": "completed",
                    "pr_reviews": "completed",
                    "pr_inline_comments": "completed",
                    "issue_comments": "completed",
                },
            )
        )
        for i in range(15):
            c_dt = ref_time - timedelta(days=20 - i)
            in_memory_db.add(
                PullRequest(
                    id=80900 + i,
                    repository_id=8090,
                    pr_number=i + 1,
                    title=f"QPR #{i + 1}",
                    state="closed" if i < 10 else "open",
                    author="qdev",
                    author_id=701,
                    created_at=c_dt,
                    closed_at=(c_dt + timedelta(days=4)) if i < 10 else None,
                    merged_at=(c_dt + timedelta(days=4)) if i < 10 else None,
                    additions=30,
                    deletions=10,
                )
            )
            in_memory_db.add(
                PullRequestReview(
                    pull_request_id=80900 + i,
                    review_id=88900 + i,
                    author="qrev",
                    author_id=702,
                    state="APPROVED",
                    submitted_at=c_dt + timedelta(hours=5),
                )
            )
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=8090, db=in_memory_db)
        claimed_run, attempt = queue.claim_job(worker_id="c4_worker", db=in_memory_db)

        with queue.open_analysis_snapshot(db=in_memory_db, repository_id=8090) as (snap_rev, snap_ref_time):
            res = calc.calculate(8090, in_memory_db, snap_ref_time, snapshot_revision=snap_rev)
            json.dumps(res.metrics)

        published = queue.publish_and_finalize(
            run_id=claimed_run.id,
            attempt_id=attempt.id,
            worker_id="c4_worker",
            expected_lease_version=claimed_run.lease_version,
            attempt_number=attempt.attempt_number,
            snapshot_revision=snap_rev,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=snap_ref_time,
            metrics_payload={"pull_request_intelligence": res.metrics},
            coverage_summary={"pull_request_intelligence": res.coverage.model_dump()},
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
        payload = claimed_run.metrics_payload["pull_request_intelligence"]
        assert payload["survival_km"]["status"] == "sufficient"
        assert payload["outcome_distribution"]["merged_count"] == 10
        assert set(claimed_run.coverage_summary["pull_request_intelligence"]["details"]["subdomains"].keys()) == {
            "ttf_resp",
            "ttfr_formal",
            "ttfr_inline",
            "merge_latency",
            "outcome_distribution",
            "survival_km",
            "size_vs_turnaround",
            "review_bottlenecks",
        }
