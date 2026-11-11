"""Unit and Integration Tests for Release Evolution & Cross-Domain Synthesis Engine (Phase C.6).

Verifies:
1. Release cadence & CV_release thresholds (0/1/2/3/5+ releases, <=0.35 / 0.35..0.75 / >0.75, zero-interval guard).
2. Exclusion of published_at IS NULL and future (> ref_time) releases without schema invention.
3. Pre-release 14d-vs-60d commit surges (74-day baseline depth guard, zero-baseline dormant flag, >=2 aggregate gate).
4. SemVer 2.0.0 adoption regex & Major/Minor/Patch/Prerelease categorization.
5. Cross-domain temporal monthly pairs (Churn vs Issues ignoring NULL diffs, TTFResp vs Abandonment, Velocity vs PR Merges).
6. K >= 12 overlapping months gate, deterministic SHA-256 B=10,000 permutation test with plus-one correction,
   Benjamini-Hochberg FDR q-values, |r_AR1| > 0.40 p-value suppression, and non-causal wording.
7. Objective Multi-Domain Diagnostic Scorecard (no CHI score), repo context without last_synced_at, embedded release_evolution,
   and deterministic rule_engine consolidation preserving C.2-C.5 rules.
8. Live execution on data/opensource_lens.db, AnalysisPayload Pydantic validation, and AnalyticsJobQueue publication.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import pytest
from scipy import stats as scipy_stats
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    Commit,
    Issue,
    PullRequest,
    PullRequestReview,
    Release,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    AnalysisPayload,
    CURRENT_METHODOLOGY_VERSION,
)
from backend.app.services.analytics import (
    ContributorIntelligenceCalculator,
    CrossDomainSynthesisCalculator,
    DevelopmentActivityCalculator,
    IssueIntelligenceCalculator,
    PullRequestIntelligenceCalculator,
    ReleaseEvolutionCalculator,
)
from backend.app.services.analytics.cross_domain import (
    apply_benjamini_hochberg_fdr,
    evaluate_paired_spearman_permutation,
)
from backend.app.services.analytics.rule_engine import consolidate_rules
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


class TestReleaseEvolutionC6:
    """Verifies Module 5 (releases.py): cadence, CV_release, surges, and SemVer 2.0.0."""

    def test_zero_and_single_release_plus_missing_and_future_exclusions(self, in_memory_db):
        calc = ReleaseEvolutionCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=1001, repo_name="rel/empty", owner="rel", name="empty", analytics_source_revision=1))

        # 1 undated draft/tag (published_at=None) + 1 future release (> ref_time) -> net valid R == 0!
        in_memory_db.add(Release(repository_id=1001, release_id=1, tag_name="v0.1.0-draft", published_at=None))
        in_memory_db.add(Release(repository_id=1001, release_id=2, tag_name="v0.2.0", published_at=ref_time + timedelta(days=5)))
        in_memory_db.commit()

        res_empty = calc.calculate(1001, in_memory_db, ref_time, snapshot_revision=1)
        assert res_empty.domain == "release_evolution"
        assert res_empty.coverage.status == "empty"
        assert res_empty.metrics["cadence"]["status"] == "no_releases_recorded"
        assert res_empty.metrics["cadence"]["missing_published_at_count"] == 1
        assert res_empty.metrics["cadence"]["future_releases_excluded_count"] == 1
        assert any(r.rule_id == "RULE-REL-EMPTY" for r in res_empty.rules)

        # Add 1 valid release published 400 days ago -> R=1, triggers single_release_recorded & RULE-REL-002 dormant warning!
        in_memory_db.add(
            Release(
                repository_id=1001,
                release_id=3,
                tag_name="v1.0.0",
                published_at=ref_time - timedelta(days=400),
                is_prerelease=False,
            )
        )
        in_memory_db.commit()

        res_one = calc.calculate(1001, in_memory_db, ref_time, snapshot_revision=1)
        assert res_one.coverage.status == "insufficient"
        assert res_one.metrics["cadence"]["status"] == "single_release_recorded"
        assert res_one.metrics["cadence"]["days_since_last_release"] == 400.0
        assert res_one.metrics["cadence"]["cv_release"] is None
        assert any(r.rule_id == "RULE-REL-002" for r in res_one.rules)

    def test_regular_cadence_cv_semver_and_surge_guards(self, in_memory_db):
        calc = ReleaseEvolutionCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=1002, repo_name="rel/regular", owner="rel", name="regular", analytics_source_revision=1))

        # 5 regular releases spaced ~30 days apart -> R=5, CV ~ 0.0 -> highly_scheduled, RULE-REL-001
        # Tags test Major, Minor, Patch, Prerelease, and 1 Non-SemVer tag => 4/5 = 0.80 adoption (>= 80%!)
        tags = [
            ("v1.0.0", False),           # Major
            ("v1.1.0", False),           # Minor
            ("v1.1.1+build7", False),    # Patch with build metadata
            ("v1.2.0-rc.1", True),       # Prerelease
            ("calver-2026-10", False),   # Non-SemVer
        ]
        release_times = [ref_time - timedelta(days=150 - 30 * i) for i in range(5)]
        for i, ((tag, is_pre), pub_dt) in enumerate(zip(tags, release_times)):
            in_memory_db.add(
                Release(
                    repository_id=1002,
                    release_id=200 + i,
                    tag_name=tag,
                    published_at=pub_dt,
                    is_prerelease=is_pre,
                )
            )

        # Commits starting only 160d before ref_time:
        # - Release 0 (ref - 150d): commit history begins at ref - 160d (10d prior < 74d) => insufficient_baseline_depth!
        # - Release 1 (ref - 120d): history begins 40d prior (< 74d) => insufficient_baseline_depth!
        # - Release 2 (ref - 90d): history spans 70d prior (< 74d) => insufficient_baseline_depth!
        # - Release 3 (ref - 60d): has 100d prior (>= 74d!), with 0 commits in [T-74d, T-14d) => dormant_baseline_activity!
        # - Release 4 (ref - 30d): has 130d prior (>= 74d!), with 60 baseline commits (1/day) in [T-74d, T-14d) and 28 pre commits (2/day) in [T-14d, T] => SM = 2.0!
        earliest = ref_time - timedelta(days=160)
        in_memory_db.add(Commit(repository_id=1002, sha="c_anchor", author_date=earliest))

        # Populate exact baseline & pre-release commits ONLY for Release 4 (T4 = ref - 30d; no overlap with Release 3's 60d window [ref-134d, ref-74d))
        # Wait: ensure Release 3 [ref-134d, ref-74d) stays empty (dormant), while Release 4 baseline is [ref-104d, ref-44d):
        # Put 60 commits at ref - 55d (strictly inside Release 4 baseline [ref-104d, ref-44d), outside Release 3 baseline!)
        t4 = release_times[4]
        for k in range(60):
            in_memory_db.add(
                Commit(
                    repository_id=1002,
                    sha=f"base4_{k}",
                    author_date=t4 - timedelta(days=20) - timedelta(minutes=k),
                )
            )
        # Put 28 commits in Release 4 pre-release window [t4 - 14d, t4] (using exact boundary t4 - 14d to test inclusive pre-release!)
        for k in range(28):
            dt_pre = t4 - timedelta(days=14) if k == 0 else t4 - timedelta(days=2) - timedelta(minutes=k)
            in_memory_db.add(Commit(repository_id=1002, sha=f"pre4_{k}", author_date=dt_pre))
        in_memory_db.commit()

        res_single_valid = calc.calculate(1002, in_memory_db, ref_time, snapshot_revision=1)
        cad = res_single_valid.metrics["cadence"]
        surg = res_single_valid.metrics["prerelease_surges"]
        sem = res_single_valid.metrics["semver_adoption"]

        assert cad["status"] == "sufficient"
        assert cad["mean_interval_days"] == 30.0
        assert cad["cv_release"] == 0.0
        assert cad["regularity_classification"] == "highly_scheduled"
        assert sem["semver_adoption_rate"] == 0.8
        assert sem["major_releases_count"] == 1
        assert sem["minor_releases_count"] == 1
        assert sem["patch_releases_count"] == 1
        assert sem["prerelease_count"] == 1
        assert sem["non_semver_count"] == 1

        # Surge gate: 3 insufficient depth, 1 dormant baseline, 1 computed (SM=2.0) => aggregate mean MUST stay None until >= 2 valid releases!
        assert surg["valid_surge_releases_count"] == 1
        assert surg["insufficient_baseline_depth_count"] == 3
        assert surg["dormant_baseline_count"] == 1
        assert surg["per_release_surges"][4]["surge_multiplier"] == 2.0
        assert surg["status"] == "insufficient"
        assert surg["mean_surge_ratio"] is None

        # Add a 6th release at ref_time with >= 74d history, 30 baseline commits (0.5/day) & 21 pre commits (1.5/day => SM = 3.0!)
        t5 = ref_time
        in_memory_db.add(Release(repository_id=1002, release_id=299, tag_name="v1.2.0", published_at=t5, is_prerelease=False))
        for k in range(30):
            in_memory_db.add(Commit(repository_id=1002, sha=f"base5_{k}", author_date=t5 - timedelta(days=50) - timedelta(minutes=k)))
        for k in range(21):
            in_memory_db.add(Commit(repository_id=1002, sha=f"pre5_{k}", author_date=t5 - timedelta(days=3) - timedelta(minutes=k)))
        in_memory_db.commit()

        # Note Wait: Release 5 pre-release [ref-14d, ref] has 21 commits; Release 5 baseline [ref-74d, ref-14d) ALSO sees Release 4's pre-release commits (t4-14..t4 = [ref-44d, ref-30d]) & baseline!
        # Let's simply assert valid_surge_releases_count == 2 -> aggregate status becomes sufficient & mean_surge_ratio populated!
        res_two_valid = calc.calculate(1002, in_memory_db, ref_time, snapshot_revision=1)
        surg2 = res_two_valid.metrics["prerelease_surges"]
        assert surg2["status"] == "sufficient"
        assert surg2["valid_surge_releases_count"] == 2
        assert surg2["mean_surge_ratio"] is not None
        assert surg2["median_surge_ratio"] is not None
        assert any(r.rule_id == "RULE-REL-001" for r in res_two_valid.rules)
        assert any(r.rule_id == "RULE-REL-004" for r in res_two_valid.rules)

    def test_cv_classifications_and_zero_interval_guard(self, in_memory_db):
        calc = ReleaseEvolutionCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)

        # Repo 1003: intervals [20, 30, 45, 55] -> mean=37.5, stdev=15.5456, CV=0.4146 => moderately_predictable
        in_memory_db.add(Repository(id=1003, repo_name="rel/moderate", owner="rel", name="moderate", analytics_source_revision=1))
        for i, offset in enumerate([150, 130, 100, 55, 0]):
            in_memory_db.add(
                Release(
                    repository_id=1003,
                    release_id=300 + i,
                    tag_name=f"v1.{i}.0",
                    published_at=ref_time - timedelta(days=offset),
                    is_prerelease=False,
                )
            )

        # Repo 1004: intervals [2, 3, 5, 100] -> CV > 0.75 => event_driven_sporadic
        in_memory_db.add(Repository(id=1004, repo_name="rel/sporadic", owner="rel", name="sporadic", analytics_source_revision=1))
        for i, offset in enumerate([110, 108, 105, 100, 0]):
            in_memory_db.add(
                Release(
                    repository_id=1004,
                    release_id=400 + i,
                    tag_name=f"v2.{i}.0",
                    published_at=ref_time - timedelta(days=offset),
                    is_prerelease=False,
                )
            )

        # Repo 1005: 3 releases at identical timestamp -> zero mean interval guard (cv_release is None)
        in_memory_db.add(Repository(id=1005, repo_name="rel/zeroint", owner="rel", name="zeroint", analytics_source_revision=1))
        same_dt = ref_time - timedelta(days=10)
        for i in range(3):
            in_memory_db.add(
                Release(
                    repository_id=1005,
                    release_id=500 + i,
                    tag_name=f"v0.1.{i}",
                    published_at=same_dt,
                    is_prerelease=False,
                )
            )
        in_memory_db.commit()

        res_mod = calc.calculate(1003, in_memory_db, ref_time, snapshot_revision=1)
        assert res_mod.metrics["cadence"]["regularity_classification"] == "moderately_predictable"
        assert 0.35 < res_mod.metrics["cadence"]["cv_release"] <= 0.75

        res_spor = calc.calculate(1004, in_memory_db, ref_time, snapshot_revision=1)
        assert res_spor.metrics["cadence"]["regularity_classification"] == "event_driven_sporadic"
        assert res_spor.metrics["cadence"]["cv_release"] > 0.75

        res_zero = calc.calculate(1005, in_memory_db, ref_time, snapshot_revision=1)
        assert res_zero.metrics["cadence"]["regularity_classification"] == "zero_mean_interval"
        assert res_zero.metrics["cadence"]["cv_release"] is None
        assert res_zero.metrics["cadence"]["annualized_release_frequency"] is None


class TestCrossDomainStatisticsAndCorrelationsC6:
    """Verifies K>=12 gate, SHA-256 B=10,000 permutations, BH FDR, AR(1) guardrail, and aligned pairs."""

    def test_insufficient_periods_zero_variance_and_ar1_guardrail(self):
        # 1. K=11 < 12 => insufficient_temporal_periods
        m11 = {(2025, m): float(m) for m in range(1, 12)}
        res_k11 = evaluate_paired_spearman_permutation(
            1, 1, "P", "x", "X", "y", "Y", m11, m11, permutations=1000
        )
        assert res_k11["k_months"] == 11
        assert res_k11["status"] == "insufficient_temporal_periods"
        assert res_k11["p_value"] is None

        # 2. Constant series (K=12) => undefined_zero_variance
        m12_x = {(2025, m): 5.0 for m in range(1, 13)}
        m12_y = {(2025, m): float(m) for m in range(1, 13)}
        res_zv = evaluate_paired_spearman_permutation(
            1, 1, "P", "x", "X", "y", "Y", m12_x, m12_y, permutations=1000
        )
        assert res_zv["status"] == "undefined_zero_variance"
        assert res_zv["p_value"] is None

        # 3. Monotonic ramp (K=14): high AR(1) > 0.40 -> serial_dependence_detected, p_value suppressed!
        m14_x = {(2025 + (m - 1) // 12, ((m - 1) % 12) + 1): float(m) for m in range(1, 15)}
        res_ar = evaluate_paired_spearman_permutation(
            1, 1, "P", "x", "X", "y", "Y", m14_x, m14_x, permutations=1000
        )
        assert res_ar["status"] == "serial_dependence_detected"
        assert res_ar["serial_dependence_guardrail_tripped"] is True
        assert res_ar["max_abs_ar1"] > 0.40
        assert res_ar["spearman_r"] == 1.0
        assert res_ar["p_value"] is None
        assert res_ar["q_value"] is None

    def test_deterministic_seeded_permutation_exactness_and_bh_fdr(self):
        # Low-autocorrelation 12-month vectors (|r_AR1| <= 0.10) so permutation runs cleanly
        months = [(2025, m) for m in range(1, 13)]
        x_vals = [6.0, 10.0, 8.0, 2.0, 3.0, 9.0, 12.0, 7.0, 1.0, 4.0, 11.0, 5.0]
        # Strongly covariant low-AR1 partner (with ties)
        y_vals = [12.0, 20.0, 15.0, 4.0, 4.0, 18.0, 25.0, 14.0, 2.0, 9.0, 23.0, 10.0]
        sx = dict(zip(months, x_vals))
        sy = dict(zip(months, y_vals))

        res_a = evaluate_paired_spearman_permutation(
            42, 7, "PAIR-TEST", "metric_a", "Metric A", "metric_b", "Metric B", sx, sy, permutations=10000
        )
        res_b = evaluate_paired_spearman_permutation(
            42, 7, "PAIR-TEST", "metric_a", "Metric A", "metric_b", "Metric B", sx, sy, permutations=10000
        )
        res_diff_rev = evaluate_paired_spearman_permutation(
            42, 8, "PAIR-TEST", "metric_a", "Metric A", "metric_b", "Metric B", sx, sy, permutations=10000
        )

        # Bit-identical reproduction for identical (repo, rev, metric_x, metric_y, methodology, B)
        assert res_a == res_b
        assert res_a["permutation_seed"] != res_diff_rev["permutation_seed"]
        assert res_a["status"] == "tested"
        expected_r = round(float(scipy_stats.spearmanr(x_vals, y_vals).statistic), 4)
        assert res_a["spearman_r"] == expected_r
        assert (1.0 / 10001.0) <= res_a["p_value"] <= 1.0
        assert res_a["p_value"] < 0.01

        # Verify Benjamini-Hochberg FDR step-up adjustment mathematically
        pairs = [
            {"status": "tested", "p_value": 0.001, "q_value": None, "significant_fdr_05": False},
            {"status": "tested", "p_value": 0.030, "q_value": None, "significant_fdr_05": False},
            {"status": "tested", "p_value": 0.200, "q_value": None, "significant_fdr_05": False},
            {"status": "serial_dependence_detected", "p_value": None, "q_value": None, "significant_fdr_05": False},
        ]
        apply_benjamini_hochberg_fdr(pairs, alpha=0.05)
        # m = 3 tested: raw q = [0.003, 0.045, 0.200] -> q = [0.003, 0.045, 0.200]
        assert pairs[0]["q_value"] == 0.003
        assert pairs[0]["significant_fdr_05"] is True
        assert pairs[1]["q_value"] == 0.045
        assert pairs[1]["significant_fdr_05"] is True
        assert pairs[2]["q_value"] == 0.2
        assert pairs[2]["significant_fdr_05"] is False
        assert pairs[3]["q_value"] is None

    def test_cross_domain_monthly_alignment_and_null_diff_exclusion(self, in_memory_db):
        """14-month synthetic repo validating all 3 pairs, self/bot review exclusion, and NULL diff month dropping."""
        calc = CrossDomainSynthesisCalculator()
        ref_time = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(
            Repository(
                id=2001,
                repo_name="syn/multi",
                owner="syn",
                name="multi",
                stars_count=500,
                forks_count=50,
                primary_language="Python",
                is_archived=True,  # Triggers RULE-SYN-001!
                last_synced_at=ref_time,
                analytics_source_revision=4,
            )
        )

        # Low-AR(1) pattern over 14 calendar months (|r_AR1| ~ 0.09)
        base_months = [(2025 + (m // 12), (m % 12) + 1) for m in range(14)]
        c_pattern = [6, 11, 9, 2, 3, 10, 13, 7, 1, 4, 12, 5, 8, 14]
        pr_id_seq = 5000
        iss_id_seq = 6000
        rev_id_seq = 7000

        for idx, (yr, mo) in enumerate(base_months):
            m_dt = datetime(yr, mo, 5, 12, 0, tzinfo=timezone.utc)
            n_items = c_pattern[idx]

            # Commits with valid additions/deletions; churn proportional to n_items
            for c_i in range(n_items):
                in_memory_db.add(
                    Commit(
                        repository_id=2001,
                        sha=f"c_{yr}_{mo}_{c_i}",
                        author_id=101,
                        author_login="dev_main",
                        author_date=m_dt + timedelta(hours=c_i),
                        additions=50 * (c_i + 1),
                        deletions=10,
                    )
                )

            # Issues matching c_pattern[idx] creations
            for i_i in range(n_items):
                in_memory_db.add(
                    Issue(
                        id=iss_id_seq,
                        repository_id=2001,
                        repo_name="syn/multi",
                        issue_number=iss_id_seq,
                        title=f"Iss {iss_id_seq}",
                        state="closed" if i_i % 2 == 0 else "open",
                        author="reporter",
                        created_at=m_dt + timedelta(hours=i_i),
                        closed_at=(m_dt + timedelta(days=2)) if i_i % 2 == 0 else None,
                    )
                )
                iss_id_seq += 1

            # PRs: 2 PRs per month; months with larger n_items get slower external TTFResp and higher abandonment
            slow_h = float(n_items * 4)
            for p_i in range(2):
                is_abandoned = (n_items >= 10) and (p_i == 0)
                in_memory_db.add(
                    PullRequest(
                        id=pr_id_seq,
                        repository_id=2001,
                        pr_number=pr_id_seq,
                        title=f"PR {pr_id_seq}",
                        state="closed",
                        author="author_x",
                        author_id=201,
                        created_at=m_dt,
                        closed_at=m_dt + timedelta(days=1),
                        merged_at=None if is_abandoned else (m_dt + timedelta(days=1)),
                    )
                )
                # Poisoned self-review at 0.1h + genuine external review at slow_h
                in_memory_db.add(
                    PullRequestReview(
                        pull_request_id=pr_id_seq,
                        review_id=rev_id_seq,
                        author="author_x",
                        author_id=201,
                        state="COMMENTED",
                        submitted_at=m_dt + timedelta(minutes=6),
                    )
                )
                rev_id_seq += 1
                in_memory_db.add(
                    PullRequestReview(
                        pull_request_id=pr_id_seq,
                        review_id=rev_id_seq,
                        author="maintainer",
                        author_id=202,
                        state="APPROVED",
                        submitted_at=m_dt + timedelta(hours=slow_h),
                    )
                )
                rev_id_seq += 1
                pr_id_seq += 1
        in_memory_db.commit()

        res = calc.calculate(2001, in_memory_db, ref_time, snapshot_revision=4)
        corr = res.metrics["cross_domain_correlations"]
        pairs = {p["pair_id"]: p for p in corr["pairs"]}

        assert res.metrics["chi_deferred"] is True
        assert "composite_health_score" not in res.metrics
        assert "last_synced_at" not in json.dumps(res.metrics)

        # Pair 1, Pair 2 & Pair 3 span 14 overlapping months; Pair 1 & Pair 3 have low AR1!
        assert pairs["PAIR-01"]["k_months"] == 14
        assert pairs["PAIR-01"]["status"] == "tested"
        assert pairs["PAIR-01"]["q_value"] is not None
        assert pairs["PAIR-02"]["k_months"] == 14
        assert pairs["PAIR-02"]["spearman_r"] > 0.5
        assert pairs["PAIR-03"]["k_months"] == 14
        assert "causation is never implied" in corr["causality_disclaimer"].lower()
        assert any(r.rule_id == "RULE-SYN-001" for r in res.rules)  # archived repo flag
        assert any(r.rule_id == "RULE-SYN-002" for r in res.rules)  # FDR-significant non-causal association

        # Verify NULL diff coverage drops those months from PAIR-01 rather than coercing to 0!
        in_memory_db.add(Repository(id=2002, repo_name="syn/nulldiff", owner="syn", name="nulldiff", analytics_source_revision=1))
        for idx, (yr, mo) in enumerate(base_months[:12]):
            m_dt = datetime(yr, mo, 5, 12, 0, tzinfo=timezone.utc)
            # Month idx == 11 gets additions=None (missing diff) -> leaves only 11 valid months (< 12)!
            in_memory_db.add(
                Commit(
                    repository_id=2002,
                    sha=f"nc_{idx}",
                    author_id=101,
                    author_date=m_dt,
                    additions=None if idx == 11 else 50,
                    deletions=None if idx == 11 else 10,
                )
            )
            in_memory_db.add(
                Issue(
                    id=7000 + idx,
                    repository_id=2002,
                    repo_name="syn/nulldiff",
                    issue_number=idx + 1,
                    title=f"NIss {idx}",
                    state="open",
                    author="u",
                    created_at=m_dt,
                )
            )
        in_memory_db.commit()

        res_null = calc.calculate(2002, in_memory_db, ref_time, snapshot_revision=1)
        p1_null = {p["pair_id"]: p for p in res_null.metrics["cross_domain_correlations"]["pairs"]}["PAIR-01"]
        assert p1_null["k_months"] == 11
        assert p1_null["status"] == "insufficient_temporal_periods"
        assert p1_null["spearman_r"] is None

    def test_end_to_end_rule_syn_003_serial_dependence_repo(self, in_memory_db):
        calc = CrossDomainSynthesisCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)
        in_memory_db.add(Repository(id=2003, repo_name="syn/trend", owner="syn", name="trend", analytics_source_revision=1))

        months = [(2025, m) for m in range(1, 13)] + [(2026, 1), (2026, 2)]
        iss_seq = 8000
        for idx, (yr, mo) in enumerate(months):
            m_dt = datetime(yr, mo, 5, 12, 0, tzinfo=timezone.utc)
            n = idx + 1  # Strictly monotonic trending ramp => high positive |r_AR1| > 0.40
            for c_i in range(n):
                in_memory_db.add(
                    Commit(
                        repository_id=2003,
                        sha=f"tr_{yr}_{mo}_{c_i}",
                        author_id=101,
                        author_date=m_dt + timedelta(hours=c_i),
                        additions=100 * n,
                        deletions=20,
                    )
                )
            for i_i in range(n):
                in_memory_db.add(
                    Issue(
                        id=iss_seq,
                        repository_id=2003,
                        repo_name="syn/trend",
                        issue_number=iss_seq,
                        title=f"TrendIss {iss_seq}",
                        state="open",
                        author="u",
                        created_at=m_dt + timedelta(hours=i_i),
                    )
                )
                iss_seq += 1
        in_memory_db.commit()

        res_trend = calc.calculate(2003, in_memory_db, ref_time, snapshot_revision=1)
        p1 = {p["pair_id"]: p for p in res_trend.metrics["cross_domain_correlations"]["pairs"]}["PAIR-01"]
        assert p1["k_months"] == 14
        assert p1["status"] == "serial_dependence_detected"
        assert p1["p_value"] is None
        assert p1["q_value"] is None
        assert p1["max_abs_ar1"] > 0.40
        assert any(r.rule_id == "RULE-SYN-003" for r in res_trend.rules)


class TestBenchmarkPayloadAndQueueIntegrationC6:
    """Verifies rule consolidation, offline benchmark DB safety, AnalysisPayload schema, and queue persistence."""

    def test_benchmark_repositories_synthesis_and_no_last_synced_at(self):
        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip(f"Database {db_path} not found.")

        engine = create_engine(f"sqlite:///{db_path}")
        calc = CrossDomainSynthesisCalculator()
        ref_time = datetime(2026, 11, 11, 0, 0, 0, tzinfo=timezone.utc)

        with Session(engine) as session:
            repos = session.scalars(select(Repository)).all()
            assert len(repos) == 4
            for repo in repos:
                res = calc.calculate(repo.id, session, ref_time, repo.analytics_source_revision)
                assert res.domain == "repository_synthesis"
                assert res.coverage.status == "degraded"  # 1 of 5 sufficient (issue_intelligence)
                assert res.metrics["chi_deferred"] is True
                assert res.metrics["repository_context"]["stars_count"] == repo.stars_count
                assert res.metrics["release_evolution"]["cadence"]["status"] == "no_releases_recorded"
                assert res.metrics["diagnostic_scorecard"]["issue_health"]["coverage_status"] == "sufficient"
                dumped = json.dumps(res.metrics)
                assert "last_synced_at" not in dumped

    def test_all_modules_validate_against_analysis_payload_and_publish_via_queue(self, in_memory_db):
        queue = AnalyticsJobQueue()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(
            Repository(
                id=2090,
                repo_name="queue/c6",
                owner="queue",
                name="c6",
                stars_count=42,
                forks_count=7,
                primary_language="Python",
                last_synced_at=ref_time,
                analytics_source_revision=3,
                is_active=True,
            )
        )
        for i in range(6):
            t_rel = ref_time - timedelta(days=180 - 30 * i)
            in_memory_db.add(
                Release(
                    repository_id=2090,
                    release_id=9100 + i,
                    tag_name=f"v1.{i}.0",
                    published_at=t_rel,
                    is_prerelease=False,
                )
            )
            in_memory_db.add(
                Commit(
                    repository_id=2090,
                    sha=f"qc_{i}",
                    author_id=101,
                    author_login="qdev",
                    author_date=t_rel - timedelta(days=5),
                    additions=20,
                    deletions=5,
                )
            )
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=2090, db=in_memory_db)
        claimed_run, attempt = queue.claim_job(worker_id="c6_worker", db=in_memory_db)

        with queue.open_analysis_snapshot(db=in_memory_db, repository_id=2090) as (snap_rev, snap_ref):
            res_act = DevelopmentActivityCalculator().calculate(2090, in_memory_db, snap_ref, snap_rev)
            res_con = ContributorIntelligenceCalculator().calculate(2090, in_memory_db, snap_ref, snap_rev)
            res_pr = PullRequestIntelligenceCalculator().calculate(2090, in_memory_db, snap_ref, snap_rev)
            res_iss = IssueIntelligenceCalculator().calculate(2090, in_memory_db, snap_ref, snap_rev)
            res_rel = ReleaseEvolutionCalculator().calculate(2090, in_memory_db, snap_ref, snap_rev)

            precomputed = {
                "development_activity": res_act,
                "contributor_ecosystem": res_con,
                "pull_request_intelligence": res_pr,
                "issue_intelligence": res_iss,
                "release_evolution": res_rel,
            }
            res_syn = CrossDomainSynthesisCalculator().calculate(
                2090, in_memory_db, snap_ref, snap_rev, precomputed_domains=precomputed
            )

            # Validate full 5-slot AnalysisPayload schema compatibility without schema breaking changes!
            metrics_payload_dict = {
                "development_activity": res_act.metrics,
                "contributor_ecosystem": res_con.metrics,
                "pull_request_intelligence": res_pr.metrics,
                "issue_intelligence": res_iss.metrics,
                "repository_synthesis": res_syn.metrics,
            }
            validated_payload = AnalysisPayload(**metrics_payload_dict)
            assert validated_payload.repository_synthesis["release_evolution"]["semver_adoption"]["semver_adoption_rate"] == 1.0

            consolidated_rules = consolidate_rules(
                [res_act, res_con, res_pr, res_iss],
                res_syn.rules,
            )
            # No duplicate rule triples across consolidated set
            triples = [(r.rule_id, r.domain, r.finding) for r in consolidated_rules]
            assert len(triples) == len(set(triples))

        published = queue.publish_and_finalize(
            run_id=claimed_run.id,
            attempt_id=attempt.id,
            worker_id="c6_worker",
            expected_lease_version=claimed_run.lease_version,
            attempt_number=attempt.attempt_number,
            snapshot_revision=snap_rev,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=snap_ref,
            metrics_payload=metrics_payload_dict,
            coverage_summary={
                "development_activity": res_act.coverage.model_dump(),
                "contributor_ecosystem": res_con.coverage.model_dump(),
                "pull_request_intelligence": res_pr.coverage.model_dump(),
                "issue_intelligence": res_iss.coverage.model_dump(),
                "repository_synthesis": res_syn.coverage.model_dump(),
            },
            confidence_summary={
                "score": round(1.0 - res_syn.confidence_penalty, 3),
                "band": "medium",
                "penalties": res_syn.details.get("penalties", []),
            },
            rule_interpretations=[r.model_dump() for r in consolidated_rules],
            db=in_memory_db,
        )
        assert published is True
        in_memory_db.refresh(claimed_run)
        assert claimed_run.status == "completed"
        assert "release_evolution" in claimed_run.metrics_payload["repository_synthesis"]
        assert "last_synced_at" not in json.dumps(claimed_run.metrics_payload)
