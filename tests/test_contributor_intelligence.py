"""Unit and Integration Tests for Contributor Intelligence Engine (Phase C.3 / Module 2).

Verifies:
1. Empty and single-contributor degenerate edge cases (n=0, n=1).
2. Uniform (G=0, H_norm=1) and extreme Pareto skew (G->1, BF50=1) distributions.
3. Standardized 11-point Lorenz decile interpolation, Shannon entropy, BF50, BF80, and top-k shares.
4. Bot segregation via Contributor.is_bot, unlinked email isolation, and zero speculative display-name merging.
5. Multi-role attribution (commits, PRs authored, PRs reviewed, issues opened) and zero-commit directory counts.
6. Reference-time filtering (ignoring stale pre-aggregated totals & future events) and newcomer/established tenure.
7. Mandatory bounded-mode retention deferral (fast_snapshot, standard_diagnostic, partial, missing run).
8. Complete synthetic-history monthly retention cohort matrix strictly across commits & PRs.
9. Live offline benchmark database handling (data/opensource_lens.db).
10. End-to-end persistence via AnalyticsJobQueue lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    CollectionRun,
    Commit,
    Contributor,
    Issue,
    PullRequest,
    PullRequestReview,
    Repository,
)
from backend.app.schemas.analytics_v2 import CURRENT_METHODOLOGY_VERSION
from backend.app.services.analytics import ContributorIntelligenceCalculator
from backend.app.services.analytics_queue import AnalyticsJobQueue


@pytest.fixture
def in_memory_db():
    """Create isolated in-memory SQLite session for each test."""
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


class TestConcentrationEmptySingleUniformPareto:
    """Verifies Gini, 11-point Lorenz curve, Shannon entropy, BF50/BF80, and top-k shares."""

    def test_empty_and_single_contributor_repositories(self, in_memory_db):
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Completely empty repository
        in_memory_db.add(Repository(id=7001, repo_name="con/empty", owner="con", name="empty", analytics_source_revision=1))
        in_memory_db.commit()

        res_empty = calc.calculate(7001, in_memory_db, ref_time, snapshot_revision=1)
        conc_empty = res_empty.metrics["concentration"]
        assert res_empty.domain == "contributor_ecosystem"
        assert res_empty.coverage.status == "empty"
        assert res_empty.confidence_penalty == 0.5
        assert conc_empty["status"] == "empty"
        assert conc_empty["gini_coefficient"] is None
        assert conc_empty["lorenz_curve"] is None
        assert conc_empty["bf50"] is None
        assert res_empty.metrics["retention"]["status"] == "unavailable_requires_full_history_ingestion"
        assert res_empty.metrics["retention"]["retention_cohorts"] is None
        assert any(r.rule_id == "RULE-CON-EMPTY" for r in res_empty.rules)

        # 2. Single human commit author (degenerate G = 0.0, BF50=1, BF80=1, top1=1.0)
        in_memory_db.add(Repository(id=7002, repo_name="con/single", owner="con", name="single", analytics_source_revision=1))
        in_memory_db.add(Contributor(id=1, repository_id=7002, github_id=101, login="solo_dev"))
        for i in range(12):
            in_memory_db.add(
                Commit(
                    repository_id=7002,
                    sha=f"sha_solo_{i}",
                    author_id=101,
                    author_login="solo_dev",
                    author_date=ref_time - timedelta(days=20 - i),
                )
            )
        in_memory_db.commit()

        res_single = calc.calculate(7002, in_memory_db, ref_time, snapshot_revision=1)
        conc_single = res_single.metrics["concentration"]
        assert res_single.coverage.status == "insufficient"
        assert conc_single["status"] == "insufficient"
        assert conc_single["degenerate_single_contributor"] is True
        assert conc_single["gini_coefficient"] == 0.0
        assert conc_single["lorenz_curve"] is None
        assert conc_single["shannon_entropy_bits"] == 0.0
        assert conc_single["normalized_entropy"] == 0.0
        assert conc_single["bf50"] == 1
        assert conc_single["bf80"] == 1
        assert conc_single["top1_share"] == 1.0
        assert conc_single["top5_share"] == 1.0
        assert any(r.rule_id == "RULE-CON-001" for r in res_single.rules)

    def test_uniform_and_extreme_pareto_concentration(self, in_memory_db):
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Uniform distribution: 10 human contributors x 10 commits = 100 commits
        in_memory_db.add(Repository(id=7003, repo_name="con/uniform", owner="con", name="uniform", analytics_source_revision=1))
        in_memory_db.add(CollectionRun(repository_id=7003, mode="fast_snapshot", status="completed", is_partial=False))
        for u in range(10):
            for c in range(10):
                in_memory_db.add(
                    Commit(
                        repository_id=7003,
                        sha=f"sha_uni_{u}_{c}",
                        author_id=2000 + u,
                        author_login=f"uni_dev_{u}",
                        author_date=ref_time - timedelta(days=c + 1),
                    )
                )

        # 2. Extreme Pareto skew: 1 dominant author (901 commits) + 19 tail authors (1 commit each) -> total 920
        # Gini theoretical for [1]*19 + [901]: 19/20 * (900/920) ~ 0.9293
        in_memory_db.add(Repository(id=7004, repo_name="con/pareto", owner="con", name="pareto", analytics_source_revision=1))
        for c in range(901):
            in_memory_db.add(
                Commit(
                    repository_id=7004,
                    sha=f"sha_par_top_{c}",
                    author_id=3000,
                    author_login="dominant_dev",
                    author_date=ref_time - timedelta(days=1 + (c % 60)),
                )
            )
        for u in range(1, 20):
            in_memory_db.add(
                Commit(
                    repository_id=7004,
                    sha=f"sha_par_tail_{u}",
                    author_id=3000 + u,
                    author_login=f"tail_dev_{u}",
                    author_date=ref_time - timedelta(days=5 + u),
                )
            )
        in_memory_db.commit()

        res_uni = calc.calculate(7003, in_memory_db, ref_time, snapshot_revision=1)
        c_uni = res_uni.metrics["concentration"]
        assert res_uni.coverage.status == "sufficient"
        assert c_uni["status"] == "sufficient"
        assert c_uni["active_human_commit_authors_count"] == 10
        assert c_uni["gini_coefficient"] == 0.0
        assert c_uni["shannon_entropy_bits"] == round(math.log2(10), 4)
        assert c_uni["normalized_entropy"] == 1.0
        assert c_uni["bf50"] == 5
        assert c_uni["bf80"] == 8
        assert c_uni["top1_share"] == 0.1
        assert c_uni["top5_share"] == 0.5
        assert c_uni["is_bounded_sample"] is True
        assert len(c_uni["lorenz_curve"]) == 11
        for idx, pt in enumerate(c_uni["lorenz_curve"]):
            assert pt["population_share"] == round(idx / 10.0, 1)
            assert pt["contribution_share"] == round(idx / 10.0, 4)
        assert any(r.rule_id == "RULE-CON-001" and r.severity == "positive" for r in res_uni.rules)

        res_par = calc.calculate(7004, in_memory_db, ref_time, snapshot_revision=1)
        c_par = res_par.metrics["concentration"]
        assert c_par["active_human_commit_authors_count"] == 20
        assert c_par["gini_coefficient"] > 0.90
        assert c_par["normalized_entropy"] < 0.10
        assert c_par["bf50"] == 1
        assert c_par["bf80"] == 1
        assert c_par["top1_share"] == round(901 / 920, 4)
        assert any(r.rule_id == "RULE-CON-001" and r.severity == "warning" for r in res_par.rules)
        assert any(r.rule_id == "RULE-CON-002" and r.severity == "warning" for r in res_par.rules)

    def test_lorenz_interpolation_and_limited_pool_gate(self, in_memory_db):
        """Verify piecewise decile interpolation on asymmetric counts [10, 20, 30, 40] (n=4 -> limited/degraded)."""
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(Repository(id=7005, repo_name="con/lorenz4", owner="con", name="lorenz4", analytics_source_revision=1))
        counts = [10, 20, 30, 40]  # total = 100, cum = [0.0, 0.1, 0.3, 0.6, 1.0] at p = [0, 0.25, 0.5, 0.75, 1.0]
        for idx, cnt in enumerate(counts):
            for j in range(cnt):
                in_memory_db.add(
                    Commit(
                        repository_id=7005,
                        sha=f"sha_l4_{idx}_{j}",
                        author_id=4000 + idx,
                        author_login=f"l4_dev_{idx}",
                        author_date=ref_time - timedelta(days=1 + j % 10),
                    )
                )
        in_memory_db.commit()

        res = calc.calculate(7005, in_memory_db, ref_time, snapshot_revision=1)
        conc = res.metrics["concentration"]
        assert res.coverage.status == "degraded"  # 2 <= n < 5
        assert conc["status"] == "limited"
        # n=4, X=100 -> Gini = 2*(1*10+2*20+3*30+4*40)/(4*100) - 5/4 = 600/400 - 1.25 = 0.25
        assert conc["gini_coefficient"] == 0.25
        lorenz = {pt["population_share"]: pt["contribution_share"] for pt in conc["lorenz_curve"]}
        assert lorenz[0.0] == 0.0
        assert lorenz[0.5] == 0.3   # p=0.5 exactly hits cumulative 10+20 = 30%
        assert lorenz[0.2] == 0.08  # pos=0.8 -> 0.8 * 0.1 = 0.08
        assert lorenz[1.0] == 1.0
        assert conc["bf50"] == 2    # 40 + 30 = 70 >= 50
        assert conc["bf80"] == 3    # 40 + 30 + 20 = 90 >= 80


class TestIdentityBotRolesDirectoryAndTenure:
    """Verifies bot separation, zero speculative merging, roles, directory accounts, and ref_time tenure."""

    def test_bot_exclusion_unlinked_unattributed_and_zero_commit_directory(self, in_memory_db):
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(Repository(id=7010, repo_name="con/hygiene", owner="con", name="hygiene", analytics_source_revision=1))

        # Directory contributors (note: stale total_commits=999 must be ignored completely!)
        alice_dir = Contributor(id=101, repository_id=7010, github_id=5001, login="alice", is_bot=False, total_commits=999)
        bob_dir = Contributor(id=102, repository_id=7010, github_id=5002, login="bob", is_bot=False, total_commits=888)
        reviewer_only_dir = Contributor(id=103, repository_id=7010, github_id=5003, login="rev_only", is_bot=False)
        dormant_dir = Contributor(id=104, repository_id=7010, github_id=5004, login="zero_dir", is_bot=False)
        bot_dir = Contributor(id=105, repository_id=7010, github_id=5005, login="dependabot[bot]", is_bot=True)
        in_memory_db.add_all([alice_dir, bob_dir, reviewer_only_dir, dormant_dir, bot_dir])

        # Commits: Alice=6, Bob=4, Bot=10, Unlinked email dev=2, Unattributed display-name-only("Twin Name")=2
        for i in range(6):
            in_memory_db.add(Commit(repository_id=7010, sha=f"sha_a_{i}", author_id=5001, author_login="alice", author_date=ref_time - timedelta(days=45 - i)))
        for i in range(4):
            in_memory_db.add(Commit(repository_id=7010, sha=f"sha_b_{i}", author_id=5002, author_login="bob", author_date=ref_time - timedelta(days=10 - i)))
        for i in range(10):
            in_memory_db.add(Commit(repository_id=7010, sha=f"sha_bot_{i}", author_id=5005, author_login="dependabot[bot]", author_date=ref_time - timedelta(days=5 + i)))
        for i in range(2):
            in_memory_db.add(
                Commit(
                    repository_id=7010,
                    sha=f"sha_unlinked_{i}",
                    author_id=None,
                    author_login=None,
                    author_email="external@dev.org",
                    author_name="Ext Dev",
                    author_date=ref_time - timedelta(days=8 + i),
                )
            )
        # Display-name-only anonymous commits sharing identical display name -> must NOT be merged!
        for i in range(2):
            in_memory_db.add(
                Commit(
                    repository_id=7010,
                    sha=f"sha_anon_{i}",
                    author_id=None,
                    author_login=None,
                    author_email=None,
                    author_name="Twin Name",
                    author_date=ref_time - timedelta(days=3 + i),
                )
            )
        # Future commit strictly > ref_time -> MUST be excluded from all metrics
        in_memory_db.add(Commit(repository_id=7010, sha="sha_future_alice", author_id=5001, author_login="alice", author_date=ref_time + timedelta(days=1)))

        # PRs, Reviews, Issues across multiple roles
        pr1 = PullRequest(id=801, repository_id=7010, pr_number=1, title="PR1", state="open", author="alice", author_id=5001, created_at=ref_time - timedelta(days=12))
        pr2 = PullRequest(id=802, repository_id=7010, pr_number=2, title="PR2", state="open", author="rev_only", author_id=5003, created_at=ref_time - timedelta(days=14))
        pr_bot = PullRequest(id=803, repository_id=7010, pr_number=3, title="Bot PR", state="open", author="dependabot[bot]", author_id=5005, created_at=ref_time - timedelta(days=6))
        in_memory_db.add_all([pr1, pr2, pr_bot])
        in_memory_db.flush()

        in_memory_db.add(PullRequestReview(pull_request_id=801, review_id=901, author="rev_only", author_id=5003, state="APPROVED", submitted_at=ref_time - timedelta(days=11)))
        in_memory_db.add(Issue(id=701001, repository_id=7010, repo_name="con/hygiene", issue_number=1, title="Bug", state="open", author="rev_only", created_at=ref_time - timedelta(days=15)))
        in_memory_db.commit()

        res = calc.calculate(7010, in_memory_db, ref_time, snapshot_revision=1)
        part = res.metrics["participation"]
        hyg = res.metrics["identity_hygiene"]
        conc = res.metrics["concentration"]
        ten = res.metrics["tenure"]

        # Directory vs active commit authors vs multi-role human accounts
        assert part["directory_contributors_count"] == 5
        assert part["directory_human_contributors_count"] == 4
        assert part["directory_bot_contributors_count"] == 1
        assert part["zero_commit_directory_human_contributors_count"] == 2  # rev_only + zero_dir
        assert part["active_human_commit_authors_count"] == 3  # alice(6), bob(4), external@dev.org(2)
        assert part["active_human_multi_role_contributors_count"] == 4  # + rev_only (PR+review+issue)
        assert part["active_bot_contributors_count"] == 1

        # Activity totals reconcile cleanly
        assert part["activity_totals"]["commits"]["eligible_total"] == 24  # ignores future sha_future_alice
        assert part["activity_totals"]["commits"]["human_attributed"] == 12  # 6 + 4 + 2
        assert part["activity_totals"]["commits"]["bot_excluded"] == 10
        assert part["activity_totals"]["commits"]["unattributed"] == 2
        assert part["activity_totals"]["prs_authored"]["human_attributed"] == 2
        assert part["activity_totals"]["prs_authored"]["bot_excluded"] == 1
        assert part["activity_totals"]["prs_reviewed"]["human_attributed"] == 1
        assert part["activity_totals"]["issues_opened"]["human_attributed"] == 1

        # Identity hygiene & zero speculative merging
        assert hyg["speculative_name_merging"] is False
        assert hyg["unlinked_active_human_contributors_count"] == 1  # external@dev.org
        assert hyg["unattributed_commits_count"] == 2
        assert hyg["bot_commits_excluded_count"] == 10
        assert hyg["bot_commits_share"] == round(10 / 24, 4)

        # Concentration uses [2, 4, 6] human commit counts only (bot excluded, dormant dir excluded, stale 999 ignored!)
        assert conc["total_attributed_human_commits"] == 12
        assert conc["top1_share"] == 0.5  # Alice 6 / 12 (not 999!)

        # Tenure relative to ref_time: Alice first commit 45d ago (>30 established), Bob/Ext/rev_only <=30d (newcomers)
        assert ten["established_gt_30d_count"] == 1
        assert ten["newcomers_30d_count"] == 3
        assert any(r.rule_id == "RULE-CON-003" for r in res.rules)

        # Equivalent instant expressed with a non-UTC offset must normalize identically
        res_ist = calc.calculate(
            7010,
            in_memory_db,
            ref_time.astimezone(timezone(timedelta(hours=5, minutes=30))),
            snapshot_revision=1,
        )
        assert res_ist.metrics["participation"] == part
        assert res_ist.metrics["concentration"] == conc


class TestRetentionGuardAndCohortCalculations:
    """Verifies strict bounded/partial retention deferral and full-history synthetic cohort calculations."""

    @pytest.mark.parametrize(
        "mode,is_partial,cov_summary",
        [
            ("fast_snapshot", False, None),
            ("standard_diagnostic", False, None),
            ("full_history", True, None),
            ("full_history", False, {"commits": {"status": "truncated", "is_partial": True}}),
        ],
    )
    def test_bounded_and_partial_modes_defer_retention(self, in_memory_db, mode, is_partial, cov_summary):
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(Repository(id=7020, repo_name="con/bounded", owner="con", name="bounded", analytics_source_revision=1))
        in_memory_db.add(
            CollectionRun(
                repository_id=7020,
                mode=mode,
                status="completed",
                is_partial=is_partial,
                coverage_summary=cov_summary,
            )
        )
        # Seed multi-month commits across Aug..Nov -> MUST still defer retention under bounded/partial modes!
        for idx, m in enumerate([8, 9, 10, 11]):
            in_memory_db.add(
                Commit(
                    repository_id=7020,
                    sha=f"sha_bnd_{idx}",
                    author_id=6001 + (idx % 2),
                    author_login=f"dev_{idx % 2}",
                    author_date=datetime(2026, m, 5, 12, 0, 0, tzinfo=timezone.utc),
                )
            )
        in_memory_db.commit()

        res = calc.calculate(7020, in_memory_db, ref_time, snapshot_revision=1)
        ret = res.metrics["retention"]
        assert ret["status"] == "unavailable_requires_full_history_ingestion"
        assert ret["retention_cohorts"] is None
        assert res.coverage.details["subdomains"]["retention_cohorts"] == "unavailable_requires_full_history_ingestion"

    def test_complete_synthetic_history_monthly_retention_cohorts(self, in_memory_db):
        """Verify Plan v1.8 Section 5.1 monthly retention strictly across commits & PRs, excluding issues & reviews."""
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(Repository(id=7030, repo_name="con/full_ret", owner="con", name="full_ret", analytics_source_revision=1))
        in_memory_db.add(CollectionRun(repository_id=7030, mode="full_history", status="completed", is_partial=False))

        # Cohort 2026-08 (3 humans):
        # - Alice (7001): commit Aug, PR Sep, commit Nov -> active {2026-08, 2026-09, 2026-11}
        # - Bob (7002): PR Aug, issue+review ONLY in Sep (excluded!), commit Oct -> active {2026-08, 2026-10}
        # - Charlie (7003): commit Aug only -> active {2026-08}
        in_memory_db.add_all([
            Commit(repository_id=7030, sha="r_a1", author_id=7001, author_login="alice", author_date=datetime(2026, 8, 5, tzinfo=timezone.utc)),
            PullRequest(id=910, repository_id=7030, pr_number=1, title="A Sep PR", state="closed", author="alice", author_id=7001, created_at=datetime(2026, 9, 10, tzinfo=timezone.utc)),
            Commit(repository_id=7030, sha="r_a2", author_id=7001, author_login="alice", author_date=datetime(2026, 11, 2, tzinfo=timezone.utc)),
            # Future Dec commit after ref_time must be ignored
            Commit(repository_id=7030, sha="r_a_future", author_id=7001, author_login="alice", author_date=datetime(2026, 12, 1, tzinfo=timezone.utc)),

            PullRequest(id=911, repository_id=7030, pr_number=2, title="B Aug PR", state="open", author="bob", author_id=7002, created_at=datetime(2026, 8, 12, tzinfo=timezone.utc)),
            Commit(repository_id=7030, sha="r_b1", author_id=7002, author_login="bob", author_date=datetime(2026, 10, 14, tzinfo=timezone.utc)),

            Commit(repository_id=7030, sha="r_c1", author_id=7003, author_login="charlie", author_date=datetime(2026, 8, 20, tzinfo=timezone.utc)),

            # Bot across all months (must be excluded!)
            Commit(repository_id=7030, sha="r_bot1", author_id=7999, author_login="renovate[bot]", author_date=datetime(2026, 8, 1, tzinfo=timezone.utc)),
            Commit(repository_id=7030, sha="r_bot2", author_id=7999, author_login="renovate[bot]", author_date=datetime(2026, 9, 1, tzinfo=timezone.utc)),
        ])
        in_memory_db.flush()

        # Bob's Sep activity is strictly a PR review & an issue -> MUST be excluded from code retention!
        in_memory_db.add(PullRequestReview(pull_request_id=910, review_id=9901, author="bob", author_id=7002, state="COMMENTED", submitted_at=datetime(2026, 9, 11, tzinfo=timezone.utc)))
        in_memory_db.add(Issue(id=703001, repository_id=7030, repo_name="con/full_ret", issue_number=1, title="Sep Issue", state="open", author="bob", created_at=datetime(2026, 9, 12, tzinfo=timezone.utc)))

        # Cohort 2026-10 (1 newcomer human Dave 7004): commit Oct, commit Nov -> {2026-10, 2026-11}
        in_memory_db.add(Commit(repository_id=7030, sha="r_d1", author_id=7004, author_login="dave", author_date=datetime(2026, 10, 4, tzinfo=timezone.utc)))
        in_memory_db.add(Commit(repository_id=7030, sha="r_d2", author_id=7004, author_login="dave", author_date=datetime(2026, 11, 4, tzinfo=timezone.utc)))

        # Issue-only user Eve (commits=0, PRs=0) -> NOT a retention code cohort!
        in_memory_db.add(Issue(id=703002, repository_id=7030, repo_name="con/full_ret", issue_number=2, title="Issue Only", state="open", author="eve_issue_only", created_at=datetime(2026, 8, 15, tzinfo=timezone.utc)))
        in_memory_db.commit()

        res = calc.calculate(7030, in_memory_db, ref_time, snapshot_revision=1)
        ret = res.metrics["retention"]
        assert ret["status"] == "computed"
        cohorts = {c["cohort_month"]: c for c in ret["retention_cohorts"]}
        assert set(cohorts.keys()) == {"2026-08", "2026-10"}

        aug = cohorts["2026-08"]
        assert aug["cohort_size"] == 3  # Alice, Bob, Charlie (bot & eve excluded)
        assert aug["max_observable_offset_months"] == 3  # Aug -> Nov
        assert aug["retention_by_month"]["M+0"]["retention_rate"] == 1.0
        assert aug["retention_by_month"]["M+1"]["retention_rate"] == round(1 / 3, 4)  # Only Alice!
        assert aug["retention_by_month"]["M+2"]["retention_rate"] == round(1 / 3, 4)  # Only Bob!
        assert aug["retention_by_month"]["M+3"]["retention_rate"] == round(1 / 3, 4)  # Only Alice!

        oct_coh = cohorts["2026-10"]
        assert oct_coh["cohort_size"] == 1
        assert oct_coh["max_observable_offset_months"] == 1
        assert oct_coh["retention_by_month"]["M+1"]["retention_rate"] == 1.0

        # Overall M+1 retention = (1 + 1) / (3 + 1) = 0.50; Overall M+3 = 1 / 3 = 0.3333
        assert ret["summary"]["total_tracked_contributors"] == 4
        assert ret["summary"]["overall_m1_retention_rate"] == 0.5
        assert ret["summary"]["overall_m3_retention_rate"] == round(1 / 3, 4)
        assert any(r.rule_id == "RULE-CON-004" and "50.0%" in r.finding for r in res.rules)


class TestBenchmarkAndQueueIntegrationC3:
    """Verifies safe offline benchmark execution and atomic queue publication for C.3."""

    def test_benchmark_repositories_safe_handling(self):
        db_path = "data/opensource_lens.db"
        if not os.path.exists(db_path):
            pytest.skip(f"Database {db_path} not found.")

        engine = create_engine(f"sqlite:///{db_path}")
        calc = ContributorIntelligenceCalculator()
        ref_time = datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)

        with Session(engine) as session:
            repos = session.scalars(select(Repository)).all()
            assert len(repos) >= 4
            for repo in repos:
                res = calc.calculate(repo.id, session, ref_time, repo.analytics_source_revision)
                assert res.domain == "contributor_ecosystem"
                assert res.coverage.status == "insufficient"
                assert res.metrics["concentration"]["status"] == "empty"
                assert res.metrics["participation"]["active_human_commit_authors_count"] == 0
                assert res.metrics["participation"]["activity_totals"]["issues_opened"]["eligible_total"] > 0
                assert res.metrics["retention"]["status"] == "unavailable_requires_full_history_ingestion"
                assert res.metrics["retention"]["retention_cohorts"] is None

    def test_contributor_calculator_published_via_queue(self, in_memory_db):
        calc = ContributorIntelligenceCalculator()
        queue = AnalyticsJobQueue()
        ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

        in_memory_db.add(Repository(id=7040, repo_name="queue/c3", owner="queue", name="c3", analytics_source_revision=4, is_active=True))
        in_memory_db.add(CollectionRun(repository_id=7040, mode="standard_diagnostic", status="completed", is_partial=False))
        for u in range(6):
            for c in range(u + 1):
                in_memory_db.add(
                    Commit(
                        repository_id=7040,
                        sha=f"sha_q3_{u}_{c}",
                        author_id=8000 + u,
                        author_login=f"qdev_{u}",
                        author_date=ref_time - timedelta(days=5 + c),
                    )
                )
        in_memory_db.commit()

        queue.enqueue_analysis(repo_id=7040, db=in_memory_db)
        claimed_run, attempt = queue.claim_job(worker_id="c3_worker", db=in_memory_db)

        res = calc.calculate(7040, in_memory_db, ref_time, snapshot_revision=claimed_run.requested_source_revision)
        json.dumps(res.metrics)

        published = queue.publish_and_finalize(
            run_id=claimed_run.id,
            attempt_id=attempt.id,
            worker_id="c3_worker",
            expected_lease_version=claimed_run.lease_version,
            attempt_number=attempt.attempt_number,
            snapshot_revision=claimed_run.requested_source_revision,
            actual_methodology=CURRENT_METHODOLOGY_VERSION,
            reference_time=ref_time,
            metrics_payload={"contributor_ecosystem": res.metrics},
            coverage_summary={"contributor_ecosystem": res.coverage.model_dump()},
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
        payload = claimed_run.metrics_payload["contributor_ecosystem"]
        assert payload["concentration"]["active_human_commit_authors_count"] == 6
        assert payload["retention"]["status"] == "unavailable_requires_full_history_ingestion"
        assert set(claimed_run.coverage_summary["contributor_ecosystem"]["details"]["subdomains"].keys()) == {
            "concentration",
            "participation",
            "retention_cohorts",
        }
