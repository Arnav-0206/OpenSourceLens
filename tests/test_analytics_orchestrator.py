"""Comprehensive Unit and Integration Tests for Phase C.7 Analytics Orchestration & Publication.

Covers all 12 required Phase C.7 scenarios using isolated test databases:
1. Successful coordinated execution of all six domains (C.2–C.6).
2. Shared snapshot transaction (SQLite WAL) and identical frozen analysis_reference_time across domains.
3. Locked 5-key AnalysisPayload Pydantic v2 construction and validation (zero last_synced_at leakage).
4. Preservation of Release Evolution metrics, coverage, confidence penalties, and rules.
5. Aggregate coverage and confidence computation across all six original CalculationResults.
6. Explicit handling of missing/unavailable domain results without fabricating zeros.
7. Domain calculation failure handling (retryable retries, non-retryable tagging, exhausted retries).
8. Stale source revision prevention (superseding completed run + concurrent ingestion replacement).
9. Optimistic lease-version fencing rejecting stale/ghost workers.
10. Deterministic rule consolidation and reproducible serialized outputs.
11. Full-history vs. bounded standard_diagnostic vs. fast_snapshot representativeness preservation.
12. Full compatibility with AnalyticsJobQueue, AnalysisRunRead, and idempotent re-enqueue suppression.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, List, Optional
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.exceptions import (
    GhostWorkerRejectedException,
    InvalidParameterException,
)
from backend.app.database.base import Base
from backend.app.database.models import (
    AnalysisRun,
    AnalysisRunAttempt,
    CollectionRun,
    Commit,
    Contributor,
    Issue,
    IssueComment,
    IssueLabel,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
    Release,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    AnalysisPayload,
    AnalysisRunRead,
    CURRENT_METHODOLOGY_VERSION,
    ConfidenceSummary,
    MetricCoverageItem,
)
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.orchestrator import (
    ALL_ANALYTICS_DOMAINS,
    AnalyticsOrchestrator,
    AnalyticsService,
    NonRetryableAnalyticsError,
)
from backend.app.services.analytics_queue import AnalyticsJobQueue


@pytest.fixture
def in_memory_db():
    """Isolated in-memory SQLite database session for unit/orchestration tests."""
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


def _seed_rich_repository(
    db: Session,
    repo_id: int = 9001,
    *,
    collection_mode: str = "full_history",
    is_partial: bool = False,
    rev_status: str = "completed",
    inline_status: str = "completed",
    stars_count: int = 1200,
    small_release_sample: bool = False,
) -> Repository:
    """Populate a deterministic multi-domain synthetic repository across 14 calendar months."""
    ref_anchor = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)
    repo = Repository(
        id=repo_id,
        repo_name=f"orch/repo_{repo_id}",
        owner="orch",
        name=f"repo_{repo_id}",
        stars_count=stars_count,
        forks_count=150,
        primary_language="Python",
        is_archived=False,
        is_fork=False,
        open_issues_count=20,
        last_synced_at=ref_anchor,
        analytics_source_revision=3,
        is_active=True,
    )
    db.add(repo)

    col_run = CollectionRun(
        id=repo_id * 10 + 1,
        repository_id=repo_id,
        mode=collection_mode,
        status="completed",
        is_partial=is_partial,
        coverage_summary={
            "commits": "completed",
            "contributors": "completed",
            "pull_requests": "completed",
            "pr_reviews": rev_status,
            "pr_inline_comments": inline_status,
            "issues": "completed",
            "issue_comments": "completed",
            "releases": "completed",
        },
        started_at=ref_anchor - timedelta(hours=2),
        completed_at=ref_anchor - timedelta(hours=1),
    )
    db.add(col_run)

    # 6 human contributors across commits/PRs/reviews
    for c_i in range(6):
        db.add(
            Contributor(
                id=repo_id * 100 + c_i,
                repository_id=repo_id,
                github_id=7000 + c_i,
                login=f"dev_{c_i}",
                git_email=f"dev_{c_i}@orch.dev",
                is_bot=False,
                total_commits=10,
            )
        )

    # 14 calendar months with low-AR(1) variation so all 3 cross-domain pairs are testable
    months = [(2025 + (m // 12), (m % 12) + 1) for m in range(14)]
    c_pattern = [6, 11, 9, 2, 3, 10, 13, 7, 1, 4, 12, 5, 8, 14]
    pr_pk_seq = repo_id * 10000
    iss_pk_seq = repo_id * 10000 + 5000
    rev_id_seq = repo_id * 100000

    for m_idx, (yr, mo) in enumerate(months):
        m_dt = datetime(yr, mo, 8, 12, 0, tzinfo=timezone.utc)
        n_val = c_pattern[m_idx]

        # Commits (spread across the 6 human authors with complete additions/deletions & committer_date)
        for c_i in range(n_val):
            auth_idx = (m_idx + c_i) % 6
            c_dt = m_dt + timedelta(hours=c_i * 2)
            db.add(
                Commit(
                    repository_id=repo_id,
                    sha=f"sha_{repo_id}_{m_idx}_{c_i}",
                    author_id=7000 + auth_idx,
                    author_login=f"dev_{auth_idx}",
                    author_email=f"dev_{auth_idx}@orch.dev",
                    author_name=f"Dev {auth_idx}",
                    author_date=c_dt,
                    committer_date=c_dt + timedelta(hours=1),
                    additions=40 * (c_i + 1),
                    deletions=10,
                    total_changes=40 * (c_i + 1) + 10,
                )
            )

        # Issues (closed and open across categories & labels)
        for i_i in range(n_val):
            is_closed = i_i % 2 == 0
            iss_pk_seq += 1
            db.add(
                Issue(
                    id=iss_pk_seq,
                    repository_id=repo_id,
                    repo_name=repo.repo_name,
                    issue_number=iss_pk_seq % 10000,
                    title=f"Issue {m_idx}_{i_i}",
                    state="closed" if is_closed else "open",
                    author=f"dev_{i_i % 6}",
                    created_at=m_dt + timedelta(hours=i_i),
                    closed_at=(m_dt + timedelta(days=3)) if is_closed else None,
                    comments_count=2,
                    canonical_label="bug" if i_i % 3 == 0 else "enhancement",
                    is_unlabeled=False,
                )
            )

        # Pull requests + formal reviews + inline review comments
        slow_h = float(n_val * 3)
        for p_i in range(2):
            pr_pk_seq += 1
            is_abandoned = (n_val >= 10) and (p_i == 0)
            db.add(
                PullRequest(
                    id=pr_pk_seq,
                    repository_id=repo_id,
                    pr_number=pr_pk_seq % 10000,
                    title=f"PR {m_idx}_{p_i}",
                    state="closed",
                    author="dev_0",
                    author_id=7000,
                    created_at=m_dt,
                    closed_at=m_dt + timedelta(days=1),
                    merged_at=None if is_abandoned else (m_dt + timedelta(days=1)),
                    additions=35 + p_i * 50,
                    deletions=10,
                    changed_files=2,
                )
            )
            rev_id_seq += 1
            db.add(
                PullRequestReview(
                    pull_request_id=pr_pk_seq,
                    review_id=rev_id_seq,
                    author="dev_1",
                    author_id=7001,
                    state="APPROVED",
                    submitted_at=m_dt + timedelta(hours=slow_h),
                )
            )
            if inline_status == "completed":
                db.add(
                    PullRequestReviewComment(
                        pull_request_id=pr_pk_seq,
                        comment_id=rev_id_seq * 10 + 1,
                        author="dev_2",
                        author_id=7002,
                        path="src/app.py",
                        line=12,
                        created_at=m_dt + timedelta(hours=slow_h + 1.0),
                    )
                )

    # Releases (either 6 releases for sufficient SemVer/cadence, or 3 releases for small-sample penalty testing)
    rel_offsets = [120, 60, 15] if small_release_sample else [240, 210, 150, 90, 60, 15]
    rel_tags = ["v1.0.0", "v1.1.0", "v1.2.0"] if small_release_sample else [
        "v1.0.0", "v1.1.0", "v1.2.0", "v1.2.1", "v2.0.0-rc.1", "v2.0.0"
    ]
    for idx, (offset, tag) in enumerate(zip(rel_offsets, rel_tags)):
        db.add(
            Release(
                repository_id=repo_id,
                release_id=repo_id * 1000 + idx,
                tag_name=tag,
                name=tag,
                published_at=datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc) - timedelta(days=offset),
                is_prerelease=("-rc" in tag),
            )
        )

    db.commit()
    return repo


class TestOrchestrationCoreAndPublication:
    """Tests 1, 3, 4, 5, 10, 12: 6-domain coordination, payload validation, release preservation, determinism."""

    def test_successful_six_domain_orchestration_and_payload_validation(self, in_memory_db):
        repo = _seed_rich_repository(in_memory_db, repo_id=9001, collection_mode="full_history")
        assert AnalyticsService is AnalyticsOrchestrator
        orchestrator = AnalyticsOrchestrator()

        exec_res = orchestrator.enqueue_and_process(repository_id=repo.id, db=in_memory_db)
        assert exec_res.status == "completed"
        assert exec_res.published is True
        assert exec_res.snapshot_revision == 3
        assert exec_res.requested_source_revision == 3
        assert exec_res.bundle is not None

        # Validate persisted AnalysisRun & AnalysisRunAttempt via queue + Pydantic contract
        persisted = in_memory_db.get(AnalysisRun, exec_res.run_id)
        assert persisted is not None
        assert persisted.status == "completed"
        assert persisted.analytics_source_revision == 3
        assert persisted.methodology_version == CURRENT_METHODOLOGY_VERSION
        assert persisted.analysis_reference_time is not None

        # Verify strictly 5 top-level keys matching AnalysisPayload (no unsupported 6th top-level key!)
        expected_top_keys = {
            "development_activity",
            "contributor_ecosystem",
            "pull_request_intelligence",
            "issue_intelligence",
            "repository_synthesis",
        }
        assert set(persisted.metrics_payload.keys()) == expected_top_keys
        assert "release_evolution" not in persisted.metrics_payload
        assert "release_evolution" in persisted.metrics_payload["repository_synthesis"]
        assert "last_synced_at" not in json.dumps(persisted.metrics_payload)

        # Schema validations succeed directly on stored payloads
        validated_payload = AnalysisPayload(**persisted.metrics_payload)
        assert validated_payload.repository_synthesis["chi_deferred"] is True
        assert "composite_health_score" not in validated_payload.repository_synthesis
        assert validated_payload.repository_synthesis["orchestration_metadata"]["methodology_version"] == CURRENT_METHODOLOGY_VERSION
        assert validated_payload.repository_synthesis["orchestration_metadata"]["snapshot_revision"] == 3

        # Coverage & confidence aggregate across ALL 6 domains (including release_evolution)
        for domain in ALL_ANALYTICS_DOMAINS:
            assert domain in persisted.coverage_summary
            assert domain in persisted.confidence_summary["domain_breakdown"]
        assert persisted.coverage_summary["summary"]["total_domains"] == 6
        assert persisted.confidence_summary["total_domains_evaluated"] == 6
        validated_conf = ConfidenceSummary(**persisted.confidence_summary)
        assert 0.0 <= validated_conf.score <= 1.0
        assert validated_conf.band in {"high", "medium", "low", "unreliable"}

        # Rule consolidation contains no duplicates across domains
        triples = [(r["rule_id"], r["domain"], r["finding"]) for r in persisted.rule_interpretations]
        assert len(triples) == len(set(triples))

        # AnalysisRunRead schema round-trip compatibility
        run_read = AnalysisRunRead.model_validate(persisted)
        assert run_read.id == persisted.id
        assert len(run_read.attempts) == 1
        assert run_read.attempts[0].outcome == "completed"

        # Idempotent no-op suppression when re-enqueued without force
        second_res = orchestrator.enqueue_and_process(repository_id=repo.id, db=in_memory_db, force=False)
        assert second_res.status == "already_completed"
        assert second_res.published is False
        assert second_res.run_id == persisted.id

    def test_release_evolution_penalties_coverage_and_rules_preserved(self, in_memory_db):
        # Seed 3 releases under standard_diagnostic -> PENALTY-SMALL-RELEASE-SAMPLE (0.2) + PENALTY-BOUNDED-RELEASE-SCOPE (0.1)
        repo = _seed_rich_repository(
            in_memory_db,
            repo_id=9002,
            collection_mode="standard_diagnostic",
            small_release_sample=True,
        )
        orchestrator = AnalyticsOrchestrator()
        res = orchestrator.enqueue_and_process(repository_id=repo.id, db=in_memory_db)
        assert res.published is True

        persisted = in_memory_db.get(AnalysisRun, res.run_id)
        rel_embedded = persisted.metrics_payload["repository_synthesis"]["release_evolution"]
        assert rel_embedded["cadence"]["total_releases"] == 3
        assert rel_embedded["coverage"]["status"] == "degraded"
        assert rel_embedded["confidence"]["confidence_penalty"] == 0.3

        # Release coverage and release penalties must be preserved explicitly across coverage_summary & confidence_summary
        assert persisted.coverage_summary["release_evolution"]["status"] == "degraded"
        rel_penalties = [p for p in persisted.confidence_summary["penalties"] if p.get("domain") == "release_evolution"]
        rel_penalty_rules = {p["rule"] for p in rel_penalties}
        assert "PENALTY-SMALL-RELEASE-SAMPLE" in rel_penalty_rules
        assert "PENALTY-BOUNDED-RELEASE-SCOPE" in rel_penalty_rules
        assert persisted.confidence_summary["domain_breakdown"]["release_evolution"]["confidence_penalty"] == 0.3

        # Release rules preserved alongside C.2–C.5 and SYN rules
        rel_rules = [r for r in persisted.rule_interpretations if r["domain"] == "release_evolution"]
        assert any(r["rule_id"] == "RULE-REL-004" for r in rel_rules)

    def test_determinism_and_bounded_vs_full_history_representativeness(self, in_memory_db):
        _seed_rich_repository(in_memory_db, repo_id=9003, collection_mode="full_history")
        _seed_rich_repository(in_memory_db, repo_id=9004, collection_mode="standard_diagnostic")
        _seed_rich_repository(
            in_memory_db,
            repo_id=9005,
            collection_mode="fast_snapshot",
            inline_status="skipped",
        )

        orchestrator = AnalyticsOrchestrator()
        fixed_ref = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)

        # Deterministic repetition on identical snapshot & reference time
        b1 = orchestrator.run_snapshot_analysis(9003, in_memory_db, analysis_reference_time=fixed_ref, snapshot_revision=3)
        b2 = orchestrator.run_snapshot_analysis(9003, in_memory_db, analysis_reference_time=fixed_ref, snapshot_revision=3)
        p1 = json.loads(json.dumps(b1.metrics_payload))
        p2 = json.loads(json.dumps(b2.metrics_payload))
        p1["repository_synthesis"]["orchestration_metadata"]["execution_time_ms"] = 0.0
        p2["repository_synthesis"]["orchestration_metadata"]["execution_time_ms"] = 0.0
        assert p1 == p2
        assert b1.confidence_summary == b2.confidence_summary
        assert b1.rule_interpretations == b2.rule_interpretations

        # Full history vs. standard_diagnostic vs. fast_snapshot representativeness
        res_full = orchestrator.enqueue_and_process(9003, in_memory_db)
        res_diag = orchestrator.enqueue_and_process(9004, in_memory_db)
        res_fast = orchestrator.enqueue_and_process(9005, in_memory_db)

        run_full = in_memory_db.get(AnalysisRun, res_full.run_id)
        run_diag = in_memory_db.get(AnalysisRun, res_diag.run_id)
        run_fast = in_memory_db.get(AnalysisRun, res_fast.run_id)

        assert run_full.coverage_summary["summary"]["representativeness"] == "full_history"
        assert run_full.coverage_summary["summary"]["is_full_history"] is True
        assert run_full.coverage_summary["summary"]["is_bounded_or_partial"] is False
        assert run_full.confidence_summary["score"] > run_diag.confidence_summary["score"]

        assert run_diag.coverage_summary["summary"]["representativeness"] == "bounded_diagnostic"
        assert run_diag.coverage_summary["summary"]["is_full_history"] is False
        assert run_diag.coverage_summary["summary"]["is_bounded_or_partial"] is True
        assert run_diag.metrics_payload["repository_synthesis"]["cross_domain_correlations"]["pairs"][1]["collection_coverage"]["coverage_status"] == "bounded_mode_completed"

        assert run_fast.coverage_summary["summary"]["representativeness"] == "fast_snapshot_bounded"
        fast_p2 = next(
            p for p in run_fast.metrics_payload["repository_synthesis"]["cross_domain_correlations"]["pairs"]
            if p["pair_id"] == "PAIR-02"
        )
        assert fast_p2["status"] == "incomplete_review_coverage"
        assert fast_p2["p_value"] is None


class TestSnapshotConsistencyAndIsolation:
    """Test 2: Identical snapshot & frozen ref_time under SQLite WAL concurrent writes."""

    def test_all_six_domains_share_identical_snapshot_and_frozen_ref_time(self, tmp_path):
        db_file = tmp_path / "orchestrator_wal.db"
        db_url = f"sqlite:///{db_file}"

        setup_engine = create_engine(db_url, connect_args={"check_same_thread": False})
        with setup_engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            Base.metadata.create_all(conn)
            conn.commit()
        setup_engine.dispose()

        writer_engine = create_engine(db_url, connect_args={"check_same_thread": False})
        WriterSession = sessionmaker(autocommit=False, autoflush=False, bind=writer_engine)
        with WriterSession() as wdb:
            _seed_rich_repository(wdb, repo_id=9101, collection_mode="full_history")

        reader_engine = create_engine(
            db_url,
            isolation_level=None,
            connect_args={"check_same_thread": False},
        )
        ReaderSession = sessionmaker(autocommit=False, autoflush=False, bind=reader_engine)

        observed_calls: List[Dict[str, Any]] = []
        writer_fired = False

        class SnapshotSpyCalculator(BaseMetricCalculator):
            def __init__(self, inner: BaseMetricCalculator, mutate_after: bool = False) -> None:
                self._inner = inner
                self._mutate_after = mutate_after

            @property
            def domain_name(self) -> str:
                return self._inner.domain_name

            def calculate(
                self,
                repository_id: int,
                db: Session,
                analysis_reference_time: datetime,
                snapshot_revision: int,
                **kwargs: Any,
            ) -> CalculationResult:
                nonlocal writer_fired
                observed_calls.append(
                    {
                        "domain": self._inner.domain_name,
                        "snapshot_revision": snapshot_revision,
                        "reference_time": analysis_reference_time,
                        "db_id": id(db),
                    }
                )
                res = self._inner.calculate(
                    repository_id=repository_id,
                    db=db,
                    analysis_reference_time=analysis_reference_time,
                    snapshot_revision=snapshot_revision,
                    **kwargs,
                )
                if self._mutate_after and not writer_fired:
                    writer_fired = True
                    # Concurrent ingestion commits revision 4 and an extra commit + issue WHILE snapshot is open!
                    with WriterSession() as conc_db:
                        r_row = conc_db.get(Repository, 9101)
                        r_row.analytics_source_revision = 4
                        conc_db.add(
                            Commit(
                                repository_id=9101,
                                sha="concurrent_writer_sha_rev4",
                                author_id=7000,
                                author_login="dev_0",
                                author_date=datetime(2026, 3, 19, 0, 0, 0, tzinfo=timezone.utc),
                                additions=999,
                                deletions=1,
                            )
                        )
                        conc_db.add(
                            Issue(
                                id=999999,
                                repository_id=9101,
                                repo_name="orch/repo_9101",
                                issue_number=999999,
                                title="Concurrent Phantom Issue",
                                state="open",
                                author="dev_0",
                                created_at=datetime(2026, 3, 19, 0, 0, 0, tzinfo=timezone.utc),
                            )
                        )
                        conc_db.commit()
                return res

        from backend.app.services.analytics import (
            ContributorIntelligenceCalculator,
            CrossDomainSynthesisCalculator,
            DevelopmentActivityCalculator,
            IssueIntelligenceCalculator,
            PullRequestIntelligenceCalculator,
            ReleaseEvolutionCalculator,
        )

        spied_calculators = {
            "development_activity": SnapshotSpyCalculator(DevelopmentActivityCalculator(), mutate_after=True),
            "contributor_ecosystem": SnapshotSpyCalculator(ContributorIntelligenceCalculator()),
            "pull_request_intelligence": SnapshotSpyCalculator(PullRequestIntelligenceCalculator()),
            "issue_intelligence": SnapshotSpyCalculator(IssueIntelligenceCalculator()),
            "release_evolution": SnapshotSpyCalculator(ReleaseEvolutionCalculator()),
            "repository_synthesis": SnapshotSpyCalculator(CrossDomainSynthesisCalculator()),
        }

        try:
            with ReaderSession() as rdb:
                orchestrator = AnalyticsOrchestrator(calculators=spied_calculators)
                exec_res = orchestrator.enqueue_and_process(repository_id=9101, db=rdb, worker_id="wal_worker")

                assert len(observed_calls) == 6
                assert [c["domain"] for c in observed_calls] == list(ALL_ANALYTICS_DOMAINS)
                # Every domain observed the exact same pinned snapshot revision (3), ref_time, and db session!
                assert all(c["snapshot_revision"] == 3 for c in observed_calls)
                ref_times = {c["reference_time"] for c in observed_calls}
                assert len(ref_times) == 1
                assert len({c["db_id"] for c in observed_calls}) == 1

                # Phantom issue #999999 committed at rev 4 was NOT visible to issue_intelligence inside WAL snapshot!
                persisted = rdb.get(AnalysisRun, exec_res.run_id)
                assert persisted.analytics_source_revision == 3
                assert exec_res.status == "completed_stale_replacement_enqueued"
                assert exec_res.stale_revision_detected is True
                assert exec_res.replacement_run_enqueued is True
                # Verify phantom commit/issue were excluded from snapshot aggregates while rev 4 is queued
                issue_count_in_payload = persisted.coverage_summary["issue_intelligence"]["sample_size"]
                assert issue_count_in_payload == sum([6, 11, 9, 2, 3, 10, 13, 7, 1, 4, 12, 5, 8, 14])
        finally:
            reader_engine.dispose()
            writer_engine.dispose()


class TestUnavailableDomainsFailuresRetriesFencing:
    """Tests 6, 7, 8, 9: Missing/unavailable domains, retries, non-retryable, stale revisions, lease fencing."""

    def test_missing_and_unavailable_domain_results_preserve_unavailable_states(self, in_memory_db):
        _seed_rich_repository(in_memory_db, repo_id=9201, collection_mode="full_history")

        class _NoneCalculator(BaseMetricCalculator):
            def __init__(self, dom: str) -> None:
                self._dom = dom

            @property
            def domain_name(self) -> str:
                return self._dom

            def calculate(self, repository_id, db, analysis_reference_time, snapshot_revision):
                return None  # type: ignore[return-value]

        from backend.app.services.analytics import (
            ContributorIntelligenceCalculator,
            CrossDomainSynthesisCalculator,
            DevelopmentActivityCalculator,
            IssueIntelligenceCalculator,
        )

        # Omit pull_request_intelligence (None config) and return None from release_evolution calculator
        orchestrator = AnalyticsOrchestrator(
            calculators={
                "development_activity": DevelopmentActivityCalculator(),
                "contributor_ecosystem": ContributorIntelligenceCalculator(),
                "pull_request_intelligence": None,
                "issue_intelligence": IssueIntelligenceCalculator(),
                "release_evolution": _NoneCalculator("release_evolution"),
                "repository_synthesis": CrossDomainSynthesisCalculator(),
            }
        )

        res = orchestrator.enqueue_and_process(repository_id=9201, db=in_memory_db)
        assert res.published is True
        persisted = in_memory_db.get(AnalysisRun, res.run_id)

        assert persisted.metrics_payload["pull_request_intelligence"]["status"] == "unavailable"
        assert persisted.metrics_payload["pull_request_intelligence"]["available"] is False
        assert persisted.metrics_payload["repository_synthesis"]["release_evolution"]["cadence"]["status"] == "unavailable"
        assert persisted.coverage_summary["summary"]["overall_status"] == "degraded"
        assert set(persisted.coverage_summary["summary"]["unavailable_domains"]) == {
            "pull_request_intelligence",
            "release_evolution",
        }
        assert any(p["rule"] == "PENALTY-UNAVAILABLE-RELEASE_EVOLUTION" for p in persisted.confidence_summary["penalties"])
        assert any(r["rule_id"] == "RULE-ORCH-UNAVAILABLE" for r in persisted.rule_interpretations)

    def test_retryable_failure_recovery_and_exhausted_retries(self, in_memory_db):
        _seed_rich_repository(in_memory_db, repo_id=9202, collection_mode="full_history")
        from backend.app.services.analytics import (
            ContributorIntelligenceCalculator,
            CrossDomainSynthesisCalculator,
            DevelopmentActivityCalculator,
            IssueIntelligenceCalculator,
            PullRequestIntelligenceCalculator,
            ReleaseEvolutionCalculator,
        )

        class _FlakyIssueCalculator(BaseMetricCalculator):
            def __init__(self, fail_times: int) -> None:
                self._fail_times = fail_times
                self.calls = 0
                self._real = IssueIntelligenceCalculator()

            @property
            def domain_name(self) -> str:
                return "issue_intelligence"

            def calculate(self, repository_id, db, analysis_reference_time, snapshot_revision):
                self.calls += 1
                if self.calls <= self._fail_times:
                    raise RuntimeError(f"Transient failure #{self.calls}")
                return self._real.calculate(repository_id, db, analysis_reference_time, snapshot_revision)

        # 1. Fail once on attempt #1 -> schedules retry -> succeeds on attempt #2!
        flaky = _FlakyIssueCalculator(fail_times=1)
        orch = AnalyticsOrchestrator(
            calculators={
                "development_activity": DevelopmentActivityCalculator(),
                "contributor_ecosystem": ContributorIntelligenceCalculator(),
                "pull_request_intelligence": PullRequestIntelligenceCalculator(),
                "issue_intelligence": flaky,
                "release_evolution": ReleaseEvolutionCalculator(),
                "repository_synthesis": CrossDomainSynthesisCalculator(),
            }
        )
        res1 = orch.enqueue_and_process(repository_id=9202, db=in_memory_db, worker_id="w_retry_1")
        assert res1.status == "retry_scheduled"
        assert res1.published is False
        assert res1.failure_category == "retryable"
        run = in_memory_db.get(AnalysisRun, res1.run_id)
        assert run.status == "pending"
        assert run.retry_count == 1
        assert run.metrics_payload is None

        # Attempt #2 succeeds
        res2 = orch.process_next_job(db=in_memory_db, worker_id="w_retry_2")
        assert res2 is not None and res2.status == "completed" and res2.published is True
        in_memory_db.refresh(run)
        assert run.status == "completed"
        assert len(run.attempts) == 2
        assert [a.outcome for a in sorted(run.attempts, key=lambda x: x.attempt_number)] == ["failed", "completed"]

        # 2. Exhausted retries (always failing calculator across max_retries=3 => 4 attempts total -> failed!)
        _seed_rich_repository(in_memory_db, repo_id=9203, collection_mode="full_history")
        perm_fail = _FlakyIssueCalculator(fail_times=99)
        orch_fail = AnalyticsOrchestrator(
            calculators={
                "development_activity": DevelopmentActivityCalculator(),
                "contributor_ecosystem": ContributorIntelligenceCalculator(),
                "pull_request_intelligence": PullRequestIntelligenceCalculator(),
                "issue_intelligence": perm_fail,
                "release_evolution": ReleaseEvolutionCalculator(),
                "repository_synthesis": CrossDomainSynthesisCalculator(),
            }
        )
        first_fail = orch_fail.enqueue_and_process(repository_id=9203, db=in_memory_db, worker_id="w_perm_1", max_retries=3)
        assert first_fail.status == "retry_scheduled"
        for i in range(2, 5):
            step = orch_fail.process_next_job(db=in_memory_db, worker_id=f"w_perm_{i}")
            assert step is not None
            if i < 4:
                assert step.status == "retry_scheduled"
            else:
                assert step.status == "failed"
                assert step.failure_category == "exhausted_retries"

        assert orch_fail.process_next_job(db=in_memory_db, worker_id="w_perm_5") is None
        perm_run = in_memory_db.get(AnalysisRun, first_fail.run_id)
        assert perm_run.status == "failed"
        assert perm_run.retry_count == 4
        assert perm_run.metrics_payload is None

    def test_non_retryable_failure_classification_and_no_silent_completion(self, in_memory_db):
        _seed_rich_repository(in_memory_db, repo_id=9204, collection_mode="full_history")
        from backend.app.services.analytics import (
            ContributorIntelligenceCalculator,
            CrossDomainSynthesisCalculator,
            DevelopmentActivityCalculator,
            IssueIntelligenceCalculator,
            PullRequestIntelligenceCalculator,
            ReleaseEvolutionCalculator,
        )

        class _InvalidContractCalc(BaseMetricCalculator):
            @property
            def domain_name(self) -> str:
                return "issue_intelligence"

            def calculate(self, repository_id, db, analysis_reference_time, snapshot_revision):
                raise NonRetryableAnalyticsError("Corrupt schema invariant in test")

        orch = AnalyticsOrchestrator(
            calculators={
                "development_activity": DevelopmentActivityCalculator(),
                "contributor_ecosystem": ContributorIntelligenceCalculator(),
                "pull_request_intelligence": PullRequestIntelligenceCalculator(),
                "issue_intelligence": _InvalidContractCalc(),
                "release_evolution": ReleaseEvolutionCalculator(),
                "repository_synthesis": CrossDomainSynthesisCalculator(),
            }
        )
        res = orch.enqueue_and_process(repository_id=9204, db=in_memory_db, max_retries=0)
        assert res.status == "failed"
        assert res.published is False
        assert res.failure_category == "non_retryable"
        assert "[NON_RETRYABLE]" in (res.error_message or "")
        run = in_memory_db.get(AnalysisRun, res.run_id)
        assert run.status == "failed"
        assert run.metrics_payload is None

    def test_stale_revision_prevents_publishing_over_newer_completed_analysis(self, in_memory_db):
        repo = _seed_rich_repository(in_memory_db, repo_id=9205, collection_mode="full_history")
        queue = AnalyticsJobQueue()

        # Worker claims Run 1 at rev 3
        queue.enqueue_analysis(repo_id=repo.id, db=in_memory_db)
        run1, att1 = queue.claim_job(worker_id="stale_worker", db=in_memory_db)

        # Simulate newer revision 4 already having a completed authoritative AnalysisRun
        repo.analytics_source_revision = 4
        newer_run = AnalysisRun(
            repository_id=repo.id,
            status="completed",
            requested_source_revision=4,
            analytics_source_revision=4,
            requested_methodology_version=CURRENT_METHODOLOGY_VERSION,
            methodology_version=CURRENT_METHODOLOGY_VERSION,
            completed_at=datetime.now(timezone.utc),
            metrics_payload={"development_activity": {"marker": "rev4"}},
        )
        in_memory_db.add(newer_run)
        in_memory_db.commit()

        # Run 1 tries to calculate snapshot rev 3 (simulated via monkeypatched open_analysis_snapshot on queue)
        from contextlib import contextmanager

        @contextmanager
        def _fake_old_snapshot(db, repository_id):
            yield 3, datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)

        queue.open_analysis_snapshot = _fake_old_snapshot  # type: ignore[method-assign]
        orch = AnalyticsOrchestrator(job_queue=queue)
        res = orch.process_claimed_run(run1, att1, "stale_worker", in_memory_db)

        assert res.status == "stale_blocked"
        assert res.published is False
        assert res.stale_revision_detected is True
        assert res.failure_category == "stale_revision"

        in_memory_db.refresh(run1)
        in_memory_db.refresh(att1)
        in_memory_db.refresh(newer_run)
        assert run1.status == "cancelled"
        assert run1.metrics_payload is None
        assert att1.outcome == "interrupted"
        assert newer_run.status == "completed"
        assert newer_run.analytics_source_revision == 4

    def test_lease_fencing_blocks_ghost_worker_publication(self, in_memory_db):
        repo = _seed_rich_repository(in_memory_db, repo_id=9206, collection_mode="full_history")
        queue = AnalyticsJobQueue()
        queue.enqueue_analysis(repo_id=repo.id, db=in_memory_db)

        run, attempt = queue.claim_job(worker_id="ghost_worker", db=in_memory_db)
        assert run.lease_version == 1

        # Simulate heartbeat/competing claim advancing lease_version cleanly from 1 -> 2 before worker publishes
        stale_run_snapshot_lease = run.lease_version
        in_memory_db.expunge(run)
        in_memory_db.expunge(attempt)
        run.lease_version = stale_run_snapshot_lease
        queue.renew_lease(run_id=run.id, worker_id="ghost_worker", expected_lease_version=1, db=in_memory_db)

        orch = AnalyticsOrchestrator(job_queue=queue)
        with pytest.raises(GhostWorkerRejectedException):
            orch.process_claimed_run(run=run, attempt=attempt, worker_id="ghost_worker", db=in_memory_db)

        fresh_run = in_memory_db.get(AnalysisRun, run.id)
        assert fresh_run.status == "running"
        assert fresh_run.lease_version == 2
        assert fresh_run.metrics_payload is None

    def test_payload_validation_failure_prevents_publication(self, in_memory_db):
        _seed_rich_repository(in_memory_db, repo_id=9207, collection_mode="full_history")
        from backend.app.services.analytics import (
            ContributorIntelligenceCalculator,
            CrossDomainSynthesisCalculator,
            IssueIntelligenceCalculator,
            PullRequestIntelligenceCalculator,
            ReleaseEvolutionCalculator,
        )

        class _MalformedPayloadDevCalculator(BaseMetricCalculator):
            @property
            def domain_name(self) -> str:
                return "development_activity"

            def calculate(self, repository_id, db, analysis_reference_time, snapshot_revision):
                # Non-string metric key passes CalculationResult init/dict conversion
                # but fails Pydantic v2 AnalysisPayload schema validation (Dict[str, Any]).
                return CalculationResult(
                    domain="development_activity",
                    metrics={123: "invalid_non_string_key"},
                    coverage=MetricCoverageItem(
                        domain="development_activity",
                        status="sufficient",
                        sample_size=10,
                        min_sample_required=5,
                    ),
                    confidence_penalty=0.0,
                    rules=[],
                    details={},
                )

        orchestrator = AnalyticsOrchestrator(
            calculators={
                "development_activity": _MalformedPayloadDevCalculator(),
                "contributor_ecosystem": ContributorIntelligenceCalculator(),
                "pull_request_intelligence": PullRequestIntelligenceCalculator(),
                "issue_intelligence": IssueIntelligenceCalculator(),
                "release_evolution": ReleaseEvolutionCalculator(),
                "repository_synthesis": CrossDomainSynthesisCalculator(),
            }
        )

        res = orchestrator.enqueue_and_process(
            repository_id=9207,
            db=in_memory_db,
            worker_id="w_validation_fail",
            max_retries=0,
        )

        assert res.status == "failed"
        assert res.published is False
        assert res.failure_category == "non_retryable"
        assert res.error_message is not None
        assert res.error_message.startswith("[NON_RETRYABLE] ValidationError:")

        persisted = in_memory_db.get(AnalysisRun, res.run_id)
        assert persisted.status == "failed"
        assert persisted.metrics_payload is None
        assert persisted.coverage_summary is None
        assert persisted.confidence_summary is None
        assert persisted.rule_interpretations is None

    def test_all_empty_repo_and_force_recompute_plus_custom_six_domain_penalties(self, in_memory_db):
        # 1. Completely empty repository produces explicit unreliable empty confidence & status
        empty_repo = Repository(
            id=9208,
            repo_name="orch/empty_9208",
            owner="orch",
            name="empty_9208",
            stars_count=0,
            forks_count=0,
            primary_language="Python",
            is_archived=False,
            is_fork=False,
            open_issues_count=0,
            last_synced_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
            analytics_source_revision=1,
            is_active=True,
        )
        in_memory_db.add(empty_repo)
        in_memory_db.add(
            CollectionRun(
                repository_id=9208,
                mode="full_history",
                status="completed",
                is_partial=False,
                started_at=datetime(2026, 3, 19, tzinfo=timezone.utc),
                completed_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
            )
        )
        in_memory_db.commit()

        orch = AnalyticsOrchestrator()
        res_empty = orch.enqueue_and_process(repository_id=9208, db=in_memory_db, worker_id="w_empty")
        assert res_empty.published is True
        persisted_empty = in_memory_db.get(AnalysisRun, res_empty.run_id)
        assert persisted_empty.status == "completed"
        assert persisted_empty.confidence_summary["band"] == "unreliable"
        assert persisted_empty.coverage_summary["summary"]["overall_status"] == "empty"
        assert len(persisted_empty.confidence_summary["domain_breakdown"]) == 6
        assert any(p.get("domain") == "release_evolution" for p in persisted_empty.confidence_summary["penalties"])

        # 2. force=True recompute creates a distinct new completed AnalysisRun ID
        res_forced = orch.enqueue_and_process(
            repository_id=9208,
            db=in_memory_db,
            worker_id="w_empty_forced",
            force=True,
        )
        assert res_forced.published is True
        assert res_forced.run_id != res_empty.run_id

        # 3. Direct custom 6-domain penalty aggregation creates fallback penalties when details omit "penalties"
        custom_domains = {}
        penalties_by_domain = {
            "development_activity": 0.1,
            "contributor_ecosystem": 0.2,
            "pull_request_intelligence": 0.15,
            "issue_intelligence": 0.25,
            "release_evolution": 0.3,
            "repository_synthesis": 0.0,
        }
        for key, pen in penalties_by_domain.items():
            metrics = (
                {
                    "scope": {"status": "sufficient"},
                    "cadence": {"status": "sufficient"},
                    "prerelease_surges": {"status": "sufficient"},
                    "semver_adoption": {"status": "sufficient"},
                }
                if key == "release_evolution"
                else {"status": "sufficient", "sample": 1}
            )
            custom_domains[key] = CalculationResult(
                domain=key,
                metrics=metrics,
                coverage=MetricCoverageItem(
                    domain=key,
                    status="sufficient" if pen == 0.0 else "degraded",
                    sample_size=10,
                    min_sample_required=5,
                ),
                confidence_penalty=pen,
                rules=[],
                details={},  # Intentionally omit explicit "penalties" list
            )

        bundle = orch.build_publication_bundle(
            repository_id=9208,
            snapshot_revision=1,
            analysis_reference_time=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
            domain_results=custom_domains,
        )
        # Expected mean domain score: (0.9 + 0.8 + 0.85 + 0.75 + 0.7 + 1.0) / 6 = 5.0 / 6 = 0.833
        assert bundle.confidence_summary["score"] == pytest.approx(0.833, abs=1e-3)
        fallback_rules = {p["rule"] for p in bundle.confidence_summary["penalties"]}
        assert fallback_rules == {
            "PENALTY-DEVELOPMENT_ACTIVITY",
            "PENALTY-CONTRIBUTOR_ECOSYSTEM",
            "PENALTY-PULL_REQUEST_INTELLIGENCE",
            "PENALTY-ISSUE_INTELLIGENCE",
            "PENALTY-RELEASE_EVOLUTION",
        }

