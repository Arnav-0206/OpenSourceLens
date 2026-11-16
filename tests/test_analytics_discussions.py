"""Unit tests for Discussions Intelligence Calculator (Module 7 / Stage 3).

Verifies:
1. Empty discussions behavior (status='empty', confidence_penalty=0.5, RULE-DISC-EMPTY).
2. Q&A resolution dynamics, answer latencies quantiles (p50, p75, p90, mean), and rules.
3. Community engagement intensity (median/mean comments & upvotes, top discussions).
4. Category breakdown and dominant category rule.
5. Contributor overlap bridging (discussion participants vs repo committers).
6. Time anchoring & snapshot isolation (ignoring post-ref_time records).
7. Cross-domain scorecard integration in CrossDomainSynthesisCalculator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.base import Base
from backend.app.database.models import (
    Contributor,
    Discussion,
    DiscussionComment,
    Issue,
    Repository,
)
from backend.app.services.analytics.cross_domain import CrossDomainSynthesisCalculator
from backend.app.services.analytics.discussions import (
    DiscussionsIntelligenceCalculator,
    build_discussions_rules,
)


@pytest.fixture
def in_memory_db():
    """Create isolated in-memory SQLite session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_repository(db: Session, repo_id: int = 100) -> Repository:
    repo = Repository(
        id=repo_id,
        repo_name=f"test/repo-{repo_id}",
        owner="test",
        name=f"repo-{repo_id}",
        stars_count=100,
        forks_count=20,
        primary_language="Python",
        is_archived=False,
        is_fork=False,
        open_issues_count=5,
        analytics_source_revision=1,
    )
    db.add(repo)
    db.commit()
    return repo


class TestDiscussionsIntelligenceCalculator:
    """Test suite for DiscussionsIntelligenceCalculator."""

    def test_empty_discussions_behavior(self, in_memory_db):
        repo = _seed_repository(in_memory_db, repo_id=101)
        calc = DiscussionsIntelligenceCalculator()
        ref_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        result = calc.calculate(
            repository_id=repo.id,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        assert result.domain == "discussions_intelligence"
        assert result.coverage.status == "empty"
        assert result.coverage.sample_size == 0
        assert result.confidence_penalty == 0.5
        assert result.metrics["status"] == "empty"
        assert result.metrics["total_discussions"] == 0
        assert result.metrics["total_comments"] == 0
        assert result.metrics["dominant_category"] is None
        assert result.metrics["qna"]["qna_answered_rate"] is None
        assert result.metrics["qna"]["time_to_answer_hours"]["p50"] is None
        assert len(result.rules) == 1
        assert result.rules[0].rule_id == "RULE-DISC-EMPTY"
        assert len(result.details["penalties"]) == 1
        assert result.details["penalties"][0]["rule"] == "PENALTY-EMPTY-DISCUSSIONS"

    def test_qna_resolution_dynamics_and_latencies(self, in_memory_db):
        repo = _seed_repository(in_memory_db, repo_id=102)
        ref_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        # Seed 4 Q&A discussions: 3 answered, 1 unanswered
        # Ans 1: answered after 2 hours
        d1 = Discussion(
            id=1,
            repository_id=repo.id,
            discussion_number=1,
            title="How to install?",
            category_name="Q&A",
            category_is_answerable=True,
            is_answered=True,
            author="user1",
            comments_count=2,
            upvote_count=3,
            created_at=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
            answered_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        )
        # Ans 2: answered after 6 hours
        d2 = Discussion(
            id=2,
            repository_id=repo.id,
            discussion_number=2,
            title="How to configure?",
            category_name="Q&A",
            category_is_answerable=True,
            is_answered=True,
            author="user2",
            comments_count=4,
            upvote_count=5,
            created_at=datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc),
            answered_at=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
        )
        # Ans 3: answered after 10 hours
        d3 = Discussion(
            id=3,
            repository_id=repo.id,
            discussion_number=3,
            title="How to deploy?",
            category_name="Q&A",
            category_is_answerable=True,
            is_answered=True,
            author="user3",
            comments_count=3,
            upvote_count=2,
            created_at=datetime(2026, 3, 3, 10, 0, tzinfo=timezone.utc),
            answered_at=datetime(2026, 3, 3, 20, 0, tzinfo=timezone.utc),
        )
        # Unanswered Q&A
        d4 = Discussion(
            id=4,
            repository_id=repo.id,
            discussion_number=4,
            title="How to debug crash?",
            category_name="Q&A",
            category_is_answerable=True,
            is_answered=False,
            author="user4",
            comments_count=1,
            upvote_count=0,
            created_at=datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc),
        )
        # General discussion (non-Q&A) to reach 5 total discussions
        d5 = Discussion(
            id=5,
            repository_id=repo.id,
            discussion_number=5,
            title="General Ideas",
            category_name="General",
            category_is_answerable=False,
            is_answered=False,
            author="user1",
            comments_count=5,
            upvote_count=8,
            created_at=datetime(2026, 3, 5, 10, 0, tzinfo=timezone.utc),
        )
        in_memory_db.add_all([d1, d2, d3, d4, d5])
        in_memory_db.commit()

        calc = DiscussionsIntelligenceCalculator()
        res = calc.calculate(
            repository_id=repo.id,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        assert res.coverage.status == "sufficient"
        assert res.coverage.sample_size == 5
        assert res.confidence_penalty == 0.0

        qna = res.metrics["qna"]
        assert qna["total_qna_threads"] == 4
        assert qna["answered_qna_threads"] == 3
        assert qna["unanswered_qna_threads"] == 1
        assert qna["qna_answered_rate"] == 0.75  # 3/4 = 75%

        lat = qna["time_to_answer_hours"]
        # Latencies: [2.0, 6.0, 10.0]
        assert lat["p50"] == 6.0
        assert lat["mean"] == 6.0
        assert lat["p75"] == 8.0
        assert lat["p90"] == 9.2

        assert res.metrics["categories"]["Q&A"] == 4
        assert res.metrics["categories"]["General"] == 1
        assert res.metrics["dominant_category"] == "Q&A"

    def test_low_qna_rate_triggers_rule_disc_002(self, in_memory_db):
        repo = _seed_repository(in_memory_db, repo_id=103)
        ref_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        # 5 Q&A threads, only 1 answered (20% answer rate)
        discs = []
        for i in range(1, 6):
            discs.append(
                Discussion(
                    id=10 + i,
                    repository_id=repo.id,
                    discussion_number=i,
                    title=f"Question {i}",
                    category_name="Q&A",
                    category_is_answerable=True,
                    is_answered=(i == 1),
                    author=f"user_{i}",
                    comments_count=1,
                    upvote_count=1,
                    created_at=datetime(2026, 3, i, 10, 0, tzinfo=timezone.utc),
                    answered_at=datetime(2026, 3, i, 12, 0, tzinfo=timezone.utc) if i == 1 else None,
                )
            )
        in_memory_db.add_all(discs)
        in_memory_db.commit()

        calc = DiscussionsIntelligenceCalculator()
        res = calc.calculate(
            repository_id=repo.id,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        assert res.metrics["qna"]["qna_answered_rate"] == 0.20
        rule_ids = {r.rule_id for r in res.rules}
        assert "RULE-DISC-002" in rule_ids
        rule_002 = next(r for r in res.rules if r.rule_id == "RULE-DISC-002")
        assert rule_002.severity == "warning"
        assert "Q&A Backlog Stagnation" in rule_002.finding

    def test_community_engagement_and_contributor_overlap(self, in_memory_db):
        repo = _seed_repository(in_memory_db, repo_id=104)
        ref_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        # Add contributors to repo: "alice" and "bob"
        in_memory_db.add_all([
            Contributor(
                repository_id=repo.id,
                login="alice",
                git_email="alice@test.com",
                total_commits=50,
            ),
            Contributor(
                repository_id=repo.id,
                login="bob",
                git_email="bob@test.com",
                total_commits=30,
            ),
        ])

        # Discussions:
        # alice (contributor)
        d1 = Discussion(
            id=20,
            repository_id=repo.id,
            discussion_number=1,
            title="RFC: Architecture",
            category_name="RFC",
            author="alice",
            comments_count=10,
            upvote_count=15,
            created_at=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        )
        # charlie (external community member)
        d2 = Discussion(
            id=21,
            repository_id=repo.id,
            discussion_number=2,
            title="Plugin question",
            category_name="Q&A",
            author="charlie",
            comments_count=6,
            upvote_count=4,
            created_at=datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc),
        )
        in_memory_db.add_all([d1, d2])

        # Comments on discussions
        c1 = DiscussionComment(
            id=101,
            repository_id=repo.id,
            discussion_id=20,
            comment_id="comment_101",
            author="bob",
            upvote_count=3,
            created_at=datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc),
        )
        c2 = DiscussionComment(
            id=102,
            repository_id=repo.id,
            discussion_id=21,
            comment_id="comment_102",
            author="dana",
            upvote_count=1,
            created_at=datetime(2026, 3, 2, 11, 0, tzinfo=timezone.utc),
        )
        in_memory_db.add_all([c1, c2])
        in_memory_db.commit()

        calc = DiscussionsIntelligenceCalculator()
        res = calc.calculate(
            repository_id=repo.id,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        eng = res.metrics["engagement"]
        # Unique discussion authors: alice, charlie, bob, dana -> 4 authors
        # Alice & Bob are in Contributors table -> contributor_overlap_count = 2, rate = 0.5 (2/4)
        assert eng["unique_discussion_authors"] == 4
        assert eng["contributor_overlap_count"] == 2
        assert eng["contributor_overlap_rate"] == 0.5
        assert eng["median_comments_per_discussion"] == 8.0  # [6, 10] -> 8.0
        assert eng["median_upvotes_per_discussion"] == 9.5  # [4, 15] -> 9.5

        # 2 discussions is < 5, so status should be 'degraded'
        assert res.coverage.status == "degraded"
        assert res.confidence_penalty == 0.2
        assert len(res.details["penalties"]) == 1
        assert res.details["penalties"][0]["rule"] == "PENALTY-SMALL-DISCUSSIONS-SAMPLE"

    def test_time_anchoring_ignores_future_records(self, in_memory_db):
        repo = _seed_repository(in_memory_db, repo_id=105)
        ref_time = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)

        # Discussion before ref_time
        d_past = Discussion(
            id=30,
            repository_id=repo.id,
            discussion_number=1,
            title="Past discussion",
            category_name="General",
            author="user1",
            comments_count=2,
            upvote_count=1,
            created_at=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
        )
        # Discussion after ref_time
        d_future = Discussion(
            id=31,
            repository_id=repo.id,
            discussion_number=2,
            title="Future discussion",
            category_name="General",
            author="user2",
            comments_count=10,
            upvote_count=20,
            created_at=datetime(2026, 3, 18, 10, 0, tzinfo=timezone.utc),
        )
        in_memory_db.add_all([d_past, d_future])
        in_memory_db.commit()

        calc = DiscussionsIntelligenceCalculator()
        res = calc.calculate(
            repository_id=repo.id,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
        )

        # Only d_past should be counted
        assert res.metrics["total_discussions"] == 1
        assert res.coverage.sample_size == 1
        assert res.coverage.status == "insufficient"
        assert res.confidence_penalty == 0.4
        assert res.details["penalties"][0]["rule"] == "PENALTY-SINGLE-DISCUSSION"

    def test_synthesis_scorecard_embeds_community_discussions(self, in_memory_db):
        repo = _seed_repository(in_memory_db, repo_id=106)
        ref_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

        # Calculate discussions result
        disc_calc = DiscussionsIntelligenceCalculator()
        disc_res = disc_calc.calculate(repo.id, in_memory_db, ref_time, 1)

        syn_calc = CrossDomainSynthesisCalculator()
        syn_res = syn_calc.calculate(
            repository_id=repo.id,
            db=in_memory_db,
            analysis_reference_time=ref_time,
            snapshot_revision=1,
            precomputed_domains={"discussions_intelligence": disc_res},
        )

        scorecard = syn_res.metrics.get("diagnostic_scorecard", {})
        assert "community_discussions" in scorecard
        cd_card = scorecard["community_discussions"]
        assert cd_card["coverage_status"] == "empty"
        assert cd_card["total_discussions"] == 0
        assert cd_card["qna_answered_rate"] is None
