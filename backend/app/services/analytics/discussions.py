"""Discussions & Community Intelligence Analytics Engine for OpenSourceLens (Module 7 / Stage 3).

Implements DiscussionsIntelligenceCalculator adhering to BaseMetricCalculator:
- Discussions thread volume, category distribution, and participation metrics.
- Q&A resolution dynamics (answered rate, time-to-answer latency quantiles).
- Community engagement intensity (upvotes, comment volume, top engaged discussions).
- Cross-domain contributor bridge (discussions participants vs code contributors).
- Strict NULL vs. 0.0 honesty (zero fabrication of placeholder metrics).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    CollectionRun,
    Contributor,
    Discussion,
    DiscussionComment,
    Issue,
    PullRequest,
)
from backend.app.schemas.analytics_v2 import MetricCoverageItem, RuleInterpretation
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.development_activity import _ensure_utc


def build_discussions_rules(
    domain_name: str,
    total_discussions: int,
    qna_metrics: Dict[str, Any],
    engagement_metrics: Dict[str, Any],
) -> List[RuleInterpretation]:
    """Generate deterministic RULE-DISC-* findings from discussions metrics."""
    rules: List[RuleInterpretation] = []

    if total_discussions == 0:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-DISC-EMPTY",
                domain=domain_name,
                severity="info",
                finding="Discussions Inactive or Disabled: Repository contains zero recorded discussions.",
                evidence={"total_discussions": 0},
                recommendation="Enable and promote GitHub Discussions for community Q&A, announcements, and RFCs.",
            )
        )
        return rules

    # RULE-DISC-001: High Q&A Resolution Rate (>= 80% with >= 3 questions)
    qna_total = qna_metrics.get("total_qna_threads", 0)
    qna_rate = qna_metrics.get("qna_answered_rate")
    if qna_total >= 3 and qna_rate is not None and qna_rate >= 0.80:
        median_lat = qna_metrics.get("time_to_answer_hours", {}).get("p50")
        rules.append(
            RuleInterpretation(
                rule_id="RULE-DISC-001",
                domain=domain_name,
                severity="positive",
                finding=(
                    f"Responsive Community Q&A: {round(qna_rate * 100.0, 1)}% of questions have verified answers "
                    f"({qna_metrics.get('answered_qna_threads')}/{qna_total} answered"
                    + (f", median resolution latency {median_lat}h" if median_lat is not None else "")
                    + ")."
                ),
                evidence={
                    "total_qna_threads": qna_total,
                    "answered_qna_threads": qna_metrics.get("answered_qna_threads"),
                    "qna_answered_rate": qna_rate,
                    "median_answer_latency_hours": median_lat,
                },
                recommendation="Maintain active community triage and highlight high-quality community answers.",
            )
        )

    # RULE-DISC-002: Q&A Backlog Stagnation (< 40% with >= 5 questions)
    if qna_total >= 5 and qna_rate is not None and qna_rate < 0.40:
        unanswered = qna_metrics.get("unanswered_qna_threads", 0)
        rules.append(
            RuleInterpretation(
                rule_id="RULE-DISC-002",
                domain=domain_name,
                severity="warning",
                finding=(
                    f"Q&A Backlog Stagnation: Only {round(qna_rate * 100.0, 1)}% of questions have answers "
                    f"({unanswered} unanswered out of {qna_total})."
                ),
                evidence={
                    "total_qna_threads": qna_total,
                    "unanswered_qna_threads": unanswered,
                    "qna_answered_rate": qna_rate,
                },
                recommendation="Designate community triage rotations or maintainers to address unanswered discussions.",
            )
        )

    # RULE-DISC-003: Vibrant Community Interaction (median comments >= 2.0 or median upvotes >= 3.0)
    med_comm = engagement_metrics.get("median_comments_per_discussion", 0.0)
    med_up = engagement_metrics.get("median_upvotes_per_discussion", 0.0)
    if total_discussions >= 5 and (med_comm >= 2.0 or med_up >= 3.0):
        rules.append(
            RuleInterpretation(
                rule_id="RULE-DISC-003",
                domain=domain_name,
                severity="positive",
                finding=(
                    f"Vibrant Discussions Engagement: Discussions show strong community interaction "
                    f"(median {med_comm} comments, median {med_up} upvotes across {total_discussions} threads)."
                ),
                evidence={
                    "total_discussions": total_discussions,
                    "median_comments_per_discussion": med_comm,
                    "median_upvotes_per_discussion": med_up,
                },
                recommendation="Continue leveraging discussions for collaborative decision making and feature ideas.",
            )
        )

    return rules


class DiscussionsIntelligenceCalculator(BaseMetricCalculator):
    """Calculates discussions volume, category mix, Q&A dynamics, and community engagement."""

    @property
    def domain_name(self) -> str:
        return "discussions_intelligence"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute discussions intelligence anchored strictly to analysis_reference_time."""
        ref_time = _ensure_utc(analysis_reference_time)

        # 1. Fetch discussions
        discussions = db.scalars(
            select(Discussion).where(
                Discussion.repository_id == repository_id,
                Discussion.created_at <= ref_time,
            )
        ).all()

        total_discussions = len(discussions)

        # 2. Fetch comments
        comments = db.scalars(
            select(DiscussionComment).where(
                DiscussionComment.repository_id == repository_id,
                DiscussionComment.created_at <= ref_time,
            )
        ).all()

        total_comments = len(comments)

        # Early return if zero discussions
        if total_discussions == 0:
            coverage = MetricCoverageItem(
                domain=self.domain_name,
                status="empty",
                sample_size=0,
                min_sample_required=5,
                details={
                    "total_discussions": 0,
                    "reason": "Repository contains zero recorded discussions.",
                },
            )
            empty_metrics: Dict[str, Any] = {
                "status": "empty",
                "total_discussions": 0,
                "total_comments": 0,
                "total_upvotes": 0,
                "active_discussions_window": 0,
                "discussion_to_issue_ratio": None,
                "categories": {},
                "dominant_category": None,
                "qna": {
                    "total_qna_threads": 0,
                    "answered_qna_threads": 0,
                    "unanswered_qna_threads": 0,
                    "qna_answered_rate": None,
                    "time_to_answer_hours": {
                        "p50": None,
                        "p75": None,
                        "p90": None,
                        "mean": None,
                    },
                },
                "engagement": {
                    "mean_comments_per_discussion": 0.0,
                    "median_comments_per_discussion": 0.0,
                    "mean_upvotes_per_discussion": 0.0,
                    "median_upvotes_per_discussion": 0.0,
                    "unique_discussion_authors": 0,
                    "contributor_overlap_count": 0,
                    "contributor_overlap_rate": None,
                    "top_discussed_threads": [],
                },
            }
            rules = build_discussions_rules(self.domain_name, 0, {}, {})
            penalties = [
                {
                    "rule": "PENALTY-EMPTY-DISCUSSIONS",
                    "penalty": 0.5,
                    "reason": "Repository snapshot has zero recorded discussions.",
                }
            ]
            return CalculationResult(
                domain=self.domain_name,
                metrics=empty_metrics,
                coverage=coverage,
                confidence_penalty=0.5,
                rules=rules,
                details={"penalties": penalties, "snapshot_revision": snapshot_revision},
            )

        # 3. Volume & Activity window (past 90 days)
        cutoff_90d = ref_time - timedelta(days=90)
        active_90d = sum(
            1 for d in discussions
            if _ensure_utc(d.created_at) >= cutoff_90d
            or (_ensure_utc(d.updated_at) and _ensure_utc(d.updated_at) >= cutoff_90d)
        )

        total_issues_count = db.scalar(
            select(func.count(Issue.id)).where(
                Issue.repository_id == repository_id,
                Issue.created_at <= ref_time,
            )
        ) or 0

        disc_to_issue_ratio: Optional[float] = None
        if total_issues_count > 0:
            disc_to_issue_ratio = round(total_discussions / float(total_issues_count), 4)

        # Total upvotes across discussions and comments
        total_upvotes = sum(d.upvote_count for d in discussions) + sum(c.upvote_count for c in comments)

        # 4. Category breakdown
        category_counts: Dict[str, int] = {}
        for d in discussions:
            cat = (d.category_name or "General").strip()
            category_counts[cat] = category_counts.get(cat, 0) + 1

        dominant_category: Optional[str] = None
        if category_counts:
            dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]

        # 5. Q&A resolution dynamics
        qna_threads = [
            d for d in discussions
            if d.category_is_answerable or (d.category_name and "q&a" in d.category_name.lower())
        ]
        total_qna = len(qna_threads)
        answered_qna = sum(1 for d in qna_threads if d.is_answered)
        unanswered_qna = total_qna - answered_qna

        qna_answered_rate: Optional[float] = None
        if total_qna > 0:
            qna_answered_rate = round(answered_qna / float(total_qna), 4)

        # Latencies in hours
        answer_latencies: List[float] = []
        for d in qna_threads:
            if d.is_answered and d.answered_at and d.created_at:
                c_dt = _ensure_utc(d.created_at)
                a_dt = _ensure_utc(d.answered_at)
                if a_dt and c_dt and a_dt >= c_dt:
                    diff_h = (a_dt - c_dt).total_seconds() / 3600.0
                    answer_latencies.append(diff_h)

        time_to_answer_hours: Dict[str, Optional[float]] = {
            "p50": None,
            "p75": None,
            "p90": None,
            "mean": None,
            "min": None,
            "max": None,
        }
        if answer_latencies:
            time_to_answer_hours["p50"] = round(float(np.percentile(answer_latencies, 50)), 2)
            time_to_answer_hours["p75"] = round(float(np.percentile(answer_latencies, 75)), 2)
            time_to_answer_hours["p90"] = round(float(np.percentile(answer_latencies, 90)), 2)
            time_to_answer_hours["mean"] = round(float(np.mean(answer_latencies)), 2)
            time_to_answer_hours["min"] = round(float(np.min(answer_latencies)), 2)
            time_to_answer_hours["max"] = round(float(np.max(answer_latencies)), 2)

        qna_metrics = {
            "total_qna_threads": total_qna,
            "answered_qna_threads": answered_qna,
            "unanswered_qna_threads": unanswered_qna,
            "qna_answered_rate": qna_answered_rate,
            "time_to_answer_hours": time_to_answer_hours,
        }

        # 6. Engagement intensity
        comment_counts_list = [d.comments_count for d in discussions]
        upvote_counts_list = [d.upvote_count for d in discussions]

        mean_comm = round(float(np.mean(comment_counts_list)), 2) if comment_counts_list else 0.0
        med_comm = round(float(np.percentile(comment_counts_list, 50)), 2) if comment_counts_list else 0.0
        mean_up = round(float(np.mean(upvote_counts_list)), 2) if upvote_counts_list else 0.0
        med_up = round(float(np.percentile(upvote_counts_list, 50)), 2) if upvote_counts_list else 0.0

        unique_authors: Set[str] = set()
        for d in discussions:
            if d.author:
                unique_authors.add(d.author)
        for c in comments:
            if c.author:
                unique_authors.add(c.author)

        # Cross-domain contributor bridge
        contributor_logins = set(
            db.scalars(
                select(Contributor.login).where(
                    Contributor.repository_id == repository_id,
                    Contributor.login.is_not(None),
                )
            ).all()
        )
        overlap_authors = unique_authors.intersection(contributor_logins)
        contributor_overlap_count = len(overlap_authors)
        contributor_overlap_rate = (
            round(contributor_overlap_count / float(len(unique_authors)), 4)
            if unique_authors
            else None
        )

        # Top 5 discussed threads
        sorted_threads = sorted(
            discussions,
            key=lambda d: (d.comments_count * 2 + d.upvote_count),
            reverse=True,
        )[:5]
        top_discussed_threads = [
            {
                "discussion_number": d.discussion_number,
                "title": d.title,
                "category": d.category_name,
                "comments_count": d.comments_count,
                "upvote_count": d.upvote_count,
                "is_answered": d.is_answered,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in sorted_threads
        ]

        engagement_metrics = {
            "mean_comments_per_discussion": mean_comm,
            "median_comments_per_discussion": med_comm,
            "mean_upvotes_per_discussion": mean_up,
            "median_upvotes_per_discussion": med_up,
            "unique_discussion_authors": len(unique_authors),
            "contributor_overlap_count": contributor_overlap_count,
            "contributor_overlap_rate": contributor_overlap_rate,
            "top_discussed_threads": top_discussed_threads,
        }

        # 7. Coverage and penalties
        min_sample = 5
        penalties: List[Dict[str, Any]] = []
        if total_discussions >= min_sample:
            cov_status = "sufficient"
            confidence_penalty = 0.0
        elif total_discussions >= 2:
            cov_status = "degraded"
            confidence_penalty = 0.2
            penalties.append({
                "rule": "PENALTY-SMALL-DISCUSSIONS-SAMPLE",
                "penalty": 0.2,
                "reason": f"Observed discussions ({total_discussions}) is below high-confidence threshold ({min_sample}).",
            })
        else:
            cov_status = "insufficient"
            confidence_penalty = 0.4
            penalties.append({
                "rule": "PENALTY-SINGLE-DISCUSSION",
                "penalty": 0.4,
                "reason": "Only 1 discussion recorded; participation and Q&A dynamics require multiple discussions.",
            })

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=cov_status,
            sample_size=total_discussions,
            min_sample_required=min_sample,
            details={
                "total_discussions": total_discussions,
                "total_comments": total_comments,
                "total_qna_threads": total_qna,
                "dominant_category": dominant_category,
            },
        )

        rules = build_discussions_rules(
            domain_name=self.domain_name,
            total_discussions=total_discussions,
            qna_metrics=qna_metrics,
            engagement_metrics=engagement_metrics,
        )

        metrics: Dict[str, Any] = {
            "status": cov_status,
            "total_discussions": total_discussions,
            "total_comments": total_comments,
            "total_upvotes": total_upvotes,
            "active_discussions_window": active_90d,
            "discussion_to_issue_ratio": disc_to_issue_ratio,
            "categories": category_counts,
            "dominant_category": dominant_category,
            "qna": qna_metrics,
            "engagement": engagement_metrics,
        }

        return CalculationResult(
            domain=self.domain_name,
            metrics=metrics,
            coverage=coverage,
            confidence_penalty=confidence_penalty,
            rules=rules,
            details={"penalties": penalties, "snapshot_revision": snapshot_revision},
        )
