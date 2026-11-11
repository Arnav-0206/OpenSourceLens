"""Pull Request Lifecycle & Turnaround Intelligence Engine (Module 3 / Phase C.4).

Implements PullRequestIntelligenceCalculator adhering to BaseMetricCalculator:
- Separated responsiveness channels: Time-to-First-Response (TTFResp),
  Time-to-First-Formal-Review (TTFR_formal), and Time-to-First-Inline-Comment (TTFR_inline).
- PR-author self-interaction and bot identity exclusion.
- Collection-mode and stream-coverage awareness (never masquerading skipped/uncollected streams as zero).
- Merge latency turnaround quantiles with genuine 0.0h merge preservation and negative clock-skew isolation.
- PR outcome distribution (merged via merged_at <= ref_time, closed_unmerged, open).
- Deterministic Kaplan-Meier survival estimator (S(7d), S(30d), median survival time) gated at >= 15 PRs.
- Diff-aware PR size vs. turnaround analysis preserving NULL vs zero diffs.
- Review bottleneck diagnostics (review latency share, maintainer review load, unreviewed open PR age).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    CollectionRun,
    Contributor,
    IssueComment,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
)
from backend.app.schemas.analytics_v2 import MetricCoverageItem, RuleInterpretation
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.development_activity import (
    _ensure_utc,
    _linear_percentile,
)


FORMAL_REVIEW_STATES: Set[str] = {"APPROVED", "CHANGES_REQUESTED", "COMMENTED"}
PR_SIZE_BINS: List[Tuple[str, int, Optional[int]]] = [
    ("small", 0, 50),
    ("medium", 51, 250),
    ("large", 251, 1000),
    ("xl", 1001, None),
]


def _extract_stream_status(cov_summary: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    """Read normalized status string from either flat collector entity_status or nested dict."""
    if not isinstance(cov_summary, dict):
        return None
    val = cov_summary.get(key)
    if isinstance(val, str):
        return val.lower().strip()
    if isinstance(val, dict):
        status = val.get("status")
        return status.lower().strip() if isinstance(status, str) else None
    return None


def compute_kaplan_meier_curve(
    records: List[Tuple[float, int]],
    horizons_days: Tuple[float, ...] = (7.0, 30.0),
) -> Dict[str, Any]:
    """Compute product-limit Kaplan-Meier survival curve, median time, and horizon probabilities.

    Args:
        records: Non-empty list of (duration_days >= 0.0, event_observed in {0, 1}).
        horizons_days: Time horizons (in days) at which S(t) is evaluated.

    Returns:
        Dictionary with 'horizon_survival', 'within_followup', 'median_days',
        'max_observed_time_days', 'terminal_survival', and capped 'curve_steps'.
    """
    n_valid = len(records)
    by_time: Dict[float, List[int]] = defaultdict(list)
    for dur_d, ev in records:
        by_time[dur_d].append(ev)

    sorted_times = sorted(by_time.keys())
    at_risk = n_valid
    surv = 1.0
    curve_steps: List[Dict[str, Any]] = []
    exact_steps: List[Tuple[float, float]] = []

    if sorted_times[0] > 0.0:
        curve_steps.append({
            "time_days": 0.0,
            "survival_prob": 1.0,
            "at_risk": n_valid,
            "events": 0,
            "censored": 0,
        })

    median_days: Optional[float] = None
    for t in sorted_times:
        evs = by_time[t]
        d_i = sum(evs)
        c_i = len(evs) - d_i
        n_i = at_risk
        # Standard KM tie convention: events occur just prior to censorings at timestamp t
        if n_i > 0 and d_i > 0:
            surv = surv * (1.0 - (float(d_i) / float(n_i)))
        if median_days is None and surv <= 0.50:
            median_days = t
        exact_steps.append((t, surv))
        curve_steps.append({
            "time_days": round(t, 4),
            "survival_prob": round(surv, 4),
            "at_risk": n_i,
            "events": d_i,
            "censored": c_i,
        })
        at_risk -= (d_i + c_i)

    def _eval_step_at(target_days: float) -> float:
        val = 1.0
        for t_step, s_step in exact_steps:
            if t_step > target_days:
                break
            val = s_step
        return round(val, 4)

    max_obs_days = sorted_times[-1]
    horizon_survival = {h: _eval_step_at(h) for h in horizons_days}
    within_followup = {h: (surv == 0.0) or (max_obs_days >= h) for h in horizons_days}

    if len(curve_steps) > 200:
        event_steps = [
            s for s in curve_steps if s["events"] > 0 or s is curve_steps[0] or s is curve_steps[-1]
        ]
        if len(event_steps) <= 200:
            curve_steps = event_steps
        else:
            step_stride = max(1, math.ceil(len(event_steps) / 200))
            sampled = event_steps[::step_stride]
            if sampled[-1] is not event_steps[-1]:
                sampled.append(event_steps[-1])
            curve_steps = sampled

    return {
        "horizon_survival": horizon_survival,
        "within_followup": within_followup,
        "median_days": median_days,
        "max_observed_time_days": max_obs_days,
        "terminal_survival": surv,
        "curve_steps": curve_steps,
    }


def _summarize_hours_sample(
    durations_hours: List[float],
    min_required: int,
    confident_required: Optional[int] = None,
    include_iqr: bool = False,
    empty_status: str = "empty",
    insufficient_reason: str = "",
) -> Dict[str, Any]:
    """Compute empirical quantiles, mean, and optional IQR for non-negative durations."""
    n = len(durations_hours)
    if n == 0:
        return {
            "sample_count": 0,
            "status": empty_status,
            "min_sample_required": min_required,
            "confident_sample_required": confident_required,
            "mean_hours": None,
            "p25_hours": None,
            "p50_hours": None,
            "p75_hours": None,
            "p90_hours": None,
            **({"iqr_hours": None} if include_iqr else {}),
            "limitations": insufficient_reason or "No valid non-negative observations available.",
        }

    if n < min_required:
        return {
            "sample_count": n,
            "status": "insufficient",
            "min_sample_required": min_required,
            "confident_sample_required": confident_required,
            "mean_hours": None,
            "p25_hours": None,
            "p50_hours": None,
            "p75_hours": None,
            "p90_hours": None,
            **({"iqr_hours": None} if include_iqr else {}),
            "limitations": insufficient_reason or (
                f"Observed {n} valid PR(s); minimum {min_required} required for stable percentiles."
            ),
        }

    sorted_vals = sorted(durations_hours)
    p25 = _linear_percentile(sorted_vals, 0.25)
    p50 = _linear_percentile(sorted_vals, 0.50)
    p75 = _linear_percentile(sorted_vals, 0.75)
    p90 = _linear_percentile(sorted_vals, 0.90)

    if confident_required is not None and n < confident_required:
        status = "limited"
        limitation = (
            f"Limited sample ({n} < {confident_required}); turnaround quantiles have wide confidence bounds."
        )
    else:
        status = "sufficient"
        limitation = None

    result: Dict[str, Any] = {
        "sample_count": n,
        "status": status,
        "min_sample_required": min_required,
        "confident_sample_required": confident_required,
        "mean_hours": round(statistics.mean(sorted_vals), 2),
        "p25_hours": round(p25, 2),
        "p50_hours": round(p50, 2),
        "p75_hours": round(p75, 2),
        "p90_hours": round(p90, 2),
    }
    if include_iqr:
        result["iqr_hours"] = round(p75 - p25, 2)
    result["limitations"] = limitation
    return result


class PullRequestIntelligenceCalculator(BaseMetricCalculator):
    """Calculates PR responsiveness channels, merge turnaround, outcomes, survival curves, and bottlenecks."""

    @property
    def domain_name(self) -> str:
        return "pull_request_intelligence"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute PR lifecycle metrics anchored to frozen analysis_reference_time."""
        ref_time = _ensure_utc(analysis_reference_time)

        # 1. Query underlying snapshot entities
        prs_rows = db.execute(
            select(
                PullRequest.id,
                PullRequest.pr_number,
                PullRequest.state,
                PullRequest.author,
                PullRequest.author_id,
                PullRequest.created_at,
                PullRequest.closed_at,
                PullRequest.merged_at,
                PullRequest.comments_count,
                PullRequest.review_comments_count,
                PullRequest.additions,
                PullRequest.deletions,
                PullRequest.changed_files,
            ).where(PullRequest.repository_id == repository_id)
        ).fetchall()

        contributors_rows = db.execute(
            select(
                Contributor.github_id,
                Contributor.login,
                Contributor.is_bot,
            ).where(Contributor.repository_id == repository_id)
        ).fetchall()

        latest_collection_run = db.scalar(
            select(CollectionRun)
            .where(CollectionRun.repository_id == repository_id)
            .order_by(CollectionRun.started_at.desc(), CollectionRun.id.desc())
            .limit(1)
        )

        # Build bot identity sets from repository Contributor directory
        bot_github_ids: Set[int] = set()
        bot_logins: Set[str] = set()
        for c in contributors_rows:
            if c.is_bot:
                if c.github_id is not None:
                    bot_github_ids.add(c.github_id)
                if c.login:
                    bot_logins.add(c.login.strip().lower())

        # Filter eligible PRs created strictly <= ref_time and anchor states at ref_time
        eligible_prs: List[Dict[str, Any]] = []
        pr_ids: List[int] = []
        seen_pr_numbers: Set[int] = set()
        future_prs_excluded = 0

        for row in prs_rows:
            if row.pr_number in seen_pr_numbers:
                continue
            created_dt = _ensure_utc(row.created_at)
            if created_dt is None or created_dt > ref_time:
                future_prs_excluded += 1
                continue
            seen_pr_numbers.add(row.pr_number)

            merged_dt = _ensure_utc(row.merged_at)
            closed_dt = _ensure_utc(row.closed_at)
            raw_state = (row.state or "open").lower().strip()

            # Anchor lifecycle state at ref_time using timestamps first (never raw merged/closed booleans)
            if merged_dt is not None and merged_dt <= ref_time:
                effective_state = "merged"
                terminal_dt = merged_dt
            elif closed_dt is not None and closed_dt <= ref_time:
                effective_state = "closed_unmerged"
                terminal_dt = closed_dt
            elif raw_state == "closed" and merged_dt is None and closed_dt is None:
                effective_state = "closed_unmerged"
                terminal_dt = None
            else:
                effective_state = "open"
                terminal_dt = None

            pr_ids.append(row.id)
            eligible_prs.append({
                "id": row.id,
                "pr_number": row.pr_number,
                "author": row.author,
                "author_id": row.author_id,
                "is_bot_author": self._is_bot(row.author_id, row.author, bot_github_ids, bot_logins),
                "created_at": created_dt,
                "merged_at": merged_dt if (merged_dt is not None and merged_dt <= ref_time) else None,
                "closed_at": closed_dt if (closed_dt is not None and closed_dt <= ref_time) else None,
                "terminal_dt": terminal_dt,
                "effective_state": effective_state,
                "comments_count": row.comments_count,
                "review_comments_count": row.review_comments_count,
                "additions": row.additions,
                "deletions": row.deletions,
                "changed_files": row.changed_files,
            })

        # 2. Query interaction streams only if eligible PRs exist
        reviews_rows: List[Any] = []
        inline_rows: List[Any] = []
        issue_comment_rows: List[Any] = []

        if pr_ids:
            reviews_rows = db.execute(
                select(
                    PullRequestReview.pull_request_id,
                    PullRequestReview.review_id,
                    PullRequestReview.author,
                    PullRequestReview.author_id,
                    PullRequestReview.state,
                    PullRequestReview.submitted_at,
                ).where(PullRequestReview.pull_request_id.in_(pr_ids))
            ).fetchall()

            inline_rows = db.execute(
                select(
                    PullRequestReviewComment.pull_request_id,
                    PullRequestReviewComment.comment_id,
                    PullRequestReviewComment.author,
                    PullRequestReviewComment.author_id,
                    PullRequestReviewComment.created_at,
                ).where(PullRequestReviewComment.pull_request_id.in_(pr_ids))
            ).fetchall()

            issue_comment_rows = db.execute(
                select(
                    IssueComment.pull_request_id,
                    IssueComment.comment_id,
                    IssueComment.author,
                    IssueComment.author_id,
                    IssueComment.created_at,
                ).where(
                    IssueComment.repository_id == repository_id,
                    IssueComment.pull_request_id.isnot(None),
                    IssueComment.pull_request_id.in_(pr_ids),
                )
            ).fetchall()

        # 3. Inspect collection scope & skipped/truncated streams
        collection_scope = self._inspect_collection_scope(
            latest_run=latest_collection_run,
            eligible_pr_count=len(eligible_prs),
            reviews_count=len(reviews_rows),
            inline_count=len(inline_rows),
            issue_comments_count=len(issue_comment_rows),
        )

        # 4. Compute responsiveness channels, filtering self-interactions, bots, and future/negative timestamps
        responsiveness = self._evaluate_responsiveness(
            eligible_prs=eligible_prs,
            reviews_rows=reviews_rows,
            inline_rows=inline_rows,
            issue_comment_rows=issue_comment_rows,
            bot_github_ids=bot_github_ids,
            bot_logins=bot_logins,
            ref_time=ref_time,
            collection_scope=collection_scope,
        )

        # 5. Outcome distribution & Merge latency
        outcomes = self._evaluate_outcomes(
            eligible_prs=eligible_prs,
            future_prs_excluded=future_prs_excluded,
        )
        merge_latency = self._evaluate_merge_latency(eligible_prs=eligible_prs)

        # 6. Kaplan-Meier survival analysis (S(7d), S(30d), median survival time; >= 15 PRs required)
        survival = self._evaluate_kaplan_meier_survival(
            eligible_prs=eligible_prs,
            ref_time=ref_time,
        )

        # 7. PR size vs. turnaround impact
        size_impact = self._evaluate_size_vs_turnaround(eligible_prs=eligible_prs)

        # 8. Review bottleneck diagnostics
        bottlenecks = self._evaluate_bottlenecks(
            eligible_prs=eligible_prs,
            responsiveness=responsiveness,
            merge_latency=merge_latency,
            reviews_rows=reviews_rows,
            bot_github_ids=bot_github_ids,
            bot_logins=bot_logins,
            ref_time=ref_time,
            collection_scope=collection_scope,
        )

        # 9. Overall domain coverage, confidence penalties & deterministic rules
        coverage, confidence_penalty, penalties = self._evaluate_coverage_and_confidence(
            eligible_pr_count=len(eligible_prs),
            collection_scope=collection_scope,
            responsiveness=responsiveness,
            merge_latency=merge_latency,
            outcomes=outcomes,
            survival=survival,
            size_impact=size_impact,
            bottlenecks=bottlenecks,
        )

        rules = self._build_rules(
            eligible_pr_count=len(eligible_prs),
            collection_scope=collection_scope,
            responsiveness=responsiveness,
            outcomes=outcomes,
            merge_latency=merge_latency,
            survival=survival,
            size_impact=size_impact,
            bottlenecks=bottlenecks,
        )

        metrics: Dict[str, Any] = {
            "scope": collection_scope,
            "outcome_distribution": outcomes,
            "responsiveness": {
                "ttf_resp": responsiveness["ttf_resp"],
                "ttfr_formal": responsiveness["ttfr_formal"],
                "ttfr_inline": responsiveness["ttfr_inline"],
                "interaction_hygiene": responsiveness["hygiene"],
            },
            "merge_latency": merge_latency,
            "survival_km": survival,
            "size_vs_turnaround": size_impact,
            "review_bottlenecks": bottlenecks,
        }

        return CalculationResult(
            domain=self.domain_name,
            metrics=metrics,
            coverage=coverage,
            confidence_penalty=confidence_penalty,
            rules=rules,
            details={"penalties": penalties, "snapshot_revision": snapshot_revision},
        )

    @staticmethod
    def _is_bot(
        author_id: Optional[int],
        author_login: Optional[str],
        bot_github_ids: Set[int],
        bot_logins: Set[str],
    ) -> bool:
        """Identify bot accounts via directory flag, immutable ID, or [bot] suffix."""
        if author_id is not None and author_id in bot_github_ids:
            return True
        if author_login:
            norm = author_login.strip().lower()
            if norm in bot_logins or norm.endswith("[bot]"):
                return True
        return False

    @staticmethod
    def _is_self_interaction(
        actor_id: Optional[int],
        actor_login: Optional[str],
        pr_author_id: Optional[int],
        pr_author_login: Optional[str],
    ) -> bool:
        """Detect PR author interacting on their own PR without speculative merging."""
        if actor_id is not None and pr_author_id is not None:
            return actor_id == pr_author_id
        if actor_login and pr_author_login:
            return actor_login.strip().lower() == pr_author_login.strip().lower()
        return False

    @staticmethod
    def _inspect_collection_scope(
        latest_run: Optional[CollectionRun],
        eligible_pr_count: int,
        reviews_count: int,
        inline_count: int,
        issue_comments_count: int,
    ) -> Dict[str, Any]:
        """Inspect latest CollectionRun mode & stream statuses for coverage transparency."""
        if latest_run is None:
            return {
                "mode": "unknown_unverified",
                "collection_status": None,
                "is_partial": True,
                "is_bounded_or_partial": True,
                "pr_stream_status": None,
                "pr_reviews_stream_status": None,
                "pr_inline_comments_stream_status": None,
                "issue_comments_stream_status": None,
                "inline_comments_skipped_by_mode": False,
                "reviews_not_collected": False,
                "notes": (
                    "No CollectionRun metadata found; sample reflects available database PR records."
                ),
            }

        mode = (latest_run.mode or "unknown").lower().strip()
        cov = latest_run.coverage_summary if isinstance(latest_run.coverage_summary, dict) else {}

        pr_status = _extract_stream_status(cov, "pull_requests")
        rev_status = _extract_stream_status(cov, "pr_reviews")
        inline_status = _extract_stream_status(cov, "pr_inline_comments")
        ic_status = _extract_stream_status(cov, "issue_comments")

        # fast_snapshot skips pr_inline_comments (entity limit = 0)
        inline_skipped = (
            inline_status == "skipped"
            or (mode == "fast_snapshot" and inline_status is None)
        ) and inline_count == 0
        inline_uncollected = (
            inline_status in ("failed", "pending", "skipped") and inline_count == 0
        )

        # pr_reviews may be uncollected if skipped/failed/budget-exhausted with zero reviews collected
        reviews_not_collected = (
            eligible_pr_count > 0
            and reviews_count == 0
            and rev_status in ("skipped", "failed", "pending")
        )

        bounded_modes = {"fast_snapshot", "standard_diagnostic"}
        stream_truncated = any(
            st in ("truncated", "partial", "rate_limited")
            for st in (pr_status, rev_status, inline_status, ic_status)
        )
        is_bounded_or_partial = (
            bool(latest_run.is_partial)
            or mode in bounded_modes
            or (mode != "full_history" or latest_run.status != "completed")
            or stream_truncated
        )

        notes: List[str] = []
        if mode in bounded_modes:
            notes.append(f"Snapshot ingested under bounded mode '{mode}'.")
        if inline_skipped:
            notes.append("Inline PR diff comments are skipped under this collection mode.")
        elif inline_uncollected:
            notes.append(f"Inline PR diff comments stream status is '{inline_status}'.")
        if rev_status == "truncated" or mode in bounded_modes:
            notes.append("PR reviews are bounded to sampled PRs within API request quota.")
        if stream_truncated:
            notes.append("One or more PR/comment streams reported partial or truncated collection.")

        return {
            "mode": mode,
            "collection_status": latest_run.status,
            "is_partial": bool(latest_run.is_partial),
            "is_bounded_or_partial": is_bounded_or_partial,
            "pr_stream_status": pr_status,
            "pr_reviews_stream_status": rev_status,
            "pr_inline_comments_stream_status": inline_status,
            "issue_comments_stream_status": ic_status,
            "inline_comments_skipped_by_mode": inline_skipped,
            "inline_comments_not_collected": inline_uncollected and not inline_skipped,
            "reviews_not_collected": reviews_not_collected,
            "notes": " ".join(notes) if notes else "Verified complete collection run.",
        }

    def _evaluate_responsiveness(
        self,
        eligible_prs: List[Dict[str, Any]],
        reviews_rows: List[Any],
        inline_rows: List[Any],
        issue_comment_rows: List[Any],
        bot_github_ids: Set[int],
        bot_logins: Set[str],
        ref_time: datetime,
        collection_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute separated TTFResp, TTFR_formal, and TTFR_inline with self/bot filtering."""
        pr_by_id: Dict[int, Dict[str, Any]] = {p["id"]: p for p in eligible_prs}

        first_any_response_h: Dict[int, float] = {}
        first_formal_review_h: Dict[int, float] = {}
        first_inline_comment_h: Dict[int, float] = {}

        self_excluded = 0
        bot_excluded = 0
        unattributed_excluded = 0
        future_interactions_excluded = 0
        negative_duration_excluded = 0

        def _update_min(target: Dict[int, float], pr_id: int, val_h: float) -> None:
            if pr_id not in target or val_h < target[pr_id]:
                target[pr_id] = val_h

        def _validate_interaction(
            pr: Dict[str, Any],
            actor_id: Optional[int],
            actor_login: Optional[str],
            dt_raw: Optional[datetime],
        ) -> Optional[float]:
            nonlocal self_excluded, bot_excluded, unattributed_excluded
            nonlocal future_interactions_excluded, negative_duration_excluded

            dt = _ensure_utc(dt_raw)
            if dt is None or dt > ref_time:
                if dt is not None and dt > ref_time:
                    future_interactions_excluded += 1
                return None

            if actor_id is None and not (actor_login and actor_login.strip()):
                unattributed_excluded += 1
                return None

            if self._is_self_interaction(actor_id, actor_login, pr["author_id"], pr["author"]):
                self_excluded += 1
                return None

            if self._is_bot(actor_id, actor_login, bot_github_ids, bot_logins):
                bot_excluded += 1
                return None

            diff_sec = (dt - pr["created_at"]).total_seconds()
            if diff_sec < 0.0:
                negative_duration_excluded += 1
                return None

            return diff_sec / 3600.0

        # Channel 1: Formal reviews
        seen_review_ids: Set[int] = set()
        for rev in reviews_rows:
            if rev.review_id in seen_review_ids:
                continue
            seen_review_ids.add(rev.review_id)
            pr = pr_by_id.get(rev.pull_request_id)
            if pr is None:
                continue
            state_up = (rev.state or "").upper().strip()
            if state_up not in FORMAL_REVIEW_STATES and state_up != "DISMISSED":
                continue

            dur_h = _validate_interaction(pr, rev.author_id, rev.author, rev.submitted_at)
            if dur_h is None:
                continue

            _update_min(first_any_response_h, pr["id"], dur_h)
            if state_up in FORMAL_REVIEW_STATES:
                _update_min(first_formal_review_h, pr["id"], dur_h)

        # Channel 2: Inline diff review comments
        seen_inline_ids: Set[int] = set()
        for comm in inline_rows:
            if comm.comment_id in seen_inline_ids:
                continue
            seen_inline_ids.add(comm.comment_id)
            pr = pr_by_id.get(comm.pull_request_id)
            if pr is None:
                continue

            dur_h = _validate_interaction(pr, comm.author_id, comm.author, comm.created_at)
            if dur_h is None:
                continue

            _update_min(first_any_response_h, pr["id"], dur_h)
            _update_min(first_inline_comment_h, pr["id"], dur_h)

        # Channel 3: PR conversation comments (IssueComment linked via pull_request_id)
        seen_ic_ids: Set[int] = set()
        for ic in issue_comment_rows:
            if ic.comment_id in seen_ic_ids:
                continue
            seen_ic_ids.add(ic.comment_id)
            pr = pr_by_id.get(ic.pull_request_id)
            if pr is None:
                continue

            dur_h = _validate_interaction(pr, ic.author_id, ic.author, ic.created_at)
            if dur_h is None:
                continue

            _update_min(first_any_response_h, pr["id"], dur_h)

        n_prs = len(eligible_prs)
        unresponded_count = n_prs - len(first_any_response_h)

        # Build channel payloads adhering strictly to v1.1 §§3.1–3.3 thresholds (Decision A.8)
        ttf_resp = _summarize_hours_sample(
            durations_hours=list(first_any_response_h.values()),
            min_required=3,
            confident_required=10,
            empty_status="empty" if n_prs == 0 else "insufficient",
            insufficient_reason=(
                "No eligible PRs observed."
                if n_prs == 0
                else f"Only {len(first_any_response_h)} PR(s) received non-author human responses (minimum 3 required)."
            ),
        )
        ttf_resp["eligible_prs_count"] = n_prs
        ttf_resp["unresponded_prs_count"] = unresponded_count
        ttf_resp["response_coverage_ratio"] = round(len(first_any_response_h) / float(n_prs), 4) if n_prs > 0 else None

        # TTFR_formal (single threshold >= 10, no invented intermediate band)
        if collection_scope["reviews_not_collected"]:
            ttfr_formal = {
                "sample_count": 0,
                "eligible_prs_count": n_prs,
                "status": "reviews_not_collected",
                "min_sample_required": 10,
                "confident_sample_required": 10,
                "mean_hours": None,
                "p25_hours": None,
                "p50_hours": None,
                "p75_hours": None,
                "p90_hours": None,
                "limitations": "Formal PR reviews were not collected in this ingestion run.",
            }
        else:
            ttfr_formal = _summarize_hours_sample(
                durations_hours=list(first_formal_review_h.values()),
                min_required=10,
                confident_required=None,
                empty_status="empty" if n_prs == 0 else "insufficient",
                insufficient_reason=(
                    "No eligible PRs observed."
                    if n_prs == 0
                    else (
                        f"Only {len(first_formal_review_h)} PR(s) have non-author formal reviews "
                        "(minimum 10 required for TTFR_formal)."
                    )
                ),
            )
            ttfr_formal["eligible_prs_count"] = n_prs
            ttfr_formal["confident_sample_required"] = 10

        # TTFR_inline (single threshold >= 5, explicit fast_snapshot / skipped guard)
        if collection_scope["inline_comments_skipped_by_mode"]:
            ttfr_inline = {
                "sample_count": 0,
                "eligible_prs_count": n_prs,
                "status": "skipped_by_collection_mode",
                "min_sample_required": 5,
                "confident_sample_required": 5,
                "mean_hours": None,
                "p25_hours": None,
                "p50_hours": None,
                "p75_hours": None,
                "p90_hours": None,
                "limitations": (
                    f"Inline PR review comments are skipped under '{collection_scope['mode']}' collection mode."
                ),
            }
        elif collection_scope.get("inline_comments_not_collected"):
            ttfr_inline = {
                "sample_count": 0,
                "eligible_prs_count": n_prs,
                "status": "inline_comments_not_collected",
                "min_sample_required": 5,
                "confident_sample_required": 5,
                "mean_hours": None,
                "p25_hours": None,
                "p50_hours": None,
                "p75_hours": None,
                "p90_hours": None,
                "limitations": "Inline PR review comments stream was not collected in this run.",
            }
        else:
            ttfr_inline = _summarize_hours_sample(
                durations_hours=list(first_inline_comment_h.values()),
                min_required=5,
                confident_required=None,
                empty_status="empty" if n_prs == 0 else "insufficient",
                insufficient_reason=(
                    "No eligible PRs observed."
                    if n_prs == 0
                    else (
                        f"Only {len(first_inline_comment_h)} PR(s) have non-author inline diff comments "
                        "(minimum 5 required for TTFR_inline)."
                    )
                ),
            )
            ttfr_inline["eligible_prs_count"] = n_prs
            ttfr_inline["confident_sample_required"] = 5

        if (
            ttfr_formal["status"] == "sufficient"
            and collection_scope["pr_reviews_stream_status"] in ("truncated", "partial")
        ):
            ttfr_formal["limitations"] = (
                "Formal review stream was partially sampled/truncated; reflects sampled PRs only."
            )

        hygiene = {
            "self_interactions_excluded_count": self_excluded,
            "bot_interactions_excluded_count": bot_excluded,
            "unattributed_interactions_excluded_count": unattributed_excluded,
            "future_interactions_excluded_count": future_interactions_excluded,
            "negative_interaction_durations_excluded_count": negative_duration_excluded,
            "bot_authored_prs_count": sum(1 for p in eligible_prs if p["is_bot_author"]),
        }

        return {
            "ttf_resp": ttf_resp,
            "ttfr_formal": ttfr_formal,
            "ttfr_inline": ttfr_inline,
            "responded_pr_ids": set(first_any_response_h.keys()),
            "hygiene": hygiene,
        }

    @staticmethod
    def _evaluate_outcomes(
        eligible_prs: List[Dict[str, Any]],
        future_prs_excluded: int,
    ) -> Dict[str, Any]:
        """Compute merged (via merged_at <= ref_time), closed-unmerged, and open counts & rates."""
        total = len(eligible_prs)
        if total == 0:
            return {
                "status": "empty",
                "total_eligible_prs": 0,
                "merged_count": 0,
                "closed_unmerged_count": 0,
                "open_count": 0,
                "resolved_count": 0,
                "merge_rate": None,
                "abandonment_rate": None,
                "open_ratio": None,
                "future_prs_excluded_count": future_prs_excluded,
                "limitations": "No eligible pull requests created up to analysis_reference_time.",
            }

        merged = sum(1 for p in eligible_prs if p["effective_state"] == "merged")
        closed_unmerged = sum(1 for p in eligible_prs if p["effective_state"] == "closed_unmerged")
        open_prs = sum(1 for p in eligible_prs if p["effective_state"] == "open")
        resolved = merged + closed_unmerged

        return {
            "status": "sufficient",
            "total_eligible_prs": total,
            "merged_count": merged,
            "closed_unmerged_count": closed_unmerged,
            "open_count": open_prs,
            "resolved_count": resolved,
            "merge_rate": round(merged / float(resolved), 4) if resolved > 0 else None,
            "abandonment_rate": round(closed_unmerged / float(resolved), 4) if resolved > 0 else None,
            "open_ratio": round(open_prs / float(total), 4),
            "future_prs_excluded_count": future_prs_excluded,
            "limitations": (
                "No resolved (merged or closed) PRs available yet to compute merge/abandonment rates."
                if resolved == 0
                else None
            ),
        }

    @staticmethod
    def _evaluate_merge_latency(eligible_prs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute merge latency quantiles preserving exact 0.0h merges and rejecting negative durations."""
        valid_merged_hours: List[float] = []
        zero_hour_count = 0
        negative_count = 0

        for p in eligible_prs:
            if p["effective_state"] != "merged" or p["merged_at"] is None:
                continue
            diff_sec = (p["merged_at"] - p["created_at"]).total_seconds()
            if diff_sec < 0.0:
                negative_count += 1
                continue
            hours = diff_sec / 3600.0
            if hours == 0.0:
                zero_hour_count += 1
            valid_merged_hours.append(hours)

        n_prs = len(eligible_prs)
        summary = _summarize_hours_sample(
            durations_hours=valid_merged_hours,
            min_required=5,
            confident_required=None,
            include_iqr=True,
            empty_status="empty" if n_prs == 0 else "insufficient",
            insufficient_reason=(
                "No eligible PRs observed."
                if n_prs == 0
                else (
                    f"Only {len(valid_merged_hours)} valid non-negative merged PR(s) observed "
                    "(minimum 5 required for merge latency quantiles)."
                )
            ),
        )
        summary["valid_merged_count"] = len(valid_merged_hours)
        summary["zero_hour_merges_count"] = zero_hour_count
        summary["invalid_records_count"] = negative_count
        summary["confident_sample_required"] = 5
        if negative_count > 0 and summary["limitations"] is None:
            summary["limitations"] = f"Excluded {negative_count} merged PR(s) with negative duration (merged_at < created_at)."
        elif negative_count > 0 and summary["limitations"]:
            summary["limitations"] = (
                f"{summary['limitations']} Excluded {negative_count} merged PR(s) with negative duration."
            )
        return summary

    @staticmethod
    def _evaluate_kaplan_meier_survival(
        eligible_prs: List[Dict[str, Any]],
        ref_time: datetime,
    ) -> Dict[str, Any]:
        """Deterministic Kaplan-Meier survival estimator (S(7d), S(30d), median) gated at >= 15 valid PRs."""
        min_required = 15
        records: List[Tuple[float, int]] = []  # (duration_days, event_observed: 1=closed/merged, 0=open censored)
        invalid_negative = 0
        missing_terminal_ts = 0

        for p in eligible_prs:
            st = p["effective_state"]
            if st in ("merged", "closed_unmerged"):
                t_dt = p["terminal_dt"]
                if t_dt is None:
                    missing_terminal_ts += 1
                    continue
                dur_sec = (t_dt - p["created_at"]).total_seconds()
                if dur_sec < 0.0:
                    invalid_negative += 1
                    continue
                records.append((dur_sec / 86400.0, 1))
            else:
                # Open PR right-censored at analysis_reference_time - created_at
                dur_sec = max(0.0, (ref_time - p["created_at"]).total_seconds())
                records.append((dur_sec / 86400.0, 0))

        n_valid = len(records)
        base_meta: Dict[str, Any] = {
            "sample_count": n_valid,
            "events_observed_count": sum(ev for _, ev in records),
            "right_censored_count": sum(1 - ev for _, ev in records),
            "min_sample_required": min_required,
            "invalid_records_count": invalid_negative + missing_terminal_ts,
            "negative_duration_count": invalid_negative,
            "missing_terminal_timestamp_count": missing_terminal_ts,
        }

        if len(eligible_prs) == 0 or n_valid == 0:
            return {
                **base_meta,
                "status": "empty",
                "s7d": None,
                "s30d": None,
                "s7d_within_followup": None,
                "s30d_within_followup": None,
                "median_survival_days": None,
                "median_survival_hours": None,
                "median_reached": False,
                "max_observed_time_days": None,
                "survival_curve": None,
                "limitations": "No valid pull requests available for Kaplan-Meier survival estimation.",
            }

        if n_valid < min_required:
            return {
                **base_meta,
                "status": "insufficient",
                "s7d": None,
                "s30d": None,
                "s7d_within_followup": None,
                "s30d_within_followup": None,
                "median_survival_days": None,
                "median_survival_hours": None,
                "median_reached": False,
                "max_observed_time_days": round(max(d for d, _ in records), 4),
                "survival_curve": None,
                "limitations": (
                    f"Observed {n_valid} valid PR(s); minimum {min_required} required per specification for Kaplan-Meier survival."
                ),
            }

        km = compute_kaplan_meier_curve(records=records, horizons_days=(7.0, 30.0))
        max_obs_days = km["max_observed_time_days"]
        surv = km["terminal_survival"]
        median_days = km["median_days"]

        notes: List[str] = []
        if base_meta["invalid_records_count"] > 0:
            notes.append(f"Excluded {base_meta['invalid_records_count']} PR(s) with negative/missing terminal durations.")
        if base_meta["right_censored_count"] > 0 and surv > 0.0 and max_obs_days < 30.0:
            notes.append(
                f"Maximum observed follow-up is {round(max_obs_days, 1)} days with open PRs still active; S(30d) reflects constant hazard-free tail beyond last observation."
            )

        return {
            **base_meta,
            "status": "sufficient",
            "s7d": km["horizon_survival"][7.0],
            "s30d": km["horizon_survival"][30.0],
            "s7d_within_followup": km["within_followup"][7.0],
            "s30d_within_followup": km["within_followup"][30.0],
            "median_survival_days": round(median_days, 4) if median_days is not None else None,
            "median_survival_hours": round(median_days * 24.0, 2) if median_days is not None else None,
            "median_reached": median_days is not None,
            "max_observed_time_days": round(max_obs_days, 4),
            "survival_curve": km["curve_steps"],
            "limitations": " ".join(notes) if notes else None,
        }

    @staticmethod
    def _evaluate_size_vs_turnaround(eligible_prs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate PR size bins vs. turnaround latency strictly distinguishing NULL diffs from 0 lines."""
        min_closed_with_diff = 10
        n_total = len(eligible_prs)

        prs_with_diff = 0
        prs_zero_diff = 0
        prs_missing_diff = 0
        closed_with_diff_records: List[Dict[str, Any]] = []
        review_intensities: List[int] = []

        for p in eligible_prs:
            # Review intensity (comments + inline comments on PR metadata)
            if p["comments_count"] is not None or p["review_comments_count"] is not None:
                review_intensities.append(int(p["comments_count"] or 0) + int(p["review_comments_count"] or 0))

            adds = p["additions"]
            dels = p["deletions"]
            if adds is None or dels is None:
                prs_missing_diff += 1
                continue

            size_lines = max(0, int(adds)) + max(0, int(dels))
            prs_with_diff += 1
            if size_lines == 0:
                prs_zero_diff += 1

            if p["effective_state"] in ("merged", "closed_unmerged") and p["terminal_dt"] is not None:
                dur_sec = (p["terminal_dt"] - p["created_at"]).total_seconds()
                if dur_sec >= 0.0:
                    closed_with_diff_records.append({
                        "size_lines": size_lines,
                        "is_merged": p["effective_state"] == "merged",
                        "turnaround_hours": dur_sec / 3600.0,
                    })

        diff_cov = round(prs_with_diff / float(n_total), 4) if n_total > 0 else None
        review_intensity_summary = {
            "sample_count": len(review_intensities),
            "mean_comments_per_pr": round(statistics.mean(review_intensities), 2) if review_intensities else None,
            "median_comments_per_pr": round(statistics.median(review_intensities), 2) if review_intensities else None,
        }

        base_meta: Dict[str, Any] = {
            "eligible_prs_count": n_total,
            "prs_with_diff_count": prs_with_diff,
            "prs_zero_line_diff_count": prs_zero_diff,
            "prs_missing_diff_count": prs_missing_diff,
            "diff_coverage_ratio": diff_cov,
            "closed_prs_with_diff_count": len(closed_with_diff_records),
            "min_closed_with_diff_required": min_closed_with_diff,
            "review_intensity": review_intensity_summary,
        }

        if n_total == 0:
            return {
                **base_meta,
                "status": "empty",
                "bins": None,
                "limitations": "No eligible PRs observed.",
            }

        if prs_with_diff == 0:
            return {
                **base_meta,
                "status": "unavailable_no_diff_metrics",
                "bins": None,
                "limitations": "PR diff statistics (additions/deletions) are NULL across all eligible PRs; NULL values are never coerced to zero.",
            }

        if len(closed_with_diff_records) < min_closed_with_diff:
            return {
                **base_meta,
                "status": "insufficient",
                "bins": None,
                "limitations": (
                    f"Only {len(closed_with_diff_records)} resolved PR(s) carry non-null diff stats "
                    f"(minimum {min_closed_with_diff} required for size-bin turnaround analysis)."
                ),
            }

        bins_out: Dict[str, Dict[str, Any]] = {}
        n_closed_diff = len(closed_with_diff_records)
        for bin_name, lo, hi in PR_SIZE_BINS:
            matching = [
                rec for rec in closed_with_diff_records
                if rec["size_lines"] >= lo and (hi is None or rec["size_lines"] <= hi)
            ]
            merged_latencies = [rec["turnaround_hours"] for rec in matching if rec["is_merged"]]
            all_latencies = [rec["turnaround_hours"] for rec in matching]
            bins_out[bin_name] = {
                "line_range": f"{lo}..{hi}" if hi is not None else f">{lo - 1}",
                "pr_count": len(matching),
                "share": round(len(matching) / float(n_closed_diff), 4),
                "merged_count": len(merged_latencies),
                "median_merge_latency_hours": (
                    round(statistics.median(merged_latencies), 2) if merged_latencies else None
                ),
                "median_turnaround_hours": (
                    round(statistics.median(all_latencies), 2) if all_latencies else None
                ),
            }

        limitation = (
            f"Partial PR diff coverage ({round((diff_cov or 0.0) * 100.0, 1)}% of eligible PRs have additions/deletions)."
            if (diff_cov is not None and diff_cov < 1.0)
            else None
        )

        return {
            **base_meta,
            "status": "sufficient",
            "bins": bins_out,
            "limitations": limitation,
        }

    def _evaluate_bottlenecks(
        self,
        eligible_prs: List[Dict[str, Any]],
        responsiveness: Dict[str, Any],
        merge_latency: Dict[str, Any],
        reviews_rows: List[Any],
        bot_github_ids: Set[int],
        bot_logins: Set[str],
        ref_time: datetime,
        collection_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute review latency share, human reviewer load distribution, and unreviewed open PR age."""
        n_prs = len(eligible_prs)
        if n_prs == 0:
            return {
                "status": "empty",
                "review_latency_share": None,
                "review_latency_share_status": "empty",
                "maintainer_review_load": {
                    "active_human_reviewers_count": 0,
                    "total_human_formal_reviews": 0,
                    "top1_reviewer_share": None,
                    "top_reviewers": [],
                },
                "unreviewed_open_prs": {
                    "open_prs_count": 0,
                    "unreviewed_open_prs_count": 0,
                    "unreviewed_open_share": None,
                    "median_age_hours": None,
                    "median_age_days": None,
                },
                "limitations": "No eligible PRs observed.",
            }

        # 1. Review latency share beta = Median(TTFResp) / Median(Merge Latency)
        med_ttfr = responsiveness["ttf_resp"]["p50_hours"]
        med_merge = merge_latency["p50_hours"]
        ttfr_ok = responsiveness["ttf_resp"]["status"] in ("limited", "sufficient")
        merge_ok = merge_latency["status"] == "sufficient"

        if ttfr_ok and merge_ok and med_ttfr is not None and med_merge is not None:
            if med_merge > 0.0:
                beta_review = round(med_ttfr / med_merge, 4)
                beta_status = "computed"
            else:
                beta_review = None
                beta_status = "undefined_zero_median_merge_latency"
        else:
            beta_review = None
            beta_status = "insufficient_ttfr_or_merge_sample"

        # 2. Maintainer formal review load (human non-author formal reviews <= ref_time)
        pr_by_id: Dict[int, Dict[str, Any]] = {p["id"]: p for p in eligible_prs}
        reviewer_counts: Dict[str, Dict[str, Any]] = {}
        seen_reviews: Set[int] = set()
        total_human_reviews = 0

        for rev in reviews_rows:
            if rev.review_id in seen_reviews:
                continue
            seen_reviews.add(rev.review_id)
            pr = pr_by_id.get(rev.pull_request_id)
            if pr is None:
                continue
            dt = _ensure_utc(rev.submitted_at)
            if dt is None or dt > ref_time:
                continue
            if (rev.state or "").upper().strip() not in FORMAL_REVIEW_STATES:
                continue
            if rev.author_id is None and not (rev.author and rev.author.strip()):
                continue
            if self._is_self_interaction(rev.author_id, rev.author, pr["author_id"], pr["author"]):
                continue
            if self._is_bot(rev.author_id, rev.author, bot_github_ids, bot_logins):
                continue

            key = (
                f"gh_id:{rev.author_id}"
                if rev.author_id is not None
                else f"login:{rev.author.strip().lower()}"
            )
            label = rev.author or f"github_id:{rev.author_id}"
            if key not in reviewer_counts:
                reviewer_counts[key] = {"reviewer": label, "reviews_count": 0}
            reviewer_counts[key]["reviews_count"] += 1
            total_human_reviews += 1

        sorted_reviewers = sorted(
            reviewer_counts.values(),
            key=lambda x: (-x["reviews_count"], str(x["reviewer"])),
        )
        top_reviewers = [
            {
                "reviewer": r["reviewer"],
                "formal_reviews_count": r["reviews_count"],
                "share": round(r["reviews_count"] / float(total_human_reviews), 4),
            }
            for r in sorted_reviewers[:10]
        ]
        top1_rev_share = top_reviewers[0]["share"] if top_reviewers else None

        # 3. Unreviewed open PR age (open PRs as of ref_time with 0 external human responses)
        responded_ids: Set[int] = responsiveness["responded_pr_ids"]
        open_prs = [p for p in eligible_prs if p["effective_state"] == "open"]
        unreviewed_open_ages_h: List[float] = []
        for p in open_prs:
            if p["id"] not in responded_ids:
                age_h = max(0.0, (ref_time - p["created_at"]).total_seconds() / 3600.0)
                unreviewed_open_ages_h.append(age_h)

        n_open = len(open_prs)
        n_unreviewed_open = len(unreviewed_open_ages_h)
        med_unrev_h = round(statistics.median(unreviewed_open_ages_h), 2) if unreviewed_open_ages_h else None
        med_unrev_d = round(med_unrev_h / 24.0, 2) if med_unrev_h is not None else None

        notes: List[str] = []
        if beta_review is not None and beta_review > 1.0:
            notes.append(
                "Median TTFResp exceeds median merge latency because unreviewed/trivial merges and reviewed PRs form distinct subsets."
            )
        if collection_scope["inline_comments_skipped_by_mode"] or collection_scope["reviews_not_collected"]:
            notes.append(
                "Unreviewed status and reviewer load reflect only the interaction streams collected in this run."
            )

        return {
            "status": "sufficient" if (beta_review is not None or total_human_reviews > 0 or n_open > 0) else "limited",
            "review_latency_share": beta_review,
            "review_latency_share_status": beta_status,
            "maintainer_review_load": {
                "active_human_reviewers_count": len(sorted_reviewers),
                "total_human_formal_reviews": total_human_reviews,
                "top1_reviewer_share": top1_rev_share,
                "top_reviewers": top_reviewers,
            },
            "unreviewed_open_prs": {
                "open_prs_count": n_open,
                "unreviewed_open_prs_count": n_unreviewed_open,
                "unreviewed_open_share": round(n_unreviewed_open / float(n_open), 4) if n_open > 0 else None,
                "median_age_hours": med_unrev_h,
                "median_age_days": med_unrev_d,
            },
            "limitations": " ".join(notes) if notes else None,
        }

    def _evaluate_coverage_and_confidence(
        self,
        eligible_pr_count: int,
        collection_scope: Dict[str, Any],
        responsiveness: Dict[str, Any],
        merge_latency: Dict[str, Any],
        outcomes: Dict[str, Any],
        survival: Dict[str, Any],
        size_impact: Dict[str, Any],
        bottlenecks: Dict[str, Any],
    ) -> Tuple[MetricCoverageItem, float, List[Dict[str, Any]]]:
        """Evaluate MetricCoverageItem and confidence penalties across PR subdomains."""
        min_sample = 10
        penalties: List[Dict[str, Any]] = []
        penalty_total = 0.0

        if eligible_pr_count == 0:
            cov_status = "empty"
            p = 0.5
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-EMPTY-PULL-REQUESTS",
                "penalty": p,
                "reason": "Repository snapshot has zero eligible pull requests.",
            })
        elif eligible_pr_count < 3:
            cov_status = "insufficient"
            p = 0.4
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-INSUFFICIENT-PULL-REQUESTS",
                "penalty": p,
                "reason": f"Observed eligible PRs ({eligible_pr_count}) is below minimum 3 required for basic response metrics.",
            })
        elif eligible_pr_count < min_sample:
            cov_status = "degraded"
            p = 0.2
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-SMALL-PR-SAMPLE",
                "penalty": p,
                "reason": f"Observed eligible PRs ({eligible_pr_count}) is below high-confidence threshold ({min_sample}).",
            })
        else:
            cov_status = "sufficient"

        if eligible_pr_count > 0 and collection_scope["is_bounded_or_partial"]:
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-BOUNDED-PR-SAMPLE",
                "penalty": p,
                "reason": f"Collection mode '{collection_scope['mode']}' harvests bounded PR and review samples.",
            })

        if eligible_pr_count > 0 and (
            collection_scope["inline_comments_skipped_by_mode"]
            or collection_scope["reviews_not_collected"]
        ):
            p = 0.05
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-INCOMPLETE-REVIEW-STREAMS",
                "penalty": p,
                "reason": "One or more review interaction streams (inline comments or formal reviews) were skipped/uncollected.",
            })

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=cov_status,
            sample_size=eligible_pr_count,
            min_sample_required=min_sample,
            details={
                "eligible_prs_count": eligible_pr_count,
                "responded_prs_count": responsiveness["ttf_resp"]["sample_count"],
                "valid_merged_prs_count": merge_latency["valid_merged_count"],
                "diff_coverage_ratio": size_impact["diff_coverage_ratio"],
                "collection_mode": collection_scope["mode"],
                "subdomains": {
                    "ttf_resp": responsiveness["ttf_resp"]["status"],
                    "ttfr_formal": responsiveness["ttfr_formal"]["status"],
                    "ttfr_inline": responsiveness["ttfr_inline"]["status"],
                    "merge_latency": merge_latency["status"],
                    "outcome_distribution": outcomes["status"],
                    "survival_km": survival["status"],
                    "size_vs_turnaround": size_impact["status"],
                    "review_bottlenecks": bottlenecks["status"],
                },
            },
        )

        return coverage, round(min(1.0, penalty_total), 3), penalties

    def _build_rules(
        self,
        eligible_pr_count: int,
        collection_scope: Dict[str, Any],
        responsiveness: Dict[str, Any],
        outcomes: Dict[str, Any],
        merge_latency: Dict[str, Any],
        survival: Dict[str, Any],
        size_impact: Dict[str, Any],
        bottlenecks: Dict[str, Any],
    ) -> List[RuleInterpretation]:
        """Generate deterministic RULE-PR-* findings anchored to verified metrics."""
        rules: List[RuleInterpretation] = []

        if eligible_pr_count == 0:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-PR-EMPTY",
                    domain=self.domain_name,
                    severity="info",
                    finding="No Pull Requests Observed: Repository snapshot contains zero eligible pull requests.",
                    evidence={"eligible_prs_count": 0},
                    recommendation="Ingest pull requests to evaluate code review responsiveness, merge latency, and PR survival.",
                )
            )
            return rules

        ttf_resp = responsiveness["ttf_resp"]
        if ttf_resp["status"] in ("limited", "sufficient") and ttf_resp["p50_hours"] is not None:
            med_ttfr = ttf_resp["p50_hours"]
            if med_ttfr > 72.0:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-PR-001",
                        domain=self.domain_name,
                        severity="warning",
                        finding=f"Slow PR Review Turnaround: Median time-to-first-response is {med_ttfr} hours (> 72h) across {ttf_resp['sample_count']} responded PRs.",
                        evidence={
                            "p50_hours": med_ttfr,
                            "p90_hours": ttf_resp["p90_hours"],
                            "sample_count": ttf_resp["sample_count"],
                        },
                        recommendation="Consider automated reviewer assignment or triage SLAs to shorten initial PR feedback delays.",
                    )
                )
            elif ttf_resp["status"] == "sufficient" and med_ttfr <= 12.0:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-PR-001",
                        domain=self.domain_name,
                        severity="positive",
                        finding=f"Responsive Review Culture: Maintainers deliver initial non-author PR feedback in a median of {med_ttfr} hours (<= 12h).",
                        evidence={
                            "p50_hours": med_ttfr,
                            "p25_hours": ttf_resp["p25_hours"],
                            "sample_count": ttf_resp["sample_count"],
                        },
                        recommendation=None,
                    )
                )

        unrev = bottlenecks["unreviewed_open_prs"]
        if (
            unrev["open_prs_count"] >= 3
            and unrev["unreviewed_open_share"] is not None
            and unrev["unreviewed_open_share"] >= 0.50
            and unrev["median_age_hours"] is not None
            and unrev["median_age_hours"] > 72.0
        ):
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-PR-002",
                    domain=self.domain_name,
                    severity="warning",
                    finding=(
                        f"Unreviewed Open PR Backlog: {unrev['unreviewed_open_prs_count']} of {unrev['open_prs_count']} open PRs "
                        f"lack non-author human feedback (median age {unrev['median_age_days']} days)."
                    ),
                    evidence=unrev,
                    recommendation="Prioritize initial triage on aging unreviewed PRs before accepting new feature queues.",
                )
            )
        else:
            rev_load = bottlenecks["maintainer_review_load"]
            if (
                rev_load["total_human_formal_reviews"] >= 10
                and rev_load["top1_reviewer_share"] is not None
                and rev_load["top1_reviewer_share"] >= 0.60
            ):
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-PR-002",
                        domain=self.domain_name,
                        severity="warning",
                        finding=(
                            f"Review Bottleneck Concentration: Top reviewer accounts for "
                            f"{round(rev_load['top1_reviewer_share'] * 100.0, 1)}% of {rev_load['total_human_formal_reviews']} observed formal reviews."
                        ),
                        evidence=rev_load,
                        recommendation="Distribute formal PR review ownership across additional maintainers.",
                    )
                )

        if outcomes["resolved_count"] >= 10 and outcomes["abandonment_rate"] is not None and outcomes["abandonment_rate"] >= 0.35:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-PR-003",
                    domain=self.domain_name,
                    severity="warning",
                    finding=(
                        f"Elevated PR Abandonment Rate: {round(outcomes['abandonment_rate'] * 100.0, 1)}% "
                        f"({outcomes['closed_unmerged_count']} of {outcomes['resolved_count']}) closed PRs were closed without merging."
                    ),
                    evidence={
                        "abandonment_rate": outcomes["abandonment_rate"],
                        "merge_rate": outcomes["merge_rate"],
                        "resolved_count": outcomes["resolved_count"],
                    },
                    recommendation=None,
                )
            )

        if size_impact["status"] == "sufficient" and size_impact["bins"]:
            small_bin = size_impact["bins"]["small"]
            xl_bin = size_impact["bins"]["xl"]
            small_med = small_bin["median_merge_latency_hours"]
            xl_med = xl_bin["median_merge_latency_hours"]
            if (
                small_bin["merged_count"] >= 2
                and xl_bin["merged_count"] >= 2
                and small_med is not None
                and small_med > 0.0
                and xl_med is not None
                and xl_med >= 3.0 * small_med
            ):
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-PR-004",
                        domain=self.domain_name,
                        severity="info",
                        finding=(
                            f"PR Size Latency Escalation: XL PRs (>1000 lines) take {xl_med}h median to merge vs "
                            f"{small_med}h for small PRs (<=50 lines)."
                        ),
                        evidence={
                            "small_median_hours": small_med,
                            "xl_median_hours": xl_med,
                            "small_merged_count": small_bin["merged_count"],
                            "xl_merged_count": xl_bin["merged_count"],
                            "diff_coverage_ratio": size_impact["diff_coverage_ratio"],
                        },
                        recommendation="Encouraging smaller, focused pull requests may help lower review turnaround.",
                    )
                )

        hygiene = responsiveness["hygiene"]
        pr_neg = max(merge_latency["invalid_records_count"], survival["negative_duration_count"])
        total_neg = hygiene["negative_interaction_durations_excluded_count"] + pr_neg
        if (
            collection_scope["inline_comments_skipped_by_mode"]
            or collection_scope["reviews_not_collected"]
            or total_neg > 0
        ):
            parts: List[str] = []
            if collection_scope["inline_comments_skipped_by_mode"]:
                parts.append(f"inline diff comments skipped under '{collection_scope['mode']}'")
            if collection_scope["reviews_not_collected"]:
                parts.append("formal review stream uncollected")
            if total_neg > 0:
                parts.append(f"{total_neg} negative-duration timestamp anomalies excluded")
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-PR-005",
                    domain=self.domain_name,
                    severity="info",
                    finding=f"PR Coverage & Data Hygiene Notes: {'; '.join(parts)}.",
                    evidence={
                        "inline_comments_skipped_by_mode": collection_scope["inline_comments_skipped_by_mode"],
                        "reviews_not_collected": collection_scope["reviews_not_collected"],
                        "negative_durations_excluded": total_neg,
                        "bot_interactions_excluded": hygiene["bot_interactions_excluded_count"],
                        "self_interactions_excluded": hygiene["self_interactions_excluded_count"],
                    },
                    recommendation=None,
                )
            )

        return rules
