"""Issue & Maintenance Analytics Engine for OpenSourceLens (Module 4 / Phase C.5).

Implements IssueIntelligenceCalculator adhering to BaseMetricCalculator:
- Observed calendar-month creation vs. closure flow (net & cumulative flow differentials, flow ratio).
- Strict separation of observed sample flow from GitHub-reported combined backlog stock
  (combined_open_observed_to_reported_ratio_percentage; pure issue coverage stays None; no last_synced_at).
- Non-parametric resolution latency distributions (P25, P50, P75, P90, IQR, un-truncated mean, skewness, speed tiers)
  derived directly from timestamps, preserving 0.0h instant closures and isolating negative durations.
- Deterministic Kaplan-Meier issue survival analysis (S(30d), S(60d), S(90d), median) gated at >= 15 issues.
- Five-bin continuous-time backlog aging (Fresh, Active, Aging, Stale, Dormant) and >90d stale fraction.
- Discussion intensity (primary Issue.comments_count, supplemental IssueComment detail, zero-variance Spearman).
- Canonical category breakdown keeping 'unlabeled' strictly distinct from 'other'.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from scipy import stats as scipy_stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    CollectionRun,
    Issue,
    IssueComment,
    IssueLabel,
    PullRequest,
    Repository,
)
from backend.app.schemas.analytics_v2 import MetricCoverageItem, RuleInterpretation
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.development_activity import (
    _ensure_utc,
    _linear_percentile,
)
from backend.app.services.analytics.pull_requests import (
    _extract_stream_status,
    compute_kaplan_meier_curve,
)


SPEED_TIER_ORDER: Tuple[str, ...] = (
    "< 1 Day",
    "1-7 Days",
    "7-30 Days",
    "30+ Days",
)


def _derive_speed_tier(resolution_hours: float) -> str:
    """Classify non-negative resolution hours using the repository taxonomy."""
    days = resolution_hours / 24.0
    if days < 1.0:
        return "< 1 Day"
    if days <= 7.0:
        return "1-7 Days"
    if days <= 30.0:
        return "7-30 Days"
    return "30+ Days"


def _classify_backlog_age_bin(age_days: float) -> str:
    """Assign continuous open-issue age (days >= 0) into exhaustive mutually-exclusive bins."""
    if age_days < 7.0:
        return "fresh"
    if age_days <= 30.0:
        return "active"
    if age_days <= 90.0:
        return "aging"
    if age_days <= 365.0:
        return "stale"
    return "dormant"


def _compute_safe_skewness(values: List[float]) -> float:
    """Compute finite sample skewness with zero-variance protection."""
    if len(values) < 3:
        return 0.0
    stdev = statistics.stdev(values)
    if stdev <= 1e-12:
        return 0.0
    raw = float(scipy_stats.skew(values, bias=False))
    if not math.isfinite(raw):
        return 0.0
    return round(raw, 4)


class IssueIntelligenceCalculator(BaseMetricCalculator):
    """Calculates issue flow, backlog stock separation, SLAs, survival, aging, discussion, and categories."""

    @property
    def domain_name(self) -> str:
        return "issue_intelligence"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute issue and maintenance metrics anchored to frozen analysis_reference_time."""
        ref_time = _ensure_utc(analysis_reference_time)

        # 1. Fetch repository open backlog count ONLY (explicitly exclude last_synced_at per v1.8)
        reported_open_count = db.scalar(
            select(Repository.open_issues_count).where(Repository.id == repository_id)
        )

        # 2. Fetch underlying issues, labels, issue comments, PR counts, and collection scope
        issues_rows = db.execute(
            select(
                Issue.id,
                Issue.issue_number,
                Issue.state,
                Issue.is_unlabeled,
                Issue.canonical_label,
                Issue.created_at,
                Issue.closed_at,
                Issue.comments_count,
            ).where(Issue.repository_id == repository_id)
        ).fetchall()

        prs_rows = db.execute(
            select(
                PullRequest.pr_number,
                PullRequest.state,
                PullRequest.created_at,
                PullRequest.closed_at,
                PullRequest.merged_at,
            ).where(PullRequest.repository_id == repository_id)
        ).fetchall()

        latest_collection_run = db.scalar(
            select(CollectionRun)
            .where(CollectionRun.repository_id == repository_id)
            .order_by(CollectionRun.started_at.desc(), CollectionRun.id.desc())
            .limit(1)
        )

        # 3. Filter issues created <= ref_time & normalize temporal states (ignoring pre-aggregated SLAs)
        eligible_issue_ids: List[int] = []
        raw_eligible: List[Dict[str, Any]] = []
        seen_issue_numbers: Set[int] = set()
        future_issues_excluded = 0

        for row in issues_rows:
            if row.issue_number in seen_issue_numbers:
                continue
            created_dt = _ensure_utc(row.created_at)
            if created_dt is None or created_dt > ref_time:
                future_issues_excluded += 1
                continue
            seen_issue_numbers.add(row.issue_number)

            closed_dt = _ensure_utc(row.closed_at)
            raw_state = (row.state or "open").lower().strip()

            # Issues closed AFTER ref_time were open & active at ref_time
            if closed_dt is not None and closed_dt <= ref_time:
                effective_state = "closed"
                terminal_dt: Optional[datetime] = closed_dt
                missing_closed_ts = False
            elif closed_dt is not None and closed_dt > ref_time:
                effective_state = "open"
                terminal_dt = None
                missing_closed_ts = False
            elif raw_state == "closed":
                effective_state = "closed"
                terminal_dt = None
                missing_closed_ts = True
            else:
                effective_state = "open"
                terminal_dt = None
                missing_closed_ts = False

            eligible_issue_ids.append(row.id)
            raw_eligible.append({
                "id": row.id,
                "issue_number": row.issue_number,
                "created_at": created_dt,
                "closed_at": terminal_dt,
                "effective_state": effective_state,
                "missing_closed_ts": missing_closed_ts,
                "is_unlabeled": bool(row.is_unlabeled),
                "raw_canonical_label": row.canonical_label,
                "comments_count": row.comments_count,
            })

        # 4. Query labels & IssueComment supplemental rows only for eligible issues
        label_rows: List[Any] = []
        comment_rows: List[Any] = []
        if eligible_issue_ids:
            label_rows = db.execute(
                select(
                    IssueLabel.issue_id,
                    IssueLabel.label_name,
                    IssueLabel.is_canonical,
                    IssueLabel.canonical_category,
                ).where(IssueLabel.issue_id.in_(eligible_issue_ids))
            ).fetchall()

            comment_rows = db.execute(
                select(
                    IssueComment.issue_id,
                    IssueComment.comment_id,
                    IssueComment.author,
                    IssueComment.author_id,
                    IssueComment.created_at,
                ).where(
                    IssueComment.repository_id == repository_id,
                    IssueComment.issue_id.isnot(None),
                    IssueComment.issue_id.in_(eligible_issue_ids),
                )
            ).fetchall()

        labels_by_issue: Dict[int, List[str]] = defaultdict(list)
        for lr in label_rows:
            labels_by_issue[lr.issue_id].append(lr.label_name)

        # Keep 'unlabeled' strictly distinct from labeled 'other'
        for item in raw_eligible:
            raw_cat = (item["raw_canonical_label"] or "").strip().lower() or "other"
            has_labels = item["id"] in labels_by_issue
            if raw_cat == "unlabeled" or (item["is_unlabeled"] and not has_labels and raw_cat == "other"):
                item["effective_category"] = "unlabeled"
            else:
                item["effective_category"] = raw_cat

        # 5. Anchor PR open/closed stock at ref_time (purely for combined stock coverage comparison)
        pr_open_count, pr_closed_count = self._count_prs_at_ref_time(prs_rows, ref_time)

        # 6. Inspect collection scope metadata
        collection_scope = self._inspect_collection_scope(
            latest_run=latest_collection_run,
            comment_rows_count=len(comment_rows),
        )

        # 7. Subdomain computations
        monthly_flow = self._evaluate_monthly_flow(
            eligible_issues=raw_eligible,
            future_issues_excluded=future_issues_excluded,
        )
        backlog_stock = self._evaluate_backlog_stock(
            eligible_issues=raw_eligible,
            reported_open_count=reported_open_count,
            pr_open_count=pr_open_count,
            pr_closed_count=pr_closed_count,
        )
        resolution = self._evaluate_resolution_distribution(eligible_issues=raw_eligible)
        survival = self._evaluate_kaplan_meier_survival(
            eligible_issues=raw_eligible,
            ref_time=ref_time,
        )
        aging = self._evaluate_backlog_aging(
            eligible_issues=raw_eligible,
            ref_time=ref_time,
        )
        discussion = self._evaluate_discussion_intensity(
            eligible_issues=raw_eligible,
            comment_rows=comment_rows,
            ref_time=ref_time,
            collection_scope=collection_scope,
        )
        categories = self._evaluate_canonical_categories(
            eligible_issues=raw_eligible,
            label_rows=label_rows,
        )

        # 8. Overall domain coverage, confidence penalties, and deterministic rules
        coverage, confidence_penalty, penalties = self._evaluate_coverage_and_confidence(
            eligible_issues=raw_eligible,
            collection_scope=collection_scope,
            monthly_flow=monthly_flow,
            backlog_stock=backlog_stock,
            resolution=resolution,
            survival=survival,
            aging=aging,
            discussion=discussion,
            categories=categories,
        )

        rules = self._build_rules(
            eligible_issues=raw_eligible,
            monthly_flow=monthly_flow,
            backlog_stock=backlog_stock,
            resolution=resolution,
            aging=aging,
            categories=categories,
        )

        metrics: Dict[str, Any] = {
            "scope": collection_scope,
            "observed_flow": monthly_flow,
            "backlog_stock_summary": backlog_stock,
            "resolution_distribution": resolution,
            "survival_km": survival,
            "backlog_aging": aging,
            "discussion_intensity": discussion,
            "canonical_categories": categories,
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
    def _count_prs_at_ref_time(prs_rows: List[Any], ref_time: datetime) -> Tuple[int, int]:
        """Count open and closed/merged harvested PRs strictly as of ref_time."""
        seen_nums: Set[int] = set()
        open_prs = 0
        closed_prs = 0
        for row in prs_rows:
            if row.pr_number in seen_nums:
                continue
            created_dt = _ensure_utc(row.created_at)
            if created_dt is None or created_dt > ref_time:
                continue
            seen_nums.add(row.pr_number)

            merged_dt = _ensure_utc(row.merged_at)
            closed_dt = _ensure_utc(row.closed_at)
            raw_state = (row.state or "open").lower().strip()

            is_closed_at_ref = (
                (merged_dt is not None and merged_dt <= ref_time)
                or (closed_dt is not None and closed_dt <= ref_time)
                or (raw_state == "closed" and merged_dt is None and closed_dt is None)
            )
            if is_closed_at_ref:
                closed_prs += 1
            else:
                open_prs += 1
        return open_prs, closed_prs

    @staticmethod
    def _inspect_collection_scope(
        latest_run: Optional[CollectionRun],
        comment_rows_count: int,
    ) -> Dict[str, Any]:
        """Inspect latest CollectionRun scope metadata without touching Repository.last_synced_at."""
        if latest_run is None:
            return {
                "mode": "unknown_unverified",
                "collection_status": None,
                "is_partial": True,
                "is_bounded_or_partial": True,
                "issues_stream_status": None,
                "issue_comments_stream_status": None,
                "comment_rows_available_count": comment_rows_count,
                "notes": "No CollectionRun record found; metrics reflect currently stored database issue records.",
            }

        mode = (latest_run.mode or "unknown").lower().strip()
        cov = latest_run.coverage_summary if isinstance(latest_run.coverage_summary, dict) else {}
        iss_status = _extract_stream_status(cov, "issues")
        ic_status = _extract_stream_status(cov, "issue_comments")
        bounded_modes = {"fast_snapshot", "standard_diagnostic"}
        stream_truncated = any(
            st in ("truncated", "partial", "rate_limited") for st in (iss_status, ic_status)
        )
        is_bounded_or_partial = (
            bool(latest_run.is_partial)
            or mode in bounded_modes
            or mode != "full_history"
            or latest_run.status != "completed"
            or stream_truncated
        )

        notes: List[str] = []
        if mode in bounded_modes:
            notes.append(f"Snapshot ingested under bounded mode '{mode}'.")
        if stream_truncated:
            notes.append("Issue or comment ingestion stream reported partial/truncated coverage.")
        if ic_status in ("skipped", "failed", "pending") and comment_rows_count == 0:
            notes.append("Detailed IssueComment rows were not collected; discussion uses Issue.comments_count.")

        return {
            "mode": mode,
            "collection_status": latest_run.status,
            "is_partial": bool(latest_run.is_partial),
            "is_bounded_or_partial": is_bounded_or_partial,
            "issues_stream_status": iss_status,
            "issue_comments_stream_status": ic_status,
            "comment_rows_available_count": comment_rows_count,
            "notes": " ".join(notes) if notes else "Verified complete collection run.",
        }

    @staticmethod
    def _evaluate_monthly_flow(
        eligible_issues: List[Dict[str, Any]],
        future_issues_excluded: int,
    ) -> Dict[str, Any]:
        """Compute monthly issue creation vs closure flow (>= 10 eligible issues required)."""
        min_required = 10
        n_total = len(eligible_issues)
        created_by_ym: Dict[Tuple[int, int], int] = defaultdict(int)
        closed_by_ym: Dict[Tuple[int, int], int] = defaultdict(int)
        all_months: Set[Tuple[int, int]] = set()

        for iss in eligible_issues:
            c_ym = (iss["created_at"].year, iss["created_at"].month)
            created_by_ym[c_ym] += 1
            all_months.add(c_ym)
            if iss["effective_state"] == "closed" and iss["closed_at"] is not None:
                cl_ym = (iss["closed_at"].year, iss["closed_at"].month)
                closed_by_ym[cl_ym] += 1
                all_months.add(cl_ym)

        total_created = sum(created_by_ym.values())
        total_closed = sum(closed_by_ym.values())

        if n_total == 0:
            return {
                "status": "empty",
                "eligible_issues_count": 0,
                "future_issues_excluded_count": future_issues_excluded,
                "min_sample_required": min_required,
                "total_created_observed": 0,
                "total_closed_observed": 0,
                "net_flow_differential": None,
                "flow_ratio": None,
                "monthly_series": None,
                "scope_note": "Observed sample flow only; never represents true repository backlog stock.",
                "limitations": "No eligible issues created up to analysis_reference_time.",
            }

        if n_total < min_required:
            return {
                "status": "insufficient",
                "eligible_issues_count": n_total,
                "future_issues_excluded_count": future_issues_excluded,
                "min_sample_required": min_required,
                "total_created_observed": total_created,
                "total_closed_observed": total_closed,
                "net_flow_differential": None,
                "flow_ratio": None,
                "monthly_series": None,
                "scope_note": "Observed sample flow only; never represents true repository backlog stock.",
                "limitations": (
                    f"Observed {n_total} eligible issue(s); minimum {min_required} required per v1.1 §4.1 for monthly flow."
                ),
            }

        min_ym = min(all_months)
        max_ym = max(all_months)
        cur_y, cur_m = min_ym
        series: List[Dict[str, Any]] = []
        cum_diff = 0

        while (cur_y, cur_m) <= max_ym:
            c_cnt = created_by_ym.get((cur_y, cur_m), 0)
            cl_cnt = closed_by_ym.get((cur_y, cur_m), 0)
            net = c_cnt - cl_cnt
            cum_diff += net
            series.append({
                "month": f"{cur_y:04d}-{cur_m:02d}",
                "created_count": c_cnt,
                "closed_count": cl_cnt,
                "net_flow_differential": net,
                "cumulative_flow_differential": cum_diff,
            })
            cur_m += 1
            if cur_m > 12:
                cur_m = 1
                cur_y += 1

        flow_ratio = round(total_closed / float(total_created), 4) if total_created > 0 else None

        return {
            "status": "sufficient",
            "eligible_issues_count": n_total,
            "future_issues_excluded_count": future_issues_excluded,
            "min_sample_required": min_required,
            "total_created_observed": total_created,
            "total_closed_observed": total_closed,
            "net_flow_differential": total_created - total_closed,
            "flow_ratio": flow_ratio,
            "monthly_series": series,
            "scope_note": (
                "F_cum(m) and flow_ratio measure net flow within the observed sample, NOT total repository backlog stock."
            ),
            "limitations": None,
        }

    @staticmethod
    def _evaluate_backlog_stock(
        eligible_issues: List[Dict[str, Any]],
        reported_open_count: Optional[int],
        pr_open_count: int,
        pr_closed_count: int,
    ) -> Dict[str, Any]:
        """Separate GitHub combined reported backlog stock from harvested local open/closed issue & PR samples."""
        open_issues = sum(1 for i in eligible_issues if i["effective_state"] == "open")
        closed_issues = sum(1 for i in eligible_issues if i["effective_state"] == "closed")
        combined_open_local = open_issues + pr_open_count

        if reported_open_count is not None and reported_open_count > 0:
            ratio_pct = round((combined_open_local / float(reported_open_count)) * 100.0, 2)
        else:
            ratio_pct = None

        notes: List[str] = [
            "Observed comparison ratio against GitHub combined open_issues_count (issues + pull requests); not a true sample completeness proof."
        ]
        if pr_open_count == 0 and pr_closed_count == 0:
            notes.append("No pull_requests records in local snapshot (PR contribution to combined open count is 0).")

        return {
            "status": "computed" if reported_open_count is not None else "reported_stock_unavailable",
            "github_metadata_snapshot": {
                "github_combined_open_issues_and_prs": reported_open_count,
                "semantics_note": "GitHub's open_issues_count combines open pure issues and open pull requests into a single counter.",
            },
            "local_harvested_entities": {
                "locally_harvested_open_pure_issues": open_issues,
                "locally_harvested_closed_pure_issues": closed_issues,
                "locally_harvested_open_prs": pr_open_count,
                "locally_harvested_closed_prs": pr_closed_count,
            },
            "coverage_assessment": {
                "combined_open_observed_to_reported_ratio_percentage": ratio_pct,
                "pure_issue_sample_coverage_percentage": None,
                "coverage_limitation_note": "Pure issue coverage percentage cannot be derived because GitHub open_issues_count mixes open issues and open pull requests.",
            },
            "notes": " ".join(notes),
        }

    @staticmethod
    def _evaluate_resolution_distribution(eligible_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute non-parametric resolution SLAs directly from timestamps (preserving 0.0h, rejecting negatives)."""
        min_required = 5
        confident_required = 20

        valid_hours: List[float] = []
        zero_hour_count = 0
        negative_count = 0
        missing_closed_ts_count = 0
        speed_tiers: Dict[str, int] = {tier: 0 for tier in SPEED_TIER_ORDER}

        for iss in eligible_issues:
            if iss["effective_state"] != "closed":
                continue
            if iss["closed_at"] is None:
                missing_closed_ts_count += 1
                continue
            diff_sec = (iss["closed_at"] - iss["created_at"]).total_seconds()
            if diff_sec < 0.0:
                negative_count += 1
                continue
            hours = diff_sec / 3600.0
            if hours == 0.0:
                zero_hour_count += 1
            valid_hours.append(hours)
            speed_tiers[_derive_speed_tier(hours)] += 1

        n_valid = len(valid_hours)
        base_meta: Dict[str, Any] = {
            "valid_resolved_count": n_valid,
            "zero_hour_closures_count": zero_hour_count,
            "invalid_records_count": negative_count + missing_closed_ts_count,
            "negative_duration_count": negative_count,
            "missing_closed_timestamp_count": missing_closed_ts_count,
            "min_sample_required": min_required,
            "confident_sample_required": confident_required,
            "speed_tier_distribution": speed_tiers,
        }

        if len(eligible_issues) == 0:
            return {
                **base_meta,
                "status": "empty",
                "mean_hours": None,
                "p25_hours": None,
                "p50_hours": None,
                "p75_hours": None,
                "p90_hours": None,
                "iqr_hours": None,
                "skewness": None,
                "high_skew_warning": False,
                "limitations": "No eligible issues observed.",
            }

        if n_valid < min_required:
            return {
                **base_meta,
                "status": "insufficient",
                "mean_hours": None,
                "p25_hours": None,
                "p50_hours": None,
                "p75_hours": None,
                "p90_hours": None,
                "iqr_hours": None,
                "skewness": None,
                "high_skew_warning": False,
                "limitations": (
                    f"Observed {n_valid} valid closed issue(s); minimum {min_required} required for empirical SLA percentiles."
                ),
            }

        sorted_vals = sorted(valid_hours)
        p25 = _linear_percentile(sorted_vals, 0.25)
        p50 = _linear_percentile(sorted_vals, 0.50)
        p75 = _linear_percentile(sorted_vals, 0.75)
        p90 = _linear_percentile(sorted_vals, 0.90)
        skew = _compute_safe_skewness(sorted_vals)
        high_skew = skew > 1.0

        status = "sufficient" if n_valid >= confident_required else "limited"
        notes: List[str] = []
        if status == "limited":
            notes.append(
                f"Limited sample ({n_valid} < {confident_required} resolved issues); SLA percentiles carry wider bounds."
            )
        if negative_count > 0:
            notes.append(f"Excluded {negative_count} issue(s) with negative duration (closed_at < created_at) without mutating records.")
        if missing_closed_ts_count > 0:
            notes.append(f"Excluded {missing_closed_ts_count} closed issue(s) with missing closed_at timestamp.")
        if high_skew:
            notes.append(f"Right-skewed resolution distribution (skewness={skew} > 1.0); prefer median/IQR over mean.")

        return {
            **base_meta,
            "status": status,
            "mean_hours": round(statistics.mean(sorted_vals), 2),
            "p25_hours": round(p25, 2),
            "p50_hours": round(p50, 2),
            "p75_hours": round(p75, 2),
            "p90_hours": round(p90, 2),
            "iqr_hours": round(p75 - p25, 2),
            "skewness": skew,
            "high_skew_warning": high_skew,
            "limitations": " ".join(notes) if notes else None,
        }

    @staticmethod
    def _evaluate_kaplan_meier_survival(
        eligible_issues: List[Dict[str, Any]],
        ref_time: datetime,
    ) -> Dict[str, Any]:
        """Compute right-censored Kaplan-Meier issue survival (S(30d), S(60d), S(90d)) gated at >= 15 issues."""
        min_required = 15
        records: List[Tuple[float, int]] = []
        negative_count = 0
        missing_closed_ts = 0

        for iss in eligible_issues:
            if iss["effective_state"] == "closed":
                c_dt = iss["closed_at"]
                if c_dt is None:
                    missing_closed_ts += 1
                    continue
                dur_sec = (c_dt - iss["created_at"]).total_seconds()
                if dur_sec < 0.0:
                    negative_count += 1
                    continue
                records.append((dur_sec / 86400.0, 1))
            else:
                dur_sec = max(0.0, (ref_time - iss["created_at"]).total_seconds())
                records.append((dur_sec / 86400.0, 0))

        n_valid = len(records)
        base_meta: Dict[str, Any] = {
            "sample_count": n_valid,
            "events_observed_count": sum(ev for _, ev in records),
            "right_censored_count": sum(1 - ev for _, ev in records),
            "min_sample_required": min_required,
            "invalid_records_count": negative_count + missing_closed_ts,
            "negative_duration_count": negative_count,
            "missing_closed_timestamp_count": missing_closed_ts,
        }

        if len(eligible_issues) == 0 or n_valid == 0:
            return {
                **base_meta,
                "status": "empty",
                "s30d": None,
                "s60d": None,
                "s90d": None,
                "s30d_within_followup": None,
                "s60d_within_followup": None,
                "s90d_within_followup": None,
                "median_survival_days": None,
                "median_survival_hours": None,
                "median_reached": False,
                "max_observed_time_days": None,
                "survival_curve": None,
                "limitations": "No valid issues available for Kaplan-Meier survival estimation.",
            }

        if n_valid < min_required:
            return {
                **base_meta,
                "status": "insufficient",
                "s30d": None,
                "s60d": None,
                "s90d": None,
                "s30d_within_followup": None,
                "s60d_within_followup": None,
                "s90d_within_followup": None,
                "median_survival_days": None,
                "median_survival_hours": None,
                "median_reached": False,
                "max_observed_time_days": round(max(d for d, _ in records), 4),
                "survival_curve": None,
                "limitations": (
                    f"Observed {n_valid} valid issue(s); minimum {min_required} required per v1.1 §4.3 for Kaplan-Meier survival."
                ),
            }

        km = compute_kaplan_meier_curve(records=records, horizons_days=(30.0, 60.0, 90.0))
        max_obs = km["max_observed_time_days"]
        surv = km["terminal_survival"]
        median_d = km["median_days"]

        notes: List[str] = []
        if base_meta["invalid_records_count"] > 0:
            notes.append(f"Excluded {base_meta['invalid_records_count']} issue(s) with negative/missing closure timestamps.")
        if base_meta["right_censored_count"] > 0 and surv > 0.0 and max_obs < 90.0:
            notes.append(
                f"Maximum observed follow-up is {round(max_obs, 1)} days with open issues still active; tail survival past {round(max_obs, 1)}d holds last observed step constant."
            )

        return {
            **base_meta,
            "status": "sufficient",
            "s30d": km["horizon_survival"][30.0],
            "s60d": km["horizon_survival"][60.0],
            "s90d": km["horizon_survival"][90.0],
            "s30d_within_followup": km["within_followup"][30.0],
            "s60d_within_followup": km["within_followup"][60.0],
            "s90d_within_followup": km["within_followup"][90.0],
            "median_survival_days": round(median_d, 4) if median_d is not None else None,
            "median_survival_hours": round(median_d * 24.0, 2) if median_d is not None else None,
            "median_reached": median_d is not None,
            "max_observed_time_days": round(max_obs, 4),
            "survival_curve": km["curve_steps"],
            "limitations": " ".join(notes) if notes else None,
        }

    @staticmethod
    def _evaluate_backlog_aging(
        eligible_issues: List[Dict[str, Any]],
        ref_time: datetime,
    ) -> Dict[str, Any]:
        """Bin open issues at ref_time into 5 exhaustive continuous-time age bins and compute >90d stale fraction."""
        open_issues = [i for i in eligible_issues if i["effective_state"] == "open"]
        n_open = len(open_issues)

        empty_bins = {
            "fresh": {"label": "< 7 days", "count": 0, "share": None},
            "active": {"label": "7 to 30 days", "count": 0, "share": None},
            "aging": {"label": "> 30 to 90 days", "count": 0, "share": None},
            "stale": {"label": "> 90 to 365 days", "count": 0, "share": None},
            "dormant": {"label": "> 365 days", "count": 0, "share": None},
        }

        if len(eligible_issues) == 0:
            return {
                "status": "empty",
                "open_issues_count": 0,
                "median_age_days": None,
                "mean_age_days": None,
                "p90_age_days": None,
                "stale_gt_90d_count": 0,
                "stale_fraction": None,
                "bins": empty_bins,
                "limitations": "No eligible issues observed.",
            }

        if n_open == 0:
            zero_bins = {
                k: {"label": v["label"], "count": 0, "share": 0.0} for k, v in empty_bins.items()
            }
            return {
                "status": "no_open_issues",
                "open_issues_count": 0,
                "median_age_days": None,
                "mean_age_days": None,
                "p90_age_days": None,
                "stale_gt_90d_count": 0,
                "stale_fraction": 0.0,
                "bins": zero_bins,
                "limitations": "Zero open issues as of analysis_reference_time.",
            }

        ages_days: List[float] = []
        counts = {"fresh": 0, "active": 0, "aging": 0, "stale": 0, "dormant": 0}
        for iss in open_issues:
            age_d = max(0.0, (ref_time - iss["created_at"]).total_seconds() / 86400.0)
            ages_days.append(age_d)
            counts[_classify_backlog_age_bin(age_d)] += 1

        sorted_ages = sorted(ages_days)
        stale_gt_90 = counts["stale"] + counts["dormant"]
        bins_out = {
            k: {
                "label": empty_bins[k]["label"],
                "count": counts[k],
                "share": round(counts[k] / float(n_open), 4),
            }
            for k in ("fresh", "active", "aging", "stale", "dormant")
        }

        return {
            "status": "sufficient",
            "open_issues_count": n_open,
            "median_age_days": round(_linear_percentile(sorted_ages, 0.50), 2),
            "mean_age_days": round(statistics.mean(sorted_ages), 2),
            "p90_age_days": round(_linear_percentile(sorted_ages, 0.90), 2),
            "stale_gt_90d_count": stale_gt_90,
            "stale_fraction": round(stale_gt_90 / float(n_open), 4),
            "bins": bins_out,
            "limitations": None,
        }

    @staticmethod
    def _evaluate_discussion_intensity(
        eligible_issues: List[Dict[str, Any]],
        comment_rows: List[Any],
        ref_time: datetime,
        collection_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Measure issue discussion via Issue.comments_count, supplemental IssueComment rows, and guarded Spearman."""
        n_total = len(eligible_issues)
        if n_total == 0:
            return {
                "status": "empty",
                "eligible_issues_count": 0,
                "mean_comments_per_issue": None,
                "median_comments_per_issue": None,
                "p90_comments_per_issue": None,
                "zero_comment_issues_count": 0,
                "zero_comment_share": None,
                "supplemental_issue_comments": {
                    "rows_observed_up_to_ref_time": 0,
                    "distinct_issues_with_comment_rows": 0,
                    "distinct_commenters": 0,
                    "detail_available": False,
                    "note": "No eligible issues observed.",
                },
                "spearman_comments_vs_resolution": {
                    "status": "empty",
                    "sample_count": 0,
                    "spearman_r": None,
                    "p_value": None,
                    "note": "Empirical association only; no causality implied.",
                },
                "limitations": "No eligible issues observed.",
            }

        comment_counts = [max(0, int(i["comments_count"] or 0)) for i in eligible_issues]
        sorted_cc = sorted(float(c) for c in comment_counts)
        zero_cc = sum(1 for c in comment_counts if c == 0)

        # Supplemental IssueComment details (strictly <= ref_time; missing rows NEVER mean 0 comments)
        seen_cids: Set[int] = set()
        valid_comment_rows = 0
        commented_issue_ids: Set[int] = set()
        commenters: Set[str] = set()
        for row in comment_rows:
            if row.comment_id in seen_cids:
                continue
            dt = _ensure_utc(row.created_at)
            if dt is None or dt > ref_time:
                continue
            seen_cids.add(row.comment_id)
            valid_comment_rows += 1
            if row.issue_id is not None:
                commented_issue_ids.add(row.issue_id)
            if row.author_id is not None:
                commenters.add(f"id:{row.author_id}")
            elif row.author and row.author.strip():
                commenters.add(f"login:{row.author.strip().lower()}")

        if valid_comment_rows > 0:
            supp_note = "Supplemental comment timeline rows aggregated up to analysis_reference_time."
        else:
            supp_note = (
                "No IssueComment timeline rows present; discussion volume relies solely on Issue.comments_count "
                "(absence of comment rows does not indicate zero comments)."
            )

        # Spearman rank correlation between comments_count and valid non-negative resolution_hours
        paired_comments: List[float] = []
        paired_hours: List[float] = []
        for iss in eligible_issues:
            if iss["effective_state"] != "closed" or iss["closed_at"] is None:
                continue
            diff_sec = (iss["closed_at"] - iss["created_at"]).total_seconds()
            if diff_sec < 0.0:
                continue
            paired_comments.append(float(max(0, int(iss["comments_count"] or 0))))
            paired_hours.append(diff_sec / 3600.0)

        n_corr = len(paired_comments)
        if n_corr < 5:
            spearman_block = {
                "status": "insufficient",
                "sample_count": n_corr,
                "spearman_r": None,
                "p_value": None,
                "note": "Requires >= 5 valid resolved issues for Spearman association between discussion and duration; empirical only (non-causal).",
            }
        elif len(set(paired_comments)) < 2 or len(set(paired_hours)) < 2:
            spearman_block = {
                "status": "undefined_zero_variance",
                "sample_count": n_corr,
                "spearman_r": None,
                "p_value": None,
                "note": "Zero variance in comment counts or resolution durations prevents correlation computation; empirical only (non-causal).",
            }
        else:
            r_raw, p_raw = scipy_stats.spearmanr(paired_comments, paired_hours)
            r_val = float(r_raw)
            p_val = float(p_raw)
            if not (math.isfinite(r_val) and math.isfinite(p_val)):
                spearman_block = {
                    "status": "undefined_zero_variance",
                    "sample_count": n_corr,
                    "spearman_r": None,
                    "p_value": None,
                    "note": "Degenerate rank distribution prevented finite Spearman correlation.",
                }
            else:
                spearman_block = {
                    "status": "computed",
                    "sample_count": n_corr,
                    "spearman_r": round(r_val, 4),
                    "p_value": round(p_val, 6),
                    "note": "Empirical monotonic association between comment count and closure time only; does NOT imply causal direction.",
                }

        return {
            "status": "sufficient",
            "eligible_issues_count": n_total,
            "mean_comments_per_issue": round(statistics.mean(sorted_cc), 2),
            "median_comments_per_issue": round(_linear_percentile(sorted_cc, 0.50), 2),
            "p90_comments_per_issue": round(_linear_percentile(sorted_cc, 0.90), 2),
            "zero_comment_issues_count": zero_cc,
            "zero_comment_share": round(zero_cc / float(n_total), 4),
            "supplemental_issue_comments": {
                "rows_observed_up_to_ref_time": valid_comment_rows,
                "distinct_issues_with_comment_rows": len(commented_issue_ids),
                "distinct_commenters": len(commenters),
                "detail_available": valid_comment_rows > 0,
                "stream_status": collection_scope["issue_comments_stream_status"],
                "note": supp_note,
            },
            "spearman_comments_vs_resolution": spearman_block,
            "limitations": None if valid_comment_rows > 0 else "Detailed IssueComment timeline unavailable; relying on Issue.comments_count.",
        }

    @staticmethod
    def _evaluate_canonical_categories(
        eligible_issues: List[Dict[str, Any]],
        label_rows: List[Any],
    ) -> Dict[str, Any]:
        """Compute canonical issue category breakdown keeping 'unlabeled' strictly distinct from 'other'."""
        n_total = len(eligible_issues)
        if n_total == 0:
            return {
                "status": "empty",
                "eligible_issues_count": 0,
                "is_unlabeled_count": 0,
                "labeled_issues_count": 0,
                "unlabeled_share": None,
                "categories": {},
                "top_raw_labels": [],
                "limitations": "No eligible issues observed.",
            }

        cat_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        unlabeled_flag_count = 0
        for iss in eligible_issues:
            cat_groups[iss["effective_category"]].append(iss)
            if iss["is_unlabeled"] or iss["effective_category"] == "unlabeled":
                unlabeled_flag_count += 1

        categories_out: Dict[str, Dict[str, Any]] = {}
        for cat_name, items in sorted(cat_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            open_c = sum(1 for i in items if i["effective_state"] == "open")
            closed_c = sum(1 for i in items if i["effective_state"] == "closed")
            durations_h: List[float] = []
            for i in items:
                if i["effective_state"] == "closed" and i["closed_at"] is not None:
                    diff_sec = (i["closed_at"] - i["created_at"]).total_seconds()
                    if diff_sec >= 0.0:
                        durations_h.append(diff_sec / 3600.0)
            categories_out[cat_name] = {
                "total_count": len(items),
                "share": round(len(items) / float(n_total), 4),
                "open_count": open_c,
                "closed_count": closed_c,
                "valid_resolved_count": len(durations_h),
                "median_resolution_hours": round(statistics.median(durations_h), 2) if durations_h else None,
            }

        freq_map: Dict[str, int] = defaultdict(int)
        cat_map: Dict[str, Optional[str]] = {}
        for lr in label_rows:
            freq_map[lr.label_name] += 1
            if lr.label_name not in cat_map:
                cat_map[lr.label_name] = lr.canonical_category
        top_raw_labels = [
            {
                "label_name": lbl,
                "count": cnt,
                "canonical_category": cat_map.get(lbl),
            }
            for lbl, cnt in sorted(freq_map.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        ]

        return {
            "status": "sufficient",
            "eligible_issues_count": n_total,
            "is_unlabeled_count": unlabeled_flag_count,
            "labeled_issues_count": n_total - unlabeled_flag_count,
            "unlabeled_share": round(unlabeled_flag_count / float(n_total), 4),
            "categories": categories_out,
            "top_raw_labels": top_raw_labels,
            "limitations": None,
        }

    def _evaluate_coverage_and_confidence(
        self,
        eligible_issues: List[Dict[str, Any]],
        collection_scope: Dict[str, Any],
        monthly_flow: Dict[str, Any],
        backlog_stock: Dict[str, Any],
        resolution: Dict[str, Any],
        survival: Dict[str, Any],
        aging: Dict[str, Any],
        discussion: Dict[str, Any],
        categories: Dict[str, Any],
    ) -> Tuple[MetricCoverageItem, float, List[Dict[str, Any]]]:
        """Evaluate overall issue_intelligence coverage status and deterministic penalties."""
        n_total = len(eligible_issues)
        min_sample = 10
        penalties: List[Dict[str, Any]] = []
        penalty_total = 0.0

        if n_total == 0:
            cov_status = "empty"
            p = 0.5
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-EMPTY-ISSUES",
                "penalty": p,
                "reason": "Repository snapshot has zero eligible issues.",
            })
        elif n_total < 5:
            cov_status = "insufficient"
            p = 0.4
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-INSUFFICIENT-ISSUES",
                "penalty": p,
                "reason": f"Observed eligible issues ({n_total}) is below minimum 5 required for baseline issue analytics.",
            })
        elif n_total < min_sample:
            cov_status = "degraded"
            p = 0.2
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-SMALL-ISSUE-SAMPLE",
                "penalty": p,
                "reason": f"Observed eligible issues ({n_total}) is below high-confidence threshold ({min_sample}).",
            })
        else:
            cov_status = "sufficient"

        if n_total > 0 and resolution["status"] in ("insufficient", "empty"):
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-LOW-RESOLVED-ISSUES",
                "penalty": p,
                "reason": f"Valid resolved issues ({resolution['valid_resolved_count']}) is below 5 required for empirical SLAs.",
            })

        if n_total > 0 and collection_scope["is_bounded_or_partial"]:
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-BOUNDED-ISSUE-SAMPLE",
                "penalty": p,
                "reason": f"Collection mode '{collection_scope['mode']}' represents a bounded/unverified issue sample.",
            })

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=cov_status,
            sample_size=n_total,
            min_sample_required=min_sample,
            details={
                "eligible_issues_count": n_total,
                "valid_resolved_issues_count": resolution["valid_resolved_count"],
                "open_issues_count": aging["open_issues_count"],
                "collection_mode": collection_scope["mode"],
                "subdomains": {
                    "observed_flow": monthly_flow["status"],
                    "backlog_stock": backlog_stock["status"],
                    "resolution_distribution": resolution["status"],
                    "survival_km": survival["status"],
                    "backlog_aging": aging["status"],
                    "discussion_intensity": discussion["status"],
                    "canonical_categories": categories["status"],
                },
            },
        )

        return coverage, round(min(1.0, penalty_total), 3), penalties

    def _build_rules(
        self,
        eligible_issues: List[Dict[str, Any]],
        monthly_flow: Dict[str, Any],
        backlog_stock: Dict[str, Any],
        resolution: Dict[str, Any],
        aging: Dict[str, Any],
        categories: Dict[str, Any],
    ) -> List[RuleInterpretation]:
        """Generate deterministic RULE-ISS-* diagnostics with verified evidence."""
        rules: List[RuleInterpretation] = []
        n_total = len(eligible_issues)

        if n_total == 0:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ISS-EMPTY",
                    domain=self.domain_name,
                    severity="info",
                    finding="No Issues Observed: Repository snapshot contains zero eligible issues.",
                    evidence={"eligible_issues_count": 0},
                    recommendation="Ingest issue history to analyze resolution velocity, backlog aging, and issue categories.",
                )
            )
            return rules

        # RULE-ISS-001: Observed Issue Flow Ratio (when sufficient & >= 20 created observed)
        flow_ratio = monthly_flow["flow_ratio"]
        total_created = monthly_flow["total_created_observed"]
        if monthly_flow["status"] == "sufficient" and flow_ratio is not None and total_created >= 20:
            if flow_ratio < 0.70:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-ISS-001",
                        domain=self.domain_name,
                        severity="warning",
                        finding=(
                            f"Observed Sample Backlog Expansion: Closure-to-creation flow ratio is {round(flow_ratio * 100.0, 1)}% "
                            f"({monthly_flow['total_closed_observed']} closed vs {total_created} created in observed sample)."
                        ),
                        evidence={
                            "flow_ratio": flow_ratio,
                            "total_created_observed": total_created,
                            "total_closed_observed": monthly_flow["total_closed_observed"],
                            "net_flow_differential": monthly_flow["net_flow_differential"],
                        },
                        recommendation="Review incoming issue triage throughput relative to closure capacity.",
                    )
                )
            elif flow_ratio >= 1.0:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-ISS-001",
                        domain=self.domain_name,
                        severity="positive",
                        finding=(
                            f"Net Issue Closure Surplus: Observed closures match or exceed creations "
                            f"(flow ratio {round(flow_ratio * 100.0, 1)}% across {total_created} created issues)."
                        ),
                        evidence={
                            "flow_ratio": flow_ratio,
                            "total_created_observed": total_created,
                            "total_closed_observed": monthly_flow["total_closed_observed"],
                        },
                        recommendation=None,
                    )
                )

        # RULE-ISS-002: Stale Backlog Accumulation (> 90d)
        stale_frac = aging["stale_fraction"]
        if aging["open_issues_count"] >= 10 and stale_frac is not None and stale_frac > 0.50:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ISS-002",
                    domain=self.domain_name,
                    severity="warning",
                    finding=(
                        f"Stale Open Backlog Accumulation: {round(stale_frac * 100.0, 1)}% of observed open issues "
                        f"({aging['stale_gt_90d_count']} of {aging['open_issues_count']}) exceed 90 days open (median age {aging['median_age_days']}d)."
                    ),
                    evidence={
                        "open_issues_count": aging["open_issues_count"],
                        "stale_gt_90d_count": aging["stale_gt_90d_count"],
                        "stale_fraction": stale_frac,
                        "median_age_days": aging["median_age_days"],
                    },
                    recommendation="Triage aging/stale open issues to separate unresolved defects from obsolete requests.",
                )
            )

        # RULE-ISS-003: Heavy-Tailed Resolution Latency Warning
        if resolution["status"] in ("limited", "sufficient") and resolution["high_skew_warning"]:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ISS-003",
                    domain=self.domain_name,
                    severity="info",
                    finding=(
                        f"Heavy-Tailed Resolution Latency: Sample skewness is {resolution['skewness']} (> 1.0); "
                        f"median resolution ({resolution['p50_hours']}h) differs substantially from mean ({resolution['mean_hours']}h)."
                    ),
                    evidence={
                        "p50_hours": resolution["p50_hours"],
                        "p90_hours": resolution["p90_hours"],
                        "mean_hours": resolution["mean_hours"],
                        "skewness": resolution["skewness"],
                    },
                    recommendation=None,
                )
            )

        # RULE-ISS-004: Unlabeled Issue Share
        unlabeled_share = categories["unlabeled_share"]
        if n_total >= 10 and unlabeled_share is not None and unlabeled_share >= 0.25:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ISS-004",
                    domain=self.domain_name,
                    severity="info",
                    finding=(
                        f"Substantial Unlabeled Issue Volume: {round(unlabeled_share * 100.0, 1)}% "
                        f"({categories['is_unlabeled_count']} of {n_total}) issues lack repository labels."
                    ),
                    evidence={
                        "is_unlabeled_count": categories["is_unlabeled_count"],
                        "eligible_issues_count": n_total,
                        "unlabeled_share": unlabeled_share,
                    },
                    recommendation="Automated triage or issue templates can improve label hygiene for downstream routing.",
                )
            )

        # RULE-ISS-005: Timestamp Hygiene / Partial Stock Observation Transparency
        neg_count = resolution["negative_duration_count"]
        miss_closed = resolution["missing_closed_timestamp_count"]
        cov_ratio = backlog_stock["coverage_assessment"]["combined_open_observed_to_reported_ratio_percentage"]
        if neg_count > 0 or miss_closed > 0 or (cov_ratio is not None and cov_ratio < 25.0):
            parts: List[str] = []
            if neg_count > 0:
                parts.append(f"{neg_count} negative-duration closed issue(s) excluded")
            if miss_closed > 0:
                parts.append(f"{miss_closed} closed issue(s) lacked closed_at timestamp")
            if cov_ratio is not None and cov_ratio < 25.0:
                parts.append(
                    f"local open issues+PRs equal {cov_ratio}% of GitHub reported open_issues_count ({backlog_stock['github_metadata_snapshot']['github_combined_open_issues_and_prs']})"
                )
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ISS-005",
                    domain=self.domain_name,
                    severity="info",
                    finding=f"Issue Coverage & Timestamp Hygiene: {'; '.join(parts)}.",
                    evidence={
                        "negative_duration_count": neg_count,
                        "missing_closed_timestamp_count": miss_closed,
                        "combined_open_observed_to_reported_ratio_percentage": cov_ratio,
                    },
                    recommendation=None,
                )
            )

        return rules
