"""Development Activity Intelligence Module for OpenSourceLens.

Implements C.2.1 Activity Aggregation, C.2.2 Consistency/Rolling/Anomaly/Change-Point Detection,
and C.2.3 Activity Rhythms, Diff-Aware Code Churn & Commit Landing Latency:
- Daily and weekly commit activity time-series aggregation.
- Continuous zero-filled calendar intervals.
- Multi-horizon weekly commit velocity (exact rolling 28d/84d/364d windows, overall).
- Contributor activity & active author aggregation.
- Release cadence, interval statistics (mean, median, min, max, stdev), and classification.
- Weekly activity consistency (CV, IQR, MAD, active week ratio, regime classification).
- Rolling weekly statistics (4w/12w SMA, stdev, median, 4w EWMA) excluding incomplete current week.
- Adaptive anomaly detection (MAD modified z-score with z-score/Poisson dispersion fallbacks).
- Justified structural change-point detection (deterministic max-SSB permutation test across eligible splits).
- 7x24 UTC activity rhythms (explicitly noting UTC timestamps do not capture contributor local hours).
- Diff-aware code churn coverage, net/gross churn, churn-per-commit, and completed-week volatility.
- Commit authoring-to-landing latency (committer_date - author_date) with strict non-substitution & skew isolation.
- Robust timezone normalization to UTC and deterministic rule-based diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Commit, Contributor, Release, Repository
from backend.app.schemas.analytics_v2 import MetricCoverageItem, RuleInterpretation
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure datetime is timezone-aware and normalized to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_week_start_utc(dt: datetime) -> datetime:
    """Return Monday 00:00:00 UTC for the week containing dt."""
    utc_dt = _ensure_utc(dt)
    monday = utc_dt - timedelta(days=utc_dt.weekday())
    return datetime(monday.year, monday.month, monday.day, 0, 0, 0, tzinfo=timezone.utc)


def _linear_percentile(sorted_vals: List[float], q: float) -> float:
    """Compute linear-interpolated quantile q in [0.0, 1.0] on non-empty sorted values."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


class DevelopmentActivityCalculator(BaseMetricCalculator):
    """Calculates development activity, commit velocity, contributor participation, and release cadence."""

    @property
    def domain_name(self) -> str:
        return "development_activity"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute activity metrics anchored to analysis_reference_time.

        Args:
            repository_id: Foreign key ID of the target Repository.
            db: SQLAlchemy session inside the pinned read snapshot.
            analysis_reference_time: Frozen UTC reference timestamp.
            snapshot_revision: Repository revision pinned by the snapshot.

        Returns:
            CalculationResult containing activity metrics, coverage evaluation, and rule interpretations.
        """
        ref_time = _ensure_utc(analysis_reference_time)

        # 1. Query Commits within snapshot
        commits_query = (
            select(
                Commit.id,
                Commit.sha,
                Commit.author_date,
                Commit.committer_date,
                Commit.author_login,
                Commit.author_name,
                Commit.author_email,
                Commit.additions,
                Commit.deletions,
                Commit.total_changes,
            )
            .where(Commit.repository_id == repository_id)
            .order_by(Commit.author_date.asc())
        )
        commits_rows = db.execute(commits_query).fetchall()

        # 2. Query Releases within snapshot
        releases_query = (
            select(
                Release.id,
                Release.tag_name,
                Release.name,
                Release.published_at,
                Release.is_prerelease,
            )
            .where(Release.repository_id == repository_id)
            .order_by(Release.published_at.asc())
        )
        releases_rows = db.execute(releases_query).fetchall()

        # 3. Query Contributor entities within snapshot
        contributors_count = db.scalar(
            select(func.count(Contributor.id)).where(Contributor.repository_id == repository_id)
        ) or 0

        # Filter commits up to frozen reference time and sort chronologically in UTC
        valid_commits = []
        for row in commits_rows:
            commit_date = _ensure_utc(row.author_date or row.committer_date)
            if commit_date is not None and commit_date <= ref_time:
                valid_commits.append((row, commit_date))
        valid_commits.sort(key=lambda item: item[1])

        total_commits = len(valid_commits)

        # Filter releases up to frozen reference time and sort chronologically in UTC
        valid_releases = []
        for rel in releases_rows:
            pub_date = _ensure_utc(rel.published_at)
            if pub_date is not None and pub_date <= ref_time:
                valid_releases.append((rel, pub_date))
        valid_releases.sort(key=lambda item: item[1])

        # Handle Empty Commit History Edge Case
        if total_commits == 0:
            return self._build_empty_result(
                repository_id=repository_id,
                releases_rows=valid_releases,
                contributors_count=contributors_count,
                ref_time=ref_time,
                snapshot_revision=snapshot_revision,
            )

        # 4. Aggregate Commit Time-Series & Contributor Activity
        daily_counts: Dict[str, int] = defaultdict(int)
        weekly_commits: Dict[str, int] = defaultdict(int)
        weekly_additions: Dict[str, int] = defaultdict(int)
        weekly_deletions: Dict[str, int] = defaultdict(int)
        weekly_changes: Dict[str, int] = defaultdict(int)
        weekly_authors: Dict[str, set] = defaultdict(set)

        all_authors: Dict[str, int] = defaultdict(int)
        recent_30d_authors: set = set()
        recent_90d_authors: set = set()

        cutoff_30d = ref_time - timedelta(days=30)
        cutoff_90d = ref_time - timedelta(days=90)

        earliest_date = valid_commits[0][1]
        latest_date = valid_commits[-1][1]

        total_additions = 0
        total_deletions = 0
        total_changes = 0

        for row, c_date in valid_commits:
            day_str = c_date.strftime("%Y-%m-%d")
            week_start = _get_week_start_utc(c_date)
            week_str = week_start.strftime("%Y-%m-%d")

            daily_counts[day_str] += 1
            weekly_commits[week_str] += 1

            adds = row.additions or 0
            dels = row.deletions or 0
            chgs = row.total_changes or (adds + dels)

            weekly_additions[week_str] += adds
            weekly_deletions[week_str] += dels
            weekly_changes[week_str] += chgs

            total_additions += adds
            total_deletions += dels
            total_changes += chgs

            # Author identity resolution
            author_key = row.author_login or row.author_email or row.author_name or f"unlinked_{row.id}"
            all_authors[author_key] += 1
            weekly_authors[week_str].add(author_key)

            if c_date >= cutoff_30d:
                recent_30d_authors.add(author_key)
            if c_date >= cutoff_90d:
                recent_90d_authors.add(author_key)

        # 5. Continuous Zero-Filled Weekly Time-Series
        # Generate full continuous sequence from earliest week to reference time week
        first_week = _get_week_start_utc(earliest_date)
        last_week = _get_week_start_utc(ref_time)

        weekly_time_series: List[Dict[str, Any]] = []
        curr_week = first_week
        total_weeks_span = 0

        while curr_week <= last_week:
            total_weeks_span += 1
            w_str = curr_week.strftime("%Y-%m-%d")
            weekly_time_series.append({
                "week_start": w_str,
                "commit_count": weekly_commits.get(w_str, 0),
                "additions": weekly_additions.get(w_str, 0),
                "deletions": weekly_deletions.get(w_str, 0),
                "total_changes": weekly_changes.get(w_str, 0),
                "active_contributors_count": len(weekly_authors.get(w_str, set())),
            })
            curr_week += timedelta(days=7)

        # Trailing 90 Days Daily Time-Series (Zero-filled)
        daily_time_series: List[Dict[str, Any]] = []
        for day_offset in range(89, -1, -1):
            target_day = ref_time.date() - timedelta(days=day_offset)
            d_str = target_day.strftime("%Y-%m-%d")
            daily_time_series.append({
                "date": d_str,
                "commit_count": daily_counts.get(d_str, 0),
            })

        # 6. Weekly Commit Velocity & Moving Averages
        # Use exact half-open trailing windows (ref_time - N days, ref_time] to avoid
        # incomplete-calendar-week day-of-week boundary bias.
        cutoff_28d = ref_time - timedelta(days=28)
        cutoff_84d = ref_time - timedelta(days=84)
        cutoff_364d = ref_time - timedelta(days=364)

        # Trailing 4 weeks (exact trailing 28-day window divided by 4)
        trailing_4w_commits = sum(1 for _, c_date in valid_commits if c_date > cutoff_28d)
        trailing_4w_velocity = round(trailing_4w_commits / 4.0, 2)

        # Trailing 12 weeks (exact trailing 84-day window divided by 12)
        trailing_12w_commits = sum(1 for _, c_date in valid_commits if c_date > cutoff_84d)
        trailing_12w_velocity = round(trailing_12w_commits / 12.0, 2)

        # Trailing 52 weeks (exact trailing 364-day window normalized by observable history in window)
        trailing_52w_commits = sum(1 for _, c_date in valid_commits if c_date > cutoff_364d)
        effective_52w_start = max(cutoff_364d, earliest_date)
        observed_52w_days = (ref_time - effective_52w_start).total_seconds() / 86400.0
        divisor_52w = min(52.0, max(1.0, observed_52w_days / 7.0))
        trailing_52w_velocity = round(trailing_52w_commits / divisor_52w, 2)

        # Overall historical weekly velocity
        overall_weekly_velocity = round(total_commits / max(1.0, float(total_weeks_span)), 2)

        # Velocity ratio (recent 4w vs baseline 12w)
        baseline_velocity = max(0.1, trailing_12w_velocity)
        velocity_trend_ratio = round(trailing_4w_velocity / baseline_velocity, 2)

        # Days since last commit
        days_since_last_commit = round((ref_time - latest_date).total_seconds() / 86400.0, 2)
        total_span_days = max(1, (latest_date.date() - earliest_date.date()).days + 1)

        # 7. Contributor Activity Metrics
        top_authors_list = sorted(
            [{"author": k, "commit_count": v, "share": round(v / total_commits, 4)} for k, v in all_authors.items()],
            key=lambda x: x["commit_count"],
            reverse=True,
        )[:10]

        # 8. Release Cadence Calculations
        release_metrics = self._calculate_release_cadence(valid_releases, ref_time)

        # 8b. C.2.2 Consistency, Rolling Statistics, Anomaly Detection, Change-Point Analysis
        # Strictly exclude the ongoing calendar week [last_week, ref_time] from completed-week
        # baseline statistics to avoid partial-week day-of-week downward bias.
        completed_weeks = weekly_time_series[:-1] if len(weekly_time_series) >= 1 else []
        current_week_entry = weekly_time_series[-1] if weekly_time_series else {
            "week_start": last_week.strftime("%Y-%m-%d"),
            "commit_count": 0,
        }
        elapsed_current_week_days = round(
            max(0.0, (ref_time - last_week).total_seconds() / 86400.0), 2
        )
        partial_current_week = {
            "week_start": current_week_entry["week_start"],
            "commit_count": current_week_entry["commit_count"],
            "elapsed_days": elapsed_current_week_days,
            "excluded_from_baseline": True,
        }

        consistency_metrics = self._calculate_consistency(completed_weeks)
        rolling_stats = self._calculate_rolling_statistics(completed_weeks, partial_current_week)
        anomaly_results = self._detect_anomalies(completed_weeks, consistency_metrics, partial_current_week)
        change_point_results = self._detect_change_points(completed_weeks)

        # 8c. C.2.3 Activity Rhythms, Diff-Aware Code Churn, Commit Landing Latency
        rhythm_metrics = self._calculate_activity_rhythms(valid_commits)
        churn_metrics = self._calculate_code_churn(valid_commits, completed_weeks)
        latency_metrics = self._calculate_commit_landing_latency(valid_commits)

        # 9. Coverage and Confidence Evaluation
        min_sample = 5
        span_days_threshold = 14
        penalties: List[Dict[str, Any]] = []
        confidence_penalty = 0.0

        if total_commits < min_sample:
            coverage_status = "insufficient"
            penalty = 0.3
            confidence_penalty += penalty
            penalties.append({
                "rule": "PENALTY-SAMPLE-SIZE",
                "penalty": penalty,
                "reason": f"Observed commit sample size ({total_commits}) is below minimum required ({min_sample})",
            })
        elif total_span_days < span_days_threshold:
            coverage_status = "degraded"
            penalty = 0.15
            confidence_penalty += penalty
            penalties.append({
                "rule": "PENALTY-SHORT-SPAN",
                "penalty": penalty,
                "reason": f"Commit history span ({total_span_days} days) is shorter than recommended window ({span_days_threshold} days)",
            })
        else:
            coverage_status = "sufficient"

        if total_commits < 20 and total_commits >= min_sample:
            penalty = 0.1
            confidence_penalty += penalty
            penalties.append({
                "rule": "PENALTY-MODERATE-SAMPLE",
                "penalty": penalty,
                "reason": f"Sample size ({total_commits}) is sufficient for baseline analysis but below high-confidence threshold (20)",
            })

        confidence_penalty = round(min(1.0, confidence_penalty), 3)

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=coverage_status,
            sample_size=total_commits,
            min_sample_required=min_sample,
            details={
                "total_commits": total_commits,
                "active_days_count": len(daily_counts),
                "active_weeks_count": len(weekly_commits),
                "total_span_days": total_span_days,
                "total_releases": len(valid_releases),
                "total_authors": len(all_authors),
                "completed_weeks_count": len(completed_weeks),
                "subdomains": {
                    "consistency": consistency_metrics["status"],
                    "rolling_statistics": rolling_stats["status"],
                    "anomalies": anomaly_results["status"],
                    "change_points": change_point_results["status"],
                    "activity_rhythms": rhythm_metrics["status"],
                    "code_churn": churn_metrics["status"],
                    "commit_landing_latency": latency_metrics["status"],
                },
            },
        )

        # 10. Deterministic Rules & Diagnostics
        rules: List[RuleInterpretation] = []

        # Rule 1: Activity Recency
        if days_since_last_commit <= 30.0:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-001",
                    domain=self.domain_name,
                    severity="positive",
                    finding="Active Development: Repository has committed code within the last 30 days.",
                    evidence={"days_since_last_commit": days_since_last_commit, "threshold_days": 30.0},
                    recommendation=None,
                )
            )
        elif days_since_last_commit <= 90.0:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-001",
                    domain=self.domain_name,
                    severity="info",
                    finding="Moderate Activity: Last commit was within the past quarter.",
                    evidence={"days_since_last_commit": days_since_last_commit, "threshold_days": 90.0},
                    recommendation=None,
                )
            )
        elif days_since_last_commit <= 180.0:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-001",
                    domain=self.domain_name,
                    severity="warning",
                    finding="Sluggish Activity: No commits recorded in the past 90–180 days.",
                    evidence={"days_since_last_commit": days_since_last_commit, "threshold_days": 180.0},
                    recommendation="Review project roadmap and repository maintenance status.",
                )
            )
        else:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-001",
                    domain=self.domain_name,
                    severity="critical",
                    finding="Dormant Repository: No commit activity in over 180 days.",
                    evidence={"days_since_last_commit": days_since_last_commit, "threshold_days": 180.0},
                    recommendation="Check if project has moved or become unmaintained.",
                )
            )

        # Rule 2: Velocity Trajectory
        if velocity_trend_ratio >= 1.25:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-002",
                    domain=self.domain_name,
                    severity="positive",
                    finding="Accelerating Velocity: Recent commit velocity is 25%+ above the 12-week baseline.",
                    evidence={
                        "trailing_4w_velocity": trailing_4w_velocity,
                        "trailing_12w_velocity": trailing_12w_velocity,
                        "velocity_trend_ratio": velocity_trend_ratio,
                    },
                    recommendation=None,
                )
            )
        elif velocity_trend_ratio <= 0.75:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-002",
                    domain=self.domain_name,
                    severity="warning",
                    finding="Decelerating Velocity: Recent commit pace has declined by 25%+ relative to baseline.",
                    evidence={
                        "trailing_4w_velocity": trailing_4w_velocity,
                        "trailing_12w_velocity": trailing_12w_velocity,
                        "velocity_trend_ratio": velocity_trend_ratio,
                    },
                    recommendation="Investigate potential contributor bandwidth bottlenecks.",
                )
            )
        else:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-002",
                    domain=self.domain_name,
                    severity="info",
                    finding="Steady Velocity: Commit pace is consistent with historical baseline.",
                    evidence={
                        "trailing_4w_velocity": trailing_4w_velocity,
                        "trailing_12w_velocity": trailing_12w_velocity,
                        "velocity_trend_ratio": velocity_trend_ratio,
                    },
                    recommendation=None,
                )
            )

        # Rule 3: Release Regularity
        cadence_class = release_metrics.get("cadence_classification")
        if cadence_class in ("rapid", "monthly", "quarterly"):
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-003",
                    domain=self.domain_name,
                    severity="positive",
                    finding=f"Structured Release Cadence: Repository publishes regular releases ({cadence_class}).",
                    evidence={
                        "cadence_classification": cadence_class,
                        "median_interval_days": release_metrics.get("median_release_interval_days"),
                    },
                    recommendation=None,
                )
            )
        elif release_metrics.get("days_since_last_release") is not None and release_metrics["days_since_last_release"] > 365.0:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-003",
                    domain=self.domain_name,
                    severity="warning",
                    finding="Stale Releases: No software releases published in over one year.",
                    evidence={"days_since_last_release": release_metrics["days_since_last_release"]},
                    recommendation="Consider tagging a new release version for downstream consumers.",
                )
            )

        # Rule 4–6: Consistency, Anomalies & Structural Change-Points (C.2.2)
        rules.extend(
            self._build_c22_rules(
                consistency_metrics=consistency_metrics,
                anomaly_results=anomaly_results,
                change_point_results=change_point_results,
            )
        )

        # Rule 7–9: Rhythms, Churn & Commit Landing Latency (C.2.3)
        rules.extend(
            self._build_c23_rules(
                rhythm_metrics=rhythm_metrics,
                churn_metrics=churn_metrics,
                latency_metrics=latency_metrics,
            )
        )

        # Assemble Full Metrics Dictionary
        metrics: Dict[str, Any] = {
            "summary": {
                "total_commits": total_commits,
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_changes": total_changes,
                "active_days_count": len(daily_counts),
                "active_weeks_count": len(weekly_commits),
                "total_span_days": total_span_days,
                "earliest_commit_date": earliest_date.isoformat(),
                "latest_commit_date": latest_date.isoformat(),
                "days_since_last_commit": days_since_last_commit,
            },
            "velocity": {
                "trailing_4w_velocity": trailing_4w_velocity,
                "trailing_12w_velocity": trailing_12w_velocity,
                "trailing_52w_velocity": trailing_52w_velocity,
                "overall_weekly_velocity": overall_weekly_velocity,
                "velocity_trend_ratio": velocity_trend_ratio,
            },
            "contributors": {
                "total_unique_authors": len(all_authors),
                "trailing_30d_active_authors": len(recent_30d_authors),
                "trailing_90d_active_authors": len(recent_90d_authors),
                "top_authors": top_authors_list,
                "total_contributors_recorded": contributors_count,
            },
            "releases": release_metrics,
            "consistency": consistency_metrics,
            "rolling_statistics": rolling_stats,
            "anomalies": anomaly_results,
            "change_points": change_point_results,
            "activity_rhythms": rhythm_metrics,
            "code_churn": churn_metrics,
            "commit_landing_latency": latency_metrics,
            "weekly_time_series": weekly_time_series,
            "daily_time_series_90d": daily_time_series,
        }

        return CalculationResult(
            domain=self.domain_name,
            metrics=metrics,
            coverage=coverage,
            confidence_penalty=confidence_penalty,
            rules=rules,
            details={"penalties": penalties, "snapshot_revision": snapshot_revision},
        )

    def _calculate_release_cadence(
        self,
        valid_releases: List[Tuple[Any, datetime]],
        ref_time: datetime,
    ) -> Dict[str, Any]:
        """Compute release cadence, interval quantiles, and regularity."""
        total_releases = len(valid_releases)
        if total_releases == 0:
            return {
                "total_releases": 0,
                "total_prereleases": 0,
                "total_stable_releases": 0,
                "latest_release_date": None,
                "latest_release_tag": None,
                "days_since_last_release": None,
                "mean_release_interval_days": None,
                "median_release_interval_days": None,
                "min_release_interval_days": None,
                "max_release_interval_days": None,
                "stdev_release_interval_days": None,
                "cadence_classification": "no_releases",
            }

        total_prereleases = sum(1 for rel, _ in valid_releases if rel.is_prerelease)
        total_stable = total_releases - total_prereleases
        latest_rel, latest_date = valid_releases[-1]
        days_since_last_release = round((ref_time - latest_date).total_seconds() / 86400.0, 2)

        if total_releases < 2:
            return {
                "total_releases": total_releases,
                "total_prereleases": total_prereleases,
                "total_stable_releases": total_stable,
                "latest_release_date": latest_date.isoformat(),
                "latest_release_tag": latest_rel.tag_name,
                "days_since_last_release": days_since_last_release,
                "mean_release_interval_days": None,
                "median_release_interval_days": None,
                "min_release_interval_days": None,
                "max_release_interval_days": None,
                "stdev_release_interval_days": None,
                "cadence_classification": "insufficient_releases",
            }

        intervals: List[float] = []
        for i in range(len(valid_releases) - 1):
            d1 = valid_releases[i][1]
            d2 = valid_releases[i + 1][1]
            delta_days = max(0.0, (d2 - d1).total_seconds() / 86400.0)
            intervals.append(delta_days)

        mean_interval = round(statistics.mean(intervals), 2)
        median_interval = round(statistics.median(intervals), 2)
        min_interval = round(min(intervals), 2)
        max_interval = round(max(intervals), 2)
        stdev_interval = round(statistics.stdev(intervals), 2) if len(intervals) >= 2 else 0.0

        if median_interval <= 14.0:
            classification = "rapid"
        elif median_interval <= 45.0:
            classification = "monthly"
        elif median_interval <= 120.0:
            classification = "quarterly"
        elif median_interval <= 365.0:
            classification = "annual"
        else:
            classification = "infrequent"

        return {
            "total_releases": total_releases,
            "total_prereleases": total_prereleases,
            "total_stable_releases": total_stable,
            "latest_release_date": latest_date.isoformat(),
            "latest_release_tag": latest_rel.tag_name,
            "days_since_last_release": days_since_last_release,
            "mean_release_interval_days": mean_interval,
            "median_release_interval_days": median_interval,
            "min_release_interval_days": min_interval,
            "max_release_interval_days": max_interval,
            "stdev_release_interval_days": stdev_interval,
            "cadence_classification": classification,
        }

    def _build_empty_result(
        self,
        repository_id: int,
        releases_rows: List[Tuple[Any, datetime]],
        contributors_count: int,
        ref_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Construct deterministic output when repository has zero observed commits."""
        release_metrics = self._calculate_release_cadence(releases_rows, ref_time)
        last_week = _get_week_start_utc(ref_time)
        partial_current_week = {
            "week_start": last_week.strftime("%Y-%m-%d"),
            "commit_count": 0,
            "elapsed_days": round(max(0.0, (ref_time - last_week).total_seconds() / 86400.0), 2),
            "excluded_from_baseline": True,
        }
        consistency_metrics = self._calculate_consistency([])
        rolling_stats = self._calculate_rolling_statistics([], partial_current_week)
        anomaly_results = self._detect_anomalies([], consistency_metrics, partial_current_week)
        change_point_results = self._detect_change_points([])
        rhythm_metrics = self._calculate_activity_rhythms([])
        churn_metrics = self._calculate_code_churn([], [])
        latency_metrics = self._calculate_commit_landing_latency([])

        metrics: Dict[str, Any] = {
            "summary": {
                "total_commits": 0,
                "total_additions": 0,
                "total_deletions": 0,
                "total_changes": 0,
                "active_days_count": 0,
                "active_weeks_count": 0,
                "total_span_days": 0,
                "earliest_commit_date": None,
                "latest_commit_date": None,
                "days_since_last_commit": None,
            },
            "velocity": {
                "trailing_4w_velocity": 0.0,
                "trailing_12w_velocity": 0.0,
                "trailing_52w_velocity": 0.0,
                "overall_weekly_velocity": 0.0,
                "velocity_trend_ratio": 1.0,
            },
            "contributors": {
                "total_unique_authors": 0,
                "trailing_30d_active_authors": 0,
                "trailing_90d_active_authors": 0,
                "top_authors": [],
                "total_contributors_recorded": contributors_count,
            },
            "releases": release_metrics,
            "consistency": consistency_metrics,
            "rolling_statistics": rolling_stats,
            "anomalies": anomaly_results,
            "change_points": change_point_results,
            "activity_rhythms": rhythm_metrics,
            "code_churn": churn_metrics,
            "commit_landing_latency": latency_metrics,
            "weekly_time_series": [],
            "daily_time_series_90d": [],
        }

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status="empty",
            sample_size=0,
            min_sample_required=5,
            details={
                "reason": "No commits found in repository snapshot",
                "total_commits": 0,
                "completed_weeks_count": 0,
                "subdomains": {
                    "consistency": consistency_metrics["status"],
                    "rolling_statistics": rolling_stats["status"],
                    "anomalies": anomaly_results["status"],
                    "change_points": change_point_results["status"],
                    "activity_rhythms": rhythm_metrics["status"],
                    "code_churn": churn_metrics["status"],
                    "commit_landing_latency": latency_metrics["status"],
                },
            },
        )

        rules = [
            RuleInterpretation(
                rule_id="RULE-ACT-EMPTY",
                domain=self.domain_name,
                severity="warning",
                finding="No Commit History: Repository has no observed commits in this snapshot.",
                evidence={"total_commits": 0},
                recommendation="Run repository ingestion to harvest commits before analyzing development activity.",
            )
        ]

        return CalculationResult(
            domain=self.domain_name,
            metrics=metrics,
            coverage=coverage,
            confidence_penalty=0.5,
            rules=rules,
            details={
                "penalties": [
                    {
                        "rule": "PENALTY-EMPTY-COMMITS",
                        "penalty": 0.5,
                        "reason": "Repository snapshot contains 0 commits.",
                    }
                ],
                "snapshot_revision": snapshot_revision,
            },
        )

    def _calculate_consistency(
        self,
        completed_weeks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Measure week-to-week commit consistency and classify activity regularity."""
        k_weeks = len(completed_weeks)
        if k_weeks == 0:
            return {
                "status": "insufficient",
                "completed_weeks_count": 0,
                "incomplete_current_week_excluded": True,
                "mean_weekly_commits": None,
                "median_weekly_commits": None,
                "stdev_weekly_commits": None,
                "iqr_weekly_commits": None,
                "mad_weekly_commits": None,
                "coefficient_of_variation": None,
                "active_week_ratio": None,
                "zero_week_ratio": None,
                "consistency_classification": "insufficient_history",
                "limitations": "Zero completed calendar weeks observed prior to the current partial week.",
            }

        counts = [int(w["commit_count"]) for w in completed_weeks]
        total_completed_commits = sum(counts)
        active_weeks = sum(1 for c in counts if c > 0)
        alpha_raw = active_weeks / float(k_weeks)
        active_ratio = round(alpha_raw, 4)
        zero_ratio = round(1.0 - alpha_raw, 4)

        if k_weeks == 1:
            return {
                "status": "insufficient",
                "completed_weeks_count": 1,
                "incomplete_current_week_excluded": True,
                "mean_weekly_commits": round(float(counts[0]), 2),
                "median_weekly_commits": round(float(counts[0]), 2),
                "stdev_weekly_commits": None,
                "iqr_weekly_commits": None,
                "mad_weekly_commits": None,
                "coefficient_of_variation": None,
                "active_week_ratio": active_ratio,
                "zero_week_ratio": zero_ratio,
                "consistency_classification": "insufficient_history",
                "limitations": "At least 4 completed weeks and 5 commits are required to evaluate weekly consistency.",
            }

        mean_val = statistics.mean(counts)
        median_val = statistics.median(counts)
        stdev_val = statistics.stdev(counts)
        sorted_counts = sorted(float(c) for c in counts)
        iqr_val = round(_linear_percentile(sorted_counts, 0.75) - _linear_percentile(sorted_counts, 0.25), 2)
        mad_val = round(statistics.median([abs(c - median_val) for c in counts]), 2)
        cv_raw = (stdev_val / mean_val) if mean_val > 0.0 else None
        cv_rounded = round(cv_raw, 4) if cv_raw is not None else None

        if k_weeks < 4 or total_completed_commits < 5:
            status = "insufficient"
            classification = "insufficient_history"
            limitations: Optional[str] = (
                f"Observed {k_weeks} completed weeks and {total_completed_commits} completed-week commits; "
                "minimum 4 completed weeks and 5 commits required for consistency classification."
            )
        else:
            status = "limited" if k_weeks < 8 else "sufficient"
            limitations = (
                "Limited historical span (< 8 completed weeks); variability estimates may stabilize with more weeks."
                if k_weeks < 8
                else None
            )

            # Classify volume + regularity without mislabeling steady low volume as volatile or anomalous
            if mean_val == 0.0 or alpha_raw < 0.5 or (mean_val < 1.0 and (cv_raw is None or cv_raw > 1.0)):
                classification = "sparse_intermittent"
            elif mean_val < 2.0 and cv_raw is not None and cv_raw <= 0.75:
                classification = "consistently_low"
            elif alpha_raw >= 0.75 and mean_val >= 2.0 and cv_raw is not None and cv_raw <= 0.5:
                classification = "consistent_steady"
            elif cv_raw is not None and cv_raw <= 1.0:
                classification = "moderately_variable"
            else:
                classification = "highly_volatile"

        return {
            "status": status,
            "completed_weeks_count": k_weeks,
            "incomplete_current_week_excluded": True,
            "mean_weekly_commits": round(mean_val, 2),
            "median_weekly_commits": round(float(median_val), 2),
            "stdev_weekly_commits": round(stdev_val, 2),
            "iqr_weekly_commits": iqr_val,
            "mad_weekly_commits": mad_val,
            "coefficient_of_variation": cv_rounded,
            "active_week_ratio": active_ratio,
            "zero_week_ratio": zero_ratio,
            "consistency_classification": classification,
            "limitations": limitations,
        }

    def _calculate_rolling_statistics(
        self,
        completed_weeks: List[Dict[str, Any]],
        partial_current_week: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute rolling 4w/12w means, stdevs, medians, and 4w EWMA over completed calendar weeks."""
        k_weeks = len(completed_weeks)
        ewma_span = 4
        ewma_alpha = round(2.0 / (ewma_span + 1.0), 4)

        if k_weeks == 0:
            return {
                "status": "insufficient",
                "window_sizes_weeks": [4, 12],
                "ewma_span_weeks": ewma_span,
                "ewma_alpha": ewma_alpha,
                "completed_weeks_count": 0,
                "incomplete_current_week": partial_current_week,
                "latest": {
                    "week_start": None,
                    "rolling_4w_mean": None,
                    "rolling_4w_stdev": None,
                    "rolling_4w_median": None,
                    "rolling_12w_mean": None,
                    "rolling_12w_stdev": None,
                    "rolling_12w_median": None,
                    "ewma_4w": None,
                    "window_4w_complete": False,
                    "window_12w_complete": False,
                },
                "series": [],
                "limitations": "No completed calendar weeks available prior to analysis_reference_time week.",
            }

        counts = [int(w["commit_count"]) for w in completed_weeks]
        series: List[Dict[str, Any]] = []
        ewma_val = float(counts[0])

        for i in range(k_weeks):
            if i > 0:
                ewma_val = ewma_alpha * float(counts[i]) + (1.0 - ewma_alpha) * ewma_val

            w4_complete = i >= 3
            w12_complete = i >= 11

            r4_mean = r4_stdev = r4_median = None
            if w4_complete:
                slice_4 = counts[i - 3 : i + 1]
                r4_mean = round(statistics.mean(slice_4), 2)
                r4_stdev = round(statistics.stdev(slice_4), 2)
                r4_median = round(float(statistics.median(slice_4)), 2)

            r12_mean = r12_stdev = r12_median = None
            if w12_complete:
                slice_12 = counts[i - 11 : i + 1]
                r12_mean = round(statistics.mean(slice_12), 2)
                r12_stdev = round(statistics.stdev(slice_12), 2)
                r12_median = round(float(statistics.median(slice_12)), 2)

            series.append({
                "week_start": completed_weeks[i]["week_start"],
                "commit_count": counts[i],
                "rolling_4w_mean": r4_mean,
                "rolling_4w_stdev": r4_stdev,
                "rolling_4w_median": r4_median,
                "rolling_12w_mean": r12_mean,
                "rolling_12w_stdev": r12_stdev,
                "rolling_12w_median": r12_median,
                "ewma_4w": round(ewma_val, 2),
                "window_4w_complete": w4_complete,
                "window_12w_complete": w12_complete,
            })

        status = "insufficient" if k_weeks < 4 else ("limited" if k_weeks < 12 else "sufficient")
        limitations: Optional[str] = None
        if k_weeks < 4:
            limitations = f"Only {k_weeks} completed weeks observed; 4-week and 12-week rolling windows are incomplete."
        elif k_weeks < 12:
            limitations = f"Only {k_weeks} completed weeks observed; 12-week rolling statistics require 12 full weeks."

        # Retain trailing 104 completed weeks in payload while keeping full-history EWMA warmup
        bounded_series = series[-104:] if len(series) > 104 else series

        return {
            "status": status,
            "window_sizes_weeks": [4, 12],
            "ewma_span_weeks": ewma_span,
            "ewma_alpha": ewma_alpha,
            "completed_weeks_count": k_weeks,
            "incomplete_current_week": partial_current_week,
            "latest": bounded_series[-1],
            "series": bounded_series,
            "limitations": limitations,
        }

    def _evaluate_week_anomaly(
        self,
        observed: int,
        baseline_vals: List[int],
        week_start: str,
        window_start: str,
        window_end: str,
        period_status: str,
        allow_drop: bool,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single week against its trailing prior baseline using robust MAD with safe fallbacks."""
        b_median = float(statistics.median(baseline_vals))
        b_mean = float(statistics.mean(baseline_vals))
        mad = float(statistics.median([abs(x - b_median) for x in baseline_vals]))
        b_stdev = float(statistics.stdev(baseline_vals)) if len(baseline_vals) >= 2 else 0.0

        if mad > 0.0:
            method = "rolling_robust_mad"
            scale = max(1.4826 * mad, 1.0)
            center = b_median
            method_note = "Evaluated via rolling modified Z-score (1.4826 * MAD)."
        elif b_stdev > 0.0:
            method = "rolling_zscore"
            scale = max(b_stdev, 1.0)
            center = b_mean
            method_note = "Zero baseline MAD; fell back to rolling sample Z-score with dispersion floor >= 1.0."
        else:
            method = "rolling_poisson_floored"
            scale = max(1.0, math.sqrt(max(1.0, b_mean)))
            center = b_mean
            method_note = "Constant prior baseline (MAD=0, stdev=0); used Poisson dispersion floor sqrt(mean)."

        score = (observed - center) / scale
        direction: Optional[str] = None

        # High surge criteria: statistical significance + absolute volume + >=2x baseline magnification
        is_high = (
            observed >= 6
            and (observed - b_median) >= 4.0
            and observed >= 2.0 * max(1.0, b_median)
            and score >= 3.0
        )
        if is_high:
            direction = "high"
        elif allow_drop:
            # Drop criteria: established active baseline (>=75% non-zero, median & mean >= 4) + >=75% reduction
            nonzero_frac = sum(1 for x in baseline_vals if x > 0) / float(len(baseline_vals))
            if nonzero_frac >= 0.75 and b_median >= 4.0 and b_mean >= 4.0:
                if observed <= 0.25 * b_median and (
                    score <= -2.5 or (observed == 0 and b_median >= 5.0 and score <= -2.0)
                ):
                    direction = "low"

        if direction is None:
            return None

        is_high_conf = len(baseline_vals) == 8 and method == "rolling_robust_mad" and period_status == "completed_week"
        confidence = "high" if is_high_conf else "moderate"

        caveats = [method_note]
        if len(baseline_vals) < 8:
            caveats.append(f"Baseline used {len(baseline_vals)} prior weeks (<8 ideal).")
        if period_status == "partial_current_week":
            caveats.append("Detected within an in-progress partial week exceeding full-week spike threshold.")

        return {
            "week_start": week_start,
            "period_status": period_status,
            "direction": direction,
            "observed_commits": observed,
            "baseline": {
                "window_weeks": len(baseline_vals),
                "window_start": window_start,
                "window_end": window_end,
                "median": round(b_median, 2),
                "mean": round(b_mean, 2),
                "scale": round(scale, 3),
            },
            "deviation": {
                "score": round(score, 2),
                "absolute_delta": round(float(observed) - b_median, 2),
                "ratio_to_baseline": round(float(observed) / b_median, 2) if b_median > 0.0 else None,
            },
            "method": method,
            "confidence": confidence,
            "limitations": " ".join(caveats),
        }

    def _detect_anomalies(
        self,
        completed_weeks: List[Dict[str, Any]],
        consistency: Dict[str, Any],
        partial_current_week: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detect statistically anomalous weekly spikes/drops with sparsity and minimum-sample guards."""
        k_weeks = len(completed_weeks)
        counts = [int(w["commit_count"]) for w in completed_weeks]
        total_completed = sum(counts)
        active_weeks = sum(1 for c in counts if c > 0)
        alpha = (active_weeks / float(k_weeks)) if k_weeks > 0 else 0.0

        base_meta = {
            "method_selection": "adaptive_rolling_robust_mad_with_zscore_poisson_fallback",
            "min_completed_weeks_required": 8,
            "min_completed_commits_required": 15,
            "min_prior_baseline_weeks": 6,
            "max_prior_baseline_weeks": 8,
        }

        if k_weeks < 8 or total_completed < 15:
            return {
                **base_meta,
                "status": "insufficient_history",
                "evaluated_weeks_count": 0,
                "total_anomalies_detected": 0,
                "high_activity_anomalies_count": 0,
                "low_activity_anomalies_count": 0,
                "ongoing_partial_week_surge": None,
                "items": [],
                "limitations": (
                    f"Anomaly detection deferred: observed {k_weeks} completed weeks and {total_completed} commits "
                    "(requires >= 8 completed weeks and >= 15 commits)."
                ),
            }

        if alpha < 0.40 and active_weeks < 6:
            return {
                **base_meta,
                "status": "skipped_sparse_series",
                "evaluated_weeks_count": 0,
                "total_anomalies_detected": 0,
                "high_activity_anomalies_count": 0,
                "low_activity_anomalies_count": 0,
                "ongoing_partial_week_surge": None,
                "items": [],
                "limitations": (
                    f"Series is sparse/zero-heavy (active_week_ratio={round(alpha, 2)}, active_weeks={active_weeks}); "
                    "outlier flagging suppressed to prevent false positives on low-frequency intermittent activity."
                ),
            }

        items: List[Dict[str, Any]] = []
        min_prior = 6
        max_prior = 8

        for k in range(min_prior, k_weeks):
            start_idx = max(0, k - max_prior)
            baseline_slice = counts[start_idx:k]
            hit = self._evaluate_week_anomaly(
                observed=counts[k],
                baseline_vals=baseline_slice,
                week_start=completed_weeks[k]["week_start"],
                window_start=completed_weeks[start_idx]["week_start"],
                window_end=completed_weeks[k - 1]["week_start"],
                period_status="completed_week",
                allow_drop=True,
            )
            if hit is not None:
                items.append(hit)

        # Check whether the in-progress partial week already represents a statistically significant surge
        # (partial weeks are never flagged as drop anomalies because elapsed days < 7.0)
        tail_start = max(0, k_weeks - max_prior)
        ongoing_surge = self._evaluate_week_anomaly(
            observed=int(partial_current_week["commit_count"]),
            baseline_vals=counts[tail_start:k_weeks],
            week_start=partial_current_week["week_start"],
            window_start=completed_weeks[tail_start]["week_start"],
            window_end=completed_weeks[k_weeks - 1]["week_start"],
            period_status="partial_current_week",
            allow_drop=False,
        )

        high_count = sum(1 for item in items if item["direction"] == "high")
        low_count = sum(1 for item in items if item["direction"] == "low")

        return {
            **base_meta,
            "status": "analyzed",
            "evaluated_weeks_count": k_weeks - min_prior,
            "total_anomalies_detected": len(items),
            "high_activity_anomalies_count": high_count,
            "low_activity_anomalies_count": low_count,
            "ongoing_partial_week_surge": ongoing_surge,
            "items": items,
            "limitations": (
                f"Evaluates completed weeks starting at week index {min_prior} using a trailing {min_prior}-{max_prior} "
                "week prior baseline; initial 6 warm-up weeks are excluded from anomaly flagging."
            ),
        }

    def _run_max_ssb_permutation_test(
        self,
        counts: List[int],
        min_segment: int = 5,
        n_permutations: int = 999,
    ) -> Tuple[int, float, float, int]:
        """Run deterministic vectorized max-SSB permutation test across eligible splits.

        Returns:
            Tuple of (best_split_index, observed_max_ssb, empirical_p_perm, eligible_splits_count).
        """
        import numpy as np

        arr = np.asarray(counts, dtype=np.float64)
        k_weeks = len(counts)
        splits = np.arange(min_segment, k_weeks - min_segment + 1)
        n1 = splits.astype(np.float64)
        n2 = (k_weeks - splits).astype(np.float64)
        denom = float(k_weeks) * n1 * n2
        total = float(arr.sum())

        # Observed SSB across all eligible splits: (K * P_t - n1 * T)^2 / (K * n1 * n2)
        obs_cum = np.cumsum(arr)[splits - 1]
        obs_ssb = ((float(k_weeks) * obs_cum - n1 * total) ** 2) / denom
        best_idx = int(np.argmax(obs_ssb))
        best_split = int(splits[best_idx])
        obs_max_ssb = float(obs_ssb[best_idx])

        # Versioned deterministic seed derived from frozen weekly counts vector
        seed_payload = f"osl:max_ssb_perm_v1:{','.join(str(c) for c in counts)}".encode("utf-8")
        seed_int = int.from_bytes(hashlib.sha256(seed_payload).digest()[:16], "big")
        rng = np.random.default_rng(seed_int)

        # Vectorized Fisher-Yates row permutations evaluating identical eligible split indices
        perm_matrix = rng.permuted(np.broadcast_to(arr, (n_permutations, k_weeks)), axis=1)
        perm_cum = np.cumsum(perm_matrix, axis=1)[:, splits - 1]
        perm_ssb = ((float(k_weeks) * perm_cum - n1 * total) ** 2) / denom
        perm_max = np.max(perm_ssb, axis=1)

        exceed_count = int(np.sum(perm_max >= obs_max_ssb - 1e-12))
        p_perm = float((1 + exceed_count) / (n_permutations + 1))

        return best_split, obs_max_ssb, p_perm, int(len(splits))

    def _detect_change_points(
        self,
        completed_weeks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Identify statistically justified structural regime shifts via max-SSB permutation test."""
        k_weeks = len(completed_weeks)
        counts = [int(w["commit_count"]) for w in completed_weeks]
        total_commits = sum(counts)
        active_weeks = sum(1 for c in counts if c > 0)

        min_weeks = 12
        min_segment = 5
        min_commits = 30
        min_active = 6
        n_permutations = 999
        alpha_level = 0.01

        base_meta = {
            "method": "max_ssb_permutation_test",
            "permutations": n_permutations,
            "significance_level": alpha_level,
            "min_completed_weeks_required": min_weeks,
            "min_segment_weeks": min_segment,
            "min_commits_required": min_commits,
            "min_active_weeks_required": min_active,
            "evaluated_completed_weeks": k_weeks,
        }

        if k_weeks < min_weeks or total_commits < min_commits or active_weeks < min_active:
            return {
                **base_meta,
                "status": "insufficient_history",
                "justified": False,
                "detected_change_points": [],
                "limitations": (
                    f"Structural change-point analysis deferred: requires >= {min_weeks} completed weeks, "
                    f">= {min_commits} commits, and >= {min_active} active weeks "
                    f"(observed {k_weeks} weeks, {total_commits} commits, {active_weeks} active weeks)."
                ),
            }

        best_split, obs_max_ssb, p_perm, eligible_splits = self._run_max_ssb_permutation_test(
            counts=counts,
            min_segment=min_segment,
            n_permutations=n_permutations,
        )

        before = counts[:best_split]
        after = counts[best_split:]
        mean_before = float(statistics.mean(before))
        mean_after = float(statistics.mean(after))
        med_before = float(statistics.median(before))
        med_after = float(statistics.median(after))

        abs_mean_shift = abs(mean_after - mean_before)
        abs_med_shift = abs(med_after - med_before)
        rel_ratio = max(mean_after / max(0.5, mean_before), mean_before / max(0.5, mean_after))

        # Secondary diagnostic only: unadjusted Mann-Whitney U does NOT govern significance
        from scipy.stats import mannwhitneyu

        mw_pvalue = float(mannwhitneyu(before, after, alternative="two-sided").pvalue)

        is_significant = (
            p_perm <= alpha_level
            and abs_mean_shift >= 3.0
            and abs_med_shift >= 2.0
            and rel_ratio >= 1.75
        )

        if not is_significant:
            return {
                **base_meta,
                "status": "no_significant_shift",
                "justified": True,
                "detected_change_points": [],
                "limitations": (
                    f"Single-regime max-SSB permutation test (B={n_permutations}, {eligible_splits} eligible splits, "
                    "min_segment=5) evaluated; candidate shifts did not meet p_perm <= 0.01 and practical magnitude filters. "
                    "Assumes exchangeable (non-autocorrelated) observations under the null hypothesis."
                ),
            }

        direction = "step_increase" if mean_after > mean_before else "step_decrease"
        confidence = "high" if (p_perm <= 0.002 and min(len(before), len(after)) >= 7) else "moderate"

        shift_limitations = (
            "Identifies at most one dominant discrete shift with >= 5-week segments using a deterministic "
            "max-SSB permutation test (B=999) assuming exchangeable (non-autocorrelated) observations under the null. "
            "Positive serial autocorrelation, recurring release cycles, or steady gradual trends can inflate permutation "
            "significance without an abrupt step shift; short-lived bursts are isolated as point anomalies."
        )

        cp_record = {
            "change_point_week_start": completed_weeks[best_split]["week_start"],
            "direction": direction,
            "before_segment": {
                "weeks": len(before),
                "start_week": completed_weeks[0]["week_start"],
                "end_week": completed_weeks[best_split - 1]["week_start"],
                "mean_weekly_commits": round(mean_before, 2),
                "median_weekly_commits": round(med_before, 2),
            },
            "after_segment": {
                "weeks": len(after),
                "start_week": completed_weeks[best_split]["week_start"],
                "end_week": completed_weeks[-1]["week_start"],
                "mean_weekly_commits": round(mean_after, 2),
                "median_weekly_commits": round(med_after, 2),
            },
            "statistics": {
                "permutation_pvalue": round(p_perm, 6),
                "observed_max_ssb": round(obs_max_ssb, 4),
                "permutations": n_permutations,
                "eligible_splits_count": eligible_splits,
                "mann_whitney_u_diagnostic_pvalue": round(mw_pvalue, 6),
                "absolute_mean_shift": round(abs_mean_shift, 2),
                "absolute_median_shift": round(abs_med_shift, 2),
                "relative_shift_ratio": round(rel_ratio, 2),
            },
            "method": "max_ssb_permutation_test",
            "confidence": confidence,
            "limitations": shift_limitations,
        }

        return {
            **base_meta,
            "status": "detected",
            "justified": True,
            "detected_change_points": [cp_record],
            "limitations": shift_limitations,
        }

    def _build_c22_rules(
        self,
        consistency_metrics: Dict[str, Any],
        anomaly_results: Dict[str, Any],
        change_point_results: Dict[str, Any],
    ) -> List[RuleInterpretation]:
        """Generate deterministic, evidence-backed RuleInterpretation entries for C.2.2 signals."""
        rules: List[RuleInterpretation] = []

        # RULE-ACT-004: Activity Consistency Classification
        cls_name = consistency_metrics["consistency_classification"]
        if cls_name == "consistent_steady":
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-004",
                    domain=self.domain_name,
                    severity="positive",
                    finding="Consistent Weekly Activity: Repository maintains steady week-to-week commit cadence with low variability.",
                    evidence={
                        "consistency_classification": cls_name,
                        "coefficient_of_variation": consistency_metrics["coefficient_of_variation"],
                        "active_week_ratio": consistency_metrics["active_week_ratio"],
                        "mean_weekly_commits": consistency_metrics["mean_weekly_commits"],
                    },
                    recommendation=None,
                )
            )
        elif cls_name == "consistently_low":
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-004",
                    domain=self.domain_name,
                    severity="info",
                    finding="Consistently Low Activity: Regular low-volume maintenance cadence with low week-to-week variability.",
                    evidence={
                        "consistency_classification": cls_name,
                        "coefficient_of_variation": consistency_metrics["coefficient_of_variation"],
                        "active_week_ratio": consistency_metrics["active_week_ratio"],
                        "mean_weekly_commits": consistency_metrics["mean_weekly_commits"],
                    },
                    recommendation=None,
                )
            )
        elif cls_name == "highly_volatile":
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-004",
                    domain=self.domain_name,
                    severity="warning",
                    finding="Highly Variable Weekly Activity: Commit output fluctuates sharply across completed weeks.",
                    evidence={
                        "consistency_classification": cls_name,
                        "coefficient_of_variation": consistency_metrics["coefficient_of_variation"],
                        "iqr_weekly_commits": consistency_metrics["iqr_weekly_commits"],
                    },
                    recommendation="Inspect burst weeks and release-driven batches using anomaly and cadence views.",
                )
            )
        elif cls_name == "sparse_intermittent":
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-004",
                    domain=self.domain_name,
                    severity="info",
                    finding="Sparse/Intermittent Activity: Commits occur periodically in fewer than half of observed weeks.",
                    evidence={
                        "consistency_classification": cls_name,
                        "active_week_ratio": consistency_metrics["active_week_ratio"],
                        "zero_week_ratio": consistency_metrics["zero_week_ratio"],
                    },
                    recommendation=None,
                )
            )

        # RULE-ACT-005: Statistically Significant Activity Anomalies (emitted only when triggered)
        anomaly_items = anomaly_results["items"]
        ongoing_surge = anomaly_results["ongoing_partial_week_surge"]
        if anomaly_items or ongoing_surge:
            peak = (
                max(anomaly_items, key=lambda x: abs(x["deviation"]["score"]))
                if anomaly_items
                else ongoing_surge
            )
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-005",
                    domain=self.domain_name,
                    severity="info" if peak["direction"] == "high" else "warning",
                    finding=(
                        f"Activity Anomaly Detected: {len(anomaly_items)} completed-week anomaly(ies)"
                        + (" plus an ongoing partial-week surge" if ongoing_surge else "")
                        + f" identified (peak in week {peak['week_start']}: {peak['observed_commits']} vs baseline median {peak['baseline']['median']})."
                    ),
                    evidence={
                        "completed_anomalies_count": len(anomaly_items),
                        "has_ongoing_partial_week_surge": ongoing_surge is not None,
                        "peak_anomaly": peak,
                    },
                    recommendation="Review commit activity and release events around the flagged week(s).",
                )
            )

        # RULE-ACT-006: Justified Structural Change-Point
        cps = change_point_results["detected_change_points"]
        if cps:
            cp = cps[0]
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-006",
                    domain=self.domain_name,
                    severity="positive" if cp["direction"] == "step_increase" else "warning",
                    finding=(
                        f"Structural Activity Shift ({cp['direction']}): Sustained change beginning week {cp['change_point_week_start']} "
                        f"(median shifted from {cp['before_segment']['median_weekly_commits']} to {cp['after_segment']['median_weekly_commits']} commits/week, "
                        f"p_perm={cp['statistics']['permutation_pvalue']})."
                    ),
                    evidence=cp,
                    recommendation=None,
                )
            )

        return rules

    def _calculate_activity_rhythms(
        self,
        valid_commits: List[Tuple[Any, datetime]],
    ) -> Dict[str, Any]:
        """Compute 7x24 UTC weekday/hour commit distribution with sample gating and UTC caveats."""
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        total_commits = len(valid_commits)
        min_sample = 5
        confident_sample = 20
        utc_limitation = (
            "All commit timestamps are grouped in UTC; 7x24 weekday and hour distributions reflect aggregate "
            "timestamp timing in UTC and do not reveal contributors' local working hours or personal schedules."
        )

        if total_commits == 0 or total_commits < min_sample:
            status = "empty" if total_commits == 0 else "insufficient"
            return {
                "status": status,
                "total_commits": total_commits,
                "min_sample_required": min_sample,
                "confident_sample_required": confident_sample,
                "weekday_names": weekday_names,
                "weekday_hour_matrix": None,
                "weekday_totals": None,
                "hour_totals_utc": None,
                "weekday_commits": 0 if total_commits == 0 else None,
                "weekend_commits": 0 if total_commits == 0 else None,
                "weekday_ratio": None,
                "weekend_ratio": None,
                "peak_weekday": None,
                "peak_utc_hour": None,
                "limitations": (
                    f"Observed {total_commits} commits (< {min_sample} minimum); rhythm distributions deferred. "
                    + utc_limitation
                ),
            }

        matrix = [[0] * 24 for _ in range(7)]
        for _, c_date in valid_commits:
            matrix[c_date.weekday()][c_date.hour] += 1

        weekday_totals = [sum(matrix[d]) for d in range(7)]
        hour_totals = [sum(matrix[d][h] for d in range(7)) for h in range(24)]
        weekday_commits = sum(weekday_totals[:5])
        weekend_commits = sum(weekday_totals[5:])
        weekday_ratio = round(weekday_commits / float(total_commits), 4)
        weekend_ratio = round(1.0 - weekday_ratio, 4)

        peak_day_idx = max(range(7), key=lambda d: weekday_totals[d])
        peak_hour_idx = max(range(24), key=lambda h: hour_totals[h])

        status = "limited" if total_commits < confident_sample else "sufficient"
        limitations = (
            f"Limited commit sample ({total_commits} < {confident_sample}); day/hour peaks may shift with additional history. {utc_limitation}"
            if status == "limited"
            else utc_limitation
        )

        return {
            "status": status,
            "total_commits": total_commits,
            "min_sample_required": min_sample,
            "confident_sample_required": confident_sample,
            "weekday_names": weekday_names,
            "weekday_hour_matrix": matrix,
            "weekday_totals": weekday_totals,
            "hour_totals_utc": hour_totals,
            "weekday_commits": weekday_commits,
            "weekend_commits": weekend_commits,
            "weekday_ratio": weekday_ratio,
            "weekend_ratio": weekend_ratio,
            "peak_weekday": {
                "index": peak_day_idx,
                "name": weekday_names[peak_day_idx],
                "commit_count": weekday_totals[peak_day_idx],
            },
            "peak_utc_hour": {
                "hour_utc": peak_hour_idx,
                "commit_count": hour_totals[peak_hour_idx],
            },
            "limitations": limitations,
        }

    def _calculate_code_churn(
        self,
        valid_commits: List[Tuple[Any, datetime]],
        completed_weeks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute diff-coverage-aware code churn without conflating NULL diffs with zero edits."""
        eligible = len(valid_commits)
        min_diff_commits = 5
        min_completed_weeks_volatility = 4
        min_coverage_sufficient = 0.50

        with_diff = 0
        missing_diff = 0
        zero_change_diff = 0
        total_adds = 0
        total_dels = 0
        weekly_gross: Dict[str, int] = defaultdict(int)
        weekly_missing_diff: Dict[str, int] = defaultdict(int)

        for row, c_date in valid_commits:
            w_str = _get_week_start_utc(c_date).strftime("%Y-%m-%d")
            if row.additions is not None and row.deletions is not None:
                adds = int(row.additions)
                dels = int(row.deletions)
                with_diff += 1
                if adds == 0 and dels == 0:
                    zero_change_diff += 1
                total_adds += adds
                total_dels += dels
                weekly_gross[w_str] += adds + dels
            else:
                missing_diff += 1
                weekly_missing_diff[w_str] += 1

        diff_cov = round(with_diff / float(eligible), 4) if eligible > 0 else None

        base_cov = {
            "eligible_commits_count": eligible,
            "commits_with_diff_count": with_diff,
            "commits_missing_diff_count": missing_diff,
            "zero_change_diff_commits_count": zero_change_diff,
            "diff_coverage_ratio": diff_cov,
            "min_diff_commits_required": min_diff_commits,
            "min_coverage_ratio_for_sufficient": min_coverage_sufficient,
            "min_completed_weeks_for_volatility": min_completed_weeks_volatility,
        }

        if eligible == 0 or with_diff == 0 or with_diff < min_diff_commits:
            if eligible == 0:
                status = "empty"
                reason = "No commits found in repository snapshot."
            elif with_diff == 0:
                status = "diff_metrics_not_collected"
                reason = "Diff line stats (additions/deletions) are NULL across all eligible commits."
            else:
                status = "insufficient"
                reason = (
                    f"Only {with_diff} of {eligible} eligible commits have diff stats "
                    f"(minimum {min_diff_commits} required)."
                )
            return {
                **base_cov,
                "status": status,
                "total_additions": None,
                "total_deletions": None,
                "gross_code_churn": None,
                "net_code_delta": None,
                "churn_per_commit": None,
                "weekly_churn_volatility": None,
                "completed_weeks_with_complete_diffs": 0,
                "limitations": reason,
            }

        gross_churn = total_adds + total_dels
        net_delta = total_adds - total_dels
        churn_per_commit = round(gross_churn / float(with_diff), 2)

        # Exclude completed weeks that contain commits with uncollected (NULL) diffs
        # rather than fabricating zero churn for unsampled commits.
        clean_weekly_churns: List[int] = [
            weekly_gross.get(w["week_start"], 0)
            for w in completed_weeks
            if weekly_missing_diff.get(w["week_start"], 0) == 0
        ]
        completed_clean_count = len(clean_weekly_churns)

        is_sufficient_coverage = diff_cov is not None and diff_cov >= min_coverage_sufficient
        status = "sufficient" if is_sufficient_coverage else "limited"

        if is_sufficient_coverage and completed_clean_count >= min_completed_weeks_volatility:
            weekly_churn_volatility: Optional[float] = round(statistics.stdev(clean_weekly_churns), 2)
        else:
            weekly_churn_volatility = None

        limitation_notes: List[str] = []
        if not is_sufficient_coverage:
            limitation_notes.append(
                f"Partial diff sampling ({round((diff_cov or 0.0) * 100.0, 1)}% coverage across {eligible} commits); "
                "churn aggregates cover sampled commits only and weekly churn volatility is deferred."
            )
        elif weekly_churn_volatility is None:
            limitation_notes.append(
                f"Only {completed_clean_count} completed weeks with complete diff coverage "
                f"(>= {min_completed_weeks_volatility} required for weekly churn volatility)."
            )
        limitation_notes.append(
            "Line additions and deletions can be skewed by vendored dependencies, lockfiles, or code formatting."
        )

        return {
            **base_cov,
            "status": status,
            "total_additions": total_adds,
            "total_deletions": total_dels,
            "gross_code_churn": gross_churn,
            "net_code_delta": net_delta,
            "churn_per_commit": churn_per_commit,
            "weekly_churn_volatility": weekly_churn_volatility,
            "completed_weeks_with_complete_diffs": completed_clean_count,
            "limitations": " ".join(limitation_notes),
        }

    def _calculate_commit_landing_latency(
        self,
        valid_commits: List[Tuple[Any, datetime]],
    ) -> Dict[str, Any]:
        """Compute authoring-to-landing latency (committer_date - author_date) strictly without substitution."""
        eligible = len(valid_commits)
        min_sample = 5
        confident_sample = 20
        min_coverage_sufficient = 0.50

        with_committer = 0
        missing_committer = 0
        missing_author = 0
        negative_lags: List[float] = []
        nonneg_lags: List[float] = []
        positive_delay_count = 0
        zero_lag_count = 0

        for row, _ in valid_commits:
            c_dt = _ensure_utc(row.committer_date)
            if c_dt is None:
                missing_committer += 1
                continue
            with_committer += 1

            a_dt = _ensure_utc(row.author_date)
            if a_dt is None:
                missing_author += 1
                continue

            lag_hours = (c_dt - a_dt).total_seconds() / 3600.0
            if lag_hours < 0.0:
                negative_lags.append(lag_hours)
            else:
                nonneg_lags.append(lag_hours)
                if c_dt > a_dt:
                    positive_delay_count += 1
                else:
                    zero_lag_count += 1

        committer_cov = round(with_committer / float(eligible), 4) if eligible > 0 else None
        valid_nonneg = len(nonneg_lags)
        skew_count = len(negative_lags)

        base_meta = {
            "eligible_commits_count": eligible,
            "commits_with_committer_date_count": with_committer,
            "commits_missing_committer_date_count": missing_committer,
            "commits_missing_author_date_count": missing_author,
            "committer_date_coverage_ratio": committer_cov,
            "valid_non_negative_count": valid_nonneg,
            "zero_lag_count": zero_lag_count,
            "positive_delay_count": positive_delay_count,
            "negative_clock_skew_count": skew_count,
            "min_negative_clock_skew_hours": round(min(negative_lags), 2) if negative_lags else None,
            "min_sample_required": min_sample,
            "confident_sample_required": confident_sample,
        }

        if eligible == 0 or with_committer == 0 or valid_nonneg == 0 or valid_nonneg < min_sample:
            if eligible == 0:
                status = "empty"
                reason = "No commits found in repository snapshot."
            elif with_committer == 0:
                status = "committer_dates_not_collected"
                reason = "All eligible commits have committer_date IS NULL; author_date is never substituted."
            elif valid_nonneg == 0:
                status = "no_valid_non_negative_intervals"
                reason = f"All {skew_count} commit pairs with committer_date had negative clock-skew intervals."
            else:
                status = "insufficient"
                reason = (
                    f"Only {valid_nonneg} valid non-negative commit interval(s) observed "
                    f"(minimum {min_sample} required for stable latency percentiles)."
                )
            return {
                **base_meta,
                "status": status,
                "positive_delay_share": None,
                "mean_latency_hours": None,
                "p25_latency_hours": None,
                "p50_latency_hours": None,
                "p75_latency_hours": None,
                "p90_latency_hours": None,
                "limitations": reason,
            }

        sorted_lags = sorted(nonneg_lags)
        positive_share = round(positive_delay_count / float(valid_nonneg), 4)
        is_sufficient = (
            valid_nonneg >= confident_sample
            and committer_cov is not None
            and committer_cov >= min_coverage_sufficient
        )
        status = "sufficient" if is_sufficient else "limited"

        notes: List[str] = []
        if committer_cov is not None and committer_cov < 1.0:
            notes.append(f"Partial committer_date coverage ({round(committer_cov * 100.0, 1)}% of eligible commits).")
        if valid_nonneg < confident_sample:
            notes.append(f"Sample size {valid_nonneg} < {confident_sample}; quantiles may shift with more commits.")
        if skew_count > 0:
            notes.append(f"Excluded {skew_count} negative clock-skew interval(s) from non-negative latency percentiles.")
        notes.append("Strict non-substitution: missing committer_date records are never filled from author_date.")

        return {
            **base_meta,
            "status": status,
            "positive_delay_share": positive_share,
            "mean_latency_hours": round(statistics.mean(nonneg_lags), 2),
            "p25_latency_hours": round(_linear_percentile(sorted_lags, 0.25), 2),
            "p50_latency_hours": round(_linear_percentile(sorted_lags, 0.50), 2),
            "p75_latency_hours": round(_linear_percentile(sorted_lags, 0.75), 2),
            "p90_latency_hours": round(_linear_percentile(sorted_lags, 0.90), 2),
            "limitations": " ".join(notes),
        }

    def _build_c23_rules(
        self,
        rhythm_metrics: Dict[str, Any],
        churn_metrics: Dict[str, Any],
        latency_metrics: Dict[str, Any],
    ) -> List[RuleInterpretation]:
        """Generate deterministic, sample-gated RuleInterpretation findings for C.2.3 signals."""
        rules: List[RuleInterpretation] = []

        # RULE-ACT-007: UTC Activity Rhythm Concentration (sufficient >= 20 commits)
        if rhythm_metrics["status"] == "sufficient":
            weekend_ratio = rhythm_metrics["weekend_ratio"]
            weekday_ratio = rhythm_metrics["weekday_ratio"]
            if weekend_ratio is not None and weekend_ratio >= 0.40:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-ACT-007",
                        domain=self.domain_name,
                        severity="info",
                        finding=(
                            f"Notable UTC Weekend Commit Share: {round(weekend_ratio * 100.0, 1)}% of commits fall on "
                            "Saturday/Sunday UTC (UTC timestamps do not indicate contributors' local working hours)."
                        ),
                        evidence={
                            "weekend_ratio": weekend_ratio,
                            "weekday_ratio": weekday_ratio,
                            "total_commits": rhythm_metrics["total_commits"],
                            "peak_weekday": rhythm_metrics["peak_weekday"],
                        },
                        recommendation=None,
                    )
                )
            elif weekday_ratio is not None and weekday_ratio >= 0.85:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-ACT-007",
                        domain=self.domain_name,
                        severity="info",
                        finding=(
                            f"Weekday-Concentrated UTC Rhythm: {round(weekday_ratio * 100.0, 1)}% of commits occur on "
                            "Monday–Friday UTC (reflects UTC timestamps, not contributor local hours)."
                        ),
                        evidence={
                            "weekday_ratio": weekday_ratio,
                            "weekend_ratio": weekend_ratio,
                            "total_commits": rhythm_metrics["total_commits"],
                            "peak_utc_hour": rhythm_metrics["peak_utc_hour"],
                        },
                        recommendation=None,
                    )
                )

        # RULE-ACT-008: Authoring-to-Landing Commit Latency (sample >= 5 valid intervals)
        if latency_metrics["status"] in ("limited", "sufficient"):
            pos_share = latency_metrics["positive_delay_share"]
            p50 = latency_metrics["p50_latency_hours"]
            if pos_share is not None and p50 is not None:
                if pos_share >= 0.50 and p50 >= 24.0:
                    rules.append(
                        RuleInterpretation(
                            rule_id="RULE-ACT-008",
                            domain=self.domain_name,
                            severity="info",
                            finding=(
                                f"Delayed Commit Landing Latency: Median authoring-to-landing interval is {p50}h across "
                                f"{round(pos_share * 100.0, 1)}% of commits with committer_date > author_date."
                            ),
                            evidence={
                                "p50_latency_hours": p50,
                                "p90_latency_hours": latency_metrics["p90_latency_hours"],
                                "positive_delay_share": pos_share,
                                "committer_date_coverage_ratio": latency_metrics["committer_date_coverage_ratio"],
                            },
                            recommendation=None,
                        )
                    )
                elif pos_share <= 0.10:
                    rules.append(
                        RuleInterpretation(
                            rule_id="RULE-ACT-008",
                            domain=self.domain_name,
                            severity="info",
                            finding=(
                                "Immediate Commit Landing Pattern: Over 90% of valid commits record identical "
                                "author_date and committer_date timestamps."
                            ),
                            evidence={
                                "p50_latency_hours": p50,
                                "positive_delay_share": pos_share,
                                "committer_date_coverage_ratio": latency_metrics["committer_date_coverage_ratio"],
                            },
                            recommendation=None,
                        )
                    )

        # RULE-ACT-009: Diff-Aware Code Churn Summary (emitted only when diff coverage >= 50% & >= 5 diffs)
        if churn_metrics["status"] == "sufficient":
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-ACT-009",
                    domain=self.domain_name,
                    severity="info",
                    finding=(
                        f"Measured Code Churn Profile: {churn_metrics['gross_code_churn']} total changed lines across "
                        f"{churn_metrics['commits_with_diff_count']} commits with diffs "
                        f"({churn_metrics['churn_per_commit']} lines/commit, net delta {churn_metrics['net_code_delta']:+d})."
                    ),
                    evidence={
                        "gross_code_churn": churn_metrics["gross_code_churn"],
                        "net_code_delta": churn_metrics["net_code_delta"],
                        "churn_per_commit": churn_metrics["churn_per_commit"],
                        "diff_coverage_ratio": churn_metrics["diff_coverage_ratio"],
                    },
                    recommendation=None,
                )
            )

        return rules
