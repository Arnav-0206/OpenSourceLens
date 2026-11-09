"""Development Activity Intelligence Module for OpenSourceLens.

Implements C.2.1 Activity Aggregation adhering to BaseMetricCalculator:
- Daily and weekly commit activity time-series aggregation.
- Continuous zero-filled calendar intervals.
- Multi-horizon weekly commit velocity (trailing 4w, 12w, 52w, overall).
- Contributor activity & active author aggregation.
- Release cadence, interval statistics (mean, median, min, max, stdev), and classification.
- Robust timezone normalization to UTC.
- Complete coverage assessment and deterministic rule-based diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
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

        # Filter commits up to frozen reference time
        valid_commits = []
        for row in commits_rows:
            commit_date = _ensure_utc(row.author_date or row.committer_date)
            if commit_date is not None and commit_date <= ref_time:
                valid_commits.append((row, commit_date))

        total_commits = len(valid_commits)

        # Filter releases up to frozen reference time
        valid_releases = []
        for rel in releases_rows:
            pub_date = _ensure_utc(rel.published_at)
            if pub_date is not None and pub_date <= ref_time:
                valid_releases.append((rel, pub_date))

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
        # Trailing 4 weeks (last 4 elements of continuous weekly series)
        trailing_4w_series = weekly_time_series[-4:] if len(weekly_time_series) >= 4 else weekly_time_series
        trailing_4w_commits = sum(w["commit_count"] for w in trailing_4w_series)
        trailing_4w_velocity = round(trailing_4w_commits / 4.0, 2)

        # Trailing 12 weeks
        trailing_12w_series = weekly_time_series[-12:] if len(weekly_time_series) >= 12 else weekly_time_series
        trailing_12w_commits = sum(w["commit_count"] for w in trailing_12w_series)
        trailing_12w_velocity = round(trailing_12w_commits / 12.0, 2)

        # Trailing 52 weeks
        trailing_52w_series = weekly_time_series[-52:] if len(weekly_time_series) >= 52 else weekly_time_series
        trailing_52w_commits = sum(w["commit_count"] for w in trailing_52w_series)
        divisor_52w = min(52.0, max(1.0, float(len(weekly_time_series))))
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
            "weekly_time_series": [],
            "daily_time_series_90d": [],
        }

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status="empty",
            sample_size=0,
            min_sample_required=5,
            details={"reason": "No commits found in repository snapshot", "total_commits": 0},
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
