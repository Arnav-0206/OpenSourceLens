"""Release & Project Evolution Analytics Engine for OpenSourceLens (Module 5 / Phase C.6).

Implements ReleaseEvolutionCalculator adhering to BaseMetricCalculator:
- Release cadence, inter-release interval statistics (mean, median, min, max, stdev), and annualized frequency.
- Release predictability dispersion (CV_release) gated on R >= 3 releases and mean interval > 0.
- Pre-release activity surges (14-day pre-release vs preceding 60-day baseline commit rates):
  - Enforces >= 74 days of prior observed commit history per release (insufficient_baseline_depth).
  - Explicitly flags zero baseline commit rates (dormant_baseline_activity) without division by zero.
  - Requires >= 2 valid surge-eligible releases for aggregate mean/median surge ratios.
- Semantic Versioning (SemVer 2.0.0) adoption rate and Major/Minor/Patch/Prerelease classification.
- Strictly uses existing Release ORM fields (published_at <= ref_time, is_prerelease) without fallback schema inventions.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
import re
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import CollectionRun, Commit, Release
from backend.app.schemas.analytics_v2 import MetricCoverageItem
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.development_activity import _ensure_utc
from backend.app.services.analytics.pull_requests import _extract_stream_status
from backend.app.services.analytics.rule_engine import build_release_rules


SEMVER_PATTERN = re.compile(
    r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


class ReleaseEvolutionCalculator(BaseMetricCalculator):
    """Calculates release cadence regularity, pre-release commit surges, and SemVer adoption."""

    @property
    def domain_name(self) -> str:
        return "release_evolution"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute release evolution intelligence anchored strictly to frozen analysis_reference_time."""
        ref_time = _ensure_utc(analysis_reference_time)

        releases_rows = db.execute(
            select(
                Release.release_id,
                Release.tag_name,
                Release.name,
                Release.published_at,
                Release.is_prerelease,
            ).where(Release.repository_id == repository_id)
        ).fetchall()

        commits_rows = db.execute(
            select(
                Commit.sha,
                Commit.author_date,
                Commit.committer_date,
            ).where(Commit.repository_id == repository_id)
        ).fetchall()

        latest_collection_run = db.scalar(
            select(CollectionRun)
            .where(CollectionRun.repository_id == repository_id)
            .order_by(CollectionRun.started_at.desc(), CollectionRun.id.desc())
            .limit(1)
        )

        # Deduplicate releases, exclude published_at IS NULL and future (> ref_time) releases
        seen_release_ids: Set[int] = set()
        valid_releases: List[Tuple[Any, datetime]] = []
        missing_published_at_count = 0
        future_releases_excluded_count = 0

        for row in releases_rows:
            if row.release_id in seen_release_ids:
                continue
            seen_release_ids.add(row.release_id)
            pub_dt = _ensure_utc(row.published_at)
            if pub_dt is None:
                missing_published_at_count += 1
                continue
            if pub_dt > ref_time:
                future_releases_excluded_count += 1
                continue
            valid_releases.append((row, pub_dt))

        valid_releases.sort(key=lambda item: (item[1], item[0].release_id))

        # Deduplicate commits <= ref_time for pre-release surge baseline checks
        seen_shas: Set[str] = set()
        commit_dates: List[datetime] = []
        for row in commits_rows:
            if row.sha in seen_shas:
                continue
            c_dt = _ensure_utc(row.author_date or row.committer_date)
            if c_dt is None or c_dt > ref_time:
                continue
            seen_shas.add(row.sha)
            commit_dates.append(c_dt)
        commit_dates.sort()

        collection_scope = self._inspect_collection_scope(latest_collection_run)

        cadence = self._evaluate_cadence(
            valid_releases=valid_releases,
            ref_time=ref_time,
            missing_published_at_count=missing_published_at_count,
            future_releases_excluded_count=future_releases_excluded_count,
        )
        surges = self._evaluate_prerelease_surges(
            valid_releases=valid_releases,
            commit_dates=commit_dates,
        )
        semver = self._evaluate_semver(valid_releases=valid_releases)

        coverage, confidence_penalty, penalties = self._evaluate_coverage_and_confidence(
            valid_releases=valid_releases,
            cadence=cadence,
            surges=surges,
            semver=semver,
            collection_scope=collection_scope,
        )

        rules = build_release_rules(
            domain_name=self.domain_name,
            cadence=cadence,
            surges=surges,
            semver=semver,
        )

        metrics: Dict[str, Any] = {
            "scope": collection_scope,
            "cadence": cadence,
            "prerelease_surges": surges,
            "semver_adoption": semver,
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
    def _inspect_collection_scope(latest_run: Optional[CollectionRun]) -> Dict[str, Any]:
        """Inspect CollectionRun scope for releases and commits streams."""
        if latest_run is None:
            return {
                "mode": "unknown_unverified",
                "is_partial": True,
                "is_bounded_or_partial": True,
                "releases_stream_status": None,
                "commits_stream_status": None,
                "notes": "No CollectionRun metadata found; release history reflects stored database records.",
            }

        mode = (latest_run.mode or "unknown").lower().strip()
        cov = latest_run.coverage_summary if isinstance(latest_run.coverage_summary, dict) else {}
        rel_status = _extract_stream_status(cov, "releases")
        com_status = _extract_stream_status(cov, "commits")
        bounded_modes = {"fast_snapshot", "standard_diagnostic"}
        stream_truncated = any(st in ("truncated", "partial", "rate_limited") for st in (rel_status, com_status))
        is_bounded_or_partial = (
            bool(latest_run.is_partial)
            or mode in bounded_modes
            or mode != "full_history"
            or latest_run.status != "completed"
            or stream_truncated
        )

        return {
            "mode": mode,
            "is_partial": bool(latest_run.is_partial),
            "is_bounded_or_partial": is_bounded_or_partial,
            "releases_stream_status": rel_status,
            "commits_stream_status": com_status,
            "notes": (
                f"Snapshot ingested under bounded mode '{mode}'."
                if mode in bounded_modes
                else "Release and commit streams evaluated within snapshot scope."
            ),
        }

    @staticmethod
    def _evaluate_cadence(
        valid_releases: List[Tuple[Any, datetime]],
        ref_time: datetime,
        missing_published_at_count: int,
        future_releases_excluded_count: int,
    ) -> Dict[str, Any]:
        """Compute inter-release intervals, annualized frequency, CV_release, and regularity class."""
        r_count = len(valid_releases)
        prereleases_count = sum(1 for rel, _ in valid_releases if rel.is_prerelease)
        stable_count = r_count - prereleases_count

        base_meta: Dict[str, Any] = {
            "total_releases": r_count,
            "stable_releases_count": stable_count,
            "prereleases_count": prereleases_count,
            "missing_published_at_count": missing_published_at_count,
            "future_releases_excluded_count": future_releases_excluded_count,
            "min_sample_for_intervals": 2,
            "min_sample_for_cv": 3,
            "confident_sample_required": 5,
        }

        if r_count == 0:
            return {
                **base_meta,
                "status": "no_releases_recorded",
                "latest_release_tag": None,
                "latest_release_published_at": None,
                "days_since_last_release": None,
                "intervals_count": 0,
                "mean_interval_days": None,
                "median_interval_days": None,
                "min_interval_days": None,
                "max_interval_days": None,
                "stdev_interval_days": None,
                "annualized_release_frequency": None,
                "cv_release": None,
                "regularity_classification": "no_releases_recorded",
                "limitations": "No dated releases recorded up to analysis_reference_time.",
            }

        latest_rel, latest_dt = valid_releases[-1]
        days_since_last = round(max(0.0, (ref_time - latest_dt).total_seconds() / 86400.0), 2)

        if r_count == 1:
            return {
                **base_meta,
                "status": "single_release_recorded",
                "latest_release_tag": latest_rel.tag_name,
                "latest_release_published_at": latest_dt.isoformat(),
                "days_since_last_release": days_since_last,
                "intervals_count": 0,
                "mean_interval_days": None,
                "median_interval_days": None,
                "min_interval_days": None,
                "max_interval_days": None,
                "stdev_interval_days": None,
                "annualized_release_frequency": None,
                "cv_release": None,
                "regularity_classification": "single_release_recorded",
                "limitations": "Single release recorded; inter-release intervals and CV_release require >= 2 and >= 3 releases respectively.",
            }

        intervals: List[float] = []
        for idx in range(1, r_count):
            dt_prev = valid_releases[idx - 1][1]
            dt_cur = valid_releases[idx][1]
            intervals.append(max(0.0, (dt_cur - dt_prev).total_seconds() / 86400.0))

        mean_int = statistics.mean(intervals)
        median_int = statistics.median(intervals)
        min_int = min(intervals)
        max_int = max(intervals)
        annual_freq = round(365.25 / mean_int, 2) if mean_int > 0.0 else None

        cv_release: Optional[float] = None
        stdev_int: Optional[float] = None
        if r_count >= 3:
            stdev_int = statistics.stdev(intervals)
            if mean_int > 0.0:
                cv_release = round(stdev_int / mean_int, 4)
                if cv_release <= 0.35:
                    regularity = "highly_scheduled"
                elif cv_release <= 0.75:
                    regularity = "moderately_predictable"
                else:
                    regularity = "event_driven_sporadic"
            else:
                regularity = "zero_mean_interval"
        else:
            regularity = "insufficient_for_cv"

        status = "sufficient" if r_count >= 5 else "limited"
        notes: List[str] = []
        if status == "limited":
            notes.append(f"Limited release history ({r_count} < 5 releases); cadence stability metrics may vary.")
        if missing_published_at_count > 0:
            notes.append(f"Excluded {missing_published_at_count} release(s) with missing published_at timestamp.")

        return {
            **base_meta,
            "status": status,
            "latest_release_tag": latest_rel.tag_name,
            "latest_release_published_at": latest_dt.isoformat(),
            "days_since_last_release": days_since_last,
            "intervals_count": len(intervals),
            "mean_interval_days": round(mean_int, 2),
            "median_interval_days": round(median_int, 2),
            "min_interval_days": round(min_int, 2),
            "max_interval_days": round(max_int, 2),
            "stdev_interval_days": round(stdev_int, 2) if stdev_int is not None else None,
            "annualized_release_frequency": annual_freq,
            "cv_release": cv_release,
            "regularity_classification": regularity,
            "limitations": " ".join(notes) if notes else None,
        }

    @staticmethod
    def _evaluate_prerelease_surges(
        valid_releases: List[Tuple[Any, datetime]],
        commit_dates: List[datetime],
    ) -> Dict[str, Any]:
        """Compute 14d pre-release vs preceding 60d baseline commit surge ratios with depth & zero-baseline guards."""
        r_count = len(valid_releases)
        min_aggregate_releases = 2

        if r_count == 0:
            return {
                "status": "no_releases_recorded",
                "releases_evaluated_count": 0,
                "valid_surge_releases_count": 0,
                "min_aggregate_releases_required": min_aggregate_releases,
                "insufficient_baseline_depth_count": 0,
                "dormant_baseline_count": 0,
                "mean_surge_ratio": None,
                "median_surge_ratio": None,
                "per_release_surges": [],
                "limitations": "No dated releases recorded.",
            }

        if not commit_dates:
            return {
                "status": "no_commits_observed",
                "releases_evaluated_count": r_count,
                "valid_surge_releases_count": 0,
                "min_aggregate_releases_required": min_aggregate_releases,
                "insufficient_baseline_depth_count": r_count,
                "dormant_baseline_count": 0,
                "mean_surge_ratio": None,
                "median_surge_ratio": None,
                "per_release_surges": [
                    {
                        "tag_name": rel.tag_name,
                        "published_at": pub_dt.isoformat(),
                        "status": "insufficient_baseline_depth",
                        "pre_release_commits_14d": 0,
                        "baseline_commits_60d": None,
                        "r_pre_daily": 0.0,
                        "r_base_daily": None,
                        "surge_multiplier": None,
                    }
                    for rel, pub_dt in valid_releases[-50:]
                ],
                "limitations": "No commits observed in snapshot to compute 74-day pre-release baseline.",
            }

        earliest_commit = commit_dates[0]
        per_release: List[Dict[str, Any]] = []
        valid_multipliers: List[float] = []
        insufficient_depth_count = 0
        dormant_baseline_count = 0

        for rel, pub_dt in valid_releases:
            pre_start = pub_dt - timedelta(days=14)
            base_start = pub_dt - timedelta(days=74)

            # Enforce >= 74d commit history prior to release T_k
            if earliest_commit > base_start:
                insufficient_depth_count += 1
                per_release.append({
                    "tag_name": rel.tag_name,
                    "published_at": pub_dt.isoformat(),
                    "status": "insufficient_baseline_depth",
                    "pre_release_commits_14d": None,
                    "baseline_commits_60d": None,
                    "r_pre_daily": None,
                    "r_base_daily": None,
                    "surge_multiplier": None,
                })
                continue

            # [T_k - 74d, T_k - 14d) for baseline vs [T_k - 14d, T_k] for pre-release window
            idx_base_start = bisect_left(commit_dates, base_start)
            idx_pre_start = bisect_left(commit_dates, pre_start)
            idx_pub_end = bisect_right(commit_dates, pub_dt)

            base_commits = idx_pre_start - idx_base_start
            pre_commits = idx_pub_end - idx_pre_start
            r_pre = pre_commits / 14.0
            r_base = base_commits / 60.0

            if r_base == 0.0:
                dormant_baseline_count += 1
                per_release.append({
                    "tag_name": rel.tag_name,
                    "published_at": pub_dt.isoformat(),
                    "status": "dormant_baseline_activity",
                    "pre_release_commits_14d": pre_commits,
                    "baseline_commits_60d": 0,
                    "r_pre_daily": round(r_pre, 4),
                    "r_base_daily": 0.0,
                    "surge_multiplier": None,
                })
                continue

            sm = r_pre / r_base
            valid_multipliers.append(sm)
            per_release.append({
                "tag_name": rel.tag_name,
                "published_at": pub_dt.isoformat(),
                "status": "computed",
                "pre_release_commits_14d": pre_commits,
                "baseline_commits_60d": base_commits,
                "r_pre_daily": round(r_pre, 4),
                "r_base_daily": round(r_base, 4),
                "surge_multiplier": round(sm, 4),
            })

        n_valid = len(valid_multipliers)
        if n_valid >= min_aggregate_releases:
            status = "sufficient"
            mean_sr = round(statistics.mean(valid_multipliers), 4)
            median_sr = round(statistics.median(valid_multipliers), 4)
            limitation = None
        else:
            status = "insufficient" if (insufficient_depth_count + dormant_baseline_count) < r_count or n_valid == 1 else "unavailable"
            mean_sr = None
            median_sr = None
            limitation = (
                f"Only {n_valid} release(s) satisfied the >= 74d commit baseline depth and non-zero baseline checks "
                f"(minimum {min_aggregate_releases} required for aggregate mean surge)."
            )

        if len(per_release) > 50:
            per_release = per_release[-50:]

        return {
            "status": status,
            "releases_evaluated_count": r_count,
            "valid_surge_releases_count": n_valid,
            "min_aggregate_releases_required": min_aggregate_releases,
            "insufficient_baseline_depth_count": insufficient_depth_count,
            "dormant_baseline_count": dormant_baseline_count,
            "mean_surge_ratio": mean_sr,
            "median_surge_ratio": median_sr,
            "per_release_surges": per_release,
            "limitations": limitation,
        }

    @staticmethod
    def _evaluate_semver(valid_releases: List[Tuple[Any, datetime]]) -> Dict[str, Any]:
        """Measure Semantic Versioning 2.0.0 tag compliance and Major/Minor/Patch/Prerelease counts."""
        r_count = len(valid_releases)
        if r_count == 0:
            return {
                "status": "no_releases_recorded",
                "total_releases": 0,
                "semver_compliant_count": 0,
                "non_semver_count": 0,
                "semver_adoption_rate": None,
                "major_releases_count": 0,
                "minor_releases_count": 0,
                "patch_releases_count": 0,
                "prerelease_count": 0,
                "limitations": "No dated releases recorded.",
            }

        compliant = 0
        major_c = 0
        minor_c = 0
        patch_c = 0
        prerelease_c = 0

        for rel, _ in valid_releases:
            tag = (rel.tag_name or "").strip()
            m = SEMVER_PATTERN.match(tag)
            if m is None:
                if rel.is_prerelease:
                    prerelease_c += 1
                continue

            compliant += 1
            minor_num = int(m.group(2))
            patch_num = int(m.group(3))
            pre_suffix = m.group(4)

            if rel.is_prerelease or pre_suffix is not None:
                prerelease_c += 1
            elif minor_num == 0 and patch_num == 0:
                major_c += 1
            elif patch_num == 0:
                minor_c += 1
            else:
                patch_c += 1

        return {
            "status": "sufficient" if r_count >= 5 else "limited",
            "total_releases": r_count,
            "semver_compliant_count": compliant,
            "non_semver_count": r_count - compliant,
            "semver_adoption_rate": round(compliant / float(r_count), 4),
            "major_releases_count": major_c,
            "minor_releases_count": minor_c,
            "patch_releases_count": patch_c,
            "prerelease_count": prerelease_c,
            "limitations": (
                f"Small release count ({r_count} < 5); SemVer rate reflects observed tags only."
                if r_count < 5
                else None
            ),
        }

    def _evaluate_coverage_and_confidence(
        self,
        valid_releases: List[Tuple[Any, datetime]],
        cadence: Dict[str, Any],
        surges: Dict[str, Any],
        semver: Dict[str, Any],
        collection_scope: Dict[str, Any],
    ) -> Tuple[MetricCoverageItem, float, List[Dict[str, Any]]]:
        """Determine release_evolution coverage and confidence penalties."""
        r_count = len(valid_releases)
        min_sample = 5
        penalties: List[Dict[str, Any]] = []
        penalty_total = 0.0

        if r_count == 0:
            cov_status = "empty"
            p = 0.5
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-EMPTY-RELEASES",
                "penalty": p,
                "reason": "Repository snapshot has zero dated releases.",
            })
        elif r_count < 2:
            cov_status = "insufficient"
            p = 0.4
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-SINGLE-RELEASE",
                "penalty": p,
                "reason": "Only 1 release recorded; cadence intervals and CV require multiple releases.",
            })
        elif r_count < min_sample:
            cov_status = "degraded"
            p = 0.2
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-SMALL-RELEASE-SAMPLE",
                "penalty": p,
                "reason": f"Observed dated releases ({r_count}) is below high-confidence threshold ({min_sample}).",
            })
        else:
            cov_status = "sufficient"

        if r_count > 0 and collection_scope["is_bounded_or_partial"]:
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-BOUNDED-RELEASE-SCOPE",
                "penalty": p,
                "reason": f"Collection mode '{collection_scope['mode']}' may capture only recent releases.",
            })

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=cov_status,
            sample_size=r_count,
            min_sample_required=min_sample,
            details={
                "total_releases": r_count,
                "valid_surge_releases_count": surges["valid_surge_releases_count"],
                "semver_compliant_count": semver["semver_compliant_count"],
                "collection_mode": collection_scope["mode"],
                "subdomains": {
                    "cadence": cadence["status"],
                    "prerelease_surges": surges["status"],
                    "semver_adoption": semver["status"],
                },
            },
        )

        return coverage, round(min(1.0, penalty_total), 3), penalties
