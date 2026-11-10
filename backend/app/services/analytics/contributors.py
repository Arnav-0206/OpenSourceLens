"""Contributor Intelligence & Concentration Engine for OpenSourceLens (Module 2 / Phase C.3).

Implements ContributorIntelligenceCalculator adhering to BaseMetricCalculator:
- Gini inequality coefficient (with uniform, n=1 degenerate, and extreme-skew handling).
- Standardized 11-point decile Lorenz curve coordinates.
- Shannon diversity entropy (bits) and normalized entropy.
- Bus factor concentration (BF50, BF80, top-1 and top-5 contribution shares).
- Contributor identity tracking (strict bot separation, unlinked & unattributed tracking without speculative merging).
- Multi-role participation & reference-time-anchored tenure (commits, PRs authored, PRs reviewed, issues opened).
- Zero-commit directory contributor segregation vs. active commit authors.
- Monthly retention cohort algorithm (commits & PRs only) guarded against bounded/partial ingestion survivor bias.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    CollectionRun,
    Commit,
    Contributor,
    Issue,
    PullRequest,
    PullRequestReview,
)
from backend.app.schemas.analytics_v2 import MetricCoverageItem, RuleInterpretation
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.development_activity import _ensure_utc


class ContributorIntelligenceCalculator(BaseMetricCalculator):
    """Calculates contributor inequality, diversity, bus factor, participation roles, and retention cohorts."""

    @property
    def domain_name(self) -> str:
        return "contributor_ecosystem"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
    ) -> CalculationResult:
        """Compute contributor ecosystem intelligence anchored to analysis_reference_time."""
        ref_time = _ensure_utc(analysis_reference_time)

        # 1. Query underlying snapshot entities
        contributors_rows = db.scalars(
            select(Contributor).where(Contributor.repository_id == repository_id)
        ).all()

        commits_rows = db.execute(
            select(
                Commit.sha,
                Commit.author_id,
                Commit.author_login,
                Commit.author_email,
                Commit.author_name,
                Commit.author_date,
                Commit.committer_date,
            ).where(Commit.repository_id == repository_id)
        ).fetchall()

        prs_rows = db.execute(
            select(
                PullRequest.pr_number,
                PullRequest.author,
                PullRequest.author_id,
                PullRequest.created_at,
            ).where(PullRequest.repository_id == repository_id)
        ).fetchall()

        reviews_rows = db.execute(
            select(
                PullRequestReview.review_id,
                PullRequestReview.author,
                PullRequestReview.author_id,
                PullRequestReview.submitted_at,
            )
            .join(PullRequest, PullRequestReview.pull_request_id == PullRequest.id)
            .where(PullRequest.repository_id == repository_id)
        ).fetchall()

        issues_rows = db.execute(
            select(
                Issue.issue_number,
                Issue.author,
                Issue.is_bot_author,
                Issue.created_at,
            ).where(Issue.repository_id == repository_id)
        ).fetchall()

        latest_collection_run = db.scalar(
            select(CollectionRun)
            .where(CollectionRun.repository_id == repository_id)
            .order_by(CollectionRun.started_at.desc(), CollectionRun.id.desc())
            .limit(1)
        )

        # 2. Build non-speculative directory identity lookups
        by_github_id: Dict[int, Contributor] = {}
        login_groups: Dict[str, List[Contributor]] = defaultdict(list)
        email_groups: Dict[str, List[Contributor]] = defaultdict(list)
        for c in contributors_rows:
            if c.github_id is not None:
                by_github_id[c.github_id] = c
            if c.login:
                login_groups[c.login].append(c)
            if c.git_email:
                email_groups[c.git_email].append(c)
        by_login: Dict[str, Contributor] = {k: v[0] for k, v in login_groups.items() if len(v) == 1}
        by_email: Dict[str, Contributor] = {k: v[0] for k, v in email_groups.items() if len(v) == 1}

        # 3. Resolve & aggregate all underlying timestamp-filtered events (<= ref_time)
        participation = self._aggregate_participation(
            contributors_rows=contributors_rows,
            by_github_id=by_github_id,
            by_login=by_login,
            by_email=by_email,
            commits_rows=commits_rows,
            prs_rows=prs_rows,
            reviews_rows=reviews_rows,
            issues_rows=issues_rows,
            ref_time=ref_time,
        )

        human_commit_counts: List[int] = participation["human_commit_counts_list"]
        n_active_commit_authors = len(human_commit_counts)

        # Scope / bounded collection check
        collection_scope = self._determine_collection_scope(latest_collection_run)

        # 4. Concentration metrics (Gini, Lorenz, Shannon Entropy, BF50, BF80, Top-k shares)
        concentration = self._calculate_concentration(
            counts=human_commit_counts,
            is_bounded_sample=collection_scope["is_bounded_or_partial"],
            collection_mode=collection_scope["mode"],
        )

        # 5. Monthly Retention Cohorts (strictly guarded on bounded / incomplete modes)
        retention = self._evaluate_retention(
            human_code_events=participation["human_code_events_by_actor"],
            ref_time=ref_time,
            collection_scope=collection_scope,
        )

        # 6. Coverage and Confidence Evaluation
        coverage, confidence_penalty, penalties = self._evaluate_coverage_and_confidence(
            participation=participation,
            concentration=concentration,
            retention=retention,
            collection_scope=collection_scope,
        )

        # 7. Deterministic Rules
        rules = self._build_rules(
            participation=participation,
            concentration=concentration,
            retention=retention,
            collection_scope=collection_scope,
        )

        metrics: Dict[str, Any] = {
            "scope": collection_scope,
            "concentration": concentration,
            "participation": participation["summary"],
            "identity_hygiene": participation["identity_hygiene"],
            "tenure": participation["tenure"],
            "top_contributors": participation["top_contributors"],
            "retention": retention,
        }

        return CalculationResult(
            domain=self.domain_name,
            metrics=metrics,
            coverage=coverage,
            confidence_penalty=confidence_penalty,
            rules=rules,
            details={"penalties": penalties, "snapshot_revision": snapshot_revision},
        )

    def _resolve_actor(
        self,
        author_id: Optional[int],
        author_login: Optional[str],
        author_email: Optional[str],
        is_explicit_bot: bool,
        by_github_id: Dict[int, Contributor],
        by_login: Dict[str, Contributor],
        by_email: Dict[str, Contributor],
    ) -> Dict[str, Any]:
        """Resolve an author reference without speculative display-name merging.

        Anchors strictly on immutable github_id, verified Contributor login/email,
        or unlinked email/login fallback keys. Returns is_unattributed=True when no
        verifiable identifier exists.
        """
        contrib: Optional[Contributor] = None
        if author_id is not None and author_id in by_github_id:
            contrib = by_github_id[author_id]
        elif author_login and author_login in by_login:
            candidate = by_login[author_login]
            # Never merge across conflicting immutable github_ids
            if author_id is None or candidate.github_id is None or candidate.github_id == author_id:
                contrib = candidate
        elif author_id is None and author_email and author_email in by_email:
            contrib = by_email[author_email]

        if contrib is not None:
            return {
                "actor_key": f"contrib:{contrib.id}",
                "contributor_id": contrib.id,
                "label": contrib.login or contrib.git_email or f"contributor:{contrib.id}",
                "github_id": contrib.github_id,
                "is_bot": bool(contrib.is_bot),
                "is_unlinked": bool(contrib.is_unlinked),
                "is_unattributed": False,
                "identity_confidence": float(contrib.identity_confidence),
            }

        if author_id is not None:
            inferred_bot = is_explicit_bot or (bool(author_login) and author_login.endswith("[bot]"))
            return {
                "actor_key": f"gh_id:{author_id}",
                "contributor_id": None,
                "label": author_login or f"github_id:{author_id}",
                "github_id": author_id,
                "is_bot": inferred_bot,
                "is_unlinked": False,
                "is_unattributed": False,
                "identity_confidence": 1.0,
            }

        if author_login:
            inferred_bot = is_explicit_bot or author_login.endswith("[bot]")
            return {
                "actor_key": f"gh_login:{author_login}",
                "contributor_id": None,
                "label": author_login,
                "github_id": None,
                "is_bot": inferred_bot,
                "is_unlinked": False,
                "is_unattributed": False,
                "identity_confidence": 0.9,
            }

        if author_email:
            return {
                "actor_key": f"unlinked_email:{author_email}",
                "contributor_id": None,
                "label": author_email,
                "github_id": None,
                "is_bot": is_explicit_bot,
                "is_unlinked": True,
                "is_unattributed": False,
                "identity_confidence": 0.7,
            }

        # Display-name-only or fully missing author -> unattributed (never speculatively merged)
        return {
            "actor_key": None,
            "contributor_id": None,
            "label": None,
            "github_id": None,
            "is_bot": is_explicit_bot,
            "is_unlinked": True,
            "is_unattributed": True,
            "identity_confidence": 0.0,
        }

    def _aggregate_participation(
        self,
        contributors_rows: List[Contributor],
        by_github_id: Dict[int, Contributor],
        by_login: Dict[str, Contributor],
        by_email: Dict[str, Contributor],
        commits_rows: List[Any],
        prs_rows: List[Any],
        reviews_rows: List[Any],
        issues_rows: List[Any],
        ref_time: datetime,
    ) -> Dict[str, Any]:
        """Aggregate timestamp-filtered underlying commits, PRs, reviews, and issues per contributor."""
        actors: Dict[str, Dict[str, Any]] = {}
        human_code_events_by_actor: Dict[str, List[datetime]] = defaultdict(list)

        # Build unambiguous (login -> github_id) map from verified directory/event records
        login_ids_seen: Dict[str, Set[int]] = defaultdict(set)
        for c in contributors_rows:
            if c.login and c.github_id is not None:
                login_ids_seen[c.login].add(c.github_id)
        for row in commits_rows:
            if row.author_login and row.author_id is not None:
                login_ids_seen[row.author_login].add(row.author_id)
        for row in prs_rows:
            if row.author and row.author_id is not None:
                login_ids_seen[row.author].add(row.author_id)
        for row in reviews_rows:
            if row.author and row.author_id is not None:
                login_ids_seen[row.author].add(row.author_id)
        unique_login_gh_id: Dict[str, int] = {
            login: next(iter(ids)) for login, ids in login_ids_seen.items() if len(ids) == 1
        }

        def _ensure_actor(resolved: Dict[str, Any]) -> Dict[str, Any]:
            key = resolved["actor_key"]
            if key not in actors:
                actors[key] = {
                    "actor_key": key,
                    "contributor_id": resolved["contributor_id"],
                    "label": resolved["label"],
                    "github_id": resolved["github_id"],
                    "is_bot": resolved["is_bot"],
                    "is_unlinked": resolved["is_unlinked"],
                    "identity_confidence": resolved["identity_confidence"],
                    "commits": 0,
                    "prs_authored": 0,
                    "prs_reviewed": 0,
                    "issues_opened": 0,
                    "first_event_date": None,
                    "last_event_date": None,
                }
            return actors[key]

        def _touch_dates(rec: Dict[str, Any], dt: datetime) -> None:
            if rec["first_event_date"] is None or dt < rec["first_event_date"]:
                rec["first_event_date"] = dt
            if rec["last_event_date"] is None or dt > rec["last_event_date"]:
                rec["last_event_date"] = dt

        # Counters for bots & unattributed records
        seen_shas: Set[str] = set()
        eligible_commits_total = 0
        human_commits_total = 0
        bot_commits_total = 0
        unattributed_commits_total = 0

        eligible_prs_total = 0
        human_prs_total = 0
        bot_prs_total = 0
        unattributed_prs_total = 0

        eligible_reviews_total = 0
        human_reviews_total = 0
        bot_reviews_total = 0
        unattributed_reviews_total = 0

        eligible_issues_total = 0
        human_issues_total = 0
        bot_issues_total = 0
        unattributed_issues_total = 0

        # Commits stream (strictly <= ref_time)
        for row in commits_rows:
            if row.sha in seen_shas:
                continue
            c_dt = _ensure_utc(row.author_date or row.committer_date)
            if c_dt is None or c_dt > ref_time:
                continue
            seen_shas.add(row.sha)
            eligible_commits_total += 1

            resolved = self._resolve_actor(
                author_id=(
                    row.author_id
                    if row.author_id is not None
                    else (unique_login_gh_id.get(row.author_login) if row.author_login else None)
                ),
                author_login=row.author_login,
                author_email=row.author_email,
                is_explicit_bot=False,
                by_github_id=by_github_id,
                by_login=by_login,
                by_email=by_email,
            )
            if resolved["is_unattributed"]:
                unattributed_commits_total += 1
                continue

            rec = _ensure_actor(resolved)
            rec["commits"] += 1
            _touch_dates(rec, c_dt)
            if rec["is_bot"]:
                bot_commits_total += 1
            else:
                human_commits_total += 1
                human_code_events_by_actor[rec["actor_key"]].append(c_dt)

        # Pull requests authored stream (strictly <= ref_time)
        seen_prs: Set[int] = set()
        for row in prs_rows:
            if row.pr_number in seen_prs:
                continue
            pr_dt = _ensure_utc(row.created_at)
            if pr_dt is None or pr_dt > ref_time:
                continue
            seen_prs.add(row.pr_number)
            eligible_prs_total += 1

            resolved = self._resolve_actor(
                author_id=(
                    row.author_id
                    if row.author_id is not None
                    else (unique_login_gh_id.get(row.author) if row.author else None)
                ),
                author_login=row.author,
                author_email=None,
                is_explicit_bot=False,
                by_github_id=by_github_id,
                by_login=by_login,
                by_email=by_email,
            )
            if resolved["is_unattributed"]:
                unattributed_prs_total += 1
                continue

            rec = _ensure_actor(resolved)
            rec["prs_authored"] += 1
            _touch_dates(rec, pr_dt)
            if rec["is_bot"]:
                bot_prs_total += 1
            else:
                human_prs_total += 1
                human_code_events_by_actor[rec["actor_key"]].append(pr_dt)

        # Pull request reviews stream (strictly <= ref_time; excluded from retention)
        seen_reviews: Set[int] = set()
        for row in reviews_rows:
            if row.review_id in seen_reviews:
                continue
            rev_dt = _ensure_utc(row.submitted_at)
            if rev_dt is None or rev_dt > ref_time:
                continue
            seen_reviews.add(row.review_id)
            eligible_reviews_total += 1

            resolved = self._resolve_actor(
                author_id=(
                    row.author_id
                    if row.author_id is not None
                    else (unique_login_gh_id.get(row.author) if row.author else None)
                ),
                author_login=row.author,
                author_email=None,
                is_explicit_bot=False,
                by_github_id=by_github_id,
                by_login=by_login,
                by_email=by_email,
            )
            if resolved["is_unattributed"]:
                unattributed_reviews_total += 1
                continue

            rec = _ensure_actor(resolved)
            rec["prs_reviewed"] += 1
            _touch_dates(rec, rev_dt)
            if rec["is_bot"]:
                bot_reviews_total += 1
            else:
                human_reviews_total += 1

        # Issues opened stream (strictly <= ref_time; excluded from retention)
        seen_issues: Set[int] = set()
        for row in issues_rows:
            if row.issue_number in seen_issues:
                continue
            iss_dt = _ensure_utc(row.created_at)
            if iss_dt is None or iss_dt > ref_time:
                continue
            seen_issues.add(row.issue_number)
            eligible_issues_total += 1

            resolved = self._resolve_actor(
                author_id=(unique_login_gh_id.get(row.author) if row.author else None),
                author_login=row.author,
                author_email=None,
                is_explicit_bot=bool(row.is_bot_author),
                by_github_id=by_github_id,
                by_login=by_login,
                by_email=by_email,
            )
            if resolved["is_unattributed"]:
                unattributed_issues_total += 1
                continue

            rec = _ensure_actor(resolved)
            rec["issues_opened"] += 1
            _touch_dates(rec, iss_dt)
            if rec["is_bot"]:
                bot_issues_total += 1
            else:
                human_issues_total += 1

        # Segregate directory contributors vs observed active actors
        human_actors = [a for a in actors.values() if not a["is_bot"]]
        bot_actors = [a for a in actors.values() if a["is_bot"]]
        active_human_commit_actors = [a for a in human_actors if a["commits"] > 0]
        human_commit_counts = [a["commits"] for a in active_human_commit_actors]

        directory_humans = [c for c in contributors_rows if not c.is_bot]
        directory_bots = [c for c in contributors_rows if c.is_bot]
        active_commit_contrib_ids = {
            a["contributor_id"] for a in active_human_commit_actors if a["contributor_id"] is not None
        }
        zero_commit_directory_humans = sum(
            1 for c in directory_humans if c.id not in active_commit_contrib_ids
        )

        # Reference-time-anchored tenure & newcomer/established/core stats across active human contributors
        tenures_days: List[float] = []
        newcomers_count = 0
        established_count = 0
        core_count = 0
        recently_active_30d = 0
        recently_active_90d = 0

        for a in human_actors:
            f_dt = a["first_event_date"]
            l_dt = a["last_event_date"]
            if f_dt is not None:
                t_days = max(0.0, (ref_time - f_dt).total_seconds() / 86400.0)
                tenures_days.append(t_days)
                if t_days <= 30.0:
                    newcomers_count += 1
                else:
                    established_count += 1
            if l_dt is not None:
                rec_days = max(0.0, (ref_time - l_dt).total_seconds() / 86400.0)
                if rec_days <= 30.0:
                    recently_active_30d += 1
                if rec_days <= 90.0:
                    recently_active_90d += 1
            if a["commits"] >= 10 or a["prs_authored"] >= 5:
                core_count += 1

        tenure_summary = {
            "active_human_contributors_with_dates": len(tenures_days),
            "newcomers_30d_count": newcomers_count,
            "established_gt_30d_count": established_count,
            "core_contributors_count": core_count,
            "recently_active_30d_count": recently_active_30d,
            "recently_active_90d_count": recently_active_90d,
            "mean_tenure_days": round(statistics.mean(tenures_days), 2) if tenures_days else None,
            "median_tenure_days": round(statistics.median(tenures_days), 2) if tenures_days else None,
        }

        unlinked_human_actors = sum(1 for a in human_actors if a["is_unlinked"])
        unattrib_commit_share = (
            round(unattributed_commits_total / float(eligible_commits_total), 4)
            if eligible_commits_total > 0
            else 0.0
        )
        bot_commit_share = (
            round(bot_commits_total / float(eligible_commits_total), 4)
            if eligible_commits_total > 0
            else 0.0
        )

        # Leaderboard sorted by commits then total multi-role activity (human contributors only)
        attributed_human_commits = sum(human_commit_counts)
        top_contributors = sorted(
            human_actors,
            key=lambda a: (
                a["commits"],
                a["prs_authored"] + a["prs_reviewed"] + a["issues_opened"],
                str(a["label"]),
            ),
            reverse=True,
        )[:10]

        top_contributors_formatted = [
            {
                "contributor": a["label"],
                "commits": a["commits"],
                "commit_share": (
                    round(a["commits"] / float(attributed_human_commits), 4)
                    if attributed_human_commits > 0
                    else 0.0
                ),
                "prs_authored": a["prs_authored"],
                "prs_reviewed": a["prs_reviewed"],
                "issues_opened": a["issues_opened"],
                "is_unlinked": a["is_unlinked"],
                "is_core": (a["commits"] >= 10 or a["prs_authored"] >= 5),
                "tenure_days": (
                    round(max(0.0, (ref_time - a["first_event_date"]).total_seconds() / 86400.0), 2)
                    if a["first_event_date"] is not None
                    else None
                ),
            }
            for a in top_contributors
        ]

        summary = {
            "directory_contributors_count": len(contributors_rows),
            "directory_human_contributors_count": len(directory_humans),
            "directory_bot_contributors_count": len(directory_bots),
            "zero_commit_directory_human_contributors_count": zero_commit_directory_humans,
            "active_human_commit_authors_count": len(active_human_commit_actors),
            "active_human_multi_role_contributors_count": len(human_actors),
            "active_bot_contributors_count": len(bot_actors),
            "activity_totals": {
                "commits": {
                    "eligible_total": eligible_commits_total,
                    "human_attributed": human_commits_total,
                    "bot_excluded": bot_commits_total,
                    "unattributed": unattributed_commits_total,
                },
                "prs_authored": {
                    "eligible_total": eligible_prs_total,
                    "human_attributed": human_prs_total,
                    "bot_excluded": bot_prs_total,
                    "unattributed": unattributed_prs_total,
                },
                "prs_reviewed": {
                    "eligible_total": eligible_reviews_total,
                    "human_attributed": human_reviews_total,
                    "bot_excluded": bot_reviews_total,
                    "unattributed": unattributed_reviews_total,
                },
                "issues_opened": {
                    "eligible_total": eligible_issues_total,
                    "human_attributed": human_issues_total,
                    "bot_excluded": bot_issues_total,
                    "unattributed": unattributed_issues_total,
                },
            },
        }

        identity_hygiene = {
            "unlinked_active_human_contributors_count": unlinked_human_actors,
            "unattributed_commits_count": unattributed_commits_total,
            "unattributed_commits_share": unattrib_commit_share,
            "bot_commits_excluded_count": bot_commits_total,
            "bot_commits_share": bot_commit_share,
            "speculative_name_merging": False,
        }

        return {
            "summary": summary,
            "identity_hygiene": identity_hygiene,
            "tenure": tenure_summary,
            "top_contributors": top_contributors_formatted,
            "human_commit_counts_list": human_commit_counts,
            "human_code_events_by_actor": dict(human_code_events_by_actor),
            "eligible_commits_total": eligible_commits_total,
        }

    def _determine_collection_scope(
        self,
        latest_run: Optional[CollectionRun],
    ) -> Dict[str, Any]:
        """Inspect latest CollectionRun metadata to detect bounded or partial ingestion snapshots."""
        if latest_run is None:
            return {
                "mode": "unknown_unverified",
                "is_partial": True,
                "is_bounded_or_partial": True,
                "verified_full_history": False,
                "reason": "No CollectionRun record found; historical completeness cannot be verified.",
            }

        mode = (latest_run.mode or "unknown").lower().strip()
        is_partial = bool(latest_run.is_partial)
        cov = latest_run.coverage_summary or {}
        commits_cov = cov.get("commits") if isinstance(cov, dict) else None
        prs_cov = cov.get("pull_requests") if isinstance(cov, dict) else None

        commits_truncated = isinstance(commits_cov, dict) and (
            commits_cov.get("is_partial") is True or commits_cov.get("status") in ("truncated", "partial", "rate_limited")
        )
        prs_truncated = isinstance(prs_cov, dict) and (
            prs_cov.get("is_partial") is True or prs_cov.get("status") in ("truncated", "partial", "rate_limited")
        )

        bounded_modes = {"fast_snapshot", "standard_diagnostic"}
        is_full_mode = mode == "full_history" and latest_run.status == "completed"
        is_bounded_or_partial = (
            (mode in bounded_modes)
            or (not is_full_mode)
            or is_partial
            or commits_truncated
            or prs_truncated
        )

        if mode in bounded_modes:
            reason = f"Snapshot ingested under bounded mode '{mode}'; reflects recent observed activity only."
        elif not is_full_mode:
            reason = f"Latest collection mode '{mode}' (status='{latest_run.status}') is not a completed full-history run."
        elif is_partial or commits_truncated or prs_truncated:
            reason = "Collection run reported partial or truncated commit/PR coverage."
        else:
            reason = "Verified completed full_history ingestion run."

        return {
            "mode": mode,
            "collection_status": latest_run.status,
            "is_partial": is_partial,
            "is_bounded_or_partial": is_bounded_or_partial,
            "verified_full_history": not is_bounded_or_partial,
            "reason": reason,
        }

    def _calculate_concentration(
        self,
        counts: List[int],
        is_bounded_sample: bool,
        collection_mode: str,
    ) -> Dict[str, Any]:
        """Compute Gini, 11-point Lorenz deciles, Shannon entropy, BF50, BF80, and top shares."""
        n = len(counts)
        min_sample = 2
        confident_sample = 5
        sample_caveat = (
            f"Metrics reflect the observed contributor sample under '{collection_mode}' ingestion, not necessarily all-time repository history."
            if is_bounded_sample
            else "Computed across verified full-history contributor commit counts."
        )

        if n == 0 or sum(counts) == 0:
            return {
                "status": "empty",
                "active_human_commit_authors_count": n,
                "total_attributed_human_commits": sum(counts),
                "min_sample_required": min_sample,
                "confident_sample_required": confident_sample,
                "degenerate_single_contributor": False,
                "gini_coefficient": None,
                "lorenz_curve": None,
                "shannon_entropy_bits": None,
                "normalized_entropy": None,
                "bf50": None,
                "bf80": None,
                "top1_share": None,
                "top5_share": None,
                "is_bounded_sample": is_bounded_sample,
                "limitations": f"No attributed human commits available for concentration metrics. {sample_caveat}",
            }

        total_x = sum(counts)
        if n == 1:
            return {
                "status": "insufficient",
                "active_human_commit_authors_count": 1,
                "total_attributed_human_commits": total_x,
                "min_sample_required": min_sample,
                "confident_sample_required": confident_sample,
                "degenerate_single_contributor": True,
                "gini_coefficient": 0.0,
                "lorenz_curve": None,
                "shannon_entropy_bits": 0.0,
                "normalized_entropy": 0.0,
                "bf50": 1,
                "bf80": 1,
                "top1_share": 1.0,
                "top5_share": 1.0,
                "is_bounded_sample": is_bounded_sample,
                "limitations": f"Single active human commit author (n=1); Gini inequality is mathematically degenerate (0.0) and Lorenz curve is deferred. {sample_caveat}",
            }

        # Sort ascending for Gini & Lorenz
        asc = sorted(counts)
        weighted_sum = sum((idx + 1) * val for idx, val in enumerate(asc))
        gini_raw = (2.0 * weighted_sum) / (float(n) * float(total_x)) - ((n + 1.0) / float(n))
        gini = round(max(0.0, min(1.0, gini_raw)), 4)

        # Standardized 11-point Lorenz deciles u in {0.0, 0.1, ..., 1.0}
        cum_shares: List[float] = [0.0]
        running = 0
        for val in asc:
            running += val
            cum_shares.append(running / float(total_x))

        lorenz_curve: List[Dict[str, float]] = []
        for decile in range(11):
            u = decile / 10.0
            pos = u * float(n)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi or hi >= len(cum_shares):
                l_val = cum_shares[min(lo, len(cum_shares) - 1)]
            else:
                frac = pos - lo
                l_val = cum_shares[lo] + frac * (cum_shares[hi] - cum_shares[lo])
            lorenz_curve.append({
                "population_share": round(u, 1),
                "contribution_share": round(l_val, 4),
            })

        # Shannon entropy & normalized entropy
        shannon_h = 0.0
        for val in asc:
            p_i = val / float(total_x)
            if p_i > 0.0:
                shannon_h -= p_i * math.log2(p_i)
        norm_h = shannon_h / math.log2(float(n))

        # Bus factor (BF50, BF80) & Top-k shares (sorted descending)
        desc = list(reversed(asc))
        bf50 = n
        bf80 = n
        cum_desc = 0
        found_50 = False
        for k, val in enumerate(desc, start=1):
            cum_desc += val
            share = cum_desc / float(total_x)
            if not found_50 and share >= 0.50:
                bf50 = k
                found_50 = True
            if share >= 0.80:
                bf80 = k
                break

        top1_share = round(desc[0] / float(total_x), 4)
        top5_share = round(sum(desc[: min(5, n)]) / float(total_x), 4)

        status = "limited" if n < confident_sample else "sufficient"
        limitations = (
            f"Limited contributor pool ({n} < {confident_sample} authors); concentration estimates have high discrete variance. {sample_caveat}"
            if status == "limited"
            else sample_caveat
        )

        return {
            "status": status,
            "active_human_commit_authors_count": n,
            "total_attributed_human_commits": total_x,
            "min_sample_required": min_sample,
            "confident_sample_required": confident_sample,
            "degenerate_single_contributor": False,
            "gini_coefficient": gini,
            "lorenz_curve": lorenz_curve,
            "shannon_entropy_bits": round(shannon_h, 4),
            "normalized_entropy": round(norm_h, 4),
            "bf50": bf50,
            "bf80": bf80,
            "top1_share": top1_share,
            "top5_share": top5_share,
            "is_bounded_sample": is_bounded_sample,
            "limitations": limitations,
        }

    def _evaluate_retention(
        self,
        human_code_events: Dict[str, List[datetime]],
        ref_time: datetime,
        collection_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enforce mandatory bounded-mode guard and compute monthly retention cohorts when verified full history exists."""
        if not collection_scope["verified_full_history"]:
            return {
                "status": "unavailable_requires_full_history_ingestion",
                "retention_cohorts": None,
                "eligible_events": ["commits", "pull_requests_authored"],
                "excluded_events": ["issues_opened", "pr_reviews", "comments", "bots"],
                "reason": (
                    "Contributor retention requires verified complete historical event streams from repository inception. "
                    f"Current collection scope is insufficient ({collection_scope['reason']})."
                ),
            }

        if not human_code_events:
            return {
                "status": "empty",
                "retention_cohorts": [],
                "eligible_events": ["commits", "pull_requests_authored"],
                "excluded_events": ["issues_opened", "pr_reviews", "comments", "bots"],
                "summary": {
                    "total_cohorts": 0,
                    "total_tracked_contributors": 0,
                    "overall_m1_retention_rate": None,
                    "overall_m3_retention_rate": None,
                },
                "reason": "Full-history repository snapshot contains no eligible human commit or PR activity.",
            }

        return self._compute_monthly_retention_cohorts(human_code_events=human_code_events, ref_time=ref_time)

    def _compute_monthly_retention_cohorts(
        self,
        human_code_events: Dict[str, List[datetime]],
        ref_time: datetime,
    ) -> Dict[str, Any]:
        """Compute monthly retention matrix strictly across commits and PRs authored up to ref_time."""
        ref_ym = (ref_time.year, ref_time.month)

        def _month_diff(m_start: Tuple[int, int], m_end: Tuple[int, int]) -> int:
            return (m_end[0] - m_start[0]) * 12 + (m_end[1] - m_start[1])

        def _add_months(m_start: Tuple[int, int], offset: int) -> Tuple[int, int]:
            total_m = (m_start[0] * 12 + (m_start[1] - 1)) + offset
            return (total_m // 12, (total_m % 12) + 1)

        def _fmt_ym(ym: Tuple[int, int]) -> str:
            return f"{ym[0]:04d}-{ym[1]:02d}"

        cohorts_members: Dict[Tuple[int, int], List[Set[Tuple[int, int]]]] = defaultdict(list)
        for actor_key, timestamps in human_code_events.items():
            active_months = {(dt.year, dt.month) for dt in timestamps if dt <= ref_time}
            if not active_months:
                continue
            m0 = min(active_months)
            cohorts_members[m0].append(active_months)

        cohort_records: List[Dict[str, Any]] = []
        m1_retained_sum = 0
        m1_eligible_sum = 0
        m3_retained_sum = 0
        m3_eligible_sum = 0

        for m0 in sorted(cohorts_members.keys()):
            members = cohorts_members[m0]
            cohort_size = len(members)
            max_offset = max(0, _month_diff(m0, ref_ym))
            retention_by_offset: Dict[str, Dict[str, Any]] = {}

            for k in range(max_offset + 1):
                target_ym = _add_months(m0, k)
                retained = sum(1 for mset in members if target_ym in mset)
                rate = round(retained / float(cohort_size), 4)
                retention_by_offset[f"M+{k}"] = {
                    "month": _fmt_ym(target_ym),
                    "retained_contributors": retained,
                    "retention_rate": rate,
                }
                if k == 1:
                    m1_retained_sum += retained
                    m1_eligible_sum += cohort_size
                if k == 3:
                    m3_retained_sum += retained
                    m3_eligible_sum += cohort_size

            cohort_records.append({
                "cohort_month": _fmt_ym(m0),
                "cohort_size": cohort_size,
                "max_observable_offset_months": max_offset,
                "retention_by_month": retention_by_offset,
            })

        return {
            "status": "computed",
            "retention_cohorts": cohort_records,
            "eligible_events": ["commits", "pull_requests_authored"],
            "excluded_events": ["issues_opened", "pr_reviews", "comments", "bots"],
            "summary": {
                "total_cohorts": len(cohort_records),
                "total_tracked_contributors": sum(c["cohort_size"] for c in cohort_records),
                "overall_m1_retention_rate": (
                    round(m1_retained_sum / float(m1_eligible_sum), 4) if m1_eligible_sum > 0 else None
                ),
                "overall_m3_retention_rate": (
                    round(m3_retained_sum / float(m3_eligible_sum), 4) if m3_eligible_sum > 0 else None
                ),
            },
            "reason": "Verified complete history snapshot; monthly cohort retention calculated over commits and PRs.",
        }

    def _evaluate_coverage_and_confidence(
        self,
        participation: Dict[str, Any],
        concentration: Dict[str, Any],
        retention: Dict[str, Any],
        collection_scope: Dict[str, Any],
    ) -> Tuple[MetricCoverageItem, float, List[Dict[str, Any]]]:
        """Determine overall domain coverage status and deterministic confidence penalties."""
        summary = participation["summary"]
        hygiene = participation["identity_hygiene"]
        n_authors = concentration["active_human_commit_authors_count"]
        multi_role = summary["active_human_multi_role_contributors_count"]
        min_sample = 5

        penalties: List[Dict[str, Any]] = []
        penalty_total = 0.0

        if n_authors == 0 and multi_role == 0:
            cov_status = "empty"
            penalty_total += 0.5
            penalties.append({
                "rule": "PENALTY-EMPTY-CONTRIBUTORS",
                "penalty": 0.5,
                "reason": "Repository snapshot has zero observed human contributor activity.",
            })
        elif n_authors < 2:
            cov_status = "insufficient"
            p = 0.4
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-INSUFFICIENT-COMMIT-AUTHORS",
                "penalty": p,
                "reason": f"Observed active human commit authors ({n_authors}) is below minimum 2 required for concentration metrics.",
            })
        elif n_authors < min_sample:
            cov_status = "degraded"
            p = 0.2
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-SMALL-CONTRIBUTOR-POOL",
                "penalty": p,
                "reason": f"Active human commit authors ({n_authors}) is below high-confidence threshold ({min_sample}).",
            })
        else:
            cov_status = "sufficient"

        if n_authors > 0 and collection_scope["is_bounded_or_partial"]:
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-BOUNDED-SAMPLE-SCOPE",
                "penalty": p,
                "reason": f"Snapshot scope is bounded/partial ('{collection_scope['mode']}'); reflects observed sample rather than repository lifetime.",
            })

        if hygiene["unattributed_commits_share"] > 0.25:
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-UNATTRIBUTED-COMMITS",
                "penalty": p,
                "reason": f"{round(hygiene['unattributed_commits_share'] * 100.0, 1)}% of eligible commits lack verifiable author identity.",
            })

        part_status = "empty" if multi_role == 0 else ("limited" if multi_role < min_sample else "sufficient")
        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=cov_status,
            sample_size=n_authors,
            min_sample_required=min_sample,
            details={
                "active_human_commit_authors_count": n_authors,
                "active_human_multi_role_contributors_count": multi_role,
                "directory_contributors_count": summary["directory_contributors_count"],
                "zero_commit_directory_human_contributors_count": summary["zero_commit_directory_human_contributors_count"],
                "collection_mode": collection_scope["mode"],
                "subdomains": {
                    "concentration": concentration["status"],
                    "participation": part_status,
                    "retention_cohorts": retention["status"],
                },
            },
        )

        return coverage, round(min(1.0, penalty_total), 3), penalties

    def _build_rules(
        self,
        participation: Dict[str, Any],
        concentration: Dict[str, Any],
        retention: Dict[str, Any],
        collection_scope: Dict[str, Any],
    ) -> List[RuleInterpretation]:
        """Generate deterministic, evidence-backed diagnostics for contributor ecosystem metrics."""
        rules: List[RuleInterpretation] = []
        n = concentration["active_human_commit_authors_count"]

        if concentration["status"] == "empty":
            multi_role = participation["summary"]["active_human_multi_role_contributors_count"]
            if multi_role == 0:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-CON-EMPTY",
                        domain=self.domain_name,
                        severity="warning",
                        finding="No Contributor Activity: Repository snapshot contains no observed contributor activity.",
                        evidence={"directory_contributors_count": participation["summary"]["directory_contributors_count"]},
                        recommendation="Run repository ingestion to harvest commits, contributors, and pull requests.",
                    )
                )
            else:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-CON-EMPTY",
                        domain=self.domain_name,
                        severity="info",
                        finding=(
                            f"No Commit Authors Observed: Snapshot includes {multi_role} non-commit contributor(s) "
                            "(e.g., issues/reviews) but 0 human commit authors; concentration and bus factor are unavailable."
                        ),
                        evidence={"active_human_multi_role_contributors_count": multi_role},
                        recommendation="Ingest commit history to evaluate contributor concentration and bus factor.",
                    )
                )
            return rules

        scope_suffix = (
            f" (observed '{collection_scope['mode']}' sample)."
            if collection_scope["is_bounded_or_partial"]
            else "."
        )

        # RULE-CON-001: Bus Factor & Dominant Share Risk
        if concentration["degenerate_single_contributor"]:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-CON-001",
                    domain=self.domain_name,
                    severity="warning",
                    finding=f"Single Commit Author: All attributed human commits belong to 1 author{scope_suffix}",
                    evidence={"active_human_commit_authors_count": 1, "bf50": 1, "top1_share": 1.0},
                    recommendation="Broaden code review and commit delegation if maintenance continuity is required.",
                )
            )
        elif n >= 2:
            bf50 = concentration["bf50"]
            top1 = concentration["top1_share"]
            if bf50 == 1 and top1 is not None and top1 >= 0.60:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-CON-001",
                        domain=self.domain_name,
                        severity="warning",
                        finding=(
                            f"High Key-Contributor Concentration: Top author accounts for {round(top1 * 100.0, 1)}% "
                            f"of observed commits (BF50={bf50}, BF80={concentration['bf80']}){scope_suffix}"
                        ),
                        evidence={"bf50": bf50, "bf80": concentration["bf80"], "top1_share": top1, "authors": n},
                        recommendation="Distribute module ownership and PR review approvals across additional contributors.",
                    )
                )
            elif n >= 5 and bf50 is not None and bf50 >= 3 and (concentration["gini_coefficient"] or 0.0) <= 0.45:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-CON-001",
                        domain=self.domain_name,
                        severity="positive",
                        finding=(
                            f"Distributed Commit Participation: BF50={bf50} contributors account for half of observed commits "
                            f"(Gini={concentration['gini_coefficient']}){scope_suffix}"
                        ),
                        evidence={
                            "bf50": bf50,
                            "bf80": concentration["bf80"],
                            "gini_coefficient": concentration["gini_coefficient"],
                            "authors": n,
                        },
                        recommendation=None,
                    )
                )

        # RULE-CON-002: Inequality Regime (when sufficient >= 5 authors)
        if concentration["status"] == "sufficient" and concentration["gini_coefficient"] is not None:
            gini = concentration["gini_coefficient"]
            if gini >= 0.75:
                rules.append(
                    RuleInterpretation(
                        rule_id="RULE-CON-002",
                        domain=self.domain_name,
                        severity="warning",
                        finding=(
                            f"Elevated Contribution Skew: Gini coefficient is {gini} across {n} active authors{scope_suffix}"
                        ),
                        evidence={
                            "gini_coefficient": gini,
                            "normalized_entropy": concentration["normalized_entropy"],
                            "bf80": concentration["bf80"],
                        },
                        recommendation=None,
                    )
                )

        # RULE-CON-003: Bot Exclusion / Unattributed Identity Hygiene
        hygiene = participation["identity_hygiene"]
        if hygiene["bot_commits_share"] >= 0.20 or hygiene["unattributed_commits_share"] >= 0.20:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-CON-003",
                    domain=self.domain_name,
                    severity="info",
                    finding=(
                        f"Identity & Bot Filtering Applied: Excluded {hygiene['bot_commits_excluded_count']} automated bot commits "
                        f"({round(hygiene['bot_commits_share'] * 100.0, 1)}%) and isolated {hygiene['unattributed_commits_count']} "
                        f"unattributed commits ({round(hygiene['unattributed_commits_share'] * 100.0, 1)}%) from human concentration."
                    ),
                    evidence=hygiene,
                    recommendation=None,
                )
            )

        # RULE-CON-004: Retention Cohort Guard Transparency
        if retention["status"] == "unavailable_requires_full_history_ingestion" and n >= 2:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-CON-004",
                    domain=self.domain_name,
                    severity="info",
                    finding="Retention Cohorts Deferred: Historical cohort retention cannot be estimated from bounded ingestion snapshots.",
                    evidence={"retention_status": retention["status"], "collection_mode": collection_scope["mode"]},
                    recommendation=None,
                )
            )
        elif retention["status"] == "computed" and retention["summary"]["overall_m1_retention_rate"] is not None:
            m1 = retention["summary"]["overall_m1_retention_rate"]
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-CON-004",
                    domain=self.domain_name,
                    severity="info",
                    finding=f"Full-History Cohort Retention Computed: Aggregate Month+1 contributor retention is {round(m1 * 100.0, 1)}% across code commits and PRs.",
                    evidence=retention["summary"],
                    recommendation=None,
                )
            )

        return rules
