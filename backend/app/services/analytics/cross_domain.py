"""Cross-Domain Intelligence & Repository Synthesis Engine (Module 6 / Phase C.6).

Implements CrossDomainSynthesisCalculator adhering to BaseMetricCalculator:
- Multi-Domain Diagnostic Scorecard (no arbitrary 0-100 CHI composite score per v1.1 §6.2) paired with
  Repository context (stars_count, forks_count, primary_language, is_archived, is_fork; zero last_synced_at).
- Embeds ReleaseEvolutionCalculator ('release_evolution') cleanly inside 'repository_synthesis' so
  AnalysisPayload keeps backward-compatible top-level keys without schema migrations.
- Calendar-month aligned Cross-Domain Spearman correlation across 3 specified metric pairs:
  1. Monthly Code Churn vs Monthly Issue Creations (NULL commit diffs treated as missing, never 0).
  2. Monthly Median TTFResp vs Monthly PR Abandonment Rate.
  3. Monthly Commit Velocity vs Monthly PR Merge Volume.
- Strict statistical guardrails (v1.1-v1.3):
  - Minimum K >= 12 overlapping calendar months (insufficient_temporal_periods below 12).
  - B = 10,000 SHA-256 seeded deterministic Monte Carlo permutations with plus-one correction.
  - Benjamini-Hochberg FDR q-values at alpha = 0.05 across tested pairs.
  - |r_AR1| > 0.40 serial dependence p-value suppression (serial_dependence_detected).
  - Non-causal wording and deferred block-bootstrap CIs.
- Deterministic synthesis rules (RULE-REL-* + RULE-SYN-*) via rule_engine without modifying C.2-C.5.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import math
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    Commit,
    Contributor,
    Issue,
    IssueComment,
    PullRequest,
    PullRequestReview,
    PullRequestReviewComment,
    Repository,
)
from backend.app.schemas.analytics_v2 import (
    CURRENT_METHODOLOGY_VERSION,
    MetricCoverageItem,
)
from backend.app.services.analytics.base import BaseMetricCalculator, CalculationResult
from backend.app.services.analytics.contributors import ContributorIntelligenceCalculator
from backend.app.services.analytics.development_activity import (
    DevelopmentActivityCalculator,
    _ensure_utc,
)
from backend.app.services.analytics.issues import IssueIntelligenceCalculator
from backend.app.services.analytics.pull_requests import (
    FORMAL_REVIEW_STATES,
    PullRequestIntelligenceCalculator,
)
from backend.app.services.analytics.releases import ReleaseEvolutionCalculator
from backend.app.services.analytics.rule_engine import (
    build_cross_domain_rules,
    consolidate_rules,
)


PERMUTATION_COUNT: int = 10000
MIN_OVERLAPPING_MONTHS: int = 12
AR1_GUARDRAIL_THRESHOLD: float = 0.40
BH_FDR_ALPHA: float = 0.05


def _month_range_sequence(
    min_ym: Tuple[int, int],
    max_ym: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Generate contiguous (year, month) tuples from min_ym through max_ym inclusive."""
    out: List[Tuple[int, int]] = []
    cur_y, cur_m = min_ym
    while (cur_y, cur_m) <= max_ym:
        out.append((cur_y, cur_m))
        cur_m += 1
        if cur_m > 12:
            cur_m = 1
            cur_y += 1
    return out


def _lag1_autocorrelation(values: List[float]) -> float:
    """Compute finite Pearson lag-1 autocorrelation coefficient with zero-variance guard."""
    if len(values) < 3:
        return 0.0
    x0 = np.asarray(values[:-1], dtype=np.float64)
    x1 = np.asarray(values[1:], dtype=np.float64)
    c0 = x0 - np.mean(x0)
    c1 = x1 - np.mean(x1)
    denom = float(np.sqrt(np.dot(c0, c0) * np.dot(c1, c1)))
    if denom <= 1e-12:
        return 0.0
    val = float(np.dot(c0, c1) / denom)
    return val if math.isfinite(val) else 0.0


def _normalized_ranks(values: List[float]) -> Optional[np.ndarray]:
    """Return L2-normalized centered average ranks for exact vectorized Spearman permutation."""
    ranks = scipy_stats.rankdata(np.asarray(values, dtype=np.float64), method="average")
    centered = ranks - np.mean(ranks)
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-12:
        return None
    return centered / norm


def evaluate_paired_spearman_permutation(
    repository_id: int,
    snapshot_revision: int,
    pair_id: str,
    metric_x: str,
    label_x: str,
    metric_y: str,
    label_y: str,
    series_x: Dict[Tuple[int, int], Optional[float]],
    series_y: Dict[Tuple[int, int], Optional[float]],
    permutations: int = PERMUTATION_COUNT,
    methodology_version: str = CURRENT_METHODOLOGY_VERSION,
) -> Dict[str, Any]:
    """Evaluate monthly paired Spearman correlation with Monte Carlo permutations & AR(1) guardrail."""
    overlapping_months = sorted(
        ym
        for ym in (set(series_x.keys()) & set(series_y.keys()))
        if series_x[ym] is not None and series_y[ym] is not None
    )
    k = len(overlapping_months)
    x_vals = [float(series_x[ym]) for ym in overlapping_months]  # type: ignore[arg-type]
    y_vals = [float(series_y[ym]) for ym in overlapping_months]  # type: ignore[arg-type]

    base_meta: Dict[str, Any] = {
        "pair_id": pair_id,
        "metric_x": metric_x,
        "label_x": label_x,
        "metric_y": metric_y,
        "label_y": label_y,
        "k_months": k,
        "min_periods_required": MIN_OVERLAPPING_MONTHS,
        "overlapping_months": [f"{ym[0]:04d}-{ym[1]:02d}" for ym in overlapping_months],
        "permutations": permutations,
        "association_type": "Descriptive Monotonic Association (Non-Causal)",
        "confidence_intervals_deferred": True,
    }

    if k < MIN_OVERLAPPING_MONTHS:
        return {
            **base_meta,
            "status": "insufficient_temporal_periods",
            "spearman_r": None,
            "p_value": None,
            "q_value": None,
            "significant_fdr_05": False,
            "ar1_x": None,
            "ar1_y": None,
            "max_abs_ar1": None,
            "serial_dependence_guardrail_tripped": False,
            "permutation_seed": None,
            "note": f"insufficient_temporal_periods (observed {k} overlapping months; requires >= {MIN_OVERLAPPING_MONTHS} months).",
        }

    u_hat = _normalized_ranks(x_vals)
    v_hat = _normalized_ranks(y_vals)
    if u_hat is None or v_hat is None:
        return {
            **base_meta,
            "status": "undefined_zero_variance",
            "spearman_r": None,
            "p_value": None,
            "q_value": None,
            "significant_fdr_05": False,
            "ar1_x": None,
            "ar1_y": None,
            "max_abs_ar1": None,
            "serial_dependence_guardrail_tripped": False,
            "permutation_seed": None,
            "note": "Zero variance in one or both aligned monthly series prevents rank correlation computation.",
        }

    r_obs = float(np.dot(u_hat, v_hat))
    r_obs = max(-1.0, min(1.0, r_obs))

    ar1_x = _lag1_autocorrelation(x_vals)
    ar1_y = _lag1_autocorrelation(y_vals)
    max_abs_ar1 = max(abs(ar1_x), abs(ar1_y))

    if max_abs_ar1 > AR1_GUARDRAIL_THRESHOLD:
        return {
            **base_meta,
            "status": "serial_dependence_detected",
            "spearman_r": round(r_obs, 4),
            "p_value": None,
            "q_value": None,
            "significant_fdr_05": False,
            "ar1_x": round(ar1_x, 4),
            "ar1_y": round(ar1_y, 4),
            "max_abs_ar1": round(max_abs_ar1, 4),
            "serial_dependence_guardrail_tripped": True,
            "permutation_seed": None,
            "note": (
                "serial_dependence_guardrail_tripped: Moderate-to-high autocorrelation (|r_AR1| > 0.40) detected; "
                "permutation p-value suppressed. Report r_obs strictly as Descriptive Monotonic Association."
            ),
        }

    # Deterministic SHA-256 32-bit unsigned seed bound to repository, revision, pair, methodology, B
    seed_key = f"{repository_id}:{snapshot_revision}:{metric_x}:{metric_y}:{methodology_version}:{permutations}"
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # Exact rank-vector permutation across B shuffles: corr(rank(X), perm(rank(Y))) == perm(v_hat) @ u_hat
    v_matrix = np.broadcast_to(v_hat, (permutations, k)).copy()
    v_shuffled = rng.permuted(v_matrix, axis=1)
    perm_rs = v_shuffled @ u_hat

    exceed_count = int(np.sum(np.abs(perm_rs) >= (abs(r_obs) - 1e-12)))
    p_val = (exceed_count + 1) / float(permutations + 1)

    return {
        **base_meta,
        "status": "tested",
        "spearman_r": round(r_obs, 4),
        "p_value": round(p_val, 6),
        "q_value": None,  # Populated via Benjamini-Hochberg across tested pairs
        "significant_fdr_05": False,
        "ar1_x": round(ar1_x, 4),
        "ar1_y": round(ar1_y, 4),
        "max_abs_ar1": round(max_abs_ar1, 4),
        "serial_dependence_guardrail_tripped": False,
        "permutation_seed": seed,
        "note": (
            "Descriptive monotonic association only (empirical two-tailed Monte Carlo permutation p-value "
            "with plus-one correction); does NOT imply causal mechanism."
        ),
    }


def apply_benjamini_hochberg_fdr(pairs: List[Dict[str, Any]], alpha: float = BH_FDR_ALPHA) -> None:
    """Apply Benjamini-Hochberg FDR step-up q-value adjustment in-place over tested pairs."""
    tested = [p for p in pairs if p.get("status") == "tested" and p.get("p_value") is not None]
    m = len(tested)
    if m == 0:
        return

    ranked = sorted(tested, key=lambda item: float(item["p_value"]))
    raw_q: List[float] = []
    for idx_1, item in enumerate(ranked, start=1):
        raw_q.append((float(m) / float(idx_1)) * float(item["p_value"]))

    adjusted: List[float] = [0.0] * m
    running_min = 1.0
    for idx in range(m - 1, -1, -1):
        running_min = min(running_min, raw_q[idx], 1.0)
        adjusted[idx] = running_min

    for item, q_val in zip(ranked, adjusted):
        item["q_value"] = round(q_val, 6)
        item["significant_fdr_05"] = bool(q_val <= alpha)


class CrossDomainSynthesisCalculator(BaseMetricCalculator):
    """Synthesizes Modules 1-5 into a Diagnostic Scorecard, cross-domain correlations, and rules."""

    @property
    def domain_name(self) -> str:
        return "repository_synthesis"

    def calculate(
        self,
        repository_id: int,
        db: Session,
        analysis_reference_time: datetime,
        snapshot_revision: int,
        precomputed_domains: Optional[Dict[str, CalculationResult]] = None,
    ) -> CalculationResult:
        """Compute cross-domain synthesis, embedded release evolution, scorecard, and correlations."""
        ref_time = _ensure_utc(analysis_reference_time)

        # 1. Query Repository context (explicitly omitting last_synced_at per v1.8)
        repo_row = db.execute(
            select(
                Repository.repo_name,
                Repository.stars_count,
                Repository.forks_count,
                Repository.primary_language,
                Repository.is_archived,
                Repository.is_fork,
            ).where(Repository.id == repository_id)
        ).first()

        repo_context: Dict[str, Any] = {
            "repository_id": repository_id,
            "repo_name": repo_row.repo_name if repo_row else None,
            "stars_count": int(repo_row.stars_count) if repo_row else None,
            "forks_count": int(repo_row.forks_count) if repo_row else None,
            "primary_language": repo_row.primary_language if repo_row else None,
            "is_archived": bool(repo_row.is_archived) if repo_row else False,
            "is_fork": bool(repo_row.is_fork) if repo_row else False,
        }

        # 2. Compute or reuse existing Modules 1-5 without duplicating domain logic
        domain_map: Dict[str, CalculationResult] = dict(precomputed_domains or {})
        calculators: Dict[str, BaseMetricCalculator] = {
            "development_activity": DevelopmentActivityCalculator(),
            "contributor_ecosystem": ContributorIntelligenceCalculator(),
            "pull_request_intelligence": PullRequestIntelligenceCalculator(),
            "issue_intelligence": IssueIntelligenceCalculator(),
            "release_evolution": ReleaseEvolutionCalculator(),
        }
        for key, calc in calculators.items():
            if key not in domain_map:
                domain_map[key] = calc.calculate(
                    repository_id=repository_id,
                    db=db,
                    analysis_reference_time=ref_time,
                    snapshot_revision=snapshot_revision,
                )

        # 3. Extract aligned monthly time series for the 3 cross-domain Spearman pairs
        monthly_series = self._build_monthly_aligned_series(
            repository_id=repository_id,
            db=db,
            ref_time=ref_time,
        )

        pairs: List[Dict[str, Any]] = [
            evaluate_paired_spearman_permutation(
                repository_id=repository_id,
                snapshot_revision=snapshot_revision,
                pair_id="PAIR-01",
                metric_x="monthly_code_churn",
                label_x="Monthly Code Churn (Additions + Deletions)",
                metric_y="monthly_issue_creations",
                label_y="Monthly Issue Creations",
                series_x=monthly_series["monthly_code_churn"],
                series_y=monthly_series["monthly_issue_creations"],
            ),
            evaluate_paired_spearman_permutation(
                repository_id=repository_id,
                snapshot_revision=snapshot_revision,
                pair_id="PAIR-02",
                metric_x="monthly_median_ttf_resp_hours",
                label_x="Monthly Median PR TTFResp (Hours)",
                metric_y="monthly_pr_abandonment_rate",
                label_y="Monthly PR Abandonment Rate",
                series_x=monthly_series["monthly_median_ttf_resp_hours"],
                series_y=monthly_series["monthly_pr_abandonment_rate"],
            ),
            evaluate_paired_spearman_permutation(
                repository_id=repository_id,
                snapshot_revision=snapshot_revision,
                pair_id="PAIR-03",
                metric_x="monthly_commit_velocity",
                label_x="Monthly Commit Velocity",
                metric_y="monthly_pr_merge_volume",
                label_y="Monthly PR Merge Volume",
                series_x=monthly_series["monthly_commit_velocity"],
                series_y=monthly_series["monthly_pr_merge_volume"],
            ),
        ]
        apply_benjamini_hochberg_fdr(pairs, alpha=BH_FDR_ALPHA)

        tested_count = sum(1 for p in pairs if p["status"] == "tested")
        correlations: Dict[str, Any] = {
            "status": "tested" if tested_count > 0 else (
                "serial_dependence_suppressed"
                if any(p["status"] == "serial_dependence_detected" for p in pairs)
                else "insufficient_temporal_periods"
            ),
            "methodology": "Spearman rank correlation with SHA-256 seeded Monte Carlo permutation (B=10,000), BH FDR (alpha=0.05), and |r_AR1|>0.40 guardrail",
            "min_periods_required": MIN_OVERLAPPING_MONTHS,
            "fdr_alpha": BH_FDR_ALPHA,
            "tested_pairs_count": tested_count,
            "pairs": pairs,
            "causality_disclaimer": "All cross-domain correlations report descriptive sample associations only; causation is never implied.",
        }

        # 4. Multi-Domain Diagnostic Scorecard (strictly no 0-100 CHI score)
        scorecard = self._build_diagnostic_scorecard(domain_map)

        # 5. Deterministic Release & Synthesis Rules + Cross-Module Consolidation
        domain_coverages = {k: v.coverage.status for k, v in domain_map.items()}
        synthesis_rules = build_cross_domain_rules(
            domain_name=self.domain_name,
            repo_context=repo_context,
            domain_coverages=domain_coverages,
            correlations=correlations,
        )

        # ReleaseEvolutionCalculator rules stay accessible under this step alongside RULE-SYN-* without duplicating C.2-C.5 rules
        release_res = domain_map["release_evolution"]
        module_own_rules = consolidate_rules([release_res], synthesis_rules)

        all_domain_results = [
            domain_map["development_activity"],
            domain_map["contributor_ecosystem"],
            domain_map["pull_request_intelligence"],
            domain_map["issue_intelligence"],
            release_res,
        ]
        consolidated_all = consolidate_rules(all_domain_results, synthesis_rules)

        severity_counts: Dict[str, int] = defaultdict(int)
        for rule in consolidated_all:
            severity_counts[rule.severity] += 1

        # 6. Coverage & Confidence Penalty
        coverage, confidence_penalty, penalties = self._evaluate_coverage_and_confidence(
            domain_coverages=domain_coverages,
            correlations=correlations,
            release_res=release_res,
        )

        # Embed full release_evolution inside repository_synthesis cleanly for AnalysisPayload compatibility
        metrics: Dict[str, Any] = {
            "chi_deferred": True,
            "chi_deferral_reason": "Composite Health Index [0-100] is formally deferred in favor of the objective Multi-Domain Diagnostic Scorecard.",
            "repository_context": repo_context,
            "diagnostic_scorecard": scorecard,
            "cross_domain_correlations": correlations,
            "release_evolution": {
                **release_res.metrics,
                "coverage": release_res.coverage.model_dump(),
            },
            "consolidated_rules_summary": {
                "total_unique_rules": len(consolidated_all),
                "severity_counts": dict(severity_counts),
            },
        }

        return CalculationResult(
            domain=self.domain_name,
            metrics=metrics,
            coverage=coverage,
            confidence_penalty=confidence_penalty,
            rules=module_own_rules,
            details={
                "penalties": penalties,
                "snapshot_revision": snapshot_revision,
                "domain_coverages": domain_coverages,
                "consolidated_rules": [r.model_dump() for r in consolidated_all],
            },
        )

    @staticmethod
    def _build_monthly_aligned_series(
        repository_id: int,
        db: Session,
        ref_time: datetime,
    ) -> Dict[str, Dict[Tuple[int, int], Optional[float]]]:
        """Extract monthly aligned domain series strictly <= ref_time respecting NULL diff coverage."""
        commits_rows = db.execute(
            select(
                Commit.sha,
                Commit.author_date,
                Commit.committer_date,
                Commit.additions,
                Commit.deletions,
            ).where(Commit.repository_id == repository_id)
        ).fetchall()

        issues_rows = db.execute(
            select(
                Issue.issue_number,
                Issue.created_at,
            ).where(Issue.repository_id == repository_id)
        ).fetchall()

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
            ).where(PullRequest.repository_id == repository_id)
        ).fetchall()

        contrib_bots = db.execute(
            select(Contributor.github_id, Contributor.login)
            .where(Contributor.repository_id == repository_id, Contributor.is_bot.is_(True))
        ).fetchall()
        bot_ids: Set[int] = {r.github_id for r in contrib_bots if r.github_id is not None}
        bot_logins: Set[str] = {r.login.strip().lower() for r in contrib_bots if r.login}

        # 1. Commits monthly counts & NULL-aware monthly code churn
        seen_shas: Set[str] = set()
        commits_per_m: Dict[Tuple[int, int], int] = defaultdict(int)
        diffs_per_m: Dict[Tuple[int, int], int] = defaultdict(int)
        churn_per_m: Dict[Tuple[int, int], int] = defaultdict(int)
        total_commits = 0
        total_diffs = 0
        commit_months: List[Tuple[int, int]] = []

        for row in commits_rows:
            if row.sha in seen_shas:
                continue
            dt = _ensure_utc(row.author_date or row.committer_date)
            if dt is None or dt > ref_time:
                continue
            seen_shas.add(row.sha)
            ym = (dt.year, dt.month)
            commits_per_m[ym] += 1
            total_commits += 1
            commit_months.append(ym)
            if row.additions is not None and row.deletions is not None:
                diffs_per_m[ym] += 1
                churn_per_m[ym] += max(0, int(row.additions)) + max(0, int(row.deletions))
                total_diffs += 1

        monthly_commit_velocity: Dict[Tuple[int, int], Optional[float]] = {}
        monthly_code_churn: Dict[Tuple[int, int], Optional[float]] = {}
        if commit_months:
            min_cm, max_cm = min(commit_months), max(commit_months)
            overall_diff_cov = total_diffs / float(total_commits) if total_commits > 0 else 0.0
            for ym in _month_range_sequence(min_cm, max_cm):
                c_cnt = commits_per_m.get(ym, 0)
                monthly_commit_velocity[ym] = float(c_cnt)
                if total_diffs == 0:
                    monthly_code_churn[ym] = None
                elif c_cnt == 0:
                    monthly_code_churn[ym] = 0.0 if overall_diff_cov >= 0.50 else None
                else:
                    m_diff_cov = diffs_per_m.get(ym, 0) / float(c_cnt)
                    monthly_code_churn[ym] = float(churn_per_m[ym]) if m_diff_cov >= 0.50 else None

        # 2. Monthly Issue Creations
        seen_iss: Set[int] = set()
        iss_created_m: Dict[Tuple[int, int], int] = defaultdict(int)
        issue_months: List[Tuple[int, int]] = []
        for row in issues_rows:
            if row.issue_number in seen_iss:
                continue
            dt = _ensure_utc(row.created_at)
            if dt is None or dt > ref_time:
                continue
            seen_iss.add(row.issue_number)
            ym = (dt.year, dt.month)
            iss_created_m[ym] += 1
            issue_months.append(ym)

        monthly_issue_creations: Dict[Tuple[int, int], Optional[float]] = {}
        if issue_months:
            for ym in _month_range_sequence(min(issue_months), max(issue_months)):
                monthly_issue_creations[ym] = float(iss_created_m.get(ym, 0))

        # 3. PR monthly outcomes, merges, and TTFResp
        seen_prs: Set[int] = set()
        pr_map: Dict[int, Dict[str, Any]] = {}
        pr_merge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        merge_months: List[Tuple[int, int]] = []
        cohort_merged: Dict[Tuple[int, int], int] = defaultdict(int)
        cohort_closed_unmerged: Dict[Tuple[int, int], int] = defaultdict(int)
        pr_create_months: List[Tuple[int, int]] = []

        for row in prs_rows:
            if row.pr_number in seen_prs:
                continue
            c_dt = _ensure_utc(row.created_at)
            if c_dt is None or c_dt > ref_time:
                continue
            seen_prs.add(row.pr_number)
            c_ym = (c_dt.year, c_dt.month)
            pr_create_months.append(c_ym)

            merged_dt = _ensure_utc(row.merged_at)
            closed_dt = _ensure_utc(row.closed_at)
            raw_state = (row.state or "open").lower().strip()

            if merged_dt is not None and merged_dt <= ref_time:
                cohort_merged[c_ym] += 1
                m_ym = (merged_dt.year, merged_dt.month)
                pr_merge_counts[m_ym] += 1
                merge_months.append(m_ym)
            elif (closed_dt is not None and closed_dt <= ref_time) or (raw_state == "closed" and merged_dt is None and closed_dt is None):
                cohort_closed_unmerged[c_ym] += 1

            pr_map[row.id] = {
                "created_at": c_dt,
                "ym": c_ym,
                "author_id": row.author_id,
                "author": row.author,
            }

        monthly_pr_merge_volume: Dict[Tuple[int, int], Optional[float]] = {}
        if merge_months:
            for ym in _month_range_sequence(min(merge_months), max(merge_months)):
                monthly_pr_merge_volume[ym] = float(pr_merge_counts.get(ym, 0))

        # Query external human PR response timestamps <= ref_time if PRs exist
        pr_first_resp_h: Dict[int, float] = {}
        if pr_map:
            pr_ids = list(pr_map.keys())
            reviews = db.execute(
                select(
                    PullRequestReview.pull_request_id,
                    PullRequestReview.author,
                    PullRequestReview.author_id,
                    PullRequestReview.state,
                    PullRequestReview.submitted_at,
                ).where(PullRequestReview.pull_request_id.in_(pr_ids))
            ).fetchall()
            inlines = db.execute(
                select(
                    PullRequestReviewComment.pull_request_id,
                    PullRequestReviewComment.author,
                    PullRequestReviewComment.author_id,
                    PullRequestReviewComment.created_at,
                ).where(PullRequestReviewComment.pull_request_id.in_(pr_ids))
            ).fetchall()
            issue_comms = db.execute(
                select(
                    IssueComment.pull_request_id,
                    IssueComment.author,
                    IssueComment.author_id,
                    IssueComment.created_at,
                ).where(
                    IssueComment.repository_id == repository_id,
                    IssueComment.pull_request_id.isnot(None),
                    IssueComment.pull_request_id.in_(pr_ids),
                )
            ).fetchall()

            for pr_id, actor, actor_id, raw_dt in (
                [(r.pull_request_id, r.author, r.author_id, r.submitted_at) for r in reviews if (r.state or "").upper().strip() in FORMAL_REVIEW_STATES or (r.state or "").upper().strip() == "DISMISSED"]
                + [(c.pull_request_id, c.author, c.author_id, c.created_at) for c in inlines]
                + [(ic.pull_request_id, ic.author, ic.author_id, ic.created_at) for ic in issue_comms]
            ):
                pinfo = pr_map.get(pr_id)
                if pinfo is None:
                    continue
                dt = _ensure_utc(raw_dt)
                if dt is None or dt > ref_time:
                    continue
                if actor_id is None and not (actor and actor.strip()):
                    continue
                if PullRequestIntelligenceCalculator._is_self_interaction(actor_id, actor, pinfo["author_id"], pinfo["author"]):
                    continue
                if PullRequestIntelligenceCalculator._is_bot(actor_id, actor, bot_ids, bot_logins):
                    continue
                diff_sec = (dt - pinfo["created_at"]).total_seconds()
                if diff_sec < 0.0:
                    continue
                hours = diff_sec / 3600.0
                if pr_id not in pr_first_resp_h or hours < pr_first_resp_h[pr_id]:
                    pr_first_resp_h[pr_id] = hours

        resp_hours_by_m: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for pr_id, hrs in pr_first_resp_h.items():
            resp_hours_by_m[pr_map[pr_id]["ym"]].append(hrs)

        monthly_median_ttf_resp_hours: Dict[Tuple[int, int], Optional[float]] = {}
        monthly_pr_abandonment_rate: Dict[Tuple[int, int], Optional[float]] = {}
        if pr_create_months:
            for ym in _month_range_sequence(min(pr_create_months), max(pr_create_months)):
                r_list = resp_hours_by_m.get(ym, [])
                monthly_median_ttf_resp_hours[ym] = statistics.median(r_list) if r_list else None
                resolved_m = cohort_merged.get(ym, 0) + cohort_closed_unmerged.get(ym, 0)
                monthly_pr_abandonment_rate[ym] = (
                    cohort_closed_unmerged.get(ym, 0) / float(resolved_m) if resolved_m > 0 else None
                )

        return {
            "monthly_code_churn": monthly_code_churn,
            "monthly_issue_creations": monthly_issue_creations,
            "monthly_median_ttf_resp_hours": monthly_median_ttf_resp_hours,
            "monthly_pr_abandonment_rate": monthly_pr_abandonment_rate,
            "monthly_commit_velocity": monthly_commit_velocity,
            "monthly_pr_merge_volume": monthly_pr_merge_volume,
        }

    @staticmethod
    def _build_diagnostic_scorecard(domain_map: Dict[str, CalculationResult]) -> Dict[str, Any]:
        """Construct the objective, non-scoring Multi-Domain Diagnostic Scorecard per v1.1 §6.2."""
        act = domain_map["development_activity"]
        con = domain_map["contributor_ecosystem"]
        prs = domain_map["pull_request_intelligence"]
        iss = domain_map["issue_intelligence"]
        rel = domain_map["release_evolution"]

        act_vel = act.metrics.get("velocity", {})
        act_con = act.metrics.get("consistency", {})
        con_conc = con.metrics.get("concentration", {})
        con_ret = con.metrics.get("retention", {})
        pr_resp = prs.metrics.get("responsiveness", {})
        pr_out = prs.metrics.get("outcome_distribution", {})
        pr_ml = prs.metrics.get("merge_latency", {})
        iss_flow = iss.metrics.get("observed_flow", {})
        iss_res = iss.metrics.get("resolution_distribution", {})
        iss_age = iss.metrics.get("backlog_aging", {})
        iss_stk = iss.metrics.get("backlog_stock_summary", {})
        rel_cad = rel.metrics.get("cadence", {})
        rel_sem = rel.metrics.get("semver_adoption", {})

        return {
            "activity_profile": {
                "coverage_status": act.coverage.status,
                "trailing_4w_velocity": act_vel.get("trailing_4w_velocity"),
                "trailing_12w_velocity": act_vel.get("trailing_12w_velocity"),
                "velocity_trend_ratio": act_vel.get("velocity_trend_ratio"),
                "consistency_classification": act_con.get("consistency_classification"),
                "active_week_ratio": act_con.get("active_week_ratio"),
                "weekly_cv": act_con.get("coefficient_of_variation"),
            },
            "contributor_resilience": {
                "coverage_status": con.coverage.status,
                "active_human_commit_authors_count": con_conc.get("active_human_commit_authors_count"),
                "gini_coefficient": con_conc.get("gini_coefficient"),
                "bf50": con_conc.get("bf50"),
                "bf80": con_conc.get("bf80"),
                "top1_share": con_conc.get("top1_share"),
                "retention_status": con_ret.get("status"),
            },
            "pr_responsiveness": {
                "coverage_status": prs.coverage.status,
                "eligible_prs_count": pr_out.get("total_eligible_prs"),
                "ttf_resp_p50_hours": pr_resp.get("ttf_resp", {}).get("p50_hours"),
                "ttfr_formal_p50_hours": pr_resp.get("ttfr_formal", {}).get("p50_hours"),
                "merge_latency_p50_hours": pr_ml.get("p50_hours"),
                "merge_rate": pr_out.get("merge_rate"),
                "abandonment_rate": pr_out.get("abandonment_rate"),
            },
            "issue_health": {
                "coverage_status": iss.coverage.status,
                "eligible_issues_count": iss.coverage.sample_size,
                "flow_ratio": iss_flow.get("flow_ratio"),
                "resolution_p50_hours": iss_res.get("p50_hours"),
                "stale_fraction": iss_age.get("stale_fraction"),
                "github_reported_open_issues_and_prs": iss_stk.get("github_metadata_snapshot", {}).get("github_combined_open_issues_and_prs"),
                "combined_open_observed_to_reported_ratio_percentage": iss_stk.get("coverage_assessment", {}).get("combined_open_observed_to_reported_ratio_percentage"),
            },
            "release_cadence": {
                "coverage_status": rel.coverage.status,
                "total_releases": rel_cad.get("total_releases"),
                "regularity_classification": rel_cad.get("regularity_classification"),
                "cv_release": rel_cad.get("cv_release"),
                "median_interval_days": rel_cad.get("median_interval_days"),
                "days_since_last_release": rel_cad.get("days_since_last_release"),
                "semver_adoption_rate": rel_sem.get("semver_adoption_rate"),
            },
        }

    def _evaluate_coverage_and_confidence(
        self,
        domain_coverages: Dict[str, str],
        correlations: Dict[str, Any],
        release_res: CalculationResult,
    ) -> Tuple[MetricCoverageItem, float, List[Dict[str, Any]]]:
        """Evaluate overall repository_synthesis coverage across the 5 synthesized domains."""
        sufficient_domains = sum(1 for st in domain_coverages.values() if st == "sufficient")
        degraded_or_better = sum(1 for st in domain_coverages.values() if st in ("sufficient", "degraded"))
        total_domains = len(domain_coverages)

        penalties: List[Dict[str, Any]] = []
        penalty_total = 0.0

        if degraded_or_better == 0:
            cov_status = "empty"
            p = 0.5
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-NO-SUFFICIENT-DOMAINS",
                "penalty": p,
                "reason": "None of the 5 underlying analytical domains have sufficient/degraded data.",
            })
        elif sufficient_domains < 2:
            cov_status = "insufficient" if sufficient_domains == 0 else "degraded"
            p = 0.3 if cov_status == "insufficient" else 0.2
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-PARTIAL-DOMAIN-SYNTHESIS",
                "penalty": p,
                "reason": f"Only {sufficient_domains} of {total_domains} analytical domains reached sufficient coverage.",
            })
        elif sufficient_domains < total_domains:
            cov_status = "degraded"
            p = 0.1
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-INCOMPLETE-CROSS-DOMAIN-MATRIX",
                "penalty": p,
                "reason": f"{sufficient_domains} of {total_domains} domains sufficient; cross-domain synthesis reflects partial coverage.",
            })
        else:
            cov_status = "sufficient"

        if correlations["tested_pairs_count"] == 0 and cov_status in ("sufficient", "degraded"):
            p = 0.05
            penalty_total += p
            penalties.append({
                "rule": "PENALTY-NO-PERMUTATION-TESTED-PAIRS",
                "penalty": p,
                "reason": "Zero cross-domain pairs met >= 12 non-autocorrelated overlapping months for permutation testing.",
            })

        coverage = MetricCoverageItem(
            domain=self.domain_name,
            status=cov_status,
            sample_size=sufficient_domains,
            min_sample_required=2,
            details={
                "total_domains_synthesized": total_domains,
                "sufficient_domains_count": sufficient_domains,
                "degraded_or_sufficient_domains_count": degraded_or_better,
                "subdomains": {
                    **domain_coverages,
                    "cross_domain_correlations": correlations["status"],
                    "release_cadence": release_res.metrics["cadence"]["status"],
                    "release_prerelease_surges": release_res.metrics["prerelease_surges"]["status"],
                    "release_semver": release_res.metrics["semver_adoption"]["status"],
                },
            },
        )

        return coverage, round(min(1.0, penalty_total), 3), penalties
