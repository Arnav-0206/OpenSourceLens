"""Deterministic Rule Synthesis & Diagnostics Engine for OpenSourceLens (Phase C.6).

Generates evidence-backed rules without LLM text generation or causal claims:
- Release evolution rules (RULE-REL-EMPTY, RULE-REL-001 .. RULE-REL-004).
- Cross-domain synthesis rules (RULE-SYN-001 .. RULE-SYN-004).
- Deterministic deduplication and consolidation of rules across Modules 1-6 while leaving C.2-C.5 untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from backend.app.schemas.analytics_v2 import RuleInterpretation
from backend.app.services.analytics.base import CalculationResult


SEVERITY_ORDER: Dict[str, int] = {
    "critical": 0,
    "warning": 1,
    "positive": 2,
    "info": 3,
}


def build_release_rules(
    domain_name: str,
    cadence: Dict[str, Any],
    surges: Dict[str, Any],
    semver: Dict[str, Any],
) -> List[RuleInterpretation]:
    """Generate deterministic RULE-REL-* findings from release metrics."""
    rules: List[RuleInterpretation] = []
    r_count = cadence["total_releases"]

    if r_count == 0:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-REL-EMPTY",
                domain=domain_name,
                severity="info",
                finding="No Releases Observed: Repository snapshot contains zero dated releases.",
                evidence={"total_releases": 0, "missing_published_at_count": cadence["missing_published_at_count"]},
                recommendation="Publish tagged GitHub releases to track release predictability and SemVer adoption.",
            )
        )
        return rules

    # RULE-REL-001: Regular Release Cadence (CV <= 0.35 & R >= 4)
    cv_rel = cadence["cv_release"]
    if r_count >= 4 and cv_rel is not None and cv_rel <= 0.35:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-REL-001",
                domain=domain_name,
                severity="positive",
                finding=(
                    f"Regular Release Cadence: Releases show low interval variability "
                    f"(CV={cv_rel} <= 0.35 across {r_count} releases, median {cadence['median_interval_days']}d)."
                ),
                evidence={
                    "cv_release": cv_rel,
                    "total_releases": r_count,
                    "median_interval_days": cadence["median_interval_days"],
                    "regularity_classification": cadence["regularity_classification"],
                },
                recommendation=None,
            )
        )

    # RULE-REL-002: Dormant Releases (> 365 days since latest release)
    days_since = cadence["days_since_last_release"]
    if days_since is not None and days_since > 365.0:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-REL-002",
                domain=domain_name,
                severity="warning",
                finding=(
                    f"Dormant Releases: {days_since} days have elapsed since latest release '{cadence['latest_release_tag']}' (> 365d)."
                ),
                evidence={
                    "days_since_last_release": days_since,
                    "latest_release_tag": cadence["latest_release_tag"],
                    "total_releases": r_count,
                },
                recommendation="Clarify release schedule or maintenance status if ongoing fixes are shipped exclusively via default-branch commits.",
            )
        )

    # RULE-REL-003: Pre-Release Commit Surge Pattern
    mean_surge = surges["mean_surge_ratio"]
    if surges["status"] == "sufficient" and mean_surge is not None and mean_surge >= 1.50:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-REL-003",
                domain=domain_name,
                severity="info",
                finding=(
                    f"Consistent Pre-Release Commit Surge: Daily commit rate in the 14d pre-release window averages "
                    f"{mean_surge}x the preceding 60d baseline across {surges['valid_surge_releases_count']} releases."
                ),
                evidence={
                    "mean_surge_ratio": mean_surge,
                    "median_surge_ratio": surges["median_surge_ratio"],
                    "valid_surge_releases_count": surges["valid_surge_releases_count"],
                },
                recommendation=None,
            )
        )

    # RULE-REL-004: Semantic Versioning Adoption
    semver_rate = semver["semver_adoption_rate"]
    if r_count >= 3 and semver_rate is not None:
        if semver_rate >= 0.80:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-REL-004",
                    domain=domain_name,
                    severity="positive",
                    finding=(
                        f"Strong SemVer Adoption: {round(semver_rate * 100.0, 1)}% of releases "
                        f"({semver['semver_compliant_count']} of {r_count}) follow Semantic Versioning."
                    ),
                    evidence={
                        "semver_adoption_rate": semver_rate,
                        "semver_compliant_count": semver["semver_compliant_count"],
                        "total_releases": r_count,
                    },
                    recommendation=None,
                )
            )
        elif semver_rate < 0.30:
            rules.append(
                RuleInterpretation(
                    rule_id="RULE-REL-004",
                    domain=domain_name,
                    severity="info",
                    finding=(
                        f"Non-SemVer Versioning Scheme: Only {round(semver_rate * 100.0, 1)}% of releases match SemVer 2.0.0 "
                        "(project may use CalVer or custom tags)."
                    ),
                    evidence={
                        "semver_adoption_rate": semver_rate,
                        "non_semver_count": semver["non_semver_count"],
                        "total_releases": r_count,
                    },
                    recommendation=None,
                )
            )

    return rules


def build_cross_domain_rules(
    domain_name: str,
    repo_context: Dict[str, Any],
    domain_coverages: Dict[str, str],
    correlations: Dict[str, Any],
) -> List[RuleInterpretation]:
    """Generate non-causal, evidence-backed RULE-SYN-* findings."""
    rules: List[RuleInterpretation] = []

    # RULE-SYN-001: Archived Repository Context Flag
    if repo_context.get("is_archived") is True:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-SYN-001",
                domain=domain_name,
                severity="warning",
                finding="Archived Repository Status: GitHub repository metadata marks this repository as archived (read-only).",
                evidence={"is_archived": True, "repo_name": repo_context.get("repo_name")},
                recommendation="Interpret recent zero activity, dormant releases, and open backlog metrics in light of archival status.",
            )
        )

    # RULE-SYN-002: Significant FDR-Adjusted Cross-Domain Exploratory Associations (Strictly Non-Causal)
    tested_pairs = correlations.get("pairs", [])
    sig_pairs = [
        p for p in tested_pairs
        if p.get("status") == "tested" and p.get("significant_fdr_05") is True
    ]
    for p in sig_pairs:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-SYN-002",
                domain=domain_name,
                severity="info",
                finding=(
                    f"Observed Exploratory Association ({p['pair_id']}): {p['label_x']} empirically covaries with "
                    f"{p['label_y']} across K={p['k_months']} overlapping months (Spearman r_obs={p['spearman_r']}, "
                    f"BH q={p['q_value']}; empirical association only, non-causal)."
                ),
                evidence={
                    "pair_id": p["pair_id"],
                    "spearman_r": p["spearman_r"],
                    "p_value": p["p_value"],
                    "q_value": p["q_value"],
                    "k_months": p["k_months"],
                    "max_abs_ar1": p["max_abs_ar1"],
                },
                recommendation=None,
            )
        )

    # RULE-SYN-003: Serial Dependence Autocorrelation Suppression Transparency
    ar1_suppressed = [p for p in tested_pairs if p.get("status") == "serial_dependence_detected"]
    if ar1_suppressed:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-SYN-003",
                domain=domain_name,
                severity="info",
                finding=(
                    f"Autocorrelation Significance Guardrail Triggered: {len(ar1_suppressed)} cross-domain pair(s) "
                    "exhibited |r_AR1| > 0.40; permutation p-values were suppressed to prevent inflated significance."
                ),
                evidence={
                    "suppressed_pairs": [
                        {"pair_id": p["pair_id"], "spearman_r": p["spearman_r"], "max_abs_ar1": p["max_abs_ar1"]}
                        for p in ar1_suppressed
                    ]
                },
                recommendation=None,
            )
        )

    # RULE-SYN-004: Multi-Domain Coverage Completeness Summary
    empty_or_insuff = [
        dom for dom, st in domain_coverages.items() if st in ("empty", "insufficient")
    ]
    if empty_or_insuff:
        rules.append(
            RuleInterpretation(
                rule_id="RULE-SYN-004",
                domain=domain_name,
                severity="info",
                finding=(
                    f"Partial Multi-Domain Coverage: {len(empty_or_insuff)} domain(s) reported empty/insufficient data "
                    f"({', '.join(sorted(empty_or_insuff))}); scorecard dimensions reflect observed data only."
                ),
                evidence={"domain_statuses": domain_coverages, "incomplete_domains": sorted(empty_or_insuff)},
                recommendation="Run multi-stream repository ingestion if full cross-domain correlation analysis is required.",
            )
        )

    return rules


def consolidate_rules(
    domain_results: List[CalculationResult],
    synthesis_rules: List[RuleInterpretation],
) -> List[RuleInterpretation]:
    """Deterministically deduplicate and sort rules across domain calculators without modifying C.2-C.5 outputs."""
    combined: List[RuleInterpretation] = []
    for res in domain_results:
        combined.extend(res.rules)
    combined.extend(synthesis_rules)

    seen_keys: Set[Tuple[str, str, str]] = set()
    unique: List[RuleInterpretation] = []
    for rule in combined:
        key = (rule.rule_id, rule.domain, rule.finding)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(rule)

    unique.sort(
        key=lambda r: (
            SEVERITY_ORDER.get(r.severity, 99),
            r.domain,
            r.rule_id,
            r.finding,
        )
    )
    return unique
