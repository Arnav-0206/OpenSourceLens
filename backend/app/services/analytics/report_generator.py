"""Executive Intelligence Report Generator for OpenSourceLens.

Compiles complete multi-domain telemetry, statistical diagnostics, rule findings,
and cross-domain evaluations into structured Markdown audit documents and JSON exports.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional


def _format_hours(hours: Optional[float]) -> str:
    """Format duration in hours into a readable string."""
    if hours is None:
        return "N/A"
    if hours < 1:
        return f"{round(hours * 60)} min"
    if hours < 48:
        return f"{hours:.1f} hrs"
    days = hours / 24.0
    return f"{days:.1f} days"


def _format_pct(val: Optional[float]) -> str:
    """Format fractional float [0.0, 1.0] as a percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100.0:.1f}%"


class ReportGeneratorService:
    """Produces GitHub Flavored Markdown and JSON intelligence reports."""

    @classmethod
    def generate_markdown(
        cls,
        repo: Any,
        run: Any,
        payload: Dict[str, Any],
        topics: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Compile complete analytical snapshot into an executive Markdown report."""
        repo_name = getattr(repo, "repo_name", "unknown/repository")
        owner = getattr(repo, "owner", repo_name.split("/")[0] if "/" in repo_name else "")
        name = getattr(repo, "name", repo_name.split("/")[-1] if "/" in repo_name else "")
        stars = getattr(repo, "stars_count", 0)
        forks = getattr(repo, "forks_count", 0)
        lang = getattr(repo, "primary_language", "Unknown") or "Unknown"

        run_id = getattr(run, "id", "N/A")
        methodology = getattr(run, "methodology_version", "2026.2")
        completed_at = getattr(run, "completed_at", None)
        timestamp_str = (
            completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
            if isinstance(completed_at, datetime)
            else str(completed_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        )

        conf_summary = getattr(run, "confidence_summary", {}) or {}
        conf_band = conf_summary.get("band", "unreliable").upper()
        conf_score = conf_summary.get("score", 0.0)
        conf_penalties = conf_summary.get("penalties", [])

        rules = getattr(run, "rule_interpretations", []) or []

        # Domain slices
        activity = payload.get("development_activity", {}) or {}
        contributors = payload.get("contributor_ecosystem", {}) or {}
        prs = payload.get("pull_request_intelligence", {}) or {}
        issues = payload.get("issue_intelligence", {}) or {}
        synthesis = payload.get("repository_synthesis", {}) or {}
        releases = synthesis.get("release_evolution", {}) or {}
        discussions = synthesis.get("discussions_intelligence", {}) or {}
        raw_correlations = synthesis.get("cross_domain_correlations", [])
        if isinstance(raw_correlations, dict):
            correlations = raw_correlations.get("pairs", []) or []
        elif isinstance(raw_correlations, list):
            correlations = raw_correlations
        else:
            correlations = []

        # Start Markdown buffer
        md: List[str] = []
        md.append(f"# OpenSourceLens — Repository Intelligence Audit")
        md.append(f"## `{repo_name}`")
        md.append("")
        md.append(f"> **Report Generated:** {timestamp_str} | **Methodology Version:** `{methodology}` | **Run ID:** `{run_id}`")
        md.append(f"> **Overall Data Confidence:** `{conf_band}` ({conf_score * 100.0:.0f}%) | **Primary Language:** {lang} | **Stars:** {stars:,} | **Forks:** {forks:,}")
        md.append("")
        md.append("---")
        md.append("")

        # 1. Executive Summary Table
        md.append("## 1. Multi-Domain Diagnostic Scorecard")
        md.append("")
        md.append("| Domain | Core Indicator | Primary Finding | Health / Status |")
        md.append("|---|---|---|---|")

        # Activity row
        act_summary = activity.get("summary", {}) or {}
        vel_trend = (act_summary.get("velocity_trend") or "active").capitalize()
        tot_commits = act_summary.get("total_commits", 0) or 0
        md.append(f"| **Development Activity** | Commit Velocity | {tot_commits:,} commits across evaluated window | `{vel_trend}` |")

        # Contributors row
        contrib_conc = contributors.get("concentration", {}) or {}
        contrib_part = contributors.get("participation", {}) or {}
        gini = contrib_conc.get("gini_coefficient")
        if gini is None:
            gini = contributors.get("gini_coefficient")
        bf50 = contrib_conc.get("bf50")
        if bf50 is None:
            bf50 = (contributors.get("bus_factors", {}) or {}).get("bf50", "N/A")
        tot_contribs = (
            contrib_part.get("directory_contributors_count")
            or contributors.get("total_contributors")
            or contrib_conc.get("active_human_commit_authors_count")
            or 0
        )
        gini_str = f"Gini {gini:.2f}" if isinstance(gini, (int, float)) else "Gini N/A"
        md.append(f"| **Contributor Ecosystem** | Inequality & Bus Factor | {tot_contribs:,} contributors; {gini_str}; BF50 = {bf50} | `Analyzed` |")

        # PR row
        pr_outcomes = prs.get("outcome_distribution", {}) or prs.get("outcomes", {}) or {}
        tot_prs = pr_outcomes.get("total_eligible_prs") or prs.get("total_prs") or 0
        merge_rate = _format_pct(pr_outcomes.get("merge_rate"))
        pr_merge_lat = prs.get("merge_latency", {}) or {}
        pr_turnaround = prs.get("turnaround_hours", {}) or {}
        ttf_p50_val = pr_merge_lat.get("p50_hours")
        if ttf_p50_val is None:
            ttf_p50_val = pr_turnaround.get("p50")
        first_resp_val = ((prs.get("responsiveness", {}) or {}).get("ttf_resp", {}) or {}).get("p50_hours")
        pr_speed_str = (
            f"merge p50 = {_format_hours(ttf_p50_val)}"
            if ttf_p50_val is not None
            else (f"first reply p50 = {_format_hours(first_resp_val)}" if first_resp_val is not None else "N/A")
        )
        md.append(f"| **Pull Requests & Reviews** | Merge Rate & Speed | {tot_prs:,} PRs ({merge_rate} merged); {pr_speed_str} | `Analyzed` |")

        # Issues row
        iss_flow = issues.get("observed_flow", {}) or {}
        iss_res = issues.get("resolution_distribution", {}) or {}
        iss_aging = issues.get("backlog_aging", {}) or issues.get("aging", {}) or {}
        tot_issues = (
            (issues.get("scope", {}) or {}).get("eligible_issues_count")
            or iss_flow.get("total_created_observed")
            or issues.get("total_issues")
            or 0
        )
        sla_p50_val = iss_res.get("p50_hours")
        if sla_p50_val is None:
            sla_p50_val = (issues.get("resolution_sla_hours", {}) or {}).get("p50")
        sla_p50 = _format_hours(sla_p50_val)
        stale_count = iss_aging.get("stale_gt_90d_count")
        if stale_count is None:
            stale_count = iss_aging.get("open_stale_count", 0)
        md.append(f"| **Issues & Backlog** | Resolution Time & Backlog | {tot_issues:,} issues; close p50 = {sla_p50}; {stale_count} stale | `Analyzed` |")

        # Releases row
        rel_cadence = releases.get("cadence", {}) or {}
        rel_semver = releases.get("semver_adoption", {}) or {}
        semver_val = rel_semver.get("semver_adoption_rate")
        if semver_val is None:
            semver_val = releases.get("semver_adoption_rate")
        semver_pct = _format_pct(semver_val)
        tot_releases = rel_cadence.get("total_releases")
        if tot_releases is None:
            tot_releases = releases.get("total_releases", 0)
        md.append(f"| **Release Evolution** | SemVer 2.0.0 Adoption | {tot_releases} releases; {semver_pct} SemVer compliance | `Analyzed` |")

        md.append("")

        # 2. Confidence & Penalty Audit
        md.append("## 2. Data Confidence & Penalty Audit")
        md.append("")
        md.append(f"- **Final Confidence Score:** `{conf_score:.2f}` / 1.00 (`{conf_band}`)")
        if conf_penalties:
            md.append("- **Applied Penalties:**")
            for pen in conf_penalties:
                if not isinstance(pen, dict):
                    continue
                desc = pen.get("reason") or pen.get("description") or pen.get("rule") or "Data penalty applied"
                val = pen.get("penalty", 0.0)
                md.append(f"  - `{desc}`: `-{val * 100.0:.0f}%`")
        else:
            md.append("- **Applied Penalties:** None (Full data coverage across evaluated domains)")
        md.append("")

        # 3. Rule Findings
        md.append("## 3. Active Rule Interpretations")
        md.append("")
        if rules:
            md.append("| Severity | Rule Identifier | Domain | Finding & Recommendation |")
            md.append("|---|---|---|---|")
            # Sort critical -> warning -> positive -> info
            sev_order = {"critical": 0, "high": 0, "warning": 1, "positive": 2, "info": 3}
            dict_rules = [r for r in rules if isinstance(r, dict)]
            sorted_rules = sorted(dict_rules, key=lambda r: sev_order.get(str(r.get("severity", "info")).lower(), 4))
            for r in sorted_rules:
                sev = str(r.get("severity", "info")).upper()
                rid = r.get("rule_id", "RULE")
                domain = str(r.get("domain", "")).replace("_", " ").title()
                msg = r.get("finding") or r.get("message") or ""
                rec = r.get("recommendation")
                full_msg = f"{msg} — *Action: {rec}*" if rec else msg
                md.append(f"| `{sev}` | `{rid}` | {domain} | {full_msg} |")
        else:
            md.append("*No critical anomalies or threshold violations detected.*")
        md.append("")

        # 4. Cross-Domain Correlation Diagnostics
        md.append("## 4. Cross-Domain Correlation Explorer (Statistically Guarded)")
        md.append("")
        md.append("> **Scientific Disclaimer:** Associations reported below measure monotonic descriptive alignment over completed calendar months. They do **not** imply direct causality.")
        md.append("")
        dict_correlations = [c for c in correlations if isinstance(c, dict)]
        if dict_correlations:
            md.append("| Pair ID | Evaluated Metrics | Spearman $r_s$ | Permutation $p$-val | FDR $q$-val | Autocorrelation $|r_{\\text{AR1}}|$ | Significance |")
            md.append("|---|---|---|---|---|---|---|")
            for c in dict_correlations:
                pid = c.get("pair_id", "")
                lx = c.get("label_x") or c.get("metric_x") or ""
                ly = c.get("label_y") or c.get("metric_y") or ""
                pair_label = c.get("name") or f"{lx} vs. {ly}"
                rs = c.get("spearman_r")
                if rs is None:
                    rs = c.get("spearman_rho")
                rs_str = f"{rs:.3f}" if isinstance(rs, (int, float)) else "N/A"
                pval = c.get("p_value")
                if pval is None:
                    pval = c.get("permutation_p_value")
                pval_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else "N/A"
                qval = c.get("q_value")
                qval_str = f"{qval:.4f}" if isinstance(qval, (int, float)) else "N/A"
                ar1 = c.get("max_abs_ar1")
                ar1_str = f"{ar1:.2f}" if isinstance(ar1, (int, float)) else "N/A"
                sig = "Significant" if c.get("significant_fdr_05") else "Not Significant"
                md.append(f"| `{pid}` | {pair_label} | `{rs_str}` | `{pval_str}` | `{qval_str}` | `{ar1_str}` | `{sig}` |")
        else:
            md.append("*Insufficient overlapping monthly history to evaluate paired permutation tests.*")
        md.append("")

        # 5. Thematic Clusters
        if topics:
            md.append("## 5. Discovered Backlog Themes & Topic Clusters")
            md.append("")
            md.append("| Cluster # | Synthesized Theme | Representative Keywords | Total Threads | Open / Closed |")
            md.append("|---|---|---|---|---|")
            for t in topics:
                cid = t.get("cluster_id", 0) + 1
                title = t.get("title", "")
                kws = ", ".join(f"`#{w}`" for w in t.get("keywords", [])[:4])
                tot = t.get("total_count", 0)
                op = t.get("open_count", 0)
                cl = t.get("closed_count", 0)
                md.append(f"| #{cid} | **{title}** | {kws} | {tot} | {op} open / {cl} closed |")
            md.append("")

        # Footer
        md.append("---")
        md.append("")
        md.append(f"*Generated automatically by [OpenSourceLens](https://github.com/Arnav-0206/OpenSourceLens) on {timestamp_str}.*")
        md.append("")

        return "\n".join(md)

    @classmethod
    def generate_json(
        cls,
        repo: Any,
        run: Any,
        payload: Dict[str, Any],
        topics: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compile complete analytical snapshot into structured JSON export."""
        repo_name = getattr(repo, "repo_name", "unknown/repository")
        return {
            "metadata": {
                "repository": repo_name,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "methodology_version": getattr(run, "methodology_version", "2026.2"),
                "run_id": getattr(run, "id", None),
                "analytics_source_revision": getattr(run, "analytics_source_revision", None),
            },
            "repository_summary": {
                "stars": getattr(repo, "stars_count", 0),
                "forks": getattr(repo, "forks_count", 0),
                "language": getattr(repo, "primary_language", None),
                "is_archived": getattr(repo, "is_archived", False),
            },
            "confidence_summary": getattr(run, "confidence_summary", {}),
            "rule_interpretations": getattr(run, "rule_interpretations", []),
            "analytics_payload": payload,
            "discovered_topics": topics or [],
        }


report_generator = ReportGeneratorService()
