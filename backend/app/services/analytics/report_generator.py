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
        activity = payload.get("development_activity", {})
        contributors = payload.get("contributor_ecosystem", {})
        prs = payload.get("pull_request_intelligence", {})
        issues = payload.get("issue_intelligence", {})
        synthesis = payload.get("repository_synthesis", {})
        releases = synthesis.get("release_evolution", {})
        discussions = synthesis.get("discussions_intelligence", {})
        correlations = synthesis.get("cross_domain_correlations", [])

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
        act_summary = activity.get("summary", {})
        vel_trend = act_summary.get("velocity_trend", "stable").capitalize()
        tot_commits = act_summary.get("total_commits", 0)
        md.append(f"| **Development Activity** | Commit Velocity | {tot_commits:,} commits across evaluated window | `{vel_trend}` |")

        # Contributors row
        gini = contributors.get("gini_coefficient")
        bf50 = contributors.get("bus_factors", {}).get("bf50", "N/A")
        gini_str = f"Gini {gini:.2f}" if gini is not None else "Gini N/A"
        md.append(f"| **Contributor Ecosystem** | Inequality & Bus Factor | {gini_str}; BF50 = {bf50} contributors | `Analyzed` |")

        # PR row
        tot_prs = prs.get("total_prs", 0)
        merge_rate = _format_pct(prs.get("outcomes", {}).get("merge_rate"))
        ttf_p50 = _format_hours(prs.get("turnaround_hours", {}).get("p50"))
        md.append(f"| **Pull Requests & Reviews** | Merge Rate & Turnaround | {tot_prs:,} PRs ({merge_rate} merged); p50 = {ttf_p50} | `Analyzed` |")

        # Issues row
        tot_issues = issues.get("total_issues", 0)
        sla_p50 = _format_hours(issues.get("resolution_sla_hours", {}).get("p50"))
        stale_count = issues.get("aging", {}).get("open_stale_count", 0)
        md.append(f"| **Issues & Backlog** | Resolution SLA & Dormancy | {tot_issues:,} issues; SLA p50 = {sla_p50}; {stale_count} stale | `Analyzed` |")

        # Discussions row
        disc_status = discussions.get("status", "not_enabled")
        if disc_status not in ("not_enabled", "disabled") and discussions.get("total_discussions", 0) > 0:
            qna_rate = _format_pct(discussions.get("qna", {}).get("qna_answered_rate"))
            tot_disc = discussions.get("total_discussions", 0)
            md.append(f"| **Community Discussions** | Q&A Resolution Rate | {tot_disc:,} threads; {qna_rate} Q&A answered | `Active` |")
        else:
            md.append(f"| **Community Discussions** | Feature Status | Discussions inactive or not enabled | `Inactive` |")

        # Releases row
        semver_pct = _format_pct(releases.get("semver_adoption_rate"))
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
                desc = pen.get("description", "Data penalty applied")
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
            # Sort critical -> warning -> info
            sev_order = {"critical": 0, "warning": 1, "info": 2}
            sorted_rules = sorted(rules, key=lambda r: sev_order.get(r.get("severity", "info"), 3))
            for r in sorted_rules:
                sev = r.get("severity", "info").upper()
                rid = r.get("rule_id", "RULE")
                domain = r.get("domain", "").replace("_", " ").title()
                msg = r.get("message", "")
                md.append(f"| `{sev}` | `{rid}` | {domain} | {msg} |")
        else:
            md.append("*No critical anomalies or threshold violations detected.*")
        md.append("")

        # 4. Cross-Domain Correlation Diagnostics
        md.append("## 4. Cross-Domain Correlation Explorer (Statistically Guarded)")
        md.append("")
        md.append("> **Scientific Disclaimer:** Associations reported below measure monotonic descriptive alignment over completed calendar months. They do **not** imply direct causality.")
        md.append("")
        if correlations:
            md.append("| Pair ID | Evaluated Metrics | Spearman $r_s$ | Permutation $p$-val | FDR $q$-val | Autocorrelation $|r_{\\text{AR1}}|$ | Significance |")
            md.append("|---|---|---|---|---|---|---|")
            for c in correlations:
                pid = c.get("pair_id", "")
                lx = c.get("label_x", "")
                ly = c.get("label_y", "")
                rs = c.get("spearman_rho")
                rs_str = f"{rs:.3f}" if rs is not None else "N/A"
                pval = c.get("permutation_p_value")
                pval_str = f"{pval:.4f}" if pval is not None else "N/A"
                qval = c.get("q_value")
                qval_str = f"{qval:.4f}" if qval is not None else "N/A"
                ar1 = c.get("max_abs_ar1")
                ar1_str = f"{ar1:.2f}" if ar1 is not None else "N/A"
                sig = "Significant" if c.get("significant_fdr_05") else "Not Significant"
                md.append(f"| `{pid}` | {lx} vs. {ly} | `{rs_str}` | `{pval_str}` | `{qval_str}` | `{ar1_str}` | `{sig}` |")
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
