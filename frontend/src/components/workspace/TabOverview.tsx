"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Compass,
  GitCommit,
  GitPullRequest,
  Inbox,
  Info,
  Package,
  Sparkles,
  Users,
} from "lucide-react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AnalysisPayload,
  ConfidenceSummary,
  CrossDomainCorrelationPair,
  RuleInterpretation,
} from "@/lib/types";

interface TabOverviewProps {
  payload?: AnalysisPayload | null;
  confidence?: ConfidenceSummary | null;
  rules?: RuleInterpretation[] | null;
}

function formatDuration(hours?: number | null): string | null {
  if (hours == null || Number.isNaN(hours)) return null;
  if (hours < 1) {
    const mins = Math.round(hours * 60);
    return mins <= 1 ? "< 1 min" : `${mins} mins`;
  }
  if (hours >= 48) {
    return `${(hours / 24).toFixed(1)} days`;
  }
  return `${hours.toFixed(1)} hrs`;
}

function cleanFindingText(text: string): string {
  return text
    .replace(/\s*\(observed\s+'[^']+'\s+sample\)/gi, "")
    .replace(/\(BF50=(\d+),\s*BF80=(\d+)\)/gi, "($1 core contributor drives half of commits)")
    .replace(/TTFResp/gi, "first response time")
    .replace(/non-author human feedback/gi, "maintainer feedback")
    .replace(/\(reflects UTC timestamps, not contributor local hours\)/gi, "(in UTC time)")
    .trim();
}

function friendlyDomainName(domain?: string): string {
  const map: Record<string, string> = {
    development_activity: "Code Activity",
    contributor_ecosystem: "Contributors",
    pull_request_intelligence: "Pull Requests",
    issue_intelligence: "Issues",
    release_evolution: "Releases",
    repository_synthesis: "Overall Health",
  };
  if (!domain) return "General";
  const key = domain.toLowerCase().replace(/\s+/g, "_");
  return map[key] || domain.replace(/_/g, " ");
}

const NOISY_RULE_IDS = new Set([
  "RULE-DISC-EMPTY",
  "RULE-CON-004",
  "RULE-SYN-004",
  "RULE-REL-EMPTY",
  "RULE-PR-005",
]);

export default function TabOverview({
  payload,
  rules,
}: TabOverviewProps) {
  const [filterSeverity, setFilterSeverity] = useState<string>("all");

  const synthesis = payload?.repository_synthesis;
  const scorecard = synthesis?.diagnostic_scorecard;
  const crossDomain = synthesis?.cross_domain_correlations;

  // Domain raw slices
  const actData = payload?.development_activity;
  const contribData = payload?.contributor_ecosystem;
  const prData = payload?.pull_request_intelligence;
  const issueData = payload?.issue_intelligence;
  const relData = synthesis?.release_evolution;

  // Scorecard sub-objects
  const actCard = scorecard?.activity_profile;
  const contribCard = scorecard?.contributor_resilience;
  const prCard = scorecard?.pr_responsiveness;
  const issueCard = scorecard?.issue_health;
  const relCard = scorecard?.release_cadence;

  // 1. Code Activity metrics (real values only)
  const totalCommits = actData?.summary?.total_commits ?? 0;
  const activeWeeks = actData?.summary?.active_weeks_count ?? 0;
  const trailing4w = actCard?.trailing_4w_velocity ?? actData?.velocity?.trailing_4w_velocity;
  const velocityRatio = actCard?.velocity_trend_ratio ?? actData?.velocity?.velocity_trend_ratio;
  const trendLabel =
    velocityRatio != null
      ? velocityRatio >= 1.25
        ? "Accelerating"
        : velocityRatio <= 0.75
        ? "Slowing"
        : "Steady"
      : totalCommits > 0
      ? "Active"
      : "Quiet";

  // 2. Contributors metrics
  const totalContributors =
    contribData?.participation?.directory_contributors_count ??
    contribData?.total_contributors ??
    0;
  const activeAuthors =
    contribCard?.active_human_commit_authors_count ??
    contribData?.concentration?.active_human_commit_authors_count ??
    0;
  const top1Share =
    contribCard?.top1_share ?? contribData?.concentration?.top1_share;
  const bf50 = contribCard?.bf50 ?? contribData?.concentration?.bf50;

  // 3. Pull Requests metrics
  const totalPRs =
    prCard?.eligible_prs_count ??
    prData?.outcome_distribution?.total_eligible_prs ??
    prData?.total_prs ??
    0;
  const mergeRate = prCard?.merge_rate ?? prData?.outcome_distribution?.merge_rate;
  const firstResp = formatDuration(
    prCard?.ttf_resp_p50_hours ?? prData?.responsiveness?.ttf_resp?.p50_hours
  );
  const mergeMedian = formatDuration(
    prCard?.merge_latency_p50_hours ?? prData?.merge_latency?.p50_hours
  );
  const openPRs = prData?.outcome_distribution?.open_count ?? 0;

  // 4. Issues metrics (only show full card if repository actually has issues)
  const totalIssues =
    issueCard?.eligible_issues_count ??
    issueData?.observed_flow?.total_created_observed ??
    issueData?.total_issues ??
    0;
  const issueCloseTime = formatDuration(
    issueCard?.resolution_p50_hours ?? issueData?.resolution_distribution?.p50_hours
  );
  const openIssues = issueData?.backlog_aging?.open_issues_count ?? 0;
  const closeRatio = issueCard?.flow_ratio ?? issueData?.observed_flow?.flow_ratio;

  // 5. Releases metrics (only show card if releases exist)
  const totalReleases =
    relCard?.total_releases ?? relData?.cadence?.total_releases ?? 0;
  const semverRate =
    relCard?.semver_adoption_rate ?? relData?.semver_adoption?.semver_adoption_rate;
  const latestTag = relData?.cadence?.latest_release_tag;
  const daysSinceRelease =
    relCard?.days_since_last_release ?? relData?.cadence?.days_since_last_release;

  // Filter Cross-Domain pairs: ONLY show section if at least one pair actually has a computed correlation!
  const testablePairs: CrossDomainCorrelationPair[] = (
    crossDomain?.pairs || []
  ).filter((p) => p.spearman_r != null);

  // Filter & humanize findings
  const rawRules = (rules || []).filter((r) => !NOISY_RULE_IDS.has(r.rule_id));
  const warningRules = rawRules.filter(
    (r) => r.severity === "warning" || r.severity === "high" || r.severity === "critical"
  );
  const positiveRules = rawRules.filter((r) => r.severity === "positive");
  const infoRules = rawRules.filter(
    (r) =>
      r.severity !== "warning" &&
      r.severity !== "high" &&
      r.severity !== "critical" &&
      r.severity !== "positive"
  );

  const visibleRules =
    filterSeverity === "warnings"
      ? warningRules
      : filterSeverity === "positive"
      ? positiveRules
      : filterSeverity === "info"
      ? infoRules
      : rawRules;

  return (
    <div className="space-y-8">
      {/* 1. Executive Summary Grid (Zero N/A cards) */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-400" />
              <span>Repository Health at a Glance</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Key metrics extracted from commits, contributors, pull requests, and issues
            </p>
          </div>
          <span className="inline-flex items-center gap-1.5 text-[11px] px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-medium">
            <Activity className="w-3.5 h-3.5" />
            Live Repository Telemetry
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Card 1: Code Activity */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4 hover:border-slate-700 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <GitCommit className="w-4 h-4 text-indigo-400" />
                <span>Code Activity</span>
              </div>
              <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/25 font-medium">
                {trendLabel}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Recent Commits</div>
                <div className="text-2xl font-bold text-white mt-0.5">
                  {totalCommits.toLocaleString()}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Weekly Pace</div>
                <div className="text-2xl font-bold text-indigo-300 mt-0.5">
                  {trailing4w != null ? `${Math.round(trailing4w)}/wk` : `${activeWeeks} wks`}
                </div>
              </div>
            </div>

            <div className="text-xs text-slate-400 flex items-center justify-between pt-1">
              <span>Active weeks tracked: <strong className="text-slate-200">{activeWeeks}</strong></span>
              {actData?.activity_rhythms?.peak_weekday?.name && (
                <span>Busiest day: <strong className="text-slate-200">{actData.activity_rhythms.peak_weekday.name}</strong></span>
              )}
            </div>
          </div>

          {/* Card 2: Contributors */}
          <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4 hover:border-slate-700 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Users className="w-4 h-4 text-cyan-400" />
                <span>Contributors &amp; Team</span>
              </div>
              {bf50 != null && (
                <span
                  className={`text-[11px] px-2.5 py-0.5 rounded-full font-medium border ${
                    bf50 <= 2
                      ? "bg-amber-500/15 text-amber-300 border-amber-500/25"
                      : "bg-cyan-500/15 text-cyan-300 border-cyan-500/25"
                  }`}
                >
                  {bf50 === 1 ? "Single Key Maintainer" : `${bf50} Core Maintainers`}
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Total Contributors</div>
                <div className="text-2xl font-bold text-white mt-0.5">
                  {totalContributors.toLocaleString()}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Top Author Share</div>
                <div className="text-2xl font-bold text-cyan-300 mt-0.5">
                  {top1Share != null ? `${(top1Share * 100).toFixed(0)}%` : `${activeAuthors} active`}
                </div>
              </div>
            </div>

            <div className="text-xs text-slate-400 flex items-center justify-between pt-1">
              <span>Active committers: <strong className="text-slate-200">{activeAuthors}</strong></span>
              <span>
                Balance:{" "}
                <strong className="text-slate-200">
                  {top1Share != null && top1Share >= 0.5 ? "Concentrated" : "Distributed"}
                </strong>
              </span>
            </div>
          </div>

          {/* Card 3: Pull Requests */}
          {totalPRs > 0 && (
            <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4 hover:border-slate-700 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <GitPullRequest className="w-4 h-4 text-purple-400" />
                  <span>Pull Requests</span>
                </div>
                <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-purple-500/15 text-purple-300 border border-purple-500/25 font-medium">
                  {totalPRs.toLocaleString()} Tracked
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">
                    {firstResp ? "First Reply Speed" : mergeMedian ? "Median Merge Time" : "Open PRs"}
                  </div>
                  <div className="text-2xl font-bold text-emerald-400 mt-0.5">
                    {firstResp || mergeMedian || openPRs.toLocaleString()}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">
                    {mergeRate != null ? "GitHub Merge Rate" : "Open PRs"}
                  </div>
                  <div className="text-2xl font-bold text-white mt-0.5">
                    {mergeRate != null ? `${(mergeRate * 100).toFixed(1)}%` : openPRs.toLocaleString()}
                  </div>
                </div>
              </div>

              <div className="text-xs text-slate-400 flex items-center justify-between pt-1">
                <span>Currently open: <strong className="text-slate-200">{openPRs}</strong></span>
                <span>
                  {mergeMedian
                    ? `Merge time: ${mergeMedian}`
                    : mergeRate != null && mergeRate < 0.1
                    ? "External patch workflow"
                    : "Active review queue"}
                </span>
              </div>
            </div>
          )}

          {/* Card 4: Issues (only show full KPIs if repository uses GitHub Issues) */}
          {totalIssues > 0 && (
            <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4 hover:border-slate-700 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <Inbox className="w-4 h-4 text-rose-400" />
                  <span>Issues &amp; Backlog</span>
                </div>
                <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-rose-500/15 text-rose-300 border border-rose-500/25 font-medium">
                  AI Classified
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">Tracked Issues</div>
                  <div className="text-2xl font-bold text-white mt-0.5">
                    {totalIssues.toLocaleString()}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">
                    {issueCloseTime ? "Median Close Time" : "Open Issues"}
                  </div>
                  <div className="text-2xl font-bold text-cyan-300 mt-0.5">
                    {issueCloseTime || openIssues.toLocaleString()}
                  </div>
                </div>
              </div>

              <div className="text-xs text-slate-400 flex items-center justify-between pt-1">
                <span>Open now: <strong className="text-slate-200">{openIssues}</strong></span>
                {closeRatio != null && (
                  <span>Close ratio: <strong className="text-slate-200">{(closeRatio * 100).toFixed(0)}%</strong></span>
                )}
              </div>
            </div>
          )}

          {/* Card 5: Releases (only show if releases/tags were found) */}
          {totalReleases > 0 && (
            <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-4 hover:border-slate-700 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <Package className="w-4 h-4 text-amber-400" />
                  <span>Releases</span>
                </div>
                {latestTag && (
                  <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/25 font-mono">
                    {latestTag}
                  </span>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">Total Releases</div>
                  <div className="text-2xl font-bold text-white mt-0.5">
                    {totalReleases.toLocaleString()}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">
                    {semverRate != null ? "Standard Versioning" : "Last Release"}
                  </div>
                  <div className="text-2xl font-bold text-amber-300 mt-0.5">
                    {semverRate != null
                      ? `${(semverRate * 100).toFixed(0)}%`
                      : daysSinceRelease != null
                      ? `${daysSinceRelease}d ago`
                      : "Active"}
                  </div>
                </div>
              </div>

              {daysSinceRelease != null && (
                <div className="text-xs text-slate-400 pt-1">
                  Last shipped <strong className="text-slate-200">{daysSinceRelease} days ago</strong>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* 2. Historical Trend Correlations (ONLY rendered when real correlations are computed!) */}
      {testablePairs.length > 0 && (
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Compass className="w-5 h-5 text-cyan-400" />
              <span>Long-Term Pattern Discovery</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Statistically verified relationships across 12+ months of repository history
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {testablePairs.map((pair) => {
              const r = pair.spearman_r ?? 0;
              const isSignificant = Boolean(pair.significant_fdr_05);
              const chartSeries = (pair.monthly_points || []).map((pt) => ({
                month: pt.month,
                x: pt.val_x,
                y: pt.val_y,
              }));
              const monthsCount = pair.k_months ?? pair.overlapping_months?.length ?? chartSeries.length;

              return (
                <div
                  key={pair.pair_id}
                  className="p-5 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-white leading-snug">
                      {pair.name || `${pair.label_x || pair.metric_x} vs ${pair.label_y || pair.metric_y}`}
                    </span>
                    <span
                      className={`shrink-0 text-[10px] px-2 py-0.5 rounded-full font-medium ${
                        isSignificant
                          ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                          : "bg-slate-800 text-slate-300"
                      }`}
                    >
                      {isSignificant ? "Strong Link" : "Weak Link"}
                    </span>
                  </div>

                  <div className="flex items-baseline gap-3">
                    <div className="text-2xl font-bold text-indigo-300 font-mono">
                      r = {r.toFixed(2)}
                    </div>
                    <div className="text-xs text-slate-400">
                      across {monthsCount} months
                    </div>
                  </div>

                  {chartSeries.length >= 2 && (
                    <div className="h-28 w-full pt-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartSeries}>
                          <XAxis dataKey="month" hide />
                          <YAxis hide />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "0.5rem",
                              fontSize: "11px",
                            }}
                          />
                          <Line type="monotone" dataKey="x" stroke="#818cf8" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="y" stroke="#38bdf8" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 3. Plain-English Key Insights & Recommendations */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span>Key Insights &amp; Recommendations</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Automated takeaways highlighting strengths, bottlenecks, and suggested improvements
            </p>
          </div>

          {rawRules.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 text-xs">
              {[
                { id: "all", label: `All (${rawRules.length})` },
                { id: "warnings", label: `Attention (${warningRules.length})` },
                { id: "positive", label: `Strengths (${positiveRules.length})` },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setFilterSeverity(tab.id)}
                  className={`px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                    filterSeverity === tab.id
                      ? "bg-indigo-600 text-white font-semibold"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {visibleRules.length === 0 ? (
          <div className="p-8 rounded-2xl border border-slate-800 bg-slate-900/40 text-center text-xs text-slate-400">
            No insights in this category for the current snapshot.
          </div>
        ) : (
          <div className="space-y-3">
            {visibleRules.map((rule, idx) => {
              const isWarning =
                rule.severity === "warning" ||
                rule.severity === "high" ||
                rule.severity === "critical";
              const isPositive = rule.severity === "positive";

              return (
                <div
                  key={`${rule.rule_id}-${idx}`}
                  className={`p-4 rounded-xl border space-y-2 ${
                    isWarning
                      ? "border-amber-500/25 bg-amber-500/5"
                      : isPositive
                      ? "border-emerald-500/25 bg-emerald-500/5"
                      : "border-slate-800 bg-slate-900/50"
                  }`}
                >
                  <div className="flex items-center gap-2 text-xs">
                    {isWarning ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-semibold text-[11px]">
                        <AlertTriangle className="w-3 h-3" />
                        Needs Attention
                      </span>
                    ) : isPositive ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-semibold text-[11px]">
                        <CheckCircle2 className="w-3 h-3" />
                        Strength
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-medium text-[11px]">
                        <Info className="w-3 h-3" />
                        Observation
                      </span>
                    )}
                    <span className="text-slate-400 font-medium">
                      {friendlyDomainName(rule.domain)}
                    </span>
                  </div>

                  <p className="text-sm text-slate-100 leading-relaxed">
                    {cleanFindingText(rule.finding)}
                  </p>

                  {rule.recommendation && (
                    <div className="text-xs text-amber-200/90 bg-slate-950/60 border border-slate-800/80 rounded-lg px-3 py-2">
                      <strong className="text-amber-300">Suggested Action:</strong>{" "}
                      {rule.recommendation}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
