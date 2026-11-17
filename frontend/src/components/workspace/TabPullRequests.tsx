"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertCircle,
  Clock,
  GitMerge,
  Layers,
  Users,
} from "lucide-react";
import { PullRequestIntelligenceData } from "@/lib/types";

interface TabPullRequestsProps {
  pullRequests?: PullRequestIntelligenceData | null;
}

function formatDurationHours(hours?: number | null): string | null {
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

export default function TabPullRequests({ pullRequests }: TabPullRequestsProps) {
  const outcomeDist = pullRequests?.outcome_distribution;
  const legacyOutcomes = pullRequests?.outcomes;

  const total =
    outcomeDist?.total_eligible_prs ??
    pullRequests?.total_prs ??
    0;
  const mergedCount = outcomeDist?.merged_count ?? legacyOutcomes?.merged_count ?? 0;
  const closedUnmergedCount =
    outcomeDist?.closed_unmerged_count ?? legacyOutcomes?.closed_unmerged_count ?? 0;
  const openCount = outcomeDist?.open_count ?? 0;
  const mergeRate = outcomeDist?.merge_rate ?? legacyOutcomes?.merge_rate;
  const abandonmentRate = outcomeDist?.abandonment_rate;

  const mergeLatency = pullRequests?.merge_latency;
  const legacyTurnaround = pullRequests?.turnaround_hours;
  const p50 = formatDurationHours(mergeLatency?.p50_hours ?? legacyTurnaround?.p50);
  const p75 = formatDurationHours(mergeLatency?.p75_hours ?? legacyTurnaround?.p75);
  const p90 = formatDurationHours(mergeLatency?.p90_hours ?? legacyTurnaround?.p90);
  const mean = formatDurationHours(mergeLatency?.mean_hours ?? legacyTurnaround?.mean);

  const resp = pullRequests?.responsiveness;
  const legacyResp = pullRequests?.response_latencies;
  const ttfRespP50 = formatDurationHours(resp?.ttf_resp?.p50_hours ?? legacyResp?.ttfresp?.p50);
  const ttfrFormalP50 = formatDurationHours(resp?.ttfr_formal?.p50_hours ?? legacyResp?.ttfr_formal?.p50);
  const ttfrInlineP50 = formatDurationHours(resp?.ttfr_inline?.p50_hours ?? legacyResp?.ttfr_inline?.p50);

  const diffBins = pullRequests?.size_vs_turnaround?.bins || pullRequests?.diff_size_turnaround || {};
  const reviewBottlenecks = pullRequests?.review_bottlenecks;
  const legacyReviewers = pullRequests?.reviewer_metrics;

  const normalizedTopReviewers: Array<{ login: string; count: number }> = (
    reviewBottlenecks?.maintainer_review_load?.top_reviewers || []
  ).map((r) => ({
    login: r.reviewer,
    count: r.formal_reviews_count ?? r.reviews_count ?? 0,
  }));

  if (normalizedTopReviewers.length === 0 && legacyReviewers?.top_reviewers) {
    legacyReviewers.top_reviewers.forEach((r) => {
      normalizedTopReviewers.push({ login: r.login, count: r.review_count });
    });
  }

  const uniqueReviewersCount =
    reviewBottlenecks?.maintainer_review_load?.active_human_reviewers_count ??
    legacyReviewers?.unique_reviewers_count ??
    normalizedTopReviewers.length;

  // Outcome Chart Data
  const outcomeData = [
    { name: "Merged", value: mergedCount, color: "#8b5cf6" },
    { name: "Open", value: openCount, color: "#10b981" },
    { name: "Closed Unmerged", value: closedUnmergedCount, color: "#64748b" },
  ].filter((d) => d.value > 0);

  // Diff Size Chart Data (only real bins)
  const diffSizeData = [
    { name: "Small (<20 lines)", p50: diffBins["small"]?.p50_hours ?? null, count: diffBins["small"]?.count ?? 0 },
    { name: "Medium (20-99)", p50: diffBins["medium"]?.p50_hours ?? null, count: diffBins["medium"]?.count ?? 0 },
    { name: "Large (100-499)", p50: diffBins["large"]?.p50_hours ?? null, count: diffBins["large"]?.count ?? 0 },
    { name: "XL (500+ lines)", p50: diffBins["xl"]?.p50_hours ?? null, count: diffBins["xl"]?.count ?? 0 },
  ].filter((d) => d.count > 0 && d.p50 != null);

  const isExternalWorkflow =
    abandonmentRate != null && abandonmentRate >= 0.9 && closedUnmergedCount >= 10;

  // Build ONLY non-null speed cards (Zero N/A!)
  const mergeCards = [
    p50 ? { label: "Typical Merge Time", value: p50, sub: "Half of PRs merge faster" } : null,
    p75 ? { label: "75% Merged Within", value: p75, sub: "Most reviews complete here" } : null,
    p90 ? { label: "90% Merged Within", value: p90, sub: "Longer reviews" } : null,
    mean ? { label: "Average Merge Time", value: mean, sub: `Across ${mergedCount} merged` } : null,
  ].filter(Boolean) as Array<{ label: string; value: string; sub: string }>;

  const responseCards = [
    ttfRespP50 ? { label: "First Human Reply", value: ttfRespP50, color: "text-emerald-400" } : null,
    ttfrFormalP50 ? { label: "Formal Approval / Review", value: ttfrFormalP50, color: "text-indigo-300" } : null,
    ttfrInlineP50 ? { label: "Inline Code Feedback", value: ttfrInlineP50, color: "text-cyan-400" } : null,
  ].filter(Boolean) as Array<{ label: string; value: string; color: string }>;

  const unreviewedOpenCount =
    reviewBottlenecks?.unreviewed_open_prs?.unreviewed_open_prs_count ??
    reviewBottlenecks?.unreviewed_open_prs?.count;
  const unreviewedMedianDays = reviewBottlenecks?.unreviewed_open_prs?.median_age_days;

  return (
    <div className="space-y-8">
      {/* Contextual Note for Mailing-List / External Patch Workflows */}
      {isExternalWorkflow && (
        <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-100/90 leading-relaxed">
            <strong className="text-amber-300 font-semibold">External Patch Workflow Detected:</strong>{" "}
            {(abandonmentRate * 100).toFixed(0)}% of closed PRs were closed without using GitHub&apos;s merge button. Projects like Git review and apply patches via developer mailing lists rather than direct GitHub merges.
          </div>
        </div>
      )}

      {/* 1. Review Speed & PR Outcomes */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Review Speed & Queue Summary (2 cols, ZERO N/A) */}
        <div className="lg:col-span-2 space-y-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                <span>Pull Request Review Speed &amp; Activity</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                How fast maintainers respond to pull requests and current queue activity
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
              {total.toLocaleString()} PRs Tracked
            </span>
          </div>

          {/* Primary KPI Grid: show Merge Times if available, otherwise show Real Activity & Queue KPIs */}
          {mergeCards.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              {mergeCards.map((c, idx) => (
                <div key={idx} className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400">{c.label}</div>
                  <div className="text-xl font-bold text-white mt-1">{c.value}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{c.sub}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">Tracked PRs</div>
                <div className="text-xl font-bold text-white mt-1">{total.toLocaleString()}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">In current snapshot</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">Currently Open</div>
                <div className="text-xl font-bold text-emerald-400 mt-1">{openCount.toLocaleString()}</div>
                <div className="text-[10px] text-slate-500 mt-0.5">Active pull requests</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                <div className="text-[11px] text-slate-400">Closed PRs</div>
                <div className="text-xl font-bold text-slate-200 mt-1">
                  {(mergedCount + closedUnmergedCount).toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">{mergedCount} merged on GitHub</div>
              </div>
              {unreviewedOpenCount != null ? (
                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400">Awaiting Review</div>
                  <div className="text-xl font-bold text-amber-300 mt-1">
                    {unreviewedOpenCount.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">
                    {unreviewedMedianDays != null
                      ? `Median age ${Math.round(unreviewedMedianDays)}d`
                      : "Open without feedback"}
                  </div>
                </div>
              ) : (
                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
                  <div className="text-[11px] text-slate-400">Active Reviewers</div>
                  <div className="text-xl font-bold text-purple-300 mt-1">{uniqueReviewersCount}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Human reviewers</div>
                </div>
              )}
            </div>
          )}

          {/* Response Speed Section (only show available response metrics) */}
          {responseCards.length > 0 && (
            <div className="pt-4 border-t border-slate-800/80 space-y-2.5">
              <div className="text-xs font-semibold text-slate-300">
                Typical Time to First Maintainer Response:
              </div>
              <div
                className={`grid grid-cols-1 ${
                  responseCards.length === 1
                    ? "sm:grid-cols-1"
                    : responseCards.length === 2
                    ? "sm:grid-cols-2"
                    : "sm:grid-cols-3"
                } gap-3 text-xs`}
              >
                {responseCards.map((rc, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between"
                  >
                    <span className="text-slate-300">{rc.label}</span>
                    <span className={`font-bold font-mono text-sm ${rc.color}`}>
                      {rc.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Outcome Donut (1 col) */}
        <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <GitMerge className="w-4 h-4 text-purple-400" />
              <span>Pull Request Outcomes</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Breakdown of {total.toLocaleString()} observed PRs
            </p>

            {outcomeData.length === 0 ? (
              <div className="h-44 flex items-center justify-center text-slate-400 text-xs">
                No pull requests recorded yet.
              </div>
            ) : (
              <div className="h-44 w-full flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={outcomeData}
                      cx="50%"
                      cy="50%"
                      innerRadius={45}
                      outerRadius={65}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {outcomeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        borderRadius: "0.5rem",
                        color: "#fff",
                        fontSize: "12px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-xs pt-2 border-t border-slate-800">
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-purple-400">Merged</div>
              <div className="text-sm font-bold text-white">{mergedCount}</div>
              {mergeRate != null && (
                <div className="text-[10px] text-slate-400">{(mergeRate * 100).toFixed(1)}%</div>
              )}
            </div>
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-emerald-400">Open</div>
              <div className="text-sm font-bold text-white">{openCount}</div>
              {total > 0 && (
                <div className="text-[10px] text-slate-400">{((openCount / total) * 100).toFixed(0)}%</div>
              )}
            </div>
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-slate-400">Closed</div>
              <div className="text-sm font-bold text-white">{closedUnmergedCount}</div>
              {abandonmentRate != null && (
                <div className="text-[10px] text-slate-400">{(abandonmentRate * 100).toFixed(0)}%</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Merge Speed by PR Size (ONLY shown when data actually exists!) */}
      {diffSizeData.length > 0 && (
        <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              <span>Merge Speed by Pull Request Size</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Compares typical merge time for small focused changes versus large pull requests
            </p>
          </div>

          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={diffSizeData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="h" tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "0.75rem",
                    color: "#f8fafc",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="p50" name="Typical Merge Time (Hours)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* 3. Top Code Reviewers (ONLY shown when reviewers exist, with fixed review count!) */}
      {normalizedTopReviewers.length > 0 && (
        <section className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Users className="w-4 h-4 text-purple-400" />
              <span>Most Active Code Reviewers</span>
            </h3>
            <span className="text-xs text-slate-400">
              Active Reviewers:{" "}
              <strong className="text-white">{uniqueReviewersCount}</strong>
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            {normalizedTopReviewers.slice(0, 8).map((r, i) => (
              <div key={i} className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 text-xs">
                <div className="text-slate-200 font-medium truncate">
                  @{r.login || "reviewer"}
                </div>
                <div className="text-base font-bold text-purple-300 mt-1">
                  {r.count}{" "}
                  <span className="text-xs font-normal text-slate-400">
                    {r.count === 1 ? "review" : "reviews"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
