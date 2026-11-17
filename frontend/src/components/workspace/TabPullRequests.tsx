"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  GitMerge,
  GitPullRequest,
  HelpCircle,
  Layers,
  Scale,
  Sparkles,
  Users,
  Zap,
} from "lucide-react";
import { PullRequestIntelligenceData } from "@/lib/types";

interface TabPullRequestsProps {
  pullRequests?: PullRequestIntelligenceData | null;
}

function formatDurationHours(hours?: number | null): string {
  if (hours == null || Number.isNaN(hours)) return "N/A";
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
  const p50 = mergeLatency?.p50_hours ?? legacyTurnaround?.p50;
  const p75 = mergeLatency?.p75_hours ?? legacyTurnaround?.p75;
  const p90 = mergeLatency?.p90_hours ?? legacyTurnaround?.p90;
  const mean = mergeLatency?.mean_hours ?? legacyTurnaround?.mean;

  const resp = pullRequests?.responsiveness;
  const legacyResp = pullRequests?.response_latencies;
  const ttfRespP50 = resp?.ttf_resp?.p50_hours ?? legacyResp?.ttfresp?.p50;
  const ttfrFormalP50 = resp?.ttfr_formal?.p50_hours ?? legacyResp?.ttfr_formal?.p50;
  const ttfrInlineP50 = resp?.ttfr_inline?.p50_hours ?? legacyResp?.ttfr_inline?.p50;

  const diffBins = pullRequests?.size_vs_turnaround?.bins || pullRequests?.diff_size_turnaround || {};
  const reviewBottlenecks = pullRequests?.review_bottlenecks;
  const legacyReviewers = pullRequests?.reviewer_metrics;

  const normalizedTopReviewers: Array<{ login: string; count: number }> = (
    reviewBottlenecks?.maintainer_review_load?.top_reviewers || []
  ).map((r) => ({
    login: r.reviewer,
    count: r.reviews_count,
  }));

  if (normalizedTopReviewers.length === 0 && legacyReviewers?.top_reviewers) {
    legacyReviewers.top_reviewers.forEach((r) => {
      normalizedTopReviewers.push({ login: r.login, count: r.review_count });
    });
  }

  const uniqueReviewersCount =
    legacyReviewers?.unique_reviewers_count ?? normalizedTopReviewers.length;
  const hasReviewerSection = Boolean(reviewBottlenecks || legacyReviewers);

  // Outcome Chart Data
  const outcomeData = [
    { name: "Merged", value: mergedCount, color: "#8b5cf6" },
    { name: "Open", value: openCount, color: "#10b981" },
    { name: "Closed Unmerged", value: closedUnmergedCount, color: "#64748b" },
  ].filter((d) => d.value > 0);

  // Diff Size Chart Data
  const diffSizeData = [
    { name: "Small (<20 lines)", p50: diffBins["small"]?.p50_hours ?? null, count: diffBins["small"]?.count ?? 0 },
    { name: "Medium (20-99)", p50: diffBins["medium"]?.p50_hours ?? null, count: diffBins["medium"]?.count ?? 0 },
    { name: "Large (100-499)", p50: diffBins["large"]?.p50_hours ?? null, count: diffBins["large"]?.count ?? 0 },
    { name: "XL (500+ lines)", p50: diffBins["xl"]?.p50_hours ?? null, count: diffBins["xl"]?.count ?? 0 },
  ].filter((d) => d.count > 0 && d.p50 != null);

  const isExternalWorkflow = abandonmentRate != null && abandonmentRate >= 0.9 && closedUnmergedCount >= 10;

  return (
    <div className="space-y-8">
      {/* Contextual Banner for External / Mailing-List PR Workflows */}
      {isExternalWorkflow && (
        <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-200/90 leading-relaxed">
            <strong className="text-amber-300 font-semibold">External Patch / Mailing-List Workflow Detected:</strong>{" "}
            {(abandonmentRate * 100).toFixed(1)}% of closed GitHub Pull Requests ({closedUnmergedCount} of{" "}
            {closedUnmergedCount + mergedCount}) were closed without merging via GitHub. For repositories like{" "}
            <code className="text-amber-100">git/git</code>, maintainers review and apply patches via external mailing lists or git-send-email rather than GitHub&apos;s merge button.
          </div>
        </div>
      )}

      {/* 1. Response Latencies & Turnaround Quantiles */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Turnaround Quantiles (2 cols) */}
        <div className="lg:col-span-2 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                <span>Pull Request Review & Merge Speed</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                How quickly pull requests receive their first response and get merged
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
              Tracked: {total.toLocaleString()} PRs
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-center">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">Median Merge Time</div>
              <div className="text-lg font-bold text-white mt-1">
                {formatDurationHours(p50)}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">50th percentile</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">75th Percentile</div>
              <div className="text-lg font-bold text-white mt-1">
                {formatDurationHours(p75)}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">Typical upper bound</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">90th Percentile</div>
              <div className="text-lg font-bold text-white mt-1">
                {formatDurationHours(p90)}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">Slowest 10% of PRs</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">Average Merge Time</div>
              <div className="text-lg font-bold text-white mt-1">
                {formatDurationHours(mean)}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                {mergedCount < 3 ? "Needs ≥3 merged PRs" : `Based on ${mergedCount} merged`}
              </div>
            </div>
          </div>

          {/* TTFR Comparison */}
          <div className="pt-4 border-t border-slate-800/80 space-y-2.5">
            <div className="text-xs font-semibold text-slate-300">
              Time to First Human Response (Median):
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400">First Feedback (Any)</span>
                <span className="font-bold text-emerald-400 font-mono">
                  {formatDurationHours(ttfRespP50)}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400">Formal Review</span>
                <span className="font-bold text-white font-mono">
                  {formatDurationHours(ttfrFormalP50)}
                </span>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400">Inline Code Comment</span>
                <span className="font-bold text-cyan-400 font-mono">
                  {formatDurationHours(ttfrInlineP50)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Outcome Donut (1 col) */}
        <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <GitMerge className="w-4 h-4 text-purple-400" />
              <span>Pull Request Outcomes</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Status breakdown across {total.toLocaleString()} observed PRs
            </p>

            {outcomeData.length === 0 ? (
              <div className="h-44 flex items-center justify-center text-slate-500 text-xs">
                No pull requests observed in this snapshot.
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
              <div className="text-[10px] text-slate-500">
                {mergeRate != null ? `${(mergeRate * 100).toFixed(1)}%` : "—"}
              </div>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-emerald-400">Open</div>
              <div className="text-sm font-bold text-white">{openCount}</div>
              <div className="text-[10px] text-slate-500">
                {total > 0 ? `${((openCount / total) * 100).toFixed(0)}%` : "—"}
              </div>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-slate-400">Unmerged</div>
              <div className="text-sm font-bold text-white">{closedUnmergedCount}</div>
              <div className="text-[10px] text-slate-500">
                {abandonmentRate != null ? `${(abandonmentRate * 100).toFixed(0)}%` : "—"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Turnaround by Diff Size Bins */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div>
          <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>Merge Speed by PR Size (Lines Changed)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Compares median merge time for small, focused PRs versus large multi-file changes
          </p>
        </div>

        {diffSizeData.length === 0 ? (
          <div className="p-8 text-center text-slate-400 text-xs bg-slate-950/40 rounded-xl border border-slate-800/70">
            Detailed diff-size turnaround requires merged PRs with sampled diff statistics. Run{" "}
            <strong className="text-slate-200">Standard Diagnostic</strong> mode on repositories that merge via GitHub to populate size-band comparisons.
          </div>
        ) : (
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
                <Bar dataKey="p50" name="Median Turnaround (Hours)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {/* 3. Reviewer Concentration & Bottlenecks */}
      {hasReviewerSection && (
        <section className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Users className="w-4 h-4 text-purple-400" />
              <span>Code Reviewer Distribution</span>
            </h3>
            <span className="text-xs text-slate-400">
              Active Reviewers:{" "}
              <strong className="text-white">{uniqueReviewersCount}</strong>
            </span>
          </div>

          {normalizedTopReviewers.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              {normalizedTopReviewers.slice(0, 8).map((r, i) => (
                <div key={i} className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 text-xs">
                  <div className="text-slate-300 font-medium truncate">
                    @{r.login || "reviewer"}
                  </div>
                  <div className="text-base font-bold text-white mt-1">
                    {r.count}{" "}
                    <span className="text-xs font-normal text-slate-500">reviews</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-slate-400 pt-2">
              No formal GitHub code review events recorded in this snapshot.
            </div>
          )}
        </section>
      )}
    </div>
  );
}
