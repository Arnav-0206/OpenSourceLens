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

export default function TabPullRequests({ pullRequests }: TabPullRequestsProps) {
  const total = pullRequests?.total_prs ?? 0;
  const turnaround = pullRequests?.turnaround_hours;
  const responseLat = pullRequests?.response_latencies;
  const outcomes = pullRequests?.outcomes;
  const diffBins = pullRequests?.diff_size_turnaround || {};
  const reviewers = pullRequests?.reviewer_metrics;

  // Outcome Chart Data
  const outcomeData = [
    { name: "Merged", value: outcomes?.merged_count ?? 0, color: "#8b5cf6" },
    { name: "Closed / Abandoned", value: outcomes?.closed_unmerged_count ?? 0, color: "#64748b" },
  ];

  // Diff Size Chart Data
  const diffSizeData = [
    { name: "Small (<20 lines)", p50: diffBins["small"]?.p50_hours ?? null, count: diffBins["small"]?.count ?? 0 },
    { name: "Medium (20-99)", p50: diffBins["medium"]?.p50_hours ?? null, count: diffBins["medium"]?.count ?? 0 },
    { name: "Large (100-499)", p50: diffBins["large"]?.p50_hours ?? null, count: diffBins["large"]?.count ?? 0 },
    { name: "XL (500+ lines)", p50: diffBins["xl"]?.p50_hours ?? null, count: diffBins["xl"]?.count ?? 0 },
  ].filter((d) => d.count > 0 || d.p50 != null);

  return (
    <div className="space-y-8">
      {/* 1. Response Latencies & Turnaround Quantiles */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Turnaround Quantiles (2 cols) */}
        <div className="lg:col-span-2 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                <span>Pull Request Turnaround Latency Quantiles</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Time elapsed from PR creation to merge or closure across percentile bands
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
              Sampled: {total} PRs
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-center">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-mono">p50 (Median)</div>
              <div className="text-xl font-bold text-white mt-1">
                {turnaround?.p50 != null ? `${turnaround.p50.toFixed(1)}h` : "N/A"}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-mono">p75</div>
              <div className="text-xl font-bold text-white mt-1">
                {turnaround?.p75 != null ? `${turnaround.p75.toFixed(1)}h` : "N/A"}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-mono">p90</div>
              <div className="text-xl font-bold text-white mt-1">
                {turnaround?.p90 != null ? `${turnaround.p90.toFixed(1)}h` : "N/A"}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400 font-mono">Mean</div>
              <div className="text-xl font-bold text-white mt-1">
                {turnaround?.mean != null ? `${turnaround.mean.toFixed(1)}h` : "N/A"}
              </div>
            </div>
          </div>

          {/* TTFR Comparison */}
          <div className="pt-4 border-t border-slate-800/80 space-y-2">
            <div className="text-xs font-semibold text-slate-300">
              Time to First Response (TTFR) Comparison:
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400">Any Response (TTFResp):</span>
                <span className="font-bold text-white font-mono">
                  {responseLat?.ttfresp?.p50 != null ? `${responseLat.ttfresp.p50.toFixed(1)}h` : "N/A"}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400">Formal Review (TTFR_formal):</span>
                <span className="font-bold text-white font-mono">
                  {responseLat?.ttfr_formal?.p50 != null ? `${responseLat.ttfr_formal.p50.toFixed(1)}h` : "N/A"}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center justify-between">
                <span className="text-slate-400">Inline Comment (TTFR_inline):</span>
                <span className="font-bold text-white font-mono">
                  {responseLat?.ttfr_inline?.p50 != null ? `${responseLat.ttfr_inline.p50.toFixed(1)}h` : "N/A"}
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
            <p className="text-xs text-slate-400 mt-1">Merge vs Abandonment Rate</p>

            <div className="h-44 w-full flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={outcomeData}
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={65}
                    paddingAngle={4}
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
          </div>

          <div className="grid grid-cols-2 gap-2 text-center text-xs pt-2 border-t border-slate-800">
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-slate-400">Merge Rate</div>
              <div className="text-base font-bold text-purple-400">
                {outcomes?.merge_rate != null ? `${(outcomes.merge_rate * 100).toFixed(0)}%` : "N/A"}
              </div>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/70">
              <div className="text-[10px] text-slate-400">Total Merged</div>
              <div className="text-base font-bold text-white">
                {outcomes?.merged_count ?? 0}
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
            <span>Turnaround Latency by Code Diff Size (Lines Changed)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluates turnaround speed disparities between small focused PRs and large multi-file diffs
          </p>
        </div>

        {diffSizeData.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-xs">
            Detailed PR diff stats not sampled in fast_snapshot mode. Re-run in standard_diagnostic mode for diff sizing.
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
      {reviewers && (
        <section className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Users className="w-4 h-4 text-purple-400" />
              <span>Reviewer Distribution & Workload</span>
            </h3>
            <span className="text-xs text-slate-400">
              Unique Reviewers: <strong className="text-white">{reviewers.unique_reviewers_count ?? 0}</strong>
            </span>
          </div>

          {reviewers.top_reviewers && reviewers.top_reviewers.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              {reviewers.top_reviewers.slice(0, 4).map((r, i) => (
                <div key={i} className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 text-xs">
                  <div className="text-slate-400 font-medium">@{r.login}</div>
                  <div className="text-base font-bold text-white mt-1">
                    {r.review_count} <span className="text-xs font-normal text-slate-500">reviews</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-slate-500 pt-2">
              Reviewer details will populate as formal reviews are harvested.
            </div>
          )}
        </section>
      )}
    </div>
  );
}
