"use client";

import React from "react";
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertCircle,
  Calendar,
  CheckCircle2,
  Clock,
  GitCommit,
  Layers,
  Package,
  Sparkles,
  TrendingUp,
  Zap,
} from "lucide-react";
import { DevelopmentActivityData, ReleaseEvolutionData } from "@/lib/types";

interface TabActivityProps {
  activity?: DevelopmentActivityData | null;
  releases?: ReleaseEvolutionData | null;
}

const DAYS_OF_WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export default function TabActivity({ activity, releases }: TabActivityProps) {
  const summary = activity?.summary || { total_commits: 0, active_weeks_count: 0, completed_weeks_evaluated: 0 };
  const weeklySeries = activity?.weekly_series || [];
  const landing = activity?.landing_latency;
  const churn = activity?.churn_metrics;
  const cadence = releases?.cadence;
  const semver = releases?.semver_adoption;
  const surges = releases?.prerelease_surges;

  // Format weekly chart data
  const chartData = weeklySeries.map((pt) => ({
    week: pt.week_start ? pt.week_start.slice(5) : "",
    commits: pt.count,
    sma4w: pt.sma_4w != null ? Number(pt.sma_4w.toFixed(1)) : null,
    sma12w: pt.sma_12w != null ? Number(pt.sma_12w.toFixed(1)) : null,
    isAnomaly: pt.is_anomaly,
  }));

  // Heatmap matrix: 7 days x 24 hours
  const matrix = activity?.utc_rhythm?.matrix;
  let maxHeat = 1;
  if (matrix && matrix.length > 0) {
    matrix.forEach((row) => {
      row.forEach((v) => {
        if (v > maxHeat) maxHeat = v;
      });
    });
  }

  return (
    <div className="space-y-8">
      {/* 1. Commit Velocity & Moving Averages Chart */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-indigo-400" />
              <span>Weekly Commit Velocity & Trend Trajectory</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Completed-week volume paired with 4-week & 12-week smoothing and anomaly detection
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
              Total Commits: <strong className="text-white">{summary.total_commits.toLocaleString()}</strong>
            </span>
            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
              Active Weeks: <strong className="text-white">{summary.active_weeks_count}</strong>
            </span>
            <span className="px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-300 font-mono border border-indigo-500/20 capitalize">
              Trend: {summary.velocity_trend || "Stable"}
            </span>
          </div>
        </div>

        {chartData.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm">
            Insufficient commit history to construct weekly time series.
          </div>
        ) : (
          <div className="h-80 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="week" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "0.75rem",
                    color: "#f8fafc",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                <Bar dataKey="commits" name="Weekly Commits" fill="#6366f1" radius={[4, 4, 0, 0]} opacity={0.85}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.isAnomaly ? "#f43f5e" : "#6366f1"} />
                  ))}
                </Bar>
                <Line type="monotone" dataKey="sma4w" name="4w Moving Average" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="sma12w" name="12w Trend" stroke="#a855f7" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {/* 2. 7x24 UTC Heatmap & Landing Latency */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Heatmap (2 cols) */}
        <div className="lg:col-span-2 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Clock className="w-4 h-4 text-cyan-400" />
              <span>7&times;24 UTC Development Rhythm Matrix</span>
            </h3>
            <span className="text-[11px] text-slate-400">Peak hour concentration</span>
          </div>

          {!matrix || matrix.length === 0 ? (
            <div className="h-44 flex items-center justify-center text-slate-500 text-xs">
              UTC rhythm data unavailable.
            </div>
          ) : (
            <div className="space-y-1.5 overflow-x-auto pb-2">
              <div className="grid grid-cols-25 gap-1 min-w-[500px] text-[10px] text-slate-500 font-mono">
                <div className="w-8"></div>
                {Array.from({ length: 24 }).map((_, h) => (
                  <div key={h} className="text-center">
                    {h % 3 === 0 ? `${h}h` : ""}
                  </div>
                ))}
              </div>

              {matrix.map((row, dayIdx) => (
                <div key={dayIdx} className="grid grid-cols-25 gap-1 min-w-[500px] items-center">
                  <div className="w-8 text-[11px] font-mono text-slate-400">{DAYS_OF_WEEK[dayIdx]}</div>
                  {row.map((val, hourIdx) => {
                    const intensity = val === 0 ? 0 : Math.max(0.15, val / maxHeat);
                    return (
                      <div
                        key={hourIdx}
                        title={`${DAYS_OF_WEEK[dayIdx]} ${hourIdx}:00 UTC: ${val} commits`}
                        className="h-4.5 rounded-xs transition-transform hover:scale-125 hover:z-10 cursor-pointer"
                        style={{
                          backgroundColor:
                            val === 0
                              ? "#1e293b"
                              : `rgba(99, 102, 241, ${intensity})`,
                        }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Latencies & Churn Card (1 col) */}
        <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              <span>Landing Latency & Churn</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">Author-to-committer delay</p>

            <div className="grid grid-cols-3 gap-2 mt-4 text-center">
              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-mono">p50 Latency</div>
                <div className="text-base font-bold text-white mt-1">
                  {landing?.p50 != null ? `${landing.p50.toFixed(1)}h` : "N/A"}
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-mono">p75</div>
                <div className="text-base font-bold text-white mt-1">
                  {landing?.p75 != null ? `${landing.p75.toFixed(1)}h` : "N/A"}
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                <div className="text-[10px] text-slate-400 font-mono">p90</div>
                <div className="text-base font-bold text-white mt-1">
                  {landing?.p90 != null ? `${landing.p90.toFixed(1)}h` : "N/A"}
                </div>
              </div>
            </div>
          </div>

          {churn && (
            <div className="space-y-2 pt-4 border-t border-slate-800/80">
              <div className="text-xs font-semibold text-slate-300">Code Churn Volume:</div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-emerald-400">
                  +{(churn.total_additions || 0).toLocaleString()} lines
                </div>
                <div className="p-2 rounded-lg bg-rose-950/20 border border-rose-500/20 text-rose-400">
                  -{(churn.total_deletions || 0).toLocaleString()} lines
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. Release Evolution & SemVer Adoption */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Package className="w-5 h-5 text-amber-400" />
              <span>Release Cadence & SemVer 2.0.0 Evolution</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Cadence regularity, versioning discipline, and pre-release commit surge detection
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
              CV_release: {cadence?.cv_release != null ? cadence.cv_release.toFixed(2) : "N/A"}
            </span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 font-mono border border-amber-500/20 capitalize">
              {cadence?.regularity_classification || "Recorded"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
            <div className="text-xs text-slate-400">Total Tracked Releases</div>
            <div className="text-2xl font-bold text-white">{cadence?.total_releases ?? 0}</div>
            <div className="text-[11px] text-slate-500">
              Latest Tag: <strong className="text-slate-300">{cadence?.latest_release_tag || "N/A"}</strong>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
            <div className="text-xs text-slate-400">SemVer Adoption Rate</div>
            <div className="text-2xl font-bold text-indigo-300">
              {semver?.semver_adoption_rate != null ? `${(semver.semver_adoption_rate * 100).toFixed(0)}%` : "N/A"}
            </div>
            <div className="text-[11px] text-slate-500">
              Major: {semver?.major_releases ?? 0} &middot; Minor: {semver?.minor_releases ?? 0} &middot; Patch: {semver?.patch_releases ?? 0}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
            <div className="text-xs text-slate-400">Days Since Last Release</div>
            <div className="text-2xl font-bold text-white">
              {cadence?.days_since_last_release != null ? `${cadence.days_since_last_release}d` : "N/A"}
            </div>
            <div className="text-[11px] text-slate-500">
              {cadence?.days_since_last_release && cadence.days_since_last_release > 180 ? "Dormant cadence" : "Active cycle"}
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
            <div className="text-xs text-slate-400">Pre-Release Commit Surges</div>
            <div className="text-2xl font-bold text-white">
              {surges?.surge_detected ? (
                <span className="text-amber-400">Detected</span>
              ) : (
                <span className="text-slate-400">Normal</span>
              )}
            </div>
            <div className="text-[11px] text-slate-500">
              Surge Ratio: {surges?.surge_ratio != null ? `${surges.surge_ratio.toFixed(2)}x` : "N/A"}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
