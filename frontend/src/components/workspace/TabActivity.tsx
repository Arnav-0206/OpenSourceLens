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

const DEFAULT_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function formatLatency(hours?: number | null): string {
  if (hours == null || Number.isNaN(hours)) return "N/A";
  if (hours < 1) {
    const mins = Math.round(hours * 60);
    return mins <= 1 ? "< 1m" : `${mins}m`;
  }
  if (hours >= 48) {
    return `${(hours / 24).toFixed(1)}d`;
  }
  return `${hours.toFixed(1)}h`;
}

function formatRegularity(code?: string | null): string {
  if (!code) return "Tracked";
  const map: Record<string, string> = {
    no_releases_recorded: "No GitHub Releases",
    insufficient_releases: "Limited History",
    highly_regular: "Highly Predictable",
    moderately_regular: "Moderately Regular",
    irregular: "Irregular Cadence",
    sporadic: "Sporadic",
  };
  return map[code] || code.replace(/_/g, " ");
}

export default function TabActivity({ activity, releases }: TabActivityProps) {
  const summary = activity?.summary || { total_commits: 0, active_weeks_count: 0, completed_weeks_evaluated: 0 };
  const rawWeekly = activity?.weekly_time_series || activity?.weekly_series || [];
  const rollingSeries = activity?.rolling_statistics?.series || [];
  const rawDaily = activity?.daily_time_series_90d || [];

  const rollingByWeek = new Map(
    rollingSeries.map((r) => [r.week_start, r])
  );

  // Format weekly chart data
  const weeklyChartData = rawWeekly.map((pt) => {
    const roll = rollingByWeek.get(pt.week_start);
    const commits = pt.commit_count ?? pt.count ?? 0;
    const sma4 = roll?.rolling_4w_mean ?? pt.sma_4w;
    const sma12 = pt.sma_12w;
    return {
      label: pt.week_start ? pt.week_start.slice(5) : "",
      commits,
      sma4w: sma4 != null ? Number(sma4.toFixed(1)) : null,
      sma12w: sma12 != null ? Number(sma12.toFixed(1)) : null,
      isAnomaly: Boolean(pt.is_anomaly),
    };
  });

  // When fewer than 4 weeks are captured (typical in Fast Snapshot), use daily commit bars
  const useDailyFallback = weeklyChartData.length < 4 && rawDaily.length > 0;
  let dailyChartData: Array<{ label: string; commits: number; sma4w: null; sma12w: null; isAnomaly: boolean }> = [];
  if (useDailyFallback) {
    const firstActiveIdx = rawDaily.findIndex((d) => (d.commit_count ?? 0) > 0);
    const sliceStart = firstActiveIdx >= 0 ? Math.max(0, firstActiveIdx - 2) : Math.max(0, rawDaily.length - 14);
    dailyChartData = rawDaily.slice(sliceStart).map((d) => ({
      label: d.date.slice(5),
      commits: d.commit_count ?? 0,
      sma4w: null,
      sma12w: null,
      isAnomaly: false,
    }));
  }

  const chartData = useDailyFallback && dailyChartData.length > 0 ? dailyChartData : weeklyChartData;

  // Heatmap matrix: 7 days x 24 hours
  const rhythms = activity?.activity_rhythms;
  const matrix = rhythms?.weekday_hour_matrix || activity?.utc_rhythm?.matrix;
  const weekdayLabels =
    rhythms?.weekday_names && rhythms.weekday_names.length === 7
      ? rhythms.weekday_names.map((d) => d.slice(0, 3))
      : DEFAULT_WEEKDAYS;

  let maxHeat = 1;
  if (matrix && matrix.length > 0) {
    matrix.forEach((row) => {
      row.forEach((v) => {
        if (v > maxHeat) maxHeat = v;
      });
    });
  }

  // Landing Latency
  const landing = activity?.commit_landing_latency;
  const legacyLanding = activity?.landing_latency;
  const landingP50 = landing?.p50_hours ?? legacyLanding?.p50;
  const landingP75 = landing?.p75_hours ?? legacyLanding?.p75;
  const landingP90 = landing?.p90_hours ?? legacyLanding?.p90;

  // Code Churn
  const churn = activity?.code_churn || activity?.churn_metrics;
  const totalAdditions = churn?.total_additions ?? 0;
  const totalDeletions = churn?.total_deletions ?? 0;
  const hasChurn = totalAdditions > 0 || totalDeletions > 0;

  // Releases
  const cadence = releases?.cadence;
  const semver = releases?.semver_adoption;
  const surges = releases?.prerelease_surges;
  const velocityRatio = activity?.velocity_metrics?.velocity_trend_ratio;
  const trendLabel =
    summary.velocity_trend ||
    (velocityRatio != null
      ? velocityRatio >= 1.25
        ? "Accelerating"
        : velocityRatio <= 0.75
        ? "Decelerating"
        : "Steady"
      : "Active");

  return (
    <div className="space-y-8">
      {/* 1. Commit Velocity & Moving Averages Chart */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-indigo-400" />
              <span>
                {useDailyFallback ? "Recent Daily Commit Activity" : "Weekly Commit Velocity & Trend Trajectory"}
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              {useDailyFallback
                ? "Daily commit volume across the captured snapshot window (switch to Standard Diagnostic for multi-month weekly trends)"
                : "Completed-week commit volume paired with 4-week moving average and anomaly detection"}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 text-xs">
            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
              Commits: <strong className="text-white">{(summary.total_commits ?? 0).toLocaleString()}</strong>
            </span>
            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
              Active Weeks: <strong className="text-white">{summary.active_weeks_count ?? 0}</strong>
            </span>
            <span className="px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-300 font-medium border border-indigo-500/20 capitalize">
              Trend: {trendLabel}
            </span>
          </div>
        </div>

        {chartData.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
            No commits captured in this snapshot window.
          </div>
        ) : (
          <div className="h-80 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} tickLine={false} />
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
                <Bar
                  dataKey="commits"
                  name={useDailyFallback ? "Daily Commits" : "Weekly Commits"}
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                  opacity={0.85}
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.isAnomaly ? "#f43f5e" : "#6366f1"} />
                  ))}
                </Bar>
                {!useDailyFallback && (
                  <Line type="monotone" dataKey="sma4w" name="4w Moving Average" stroke="#38bdf8" strokeWidth={2} dot={false} />
                )}
                {!useDailyFallback && (
                  <Line type="monotone" dataKey="sma12w" name="12w Trend" stroke="#a855f7" strokeWidth={2} dot={false} />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {/* 2. 7x24 UTC Heatmap & Landing Latency */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Heatmap (2 cols) */}
        <div className="lg:col-span-2 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span>7&times;24 UTC Commit Rhythm Matrix</span>
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Hover over cells to inspect commit counts by weekday and UTC hour
              </p>
            </div>
            {rhythms?.peak_weekday && rhythms?.peak_utc_hour && (
              <span className="text-[11px] px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-medium">
                Peak: {rhythms.peak_weekday.name}s &middot; {String(rhythms.peak_utc_hour.hour_utc).padStart(2, "0")}:00 UTC
              </span>
            )}
          </div>

          {!matrix || matrix.length === 0 ? (
            <div className="h-44 flex items-center justify-center text-slate-500 text-xs">
              UTC rhythm data requires at least 5 commits.
            </div>
          ) : (
            <div className="space-y-1.5 overflow-x-auto pb-2">
              <div
                className="grid gap-1 min-w-[520px] text-[10px] text-slate-500 font-mono"
                style={{ gridTemplateColumns: "36px repeat(24, minmax(0, 1fr))" }}
              >
                <div></div>
                {Array.from({ length: 24 }).map((_, h) => (
                  <div key={h} className="text-center">
                    {h % 3 === 0 ? `${h}h` : ""}
                  </div>
                ))}
              </div>

              {matrix.map((row, dayIdx) => (
                <div
                  key={dayIdx}
                  className="grid gap-1 min-w-[520px] items-center"
                  style={{ gridTemplateColumns: "36px repeat(24, minmax(0, 1fr))" }}
                >
                  <div className="text-[11px] font-mono text-slate-400">{weekdayLabels[dayIdx]}</div>
                  {row.map((val, hourIdx) => {
                    const intensity = val === 0 ? 0 : Math.max(0.2, val / maxHeat);
                    return (
                      <div
                        key={hourIdx}
                        title={`${weekdayLabels[dayIdx]} ${String(hourIdx).padStart(2, "0")}:00 UTC — ${val} ${val === 1 ? "commit" : "commits"}`}
                        className="h-5 rounded-[3px] transition-transform hover:scale-125 hover:z-10 cursor-pointer"
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

              {rhythms?.weekday_ratio != null && (
                <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400">
                  <span>
                    Weekday share (Mon–Fri UTC):{" "}
                    <strong className="text-slate-200">{(rhythms.weekday_ratio * 100).toFixed(1)}%</strong>
                  </span>
                  <span>
                    Weekend share (Sat–Sun UTC):{" "}
                    <strong className="text-slate-200">{((rhythms.weekend_ratio ?? 0) * 100).toFixed(1)}%</strong>
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Side Card (1 col): Integration Speed if available, otherwise Activity Highlights (Zero N/A) */}
        <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          {landingP50 != null ? (
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                <span>Patch Integration Speed</span>
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Typical time from when code is written to when it lands in the repository
              </p>

              <div className="grid grid-cols-3 gap-2 mt-4 text-center">
                <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Typical</div>
                  <div className="text-base font-bold text-white mt-1">
                    {formatLatency(landingP50)}
                  </div>
                </div>
                {landingP75 != null && (
                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                    <div className="text-[10px] text-slate-400">75% Under</div>
                    <div className="text-base font-bold text-white mt-1">
                      {formatLatency(landingP75)}
                    </div>
                  </div>
                )}
                {landingP90 != null && (
                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                    <div className="text-[10px] text-slate-400">90% Under</div>
                    <div className="text-base font-bold text-white mt-1">
                      {formatLatency(landingP90)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>Activity Highlights</span>
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  When and how often developers push updates
                </p>
              </div>

              <div className="space-y-2.5 pt-1">
                {rhythms?.peak_weekday && (
                  <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                    <span className="text-xs text-slate-400">Most Active Day</span>
                    <span className="text-sm font-bold text-white">
                      {rhythms.peak_weekday.name} ({rhythms.peak_weekday.commit_count})
                    </span>
                  </div>
                )}
                {rhythms?.peak_utc_hour && (
                  <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                    <span className="text-xs text-slate-400">Busiest Hour (UTC)</span>
                    <span className="text-sm font-bold text-cyan-300">
                      {String(rhythms.peak_utc_hour.hour_utc).padStart(2, "0")}:00 UTC
                    </span>
                  </div>
                )}
                {rhythms?.weekday_ratio != null && (
                  <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                    <span className="text-xs text-slate-400">Workweek Pace</span>
                    <span className="text-sm font-bold text-emerald-400">
                      {(rhythms.weekday_ratio * 100).toFixed(0)}% Mon–Fri
                    </span>
                  </div>
                )}
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                  <span className="text-xs text-slate-400">Active Days Tracked</span>
                  <span className="text-sm font-bold text-indigo-300">
                    {summary.active_days_count ?? chartData.filter((d) => d.commits > 0).length} days
                  </span>
                </div>
              </div>
            </div>
          )}

          {hasChurn && (
            <div className="space-y-2 pt-4 border-t border-slate-800/80">
              <div className="text-xs font-semibold text-slate-300">Code Changes Volume:</div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-emerald-400">
                  +{totalAdditions.toLocaleString()} lines
                </div>
                <div className="p-2 rounded-lg bg-rose-950/20 border border-rose-500/20 text-rose-400">
                  -{totalDeletions.toLocaleString()} lines
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. Release Evolution (only rendered when releases exist — no empty N/A card) */}
      {(cadence?.total_releases ?? 0) > 0 && (
        <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                <Package className="w-5 h-5 text-amber-400" />
                <span>Release Schedule &amp; Versioning</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                How regularly new versions are published and how consistently version numbering is followed
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 font-medium border border-amber-500/20">
                {formatRegularity(cadence?.regularity_classification)}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
              <div className="text-xs text-slate-400">Published Releases</div>
              <div className="text-2xl font-bold text-white">{cadence?.total_releases ?? 0}</div>
              {cadence?.latest_release_tag && (
                <div className="text-[11px] text-slate-400">
                  Latest: <strong className="text-slate-200">{cadence.latest_release_tag}</strong>
                </div>
              )}
            </div>

            {semver?.semver_adoption_rate != null && (
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <div className="text-xs text-slate-400">Standard Versioning (SemVer)</div>
                <div className="text-2xl font-bold text-indigo-300">
                  {(semver.semver_adoption_rate * 100).toFixed(0)}%
                </div>
                <div className="text-[11px] text-slate-400">
                  Major: {semver.major_releases ?? 0} &middot; Minor: {semver.minor_releases ?? 0} &middot; Patch: {semver.patch_releases ?? 0}
                </div>
              </div>
            )}

            {cadence?.days_since_last_release != null && (
              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                <div className="text-xs text-slate-400">Time Since Last Release</div>
                <div className="text-2xl font-bold text-white">
                  {cadence.days_since_last_release} days
                </div>
                <div className="text-[11px] text-slate-400">
                  {cadence.days_since_last_release > 180 ? "Quiet release cycle" : "Actively shipping"}
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
