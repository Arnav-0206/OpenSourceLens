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
  AlertCircle,
  Clock,
  Flame,
  HelpCircle,
  Inbox,
  Layers,
  MessageCircle,
  MessageSquare,
  Sparkles,
  TrendingUp,
  Zap,
} from "lucide-react";
import { IssueIntelligenceData } from "@/lib/types";

interface TabIssuesProps {
  issues?: IssueIntelligenceData | null;
}

export default function TabIssues({ issues }: TabIssuesProps) {
  const total = issues?.total_issues ?? 0;
  const flow = issues?.flow;
  const sla = issues?.resolution_sla_hours;
  const aging = issues?.aging;
  const discussion = issues?.discussion_intensity;

  const monthlyFlow = flow?.monthly_flow || [];
  const chartData = monthlyFlow.map((m) => ({
    month: m.month,
    opened: m.opened,
    closed: m.closed,
  }));

  return (
    <div className="space-y-8">
      {/* 1. High-Level Backlog Indicators */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Inbox className="w-4 h-4 text-indigo-400" />
            <span>Total Tracked Issues</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">{total.toLocaleString()}</div>
          <div className="text-[11px] text-slate-500 mt-0.5">Excludes PR envelopes</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span>Resolution SLA (p50)</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">
            {sla?.p50 != null ? `${sla.p50.toFixed(1)}h` : "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            p75: {sla?.p75 != null ? `${sla.p75.toFixed(1)}h` : "N/A"} &middot; p90: {sla?.p90 != null ? `${sla.p90.toFixed(1)}h` : "N/A"}
          </div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <AlertCircle className="w-4 h-4 text-rose-400" />
            <span>Stale Backlog</span>
          </div>
          <div className="text-2xl font-bold text-rose-400 mt-1.5">
            {aging?.open_stale_count ?? 0}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">&gt; 90 days dormant</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <MessageCircle className="w-4 h-4 text-amber-400" />
            <span>Discussion Intensity</span>
          </div>
          <div className="text-2xl font-bold text-amber-300 mt-1.5">
            {discussion?.comments_per_issue_p50 ?? "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Median comments per issue</div>
        </div>
      </div>

      {/* 2. Monthly Issue Flow Chart */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-indigo-400" />
              <span>Monthly Issue Inflow vs. Close Flow</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Net backlog growth rate and maintainer resolution velocity over time
            </p>
          </div>

          {flow && (
            <div className="flex items-center gap-3 text-xs">
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
                Opened: <strong className="text-white">{flow.total_opened}</strong>
              </span>
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
                Closed: <strong className="text-white">{flow.total_closed}</strong>
              </span>
            </div>
          )}
        </div>

        {chartData.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm">
            Insufficient monthly issue records to plot timeline.
          </div>
        ) : (
          <div className="h-72 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="month" stroke="#94a3b8" fontSize={11} tickLine={false} />
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
                <Bar dataKey="opened" name="Issues Opened" fill="#6366f1" radius={[4, 4, 0, 0]} opacity={0.8} />
                <Bar dataKey="closed" name="Issues Closed" fill="#10b981" radius={[4, 4, 0, 0]} opacity={0.8} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {/* 3. Backlog Aging & Discussion Diagnostics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span>Backlog Aging Dynamics</span>
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Tracks accumulation of unresolved issues and dormancy risks across the backlog.
          </p>
          <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Median Age</div>
              <div className="text-lg font-bold text-white mt-1">
                {aging?.median_age_days != null ? `${aging.median_age_days.toFixed(0)} days` : "N/A"}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Stale Ratio</div>
              <div className="text-lg font-bold text-rose-400 mt-1">
                {total > 0 && aging?.open_stale_count != null
                  ? `${((aging.open_stale_count / total) * 100).toFixed(0)}%`
                  : "N/A"}
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-indigo-400" />
            <span>Community Collaboration Intensity</span>
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Analyzes discussion density and response engagement on bug reports and feature requests.
          </p>
          <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Mean Comments</div>
              <div className="text-lg font-bold text-white mt-1">
                {discussion?.comments_per_issue_mean != null ? discussion.comments_per_issue_mean.toFixed(1) : "N/A"}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Median Comments</div>
              <div className="text-lg font-bold text-white mt-1">
                {discussion?.comments_per_issue_p50 != null ? discussion.comments_per_issue_p50.toFixed(1) : "N/A"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
