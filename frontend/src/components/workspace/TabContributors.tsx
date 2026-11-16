"use client";

import React, { useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  Award,
  Bot,
  CheckCircle2,
  GitCommit,
  GitPullRequest,
  HelpCircle,
  Layers,
  Scale,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  UserCheck,
  Users,
  Zap,
} from "lucide-react";
import { ContributorEcosystemData, ContributorLeaderboardItem } from "@/lib/types";

interface TabContributorsProps {
  contributors?: ContributorEcosystemData | null;
}

export default function TabContributors({ contributors }: TabContributorsProps) {
  const [roleFilter, setRoleFilter] = useState<string>("all");

  const total = contributors?.total_contributors ?? 0;
  const gini = contributors?.gini_coefficient;
  const lorenzData = (contributors?.lorenz_curve || []).map((pt) => ({
    x: Number((pt.contributor_percentile * 100).toFixed(1)),
    observed: Number((pt.contribution_percentile * 100).toFixed(1)),
    equality: Number((pt.contributor_percentile * 100).toFixed(1)),
  }));

  const busFactors = contributors?.bus_factors;
  const cohorts = contributors?.cohorts;
  const rawLeaderboard = contributors?.leaderboard || [];

  const filteredLeaderboard = roleFilter === "all"
    ? rawLeaderboard
    : rawLeaderboard.filter((c) => {
        if (roleFilter === "core") return c.is_core;
        if (roleFilter === "bot") return c.is_bot;
        if (roleFilter === "newcomer") return c.is_newcomer;
        return c.primary_role === roleFilter;
      });

  return (
    <div className="space-y-8">
      {/* 1. Ecosystem High-Level Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Users className="w-4 h-4 text-cyan-400" />
            <span>Total Contributors</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">{total.toLocaleString()}</div>
          <div className="text-[11px] text-slate-500 mt-0.5">Identity reconciled</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Scale className="w-4 h-4 text-indigo-400" />
            <span>Gini Inequality</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">
            {gini != null ? gini.toFixed(3) : "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            {gini != null && gini > 0.80 ? "Severe concentration" : (gini != null && gini > 0.60 ? "Moderate inequality" : "Even distribution")}
          </div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            <span>Bus Factor (BF50)</span>
          </div>
          <div className="text-2xl font-bold text-amber-300 mt-1.5">
            {busFactors?.bf50 ?? "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">Produce 50% of total activity</div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <UserCheck className="w-4 h-4 text-emerald-400" />
            <span>Core Contributors</span>
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-1.5">
            {cohorts?.core_contributors ?? 0}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            Newcomers: {cohorts?.newcomers_30d ?? 0} &middot; Bots: {cohorts?.bot_count ?? 0}
          </div>
        </div>
      </div>

      {/* 2. Lorenz Curve Chart & Concentration Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Scale className="w-4 h-4 text-indigo-400" />
                <span>Lorenz Inequality Curve</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Cumulative contributor deciles plotted against cumulative output shares
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
              Gini = {gini != null ? gini.toFixed(3) : "N/A"}
            </span>
          </div>

          {lorenzData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-xs">
              Lorenz curve requires at least 2 active contributors.
            </div>
          ) : (
            <div className="h-72 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lorenzData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                  <XAxis
                    dataKey="x"
                    stroke="#94a3b8"
                    fontSize={11}
                    unit="%"
                    tickLine={false}
                    label={{ value: "Cumulative Contributors %", position: "insideBottom", offset: -5, fontSize: 11, fill: "#64748b" }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    fontSize={11}
                    unit="%"
                    domain={[0, 100]}
                    tickLine={false}
                  />
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
                  <Line type="monotone" dataKey="equality" name="Line of Perfect Equality" stroke="#64748b" strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="observed" name="Observed Lorenz Curve" stroke="#6366f1" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Bus Factor Analysis Card */}
        <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-400" />
              <span>Bus Factor Concentration Risk</span>
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Measures knowledge concentration and organizational fragility by counting the minimum maintainers required to cover critical percentages of repository output.
            </p>

            <div className="space-y-2.5 pt-2">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-white">BF50 Threshold</div>
                  <div className="text-[11px] text-slate-400">Covers 50% of volume</div>
                </div>
                <div className="text-xl font-bold text-amber-400 font-mono">
                  {busFactors?.bf50 ?? "N/A"}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-white">BF80 Threshold</div>
                  <div className="text-[11px] text-slate-400">Covers 80% of volume</div>
                </div>
                <div className="text-xl font-bold text-indigo-400 font-mono">
                  {busFactors?.bf80 ?? "N/A"}
                </div>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs leading-relaxed">
            <strong>Bot Isolation:</strong> Automated bots and CI service accounts are segregated from human inequality statistics.
          </div>
        </div>
      </div>

      {/* 3. Contributor Breakdown Table */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Users className="w-4 h-4 text-cyan-400" />
              <span>Contributor Directory & Multi-Role Breakdown</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Identity-clean contributor counts across commits, reviews, and PRs
            </p>
          </div>

          {/* Role Filters */}
          <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            {["all", "core", "newcomer", "bot"].map((f) => (
              <button
                key={f}
                onClick={() => setRoleFilter(f)}
                className={`px-3 py-1 rounded-lg capitalize transition-colors cursor-pointer ${
                  roleFilter === f ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-white"
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {filteredLeaderboard.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-xs">
            No contributor records available for this filter.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800 font-mono">
                <tr>
                  <th className="py-2.5 px-3">Contributor</th>
                  <th className="py-2.5 px-3">Primary Role</th>
                  <th className="py-2.5 px-3 text-right">Commits</th>
                  <th className="py-2.5 px-3 text-right">PRs Authored</th>
                  <th className="py-2.5 px-3 text-right">PRs Reviewed</th>
                  <th className="py-2.5 px-3 text-right">Tenure</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredLeaderboard.slice(0, 25).map((item, i) => (
                  <tr key={i} className="hover:bg-slate-850/40">
                    <td className="py-2.5 px-3 flex items-center gap-2 font-medium text-white">
                      {item.is_bot ? (
                        <Bot className="w-4 h-4 text-slate-400" />
                      ) : (
                        <div className="w-5 h-5 rounded-full bg-indigo-600/40 border border-indigo-500/50 flex items-center justify-center text-[10px] font-bold text-indigo-300">
                          {item.login.slice(0, 2).toUpperCase()}
                        </div>
                      )}
                      <span>{item.login}</span>
                      {item.is_core && (
                        <span className="px-1.5 py-0.2 rounded text-[9px] bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-bold uppercase">
                          Core
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 capitalize text-slate-400 font-mono">{item.primary_role}</td>
                    <td className="py-2.5 px-3 text-right font-mono font-bold text-white">{item.total_commits.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{item.total_prs_authored ?? 0}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{item.total_prs_reviewed ?? 0}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                      {item.tenure_days != null ? `${item.tenure_days}d` : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
