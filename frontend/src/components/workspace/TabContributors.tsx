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

  // Map backend schema with fallbacks
  const concentration = contributors?.concentration;
  const participation = contributors?.participation;
  const tenure = contributors?.tenure;
  const botIsolation = contributors?.bot_isolation;

  const directoryTotal =
    participation?.directory_contributors_count ??
    contributors?.scope?.directory_contributors_count ??
    contributors?.total_contributors ??
    0;
  const activeAuthors =
    concentration?.active_human_commit_authors_count ??
    participation?.active_human_commit_authors_count ??
    0;
  const multiRoleActive = participation?.active_human_multi_role_contributors_count ?? activeAuthors;

  const gini = concentration?.gini_coefficient ?? contributors?.gini_coefficient;
  const bf50 = concentration?.bf50 ?? contributors?.bus_factors?.bf50;
  const bf80 = concentration?.bf80 ?? contributors?.bus_factors?.bf80;
  const top1Share = concentration?.top1_share;
  const top5Share = concentration?.top5_share;

  const rawLorenz = concentration?.lorenz_curve || contributors?.lorenz_curve || [];
  const lorenzData = rawLorenz.map((pt) => {
    const pop = pt.population_share ?? pt.contributor_percentile ?? 0;
    const contrib = pt.contribution_share ?? pt.contribution_percentile ?? 0;
    return {
      x: Number((pop * 100).toFixed(1)),
      observed: Number((contrib * 100).toFixed(1)),
      equality: Number((pop * 100).toFixed(1)),
    };
  });

  const coreCount = tenure?.core_contributors_count ?? contributors?.cohorts?.core_contributors ?? 0;
  const newcomers30d = tenure?.newcomers_30d_count ?? contributors?.cohorts?.newcomers_30d ?? 0;
  const botCount = botIsolation?.bot_accounts_detected ?? contributors?.cohorts?.bot_count ?? 0;

  // Normalize top_contributors / leaderboard records
  const rawLeaderboard = (contributors?.top_contributors || contributors?.leaderboard || []).map(
    (item: ContributorLeaderboardItem) => {
      const login = item.contributor || item.login || "unknown";
      const commits = item.commits ?? item.total_commits ?? 0;
      const prsAuthored = item.prs_authored ?? item.total_prs_authored ?? 0;
      const prsReviewed = item.prs_reviewed ?? item.total_prs_reviewed ?? 0;
      const isCore = Boolean(item.is_core);
      const isBot = Boolean(item.is_bot);
      const isNewcomer = Boolean(item.is_newcomer) || (item.tenure_days != null && item.tenure_days <= 30);

      let inferredRole = item.primary_role;
      if (!inferredRole) {
        if (isBot) inferredRole = "Automation Bot";
        else if (isCore) inferredRole = "Core Maintainer";
        else if (prsReviewed > commits && prsReviewed > 0) inferredRole = "Reviewer";
        else if (commits > 0) inferredRole = "Code Author";
        else if (prsAuthored > 0) inferredRole = "PR Contributor";
        else inferredRole = "Community";
      }

      return {
        login,
        commits,
        commitShare: item.commit_share,
        prsAuthored,
        prsReviewed,
        issuesOpened: item.issues_opened ?? 0,
        isCore,
        isBot,
        isNewcomer,
        tenureDays: item.tenure_days,
        primaryRole: inferredRole,
      };
    }
  );

  const filteredLeaderboard =
    roleFilter === "all"
      ? rawLeaderboard
      : rawLeaderboard.filter((c) => {
          if (roleFilter === "core") return c.isCore;
          if (roleFilter === "bot") return c.isBot;
          if (roleFilter === "newcomer") return c.isNewcomer;
          return true;
        });

  return (
    <div className="space-y-8">
      {/* 1. Ecosystem High-Level Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Users className="w-4 h-4 text-cyan-400" />
            <span>Tracked Contributors</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">{directoryTotal.toLocaleString()}</div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            {activeAuthors} active committers in sample
          </div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Scale className="w-4 h-4 text-indigo-400" />
            <span>Workload Concentration (Gini)</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">
            {gini != null ? gini.toFixed(3) : "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            {gini == null
              ? "Insufficient authors"
              : gini > 0.8
              ? "Heavily concentrated on top maintainers"
              : gini > 0.6
              ? "Moderately concentrated"
              : "Well-distributed across team"}
          </div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            <span>Key-Person Risk (Bus Factor)</span>
          </div>
          <div className="text-2xl font-bold text-amber-300 mt-1.5">
            {bf50 != null ? `${bf50} ${bf50 === 1 ? "person" : "people"}` : "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            {top1Share != null
              ? `Top author accounts for ${(top1Share * 100).toFixed(0)}% of commits`
              : "Produce 50% of total activity"}
          </div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <UserCheck className="w-4 h-4 text-emerald-400" />
            <span>Core Maintainers</span>
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-1.5">
            {coreCount}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            Newcomers: {newcomers30d} &middot; Bots isolated: {botCount}
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
                <span>Contribution Distribution Curve (Lorenz)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Compares observed commit distribution against perfectly equal contribution sharing
              </p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
              Gini = {gini != null ? gini.toFixed(3) : "N/A"}
            </span>
          </div>

          {lorenzData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-xs">
              Distribution curve requires at least 2 active human commit authors.
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
                  <Line type="monotone" dataKey="equality" name="Perfect Equality Baseline" stroke="#64748b" strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="observed" name="Observed Commit Share" stroke="#6366f1" strokeWidth={2.5} dot={false} />
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
              <span>Bus Factor & Resilience Breakdown</span>
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Counts the smallest number of maintainers who produced 50% and 80% of recent commits. Lower numbers mean higher reliance on a few individuals.
            </p>

            <div className="space-y-2.5 pt-2">
              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-white">50% Output Threshold (BF50)</div>
                  <div className="text-[11px] text-slate-400">Maintainers covering half of all commits</div>
                </div>
                <div className="text-xl font-bold text-amber-400 font-mono">
                  {bf50 ?? "N/A"}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-white">80% Output Threshold (BF80)</div>
                  <div className="text-[11px] text-slate-400">Maintainers covering 80% of all commits</div>
                </div>
                <div className="text-xl font-bold text-indigo-400 font-mono">
                  {bf80 ?? "N/A"}
                </div>
              </div>

              {top5Share != null && (
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                  <div>
                    <div className="text-xs font-semibold text-white">Top 5 Share</div>
                    <div className="text-[11px] text-slate-400">Multi-role active: {multiRoleActive}</div>
                  </div>
                  <div className="text-lg font-bold text-cyan-400 font-mono">
                    {(top5Share * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs leading-relaxed">
            <strong>Automated Bot Filtering:</strong> CI bots and automated dependency accounts ({botCount} detected) are excluded from human concentration metrics.
          </div>
        </div>
      </div>

      {/* 3. Contributor Breakdown Table */}
      <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Users className="w-4 h-4 text-cyan-400" />
              <span>Top Active Contributors</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Identity-reconciled breakdown of active authors and reviewers in this snapshot
            </p>
          </div>

          {/* Role Filters */}
          <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            {[
              { id: "all", label: `All (${rawLeaderboard.length})` },
              { id: "core", label: "Core" },
              { id: "newcomer", label: "Newcomers" },
              { id: "bot", label: "Bots" },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setRoleFilter(f.id)}
                className={`px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                  roleFilter === f.id ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-white"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {filteredLeaderboard.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-xs">
            No contributor records match the selected filter.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800 font-mono">
                <tr>
                  <th className="py-2.5 px-3">Contributor</th>
                  <th className="py-2.5 px-3">Role</th>
                  <th className="py-2.5 px-3 text-right">Commits</th>
                  <th className="py-2.5 px-3 text-right">Commit Share</th>
                  <th className="py-2.5 px-3 text-right">PRs Authored</th>
                  <th className="py-2.5 px-3 text-right">PRs Reviewed</th>
                  <th className="py-2.5 px-3 text-right">Observed Span</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredLeaderboard.slice(0, 25).map((item, i) => (
                  <tr key={i} className="hover:bg-slate-850/40">
                    <td className="py-2.5 px-3 flex items-center gap-2 font-medium text-white">
                      {item.isBot ? (
                        <Bot className="w-4 h-4 text-slate-400" />
                      ) : (
                        <div className="w-5 h-5 rounded-full bg-indigo-600/40 border border-indigo-500/50 flex items-center justify-center text-[10px] font-bold text-indigo-300">
                          {item.login.slice(0, 2).toUpperCase()}
                        </div>
                      )}
                      <span>{item.login}</span>
                      {item.isCore && (
                        <span className="px-1.5 py-0.2 rounded text-[9px] bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-bold uppercase">
                          Core
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{item.primaryRole}</td>
                    <td className="py-2.5 px-3 text-right font-mono font-bold text-white">{item.commits.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-indigo-300">
                      {item.commitShare != null ? `${(item.commitShare * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{item.prsAuthored}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{item.prsReviewed}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                      {item.tenureDays != null ? `${Math.round(item.tenureDays)}d` : "—"}
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
