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
  Activity,
  AlertCircle,
  AlertTriangle,
  Award,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Compass,
  Cpu,
  GitCommit,
  GitPullRequest,
  HelpCircle,
  Info,
  Layers,
  Lightbulb,
  MessageSquare,
  Package,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Users,
  Zap,
} from "lucide-react";
import { AnalysisPayload, ConfidenceSummary, CrossDomainCorrelationPair, RuleInterpretation } from "@/lib/types";

interface TabOverviewProps {
  payload?: AnalysisPayload | null;
  confidence?: ConfidenceSummary | null;
  rules?: RuleInterpretation[] | null;
}

export default function TabOverview({ payload, confidence, rules = [] }: TabOverviewProps) {
  const [severityFilter, setSeverityFilter] = useState<string>("all");

  const syn = payload?.repository_synthesis;
  const scorecard = syn?.diagnostic_scorecard || {};
  const rel = syn?.release_evolution || payload?.repository_synthesis?.release_evolution;
  const disc = syn?.discussions_intelligence || payload?.repository_synthesis?.discussions_intelligence;
  const correlations = syn?.cross_domain_correlations;
  const pairs: CrossDomainCorrelationPair[] = correlations?.pairs || [];

  // Domain scorecards
  const devCard = scorecard["development_activity"] || {};
  const contribCard = scorecard["contributor_ecosystem"] || {};
  const prCard = scorecard["pull_request_intelligence"] || {};
  const issueCard = scorecard["issue_intelligence"] || {};
  const relCard = scorecard["release_evolution"] || {};
  const discCard = scorecard["community_discussions"] || {};

  const allRules = rules || [];
  const filteredRules = severityFilter === "all"
    ? allRules
    : allRules.filter((r) => r.severity === severityFilter || (severityFilter === "warning" && (r.severity === "high" || r.severity === "warning")));

  return (
    <div className="space-y-8">
      {/* 1. Multi-Domain Diagnostic Scorecard Grid */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-400" />
              <span>Multi-Domain Diagnostic Scorecard</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Objective operational indicators synthesized across all 7 analytical domains
            </p>
          </div>
          <span className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
            No Arbitrary 0-100 CHI Scoring
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Card 1: Development Activity */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <GitCommit className="w-4 h-4 text-indigo-400" />
                <span>Development Activity</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300">
                {devCard.velocity_classification || "Active"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Total Commits</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {(devCard.total_commits ?? payload?.development_activity?.summary?.total_commits ?? 0).toLocaleString()}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Velocity Trend</div>
                <div className="text-base font-bold text-white mt-0.5 capitalize">
                  {devCard.velocity_trend || payload?.development_activity?.summary?.velocity_trend || "Stable"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Completed Weeks: {devCard.completed_weeks ?? "N/A"}</span>
              <span>Cadence: {devCard.cadence_days ? `${devCard.cadence_days}d` : "N/A"}</span>
            </div>
          </div>

          {/* Card 2: Contributor Ecosystem */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <Users className="w-4 h-4 text-cyan-400" />
                <span>Contributor Ecosystem</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300">
                {contribCard.concentration_classification || "Analyzed"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Total Contributors</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {(contribCard.total_contributors ?? payload?.contributor_ecosystem?.total_contributors ?? 0).toLocaleString()}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Gini Inequality</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {contribCard.gini_coefficient != null ? contribCard.gini_coefficient.toFixed(3) : "N/A"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Bus Factor BF50: {contribCard.bf50 ?? "N/A"}</span>
              <span>Core Ratio: {contribCard.core_ratio != null ? `${(contribCard.core_ratio * 100).toFixed(0)}%` : "N/A"}</span>
            </div>
          </div>

          {/* Card 3: Pull Request Turnaround */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <GitPullRequest className="w-4 h-4 text-indigo-400" />
                <span>Pull Request Intelligence</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300">
                {prCard.speed_tier || "Tracked"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Turnaround p50</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {prCard.turnaround_p50_hours != null ? `${prCard.turnaround_p50_hours.toFixed(1)}h` : "N/A"}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Merge Rate</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {prCard.merge_rate != null ? `${(prCard.merge_rate * 100).toFixed(0)}%` : "N/A"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>First Response p50: {prCard.ttfresp_p50_hours ? `${prCard.ttfresp_p50_hours.toFixed(1)}h` : "N/A"}</span>
              <span>Sampled: {prCard.total_prs ?? 0} PRs</span>
            </div>
          </div>

          {/* Card 4: Issue Intelligence & SLA */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                <span>Issue & Backlog Intelligence</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/10 text-rose-300">
                {issueCard.backlog_health || "Gauged"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Resolution SLA p50</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {issueCard.resolution_sla_p50_hours != null ? `${issueCard.resolution_sla_p50_hours.toFixed(1)}h` : "N/A"}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Close Ratio</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {issueCard.close_ratio != null ? `${(issueCard.close_ratio * 100).toFixed(0)}%` : "N/A"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Stale Backlog: {issueCard.stale_count ?? 0}</span>
              <span>Intensity: {issueCard.discussion_intensity_p50 ? `${issueCard.discussion_intensity_p50} comms` : "N/A"}</span>
            </div>
          </div>

          {/* Card 5: Release Evolution */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <Package className="w-4 h-4 text-amber-400" />
                <span>Release Evolution</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300">
                {relCard.regularity_classification || "Recorded"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Total Releases</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {relCard.total_releases ?? rel?.cadence?.total_releases ?? 0}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">SemVer Adoption</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {relCard.semver_adoption_rate != null ? `${(relCard.semver_adoption_rate * 100).toFixed(0)}%` : "N/A"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>CV Cadence: {relCard.cv_release != null ? relCard.cv_release.toFixed(2) : "N/A"}</span>
              <span>Last: {relCard.days_since_last_release ? `${relCard.days_since_last_release}d ago` : "N/A"}</span>
            </div>
          </div>

          {/* Card 6: Community Discussions */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <MessageSquare className="w-4 h-4 text-emerald-400" />
                <span>Community Discussions</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 capitalize">
                {discCard.coverage_status || disc?.status || "Active"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Total Discussions</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {discCard.total_discussions ?? disc?.total_discussions ?? 0}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Q&A Answered Rate</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {discCard.qna_answered_rate != null
                    ? `${(discCard.qna_answered_rate * 100).toFixed(0)}%`
                    : (disc?.qna?.qna_answered_rate != null ? `${(disc.qna.qna_answered_rate * 100).toFixed(0)}%` : "N/A")}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Time-to-Answer: {discCard.time_to_answer_p50_hours ? `${discCard.time_to_answer_p50_hours.toFixed(1)}h` : "N/A"}</span>
              <span>Category: {discCard.dominant_category || disc?.dominant_category || "N/A"}</span>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Cross-Domain Discovery Explorer (Feature Slice 4: Synthesis & Scale) */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Compass className="w-5 h-5 text-indigo-400" />
              <span>Cross-Domain Discovery Explorer</span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                Synthesis & Scale
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Spearman monotonic alignment testing with Monte Carlo permutations (B=10,000), BH FDR correction (α=0.05), and AR(1) serial dependence guardrails
            </p>
          </div>
          <span className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
            {pairs.length} Tested Pairs
          </span>
        </div>

        {/* Scientific Non-Causal Disclaimer */}
        <div className="flex items-start gap-3 p-3.5 rounded-xl bg-indigo-950/30 border border-indigo-500/20 text-xs text-indigo-200/90 leading-relaxed">
          <Info className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
          <div>
            <strong className="text-white">Scientific Association Guardrail: </strong>
            Statistical association does not establish direct operational causation. Correlation estimates reflect calendar-month alignment and may be influenced by seasonal cadence, team expansion, or external repository events.
          </div>
        </div>

        {pairs.length === 0 ? (
          <div className="p-8 text-center rounded-xl border border-slate-800 bg-slate-900/40 text-slate-400 text-sm">
            No cross-domain correlation pairs available for this repository. At least 6 overlapping calendar months with complete telemetry are required for inferential testing.
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            {pairs.map((pair) => {
              const isSignificant = pair.significant_fdr_05;
              const isGuarded = pair.serial_dependence_guardrail_tripped || pair.status === "serial_dependence_detected";
              const isCoverageGuard = pair.status === "incomplete_review_coverage";
              const rVal = pair.spearman_r;

              return (
                <div
                  key={pair.pair_id}
                  className={`p-4 rounded-xl border flex flex-col justify-between transition-all ${
                    isSignificant
                      ? "border-emerald-500/30 bg-emerald-950/10"
                      : isGuarded
                      ? "border-amber-500/30 bg-amber-950/10"
                      : "border-slate-800 bg-slate-900/60"
                  }`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-indigo-300 border border-slate-700">
                        {pair.pair_id}
                      </span>
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                          isSignificant
                            ? "bg-emerald-500/20 text-emerald-300"
                            : isGuarded
                            ? "bg-amber-500/20 text-amber-300"
                            : isCoverageGuard
                            ? "bg-amber-500/20 text-amber-300"
                            : "bg-slate-800 text-slate-400"
                        }`}
                      >
                        {isSignificant
                          ? "Significant (FDR 5%)"
                          : isGuarded
                          ? "AR(1) Guarded"
                          : isCoverageGuard
                          ? "Incomplete Reviews"
                          : pair.status}
                      </span>
                    </div>

                    <div>
                      <h3 className="text-sm font-bold text-white">
                        {pair.label_x || pair.metric_x} vs {pair.label_y || pair.metric_y}
                      </h3>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        Aligned Window: {pair.k_months || pair.overlapping_months?.length || 0} overlapping months
                      </p>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-4 gap-1.5 p-2 rounded-lg bg-slate-950/70 border border-slate-800/80 text-center">
                      <div>
                        <div className="text-[9px] text-slate-400">Spearman r</div>
                        <div
                          className={`text-xs font-mono font-bold mt-0.5 ${
                            rVal != null
                              ? rVal > 0.3
                                ? "text-emerald-400"
                                : rVal < -0.3
                                ? "text-rose-400"
                                : "text-slate-200"
                              : "text-slate-500"
                          }`}
                        >
                          {rVal != null ? (rVal > 0 ? `+${rVal.toFixed(2)}` : rVal.toFixed(2)) : "N/A"}
                        </div>
                      </div>
                      <div>
                        <div className="text-[9px] text-slate-400">p-value</div>
                        <div className="text-xs font-mono font-bold text-slate-200 mt-0.5">
                          {pair.p_value != null ? pair.p_value.toFixed(3) : "Suppr."}
                        </div>
                      </div>
                      <div>
                        <div className="text-[9px] text-slate-400">q (FDR)</div>
                        <div className="text-xs font-mono font-bold text-slate-200 mt-0.5">
                          {pair.q_value != null ? pair.q_value.toFixed(3) : "N/A"}
                        </div>
                      </div>
                      <div>
                        <div className="text-[9px] text-slate-400">|r_AR1|</div>
                        <div
                          className={`text-xs font-mono font-bold mt-0.5 ${
                            pair.max_abs_ar1 != null && pair.max_abs_ar1 > 0.40
                              ? "text-amber-400"
                              : "text-slate-300"
                          }`}
                        >
                          {pair.max_abs_ar1 != null ? pair.max_abs_ar1.toFixed(2) : "N/A"}
                        </div>
                      </div>
                    </div>

                    {pair.note && (
                      <p className="text-[11px] text-slate-400 italic line-clamp-2" title={pair.note}>
                        {pair.note}
                      </p>
                    )}
                  </div>

                  {/* Dual-Line Trend Chart */}
                  {pair.monthly_points && pair.monthly_points.length > 0 && (
                    <div className="h-36 w-full mt-3 pt-2 border-t border-slate-800/60">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={pair.monthly_points} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                          <XAxis dataKey="month" stroke="#64748b" fontSize={9} tickLine={false} />
                          <YAxis yAxisId="left" stroke="#818cf8" fontSize={9} tickLine={false} />
                          <YAxis yAxisId="right" orientation="right" stroke="#38bdf8" fontSize={9} tickLine={false} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "0.5rem",
                              fontSize: "10px",
                            }}
                            labelStyle={{ color: "#f8fafc", fontWeight: "bold" }}
                          />
                          <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="val_x"
                            name={pair.label_x || "Series X"}
                            stroke="#818cf8"
                            strokeWidth={1.5}
                            dot={false}
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="val_y"
                            name={pair.label_y || "Series Y"}
                            stroke="#38bdf8"
                            strokeWidth={1.5}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 3. Deterministic Rule Findings */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-400" />
              <span>Synthesized Intelligence Findings ({filteredRules.length})</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Deterministic rule detections derived directly from multi-domain telemetry
            </p>
          </div>

          {/* Severity Filter Tabs */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 text-xs">
            <button
              onClick={() => setSeverityFilter("all")}
              className={`px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                severityFilter === "all" ? "bg-indigo-600 text-white font-semibold" : "text-slate-400 hover:text-white"
              }`}
            >
              All ({allRules.length})
            </button>
            <button
              onClick={() => setSeverityFilter("warning")}
              className={`px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                severityFilter === "warning" ? "bg-amber-600/80 text-white font-semibold" : "text-slate-400 hover:text-white"
              }`}
            >
              Warnings & Bottlenecks
            </button>
            <button
              onClick={() => setSeverityFilter("positive")}
              className={`px-3 py-1 rounded-lg transition-colors cursor-pointer ${
                severityFilter === "positive" ? "bg-emerald-600/80 text-white font-semibold" : "text-slate-400 hover:text-white"
              }`}
            >
              Positive Signals
            </button>
          </div>
        </div>

        {filteredRules.length === 0 ? (
          <div className="p-8 text-center rounded-xl border border-slate-800 bg-slate-900/40 text-slate-400 text-sm">
            No specific rule findings matching this filter.
          </div>
        ) : (
          <div className="space-y-3">
            {filteredRules.map((rule, idx) => {
              const isWarning = rule.severity === "high" || rule.severity === "warning";
              const isPositive = rule.severity === "positive";

              return (
                <div
                  key={`${rule.rule_id}-${idx}`}
                  className={`p-4 rounded-xl border transition-all ${
                    isWarning
                      ? "border-amber-500/25 bg-amber-950/15"
                      : isPositive
                      ? "border-emerald-500/25 bg-emerald-950/15"
                      : "border-slate-800 bg-slate-900/50"
                  }`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                    <div className="space-y-1.5 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-indigo-300 border border-slate-700">
                          {rule.rule_id}
                        </span>
                        <span
                          className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded ${
                            isWarning
                              ? "bg-amber-500/20 text-amber-300"
                              : isPositive
                              ? "bg-emerald-500/20 text-emerald-300"
                              : "bg-slate-800 text-slate-300"
                          }`}
                        >
                          {rule.severity}
                        </span>
                        <span className="text-xs text-slate-500 capitalize">
                          Domain: {rule.domain.replace("_", " ")}
                        </span>
                      </div>

                      <p className="text-sm font-semibold text-white leading-relaxed">
                        {rule.finding}
                      </p>

                      {rule.recommendation && (
                        <div className="flex items-start gap-2 pt-2 text-xs text-slate-300 leading-relaxed border-t border-slate-800/80">
                          <Lightbulb className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                          <div>
                            <span className="font-semibold text-amber-300">Prescriptive Action: </span>
                            {rule.recommendation}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 3. Statistical Confidence & Applied Penalty Audit Log */}
      {confidence?.penalties && confidence.penalties.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-indigo-400" />
            <span>Confidence Assessment & Penalty Audit Log</span>
          </h2>

          <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/50">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-semibold">Rule</th>
                  <th className="py-2.5 px-4 font-semibold">Domain</th>
                  <th className="py-2.5 px-4 font-semibold">Penalty Impact</th>
                  <th className="py-2.5 px-4 font-semibold">Underlying Trigger Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                {confidence.penalties.map((p, i) => (
                  <tr key={i} className="hover:bg-slate-850/50">
                    <td className="py-2.5 px-4 font-bold text-amber-300">{p.rule}</td>
                    <td className="py-2.5 px-4 text-slate-400 capitalize">{p.domain?.replace("_", " ") || "Global"}</td>
                    <td className="py-2.5 px-4 text-rose-400 font-bold">-{(p.penalty * 100).toFixed(0)}%</td>
                    <td className="py-2.5 px-4 font-sans text-slate-300">{p.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
