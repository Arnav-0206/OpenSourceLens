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

  // Domain scorecards from backend diagnostic_scorecard
  const devCard = scorecard["activity_profile"] || scorecard["development_activity"] || {};
  const contribCard = scorecard["contributor_resilience"] || scorecard["contributor_ecosystem"] || {};
  const prCard = scorecard["pr_responsiveness"] || scorecard["pull_request_intelligence"] || {};
  const issueCard = scorecard["issue_health"] || scorecard["issue_intelligence"] || {};
  const relCard = scorecard["release_cadence"] || scorecard["release_evolution"] || {};
  const discCard = scorecard["community_discussions"] || {};

  const allRules = rules || [];
  const filteredRules = severityFilter === "all"
    ? allRules
    : allRules.filter((r) => r.severity === severityFilter || (severityFilter === "warning" && (r.severity === "high" || r.severity === "warning")));

  // Derive human-readable metrics with safe fallbacks
  const devTotalCommits = devCard.total_commits ?? payload?.development_activity?.summary?.total_commits ?? 0;
  const devVelocityTrend = devCard.velocity_trend ?? payload?.development_activity?.summary?.velocity_trend ?? "Stable";
  const devCompletedWeeks = devCard.completed_weeks ?? payload?.development_activity?.summary?.completed_weeks_count ?? "N/A";
  const devCadence = devCard.cadence_days ? `${devCard.cadence_days}d` : (payload?.development_activity?.summary?.current_cadence_days ? `${payload.development_activity.summary.current_cadence_days}d` : "N/A");

  const contribTotal = payload?.contributor_ecosystem?.participation?.directory_contributors_count
    ?? payload?.contributor_ecosystem?.scope?.directory_contributors_count
    ?? contribCard.active_human_commit_authors_count
    ?? payload?.contributor_ecosystem?.total_contributors
    ?? 0;
  const contribGini = contribCard.gini_coefficient ?? payload?.contributor_ecosystem?.concentration?.gini_coefficient;
  const contribBf50 = contribCard.bf50 ?? payload?.contributor_ecosystem?.concentration?.bf50;
  const contribTop1 = contribCard.top1_share ?? payload?.contributor_ecosystem?.concentration?.top1_share;

  const prTotal = prCard.eligible_prs_count ?? payload?.pull_request_intelligence?.outcome_distribution?.total_eligible_prs ?? 0;
  const prMergeRate = prCard.merge_rate ?? payload?.pull_request_intelligence?.outcome_distribution?.merge_rate;
  const prTurnaroundP50 = prCard.merge_latency_p50_hours ?? payload?.pull_request_intelligence?.merge_latency?.p50_hours;
  const prTtfRespP50 = prCard.ttf_resp_p50_hours ?? payload?.pull_request_intelligence?.responsiveness?.ttf_resp?.p50_hours;

  const isIssueEmpty = issueCard.coverage_status === "empty" || (payload?.issue_intelligence?.observed_flow?.total_created_observed === 0 && payload?.issue_intelligence?.scope?.mode !== undefined);
  const isDiscEmpty = discCard.coverage_status === "empty" || (disc?.total_discussions === 0);

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
            Evidence-Based Diagnostics
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
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 capitalize">
                {devCard.consistency_classification ? devCard.consistency_classification.replace(/_/g, " ") : "Active"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Total Commits</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {devTotalCommits.toLocaleString()}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Velocity Trend</div>
                <div className="text-base font-bold text-white mt-0.5 capitalize">
                  {devVelocityTrend}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Completed Weeks: {devCompletedWeeks}</span>
              <span>Cadence: {devCadence}</span>
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
                {contribBf50 != null ? `Bus Factor: ${contribBf50}` : "Analyzed"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Total Contributors</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {contribTotal.toLocaleString()}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Gini Inequality</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {contribGini != null ? contribGini.toFixed(3) : "N/A"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Active Committers: {contribCard.active_human_commit_authors_count ?? "N/A"}</span>
              <span>Top Author Share: {contribTop1 != null ? `${(contribTop1 * 100).toFixed(0)}%` : "N/A"}</span>
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
                {prTotal > 0 ? `${prTotal} Tracked` : "Tracked"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Turnaround p50</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {prTurnaroundP50 != null ? `${prTurnaroundP50.toFixed(1)}h` : "N/A (Pending)"}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Merge Rate</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {prMergeRate != null ? `${(prMergeRate * 100).toFixed(1)}%` : "N/A"}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>First Response p50: {prTtfRespP50 != null ? `${prTtfRespP50.toFixed(1)}h` : "N/A"}</span>
              <span>Sampled: {prTotal} PRs</span>
            </div>
          </div>

          {/* Card 4: Issue Intelligence & SLA */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                <span>Issue & Backlog Intelligence</span>
              </span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${isIssueEmpty ? "bg-slate-800 text-slate-400" : "bg-rose-500/10 text-rose-300"}`}>
                {isIssueEmpty ? "Not Active" : (issueCard.backlog_health || "Gauged")}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Resolution SLA p50</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {issueCard.resolution_p50_hours != null
                    ? `${issueCard.resolution_p50_hours.toFixed(1)}h`
                    : (isIssueEmpty ? "External / 0" : "N/A")}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-[10px] text-slate-400">Close Ratio</div>
                <div className="text-base font-bold text-white mt-0.5">
                  {issueCard.flow_ratio != null
                    ? `${(issueCard.flow_ratio * 100).toFixed(0)}%`
                    : (isIssueEmpty ? "N/A" : "N/A")}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>{isIssueEmpty ? "GitHub Issues Disabled" : `Stale Backlog: ${issueCard.stale_fraction ? `${(issueCard.stale_fraction * 100).toFixed(0)}%` : "0"}`}</span>
              <span>{isIssueEmpty ? "Mailing List Driven" : "Active"}</span>
            </div>
          </div>

          {/* Card 5: Release Evolution */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all space-y-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <Package className="w-4 h-4 text-amber-400" />
                <span>Release Evolution</span>
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 capitalize">
                {relCard.regularity_classification ? relCard.regularity_classification.replace(/_/g, " ") : "Recorded"}
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
                  {relCard.semver_adoption_rate != null
                    ? `${(relCard.semver_adoption_rate * 100).toFixed(0)}%`
                    : (rel?.semver_adoption?.semver_adoption_rate != null ? `${(rel.semver_adoption.semver_adoption_rate * 100).toFixed(0)}%` : "N/A")}
                </div>
              </div>
            </div>
            <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 border-t border-slate-800/60">
              <span>Cadence CV: {relCard.cv_release != null ? relCard.cv_release.toFixed(2) : "N/A"}</span>
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
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded capitalize ${isDiscEmpty ? "bg-slate-800 text-slate-400" : "bg-emerald-500/10 text-emerald-300"}`}>
                {isDiscEmpty ? "Not Enabled" : (discCard.coverage_status || "Active")}
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

      {/* 2. Cross-Domain Discovery Explorer */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Compass className="w-5 h-5 text-indigo-400" />
              <span>Cross-Domain Discovery Explorer</span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                Longitudinal Modeling
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Spearman monotonic alignment testing with Monte Carlo permutations (B=10,000), BH FDR correction (α=0.05), and AR(1) serial dependence guardrails
            </p>
          </div>
          <span className="text-[11px] font-mono px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
            {pairs.length} Evaluated Pairs
          </span>
        </div>

        {/* Scientific Disclaimer Banner */}
        <div className="p-3.5 rounded-xl border border-indigo-500/20 bg-indigo-950/20 text-xs text-indigo-200/90 flex items-start gap-3">
          <Info className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
          <p className="leading-relaxed">
            <strong>Scientific Association Guardrail:</strong> Statistical association does not establish direct operational causation. Correlation estimates reflect calendar-month alignment and may be influenced by seasonal cadence, team expansion, or external repository events.
          </p>
        </div>

        {/* 3 Aligned Correlation Cards Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {pairs.map((pair) => {
            const isSignificant = pair.significant_fdr_05;
            const isGuarded = pair.serial_dependence_guardrail_tripped;
            const isCoverageGuard = pair.status === "incomplete_review_coverage";
            const isInsufficientPeriods = pair.status === "insufficient_temporal_periods";
            const rVal = pair.spearman_r;

            const friendlyStatus = isSignificant
              ? "Significant (FDR 5%)"
              : isGuarded
              ? "AR(1) Guarded"
              : isCoverageGuard
              ? "Incomplete Reviews"
              : isInsufficientPeriods
              ? "Requires 12+ Months"
              : pair.status;

            return (
              <div
                key={pair.pair_id}
                className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-all flex flex-col justify-between space-y-3"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-bold border border-slate-700">
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
                      {friendlyStatus}
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

                  {isInsufficientPeriods ? (
                    <p className="text-[11px] text-slate-400 italic">
                      Fast Snapshot samples a 14-day window ({pair.k_months || 0} months). Longitudinal Spearman correlation requires 12+ calendar months of history.
                    </p>
                  ) : pair.note ? (
                    <p className="text-[11px] text-slate-400 italic line-clamp-2" title={pair.note}>
                      {pair.note}
                    </p>
                  ) : null}
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
      </section>

      {/* 3. Synthesized Findings Rules Feed */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-400" />
              <span>Synthesized Intelligence Findings ({allRules.length})</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Deterministic rule detections derived directly from multi-domain telemetry
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs">
            {["all", "warning", "positive", "info"].map((f) => (
              <button
                key={f}
                onClick={() => setSeverityFilter(f)}
                className={`px-3 py-1 rounded-lg font-medium capitalize transition-all cursor-pointer ${
                  severityFilter === f
                    ? "bg-indigo-600 text-white"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {f === "all" ? `All (${allRules.length})` : f === "warning" ? "Warnings & Bottlenecks" : f === "positive" ? "Positive Signals" : "Context"}
              </button>
            ))}
          </div>
        </div>

        {filteredRules.length === 0 ? (
          <div className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800 text-center text-slate-400 text-sm">
            No findings matching the selected filter.
          </div>
        ) : (
          <div className="space-y-3">
            {filteredRules.map((rule, idx) => {
              const sev = rule.severity;
              const isWarning = sev === "warning" || sev === "critical" || sev === "high";
              const isPositive = sev === "positive";

              return (
                <div
                  key={`${rule.rule_id}-${idx}`}
                  className={`p-4 rounded-xl border transition-all space-y-2 ${
                    isWarning
                      ? "border-amber-500/30 bg-amber-950/10 hover:border-amber-500/50"
                      : isPositive
                      ? "border-emerald-500/30 bg-emerald-950/10 hover:border-emerald-500/50"
                      : "border-slate-800 bg-slate-900/60 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        {rule.rule_id}
                      </span>
                      <span
                        className={`text-[10px] uppercase font-extrabold tracking-wider px-2 py-0.5 rounded ${
                          isWarning
                            ? "bg-amber-500/20 text-amber-300"
                            : isPositive
                            ? "bg-emerald-500/20 text-emerald-300"
                            : "bg-slate-800 text-slate-400"
                        }`}
                      >
                        {sev}
                      </span>
                      <span className="text-xs text-slate-400">
                        Domain: <span className="text-slate-300 capitalize">{rule.domain.replace(/_/g, " ")}</span>
                      </span>
                    </div>
                  </div>

                  <p className="text-sm font-semibold text-slate-200">
                    {rule.finding}
                  </p>

                  {rule.recommendation && (
                    <div className="flex items-start gap-2 pt-1 text-xs text-amber-200/90 bg-amber-500/5 p-2 rounded-lg border border-amber-500/10">
                      <Lightbulb className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
                      <span><strong>Prescriptive Action:</strong> {rule.recommendation}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {/* 4. Confidence Penalty Audit Log */}
      {confidence?.penalties && confidence.penalties.length > 0 && (
        <section className="space-y-3 pt-4 border-t border-slate-800">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Confidence Assessment & Penalty Audit Log
            </h3>
          </div>
          <div className="rounded-xl border border-slate-800 overflow-hidden bg-slate-900/40">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-[10px] font-mono uppercase text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="px-4 py-2.5">Rule</th>
                  <th className="px-4 py-2.5">Domain</th>
                  <th className="px-4 py-2.5">Penalty Impact</th>
                  <th className="px-4 py-2.5">Underlying Trigger Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                {confidence.penalties.map((p, i) => (
                  <tr key={i} className="hover:bg-slate-850/50">
                    <td className="px-4 py-2 text-cyan-400 font-bold">{p.rule}</td>
                    <td className="px-4 py-2 text-slate-300 font-sans capitalize">{p.domain?.replace(/_/g, " ") || "Global"}</td>
                    <td className="px-4 py-2 text-rose-400 font-bold">-{Math.round((p.penalty > 1 ? p.penalty / 100 : p.penalty) * 100)}%</td>
                    <td className="px-4 py-2 text-slate-400 font-sans text-xs">{p.reason}</td>
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
