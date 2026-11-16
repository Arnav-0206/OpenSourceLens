"use client";

import React, { useState } from "react";
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
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  ExternalLink,
  Flame,
  HelpCircle,
  Inbox,
  Layers,
  Loader2,
  MessageCircle,
  MessageSquare,
  Search,
  Sparkles,
  Tag,
  TrendingUp,
  Zap,
} from "lucide-react";
import {
  DiscoveredTopicItem,
  IssueIntelligenceData,
  SimilarCandidateItem,
  SimilarQueryResponse,
  TopicDiscoveryResponse,
} from "@/lib/types";
import { getDiscoveredTopics, searchSimilarIssues } from "@/lib/api";

interface TabIssuesProps {
  issues?: IssueIntelligenceData | null;
  owner?: string;
  name?: string;
}

export default function TabIssues({ issues, owner, name }: TabIssuesProps) {
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

  // ─── Semantic Duplicate Search State ─────────────────────────────────────────
  const [queryText, setQueryText] = useState("");
  const [threshold, setThreshold] = useState(0.45);
  const [includeDiscussions, setIncludeDiscussions] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [similarityResult, setSimilarityResult] =
    useState<SimilarQueryResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  // ─── Topic Discovery State ───────────────────────────────────────────────────
  const [topicsResult, setTopicsResult] =
    useState<TopicDiscoveryResponse | null>(null);
  const [isLoadingTopics, setIsLoadingTopics] = useState(false);
  const [topicError, setTopicError] = useState<string | null>(null);
  const [expandedTopicId, setExpandedTopicId] = useState<number | null>(null);

  const handleSearchSimilar = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!owner || !name || !queryText.trim()) return;

    setIsSearching(true);
    setSearchError(null);
    try {
      const res = await searchSimilarIssues(owner, name, {
        query_text: queryText.trim(),
        top_k: 5,
        threshold: threshold,
        include_discussions: includeDiscussions,
      });
      setSimilarityResult(res);
    } catch (err: any) {
      setSearchError(
        err.message || "Failed to search semantically similar issues."
      );
    } finally {
      setIsSearching(false);
    }
  };

  const handleLoadTopics = async () => {
    if (!owner || !name) return;
    setIsLoadingTopics(true);
    setTopicError(null);
    try {
      const res = await getDiscoveredTopics(owner, name, 8);
      setTopicsResult(res);
    } catch (err: any) {
      setTopicError(err.message || "Failed to discover backlog topics.");
    } finally {
      setIsLoadingTopics(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* 1. High-Level Backlog Indicators */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Inbox className="w-4 h-4 text-indigo-400" />
            <span>Total Tracked Issues</span>
          </div>
          <div className="text-2xl font-bold text-white mt-1.5">
            {total.toLocaleString()}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            Excludes PR envelopes
          </div>
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
            p75: {sla?.p75 != null ? `${sla.p75.toFixed(1)}h` : "N/A"} &middot;
            p90: {sla?.p90 != null ? `${sla.p90.toFixed(1)}h` : "N/A"}
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
          <div className="text-[11px] text-slate-500 mt-0.5">
            &gt; 90 days dormant
          </div>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <MessageCircle className="w-4 h-4 text-amber-400" />
            <span>Discussion Intensity</span>
          </div>
          <div className="text-2xl font-bold text-amber-300 mt-1.5">
            {discussion?.comments_per_issue_p50 ?? "N/A"}
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            Median comments per issue
          </div>
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
                Opened:{" "}
                <strong className="text-white">{flow.total_opened}</strong>
              </span>
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
                Closed:{" "}
                <strong className="text-white">{flow.total_closed}</strong>
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
              <ComposedChart
                data={chartData}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#334155"
                  opacity={0.5}
                />
                <XAxis
                  dataKey="month"
                  stroke="#94a3b8"
                  fontSize={11}
                  tickLine={false}
                />
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
                  dataKey="opened"
                  name="Issues Opened"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                  opacity={0.8}
                />
                <Bar
                  dataKey="closed"
                  name="Issues Closed"
                  fill="#10b981"
                  radius={[4, 4, 0, 0]}
                  opacity={0.8}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      {/* 3. Applied Intelligence: Semantic Duplicate Finder */}
      <section className="space-y-5 rounded-2xl border border-indigo-500/30 bg-slate-900/80 p-6 backdrop-blur-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <span>Semantic Duplicate &amp; Related Issue Finder</span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  Dense Vectors &middot; all-MiniLM-L6-v2
                </span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Query candidate bug reports or feature ideas to detect duplicates
                or find overlapping discussion threads in this repository
              </p>
            </div>
          </div>
        </div>

        {/* Search Input Bar */}
        <form onSubmit={handleSearchSimilar} className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-400" />
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="Paste a bug report title, description, or stack trace error..."
              className="w-full pl-10 pr-32 py-3 rounded-xl bg-slate-950/80 border border-slate-700/80 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 text-sm text-slate-100 placeholder-slate-500 outline-none transition-all"
            />
            <button
              type="submit"
              disabled={isSearching || !queryText.trim()}
              className="absolute right-2 top-2 bottom-2 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all disabled:opacity-40 cursor-pointer flex items-center gap-1.5"
            >
              {isSearching ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Zap className="w-3.5 h-3.5" />
              )}
              <span>Find Matches</span>
            </button>
          </div>

          <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 gap-3 pt-1">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <span>Sensitivity Cutoff:</span>
                <select
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-slate-200 text-xs outline-none"
                >
                  <option value={0.7}>High / Duplicate Only (&ge; 70%)</option>
                  <option value={0.55}>Medium Related (&ge; 55%)</option>
                  <option value={0.45}>Broad Context (&ge; 45%)</option>
                </select>
              </label>

              <label className="flex items-center gap-1.5 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={includeDiscussions}
                  onChange={(e) => setIncludeDiscussions(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-800 text-indigo-500 focus:ring-0"
                />
                <span>Include Community Discussions</span>
              </label>
            </div>

            {similarityResult && (
              <span className="font-mono text-[11px] text-slate-500">
                Latency: {similarityResult.latency_ms}ms
              </span>
            )}
          </div>
        </form>

        {/* Search Error */}
        {searchError && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs">
            {searchError}
          </div>
        )}

        {/* Results Container */}
        {similarityResult && (
          <div className="space-y-4 pt-2 border-t border-slate-800">
            {/* Recommendation Alert Banner */}
            <div
              className={`p-4 rounded-xl border text-xs flex items-start gap-3 ${
                similarityResult.has_potential_duplicate
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-200"
                  : similarityResult.total_candidates_found > 0
                  ? "bg-indigo-500/10 border-indigo-500/25 text-indigo-200"
                  : "bg-slate-800/60 border-slate-700 text-slate-300"
              }`}
            >
              {similarityResult.has_potential_duplicate ? (
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              ) : (
                <Sparkles className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
              )}
              <div>
                <strong className="font-semibold block mb-0.5">
                  {similarityResult.has_potential_duplicate
                    ? "Potential Duplicate Candidate Detected"
                    : "Semantic Search Analysis"}
                </strong>
                <p className="leading-relaxed">
                  {similarityResult.primary_recommendation}
                </p>
              </div>
            </div>

            {/* Candidate List */}
            {similarityResult.candidates.length > 0 && (
              <div className="space-y-2.5">
                {similarityResult.candidates.map((cand) => {
                  const pct = Math.round(cand.similarity_score * 100);
                  const isDup = cand.duplicate_candidate;
                  return (
                    <div
                      key={`${cand.entity_type}-${cand.id}`}
                      className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700 transition-colors flex items-center justify-between gap-4"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <span className="font-mono text-xs text-slate-400">
                            #{cand.number}
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] uppercase font-semibold tracking-wider bg-slate-800 text-slate-300 border border-slate-700">
                            {cand.entity_type}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-medium capitalize ${
                              cand.state === "open"
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {cand.state}
                          </span>
                          {cand.canonical_label && (
                            <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                              {cand.canonical_label}
                            </span>
                          )}
                          {cand.category && (
                            <span className="px-2 py-0.5 rounded text-[10px] bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                              {cand.category}
                            </span>
                          )}
                        </div>
                        <div className="text-sm font-medium text-slate-100 truncate">
                          {cand.title}
                        </div>
                      </div>

                      <div className="flex items-center gap-3 shrink-0">
                        {/* Similarity Pill */}
                        <div
                          className={`px-3 py-1.5 rounded-xl border text-xs font-mono font-bold flex items-center gap-1.5 ${
                            isDup
                              ? "bg-amber-500/15 text-amber-300 border-amber-500/40"
                              : cand.confidence_tier === "MEDIUM"
                              ? "bg-cyan-500/15 text-cyan-300 border-cyan-500/30"
                              : "bg-slate-800 text-slate-300 border-slate-700"
                          }`}
                        >
                          <span>{pct}%</span>
                          <span className="text-[10px] font-sans font-normal opacity-80">
                            {isDup ? "Duplicate" : "Related"}
                          </span>
                        </div>

                        {/* GitHub External Link */}
                        {owner && name && (
                          <a
                            href={
                              cand.entity_type === "discussion"
                                ? `https://github.com/${owner}/${name}/discussions/${cand.number}`
                                : `https://github.com/${owner}/${name}/issues/${cand.number}`
                            }
                            target="_blank"
                            rel="noopener noreferrer"
                            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
                            title="View on GitHub"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </section>

      {/* 4. Applied Intelligence: Unsupervised Backlog Topics Explorer */}
      <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              <span>Discovered Backlog Themes &amp; Topic Clusters</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Unsupervised Spherical KMeans + c-TF-IDF keyword discovery across
              all issues and discussions
            </p>
          </div>

          <button
            onClick={handleLoadTopics}
            disabled={isLoadingTopics}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 text-xs font-semibold border border-slate-700 hover:border-slate-600 transition-all cursor-pointer disabled:opacity-50 self-start sm:self-auto"
          >
            {isLoadingTopics ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
            ) : (
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            )}
            <span>
              {topicsResult ? "Refresh Themes" : "Discover Themes"}
            </span>
          </button>
        </div>

        {topicError && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs">
            {topicError}
          </div>
        )}

        {/* Topic Grid */}
        {topicsResult && (
          <div className="space-y-4 pt-1">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>
                Discovered{" "}
                <strong className="text-slate-200">
                  {topicsResult.total_topics_discovered}
                </strong>{" "}
                thematic clusters across{" "}
                <strong className="text-slate-200">
                  {topicsResult.total_items_clustered}
                </strong>{" "}
                threads
              </span>
              <span className="font-mono text-[11px] text-slate-500">
                {topicsResult.execution_time_ms}ms &middot; {topicsResult.methodology}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {topicsResult.topics.map((topic) => {
                const isExpanded = expandedTopicId === topic.cluster_id;
                return (
                  <div
                    key={topic.cluster_id}
                    className="p-5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700/80 transition-all space-y-3"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-xs font-mono text-indigo-400 mb-0.5">
                          Cluster #{topic.cluster_id + 1}
                        </div>
                        <h3 className="text-sm font-bold text-white tracking-tight">
                          {topic.title}
                        </h3>
                      </div>
                      <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800 text-slate-200 border border-slate-700 shrink-0">
                        {topic.total_count} items
                      </span>
                    </div>

                    {/* Keywords pills */}
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {topic.keywords.map((kw, i) => (
                        <span
                          key={i}
                          className="px-2 py-0.5 rounded text-[11px] font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20"
                        >
                          #{kw}
                        </span>
                      ))}
                    </div>

                    {/* Open vs Closed ratio */}
                    <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                      <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1 text-emerald-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          {topic.open_count} open
                        </span>
                        <span className="flex items-center gap-1 text-slate-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                          {topic.closed_count} closed
                        </span>
                      </div>

                      <button
                        onClick={() =>
                          setExpandedTopicId(isExpanded ? null : topic.cluster_id)
                        }
                        className="inline-flex items-center gap-1 text-indigo-400 hover:text-indigo-300 cursor-pointer"
                      >
                        <span>{isExpanded ? "Hide Samples" : "View Samples"}</span>
                        {isExpanded ? (
                          <ChevronUp className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>

                    {/* Expandable Representative Items */}
                    {isExpanded && topic.representative_items.length > 0 && (
                      <div className="pt-2 space-y-1.5 border-t border-slate-800/60">
                        <div className="text-[10px] uppercase font-semibold text-slate-500 tracking-wider mb-1">
                          Representative Issues (Closest to Centroid)
                        </div>
                        {topic.representative_items.map((item, idx) => (
                          <div
                            key={idx}
                            className="flex items-center justify-between gap-2 p-2 rounded-lg bg-slate-900 border border-slate-800 text-xs"
                          >
                            <div className="truncate flex-1">
                              <span className="font-mono text-slate-400 mr-1.5">
                                #{item.number}
                              </span>
                              <span className="text-slate-200">{item.title}</span>
                            </div>
                            {owner && name && (
                              <a
                                href={
                                  item.entity_type === "discussion"
                                    ? `https://github.com/${owner}/${name}/discussions/${item.number}`
                                    : `https://github.com/${owner}/${name}/issues/${item.number}`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-400 hover:text-indigo-400 transition-colors shrink-0"
                              >
                                <ExternalLink className="w-3.5 h-3.5" />
                              </a>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* 5. Backlog Aging Dynamics & Collaboration Diagnostics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/60 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <span>Backlog Aging Dynamics</span>
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Tracks accumulation of unresolved issues and dormancy risks across the
            backlog.
          </p>
          <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Median Age</div>
              <div className="text-lg font-bold text-white mt-1">
                {aging?.median_age_days != null
                  ? `${aging.median_age_days.toFixed(0)} days`
                  : "N/A"}
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
            Analyzes discussion density and response engagement on bug reports and
            feature requests.
          </p>
          <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Mean Comments</div>
              <div className="text-lg font-bold text-white mt-1">
                {discussion?.comments_per_issue_mean != null
                  ? discussion.comments_per_issue_mean.toFixed(1)
                  : "N/A"}
              </div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
              <div className="text-[10px] text-slate-400">Median Comments</div>
              <div className="text-lg font-bold text-white mt-1">
                {discussion?.comments_per_issue_p50 != null
                  ? discussion.comments_per_issue_p50.toFixed(1)
                  : "N/A"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
