"use client";

import React, { useEffect, useState } from "react";
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertCircle,
  BookOpen,
  Bug,
  Clock,
  ExternalLink,
  HelpCircle,
  Inbox,
  Layers,
  Loader2,
  MessageCircle,
  Sparkles,
  Tag,
  TrendingUp,
  Wrench,
} from "lucide-react";
import {
  ClassifiedIssuesResponse,
  IssueIntelligenceData,
  TopicDiscoveryResponse,
} from "@/lib/types";
import { getClassifiedIssues, getDiscoveredTopics } from "@/lib/api";

interface TabIssuesProps {
  issues?: IssueIntelligenceData | null;
  owner?: string;
  name?: string;
}

const CATEGORY_META: Record<
  string,
  { label: string; color: string; badgeClass: string; icon: React.ComponentType<{ className?: string }> }
> = {
  bug: {
    label: "Bug Report",
    color: "#f43f5e",
    badgeClass: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    icon: Bug,
  },
  feature: {
    label: "Feature Request",
    color: "#8b5cf6",
    badgeClass: "bg-purple-500/15 text-purple-300 border-purple-500/30",
    icon: Sparkles,
  },
  documentation: {
    label: "Documentation",
    color: "#38bdf8",
    badgeClass: "bg-sky-500/15 text-sky-300 border-sky-500/30",
    icon: BookOpen,
  },
  question: {
    label: "Question / Support",
    color: "#f59e0b",
    badgeClass: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    icon: HelpCircle,
  },
  other: {
    label: "Maintenance / Other",
    color: "#94a3b8",
    badgeClass: "bg-slate-700/50 text-slate-300 border-slate-600/50",
    icon: Wrench,
  },
};

function formatDuration(hours?: number | null): string | null {
  if (hours == null || Number.isNaN(hours)) return null;
  if (hours < 1) {
    const mins = Math.round(hours * 60);
    return mins <= 1 ? "< 1 min" : `${mins} mins`;
  }
  if (hours >= 48) {
    return `${(hours / 24).toFixed(1)} days`;
  }
  return `${hours.toFixed(1)} hrs`;
}

function formatRelativeDate(iso?: string | null): string {
  if (!iso) return "";
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return "";
  const days = Math.floor((Date.now() - dt.getTime()) / (1000 * 60 * 60 * 24));
  if (days <= 0) return "today";
  if (days === 1) return "1d ago";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${(days / 365).toFixed(1)}y ago`;
}

export default function TabIssues({ issues, owner, name }: TabIssuesProps) {
  const obsFlow = issues?.observed_flow;
  const legacyFlow = issues?.flow;
  const resDist = issues?.resolution_distribution;
  const legacySla = issues?.resolution_sla_hours;
  const aging = issues?.backlog_aging;
  const legacyAging = issues?.aging;
  const discussion = issues?.discussion_intensity;

  const totalOpened = obsFlow?.total_created_observed ?? legacyFlow?.total_opened ?? 0;
  const totalClosed = obsFlow?.total_closed_observed ?? legacyFlow?.total_closed ?? 0;
  const totalTracked =
    issues?.scope?.eligible_issues_count ??
    obsFlow?.total_created_observed ??
    issues?.total_issues ??
    0;

  const closeTimeMedian = formatDuration(resDist?.p50_hours ?? legacySla?.p50);
  const staleCount = aging?.stale_gt_90d_count ?? legacyAging?.open_stale_count ?? 0;
  const openCount = aging?.open_issues_count ?? legacyAging?.open_count ?? 0;
  const medianComments = discussion?.median_comments_per_issue;

  const rawMonthly: Array<{
    month: string;
    created_count?: number;
    closed_count?: number;
    created?: number;
    opened?: number;
    closed?: number;
  }> = obsFlow?.monthly_series || legacyFlow?.monthly_flow || [];

  const allMonthlyData = rawMonthly.map((m) => ({
    month: m.month,
    opened: m.created_count ?? m.created ?? m.opened ?? 0,
    closed: m.closed_count ?? m.closed ?? 0,
  }));

  // If history spans > 24 months with mostly empty months, show active months (capped at latest 24) so bars stay wide & readable
  const chartData =
    allMonthlyData.length > 24
      ? allMonthlyData
          .filter((pt) => pt.opened > 0 || pt.closed > 0)
          .slice(-24)
      : allMonthlyData;

  // ─── Auto-Fetched AI Classified Issues State ───────────────────────────────
  const [stateFilter, setStateFilter] = useState<"open" | "closed" | "all">("open");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [classifiedData, setClassifiedData] = useState<ClassifiedIssuesResponse | null>(null);
  const [isLoadingIssues, setIsLoadingIssues] = useState<boolean>(false);
  const [issuesError, setIssuesError] = useState<string | null>(null);

  // ─── Auto-Loaded AI Topic Clusters State ───────────────────────────────────
  const [topicsData, setTopicsData] = useState<TopicDiscoveryResponse | null>(null);

  useEffect(() => {
    if (!owner || !name) return;
    let cancelled = false;

    const loadClassified = async () => {
      setIsLoadingIssues(true);
      setIssuesError(null);
      try {
        const res = await getClassifiedIssues(owner, name, stateFilter, 60);
        if (!cancelled) {
          setClassifiedData(res);
          // If open issues is 0 but closed issues exist, automatically switch to all on initial load
          if (stateFilter === "open" && res.total_open === 0 && res.total_closed > 0) {
            setStateFilter("all");
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          setIssuesError(err.message || "Could not load classified issues.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingIssues(false);
        }
      }
    };

    loadClassified();
    return () => {
      cancelled = true;
    };
  }, [owner, name, stateFilter]);

  useEffect(() => {
    if (!owner || !name || totalTracked < 6) return;
    let cancelled = false;
    getDiscoveredTopics(owner, name, 6)
      .then((res) => {
        if (!cancelled && res.topics && res.topics.length > 0) {
          setTopicsData(res);
        }
      })
      .catch(() => {
        // Silently omit topic clusters if insufficient issues
      });
    return () => {
      cancelled = true;
    };
  }, [owner, name, totalTracked]);

  const hasAnyIssues =
    totalTracked > 0 ||
    openCount > 0 ||
    (classifiedData != null && (classifiedData.total_open > 0 || classifiedData.total_closed > 0));

  // Clean single-card empty state (ZERO N/A clutter!)
  if (!isLoadingIssues && !hasAnyIssues) {
    return (
      <div className="p-10 rounded-2xl border border-slate-800 bg-slate-900/60 text-center max-w-2xl mx-auto my-8 space-y-4">
        <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/25 flex items-center justify-center mx-auto text-indigo-400">
          <Inbox className="w-7 h-7" />
        </div>
        <h3 className="text-lg font-bold text-white">
          GitHub Issues Are Not Used by This Repository
        </h3>
        <p className="text-sm text-slate-400 leading-relaxed">
          This repository has <strong className="text-slate-200">0 issues on GitHub</strong>. Projects like Git, Linux, and GCC track bug reports and feature discussions on external mailing lists or bug trackers. Try any GitHub-native repository (such as <code className="text-indigo-300">pallets/flask</code> or <code className="text-indigo-300">fastapi/fastapi</code>) to see automatic AI issue classification and topic discovery in action.
        </p>
      </div>
    );
  }

  // Build ONLY non-null KPI cards
  const kpiCards: Array<{
    label: string;
    value: string;
    sub: string;
    color: string;
    icon: React.ComponentType<{ className?: string }>;
  }> = [
    {
      label: "Open Issues",
      value: (classifiedData?.total_open ?? openCount).toLocaleString(),
      sub: `${(classifiedData?.total_closed ?? totalClosed).toLocaleString()} closed in snapshot`,
      color: "text-emerald-400",
      icon: Inbox,
    },
    ...(closeTimeMedian
      ? [
          {
            label: "Typical Close Time",
            value: closeTimeMedian,
            sub: "Median time to resolve",
            color: "text-cyan-300",
            icon: Clock,
          },
        ]
      : []),
    ...(staleCount > 0
      ? [
          {
            label: "Aging Backlog (>90d)",
            value: staleCount.toLocaleString(),
            sub: "Open longer than 3 months",
            color: "text-rose-400",
            icon: AlertCircle,
          },
        ]
      : []),
    ...(medianComments != null
      ? [
          {
            label: "Comments per Issue",
            value: medianComments.toFixed(1),
            sub: "Typical discussion depth",
            color: "text-amber-300",
            icon: MessageCircle,
          },
        ]
      : []),
  ];

  // Category distribution for AI summary bar
  const categoryCounts = classifiedData?.category_counts || {};
  const categoryBreakdown = Object.entries(CATEGORY_META)
    .map(([key, meta]) => ({
      key,
      label: meta.label,
      count: categoryCounts[key] ?? 0,
      color: meta.color,
      badgeClass: meta.badgeClass,
      icon: meta.icon,
    }))
    .filter((c) => c.count > 0)
    .sort((a, b) => b.count - a.count);

  const visibleItems = (classifiedData?.items || []).filter(
    (item) => categoryFilter === "all" || item.predicted_category === categoryFilter
  );

  return (
    <div className="space-y-8">
      {/* 1. Dynamic KPI Row (ZERO N/A!) */}
      <div
        className={`grid grid-cols-2 ${
          kpiCards.length >= 4
            ? "sm:grid-cols-4"
            : kpiCards.length === 3
            ? "sm:grid-cols-3"
            : "sm:grid-cols-2"
        } gap-4`}
      >
        {kpiCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <div key={i} className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Icon className="w-4 h-4 text-indigo-400" />
                <span>{card.label}</span>
              </div>
              <div className={`text-2xl font-bold mt-1.5 ${card.color}`}>
                {card.value}
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">{card.sub}</div>
            </div>
          );
        })}
      </div>

      {/* 2. Automatic AI-Classified Open Issues Section */}
      <section className="space-y-5 rounded-2xl border border-indigo-500/30 bg-slate-900/70 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex flex-wrap items-center gap-2">
                <span>Automatically Classified GitHub Issues</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  Powered by ML / NLP
                </span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Every fetched issue is automatically sorted into Bug, Feature Request, Documentation, Question, or Maintenance
              </p>
            </div>
          </div>

          {/* Open / Closed / All Switch */}
          <div className="flex items-center gap-1 p-1 rounded-xl bg-slate-950 border border-slate-800 text-xs shrink-0">
            {(
              [
                { id: "open", label: `Open (${classifiedData?.total_open ?? openCount})` },
                { id: "closed", label: `Closed (${classifiedData?.total_closed ?? totalClosed})` },
                { id: "all", label: "All" },
              ] as const
            ).map((s) => (
              <button
                key={s.id}
                onClick={() => {
                  setStateFilter(s.id);
                  setCategoryFilter("all");
                }}
                className={`px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
                  stateFilter === s.id
                    ? "bg-indigo-600 text-white font-semibold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Loading state */}
        {isLoadingIssues && (
          <div className="py-12 flex flex-col items-center justify-center gap-3 text-slate-400 text-xs">
            <Loader2 className="w-6 h-6 animate-spin text-indigo-400" />
            <span>Running AI classifier across repository issues...</span>
          </div>
        )}

        {/* Error state */}
        {issuesError && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs">
            {issuesError}
          </div>
        )}

        {/* Category Breakdown Pills + Filter */}
        {!isLoadingIssues && classifiedData && classifiedData.items.length > 0 && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => setCategoryFilter("all")}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition-all cursor-pointer ${
                  categoryFilter === "all"
                    ? "bg-indigo-600 text-white border-indigo-500"
                    : "bg-slate-950/70 text-slate-300 border-slate-800 hover:border-slate-700"
                }`}
              >
                All Categories ({classifiedData.items.length})
              </button>

              {categoryBreakdown.map((cat) => {
                const Icon = cat.icon;
                const active = categoryFilter === cat.key;
                const pct = Math.round((cat.count / classifiedData.items.length) * 100);
                return (
                  <button
                    key={cat.key}
                    onClick={() => setCategoryFilter(active ? "all" : cat.key)}
                    className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all cursor-pointer ${
                      active
                        ? "bg-indigo-600 text-white border-indigo-500"
                        : `${cat.badgeClass} hover:brightness-110`
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{cat.label}</span>
                    <span className="font-mono font-bold">
                      {cat.count} ({pct}%)
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Classified Issues List */}
            <div className="divide-y divide-slate-800/70 rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden max-h-[560px] overflow-y-auto">
              {visibleItems.map((item) => {
                const meta = CATEGORY_META[item.predicted_category] || CATEGORY_META.other;
                const Icon = meta.icon;
                const confPct = Math.round((item.confidence || 0) * 100);

                return (
                  <div
                    key={item.issue_number}
                    className="p-4 hover:bg-slate-900/70 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                  >
                    <div className="space-y-1.5 min-w-0 flex-1">
                      <div className="flex items-start gap-2">
                        <span className="text-xs font-mono text-slate-500 shrink-0 mt-0.5">
                          #{item.issue_number}
                        </span>
                        <a
                          href={item.html_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm font-medium text-slate-100 hover:text-indigo-300 transition-colors line-clamp-2 inline-flex items-center gap-1.5"
                        >
                          <span>{item.title}</span>
                          <ExternalLink className="w-3 h-3 shrink-0 opacity-60" />
                        </a>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                        <span>by @{item.author}</span>
                        {item.created_at && <span>&middot; {formatRelativeDate(item.created_at)}</span>}
                        {item.comments_count > 0 && (
                          <span className="inline-flex items-center gap-1">
                            &middot; <MessageCircle className="w-3 h-3" /> {item.comments_count}
                          </span>
                        )}
                        {item.github_labels.slice(0, 3).map((lbl, idx) => (
                          <span
                            key={idx}
                            className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/70 text-[10px]"
                          >
                            {lbl}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* AI Predicted Category Badge */}
                    <div className="flex items-center gap-2 shrink-0">
                      <span
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${meta.badgeClass}`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        <span>{meta.label}</span>
                        {confPct > 0 && (
                          <span className="text-[10px] opacity-80 font-mono">{confPct}%</span>
                        )}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {!isLoadingIssues && classifiedData && classifiedData.items.length === 0 && (
          <div className="p-6 text-center text-xs text-slate-400 bg-slate-950/50 rounded-xl border border-slate-800">
            No {stateFilter === "all" ? "" : stateFilter} issues found in this snapshot.
          </div>
        )}
      </section>

      {/* 3. Auto-Discovered AI Topic Themes (Sentence-Transformers + KMeans, shown only when topics exist) */}
      {topicsData && topicsData.topics.length > 0 && (
        <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>Main Recurring Themes in Issues</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  Deep Learning Clustering
                </span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Automatically grouped similar issues using neural sentence embeddings
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {topicsData.topics.map((topic) => (
              <div
                key={topic.cluster_id}
                className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold text-white truncate">
                    {topic.title}
                  </h3>
                  <span className="shrink-0 text-xs font-mono px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300">
                    {topic.total_count} issues
                  </span>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {topic.keywords.slice(0, 5).map((kw, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 text-[10px]"
                    >
                      <Tag className="w-2.5 h-2.5 text-cyan-400" />
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 4. Monthly Issue Opened vs Closed Chart (only when data exists) */}
      {chartData.length > 0 && (
        <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                <span>Issues Opened vs. Closed Over Time</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Shows whether maintainers are keeping pace with incoming issues
              </p>
            </div>

            <div className="flex items-center gap-3 text-xs">
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
                Opened: <strong className="text-white">{totalOpened.toLocaleString()}</strong>
              </span>
              <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
                Closed: <strong className="text-white">{totalClosed.toLocaleString()}</strong>
              </span>
            </div>
          </div>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={chartData}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
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
                <Bar dataKey="opened" name="Issues Opened" fill="#6366f1" radius={[4, 4, 0, 0]} opacity={0.85} />
                <Bar dataKey="closed" name="Issues Closed" fill="#10b981" radius={[4, 4, 0, 0]} opacity={0.85} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}
    </div>
  );
}
