"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from "recharts";
import {
  MessageSquare,
  CheckCircle2,
  HelpCircle,
  Clock,
  Users,
  ThumbsUp,
  MessageCircle,
  Sparkles,
  Info,
  Layers,
  ArrowUpRight,
} from "lucide-react";
import { DiscussionsIntelligenceData } from "@/lib/types";

interface TabDiscussionsProps {
  discussions?: DiscussionsIntelligenceData | null;
}

const CATEGORY_COLORS = [
  "#3b82f6", // blue
  "#10b981", // emerald
  "#8b5cf6", // purple
  "#f59e0b", // amber
  "#ec4899", // pink
  "#06b6d4", // cyan
  "#6366f1", // indigo
  "#14b8a6", // teal
];

function formatHours(hours?: number | null): string {
  if (hours === undefined || hours === null) return "N/A";
  if (hours < 1) return `${Math.round(hours * 60)} min`;
  if (hours < 48) return `${hours.toFixed(1)} hrs`;
  const days = hours / 24;
  return `${days.toFixed(1)} days`;
}

export default function TabDiscussions({ discussions }: TabDiscussionsProps) {
  const isEnabled =
    discussions &&
    discussions.status !== "not_enabled" &&
    discussions.status !== "disabled" &&
    discussions.total_discussions > 0;

  if (!isEnabled) {
    return (
      <div className="space-y-6">
        <div className="p-8 rounded-2xl bg-neutral-900/60 border border-neutral-800 text-center max-w-2xl mx-auto my-12">
          <div className="w-16 h-16 rounded-2xl bg-neutral-800/80 border border-neutral-700/60 flex items-center justify-center mx-auto mb-4 text-neutral-400">
            <MessageSquare className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-semibold text-neutral-200 mb-2">
            GitHub Discussions Not Enabled or Active
          </h3>
          <p className="text-neutral-400 text-sm leading-relaxed mb-6">
            This repository does not have public GitHub Discussions enabled, or no discussion threads have been created yet. GitHub Discussions is an optional repository feature for community Q&amp;A, RFCs, and open dialogue.
          </p>
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-neutral-800 text-neutral-300 text-xs font-mono border border-neutral-700">
            <Info className="w-3.5 h-3.5 text-blue-400" />
            Status: {discussions?.status ?? "not_enabled"}
          </div>
        </div>
      </div>
    );
  }

  const qna = discussions.qna;
  const engagement = discussions.engagement;
  const categoriesRecord = discussions.categories || {};
  const categoryData = Object.entries(categoriesRecord)
    .map(([name, val]) => ({
      name,
      count: typeof val === "number" ? val : (val as { count?: number })?.count ?? 0,
    }))
    .sort((a, b) => b.count - a.count);

  const answeredRatePct =
    qna?.qna_answered_rate !== undefined && qna?.qna_answered_rate !== null
      ? (qna.qna_answered_rate * 100).toFixed(1)
      : null;

  const overlapRatePct =
    engagement?.contributor_overlap_rate !== undefined &&
    engagement?.contributor_overlap_rate !== null
      ? (engagement.contributor_overlap_rate * 100).toFixed(1)
      : null;

  return (
    <div className="space-y-8">
      {/* 1. Header KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Threads */}
        <div className="p-5 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center justify-between text-neutral-400 text-xs mb-2">
            <span>Community Discussions</span>
            <MessageSquare className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-neutral-100 font-mono">
            {(discussions.total_discussions ?? 0).toLocaleString()}
          </div>
          <div className="text-xs text-neutral-400 mt-1 flex items-center gap-1.5">
            <span>{(discussions.total_comments ?? 0).toLocaleString()} total comments</span>
            <span>•</span>
            <span className="flex items-center gap-0.5 text-neutral-300">
              <ThumbsUp className="w-3 h-3 text-amber-400" />
              {discussions.total_upvotes ?? 0}
            </span>
          </div>
        </div>

        {/* Q&A Answered Rate */}
        <div className="p-5 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center justify-between text-neutral-400 text-xs mb-2">
            <span>Q&amp;A Resolution Rate</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 font-mono">
            {answeredRatePct !== null ? `${answeredRatePct}%` : "N/A"}
          </div>
          <div className="text-xs text-neutral-400 mt-1">
            {qna ? `${qna.answered_qna_threads} of ${qna.total_qna_threads} Q&A resolved` : "No Q&A category"}
          </div>
        </div>

        {/* Median Time to Answer */}
        <div className="p-5 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center justify-between text-neutral-400 text-xs mb-2">
            <span>Median Time to Answer</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-neutral-100 font-mono">
            {formatHours(qna?.time_to_answer_hours?.p50)}
          </div>
          <div className="text-xs text-neutral-400 mt-1">
            Mean: {formatHours(qna?.time_to_answer_hours?.mean)}
          </div>
        </div>

        {/* Contributor Bridge Overlap */}
        <div className="p-5 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center justify-between text-neutral-400 text-xs mb-2">
            <span>Contributor Overlap Rate</span>
            <Users className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-400 font-mono">
            {overlapRatePct !== null ? `${overlapRatePct}%` : "N/A"}
          </div>
          <div className="text-xs text-neutral-400 mt-1">
            {engagement?.contributor_overlap_count ?? 0} bridged of{" "}
            {engagement?.unique_discussion_authors ?? 0} authors
          </div>
        </div>
      </div>

      {/* 2. Middle Row: Category Distribution & Q&A Latency Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Category Breakdown (2 Cols) */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-semibold text-neutral-200">
                Category Distribution
              </h3>
              <p className="text-xs text-neutral-400">
                Volume of discussions across community categories
              </p>
            </div>
            {discussions.dominant_category && (
              <span className="px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-medium">
                Dominant: {discussions.dominant_category}
              </span>
            )}
          </div>

          {categoryData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={categoryData}
                  layout="vertical"
                  margin={{ top: 8, right: 24, left: 24, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" horizontal={false} />
                  <XAxis type="number" stroke="#737373" fontSize={11} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    stroke="#a3a3a3"
                    fontSize={11}
                    width={110}
                    tickFormatter={(val) => (val.length > 14 ? `${val.slice(0, 12)}…` : val)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#171717",
                      borderColor: "#404040",
                      borderRadius: "0.75rem",
                      fontSize: "12px",
                      color: "#e5e5e5",
                    }}
                    formatter={(val: any) => [`${val ?? 0} threads`, "Discussions"]}
                  />
                  <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                    {categoryData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={CATEGORY_COLORS[index % CATEGORY_COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center text-neutral-500 text-sm">
              No category data recorded.
            </div>
          )}
        </div>

        {/* Q&A Resolution & Latency Quantiles */}
        <div className="p-6 rounded-2xl bg-neutral-900/60 border border-neutral-800/80 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-semibold text-neutral-200 mb-1">
              Q&amp;A Resolution SLA
            </h3>
            <p className="text-xs text-neutral-400 mb-4">
              Time elapsed before an accepted answer is marked
            </p>

            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-neutral-800/40 border border-neutral-800 flex items-center justify-between">
                <span className="text-xs text-neutral-400">Total Q&amp;A Threads</span>
                <span className="text-sm font-mono font-semibold text-neutral-200">
                  {qna?.total_qna_threads ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-neutral-800/40 border border-neutral-800 flex items-center justify-between">
                <span className="text-xs text-neutral-400">Answered (Resolved)</span>
                <span className="text-sm font-mono font-semibold text-emerald-400">
                  {qna?.answered_qna_threads ?? 0}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-neutral-800/40 border border-neutral-800 flex items-center justify-between">
                <span className="text-xs text-neutral-400">Unanswered</span>
                <span className="text-sm font-mono font-semibold text-amber-400">
                  {qna?.unanswered_qna_threads ?? 0}
                </span>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-neutral-800/80">
            <div className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-2">
              Answer Latency Quantiles
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2 rounded-lg bg-neutral-800/30">
                <div className="text-[10px] text-neutral-500">P50 (Median)</div>
                <div className="text-xs font-mono font-semibold text-neutral-200 mt-0.5">
                  {formatHours(qna?.time_to_answer_hours?.p50)}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-neutral-800/30">
                <div className="text-[10px] text-neutral-500">P75</div>
                <div className="text-xs font-mono font-semibold text-neutral-200 mt-0.5">
                  {formatHours(qna?.time_to_answer_hours?.p75)}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-neutral-800/30">
                <div className="text-[10px] text-neutral-500">P90</div>
                <div className="text-xs font-mono font-semibold text-neutral-200 mt-0.5">
                  {formatHours(qna?.time_to_answer_hours?.p90)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Community Bridge & Top Engaged Discussions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Community Bridge Context Card */}
        <div className="p-6 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Sparkles className="w-4 h-4" />
            </div>
            <h3 className="text-base font-semibold text-neutral-200">
              Community Bridge
            </h3>
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed mb-4">
            Measures the proportion of discussion participants who also submit code, pull requests, or reviews. A lower overlap rate indicates a repository functioning as an open public support channel; a higher overlap indicates maintainer-dominated discussion.
          </p>

          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs py-2 border-b border-neutral-800/60">
              <span className="text-neutral-400">Unique Discussion Authors</span>
              <span className="font-mono text-neutral-200">
                {engagement?.unique_discussion_authors ?? 0}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs py-2 border-b border-neutral-800/60">
              <span className="text-neutral-400">Codebase Contributors Bridged</span>
              <span className="font-mono text-cyan-400 font-semibold">
                {engagement?.contributor_overlap_count ?? 0}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs py-2 border-b border-neutral-800/60">
              <span className="text-neutral-400">Mean Comments / Thread</span>
              <span className="font-mono text-neutral-200">
                {engagement?.mean_comments_per_discussion?.toFixed(1) ?? "N/A"}
              </span>
            </div>
            <div className="flex items-center justify-between text-xs py-2">
              <span className="text-neutral-400">Mean Upvotes / Thread</span>
              <span className="font-mono text-amber-400">
                {engagement?.mean_upvotes_per_discussion?.toFixed(1) ?? "N/A"}
              </span>
            </div>
          </div>
        </div>

        {/* Top Engaged Threads List */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-neutral-900/60 border border-neutral-800/80">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-semibold text-neutral-200">
                Top Engaged Discussion Threads
              </h3>
              <p className="text-xs text-neutral-400">
                Discussions with the highest community participation and comments
              </p>
            </div>
          </div>

          {engagement?.top_discussed_threads && engagement.top_discussed_threads.length > 0 ? (
            <div className="space-y-2.5">
              {engagement.top_discussed_threads.map((thread) => (
                <div
                  key={thread.discussion_number}
                  className="p-3.5 rounded-xl bg-neutral-800/30 border border-neutral-800/70 hover:border-neutral-700/80 transition-colors flex items-center justify-between gap-4"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-mono text-neutral-500">
                        #{thread.discussion_number}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-neutral-800 text-neutral-300 border border-neutral-700">
                        {thread.category}
                      </span>
                      {thread.is_answered && (
                        <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3 h-3" /> Answered
                        </span>
                      )}
                    </div>
                    <div className="text-sm font-medium text-neutral-200 truncate">
                      {thread.title}
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0 text-xs">
                    <span className="inline-flex items-center gap-1 text-neutral-400">
                      <MessageSquare className="w-3.5 h-3.5 text-blue-400" />
                      {thread.comments_count}
                    </span>
                    <span className="inline-flex items-center gap-1 text-neutral-400">
                      <ThumbsUp className="w-3.5 h-3.5 text-amber-400" />
                      {thread.upvote_count}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-40 flex items-center justify-center text-neutral-500 text-sm">
              No active discussion threads recorded.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
