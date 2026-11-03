"use client";

import { MessageSquare, Smile, User } from "lucide-react";
import { IssueSummary } from "@/lib/types";
import { formatDate, getCategoryBadge, getSpeedTierBadge } from "@/lib/utils";

interface IssueTableProps {
  issues: IssueSummary[];
  onSelectIssue: (issue: IssueSummary) => void;
  isLoading?: boolean;
}

export default function IssueTable({
  issues,
  onSelectIssue,
  isLoading = false,
}: IssueTableProps) {
  if (isLoading) {
    return (
      <div className="glass-panel rounded-2xl p-12 border border-slate-800 text-center space-y-3">
        <div className="w-8 h-8 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin mx-auto" />
        <p className="text-xs text-slate-400">Loading issues...</p>
      </div>
    );
  }

  if (issues.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-12 border border-slate-800 text-center space-y-2">
        <h4 className="text-sm font-semibold text-white">No matching issues</h4>
        <p className="text-xs text-slate-400">
          Try clearing some filters or searching for different keywords.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-2xl border border-slate-800 bg-slate-900/60 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="text-[11px] uppercase tracking-wider text-slate-400 bg-slate-950/60 border-b border-slate-800">
            <tr>
              <th scope="col" className="py-3 px-4 w-16">#</th>
              <th scope="col" className="py-3 px-4 min-w-[300px]">Title</th>
              <th scope="col" className="py-3 px-3 text-center">State</th>
              <th scope="col" className="py-3 px-3 text-center">Category</th>
              <th scope="col" className="py-3 px-3 text-center">Velocity</th>
              <th scope="col" className="py-3 px-3 text-center">Author</th>
              <th scope="col" className="py-3 px-3 text-center">Activity</th>
              <th scope="col" className="py-3 px-4 text-right">Created</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/60">
            {issues.map((issue) => {
              const categoryBadge = getCategoryBadge(issue.canonical_label);
              const speedBadge = getSpeedTierBadge(issue.speed_tier);
              const isOpen = issue.state === "open";

              return (
                <tr
                  key={issue.id}
                  onClick={() => onSelectIssue(issue)}
                  className="hover:bg-slate-800/50 cursor-pointer transition-colors group"
                >
                  {/* Issue Number */}
                  <td className="py-3.5 px-4 font-mono text-[11px] text-slate-500 group-hover:text-indigo-400">
                    #{issue.issue_number}
                  </td>

                  {/* Title */}
                  <td className="py-3.5 px-4 font-medium text-slate-200 group-hover:text-white transition-colors">
                    <span className="line-clamp-1">{issue.title}</span>
                  </td>

                  {/* State Badge */}
                  <td className="py-3.5 px-3 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 text-[10px] font-semibold rounded-md border ${
                        isOpen
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
                          : "bg-purple-500/10 text-purple-400 border-purple-500/25"
                      }`}
                    >
                      {isOpen ? "Open" : "Closed"}
                    </span>
                  </td>

                  {/* Category Badge */}
                  <td className="py-3.5 px-3 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 text-[10px] font-semibold border rounded-md ${categoryBadge.className}`}
                    >
                      {categoryBadge.label}
                    </span>
                  </td>

                  {/* Velocity Speed Tier */}
                  <td className="py-3.5 px-3 text-center">
                    <span
                      className={`inline-block px-2 py-0.5 text-[10px] font-semibold border rounded-md ${speedBadge.className}`}
                    >
                      {speedBadge.label}
                    </span>
                  </td>

                  {/* Author */}
                  <td className="py-3.5 px-3 text-center">
                    <span className="inline-flex items-center gap-1 text-[11px] text-slate-400 truncate max-w-[100px]">
                      <User className="w-3 h-3 text-slate-500 shrink-0" />
                      <span className="truncate">{issue.author}</span>
                    </span>
                  </td>

                  {/* Activity (Comments / Reactions) */}
                  <td className="py-3.5 px-3 text-center">
                    <div className="inline-flex items-center gap-2 text-[11px] text-slate-400">
                      <span className="flex items-center gap-0.5" title="Comments">
                        <MessageSquare className="w-3 h-3 text-slate-500" />
                        {issue.comments_count}
                      </span>
                      {issue.reactions_count > 0 && (
                        <span className="flex items-center gap-0.5 text-amber-400/80" title="Reactions">
                          <Smile className="w-3 h-3" />
                          {issue.reactions_count}
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Created Date */}
                  <td className="py-3.5 px-4 text-right text-[11px] text-slate-500 whitespace-nowrap">
                    {formatDate(issue.created_at)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
