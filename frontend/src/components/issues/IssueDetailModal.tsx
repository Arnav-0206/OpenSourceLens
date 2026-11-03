"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Clock,
  ExternalLink,
  MessageSquare,
  Search,
  Sparkles,
  Tag,
  User,
  X,
} from "lucide-react";
import { getIssue } from "@/lib/api";
import { IssueDetail, IssueSummary } from "@/lib/types";
import { formatDate, formatHours, getCategoryBadge, getSpeedTierBadge } from "@/lib/utils";

interface IssueDetailModalProps {
  issue: IssueSummary | null;
  owner: string;
  name: string;
  onClose: () => void;
}

export default function IssueDetailModal({
  issue,
  owner,
  name,
  onClose,
}: IssueDetailModalProps) {
  const [detail, setDetail] = useState<IssueDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    if (issue) {
      setLoading(true);
      getIssue(owner, name, issue.issue_number)
        .then((data) => {
          if (mounted) {
            setDetail(data);
            setLoading(false);
          }
        })
        .catch(() => {
          if (mounted) {
            setLoading(false);
          }
        });
    }

    return () => {
      mounted = false;
    };
  }, [issue, owner, name]);

  // Handle Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!issue) return null;

  const categoryBadge = getCategoryBadge(issue.canonical_label);
  const speedBadge = getSpeedTierBadge(issue.speed_tier);
  const isOpen = issue.state === "open";
  const bodyText = detail?.body || "No description provided for this issue.";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div
        className="glass-panel w-full max-w-3xl max-h-[90vh] rounded-3xl border border-slate-700 bg-slate-900/95 shadow-2xl flex flex-col overflow-hidden animate-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
      >
        {/* Modal Header */}
        <div className="p-6 border-b border-slate-800 flex items-start justify-between gap-4">
          <div className="space-y-2 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs text-indigo-400 font-semibold">
                {owner}/{name} #{issue.issue_number}
              </span>
              <span
                className={`px-2 py-0.5 text-[10px] font-semibold rounded-md border ${
                  isOpen
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
                    : "bg-purple-500/10 text-purple-400 border-purple-500/25"
                }`}
              >
                {isOpen ? "Open" : "Closed"}
              </span>
              <span
                className={`px-2 py-0.5 text-[10px] font-semibold rounded-md border ${categoryBadge.className}`}
              >
                {categoryBadge.label}
              </span>
              <span
                className={`px-2 py-0.5 text-[10px] font-semibold rounded-md border ${speedBadge.className}`}
              >
                {speedBadge.label}
              </span>
            </div>

            <h2 className="text-xl font-bold text-white tracking-tight leading-snug">
              {issue.title}
            </h2>

            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400 pt-1">
              <span className="flex items-center gap-1">
                <User className="w-3.5 h-3.5 text-slate-500" />
                <span className="font-medium text-slate-300">{issue.author}</span>
                {issue.author_association !== "NONE" && (
                  <span className="text-[10px] uppercase font-bold text-slate-500">
                    [{issue.author_association}]
                  </span>
                )}
              </span>
              <span>•</span>
              <span>Opened {formatDate(issue.created_at)}</span>
              {issue.closed_at && (
                <>
                  <span>•</span>
                  <span className="flex items-center gap-1 text-emerald-400 font-medium">
                    <Clock className="w-3 h-3" />
                    Closed {formatDate(issue.closed_at)}
                    {detail?.resolution_hours && ` (${formatHours(detail.resolution_hours)})`}
                  </span>
                </>
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close issue details modal"
            className="p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center shrink-0"
            title="Close modal (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Scrollable Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs text-slate-300">
          {/* Issue Description */}
          <div className="space-y-2">
            <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400">
              Description Body
            </h3>
            <div className="p-4 rounded-2xl bg-slate-950/80 border border-slate-800 font-mono text-[11px] leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto">
              {loading ? "Loading full description..." : bodyText}
            </div>
          </div>

          {/* Attached Labels */}
          {detail?.labels && detail.labels.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-xs uppercase font-bold tracking-wider text-slate-400 flex items-center gap-1.5">
                <Tag className="w-3.5 h-3.5 text-slate-400" />
                <span>Normalized GitHub Labels ({detail.labels.length})</span>
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {detail.labels.map((lbl) => (
                  <span
                    key={lbl.label_name}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium bg-slate-800/80 text-slate-300 border border-slate-700/80"
                  >
                    <span>{lbl.label_name}</span>
                    {lbl.canonical_category && (
                      <span className="text-indigo-400 text-[9px] uppercase font-bold">
                        → {lbl.canonical_category}
                      </span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer Actions */}
        <div className="p-4 sm:p-6 border-t border-slate-800 bg-slate-950/50 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Link
              href={`/analyzer?title=${encodeURIComponent(issue.title)}&repo=${encodeURIComponent(`${owner}/${name}`)}`}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition-colors"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Analyze in AI Pipeline</span>
            </Link>

            <Link
              href={`/analyzer#related?title=${encodeURIComponent(issue.title)}`}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs transition-colors border border-slate-700"
            >
              <Search className="w-3.5 h-3.5 text-cyan-400" />
              <span>Find Related Issues</span>
            </Link>
          </div>

          <a
            href={`https://github.com/${owner}/${name}/issues/${issue.issue_number}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
          >
            <span>View on GitHub</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </div>
  );
}
