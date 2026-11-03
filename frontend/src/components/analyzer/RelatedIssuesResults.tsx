"use client";

import React from "react";
import Link from "next/link";
import {
  Search,
  ExternalLink,
  Zap,
  Cpu,
  Layers,
  AlertTriangle,
  Sparkles,
  GitPullRequest,
  CheckCircle2,
  CircleDot,
  FileCode,
} from "lucide-react";
import { RelatedIssuesResponse, RelatedIssueMatch } from "@/lib/types";
import { getCategoryBadge } from "@/lib/utils";

export interface RelatedIssuesResultsProps {
  response: RelatedIssuesResponse | null;
  isLoading: boolean;
  error: string | null;
  isOfflineSimulated?: boolean;
  onAnalyzeIssue?: (title: string, repo: string) => void;
}

export default function RelatedIssuesResults({
  response,
  isLoading,
  error,
  isOfflineSimulated = false,
  onAnalyzeIssue,
}: RelatedIssuesResultsProps) {
  // 1. Loading State
  if (isLoading) {
    return (
      <div className="bg-slate-900/60 border border-cyan-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm space-y-4">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/5 to-transparent animate-pulse" />
        <div className="flex items-center justify-between">
          <div className="h-5 w-48 bg-slate-800 rounded animate-pulse" />
          <div className="h-5 w-24 bg-slate-800 rounded-full animate-pulse" />
        </div>
        <div className="space-y-3 pt-2">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2.5"
            >
              <div className="flex items-center justify-between">
                <div className="h-4 w-3/4 bg-slate-800 rounded animate-pulse" />
                <div className="h-4 w-16 bg-slate-800 rounded animate-pulse" />
              </div>
              <div className="h-3 w-1/2 bg-slate-800/60 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // 2. Error State
  if (error) {
    return (
      <div className="bg-rose-950/20 border border-rose-500/30 rounded-2xl p-6 text-rose-200 backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-rose-300">
              Semantic Retrieval Query Failed
            </h3>
            <p className="text-xs text-rose-200/80 mt-1 leading-relaxed">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  // 3. Awaiting Query
  if (!response) {
    return (
      <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl p-8 text-center">
        <div className="w-12 h-12 rounded-2xl bg-slate-800/60 text-slate-500 flex items-center justify-center mx-auto mb-3">
          <Search className="w-6 h-6" />
        </div>
        <h3 className="text-sm font-semibold text-slate-300">
          Awaiting Retrieval Query
        </h3>
        <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
          Enter an issue title or natural language symptom above, or select a sample query to retrieve nearest neighbors.
        </p>
      </div>
    );
  }

  // 4. No Matches Found
  if (response.matches.length === 0) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-8 text-center">
        <div className="w-12 h-12 rounded-2xl bg-amber-500/10 text-amber-400 flex items-center justify-center mx-auto mb-3">
          <Layers className="w-6 h-6" />
        </div>
        <h3 className="text-sm font-semibold text-slate-200">
          No Similar Issues Found Above Threshold
        </h3>
        <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
          No issues in the embeddings index exceeded the similarity threshold for query:
          <span className="block font-mono text-cyan-300 mt-1">&quot;{response.query_text || "query"}&quot;</span>
        </p>
        <p className="text-[11px] text-slate-500 mt-3">
          Try lowering the minimum similarity threshold slider or searching for broader terms.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-6">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <span>Retrieved Semantically Related Issues</span>
            </h3>
            <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              {response.total_matches} {response.total_matches === 1 ? "match" : "matches"}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Ranked by cosine similarity over 384-dimensional dense sentence embeddings.
          </p>
        </div>

        {/* Latency & Model Badges */}
        <div className="flex items-center gap-2 sm:self-center">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 font-mono">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>{response.retrieval_time_ms.toFixed(1)} ms</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>{response.model_version || "all-MiniLM-L6-v2"}</span>
          </div>

          {isOfflineSimulated && (
            <span className="px-2 py-1 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Offline Simulated
            </span>
          )}
        </div>
      </div>

      {/* Match Cards List */}
      <div className="space-y-3.5">
        {response.matches.map((match, idx) => {
          const simPct = Math.round(match.similarity_score * 1000) / 10;
          const isHigh = match.similarity_score >= 0.7;
          const isMedium = match.similarity_score >= 0.5 && match.similarity_score < 0.7;
          const categoryBadge = getCategoryBadge(match.canonical_label);
          const isOpen = match.state.toLowerCase() === "open";

          return (
            <div
              key={`${match.repo_name}-${match.issue_number}-${idx}`}
              className="group p-4 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700 transition-all hover:bg-slate-900/80"
            >
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                {/* Left: Issue Meta & Title */}
                <div className="space-y-1.5 flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    {/* Rank index */}
                    <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-400 text-[10px] font-mono font-bold flex items-center justify-center">
                      #{idx + 1}
                    </span>

                    {/* Repository Name & Issue Number */}
                    <span className="font-semibold text-slate-300">
                      {match.repo_name}
                      <span className="text-cyan-400 ml-1">#{match.issue_number}</span>
                    </span>

                    {/* State Badge */}
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
                        isOpen
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                          : "bg-purple-500/10 text-purple-400 border-purple-500/20"
                      }`}
                    >
                      {isOpen ? (
                        <CircleDot className="w-2.5 h-2.5" />
                      ) : (
                        <CheckCircle2 className="w-2.5 h-2.5" />
                      )}
                      <span>{match.state}</span>
                    </span>

                    {/* Category Badge */}
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold border ${categoryBadge.className}`}
                    >
                      {categoryBadge.label}
                    </span>
                  </div>

                  {/* Title */}
                  <h4 className="text-sm font-semibold text-white group-hover:text-cyan-300 transition-colors leading-snug">
                    {match.title}
                  </h4>
                </div>

                {/* Right: Similarity Score & Progress */}
                <div className="sm:text-right shrink-0 flex sm:flex-col items-center sm:items-end justify-between gap-2 sm:gap-1">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`text-xs font-bold font-mono px-2 py-0.5 rounded-lg border ${
                        isHigh
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                          : isMedium
                          ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/30"
                          : "bg-slate-800 text-slate-400 border-slate-700"
                      }`}
                    >
                      {simPct}% match
                    </span>
                  </div>

                  <div className="w-24 h-1.5 rounded-full bg-slate-800 overflow-hidden hidden sm:block">
                    <div
                      className={`h-full rounded-full ${
                        isHigh
                          ? "bg-emerald-400"
                          : isMedium
                          ? "bg-cyan-400"
                          : "bg-slate-500"
                      }`}
                      style={{ width: `${Math.max(simPct, 5)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Bottom Actions Row */}
              <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex items-center justify-between text-xs">
                <span className="text-[11px] text-slate-500">
                  Cosine score: <span className="font-mono text-slate-400">{match.similarity_score.toFixed(4)}</span>
                </span>

                <div className="flex items-center gap-3">
                  {onAnalyzeIssue && (
                    <button
                      type="button"
                      onClick={() => onAnalyzeIssue(match.title, match.repo_name)}
                      className="inline-flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 transition-colors"
                    >
                      <Sparkles className="w-3 h-3" />
                      <span>Classify this issue</span>
                    </button>
                  )}

                  <a
                    href={`https://github.com/${match.repo_name}/issues/${match.issue_number}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-white transition-colors"
                  >
                    <span>GitHub #{match.issue_number}</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
