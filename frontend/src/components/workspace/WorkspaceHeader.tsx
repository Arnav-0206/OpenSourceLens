"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  Calendar,
  CheckCircle2,
  ExternalLink,
  GitBranch,
  GitCommit,
  GitFork,
  Layers,
  Loader2,
  RefreshCw,
  RotateCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Star,
} from "lucide-react";
import {
  AnalysisLatestResponse,
  AnalysisRecomputeResponse,
} from "@/lib/types";
import { triggerRecompute } from "@/lib/api";

interface WorkspaceHeaderProps {
  data: AnalysisLatestResponse;
  onRefresh: () => void;
}

export default function WorkspaceHeader({ data, onRefresh }: WorkspaceHeaderProps) {
  const { repository, run, is_stale } = data;
  const [isRecomputing, setIsRecomputing] = useState(false);
  const [recomputeMsg, setRecomputeMsg] = useState<string | null>(null);

  const confidence = run.confidence_summary;
  const confBand = confidence?.band || "unreliable";
  const confScorePct = Math.round((confidence?.score ?? 0) * 100);

  const bandColors = {
    high: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    medium: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
    low: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    unreliable: "bg-rose-500/10 text-rose-400 border-rose-500/30",
  };

  const handleRecompute = async (force: boolean = true) => {
    const [owner, name] = repository.repo_name.split("/");
    setIsRecomputing(true);
    setRecomputeMsg(null);
    try {
      const res: AnalysisRecomputeResponse = await triggerRecompute(owner, name, { force });
      setRecomputeMsg(res.message);
      // Wait briefly then refresh
      setTimeout(() => {
        onRefresh();
      }, 2000);
    } catch (err: any) {
      setRecomputeMsg(err.message || "Failed to trigger analysis recomputation.");
    } finally {
      setIsRecomputing(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Banner Card */}
      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 p-6 backdrop-blur-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          {/* Repository Identity */}
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                {repository.repo_name}
              </h1>
              <a
                href={`https://github.com/${repository.repo_name}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-400 hover:text-indigo-400 transition-colors"
                title="View on GitHub"
              >
                <ExternalLink className="w-5 h-5" />
              </a>

              {/* Confidence Band Pill */}
              <div
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold capitalize ${
                  bandColors[confBand]
                }`}
              >
                {confBand === "high" || confBand === "medium" ? (
                  <ShieldCheck className="w-4 h-4" />
                ) : (
                  <ShieldAlert className="w-4 h-4" />
                )}
                <span>
                  {confBand} Confidence ({confScorePct}%)
                </span>
              </div>

              {/* Methodology Pill */}
              <span className="px-2.5 py-1 rounded-full text-xs font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                Methodology {run.methodology_version || "2026.2"}
              </span>

              {repository.is_archived && (
                <span className="px-2 py-0.5 rounded text-xs bg-amber-500/10 text-amber-300 border border-amber-500/20">
                  Archived
                </span>
              )}
            </div>

            {/* Surface Indicators */}
            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-indigo-400" />
                Language: <strong className="text-slate-200">{repository.primary_language || "N/A"}</strong>
              </span>
              <span className="flex items-center gap-1">
                <Star className="w-3.5 h-3.5 text-amber-400" />
                {repository.stars_count.toLocaleString()} stars
              </span>
              <span className="flex items-center gap-1">
                <GitFork className="w-3.5 h-3.5 text-cyan-400" />
                {repository.forks_count.toLocaleString()} forks
              </span>
              <span className="flex items-center gap-1 font-mono text-[11px] text-slate-500">
                Source Rev #{repository.analytics_source_revision} &middot; Snapshot Rev #{run.analytics_source_revision}
              </span>
            </div>
          </div>

          {/* Action Tools */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => handleRecompute(true)}
              disabled={isRecomputing}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 hover:text-white border border-slate-700 hover:border-slate-600 text-xs font-semibold transition-all cursor-pointer disabled:opacity-50"
              title="Force full recomputation of analytical snapshot"
            >
              <RotateCw className={`w-3.5 h-3.5 ${isRecomputing ? "animate-spin text-indigo-400" : ""}`} />
              <span>{isRecomputing ? "Recomputing..." : "Recompute Snapshot"}</span>
            </button>
          </div>
        </div>

        {/* Recompute Alert Message */}
        {recomputeMsg && (
          <div className="mt-4 p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs">
            {recomputeMsg}
          </div>
        )}
      </div>

      {/* Stale Revision Alert Banner */}
      {is_stale && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/25 text-amber-300 text-xs">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>
              <strong>Stale Analytics Snapshot:</strong> Repository source data has advanced to revision #
              {repository.analytics_source_revision} (current analysis reflects revision #{run.analytics_source_revision}).
            </span>
          </div>
          <button
            onClick={() => handleRecompute(false)}
            disabled={isRecomputing}
            className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 font-semibold transition-colors cursor-pointer self-start sm:self-auto whitespace-nowrap"
          >
            Update to Revision #{repository.analytics_source_revision}
          </button>
        </div>
      )}
    </div>
  );
}
