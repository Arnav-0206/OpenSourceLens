"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  Calendar,
  CheckCircle2,
  ChevronDown,
  Code,
  Download,
  ExternalLink,
  FileText,
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
import { downloadReport, triggerRecompute } from "@/lib/api";

interface WorkspaceHeaderProps {
  data: AnalysisLatestResponse;
  onRefresh: () => void;
}

export default function WorkspaceHeader({ data, onRefresh }: WorkspaceHeaderProps) {
  const { repository, run, is_stale } = data;
  const [isRecomputing, setIsRecomputing] = useState(false);
  const [recomputeMsg, setRecomputeMsg] = useState<string | null>(null);
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [exportingFormat, setExportingFormat] = useState<string | null>(null);
  const [exportErr, setExportErr] = useState<string | null>(null);

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

  const handleExport = async (format: "markdown" | "json") => {
    const [owner, name] = repository.repo_name.split("/");
    setExportingFormat(format);
    setExportErr(null);
    setIsExportOpen(false);
    try {
      await downloadReport(owner, name, format);
    } catch (err: any) {
      setExportErr(err.message || "Failed to download intelligence report.");
    } finally {
      setExportingFormat(null);
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

              {/* Data Coverage Pill */}
              <div
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${
                  bandColors[confBand]
                }`}
              >
                {confBand === "high" || confBand === "medium" ? (
                  <ShieldCheck className="w-4 h-4" />
                ) : (
                  <ShieldAlert className="w-4 h-4" />
                )}
                <span>Data Coverage: {confScorePct}%</span>
              </div>

              {repository.is_archived && (
                <span className="px-2 py-0.5 rounded text-xs bg-amber-500/10 text-amber-300 border border-amber-500/20">
                  Archived
                </span>
              )}
            </div>

            {/* Surface Indicators */}
            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-400">
              {repository.primary_language && (
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-indigo-400" />
                  <strong className="text-slate-200">{repository.primary_language}</strong>
                </span>
              )}
              <span className="flex items-center gap-1">
                <Star className="w-3.5 h-3.5 text-amber-400" />
                {repository.stars_count.toLocaleString()} stars
              </span>
              <span className="flex items-center gap-1">
                <GitFork className="w-3.5 h-3.5 text-cyan-400" />
                {repository.forks_count.toLocaleString()} forks
              </span>
            </div>
          </div>

          {/* Action Tools */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Export Report Dropdown */}
            <div className="relative">
              <button
                onClick={() => setIsExportOpen(!isExportOpen)}
                disabled={!!exportingFormat}
                className="inline-flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-indigo-600/90 hover:bg-indigo-600 text-white border border-indigo-500/40 text-xs font-semibold transition-all cursor-pointer disabled:opacity-50 shadow-sm shadow-indigo-950"
                title="Download repository report"
              >
                {exportingFormat ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                <span>{exportingFormat ? `Exporting...` : "Export Report"}</span>
                <ChevronDown className="w-3 h-3 text-indigo-200" />
              </button>

              {isExportOpen && (
                <div className="absolute right-0 mt-2 w-64 rounded-xl border border-slate-700 bg-slate-900 shadow-2xl p-1.5 z-50 space-y-1">
                  <button
                    onClick={() => handleExport("markdown")}
                    className="w-full flex items-start gap-2.5 px-3 py-2 text-xs text-left rounded-lg text-slate-200 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
                  >
                    <FileText className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-semibold text-white">Summary Report (.md)</div>
                      <div className="text-[10px] text-slate-400">Readable Markdown report with insights &amp; tables</div>
                    </div>
                  </button>
                  <button
                    onClick={() => handleExport("json")}
                    className="w-full flex items-start gap-2.5 px-3 py-2 text-xs text-left rounded-lg text-slate-200 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
                  >
                    <Code className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-semibold text-white">Raw Data (.json)</div>
                      <div className="text-[10px] text-slate-400">Structured JSON metrics &amp; insights</div>
                    </div>
                  </button>
                </div>
              )}
            </div>

            <button
              onClick={() => handleRecompute(true)}
              disabled={isRecomputing}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 hover:text-white border border-slate-700 hover:border-slate-600 text-xs font-semibold transition-all cursor-pointer disabled:opacity-50"
              title="Re-run analytics on the latest collected data"
            >
              <RotateCw className={`w-3.5 h-3.5 ${isRecomputing ? "animate-spin text-indigo-400" : ""}`} />
              <span>{isRecomputing ? "Refreshing..." : "Refresh Analysis"}</span>
            </button>
          </div>
        </div>

        {/* Feedback Messages */}
        {exportErr && (
          <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs">
            {exportErr}
          </div>
        )}
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
