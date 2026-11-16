"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Database,
  ExternalLink,
  Flame,
  GitBranch,
  GitFork,
  GitPullRequest,
  Layers,
  Loader2,
  MessageSquare,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Star,
  Users,
  Zap,
} from "lucide-react";
import {
  IngestJobStatusResponse,
  IngestPreviewResponse,
} from "@/lib/types";
import {
  getAnalyticsStatus,
  getIngestionJobStatus,
  previewRepository,
  submitIngestion,
} from "@/lib/api";

const POPULAR_SUGGESTIONS = [
  { slug: "pallets/click", label: "pallets/click", desc: "Python CLI toolkit (Fast Snapshot)" },
  { slug: "psf/requests", label: "psf/requests", desc: "HTTP for Humans" },
  { slug: "pallets/flask", label: "pallets/flask", desc: "Lightweight WSGI framework" },
  { slug: "facebook/react", label: "facebook/react", desc: "Web UI Library" },
];

export default function RepoSearchFlow() {
  const router = useRouter();

  const [inputVal, setInputVal] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [previewData, setPreviewData] = useState<IngestPreviewResponse | null>(null);

  // Ingestion submission & polling
  const [selectedMode, setSelectedMode] = useState<"fast_snapshot" | "standard_diagnostic">("fast_snapshot");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const [jobStatus, setJobStatus] = useState<IngestJobStatusResponse | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [isPollingAnalytics, setIsPollingAnalytics] = useState(false);

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  // Poll active ingestion job
  useEffect(() => {
    if (!activeJobId) return;

    const pollJob = async () => {
      try {
        const status = await getIngestionJobStatus(activeJobId);
        setJobStatus(status);

        if (status.status === "completed" || status.status === "completed_partial") {
          // Stop ingestion polling, check analytics status
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setIsPollingAnalytics(true);
        } else if (status.status === "failed") {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setPollError(status.error_message || "Ingestion collection failed.");
        } else if (status.status === "rate_limited") {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setPollError(
            `GitHub API primary rate limit exceeded. Collection paused until ${
              status.resumes_at ? new Date(status.resumes_at).toLocaleTimeString() : "reset"
            }.`
          );
        }
      } catch (err: any) {
        setPollError(err.message || "Failed to poll collection job status.");
      }
    };

    pollJob();
    pollIntervalRef.current = setInterval(pollJob, 2500);

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [activeJobId]);

  // Once collection completes, poll analytics until latest run is published
  useEffect(() => {
    if (!isPollingAnalytics || !jobStatus) return;

    const [owner, name] = jobStatus.repo_name.split("/");
    let attempts = 0;

    const checkAnalytics = async () => {
      attempts++;
      try {
        const anaStatus = await getAnalyticsStatus(owner, name);
        if (anaStatus.has_analysis && anaStatus.status === "completed") {
          // Analytical snapshot is published! Redirect to workspace.
          router.push(`/repo/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`);
        } else if (anaStatus.status === "failed") {
          setPollError("Analytics computation failed after collection.");
          setIsPollingAnalytics(false);
        } else if (attempts > 30) {
          // Timeout after ~60s, redirect anyway (workspace handles in-progress state)
          router.push(`/repo/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`);
        }
      } catch {
        // Retry
        if (attempts > 30) {
          router.push(`/repo/${encodeURIComponent(owner)}/${encodeURIComponent(name)}`);
        }
      }
    };

    const interval = setInterval(checkAnalytics, 2000);
    return () => clearInterval(interval);
  }, [isPollingAnalytics, jobStatus, router]);

  const handlePreview = async (repoSlug: string) => {
    const trimmed = repoSlug.trim();
    if (!trimmed) {
      setErrorMsg("Please enter a valid GitHub repository URL or owner/repo slug.");
      return;
    }

    setErrorMsg(null);
    setIsPreviewing(true);
    setPreviewData(null);
    setPollError(null);
    setActiveJobId(null);
    setJobStatus(null);

    try {
      const result = await previewRepository(trimmed);
      setPreviewData(result);
      if (!result.is_eligible) {
        setErrorMsg(result.rejection_reason || "Repository does not meet eligibility criteria.");
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to evaluate repository eligibility. Check network or rate limits.");
    } finally {
      setIsPreviewing(false);
    }
  };

  const handleStartIngestion = async () => {
    if (!previewData || !previewData.is_eligible) return;

    setIsSubmitting(true);
    setPollError(null);

    try {
      const submitRes = await submitIngestion({
        repo: previewData.repo_name,
        mode: selectedMode,
      });
      setActiveJobId(submitRes.job_id);
    } catch (err: any) {
      setPollError(err.message || "Failed to submit ingestion job to the queue.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      {/* Search Input Box */}
      <div className="relative rounded-2xl border border-slate-800 bg-slate-900/90 p-4 sm:p-6 shadow-2xl backdrop-blur-xl">
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Universal Repository Intelligence Entry
        </label>
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
              <Search className="w-5 h-5" />
            </div>
            <input
              type="text"
              value={inputVal}
              onChange={(e) => {
                setInputVal(e.target.value);
                if (errorMsg) setErrorMsg(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") handlePreview(inputVal);
              }}
              placeholder="e.g. pallets/click or https://github.com/facebook/react"
              className="w-full pl-11 pr-4 py-3 bg-slate-950/80 border border-slate-700/80 rounded-xl text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all"
            />
          </div>

          <button
            onClick={() => handlePreview(inputVal)}
            disabled={isPreviewing || !inputVal.trim()}
            className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/25 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer whitespace-nowrap"
          >
            {isPreviewing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Evaluating...</span>
              </>
            ) : (
              <>
                <span>Inspect Repository</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

        {/* Suggestion Pills */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-slate-800/80 text-xs">
          <span className="text-slate-400 font-medium">Quick examples:</span>
          {POPULAR_SUGGESTIONS.map((item) => (
            <button
              key={item.slug}
              onClick={() => {
                setInputVal(item.slug);
                handlePreview(item.slug);
              }}
              className="px-2.5 py-1 rounded-lg bg-slate-800/60 hover:bg-slate-750 text-slate-300 hover:text-indigo-300 border border-slate-700/50 hover:border-indigo-500/30 transition-all font-mono"
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Inline Error */}
        {errorMsg && (
          <div className="mt-4 flex items-start gap-2.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* Pre-Flight Eligibility & Mode Selection Card */}
      {previewData && previewData.is_eligible && !activeJobId && (
        <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-b from-slate-900 via-slate-900/90 to-slate-950 p-6 shadow-2xl space-y-6 animate-in fade-in slide-in-from-top-4 duration-300">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2.5">
                <h2 className="text-xl font-bold text-white tracking-tight">
                  {previewData.repo_name}
                </h2>
                <a
                  href={previewData.canonical_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 hover:text-indigo-400 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Verified Public GitHub Repository &middot; Primary Language:{" "}
                <span className="text-slate-200 font-medium">
                  {previewData.metadata.primary_language || "N/A"}
                </span>
              </p>
            </div>

            {/* Quick Badges */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Eligible for Analysis</span>
              </span>
              {previewData.badges.map((b) => (
                <span
                  key={b}
                  className="px-2 py-0.5 rounded-md bg-slate-800 text-slate-300 text-xs border border-slate-700"
                >
                  {b}
                </span>
              ))}
            </div>
          </div>

          {/* Key Indicators */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 text-xs">
                <Star className="w-4 h-4 text-amber-400" />
                <span>Stargazers</span>
              </div>
              <div className="text-lg font-bold text-white mt-1">
                {(previewData.metadata.stars_count || 0).toLocaleString()}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 text-xs">
                <GitFork className="w-4 h-4 text-cyan-400" />
                <span>Forks</span>
              </div>
              <div className="text-lg font-bold text-white mt-1">
                {(previewData.metadata.forks_count || 0).toLocaleString()}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 text-xs">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                <span>Open Issues</span>
              </div>
              <div className="text-lg font-bold text-white mt-1">
                {(previewData.metadata.open_issues_count || 0).toLocaleString()}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-400 text-xs">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                <span>Confidence Preview</span>
              </div>
              <div className="text-lg font-bold text-indigo-300 mt-1 capitalize">
                {previewData.confidence_preview?.overall_confidence?.replace("_", " ") || "Adequate"}
              </div>
            </div>
          </div>

          {/* Collection Mode Selection */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Select Ingestion Depth & Rate Budget</h3>
              <span className="text-xs text-slate-400">Strict Rate-Budget Ceilings Guaranteed</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div
                onClick={() => setSelectedMode("fast_snapshot")}
                className={`p-4 rounded-xl border cursor-pointer transition-all ${
                  selectedMode === "fast_snapshot"
                    ? "border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50"
                    : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-amber-400" />
                    <span className="font-semibold text-white text-sm">Fast Snapshot</span>
                  </div>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
                    &le; 10 requests
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  Fastest turnaround (~10-15s). Harvests recent commits, PRs, issues, releases, and discussions for immediate repository diagnostics.
                </p>
              </div>

              <div
                onClick={() => setSelectedMode("standard_diagnostic")}
                className={`p-4 rounded-xl border cursor-pointer transition-all ${
                  selectedMode === "standard_diagnostic"
                    ? "border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50"
                    : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Compass className="w-4 h-4 text-cyan-400" />
                    <span className="font-semibold text-white text-sm">Standard Diagnostic</span>
                  </div>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
                    &le; 50 requests
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  Deep intelligence. Includes individual commit diffs, sampled PR review turnaround, inline comments, and full Discussions threads.
                </p>
              </div>
            </div>
          </div>

          {/* Action Trigger */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-800">
            <button
              onClick={() => setPreviewData(null)}
              className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleStartIngestion}
              disabled={isSubmitting}
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-600/30 cursor-pointer disabled:opacity-50"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Enqueuing Collection...</span>
                </>
              ) : (
                <>
                  <Flame className="w-4 h-4 text-amber-300" />
                  <span>Start Ingestion & Analysis</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Live Ingestion & Analytics Progress Tracker */}
      {activeJobId && (
        <div className="rounded-2xl border border-indigo-500/40 bg-slate-900/95 p-6 shadow-2xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white">
                  Ingesting: {previewData?.repo_name || jobStatus?.repo_name}
                </h3>
                <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  Job #{activeJobId}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {isPollingAnalytics
                  ? "Harvesting completed. Synthesizing multi-domain statistical intelligence..."
                  : `Harvesting GitHub streams in ${selectedMode.replace("_", " ")} mode...`}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
              </span>
              <span className="text-xs font-medium text-cyan-300 capitalize">
                {isPollingAnalytics ? "Computing Analytics" : (jobStatus?.status?.replace("_", " ") || "Queued")}
              </span>
            </div>
          </div>

          {/* Progress Stage Tracker */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-emerald-500/30 text-emerald-400">
              <CheckCircle2 className="w-5 h-5 mx-auto mb-1 text-emerald-400" />
              <div className="text-xs font-semibold">1. Pre-Flight Check</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Eligibility Verified</div>
            </div>

            <div className={`p-3 rounded-xl border ${
              jobStatus && jobStatus.status !== "pending"
                ? "bg-slate-950/80 border-emerald-500/30 text-emerald-400"
                : "bg-slate-950/40 border-slate-800 text-slate-500"
            }`}>
              {jobStatus && jobStatus.status !== "pending" ? (
                <CheckCircle2 className="w-5 h-5 mx-auto mb-1 text-emerald-400" />
              ) : (
                <Clock className="w-5 h-5 mx-auto mb-1 text-slate-500" />
              )}
              <div className="text-xs font-semibold">2. Worker Claim</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Lease Version Fenced</div>
            </div>

            <div className={`p-3 rounded-xl border ${
              jobStatus?.status === "completed" || jobStatus?.status === "completed_partial"
                ? "bg-slate-950/80 border-emerald-500/30 text-emerald-400"
                : "bg-slate-950/80 border-indigo-500/40 text-indigo-300"
            }`}>
              {jobStatus?.status === "completed" || jobStatus?.status === "completed_partial" ? (
                <CheckCircle2 className="w-5 h-5 mx-auto mb-1 text-emerald-400" />
              ) : (
                <Loader2 className="w-5 h-5 mx-auto mb-1 animate-spin text-indigo-400" />
              )}
              <div className="text-xs font-semibold">3. Stream Harvesting</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Commits, PRs, Discussions</div>
            </div>

            <div className={`p-3 rounded-xl border ${
              isPollingAnalytics
                ? "bg-slate-950/80 border-cyan-500/40 text-cyan-300"
                : "bg-slate-950/40 border-slate-800 text-slate-500"
            }`}>
              {isPollingAnalytics ? (
                <Loader2 className="w-5 h-5 mx-auto mb-1 animate-spin text-cyan-400" />
              ) : (
                <Cpu className="w-5 h-5 mx-auto mb-1 text-slate-500" />
              )}
              <div className="text-xs font-semibold">4. Analytics Engine</div>
              <div className="text-[10px] text-slate-400 mt-0.5">7-Domain Synthesis</div>
            </div>
          </div>

          {/* Harvested Counts Telemetry */}
          {jobStatus?.items_collected && Object.keys(jobStatus.items_collected).length > 0 && (
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-xs font-semibold text-slate-300 mb-2">Live Harvested Entities:</div>
              <div className="flex flex-wrap gap-4 text-xs font-mono text-slate-300">
                {Object.entries(jobStatus.items_collected).map(([k, v]) => (
                  <div key={k} className="flex items-center gap-1.5">
                    <span className="text-slate-500 capitalize">{k.replace("_", " ")}:</span>
                    <span className="text-indigo-400 font-bold">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Polling Error Alert */}
          {pollError && (
            <div className="flex items-start gap-2.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
              <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
              <div>
                <p className="font-semibold">Collection Notice</p>
                <p className="text-xs mt-0.5">{pollError}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
