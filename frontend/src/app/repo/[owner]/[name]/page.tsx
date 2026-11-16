"use client";

import React, { use, useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  GitCommit,
  GitPullRequest,
  Inbox,
  Layers,
  Loader2,
  MessageSquare,
  RefreshCw,
  Search,
  Sparkles,
  Users,
} from "lucide-react";
import {
  AnalysisLatestResponse,
  AnalysisPayload,
  AnalysisStatusResponse,
} from "@/lib/types";
import { getLatestAnalytics, getAnalyticsStatus, triggerRecompute } from "@/lib/api";
import WorkspaceHeader from "@/components/workspace/WorkspaceHeader";
import TabOverview from "@/components/workspace/TabOverview";
import TabActivity from "@/components/workspace/TabActivity";
import TabContributors from "@/components/workspace/TabContributors";
import TabPullRequests from "@/components/workspace/TabPullRequests";
import TabIssues from "@/components/workspace/TabIssues";
import TabDiscussions from "@/components/workspace/TabDiscussions";

interface PageProps {
  params: Promise<{
    owner: string;
    name: string;
  }>;
}

type TabKey =
  | "overview"
  | "activity"
  | "contributors"
  | "pull_requests"
  | "issues"
  | "discussions";

interface TabItem {
  id: TabKey;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string | number | null;
}

export default function RepoWorkspacePage({ params }: PageProps) {
  const resolvedParams = use(params);
  const { owner, name } = resolvedParams;

  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [latestData, setLatestData] = useState<AnalysisLatestResponse | null>(null);
  const [statusData, setStatusData] = useState<AnalysisStatusResponse | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const [isStartingRun, setIsStartingRun] = useState<boolean>(false);

  const fetchAnalytics = useCallback(async () => {
    try {
      setError(null);
      const data = await getLatestAnalytics(owner, name);
      setLatestData(data);
      setLoading(false);

      if (data.run.status === "pending" || data.run.status === "running") {
        setIsPolling(true);
      } else {
        setIsPolling(false);
      }
    } catch (err: any) {
      // If 404 or no analysis snapshot found, check status endpoint
      try {
        const stat = await getAnalyticsStatus(owner, name);
        setStatusData(stat);
        if (stat.status === "pending" || stat.status === "running") {
          setIsPolling(true);
        }
      } catch (statErr) {
        // Status also failed or repo not found
      }
      setError(err.message || "Failed to load repository analytics.");
      setLoading(false);
    }
  }, [owner, name]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  // Polling effect when an analysis run or ingestion is in progress
  useEffect(() => {
    if (!isPolling) return;

    const interval = setInterval(async () => {
      try {
        const stat = await getAnalyticsStatus(owner, name);
        setStatusData(stat);

        if (stat.status === "completed") {
          setIsPolling(false);
          await fetchAnalytics();
        } else if (stat.status === "failed") {
          setIsPolling(false);
          setError(
            stat.current_run?.error_message ||
              stat.failure_category ||
              "Analysis run failed."
          );
        }
      } catch (e) {
        // Continue polling until timeout or completion
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [isPolling, owner, name, fetchAnalytics]);

  const handleStartAnalysis = async () => {
    setIsStartingRun(true);
    setError(null);
    try {
      await triggerRecompute(owner, name, { force: true });
      setIsPolling(true);
      // Wait briefly then re-check status
      setTimeout(() => {
        fetchAnalytics();
        setIsStartingRun(false);
      }, 1500);
    } catch (err: any) {
      setError(err.message || "Failed to trigger analysis run.");
      setIsStartingRun(false);
    }
  };

  const payload: AnalysisPayload | undefined | null =
    latestData?.run?.metrics_payload;

  const discussionCount =
    payload?.repository_synthesis?.discussions_intelligence?.total_discussions;

  const tabs: TabItem[] = [
    { id: "overview", label: "Overview & Scorecard", icon: Layers },
    {
      id: "activity",
      label: "Development Activity",
      icon: GitCommit,
      badge: payload?.development_activity?.summary?.total_commits,
    },
    {
      id: "contributors",
      label: "Contributor Ecosystem",
      icon: Users,
      badge: payload?.contributor_ecosystem?.total_contributors,
    },
    {
      id: "pull_requests",
      label: "Pull Requests",
      icon: GitPullRequest,
      badge: payload?.pull_request_intelligence?.total_prs,
    },
    {
      id: "issues",
      label: "Issues & Backlog",
      icon: Inbox,
      badge: payload?.issue_intelligence?.total_issues,
    },
    {
      id: "discussions",
      label: "Discussions",
      icon: MessageSquare,
      badge:
        discussionCount !== undefined && discussionCount > 0
          ? discussionCount
          : undefined,
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-slate-100 px-3 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700 bg-slate-900/50 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Search Repositories</span>
            </Link>

            <div className="h-4 w-px bg-slate-800" />

            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono">
              <span className="text-slate-500">repo /</span>
              <span className="text-slate-300 font-semibold">{owner}</span>
              <span className="text-slate-500">/</span>
              <span className="text-indigo-400 font-semibold">{name}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {isPolling && (
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 animate-pulse">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Crunching Snapshot...</span>
              </span>
            )}
            <button
              onClick={() => {
                setLoading(true);
                fetchAnalytics();
              }}
              disabled={loading || isPolling}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white bg-slate-900 border border-slate-800 hover:border-slate-700 transition-colors disabled:opacity-50 cursor-pointer"
              title="Refresh Workspace"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Loading Spinner */}
        {loading && !latestData && (
          <div className="py-24 flex flex-col items-center justify-center text-center space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 animate-spin">
              <Loader2 className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-200">
                Loading Intelligence Workspace...
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Retrieving multi-domain telemetry and analytical metrics for{" "}
                <span className="font-mono text-slate-300">
                  {owner}/{name}
                </span>
              </p>
            </div>
          </div>
        )}

        {/* Polling / In-Progress State when no snapshot is completed yet */}
        {!latestData && isPolling && (
          <div className="p-8 rounded-2xl bg-slate-900/60 border border-indigo-500/30 text-center max-w-xl mx-auto my-12">
            <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto mb-4 text-indigo-400">
              <Loader2 className="w-8 h-8 animate-spin" />
            </div>
            <h3 className="text-xl font-bold text-slate-100 mb-2">
              Computation in Progress
            </h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-6">
              The analytical engine is currently processing commits, contributor
              networks, pull requests, issues, and discussions for{" "}
              <strong className="text-slate-200">
                {owner}/{name}
              </strong>
              . This page updates automatically upon completion.
            </p>
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-slate-300 text-xs font-mono">
              <span>Status: {statusData?.status || "running"}</span>
              {(statusData?.current_run?.requested_source_revision ??
                statusData?.repository?.analytics_source_revision) && (
                <span>
                  &bull; Rev #
                  {statusData?.current_run?.requested_source_revision ??
                    statusData?.repository?.analytics_source_revision}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Not Found / Needs Analysis State */}
        {!loading && !latestData && !isPolling && error && (
          <div className="p-8 rounded-2xl bg-slate-900/60 border border-slate-800 text-center max-w-xl mx-auto my-12">
            <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center mx-auto mb-4 text-amber-400">
              <AlertCircle className="w-8 h-8" />
            </div>
            <h3 className="text-xl font-bold text-slate-100 mb-2">
              No Analysis Snapshot Available
            </h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-6">
              {error ||
                `We couldn't locate an existing analytical run for ${owner}/${name}. The repository may still need data collection or snapshot computation.`}
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <button
                onClick={handleStartAnalysis}
                disabled={isStartingRun}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all cursor-pointer disabled:opacity-50"
              >
                {isStartingRun ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                <span>Run Analysis Now</span>
              </button>
              <Link
                href="/"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors"
              >
                <Search className="w-4 h-4" />
                <span>Return to Search</span>
              </Link>
            </div>
          </div>
        )}

        {/* Loaded Workspace View */}
        {latestData && (
          <div className="space-y-6">
            {/* Header with identity, confidence band, and recompute actions */}
            <WorkspaceHeader data={latestData} onRefresh={fetchAnalytics} />

            {/* Tab Navigation Pill Bar */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-slate-800/80 no-scrollbar">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium whitespace-nowrap transition-all cursor-pointer ${
                      isActive
                        ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/25"
                        : "bg-slate-900/60 text-slate-400 hover:text-slate-200 hover:bg-slate-850 border border-slate-800/80"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                    {tab.badge !== undefined && tab.badge !== null && (
                      <span
                        className={`px-1.5 py-0.5 rounded-full text-[10px] font-mono ${
                          isActive
                            ? "bg-indigo-700/60 text-indigo-100"
                            : "bg-slate-800 text-slate-400"
                        }`}
                      >
                        {typeof tab.badge === "number"
                          ? tab.badge.toLocaleString()
                          : tab.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Active Tab View */}
            <div className="pt-2">
              {activeTab === "overview" && (
                <TabOverview
                  payload={payload}
                  confidence={latestData.run?.confidence_summary}
                  rules={latestData.run?.rule_interpretations}
                />
              )}

              {activeTab === "activity" && (
                <TabActivity
                  activity={payload?.development_activity}
                  releases={payload?.repository_synthesis?.release_evolution}
                />
              )}

              {activeTab === "contributors" && (
                <TabContributors
                  contributors={payload?.contributor_ecosystem}
                />
              )}

              {activeTab === "pull_requests" && (
                <TabPullRequests
                  pullRequests={payload?.pull_request_intelligence}
                />
              )}

              {activeTab === "issues" && (
                <TabIssues
                  issues={payload?.issue_intelligence}
                  owner={owner}
                  name={name}
                />
              )}

              {activeTab === "discussions" && (
                <TabDiscussions
                  discussions={
                    payload?.repository_synthesis?.discussions_intelligence
                  }
                />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
