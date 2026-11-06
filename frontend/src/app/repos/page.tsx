"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Clock,
  ExternalLink,
  Flame,
  GitPullRequest,
  Star,
  Terminal,
} from "lucide-react";
import RepoSearchInput from "@/components/repository/RepoSearchInput";
import { getAnalyticsComparison, getRepositories } from "@/lib/api";
import { RepositoryComparisonItem, RepositorySummary } from "@/lib/types";
import { formatHours, formatNumber, formatPercent } from "@/lib/utils";

interface DirectoryRepo {
  owner: string;
  name: string;
  repo_name: string;
  domain: string;
  description: string;
  stars_count: number;
  forks_count: number;
  total_issues: number;
  closed_issues: number;
  closure_rate: number;
  median_hours: number;
  rapid_pct: number;
  primary_language: string;
  color: string;
  border: string;
  accent: string;
}

type SortKey = "stars" | "median_hours" | "closure_rate" | "name";

export default function RepositoriesPage() {
  const [repos, setRepos] = useState<DirectoryRepo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("stars");

  useEffect(() => {
    let mounted = true;

    async function fetchLiveRepos() {
      try {
        setLoading(true);
        const [reposRes, compRes] = await Promise.allSettled([
          getRepositories(),
          getAnalyticsComparison(),
        ]);

        const liveList =
          reposRes.status === "fulfilled" && Array.isArray(reposRes.value)
            ? reposRes.value
            : [];
        const compList =
          compRes.status === "fulfilled" &&
          Array.isArray(compRes.value?.repositories)
            ? compRes.value.repositories
            : [];

        if (mounted && liveList.length > 0) {
          const updated: DirectoryRepo[] = liveList.map((r) => {
            const matchComp = compList.find(
              (c) => c.repo_name.toLowerCase() === r.repo_name.toLowerCase()
            );

            return {
              owner: r.owner,
              name: r.name,
              repo_name: r.repo_name,
              domain: matchComp?.dominant_category ? `${matchComp.dominant_category} focus` : "Software Engineering",
              description: r.description || "Public open-source repository.",
              stars_count: r.stars_count,
              forks_count: r.forks_count,
              total_issues: matchComp?.total_issues ?? r.open_issues_count,
              closed_issues: matchComp?.closed_issues ?? 0,
              closure_rate: matchComp?.closure_rate ?? 0,
              median_hours: matchComp?.median_resolution_hours ?? 0,
              rapid_pct:
                matchComp?.rapid_closure_percentage !== undefined
                  ? matchComp.rapid_closure_percentage / 100
                  : 0,
              primary_language: r.primary_language || "General",
              color: "from-indigo-500/15 via-slate-900 to-slate-950",
              border: "border-slate-800",
              accent: "text-indigo-400",
            };
          });
          setRepos(updated);
        }
      } catch {
        // Fallback gracefully on fetch errors
      } finally {
        if (mounted) setLoading(false);
      }
    }

    fetchLiveRepos();
    return () => {
      mounted = false;
    };
  }, []);

  const filteredAndSortedRepos = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    let result = repos.filter(
      (r) =>
        r.repo_name.toLowerCase().includes(q) ||
        r.domain.toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q) ||
        r.primary_language.toLowerCase().includes(q)
    );

    result.sort((a, b) => {
      switch (sortBy) {
        case "stars":
          return b.stars_count - a.stars_count;
        case "median_hours":
          return a.median_hours - b.median_hours;
        case "closure_rate":
          return b.closure_rate - a.closure_rate;
        case "name":
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    return result;
  }, [repos, searchQuery, sortBy]);

  return (
    <div className="space-y-8 pb-12">
      {/* Page Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
          <Boxes className="w-3.5 h-3.5" />
          <span>Repository Directory</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
          Analyzed Repositories
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-3xl leading-relaxed">
          Explore analytics across ingested open-source repositories.
          View resolution SLA percentiles, category breakdowns, and issue distributions.
        </p>
      </div>

      {/* Control Bar: Search & Sort */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 p-4 rounded-2xl glass-panel border border-slate-800">
        <RepoSearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          className="flex-1 max-w-lg"
        />

        <div className="flex items-center gap-3">
          <label htmlFor="sort-select" className="text-xs font-medium text-slate-400 whitespace-nowrap">
            Sort by:
          </label>
          <select
            id="sort-select"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortKey)}
            className="px-3 py-2 text-xs font-medium bg-slate-900 border border-slate-800 rounded-xl text-slate-200 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
          >
            <option value="stars">Stars (Highest First)</option>
            <option value="median_hours">Median SLA (Fastest First)</option>
            <option value="closure_rate">Closure Rate (Highest First)</option>
            <option value="name">Repository Name (A–Z)</option>
          </select>
        </div>
      </div>

      {/* Loading Skeleton */}
      {loading && repos.length === 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="glass-panel rounded-3xl p-6 sm:p-8 border border-slate-800 animate-pulse space-y-6">
              <div className="h-6 bg-slate-800 rounded w-1/3" />
              <div className="h-4 bg-slate-800/60 rounded w-2/3" />
              <div className="grid grid-cols-4 gap-3 h-16 bg-slate-900/50 rounded-2xl" />
            </div>
          ))}
        </div>
      )}

      {/* Repository Cards Grid */}
      {!loading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredAndSortedRepos.map((repo) => (
            <div
              key={repo.repo_name}
              className={`glass-panel glass-panel-hover rounded-3xl p-6 sm:p-8 border ${repo.border} bg-gradient-to-br ${repo.color} flex flex-col justify-between space-y-6`}
            >
              {/* Header */}
              <div>
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1">
                    <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-400">
                      {repo.domain}
                    </span>
                    <h2 className="text-2xl font-extrabold text-white tracking-tight">
                      <span className="text-slate-400">{repo.owner}/</span>
                      <span className={repo.accent}>{repo.name}</span>
                    </h2>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-slate-900/90 border border-slate-800 text-slate-200">
                      <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />
                      {formatNumber(repo.stars_count)}
                    </span>
                    <span className="px-2.5 py-1 text-[11px] font-medium bg-slate-900/60 border border-slate-800 rounded-lg text-slate-400">
                      {repo.primary_language}
                    </span>
                  </div>
                </div>

                <p className="text-xs sm:text-sm text-slate-300 mt-3 leading-relaxed">
                  {repo.description}
                </p>
              </div>

              {/* Metrics Snapshot Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-4 px-4 rounded-2xl bg-slate-950/70 border border-slate-800/80 text-center">
                <div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">
                    Total Issues
                  </div>
                  <div className="text-base font-bold text-white mt-0.5">
                    {repo.total_issues}
                  </div>
                </div>

                <div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">
                    Resolved
                  </div>
                  <div className="text-base font-bold text-cyan-300 mt-0.5">
                    {repo.closed_issues} ({formatPercent(repo.closure_rate)})
                  </div>
                </div>

                <div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">
                    Median SLA
                  </div>
                  <div className="text-base font-bold text-emerald-400 mt-0.5 flex items-center justify-center gap-1">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{formatHours(repo.median_hours)}</span>
                  </div>
                </div>

                <div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">
                    &lt; 24h Resolved
                  </div>
                  <div className="text-base font-bold text-indigo-300 mt-0.5 flex items-center justify-center gap-1">
                    <Flame className="w-3.5 h-3.5 text-indigo-400" />
                    <span>{formatPercent(repo.rapid_pct)}</span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800/80">
                <div className="flex items-center gap-3">
                  <a
                    href={`https://github.com/${repo.repo_name}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-slate-200 transition-colors"
                  >
                    <span>View on GitHub</span>
                    <ExternalLink className="w-3 h-3 ml-0.5" />
                  </a>
                </div>

                <Link
                  href="/analyzer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 text-xs font-medium border border-indigo-500/30 transition-colors"
                >
                  <Terminal className="w-3.5 h-3.5" />
                  <span>AI Issue Analyzer</span>
                  <ArrowRight className="w-3 h-3" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && filteredAndSortedRepos.length === 0 && (
        <div className="p-12 text-center rounded-3xl glass-panel border border-slate-800 space-y-3">
          <Boxes className="w-10 h-10 text-slate-500 mx-auto" />
          <h3 className="text-base font-semibold text-white">
            No matching repositories
          </h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Try adjusting your search query or clear the filter.
          </p>
          <button
            type="button"
            onClick={() => setSearchQuery("")}
            className="px-4 py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors"
          >
            Clear Search Filter
          </button>
        </div>
      )}
    </div>
  );
}
