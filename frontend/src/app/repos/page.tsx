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
import { getRepositories } from "@/lib/api";
import { RepositorySummary } from "@/lib/types";
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

const initialRepos: DirectoryRepo[] = [
  {
    owner: "microsoft",
    name: "vscode",
    repo_name: "microsoft/vscode",
    domain: "Developer Tools & IDE",
    description: "Visual Studio Code is a code editor redefined and optimized for building and debugging modern web and cloud applications.",
    stars_count: 165400,
    forks_count: 32000,
    total_issues: 582,
    closed_issues: 242,
    closure_rate: 0.416,
    median_hours: 7.7,
    rapid_pct: 0.698,
    primary_language: "TypeScript",
    color: "from-blue-500/15 via-indigo-500/5 to-slate-950",
    border: "border-blue-500/30",
    accent: "text-blue-400",
  },
  {
    owner: "pytorch",
    name: "pytorch",
    repo_name: "pytorch/pytorch",
    domain: "Machine Learning Framework",
    description: "PyTorch is an open-source machine learning library primarily used for applications such as computer vision and natural language processing.",
    stars_count: 85200,
    forks_count: 22000,
    total_issues: 582,
    closed_issues: 310,
    closure_rate: 0.533,
    median_hours: 53.1,
    rapid_pct: 0.394,
    primary_language: "Python / C++",
    color: "from-amber-500/15 via-orange-500/5 to-slate-950",
    border: "border-amber-500/30",
    accent: "text-amber-400",
  },
  {
    owner: "kubernetes",
    name: "kubernetes",
    repo_name: "kubernetes/kubernetes",
    domain: "Cloud Native Orchestration",
    description: "Production-grade container orchestration system automating software deployment, scaling, and operational management.",
    stars_count: 110300,
    forks_count: 39000,
    total_issues: 582,
    closed_issues: 380,
    closure_rate: 0.653,
    median_hours: 141.2,
    rapid_pct: 0.281,
    primary_language: "Go",
    color: "from-cyan-500/15 via-sky-500/5 to-slate-950",
    border: "border-cyan-500/30",
    accent: "text-cyan-400",
  },
  {
    owner: "huggingface",
    name: "transformers",
    repo_name: "huggingface/transformers",
    domain: "NLP & Foundation Models",
    description: "Transformers provides thousands of pretrained models to perform tasks on texts such as classification, extraction, and generation.",
    stars_count: 135800,
    forks_count: 26000,
    total_issues: 582,
    closed_issues: 381,
    closure_rate: 0.655,
    median_hours: 48.2,
    rapid_pct: 0.426,
    primary_language: "Python",
    color: "from-violet-500/15 via-purple-500/5 to-slate-950",
    border: "border-violet-500/30",
    accent: "text-violet-400",
  },
];

type SortKey = "stars" | "median_hours" | "closure_rate" | "name";

export default function RepositoriesPage() {
  const [repos, setRepos] = useState<DirectoryRepo[]>(initialRepos);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("stars");

  useEffect(() => {
    let mounted = true;

    async function fetchLiveRepos() {
      try {
        const liveList: RepositorySummary[] = await getRepositories();
        if (mounted && Array.isArray(liveList) && liveList.length > 0) {
          const updated = initialRepos.map((base) => {
            const match = liveList.find(
              (r) => r.repo_name.toLowerCase() === base.repo_name.toLowerCase()
            );
            if (match) {
              return {
                ...base,
                stars_count: match.stars_count,
                forks_count: match.forks_count,
                description: match.description || base.description,
                primary_language: match.primary_language || base.primary_language,
              };
            }
            return base;
          });
          setRepos(updated);
        }
      } catch {
        // Retain fallback data gracefully if backend is offline
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
          <span>Repository Directory • Step 10.3</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
          Monitored Flagship Repositories
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-3xl leading-relaxed">
          Explore benchmark analytics across 4 high-profile open-source software domains.
          Select a repository to drill down into resolution SLA percentiles, Kaplan-Meier
          survival curves, and full-text issue analytics.
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

      {/* Repository Cards Grid */}
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
                  Analyzed Issues
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
                <Link
                  href={`/repos/${repo.owner}/${repo.name}`}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition-colors shadow-sm"
                >
                  <Terminal className="w-3.5 h-3.5" />
                  <span>View Overview</span>
                  <ArrowRight className="w-3 h-3 ml-0.5" />
                </Link>

                <Link
                  href={`/repos/${repo.owner}/${repo.name}/issues`}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white font-medium text-xs transition-colors"
                >
                  <GitPullRequest className="w-3.5 h-3.5 text-slate-400" />
                  <span>Browse Issues (582)</span>
                </Link>
              </div>

              <a
                href={`https://github.com/${repo.repo_name}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-slate-200 transition-colors"
              >
                <span>GitHub</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        ))}
      </div>

      {filteredAndSortedRepos.length === 0 && (
        <div className="p-12 text-center rounded-3xl glass-panel border border-slate-800 space-y-3">
          <Boxes className="w-10 h-10 text-slate-500 mx-auto" />
          <h3 className="text-base font-semibold text-white">
            No matching repositories
          </h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Try adjusting your search query or reset the filter to view all 4 supported repositories.
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
