"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, Clock, Star, Terminal } from "lucide-react";
import { getRepositories } from "@/lib/api";
import { RepositorySummary } from "@/lib/types";

interface RepoCardData {
  owner: string;
  name: string;
  repo_name: string;
  domain: string;
  description: string;
  stars_count: number;
  total_issues: number;
  median_hours: number;
  rapid_pct: number;
  color: string;
  border: string;
  accent: string;
}

const fallbackRepos: RepoCardData[] = [
  {
    owner: "microsoft",
    name: "vscode",
    repo_name: "microsoft/vscode",
    domain: "Developer Tools & IDE",
    description: "Visual Studio Code editor codebase written in TypeScript & Electron.",
    stars_count: 165400,
    total_issues: 582,
    median_hours: 7.7,
    rapid_pct: 69.8,
    color: "from-blue-500/15 via-indigo-500/5 to-slate-950",
    border: "border-blue-500/30",
    accent: "text-blue-400",
  },
  {
    owner: "pytorch",
    name: "pytorch",
    repo_name: "pytorch/pytorch",
    domain: "Machine Learning Framework",
    description: "Tensors and dynamic neural networks in Python with strong GPU acceleration.",
    stars_count: 85200,
    total_issues: 582,
    median_hours: 53.1,
    rapid_pct: 39.4,
    color: "from-amber-500/15 via-orange-500/5 to-slate-950",
    border: "border-amber-500/30",
    accent: "text-amber-400",
  },
  {
    owner: "kubernetes",
    name: "kubernetes",
    repo_name: "kubernetes/kubernetes",
    domain: "Cloud Native Orchestration",
    description: "Production-grade container orchestration system powering modern cloud clusters.",
    stars_count: 110300,
    total_issues: 582,
    median_hours: 141.2,
    rapid_pct: 28.1,
    color: "from-cyan-500/15 via-sky-500/5 to-slate-950",
    border: "border-cyan-500/30",
    accent: "text-cyan-400",
  },
  {
    owner: "huggingface",
    name: "transformers",
    repo_name: "huggingface/transformers",
    domain: "NLP & Foundation Models",
    description: "State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.",
    stars_count: 135800,
    total_issues: 582,
    median_hours: 48.2,
    rapid_pct: 42.6,
    color: "from-violet-500/15 via-purple-500/5 to-slate-950",
    border: "border-violet-500/30",
    accent: "text-violet-400",
  },
];

export default function EcosystemShowcase() {
  const [repos, setRepos] = useState<RepoCardData[]>(fallbackRepos);

  useEffect(() => {
    let mounted = true;

    async function loadLiveData() {
      try {
        const liveList: RepositorySummary[] = await getRepositories();
        if (mounted && Array.isArray(liveList) && liveList.length > 0) {
          const merged = fallbackRepos.map((fb) => {
            const found = liveList.find(
              (r) => r.repo_name.toLowerCase() === fb.repo_name.toLowerCase()
            );
            if (found) {
              return {
                ...fb,
                stars_count: found.stars_count,
              };
            }
            return fb;
          });
          setRepos(merged);
        }
      } catch {
        // Fallback to static dataset if backend is unreachable
      }
    }

    loadLiveData();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <section className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
        <div>
          <span className="text-xs uppercase font-bold tracking-wider text-indigo-400">
            Target Ecosystems
          </span>
          <h2 className="text-2xl font-bold tracking-tight text-white mt-1">
            Standardized Open-Source Benchmarks
          </h2>
          <p className="text-sm text-slate-400 mt-1 max-w-xl">
            4 distinct developer domains analyzed with normalized schema taxonomy and SLA tracking.
          </p>
        </div>
        <Link
          href="/repos"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
        >
          <span>View Detailed Benchmark Matrix</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {repos.map((repo) => (
          <div
            key={repo.repo_name}
            className={`glass-panel glass-panel-hover rounded-2xl p-5 border ${repo.border} bg-gradient-to-b ${repo.color} flex flex-col justify-between`}
          >
            {/* Header: Domain & Name */}
            <div>
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] font-semibold tracking-wider uppercase text-slate-400">
                  {repo.domain}
                </span>
                <span className="flex items-center gap-1 text-[11px] font-medium text-slate-300">
                  <Star className="w-3 h-3 text-amber-400 fill-amber-400" />
                  {(repo.stars_count / 1000).toFixed(1)}k
                </span>
              </div>

              <h3 className="text-lg font-bold text-white mt-2 group">
                <span className="text-slate-400">{repo.owner}/</span>
                <span className={repo.accent}>{repo.name}</span>
              </h3>

              <p className="text-xs text-slate-400 mt-2 line-clamp-2 leading-relaxed">
                {repo.description}
              </p>
            </div>

            {/* Metrics Grid */}
            <div className="my-4 pt-4 border-t border-slate-800/80 grid grid-cols-2 gap-3">
              <div className="space-y-0.5">
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">
                  Median SLA
                </div>
                <div className="text-sm font-bold text-white flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{repo.median_hours}h</span>
                </div>
              </div>

              <div className="space-y-0.5">
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">
                  &lt; 24h Closures
                </div>
                <div className="text-sm font-bold text-cyan-300">
                  {repo.rapid_pct.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Action link */}
            <Link
              href={`/repos/${repo.owner}/${repo.name}`}
              className="inline-flex items-center justify-between w-full pt-2 border-t border-slate-800/50 text-xs font-medium text-slate-300 hover:text-white transition-colors group"
            >
              <span className="flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-slate-500 group-hover:text-indigo-400" />
                <span>Inspect Repository</span>
              </span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-0.5 transition-all" />
            </Link>
          </div>
        ))}
      </div>
    </section>
  );
}
