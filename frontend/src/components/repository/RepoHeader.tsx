import Link from "next/link";
import {
  ArrowLeft,
  Boxes,
  ChevronRight,
  ExternalLink,
  GitFork,
  GitPullRequest,
  Sparkles,
  Star,
} from "lucide-react";
import { formatNumber } from "@/lib/utils";

interface RepoHeaderProps {
  owner: string;
  name: string;
  domain: string;
  description: string;
  starsCount: number;
  forksCount: number;
  openIssuesCount: number;
  primaryLanguage: string;
}

export default function RepoHeader({
  owner,
  name,
  domain,
  description,
  starsCount,
  forksCount,
  openIssuesCount,
  primaryLanguage,
}: RepoHeaderProps) {
  const fullName = `${owner}/${name}`;

  return (
    <div className="space-y-4">
      {/* Breadcrumb Navigation */}
      <nav className="flex items-center gap-2 text-xs text-slate-400">
        <Link
          href="/repos"
          className="flex items-center gap-1 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Repositories</span>
        </Link>
        <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
        <span className="text-slate-400">{owner}</span>
        <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
        <span className="font-semibold text-white">{name}</span>
      </nav>

      {/* Main Header Card */}
      <div className="glass-panel rounded-3xl p-6 sm:p-8 border border-slate-800 bg-gradient-to-b from-slate-900/90 via-slate-900/50 to-slate-950 space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-3 max-w-3xl">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-0.5 text-[11px] font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/25 rounded-full">
                {domain}
              </span>
              <span className="px-2.5 py-0.5 text-[11px] font-medium bg-slate-800 text-slate-300 border border-slate-700 rounded-full">
                {primaryLanguage}
              </span>
              <span className="px-2.5 py-0.5 text-[11px] font-medium bg-slate-800/60 text-slate-400 border border-slate-800 rounded-full">
                582 Analyzed Issues
              </span>
            </div>

            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              <span className="text-slate-400">{owner}/</span>
              <span className="text-indigo-400">{name}</span>
            </h1>

            <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
              {description}
            </p>
          </div>

          {/* GitHub Metadata Chips */}
          <div className="flex flex-wrap lg:flex-col items-start lg:items-end gap-3 shrink-0">
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800 text-xs font-semibold text-slate-200">
                <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />
                <span>{formatNumber(starsCount)} stars</span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800 text-xs font-semibold text-slate-300">
                <GitFork className="w-3.5 h-3.5 text-slate-400" />
                <span>{formatNumber(forksCount)} forks</span>
              </div>
            </div>

            <div className="text-xs text-slate-400">
              Reported open issues on GitHub:{" "}
              <span className="font-semibold text-slate-200">
                {formatNumber(openIssuesCount)}
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons Toolbar */}
        <div className="pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href={`/repos/${owner}/${name}/issues`}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition-all shadow-sm shadow-indigo-600/30"
            >
              <GitPullRequest className="w-4 h-4" />
              <span>Browse Issues (582)</span>
            </Link>

            <Link
              href={`/analyzer?repo=${encodeURIComponent(fullName)}`}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/80 text-slate-200 font-medium text-xs transition-colors"
            >
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>AI Issue Analyzer</span>
            </Link>
          </div>

          <div className="flex items-center gap-3">
            <a
              href={`https://github.com/${fullName}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-white transition-colors"
            >
              <span>GitHub Repository</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
