import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Cpu,
  Database,
  ExternalLink,
  Search,
  Sparkles,
  Zap,
} from "lucide-react";

export default function HeroSection() {
  return (
    <section className="relative overflow-hidden rounded-3xl border border-slate-800 bg-gradient-to-b from-slate-900/90 via-slate-900/50 to-slate-950 p-8 sm:p-12 lg:p-16">
      {/* Background radial glow accents */}
      <div className="absolute top-0 right-0 -mt-16 -mr-16 w-[30rem] h-[30rem] bg-indigo-500/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 -mb-16 -ml-16 w-[28rem] h-[28rem] bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-4xl space-y-6">
        {/* Release / Phase pill */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 text-xs font-semibold tracking-wide shadow-sm">
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
          <span>Production ML Dashboard • Phase 10 Interactive UI</span>
        </div>

        {/* Hero Title */}
        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-[1.15]">
          Intelligent GitHub Analytics Powered by{" "}
          <span className="bg-gradient-to-r from-indigo-400 via-cyan-400 to-indigo-300 bg-clip-text text-transparent">
            Transformers & Semantic Search
          </span>
        </h1>

        {/* Hero Description */}
        <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl">
          OpenSourceLens applies data science, classical ML baselines, fine-tuned DistilBERT NLP,
          and dense sentence embeddings to analyze GitHub issues across flagship open-source repositories.
          Predict categories, detect duplicates, and benchmark resolution velocity with empirical rigor.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-4 pt-3">
          <Link
            href="/repos"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-all shadow-lg shadow-indigo-600/25 hover:shadow-indigo-600/40 hover:-translate-y-0.5"
          >
            <Boxes className="w-4 h-4" />
            <span>Explore Repositories</span>
            <ArrowRight className="w-4 h-4 ml-0.5" />
          </Link>

          <Link
            href="/analyzer"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700/80 hover:border-slate-600 text-slate-200 font-medium text-sm transition-all hover:-translate-y-0.5"
          >
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span>Launch AI Issue Analyzer</span>
          </Link>

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-3 rounded-xl text-slate-400 hover:text-white font-medium text-sm transition-colors"
          >
            <span>FastAPI Docs</span>
            <ExternalLink className="w-3.5 h-3.5 text-slate-400" />
          </a>
        </div>

        {/* Key Benchmark Metrics Row */}
        <div className="pt-6 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Boxes className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">4 Flagships</div>
              <div className="text-[11px] text-slate-400">Target Ecosystems</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">2,328 Issues</div>
              <div className="text-[11px] text-slate-400">Curated & Labeled</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">81.3% Acc</div>
              <div className="text-[11px] text-slate-400">DistilBERT Champion</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center text-violet-400">
              <Search className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">&lt; 20ms</div>
              <div className="text-[11px] text-slate-400">Dense Vector Search</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
