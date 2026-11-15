import {
  Activity,
  ArrowRight,
  BarChart3,
  Cpu,
  Database,
  ExternalLink,
  GitBranch,
  Layers,
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
        {/* Product Badge */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 text-xs font-semibold tracking-wide shadow-sm">
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
          <span>Universal GitHub Repository Intelligence & Analytics Platform</span>
        </div>

        {/* Hero Title */}
        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-[1.15]">
          Deep GitHub Repository Intelligence Powered by{" "}
          <span className="bg-gradient-to-r from-indigo-400 via-cyan-400 to-indigo-300 bg-clip-text text-transparent">
            Statistical Rigor & ML
          </span>
        </h1>

        {/* Hero Description */}
        <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl">
          OpenSourceLens goes beyond surface-level counters. Ingest any public GitHub repository
          to uncover hidden velocity trends, contributor inequality, review bottlenecks, backlog survival dynamics,
          and fine-tuned NLP issue intelligence.
        </p>

        {/* Action / Links */}
        <div className="flex flex-wrap items-center gap-4 pt-3">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-all shadow-lg shadow-indigo-600/25 hover:shadow-indigo-600/40 hover:-translate-y-0.5"
          >
            <span>Explore OpenAPI / Swagger</span>
            <ExternalLink className="w-4 h-4 ml-0.5" />
          </a>

          <a
            href="https://github.com/Arnav-0206/OpenSourceLens"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700/80 hover:border-slate-600 text-slate-200 font-medium text-sm transition-all hover:-translate-y-0.5"
          >
            <GitBranch className="w-4 h-4 text-indigo-400" />
            <span>GitHub Repository</span>
          </a>
        </div>

        {/* Core Architectural Pillars */}
        <div className="pt-6 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">Universal</div>
              <div className="text-[11px] text-slate-400">Any Public Repo</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">6 Domains</div>
              <div className="text-[11px] text-slate-400">Statistical Engine</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">Transformers</div>
              <div className="text-[11px] text-slate-400">DistilBERT NLP</div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center text-violet-400">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <div className="text-lg font-bold text-white">Interactive</div>
              <div className="text-[11px] text-slate-400">Workspace (Stage 4)</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
