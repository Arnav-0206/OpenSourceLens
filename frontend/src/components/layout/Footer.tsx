import { GitBranch, GitPullRequest, ShieldCheck, Sparkles, Terminal } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-900 bg-slate-950/90 text-slate-400 text-xs py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Col 1: Identity */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2">
              <span className="font-bold text-sm tracking-tight text-white">
                OpenSource<span className="text-indigo-400">Lens</span>
              </span>
              <span className="px-1.5 py-0.5 text-[9px] uppercase font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded">
                Intelligence Platform
              </span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed max-w-md">
              Universal GitHub repository intelligence and analytics platform applying statistical analysis,
              data science, fine-tuned transformer NLP, and machine learning to uncover deep repository dynamics.
            </p>
            <div className="flex items-center gap-3 text-[11px] text-slate-500">
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                Anti-Leakage Validation
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                DistilBERT + Statistical Engine
              </span>
            </div>
          </div>

          {/* Col 2: Documentation & Resources */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Resources & Specs
            </h4>
            <ul className="space-y-1.5 text-slate-400 text-xs">
              <li>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-indigo-400 transition-colors"
                >
                  FastAPI Swagger UI &rarr;
                </a>
              </li>
              <li>
                <a
                  href="http://localhost:8000/redoc"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-indigo-400 transition-colors"
                >
                  ReDoc API Reference &rarr;
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/Arnav-0206/OpenSourceLens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-indigo-400 transition-colors"
                >
                  GitHub Project Repository &rarr;
                </a>
              </li>
            </ul>
          </div>

          {/* Col 3: Architecture */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Technology Stack
            </h4>
            <ul className="space-y-1.5 text-slate-400 text-xs">
              <li className="flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-slate-500" />
                FastAPI 0.115 + SQLite WAL / Postgres
              </li>
              <li className="flex items-center gap-1.5">
                <GitPullRequest className="w-3.5 h-3.5 text-slate-500" />
                Next.js 15.5 + TypeScript 5.7
              </li>
              <li className="flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5 text-slate-500" />
                PyTorch & Hugging Face Transformers
              </li>
            </ul>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-500">
          <p>© {new Date().getFullYear()} OpenSourceLens Project. MIT License.</p>
          <p className="text-[11px]">
            Universal GitHub Intelligence • Powered by Data Science & ML
          </p>
        </div>
      </div>
    </footer>
  );
}
