import Link from "next/link";
import { GitPullRequest, ShieldCheck, Sparkles, Terminal } from "lucide-react";

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
                Production Dual-Driver
              </span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed max-w-md">
              Intelligent open-source analytics platform applying data science,
              classical ML baselines, fine-tuned transformer NLP, and semantic retrieval
              to GitHub issues across major developer ecosystems.
            </p>
            <div className="flex items-center gap-3 text-[11px] text-slate-500">
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                Anti-Leakage Validation
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                DistilBERT + all-MiniLM-L6-v2
              </span>
            </div>
          </div>

          {/* Col 2: Target Repositories */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Monitored Ecosystems
            </h4>
            <ul className="space-y-1.5 text-slate-400 text-xs">
              <li>
                <Link href="/repos/microsoft/vscode" className="hover:text-indigo-400 transition-colors">
                  microsoft / vscode
                </Link>
              </li>
              <li>
                <Link href="/repos/pytorch/pytorch" className="hover:text-indigo-400 transition-colors">
                  pytorch / pytorch
                </Link>
              </li>
              <li>
                <Link href="/repos/kubernetes/kubernetes" className="hover:text-indigo-400 transition-colors">
                  kubernetes / kubernetes
                </Link>
              </li>
              <li>
                <Link href="/repos/huggingface/transformers" className="hover:text-indigo-400 transition-colors">
                  huggingface / transformers
                </Link>
              </li>
            </ul>
          </div>

          {/* Col 3: Architecture */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Architecture Stack
            </h4>
            <ul className="space-y-1.5 text-slate-400 text-xs">
              <li className="flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-slate-500" />
                FastAPI 0.115 + PostgreSQL 16
              </li>
              <li className="flex items-center gap-1.5">
                <GitPullRequest className="w-3.5 h-3.5 text-slate-500" />
                Next.js 15 + TypeScript
              </li>
              <li>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-indigo-400 hover:text-indigo-300 underline underline-offset-2"
                >
                  REST API Documentation →
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-500">
          <p>© {new Date().getFullYear()} OpenSourceLens Project. MIT License.</p>
          <p className="text-[11px]">
            Phase 10: Frontend UI Shell • Powered by Deep Learning & Empirical Data Science
          </p>
        </div>
      </div>
    </footer>
  );
}
