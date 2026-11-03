import Link from "next/link";
import { ArrowLeft, Boxes, FileQuestion, Home, Sparkles } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center text-center px-4 py-16">
      <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mb-6 shadow-lg shadow-indigo-500/5">
        <FileQuestion className="w-8 h-8" />
      </div>

      <span className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-2">
        404 — Resource Not Found
      </span>

      <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight mb-3">
        Page or Repository Not Found
      </h1>

      <p className="text-sm text-slate-400 max-w-md mx-auto mb-8 leading-relaxed">
        The requested URL does not match any route in the OpenSourceLens analytics platform. It may have been relocated or is outside the four target repositories.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-colors shadow-sm"
        >
          <Home className="w-4 h-4" />
          <span>Return Home</span>
        </Link>

        <Link
          href="/repos"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-xs transition-colors border border-slate-800"
        >
          <Boxes className="w-4 h-4 text-slate-400" />
          <span>Explore Repositories</span>
        </Link>

        <Link
          href="/analyzer"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-xs transition-colors border border-slate-800"
        >
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span>AI Analyzer</span>
        </Link>
      </div>
    </div>
  );
}
