"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, Home, RotateCcw } from "lucide-react";

export interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: ErrorProps) {
  useEffect(() => {
    // Log unexpected runtime error
    console.error("OpenSourceLens runtime exception:", error);
  }, [error]);

  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center text-center px-4 py-16">
      <div className="w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center mb-6 shadow-lg shadow-rose-500/5">
        <AlertTriangle className="w-8 h-8" />
      </div>

      <span className="text-xs font-bold uppercase tracking-wider text-rose-400 mb-2">
        Application Error Boundary
      </span>

      <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight mb-3">
        Unexpected Application Exception
      </h1>

      <p className="text-sm text-slate-400 max-w-md mx-auto mb-4 leading-relaxed">
        An error occurred during page rendering. This may be caused by an unexpected data format or temporary network disconnection.
      </p>

      {error.message && (
        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-rose-300 max-w-md mx-auto mb-6 text-left break-all">
          <span className="text-[10px] uppercase font-bold text-slate-500 block mb-1">
            Error Details:
          </span>
          {error.message}
          {error.digest && (
            <span className="text-[10px] text-slate-500 block mt-1">
              Digest: {error.digest}
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          onClick={() => reset()}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs transition-colors shadow-sm"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Try Again</span>
        </button>

        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-xs transition-colors border border-slate-800"
        >
          <Home className="w-4 h-4 text-slate-400" />
          <span>Return Home</span>
        </Link>
      </div>
    </div>
  );
}
