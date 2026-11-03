"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, RefreshCw, Server, Terminal, XCircle } from "lucide-react";
import { getHealthStatus } from "@/lib/api";
import { HealthResponse } from "@/lib/types";

export default function BackendStatusBanner() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [isRetrying, setIsRetrying] = useState(false);

  async function checkBackend() {
    setIsRetrying(true);
    try {
      const data = await getHealthStatus();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setLoading(false);
      setIsRetrying(false);
    }
  }

  useEffect(() => {
    checkBackend();
  }, []);

  if (loading) {
    return null;
  }

  const isConnected = health && health.status === "healthy";

  return (
    <div
      className={`rounded-2xl p-4 sm:p-5 border transition-all ${
        isConnected
          ? "bg-slate-900/60 border-slate-800 text-slate-300"
          : "bg-amber-950/20 border-amber-800/40 text-amber-200"
      }`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start sm:items-center gap-3">
          <div
            className={`p-2 rounded-xl shrink-0 ${
              isConnected
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
            }`}
          >
            {isConnected ? (
              <CheckCircle2 className="w-5 h-5" />
            ) : (
              <Server className="w-5 h-5" />
            )}
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-white">
                {isConnected
                  ? "OpenSourceLens Backend Active"
                  : "Backend Service Unavailable"}
              </span>
              <span
                className={`px-2 py-0.5 text-[10px] font-semibold rounded-full border ${
                  isConnected
                    ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                    : "bg-amber-500/10 text-amber-300 border-amber-500/30"
                }`}
              >
                {isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              {isConnected
                ? `FastAPI v${health?.version} connected to ${health?.database.dialect} (${health?.database.latency_ms.toFixed(1)}ms latency) with ${health?.database.tables_count} relational tables.`
                : "The frontend is currently using static precomputed benchmarks. To enable live ML inference and real-time database queries, start the FastAPI service."}
            </p>

            {!isConnected && (
              <div className="pt-2 flex items-center gap-2 text-xs font-mono text-slate-300">
                <Terminal className="w-3.5 h-3.5 text-indigo-400" />
                <span className="bg-slate-900 px-2 py-1 rounded border border-slate-800">
                  python -m backend.app.main
                </span>
              </div>
            )}
          </div>
        </div>

        <button
          onClick={checkBackend}
          disabled={isRetrying}
          className="self-start sm:self-center inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium bg-slate-900 hover:bg-slate-800 border border-slate-700/80 rounded-lg text-slate-300 hover:text-white transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRetrying ? "animate-spin" : ""}`} />
          <span>{isRetrying ? "Checking..." : "Verify Connection"}</span>
        </button>
      </div>
    </div>
  );
}
