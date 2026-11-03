"use client";

import { useEffect, useState } from "react";
import { getHealthStatus } from "@/lib/api";
import { HealthResponse } from "@/lib/types";

export default function HealthIndicator() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function checkHealth() {
      try {
        const data = await getHealthStatus();
        if (mounted) {
          setHealth(data);
          setIsError(false);
          setLoading(false);
        }
      } catch {
        if (mounted) {
          setIsError(true);
          setLoading(false);
        }
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // 30s probe
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 px-2.5 py-1 text-xs font-medium text-slate-400 bg-slate-900/80 border border-slate-800 rounded-full">
        <span className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" />
        <span>Checking API...</span>
      </div>
    );
  }

  if (isError || !health || health.status !== "healthy") {
    return (
      <div
        className="flex items-center gap-2 px-2.5 py-1 text-xs font-medium text-rose-300 bg-rose-950/40 border border-rose-800/60 rounded-full cursor-help"
        title="FastAPI backend is offline or unreachable at http://localhost:8000"
      >
        <span className="w-2 h-2 rounded-full bg-rose-500" />
        <span>API Offline</span>
      </div>
    );
  }

  return (
    <div
      className="flex items-center gap-2 px-2.5 py-1 text-xs font-medium text-emerald-300 bg-emerald-950/30 border border-emerald-800/40 rounded-full cursor-default"
      title={`Database: ${health.database.dialect} (${health.database.latency_ms.toFixed(1)}ms) | Tables: ${health.database.tables_count}`}
    >
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
      </span>
      <span>Live v{health.version}</span>
    </div>
  );
}
