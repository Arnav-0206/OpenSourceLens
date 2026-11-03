"use client";

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { BarChart3, Clock, Zap } from "lucide-react";
import { RepositoryComparisonItem } from "@/lib/types";
import { formatHours, formatPercent } from "@/lib/utils";

export interface ResolutionBenchmarkBarChartProps {
  repositories: RepositoryComparisonItem[];
  height?: number;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: {
      name: string;
      fullName: string;
      medianHours: number;
      closureRate: number;
      rapidClosure: number;
    };
  }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;

  return (
    <div className="bg-slate-950/95 border border-slate-800 rounded-xl p-3.5 shadow-2xl backdrop-blur-md text-xs space-y-2 min-w-[200px]">
      <div className="font-bold text-white pb-1.5 border-b border-slate-800">
        {data.fullName}
      </div>

      <div className="flex items-center justify-between text-slate-300">
        <span className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>Median Resolution:</span>
        </span>
        <span className="font-mono font-bold text-cyan-300">
          {formatHours(data.medianHours)} ({data.medianHours.toFixed(1)}h)
        </span>
      </div>

      <div className="flex items-center justify-between text-slate-400">
        <span>Closure Rate:</span>
        <span className="font-mono font-bold text-emerald-400">
          {formatPercent(data.closureRate * 100)}
        </span>
      </div>

      <div className="flex items-center justify-between text-slate-400">
        <span>Rapid (&lt;24h) Closures:</span>
        <span className="font-mono font-bold text-indigo-400">
          {formatPercent(data.rapidClosure)}
        </span>
      </div>
    </div>
  );
}

export default function ResolutionBenchmarkBarChart({
  repositories,
  height = 280,
}: ResolutionBenchmarkBarChartProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div
        className="w-full bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between animate-pulse"
        style={{ height }}
      >
        <div className="h-5 w-48 bg-slate-800 rounded" />
        <div className="h-44 w-full bg-slate-800/40 rounded-xl" />
        <div className="h-4 w-32 bg-slate-800/60 rounded" />
      </div>
    );
  }

  const chartData = (repositories || []).map((repo) => ({
    name: repo.name,
    fullName: repo.repo_name,
    medianHours: repo.median_resolution_hours || 0,
    closureRate: repo.closure_rate,
    rapidClosure: repo.rapid_closure_percentage,
  }));

  // Find min for fastest highlighting
  const minHours = Math.min(...chartData.map((d) => d.medianHours || Infinity));

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            <span>Cross-Repository Resolution Benchmark</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Median hours required to triage and resolve issues across target ecosystems.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
            <Zap className="w-3 h-3" />
            <span>Fastest: VS Code (~7.7h)</span>
          </span>
        </div>
      </div>

      {/* BarChart Container */}
      <div className="w-full" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e293b"
              vertical={false}
            />

            <XAxis
              dataKey="name"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#334155" }}
            />

            <YAxis
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#334155" }}
              tickFormatter={(val) => `${val}h`}
            />

            <Tooltip content={<CustomTooltip />} />

            <Bar
              dataKey="medianHours"
              radius={[6, 6, 0, 0]}
              animationDuration={800}
            >
              {chartData.map((entry, index) => {
                const isFastest = entry.medianHours === minHours;
                return (
                  <Cell
                    key={`cell-${index}`}
                    fill={isFastest ? "#10b981" : "#6366f1"}
                    className="hover:opacity-80 transition-opacity cursor-pointer"
                  />
                );
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-500">
        <span>Lower median hours indicate higher maintenance velocity</span>
        <span>Empirical PostgreSQL analytics dataset</span>
      </div>
    </div>
  );
}
