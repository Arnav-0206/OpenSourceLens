"use client";

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { TrendingUp, Activity, ArrowDownRight, ArrowUpRight } from "lucide-react";
import { TrendPoint } from "@/lib/types";

export interface IssueTrendsChartProps {
  data: TrendPoint[];
  repoName?: string;
  height?: number;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const created = payload.find((p) => p.name === "Created")?.value || 0;
  const closed = payload.find((p) => p.name === "Closed")?.value || 0;
  const netDelta = closed - created;

  return (
    <div className="bg-slate-950/95 border border-slate-800 rounded-xl p-3 shadow-2xl backdrop-blur-md text-xs space-y-1.5 min-w-[170px]">
      <div className="font-semibold text-slate-300 pb-1 border-b border-slate-800/80">
        {label}
      </div>

      <div className="flex items-center justify-between text-indigo-400">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-indigo-500" />
          <span>Created:</span>
        </span>
        <span className="font-mono font-bold">{created.toLocaleString()}</span>
      </div>

      <div className="flex items-center justify-between text-emerald-400">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>Closed:</span>
        </span>
        <span className="font-mono font-bold">{closed.toLocaleString()}</span>
      </div>

      <div className="pt-1 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
        <span className="text-slate-400">Net Delta:</span>
        <span
          className={`font-mono font-bold flex items-center gap-0.5 ${
            netDelta >= 0 ? "text-emerald-400" : "text-amber-400"
          }`}
        >
          {netDelta >= 0 ? (
            <ArrowDownRight className="w-3 h-3 text-emerald-400" />
          ) : (
            <ArrowUpRight className="w-3 h-3 text-amber-400" />
          )}
          <span>{netDelta > 0 ? `+${netDelta}` : netDelta}</span>
        </span>
      </div>
    </div>
  );
}

export default function IssueTrendsChart({
  data,
  repoName,
  height = 320,
}: IssueTrendsChartProps) {
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
        <div className="h-40 w-full bg-slate-800/40 rounded-xl" />
        <div className="h-4 w-32 bg-slate-800/60 rounded" />
      </div>
    );
  }

  // Format data for chart
  const chartData = (data || []).map((point) => ({
    period: point.period,
    Created: point.created_count,
    Closed: point.closed_count,
  }));

  const totalCreated = chartData.reduce((sum, p) => sum + p.Created, 0);
  const totalClosed = chartData.reduce((sum, p) => sum + p.Closed, 0);

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-4">
      {/* Chart Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            <span>Issue Velocity & Lifecycle Trends</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Monthly issue creation versus triage resolution velocity
            {repoName && ` for ${repoName}`}.
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-300">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            <span>Total Inflow: {totalCreated.toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Total Resolved: {totalClosed.toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* AreaChart Container */}
      <div className="w-full" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <defs>
              <linearGradient id="createdGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="closedGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e293b"
              vertical={false}
            />

            <XAxis
              dataKey="period"
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
            />

            <Tooltip content={<CustomTooltip />} />

            <Legend
              verticalAlign="top"
              align="right"
              iconType="circle"
              wrapperStyle={{ paddingBottom: 12, fontSize: 12 }}
            />

            <Area
              type="monotone"
              dataKey="Created"
              stroke="#6366f1"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#createdGradient)"
            />

            <Area
              type="monotone"
              dataKey="Closed"
              stroke="#10b981"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#closedGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
