"use client";

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from "recharts";
import { PieChart as PieIcon, Layers } from "lucide-react";
import { formatNumber, formatPercent } from "@/lib/utils";

export interface CategoryDistributionPieChartProps {
  distribution: Record<string, number>;
  title?: string;
  subtitle?: string;
  height?: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  bug: "#f43f5e", // rose-500
  feature: "#10b981", // emerald-500
  other: "#6366f1", // indigo-500
  documentation: "#0ea5e9", // sky-500
  question: "#f59e0b", // amber-500
};

interface TooltipPayloadItem {
  name: string;
  value: number;
  payload: {
    name: string;
    value: number;
    percent: number;
    color: string;
  };
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const item = payload[0];

  return (
    <div className="bg-slate-950/95 border border-slate-800 rounded-xl p-3 shadow-2xl backdrop-blur-md text-xs space-y-1">
      <div className="flex items-center gap-2">
        <span
          className="w-2.5 h-2.5 rounded-full"
          style={{ backgroundColor: item.payload.color }}
        />
        <span className="font-bold text-white capitalize">{item.name}</span>
      </div>
      <div className="flex items-center justify-between gap-4 pt-1 border-t border-slate-800 text-slate-300">
        <span>Count:</span>
        <span className="font-mono font-bold text-white">
          {item.value.toLocaleString()}
        </span>
      </div>
      <div className="flex items-center justify-between gap-4 text-slate-400">
        <span>Share:</span>
        <span className="font-mono font-bold text-indigo-400">
          {(item.payload.percent * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

export default function CategoryDistributionPieChart({
  distribution,
  title = "Taxonomy Category Breakdown",
  subtitle = "Distribution across normalized taxonomy classes",
  height = 260,
}: CategoryDistributionPieChartProps) {
  const [mounted, setMounted] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div
        className="w-full bg-slate-900/60 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between animate-pulse"
        style={{ height }}
      >
        <div className="h-4 w-40 bg-slate-800 rounded" />
        <div className="w-32 h-32 rounded-full bg-slate-800/40 mx-auto" />
        <div className="h-4 w-48 bg-slate-800/60 rounded" />
      </div>
    );
  }

  const total = Object.values(distribution || {}).reduce((s, v) => s + v, 0);

  const chartData = Object.entries(distribution || {}).map(([key, value]) => {
    const normKey = key.toLowerCase();
    const color = CATEGORY_COLORS[normKey] || "#94a3b8";
    const pct = total > 0 ? value / total : 0;
    return {
      name: key.charAt(0).toUpperCase() + key.slice(1),
      value,
      percent: pct,
      color,
    };
  });

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-4">
      {/* Header */}
      <div className="pb-3 border-b border-slate-800/80">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <PieIcon className="w-4 h-4 text-indigo-400" />
          <span>{title}</span>
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
        {/* Doughnut Pie Container */}
        <div className="relative w-48 shrink-0" style={{ height: height - 80 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Tooltip content={<CustomTooltip />} />
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={75}
                paddingAngle={4}
                onMouseEnter={(_, index) => setActiveIndex(index)}
                onMouseLeave={() => setActiveIndex(null)}
              >
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.color}
                    opacity={
                      activeIndex === null || activeIndex === index ? 1 : 0.4
                    }
                    stroke="#0f172a"
                    strokeWidth={2}
                    className="transition-opacity duration-300 cursor-pointer"
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>

          {/* Centered Total */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center">
            <span className="text-xs font-semibold text-slate-400">Total</span>
            <span className="text-base font-extrabold text-white tracking-tight">
              {formatNumber(total)}
            </span>
          </div>
        </div>

        {/* Legend List */}
        <div className="flex-1 w-full space-y-2.5">
          {chartData.map((item, idx) => (
            <div
              key={item.name}
              onMouseEnter={() => setActiveIndex(idx)}
              onMouseLeave={() => setActiveIndex(null)}
              className={`p-2 rounded-xl transition-all border cursor-pointer ${
                activeIndex === idx
                  ? "bg-slate-800/80 border-slate-700 shadow-md"
                  : "bg-slate-950/60 border-slate-800/80 hover:bg-slate-900/60"
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="font-semibold text-slate-200">
                    {item.name}
                  </span>
                </div>
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-slate-400 font-medium">
                    {item.value.toLocaleString()}
                  </span>
                  <span className="text-[11px] font-bold text-slate-300 bg-slate-800 px-1.5 py-0.5 rounded">
                    {(item.percent * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
