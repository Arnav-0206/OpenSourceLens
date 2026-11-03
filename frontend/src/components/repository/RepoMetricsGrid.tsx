import {
  CheckCircle2,
  Clock,
  Database,
  Flame,
  ShieldCheck,
  TrendingDown,
} from "lucide-react";
import { formatHours, formatPercent } from "@/lib/utils";

interface RepoMetricsGridProps {
  totalAnalyzed: number;
  closedCount: number;
  medianHours: number | null;
  rapidClosurePct: number;
  survivalRate30d?: number | null;
}

export default function RepoMetricsGrid({
  totalAnalyzed,
  closedCount,
  medianHours,
  rapidClosurePct,
  survivalRate30d,
}: RepoMetricsGridProps) {
  const closureRate = totalAnalyzed > 0 ? closedCount / totalAnalyzed : 0;
  const openCount = totalAnalyzed - closedCount;

  const kpis = [
    {
      title: "Standardized Issues",
      value: totalAnalyzed.toLocaleString(),
      subtext: `${openCount} Open / ${closedCount} Closed`,
      icon: Database,
      accent: "text-indigo-400",
      bg: "bg-indigo-500/10 border-indigo-500/25",
    },
    {
      title: "Resolution Rate",
      value: formatPercent(closureRate),
      subtext: `${closedCount} verified closures`,
      icon: CheckCircle2,
      accent: "text-cyan-400",
      bg: "bg-cyan-500/10 border-cyan-500/25",
    },
    {
      title: "Median SLA Duration",
      value: formatHours(medianHours),
      subtext: "P50 resolution duration",
      icon: Clock,
      accent: "text-emerald-400",
      bg: "bg-emerald-500/10 border-emerald-500/25",
    },
    {
      title: "Rapid Closure Velocity",
      value: formatPercent(rapidClosurePct),
      subtext: "Resolved in under 24 hours",
      icon: Flame,
      accent: "text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/25",
    },
    {
      title: "30-Day Survival Rate",
      value: survivalRate30d ? formatPercent(survivalRate30d) : "N/A",
      subtext: "Kaplan-Meier survival curve",
      icon: TrendingDown,
      accent: "text-rose-400",
      bg: "bg-rose-500/10 border-rose-500/25",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {kpis.map((kpi) => {
        const Icon = kpi.icon;
        return (
          <div
            key={kpi.title}
            className="glass-panel rounded-2xl p-4 sm:p-5 border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-3"
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                {kpi.title}
              </span>
              <div className={`p-1.5 rounded-lg border ${kpi.bg} ${kpi.accent}`}>
                <Icon className="w-3.5 h-3.5" />
              </div>
            </div>

            <div>
              <div className="text-2xl font-extrabold text-white tracking-tight">
                {kpi.value}
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5 truncate">
                {kpi.subtext}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
