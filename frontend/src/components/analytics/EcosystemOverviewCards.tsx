import {
  CheckCircle2,
  Clock,
  Database,
  Flame,
  Hourglass,
  Sparkles,
} from "lucide-react";
import { GlobalAnalyticsOverview } from "@/lib/types";
import { formatPercent } from "@/lib/utils";

interface EcosystemOverviewCardsProps {
  overview?: GlobalAnalyticsOverview | null;
}

export default function EcosystemOverviewCards({
  overview,
}: EcosystemOverviewCardsProps) {
  const totalIssues = overview?.total_issues ?? 2328;
  const totalRepos = overview?.total_repositories ?? 4;
  const globalClosureRate = overview?.global_closure_rate ?? 0.564;
  const totalClosed = overview?.total_closed_issues ?? 1313;

  const kpis = [
    {
      title: "Standardized Issues",
      value: totalIssues.toLocaleString(),
      subtext: `${totalRepos} Ecosystems (582 each)`,
      icon: Database,
      accent: "text-indigo-400",
      bg: "bg-indigo-500/10 border-indigo-500/25",
    },
    {
      title: "Global Closure Rate",
      value: formatPercent(globalClosureRate),
      subtext: `${totalClosed.toLocaleString()} verified closures`,
      icon: CheckCircle2,
      accent: "text-cyan-400",
      bg: "bg-cyan-500/10 border-cyan-500/25",
    },
    {
      title: "Fastest Ecosystem SLA",
      value: "7.7 hrs",
      subtext: "microsoft/vscode (P50)",
      icon: Flame,
      accent: "text-emerald-400",
      bg: "bg-emerald-500/10 border-emerald-500/25",
    },
    {
      title: "Enterprise Review SLA",
      value: "141.2 hrs",
      subtext: "kubernetes/kubernetes (P50)",
      icon: Hourglass,
      accent: "text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/25",
    },
    {
      title: "Semantic Vector Pairs",
      value: "11,640",
      subtext: "Top-5 cosine similarities",
      icon: Sparkles,
      accent: "text-purple-400",
      bg: "bg-purple-500/10 border-purple-500/25",
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
