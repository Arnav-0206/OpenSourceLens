import {
  Activity,
  CheckCircle,
  Database,
  Flame,
  GitPullRequest,
  Hourglass,
  Layers,
  Sparkles,
} from "lucide-react";

export default function GlobalMetricsGrid() {
  const kpis = [
    {
      label: "Total Standardized Issues",
      value: "2,328",
      subtext: "582 issues per repository",
      icon: Database,
      accent: "text-indigo-400",
      bg: "bg-indigo-500/10 border-indigo-500/20",
    },
    {
      label: "Fastest Resolution SLA",
      value: "7.7 hrs",
      subtext: "microsoft/vscode (P50)",
      icon: Flame,
      accent: "text-emerald-400",
      bg: "bg-emerald-500/10 border-emerald-500/20",
    },
    {
      label: "Deep Enterprise SLA",
      value: "141.2 hrs",
      subtext: "kubernetes/kubernetes (P50)",
      icon: Hourglass,
      accent: "text-amber-400",
      bg: "bg-amber-500/10 border-amber-500/20",
    },
    {
      label: "Global Resolution Rate",
      value: "56.4%",
      subtext: "1,313 total resolved issues",
      icon: CheckCircle,
      accent: "text-cyan-400",
      bg: "bg-cyan-500/10 border-cyan-500/20",
    },
    {
      label: "Dominant Taxonomy Class",
      value: "Bug (41.2%)",
      subtext: "Canonical taxonomy mapping",
      icon: Layers,
      accent: "text-rose-400",
      bg: "bg-rose-500/10 border-rose-500/20",
    },
    {
      label: "Precomputed Semantic Pairs",
      value: "11,640",
      subtext: "Top-5 cosine pairings indexed",
      icon: Sparkles,
      accent: "text-purple-400",
      bg: "bg-purple-500/10 border-purple-500/20",
    },
  ];

  return (
    <section className="space-y-4">
      <div>
        <span className="text-xs uppercase font-bold tracking-wider text-indigo-400">
          Macro Analytics
        </span>
        <h2 className="text-2xl font-bold tracking-tight text-white mt-1">
          Dataset & System Key Performance Indicators
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Aggregated empirical metrics derived from repository snapshots and offline ML models.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {kpis.map((kpi) => {
          const Icon = kpi.icon;
          return (
            <div
              key={kpi.label}
              className="glass-panel rounded-2xl p-4 border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-slate-400 leading-tight">
                  {kpi.label}
                </span>
                <div className={`p-1.5 rounded-lg border ${kpi.bg} ${kpi.accent}`}>
                  <Icon className="w-3.5 h-3.5" />
                </div>
              </div>

              <div>
                <div className="text-xl font-extrabold text-white tracking-tight">
                  {kpi.value}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5 truncate">
                  {kpi.subtext}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
