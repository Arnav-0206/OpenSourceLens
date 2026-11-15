import {
  Activity,
  BarChart3,
  Cpu,
  GitPullRequest,
  Search,
  Users,
} from "lucide-react";

export default function FeaturesShowcase() {
  const features = [
    {
      title: "Dynamic Ingestion & 6-Domain Analytics",
      badge: "Universal Engine",
      icon: BarChart3,
      accent: "text-indigo-400",
      accentBg: "bg-indigo-500/10 border-indigo-500/25",
      description:
        "Full telemetry extraction across development activity, contributor ecosystems, pull requests, backlog flow, and release cadence with Benjamini-Hochberg FDR guarded synthesis.",
      stats: [
        { label: "Engine Domains", value: "6 Domains" },
        { label: "Permutation Tests", value: "Spearman r" },
        { label: "Storage Architecture", value: "WAL Isolation" },
      ],
    },
    {
      title: "Fine-Tuned DistilBERT Issue Classification",
      badge: "Transformer Champion",
      icon: Cpu,
      accent: "text-cyan-400",
      accentBg: "bg-cyan-500/10 border-cyan-500/25",
      description:
        "Production PyTorch sequence classifier categorizing incoming issues into bug, feature request, and discussion types with calibrated confidence probabilities.",
      stats: [
        { label: "Architecture", value: "DistilBERT" },
        { label: "Inference Latency", value: "< 50ms" },
        { label: "Audit Logging", value: "Active" },
      ],
    },
    {
      title: "Contributor Ecosystem & Bus Factor Analysis",
      badge: "Inequality Metrics",
      icon: Users,
      accent: "text-emerald-400",
      accentBg: "bg-emerald-500/10 border-emerald-500/25",
      description:
        "Identity-resolved contributor profiling computing Lorenz curves, Gini coefficient, Shannon entropy, and BF50/BF80 retention thresholds to pinpoint community risk.",
      stats: [
        { label: "Inequality Metric", value: "Gini Index" },
        { label: "Concentration", value: "BF50 / BF80" },
        { label: "Role Separation", value: "Bot-Filtered" },
      ],
    },
    {
      title: "Review Bottlenecks & Survival Dynamics",
      badge: "PR & Backlog Intelligence",
      icon: GitPullRequest,
      accent: "text-purple-400",
      accentBg: "bg-purple-500/10 border-purple-500/25",
      description:
        "Time-to-first-response (TTFR) metrics, reviewer workload balance, diff churn bins, and Kaplan-Meier survival curves quantifying issue and PR resolution velocity.",
      stats: [
        { label: "Latency Metrics", value: "TTFR / TTMerge" },
        { label: "Survival Window", value: "Kaplan-Meier" },
        { label: "Size Stratification", value: "Churn Bins" },
      ],
    },
  ];

  return (
    <section className="space-y-6">
      <div>
        <span className="text-xs uppercase font-bold tracking-wider text-indigo-400">
          Core Analytical Capabilities
        </span>
        <h2 className="text-2xl font-bold tracking-tight text-white mt-1">
          Deep Data Science & Statistical Engine
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Engineered for universal repository analysis with statistical safeguards against false discoveries,
          rate limit budgets, and rigorous ML evaluation.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feat) => {
          const Icon = feat.icon;
          return (
            <div
              key={feat.title}
              className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 flex flex-col justify-between hover:border-slate-700/80 transition-colors group"
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div
                    className={`w-10 h-10 rounded-xl ${feat.accentBg} border flex items-center justify-center ${feat.accent} group-hover:scale-105 transition-transform`}
                  >
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700/60">
                    {feat.badge}
                  </span>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-white group-hover:text-indigo-300 transition-colors">
                    {feat.title}
                  </h3>
                  <p className="text-sm text-slate-400 mt-1.5 leading-relaxed">
                    {feat.description}
                  </p>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-800/80 grid grid-cols-3 gap-2">
                {feat.stats.map((s) => (
                  <div key={s.label}>
                    <div className="text-xs font-bold text-slate-200">
                      {s.value}
                    </div>
                    <div className="text-[10px] text-slate-400 truncate">
                      {s.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
