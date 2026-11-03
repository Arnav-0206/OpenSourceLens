import Link from "next/link";
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  Search,
  Shield,
  Sparkles,
} from "lucide-react";

export default function FeaturesShowcase() {
  const features = [
    {
      title: "Dual-Engine Issue Classification",
      badge: "Transformer Champion (81.3% Acc)",
      icon: Cpu,
      accent: "text-indigo-400",
      accentBg: "bg-indigo-500/10 border-indigo-500/25",
      description:
        "Fine-tuned DistilBERT transformer benchmarked against classical TF-IDF + Linear SVM and Logistic Regression baselines. Normalizes issue titles and bodies to categorize bugs, features, and discussions.",
      stats: [
        { label: "Champion Accuracy", value: "81.3%" },
        { label: "Macro F1-Score", value: "0.812" },
        { label: "Baseline Margin", value: "+4.5%" },
      ],
      linkText: "Test Classification Model",
      href: "/analyzer",
    },
    {
      title: "Dense Semantic Vector Retrieval",
      badge: "all-MiniLM-L6-v2 Embeddings",
      icon: Search,
      accent: "text-cyan-400",
      accentBg: "bg-cyan-500/10 border-cyan-500/25",
      description:
        "Sub-20ms duplicate and related issue detection utilizing 384-dimensional sentence transformer embeddings. Supports instant cached top-5 lookups and dynamic real-time query vectorization.",
      stats: [
        { label: "Vector Dimension", value: "384-d" },
        { label: "Indexed Vectors", value: "2,328" },
        { label: "Retrieval Latency", value: "< 20ms" },
      ],
      linkText: "Search Related Issues",
      href: "/analyzer#related",
    },
    {
      title: "Advisory Resolution Velocity Prediction",
      badge: "Random Forest Champion",
      icon: Clock,
      accent: "text-emerald-400",
      accentBg: "bg-emerald-500/10 border-emerald-500/25",
      description:
        "Multi-class velocity tier prediction (< 1 Day, 1-7 Days, 7-30 Days, 30+ Days) with empirical ±1 tier error bounds. Built with strict anti-leakage validation and mandatory maintainer advisories.",
      stats: [
        { label: "Adjacent Accuracy", value: "88.1%" },
        { label: "Exact Tier Acc", value: "53.2%" },
        { label: "Validation Metric", value: "Stratified K-Fold" },
      ],
      linkText: "Predict Issue Velocity",
      href: "/analyzer#resolution",
    },
    {
      title: "Comparative Ecosystem Analytics",
      badge: "4-Repo Benchmark Matrix",
      icon: BarChart3,
      accent: "text-purple-400",
      accentBg: "bg-purple-500/10 border-purple-500/25",
      description:
        "Non-parametric resolution duration statistics (P25, Median, P75, IQR) paired with 30-day Kaplan-Meier survival curves and chronological monthly creation vs closure trends.",
      stats: [
        { label: "Fastest Median", value: "7.7h (VS Code)" },
        { label: "Enterprise Median", value: "141.2h (K8s)" },
        { label: "Survival Window", value: "30-Day KM" },
      ],
      linkText: "View Analytics Dashboards",
      href: "/repos",
    },
  ];

  return (
    <section className="space-y-6">
      <div>
        <span className="text-xs uppercase font-bold tracking-wider text-indigo-400">
          Core ML Architectures
        </span>
        <h2 className="text-2xl font-bold tracking-tight text-white mt-1">
          Rigorous Data Science & NLP Pipeline
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Designed with empirical reproducibility: raw data isolation, leakage prevention,
          and transparent evaluation metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feat) => {
          const Icon = feat.icon;
          return (
            <div
              key={feat.title}
              className="glass-panel glass-panel-hover rounded-2xl p-6 border border-slate-800 bg-slate-900/60 flex flex-col justify-between space-y-5"
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div className={`p-2.5 rounded-xl border ${feat.accentBg} ${feat.accent}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="px-2.5 py-1 text-[11px] font-semibold bg-slate-800 text-slate-300 border border-slate-700 rounded-full">
                    {feat.badge}
                  </span>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-white tracking-tight">
                    {feat.title}
                  </h3>
                  <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                    {feat.description}
                  </p>
                </div>

                {/* Key Metric Chips */}
                <div className="grid grid-cols-3 gap-2 py-3 px-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
                  {feat.stats.map((s) => (
                    <div key={s.label}>
                      <div className="text-[10px] text-slate-400">{s.label}</div>
                      <div className="text-xs font-bold text-slate-200 mt-0.5">{s.value}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action */}
              <Link
                href={feat.href}
                className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors pt-2 border-t border-slate-800/60 group"
              >
                <span>{feat.linkText}</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          );
        })}
      </div>
    </section>
  );
}
