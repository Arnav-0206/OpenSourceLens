import { AlertCircle, CheckCircle, Info, Lightbulb, Zap } from "lucide-react";

export default function BenchmarkInsightsBanner() {
  const insights = [
    {
      title: "VS Code Rapid Turnaround (7.7h P50)",
      description:
        "Microsoft's vscode achieves the fastest resolution velocity across all benchmarks, with 69.8% of resolved issues closing within 24 hours. Automated bot triaging and strict daily triage rotations drive rapid backlog pruning.",
      badge: "High Automation",
      accent: "border-blue-500/30 bg-blue-500/5 text-blue-300",
    },
    {
      title: "ML Frameworks Reproducibility Lag (48h–53h P50)",
      description:
        "PyTorch and Hugging Face Transformers exhibit moderate turnaround times (48.2h and 53.1h). Resolution duration strongly correlates with CUDA hardware environment reproduction and distributed multi-GPU CI test cycles.",
      badge: "Hardware Matrix",
      accent: "border-amber-500/30 bg-amber-500/5 text-amber-300",
    },
    {
      title: "Kubernetes Multi-SIG Review Depth (141.2h P50)",
      description:
        "Kubernetes requires 141.2 hours median resolution time. The extended duration reflects decentralized Special Interest Group (SIG) peer consensus, multi-stage approval requirements, and mission-critical cloud stability criteria.",
      badge: "Consensus Governance",
      accent: "border-cyan-500/30 bg-cyan-500/5 text-cyan-300",
    },
  ];

  return (
    <div className="glass-panel rounded-3xl p-6 sm:p-8 border border-slate-800 bg-slate-900/40 space-y-5">
      <div className="flex items-center gap-2.5">
        <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
          <Lightbulb className="w-4 h-4" />
        </div>
        <div>
          <h3 className="text-base font-bold text-white tracking-tight">
            Key Empirical Analytics Insights
          </h3>
          <p className="text-xs text-slate-400">
            Qualitative findings from exploratory data analysis and resolution modeling (Phases 3 & 8)
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
        {insights.map((item) => (
          <div
            key={item.title}
            className={`p-4 rounded-2xl border ${item.accent} flex flex-col justify-between space-y-3`}
          >
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold text-white leading-snug">
                  {item.title}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {item.description}
              </p>
            </div>

            <div>
              <span className="inline-block px-2 py-0.5 text-[10px] font-semibold rounded-md bg-slate-900/80 border border-slate-800 text-slate-400">
                {item.badge}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
