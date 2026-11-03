import { Layers } from "lucide-react";
import { getCategoryBadge } from "@/lib/utils";

interface CategoryDistributionProps {
  distribution: Record<string, number> | null;
  total: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  bug: "bg-rose-500",
  feature: "bg-emerald-500",
  documentation: "bg-sky-500",
  question: "bg-amber-500",
  other: "bg-slate-500",
};

export default function CategoryDistributionCard({
  distribution,
  total,
}: CategoryDistributionProps) {
  const dist = distribution || {
    bug: Math.round(total * 0.42),
    feature: Math.round(total * 0.25),
    documentation: Math.round(total * 0.12),
    question: Math.round(total * 0.08),
    other: Math.round(total * 0.13),
  };

  const categories = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const safeTotal = total > 0 ? total : 1;

  return (
    <div className="glass-panel rounded-3xl p-6 border border-slate-800 bg-slate-900/60 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Canonical Category Taxonomy
            </h3>
            <p className="text-xs text-slate-400">
              Distribution of standardized issue types
            </p>
          </div>
        </div>
        <span className="text-xs font-semibold text-slate-400">
          {total} issues labeled
        </span>
      </div>

      {/* Multi-segment stacked bar */}
      <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden flex border border-slate-800">
        {categories.map(([cat, count]) => {
          const pct = (count / safeTotal) * 100;
          if (pct <= 0) return null;
          const bg = CATEGORY_COLORS[cat.toLowerCase()] || "bg-slate-600";
          return (
            <div
              key={cat}
              style={{ width: `${pct}%` }}
              className={`h-full ${bg} transition-all`}
              title={`${cat}: ${count} (${pct.toFixed(1)}%)`}
            />
          );
        })}
      </div>

      {/* Category breakdown rows */}
      <div className="space-y-3 pt-2">
        {categories.map(([cat, count]) => {
          const pct = (count / safeTotal) * 100;
          const badge = getCategoryBadge(cat);
          const barColor = CATEGORY_COLORS[cat.toLowerCase()] || "bg-slate-500";

          return (
            <div key={cat} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span
                  className={`px-2 py-0.5 text-[10px] font-semibold border rounded-md ${badge.className}`}
                >
                  {badge.label}
                </span>
                <span className="font-semibold text-slate-300">
                  {count} <span className="text-slate-500">({pct.toFixed(1)}%)</span>
                </span>
              </div>
              <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800/80">
                <div
                  style={{ width: `${pct}%` }}
                  className={`h-full ${barColor} rounded-full`}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
