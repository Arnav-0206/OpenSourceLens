"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  ArrowUpDown,
  Clock,
  ExternalLink,
  Flame,
  Star,
  TrendingDown,
} from "lucide-react";
import { RepositoryComparisonItem } from "@/lib/types";
import { formatHours, formatNumber, formatPercent, getCategoryBadge } from "@/lib/utils";

const defaultComparisons: RepositoryComparisonItem[] = [
  {
    repo_name: "microsoft/vscode",
    owner: "microsoft",
    name: "vscode",
    stars_count: 165400,
    total_issues: 582,
    open_issues: 340,
    closed_issues: 242,
    closure_rate: 0.416,
    median_resolution_hours: 7.7,
    survival_rate_30d: 0.28,
    dominant_category: "bug",
    rapid_closure_percentage: 69.8,
  },
  {
    repo_name: "pytorch/pytorch",
    owner: "pytorch",
    name: "pytorch",
    stars_count: 85200,
    total_issues: 582,
    open_issues: 272,
    closed_issues: 310,
    closure_rate: 0.533,
    median_resolution_hours: 53.1,
    survival_rate_30d: 0.52,
    dominant_category: "bug",
    rapid_closure_percentage: 39.4,
  },
  {
    repo_name: "kubernetes/kubernetes",
    owner: "kubernetes",
    name: "kubernetes",
    stars_count: 110300,
    total_issues: 582,
    open_issues: 202,
    closed_issues: 380,
    closure_rate: 0.653,
    median_resolution_hours: 141.2,
    survival_rate_30d: 0.65,
    dominant_category: "bug",
    rapid_closure_percentage: 28.1,
  },
  {
    repo_name: "huggingface/transformers",
    owner: "huggingface",
    name: "transformers",
    stars_count: 135800,
    total_issues: 582,
    open_issues: 201,
    closed_issues: 381,
    closure_rate: 0.655,
    median_resolution_hours: 48.2,
    survival_rate_30d: 0.48,
    dominant_category: "bug",
    rapid_closure_percentage: 42.6,
  },
];

interface ComparativeMatrixTableProps {
  items?: RepositoryComparisonItem[];
}

type SortColumn = "name" | "stars" | "median" | "rapid" | "survival" | "closure";

export default function ComparativeMatrixTable({
  items,
}: ComparativeMatrixTableProps) {
  const [sortBy, setSortBy] = useState<SortColumn>("median");
  const [sortAsc, setSortAsc] = useState<boolean>(true);

  const dataList = items && items.length > 0 ? items : defaultComparisons;

  function toggleSort(col: SortColumn) {
    if (sortBy === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(col);
      // For median, ascending (fastest) is usually desired
      setSortAsc(col === "median" || col === "name");
    }
  }

  const sortedList = useMemo(() => {
    return [...dataList].sort((a, b) => {
      let diff = 0;
      switch (sortBy) {
        case "name":
          diff = a.name.localeCompare(b.name);
          break;
        case "stars":
          diff = a.stars_count - b.stars_count;
          break;
        case "median":
          diff = (a.median_resolution_hours ?? 9999) - (b.median_resolution_hours ?? 9999);
          break;
        case "rapid":
          diff = a.rapid_closure_percentage - b.rapid_closure_percentage;
          break;
        case "survival":
          diff = (a.survival_rate_30d ?? 0) - (b.survival_rate_30d ?? 0);
          break;
        case "closure":
          diff = a.closure_rate - b.closure_rate;
          break;
      }
      return sortAsc ? diff : -diff;
    });
  }, [dataList, sortBy, sortAsc]);

  // Max median hours for relative scaling bar
  const maxMedian = Math.max(...dataList.map((d) => d.median_resolution_hours || 150));

  return (
    <div className="glass-panel rounded-3xl border border-slate-800 bg-slate-900/60 overflow-hidden space-y-4">
      {/* Table Header Controls */}
      <div className="p-6 pb-2 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-white tracking-tight">
            Cross-Repository Benchmark Matrix
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Empirical turnaround comparison derived from 2,328 standardized GitHub issues.
          </p>
        </div>
        <div className="text-xs text-slate-400">
          Showing <span className="font-semibold text-white">{dataList.length}</span> flagship ecosystems
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="text-[11px] uppercase tracking-wider text-slate-400 bg-slate-950/60 border-y border-slate-800">
            <tr>
              <th scope="col" className="py-3.5 px-6">
                <button
                  type="button"
                  onClick={() => toggleSort("name")}
                  className="flex items-center gap-1.5 font-bold hover:text-white"
                >
                  <span>Repository</span>
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th scope="col" className="py-3.5 px-4 text-center">
                <button
                  type="button"
                  onClick={() => toggleSort("stars")}
                  className="flex items-center gap-1.5 font-bold hover:text-white mx-auto"
                >
                  <span>Stars</span>
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th scope="col" className="py-3.5 px-4">
                <button
                  type="button"
                  onClick={() => toggleSort("median")}
                  className="flex items-center gap-1.5 font-bold hover:text-white"
                >
                  <span>Median SLA (P50)</span>
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th scope="col" className="py-3.5 px-4 text-center">
                <button
                  type="button"
                  onClick={() => toggleSort("rapid")}
                  className="flex items-center gap-1.5 font-bold hover:text-white mx-auto"
                >
                  <span>&lt; 24h Closures</span>
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th scope="col" className="py-3.5 px-4 text-center">
                <button
                  type="button"
                  onClick={() => toggleSort("survival")}
                  className="flex items-center gap-1.5 font-bold hover:text-white mx-auto"
                >
                  <span>30d Survival</span>
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th scope="col" className="py-3.5 px-4 text-center">
                <span>Dominant Class</span>
              </th>
              <th scope="col" className="py-3.5 px-6 text-right">
                <span>Drilldown</span>
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/60">
            {sortedList.map((row) => {
              const categoryBadge = getCategoryBadge(row.dominant_category);
              const medianVal = row.median_resolution_hours ?? 0;
              const barWidth = Math.min(100, Math.max(10, (medianVal / maxMedian) * 100));

              // Color based on velocity
              let speedColor = "text-emerald-400 bg-emerald-500";
              if (medianVal > 100) speedColor = "text-amber-400 bg-amber-500";
              else if (medianVal > 24) speedColor = "text-sky-400 bg-sky-500";

              return (
                <tr
                  key={row.repo_name}
                  className="hover:bg-slate-800/40 transition-colors group"
                >
                  {/* Repo Name */}
                  <td className="py-4 px-6 font-semibold">
                    <Link
                      href={`/repos/${row.owner}/${row.name}`}
                      className="flex flex-col group-hover:text-indigo-400 transition-colors"
                    >
                      <span className="text-white text-sm font-bold">
                        {row.owner}/{row.name}
                      </span>
                      <span className="text-[11px] text-slate-500 font-normal">
                        582 standardized issues
                      </span>
                    </Link>
                  </td>

                  {/* Stars */}
                  <td className="py-4 px-4 text-center">
                    <span className="inline-flex items-center gap-1 font-bold text-slate-200">
                      <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />
                      {formatNumber(row.stars_count)}
                    </span>
                  </td>

                  {/* Median SLA */}
                  <td className="py-4 px-4">
                    <div className="space-y-1.5 max-w-[140px]">
                      <div className="flex items-center justify-between">
                        <span className="font-extrabold text-sm text-white">
                          {formatHours(row.median_resolution_hours)}
                        </span>
                        {medianVal <= 24 && (
                          <span className="px-1.5 py-0.2 text-[9px] font-bold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 rounded">
                            Rapid
                          </span>
                        )}
                      </div>
                      <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                        <div
                          style={{ width: `${barWidth}%` }}
                          className={`h-full ${speedColor.split(" ")[1]} rounded-full`}
                        />
                      </div>
                    </div>
                  </td>

                  {/* Rapid Closures */}
                  <td className="py-4 px-4 text-center">
                    <span className="inline-flex items-center gap-1 font-bold text-indigo-300">
                      <Flame className="w-3.5 h-3.5 text-indigo-400" />
                      {row.rapid_closure_percentage.toFixed(1)}%
                    </span>
                  </td>

                  {/* 30-Day Survival */}
                  <td className="py-4 px-4 text-center">
                    <span className="inline-flex items-center gap-1 font-medium text-slate-300">
                      <TrendingDown className="w-3.5 h-3.5 text-slate-500" />
                      {row.survival_rate_30d ? formatPercent(row.survival_rate_30d) : "N/A"}
                    </span>
                  </td>

                  {/* Dominant Category */}
                  <td className="py-4 px-4 text-center">
                    <span
                      className={`inline-block px-2.5 py-0.5 text-[10px] font-semibold border rounded-md ${categoryBadge.className}`}
                    >
                      {categoryBadge.label}
                    </span>
                  </td>

                  {/* Actions */}
                  <td className="py-4 px-6 text-right">
                    <Link
                      href={`/repos/${row.owner}/${row.name}`}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white font-medium text-xs transition-colors"
                    >
                      <span>Explore</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
