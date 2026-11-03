"use client";

import { Filter, RotateCcw, Search, X } from "lucide-react";

interface IssueFilterBarProps {
  stateFilter: string;
  onStateChange: (state: string) => void;
  categoryFilter: string;
  onCategoryChange: (category: string) => void;
  speedTierFilter: string;
  onSpeedTierChange: (tier: string) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onReset: () => void;
  totalFiltered: number;
  totalRepoIssues: number;
}

export default function IssueFilterBar({
  stateFilter,
  onStateChange,
  categoryFilter,
  onCategoryChange,
  speedTierFilter,
  onSpeedTierChange,
  searchQuery,
  onSearchChange,
  onReset,
  totalFiltered,
  totalRepoIssues,
}: IssueFilterBarProps) {
  const hasActiveFilters =
    Boolean(stateFilter) ||
    Boolean(categoryFilter) ||
    Boolean(speedTierFilter) ||
    Boolean(searchQuery);

  return (
    <div className="glass-panel rounded-2xl p-4 border border-slate-800 bg-slate-900/60 space-y-4">
      <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3">
        {/* Search Bar */}
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search issue title or keywords..."
            className="w-full pl-10 pr-9 py-2 text-xs bg-slate-950/80 border border-slate-800 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/60"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => onSearchChange("")}
              className="absolute right-2.5 top-2.5 p-0.5 rounded text-slate-400 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Multi-facet dropdown selectors */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* State Filter */}
          <select
            value={stateFilter}
            onChange={(e) => onStateChange(e.target.value)}
            className="px-3 py-2 text-xs font-medium bg-slate-950/80 border border-slate-800 rounded-xl text-slate-200 focus:outline-none focus:border-indigo-500/60"
          >
            <option value="">All States</option>
            <option value="open">Open</option>
            <option value="closed">Closed</option>
          </select>

          {/* Canonical Category Filter */}
          <select
            value={categoryFilter}
            onChange={(e) => onCategoryChange(e.target.value)}
            className="px-3 py-2 text-xs font-medium bg-slate-950/80 border border-slate-800 rounded-xl text-slate-200 focus:outline-none focus:border-indigo-500/60"
          >
            <option value="">All Categories</option>
            <option value="bug">Bug</option>
            <option value="feature">Feature</option>
            <option value="documentation">Documentation</option>
            <option value="question">Question</option>
            <option value="other">Other</option>
          </select>

          {/* Velocity Speed Tier Filter */}
          <select
            value={speedTierFilter}
            onChange={(e) => onSpeedTierChange(e.target.value)}
            className="px-3 py-2 text-xs font-medium bg-slate-950/80 border border-slate-800 rounded-xl text-slate-200 focus:outline-none focus:border-indigo-500/60"
          >
            <option value="">All Velocities</option>
            <option value="< 1 Day">&lt; 1 Day (Rapid)</option>
            <option value="1-7 Days">1-7 Days (Standard)</option>
            <option value="7-30 Days">7-30 Days (Extended)</option>
            <option value="30+ Days">30+ Days (Backlog)</option>
          </select>

          {/* Reset button */}
          {hasActiveFilters && (
            <button
              type="button"
              onClick={onReset}
              className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-rose-300 hover:text-white bg-rose-950/30 hover:bg-rose-900/50 border border-rose-800/40 rounded-xl transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>

      {/* Count Indicator */}
      <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/60">
        <span className="flex items-center gap-1.5">
          <Filter className="w-3.5 h-3.5 text-indigo-400" />
          <span>
            Showing <strong className="text-white">{totalFiltered}</strong> matching issues{" "}
            <span className="text-slate-500">(out of {totalRepoIssues} total in dataset)</span>
          </span>
        </span>

        {hasActiveFilters && (
          <span className="text-[11px] text-indigo-300 font-medium">
            Active filters applied
          </span>
        )}
      </div>
    </div>
  );
}
