import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge multiple Tailwind CSS classes with resolution of conflicting rules.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format duration in hours into human-readable duration strings (e.g., "7.7h", "2.4d").
 */
export function formatHours(hours: number | null | undefined): string {
  if (hours === null || hours === undefined || isNaN(hours)) {
    return "N/A";
  }
  if (hours < 1) {
    const mins = Math.round(hours * 60);
    return `${mins}m`;
  }
  if (hours < 24) {
    return `${hours.toFixed(1)}h`;
  }
  const days = hours / 24;
  if (days < 30) {
    return `${days.toFixed(1)}d`;
  }
  const months = days / 30;
  return `${months.toFixed(1)}mo`;
}

/**
 * Format large numbers with compact SI notation (e.g., 160500 -> "160.5k").
 */
export function formatNumber(num: number | null | undefined): string {
  if (num === null || num === undefined || isNaN(num)) {
    return "0";
  }
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M`;
  }
  if (num >= 1_000) {
    return `${(num / 1_000).toFixed(1)}k`;
  }
  return num.toLocaleString();
}

/**
 * Format fractional value [0.0, 1.0] to clean percentage string (e.g., 0.842 -> "84.2%").
 */
export function formatPercent(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) {
    return "0.0%";
  }
  // If already scaled to [0, 100], handle gracefully
  const pct = val <= 1.0 && val >= 0.0 ? val * 100 : val;
  return `${pct.toFixed(1)}%`;
}

/**
 * Format ISO datetime string into concise localized date.
 */
export function formatDate(isoString: string | null | undefined): string {
  if (!isoString) return "N/A";
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return isoString;
  }
}

/**
 * Category styling tokens mapping canonical issue classifications to accessible color schemes.
 */
export function getCategoryBadge(category: string): {
  label: string;
  className: string;
} {
  const norm = category.toLowerCase();
  switch (norm) {
    case "bug":
      return {
        label: "Bug",
        className: "bg-rose-500/15 text-rose-400 border-rose-500/30",
      };
    case "feature":
      return {
        label: "Feature",
        className: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
      };
    case "documentation":
      return {
        label: "Documentation",
        className: "bg-sky-500/15 text-sky-400 border-sky-500/30",
      };
    case "question":
      return {
        label: "Question",
        className: "bg-amber-500/15 text-amber-400 border-amber-500/30",
      };
    default:
      return {
        label: "Other",
        className: "bg-slate-500/15 text-slate-400 border-slate-500/30",
      };
  }
}

/**
 * Speed tier styling tokens mapping velocity brackets to intuitive visual cues.
 */
export function getSpeedTierBadge(tier: string | null | undefined): {
  label: string;
  className: string;
} {
  if (!tier) {
    return {
      label: "Pending",
      className: "bg-slate-800 text-slate-400 border-slate-700",
    };
  }

  switch (tier) {
    case "< 1 Day":
      return {
        label: "< 1 Day (Rapid)",
        className: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
      };
    case "1-7 Days":
      return {
        label: "1-7 Days (Standard)",
        className: "bg-sky-500/15 text-sky-300 border-sky-500/30",
      };
    case "7-30 Days":
      return {
        label: "7-30 Days (Extended)",
        className: "bg-amber-500/15 text-amber-300 border-amber-500/30",
      };
    case "30+ Days":
      return {
        label: "30+ Days (Backlog)",
        className: "bg-rose-500/15 text-rose-300 border-rose-500/30",
      };
    default:
      return {
        label: tier,
        className: "bg-slate-800 text-slate-300 border-slate-700",
      };
  }
}
