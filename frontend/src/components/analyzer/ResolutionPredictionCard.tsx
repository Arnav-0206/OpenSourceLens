"use client";

import React from "react";
import {
  Clock,
  Zap,
  Cpu,
  ShieldCheck,
  AlertTriangle,
  Sparkles,
  Calendar,
  Hourglass,
  Activity,
  ArrowRight,
  TrendingDown,
  Info,
  CheckCircle2,
} from "lucide-react";
import { ResolutionPredictionResponse } from "@/lib/types";
import { getSpeedTierBadge } from "@/lib/utils";

export interface ResolutionPredictionCardProps {
  result: ResolutionPredictionResponse | null;
  isLoading: boolean;
  error: string | null;
  isOfflineSimulated?: boolean;
  repoName?: string;
}

export default function ResolutionPredictionCard({
  result,
  isLoading,
  error,
  isOfflineSimulated = false,
  repoName,
}: ResolutionPredictionCardProps) {
  // 1. Loading State
  if (isLoading) {
    return (
      <div className="bg-slate-900/60 border border-emerald-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm space-y-4">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-emerald-500/5 to-transparent animate-pulse" />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-lg bg-emerald-500/20 flex items-center justify-center animate-spin">
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <div className="h-4 w-32 bg-slate-800 rounded animate-pulse" />
              <div className="h-3 w-48 bg-slate-800/60 rounded mt-1.5 animate-pulse" />
            </div>
          </div>
          <div className="h-6 w-20 bg-slate-800 rounded-full animate-pulse" />
        </div>

        <div className="pt-4 border-t border-slate-800 space-y-3">
          <div className="h-8 w-1/2 bg-slate-800 rounded animate-pulse" />
          <div className="h-3 w-full bg-slate-800 rounded animate-pulse" />
          <div className="h-3 w-2/3 bg-slate-800 rounded animate-pulse" />
        </div>

        <div className="flex items-center justify-center pt-3 text-xs text-emerald-300 gap-2">
          <Clock className="w-4 h-4 animate-bounce text-emerald-400" />
          <span>Computing velocity features & running regression tree ensembles...</span>
        </div>
      </div>
    );
  }

  // 2. Error State
  if (error) {
    return (
      <div className="bg-rose-950/20 border border-rose-500/30 rounded-2xl p-6 text-rose-200 backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-rose-300">
              Resolution Prediction Failed
            </h3>
            <p className="text-xs text-rose-200/80 mt-1 leading-relaxed">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  // 3. Empty State
  if (!result) {
    return (
      <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl p-8 text-center">
        <div className="w-12 h-12 rounded-2xl bg-slate-800/60 text-slate-500 flex items-center justify-center mx-auto mb-3">
          <Clock className="w-6 h-6" />
        </div>
        <h3 className="text-sm font-semibold text-slate-300">
          Awaiting Issue SLA Parameters
        </h3>
        <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
          Select a target repository and enter an issue title or choose a scenario above, then click &quot;Predict Resolution Velocity&quot;.
        </p>
      </div>
    );
  }

  // 4. Result Available
  const tierBadge = getSpeedTierBadge(result.predicted_speed_tier);
  const lowerBadge = getSpeedTierBadge(result.lower_bound_tier);
  const upperBadge = getSpeedTierBadge(result.upper_bound_tier);

  const confidencePct = Math.round(result.confidence_score * 1000) / 10;
  const isHighConfidence = result.confidence_score >= 0.7;

  const probEntries = Object.entries(result.tier_probabilities || {}).sort(
    (a, b) => (Number(b[1]) || 0) - (Number(a[1]) || 0)
  );

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-6">
      {/* Top Banner: Speed Tier & Confidence */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Predicted Resolution Velocity Tier
          </span>
          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold text-base border shadow-sm ${tierBadge.className}`}
            >
              <span>{tierBadge.label}</span>
            </span>

            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-xs">
              <ShieldCheck
                className={`w-3.5 h-3.5 ${
                  isHighConfidence ? "text-emerald-400" : "text-amber-400"
                }`}
              />
              <span className="text-slate-300 font-semibold">{confidencePct}%</span>
              <span className="text-slate-400">confidence</span>
            </div>

            {isOfflineSimulated && (
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                Offline Simulated
              </span>
            )}
          </div>
        </div>

        {/* Latency & Model Used */}
        <div className="flex items-center gap-2 sm:self-center">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 font-mono">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>{result.inference_time_ms.toFixed(1)} ms</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>{result.model_used}</span>
          </div>
        </div>
      </div>

      {/* Bound Horizon Window */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Optimistic Lower Bound */}
        <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Hourglass className="w-3.5 h-3.5 text-emerald-400" />
            <span>Optimistic Lower Bound</span>
          </span>
          <div className="pt-1.5">
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-bold border ${lowerBadge.className}`}
            >
              {lowerBadge.label}
            </span>
            <p className="text-[11px] text-slate-400 mt-1">Fast-track triage window</p>
          </div>
        </div>

        {/* Conservative Upper Bound */}
        <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 space-y-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            <span>Conservative Upper Bound</span>
          </span>
          <div className="pt-1.5">
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-bold border ${upperBadge.className}`}
            >
              {upperBadge.label}
            </span>
            <p className="text-[11px] text-slate-400 mt-1">Extended review SLA ceiling</p>
          </div>
        </div>
      </div>

      {/* Advisory Factors Signals */}
      {result.advisory_factors && result.advisory_factors.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Contributing Advisory Factors:</span>
          </span>
          <ul className="space-y-1.5">
            {result.advisory_factors.map((factor, i) => (
              <li
                key={i}
                className="text-xs text-slate-400 flex items-start gap-2 leading-relaxed"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                <span>{factor}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Speed Tier Probability Distribution */}
      {probEntries.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Velocity Tier Probabilities
            </h4>
            <span className="text-[11px] text-slate-400">Model likelihood distribution</span>
          </div>

          <div className="space-y-3">
            {probEntries.map(([tierKey, rawProb]) => {
              const probNum = Number(rawProb) || 0;
              const probPct = Math.round(probNum * 1000) / 10;
              const isWinner =
                tierKey.toLowerCase() === result.predicted_speed_tier.toLowerCase();
              const badge = getSpeedTierBadge(tierKey);

              return (
                <div key={tierKey} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span
                        className={`font-semibold ${
                          isWinner ? "text-white" : "text-slate-400"
                        }`}
                      >
                        {badge.label}
                      </span>
                      {isWinner && (
                        <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.2 rounded">
                          Winner
                        </span>
                      )}
                    </div>
                    <span className="font-mono text-slate-300 font-medium">
                      {probPct}% ({probNum.toFixed(3)})
                    </span>
                  </div>

                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isWinner
                          ? "bg-emerald-500 opacity-100 shadow-sm"
                          : "bg-slate-600 opacity-40"
                      }`}
                      style={{ width: `${Math.max(probPct, 2)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Reliability Notice */}
      {result.reliability_notice && (
        <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-500 flex items-start gap-1.5">
          <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
          <span>{result.reliability_notice}</span>
        </div>
      )}
    </div>
  );
}
