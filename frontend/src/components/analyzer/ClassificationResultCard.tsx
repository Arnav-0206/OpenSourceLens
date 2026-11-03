"use client";

import React, { useState } from "react";
import {
  Sparkles,
  Zap,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Search,
  Clock,
  Code2,
  Cpu,
  ShieldCheck,
  Activity,
} from "lucide-react";
import { ClassificationResponse } from "@/lib/types";
import { formatPercent, getCategoryBadge } from "@/lib/utils";

export interface ClassificationResultCardProps {
  result: ClassificationResponse | null;
  isLoading: boolean;
  error: string | null;
  onFindRelated?: (title: string) => void;
  onPredictResolution?: (title: string, category: string) => void;
  isOfflineSimulated?: boolean;
}

export default function ClassificationResultCard({
  result,
  isLoading,
  error,
  onFindRelated,
  onPredictResolution,
  isOfflineSimulated = false,
}: ClassificationResultCardProps) {
  const [showPreprocessed, setShowPreprocessed] = useState(false);

  // 1. Loading State
  if (isLoading) {
    return (
      <div className="bg-slate-900/60 border border-indigo-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden backdrop-blur-sm">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-indigo-500/5 to-transparent animate-pulse" />
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-lg bg-indigo-500/20 flex items-center justify-center animate-spin">
                <Activity className="w-4 h-4 text-indigo-400" />
              </div>
              <div>
                <div className="h-4 w-32 bg-slate-800 rounded animate-pulse" />
                <div className="h-3 w-48 bg-slate-800/60 rounded mt-1.5 animate-pulse" />
              </div>
            </div>
            <div className="h-6 w-20 bg-slate-800 rounded-full animate-pulse" />
          </div>

          <div className="pt-4 border-t border-slate-800 space-y-3">
            <div className="h-3 w-full bg-slate-800 rounded animate-pulse" />
            <div className="h-3 w-4/5 bg-slate-800 rounded animate-pulse" />
            <div className="h-3 w-2/3 bg-slate-800 rounded animate-pulse" />
          </div>

          <div className="flex items-center justify-center pt-3 text-xs text-indigo-300 gap-2">
            <Sparkles className="w-4 h-4 animate-bounce text-indigo-400" />
            <span>Tokenizing text & executing model forward pass...</span>
          </div>
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
            <h3 className="text-sm font-semibold text-rose-300">Classification Inference Failed</h3>
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
          <Sparkles className="w-6 h-6" />
        </div>
        <h3 className="text-sm font-semibold text-slate-300">Awaiting Issue Submission</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
          Enter an issue title and description or load a sample preset above, then click &quot;Classify Issue&quot;.
        </p>
      </div>
    );
  }

  // 4. Result Available
  const badge = getCategoryBadge(result.predicted_label);
  const confidencePct = Math.round(result.confidence_score * 1000) / 10;
  const isHighConfidence = result.confidence_score >= 0.8;
  const isModerateConfidence = result.confidence_score >= 0.5 && result.confidence_score < 0.8;

  // Normalized probability entries sorted descending
  const probEntries = Object.entries(result.probabilities || {}).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm space-y-6">
      {/* Top Banner: Predicted Label & Confidence */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Predicted Taxonomy Category
          </span>
          <div className="flex items-center gap-3">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl font-bold text-base border shadow-sm ${badge.className}`}
            >
              <span>{badge.label}</span>
            </span>

            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-xs">
              <ShieldCheck
                className={`w-3.5 h-3.5 ${
                  isHighConfidence
                    ? "text-emerald-400"
                    : isModerateConfidence
                    ? "text-amber-400"
                    : "text-rose-400"
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

        {/* Inference Latency & Model Used */}
        <div className="flex items-center gap-2 sm:self-center">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 font-mono">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>{result.inference_time_ms.toFixed(1)} ms</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span className="capitalize">{result.model_used}</span>
          </div>
        </div>
      </div>

      {/* Probabilities Distribution Breakdown */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Probability Distribution
          </h4>
          <span className="text-[11px] text-slate-400">Softmax output vector</span>
        </div>

        <div className="space-y-3">
          {probEntries.map(([category, prob]) => {
            const catBadge = getCategoryBadge(category);
            const probPct = Math.round(prob * 1000) / 10;
            const isWinner = category === result.predicted_label;

            let barColor = "bg-slate-600";
            if (category === "bug") barColor = "bg-rose-500";
            if (category === "feature") barColor = "bg-emerald-500";
            if (category === "other") barColor = "bg-indigo-500";

            return (
              <div key={category} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-semibold capitalize ${
                        isWinner ? "text-white" : "text-slate-400"
                      }`}
                    >
                      {category}
                    </span>
                    {isWinner && (
                      <span className="text-[10px] font-bold text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-1.5 py-0.2 rounded">
                        Winner
                      </span>
                    )}
                  </div>
                  <span className="font-mono text-slate-300 font-medium">
                    {probPct}% ({prob.toFixed(4)})
                  </span>
                </div>

                <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${barColor} ${
                      isWinner ? "opacity-100 shadow-sm" : "opacity-40"
                    }`}
                    style={{ width: `${Math.max(probPct, 2)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Preprocessing Inspector Drawer (Collapsible) */}
      <div className="pt-2 border-t border-slate-800">
        <button
          type="button"
          onClick={() => setShowPreprocessed(!showPreprocessed)}
          className="w-full flex items-center justify-between py-2 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          <span className="flex items-center gap-1.5 font-medium">
            <Code2 className="w-3.5 h-3.5 text-indigo-400" />
            Inspect Preprocessed / Cleaned Text
          </span>
          {showPreprocessed ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {showPreprocessed && (
          <div className="mt-3 p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs space-y-3">
            <div>
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                Clean Title (Noise stripped & normalized):
              </span>
              <p className="text-slate-200 bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                {result.clean_title || "(empty)"}
              </p>
            </div>

            {result.clean_body && (
              <div>
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  Clean Body (Code blocks & markdown processed):
                </span>
                <p className="text-slate-300 bg-slate-900/80 p-2.5 rounded-lg border border-slate-800 max-h-40 overflow-y-auto whitespace-pre-wrap">
                  {result.clean_body}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Pipeline Navigation Quick-Links */}
      <div className="pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-slate-400">Next Pipeline Steps:</span>

        <div className="flex flex-wrap items-center gap-2">
          {onFindRelated && (
            <button
              type="button"
              onClick={() => onFindRelated(result.clean_title || "")}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
            >
              <Search className="w-3.5 h-3.5 text-cyan-400" />
              <span>Search Related Issues</span>
            </button>
          )}

          {onPredictResolution && (
            <button
              type="button"
              onClick={() =>
                onPredictResolution(
                  result.clean_title || "",
                  result.predicted_label
                )
              }
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
            >
              <Clock className="w-3.5 h-3.5 text-emerald-400" />
              <span>Predict Resolution Velocity</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
