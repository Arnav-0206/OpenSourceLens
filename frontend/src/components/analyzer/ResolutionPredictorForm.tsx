"use client";

import React from "react";
import {
  Clock,
  RotateCcw,
  Sparkles,
  GitBranch,
  Tag,
  UserCheck,
  HelpCircle,
  AlertCircle,
} from "lucide-react";

export interface ResolutionPredictorFormProps {
  repoName: string;
  setRepoName: (val: string) => void;
  title: string;
  setTitle: (val: string) => void;
  body: string;
  setBody: (val: string) => void;
  canonicalLabel: string;
  setCanonicalLabel: (val: string) => void;
  authorAssociation: string;
  setAuthorAssociation: (val: string) => void;
  onSubmit: (e?: React.FormEvent) => void;
  isLoading: boolean;
  onReset: () => void;
}

const SAMPLE_SCENARIOS = [
  {
    label: "Critical Crash Hotfix",
    repo: "microsoft/vscode",
    title: "Critical regression: Extension host crashes on startup with SIGSEGV",
    body: "Uncaught segmentation fault in node terminal worker during window initialization. Prevents all extensions from starting.",
    canonicalLabel: "bug",
    authorAssociation: "MEMBER",
  },
  {
    label: "Complex Feature RFC",
    repo: "pytorch/pytorch",
    title: "Design RFC: Distributed tensor parallelism for heterogeneous accelerator clusters",
    body: "Proposal to integrate cross-node tensor sharding kernels with heterogeneous memory topology awareness.",
    canonicalLabel: "feature",
    authorAssociation: "CONTRIBUTOR",
  },
  {
    label: "Docs / RBAC Fix",
    repo: "kubernetes/kubernetes",
    title: "Fix incorrect cluster-admin RBAC role reference in kube-dns guide",
    body: "Updating documentation to reference correct ClusterRole verbs for coredns configmap access.",
    canonicalLabel: "other",
    authorAssociation: "COLLABORATOR",
  },
];

const TARGET_REPOSITORIES = [
  { value: "microsoft/vscode", label: "microsoft / vscode (Triage: ~7.7h median)" },
  { value: "pytorch/pytorch", label: "pytorch / pytorch (Triage: ~14.2h median)" },
  { value: "kubernetes/kubernetes", label: "kubernetes / kubernetes (Triage: ~46.5h median)" },
  { value: "huggingface/transformers", label: "huggingface / transformers (Triage: ~28.0h median)" },
];

const CATEGORIES = [
  { value: "bug", label: "Bug Report (Defect)" },
  { value: "feature", label: "Feature Request (Enhancement)" },
  { value: "other", label: "Other / Documentation / RFC" },
];

const AUTHOR_ASSOCIATIONS = [
  { value: "MEMBER", label: "MEMBER — Core repository team member" },
  { value: "COLLABORATOR", label: "COLLABORATOR — Project maintainer / collaborator" },
  { value: "CONTRIBUTOR", label: "CONTRIBUTOR — Prior accepted contributor" },
  { value: "NONE", label: "NONE — First-time or external reporter" },
];

export default function ResolutionPredictorForm({
  repoName,
  setRepoName,
  title,
  setTitle,
  body,
  setBody,
  canonicalLabel,
  setCanonicalLabel,
  authorAssociation,
  setAuthorAssociation,
  onSubmit,
  isLoading,
  onReset,
}: ResolutionPredictorFormProps) {
  const isValidTitle = title.trim().length >= 3;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (isValidTitle && !isLoading) {
        onSubmit();
      }
    }
  };

  const applyScenario = (s: (typeof SAMPLE_SCENARIOS)[number]) => {
    setRepoName(s.repo);
    setTitle(s.title);
    setBody(s.body);
    setCanonicalLabel(s.canonicalLabel);
    setAuthorAssociation(s.authorAssociation);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
      {/* Header & Sample Chips */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Clock className="w-5 h-5 text-emerald-400" />
            <span>Resolution Velocity & SLA Forecast</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Predict expected time-to-close hours, speed tier bracket, and IQR confidence intervals.
          </p>
        </div>

        {/* Preset scenario buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-400">Scenarios:</span>
          {SAMPLE_SCENARIOS.map((s) => (
            <button
              key={s.label}
              type="button"
              onClick={() => applyScenario(s)}
              className="px-2.5 py-1 text-xs font-medium rounded-lg border border-slate-700 bg-slate-800 text-slate-300 hover:text-white hover:border-emerald-500/50 hover:bg-slate-800/80 transition-all hover:scale-105 active:scale-95"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={onSubmit} onKeyDown={handleKeyDown} className="mt-6 space-y-6">
        {/* Repository Scope & Canonical Category */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="pred-repo"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center gap-1.5"
            >
              <GitBranch className="w-3.5 h-3.5 text-slate-400" />
              <span>Target Repository</span>
              <span className="text-rose-400">*</span>
            </label>
            <select
              id="pred-repo"
              required
              value={repoName}
              onChange={(e) => setRepoName(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
            >
              {TARGET_REPOSITORIES.map((repo) => (
                <option key={repo.value} value={repo.value}>
                  {repo.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label
              htmlFor="pred-category"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center gap-1.5"
            >
              <Tag className="w-3.5 h-3.5 text-slate-400" />
              <span>Issue Category</span>
              <span className="text-rose-400">*</span>
            </label>
            <select
              id="pred-category"
              value={canonicalLabel}
              onChange={(e) => setCanonicalLabel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat.value} value={cat.value}>
                  {cat.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Issue Title Input */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label
              htmlFor="pred-title"
              className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1"
            >
              <span>Issue Title</span>
              <span className="text-rose-400">*</span>
            </label>
            <span
              className={`text-xs ${
                title.length === 0
                  ? "text-slate-500"
                  : title.length < 3
                  ? "text-amber-400 font-medium"
                  : "text-slate-400"
              }`}
            >
              {title.length} characters
            </span>
          </div>
          <input
            id="pred-title"
            type="text"
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Memory leak during tensor forward pass with autograd..."
            className={`w-full bg-slate-950 border rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition-colors ${
              title.length > 0 && title.trim().length < 3
                ? "border-amber-500 focus:border-amber-400 focus:ring-1 focus:ring-amber-400"
                : "border-slate-800 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
            }`}
          />
        </div>

        {/* Author Association & Description */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="sm:col-span-1">
            <label
              htmlFor="pred-author"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center gap-1.5"
            >
              <UserCheck className="w-3.5 h-3.5 text-slate-400" />
              <span>Author Association</span>
            </label>
            <select
              id="pred-author"
              value={authorAssociation}
              onChange={(e) => setAuthorAssociation(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
            >
              {AUTHOR_ASSOCIATIONS.map((assoc) => (
                <option key={assoc.value} value={assoc.value}>
                  {assoc.label}
                </option>
              ))}
            </select>
          </div>

          <div className="sm:col-span-2">
            <label
              htmlFor="pred-body"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center justify-between"
            >
              <span>Issue Description / Context (Optional)</span>
              <span className="text-slate-500 lowercase font-normal">text length feature</span>
            </label>
            <input
              id="pred-body"
              type="text"
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder="Paste stack traces or reproduction context..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
            />
          </div>
        </div>

        {/* Action Row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <HelpCircle className="w-4 h-4" />
            <span>
              Predictions combine Random Forest regression with empirical percentile windows
            </span>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={onReset}
              disabled={isLoading || (!title && !body)}
              aria-label="Reset resolution predictor fields"
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-800 bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-sm transition-colors min-h-[44px] disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>

            <button
              type="submit"
              disabled={isLoading || !isValidTitle}
              aria-label="Execute resolution velocity prediction"
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white font-semibold text-sm shadow-lg shadow-emerald-600/25 transition-all hover:scale-[1.02] active:scale-[0.98] min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Estimating Velocity...</span>
                </>
              ) : (
                <>
                  <Clock className="w-4 h-4" />
                  <span>Predict Resolution Velocity</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
