"use client";

import React from "react";
import {
  Sparkles,
  RotateCcw,
  Cpu,
  Brain,
  Sliders,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  GitBranch,
} from "lucide-react";

export interface ClassificationFormProps {
  title: string;
  setTitle: (value: string) => void;
  body: string;
  setBody: (value: string) => void;
  repoName: string;
  setRepoName: (value: string) => void;
  modelType: "distilbert" | "svm" | "logistic_regression";
  setModelType: (value: "distilbert" | "svm" | "logistic_regression") => void;
  onSubmit: (e?: React.FormEvent) => void;
  isLoading: boolean;
  onReset: () => void;
}

const SAMPLE_PRESETS = [
  {
    label: "Bug Report",
    badge: "bug",
    badgeClass: "bg-rose-500/10 text-rose-400 border-rose-500/20",
    title: "Crash in extension host when reloading window with active terminal",
    body: "Steps to Reproduce:\n1. Open integrated terminal running node worker.\n2. Execute Developer: Reload Window.\n3. The extension host process terminates unexpectedly with SIGSEGV code 139.\n\nOS: Ubuntu 22.04 LTS / x86_64\nExpected: Clean teardown of terminal worker threads without segmentation fault.",
    repo: "microsoft/vscode",
    model: "distilbert" as const,
  },
  {
    label: "Feature Request",
    badge: "feature",
    badgeClass: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    title: "Add support for FlashAttention-2 in transformer forward pass",
    body: "Proposal:\nFlashAttention-2 provides up to 2x speedup compared to standard scaled dot-product attention on Ampere and Hopper GPUs.\n\nProposed API:\nWould love to expose torch.nn.functional.scaled_dot_product_attention backend flags to explicitly enable flash-attn v2 kernel bindings in native dispatch.",
    repo: "pytorch/pytorch",
    model: "distilbert" as const,
  },
  {
    label: "Docs / RFC",
    badge: "other",
    badgeClass: "bg-slate-500/10 text-slate-400 border-slate-500/20",
    title: "Clarify RBAC permissions required for ingress controller Helm chart",
    body: "The documentation page on security contexts does not state which ClusterRole bindings are mandatory when running in unprivileged namespace mode.\n\nPlease add a reference table detailing each required API group and verb for custom service accounts.",
    repo: "kubernetes/kubernetes",
    model: "svm" as const,
  },
];

const TARGET_REPOSITORIES = [
  { value: "", label: "Generic / Any Repository" },
  { value: "microsoft/vscode", label: "microsoft / vscode" },
  { value: "pytorch/pytorch", label: "pytorch / pytorch" },
  { value: "kubernetes/kubernetes", label: "kubernetes / kubernetes" },
  { value: "huggingface/transformers", label: "huggingface / transformers" },
];

const MODEL_OPTIONS: Array<{
  id: "distilbert" | "svm" | "logistic_regression";
  name: string;
  tag: string;
  accuracy: string;
  description: string;
  icon: typeof Brain;
}> = [
  {
    id: "distilbert",
    name: "DistilBERT Champion",
    tag: "Fine-Tuned Transformer",
    accuracy: "81.3% test acc",
    description: "Deep contextual bidirectionality with fine-tuned self-attention",
    icon: Brain,
  },
  {
    id: "svm",
    name: "Linear SVM",
    tag: "TF-IDF Baseline",
    accuracy: "76.8% test acc",
    description: "Max-margin linear hyperplane with sublinear TF-IDF word n-grams",
    icon: Sliders,
  },
  {
    id: "logistic_regression",
    name: "Logistic Regression",
    tag: "TF-IDF Baseline",
    accuracy: "75.4% test acc",
    description: "L2-regularized multinomial probability estimation",
    icon: Cpu,
  },
];

export default function ClassificationForm({
  title,
  setTitle,
  body,
  setBody,
  repoName,
  setRepoName,
  modelType,
  setModelType,
  onSubmit,
  isLoading,
  onReset,
}: ClassificationFormProps) {
  const isTitleValid = title.trim().length >= 3;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (isTitleValid && !isLoading) {
        onSubmit();
      }
    }
  };

  const applyPreset = (preset: (typeof SAMPLE_PRESETS)[number]) => {
    setTitle(preset.title);
    setBody(preset.body);
    setRepoName(preset.repo);
    setModelType(preset.model);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
      {/* Header & Quick Samples */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-indigo-400" />
            <span>Issue Text & Inference Configuration</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Submit GitHub issue content to classify into taxonomy categories (bug, feature, other).
          </p>
        </div>

        {/* Preset sample buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-400">Load Sample:</span>
          {SAMPLE_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => applyPreset(preset)}
              className={`px-2.5 py-1 text-xs font-medium rounded-lg border transition-all hover:scale-105 active:scale-95 ${preset.badgeClass}`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={onSubmit} onKeyDown={handleKeyDown} className="mt-6 space-y-6">
        {/* Model Selection Cards */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2.5">
            Select Model Architecture
          </label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {MODEL_OPTIONS.map((model) => {
              const isSelected = modelType === model.id;
              const Icon = model.icon;
              return (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => setModelType(model.id)}
                  className={`text-left p-3.5 rounded-xl border transition-all ${
                    isSelected
                      ? "bg-indigo-950/40 border-indigo-500/70 shadow-md shadow-indigo-950/50 ring-1 ring-indigo-500/40"
                      : "bg-slate-900/80 border-slate-800 hover:border-slate-700 hover:bg-slate-800/50"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="flex items-center gap-1.5 font-bold text-sm text-white">
                      <Icon
                        className={`w-4 h-4 ${
                          isSelected ? "text-indigo-400" : "text-slate-400"
                        }`}
                      />
                      {model.name}
                    </span>
                    {isSelected && (
                      <CheckCircle2 className="w-4 h-4 text-indigo-400 shrink-0" />
                    )}
                  </div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                      {model.tag}
                    </span>
                    <span className="text-[10px] font-bold text-emerald-400">
                      {model.accuracy}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-tight">
                    {model.description}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Repository Context (Optional) */}
        <div>
          <label
            htmlFor="repo-context"
            className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2 flex items-center justify-between"
          >
            <span className="flex items-center gap-1.5">
              <GitBranch className="w-3.5 h-3.5 text-slate-400" />
              Repository Context (Optional)
            </span>
            <span className="text-[11px] font-normal text-slate-500 lowercase">
              audits inference in repository log
            </span>
          </label>
          <select
            id="repo-context"
            value={repoName}
            onChange={(e) => setRepoName(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors"
          >
            {TARGET_REPOSITORIES.map((repo) => (
              <option key={repo.value} value={repo.value}>
                {repo.label}
              </option>
            ))}
          </select>
        </div>

        {/* Issue Title Input */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label
              htmlFor="issue-title"
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
            id="issue-title"
            type="text"
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Memory leak in renderer on display reconnect..."
            className={`w-full bg-slate-950 border rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition-colors ${
              title.length > 0 && title.trim().length < 3
                ? "border-amber-500 focus:border-amber-400 focus:ring-1 focus:ring-amber-400"
                : "border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            }`}
          />
          {title.length > 0 && title.trim().length < 3 && (
            <p className="mt-1.5 text-xs text-amber-400 flex items-center gap-1">
              <AlertCircle className="w-3.5 h-3.5" />
              Title should be at least 3 characters long for reliable tokenization.
            </p>
          )}
        </div>

        {/* Issue Body Textarea */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label
              htmlFor="issue-body"
              className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5"
            >
              <span>Issue Body / Description</span>
              <span className="text-slate-500 font-normal lowercase">(optional)</span>
            </label>
            <span className="text-xs text-slate-500">
              Stack traces, code blocks & Markdown supported
            </span>
          </div>
          <textarea
            id="issue-body"
            rows={5}
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Paste complete issue description, reproduction steps, error logs, or proposed design..."
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors resize-y min-h-[120px]"
          />
        </div>

        {/* Actions Row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800/80">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <HelpCircle className="w-4 h-4" />
            <span>
              Tip: Press <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300 font-mono text-[11px]">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300 font-mono text-[11px]">Enter</kbd> to classify instantly
            </span>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={onReset}
              disabled={isLoading || (!title && !body)}
              aria-label="Clear form fields"
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-800 bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-sm transition-colors min-h-[44px] disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Clear</span>
            </button>

            <button
              type="submit"
              disabled={isLoading || !isTitleValid}
              aria-label="Submit issue for classification"
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-semibold text-sm shadow-lg shadow-indigo-600/25 transition-all hover:scale-[1.02] active:scale-[0.98] min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Inferring Category...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Classify Issue</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
