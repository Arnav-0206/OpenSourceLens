"use client";

import React from "react";
import {
  Search,
  RotateCcw,
  SlidersHorizontal,
  GitBranch,
  Layers,
  Sparkles,
  HelpCircle,
  Hash,
} from "lucide-react";

export interface RelatedIssuesSearchParams {
  queryText: string;
  setQueryText: (val: string) => void;
  repoName: string;
  setRepoName: (val: string) => void;
  stateFilter: string;
  setStateFilter: (val: string) => void;
  topK: number;
  setTopK: (val: number) => void;
  similarityThreshold: number;
  setSimilarityThreshold: (val: number) => void;
  onSearch: (e?: React.FormEvent) => void;
  isLoading: boolean;
  onReset: () => void;
}

const SAMPLE_QUERIES = [
  {
    label: "VS Code Memory Leak",
    query: "Memory leak in extension host terminal worker on window reload",
    repo: "microsoft/vscode",
  },
  {
    label: "PyTorch CUDA OOM",
    query: "CUDA out of memory during backward pass with gradient checkpointing",
    repo: "pytorch/pytorch",
  },
  {
    label: "K8s Pod Scheduling",
    query: "Pod stuck in Pending state due to node untolerated taint",
    repo: "kubernetes/kubernetes",
  },
  {
    label: "Transformers Tokenizer",
    query: "Rust fast tokenizer panic during byte pair decoding with special tokens",
    repo: "huggingface/transformers",
  },
];

const TARGET_REPOSITORIES = [
  { value: "", label: "All Repositories (Global Corpus)" },
  { value: "microsoft/vscode", label: "microsoft / vscode" },
  { value: "pytorch/pytorch", label: "pytorch / pytorch" },
  { value: "kubernetes/kubernetes", label: "kubernetes / kubernetes" },
  { value: "huggingface/transformers", label: "huggingface / transformers" },
];

export default function RelatedIssuesSearch({
  queryText,
  setQueryText,
  repoName,
  setRepoName,
  stateFilter,
  setStateFilter,
  topK,
  setTopK,
  similarityThreshold,
  setSimilarityThreshold,
  onSearch,
  isLoading,
  onReset,
}: RelatedIssuesSearchParams) {
  const isValidQuery = queryText.trim().length >= 3;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isValidQuery && !isLoading) {
        onSearch();
      }
    }
  };

  const applyPreset = (preset: (typeof SAMPLE_QUERIES)[number]) => {
    setQueryText(preset.query);
    setRepoName(preset.repo);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
      {/* Header & Sample Chips */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Search className="w-5 h-5 text-cyan-400" />
            <span>Semantic Retrieval & Duplicate Search</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Query dense vector embeddings (384-d `all-MiniLM-L6-v2`) to retrieve duplicate or related issues.
          </p>
        </div>

        {/* Preset chips */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-400">Quick Samples:</span>
          {SAMPLE_QUERIES.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => applyPreset(preset)}
              className="px-2.5 py-1 text-xs font-medium rounded-lg border border-slate-700 bg-slate-800 text-slate-300 hover:text-white hover:border-cyan-500/50 hover:bg-slate-800/80 transition-all hover:scale-105 active:scale-95"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={onSearch} onKeyDown={handleKeyDown} className="mt-6 space-y-6">
        {/* Search Query Input */}
        <div>
          <label
            htmlFor="query-text"
            className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2 flex items-center justify-between"
          >
            <span className="flex items-center gap-1.5">
              <span>Natural Language Query / Issue Title</span>
              <span className="text-rose-400">*</span>
            </span>
            <span className="text-xs text-slate-500 font-normal lowercase">
              semantic similarity matching
            </span>
          </label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
              <Search className="w-4 h-4" />
            </div>
            <input
              id="query-text"
              type="text"
              required
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="e.g. Memory leak during tensor forward pass with autograd..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
            />
          </div>
        </div>

        {/* Filter Controls Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Repository Selector */}
          <div>
            <label
              htmlFor="repo-scope"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center gap-1.5"
            >
              <GitBranch className="w-3.5 h-3.5 text-slate-400" />
              <span>Target Repository</span>
            </label>
            <select
              id="repo-scope"
              value={repoName}
              onChange={(e) => setRepoName(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            >
              {TARGET_REPOSITORIES.map((repo) => (
                <option key={repo.value} value={repo.value}>
                  {repo.label}
                </option>
              ))}
            </select>
          </div>

          {/* Issue State Filter */}
          <div>
            <label
              htmlFor="state-filter"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center gap-1.5"
            >
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <span>Issue State</span>
            </label>
            <select
              id="state-filter"
              value={stateFilter}
              onChange={(e) => setStateFilter(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            >
              <option value="">All States (Open & Closed)</option>
              <option value="open">Open Only</option>
              <option value="closed">Closed Only</option>
            </select>
          </div>

          {/* Top-K Results */}
          <div>
            <label
              htmlFor="top-k"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5 flex items-center gap-1.5"
            >
              <Hash className="w-3.5 h-3.5 text-slate-400" />
              <span>Top-K Neighbors</span>
            </label>
            <select
              id="top-k"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            >
              <option value={5}>Top 5 Nearest</option>
              <option value={10}>Top 10 Nearest</option>
              <option value={20}>Top 20 Nearest</option>
            </select>
          </div>

          {/* Cosine Similarity Threshold Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label
                htmlFor="threshold"
                className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5"
              >
                <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                <span>Min Similarity</span>
              </label>
              <span className="font-mono text-xs font-bold text-cyan-400">
                {(similarityThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <div className="pt-1">
              <input
                id="threshold"
                type="range"
                min="0.2"
                max="0.85"
                step="0.05"
                value={similarityThreshold}
                onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 px-0.5">
                <span>20% (Broad)</span>
                <span>85% (Strict)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <HelpCircle className="w-4 h-4" />
            <span>
              Matches are scored using cosine similarity over normalized Sentence Transformers embeddings
            </span>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={onReset}
              disabled={isLoading || !queryText}
              aria-label="Reset search filters and query"
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-slate-800 bg-slate-900 hover:bg-slate-800 text-slate-300 font-medium text-sm transition-colors min-h-[44px] disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>

            <button
              type="submit"
              disabled={isLoading || !isValidQuery}
              aria-label="Execute semantic duplicate search"
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-lg shadow-cyan-600/25 transition-all hover:scale-[1.02] active:scale-[0.98] min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Searching Vectors...</span>
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  <span>Search Related Issues</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
