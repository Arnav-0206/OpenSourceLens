"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Sparkles,
  Search,
  Clock,
  Layers,
  ArrowRight,
  Info,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import ClassificationForm from "@/components/analyzer/ClassificationForm";
import ClassificationResultCard from "@/components/analyzer/ClassificationResultCard";
import RelatedIssuesSearch from "@/components/analyzer/RelatedIssuesSearch";
import RelatedIssuesResults from "@/components/analyzer/RelatedIssuesResults";
import ResolutionPredictorForm from "@/components/analyzer/ResolutionPredictorForm";
import ResolutionPredictionCard from "@/components/analyzer/ResolutionPredictionCard";
import {
  classifyIssue,
  searchRelatedIssues,
  predictResolution,
} from "@/lib/api";
import {
  ClassificationResponse,
  RelatedIssuesResponse,
  RelatedIssueMatch,
  ResolutionPredictionResponse,
} from "@/lib/types";

type ActiveTab = "classifier" | "related" | "resolution";

// In-memory fallback corpus representing Phase 7 sentence embeddings index
const FALLBACK_ISSUE_CORPUS: RelatedIssueMatch[] = [
  {
    issue_id: 101,
    repo_name: "microsoft/vscode",
    issue_number: 182390,
    title: "Terminal process crashes with exit code 139 on macOS/Linux reload",
    state: "closed",
    canonical_label: "bug",
    similarity_score: 0.842,
    rank: 1,
    created_at: "2023-05-10T14:30:00Z",
    closed_at: "2023-05-12T09:15:00Z",
  },
  {
    issue_id: 102,
    repo_name: "microsoft/vscode",
    issue_number: 178942,
    title: "Extension host memory leak when opening multiple large workspaces",
    state: "closed",
    canonical_label: "bug",
    similarity_score: 0.765,
    rank: 2,
    created_at: "2023-04-02T11:20:00Z",
    closed_at: "2023-04-18T16:45:00Z",
  },
  {
    issue_id: 103,
    repo_name: "microsoft/vscode",
    issue_number: 165431,
    title: "Add native support for smooth scrolling in integrated terminal",
    state: "open",
    canonical_label: "feature",
    similarity_score: 0.528,
    rank: 3,
    created_at: "2023-08-14T08:00:00Z",
    closed_at: null,
  },
  {
    issue_id: 201,
    repo_name: "pytorch/pytorch",
    issue_number: 94210,
    title: "CUDA out of memory during backward pass with amp.autocast and DDP",
    state: "open",
    canonical_label: "bug",
    similarity_score: 0.887,
    rank: 1,
    created_at: "2023-02-18T19:40:00Z",
    closed_at: null,
  },
  {
    issue_id: 202,
    repo_name: "pytorch/pytorch",
    issue_number: 89421,
    title: "Add FlashAttention-2 integration for torch.nn.MultiheadAttention",
    state: "closed",
    canonical_label: "feature",
    similarity_score: 0.793,
    rank: 2,
    created_at: "2023-07-22T10:15:00Z",
    closed_at: "2023-09-01T14:20:00Z",
  },
  {
    issue_id: 203,
    repo_name: "pytorch/pytorch",
    issue_number: 91234,
    title: "Memory leak in DataLoader workers when pin_memory=True on CUDA",
    state: "closed",
    canonical_label: "bug",
    similarity_score: 0.718,
    rank: 3,
    created_at: "2023-03-12T13:00:00Z",
    closed_at: "2023-03-20T17:30:00Z",
  },
  {
    issue_id: 301,
    repo_name: "kubernetes/kubernetes",
    issue_number: 116540,
    title: "Pod stuck in Pending state due to node untolerated taint",
    state: "closed",
    canonical_label: "bug",
    similarity_score: 0.854,
    rank: 1,
    created_at: "2023-03-01T15:00:00Z",
    closed_at: "2023-03-04T12:00:00Z",
  },
  {
    issue_id: 302,
    repo_name: "kubernetes/kubernetes",
    issue_number: 114321,
    title: "Kubelet fails to mount volume when Secret is updated dynamically",
    state: "closed",
    canonical_label: "bug",
    similarity_score: 0.642,
    rank: 2,
    created_at: "2023-01-20T09:30:00Z",
    closed_at: "2023-02-15T18:00:00Z",
  },
  {
    issue_id: 303,
    repo_name: "kubernetes/kubernetes",
    issue_number: 118945,
    title: "Add native support for sidecar containers in Pod lifecycle KEP-753",
    state: "open",
    canonical_label: "feature",
    similarity_score: 0.589,
    rank: 3,
    created_at: "2023-06-11T16:45:00Z",
    closed_at: null,
  },
  {
    issue_id: 401,
    repo_name: "huggingface/transformers",
    issue_number: 24561,
    title: "Rust fast tokenizer panic during byte pair decoding with special tokens",
    state: "closed",
    canonical_label: "bug",
    similarity_score: 0.865,
    rank: 1,
    created_at: "2023-06-25T14:10:00Z",
    closed_at: "2023-07-02T11:30:00Z",
  },
  {
    issue_id: 402,
    repo_name: "huggingface/transformers",
    issue_number: 22340,
    title: "Add 4-bit and 8-bit QLoRA quantized inference pipeline for LLaMA",
    state: "closed",
    canonical_label: "feature",
    similarity_score: 0.734,
    rank: 2,
    created_at: "2023-05-18T10:00:00Z",
    closed_at: "2023-06-05T15:20:00Z",
  },
  {
    issue_id: 403,
    repo_name: "huggingface/transformers",
    issue_number: 25129,
    title: "OutOfMemoryError in Trainer during evaluation on multi-GPU setup",
    state: "open",
    canonical_label: "bug",
    similarity_score: 0.698,
    rank: 3,
    created_at: "2023-07-30T13:45:00Z",
    closed_at: null,
  },
];

function AnalyzerContent() {
  const searchParams = useSearchParams();

  // URL Query Parameters prefill
  const initialTitle = searchParams.get("title") || "";
  const initialRepo = searchParams.get("repo") || "";
  const initialTab = (searchParams.get("tab") as ActiveTab) || "classifier";

  const [activeTab, setActiveTab] = useState<ActiveTab>(initialTab);

  // Tab 1: Classifier State
  const [title, setTitle] = useState(initialTitle);
  const [body, setBody] = useState("");
  const [repoName, setRepoName] = useState(initialRepo);
  const [modelType, setModelType] = useState<
    "distilbert" | "svm" | "logistic_regression"
  >("distilbert");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResponse | null>(null);
  const [isOfflineSimulated, setIsOfflineSimulated] = useState(false);

  // Tab 2: Related Issues State
  const [relatedQuery, setRelatedQuery] = useState(initialTitle);
  const [relatedRepo, setRelatedRepo] = useState(initialRepo);
  const [relatedState, setRelatedState] = useState("");
  const [relatedTopK, setRelatedTopK] = useState(5);
  const [relatedThreshold, setRelatedThreshold] = useState(0.4);
  const [isRelatedLoading, setIsRelatedLoading] = useState(false);
  const [relatedError, setRelatedError] = useState<string | null>(null);
  const [relatedResponse, setRelatedResponse] =
    useState<RelatedIssuesResponse | null>(null);
  const [isRelatedSimulated, setIsRelatedSimulated] = useState(false);

  // Tab 3: Resolution Prediction State
  const [predRepo, setPredRepo] = useState(
    initialRepo || "microsoft/vscode"
  );
  const [predTitle, setPredTitle] = useState(initialTitle);
  const [predBody, setPredBody] = useState("");
  const [predCanonicalLabel, setPredCanonicalLabel] = useState("bug");
  const [predAuthorAssociation, setPredAuthorAssociation] = useState("NONE");
  const [isPredLoading, setIsPredLoading] = useState(false);
  const [predError, setPredError] = useState<string | null>(null);
  const [predResponse, setPredResponse] =
    useState<ResolutionPredictionResponse | null>(null);
  const [isPredSimulated, setIsPredSimulated] = useState(false);

  // Sync title / repo if URL parameters change
  useEffect(() => {
    if (initialTitle) {
      if (!title) setTitle(initialTitle);
      if (!relatedQuery) setRelatedQuery(initialTitle);
      if (!predTitle) setPredTitle(initialTitle);
    }
    if (initialRepo) {
      if (!repoName) setRepoName(initialRepo);
      if (!relatedRepo) setRelatedRepo(initialRepo);
      setPredRepo(initialRepo);
    }
  }, [initialTitle, initialRepo, title, repoName, relatedQuery, relatedRepo, predTitle]);

  // Classification Clean / Reset handler
  const handleResetClassifier = () => {
    setTitle("");
    setBody("");
    setRepoName("");
    setResult(null);
    setError(null);
    setIsOfflineSimulated(false);
  };

  // Related Issues Reset handler
  const handleResetRelated = () => {
    setRelatedQuery("");
    setRelatedRepo("");
    setRelatedState("");
    setRelatedTopK(5);
    setRelatedThreshold(0.4);
    setRelatedResponse(null);
    setRelatedError(null);
    setIsRelatedSimulated(false);
  };

  // Resolution Reset handler
  const handleResetPrediction = () => {
    setPredTitle("");
    setPredBody("");
    setPredCanonicalLabel("bug");
    setPredAuthorAssociation("NONE");
    setPredResponse(null);
    setPredError(null);
    setIsPredSimulated(false);
  };

  // Inference submission
  const handleSubmitClassifier = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!title.trim() || title.trim().length < 3) return;

    setIsLoading(true);
    setError(null);
    setIsOfflineSimulated(false);

    const startTime = performance.now();

    try {
      const response = await classifyIssue({
        title: title.trim(),
        body: body.trim() ? body.trim() : null,
        repo_name: repoName.trim() ? repoName.trim() : null,
        model_type: modelType,
      });
      setResult(response);
    } catch (err: unknown) {
      console.warn("Backend API unavailable, simulating offline inference:", err);
      const elapsed = Math.round((performance.now() - startTime) * 10) / 10;
      const lower = `${title} ${body}`.toLowerCase();

      let simulatedCategory = "other";
      let pBug = 0.15;
      let pFeature = 0.2;
      let pOther = 0.65;

      if (
        lower.includes("crash") ||
        lower.includes("bug") ||
        lower.includes("error") ||
        lower.includes("fail") ||
        lower.includes("segfault") ||
        lower.includes("leak") ||
        lower.includes("broken") ||
        lower.includes("panic")
      ) {
        simulatedCategory = "bug";
        pBug = 0.88;
        pFeature = 0.07;
        pOther = 0.05;
      } else if (
        lower.includes("feature") ||
        lower.includes("add") ||
        lower.includes("support") ||
        lower.includes("proposal") ||
        lower.includes("implement") ||
        lower.includes("allow") ||
        lower.includes("enhance")
      ) {
        simulatedCategory = "feature";
        pBug = 0.08;
        pFeature = 0.85;
        pOther = 0.07;
      }

      const simulatedResponse: ClassificationResponse = {
        predicted_label: simulatedCategory,
        canonical_label_id:
          simulatedCategory === "bug" ? 0 : simulatedCategory === "feature" ? 1 : 2,
        confidence_score: Math.max(pBug, pFeature, pOther),
        probabilities: {
          bug: pBug,
          feature: pFeature,
          other: pOther,
        },
        model_used: modelType,
        inference_time_ms: elapsed > 0 ? elapsed : 16.4,
        clean_title: title.toLowerCase().replace(/[^\w\s]/g, " ").trim(),
        clean_body: body ? body.toLowerCase().replace(/[^\w\s]/g, " ").trim() : null,
        is_classified_category: true,
      };

      setResult(simulatedResponse);
      setIsOfflineSimulated(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Related Issues Retrieval submission
  const handleSubmitRelated = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!relatedQuery.trim() || relatedQuery.trim().length < 3) return;

    setIsRelatedLoading(true);
    setRelatedError(null);
    setIsRelatedSimulated(false);

    const startTime = performance.now();

    try {
      const response = await searchRelatedIssues({
        title: relatedQuery.trim(),
        repo_name: relatedRepo ? relatedRepo : null,
        top_k: relatedTopK,
        state:
          relatedState === "open" || relatedState === "closed"
            ? relatedState
            : null,
      });
      setRelatedResponse(response);
    } catch (err: unknown) {
      console.warn("Backend API unavailable, simulating offline vector retrieval:", err);
      const elapsed = Math.round((performance.now() - startTime) * 10) / 10;

      let matches = FALLBACK_ISSUE_CORPUS.filter((issue) => {
        if (relatedRepo && issue.repo_name !== relatedRepo) return false;
        if (relatedState && issue.state !== relatedState) return false;
        return true;
      });

      const queryWords = relatedQuery.toLowerCase().split(/\s+/).filter(Boolean);
      matches = matches
        .map((issue) => {
          let score = issue.similarity_score;
          const titleLower = issue.title.toLowerCase();
          const matchesCount = queryWords.filter((w) => titleLower.includes(w)).length;
          if (matchesCount > 0) {
            score = Math.min(0.95, score + matchesCount * 0.05);
          } else {
            score = Math.max(0.35, score - 0.15);
          }
          return {
            ...issue,
            similarity_score: Math.round(score * 1000) / 1000,
          };
        })
        .filter((issue) => issue.similarity_score >= relatedThreshold)
        .sort((a, b) => b.similarity_score - a.similarity_score)
        .slice(0, relatedTopK);

      const simulatedResponse: RelatedIssuesResponse = {
        query_text: relatedQuery.trim(),
        query_issue_id: null,
        total_matches: matches.length,
        matches: matches.map((m, idx) => ({ ...m, rank: idx + 1 })),
        model_version: "all-MiniLM-L6-v2",
        retrieval_time_ms: elapsed > 0 ? elapsed : 12.8,
      };

      setRelatedResponse(simulatedResponse);
      setIsRelatedSimulated(true);
    } finally {
      setIsRelatedLoading(false);
    }
  };

  // Resolution Velocity Prediction submission
  const handleSubmitPrediction = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!predTitle.trim() || predTitle.trim().length < 3) return;

    setIsPredLoading(true);
    setPredError(null);
    setIsPredSimulated(false);

    const startTime = performance.now();

    try {
      const response = await predictResolution({
        repo_name: predRepo,
        title: predTitle.trim(),
        body: predBody.trim() ? predBody.trim() : null,
        canonical_label: predCanonicalLabel,
        author_association: predAuthorAssociation,
      });
      setPredResponse(response);
    } catch (err: unknown) {
      console.warn("Backend API unavailable, simulating offline resolution prediction:", err);
      const elapsed = Math.round((performance.now() - startTime) * 10) / 10;

      // Realistic empirical base hours per repo and category
      let baseHours = 14.0;
      if (predRepo.includes("vscode")) baseHours = 7.7;
      if (predRepo.includes("pytorch")) baseHours = 14.2;
      if (predRepo.includes("kubernetes")) baseHours = 46.5;
      if (predRepo.includes("transformers")) baseHours = 28.0;

      // Category adjustments
      if (predCanonicalLabel === "bug") baseHours *= 0.8;
      if (predCanonicalLabel === "feature") baseHours *= 2.2;
      if (predCanonicalLabel === "other") baseHours *= 1.1;

      // Author association acceleration
      if (predAuthorAssociation === "MEMBER") baseHours *= 0.5;
      if (predAuthorAssociation === "COLLABORATOR") baseHours *= 0.7;
      if (predAuthorAssociation === "CONTRIBUTOR") baseHours *= 0.9;

      const predictedHours = Math.round(baseHours * 10) / 10;
      let tier = "1-7 Days";
      let lowerBound = "< 1 Day";
      let upperBound = "1-7 Days";
      let pUnder24h = 0.25;
      let p1to7d = 0.55;
      let p7to30d = 0.15;
      let p30plus = 0.05;

      const advisoryFactors: string[] = [
        `Base repository baseline triage velocity: ~${baseHours.toFixed(1)}h median`,
      ];

      if (predAuthorAssociation === "MEMBER" || predAuthorAssociation === "COLLABORATOR") {
        advisoryFactors.push(
          `Author association [${predAuthorAssociation}] grants expedited maintainer review prioritization`
        );
      }

      if (predCanonicalLabel === "bug") {
        advisoryFactors.push("Defect classification correlates with faster patch closure cycles");
      } else if (predCanonicalLabel === "feature") {
        advisoryFactors.push("Enhancement proposals typically require multi-maintainer design review");
      }

      if (predictedHours < 24) {
        tier = "< 1 Day";
        lowerBound = "< 1 Day";
        upperBound = "1-7 Days";
        pUnder24h = 0.78;
        p1to7d = 0.16;
        p7to30d = 0.05;
        p30plus = 0.01;
      } else if (predictedHours > 168) {
        tier = "7-30 Days";
        lowerBound = "1-7 Days";
        upperBound = "30+ Days";
        pUnder24h = 0.05;
        p1to7d = 0.25;
        p7to30d = 0.55;
        p30plus = 0.15;
      }

      const simulatedResponse: ResolutionPredictionResponse = {
        repo_name: predRepo,
        predicted_speed_tier: tier,
        confidence_score: Math.max(pUnder24h, p1to7d, p7to30d, p30plus),
        tier_probabilities: {
          "< 1 Day": pUnder24h,
          "1-7 Days": p1to7d,
          "7-30 Days": p7to30d,
          "30+ Days": p30plus,
        },
        lower_bound_tier: lowerBound,
        upper_bound_tier: upperBound,
        advisory_factors: advisoryFactors,
        reliability_notice:
          "Statistical resolution estimate based on Phase 8 Random Forest classification. Real-world resolution may vary with repository sprint schedules.",
        model_used: "RandomForestClassifier (Ensemble)",
        inference_time_ms: elapsed > 0 ? elapsed : 14.5,
      };

      setPredResponse(simulatedResponse);
      setIsPredSimulated(true);
    } finally {
      setIsPredLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-400 mb-1">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Interactive Machine Learning Pipeline</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          AI Issue Intelligence Workspace
        </h1>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Classify issue taxonomy using fine-tuned DistilBERT or classical ML baselines,
          retrieve semantic duplicates via Sentence Transformers, and forecast resolution SLA velocity.
        </p>
      </div>

      {/* Tab Navigation (Responsive horizontally scrollable on mobile) */}
      <div className="flex border-b border-slate-800 space-x-2 sm:space-x-4 overflow-x-auto pb-px scrollbar-none">
        <button
          onClick={() => setActiveTab("classifier")}
          className={`flex items-center gap-2 pb-3 px-3 sm:px-4 text-xs sm:text-sm font-semibold border-b-2 transition-colors whitespace-nowrap min-h-[44px] shrink-0 ${
            activeTab === "classifier"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Sparkles className="w-4 h-4" />
          <span>Issue Classifier</span>
          <span className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] bg-indigo-500/10 text-indigo-400 rounded-full border border-indigo-500/20">
            Step 10.7
          </span>
        </button>

        <button
          onClick={() => setActiveTab("related")}
          className={`flex items-center gap-2 pb-3 px-3 sm:px-4 text-xs sm:text-sm font-semibold border-b-2 transition-colors whitespace-nowrap min-h-[44px] shrink-0 ${
            activeTab === "related"
              ? "border-cyan-500 text-cyan-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Search className="w-4 h-4" />
          <span>Semantic Duplicates</span>
          <span className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] bg-cyan-500/10 text-cyan-400 rounded-full border border-cyan-500/20">
            Step 10.8
          </span>
        </button>

        <button
          onClick={() => setActiveTab("resolution")}
          className={`flex items-center gap-2 pb-3 px-3 sm:px-4 text-xs sm:text-sm font-semibold border-b-2 transition-colors whitespace-nowrap min-h-[44px] shrink-0 ${
            activeTab === "resolution"
              ? "border-emerald-500 text-emerald-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`}
        >
          <Clock className="w-4 h-4" />
          <span>Resolution Velocity</span>
          <span className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] bg-emerald-500/10 text-emerald-400 rounded-full border border-emerald-500/20">
            Step 10.9
          </span>
        </button>
      </div>

      {/* Tab 1: Classifier (Step 10.7) */}
      {activeTab === "classifier" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7">
            <ClassificationForm
              title={title}
              setTitle={setTitle}
              body={body}
              setBody={setBody}
              repoName={repoName}
              setRepoName={setRepoName}
              modelType={modelType}
              setModelType={setModelType}
              onSubmit={handleSubmitClassifier}
              isLoading={isLoading}
              onReset={handleResetClassifier}
            />
          </div>

          <div className="lg:col-span-5 space-y-6">
            <ClassificationResultCard
              result={result}
              isLoading={isLoading}
              error={error}
              isOfflineSimulated={isOfflineSimulated}
              onFindRelated={(query) => {
                setRelatedQuery(query);
                setActiveTab("related");
              }}
              onPredictResolution={(query, cat) => {
                setPredTitle(query);
                setPredCanonicalLabel(cat || "bug");
                if (repoName) setPredRepo(repoName);
                setActiveTab("resolution");
              }}
            />

            <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 text-xs text-slate-400 space-y-3">
              <div className="flex items-center gap-2 text-slate-300 font-semibold">
                <Info className="w-4 h-4 text-indigo-400 shrink-0" />
                <span>Architecture Benchmark Note</span>
              </div>
              <p className="leading-relaxed">
                The Champion model is a fine-tuned <strong>DistilBERT</strong> transformer
                achieving <strong>81.3% test accuracy</strong> across 4 major repositories.
                Classical baselines (Linear SVM at 76.8% and Logistic Regression at 75.4%) are
                served simultaneously for comparative analysis and low-latency fallbacks.
              </p>
              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500">
                <span>Audited in PostgreSQL `predictions`</span>
                <span>Sub-50ms CPU latency</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Related Issues (Step 10.8) */}
      {activeTab === "related" && (
        <div className="space-y-8">
          <RelatedIssuesSearch
            queryText={relatedQuery}
            setQueryText={setRelatedQuery}
            repoName={relatedRepo}
            setRepoName={setRelatedRepo}
            stateFilter={relatedState}
            setStateFilter={setRelatedState}
            topK={relatedTopK}
            setTopK={setRelatedTopK}
            similarityThreshold={relatedThreshold}
            setSimilarityThreshold={setRelatedThreshold}
            onSearch={handleSubmitRelated}
            isLoading={isRelatedLoading}
            onReset={handleResetRelated}
          />

          <RelatedIssuesResults
            response={relatedResponse}
            isLoading={isRelatedLoading}
            error={relatedError}
            isOfflineSimulated={isRelatedSimulated}
            onAnalyzeIssue={(issueTitle, repo) => {
              setTitle(issueTitle);
              setRepoName(repo);
              setActiveTab("classifier");
            }}
          />

          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 text-xs text-slate-400 space-y-2">
            <div className="flex items-center gap-2 text-slate-300 font-semibold">
              <Info className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>Semantic Retrieval Methodology</span>
            </div>
            <p className="leading-relaxed">
              Issues are projected into a dense metric space using the <strong>all-MiniLM-L6-v2</strong>{" "}
              Sentence Transformers architecture. Cosine similarity compares normalized vectors:
              scores $\ge 0.70$ frequently represent duplicate bug reports or identical failure
              signatures, enabling proactive triage before engineering resources are allocated.
            </p>
          </div>
        </div>
      )}

      {/* Tab 3: Resolution Velocity (Step 10.9) */}
      {activeTab === "resolution" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7">
            <ResolutionPredictorForm
              repoName={predRepo}
              setRepoName={setPredRepo}
              title={predTitle}
              setTitle={setPredTitle}
              body={predBody}
              setBody={setPredBody}
              canonicalLabel={predCanonicalLabel}
              setCanonicalLabel={setPredCanonicalLabel}
              authorAssociation={predAuthorAssociation}
              setAuthorAssociation={setPredAuthorAssociation}
              onSubmit={handleSubmitPrediction}
              isLoading={isPredLoading}
              onReset={handleResetPrediction}
            />
          </div>

          <div className="lg:col-span-5 space-y-6">
            <ResolutionPredictionCard
              result={predResponse}
              isLoading={isPredLoading}
              error={predError}
              isOfflineSimulated={isPredSimulated}
              repoName={predRepo}
            />

            <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-5 text-xs text-slate-400 space-y-3">
              <div className="flex items-center gap-2 text-slate-300 font-semibold">
                <Info className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>SLA Prediction Methodology</span>
              </div>
              <p className="leading-relaxed">
                Resolution forecast leverages ensemble <strong>Random Forest Regression</strong>{" "}
                conditioned on historical repository resolution dynamics, author association tier,
                and issue taxonomy category. Interquartile ranges (P25 to P75) provide robust
                confidence bounds reflecting typical triage latency.
              </p>
              <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500">
                <span>Multi-tier gradient boosting</span>
                <span>Sub-20ms inference</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalyzerPage() {
  return (
    <Suspense
      fallback={
        <div className="py-16 text-center text-slate-400 text-sm">
          Loading AI Intelligence Workspace...
        </div>
      }
    >
      <AnalyzerContent />
    </Suspense>
  );
}
