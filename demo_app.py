"""
OpenSourceLens — Milestone Demonstration Application (Phases 0–7)
Interactive Streamlit Dashboard for Academic Milestone Evaluation.

Features:
1. Live Dual-Model Classification: Fine-Tuned DistilBERT vs. Classical Champion LinearSVC
   - Confidence scoring, uncertainty triage thresholds (<0.55), and local feature explainability.
2. Semantic Related-Issue Retrieval & Duplicate Detection:
   - Real-time cosine similarity search across 2,328 curated issues via all-MiniLM-L6-v2.
   - Confidence tiers (HIGH / MEDIUM / LOW), duplicate warning alerts (>=0.85), and direct GitHub links.
3. Research Analytics & Benchmark Gallery:
   - Empirical performance comparison tables (Phases 4, 5, 6, and 7).
   - Publication-quality evaluation figures (confusion matrices, ROC curves, loss curves, retrieval benchmarks).
4. Interactive Preset Scenarios:
   - 1-click test cases for Bug, Feature, Documentation, and Duplicate queries.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline classes
from ml.deep_learning.pipeline import TransformerIssueClassifier
from ml.nlp.final_pipeline import FinalClassicalNLPPipeline
from ml.related_issues.pipeline import RelatedIssuePipeline

# Ensure pickle compatibility for classical pipeline
sys.modules["__main__"].FinalClassicalNLPPipeline = FinalClassicalNLPPipeline

# ---------------------------------------------------------------------------
# Preset Queries for Quick Academic Demonstrations
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, str]] = {
    "Select a preset or custom input...": {
        "title": "",
        "body": "",
        "description": "Choose a preset from the dropdown or type custom text below.",
    },
    "🐛 Preset 1: Bug Report (VS Code Crash)": {
        "title": "VS Code editor crashes when opening large TypeScript workspace with out of memory error",
        "body": (
            "Steps to Reproduce:\n"
            "1. Open a monorepo workspace containing >50,000 TypeScript files.\n"
            "2. Trigger global symbol search via Ctrl+Shift+F.\n"
            "3. The renderer process hangs and immediately crashes with: "
            "'FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory'.\n"
            "Expected: Graceful truncation of search index without terminating the main application."
        ),
        "description": "Demonstrates bug detection with high confidence and memory leak keywords.",
    },
    "✨ Preset 2: Feature Request (PyTorch Quantization)": {
        "title": "Add native GPU acceleration support for quantized Int4 matrix multiplications in PyTorch",
        "body": (
            "Motivation:\n"
            "Currently, running 4-bit quantized LLM inference requires custom external kernels. "
            "We would like to propose a first-class torch.nn.functional.qlinear_int4 API that leverages "
            "Tensor Core hardware acceleration on modern CUDA architectures.\n"
            "Proposed API:\n"
            "output = torch.ops.aten.int4_matmul(input_tensor, quantized_weights, scales, zeros)"
        ),
        "description": "Demonstrates feature enhancement detection with high confidence.",
    },
    "📖 Preset 3: Documentation / Other (Hugging Face)": {
        "title": "Update documentation links and fix typos in transformers quickstart tutorial guide",
        "body": (
            "The quickstart tutorial documentation page for AutoModelForCausalLM contains a broken hyperlink "
            "pointing to an outdated v4.28 example notebook. Also fixed two grammatical typos in the tokenizer "
            "truncation section on line 42 of docs/source/en/quickstart.md."
        ),
        "description": "Demonstrates documentation/other classification.",
    },
    "🚨 Preset 4: Potential Duplicate (Memory / CUDA Issue)": {
        "title": "CUDA error: out of memory when running multi-gpu distributed training with torch.cuda.empty_cache",
        "body": (
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.76 GiB total capacity; "
            "9.50 GiB already allocated; 1.20 GiB free; 9.80 GiB reserved in total by PyTorch). "
            "Calling torch.cuda.empty_cache() does not release reserved cache blocks."
        ),
        "description": "Demonstrates semantic similarity retrieval and duplicate detection alerting (>=85% score).",
    },
}


# ---------------------------------------------------------------------------
# Figure Catalog for Research Tab
# ---------------------------------------------------------------------------
FIGURE_GALLERY = {
    "🏆 Model Comparison & Tradeoffs": [
        {
            "title": "Figure 22: Comprehensive Model Comparison (Radar & Performance Bars)",
            "path": "ml/deep_learning/figures/fig22_model_comparison_radar_bars.png",
            "caption": (
                "Multi-metric comparison (Accuracy, Macro F1, Precision, Recall, Inference Speed) across "
                "Phase 4 Baseline Logistic Regression, Phase 4 Baseline LinearSVC, Phase 5 Enhanced LinearSVC, "
                "and Phase 6 Fine-Tuned DistilBERT."
            ),
        },
        {
            "title": "Figure 16: Phase 4 vs. Phase 5 Performance Gains",
            "path": "ml/nlp/figures/fig16_phase4_vs_phase5_performance.png",
            "caption": (
                "Empirical validation of NLP enhancements: 3x title weighting, software token normalization, "
                "and sublinear TF-IDF scaling yield significant F1 gains across all issue categories."
            ),
        },
    ],
    "🎯 Confusion Matrices & Calibration": [
        {
            "title": "Figure 20: Fine-Tuned DistilBERT Confusion Matrix",
            "path": "ml/deep_learning/figures/fig20_transformer_confusion_matrix.png",
            "caption": (
                "Normalized confusion matrix on unseen test split (N=350) showing 76.5% test accuracy "
                "with balanced per-class diagonal concentration across bug, feature, and other categories."
            ),
        },
        {
            "title": "Figure 21: DistilBERT ROC and Precision-Recall Curves",
            "path": "ml/deep_learning/figures/fig21_transformer_roc_pr_curves.png",
            "caption": (
                "Multi-class ROC curves (Macro-AUC: 0.90) and Precision-Recall curves verifying robust separation "
                "across varying decision thresholds."
            ),
        },
        {
            "title": "Figure 10: Baseline Model Confusion Matrices (Phase 4)",
            "path": "ml/classification/figures/fig10_baseline_confusion_matrices.png",
            "caption": (
                "Side-by-side comparison of baseline Logistic Regression and LinearSVC classifiers on raw TF-IDF."
            ),
        },
    ],
    "📉 Deep Learning Training Dynamics": [
        {
            "title": "Figure 19: DistilBERT Fine-Tuning Loss & Accuracy Curves",
            "path": "ml/deep_learning/figures/fig19_transformer_training_curves.png",
            "caption": (
                "Training vs. validation loss and accuracy progression over 4 epochs (AdamW optimizer, lr=3e-5, "
                "linear warm-up, weight decay 0.01) confirming stable convergence without overfitting."
            ),
        },
        {
            "title": "Figure 18: Token Length Distribution & Truncation Budget",
            "path": "ml/deep_learning/figures/fig18_token_length_distribution.png",
            "caption": (
                "Distribution of issue token lengths supporting the chosen max_sequence_length=256 budget, "
                "capturing >92% of complete issue titles and descriptions."
            ),
        },
    ],
    "🔍 Semantic Retrieval & Duplicate Benchmarks": [
        {
            "title": "Figure 28: Retrieval Evaluation & Precision-Recall Benchmark",
            "path": "ml/related_issues/figures/fig28_retrieval_evaluation_benchmark.png",
            "caption": (
                "Precision@K, Recall@K, and MRR@K evaluation across K=[1, 3, 5, 10] on the 2,328 issue vector store. "
                "Achieves 84.2% Top-1 Precision and 96.8% Top-5 Recall."
            ),
        },
        {
            "title": "Figure 27: Similarity Threshold Calibration & Confidence Tiers",
            "path": "ml/related_issues/figures/fig27_similarity_threshold_calibration.png",
            "caption": (
                "Empirical calibration of similarity cutoff boundaries: HIGH (>=0.70), MEDIUM (0.55-0.70), "
                "LOW (0.45-0.55), and Duplicate Alert Threshold (>=0.85)."
            ),
        },
        {
            "title": "Figure 26: Top-K Nearest Neighbor Semantic Distribution",
            "path": "ml/related_issues/figures/fig26_topk_retrieval_analysis.png",
            "caption": (
                "Cosine similarity score decay profiles across rank positions 1 through 10, demonstrating sharp "
                "discrimination between true related issues and background corpus noise."
            ),
        },
        {
            "title": "Figure 24: Embedding Norms & Inference Latency Distribution",
            "path": "ml/related_issues/figures/fig24_embedding_norm_and_latency.png",
            "caption": (
                "Verification of L2 unit-normalization (norm=1.000) and query latency distribution "
                "(mean: ~14 ms on CPU), proving production readiness."
            ),
        },
    ],
    "📊 Dataset & Exploratory Analysis": [
        {
            "title": "Figure 9: Master Exploratory Data Analysis Dashboard",
            "path": "ml/eda/figures/fig9_master_eda_dashboard.png",
            "caption": (
                "Comprehensive overview of dataset characteristics across target repositories: "
                "microsoft/vscode, pytorch/pytorch, kubernetes/kubernetes, huggingface/transformers."
            ),
        },
        {
            "title": "Figure 1: Target Label Distribution & Class Proportions",
            "path": "ml/eda/figures/fig1_label_distribution.png",
            "caption": (
                "Empirical class balance: Bug (42.5%), Feature (34.1%), Other (23.4%) across 2,328 curated issues."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Cached Model Loaders (Streamlit Resource Cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Fine-Tuned DistilBERT Classifier (Phase 6)...")
def load_transformer_classifier() -> Optional[TransformerIssueClassifier]:
    """Load production TransformerIssueClassifier."""
    try:
        model_path = PROJECT_ROOT / "models" / "distilbert_finetuned"
        if not model_path.exists():
            return None
        return TransformerIssueClassifier(model_dir=model_path)
    except Exception as exc:
        st.error(f"Failed to load DistilBERT model: {exc}")
        return None


@st.cache_resource(show_spinner="Loading Classical NLP Pipeline (Phase 5)...")
def load_classical_pipeline() -> Optional[FinalClassicalNLPPipeline]:
    """Load production FinalClassicalNLPPipeline."""
    try:
        model_path = PROJECT_ROOT / "models" / "final_classical_nlp_pipeline.joblib"
        if not model_path.exists():
            return None
        return FinalClassicalNLPPipeline.load(model_path=model_path)
    except Exception as exc:
        st.error(f"Failed to load Classical NLP pipeline: {exc}")
        return None


@st.cache_resource(show_spinner="Loading Semantic Retrieval Vector Index (Phase 7)...")
def load_retrieval_pipeline() -> Optional[RelatedIssuePipeline]:
    """Load production RelatedIssuePipeline with precomputed embeddings."""
    try:
        store_path = PROJECT_ROOT / "models" / "related_issues"
        if not store_path.exists():
            return None
        pipeline = RelatedIssuePipeline(store_dir=store_path)
        pipeline._ensure_initialized()
        return pipeline
    except Exception as exc:
        st.error(f"Failed to load Related Issue pipeline: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helper Inference & Formatting Functions (Testable)
# ---------------------------------------------------------------------------
def get_preset_issues() -> Dict[str, Dict[str, str]]:
    """Return catalog of interactive preset issues."""
    return PRESETS


def classify_with_transformer(
    classifier: TransformerIssueClassifier,
    title: str,
    body: str = "",
) -> Dict[str, Any]:
    """Execute single-issue inference using the fine-tuned DistilBERT classifier."""
    return classifier.predict_one(title=title, body=body)


def classify_with_classical(
    pipeline: FinalClassicalNLPPipeline,
    title: str,
    body: str = "",
    top_k_features: int = 6,
) -> Dict[str, Any]:
    """Execute single-issue inference and explainability using the classical pipeline."""
    return pipeline.explain_prediction(title=title, body=body, top_k=top_k_features)


def retrieve_related_issues(
    pipeline: RelatedIssuePipeline,
    title: str,
    body: str = "",
    repo_filter: Optional[str] = None,
    top_k: int = 5,
    min_confidence: str = "ALL",
    state_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute semantic issue retrieval and duplicate detection."""
    actual_repo = None if repo_filter in (None, "All Repositories") else repo_filter
    actual_state = None if state_filter in (None, "all") else state_filter
    return pipeline.find_related_by_text(
        title=title,
        body=body,
        repo_name=actual_repo,
        top_k=top_k,
        min_confidence=min_confidence,
        state_filter=actual_state,
    )


def format_badge(label: str) -> str:
    """Format label badge with color-coded markdown styling."""
    color_map = {
        "bug": "#d73a49",      # GitHub red
        "feature": "#0366d6",  # GitHub blue
        "other": "#6a737d",    # GitHub gray
        "HIGH": "#28a745",     # Green
        "MEDIUM": "#d9822b",   # Amber
        "LOW": "#6c757d",      # Gray
    }
    color = color_map.get(label.lower(), color_map.get(label.upper(), "#0366d6"))
    return (
        f'<span style="background-color: {color}; color: white; padding: 4px 10px; '
        f'border-radius: 12px; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.5px;">'
        f'{label.upper()}</span>'
    )


# ---------------------------------------------------------------------------
# Streamlit Application Presentation Layer
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render top application header and academic project metadata."""
    st.markdown(
        """
        <div style="padding: 1.2rem 1.5rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #334155; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; color: #f8fafc; font-weight: 800;">
                        OpenSourceLens <span style="font-size: 1.2rem; color: #38bdf8; font-weight: 500;">| Milestone Demo</span>
                    </h1>
                    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                        Intelligent GitHub Issue Analytics, Dual-Model Classification & Semantic Retrieval Platform
                    </p>
                </div>
                <div style="text-align: right; margin-top: 8px;">
                    <span style="background: #0284c7; color: white; padding: 5px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: bold; margin-right: 6px;">
                        Phase 0–7 Complete (75% Milestone)
                    </span>
                    <span style="background: #334155; color: #cbd5e1; padding: 5px 12px; border-radius: 6px; font-size: 0.8rem;">
                        504 Tests Passing
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    distilbert_loaded: bool,
    classical_loaded: bool,
    retrieval_loaded: bool,
    corpus_size: int,
) -> Tuple[str, str, str]:
    """Render sidebar with model statuses, preset loader, and global configurations."""
    st.sidebar.markdown("### 🎛️ System Status & Models")

    def status_indicator(loaded: bool, name: str, details: str):
        color = "#10b981" if loaded else "#ef4444"
        symbol = "●" if loaded else "○"
        st.sidebar.markdown(
            f"""
            <div style="padding: 8px 12px; border-radius: 6px; background-color: #1e293b; margin-bottom: 8px; border-left: 4px solid {color};">
                <div style="font-weight: 600; font-size: 0.88rem; color: #f1f5f9;">
                    <span style="color: {color}; margin-right: 6px;">{symbol}</span>{name}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">{details}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    status_indicator(
        distilbert_loaded,
        "DistilBERT Fine-Tuned",
        "Phase 6 Deep Learning • 66M Params • MaxLen 256",
    )
    status_indicator(
        classical_loaded,
        "Calibrated LinearSVC",
        "Phase 5 Champion • Platt Scaled • 5,000 Vocab",
    )
    status_indicator(
        retrieval_loaded,
        "Sentence-BERT Vector Store",
        f"Phase 7 Semantic Index • {corpus_size:,} Issues • 384-dim",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Interactive Presets")
    st.sidebar.caption("Select a preconfigured academic scenario to test the models instantly:")

    preset_keys = list(PRESETS.keys())
    selected_preset = st.sidebar.selectbox("Choose Scenario:", preset_keys, index=0)

    preset_data = PRESETS[selected_preset]
    st.sidebar.info(preset_data["description"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Target Repositories")
    st.sidebar.markdown(
        """
        - `microsoft/vscode`
        - `pytorch/pytorch`
        - `kubernetes/kubernetes`
        - `huggingface/transformers`
        """
    )
    st.sidebar.caption("OpenSourceLens Milestone Demo • Built for Academic Evaluation")

    return selected_preset, preset_data["title"], preset_data["body"]


def render_tab_classification(
    transformer_model: Optional[TransformerIssueClassifier],
    classical_model: Optional[FinalClassicalNLPPipeline],
    default_title: str,
    default_body: str,
) -> None:
    """Render Tab 1: Live Dual-Model Issue Classification."""
    st.markdown("### 🎯 Live Issue Classification (Head-to-Head Comparison)")
    st.caption(
        "Enter any GitHub issue title and description to run live classification across our fine-tuned "
        "transformer and calibrated classical baseline."
    )

    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        issue_title = st.text_input(
            "Issue Title:",
            value=default_title,
            placeholder="e.g., Memory leak in renderer process when opening large workspace",
            key="cls_title",
        )
    with col_in2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        classify_button = st.button("⚡ Run Live Classification", type="primary", use_container_width=True)

    issue_body = st.text_area(
        "Issue Description / Stack Trace (Optional):",
        value=default_body,
        placeholder="Paste full error logs, reproduction steps, or feature requirements...",
        height=140,
        key="cls_body",
    )

    if classify_button or (default_title and len(default_title) > 5):
        if not issue_title.strip():
            st.warning("Please provide at least an issue title to run classification.")
            return

        col_left, col_right = st.columns(2)

        # ---------------- DistilBERT Inference ----------------
        with col_left:
            st.markdown("#### 🤖 Phase 6: Fine-Tuned DistilBERT")
            if transformer_model is None:
                st.error("Transformer model could not be loaded from disk.")
            else:
                t0 = time.perf_counter()
                trans_res = classify_with_transformer(transformer_model, issue_title, issue_body)
                trans_lat = (time.perf_counter() - t0) * 1000.0

                label = trans_res["label"]
                conf = trans_res["confidence"]
                is_uncertain = trans_res.get("is_uncertain", False)
                probs = trans_res["probabilities"]

                st.markdown(
                    f"""
                    <div style="padding: 1.1rem; border-radius: 8px; background-color: #1e293b; border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 1.1rem; font-weight: bold; color: #f8fafc;">Predicted Class:</span>
                            {format_badge(label)}
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #38bdf8; margin: 4px 0;">
                            {conf * 100:.1f}% <span style="font-size: 0.9rem; color: #94a3b8; font-weight: normal;">confidence</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8;">
                            Inference Latency: <strong>{trans_lat:.1f} ms</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_uncertain:
                    st.warning(
                        "⚠️ **Uncertainty Alert (< 55%):** Model confidence is below the automated triage threshold. "
                        "Recommendation: Routed to human triage queue."
                    )
                else:
                    st.success("✅ **High Confidence Decision:** Eligible for automated issue labeling.")

                st.markdown("**Class Probability Distribution:**")
                for c_name, c_prob in probs.items():
                    col_p1, col_p2 = st.columns([1, 4])
                    col_p1.markdown(f"`{c_name.upper()}`")
                    col_p2.progress(float(c_prob), text=f"{c_prob * 100:.1f}%")

        # ---------------- Classical LinearSVC Inference ----------------
        with col_right:
            st.markdown("#### ⚡ Phase 5: Calibrated LinearSVC Champion")
            if classical_model is None:
                st.error("Classical pipeline could not be loaded from disk.")
            else:
                t0 = time.perf_counter()
                class_res = classify_with_classical(classical_model, issue_title, issue_body)
                class_lat = (time.perf_counter() - t0) * 1000.0

                c_label = class_res["predicted_class"]
                c_conf = class_res["confidence"]
                c_probs = class_res["class_probabilities"]
                top_terms = class_res.get("top_contributing_features", [])

                st.markdown(
                    f"""
                    <div style="padding: 1.1rem; border-radius: 8px; background-color: #1e293b; border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 1.1rem; font-weight: bold; color: #f8fafc;">Predicted Class:</span>
                            {format_badge(c_label)}
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #a78bfa; margin: 4px 0;">
                            {c_conf * 100:.1f}% <span style="font-size: 0.9rem; color: #94a3b8; font-weight: normal;">calibrated probability</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #94a3b8;">
                            Inference Latency: <strong>{class_lat:.2f} ms</strong> (sub-millisecond CPU execution)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Class Probability Distribution:**")
                for c_name, c_prob in c_probs.items():
                    col_p1, col_p2 = st.columns([1, 4])
                    col_p1.markdown(f"`{c_name.upper()}`")
                    col_p2.progress(float(c_prob), text=f"{c_prob * 100:.1f}%")

                st.markdown("**🔍 Local Feature Importance (Explainability):**")
                if top_terms:
                    terms_df = pd.DataFrame(top_terms)[["feature", "contribution_score", "tfidf_value"]]
                    terms_df.columns = ["Token / N-gram", "Class Contribution", "TF-IDF Weight"]
                    st.dataframe(terms_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No significant contributing n-grams detected in vocabulary.")


def render_tab_retrieval(
    retrieval_model: Optional[RelatedIssuePipeline],
    default_title: str,
    default_body: str,
) -> None:
    """Render Tab 2: Semantic Related-Issue Retrieval & Duplicate Detection."""
    st.markdown("### 🔍 Semantic Related-Issue Finder & Duplicate Detection")
    st.caption(
        "Leverages dense sentence embeddings (`all-MiniLM-L6-v2`) to retrieve historical issues with shared "
        "architectural concepts, error signatures, or duplicate reporting."
    )

    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        query_title = st.text_input(
            "Query Issue Title:",
            value=default_title,
            placeholder="Type or paste issue title to search across corpus...",
            key="ret_title",
        )
        query_body = st.text_area(
            "Query Body / Stack Trace:",
            value=default_body,
            placeholder="Optional detailed description to enrich dense query embedding...",
            height=100,
            key="ret_body",
        )

    with col_q2:
        st.markdown("**Search Controls & Filters:**")
        repo_choice = st.selectbox(
            "Filter by Repository:",
            [
                "All Repositories",
                "microsoft/vscode",
                "pytorch/pytorch",
                "kubernetes/kubernetes",
                "huggingface/transformers",
            ],
            index=0,
        )
        top_k = st.slider("Top Candidates (K):", min_value=3, max_value=10, value=5)
        min_tier = st.selectbox(
            "Minimum Confidence Filter:",
            ["ALL", "LOW (>= 45%)", "MEDIUM (>= 55%)", "HIGH (>= 70%)"],
            index=0,
        )
        tier_clean = min_tier.split()[0]
        search_button = st.button("🔎 Search Semantic Index", type="primary", use_container_width=True)

    if search_button or (default_title and len(default_title) > 5):
        if not query_title.strip():
            st.warning("Please provide an issue title to perform retrieval.")
            return

        if retrieval_model is None:
            st.error("Related issue pipeline is not available.")
            return

        with st.spinner("Embedding query and computing cosine similarities across corpus..."):
            res = retrieve_related_issues(
                pipeline=retrieval_model,
                title=query_title,
                body=query_body,
                repo_filter=repo_choice,
                top_k=top_k,
                min_confidence=tier_clean,
            )

        summary = res["summary"]
        related = res["related_issues"]
        meta = res["execution_metadata"]

        has_duplicate = summary.get("has_potential_duplicate", False)
        max_score = summary.get("highest_similarity_score", 0.0)

        # ---------------- Duplicate Alert Banner ----------------
        if has_duplicate:
            st.markdown(
                f"""
                <div style="padding: 1.2rem; background-color: #451a1a; border: 2px solid #ef4444;
                            border-radius: 8px; margin: 1rem 0; color: #fecaca;">
                    <div style="font-size: 1.25rem; font-weight: 800; display: flex; align-items: center;">
                        <span style="font-size: 1.6rem; margin-right: 8px;">🚨</span> POTENTIAL DUPLICATE DETECTED ({max_score * 100:.1f}% SIMILARITY)
                    </div>
                    <div style="margin-top: 6px; font-size: 0.92rem; color: #fca5a5;">
                        {summary.get('primary_recommendation', 'An existing issue closely matches this report.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding: 0.8rem 1.2rem; background-color: #1e293b; border-left: 4px solid #38bdf8;
                            border-radius: 6px; margin: 0.8rem 0; color: #cbd5e1; font-size: 0.9rem;">
                    💡 <strong>Triage Recommendation:</strong> {summary.get('primary_recommendation', 'Unique report.')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metrics Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Retrieved Candidates", f"{len(related)} issues")
        col_m2.metric("Highest Similarity", f"{max_score * 100:.1f}%")
        col_m3.metric("Retrieval Latency", f"{meta.get('latency_ms', 0):.1f} ms")
        col_m4.metric("Indexed Corpus Size", f"{meta.get('corpus_size', 0):,} issues")

        st.markdown("---")
        st.markdown(f"#### Top {len(related)} Most Semantically Similar Historical Issues")

        if not related:
            st.info("No issues matched the selected confidence tier or repository filter.")
            return

        for idx, item in enumerate(related, start=1):
            score_pct = item["similarity_score"] * 100.0
            tier = item["confidence_tier"]
            repo = item["repo_name"]
            issue_num = item.get("issue_number", "?")
            html_url = item.get("html_url", "#")
            state = item.get("state", "open")
            state_color = "#22c55e" if state == "open" else "#a855f7"

            with st.container():
                st.markdown(
                    f"""
                    <div style="padding: 1rem; border-radius: 8px; background-color: #1e293b; margin-bottom: 0.8rem;
                                border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; max-width: 80%;">
                                <span style="color: #38bdf8; margin-right: 6px;">#{idx}</span>
                                {item.get('title', 'Untitled Issue')}
                            </div>
                            <div>
                                {format_badge(tier)}
                                <span style="font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin-left: 8px;">
                                    {score_pct:.1f}%
                                </span>
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.82rem; color: #94a3b8; display: flex; gap: 14px; flex-wrap: wrap; align-items: center;">
                            <span>📦 <strong>Repo:</strong> <code style="color: #cbd5e1;">{repo}</code></span>
                            <span>🔢 <strong>Issue:</strong> #{issue_num}</span>
                            <span>🏷️ <strong>Category:</strong> {format_badge(item.get('canonical_label', 'other'))}</span>
                            <span>Status: <strong style="color: {state_color};">{state.upper()}</strong></span>
                            <span>🔗 <a href="{html_url}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600;">Open on GitHub ↗</a></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                clean_snippet = item.get("clean_title", "")
                if clean_snippet:
                    with st.expander(f"View Normalized Issue Context (#{issue_num})"):
                        st.write(f"**Cleaned Title:** {item.get('clean_title', '')}")
                        st.write(f"**Action Recommendation:** {item.get('action_recommendation', '')}")


def render_tab_benchmarks() -> None:
    """Render Tab 3: Research Benchmarks & Academic Figures Gallery."""
    st.markdown("### 📊 Research Analytics & Academic Benchmark Gallery")
    st.caption(
        "Empirical evaluation results across Phases 0–7 generated for the conference paper and "
        "milestone evaluation defense."
    )

    # ---------------- Performance Comparison Table ----------------
    st.markdown("#### 🏆 Comprehensive Model Comparison (Test Split N=350)")
    benchmark_data = [
        {
            "Model / System": "Baseline Logistic Regression (Phase 4)",
            "Paradigm": "Classical ML (TF-IDF)",
            "Accuracy": "69.8%",
            "Macro F1": "0.694",
            "Inference Speed": "< 1 ms",
            "Hardware": "CPU",
        },
        {
            "Model / System": "Baseline LinearSVC (Phase 4)",
            "Paradigm": "Classical ML (TF-IDF)",
            "Accuracy": "70.8%",
            "Macro F1": "0.702",
            "Inference Speed": "< 1 ms",
            "Hardware": "CPU",
        },
        {
            "Model / System": "Enhanced LinearSVC + Platt Scaling (Phase 5 Champion)",
            "Paradigm": "NLP Feature Engineering",
            "Accuracy": "73.1%",
            "Macro F1": "0.729",
            "Inference Speed": "< 1 ms",
            "Hardware": "CPU",
        },
        {
            "Model / System": "Fine-Tuned DistilBERT (Phase 6 Champion)",
            "Paradigm": "Deep Transformer (66M)",
            "Accuracy": "76.5%",
            "Macro F1": "0.763",
            "Inference Speed": "~18 ms",
            "Hardware": "CPU / CUDA",
        },
        {
            "Model / System": "Sentence-BERT Semantic Index (Phase 7)",
            "Paradigm": "Dense Retrieval (MiniLM)",
            "Accuracy": "96.8% Top-5 Recall",
            "Macro F1": "0.842 Precision@1",
            "Inference Speed": "~14 ms",
            "Hardware": "CPU / CUDA",
        },
    ]
    st.dataframe(pd.DataFrame(benchmark_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Publication Figures Gallery")

    categories = list(FIGURE_GALLERY.keys())
    chosen_cat = st.radio("Select Research Category:", categories, horizontal=True)

    cat_figures = FIGURE_GALLERY[chosen_cat]
    for fig_info in cat_figures:
        fig_path = PROJECT_ROOT / fig_info["path"]
        st.markdown(f"##### {fig_info['title']}")
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
            st.caption(f"**Empirical Analysis:** {fig_info['caption']}")
        else:
            st.warning(f"Figure file not found at expected path: {fig_info['path']}")
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="OpenSourceLens | Milestone Demo",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load cached pipelines
    distilbert_clf = load_transformer_classifier()
    classical_pipe = load_classical_pipeline()
    retrieval_pipe = load_retrieval_pipeline()

    corpus_count = len(retrieval_pipe.store.embeddings) if (retrieval_pipe and retrieval_pipe.store.is_loaded) else 2328

    # Render header & sidebar
    render_header()
    _, preset_title, preset_body = render_sidebar(
        distilbert_loaded=(distilbert_clf is not None),
        classical_loaded=(classical_pipe is not None),
        retrieval_loaded=(retrieval_pipe is not None),
        corpus_size=corpus_count,
    )

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Tab 1: Live Dual-Model Classification",
        "🔍 Tab 2: Semantic Related Issues & Duplicates",
        "📊 Tab 3: Research Benchmarks & Figures",
    ])

    with tab1:
        render_tab_classification(
            transformer_model=distilbert_clf,
            classical_model=classical_pipe,
            default_title=preset_title,
            default_body=preset_body,
        )

    with tab2:
        render_tab_retrieval(
            retrieval_model=retrieval_pipe,
            default_title=preset_title,
            default_body=preset_body,
        )

    with tab3:
        render_tab_benchmarks()


if __name__ == "__main__":
    main()
