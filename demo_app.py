"""
OpenSourceLens — Interactive Demonstration Platform (Phases 0–7)
Clean Two-Mode Application:
1. 🌐 Repository Analyzer (The Live Product):
   - Real-time GitHub repository ingestion (25-30 active issues).
   - Automated category breakdown (Bug, Feature, Other) via Fine-Tuned DistilBERT.
   - Intra-repository semantic duplicate detection via Sentence-BERT.
   - Interactive issue explorer with direct GitHub links.
2. 🔬 Model Lab & Benchmarks (Supporting Technical Proof):
   - Single-issue dual-model classification & explainability.
   - Historical semantic retrieval across the 2,328-issue training corpus.
   - Publication figures gallery & empirical benchmark tables.
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

# Core pipeline imports
from ml.deep_learning.pipeline import TransformerIssueClassifier
from ml.nlp.final_pipeline import FinalClassicalNLPPipeline
from ml.related_issues.embedder import IssueEmbedder
from ml.related_issues.pipeline import RelatedIssuePipeline
from ml.repository_analyzer import (
    analyze_repository,
    calculate_category_breakdown,
    classify_issues_batch,
    fetch_repo_metadata_and_issues,
    find_intra_repo_duplicates,
    normalize_repo_input,
)

# Pickle alias for classical pipeline
sys.modules["__main__"].FinalClassicalNLPPipeline = FinalClassicalNLPPipeline

# Quick Presets for Repository Analyzer
REPO_PRESETS = [
    {
        "name": "psf/requests",
        "label": "📦 psf/requests",
        "desc": "Standard Python HTTP library (>54k stars)",
    },
    {
        "name": "pallets/flask",
        "label": "📦 pallets/flask",
        "desc": "Lightweight WSGI web application framework (>67k stars)",
    },
    {
        "name": "scikit-learn/scikit-learn",
        "label": "📦 scikit-learn",
        "desc": "Industry-standard machine learning library (>60k stars)",
    },
]

# Single-issue testing presets for Model Lab
SINGLE_ISSUE_PRESETS = {
    "Select an issue scenario...": {"title": "", "body": ""},
    "🐛 Memory Leak Crash (VS Code)": {
        "title": "VS Code renderer crashes with JavaScript heap out of memory error",
        "body": "Opening a repository with >50k files causes the renderer process to hang and crash with 'Allocation failed - JavaScript heap out of memory'.",
    },
    "✨ Hardware Quantization (PyTorch)": {
        "title": "Add native GPU acceleration for 4-bit quantized matrix multiplication",
        "body": "Propose first-class torch.nn.functional.qlinear_int4 kernel utilizing Tensor Core hardware acceleration.",
    },
    "📖 Documentation Deadlink (Hugging Face)": {
        "title": "Fix broken quickstart documentation tutorial hyperlinks in quickstart.md",
        "body": "Quickstart tutorial page contains a 404 broken URL pointing to outdated example notebook on line 42.",
    },
}

# ---------------------------------------------------------------------------
# Figure Catalog for Academic Tab
# ---------------------------------------------------------------------------
FIGURE_GALLERY = {
    "🏆 Model Comparison & Tradeoffs": [
        {
            "title": "Figure 22: Comprehensive Model Comparison (Radar & Performance Bars)",
            "path": "ml/deep_learning/figures/fig22_model_comparison_radar_bars.png",
            "caption": "Multi-metric comparison across Phase 4 Baselines, Phase 5 Enhanced LinearSVC, and Phase 6 DistilBERT.",
        },
        {
            "title": "Figure 16: Phase 4 vs. Phase 5 Performance Gains",
            "path": "ml/nlp/figures/fig16_phase4_vs_phase5_performance.png",
            "caption": "Validation of domain NLP enhancements: 3x title weighting, code token handling, and sublinear TF-IDF.",
        },
    ],
    "🎯 Confusion Matrices & Calibration": [
        {
            "title": "Figure 20: Fine-Tuned DistilBERT Confusion Matrix",
            "path": "ml/deep_learning/figures/fig20_transformer_confusion_matrix.png",
            "caption": "Test split (N=350) confusion matrix showing 76.5% accuracy with balanced per-class separation.",
        },
        {
            "title": "Figure 21: DistilBERT ROC and Precision-Recall Curves",
            "path": "ml/deep_learning/figures/fig21_transformer_roc_pr_curves.png",
            "caption": "Multi-class ROC curves (Macro-AUC: 0.90) demonstrating sharp class boundary discrimination.",
        },
        {
            "title": "Figure 10: Baseline Confusion Matrices (Phase 4)",
            "path": "ml/classification/figures/fig10_baseline_confusion_matrices.png",
            "caption": "Phase 4 baseline Logistic Regression and LinearSVC confusion matrices on raw TF-IDF.",
        },
    ],
    "📉 Training Dynamics & Loss": [
        {
            "title": "Figure 19: DistilBERT Fine-Tuning Loss & Accuracy Progression",
            "path": "ml/deep_learning/figures/fig19_transformer_training_curves.png",
            "caption": "Training vs. validation loss over 4 epochs (lr=3e-5, AdamW, linear warm-up) showing stable convergence.",
        },
        {
            "title": "Figure 18: Token Length Distribution & Truncation Budget",
            "path": "ml/deep_learning/figures/fig18_token_length_distribution.png",
            "caption": "Sequence length distribution confirming max_length=256 captures >92% of issue content.",
        },
    ],
    "🔍 Semantic Retrieval Benchmarks": [
        {
            "title": "Figure 28: Retrieval Evaluation & Precision-Recall Benchmark",
            "path": "ml/related_issues/figures/fig28_retrieval_evaluation_benchmark.png",
            "caption": "Precision@K, Recall@K, and MRR@K across K=[1, 3, 5, 10] on the 2,328 issue vector store (96.8% Top-5 Recall).",
        },
        {
            "title": "Figure 27: Similarity Threshold Calibration & Confidence Tiers",
            "path": "ml/related_issues/figures/fig27_similarity_threshold_calibration.png",
            "caption": "Empirical calibration: HIGH (>=0.70), MEDIUM (0.55-0.70), LOW (0.45-0.55), Duplicate Alert (>=0.80).",
        },
        {
            "title": "Figure 26: Top-K Nearest Neighbor Distribution",
            "path": "ml/related_issues/figures/fig26_topk_retrieval_analysis.png",
            "caption": "Cosine similarity decay across top-10 neighbors illustrating clean discrimination from corpus noise.",
        },
    ],
    "📊 Dataset & Exploratory Analysis": [
        {
            "title": "Figure 9: Master Exploratory Data Analysis Dashboard",
            "path": "ml/eda/figures/fig9_master_eda_dashboard.png",
            "caption": "Overview of dataset distributions across target repositories.",
        },
        {
            "title": "Figure 1: Target Label Distribution & Class Proportions",
            "path": "ml/eda/figures/fig1_label_distribution.png",
            "caption": "Corpus class distribution: Bug (42.5%), Feature (34.1%), Other (23.4%).",
        },
    ],
}


# ---------------------------------------------------------------------------
# Cached Model Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Fine-Tuned DistilBERT Classifier (Phase 6)...")
def get_transformer_classifier() -> Optional[TransformerIssueClassifier]:
    """Load production TransformerIssueClassifier."""
    try:
        model_path = PROJECT_ROOT / "models" / "distilbert_finetuned"
        if not model_path.exists():
            return None
        return TransformerIssueClassifier(model_dir=model_path)
    except Exception as exc:
        st.error(f"Error loading DistilBERT model: {exc}")
        return None


@st.cache_resource(show_spinner="Loading Classical NLP Pipeline (Phase 5)...")
def get_classical_pipeline() -> Optional[FinalClassicalNLPPipeline]:
    """Load production FinalClassicalNLPPipeline."""
    try:
        model_path = PROJECT_ROOT / "models" / "final_classical_nlp_pipeline.joblib"
        if not model_path.exists():
            return None
        return FinalClassicalNLPPipeline.load(model_path=model_path)
    except Exception as exc:
        st.error(f"Error loading Classical pipeline: {exc}")
        return None


@st.cache_resource(show_spinner="Loading Sentence-BERT Embedder (Phase 7)...")
def get_sentence_embedder() -> Optional[IssueEmbedder]:
    """Load production IssueEmbedder."""
    try:
        return IssueEmbedder(model_name="all-MiniLM-L6-v2", normalize_embeddings=True)
    except Exception as exc:
        st.error(f"Error loading Sentence Embedder: {exc}")
        return None


@st.cache_resource(show_spinner="Loading Related Issues Vector Index (Phase 7)...")
def get_retrieval_pipeline() -> Optional[RelatedIssuePipeline]:
    """Load historical vector index pipeline."""
    try:
        store_path = PROJECT_ROOT / "models" / "related_issues"
        if not store_path.exists():
            return None
        pipe = RelatedIssuePipeline(store_dir=store_path)
        pipe._ensure_initialized()
        return pipe
    except Exception as exc:
        st.error(f"Error loading Related Issue pipeline: {exc}")
        return None


@st.cache_data(show_spinner="Ingesting repository data and running inference...", ttl=600)
def run_cached_repository_analysis(repo_name: str, target_count: int = 25) -> Dict[str, Any]:
    """Cached wrapper executing end-to-end repository analysis."""
    classifier = get_transformer_classifier()
    embedder = get_sentence_embedder()
    if classifier is None or embedder is None:
        raise RuntimeError("Required ML models could not be loaded.")
    return analyze_repository(
        repo_str=repo_name,
        classifier=classifier,
        embedder=embedder,
        target_count=target_count,
    )


# ---------------------------------------------------------------------------
# UI Helpers & Badges
# ---------------------------------------------------------------------------
def render_category_badge(category: str) -> str:
    """Format category with high-contrast color badge."""
    c = category.lower()
    if c == "bug":
        color = "#ef4444"
        icon = "🐛"
    elif c == "feature":
        color = "#3b82f6"
        icon = "✨"
    else:
        color = "#6b7280"
        icon = "📖"

    return (
        f'<span style="background-color: {color}; color: white; padding: 3px 10px; '
        f'border-radius: 12px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.3px;">'
        f'{icon} {category.upper()}</span>'
    )


# ---------------------------------------------------------------------------
# MODE 1: 🌐 REPOSITORY ANALYZER (THE PRODUCT)
# ---------------------------------------------------------------------------
def render_repository_analyzer_view() -> None:
    """Render the primary repository issue ecosystem analyzer view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #f8fafc;">
                🌐 GitHub Repository Issue Analyzer
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Ingest any public open-source repository to automatically classify issue types, 
                detect intra-repository duplicates, and map maintenance workload.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for input
    if "repo_input_val" not in st.session_state:
        st.session_state["repo_input_val"] = "psf/requests"

    # Input and Action Row
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        repo_input = st.text_input(
            "Enter Repository (Slug or GitHub URL):",
            value=st.session_state["repo_input_val"],
            placeholder="e.g. psf/requests, pallets/flask, or https://github.com/scikit-learn/scikit-learn",
            key="repo_text_field",
        )
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 Analyze Repository", type="primary", use_container_width=True)

    # Quick Preset Buttons
    col_p_label, col_p1, col_p2, col_p3 = st.columns([1.2, 1.5, 1.5, 1.5])
    col_p_label.markdown("<div style='padding-top: 6px; font-weight: 600; font-size: 0.85rem; color: #94a3b8;'>Quick Presets:</div>", unsafe_allow_html=True)
    
    if col_p1.button("📦 psf/requests", use_container_width=True):
        st.session_state["repo_input_val"] = "psf/requests"
        st.rerun()
    if col_p2.button("📦 pallets/flask", use_container_width=True):
        st.session_state["repo_input_val"] = "pallets/flask"
        st.rerun()
    if col_p3.button("📦 scikit-learn", use_container_width=True):
        st.session_state["repo_input_val"] = "scikit-learn/scikit-learn"
        st.rerun()

    active_repo = repo_input.strip() or st.session_state["repo_input_val"]

    # Trigger Analysis
    if analyze_clicked or active_repo:
        try:
            with st.spinner(f"Connecting to GitHub API and analyzing issues in '{active_repo}'..."):
                result = run_cached_repository_analysis(active_repo, target_count=25)
        except Exception as exc:
            st.error(f"Failed to analyze repository '{active_repo}': {exc}")
            st.info("Tip: Enter repository as 'owner/repo' (e.g. 'psf/requests') or full URL.")
            return

        repo_meta = result["repository"]
        issues = result["issues"]
        breakdown = result["breakdown"]
        duplicates = result["duplicates"]
        exec_meta = result["execution_metadata"]

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # ---------------- 1. Summary Cards ----------------
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric(
            "📦 Repository",
            repo_meta["full_name"],
            f"{repo_meta['language']}",
        )
        col_c2.metric(
            "⭐ GitHub Stars",
            f"{repo_meta['stars']:,}",
            f"{repo_meta['forks']:,} forks",
        )
        col_c3.metric(
            "📂 Open Issues",
            f"{repo_meta['open_issues_count']:,}",
            "on GitHub",
        )
        col_c4.metric(
            "⚡ Analyzed Issues",
            f"{exec_meta['analyzed_count']} issues",
            f"in {exec_meta['latency_ms']} ms",
        )

        st.markdown("---")

        # ---------------- 2. Automated Issue Breakdown ----------------
        st.markdown("#### 📊 Automated Issue Breakdown & Workload Distribution")
        pcts = breakdown["percentages"]
        counts = breakdown["counts"]
        total_iss = breakdown["total"]

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 8px; background: #1e293b; border-left: 5px solid #ef4444;">
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">BUG REPORTS</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ef4444;">{pcts['bug']}%</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">{counts['bug']} of {total_iss} issues classified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_b2.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 8px; background: #1e293b; border-left: 5px solid #3b82f6;">
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">FEATURE REQUESTS</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #3b82f6;">{pcts['feature']}%</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">{counts['feature']} of {total_iss} issues classified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_b3.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 8px; background: #1e293b; border-left: 5px solid #6b7280;">
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">DOCS / OTHER</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #9ca3af;">{pcts['other']}%</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">{counts['other']} of {total_iss} issues classified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # ---------------- 3. Duplicate Warning Alert ----------------
        st.markdown("#### 🔍 Intra-Repository Duplicate Detection")
        if duplicates["has_duplicates"]:
            dup_pairs = duplicates["duplicate_pairs"]
            top_dup = dup_pairs[0]
            iss1 = top_dup["issue_1"]
            iss2 = top_dup["issue_2"]
            sim_score = top_dup["similarity_pct"]

            st.markdown(
                f"""
                <div style="padding: 1.2rem; background-color: #451a1a; border: 2px solid #ef4444; border-radius: 8px; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; font-size: 1.15rem; font-weight: 800; color: #fca5a5;">
                        <span style="font-size: 1.5rem; margin-right: 8px;">🚨</span>
                        POTENTIAL INTRA-REPO DUPLICATE DETECTED ({sim_score}% SIMILARITY)
                    </div>
                    <div style="margin-top: 8px; font-size: 0.92rem; color: #fee2e2;">
                        <strong>Pair 1:</strong> Issue #{iss1.get('issue_number')} — <em>"{iss1.get('title')}"</em><br>
                        <strong>Pair 2:</strong> Issue #{iss2.get('issue_number')} — <em>"{iss2.get('title')}"</em>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.85rem; color: #fca5a5; background: #2f1212; padding: 6px 12px; border-radius: 6px;">
                        💡 <strong>Action Recommendation:</strong> Issues discuss the same component defect. Merge discussion into #{min(iss1.get('issue_number'), iss2.get('issue_number'))} before developer triage.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            highest = duplicates.get("highest_similarity_pair")
            top_msg = ""
            if highest:
                top_msg = (
                    f" (Closest semantic pair: #{highest['issue_1'].get('issue_number')} and "
                    f"#{highest['issue_2'].get('issue_number')} at {highest['similarity_pct']}%)"
                )
            st.markdown(
                f"""
                <div style="padding: 0.9rem 1.2rem; background: #132e22; border-left: 5px solid #10b981; border-radius: 6px; margin-bottom: 1rem;">
                    <span style="font-weight: 700; color: #6ee7b7;">✅ Zero High-Confidence Duplicates Detected:</span>
                    <span style="color: #d1fae5; font-size: 0.9rem;"> All analyzed issues describe distinct defects or feature requests.{top_msg}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------- 4. Issue Explorer Table ----------------
        st.markdown("#### 📋 Ingested Issue Explorer")

        # Category Filter
        filter_col, _ = st.columns([2, 3])
        with filter_col:
            cat_filter = st.selectbox(
                "Filter Issues by Predicted Category:",
                ["All Categories", "Bug", "Feature", "Other"],
                index=0,
            )

        filtered_issues = issues
        if cat_filter != "All Categories":
            filtered_issues = [i for i in issues if i.get("predicted_category", "").lower() == cat_filter.lower()]

        # Prepare Table Data
        table_rows = []
        for it in filtered_issues:
            table_rows.append({
                "Issue #": f"#{it.get('issue_number')}",
                "Title": it.get("title", ""),
                "Predicted Category": it.get("predicted_category", "").upper(),
                "Confidence": f"{it.get('confidence', 0.0) * 100:.1f}%",
                "State": it.get("state", "open").upper(),
                "GitHub Link": it.get("html_url", "#"),
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            column_config={
                "GitHub Link": st.column_config.LinkColumn("GitHub Link", display_text="Open on GitHub ↗"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Detailed item expander
        with st.expander("🔍 Inspect Full Issue Content & Model Probabilities"):
            selected_num = st.selectbox(
                "Select Issue # to Inspect:",
                [f"#{it.get('issue_number')} - {it.get('title')[:60]}..." for it in issues],
            )
            sel_idx = [f"#{it.get('issue_number')} - {it.get('title')[:60]}..." for it in issues].index(selected_num)
            sel_issue = issues[sel_idx]

            st.write(f"**Title:** {sel_issue.get('title')}")
            st.write(f"**Predicted Category:** {sel_issue.get('predicted_category', '').upper()} ({sel_issue.get('confidence', 0.0)*100:.1f}% confidence)")
            st.markdown(f"[🔗 View #{sel_issue.get('issue_number')} directly on GitHub]({sel_issue.get('html_url')})")
            st.text_area("Body / Description:", value=sel_issue.get("body", "No description"), height=120, disabled=True)


# ---------------------------------------------------------------------------
# MODE 2: 🔬 MODEL LAB & BENCHMARKS (SUPPORTING PROOF)
# ---------------------------------------------------------------------------
def render_model_lab_view() -> None:
    """Render the secondary technical evaluation and model laboratory view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #f8fafc;">
                🔬 Model Lab & Academic Evaluation
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Single-issue testing, local feature explainability, historical retrieval across the 2,328-issue training corpus, 
                and publication benchmark figures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lab_tab1, lab_tab2, lab_tab3 = st.tabs([
        "🎯 Single-Issue Dual Classifier",
        "🔍 Historical Semantic Retrieval",
        "📊 Benchmark Figures Gallery",
    ])

    # ---------------- Sub-Tab 1: Single Issue Classifier ----------------
    with lab_tab1:
        st.markdown("##### Test Any Custom Issue Text (DistilBERT vs. Calibrated LinearSVC)")
        
        # Scenario presets
        preset_names = list(SINGLE_ISSUE_PRESETS.keys())
        chosen_preset = st.selectbox("Choose a Test Scenario:", preset_names, index=0)
        p_data = SINGLE_ISSUE_PRESETS[chosen_preset]

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            inp_title = st.text_input("Issue Title:", value=p_data["title"], key="lab_title")
        with col_t2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            classify_btn = st.button("⚡ Run Dual Classification", type="primary", use_container_width=True)

        inp_body = st.text_area("Issue Body / Stack Trace:", value=p_data["body"], height=100, key="lab_body")

        if classify_btn or (inp_title and len(inp_title) > 5):
            distilbert = get_transformer_classifier()
            classical = get_classical_pipeline()

            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("###### 🤖 Phase 6: Fine-Tuned DistilBERT")
                if distilbert:
                    res = distilbert.predict_one(title=inp_title, body=inp_body)
                    st.markdown(
                        f"""
                        <div style="padding: 1rem; border-radius: 8px; background: #1e293b; border: 1px solid #334155;">
                            <div>Predicted: {render_category_badge(res['label'])}</div>
                            <div style="font-size: 1.8rem; font-weight: 800; color: #38bdf8; margin: 4px 0;">
                                {res['confidence']*100:.1f}% confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Class Probabilities:**")
                    for k, v in res["probabilities"].items():
                        c1, c2 = st.columns([1, 4])
                        c1.markdown(f"`{k.upper()}`")
                        c2.progress(float(v), text=f"{v*100:.1f}%")

            with c_right:
                st.markdown("###### ⚡ Phase 5: Calibrated LinearSVC Champion")
                if classical:
                    c_res = classical.explain_prediction(title=inp_title, body=inp_body, top_k=5)
                    st.markdown(
                        f"""
                        <div style="padding: 1rem; border-radius: 8px; background: #1e293b; border: 1px solid #334155;">
                            <div>Predicted: {render_category_badge(c_res['predicted_class'])}</div>
                            <div style="font-size: 1.8rem; font-weight: 800; color: #a78bfa; margin: 4px 0;">
                                {c_res['confidence']*100:.1f}% probability
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Top Contributing Keywords (Explainability):**")
                    if c_res.get("top_contributing_features"):
                        terms_df = pd.DataFrame(c_res["top_contributing_features"])[["feature", "contribution_score"]]
                        terms_df.columns = ["Token / N-gram", "Contribution"]
                        st.dataframe(terms_df, hide_index=True, use_container_width=True)

    # ---------------- Sub-Tab 2: Historical Semantic Retrieval ----------------
    with lab_tab2:
        st.markdown("##### Semantic Retrieval across the 2,328-Issue Curated Corpus")
        ret_pipe = get_retrieval_pipeline()
        
        q_title = st.text_input("Query Title:", value="CUDA out of memory error during backward pass", key="ret_pipe_title")
        q_repo = st.selectbox("Constrain to Repository:", ["All Repositories", "microsoft/vscode", "pytorch/pytorch", "kubernetes/kubernetes", "huggingface/transformers"])
        q_k = st.slider("Top Candidates (K):", 3, 10, 5)

        if st.button("🔎 Search Training Vector Index", type="primary"):
            if ret_pipe:
                repo_filt = None if q_repo == "All Repositories" else q_repo
                ret_res = ret_pipe.find_related_by_text(title=q_title, top_k=q_k, repo_name=repo_filt)
                for item in ret_res["related_issues"]:
                    st.markdown(
                        f"""
                        <div style="padding: 0.8rem; background: #1e293b; border-radius: 6px; margin-bottom: 6px; border: 1px solid #334155;">
                            <strong>{item.get('title')}</strong> — <span style="color: #38bdf8; font-weight: bold;">{item.get('similarity_score')*100:.1f}% similarity</span><br>
                            <span style="font-size: 0.8rem; color: #94a3b8;">Repo: <code>{item.get('repo_name')}</code> | #{item.get('issue_number')} | {item.get('confidence_tier')} confidence</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ---------------- Sub-Tab 3: Academic Figures Gallery ----------------
    with lab_tab3:
        st.markdown("##### Empirical Model Comparison (Test Split N=350)")
        bench_df = pd.DataFrame([
            {"Model": "Baseline Logistic Regression (Phase 4)", "Paradigm": "TF-IDF + LR", "Accuracy": "69.8%", "Macro F1": "0.694", "Latency": "< 1 ms"},
            {"Model": "Baseline LinearSVC (Phase 4)", "Paradigm": "TF-IDF + SVM", "Accuracy": "70.8%", "Macro F1": "0.702", "Latency": "< 1 ms"},
            {"Model": "Calibrated LinearSVC (Phase 5 Champion)", "Paradigm": "Domain NLP + Platt Scaling", "Accuracy": "73.1%", "Macro F1": "0.729", "Latency": "< 1 ms"},
            {"Model": "Fine-Tuned DistilBERT (Phase 6 Champion)", "Paradigm": "Deep Transformer (66M)", "Accuracy": "76.5%", "Macro F1": "0.763", "Latency": "~18 ms"},
            {"Model": "Sentence-BERT Index (Phase 7)", "Paradigm": "Dense Retrieval (MiniLM)", "Accuracy": "96.8% Top-5 Recall", "Macro F1": "0.842 P@1", "Latency": "~14 ms"},
        ])
        st.dataframe(bench_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Publication Evaluation Figures")
        chosen_cat = st.radio("Figure Category:", list(FIGURE_GALLERY.keys()), horizontal=True)
        for fig in FIGURE_GALLERY[chosen_cat]:
            f_path = PROJECT_ROOT / fig["path"]
            st.markdown(f"**{fig['title']}**")
            if f_path.exists():
                st.image(str(f_path), use_container_width=True)
                st.caption(fig["caption"])


# ---------------------------------------------------------------------------
# Main App Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit entrypoint with top mode selector."""
    st.set_page_config(
        page_title="OpenSourceLens | Repository Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Top App Header
    st.markdown(
        """
        <div style="padding: 1rem 1.4rem; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    border-radius: 8px; margin-bottom: 1.2rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 style="margin: 0; font-size: 1.9rem; color: #f8fafc; font-weight: 800;">
                        OpenSourceLens <span style="font-size: 1.1rem; color: #38bdf8; font-weight: 500;">| Intelligent Issue Triage</span>
                    </h1>
                </div>
                <div>
                    <span style="background: #0284c7; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: bold; margin-right: 6px;">
                        Phase 0–7 Complete
                    </span>
                    <span style="background: #334155; color: #cbd5e1; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem;">
                        515 Tests Passing
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Mode Selector
    mode = st.radio(
        "Navigation Mode:",
        ["🌐 Repository Analyzer (The Product)", "🔬 Model Lab & Benchmarks (Supporting Proof)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    if mode.startswith("🌐"):
        render_repository_analyzer_view()
    else:
        render_model_lab_view()


if __name__ == "__main__":
    main()
