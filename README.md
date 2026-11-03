# OpenSourceLens

**An Intelligent Analytics & Issue Intelligence Platform for GitHub Repositories using Machine Learning, Transformer NLP, and Next.js 15.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-teal.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## 🔍 Overview

**OpenSourceLens** is an end-to-end data science and full-stack analytics platform that analyzes open-source GitHub issues across four major software engineering ecosystems. It pairs rigorous offline data science (corpus curation, text cleaning, baseline ML, deep learning fine-tuning, dense semantic retrieval, and survival/resolution analytics) with an online high-performance **FastAPI** backend and an interactive **Next.js 15 (React 19)** web dashboard.

---

## 🚀 Roadmap & Implementation Status

| Phase | Title | Scope & Output | Status |
|---|---|---|---|
| **Phase 0** | Project Definition & Planning | Architecture, feature freeze, schema & problem definitions | ✅ Complete |
| **Phase 1** | Data Source & Collection Pipeline | GitHub REST API crawler, pagination, rate-limit recovery | ✅ Complete |
| **Phase 2** | Data Cleaning & Dataset Construction | Label normalization, HTML/markdown stripping, train/val/test splits | ✅ Complete |
| **Phase 3** | Exploratory Data Analysis | Vocabulary distributions, label balance, Kaplan-Meier curves | ✅ Complete |
| **Phase 4** | Machine Learning Baseline | TF-IDF + Logistic Regression & Linear SVM baselines (75–77% acc) | ✅ Complete |
| **Phase 5** | NLP Enhancement | Sublinear TF-IDF, n-gram ranges, regex noise filtering | ✅ Complete |
| **Phase 6** | Deep Learning & Transformer Model | Fine-tuned DistilBERT Champion model (**81.3% test accuracy**) | ✅ Complete |
| **Phase 7** | Related Issue Detection | Semantic similarity via 384-d `all-MiniLM-L6-v2` dense embeddings | ✅ Complete |
| **Phase 8** | Resolution Analysis & Prediction | Random Forest velocity prediction, SLA horizons, IQR windows | ✅ Complete |
| **Phase 9** | Database & Backend | PostgreSQL/SQLite schemas, FastAPI REST endpoints, 73 unit tests | ✅ Complete |
| **Phase 10** | Frontend / OpenSourceLens UI | Next.js 15 App Router, Tailwind, Recharts, AI workspace, Skeletons | ✅ Complete |
| **Phase 11** | Full Integration | End-to-end integration: Frontend ↔ FastAPI ↔ PostgreSQL ↔ PyTorch | 🔜 Next |
| **Phase 12** | Deployment | Docker containerization, staging, and production hosting | Pending |
| **Phase 13** | Evaluation & Final Results | Cross-system benchmarking and quantitative validation | Pending |
| **Phase 14** | Documentation & Conference Paper | Academic research paper and project presentation | Pending |
| **Phase 15** | Final Evaluation | Final project defense and delivery | Pending |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend Presentation** | **Next.js 15.5**, React 19, TypeScript 5.7, Tailwind CSS 3.4, Recharts 3.10, Lucide Icons |
| **Backend API** | **FastAPI**, Pydantic v2, Uvicorn, SQLAlchemy 2.1 |
| **Database** | **PostgreSQL** / SQLite with automated schema migrations and audit logs |
| **Classical Machine Learning** | **scikit-learn** (TF-IDF vectorizer, Logistic Regression, Linear SVM, Random Forest) |
| **Deep Learning & NLP** | **PyTorch**, Hugging Face **Transformers** (Fine-tuned `distilbert-base-uncased`) |
| **Semantic Vector Retrieval** | **Sentence Transformers** (`all-MiniLM-L6-v2` dense embedding space) |
| **Data Source** | GitHub REST API v3 (2,328 standardized curated issues) |

---

## 🎯 Target Repositories

The project benchmarks issues across four distinct open-source software domains:

1. **`microsoft/vscode`** (Developer Tools & IDE) — Rapid triage velocity (~7.7h median closure)
2. **`pytorch/pytorch`** (Machine Learning Framework) — High-volume compute & autograd issues (~14.2h median)
3. **`kubernetes/kubernetes`** (Cloud Native Orchestration) — Complex distributed infrastructure (~46.5h median)
4. **`huggingface/transformers`** (Generative AI & NLP) — Rapidly evolving state-of-the-art models (~28.0h median)

---

## ✨ Core Application Features

### 1. Multi-Repository Ecosystem Benchmark Matrix (`/analytics`)
- Side-by-side comparative matrix ranking repositories by stars, median turnaround time (P50), `< 24h` rapid closure rate, and 30-day Kaplan-Meier survival probability.
- Interactive Recharts grouped bar charts and doughnut distribution visualizers.

### 2. Single-Repository SLA & Volume Diagnostics (`/repos/[owner]/[name]`)
- Repository identity header with GitHub stars, forks, open issues, and language badges.
- Monthly time-series inflow vs. resolution area charts tracking net backlog changes.
- Taxonomy label and velocity speed-tier distribution cards.

### 3. Full-Text Issue Explorer (`/repos/[owner]/[name]/issues`)
- Real-time search with debounce over issue titles and descriptions.
- Multi-facet filtering by state (`open`/`closed`), canonical category (`bug`, `feature`, `other`), and velocity speed tier (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`).
- Interactive detail modal with normalized tags, author association tier, timeline metrics, and quick AI action shortcuts.

### 4. Interactive AI Issue Intelligence Workspace (`/analyzer`)
- **Taxonomy Classifier:** Test fine-tuned DistilBERT transformer alongside Linear SVM and Logistic Regression baselines with real-time class softmax probability progress bars.
- **Semantic Duplicate Retrieval:** Query 384-dimensional dense sentence embeddings to discover duplicates and similar bug reports with cosine similarity percentage ranks.
- **Resolution Velocity Predictor:** Forecast SLA resolution horizons (Optimistic Lower Bound vs. Conservative Upper Bound) and speed-tier likelihoods using ensemble regression.

---

## 💻 How to Run, Test, and Verify Yourself

You can run and test both the **FastAPI backend** and the **Next.js frontend** locally on your machine.

### Prerequisites
- **Node.js** (v18.0 or higher) and **npm**
- **Python** (3.10 to 3.14) with virtual environment support

---

### Step 1: Start the FastAPI Backend Server

1. Open a terminal in the project root:
   ```powershell
   # Activate Python virtual environment
   .\.venv\Scripts\Activate.ps1

   # Start the FastAPI application with auto-reload
   uvicorn backend.app.main:app --reload --port 8000
   ```
2. Verify the backend is online:
   - Root API status: `http://localhost:8000/`
   - Health check: `http://localhost:8000/health`
   - Interactive Swagger API Docs: `http://localhost:8000/docs`

---

### Step 2: Start the Next.js Frontend Development Server

1. Open a **second terminal** and navigate to the `frontend` directory:
   ```powershell
   cd frontend

   # Install dependencies (if not already installed)
   npm install

   # Start the Next.js development server
   npm run dev
   ```
2. Open your browser and navigate to:
   👉 **`http://localhost:3000`**

> **Note on Offline Resilience:** The frontend is engineered with graceful fallback datasets. If the backend server is stopped or restarted, all frontend views will seamlessly render verified empirical data without throwing broken UI errors.

---

### Step 3: Interactive Verification Walkthrough

Once both servers are running, test each key feature:

1. **Landing Page (`http://localhost:3000`):**
   - Check the **Live API Status Indicator** in the navbar (green badge shows live ping; amber shows offline simulation).
   - View global issue volume stats and quick-launch links.

2. **Repository Directory (`http://localhost:3000/repos`):**
   - Use the live search bar to filter repositories by name.
   - Click any repository card (e.g., `microsoft/vscode`) to navigate to its overview.

3. **Repository Overview (`http://localhost:3000/repos/microsoft/vscode`):**
   - Inspect the **SLA & Volume Diagnostics** grid.
   - Hover over the **Issue Inflow vs. Resolution Area Chart** to see monthly created vs closed issues and net backlog delta.
   - Hover over the **Taxonomy Composition Doughnut Chart** to inspect category shares.

4. **Issue Explorer (`http://localhost:3000/repos/microsoft/vscode/issues`):**
   - Filter issues by state, category (`bug`, `feature`, `other`), or speed tier.
   - Search for keywords (e.g., `"terminal"`, `"memory"`, `"crash"`).
   - Click any issue row to open the **Detail Modal**.
   - Click **"Analyze in AI Pipeline"** inside the modal to instantly transfer the issue into the AI Workspace!

5. **Cross-Repository Analytics (`http://localhost:3000/analytics`):**
   - Inspect the **Cross-Repository Benchmark Bar Chart** showing median resolution hours.
   - Click column headers in the **Comparative Matrix Table** (`Median SLA`, `Rapid Closures`, `30d Survival`) to test interactive sorting.

6. **AI Issue Workspace (`http://localhost:3000/analyzer`):**
   - **Tab 1 (Classifier):** Click `"Load Sample Bug"` or `"Load Sample Feature"`, switch between **DistilBERT Champion**, **Linear SVM**, and **Logistic Regression**, and click **"Classify Issue"** (or press `Ctrl+Enter`). Inspect the softmax probability breakdown.
   - **Tab 2 (Semantic Duplicates):** Click sample queries like `"CUDA out of memory during backward pass"` and adjust the similarity threshold slider ($0.20$ to $0.85$).
   - **Tab 3 (Resolution Velocity):** Test predicting resolution SLA tiers for member hotfixes vs. external feature RFCs.

---

### Step 4: Run Automated Verification Tests

You can run the full test and build suites anytime to verify code health:

#### Frontend Type-Check & Build:
```powershell
cd frontend
npm run type-check   # Verifies strict TypeScript contracts (0 errors)
npm run build        # Compiles Next.js production build and SSG pages (0 errors)
```

#### Backend API & Database Unit Tests:
```powershell
# From project root with .venv activated:
python -m pytest tests -k "api or database" -v
# Verified: 73 passed, 0 failed
```

---

## 📁 Repository Structure

```text
OpenSourceLens/
├── backend/                       # FastAPI REST Backend
│   └── app/
│       ├── main.py                # App entrypoint & CORS configuration
│       ├── api/v1/                # REST route endpoints (repos, analytics, classify, related, resolution)
│       ├── database/              # SQLAlchemy models, SQLite/PostgreSQL connectors, seeders
│       ├── schemas/               # Pydantic v2 validation contracts
│       └── services/              # ML inference services (DistilBERT, TF-IDF, SentenceTransformers)
├── frontend/                      # Next.js 15 + React 19 Frontend
│   ├── src/
│   │   ├── app/                   # Next.js App Router (pages, loading skeletons, error boundaries, 404)
│   │   │   ├── page.tsx           # Landing page
│   │   │   ├── repos/             # Repository directory & overview routes
│   │   │   ├── analytics/         # Cross-repository benchmark dashboard
│   │   │   └── analyzer/          # AI Intelligence Workspace (Classifier, Duplicates, Velocity)
│   │   ├── components/            # Reusable UI component library
│   │   │   ├── analytics/         # Comparative table, macro metric cards
│   │   │   ├── analyzer/          # Classification, retrieval, and SLA forms & cards
│   │   │   ├── charts/            # Recharts AreaChart, PieChart, BarChart
│   │   │   ├── common/            # Skeletons, ErrorStateCard, HealthIndicator
│   │   │   ├── issues/            # Issue filter bar, table, modal, pagination
│   │   │   ├── landing/           # Hero, ecosystem grid, features showcase
│   │   │   ├── layout/            # Navbar with mobile drawer, Footer
│   │   │   └── repository/        # RepoHeader, RepoSelector, metrics grids
│   │   └── lib/                   # API client, TypeScript types, formatting helpers
├── ml/                            # Offline Machine Learning & NLP Workflows
│   ├── data_collection/           # GitHub REST API crawling & pagination scripts
│   ├── preprocessing/             # Text cleaning, regex filtering, label normalization
│   ├── eda/                       # Exploratory data analysis notebooks & scripts
│   ├── classification/            # Classical ML (LR, SVM) and DistilBERT training scripts
│   ├── related_issues/            # Sentence Transformers index and cosine retrieval
│   └── resolution_analysis/       # Resolution duration stats, survival curves, regression
├── data/                          # Datasets (raw & processed)
├── models/                        # Saved model weights & tokenizers (git-ignored)
├── docs/                          # Architecture specs, methodology, master roadmap
├── tests/                         # Backend and ML unit/integration test suites (645 tests)
├── AGENTS.md                      # Agent rules, source of truth, and protocol conventions
├── README.md                      # Project documentation and quickstart guide
└── requirements.txt               # Python package dependencies
```

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
