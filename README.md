# OpenSourceLens

**A Data Science-Powered GitHub Repository Intelligence Platform using Machine Learning, Transformer NLP, and Next.js 15.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162-teal.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## 🔍 Overview

**OpenSourceLens** is an intelligent, data science-driven analytics platform designed to analyze open-source software repositories on GitHub. Moving beyond simple vanity metrics and raw counts, OpenSourceLens applies data engineering, time-series analysis, statistical modeling, machine learning, transformer-based NLP, and semantic retrieval to extract deep, evidence-backed intelligence regarding repository evolution, community sustainability, and maintenance dynamics.

The platform is designed to operate with **any eligible public GitHub repository** entered by an end-user. Users supply a repository URL, and the platform dynamically ingests, structures, and analyzes its commits, contributors, pull requests, issues, and releases within GitHub API rate-limit budgets.

To validate its analytical pipelines and train universal natural language processing models, OpenSourceLens utilizes an initial benchmark corpus composed of four major open-source ecosystems:
- **`microsoft/vscode`** (Developer Tools & GUI Application)
- **`pytorch/pytorch`** (Machine Learning Framework)
- **`kubernetes/kubernetes`** (Cloud-Native Infrastructure)
- **`huggingface/transformers`** (Deep Learning & NLP)

These four repositories serve as our benchmark laboratory and validation baseline; they do not define the functional boundaries of the end product.

---

## 💡 The Core Differentiator: Raw Data vs. Derived Intelligence

GitHub already provides raw lists and counters. OpenSourceLens applies Data Science to convert raw observations into actionable intelligence:

| Raw GitHub Metric | The OpenSourceLens Derived Intelligence |
|---|---|
| **Total Commits** | Development velocity time-series, 4/12-week rolling trends, burstiness indices, sprint surges, and activity freeze anomalies |
| **Contributor List** | Gini coefficient concentration, Lorenz curve, Shannon entropy, bus-factor vulnerability, core vs. peripheral community segmentation, newcomer retention half-life |
| **Pull Requests** | PR lifecycle turnaround, review response velocity, merge ratios, abandonment rates, and code-review bottleneck detection |
| **Issues & Labels** | Domain-agnostic taxonomy classification (`bug`, `feature`, `other`), 384-d semantic duplicate retrieval, backlog aging, and net inflow/outflow balance |
| **Repository Stats** | **Cross-domain synthesis:** Correlating contributor attrition with PR review slowdowns and rising bug volume to diagnose underlying project health |

---

## 🚀 Roadmap & Implementation Status

The project is structured into two core evolutionary tracks:
- **Phases 0–11: Benchmark Lab Foundation (Completed)** — Full offline ML lab, fine-tuned DistilBERT classifier (81.3% accuracy), Sentence-Transformers semantic retrieval, relational database, FastAPI backend, Next.js UI, and 145 integration tests.
- **Phases 11-R–16-R: Universal Repository Intelligence Platform (Realigned Execution Plan)** — Dynamic ingestion for any public repository, multi-domain intelligence engines, universal search workspace, and cross-repo validation.

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
| **Phase 8** | Resolution Analysis & Prediction | Empirical SLA horizons, turnaround statistics, Random Forest velocity | ✅ Complete |
| **Phase 9** | Database & Backend | PostgreSQL/SQLite schemas, FastAPI REST endpoints, 73 unit tests | ✅ Complete |
| **Phase 10** | Frontend / OpenSourceLens UI | Next.js 15 App Router, Tailwind, Recharts, AI workspace, Skeletons | ✅ Complete |
| **Phase 11** | Full Integration | End-to-end integration: Frontend ↔ FastAPI ↔ SQLite/Postgres ↔ ML Ensembles | ✅ Complete |
| **Phase 11-R** | Universal Ingestion Engine | Dynamic URL validator, multi-entity collectors, tiered rate budgeter | 🔜 Next |
| **Phase 12-R** | Multi-Domain Intelligence Engines | Time-series, Gini concentration, PR bottlenecks, dynamic embeddings | Pending |
| **Phase 13-R** | Dynamic Frontend Workspace | Universal URL search bar, 6-domain tabbed workspace, findings report | Pending |
| **Phase 14-R** | Generalization & Verification | Cross-repo validation on unseen repositories, rate-limit stress tests | Pending |
| **Phase 15-R** | Production Deployment | Docker containerization, staging, and production hosting | Pending |
| **Phase 16-R** | Research Paper & Final Defense | Academic research paper and project presentation | Pending |

---

## 🛠️ Technology Stack

| Layer | Technologies | Architectural Role |
|---|---|---|
| **Frontend Presentation** | **Next.js 15.5**, React 19, TypeScript 5.7, Tailwind CSS 3.4, Recharts 3.10, Lucide Icons | Responsive UI, dynamic URL routing, interactive time-series & Lorenz charts |
| **Backend API** | **FastAPI 0.162**, Pydantic v2, Uvicorn, SQLAlchemy 2.0 | High-concurrency asynchronous REST endpoints, request validation |
| **Database** | **PostgreSQL** (Production) / **SQLite (WAL mode)** (Local Dev & Testing) | Normalized storage of repositories, commits, contributors, PRs, issues, analytics |
| **Data Ingestion** | Python `httpx`, `asyncio`, GitHub REST API v3 | Dynamic bounded ingestion, rate-limit budgeting, pagination, backoff |
| **Classical Machine Learning** | **scikit-learn** (TF-IDF vectorizer, Logistic Regression, Linear SVM) | Baseline text classification, statistical feature modeling |
| **Deep Learning & NLP** | **PyTorch 2.14**, Hugging Face **Transformers** (`distilbert-base-uncased`, 66.9M params) | Universal sequence classification into `bug`, `feature`, and `other` |
| **Semantic Vector Retrieval** | **Sentence Transformers** (`all-MiniLM-L6-v2` 384-d dense embedding space) | On-demand issue embeddings, cosine similarity search |

---

## 💻 How to Run, Test, and Verify Yourself

You can run and test both the **FastAPI backend** and the **Next.js frontend** locally on your machine.

### Prerequisites
- **Python 3.10+** (Python 3.14 supported) with virtual environment
- **Node.js 18+** and **npm**

---

### Step 1: Run the Backend Server (FastAPI)

1. Open a terminal in the project root (`OpenSourceLens`).
2. Activate your virtual environment:
   - **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
   - **Linux/macOS:** `source .venv/bin/activate`
3. Launch the FastAPI server:
   ```bash
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```
   > [!NOTE]
   > Running `python -m uvicorn` ensures the server executes through the trusted Python binary, avoiding Windows Application Control (AppLocker) restrictions.
4. Verify the backend is live:
   - **Interactive API Docs (Swagger UI):** Open [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Check:** Open [http://localhost:8000/health](http://localhost:8000/health) (should return `{"status": "healthy"}`)

---

### Step 2: Run the Frontend Application (Next.js 15)

1. Open a **second terminal** in the project root:
   ```bash
   cd frontend
   npm run dev
   ```
2. Open your browser and navigate to:
   - **Landing Page:** [http://localhost:3000](http://localhost:3000)
   - **AI Issue Intelligence Workspace:** [http://localhost:3000/analyzer](http://localhost:3000/analyzer)
   - **Benchmark Repositories Matrix:** [http://localhost:3000/analytics](http://localhost:3000/analytics)

---

### Step 3: Run Backend Automated Verification Tests

To verify that all backend integration tests pass against the SQLite database and loaded machine learning models:
```bash
python -m pytest -k "test_api or test_integration" -v
```
*(All 145 integration tests pass in ~12 seconds).*

---

### Step 4: Run Diagnostic Verification Scripts

We provide 9 standalone test scripts in `scripts/` to verify each subsystem individually:
```bash
python scripts/verify_phase11_db.py              # Verify DB connectivity & schema
python scripts/verify_phase11_api.py             # Verify FastAPI REST endpoints
python scripts/verify_phase11_classifier.py      # Verify classical ML baseline inference
python scripts/verify_phase11_transformer.py     # Verify fine-tuned DistilBERT inference
python scripts/verify_phase11_similarity.py      # Verify Sentence Transformers semantic search
python scripts/verify_phase11_resolution.py      # Verify resolution analytics
python scripts/verify_phase11_e2e.py             # Verify full end-to-end user workflows
python scripts/verify_phase11_edge_cases.py      # Verify error handling & edge cases
python scripts/verify_phase11_performance.py     # Verify sub-100ms API response latency
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
