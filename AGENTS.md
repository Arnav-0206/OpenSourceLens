# OpenSourceLens — Agent Instructions

## Project Identity

**OpenSourceLens** is an intelligent analytics platform that collects GitHub repository data and applies data science, classical ML, transformer-based NLP, and semantic similarity to analyze open-source issues. The dashboard is the presentation layer; the core project is the data science workflow.

## Source-of-Truth Hierarchy

When interpreting project decisions, use this priority order:

1. **Explicit user instructions** in the current interaction
2. **Canonical documentation** in `docs/` (including this roadmap)
3. **Existing implementation**, configuration, tests, and repository conventions
4. **Root-level contextual documents** in the parent `DS-PROJECT/` directory (Word documents from planning sessions)
5. **General technical knowledge** and recommendations

If sources disagree, identify the conflict, explain why it matters, and ask the user to resolve it before implementing.

## Development Protocol

### Phase-by-Phase Execution

- The project follows a 16-phase master roadmap (Phase 0–15). See `docs/roadmap.md`.
- Implement only the step explicitly requested by the user.
- Do not implement an entire phase when only one step was requested.
- Do not jump ahead to later phases.

### Plan Before Every Change

Before modifying any project file:

1. Inspect relevant documentation and current code.
2. Produce a downloadable implementation-plan artifact.
3. **Wait for user approval** before executing.

The plan must include: step identification, existing-state inspection, requirements, proposed implementation, risks, verification plan, completion criteria, and user decisions.

### After Implementation

Provide a final report covering: step number, status, files changed, summary, verification results, limitations, user actions needed, and next step.

## Anti-Hallucination Rules

1. Never claim to have read, run, tested, or deployed something unless you actually did.
2. Never invent statistics, metrics, API responses, or file paths.
3. Inspect existing code before modifying it.
4. Verify technical claims against documentation when the claim matters to implementation.
5. Distinguish facts from assumptions, estimates, and recommendations.
6. If a dependency, credential, or dataset is missing, report it.
7. Never fabricate placeholder metrics as real results.
8. Do not silently change project requirements.
9. If you cannot verify something, say so.
10. When implementation fails, investigate the actual error before applying fixes.

## Architecture Conventions

### Proposed Directory Structure

```text
OpenSourceLens/
├── frontend/          # Next.js + TypeScript
├── backend/           # FastAPI + Python
│   └── app/
│       ├── main.py
│       ├── api/
│       ├── services/
│       ├── database/
│       └── schemas/
├── data/
│   ├── raw/           # Original GitHub API responses (git-ignored)
│   └── processed/     # Cleaned datasets (git-ignored)
├── ml/
│   ├── data_collection/
│   ├── preprocessing/
│   ├── eda/
│   ├── classification/
│   ├── related_issues/
│   └── resolution_analysis/
├── models/            # Saved model artifacts (git-ignored)
├── docs/              # Project documentation
├── .gitignore
├── .env.example
├── AGENTS.md
└── README.md
```

### Technology Stack

- **Python** — Primary language for data collection, processing, ML/NLP
- **Pandas** — Data manipulation and analysis
- **scikit-learn** — TF-IDF, Logistic Regression, SVM
- **PyTorch** — Deep learning training and inference
- **Hugging Face Transformers** — DistilBERT/BERT fine-tuning
- **Sentence Transformers** — Sentence embeddings for related-issue detection
- **FastAPI** — Backend API
- **PostgreSQL** — Structured data storage
- **Next.js + TypeScript** — Frontend application
- **GitHub REST API** — Data source

### Separation of Concerns

- Data collection, preprocessing, and model training are **offline** processes.
- The FastAPI backend loads pre-trained model artifacts for **online** inference.
- Raw data and processed data are kept **separate**.
- Model weights are **not** stored in version control.

## Data Science Integrity

- Preserve raw data separately from processed datasets.
- Document all cleaning and label-normalization decisions.
- Validate label availability and quality before training.
- Establish a classical ML baseline before comparing with transformer models.
- Use proper train/validation/test splits.
- Prevent data leakage — labels used as targets must not be model input features.
- Do not fabricate training results or expected accuracy.
- Evaluate class imbalance and inspect model errors.
- Treat semantic similarity as retrieval of *potentially* related issues, not guaranteed duplicate detection.
- Resolution-time prediction is optional until data feasibility is established.

## Security Rules

- Store credentials and API tokens in `.env` (git-ignored).
- Never commit real secrets to Git.
- Use `.env.example` to document required variables with placeholder values.
- Validate user-supplied inputs in API endpoints.

## Target Repositories

1. `microsoft/vscode` — Developer tools
2. `pytorch/pytorch` — Machine learning framework
3. `kubernetes/kubernetes` — Cloud infrastructure
4. `huggingface/transformers` — NLP/AI library

## Core ML/NLP Tasks

| # | Task | Type | Priority |
|---|---|---|---|
| 1 | Issue Classification | Supervised text classification (TF-IDF + LR/SVM vs. fine-tuned transformer) | Core |
| 2 | Related-Issue Detection | Semantic retrieval (sentence embeddings + cosine similarity) | Core |
| 3 | Resolution Analysis | Descriptive analytics (resolution-time statistics) | Core |
| 4 | Resolution-Time Prediction | Regression | Optional |

## Roadmap Status

| Phase | Title | Status |
|---|---|---|
| 0 | Project Definition & Planning | ✅ Complete |
| 1 | Data Source & Collection Pipeline | 🔜 Next |
| 2 | Data Cleaning & Dataset Construction | Pending |
| 3 | Exploratory Data Analysis | Pending |
| 4 | Machine Learning Baseline | Pending |
| 5 | NLP Enhancement | Pending |
| 6 | Deep Learning & Transformer Model | Pending |
| 7 | Related Issue Detection | Pending |
| 8 | Resolution Analysis & Prediction | Pending |
| 9 | Database & Backend | Pending |
| 10 | Frontend / UI | Pending |
| 11 | Full Integration | Pending |
| 12 | Deployment | Pending |
| 13 | Evaluation & Final Results | Pending |
| 14 | Documentation & Conference Paper | Pending |
| 15 | Final Evaluation | Pending |
