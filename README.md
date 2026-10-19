# OpenSourceLens

**An intelligent analytics platform for GitHub repository issues using machine learning and natural language processing.**

---

## Overview

OpenSourceLens is a data-driven platform that collects and analyzes GitHub repository data, applying machine learning and NLP techniques to provide issue classification, related-issue detection, resolution analysis, and repository-level insights through an interactive web application.

## Core Features

| Feature | Description |
|---|---|
| **Repository Overview** | Repository statistics, issue counts, activity trends, and label distributions |
| **Issue Analytics** | Issue patterns, category distributions, trends over time, and cross-repository comparisons |
| **Issue Classification** | AI-based categorization of issues (Bug, Feature Request, Documentation, Question, Enhancement) comparing classical ML and transformer approaches |
| **Related-Issue Detection** | Semantic retrieval of potentially related issues using sentence embeddings and cosine similarity |
| **Resolution Analysis** | Resolution-time statistics, distributions, and comparisons across categories and repositories |

## Technology Stack

| Component | Technology |
|---|---|
| Data Processing | Python, Pandas |
| Classical ML | scikit-learn (TF-IDF, Logistic Regression, SVM) |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Semantic Retrieval | Sentence Transformers |
| Backend | FastAPI |
| Database | PostgreSQL |
| Frontend | Next.js, TypeScript |
| Data Source | GitHub REST API |

## Project Status

| Phase | Title | Status |
|---|---|---|
| Phase 0 | Project Definition & Planning | ✅ Complete |
| Phase 1 | Data Source & Collection Pipeline | 🔜 Next |
| Phase 2–15 | Remaining phases | Pending |

See [`docs/`](docs/) for complete project documentation.

## Target Repositories

The project analyzes issues from these open-source repositories:

1. `microsoft/vscode` — Developer tools
2. `pytorch/pytorch` — Machine learning framework
3. `kubernetes/kubernetes` — Cloud infrastructure
4. `huggingface/transformers` — NLP/AI library

## Setup

### Prerequisites

- Python 3.10+
- Git

### Environment Configuration

```bash
cp .env.example .env
# Edit .env and add your GitHub Personal Access Token
```

A GitHub Personal Access Token is required for API data collection. Create one at [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens).

## Documentation

All project documentation is in the [`docs/`](docs/) directory:

- [Project Definition](docs/0.1-project-definition.md)
- [Project Features](docs/0.2-project-features.md)
- [Data Definition](docs/0.3-data-definition.md)
- [Repository Selection](docs/0.4-repository-selection.md)
- [ML Problem Definition](docs/0.5-ml-problem-definition.md)
- [Methodology](docs/0.6-methodology.md)
- [Architecture](docs/0.7-architecture.md)
- [Stage 1 Proposal](docs/0.8-stage-1-proposal.md)
- [Master Roadmap](docs/roadmap.md)

## License

*To be determined.*
