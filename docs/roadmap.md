# OpenSourceLens — Master Execution Roadmap

This roadmap defines the complete project execution plan from definition through final evaluation.

---

## Phase 0 — Project Definition & Planning ✅

**Goal:** Establish exactly what we're building and how, before writing project code.

| Step | Title | Status |
|---|---|---|
| 0.1 | Freeze the problem | ✅ `docs/0.1-project-definition.md` |
| 0.2 | Freeze the features | ✅ `docs/0.2-project-features.md` |
| 0.3 | Define the data we need | ✅ `docs/0.3-data-definition.md` |
| 0.4 | Decide candidate repositories | ✅ `docs/0.4-repository-selection.md` |
| 0.5 | Define the ML problems | ✅ `docs/0.5-ml-problem-definition.md` |
| 0.6 | Define the methodology | ✅ `docs/0.6-methodology.md` |
| 0.7 | Design system architecture | ✅ `docs/0.7-architecture.md` |
| 0.8 | Stage 1 material | ✅ `docs/0.8-stage-1-proposal.md` |

**Output:** A fully defined project with documentation, but no implementation yet.

---

## Phase 1 — Data Source & Collection Pipeline ✅

**Goal:** Build the pipeline: GitHub repository → GitHub API → raw dataset.

**Prerequisites:** Python basics, JSON, HTTP requests, REST API basics, GitHub API basics, API authentication, basic Pandas.

| Step | Title | Description | Status |
|---|---|---|---|
| 1.1 | GitHub API exploration | Understand repository, issues, labels, comments, events, pagination, rate limit endpoints | ✅ Complete (`ml/data_collection/explore_github_api.py`) |
| 1.2 | Build repository collector | Input: owner/repo → Output: repository metadata | ✅ Complete (`ml/data_collection/repository_collector.py`) |
| 1.3 | Build issue collector | Collect issue ID, title, body, state, timestamps, author, labels | ✅ Complete (`ml/data_collection/issue_collector.py`) |
| 1.4 | Build comment collector | Collect issue comments | ✅ Complete (`ml/data_collection/comment_collector.py`) |
| 1.5 | Build event collector | Collect useful issue events | ✅ Complete (`ml/data_collection/event_collector.py`) |
| 1.6 | Handle pagination | Retrieve thousands of records, not just the first page | ✅ Complete (`ml/data_collection/paginator.py`) |
| 1.7 | Handle rate limits/errors | Authentication, retry handling, rate-limit handling, API errors | ✅ Complete (`ml/data_collection/github_client.py`) |
| 1.8 | Save raw data | Create `data/raw/` structure for repositories, issues, comments, events | ✅ Complete (`ml/data_collection/storage.py`) |
| 1.9 | Run collection on selected repositories | Collect initial dataset from candidate repos | ✅ Complete (`ml/data_collection/pipeline.py`) |
| 1.10 | Data inventory | Generate report: repo stats, issue counts, labels, comments, time range, missing data | ✅ Complete (`ml/data_collection/inventory.py`) |

**Output:** A reproducible GitHub → raw-data pipeline.

---

## Phase 2 — Data Cleaning & Dataset Construction ✅

**Goal:** Turn messy GitHub data into a proper dataset for analysis and modeling.

**Prerequisites:** Pandas, data cleaning, missing values, duplicates, datetime manipulation, string processing.

| Step | Title | Status |
|---|---|---|
| 2.1 | Load raw data | ✅ Complete (`ml/preprocessing/loader.py`) |
| 2.2 | Inspect dataset | ✅ Complete (`ml/preprocessing/inspector.py`) |
| 2.3 | Clean issue text | ✅ Complete (`ml/preprocessing/text_cleaner.py`) |
| 2.4 | Clean timestamps | ✅ Complete (`ml/preprocessing/timestamp_cleaner.py`) |
| 2.5 | Handle missing values | ✅ Complete (`ml/preprocessing/missing_handler.py`) |
| 2.6 | Remove irrelevant data | ✅ Complete (`ml/preprocessing/filter.py`) |
| 2.7 | Normalize labels | ✅ Complete (`ml/preprocessing/label_normalizer.py`) |
| 2.8 | Construct issue dataset | ✅ Complete (`ml/preprocessing/dataset_builder.py`) |
| 2.9 | Construct derived features | ✅ Complete (`ml/preprocessing/features.py`) |
| 2.10 | Save processed dataset | ✅ Complete (`ml/preprocessing/storage_processed.py`) |

**Output:** A clean, documented OpenSourceLens dataset.

---

## Phase 3 — Exploratory Data Analysis ✅

**Goal:** Understand the data before building any model. This is the core Data Science phase.

**Prerequisites:** Descriptive statistics, Pandas, Matplotlib/Seaborn, distributions, correlation.

| Step | Title | Status |
|---|---|---|
| 3.1 | Dataset overview | ✅ Complete (`ml/eda/overview.py`) |
| 3.2 | Issue state analysis | ✅ Complete (`ml/eda/state_analysis.py`) |
| 3.3 | Label analysis | ✅ Complete (`ml/eda/label_analysis.py`) |
| 3.4 | Issue volume over time | ✅ Complete (`ml/eda/temporal_analysis.py`) |
| 3.5 | Repository comparison | ✅ Complete (`ml/eda/repo_comparison.py`) |
| 3.6 | Issue text analysis | ✅ Complete (`ml/eda/text_analysis.py`) |
| 3.7 | Comment analysis | ✅ Complete (`ml/eda/comment_analysis.py`) |
| 3.8 | Resolution-time analysis | ✅ Complete (`ml/eda/resolution_analysis.py`) |
| 3.9 | Correlation analysis | ✅ Complete (`ml/eda/correlation_analysis.py`) |
| 3.10 | Generate final visualizations | ✅ Complete (`ml/eda/visualizations.py`) |
| 3.11 | Write actual findings | ✅ Complete (`ml/eda/report.py`, `ml/eda/eda_findings.ipynb`, `ml/eda/FINDINGS.md`) |

**Output:** An EDA notebook/report with actual findings.

---

## Phase 4 — Machine Learning Baseline ✅

**Goal:** Build the first predictive model. No Transformers yet.

**Prerequisites:** Supervised learning, classification, features/labels, train/test split, Logistic Regression, SVM, evaluation metrics, overfitting, data leakage.

| Step | Title | Status |
|---|---|---|
| 4.1 | Define classification target | ✅ Complete (`ml/classification/target_definition.py`) |
| 4.2 | Define training/test strategy | ✅ Complete (`ml/classification/split.py`) |
| 4.3 | Build feature representation (TF-IDF) | ✅ Complete (`ml/classification/features.py`) |
| 4.4 | Train baseline model (Logistic Regression) | ✅ Complete (`ml/classification/logistic_regression.py`) |
| 4.5 | Train second baseline (SVM) | ✅ Complete (`ml/classification/svm.py`) |
| 4.6 | Evaluate | ✅ Complete (`ml/classification/evaluator.py`) |
| 4.7 | Error analysis | ✅ Complete (`ml/classification/error_analysis.py`) |

**Output:** A working classical ML issue classifier.

---

## Phase 5 — NLP Enhancement

**Goal:** Improve how we represent GitHub issue text.

**Prerequisites:** NLP basics, tokenization, TF-IDF, text preprocessing, embeddings, basic neural-network concepts.

| Step | Title | Status |
|---|---|---|
| 5.1 | Improve text preprocessing | ✅ Complete (`ml/nlp/enhanced_preprocessor.py`) |
| 5.2 | Experiment with title + body | ✅ Complete (`ml/nlp/title_body_experiment.py`) |
| 5.3 | Experiment with different text representations | ✅ Complete (`ml/nlp/representations.py`) |
| 5.4 | Analyze vocabulary/features | Pending |
| 5.5 | Establish final classical NLP pipeline | Pending |

**Output:** A strong classical NLP baseline.

---

## Phase 6 — Deep Learning & Transformer Model

**Goal:** Add a proper deep-learning/transformer component.

**Prerequisites:** Neural networks, tensors, training, loss, backpropagation, PyTorch basics, attention, Transformers, BERT, tokenizers, fine-tuning.

| Step | Title | Status |
|---|---|---|
| 6.1 | Choose transformer (DistilBERT/BERT) | Pending |
| 6.2 | Prepare transformer dataset | Pending |
| 6.3 | Tokenization | Pending |
| 6.4 | Fine-tune pretrained model | Pending |
| 6.5 | Evaluate | Pending |
| 6.6 | Compare with classical ML | Pending |
| 6.7 | Error analysis | Pending |
| 6.8 | Save trained model | Pending |

**Output:** A working transformer-based issue classifier and model comparison.

---

## Phase 7 — Related Issue Detection

**Goal:** Given a new issue, find potentially related historical issues.

**Prerequisites:** Embeddings, vector representations, Sentence Transformers, cosine similarity, similarity search.

| Step | Title | Status |
|---|---|---|
| 7.1 | Generate embeddings for issues | Pending |
| 7.2 | Store embeddings | Pending |
| 7.3 | Build similarity calculation | Pending |
| 7.4 | Retrieve top-K similar issues | Pending |
| 7.5 | Set/validate similarity threshold | Pending |
| 7.6 | Evaluate retrieval quality | Pending |
| 7.7 | Build related-issue output | Pending |

**Output:** Working semantic related-issue detection.

---

## Phase 8 — Resolution Analysis & Prediction

**Goal:** Understand and potentially predict issue resolution behavior.

**Prerequisites:** Regression/classification, feature engineering, correlation, data leakage, time-based data, model evaluation.

| Step | Title | Status |
|---|---|---|
| 8.1 | Define resolution time | Pending |
| 8.2 | Analyze resolution distribution | Pending |
| 8.3 | Identify useful features | Pending |
| 8.4 | Investigate resolution-time factors | Pending |
| 8.5 | Build baseline prediction model | Pending |
| 8.6 | Evaluate | Pending |
| 8.7 | Decide whether prediction is reliable | Pending |

**Output:** Resolution analytics, and potentially a resolution prediction model. Phase 8 prediction steps are **optional**.

---

## Phase 9 — Database & Backend

**Goal:** Build the backend that exposes data and models.

**Prerequisites:** Basic SQL, PostgreSQL, REST APIs, FastAPI, Pydantic, JSON, environment variables, basic backend architecture.

| Step | Title | Status |
|---|---|---|
| 9.1 | Design database schema | Pending |
| 9.2 | Create PostgreSQL database | Pending |
| 9.3 | Load processed data | Pending |
| 9.4 | Create FastAPI project | Pending |
| 9.5 | Repository endpoints | Pending |
| 9.6 | Analytics endpoints | Pending |
| 9.7 | Classification endpoint | Pending |
| 9.8 | Related-issue endpoint | Pending |
| 9.9 | Resolution endpoint | Pending |
| 9.10 | Integrate trained models | Pending |
| 9.11 | Error handling | Pending |
| 9.12 | API testing | Pending |

**Output:** A working ML/data backend.

---

## Phase 10 — Frontend / OpenSourceLens UI

**Goal:** Turn the backend into an actual usable application.

**Prerequisites:** React, Next.js, TypeScript, Tailwind, API requests, basic component architecture.

| Step | Title | Status |
|---|---|---|
| 10.1 | Design UI structure | Pending |
| 10.2 | Landing page | Pending |
| 10.3 | Repository input | Pending |
| 10.4 | Repository overview | Pending |
| 10.5 | Analytics dashboard | Pending |
| 10.6 | Issue analytics | Pending |
| 10.7 | AI issue analyzer | Pending |
| 10.8 | Related issues interface | Pending |
| 10.9 | Resolution analytics | Pending |
| 10.10 | Charts | Pending |
| 10.11 | Loading/error states | Pending |
| 10.12 | Responsive UI | Pending |

**Output:** The OpenSourceLens frontend.

---

## Phase 11 — Full Integration

**Goal:** Connect frontend ↔ API ↔ DB ↔ models.

| Step | Title | Status |
|---|---|---|
| 11.1 | Connect frontend/backend | Pending |
| 11.2 | Connect database | Pending |
| 11.3 | Connect classifier | Pending |
| 11.4 | Connect transformer | Pending |
| 11.5 | Connect similarity system | Pending |
| 11.6 | Connect resolution analysis | Pending |
| 11.7 | End-to-end testing | Pending |
| 11.8 | Handle edge cases | Pending |
| 11.9 | Performance cleanup | Pending |

**Output:** A complete local OpenSourceLens application.

---

## Phase 12 — Deployment

**Goal:** Get the complete application online.

**Prerequisites:** Environment variables, production builds, CORS, hosting basics, API/database/frontend deployment.

| Step | Title | Status |
|---|---|---|
| 12.1 | Deploy database | Pending |
| 12.2 | Deploy FastAPI backend | Pending |
| 12.3 | Deploy frontend | Pending |
| 12.4 | Configure environment variables | Pending |
| 12.5 | Configure CORS | Pending |
| 12.6 | Deploy model files/inference | Pending |
| 12.7 | Test production | Pending |
| 12.8 | Fix production issues | Pending |

**Output:** A publicly accessible OpenSourceLens application.

---

## Phase 13 — Evaluation & Final Results

**Goal:** Produce the results for academic evaluation.

| Step | Title | Status |
|---|---|---|
| 13.1 | Final dataset statistics | Pending |
| 13.2 | Final EDA | Pending |
| 13.3 | Classical ML results | Pending |
| 13.4 | Transformer results | Pending |
| 13.5 | Similarity results | Pending |
| 13.6 | Resolution analysis | Pending |
| 13.7 | Model comparison | Pending |
| 13.8 | Error analysis | Pending |
| 13.9 | Limitations | Pending |
| 13.10 | Final conclusions | Pending |

**Output:** Final experimental results.

---

## Phase 14 — Documentation & Conference Paper

**Goal:** Convert existing work into a conference paper.

| Step | Title | Status |
|---|---|---|
| 14.1 | Finalize title | Pending |
| 14.2 | Write abstract | Pending |
| 14.3 | Introduction | Pending |
| 14.4 | Related work | Pending |
| 14.5 | Dataset | Pending |
| 14.6 | Methodology | Pending |
| 14.7 | Model architecture | Pending |
| 14.8 | Experiments | Pending |
| 14.9 | Results | Pending |
| 14.10 | Discussion | Pending |
| 14.11 | Limitations | Pending |
| 14.12 | Conclusion | Pending |
| 14.13 | References | Pending |
| 14.14 | Formatting | Pending |

**Output:** Conference-paper version of the project.

---

## Phase 15 — Final Evaluation

**Goal:** Final presentation and demo preparation.

| Step | Title | Status |
|---|---|---|
| 15.1 | Stage 3 demo flow | Pending |
| 15.2 | Final PPT | Pending |
| 15.3 | Technical explanation | Pending |
| 15.4 | ML/DL viva preparation | Pending |
| 15.5 | Dataset questions | Pending |
| 15.6 | Architecture questions | Pending |
| 15.7 | Model comparison questions | Pending |
| 15.8 | Limitations/future scope | Pending |
| 15.9 | Live demo preparation | Pending |
| 15.10 | Final submission | Pending |
