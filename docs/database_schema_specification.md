# OpenSourceLens — Database Schema Specification

> [!NOTE]
> **Historical Planning Document (Pre-Phase B/C):** This early design spec omits implemented Phase B/C tables (`collection_runs`, `analysis_runs`, `analysis_run_attempts`, `pull_request_reviews`, `pull_request_review_comments`, `releases`, `issue_comments`, `contributor_activities`) and lists draft columns (`ingestion_status`, `repo_name` on commits/PRs, etc.) that were not added. For the actual 17-table SQLAlchemy schema and Alembic migrations, see [docs/database.md](./database.md).

## 1. Executive Summary & Design Principles

The OpenSourceLens relational database schema serves as the primary persistence and retrieval layer connecting offline data science workflows (Track A) and online dynamic repository intelligence (Track B) with the high-performance FastAPI backend and Next.js frontend.

### Core Design Principles
1. **Universal Multi-Entity Architecture:**
   - The schema models all critical open-source software engineering entities: **Repositories**, **Commits**, **Contributors**, **Pull Requests**, **Issues**, **Issue Labels**, and **Derived Analytics**.
   - Tables are fully generic and relational; no tables or views are partitioned or hardcoded to specific repository names.
2. **Dual-Driver Dialect Compatibility (PostgreSQL + SQLite):**
   - **PostgreSQL (Production):** The production target utilizes `BIGSERIAL`, `TIMESTAMP WITH TIME ZONE`, and native `JSON`/`JSONB` columns with connection pooling (`pool_size=10, max_overflow=20`).
   - **SQLite (Development / Testing / Offline):** The zero-configuration SQLite fallback allows instant unit testing and offline local execution without external server dependencies. SQLAlchemy 2.0 generic types (`BigInteger`, `DateTime(timezone=True)`, `JSON`) adapt transparently to both dialects.
3. **Optimized for High-Throughput REST API Access:**
   - Composite B-tree indexes are tailored to access patterns: `(repo_name, issue_number)`, `(repo_name, state)`, `(repo_name, canonical_label)`, `(repository_id, author_date)`, `(repository_id, merged_at)`, and `(source_issue_id, rank)`.
4. **Relational Integrity with Cascading Life-Cycles:**
   - Deleting or re-ingesting a repository cascades cleanly to child commits, contributors, pull requests, issues, and analytics.
   - Prediction audit logs are preserved (`ON DELETE SET NULL`) for model evaluation and latency tracking.

---

## 2. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    REPOSITORIES ||--o{ COMMITS : "has (1:N)"
    REPOSITORIES ||--o{ CONTRIBUTORS : "has (1:N)"
    REPOSITORIES ||--o{ PULL_REQUESTS : "contains (1:N)"
    REPOSITORIES ||--o{ ISSUES : "tracks (1:N)"
    REPOSITORIES ||--o{ REPOSITORY_ANALYTICS : "snapshots (1:N)"
    ISSUES ||--o{ ISSUE_LABELS : "tagged_with (1:N)"
    ISSUES ||--o{ PREDICTIONS : "evaluated_by (1:N)"
    ISSUES ||--o{ RELATED_ISSUES_CACHE : "source_issue (1:N)"
    ISSUES ||--o{ RELATED_ISSUES_CACHE : "target_issue (1:N)"

    REPOSITORIES {
        bigint id PK "GitHub Repo ID or Auto-increment"
        varchar repo_name UK "owner/repo slug"
        varchar owner "Repository owner"
        varchar name "Repository name"
        text description "Project description"
        varchar primary_language "Main programming language"
        integer stars_count "Stargazer count"
        integer forks_count "Fork count"
        integer open_issues_count "Open issue count"
        timestamp github_created_at "Creation timestamp on GitHub"
        timestamp github_updated_at "Last update timestamp on GitHub"
        timestamp collected_at "Data snapshot timestamp"
        varchar ingestion_status "pending, completed, failed"
        float ingestion_progress "Progress percentage (0-100)"
    }

    COMMITS {
        bigint id PK "Auto-increment ID"
        bigint repository_id FK "References repositories(id)"
        varchar repo_name "Denormalized repo_name"
        varchar sha UK "Commit Git SHA hash"
        varchar author_name "Commit author name"
        varchar author_email "Commit author email"
        varchar author_login "GitHub author username"
        timestamp author_date "Commit timestamp"
        text message "Commit message summary"
        integer additions "Lines added"
        integer deletions "Lines deleted"
        integer total_changes "Total line changes"
    }

    CONTRIBUTORS {
        bigint id PK "Auto-increment ID"
        bigint repository_id FK "References repositories(id)"
        varchar repo_name "Denormalized repo_name"
        varchar login "GitHub username"
        bigint github_id "GitHub User ID"
        varchar avatar_url "Avatar image URL"
        varchar html_url "GitHub profile URL"
        integer total_commits "Aggregate commit count"
        timestamp first_commit_date "First observed contribution"
        timestamp last_commit_date "Most recent contribution"
        boolean is_core "Flag: member of core contributor tier"
    }

    PULL_REQUESTS {
        bigint id PK "GitHub PR ID"
        bigint repository_id FK "References repositories(id)"
        varchar repo_name "Denormalized repo_name"
        integer pr_number "GitHub PR number"
        text title "PR title"
        varchar state "open, closed, or merged"
        varchar author "Author login"
        timestamp created_at "PR creation timestamp"
        timestamp updated_at "PR last updated timestamp"
        timestamp closed_at "PR closure timestamp"
        timestamp merged_at "PR merge timestamp (null if unmerged)"
        float time_to_merge_hours "Duration from creation to merge"
        integer comments_count "Discussion comment count"
        integer review_comments_count "Code review inline comments"
        integer additions "Lines added"
        integer deletions "Lines deleted"
        integer changed_files "Number of files changed"
    }

    ISSUES {
        bigint id PK "GitHub Issue ID"
        bigint repository_id FK "References repositories(id)"
        varchar repo_name "Denormalized repo_name"
        integer issue_number "GitHub issue number"
        text title "Original issue title"
        text body "Original issue description"
        text clean_title "Cleaned title"
        text clean_body "Cleaned body"
        text combined_text "Combined NLP text"
        integer clean_title_word_count "Word count in title"
        integer clean_body_word_count "Word count in body"
        varchar state "open or closed"
        varchar state_reason "completed, not_planned, etc."
        varchar author "Author login"
        varchar author_type "User or Bot"
        varchar author_association "MEMBER, CONTRIBUTOR, etc."
        boolean is_internal_author "Maintainer/collaborator flag"
        boolean is_bot_author "Bot flag"
        timestamp created_at "Issue creation timestamp"
        timestamp updated_at "Issue last update"
        timestamp closed_at "Closure timestamp"
        integer comments_count "Number of comments"
        integer reactions_count "Total reactions"
        integer labels_count "Label count"
        boolean has_code_block "Code snippet indicator"
        integer code_block_count "Code block count"
        boolean has_url "URL indicator"
        boolean has_stack_trace "Traceback indicator"
        varchar canonical_label "bug, feature, other"
        integer canonical_label_id "Numeric category ID (0-2)"
        boolean is_classified_category "Core class member"
        boolean is_unlabeled "Unlabeled issue flag"
        float resolution_time_hours "Continuous resolution duration"
        float resolution_time_days "Duration in days"
        varchar speed_tier "Advisory tier (<1d, 1-7d, 7-30d, 30+d)"
        varchar resolution_type "rapid, standard, prolonged, unresolved"
        boolean is_censored "Right-censored open issue indicator"
        boolean is_immediate_closure "Bot / duplicate immediate closure"
        float duration_at_risk_days "Observed or censored duration"
    }

    ISSUE_LABELS {
        bigint id PK "Auto-increment ID"
        bigint issue_id FK "References issues(id)"
        varchar label_name "Raw label name from GitHub"
        boolean is_canonical "True if mapped to standard taxonomy"
        varchar canonical_category "Mapped standard category"
    }

    REPOSITORY_ANALYTICS {
        bigint id PK "Auto-increment ID"
        bigint repository_id FK "References repositories(id)"
        varchar repo_name "Denormalized repo_name"
        integer total_issues "Total collected issues"
        integer open_issues "Open issues count"
        integer closed_issues "Closed issues count"
        float median_resolution_hours "Median resolution turnaround (P50)"
        float mean_resolution_hours "Mean resolution turnaround"
        float p90_resolution_hours "90th percentile resolution duration"
        float rapid_closure_rate "Proportion closed in < 24 hours"
        float survival_rate_30d "Kaplan-Meier 30-day survival probability"
        float gini_coefficient "Contributor commit concentration (0.0 - 1.0)"
        float shannon_entropy "Normalized contributor entropy"
        integer total_commits "Total commits analyzed"
        integer total_contributors "Total unique contributors"
        integer total_prs "Total pull requests analyzed"
        float median_pr_merge_hours "Median time to merge PRs"
        float pr_merge_rate "Proportion of PRs successfully merged"
        json monthly_activity "Monthly time-series commit and issue counts"
        json label_distribution "Distribution of canonical labels"
        json speed_tier_distribution "Distribution of resolution tiers"
        json findings_report "Structured synthesized diagnostic insights"
        timestamp generated_at "Analytics computation timestamp"
    }

    PREDICTIONS {
        bigint id PK "Auto-increment ID"
        bigint issue_id FK "Optional reference to issues(id)"
        varchar repo_name "Repository context"
        varchar model_name "distilbert, svm, logistic_regression"
        varchar model_version "Artifact identifier"
        text input_text "Submitted text sample"
        varchar predicted_label "Predicted canonical label"
        float confidence_score "Softmax confidence probability"
        json class_probabilities "Per-class probability distribution"
        float inference_latency_ms "Execution latency in milliseconds"
        timestamp created_at "Inference timestamp"
    }

    RELATED_ISSUES_CACHE {
        bigint id PK "Auto-increment ID"
        bigint source_issue_id FK "References issues(id)"
        bigint target_issue_id FK "References issues(id)"
        float similarity_score "Cosine similarity (0.0 - 1.0)"
        integer rank "Retrieval rank (1 to K)"
        timestamp created_at "Cache computation timestamp"
    }
```

---

## 3. Entity Definitions & Attribute Specifications

### 3.1 `repositories` Table
Stores high-level metadata and ingestion status for every analyzed repository.
- `id` (BigInteger, PK): GitHub numeric repository ID or internal sequential identifier.
- `repo_name` (String(255), Unique, Indexed): Canonical `{owner}/{name}` identifier.
- `ingestion_status` (String(32)): Workflow state (`pending`, `ingesting`, `completed`, `failed`).
- `ingestion_progress` (Float): Ingestion completion percentage (0.0 to 100.0).

### 3.2 `commits` Table
Stores commit activity records for development velocity, burstiness, and time-series trend analysis.
- `sha` (String(40), Unique, Indexed): 40-character Git commit hash.
- `repository_id` (BigInteger, FK): Foreign key referencing `repositories.id` (`ON DELETE CASCADE`).
- `author_date` (DateTime): Commit authoring timestamp for temporal bucketing.
- `additions` / `deletions` / `total_changes` (Integer): Lines added, deleted, and modified.

### 3.3 `contributors` Table
Maintains contributor profiles and aggregate contribution statistics for ecosystem modeling.
- `repository_id` (BigInteger, FK): Foreign key referencing `repositories.id` (`ON DELETE CASCADE`).
- `login` (String(255), Indexed): GitHub username.
- `total_commits` (Integer): Aggregate number of commits by this contributor.
- `first_commit_date` / `last_commit_date` (DateTime): Boundary timestamps for tenure and retention calculation.
- `is_core` (Boolean): Flag indicating membership in the core contributor tier (top 10% volume).

### 3.4 `pull_requests` Table
Records pull request lifecycles to model code-review turnaround, velocity, and backlog growth.
- `pr_number` (Integer, Indexed): GitHub repository-scoped pull request number.
- `state` (String(32)): `open`, `closed`, or `merged`.
- `merged_at` (DateTime, Nullable): Timestamp of merge closure (null if unmerged or rejected).
- `time_to_merge_hours` (Float, Nullable): Elapsed hours from `created_at` to `merged_at`.
- `comments_count` / `review_comments_count` (Integer): Quantity of discussion and inline code review.

### 3.5 `issues` Table
The central corpus table containing cleaned text, extracted engineering features, classification targets, and resolution metrics.
- `issue_number` (Integer, Indexed): GitHub repository-scoped issue number.
- `combined_text` (Text): Cleaned title concatenated with cleaned body for NLP models.
- `canonical_label` (String(32), Indexed): Normalized class (`bug`, `feature`, `other`).
- `resolution_time_hours` (Float, Nullable): Elapsed duration from `created_at` to `closed_at`.
- `speed_tier` (String(32)): Resolution velocity bracket (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`).

### 3.6 `repository_analytics` Table
Persists precalculated statistical snapshots, time-series aggregations, and synthesized findings for instant dashboard rendering.
- `monthly_activity` (JSON): Temporal array of `{month, commits, issues_created, issues_closed, prs_merged}`.
- `gini_coefficient` (Float): Contributor commit concentration metric (0.0 to 1.0).
- `findings_report` (JSON): Array of structured diagnostic observations with evidence tags.

---

## 4. Indexing Strategy & Performance Optimization

To guarantee sub-100ms API response times across tables containing tens of thousands of rows:
1. **Repository-Scoped Lookups:**
   - `idx_issues_repo_number`: `(repo_name, issue_number)` UNIQUE.
   - `idx_commits_repo_date`: `(repository_id, author_date)`.
   - `idx_prs_repo_merged`: `(repository_id, merged_at)`.
2. **Faceted Filtering Indexes:**
   - `idx_issues_repo_state_label`: `(repo_name, state, canonical_label)`.
   - `idx_issues_speed_tier`: `(repo_name, speed_tier)`.
3. **Similarity Search Ranking:**
   - `idx_related_source_rank`: `(source_issue_id, rank)`.
