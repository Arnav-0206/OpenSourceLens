# OpenSourceLens — Dynamic GitHub Ingestion Pipeline

OpenSourceLens dynamically collects commits, contributors, pull requests, PR reviews, inline review comments, pure issues, conversation comments, releases (with git tags fallback), and GitHub Discussions from **any eligible public GitHub repository** via the GitHub REST API v3 and GraphQL API v4.

> **Operational Status (Stages 1 through 6 Completed):** PR depth fields (`body`, `head_branch`, `base_branch`, `is_draft`, `labels`, `requested_reviewers`), PR diff metrics (`additions`, `deletions`, `changed_files`), git tags fallback for unannounced releases (`/tags`), and **GitHub Discussions** GraphQL API v4 ingestion are fully operational and integrated with automatic post-ingestion analytics enqueueing.

---

## 1. Complete Ingestion-to-Analytics Lifecycle

```
GitHub URL / Slug
  └─▶ GitHubURLValidator (syntax + host + reserved-path checks)
        └─▶ RepositoryEligibilityChecker (GET /repos/{owner}/{repo}, public & non-empty gates)
              └─▶ Preview Response (metadata + budget estimate + confidence preview)
                    └─▶ Submit (POST /api/v1/ingest/submit -> DurableJobQueue.submit_job -> CollectionRun pending)
                          └─▶ IngestionWorker (embedded in lifespan or standalone CLI)
                                ├─▶ AsyncGitHubClient (concurrency=2, 100ms pacing, rate budgeting)
                                │     ├─▶ REST API v3: Commits, PRs, Issues, Releases, Tags
                                │     └─▶ GraphQL API v4: Discussions, Categories, Q&A answers
                                └─▶ RepositoryCollector (normalization & atomic mutation tracking)
                                      └─▶ Atomic analytics_source_revision increment (only if mutated)
                                            └─▶ DurableJobQueue.complete_job (status + coverage_summary)
                                                  └─▶ _enqueue_post_ingestion_analytics -> AnalyticsJobQueue
```

---

## 2. Stage-by-Stage Harvesting Behavior

1. **URL Validation ([`validator.py`](../backend/app/services/validator.py)):** Accepts full URLs (`https://github.com/owner/repo`, `.git` suffix stripped), scheme-less URLs (`github.com/owner/repo`), or canonical slugs (`owner/repo`). Rejects npm scopes (`@owner/repo`), non-GitHub hosts, sub-paths (`/issues`, `/pulls`), reserved owner names (`explore`, `marketplace`, `settings`), and invalid names.
2. **Eligibility & Preview ([`repository_eligibility.py`](../backend/app/services/repository_eligibility.py)):**
   - **Gates:** Repository must be **public** (`private == False`) and **non-empty** (`size > 0` or `commits_count >= 1`). Archived repositories and forks are accepted with informational status flags.
   - `POST /api/v1/ingest/preview` inspects only repository metadata to estimate request budgets and evaluate initial sample sufficiency without consuming quota on entity streams.
3. **Submission & Deduplication (`POST /api/v1/ingest/submit`):**
   - Upserts the `Repository` record and creates a `pending` `CollectionRun`.
   - If an active (`pending` or `running`) `CollectionRun` already exists for the repository, the submission returns the existing active job (`already_active` semantics) to prevent redundant upstream requests.
4. **Worker Claim & Heartbeat (`IngestionWorker`):**
   - Recovers expired jobs (`recover_expired_jobs()`) and re-opens resumed rate-limited jobs (`reopen_resumed_rate_limited_jobs()`).
   - Claims the next job (`lease_duration_seconds=600`, `lease_version += 1`) and maintains a 15-second async heartbeat (`_heartbeat_loop`).
5. **Multi-Entity Harvesting Depth (`RepositoryCollector`):**
   1. `collect_repository_metadata`: Core repository details, stargazers, forks, open issues count, and primary language.
   2. `collect_commits`: Harvests commits and queries `/commits/{sha}` diff statistics (`additions`, `deletions`, `total_changes`) within the allocated diff budget.
   3. `collect_contributors`: Maps contributor profiles anchored on `github_id` or unlinked `git_email`, separating bot identities.
   4. `collect_pull_requests`: Harvests PR streams, extracting PR depth attributes (`body`, `head_branch`, `base_branch`, `is_draft`, `labels`, `requested_reviewers`) and diff churn stats (`additions`, `deletions`, `changed_files`).
   5. `collect_pr_reviews`: Harvests formal review submissions (`APPROVED`, `CHANGES_REQUESTED`, `COMMENTED`, `DISMISSED`) on sampled PRs.
   6. `collect_pr_inline_comments`: Harvests inline diff comments via `/pulls/comments`.
   7. `collect_issues`: Harvests pure issues (filtering `"pull_request" in item`), applying text cleaning, feature extraction, and speed tiers.
   8. `collect_issue_comments`: Harvests conversation comments, enforcing mutual exclusivity to either an issue or a PR.
   9. `collect_releases`: Queries `/releases`; if zero releases are found, automatically falls back to `/tags` commits, tagging them with `is_tag_fallback=True`.
   10. `collect_discussions`: Executes GraphQL API v4 queries to harvest repository discussions, categories, Q&A status, marked answers (`answerChosenAt` / `answer { createdAt }`), and comment threads.
   11. `reconcile_contributor_metrics`: Computes participation counts, tenure, and core/newcomer cohorts.
6. **Atomic Mutation Tracking:** Populates `mutated_records`. If at least one row was inserted or modified, `repositories.analytics_source_revision` is atomically incremented (+1).
7. **Post-Ingestion Analytics Enqueue:** Upon clean completion (`completed` or non-errored `completed_partial`), `IngestionWorker._enqueue_post_ingestion_analytics()` automatically enqueues an `AnalysisRun` in `AnalyticsJobQueue`.

---

## 3. Request Budgets & Collection Modes

Defined in [`BUDGET_CONFIGS`](../backend/app/services/rate_budgeter.py):

| Entity Stream | `fast_snapshot` (Max `10` requests) | `standard_diagnostic` (Max `50` requests) | Notes |
|---|---|---|---|
| `repository_metadata` | `1` | `1` | `/repos/{owner}/{repo}` |
| `commits` | `1` | `3` | `per_page=100` (100–300 commits) |
| `commit_details` | `0` | `5` | `/commits/{sha}` diff lines |
| `contributors` | `1` | `2` | `/contributors?per_page=100` |
| `pull_requests` | `1` | `3` | `/pulls?state=all&per_page=100` |
| `pr_details` | `0` | `5` | `/pulls/{n}` diff stats (`additions`/`deletions`/`changed_files`) |
| `pr_reviews` | `2` | `15` | Sampled `/pulls/{n}/reviews` |
| `pr_inline_comments` | `0` | `2` | `/pulls/comments` |
| `issues` | `1` | `4` | `/issues?state=all&per_page=100` |
| `issue_comments` | `1` | `2` | `/issues/comments?per_page=100` |
| `releases` | `1` | `2` | `/releases` (with `/tags` fallback) |
| `discussions` | `1` | `3` | GraphQL API v4 discussions query |
| `headroom` | `1` | `5` | Reserved safety buffer |

---

## 4. Rate Limiting, Resilience & Resumption

- **Concurrency & Pacing:** [`AsyncGitHubClient`](../backend/app/services/github_client.py) enforces a concurrency semaphore of 2 concurrent upstream requests and a 100ms inter-request pacing delay.
- **Primary Rate-Limit Safety:** If `X-RateLimit-Remaining < 10`, the client halts further requests, flags `is_primary_rate_limited=True`, and records `resumes_at = X-RateLimit-Reset`.
- **403/429 Handling:** Inspects `Retry-After`. If `<= 300s`, sleeps asynchronously before retrying. If `> 300s`, raises `GitHubRateLimitExceededException`.
- **Automatic Rate-Limit Resumption:** Jobs in `rate_limited` status record `coverage_summary["resumes_at"]`. `DurableJobQueue.reopen_resumed_rate_limited_jobs()` periodically checks timestamps and transitions eligible jobs back to `pending` once reset time has elapsed.

---

## 5. Execution Modes (Embedded vs. Standalone)

OpenSourceLens supports two worker execution modes:

1. **Embedded Mode (Default for Local Development):**
   - Enabled by default via `ENABLE_INGESTION_WORKER=true` in `config.py`.
   - FastAPI automatically starts `IngestionWorker.run_worker_loop()` as a background task during application boot. Single-command startup (`uvicorn backend.app.main:app`) handles both API requests and background ingestion.
2. **Standalone Mode (Production & Distributed Scale):**
   - Run as an independent process daemon:
     ```powershell
     .\.venv\Scripts\python.exe -m backend.app.services.ingestion_worker
     ```
   - In containerized production deployments, `docker-compose.yml` orchestrates decoupled worker containers with dedicated worker IDs and resource allocations.
