# OpenSourceLens — Dynamic GitHub Ingestion Pipeline

OpenSourceLens dynamically collects commits, contributors, pull requests, PR reviews, inline review comments, pure issues, conversation comments, and releases from **any eligible public GitHub repository** via GitHub REST API v3.

> **Product Scope & Collection Depth Gaps:** Per [product-vision.md](./product-vision.md), OpenSourceLens targets rich public GitHub API intelligence rather than shallow counters. Current Phase B collection has three known depth gaps scheduled for expansion: (1) `/pulls` list calls do not populate per-PR `additions`, `deletions`, `changed_files`, `body`, `head`/`base` branches, `draft` flag, or requested reviewers; (2) `commit_details` diffs are capped at `0` (`fast_snapshot`) or `5` (`standard_diagnostic`); and (3) issue `reactions_count`/assignees, repository tags fallback, and **GitHub Discussions** (GraphQL/REST) are not yet harvested.

---

## 1. Complete Ingestion-to-Analytics Lifecycle

```
GitHub URL / Slug
  └─▶ GitHubURLValidator (syntax + host + reserved-path checks)
        └─▶ RepositoryEligibilityChecker (GET /repos/{owner}/{repo}, public & non-empty gates)
              └─▶ Preview Response (metadata + budget estimate + 5-domain confidence preview)
                    └─▶ Submit (POST /api/ingest/submit -> DurableJobQueue.submit_job -> CollectionRun pending)
                          └─▶ IngestionWorker (claim + 15s heartbeat + lease_version fencing)
                                └─▶ RepositoryCollector + AsyncGitHubClient + EntityRateBudgeter
                                      └─▶ Entity Normalization & Upsert + Mutated Record Tracking
                                            └─▶ Atomic analytics_source_revision Bump (only if mutated)
                                                  └─▶ DurableJobQueue.complete_job (status + coverage_summary)
                                                        └─▶ _enqueue_post_ingestion_analytics -> AnalyticsJobQueue
```

---

## 2. Stage-by-Stage Behavior

1. **URL Validation ([`validator.py`](../backend/app/services/validator.py)):** Accepts full URLs (`https://github.com/owner/repo`, `.git` suffix stripped), scheme-less URLs (`github.com/owner/repo`), or canonical slugs (`owner/repo`). Rejects `@owner/repo` npm scopes, non-GitHub hosts (only `github.com` and `www.github.com` allowed), deep sub-paths (`/issues`, `/tree/...`), reserved owner paths (`explore`, `marketplace`, `settings`, etc.), owner names > 39 chars or containing `--`, and invalid repo names.
2. **Eligibility & Preview ([`repository_eligibility.py`](../backend/app/services/repository_eligibility.py), [`ingest.py`](../backend/app/api/v1/endpoints/ingest.py)):**
   - **Gates:** Repository must be **public** (`private == False`) and **non-empty** (`commits_count >= 1` if supplied, otherwise GitHub `size > 0` KB). Archived and forked repositories remain **eligible** and receive informational badges.
   - `POST /api/ingest/preview` does **not** fetch commit/PR/issue streams; it inspects only `/repos/{owner}/{repo}` metadata and passes `open_issues_count` into [`ConfidenceMatrixEvaluator`](../backend/app/services/confidence_matrix.py) to preview data sufficiency expectations and budget allocations for both modes.
3. **Submission & Deduplication (`POST /api/ingest/submit` -> `DurableJobQueue.submit_job()`):**
   - Validates eligibility live on GitHub (unless mocked/cached), upserts the `Repository` row (using GitHub `id` if present, or a deterministic per-process fallback `abs(hash(repo_name)) % 1_000_000_000` if absent), and inserts a `pending` `CollectionRun`.
   - If an active (`pending` or `running`) `CollectionRun` already exists for `repository_id` (enforced by `uq_collection_runs_active_repo`), submission returns the existing active job (`already_active` semantics in HTTP `200/202` contract; see [api.md](./api.md)).
4. **Worker Claim & Heartbeat (`IngestionWorker`, `DurableJobQueue`):**
   - [`IngestionWorker.process_one_job()`](../backend/app/services/ingestion_worker.py#L58-L248) first runs `recover_expired_jobs()` (marking expired jobs with `retry_count < max_retries` back to `pending`, or `failed` once `max_retries=2` is reached), then claims the next pending job (`status='running'`, `lease_version += 1`, `leased_until = now + 600s`).
   - An async `_heartbeat_loop` task renews the lease every `15.0s` checking `(id, worker_id, lease_version)`.
5. **Collection & Normalization (`RepositoryCollector`):** Collects streams in a fixed order within `EntityRateBudgeter`:
   1. `collect_repository_metadata`
   2. `collect_commits` + up to `commit_details` detailed diff calls (`/commits/{sha}` for `additions`/`deletions`/`total_changes`)
   3. `collect_contributors` (anchoring strictly on `github_id` or unlinked `git_email` without speculative display-name merging)
   4. `collect_pull_requests` (`/pulls?state=all&sort=updated&direction=desc`)
   5. `collect_pr_reviews` (sampled PRs up to 2 or 20; updates `time_to_first_review_hours`)
   6. `collect_pr_inline_comments` (`/pulls/comments`; **0** budget in `fast_snapshot`)
   7. `collect_issues` (`/issues?state=all&sort=updated&direction=desc`, discarding items with `"pull_request" in item`, running `IssueTextCleaner`, technical feature flags, canonical regex category extraction, and `speed_tier`)
   8. `collect_issue_comments` (`/issues/comments`, linking exclusively to either `issue_id` OR `pull_request_id` per `chk_issue_comment_parent`)
   9. `collect_releases` (`/releases`)
   10. `reconcile_contributor_metrics()` (recomputes distinct commit/PR/review/issue counts, tenure, `is_newcomer <= 30d`, `is_core`).
6. **Atomic Mutation & Revision Tracking:** Every helper populates `mutated_records`. Only when `mutated_records` is non-empty does `collect_repository()` atomically bump `repositories.analytics_source_revision += 1`.
7. **Completion & Post-Ingestion Analytics Enqueue:**
   - `DurableJobQueue.complete_job()` assigns terminal `CollectionRun.status`:
     - `summary.errors` non-empty → `"failed"`
     - `summary.resumes_at is not None` → `"rate_limited"`
     - `summary.is_partial` is True → `"completed_partial"`
     - otherwise → `"completed"`
   - After commit, `IngestionWorker._enqueue_post_ingestion_analytics()` calls `AnalyticsJobQueue.enqueue_analysis()` **only if** `auto_enqueue_analytics=True`, `not summary.errors`, `summary.resumes_at is None`, and persisted `CollectionRun.status in ("completed", "completed_partial")`. Enqueue failures are caught, rolled back cleanly, logged, and recoverable via `AnalyticsJobQueue.reconcile_analytics_jobs()`.

---

## 3. Collection Modes & Request Budgets

Defined in [`BUDGET_CONFIGS`](../backend/app/services/rate_budgeter.py#L17-L50):

| Entity Stream | `fast_snapshot` (Max `10` reqs) | `standard_diagnostic` (Max `50` reqs) | Notes |
|---|---|---|---|
| `repository_metadata` | `1` | `1` | `/repos/{owner}/{repo}` |
| `commits` | `1` | `3` | `per_page=100` (100–300 commits) |
| `commit_details` | `0` | `5` | Individual `/commits/{sha}` diff calls |
| `contributors` | `1` | `2` | `/contributors?per_page=100` |
| `pull_requests` | `1` | `3` | `/pulls?state=all&per_page=100` |
| `pr_reviews` | `2` | `20` | Sampled per-PR `/pulls/{n}/reviews` |
| `pr_inline_comments` | `0` | `2` | `/pulls/comments` (`0` = skipped by mode) |
| `issues` | `1` | `5` | `/issues?state=all&per_page=100` |
| `issue_comments` | `1` | `3` | `/issues/comments?per_page=100` |
| `releases` | `1` | `1` | `/releases?per_page=100` |
| `headroom` | `1` | `5` | Reserved headroom buffer |

*(Note: `CollectionRun.mode` column also accommodates `"full_history"` as recognized by analytics calculators, though `IngestionSubmitRequest` pattern and `BUDGET_CONFIGS` currently accept only `fast_snapshot` and `standard_diagnostic`.)*

---

## 4. Pagination, Rate Limits, Partiality & Stream Coverage

- **Pagination (`AsyncGitHubClient.paginate`):** Traverses RFC 5988 `Link` headers (`rel="next"`), halting cleanly as `is_partial=True` if `EntityRateBudgeter.can_consume()` becomes false or `is_primary_rate_limited` triggers.
- **Rate Limits & Retries (`AsyncGitHubClient.request`):**
  - Primary limit guard: If `X-RateLimit-Remaining < 10`, marks `is_primary_rate_limited=True` and sets `resumes_at` to `X-RateLimit-Reset`.
  - On HTTP `403`/`429`: Parses `Retry-After` (integer seconds or HTTP date). If `retry_seconds > 300` (5 min), immediately halts and raises `GitHubRateLimitExceededException` with `resumes_at`; if `<= 300` and budget permits retry (`max_retries=3`), sleeps full duration asynchronously before retrying.
  - Note: Rate-limited `CollectionRun` jobs (`status="rate_limited"`) persist `resumes_at` in `coverage_summary`, but **no built-in scheduler automatically wakes and re-enqueues `rate_limited` jobs after `resumes_at` passes**.
- **Stream Coverage Format (`coverage_summary`):**
  - `DurableJobQueue.complete_job()` persists each entity's status (`"complete"`, `"truncated"`, `"skipped"`, `"failed"`, etc.) BOTH at top-level keys (with reserved metadata protection) AND nested under `coverage_summary["entity_status"]` so downstream C.4/C.6 readers work consistently.
  - Direct calls to `RepositoryCollector.collect_repository(..., collection_run_id=...)` bypassing `DurableJobQueue.complete_job()` write a slightly different flat structure and check `resumes_at` before `errors`.
