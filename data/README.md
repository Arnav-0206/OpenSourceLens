# OpenSourceLens — Data Storage Architecture

This directory manages all local database files, backup baselines, and historical datasets for OpenSourceLens.

---

## Directory Layout

```text
data/
├── opensource_lens.db        # Primary SQLite database (WAL mode, 21 ORM tables)
├── opensource_lens.db-wal    # SQLite Write-Ahead Log
├── opensource_lens.db-shm    # SQLite shared-memory index
│
├── backups/                  # Immutable benchmark baseline backups
│   └── benchmark_baseline.db # Baseline SQLite database for offline research
│
├── raw/                      # Historical raw GitHub REST API responses (Git-ignored)
│   ├── repositories/         # Repository metadata ({owner}__{repo}.json)
│   ├── issues/               # Issue batches ({owner}__{repo}__issues_{batch}.json)
│   ├── comments/             # Discussion threads ({owner}__{repo}__issue_{number}__comments.json)
│   └── events/               # Lifecycle activity ({owner}__{repo}__issue_{number}__events.json)
│
├── processed/                # Normalized datasets for benchmark evaluation
│   ├── issues.parquet        # Master processed issue table
│   └── label_mappings.json   # Repository-to-canonical label taxonomy maps
│
└── README.md                 # Data documentation and conventions
```

---

## Data Persistence & Principles

1. **Relational Database (`opensource_lens.db`):**
   - Managed via SQLAlchemy 2.0 and Alembic migrations (`001` through `005`).
   - Runs in SQLite `WAL` mode (`PRAGMA journal_mode=WAL`) with foreign key constraints enabled.
   - Holds repository entities, collection runs, analytics runs, attempt logs, and dense vector embeddings.
2. **Dense Vector Embeddings Storage:**
   - Dense sentence embeddings (`all-MiniLM-L6-v2`, 384 dimensions) are stored directly inside the relational database in `issue_embeddings` and `discussion_embeddings` as raw IEEE-754 binary BLOBs (1,536 bytes per vector).
   - This enables portable, zero-external-vector-database operation on both SQLite and PostgreSQL.
3. **Separation of Raw API Data and Benchmark Datasets:**
   - `data/raw/` stores unmodified GitHub JSON responses for benchmark baseline analysis.
   - Production dynamic ingestion operates directly through `RepositoryCollector` into the relational database.
4. **Encoding Standard:**
   - All text and JSON processing enforces `UTF-8` encoding to preserve technical symbols, international text, and emojis.
