# OpenSourceLens — Data Storage

This directory manages all datasets for OpenSourceLens, maintaining strict separation between raw API responses and processed datasets.

---

## Directory Layout

```text
data/
├── raw/                      # Unmodified GitHub REST API responses (Git-ignored)
│   ├── repositories/         # Repository metadata ({owner}__{repo}.json)
│   ├── issues/               # Issue batches ({owner}__{repo}__issues_{batch}.json)
│   ├── comments/             # Discussion threads ({owner}__{repo}__issue_{number}__comments.json)
│   └── events/               # Lifecycle activity ({owner}__{repo}__issue_{number}__events.json)
│
├── processed/                # Cleaned and normalized datasets for ML/NLP (Phase 2+)
│   ├── issues.parquet        # Master processed issue table
│   └── label_mappings.json   # Repository-to-canonical label taxonomy maps
│
└── README.md                 # Data documentation and conventions
```

---

## Data Pipeline Principles

1. **Separation of Raw and Processed:**
   - Raw data represents the original truth as returned by GitHub.
   - Processing, cleaning, label normalization, and feature extraction (Phase 2) read from `raw/` and write to `processed/`.
   - Raw files are **never modified in-place**.
2. **Version Control Exclusion:**
   - `data/raw/` and `data/processed/` are git-ignored via `.gitignore` to prevent committing large data files.
3. **Encoding Standard:**
   - All JSON files must be encoded in `UTF-8` (`ensure_ascii=False`) to preserve technical symbols, international text, and emojis (e.g. Kubernetes labels `¯\_(ツ)_/¯`).
4. **Atomic Writes:**
   - Files are written to a temporary buffer before atomic replacement to prevent partial or corrupted files during interruption.
