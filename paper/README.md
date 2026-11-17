# OpenSourceLens — IEEE Conference Paper Package

This directory contains the complete, camera-ready **IEEEtran** 2-column LaTeX manuscript, verified BibTeX bibliography, and reproducible vector figures.

## File Inventory

- `main.tex` — Complete IEEE conference manuscript (`\documentclass[conference]{IEEEtran}`, 6–8 pages compiled).
- `references.bib` — 15 verified, peer-reviewed BibTeX citations (`IEEEtran.bst`).
- `generate_figures.py` — Reproducible `matplotlib` script generating all four publication figures directly from verified evaluation artifacts in `data/processed/evaluation/` and `models/`.
- `figures/` — Vector PDF (`.pdf`) and 300-DPI PNG (`.png`) figures:
  - `fig1_framework.pdf` — Guarded OSS analytical framework architecture (`v2026.2`).
  - `fig2_classification_loro.pdf` — DistilBERT truncation decay by length quartile & strict LORO abstention transfer.
  - `fig3_retrieval_calibration.pdf` — Dense (`all-MiniLM-L6-v2`) vs.\ lexical TF-IDF retrieval & cosine threshold calibration.
  - `fig4_resolution_feasibility.pdf` — ANOVA Type-III variance attribution (`71.11%` Repository Governance) & pooled vs.\ within-repo $R^2$ collapse.

## Compiling on Overleaf (Recommended)

1. Zip the entire `paper/` directory (`main.tex`, `references.bib`, and `figures/*.pdf`).
2. In Overleaf, click **New Project → Upload Project** and upload the `.zip` archive.
3. Ensure compiler is set to **pdfLaTeX** (default) with `main.tex` as the main document.
4. Edit the `\author{...}` block near the top of `main.tex` with your real author name(s), department, institution, and email address(es).

## Compiling Locally (TeX Live / MiKTeX)

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Regenerating Figures

```powershell
.\.venv\Scripts\python.exe paper\generate_figures.py
```
