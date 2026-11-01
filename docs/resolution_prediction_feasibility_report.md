# OpenSourceLens — Resolution Prediction Feasibility & Scientific Reliability Report

**Phase:** Phase 8 (Resolution Analysis & Prediction)  
**Document Type:** Milestone Synthesis & Scientific Decision Report  
**Date:** November 2026  
**Status:** Approved — Advisory Velocity Classification & Descriptive Analytics Deployed; Continuous Hard Regression Quarantined

---

## 1. Executive Summary

As part of OpenSourceLens's master roadmap, **Phase 8 (Resolution Analysis & Prediction)** was designated as an empirical investigation to determine whether the lifespan of GitHub issues can be meaningfully and reliably predicted at creation time ($t = 0$).

Over seven sequential steps (Steps 8.1 through 8.7), we formulated an exact survival definition across 2,328 issues, engineered 28 strictly isolated creation-time features, performed econometric factor analysis, benchmarked dual-track machine learning models, and executed rigorous statistical diagnostics on holdout test data.

### The Scientific Verdict
1. **Continuous Hour Prediction ($\hat{T} = \text{hours}$): QUARANTINED FROM PRODUCTION SLA PROMISES.**  
   Continuous regression achieves an overall holdout $R^2 = 0.276$. However, **within-repository $R^2$ drops to $-0.04$ to $+0.18$**. Over $72\%$ of resolution time variance is driven by unobservable post-creation maintainer availability, asynchronous PR discussions, and release cadences. Presenting developers with exact countdown timers or completion hours generates false precision and damages platform trust.
2. **Discrete Velocity Classification (Speed Tiers): APPROVED FOR PRODUCTION AS AN ADVISORY SERVICE.**  
   Classifying issues into 4 operational brackets (`< 1 Day`, `1-7 Days`, `7-30 Days`, `30+ Days`) achieves **$51.10\%$ exact accuracy** and **$88.11\%$ adjacent accuracy ($\pm 1$ tier)**, outperforming naive zero-information baselines by $+12.33\%$ absolute accuracy. Severe errors ($\ge 2$ tiers) occur in only $11.89\%$ of cases. This provides actionable velocity guidance without overpromising.
3. **Descriptive Historical Analytics: CORE PRODUCTION CAPABILITY.**  
   Kaplan-Meier survival curves, empirical quartiles, and repository triage rates represent factual, reproducible ground truth and form the primary presentation layer for the analytics dashboard.

---

## 2. Empirical Evidence Across Phase 8 Milestones

### 2.1 Distribution Characteristics & Censoring (Steps 8.1 & 8.2)
- Across $n = 2,328$ harvested issues, $1,135$ are closed resolution events and $1,193$ are open right-censored issues.
- Raw resolution duration exhibits extreme positive skewness ($S = 2.616$, excess kurtosis $7.246$).
- Logarithmic transformation $y = \ln(\text{resolution\_hours} + 1)$ normalizes residuals ($S = -0.158$), stabilizing variance for parametric inference.
- Non-parametric Kruskal-Wallis tests prove highly statistically significant differences in resolution speed across repositories ($p = 1.71 \times 10^{-48}$) and categories ($p = 1.07 \times 10^{-17}$).

### 2.2 Strict Creation-Time Feature Quarantine (Step 8.3)
- All post-creation metrics (`comments_count`, `reactions_count`, `closed_at`) were quarantined to eliminate data leakage.
- 28 pure $t = 0$ predictors were engineered: structural text metrics, code presence, stack trace signatures, URL references, title bracket tags, author affiliation, timing, and one-hot repository and category indicators.

### 2.3 Econometric Factor Decomposition (Step 8.4)
Multivariate OLS modeling ($R^2 = 0.2523, F = 13.33, p < 10^{-50}$) decomposed explained variance across 5 factor families via Partial ANOVA:
- **Repository Governance: 71.1% of explained variance ($F = 45.27, p < 10^{-20}$)**. The organizational ecosystem dominates issue lifespan.
- **Structural Text Complexity: 9.4% of explained variance ($F = 1.99, p = 0.0373$)**.
- **Technical Evidence (Code blocks, URLs): 8.3% of explained variance ($F = 3.16, p = 0.0077$)**. Code blocks increase resolution time by $+22.5\%$ per standard deviation.
- **Issue Typology (Bug vs. Feature vs. Doc): 7.4% of explained variance ($F = 2.84, p = 0.0147$)**. Doc issues resolve $76.7\%$ faster than core defects.
- **Temporal Filing Context: 3.8% of explained variance ($F = 1.20, p = 0.3013$, not significant)**. Weekend filing does not statistically prolong total resolution time ($p = 0.696$).

### 2.4 Holdout Benchmark & Diagnostics (Steps 8.5 & 8.6)
On the unseen holdout test set ($n = 227$):
- **Ridge Regressor:** Test $R^2 = 0.2759$, Test $\text{MAE} = 1.3915$ log-hours, Spearman $\rho = +0.518$ ($p = 7.15 \times 10^{-17}$).
- **Random Forest Classifier:** Test $\text{Accuracy} = 51.10\%$, Test $\text{Macro F1} = 0.4032$, Balanced $\text{Accuracy} = 44.30\%$.
- **Ordinal Proximity:** In **$88.11\%$** of cases, the predicted tier is either exact or off by at most one neighboring tier.
- **Residual Diagnostics:** Gaussian residual hypothesis is retained (D’Agostino-Pearson $p = 0.0711$), and Breusch-Pagan test confirms homoskedasticity ($p = 0.1274$).

---

## 3. Production Architecture & API Policy (Phases 9 & 10)

### 3.1 Backend Service Implementation: `ResolutionPredictor`
The production inference engine [`ml/resolution_analysis/predictor.py`](file:///d:/Dev/CODING/Projects/DS-PROJECT/OpenSourceLens/ml/resolution_analysis/predictor.py) exposes a clean, synchronous inference interface:

```python
from ml.resolution_analysis import ResolutionPredictor

predictor = ResolutionPredictor()
result = predictor.predict_velocity(
    title="[API] Null pointer exception in tokenizer",
    body="Reproduction snippet:\n```python\ntokenizer.encode(None)\n```",
    repo_name="huggingface/transformers",
    canonical_label="bug"
)
```

### 3.2 FastAPI Endpoint Contract (Phase 9)
```json
POST /api/issues/predict-velocity
{
  "predicted_tier": "< 1 Day",
  "confidence": 0.575,
  "tier_probabilities": {
    "< 1 Day": 0.575,
    "1-7 Days": 0.306,
    "7-30 Days": 0.076,
    "30+ Days": 0.043
  },
  "adjacent_window": {
    "lower_bound": "< 1 Day",
    "upper_bound": "1-7 Days",
    "confidence_coverage": "88.1% empirical proximity guarantee (±1 tier)"
  },
  "advisory_factors": [
    "Component tag detected in title (+16.4% faster historical triage).",
    "Technical code snippets present (+22.5% longer expected reproduction duration)."
  ],
  "disclaimer": "Advisory estimate based strictly on filing-time characteristics and historical project velocity. Does not constitute an SLA guarantee."
}
```

### 3.3 UI Presentation Rules (Phase 10)
1. **Permitted UI Elements:**
   - Velocity Bracket Badges: `Fast (< 1 Day)` (Green), `Moderate (1-7 Days)` (Blue), `Slow (7-30 Days)` (Orange), `Very Slow (30+ Days)` (Red).
   - Confidence meter (e.g. `58% confidence`).
   - Descriptive Kaplan-Meier survival curves showing historical milestone completion rates.
2. **Strictly Prohibited UI Elements:**
   - Exact hour/day countdown timers (e.g., *"This issue will close in 14.2 hours"*).
   - Binding SLA commitment statements.

---

## 4. Phase 8 Synthesis Figures & Artifacts

| Artifact | Path | Description |
|---|---|---|
| **Figure 29** | `ml/resolution_analysis/figures/fig29_resolution_time_distribution_and_survival.png` | Distribution moments & Kaplan-Meier survival curves |
| **Figure 30** | `ml/resolution_analysis/figures/fig30_feature_correlations_and_importance.png` | Feature correlations, leakage audit, mutual info rankings |
| **Figure 31** | `ml/resolution_analysis/figures/fig31_resolution_time_factors.png` | Econometric factor analysis, ANOVA decomposition, interaction plot |
| **Figure 32** | `ml/resolution_analysis/figures/fig32_baseline_model_benchmarks.png` | Dual-track benchmark comparisons & confusion matrix |
| **Figure 33** | `ml/resolution_analysis/figures/fig33_resolution_model_evaluation.png` | Residual diagnostics, Q-Q plot, subgroup disaggregation |
| **Figure 34** | `ml/resolution_analysis/figures/fig34_resolution_decision_and_architecture.png` | Milestone decision matrix & Phase 9 inference architecture |
| **Decision Manifest** | `data/processed/evaluation/resolution_reliability_decision.json` | Machine-readable scientific verdict and integration parameters |
| **Champion Models** | `models/resolution_regressor_champion.joblib`<br>`models/resolution_classifier_champion.joblib` | Serialized champion models and scaler parameters |
