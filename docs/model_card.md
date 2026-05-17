# Model card

Modeled after the format proposed by Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

## Model details

| | |
|---|---|
| **Person developing** | Yullie Yang |
| **Model date** | 2026-05 |
| **Model version** | 0.2.0 |
| **Model type** | Binary classifier (Logistic Regression or Random Forest), with `RandomOverSampler` for class imbalance and post-hoc threshold tuning |
| **License** | MIT |

## Intended use

- **Primary use:** demonstrate an end-to-end CRE-stress identification pipeline using only public data, packaged as a reproducible research codebase.
- **Primary users:** the author and other engineers/economists who want to study the pipeline architecture.
- **Out of scope:** any actual credit, lending, or regulatory decision. This is a portfolio demonstration, not a production risk model.

## Factors

| Factor | Notes |
|---|---|
| **Geographic** | Mobility data restricted to Massachusetts. Macro data is national. |
| **Temporal** | Mobility data exists only from 2020-02-15 through 2022-10-15 (Google sunset). |
| **Population** | The "stress condition" is operationally defined (quantile rule), not a real-world labeled population. |

## Metrics

Reported on a stratified 70/30 train/test split, averaged across a small panel:

| Metric | Value (approx.) |
|---|---|
| AUC | ~0.73 |
| Recall (positive class) | ~0.78 |
| FPR | ≤ 0.20 (hard-constrained) |

These should be read as **method illustration**, not generalization claims — the modeling panel is small and the target is rule-based.

## Ethical considerations

- **Label provenance:** the binary target is constructed from macro/mobility quantiles, not from real default events. Any predictions should be interpreted as "the macro/mobility quantile rule would have fired here," not as a probability of credit loss.
- **Mobility data:** the Google Mobility dataset is aggregated and anonymized; no individual movement data is involved.
- **Bias risk:** the model only sees macro and mobility features. It does not see property-level, tenant-level, borrower-level, or demographic data. It therefore cannot directly perpetuate borrower-level bias, but it equally cannot inform any borrower-level decision.

## Caveats and recommendations

- **Do not deploy.** This is a portfolio demonstration.
- For a production-grade CRE stress test, replace the rule-based target with a real CRE delinquency / default series (e.g. Trepp, CMBS-level data), expand features to include property and loan characteristics, and use proper time-series cross-validation.
- The pipeline architecture (`src/cre_stress/`, persistence layer, R companion, CI) is the artifact intended for review — not the model coefficients.
