# Methodology

## Goal

Identify periods of **elevated CRE credit-risk conditions** using public macro and mobility data, with explicit control over false-positive rate (a key requirement for any operational risk-monitoring tool).

## Data sources

| Source | Series | Frequency | Notes |
|---|---|---|---|
| FRED (St. Louis Fed) | `FEDFUNDS`, `UNRATE`, `GDP` | Monthly / Quarterly | Authoritative macro indicators |
| Google COVID-19 Community Mobility Reports | Retail, grocery, transit, workplace, residential, parks (% change from baseline) | Daily | Filtered to Massachusetts |
| City of Boston Open Data (CKAN) | Public zoning subdistricts | Static | Used as a context layer; not currently in modeling features |

All data is 100% public. The Google COVID-19 Mobility dataset was sunset in 2022, so the analysis is bounded by what was published.

## Target definition

There is no canonical, publicly labeled "CRE stress period" series, so the pipeline defines stress conditions via a transparent quantile rule on observed macro/mobility behavior:

- `target_or` = 1 if `UNRATE ≥ Q75(UNRATE)` **OR** `retail_mobility ≤ Q25(retail_mobility)`
- `target_hard` = 1 if both conditions hold (stricter; positive class is rare)

The modeling code uses `target_hard` by default. This is openly an *operational* definition (not an economic claim about default rates); a future revision would calibrate against an actual CRE delinquency series.

## Sampling and label noise

A configurable fraction (`label_noise_frac = 0.10`) of labels is flipped at random to stress-test the classifier under label uncertainty — useful for surfacing models that have over-fit to a clean (and unrealistic) target.

## Modeling

- **Estimators:** Logistic Regression (L2, `C=0.5`) and Random Forest (depth-2, 10 trees). Both are kept deliberately small to suit the very limited sample size of the toy panel.
- **Class imbalance:** handled with `RandomOverSampler` from `imbalanced-learn`, sampling strategy = 0.2.
- **Pipeline:** `SimpleImputer(median) → RandomOverSampler → estimator`. Imputation precedes oversampling so synthetic positive rows are based on imputed features, not NaNs.
- **Threshold selection:** post-hoc — picks the lowest threshold satisfying `FPR ≤ 0.2` and `recall ≥ 0.8`; if no threshold satisfies both, picks the threshold closest to `recall = 0.8`.

## Reported metrics

| Metric | Why it's reported |
|---|---|
| AUC | Threshold-free measure of separability |
| Recall (positive class) | We care more about catching stress than calling false alarms |
| FPR | The operational cost: each false-positive is investigative work |
| Precision / F1 | Reported but not optimized |
| SHAP | Per-feature attribution; helps explain why a flag fired |

## Scenario forecasting (R companion)

`R/forecast_stress_scenarios.R` reads the same SQLite database and projects each macro series forward 8 quarters with `forecast::auto.arima`, producing baseline / adverse / severely-adverse paths. The Python pipeline can then be re-run on those projected covariates to score scenarios.

## Limitations

- Toy-scale data; AUC of ~0.73 should not be taken as a generalization claim.
- Mobility data ends in 2022; for current monitoring a different proxy is needed.
- Target is rule-based; would benefit from calibration against a real-world stress label.
- No spatial CRE features (zoning data ingested but not in modeling features yet).
