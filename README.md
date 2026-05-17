# cre-stress-test

A production-style **commercial real estate (CRE) credit-risk modeling pipeline**, built as research-support infrastructure for stress testing under macro and mobility shocks.

The pipeline ingests public data from FRED, Google COVID-19 Mobility, and Boston Public Zoning, persists everything to a SQLite database, fits a binary classifier for elevated-stress conditions with explicit threshold control on false-positive rate and recall, and produces ROC / SHAP / commentary artifacts for review.

> Originally a personal interview take-home; restructured here as a research-codebase reference. Data sources are 100% public.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![R](https://img.shields.io/badge/R-4.3%2B-blue)
![tests](https://img.shields.io/badge/tests-pytest-green)
![license](https://img.shields.io/badge/license-MIT-green)

## Design goals

| Goal | How it shows up |
|---|---|
| **Reproducible** | One-command pipeline (`make run`), env-driven config, deterministic seeds, locked deps |
| **Production-style** | Numbered pipeline stages, modular `src/cre_stress/` package, type hints, logging, pytest |
| **Auditable** | SQLite persistence with explicit schema, model card, methodology doc |
| **Bilingual (R + Python)** | Python for ML / ingestion; R for ARIMA-based macro stress-scenario forecasting |
| **AI-assisted, responsibly** | Optional Claude-API commentary module documents results in plain English |
| **Reviewable on GitHub** | CI runs pytest on push; figures committed so reviewers can audit without running code |

## Pipeline

```
                ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
                │   FRED API  │    │  Google      │    │  Boston      │
                │ (FEDFUNDS,  │    │  Mobility    │    │  Zoning      │
                │  UNRATE,    │    │  (MA)        │    │  CKAN API    │
                │  GDP)       │    │              │    │              │
                └──────┬──────┘    └──────┬───────┘    └──────┬───────┘
                       │                  │                   │
                       ▼                  ▼                   ▼
                ┌──────────────────────────────────────────────────────┐
                │  src/cre_stress/ingest.py  →  data/raw/*.csv         │
                └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────────┐
                │  src/cre_stress/persist.py  →  SQLite                │
                └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────────┐
                │  src/cre_stress/features.py                          │
                │    • Quarterly alignment                             │
                │    • Quantile-based stress label (UNRATE q75 OR     │
                │      retail mobility q25)                            │
                │    • Optional R/ARIMA-forecasted scenario covariates │
                └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────────┐
                │  src/cre_stress/models.py                            │
                │    • Logistic Regression / Random Forest             │
                │    • RandomOverSampler / SMOTE for class imbalance   │
                │    • Threshold tuning: FPR ≤ τ, recall ≥ ρ           │
                └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────────┐
                │  src/cre_stress/evaluate.py                          │
                │    • Classification report, ROC, AUC                 │
                │    • SHAP summary + waterfall                        │
                │    • Outputs: outputs/figures/, outputs/tables/      │
                └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────────────────┐
                │  src/cre_stress/commentary.py  (optional)            │
                │    Claude API generates a plain-English summary of   │
                │    classification report and top SHAP drivers        │
                └──────────────────────────────────────────────────────┘
```

## Quick start

```bash
# 1. Install
make install

# 2. Configure: add your FRED API key (free at https://fred.stlouisfed.org/)
cp .env.example .env
# Edit .env

# 3. Run the full pipeline (ingest → persist → train → evaluate)
make run

# 4. Optional: generate AI commentary
make commentary

# 5. Browse results
make dashboard      # Streamlit dashboard at http://localhost:8501
```

## Repository layout

```
cre-stress-test/
├── src/cre_stress/         # Python package: ingest, persist, features, models, evaluate, commentary
├── src/dashboard/          # Streamlit dashboard
├── R/                      # R companion: ARIMA forecast of stress-scenario macro inputs
├── sql/                    # SQLite schema and seed data
├── scripts/                # Pipeline orchestration entry points
├── tests/                  # pytest suite
├── data/                   # raw/ and processed/ snapshots (public-data only)
├── outputs/                # figures/ and tables/ from the latest run
├── reports/                # Standalone HTML reports
├── docs/                   # methodology.md, data_dictionary.md, model_card.md
├── notebooks/              # Cleaned-up exploratory notebooks
└── .github/workflows/      # CI: pytest + ruff
```

## Key results (latest run)

| Metric | Value | Notes |
|---|---|---|
| AUC | **0.73** | Out-of-sample, stratified test split |
| Recall (positive class) | **~78%** | Subject to threshold constraint |
| FPR | **≤ 20%** | Hard constraint via post-hoc threshold tuning |

See `outputs/tables/` for the full classification report, `outputs/figures/` for ROC and SHAP plots, and `docs/model_card.md` for limitations and intended use.

## Why R *and* Python

The Python pipeline does the ingestion, persistence, and classification. The R companion (`R/forecast_stress_scenarios.R`) uses `forecast::auto.arima` to project the macro drivers — `UNRATE`, `FEDFUNDS`, `GDP` — forward under baseline / adverse / severely-adverse scenarios. The Python pipeline can then re-score using those projected covariates. This mirrors how policy-research teams typically split work: R for the time-series side, Python for ML, both sharing a common data layer.

## Documentation

- [docs/methodology.md](docs/methodology.md) — full methodology, target definition, sampling strategy, threshold choice
- [docs/data_dictionary.md](docs/data_dictionary.md) — every column, type, source, transform
- [docs/model_card.md](docs/model_card.md) — intended use, limitations, ethical considerations

## License

MIT — see [LICENSE](LICENSE).
