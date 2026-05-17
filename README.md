# cre-stress-test

A production-style **commercial real estate (CRE) credit-risk modeling pipeline**, built as research-support infrastructure for stress testing under macro and mobility shocks.

The pipeline ingests public data from FRED, Google COVID-19 Mobility, and Boston Public Zoning, persists everything to a SQLite database, fits a binary classifier for elevated-stress conditions with explicit threshold control on false-positive rate and recall, and produces ROC / SHAP / commentary artifacts for review.

> **Scope and disclaimer.** This is a personal portfolio project, originally
> built as an interview take-home and restructured here as a research-codebase
> reference. **All data sources are 100% public** (FRED, Google COVID-19
> Mobility, Boston Public Zoning). The repository does **not** reflect any
> employer's internal data, methodology, model logic, or assumptions. It is
> not a deployed credit-risk system and should not be used as one.

[![ci](https://github.com/yullieyang/cre_stress_test/actions/workflows/ci.yml/badge.svg)](https://github.com/yullieyang/cre_stress_test/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![R](https://img.shields.io/badge/R-4.3%2B-blue)
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

## Limitations

- **Synthetic target.** The "stress" label is constructed from quantile cuts
  on `UNRATE` and Google Mobility, not from observed CRE defaults or
  delinquencies. The classifier predicts a defined macro/mobility regime,
  not credit losses.
- **Sample size.** Quarterly alignment of the source series yields a small
  panel; AUC and recall numbers are illustrative and should not be read as
  out-of-sample performance estimates for any production use.
- **No causal identification.** The classifier and SHAP plots describe
  predictive associations, not causal effects of macro shocks on CRE risk.
- **Scenario forecasts are univariate.** The R companion fits independent
  `auto.arima` models per macro driver; cross-driver dependencies are not
  modeled.
- **No live data refresh.** Snapshots in `data/processed/` are committed as
  reference fixtures and must be regenerated via `make run` to refresh.
- **AI-generated commentary is a draft.** The optional Claude-API commentary
  module produces a plain-English summary of the latest run; a human
  reviewer must verify it against the source figures and tables.

## Future improvements

- Add a small evaluation harness comparing AI-generated commentary against a
  hand-graded reference set, so commentary quality is measurable over time.
- Replace the quantile-based label with a published distress indicator
  (e.g., aggregated CMBS delinquency rates from public sources).
- Add a temporal cross-validation strategy (expanding-window) so reported
  metrics reflect realistic deployment conditions.
- Add a Quarto briefing template that renders ROC, SHAP, and commentary
  into a one-page memo.
- Add a `renv.lock` for the R companion so package versions are pinned
  alongside the Python `pyproject.toml`.

## Skills demonstrated

- **Bilingual research workflows** — Python for ingestion / persistence /
  classification, R for ARIMA scenario forecasting, shared SQLite layer
  between them.
- **Production-style code organization** — installable Python package under
  `src/cre_stress/`, numbered pipeline stages, type hints, structured
  logging, `make` targets, pytest with deterministic fixtures, CI on push.
- **Data engineering** — three public sources merged through a documented
  schema; SQLAlchemy ORM persistence; idempotent re-runs.
- **Classification under class imbalance** — RandomOverSampler / SMOTE,
  explicit threshold tuning to a target FPR and recall, ROC and SHAP
  diagnostics committed to the repo for review.
- **Model documentation** — methodology, data dictionary, and a model card
  spelling out intended use, limitations, and ethical considerations.
- **Responsible LLM use** — optional Claude-API commentary is framed as a
  draft for human review, not a finding; `CLAUDE.md` codifies code-style
  and review rules for AI-generated diffs.

## Connects to research-support workflows

The pipeline is laid out the way a recurring research-support workflow would
be packaged for review: ingestion, persistence, feature construction,
modeling, evaluation, and (optional) AI-assisted commentary, with each stage
isolated, tested, and regenerable from one command. The classifier itself is
a placeholder; the value of the repository is the *scaffolding around it* —
schema discipline, threshold control, explainability, documented
limitations, and a separate channel for AI-assisted narrative that a human
reviewer must approve before sharing.

For the reusable AI-assisted review templates (prompts, sample QC reports,
human-in-the-loop checklist), see
[llm-research-workflow-assistant](https://github.com/yullieyang/llm-research-workflow-assistant).

## License

MIT — see [LICENSE](LICENSE).
