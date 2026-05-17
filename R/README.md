# R companion

The Python pipeline does ingestion, persistence, feature engineering, and binary classification. The R companion handles the **time-series forecasting side** — projecting the macro drivers (`UNRATE`, `FEDFUNDS`, `GDP`) forward to define stress scenarios that the classifier can re-score.

This split mirrors how policy-research teams typically divide work: R for time-series / forecasting, Python for ML / engineering, both reading the same shared data layer.

## Why a separate R script

- `forecast::auto.arima` is the de-facto baseline for univariate macro forecasting in econ research; the R ecosystem (`forecast`, `fable`, `tsibble`) is much richer for time-series than Python's.
- The two languages share state through the **SQLite database** that Python's persistence layer writes. R reads from the same `macro_observations` table.

## How to run

```bash
# 1. Make sure Python has populated the DB
make persist

# 2. Run the R forecaster
make forecast-scenarios
# or, equivalently:
Rscript R/forecast_stress_scenarios.R
```

Output: `outputs/tables/scenario_forecasts.csv` with columns `series_id`, `horizon_q`, `baseline`, `adverse`, `severely_adverse`.

## Required R packages

```r
install.packages(c("DBI", "RSQLite", "dplyr", "tidyr", "lubridate",
                   "forecast", "here", "readr"))
```

## Scenario design

For each FRED series the script fits `auto.arima` and then constructs:

- **Baseline**   = point forecast of `auto.arima`
- **Adverse**    = baseline shifted by ±1 in-sample residual std-dev (sign depending on whether higher or lower is "worse" for credit risk — e.g. higher `UNRATE`, lower `GDP`)
- **Severely-adverse** = ±2 residual std-dev shift

This is a deliberately simple parametric stress design; for production policy work a richer specification (VAR, BVAR, conditional forecasts) would be appropriate.
