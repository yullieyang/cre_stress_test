#' Forecast stress-scenario macro covariates
#'
#' Reads the wide macro panel that the Python pipeline persists into the SQLite
#' database, projects each FRED series forward N quarters with `forecast::auto.arima`,
#' and writes baseline / adverse / severely-adverse scenarios to
#' `outputs/tables/scenario_forecasts.csv` for the Python pipeline to re-score.
#'
#' Adverse and severely-adverse paths are built as additive shocks to the point
#' forecast: -1 and -2 standard-deviations of in-sample residuals applied to GDP
#' and FEDFUNDS, with corresponding positive shocks to UNRATE.
#'
#' Run with:  Rscript R/forecast_stress_scenarios.R

suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(forecast)
  library(here)
  library(readr)
})

DB_PATH <- here::here("cre_stress.db")
OUT_PATH <- here::here("outputs", "tables", "scenario_forecasts.csv")
HORIZON_Q <- 8L

if (!file.exists(DB_PATH)) {
  stop("Database not found at ", DB_PATH,
       ". Run `make persist` (Python) first to populate it.")
}

# --- 1. Load wide macro panel from SQLite ----------------------------------
con <- dbConnect(SQLite(), DB_PATH)
on.exit(dbDisconnect(con), add = TRUE)

macro <- dbGetQuery(con, "SELECT date, series_id, value FROM macro_observations") %>%
  mutate(date = as.Date(date)) %>%
  pivot_wider(names_from = series_id, values_from = value) %>%
  arrange(date)

if (nrow(macro) == 0) {
  stop("macro_observations table is empty. Run the Python persist stage.")
}

# --- 2. Quarterly aggregation ----------------------------------------------
quarterly <- macro %>%
  mutate(quarter = floor_date(date, "quarter")) %>%
  group_by(quarter) %>%
  summarize(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  filter(!is.na(quarter))

series_cols <- setdiff(names(quarterly), "quarter")
message("Forecasting series: ", paste(series_cols, collapse = ", "))

# --- 3. Fit auto.arima per series, build scenario paths ---------------------
forecast_one <- function(series_name) {
  vec <- quarterly[[series_name]]
  vec <- vec[!is.na(vec)]
  if (length(vec) < 8) {
    warning("Series ", series_name, " has fewer than 8 obs; skipping.")
    return(NULL)
  }
  ts_q <- ts(vec, frequency = 4)
  fit <- forecast::auto.arima(ts_q, seasonal = FALSE)
  fc <- forecast(fit, h = HORIZON_Q)
  resid_sd <- sd(residuals(fit), na.rm = TRUE)

  baseline <- as.numeric(fc$mean)
  adverse <- baseline + sign_for_stress(series_name) * 1 * resid_sd
  severe <- baseline + sign_for_stress(series_name) * 2 * resid_sd

  tibble(
    series_id = series_name,
    horizon_q = seq_len(HORIZON_Q),
    baseline,
    adverse,
    severely_adverse = severe
  )
}

# Higher UNRATE is "worse"; lower GDP and lower FEDFUNDS shocks too.
sign_for_stress <- function(series_name) {
  if (series_name == "UNRATE") 1 else -1
}

forecasts <- bind_rows(lapply(series_cols, forecast_one))

# --- 4. Write to outputs/tables/ -------------------------------------------
dir.create(dirname(OUT_PATH), showWarnings = FALSE, recursive = TRUE)
write_csv(forecasts, OUT_PATH)
message("Wrote ", nrow(forecasts), " forecast rows to ", OUT_PATH)
