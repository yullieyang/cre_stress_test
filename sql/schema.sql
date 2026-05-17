-- Canonical schema for the CRE stress test pipeline.
-- Source of truth: src/cre_stress/persist.py (SQLAlchemy models).
-- This file is provided so the schema is reviewable without reading Python.

CREATE TABLE IF NOT EXISTS macro_observations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       DATE        NOT NULL,
    series_id  VARCHAR(50) NOT NULL,
    value      REAL,
    source     VARCHAR(50) DEFAULT 'FRED'
);
CREATE INDEX IF NOT EXISTS idx_macro_date      ON macro_observations (date);
CREATE INDEX IF NOT EXISTS idx_macro_series    ON macro_observations (series_id);

CREATE TABLE IF NOT EXISTS mobility_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            DATE         NOT NULL,
    region          VARCHAR(100) NOT NULL,
    category        VARCHAR(80)  NOT NULL,
    percent_change  REAL
);
CREATE INDEX IF NOT EXISTS idx_mob_date        ON mobility_observations (date);
CREATE INDEX IF NOT EXISTS idx_mob_region      ON mobility_observations (region);
CREATE INDEX IF NOT EXISTS idx_mob_category    ON mobility_observations (category);

CREATE TABLE IF NOT EXISTS zoning_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_subdistrict    VARCHAR(120),
    zone_district       VARCHAR(120),
    area_sqft           REAL,
    region              VARCHAR(80) DEFAULT 'Boston'
);

-- Example queries:

-- (Q1) Quarterly average UNRATE since 2018:
-- SELECT strftime('%Y-Q%m', date) AS quarter, AVG(value) AS unrate
--   FROM macro_observations
--  WHERE series_id = 'UNRATE' AND date >= '2018-01-01'
--  GROUP BY quarter;

-- (Q2) Retail mobility z-score by month for Massachusetts:
-- SELECT date, percent_change,
--        (percent_change - AVG(percent_change) OVER ()) /
--        (NULLIF((SELECT (avg(percent_change*percent_change) - avg(percent_change)*avg(percent_change))
--                FROM mobility_observations
--               WHERE region = 'Massachusetts' AND category = 'retail_and_recreation'), 0))
--          AS z_retail
--   FROM mobility_observations
--  WHERE region = 'Massachusetts' AND category = 'retail_and_recreation';
