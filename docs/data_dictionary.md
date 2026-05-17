# Data dictionary

All persisted columns, types, sources, and transforms.

## `macro_observations` (long format)

| Column | Type | Source | Notes |
|---|---|---|---|
| `id` | INTEGER | (generated) | Surrogate primary key |
| `date` | DATE | FRED | Observation date as published |
| `series_id` | VARCHAR(50) | FRED | Series ID (`FEDFUNDS`, `UNRATE`, `GDP`, …) |
| `value` | REAL | FRED | Numeric; nullable (some FRED rows are `"."`) |
| `source` | VARCHAR(50) | constant | Default `'FRED'` |

### Series covered

| `series_id` | Description | Native frequency |
|---|---|---|
| `FEDFUNDS` | Effective Federal Funds Rate | Monthly |
| `UNRATE` | Civilian Unemployment Rate | Monthly |
| `GDP` | Gross Domestic Product (nominal) | Quarterly |

## `mobility_observations` (long format)

| Column | Type | Source | Notes |
|---|---|---|---|
| `id` | INTEGER | (generated) | |
| `date` | DATE | Google Mobility | Daily |
| `region` | VARCHAR(100) | Google Mobility | Currently `'Massachusetts'` only |
| `category` | VARCHAR(80) | Google Mobility | One of `retail_and_recreation`, `grocery_and_pharmacy`, `parks`, `transit_stations`, `workplaces`, `residential` |
| `percent_change` | REAL | Google Mobility | % change from a 2020 baseline window |

## `zoning_records`

| Column | Type | Source | Notes |
|---|---|---|---|
| `id` | INTEGER | (generated) | |
| `zone_subdistrict` | VARCHAR(120) | Boston CKAN | Sub-district code |
| `zone_district` | VARCHAR(120) | Boston CKAN | District code |
| `area_sqft` | REAL | Boston CKAN | Polygon area, computed |
| `region` | VARCHAR(80) | constant | Default `'Boston'` |

## Derived columns (modeling panel)

These exist in memory after `features.apply_quantile_stress_labels()`; they are not persisted.

| Column | Definition |
|---|---|
| `target_or` | `1` if `UNRATE ≥ Q75(UNRATE)` OR `retail_mobility ≤ Q25(retail_mobility)`; else `0` |
| `target_hard` | `1` if both conditions hold; else `0`. **Default modeling target.** |

## Provenance

| Field | Authority |
|---|---|
| FRED API | <https://fred.stlouisfed.org/docs/api/fred/> |
| Google Mobility (archived) | <https://www.google.com/covid19/mobility/> |
| Boston Open Data (CKAN) | <https://data.boston.gov/> |
