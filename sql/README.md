# SQL schema

The pipeline persists all ingested data into a local SQLite database (default: `cre_stress.db`). The schema is canonically defined in `src/cre_stress/persist.py` via SQLAlchemy; `schema.sql` is a reviewable mirror of the same tables for engineers who prefer to read SQL.

## Tables

| Table | Purpose |
|---|---|
| `macro_observations` | Long-format FRED series — one row per (date, series_id, value) |
| `mobility_observations` | Long-format Google Mobility — one row per (date, region, category, percent_change) |
| `zoning_records` | Flattened Boston public zoning properties |

## Why long format?

Long-format storage makes it trivial to add a new FRED series (just insert rows — no `ALTER TABLE`) and lets the R and Python sides share a single schema. The wide-format DataFrame used by the modeling code is materialized on read via `load_macro_wide()`.

## Connection

```python
from cre_stress.config import get_settings
from cre_stress.persist import make_engine

eng = make_engine(get_settings())
# eng is a SQLAlchemy 2.0 Engine — use sessions via `session_scope()`
```

## Manual inspection

```bash
sqlite3 cre_stress.db
sqlite> .schema
sqlite> SELECT series_id, COUNT(*), MIN(date), MAX(date)
   ...> FROM macro_observations GROUP BY series_id;
```
