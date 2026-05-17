"""Data ingestion from public sources.

All raw fetches write to ``settings.raw_data_dir``. Each ingestion function is
idempotent: the same call on the same date produces the same file.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from cre_stress.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_FRED_SERIES: tuple[str, ...] = ("FEDFUNDS", "UNRATE", "GDP")

GOOGLE_MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
GOOGLE_MOBILITY_COLS = [
    "country_region",
    "sub_region_1",
    "date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

BOSTON_CKAN_RESOURCE_SHOW = "https://data.boston.gov/api/3/action/resource_show"


def fetch_macro(
    settings: Settings | None = None,
    series_ids: Iterable[str] = DEFAULT_FRED_SERIES,
) -> pd.DataFrame:
    """Fetch macro time series from FRED and write to ``data/raw/macro.csv``.

    Returns the merged wide-format DataFrame with one row per ``date``.
    Raises ``RuntimeError`` if ``FRED_API_KEY`` is not configured.
    """
    s = settings or get_settings()
    s.validate_required("fred_api_key")

    out_path = s.raw_data_dir / "macro.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={sid}&api_key={s.fred_api_key}&file_type=json"
        )
        logger.info("Fetching FRED series %s", sid)
        observations = requests.get(url, timeout=30).json().get("observations", [])
        if not observations:
            raise RuntimeError(f"FRED returned no observations for series {sid}")
        df = pd.DataFrame(observations)[["date", "value"]].rename(columns={"value": sid})
        df["date"] = pd.to_datetime(df["date"])
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        frames.append(df)

    macro = frames[0]
    for df in frames[1:]:
        macro = macro.merge(df, on="date", how="outer")
    macro = macro.sort_values("date").reset_index(drop=True)
    macro.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(macro), out_path)
    return macro


def fetch_mobility(
    settings: Settings | None = None,
    region: str = "Massachusetts",
) -> pd.DataFrame:
    """Fetch Google Mobility data filtered to ``region`` (US state).

    Writes to ``data/raw/mobility.csv``. Source is a snapshot of the COVID-19
    Community Mobility Reports (Google, sunset 2022) — the file is large but
    historical and stable.
    """
    s = settings or get_settings()
    out_path = s.raw_data_dir / "mobility.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Google Mobility data (filtered to %s)", region)
    df = pd.read_csv(GOOGLE_MOBILITY_URL, usecols=GOOGLE_MOBILITY_COLS, parse_dates=["date"])
    df = df[(df["country_region"] == "United States") & (df["sub_region_1"] == region)]
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return df


def fetch_zoning(settings: Settings | None = None) -> pd.DataFrame:
    """Fetch Boston public zoning data via the CKAN resource_show endpoint.

    Writes to ``data/raw/zoning.csv``. Returns the raw property attributes
    flattened from the GeoJSON features.
    """
    s = settings or get_settings()
    s.validate_required("boston_zoning_resource_id")

    out_path = s.raw_data_dir / "zoning.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Resolving Boston zoning CKAN resource %s", s.boston_zoning_resource_id)
    resource_meta = requests.get(
        BOSTON_CKAN_RESOURCE_SHOW,
        params={"id": s.boston_zoning_resource_id},
        timeout=30,
    ).json()
    signed_url = resource_meta["result"]["url"]

    logger.info("Downloading zoning GeoJSON")
    geojson = requests.get(signed_url, timeout=60).json()
    props = [feature["properties"] for feature in geojson.get("features", [])]
    df = pd.DataFrame(props)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d zoning records to %s", len(df), out_path)
    return df


def load_processed_snapshot(name: str, settings: Settings | None = None) -> pd.DataFrame:
    """Load a committed reference CSV from ``data/processed/``.

    Used by tests and the dashboard when live data fetches are not desired.
    """
    s = settings or get_settings()
    path: Path = s.processed_data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"No processed snapshot at {path}")
    return pd.read_csv(path, parse_dates=["date"] if "date" in path.read_text(encoding="utf-8")[:200] else None)
