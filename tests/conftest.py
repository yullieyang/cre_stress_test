"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable without installing the package (handy in CI before `pip install -e .`).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def macro_wide() -> pd.DataFrame:
    """Tiny 6-row wide-format macro panel."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "FEDFUNDS": [1.55, 1.58, 0.65, 0.05, 0.05, 0.08],
            "UNRATE": [3.6, 3.5, 4.4, 14.7, 13.3, 11.1],
            "GDP": [21747.4, 21747.4, 19636.7, 19636.7, 20566.0, 20566.0],
        }
    )


@pytest.fixture
def mobility_wide() -> pd.DataFrame:
    """Tiny mobility panel (wide, single state)."""
    return pd.DataFrame(
        {
            "country_region": ["United States"] * 6,
            "sub_region_1": ["Massachusetts"] * 6,
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "retail_and_recreation_percent_change_from_baseline": [2.0, 1.0, -30.0, -50.0, -25.0, -10.0],
            "grocery_and_pharmacy_percent_change_from_baseline": [1.0, 0.0, -5.0, -10.0, 0.0, 5.0],
            "parks_percent_change_from_baseline": [0.0, 5.0, -20.0, -30.0, -10.0, 5.0],
            "transit_stations_percent_change_from_baseline": [3.0, 0.0, -40.0, -60.0, -35.0, -15.0],
            "workplaces_percent_change_from_baseline": [0.0, 0.0, -50.0, -60.0, -40.0, -20.0],
            "residential_percent_change_from_baseline": [0.0, 1.0, 15.0, 20.0, 12.0, 5.0],
        }
    )


@pytest.fixture
def integrated_panel(macro_wide: pd.DataFrame, mobility_wide: pd.DataFrame) -> pd.DataFrame:
    """A small joined panel suitable for feature / model tests."""
    from cre_stress.features import integrate_macro_mobility

    return integrate_macro_mobility(macro_wide, mobility_wide)
