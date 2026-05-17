"""Tests for the SQLAlchemy persistence layer (uses an in-memory SQLite)."""

from __future__ import annotations

import pandas as pd

from cre_stress.config import Settings
from cre_stress.persist import (
    init_db,
    load_macro_wide,
    make_engine,
    persist_macro,
    persist_mobility,
)


def _in_memory_settings(tmp_path) -> Settings:
    db_path = tmp_path / "test.db"
    return Settings(_env_file=None, database_url=f"sqlite:///{db_path}")


def test_persist_macro_roundtrip(tmp_path, macro_wide: pd.DataFrame) -> None:
    s = _in_memory_settings(tmp_path)
    init_db(make_engine(s), settings=s)

    inserted = persist_macro(macro_wide, settings=s)
    assert inserted == len(macro_wide) * 3  # 3 series

    out = load_macro_wide(settings=s)
    assert {"FEDFUNDS", "UNRATE", "GDP"}.issubset(out.columns)
    assert len(out) == len(macro_wide)


def test_persist_mobility_long_format(tmp_path, mobility_wide: pd.DataFrame) -> None:
    s = _in_memory_settings(tmp_path)
    init_db(make_engine(s), settings=s)

    inserted = persist_mobility(mobility_wide, settings=s)
    # 6 dates × 6 category columns
    assert inserted == 6 * 6
