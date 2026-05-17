#!/usr/bin/env python
"""Stage 2: load raw CSVs into the SQLite database."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cre_stress.config import get_settings
from cre_stress.persist import init_db, persist_macro, persist_mobility


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    s = get_settings()
    init_db(settings=s)

    macro = pd.read_csv(s.raw_data_dir / "macro.csv", parse_dates=["date"])
    persist_macro(macro, settings=s)

    mobility = pd.read_csv(s.raw_data_dir / "mobility.csv", parse_dates=["date"])
    persist_mobility(mobility, settings=s)


if __name__ == "__main__":
    main()
