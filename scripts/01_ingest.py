#!/usr/bin/env python
"""Stage 1: pull raw FRED + Mobility data into ``data/raw/``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cre_stress.config import get_settings
from cre_stress.ingest import fetch_macro, fetch_mobility, fetch_zoning


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    s = get_settings()
    fetch_macro(s)
    fetch_mobility(s)
    if s.boston_zoning_resource_id:
        fetch_zoning(s)


if __name__ == "__main__":
    main()
