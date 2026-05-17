#!/usr/bin/env python
"""Stage 4: load the latest trained pipeline and write ROC / SHAP / report artifacts."""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cre_stress.config import get_settings
from cre_stress.evaluate import save_artifacts


def latest_model_path(tables_dir: Path) -> Path:
    candidates = sorted(tables_dir.glob("model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pkl found in {tables_dir}; run scripts/03_train.py first.")
    return candidates[-1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    s = get_settings()

    model_path = latest_model_path(s.tables_dir)
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    paths = save_artifacts(
        bundle["result"],
        bundle["X_tr"],
        bundle["X_te"],
        bundle["y_tr"],
        bundle["y_te"],
        settings=s,
    )
    print("Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
