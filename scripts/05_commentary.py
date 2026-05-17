#!/usr/bin/env python
"""Stage 5 (optional): generate a plain-English commentary on the latest run."""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cre_stress.commentary import summarize_run, write_commentary
from cre_stress.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    s = get_settings()

    pkls = sorted(s.tables_dir.glob("model_*.pkl"))
    if not pkls:
        raise SystemExit("No model artifacts found. Run scripts/03_train.py first.")
    with pkls[-1].open("rb") as f:
        bundle = pickle.load(f)
    result = bundle["result"]

    text = summarize_run(
        report=result.classification_report,
        auc=result.auc,
        threshold=result.threshold,
        top_shap_features=None,
        model_type="logistic",
        max_fpr=s.max_fpr,
        target_recall=s.target_recall,
    )
    out = s.outputs_dir / "commentary.md"
    write_commentary(text, out)
    print(f"Wrote commentary to {out}")


if __name__ == "__main__":
    main()
