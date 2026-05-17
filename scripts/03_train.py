#!/usr/bin/env python
"""Stage 3: train classifier on the engineered features. Writes pickled model artifacts."""

from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cre_stress.config import get_settings
from cre_stress.features import (
    add_label_noise,
    apply_quantile_stress_labels,
    integrate_macro_mobility,
    select_modeling_features,
)
from cre_stress.models import stratified_split, train_classifier


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    s = get_settings()

    macro = pd.read_csv(s.raw_data_dir / "macro.csv", parse_dates=["date"])
    mobility = pd.read_csv(s.raw_data_dir / "mobility.csv", parse_dates=["date"])

    panel = integrate_macro_mobility(macro, mobility)
    panel = apply_quantile_stress_labels(panel)
    panel = add_label_noise(panel, "target_hard", noise_frac=0.1, random_state=s.random_state)

    X = select_modeling_features(panel, exclude=["target_or", "target_hard"])
    y = panel["target_hard"]

    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=s.test_size, random_state=s.random_state)
    result = train_classifier(
        X_tr,
        X_te,
        y_tr,
        y_te,
        model_type="logistic",
        sampling_strategy=s.sampling_strategy,
        max_fpr=s.max_fpr,
        target_recall=s.target_recall,
        random_state=s.random_state,
    )

    s.tables_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = s.tables_dir / f"model_{ts}.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"result": result, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te}, f)
    print(f"Saved trained pipeline to {model_path}")


if __name__ == "__main__":
    main()
