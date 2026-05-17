"""End-to-end orchestration: ingest → persist → train → evaluate.

Exposed as the ``cre-stress`` CLI via ``pyproject.toml``. Each step is
individually runnable from ``scripts/0N_*.py`` for debugging.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cre_stress import config as config_module
from cre_stress import evaluate, features, ingest, models, persist

logger = logging.getLogger(__name__)


def configure_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def run(model_type: str = "logistic", skip_zoning: bool = True) -> dict[str, Path]:
    """Execute the full pipeline and return artifact paths."""
    s = config_module.get_settings()

    logger.info("=== 1. Ingest ===")
    macro = ingest.fetch_macro(s)
    mobility = ingest.fetch_mobility(s)
    if not skip_zoning:
        ingest.fetch_zoning(s)

    logger.info("=== 2. Persist ===")
    persist.init_db(settings=s)
    persist.persist_macro(macro, settings=s)
    persist.persist_mobility(mobility, settings=s)

    logger.info("=== 3. Feature engineering ===")
    panel = features.integrate_macro_mobility(macro, mobility)
    panel = features.apply_quantile_stress_labels(panel)
    panel = features.add_label_noise(panel, "target_hard", noise_frac=0.1, random_state=s.random_state)
    X = features.select_modeling_features(panel, exclude=["target_or", "target_hard"])
    y: pd.Series = panel["target_hard"]

    logger.info("=== 4. Train ===")
    X_tr, X_te, y_tr, y_te = models.stratified_split(X, y, test_size=s.test_size, random_state=s.random_state)
    result = models.train_classifier(
        X_tr,
        X_te,
        y_tr,
        y_te,
        model_type=model_type,  # type: ignore[arg-type]
        sampling_strategy=s.sampling_strategy,
        max_fpr=s.max_fpr,
        target_recall=s.target_recall,
        random_state=s.random_state,
    )

    logger.info("=== 5. Evaluate ===")
    paths = evaluate.save_artifacts(result, X_tr, X_te, y_tr, y_te, settings=s)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cre-stress", description="Run the CRE stress test pipeline.")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--zoning", action="store_true", help="Also fetch Boston zoning data (requires CKAN id)")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    paths = run(model_type=args.model, skip_zoning=not args.zoning)
    print("\nArtifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
