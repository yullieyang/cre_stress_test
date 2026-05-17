"""Evaluation: ROC, SHAP, and saved-result artifacts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from cre_stress.config import Settings, get_settings
from cre_stress.models import TrainResult

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_roc(
    y_true: pd.Series,
    y_score: np.ndarray,
    output_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Render and save the ROC curve. Returns the PNG path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or _timestamp()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png = output_dir / f"roc_curve_{ts}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve to %s", png)
    return png


def plot_shap_summary(
    pipeline: ImbPipeline,
    X_te: pd.DataFrame,
    output_dir: Path,
    timestamp: str | None = None,
) -> Path | None:
    """Render and save a SHAP summary plot. Returns the PNG path (or None on failure)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or _timestamp()

    final_step = pipeline.steps[-1][1]
    try:
        if hasattr(final_step, "coef_"):
            explainer = shap.LinearExplainer(final_step, X_te, feature_perturbation="interventional")
        else:
            explainer = shap.TreeExplainer(final_step)
        shap_values = explainer.shap_values(X_te)
    except Exception as exc:  # SHAP is finicky on some sklearn versions
        logger.warning("Could not compute SHAP values: %s", exc)
        return None

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_te, show=False)
    png = output_dir / f"shap_summary_{ts}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary to %s", png)
    return png


def save_artifacts(
    result: TrainResult,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Write ROC, SHAP, classification report, and metrics summary. Returns paths."""
    s = settings or get_settings()
    figures_dir = s.figures_dir
    tables_dir = s.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    paths: dict[str, Path] = {}

    paths["roc"] = plot_roc(y_te, result.y_score, figures_dir, ts)
    shap_path = plot_shap_summary(result.pipeline, X_te, figures_dir, ts)
    if shap_path is not None:
        paths["shap"] = shap_path

    report_df = pd.DataFrame(result.classification_report).transpose()
    paths["report"] = tables_dir / f"classification_report_{ts}.csv"
    report_df.to_csv(paths["report"])

    metrics = {
        "timestamp": ts,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "auc": result.auc,
        "threshold": result.threshold,
        "train_accuracy": float(result.pipeline.score(X_tr, y_tr)),
        "test_accuracy": float(result.pipeline.score(X_te, y_te)),
    }
    paths["metrics"] = tables_dir / f"model_metrics_{ts}.csv"
    pd.DataFrame([metrics]).to_csv(paths["metrics"], index=False)
    logger.info("Saved %d artifacts to %s and %s", len(paths), figures_dir, tables_dir)
    return paths
