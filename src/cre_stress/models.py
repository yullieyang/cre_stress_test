"""Classification model training with class-imbalance handling and threshold tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ModelType = Literal["logistic", "random_forest"]


@dataclass
class TrainResult:
    """Bundle of artifacts produced by ``train_classifier``."""

    pipeline: ImbPipeline
    y_pred: np.ndarray
    y_score: np.ndarray
    threshold: float
    auc: float
    classification_report: dict


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    max_trials: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified-ish split that retries until both classes appear in train and test.

    For very small / very imbalanced samples a single ``train_test_split`` can
    leave one class out of a fold. This wrapper retries with bumped seeds.
    """
    for trial in range(max_trials):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state + trial, stratify=None
        )
        if y_tr.nunique() == 2 and y_te.nunique() == 2:
            logger.info(
                "Split succeeded on trial %d: train=%s, test=%s",
                trial,
                y_tr.value_counts().to_dict(),
                y_te.value_counts().to_dict(),
            )
            return X_tr, X_te, y_tr, y_te
    raise RuntimeError("Failed to produce a two-class train/test split after retries")


def _build_pipeline(
    model_type: ModelType,
    sampling_strategy: float,
    random_state: int,
) -> ImbPipeline:
    if model_type == "logistic":
        clf = LogisticRegression(penalty="l2", C=0.5, max_iter=1000, random_state=random_state, n_jobs=1)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=random_state, n_jobs=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return ImbPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("ros", RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)),
            ("clf", clf),
        ]
    )


def select_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    max_fpr: float = 0.2,
    target_recall: float = 0.8,
) -> float:
    """Pick the lowest threshold satisfying (FPR ≤ max_fpr) AND (TPR ≥ target_recall).

    Falls back to the threshold whose TPR is closest to ``target_recall`` if no
    threshold satisfies both constraints.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    feasible = [
        (t, fp, rc)
        for t, fp, rc in zip(thresholds, fpr, tpr, strict=False)
        if fp <= max_fpr and rc >= target_recall
    ]
    if feasible:
        return float(feasible[0][0])
    idx = int(np.argmin(np.abs(tpr - target_recall)))
    return float(thresholds[idx])


def train_classifier(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    model_type: ModelType = "logistic",
    sampling_strategy: float = 0.2,
    max_fpr: float = 0.2,
    target_recall: float = 0.8,
    random_state: int = 42,
) -> TrainResult:
    """Fit a classification pipeline and pick a threshold with explicit FPR / recall constraints."""
    pipeline = _build_pipeline(model_type, sampling_strategy, random_state)
    pipeline.fit(X_tr, y_tr)

    y_score = pipeline.predict_proba(X_te)[:, 1]
    threshold = select_threshold(y_te, y_score, max_fpr=max_fpr, target_recall=target_recall)
    y_pred = (y_score >= threshold).astype(int)

    auc = float(roc_auc_score(y_te, y_score))
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    logger.info("Trained %s; threshold=%.3f; AUC=%.3f", model_type, threshold, auc)
    return TrainResult(
        pipeline=pipeline,
        y_pred=np.asarray(y_pred),
        y_score=np.asarray(y_score),
        threshold=threshold,
        auc=auc,
        classification_report=report,
    )
