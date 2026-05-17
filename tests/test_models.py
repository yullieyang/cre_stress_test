"""Smoke tests for the modeling layer.

These do not assert specific metric values (random data is too small for that);
they verify that the training pipeline runs end-to-end and that the threshold
selector respects its constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cre_stress.models import select_threshold, stratified_split, train_classifier


@pytest.fixture
def small_classification_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic imbalanced synthetic data: ~5-7% positives, signal in feat_a.

    The hard-coded positive count keeps the test stable across rng draws and
    leaves enough headroom for ``RandomOverSampler(sampling_strategy=0.2)`` on
    a 70/30 split.
    """
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "feat_c": rng.normal(size=n),
        }
    )
    # Force a deterministic ~6% positive rate by ranking on a noisy logit signal.
    logits = 1.5 * X["feat_a"] - 0.5 * X["feat_b"] + rng.normal(0, 0.5, size=n)
    n_positive = int(0.06 * n)  # exactly 30 positives
    threshold = np.partition(logits, -n_positive)[-n_positive]
    y = pd.Series((logits >= threshold).astype(int), name="target")
    return X, y


def test_stratified_split_keeps_both_classes(small_classification_dataset) -> None:
    X, y = small_classification_dataset
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.25, random_state=1)
    assert y_tr.nunique() == 2
    assert y_te.nunique() == 2


def test_train_classifier_returns_valid_result(small_classification_dataset) -> None:
    X, y = small_classification_dataset
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.3, random_state=1)
    result = train_classifier(X_tr, X_te, y_tr, y_te, model_type="logistic")
    assert 0.0 <= result.auc <= 1.0
    assert result.y_pred.shape == (len(X_te),)
    assert result.y_score.shape == (len(X_te),)
    assert set(np.unique(result.y_pred)).issubset({0, 1})


def test_select_threshold_satisfies_constraints_when_possible() -> None:
    """With a clearly separable score, the threshold should land in the empty gap."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = pd.Series(rng.integers(0, 2, size=n))
    # Wide separation gap (negatives in [0, 0.3], positives in [0.7, 1.0]).
    y_score = np.where(y_true == 1, rng.uniform(0.7, 1.0, size=n), rng.uniform(0.0, 0.3, size=n))
    threshold = select_threshold(y_true, y_score, max_fpr=0.1, target_recall=0.9)
    # Allow a small numerical tolerance at the gap boundaries.
    assert 0.3 <= threshold <= 0.71
