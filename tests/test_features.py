"""Tests for the pure-function feature engineering layer."""

from __future__ import annotations

import pandas as pd
import pytest

from cre_stress.features import (
    add_label_noise,
    apply_quantile_stress_labels,
    integrate_macro_mobility,
    select_modeling_features,
)


def test_integrate_inner_joins_on_date(macro_wide: pd.DataFrame, mobility_wide: pd.DataFrame) -> None:
    merged = integrate_macro_mobility(macro_wide, mobility_wide)
    assert len(merged) == len(macro_wide)
    assert {"FEDFUNDS", "UNRATE", "GDP"}.issubset(merged.columns)
    assert "retail_and_recreation_percent_change_from_baseline" in merged.columns


def test_integrate_drops_non_overlapping_dates() -> None:
    a = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "UNRATE": [3.5, 3.6]})
    b = pd.DataFrame(
        {
            "date": ["2020-02-01"],
            "sub_region_1": ["MA"],
            "country_region": ["United States"],
            "retail_and_recreation_percent_change_from_baseline": [-5.0],
        }
    )
    out = integrate_macro_mobility(a, b)
    assert len(out) == 1
    assert out["date"].iloc[0] == pd.Timestamp("2020-02-01")


def test_quantile_labels_are_binary(integrated_panel: pd.DataFrame) -> None:
    labeled = apply_quantile_stress_labels(integrated_panel)
    assert labeled["target_or"].isin([0, 1]).all()
    assert labeled["target_hard"].isin([0, 1]).all()


def test_target_hard_is_subset_of_target_or(integrated_panel: pd.DataFrame) -> None:
    labeled = apply_quantile_stress_labels(integrated_panel)
    # target_hard == 1 ⇒ target_or == 1 (AND ⇒ OR)
    assert ((labeled["target_hard"] == 1) <= (labeled["target_or"] == 1)).all()


def test_quantile_labels_raise_when_required_column_missing() -> None:
    bad = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "FEDFUNDS": [1, 2, 3]})
    with pytest.raises(ValueError):
        apply_quantile_stress_labels(bad)


def test_label_noise_is_deterministic_under_same_seed() -> None:
    df = pd.DataFrame({"target_hard": [0, 0, 0, 1, 1, 1, 0, 1]})
    a = add_label_noise(df, noise_frac=0.5, random_state=7)
    b = add_label_noise(df, noise_frac=0.5, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_label_noise_does_not_mutate_input() -> None:
    df = pd.DataFrame({"target_hard": [0, 1, 0, 1]})
    snapshot = df.copy()
    _ = add_label_noise(df, noise_frac=0.5, random_state=0)
    pd.testing.assert_frame_equal(df, snapshot)


def test_label_noise_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        add_label_noise(pd.DataFrame({"target_hard": [0, 1]}), noise_frac=1.0)


def test_select_modeling_features_drops_dates_and_excluded() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "UNRATE": [3.0, 3.5, 4.0],
            "GDP": [21000.0, 21100.0, 21200.0],
            "target_or": [0, 1, 0],
            "target_hard": [0, 0, 0],
        }
    )
    feats = select_modeling_features(df, exclude=["target_or", "target_hard"])
    assert "date" not in feats.columns
    assert "target_or" not in feats.columns
    assert "target_hard" not in feats.columns
    assert set(feats.columns) == {"UNRATE", "GDP"}
