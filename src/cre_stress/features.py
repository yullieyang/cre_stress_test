"""Feature engineering for the CRE stress test.

Pure functions: no I/O, no globals, no `print`. Every transform is testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressLabelConfig:
    """Configuration for the stress-condition label rules."""

    unrate_quantile: float = 0.75
    retail_mobility_quantile: float = 0.25
    label_noise_frac: float = 0.10
    random_state: int = 42


def integrate_macro_mobility(macro: pd.DataFrame, mobility: pd.DataFrame) -> pd.DataFrame:
    """Inner-join macro (wide) and mobility (wide) on date.

    Both inputs must have a ``date`` column. Mobility is aggregated to one row
    per date by averaging across sub-categories (the input is already filtered
    to a single state).
    """
    if "date" not in macro.columns or "date" not in mobility.columns:
        raise ValueError("Both inputs must have a `date` column")

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    mobility = mobility.copy()
    mobility["date"] = pd.to_datetime(mobility["date"])
    mobility_daily = (
        mobility.drop(columns=[c for c in ["country_region", "sub_region_1"] if c in mobility.columns])
        .groupby("date", as_index=False)
        .mean(numeric_only=True)
    )

    merged = macro.merge(mobility_daily, on="date", how="inner")
    return merged.sort_values("date").reset_index(drop=True)


def apply_quantile_stress_labels(
    df: pd.DataFrame,
    config: StressLabelConfig | None = None,
) -> pd.DataFrame:
    """Add two stress-condition labels.

    ``target_or``  : 1 if UNRATE in the top quartile **or** retail mobility in the bottom quartile.
    ``target_hard``: 1 if both conditions hold (stricter; positive class is rarer).
    """
    cfg = config or StressLabelConfig()
    out = df.copy()

    if "UNRATE" not in out.columns:
        raise ValueError("Expected column `UNRATE` in the integrated panel")
    retail_col = "retail_and_recreation_percent_change_from_baseline"
    if retail_col not in out.columns:
        raise ValueError(f"Expected column `{retail_col}` in the integrated panel")

    unr_q = out["UNRATE"].quantile(cfg.unrate_quantile)
    ret_q = out[retail_col].quantile(cfg.retail_mobility_quantile)

    out["target_or"] = ((out["UNRATE"] >= unr_q) | (out[retail_col] <= ret_q)).astype(int)
    out["target_hard"] = ((out["UNRATE"] >= unr_q) & (out[retail_col] <= ret_q)).astype(int)
    return out


def add_label_noise(
    df: pd.DataFrame,
    target_col: str = "target_hard",
    noise_frac: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Flip ``noise_frac`` of labels at random.

    Used to stress-test downstream classifiers by simulating mis-labeled rows.
    Returns a new DataFrame; does not mutate the input.
    """
    if not 0.0 <= noise_frac < 1.0:
        raise ValueError("noise_frac must be in [0, 1)")
    rng = np.random.default_rng(random_state)
    out = df.copy()
    flip_idx = rng.choice(out.index, size=int(len(out) * noise_frac), replace=False)
    out.loc[flip_idx, target_col] = 1 - out.loc[flip_idx, target_col]
    return out


def select_modeling_features(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Return a frame containing only numeric / boolean modeling features.

    Drops the date column and any explicitly excluded targets / identifiers.
    """
    drop = set(exclude or []) | {"date"}
    keep = [c for c in df.columns if c not in drop]
    sub = df[keep]
    numeric_bool = sub.select_dtypes(include=[np.number, "bool"]).columns
    return sub[numeric_bool]
