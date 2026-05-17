"""Streamlit dashboard for the CRE stress test pipeline.

Run with:  ``streamlit run src/dashboard/app.py``

Reads from the SQLite database populated by the Python pipeline and from the
``outputs/`` directory. Falls back to a friendly message when the pipeline has
not yet been run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the package importable when running via `streamlit run` from the repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from cre_stress.config import get_settings  # noqa: E402
from cre_stress.persist import load_macro_wide, load_mobility_long  # noqa: E402

st.set_page_config(page_title="CRE Stress Test", page_icon="🏢", layout="wide")
st.title("CRE Credit Risk — Stress Test Dashboard")
st.caption(
    "Interactive view over the same data the Python pipeline uses. "
    "Numbers below come from the SQLite database; figures from the most recent run."
)

settings = get_settings()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Run info")
    st.write(f"Database: `{settings.database_url}`")
    st.write(f"Outputs:  `{settings.outputs_dir}`")
    st.divider()
    st.markdown(
        "**Pipeline stages**\n\n"
        "1. Ingest (FRED, Mobility, Zoning)\n"
        "2. Persist (SQLite)\n"
        "3. Train (LR / RF with SMOTE)\n"
        "4. Evaluate (ROC + SHAP)\n"
        "5. (Optional) AI commentary"
    )

# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------
data_tab, model_tab = st.tabs(["Data", "Latest model run"])

with data_tab:
    try:
        macro = load_macro_wide()
        st.subheader("Macroeconomic series (FRED)")
        if macro.empty:
            st.warning("No macro data persisted yet. Run `make ingest && make persist`.")
        else:
            macro["date"] = pd.to_datetime(macro["date"])
            st.line_chart(macro.set_index("date"))

        mobility = load_mobility_long()
        st.subheader("Mobility by category (MA)")
        if mobility.empty:
            st.warning("No mobility data persisted yet.")
        else:
            mobility["date"] = pd.to_datetime(mobility["date"])
            pivoted = mobility.pivot_table(index="date", columns="category", values="percent_change", aggfunc="mean")
            st.line_chart(pivoted)
    except Exception as exc:  # pragma: no cover  — dashboard runtime errors
        st.error(f"Could not load data: {exc}")

# ---------------------------------------------------------------------------
# Latest run tab
# ---------------------------------------------------------------------------
with model_tab:
    st.subheader("Latest classification report")
    tables = sorted(settings.tables_dir.glob("classification_report_*.csv"))
    if not tables:
        st.info("No model artifacts found. Run `make run` to populate.")
    else:
        latest = tables[-1]
        st.caption(f"Source: `{latest.name}`")
        st.dataframe(pd.read_csv(latest, index_col=0))

    metrics = sorted(settings.tables_dir.glob("model_metrics_*.csv"))
    if metrics:
        st.subheader("Run metrics")
        st.dataframe(pd.read_csv(metrics[-1]))

    figures = sorted(settings.figures_dir.glob("roc_curve_*.png"))
    if figures:
        st.subheader("ROC curve")
        st.image(str(figures[-1]))

    shap_figs = sorted(settings.figures_dir.glob("shap_summary_*.png"))
    if shap_figs:
        st.subheader("SHAP summary")
        st.image(str(shap_figs[-1]))
