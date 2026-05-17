"""SQLAlchemy persistence layer.

All ingested data is normalized into long-format tables in a SQLite database
(``cre_stress.db`` by default). Downstream feature engineering reads from this
layer rather than from CSV, so the schema is the contract.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    Engine,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from cre_stress.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class MacroObservation(Base):
    """One macroeconomic observation (long format)."""

    __tablename__ = "macro_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, nullable=False)
    series_id = Column(String(50), index=True, nullable=False)
    value = Column(Float, nullable=True)
    source = Column(String(50), default="FRED")


class MobilityObservation(Base):
    """One Google Mobility observation (long format)."""

    __tablename__ = "mobility_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, nullable=False)
    region = Column(String(100), index=True, nullable=False)
    category = Column(String(80), index=True, nullable=False)
    percent_change = Column(Float, nullable=True)


class ZoningRecord(Base):
    """One Boston zoning area record."""

    __tablename__ = "zoning_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_subdistrict = Column(String(120))
    zone_district = Column(String(120))
    area_sqft = Column(Float)
    region = Column(String(80), default="Boston")


def make_engine(settings: Settings | None = None) -> Engine:
    s = settings or get_settings()
    return create_engine(s.database_url, future=True)


def init_db(engine: Engine | None = None, settings: Settings | None = None) -> None:
    """Create tables if they do not exist. Idempotent."""
    eng = engine or make_engine(settings)
    Base.metadata.create_all(eng)
    logger.info("Initialized schema in %s", eng.url)


@contextmanager
def session_scope(engine: Engine | None = None, settings: Settings | None = None) -> Iterator[Session]:
    eng = engine or make_engine(settings)
    SessionLocal = sessionmaker(bind=eng, expire_on_commit=False, future=True)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def persist_macro(macro_wide: pd.DataFrame, settings: Settings | None = None) -> int:
    """Persist a wide-format macro DataFrame as long rows.

    Expected columns: ``date`` plus one column per FRED series ID.
    Returns the number of rows inserted.
    """
    series_cols = [c for c in macro_wide.columns if c != "date"]
    long = macro_wide.melt(id_vars=["date"], value_vars=series_cols, var_name="series_id", value_name="value")
    long["date"] = pd.to_datetime(long["date"]).dt.date

    inserted = 0
    with session_scope(settings=settings) as session:
        for row in long.itertuples(index=False):
            session.add(
                MacroObservation(
                    date=row.date,
                    series_id=row.series_id,
                    value=None if pd.isna(row.value) else float(row.value),
                )
            )
            inserted += 1
    logger.info("Persisted %d macro observations", inserted)
    return inserted


def persist_mobility(mobility: pd.DataFrame, settings: Settings | None = None) -> int:
    """Persist Google Mobility records as long rows. Returns inserted count."""
    category_cols = [c for c in mobility.columns if c.endswith("_percent_change_from_baseline")]
    long = mobility.melt(
        id_vars=["date", "sub_region_1"],
        value_vars=category_cols,
        var_name="category",
        value_name="percent_change",
    ).rename(columns={"sub_region_1": "region"})
    long["category"] = long["category"].str.replace("_percent_change_from_baseline", "", regex=False)
    long["date"] = pd.to_datetime(long["date"]).dt.date

    inserted = 0
    with session_scope(settings=settings) as session:
        for row in long.itertuples(index=False):
            session.add(
                MobilityObservation(
                    date=row.date,
                    region=row.region,
                    category=row.category,
                    percent_change=None if pd.isna(row.percent_change) else float(row.percent_change),
                )
            )
            inserted += 1
    logger.info("Persisted %d mobility observations", inserted)
    return inserted


def load_macro_wide(settings: Settings | None = None) -> pd.DataFrame:
    """Read macro observations back as a wide-format DataFrame for modeling."""
    eng = make_engine(settings)
    with session_scope(eng, settings) as session:
        rows = session.execute(select(MacroObservation)).scalars().all()
    df = pd.DataFrame(
        [{"date": r.date, "series_id": r.series_id, "value": r.value} for r in rows]
    )
    if df.empty:
        return df
    return df.pivot(index="date", columns="series_id", values="value").reset_index()


def load_mobility_long(settings: Settings | None = None) -> pd.DataFrame:
    """Read mobility observations as a long DataFrame."""
    eng = make_engine(settings)
    with session_scope(eng, settings) as session:
        rows = session.execute(select(MobilityObservation)).scalars().all()
    return pd.DataFrame(
        [
            {"date": r.date, "region": r.region, "category": r.category, "percent_change": r.percent_change}
            for r in rows
        ]
    )
