"""Tests for the Settings configuration layer."""

from __future__ import annotations

import pytest

from cre_stress.config import Settings


def test_defaults_fall_back_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FRED_API_KEY", "BOSTON_ZONING_RESOURCE_ID", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.fred_api_key == ""
    assert s.database_url.startswith("sqlite:///")


def test_validate_required_raises_when_secret_missing() -> None:
    s = Settings(_env_file=None, fred_api_key="")
    with pytest.raises(RuntimeError, match="Missing required configuration"):
        s.validate_required("fred_api_key")


def test_validate_required_passes_when_present() -> None:
    s = Settings(_env_file=None, fred_api_key="abc")
    s.validate_required("fred_api_key")  # should not raise


def test_paths_derive_from_data_dir(tmp_path) -> None:
    s = Settings(_env_file=None, data_dir=tmp_path, outputs_dir=tmp_path / "outs")
    assert s.raw_data_dir == tmp_path / "raw"
    assert s.processed_data_dir == tmp_path / "processed"
    assert s.figures_dir == tmp_path / "outs" / "figures"
    assert s.tables_dir == tmp_path / "outs" / "tables"
