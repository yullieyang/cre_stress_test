"""Centralized, type-checked configuration.

Loads values from the environment (and `.env`) via Pydantic Settings.
Library and script code MUST go through `get_settings()` rather than reading
`os.environ` directly — this is the only safe path for secrets in tests, CI,
and production.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide configuration.

    Reads from the process environment first, then falls back to a project-level
    `.env` file. Sensitive values default to empty strings rather than `None`
    so consumers can fail loudly with `validate_required(...)` when the secret
    is actually needed (vs. on import).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets ---
    fred_api_key: str = Field(default="", description="FRED API key")
    boston_zoning_resource_id: str = Field(default="", description="CKAN resource UUID")
    anthropic_api_key: str = Field(default="", description="Anthropic API key for commentary")

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"))
    outputs_dir: Path = Field(default=Path("outputs"))
    database_url: str = Field(default="sqlite:///cre_stress.db")

    # --- Modeling defaults ---
    random_state: int = 42
    test_size: float = 0.3
    max_fpr: float = 0.2
    target_recall: float = 0.8
    sampling_strategy: float = 0.2

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    def validate_required(self, *fields: str) -> None:
        """Raise if any named field is empty. Use at the entry point of code that needs it."""
        missing = [f for f in fields if not getattr(self, f, None)]
        if missing:
            raise RuntimeError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Set these in your environment or `.env` file."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Clear with `get_settings.cache_clear()` in tests."""
    return Settings()
