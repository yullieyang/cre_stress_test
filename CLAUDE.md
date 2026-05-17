# CLAUDE.md

This file documents the project's conventions for working with [Claude Code](https://claude.com/claude-code) — the AI coding agent used to refactor and extend this codebase.

## Project context

Production-style commercial real estate credit risk modeling pipeline. Three public data sources (FRED, Google Mobility, Boston Zoning) → SQLite → binary classifier → SHAP explanations → optional LLM commentary.

## Code-style rules Claude Code must follow

- **Python:** type-hint every function signature; use `pathlib.Path` (never raw string paths); log via `logging` (never `print`) in library code; CLI scripts may print user-facing output. Target Python 3.10+.
- **R:** all paths via `here::here()`; tidyverse-first; forecast functions return tidied tibbles, not raw forecast objects.
- **SQL:** parameterized queries only; use the SQLAlchemy ORM in `src/cre_stress/persist.py`, not raw strings, when accessing application tables.
- **Tests:** every module under `src/cre_stress/` has a paired test under `tests/`. New transforms or model code without tests will not be accepted.
- **Secrets:** never read API keys from inline literals or commit `.env`. Use `src.cre_stress.config.Settings`, which loads from environment.

## Pipeline conventions

- Numbered scripts run in order: 01 → 02 → 03 → ...; each is idempotent.
- All artifacts under `outputs/figures/` and `outputs/tables/` are timestamped.
- Raw data lives under `data/raw/` and is **not** committed (regenerable by `scripts/01_ingest.py`).
- The SQLite database `cre_stress.db` is also `.gitignore`d — it is rebuilt by `scripts/02_persist.py`.

## When using Claude Code on this repo

Useful starting prompts:

- *"Add a new ingestion source under `src/cre_stress/ingest.py` for {dataset}, with a paired test in `tests/test_ingest.py`."*
- *"Refactor `train_classifier` in `src/cre_stress/models.py` to accept a sklearn estimator factory instead of a string switch."*
- *"Write a commentary section in `src/cre_stress/commentary.py` summarizing the latest classification report."*

Avoid:

- Asking Claude to add dependencies without checking `pyproject.toml`.
- Asking Claude to bypass the `Settings` config object for env vars.
- Asking Claude to modify committed CSVs in `data/processed/` — those snapshots are reference fixtures for tests.

## Review checklist for AI-generated diffs

Before merging anything Claude wrote:

1. Does every new function have a type hint and a docstring?
2. Does the change include a paired test?
3. Does `make test` pass locally?
4. Is there a `print()` call where there should be `logger.info()`?
5. Are any new secrets read directly from `os.environ` instead of `Settings`?
