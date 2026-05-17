.PHONY: install install-dev test lint format ingest persist train evaluate run commentary dashboard clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[all]"

test:
	pytest tests/

lint:
	ruff check src tests

format:
	ruff format src tests

ingest:
	$(PYTHON) scripts/01_ingest.py

persist:
	$(PYTHON) scripts/02_persist.py

train:
	$(PYTHON) scripts/03_train.py

evaluate:
	$(PYTHON) scripts/04_evaluate.py

run: ingest persist train evaluate

commentary:
	$(PYTHON) scripts/05_commentary.py

dashboard:
	streamlit run src/dashboard/app.py

# R companion: project stress-scenario macro inputs forward
forecast-scenarios:
	Rscript R/forecast_stress_scenarios.R

clean:
	rm -rf data/raw/*.csv data/processed/*.csv cre_stress.db outputs/figures/*.png outputs/tables/*.csv
	@echo "Cleaned ephemeral artifacts. Committed reference snapshots preserved."
