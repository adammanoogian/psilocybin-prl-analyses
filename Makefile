.PHONY: install install-dev test lint format demo design clean help

PYTHON ?= python
CONDA_ENV ?= ds_env

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	$(PYTHON) -m pip install -e .

install-dev:  ## Install with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run unit tests (fast, local-safe)
	$(PYTHON) -m pytest tests/unit/ -x -q

test-all:  ## Run full test suite incl. integration + scientific (route to cluster)
	$(PYTHON) -m pytest tests/ -x -q

lint:  ## Run ruff linter + mypy type checks
	$(PYTHON) -m ruff check src/ scripts/ tests/
	$(PYTHON) -m mypy src/prl_hgf/ --ignore-missing-imports

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format src/ scripts/ tests/
	$(PYTHON) -m ruff check --fix src/ scripts/ tests/

demo:  ## Run quickstart demo (simulate + fit + recover, ~2 min on CPU)
	$(PYTHON) scripts/demo_quickstart.py

design:  ## Run experiment design helper (power analysis, ~5 min quick mode)
	$(PYTHON) scripts/design_experiment.py --quick

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
