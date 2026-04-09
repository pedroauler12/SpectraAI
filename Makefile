.PHONY: setup-dev test coverage run-a11 consolidate-a11-metrics

setup-dev:
	python -m pip install -U pip
	python -m pip install -r requirements-dev.txt

test:
	pytest -q src/tests

coverage:
	pytest -q --cov=src --cov-report=term-missing src/tests

# ──────────────────────────────────────────────────────────────
# A11 — Pipeline Modelo Final
# ──────────────────────────────────────────────────────────────

run-a11:
	python -m artefatos.a11_pipeline_e2e

consolidate-a11-metrics:
	python -m src.consolidate_a11_metrics
