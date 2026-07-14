.PHONY: test lint typecheck ci train api streamlit

test:
	pytest tests/ -v --tb=short --cov=src --cov=benchmarks --cov=api --cov-report=term-missing --cov-fail-under=88

# Mirrors CI's lint job (same paths) — a `make lint` that checks fewer
# trees than CI just moves the failure to the slower feedback loop.
lint:
	ruff check src/ api/ tests/ benchmarks/ run_training.py
	ruff format --check src/ api/ tests/ benchmarks/ run_training.py

typecheck:
	mypy src/ api/ benchmarks/ --ignore-missing-imports

ci: lint typecheck test

# The one real training entrypoint. (The old target invoked
# `python -m src.models.train_classification` / `train_regression`,
# whose __main__ blocks trained nothing — runbook theater.) Requires the
# Kaggle dataset at data/raw/NY-House-Dataset.csv; artifacts land in
# models/ with provenance in reports/training_metrics.json.
train:
	python run_training.py

api:
	uvicorn api.main:app --reload --port 8000

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501
