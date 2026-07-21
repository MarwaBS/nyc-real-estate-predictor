.PHONY: test lint typecheck ci train api streamlit

test:
	pytest tests/ -v --tb=short --cov=src --cov=benchmarks --cov=api --cov=run_training --cov-report=term-missing --cov-fail-under=78

# Mirrors CI's lint job exactly — a `make lint` that checks less than CI
# just moves the failure to the slower feedback loop.
lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/ api/ benchmarks/ --ignore-missing-imports

ci: lint typecheck test

# The one training entrypoint. Requires the committed raw dataset at
# Resources/NY-House-Dataset.csv; artifacts land
# in models/ with provenance in reports/training_metrics.json.
train:
	python run_training.py

api:
	uvicorn api.main:app --reload --port 8000

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501
