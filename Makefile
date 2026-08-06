.PHONY: test lint typecheck security ci train api streamlit

test:
	pytest tests/ -v --tb=short --cov=src --cov=benchmarks --cov=api --cov=run_training --cov=streamlit_app --cov-report=term-missing --cov-fail-under=85

# `ci` runs what CI's lint + test jobs run, split so each can run alone.
lint:
	ruff check .
	ruff format --check .
	codespell $$(git ls-files '*.md' '*.py' '*.yml' '*.toml' '*.txt' '*.cfg')

typecheck:
	mypy src/ api/ benchmarks/ streamlit_app/ scripts/ run_training.py --ignore-missing-imports

security:
	bandit -r src/ api/ benchmarks/ streamlit_app/ scripts/ run_training.py -n 3 -ll

ci: lint typecheck security test
	python scripts/verify_gates.py

# The one training entrypoint. Requires the committed raw dataset at
# Resources/NY-House-Dataset.csv; artifacts land
# in models/ with provenance in reports/training_metrics.json.
train:
	python run_training.py

api:
	uvicorn api.main:app --reload --port 8000

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501
