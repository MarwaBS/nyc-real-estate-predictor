.PHONY: test lint typecheck ci train predict api streamlit

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	ruff check src/ api/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

ci: lint typecheck test

train:
	python -m src.models.train_classification
	python -m src.models.train_regression

predict:
	python -m src.models.predict

api:
	uvicorn api.main:app --reload --port 8000

streamlit:
	streamlit run streamlit_app/app.py --server.port 8501
