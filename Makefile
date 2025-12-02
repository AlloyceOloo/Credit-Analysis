.PHONY: venv install up down run-tests lint

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down --volumes

lint:
	flake8 src

build-features:
	python scripts/run_feature_build.py --src data/processed/customers.parquet --out data/features/features.parquet

test-features:
	pytest tests/unit/test_feature_engineering.py -q

generate-data:
	python scripts/generate_synthetic_data.py --n 10000 --out data/synthetic/customers.csv

ingest-data:
	python src/ingestion/batch_ingest.py --src data/synthetic/customers.csv

run-tests:
	pytest -q

.PHONY: train-model serve-api docker-build docker-run

train-model:
	python scripts/train_model.py --features data/features/features.parquet --out_dir models --artifacts_dir artifacts

serve-api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t creditx-api .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models creditx-api

predict:
	python scripts/run_inference.py --input_csv data/synthetic/customers.csv --out_csv artifacts/predictions.csv

dashboard:
	streamlit run src/dashboard/app.py