.PHONY: install data train sweep evaluate export visualize reference examples repro test lint serve docker-build docker-run mlflow-ui drift-demo

install:
	pip install -r requirements.txt -r requirements-dev.txt

data:
	python -m src.prepare_data

train:
	python -m src.train

sweep:
	python -m src.sweep --promote

evaluate:
	python -m src.evaluate

export:
	python -m src.export_onnx

visualize:
	python -m src.visualize

reference:
	python -m src.make_reference

examples:
	python -m src.make_examples

repro:
	dvc repro

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t defect-api:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/logs:/app/logs defect-api:latest

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

drift-demo:
	python -m src.simulate_traffic --mode normal --n 200 --reset
	python -m src.monitor
	python -m src.simulate_traffic --mode drifted --n 200 --reset
	python -m src.monitor
