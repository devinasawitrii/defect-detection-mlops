.PHONY: install data train sweep best evaluate export visualize reference examples repro test lint serve docker-build docker-run mlflow-ui drift-demo

install:
	pip install -r requirements.txt -r requirements-dev.txt

data:
	python -m src.prepare_data

train:
	python -m src.train

sweep:
	python -m src.sweep --promote

# Unattended full run: sweep (cheap variants) -> pick winner on VALIDATION -> refit on train+val -> test ONCE
# -> export + figures. Override the variants with: make best SWEEP="n_640 n_960"
SWEEP ?= n_640 n_640_seed2 n_640_scale n_960
best:
	python -m src.sweep --only $(SWEEP) --promote
	python -m src.refit
	python -m src.evaluate
	python -m src.export_onnx
	python -m src.visualize
	python -m src.make_reference
	python -m src.make_examples

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
