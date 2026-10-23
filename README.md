# Steel Surface Defect Detection: YOLO26 + MLOps

End-to-end computer vision project: fine-tune **YOLO26** to **detect and localise 6 types of hot-rolled steel
surface defects** (NEU-DET), then ship it the way a production team would: reproducible pipeline, experiment
tracking, tested code, containerised API, CI/CD, and drift monitoring.

> **Live demo:** _add your Hugging Face Space link here_ &nbsp;|&nbsp; **API image:** `ghcr.io/<you>/defect-detection-mlops/defect-api`

## Why this project

Manual visual inspection on a steel line is slow and inconsistent. A detector that says *what* defect is present
and *where* lets the line react early (and handles strips with several defects). The model is intentionally the
smaller half of this repo; the point is the full lifecycle around it.

## Architecture

```mermaid
flowchart LR
    A[NEU-DET<br/>images + VOC XML<br/>DVC-tracked] --> B[prepare_data<br/>VOC→YOLO · dedupe · stratified split]
    B --> C[train<br/>YOLO26 transfer learning<br/>from COCO weights]
    C -->|params, metrics, artifacts| M[(MLflow)]
    C --> D[evaluate<br/>mAP@50 / mAP@50-95 on test]
    D --> G{Quality gate<br/>mAP@50 ≥ 0.65}
    G --> E[Export ONNX<br/>NMS-free · parity check]
    E --> F[FastAPI + ONNX Runtime<br/>Docker image]
    E --> H[Gradio demo<br/>Hugging Face Space]
    F -->|prediction log| L[(logs/predictions.jsonl)]
    L --> K[Drift monitor<br/>KS test vs validation reference]
```

## Results

> Fill this table in after your training run (`metrics/eval.json` and `reports/` are produced by the pipeline).

| Metric (held-out test set) | Value |
|---|---|
| mAP@50 (PyTorch) | _TBD_ |
| mAP@50 (exported ONNX, `metrics/export.json`) | _TBD_ |
| mAP@50-95 | _TBD_ |
| Precision / Recall | _TBD_ / _TBD_ |
| ONNX model size | _TBD_ MB |
| CPU latency / image | _TBD_ ms |

Per-class AP: `reports/detection_report.txt` · Confusion matrix and PR curves: `reports/` ·
Predictions from the **exported ONNX model** vs ground truth: `reports/predictions_grid.png`

## What's inside

| Area | Tooling | Where |
|---|---|---|
| Data versioning & pipeline | DVC (`dvc repro`) | `dvc.yaml`, `params.yaml` |
| Data hygiene | Pascal VOC → YOLO conversion, box clipping, MD5 de-duplication, stratified split (no leakage) | `src/prepare_data.py` |
| Modelling | Ultralytics YOLO26, COCO-pretrained transfer learning, AdamW, early stopping | `src/train.py` |
| Experiment tracking | MLflow (params, per-epoch metrics, git SHA, checkpoint artifact) | `make mlflow-ui` |
| Model selection | Config sweep (model size, resolution) ranked on the **validation** split; test set used once for the winner | `src/sweep.py`, `make sweep` |
| Evaluation | mAP@50, mAP@50-95, per-class AP, confusion matrix, PR curves | `src/evaluate.py` |
| Serving | NMS-free ONNX export with parity check, ONNX Runtime, FastAPI (`/predict`, `/predict/image`, `/health`, `/stats`) | `src/export_onnx.py`, `serving/` |
| Packaging | Slim Docker image (no PyTorch / Ultralytics), non-root user, healthcheck | `Dockerfile` |
| CI | Ruff + pytest on synthetic data (trains a tiny YOLO26, no downloads) + model quality gate | `.github/workflows/ci.yml` |
| CD | Build, container smoke test, push to GHCR, deploy demo to Hugging Face Spaces | `.github/workflows/cd.yml` |
| Monitoring | Every prediction logged; KS-test drift on brightness, contrast, detections/image, top confidence | `src/monitor.py`, `src/simulate_traffic.py` |

## Quickstart

```bash
git clone https://github.com/<you>/defect-detection-mlops && cd defect-detection-mlops
make install

# 1. Put NEU-DET (images + .xml annotations) under data/raw/  (see notebooks/colab_train.ipynb)
make data          # VOC -> YOLO format, train/val/test split
make train         # fine-tune YOLO26, logs to MLflow
make sweep         # optional: compare variants on validation, promote the winner to models/best.pt
make evaluate      # metrics/eval.json (mAP) + plots
make export        # models/model.onnx (+ parity check)
make visualize     # reports/predictions_grid.png
make reference     # drift reference stats
make examples      # sample images for the demo

make mlflow-ui     # http://localhost:5000
make serve         # http://localhost:8000/docs
```

No GPU? Train on Colab/Kaggle with `notebooks/colab_train.ipynb`, download the artifacts, commit them.

### Call the API

```bash
curl -F "file=@examples/scratches.jpg" "http://localhost:8000/predict?conf=0.25"
curl -F "file=@examples/scratches.jpg" http://localhost:8000/predict/image -o annotated.png
```

```json
{"detections": [{"label": "scratches", "confidence": 0.91, "box": [12.0, 40.5, 118.2, 96.0]}],
 "n_detections": 1, "image_size": [200, 200], "latency_ms": 38.2}
```

### Docker

```bash
make docker-build && make docker-run
```

### Drift monitoring demo

```bash
make drift-demo
```

Logs 200 normal predictions (monitor says `ok`), then 200 with simulated lighting change + sensor noise
(monitor reports `drift` and exits non-zero, so it can gate a scheduled job or trigger retraining).

### Reproduce the whole pipeline

```bash
dvc repro
```

## Tests

```bash
make test
```

The suite generates a synthetic NEU-DET-style dataset (images + VOC XML) and exercises the real path end to end:
VOC conversion → train a tiny YOLO26 → evaluate → ONNX export → API → drift detection. It also asserts that the
serving letterbox is **pixel-identical** to the one Ultralytics applies at train time, a classic source of silent
train/serve skew.

## Design decisions & trade-offs

- **Detection, not just classification.** Real lines have several defects per strip and care where they are.
  YOLO is still a CNN under the hood, and it is fine-tuned from COCO weights (transfer learning).
- **NMS-free head, used consistently.** YOLO26 can export a head whose outputs are already final detections.
  Training-time model selection, evaluation, parity checks and export all use that same head (`nms=False`), so the
  reported mAP describes the model that is actually deployed. The export stage also re-evaluates the ONNX file on
  the test split and fails if its mAP@50 drifts more than 0.02 from PyTorch (`metrics/export.json`).
- **ONNX Runtime for serving.** The API image doesn't ship PyTorch or Ultralytics, so it stays small. The
  export step fails the build if ONNX and PyTorch outputs disagree.
- **Train/serve preprocessing parity.** The serving letterbox reimplements Ultralytics' (OpenCV bilinear, grey 114
  padding, same rounding) and is tested pixel-for-pixel. An earlier PIL-based resize differed enough to change
  detection counts.
- **Model selection on validation only.** The sweep ranks variants by validation mAP; the test split is evaluated once,
  for the chosen model. Reporting the best of N test scores would quietly overstate performance.
- **De-duplication before splitting.** Mirrors of this dataset contain repeated files; leaving them in inflates test scores.
- **Drift on cheap statistics.** Brightness, contrast, detections per image and top confidence need no labels, so
  they work in production where ground truth arrives late (if at all).

## Limitations & next steps

- NEU-DET images are clean, uniform, 200×200 grayscale. Real lines have lighting changes, motion blur and new
  defect types; expect a domain gap and plan for fine-tuning on plant data.
- Crazing (a faint network of fine cracks) and rolled-in scale have diffuse boundaries, so their boxes are inherently
  ambiguous. They are the weakest classes in published YOLO baselines too, and mAP@50-95 is capped by annotation noise.
- The gate threshold (mAP@50 ≥ 0.65) is a starting point; set it from your own baseline.
- Automate retraining when the drift monitor fires; promote models through the MLflow Model Registry.

## Project layout

```
├── src/            data prep, training, evaluation, ONNX export, visualisation, monitoring
├── serving/        ONNX detector, FastAPI app, prediction logging
├── app.py          Gradio demo (Hugging Face Space entrypoint)
├── tests/          pytest suite on synthetic data
├── notebooks/      Colab/Kaggle training notebook
├── deploy/         Hugging Face Space config
├── .github/        CI and CD workflows
├── dvc.yaml        reproducible pipeline stages
└── params.yaml     all hyperparameters in one place
```

## Dataset, license & credits

NEU-DET (Northeastern University, China): 1,800 grayscale images, 6 defect classes (crazing, inclusion, patches,
pitted surface, rolled-in scale, scratches) with bounding-box annotations. Please cite the original authors
(K. Song and Y. Yan, 2013) if you publish results, and check the dataset's license before redistributing images.

The training code uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed **AGPL-3.0**. That
is fine for an open-source portfolio project, but a closed-source commercial deployment would need an Ultralytics
enterprise license (or a different detector).
