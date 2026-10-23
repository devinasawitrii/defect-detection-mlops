"""FastAPI service: POST an image, get defect boxes back."""
from __future__ import annotations

import io
import os
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from PIL import Image

from serving.inference import Detector, draw_detections
from serving.prediction_log import log_prediction

MAX_BYTES = 5 * 1024 * 1024

state: dict = {"detector": None, "requests": 0, "detections_by_class": Counter()}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["detector"] = Detector(
        os.getenv("MODEL_PATH", "models/model.onnx"),
        os.getenv("META_PATH", "models/classes.json"),
    )
    yield


app = FastAPI(
    title="Steel Surface Defect Detector",
    description="Detects and localises hot-rolled steel surface defects (NEU-DET) with YOLO26 + ONNX Runtime.",
    version="2.0.0",
    lifespan=lifespan,
)


async def _load_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:  # untrusted bytes: any decode failure (bad format, bomb, patched loaders) is a 400
        raise HTTPException(400, "File is not a valid image.")
    return img


def _run(img: Image.Image, conf: float | None) -> dict:
    result = state["detector"].predict(img, conf)
    log_prediction(result)
    state["requests"] += 1
    state["detections_by_class"].update(d["label"] for d in result["detections"])
    return result


@app.get("/health")
def health():
    d: Detector = state["detector"]
    return {
        "status": "ok",
        "architecture": d.meta.get("architecture"),
        "test_map50": d.meta.get("test_map50"),
        "classes": d.class_names,
        "default_conf": d.conf,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float | None = Query(None, ge=0.0, le=1.0)):
    result = _run(await _load_image(file), conf)
    return {k: result[k] for k in ("detections", "n_detections", "image_size", "latency_ms")}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float | None = Query(None, ge=0.0, le=1.0)):
    """Same as /predict but returns the image with boxes drawn on it (PNG)."""
    img = await _load_image(file)
    result = _run(img, conf)
    buf = io.BytesIO()
    draw_detections(img, result["detections"], state["detector"].class_names).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"X-Detections": str(result["n_detections"])})


@app.get("/stats")
def stats():
    return {"requests": state["requests"], "detections_by_class": dict(state["detections_by_class"])}
