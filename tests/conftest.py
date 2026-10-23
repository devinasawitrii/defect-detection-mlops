"""Shared fixtures: a synthetic NEU-DET-like dataset and a tiny end-to-end trained YOLO26."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("YOLO_OFFLINE", "1")  # never try to download weights/fonts during tests
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.prepare_data import build  # noqa: E402
from tests.synthetic import write_dataset  # noqa: E402

IMGSZ = 64


@pytest.fixture(scope="session")
def raw_dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("raw"), per_class=20)


@pytest.fixture(scope="session")
def processed_dataset(raw_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    build(raw_dataset, out, val_size=0.2, test_size=0.2, seed=0)
    return out


@pytest.fixture(scope="session")
def tiny_params(processed_dataset, tmp_path_factory):
    base = tmp_path_factory.mktemp("artifacts")
    return {
        "data": {"processed_dir": str(processed_dataset)},
        "train": {
            "model": "yolo26n.yaml", "imgsz": IMGSZ, "epochs": 3, "batch": 16, "patience": 10,
            "lr0": 0.002, "optimizer": "AdamW", "workers": 0, "amp": False, "seed": 0, "plots": False,
        },
        "eval": {"conf": 0.001, "iou": 0.6, "plots": False},
        "serve": {"conf": 0.05, "iou": 0.5},
        "mlflow": {"tracking_uri": f"sqlite:///{base / 'mlflow.db'}", "experiment": "test"},
        "paths": {
            "models_dir": str(base / "models"),
            "metrics_dir": str(base / "metrics"),
            "reports_dir": str(base / "reports"),
            "runs_dir": str(base / "runs"),
        },
    }


@pytest.fixture(scope="session")
def trained(tiny_params, processed_dataset):
    """Train a few epochs and export to ONNX; reused by several test modules."""
    from src.export_onnx import export
    from src.train import run

    result = run(tiny_params)
    export(
        tiny_params["paths"]["models_dir"],
        IMGSZ,
        tiny_params["serve"]["conf"],
        sample_dir=processed_dataset / "images" / "test",
        metrics_dir=tiny_params["paths"]["metrics_dir"],
        runs_dir=tiny_params["paths"]["runs_dir"],
    )
    return tiny_params, result
