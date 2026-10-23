"""End-to-end: train -> evaluate -> export -> visualize -> serve -> monitor."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from serving.inference import Detector, draw_detections
from src.monitor import check
from src.prepare_data import CLASSES


def _detector(params, conf=0.01):
    models = Path(params["paths"]["models_dir"])
    return Detector(models / "model.onnx", models / "classes.json", conf=conf)


def test_training_produces_checkpoint_and_metrics(trained):
    params, result = trained
    assert (Path(params["paths"]["models_dir"]) / "best.pt").exists()
    assert result["epochs_run"] >= 1
    saved = json.loads((Path(params["paths"]["metrics_dir"]) / "train.json").read_text())
    assert saved["imgsz"] == 64


def test_mlflow_run_is_logged(trained):
    import mlflow

    params, _ = trained
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[params["mlflow"]["experiment"]])
    assert len(runs) >= 1
    assert "params.train.imgsz" in runs.columns


def test_evaluate_writes_detection_metrics(trained):
    from src.evaluate import run

    params, _ = trained
    m = run(params)
    for key in ("test_map50", "test_map50_95", "test_precision", "test_recall"):
        assert 0.0 <= m[key] <= 1.0
    assert set(m["per_class_ap50"]) <= set(CLASSES)
    assert (Path(params["paths"]["reports_dir"]) / "detection_report.txt").exists()
    assert (Path(params["paths"]["metrics_dir"]) / "eval.json").exists()


def test_onnx_map_matches_pytorch(trained, processed_dataset):
    from src.evaluate import run as evaluate
    from src.export_onnx import check_map_parity

    params, _ = trained
    pt = evaluate(params)["test_map50"]
    onnx_map = check_map_parity(
        Path(params["paths"]["models_dir"]) / "model.onnx", Path(processed_dataset) / "data.yaml",
        params["train"]["imgsz"], params["eval"], params["paths"]["runs_dir"], pt,
    )
    assert abs(onnx_map - pt) <= 0.02
    with pytest.raises(RuntimeError):  # a clearly wrong reference must trip the gate
        check_map_parity(
            Path(params["paths"]["models_dir"]) / "model.onnx", Path(processed_dataset) / "data.yaml",
            params["train"]["imgsz"], params["eval"], params["paths"]["runs_dir"], pt + 0.5,
        )


def test_onnx_export_is_end_to_end(trained):
    params, _ = trained
    det = _detector(params)
    canvas = np.full((det.size, det.size, 3), 120, dtype=np.uint8)
    rows = det.raw(canvas)
    assert rows.ndim == 2 and rows.shape[1] == 6
    meta = json.loads((Path(params["paths"]["models_dir"]) / "classes.json").read_text())
    assert meta["img_size"] == 64 and len(meta["class_names"]) == 6


@pytest.mark.parametrize("size", [(200, 200), (320, 160), (90, 240), (201, 133)])
def test_letterbox_is_pixel_identical_to_ultralytics(trained, size):
    """Train/serve skew guard: our numpy letterbox must equal the one Ultralytics applies at train/val time."""
    from ultralytics.data.augment import LetterBox

    params, _ = trained
    det = _detector(params)
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    ours, _, _ = det.letterbox(img)
    theirs = LetterBox(new_shape=(det.size, det.size), auto=False)(image=np.asarray(img))
    assert ours.shape == theirs.shape and np.array_equal(ours, theirs)


@pytest.mark.parametrize("size", [(200, 200), (320, 160), (90, 240)])
def test_letterbox_roundtrip_maps_boxes_back_to_original(trained, size):
    """A box placed in letterboxed space must decode back to the same place in the original image."""
    params, _ = trained
    det = _detector(params)
    img = Image.new("RGB", size)
    _, r, (left, top) = det.letterbox(img)
    ox1, oy1, ox2, oy2 = size[0] * 0.2, size[1] * 0.3, size[0] * 0.7, size[1] * 0.8
    rows = np.array([[ox1 * r + left, oy1 * r + top, ox2 * r + left, oy2 * r + top, 0.9, 2]])
    (d,) = det.decode(rows, r, (left, top), size, conf=0.5)
    assert d["label"] == det.class_names[2]
    assert d["box"] == pytest.approx([ox1, oy1, ox2, oy2], abs=1.0)


def test_decode_filters_by_confidence_and_sorts(trained):
    params, _ = trained
    det = _detector(params)
    rows = np.array([[0, 0, 30, 30, 0.2, 0], [10, 10, 40, 40, 0.9, 1], [5, 5, 6, 5.5, 0.99, 2]])
    out = det.decode(rows, 1.0, (0, 0), (64, 64), conf=0.1)
    assert [d["confidence"] for d in out] == [0.9, 0.2]  # degenerate (1px) box dropped, sorted by score


def test_predict_output_schema(trained, raw_dataset):
    params, _ = trained
    det = _detector(params, conf=0.0)
    res = det.predict(Image.open(next(raw_dataset.glob("crazing_1.jpg"))))
    assert res["n_detections"] == len(res["detections"])
    for d in res["detections"]:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 200
        assert d["label"] in det.class_names
    assert 0.0 <= res["image_stats"]["brightness"] <= 1.0


def test_draw_detections_changes_pixels():
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    out = draw_detections(img, [{"label": "crazing", "confidence": 0.9, "box": [10, 20, 60, 70]}], ["crazing"])
    assert out.size == img.size and np.asarray(out).sum() > 0 and np.asarray(img).sum() == 0


def test_visualize_creates_grid(trained):
    from src.visualize import run

    params, _ = trained
    assert run(params, n_images=3).exists()


def test_api_endpoints(trained, raw_dataset, tmp_path, monkeypatch):
    params, _ = trained
    models_dir = Path(params["paths"]["models_dir"])
    monkeypatch.setenv("MODEL_PATH", str(models_dir / "model.onnx"))
    monkeypatch.setenv("META_PATH", str(models_dir / "classes.json"))
    monkeypatch.setenv("PRED_LOG", str(tmp_path / "preds.jsonl"))
    from serving.api import app

    img_bytes = next(raw_dataset.glob("patches_1.jpg")).read_bytes()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        ok = client.post("/predict?conf=0.0", files={"file": ("x.jpg", img_bytes, "image/jpeg")})
        assert ok.status_code == 200
        body = ok.json()
        assert set(body) == {"detections", "n_detections", "image_size", "latency_ms"}
        assert body["image_size"] == [200, 200]
        img_resp = client.post("/predict/image", files={"file": ("x.jpg", img_bytes, "image/jpeg")})
        assert img_resp.status_code == 200 and img_resp.headers["content-type"] == "image/png"
        assert client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
        assert client.post("/predict?conf=7", files={"file": ("x.jpg", img_bytes, "image/jpeg")}).status_code == 422
        assert client.get("/stats").json()["requests"] >= 2
    assert (tmp_path / "preds.jsonl").exists()


def test_reference_stats_and_monitor_on_real_logs(trained, processed_dataset, tmp_path):
    from serving.prediction_log import log_prediction
    from src.make_reference import build as build_reference
    from src.monitor import load_log

    params, _ = trained
    ref = build_reference(processed_dataset, params["paths"]["models_dir"], tmp_path / "ref.json", conf=0.05)
    assert ref["n"] > 0 and set(ref) >= {"brightness", "contrast", "n_detections", "max_confidence"}

    det = _detector(params, conf=0.05)
    log = tmp_path / "log.jsonl"
    for p in sorted((Path(processed_dataset) / "images" / "test").glob("*.jpg"))[:40]:
        log_prediction(det.predict(Image.open(p)), log)
    cols = load_log(log)
    assert len(cols["brightness"]) == min(40, len(list((Path(processed_dataset) / "images" / "test").glob("*.jpg"))))


def test_monitor_detects_drift():
    rng = np.random.default_rng(0)
    ref = {
        "n": 300,
        "brightness": rng.normal(0.45, 0.03, 300).tolist(),
        "contrast": rng.normal(0.05, 0.01, 300).tolist(),
        "n_detections": rng.poisson(1.5, 300).tolist(),
        "max_confidence": rng.beta(8, 2, 300).tolist(),
    }
    feats = ("brightness", "contrast", "n_detections", "max_confidence")
    same = {k: rng.permutation(ref[k])[:150].tolist() for k in feats}
    shifted = dict(same, brightness=(np.array(same["brightness"]) - 0.15).tolist())
    assert check(ref, same)["drift_detected"] is False
    report = check(ref, shifted)
    assert report["drift_detected"] is True and report["features"]["brightness"]["drift"] is True
    assert check(ref, {k: v[:5] for k, v in same.items()})["status"] == "insufficient_data"
