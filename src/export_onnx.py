"""Export the best checkpoint to an NMS-free ONNX graph and verify parity with PyTorch."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from serving.inference import Detector
from src.prepare_data import IMG_EXT


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def check_parity(model, detector: Detector, sample: Path, imgsz: int) -> None:
    """Feed the *same* letterboxed image to PyTorch and ONNX Runtime and compare the top detections.

    Using an already-letterboxed input removes preprocessing differences, so this isolates the export itself.
    """
    canvas, _, _ = detector.letterbox(Image.open(sample))
    onnx_rows = detector.raw(canvas)
    onnx_rows = onnx_rows[onnx_rows[:, 4] >= 0.01]
    bgr = np.ascontiguousarray(canvas[..., ::-1])  # Ultralytics expects BGR for numpy input
    res = model.predict(bgr, imgsz=imgsz, conf=0.01, nms=False, verbose=False, device="cpu")[0]
    torch_boxes = res.boxes
    if len(torch_boxes) == 0 and len(onnx_rows) == 0:
        return
    top_t = float(torch_boxes.conf.max()) if len(torch_boxes) else 0.0
    top_o = float(onnx_rows[:, 4].max()) if len(onnx_rows) else 0.0
    if abs(top_t - top_o) > 0.02:
        raise RuntimeError(f"ONNX/PyTorch mismatch: top score {top_o:.4f} vs {top_t:.4f}")
    if top_t > 0.05:
        i_t = int(torch_boxes.conf.argmax())
        i_o = int(onnx_rows[:, 4].argmax())
        iou = _iou(torch_boxes.xyxy[i_t].tolist(), onnx_rows[i_o, :4].tolist())
        if iou < 0.9 or int(torch_boxes.cls[i_t]) != int(onnx_rows[i_o, 5]):
            raise RuntimeError(f"ONNX/PyTorch top box mismatch (IoU {iou:.3f})")


def check_map_parity(
    onnx_path: Path, data_yaml: Path, imgsz: int, eval_cfg: dict, runs_dir: str | Path, pytorch_map50: float,
    tol: float = 0.02,
) -> float:
    """Evaluate the exported ONNX model on the test split and require mAP@50 within ``tol`` of PyTorch's."""
    from ultralytics import YOLO

    res = YOLO(str(onnx_path), task="detect").val(
        data=str(data_yaml), split="test", imgsz=imgsz, batch=1, conf=eval_cfg["conf"], iou=eval_cfg["iou"],
        nms=False, device="cpu", plots=False, verbose=False,
        project=str(Path(runs_dir).resolve()), name="onnx_eval", exist_ok=True,
    )
    onnx_map50 = float(res.box.map50)
    if abs(onnx_map50 - pytorch_map50) > tol:
        raise RuntimeError(f"ONNX mAP@50 {onnx_map50:.4f} differs from PyTorch {pytorch_map50:.4f} by more than {tol}")
    return onnx_map50


def export(
    models_dir: str | Path,
    imgsz: int,
    conf: float = 0.25,
    sample_dir: str | Path | None = None,
    data_yaml: str | Path | None = None,
    eval_cfg: dict | None = None,
    metrics_dir: str | Path = "metrics",
    runs_dir: str | Path = "runs",
) -> Path:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    models_dir, metrics_dir = Path(models_dir), Path(metrics_dir)
    model = YOLO(models_dir / "best.pt")
    # nms=False -> NMS-free head: every output row is already a final detection (no NMS needed when serving)
    exported = model.export(format="onnx", imgsz=imgsz, nms=False, simplify=False, verbose=False)
    onnx_path = models_dir / "model.onnx"
    shutil.move(exported, onnx_path)

    names = [model.names[i] for i in sorted(model.names)]
    eval_file = metrics_dir / "eval.json"
    pytorch_map50 = json.loads(eval_file.read_text()).get("test_map50") if eval_file.exists() else None
    meta = {
        "class_names": names,
        "img_size": imgsz,
        "conf": conf,
        "model_name": Path(str(model.ckpt_path)).stem if getattr(model, "ckpt_path", None) else "yolo26",
        "architecture": "YOLO26 (NMS-free end-to-end head)",
        "test_map50": pytorch_map50,
    }
    (models_dir / "classes.json").write_text(json.dumps(meta, indent=2))

    detector = Detector(onnx_path, models_dir / "classes.json")
    if sample_dir is not None:
        samples = [p for p in sorted(Path(sample_dir).rglob("*")) if p.suffix.lower() in IMG_EXT][:3]
        for s in samples:
            check_parity(model, detector, s, imgsz)
        print(f"parity OK on {len(samples)} sample image(s)")

    if data_yaml is not None and eval_cfg is not None and pytorch_map50 is not None:
        onnx_map50 = check_map_parity(onnx_path, Path(data_yaml), imgsz, eval_cfg, runs_dir, pytorch_map50)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "export.json").write_text(json.dumps({
            "pytorch_test_map50": pytorch_map50,
            "onnx_test_map50": onnx_map50,
            "abs_diff": abs(onnx_map50 - pytorch_map50),
            "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        }, indent=2))
        print(f"mAP@50 parity OK: PyTorch {pytorch_map50:.4f} vs ONNX {onnx_map50:.4f}")
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    p = yaml.safe_load(open(args.params))
    export(
        p["paths"]["models_dir"],
        p["train"]["imgsz"],
        p["serve"]["conf"],
        Path(p["data"]["processed_dir"]) / "images" / "test",
        data_yaml=Path(p["data"]["processed_dir"]).resolve() / "data.yaml",
        eval_cfg=p["eval"],
        metrics_dir=p["paths"]["metrics_dir"],
        runs_dir=p["paths"]["runs_dir"],
    )


if __name__ == "__main__":
    main()
