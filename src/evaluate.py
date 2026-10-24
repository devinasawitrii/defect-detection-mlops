"""Evaluate the best checkpoint on the held-out test split (mAP, precision, recall, per-class AP)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from src.ckpt import checkpoint_imgsz

PLOT_FILES = ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png")


def run(params: dict) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    d, t, ev, paths = params["data"], params["train"], params["eval"], params["paths"]
    data_yaml = (Path(d["processed_dir"]) / "data.yaml").resolve()
    reports_dir, metrics_dir = Path(paths["reports_dir"]), Path(paths["metrics_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(Path(paths["models_dir"]) / "best.pt")
    imgsz = checkpoint_imgsz(model, t["imgsz"])  # the size it was trained at, not necessarily params.yaml's
    batch = t["batch"] if t["batch"] > 0 else 16  # train.batch may be -1 (auto-batch), which val does not support
    res = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        conf=ev["conf"],
        iou=ev["iou"],
        nms=False,  # NMS-free head = the exact model that gets exported to ONNX and served
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(Path(paths["runs_dir"]).resolve()),
        name="eval",
        exist_ok=True,
        plots=bool(ev.get("plots", True)),
        verbose=False,
    )

    names = res.names
    idx = [int(i) for i in res.box.ap_class_index]
    ap50 = {names[i]: float(v) for i, v in zip(idx, res.box.ap50)}
    ap = {names[i]: float(res.box.maps[i]) for i in idx}
    n_images = len(list((Path(d["processed_dir"]) / "images" / "test").iterdir()))

    metrics = {
        "test_map50": float(res.box.map50),
        "test_map50_95": float(res.box.map),
        "test_precision": float(res.box.mp),
        "test_recall": float(res.box.mr),
        "imgsz": imgsz,
        "n_test_images": n_images,
        "per_class_ap50": ap50,
        "per_class_ap50_95": ap,
    }
    (metrics_dir / "eval.json").write_text(json.dumps(metrics, indent=2))

    report = ["class              AP50    AP50-95"]
    report += [f"{c:<18} {ap50.get(c, 0):.3f}   {ap.get(c, 0):.3f}" for c in names.values()]
    report += [f"{'ALL':<18} {metrics['test_map50']:.3f}   {metrics['test_map50_95']:.3f}"]
    (reports_dir / "detection_report.txt").write_text("\n".join(report) + "\n")

    for name in PLOT_FILES:
        src = Path(res.save_dir) / name
        if src.exists():
            shutil.copy2(src, reports_dir / name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    m = run(yaml.safe_load(open(args.params)))
    print(f"test mAP@50={m['test_map50']:.4f}  mAP@50-95={m['test_map50_95']:.4f}")
    print(open(Path(yaml.safe_load(open(args.params))["paths"]["reports_dir"]) / "detection_report.txt").read())


if __name__ == "__main__":
    main()
