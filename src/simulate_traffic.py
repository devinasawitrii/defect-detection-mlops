"""Generate prediction logs from test images, optionally with simulated drift.

`--mode drifted` darkens images and adds sensor noise, mimicking a lighting
change or camera degradation on the production line. Run src.monitor afterwards
to see the drift alert fire.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from serving.inference import Detector
from serving.prediction_log import log_path, log_prediction
from src.prepare_data import IMG_EXT


def degrade(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img.convert("L"), dtype=np.float32) * 0.55  # darker
    arr += rng.normal(0, 18, arr.shape)                           # sensor noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["normal", "drifted"], default="normal")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--data", default="data/processed/images/test")
    parser.add_argument("--models", default="models")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--reset", action="store_true", help="clear the log first")
    args = parser.parse_args()

    random.seed(0)
    rng = np.random.default_rng(0)
    files = [p for p in Path(args.data).rglob("*") if p.suffix.lower() in IMG_EXT]
    if not files:
        raise SystemExit(f"No images under {args.data}")
    path = log_path()
    if args.reset and path.exists():
        path.unlink()

    detector = Detector(Path(args.models) / "model.onnx", Path(args.models) / "classes.json", conf=args.conf)
    for p in random.choices(files, k=args.n):
        img = Image.open(p)
        if args.mode == "drifted":
            img = degrade(img, rng)
        log_prediction(detector.predict(img), path)
    print(f"logged {args.n} {args.mode} predictions -> {path}")


if __name__ == "__main__":
    main()
