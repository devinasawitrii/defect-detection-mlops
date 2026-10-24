"""Build the reference distribution (held-out val set) that drift monitoring compares against."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from serving.inference import Detector
from src.prepare_data import IMG_EXT

FEATURES = ("brightness", "contrast", "n_detections", "max_confidence")


def build(processed_dir: str | Path, models_dir: str | Path, out: str | Path, conf: float = 0.25) -> dict:
    detector = Detector(Path(models_dir) / "model.onnx", Path(models_dir) / "classes.json", conf=conf)
    cols: dict[str, list[float]] = {f: [] for f in FEATURES}
    for p in sorted((Path(processed_dir) / "images" / "val").iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        res = detector.predict(Image.open(p))
        cols["brightness"].append(res["image_stats"]["brightness"])
        cols["contrast"].append(res["image_stats"]["contrast"])
        cols["n_detections"].append(res["n_detections"])
        cols["max_confidence"].append(res["max_confidence"])
    ref = {"n": len(cols["brightness"]), **cols}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(ref))
    print(f"reference stats from {ref['n']} validation images -> {out}")
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    params = yaml.safe_load(open(args.params))
    build(
        params["data"]["processed_dir"],
        params["paths"]["models_dir"],
        Path(params["paths"]["reports_dir"]) / "reference_stats.json",
        params["serve"]["conf"],
    )


if __name__ == "__main__":
    main()
