"""Ground truth vs. predictions from the *exported ONNX model* on test images (portfolio figure)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from serving.inference import Detector, draw_detections  # noqa: E402
from src.prepare_data import IMG_EXT  # noqa: E402


def read_gt(label_file: Path, class_names: list[str], size: tuple[int, int]) -> list[dict]:
    w, h = size
    dets = []
    for line in label_file.read_text().splitlines():
        c, xc, yc, bw, bh = line.split()
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        box = [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]
        dets.append({"label": class_names[int(c)], "confidence": 1.0, "box": box})
    return dets


def run(params: dict, n_images: int = 6) -> Path:
    paths, d = params["paths"], params["data"]
    models_dir = Path(paths["models_dir"])
    detector = Detector(models_dir / "model.onnx", models_dir / "classes.json", conf=params["serve"]["conf"])
    img_dir = Path(d["processed_dir"]) / "images" / "test"
    lbl_dir = Path(d["processed_dir"]) / "labels" / "test"

    # one image per primary class first, so every defect type appears in the figure
    picked, seen = [], set()
    files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
    for p in files:
        first = (lbl_dir / f"{p.stem}.txt").read_text().split()[0]
        if first not in seen:
            seen.add(first)
            picked.append(p)
    picked = (picked + [p for p in files if p not in picked])[:n_images]

    fig, axes = plt.subplots(len(picked), 2, figsize=(6, 3 * len(picked)), squeeze=False)
    for row, p in enumerate(picked):
        img = Image.open(p).convert("RGB")
        gt = read_gt(lbl_dir / f"{p.stem}.txt", detector.class_names, img.size)
        pred = detector.predict(img)["detections"]
        axes[row, 0].imshow(draw_detections(img, gt, detector.class_names))
        axes[row, 0].set_title("ground truth", fontsize=9)
        axes[row, 1].imshow(draw_detections(img, pred, detector.class_names))
        axes[row, 1].set_title(f"prediction ({len(pred)} boxes)", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.tight_layout()
    out = Path(paths["reports_dir"]) / "predictions_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()
    run(yaml.safe_load(open(args.params)), args.n)


if __name__ == "__main__":
    main()
