"""Convert NEU-DET (Pascal VOC XML boxes) into a YOLO dataset with train/val/test splits.

Raw archives differ between mirrors, so the folder is scanned recursively: every
``*.xml`` annotation is paired with the image of the same stem. Exact duplicate
images (same MD5) are dropped before splitting to avoid train/test leakage.
Splitting is stratified by the image's first defect class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png"}
_ALIASES = {"pitted-surface": "pitted_surface", "rolled_in_scale": "rolled-in_scale"}


def normalize_class(name: str) -> str | None:
    n = name.strip().lower()
    n = _ALIASES.get(n, n)
    return n if n in CLASSES else None


def parse_voc(xml_path: Path) -> tuple[tuple[int, int], list[tuple[int, float, float, float, float]]]:
    """Return ((width, height), [(class_id, xc, yc, w, h), ...]) with normalised YOLO boxes."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(float(size.findtext("width")))
    H = int(float(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        cls = normalize_class(obj.findtext("name", ""))
        if cls is None:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append(
            (CLASSES.index(cls), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
        )
    return (W, H), boxes


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build(
    raw_dir: str | Path,
    out_dir: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    raw, out = Path(raw_dir), Path(out_dir)
    images: dict[str, Path] = {}
    for p in sorted(raw.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            images.setdefault(p.stem, p)

    seen: set[str] = set()
    items: list[dict] = []
    no_image = no_boxes = duplicates = 0
    for xml_path in sorted(raw.rglob("*.xml")):
        img = images.get(xml_path.stem)
        if img is None:
            no_image += 1
            continue
        _, boxes = parse_voc(xml_path)
        if not boxes:
            no_boxes += 1
            continue
        digest = _md5(img)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        items.append({"img": img, "boxes": boxes, "digest": digest, "primary": boxes[0][0]})

    if not items:
        raise SystemExit(
            f"No annotated images found under {raw}. Expected NEU-DET: images plus Pascal VOC .xml files."
        )

    idx = list(range(len(items)))
    strat = [it["primary"] for it in items]
    tv_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=strat, random_state=seed)
    tr_idx, va_idx = train_test_split(
        tv_idx,
        test_size=val_size / (1.0 - test_size),
        stratify=[strat[i] for i in tv_idx],
        random_state=seed,
    )

    if out.exists():
        shutil.rmtree(out)
    summary: dict = {
        "xml_without_image": no_image,
        "xml_without_valid_boxes": no_boxes,
        "duplicates_dropped": duplicates,
    }
    for split, split_idx in (("train", tr_idx), ("val", va_idx), ("test", te_idx)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for i in split_idx:
            it = items[i]
            stem = f"{it['img'].stem}_{it['digest'][:6]}"
            shutil.copy2(it["img"], out / "images" / split / f"{stem}{it['img'].suffix.lower()}")
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in it["boxes"]]
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_boxes += len(lines)
        summary[split] = {"images": len(split_idx), "boxes": n_boxes}

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    d = yaml.safe_load(open(args.params))["data"]
    print(json.dumps(build(d["raw_dir"], d["processed_dir"], d["val_size"], d["test_size"], d["seed"]), indent=2))


if __name__ == "__main__":
    main()
