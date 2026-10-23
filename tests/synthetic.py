"""Synthetic NEU-DET-like data (images + Pascal VOC XML) so tests never download anything."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.prepare_data import CLASSES

VOC_TMPL = """<annotation>
  <filename>{fname}</filename>
  <size><width>{w}</width><height>{h}</height><depth>1</depth></size>
{objects}</annotation>
"""
OBJ_TMPL = """  <object>
    <name>{name}</name>
    <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>
  </object>
"""


def draw_defect(arr: np.ndarray, cls_idx: int, x1: int, y1: int, x2: int, y2: int) -> None:
    """Class-dependent pattern inside the box so a tiny detector has something to learn."""
    if cls_idx % 3 == 0:
        arr[y1:y2, x1:x2:4] += 70      # vertical stripes
    elif cls_idx % 3 == 1:
        arr[y1:y2:4, x1:x2] += 70      # horizontal stripes
    else:
        arr[y1:y2, x1:x2] += 55        # bright patch


def make_image(rng: np.random.Generator, cls_idx: int, size: int = 200):
    arr = rng.normal(110, 10, (size, size))
    n_obj = int(rng.integers(1, 3))
    boxes = []
    for _ in range(n_obj):
        bw, bh = (int(v) for v in rng.integers(40, 90, 2))
        x1 = int(rng.integers(0, size - bw))
        y1 = int(rng.integers(0, size - bh))
        draw_defect(arr, cls_idx, x1, y1, x1 + bw, y1 + bh)
        boxes.append((x1, y1, x1 + bw, y1 + bh))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), boxes


def write_dataset(root: Path, per_class: int = 20, seed: int = 0) -> Path:
    """Write images + VOC XML files into ``root`` (flat, like a raw NEU-DET dump)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for ci, cls in enumerate(CLASSES):
        for i in range(per_class):
            img, boxes = make_image(rng, ci)
            fname = f"{cls}_{i + 1}.jpg"
            img.save(root / fname, quality=95)
            objs = "".join(OBJ_TMPL.format(name=cls, x1=a, y1=b, x2=c, y2=d) for a, b, c, d in boxes)
            xml = VOC_TMPL.format(fname=fname, w=img.width, h=img.height, objects=objs)
            (root / f"{cls}_{i + 1}.xml").write_text(xml)
    return root
