import shutil
from pathlib import Path

import pytest

from src.prepare_data import CLASSES, build, normalize_class, parse_voc
from tests.conftest import IMGSZ  # noqa: F401  (ensures env vars are set before ultralytics import)
from tests.synthetic import OBJ_TMPL, VOC_TMPL, write_dataset


@pytest.mark.parametrize(
    "raw,expected",
    [("crazing", "crazing"), ("Inclusion", "inclusion"), ("pitted-surface", "pitted_surface"),
     ("rolled_in_scale", "rolled-in_scale"), ("  scratches ", "scratches"), ("banana", None)],
)
def test_normalize_class(raw, expected):
    assert normalize_class(raw) == expected


def test_parse_voc_to_normalised_yolo(tmp_path):
    xml = VOC_TMPL.format(
        fname="a.jpg", w=200, h=100,
        objects=OBJ_TMPL.format(name="patches", x1=20, y1=10, x2=120, y2=60),
    )
    p = tmp_path / "a.xml"
    p.write_text(xml)
    (w, h), boxes = parse_voc(p)
    assert (w, h) == (200, 100)
    cid, xc, yc, bw, bh = boxes[0]
    assert cid == CLASSES.index("patches")
    assert (xc, yc, bw, bh) == pytest.approx((0.35, 0.35, 0.5, 0.5))


def test_parse_voc_clips_and_drops_degenerate(tmp_path):
    objs = (
        OBJ_TMPL.format(name="crazing", x1=-10, y1=-5, x2=300, y2=90)       # clipped to the image
        + OBJ_TMPL.format(name="crazing", x1=50, y1=50, x2=51, y2=51)        # too tiny
        + OBJ_TMPL.format(name="unknown_defect", x1=0, y1=0, x2=50, y2=50)   # unknown class
    )
    p = tmp_path / "b.xml"
    p.write_text(VOC_TMPL.format(fname="b.jpg", w=200, h=100, objects=objs))
    _, boxes = parse_voc(p)
    assert len(boxes) == 1
    assert boxes[0][1:] == pytest.approx((0.5, 0.45, 1.0, 0.9))


def test_build_writes_yolo_layout(processed_dataset):
    root = Path(processed_dataset)
    assert (root / "data.yaml").exists()
    total = 0
    for split in ("train", "val", "test"):
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        lbls = sorted((root / "labels" / split).glob("*.txt"))
        assert imgs and [p.stem for p in imgs] == [p.stem for p in lbls]
        total += len(imgs)
        for lbl in lbls:
            for line in lbl.read_text().splitlines():
                cls, *coords = line.split()
                assert 0 <= int(cls) < len(CLASSES)
                assert all(0.0 <= float(c) <= 1.0 for c in coords)
    assert total == 20 * len(CLASSES)


def test_splits_are_disjoint_and_cover_all_classes(processed_dataset):
    root = Path(processed_dataset)
    names = {s: {p.stem for p in (root / "images" / s).iterdir()} for s in ("train", "val", "test")}
    assert not names["train"] & names["val"] and not names["train"] & names["test"] and not names["val"] & names["test"]
    for split in ("train", "val", "test"):
        lines = [ln for p in (root / "labels" / split).glob("*.txt") for ln in p.read_text().splitlines()]
        classes = {int(ln.split()[0]) for ln in lines}
        assert classes == set(range(len(CLASSES)))


def test_duplicates_are_dropped(tmp_path):
    raw = write_dataset(tmp_path / "raw", per_class=10)
    shutil.copy2(raw / "crazing_1.jpg", raw / "crazing_dup.jpg")
    shutil.copy2(raw / "crazing_1.xml", raw / "crazing_dup.xml")
    summary = build(raw, tmp_path / "out", 0.2, 0.2, seed=0)
    assert summary["duplicates_dropped"] == 1
