"""Copy one test image per defect class into examples/ (used by the Gradio demo and CD smoke test)."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.prepare_data import CLASSES, IMG_EXT


def main(src: str = "data/processed", dst: str = "examples") -> None:
    img_dir, lbl_dir = Path(src) / "images" / "test", Path(src) / "labels" / "test"
    Path(dst).mkdir(exist_ok=True)
    done: set[int] = set()
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        cls = int((lbl_dir / f"{p.stem}.txt").read_text().split()[0])
        if cls in done:
            continue
        done.add(cls)
        shutil.copy2(p, Path(dst) / f"{CLASSES[cls]}{p.suffix.lower()}")
    print(f"{len(done)} examples written to {dst}/")


if __name__ == "__main__":
    main()
