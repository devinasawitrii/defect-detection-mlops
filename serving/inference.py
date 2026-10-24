"""ONNX Runtime defect detector. No PyTorch / Ultralytics dependency, so the Docker image stays small.

The exported YOLO26 graph is NMS-free (exported with ``nms=False``): each output row is already
``[x1, y1, x2, y2, score, class]`` in letterboxed-input pixels, sorted by the model's top-k.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

PAD_COLOR = (114, 114, 114)  # same grey Ultralytics pads with
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 165, 0), (0, 130, 200), (145, 30, 180), (0, 200, 200)]


class Detector:
    def __init__(
        self,
        model_path: str | Path = "models/model.onnx",
        meta_path: str | Path = "models/classes.json",
        conf: float | None = None,
    ):
        self.meta = json.loads(Path(meta_path).read_text())
        self.class_names: list[str] = self.meta["class_names"]
        self.size: int = self.meta["img_size"]
        self.conf = conf if conf is not None else self.meta.get("conf", 0.25)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # ---------------------------------------------------------------- preprocessing
    def letterbox(self, img: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize keeping aspect ratio and pad to a square canvas.

        Mirrors Ultralytics' LetterBox exactly (cv2.INTER_LINEAR, grey 114 padding, same rounding), because any
        difference in preprocessing between training and serving silently degrades accuracy.
        Returns an RGB uint8 array [S, S, 3], the scale ratio, and the (left, top) padding.
        """
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = round(w * r), round(h * r)
        dw, dh = (self.size - nw) / 2, (self.size - nh) / 2
        if (w, h) != (nw, nh):
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        arr = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
        return arr, r, (left, top)

    @staticmethod
    def to_tensor(canvas: np.ndarray) -> np.ndarray:
        return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()

    @staticmethod
    def image_stats(img: Image.Image) -> dict:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return {"brightness": float(gray.mean()), "contrast": float(gray.std())}

    # ---------------------------------------------------------------- inference
    def raw(self, canvas: np.ndarray) -> np.ndarray:
        """Model output rows [N, 6] for an already-letterboxed RGB array."""
        out = self.session.run(None, {self.input_name: self.to_tensor(canvas)})[0]
        if out.ndim != 3 or out.shape[-1] != 6:
            raise RuntimeError(f"Expected an end-to-end [1, N, 6] output, got {out.shape}. Re-export the model.")
        return out[0]

    def decode(self, rows, r, pad, orig_size, conf) -> list[dict]:
        w, h = orig_size
        detections = []
        for x1, y1, x2, y2, score, cls in rows:
            if score < conf:
                continue
            box = [
                float(np.clip((x1 - pad[0]) / r, 0, w)),
                float(np.clip((y1 - pad[1]) / r, 0, h)),
                float(np.clip((x2 - pad[0]) / r, 0, w)),
                float(np.clip((y2 - pad[1]) / r, 0, h)),
            ]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            detections.append(
                {"label": self.class_names[int(cls)], "confidence": float(score), "box": [round(v, 1) for v in box]}
            )
        return sorted(detections, key=lambda d: -d["confidence"])

    def predict(self, img: Image.Image, conf: float | None = None) -> dict:
        start = time.perf_counter()
        canvas, r, pad = self.letterbox(img)
        detections = self.decode(self.raw(canvas), r, pad, img.size, self.conf if conf is None else conf)
        return {
            "detections": detections,
            "n_detections": len(detections),
            "max_confidence": detections[0]["confidence"] if detections else 0.0,
            "image_size": list(img.size),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "image_stats": self.image_stats(img),
        }


def draw_detections(img: Image.Image, detections: list[dict], class_names: list[str]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        color = PALETTE[class_names.index(det["label"]) % len(PALETTE)]
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{det['label']} {det['confidence']:.2f}"
        tw = draw.textlength(text)
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + 12], fill=color)
        draw.text((x1 + 2, ty), text, fill=(255, 255, 255))
    return out
