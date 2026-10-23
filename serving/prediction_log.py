"""Append-only JSONL log of predictions, the input to drift monitoring."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def log_path() -> Path:
    return Path(os.getenv("PRED_LOG", "logs/predictions.jsonl"))


def log_prediction(result: dict, path: Path | None = None) -> None:
    record = {
        "ts": time.time(),
        "n_detections": result["n_detections"],
        "max_confidence": result["max_confidence"],
        "labels": [d["label"] for d in result["detections"]],
        "latency_ms": result["latency_ms"],
        **result["image_stats"],
    }
    path = path or log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # logging must never break serving
