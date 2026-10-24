"""Helpers for reading settings back from a trained checkpoint."""
from __future__ import annotations


def checkpoint_imgsz(model, default: int) -> int:
    """Image size the checkpoint was TRAINED at.

    Evaluation and export must use it, not whatever ``params.yaml`` currently says: after a sweep the promoted
    winner may have been trained at a different resolution than the default recipe.
    """
    return int(model.overrides.get("imgsz", default))
