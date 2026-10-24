"""Regression: after a sweep, params.yaml may disagree with the promoted checkpoint's training resolution."""
import json
from pathlib import Path

from tests.conftest import IMGSZ


def test_evaluate_uses_checkpoint_imgsz_not_params(trained):
    from src.evaluate import run

    params, _ = trained
    wrong = {**params, "train": {**params["train"], "imgsz": 96}}  # params claim 96, model was trained at 64
    assert run(wrong)["imgsz"] == IMGSZ


def test_evaluate_handles_auto_batch(trained):
    from src.evaluate import run

    params, _ = trained
    auto = {**params, "train": {**params["train"], "batch": -1}}  # train.batch=-1 (auto) must not reach val()
    assert run(auto)["imgsz"] == IMGSZ


def test_export_uses_checkpoint_imgsz_when_not_given_or_wrong(trained, processed_dataset):
    from src.export_onnx import export

    params, _ = trained
    models_dir = Path(params["paths"]["models_dir"])
    for requested in (None, 96):
        export(models_dir, requested, params["serve"]["conf"], metrics_dir=params["paths"]["metrics_dir"])
        meta = json.loads((models_dir / "classes.json").read_text())
        assert meta["img_size"] == IMGSZ
