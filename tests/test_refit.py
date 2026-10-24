import copy
import json
from pathlib import Path

import pytest
import yaml

from src.refit import MIN_EPOCHS, refit_params, run, write_trainval_yaml
from src.sweep import run as sweep_run
from tests.conftest import IMGSZ  # noqa: F401  (sets offline env vars before ultralytics is imported)


def test_trainval_yaml_lists_both_splits(processed_dataset, tmp_path):
    out = write_trainval_yaml(processed_dataset, tmp_path / "tv.yaml")
    data = yaml.safe_load(out.read_text())
    assert data["train"] == ["images/train", "images/val"]
    assert data["val"] == "images/val" and Path(data["path"]).is_absolute()
    assert (Path(data["path"]) / "images" / "val").is_dir()


def test_refit_params_fixes_epochs_and_disables_early_stopping(tiny_params, tmp_path):
    config = {**tiny_params["train"], "epochs": 150, "patience": 40, "extra": {"close_mosaic": 20, "cos_lr": True}}
    winner = {"name": "w", "best_epoch": 40, "train_config": config}
    p = refit_params(tiny_params, winner, tmp_path / "tv.yaml")
    assert p["train"]["epochs"] == 44 and p["train"]["patience"] == 44  # ceil(40 * 1.1)
    assert p["train"]["checkpoint"] == "last"
    assert p["train"]["extra"] == {"close_mosaic": 14, "cos_lr": True}  # capped at epochs // 3
    assert p["data"]["yaml"].endswith("tv.yaml") and p["mlflow"]["run_name"] == "refit-w"
    assert tiny_params["train"]["epochs"] == 3 and "yaml" not in tiny_params["data"]  # originals untouched
    short = refit_params(tiny_params, {**winner, "best_epoch": 3}, tmp_path / "tv.yaml")
    assert short["train"]["epochs"] == MIN_EPOCHS


def test_refit_rejects_rows_from_older_sweeps(tiny_params, tmp_path):
    with pytest.raises(SystemExit):
        refit_params(tiny_params, {"name": "old", "checkpoint": "x"}, tmp_path / "tv.yaml")


def test_refit_end_to_end(tiny_params, tmp_path, monkeypatch):
    monkeypatch.setattr("src.refit.MIN_EPOCHS", 2)  # keep the test fast
    params = copy.deepcopy(tiny_params)
    params["paths"] = {k: str(tmp_path / k) for k in ("models_dir", "metrics_dir", "reports_dir", "runs_dir")}
    params["sweep"] = [{"name": "only", "epochs": 3}]
    summary = sweep_run(params, root=tmp_path / "sweeps")
    assert summary["variants"][0]["best_epoch"] >= 1 and "train_config" in summary["variants"][0]

    from src.sweep import promote

    promote(tmp_path / "sweeps" / "summary.json", params["paths"]["models_dir"])
    result = run(params, tmp_path / "sweeps" / "summary.json")

    assert result["val_contaminated"] is True and result["refit_of"] == "only"
    models_dir = Path(params["paths"]["models_dir"])
    assert (models_dir / "best.pt").exists() and (models_dir / "best_sweep_winner.pt").exists()
    saved = json.loads((Path(params["paths"]["metrics_dir"]) / "train.json").read_text())
    assert saved["val_contaminated"] is True
    assert (Path(params["paths"]["runs_dir"]) / "data_trainval.yaml").exists()
