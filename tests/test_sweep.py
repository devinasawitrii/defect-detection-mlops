import json
from pathlib import Path

import yaml

from src.sweep import promote, run, variant_params
from tests.conftest import IMGSZ  # noqa: F401  (sets offline env vars before ultralytics is imported)

REPO = Path(__file__).resolve().parents[1]


def test_variant_params_overrides_and_isolates_paths(tiny_params, tmp_path):
    variant = {"name": "v1", "epochs": 7, "extra": {"flipud": 0.3}}
    p = variant_params({**tiny_params, "train": {**tiny_params["train"], "extra": {"cos_lr": True}}}, variant, tmp_path)
    assert p["train"]["epochs"] == 7
    assert p["train"]["extra"] == {"cos_lr": True, "flipud": 0.3}  # merged, not replaced
    assert p["paths"]["models_dir"] == str(tmp_path / "v1" / "models")
    assert p["mlflow"]["run_name"] == "sweep-v1"
    assert tiny_params["train"]["epochs"] == 3  # original params untouched


def test_params_yaml_is_valid_for_ultralytics():
    """Typos in train.extra would only surface after minutes on Colab; catch them here."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    params = yaml.safe_load((REPO / "params.yaml").read_text())
    assert set(params["train"].get("extra", {})) <= set(DEFAULT_CFG_DICT)
    names = [v["name"] for v in params["sweep"]]
    assert len(names) == len(set(names))
    for v in params["sweep"]:
        for key in v:
            assert key in {"name", "extra"} or key in params["train"], f"unknown sweep key {key}"
        assert set(v.get("extra", {})) <= set(DEFAULT_CFG_DICT)


def test_sweep_end_to_end_picks_on_validation_and_promotes(tiny_params, tmp_path):
    params = {
        **tiny_params,
        "sweep": [
            {"name": "short", "epochs": 1},
            {"name": "longer", "epochs": 3, "extra": {"cos_lr": True, "flipud": 0.5, "close_mosaic": 1}},
        ],
    }
    summary = run(params, root=tmp_path / "sweeps")
    assert summary["winner"] in {"short", "longer"}
    assert "validation" in summary["selected_on"]
    assert {r["name"] for r in summary["variants"]} == {"short", "longer"}
    for r in summary["variants"]:
        assert Path(r["checkpoint"]).exists()
        assert "test" not in " ".join(r)  # sweep rows carry no test-set metrics

    saved = json.loads((tmp_path / "sweeps" / "summary.json").read_text())
    assert saved["winner"] == summary["winner"]

    dest = promote(tmp_path / "sweeps" / "summary.json", tmp_path / "promoted")
    assert dest.exists() and dest.name == "best.pt"


def test_sweep_only_filter(tiny_params, tmp_path):
    params = {**tiny_params, "sweep": [{"name": "a", "epochs": 1}, {"name": "b", "epochs": 1}]}
    summary = run(params, only=["b"], root=tmp_path / "s")
    assert [r["name"] for r in summary["variants"]] == ["b"]


def test_sweep_merges_with_earlier_results(tiny_params, tmp_path):
    params = {**tiny_params, "sweep": [{"name": n, "epochs": 1} for n in "abc"]}
    root = tmp_path / "sw"
    run(params, only=["a"], root=root)
    second = run(params, only=["b", "c"], root=root)
    assert {r["name"] for r in second["variants"]} == {"a", "b", "c"}  # 'a' from the first call is kept
    assert {r["name"] for r in json.loads((root / "summary.json").read_text())["variants"]} == {"a", "b", "c"}
    rerun = run(params, only=["a"], root=root)  # re-running a variant replaces its row instead of duplicating it
    assert sorted(r["name"] for r in rerun["variants"]) == ["a", "b", "c"]
