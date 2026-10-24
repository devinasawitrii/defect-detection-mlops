"""Retrain the sweep winner on train+val ("refit"), then evaluate it once on the untouched test split.

Why: after model selection the validation images are no longer needed for choosing anything, so they can
become training data (+~18% images on NEU-DET). Typical gain is small but real, and it costs no extra test peeking.

Rules that keep it honest:
  * The recipe is the winner's exact config. Epochs are fixed from its best validation epoch (x1.1 for the extra
    data), early stopping is off, and the LAST epoch is kept, because validation can no longer judge a checkpoint.
  * Validation metrics from this run are contaminated (flagged in metrics/train.json). Only the test split counts.
  * Decide beforehand that the refit is what you report, and evaluate it once.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import yaml

from src.train import run as train_run

MIN_EPOCHS = 20


def write_trainval_yaml(processed_dir: str | Path, out: str | Path) -> Path:
    processed = Path(processed_dir).resolve()
    data = yaml.safe_load((processed / "data.yaml").read_text())
    data["path"] = str(processed)
    data["train"] = ["images/train", "images/val"]
    data["val"] = "images/val"  # still needed by the trainer for its per-epoch report, but no longer held out
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def refit_params(params: dict, winner: dict, data_yaml: Path) -> dict:
    if "train_config" not in winner or not winner.get("best_epoch"):
        raise SystemExit(
            f"Sweep row '{winner['name']}' has no train_config/best_epoch (it comes from an older sweep). "
            f"Re-run it: python -m src.sweep --only {winner['name']}"
        )
    p = copy.deepcopy(params)
    t = {**p["train"], **copy.deepcopy(winner["train_config"])}
    epochs = max(MIN_EPOCHS, math.ceil(winner["best_epoch"] * 1.1))
    t["epochs"] = epochs
    t["patience"] = epochs  # early stopping off: there is no held-out validation set any more
    t["checkpoint"] = "last"
    extra = dict(t.get("extra", {}))
    if "close_mosaic" in extra:
        extra["close_mosaic"] = min(extra["close_mosaic"], max(1, epochs // 3))
    t["extra"] = extra
    p["train"] = t
    p["data"] = {**p["data"], "yaml": str(data_yaml)}
    p["mlflow"] = {**p["mlflow"], "run_name": f"refit-{winner['name']}"}
    return p


def run(params: dict, summary_path: str | Path | None = None) -> dict:
    summary_path = Path(summary_path or Path(params["paths"].get("sweeps_dir", "sweeps")) / "summary.json")
    winner = json.loads(summary_path.read_text())["variants"][0]
    models_dir = Path(params["paths"]["models_dir"])

    data_yaml = write_trainval_yaml(
        params["data"]["processed_dir"], Path(params["paths"]["runs_dir"]) / "data_trainval.yaml"
    )
    p = refit_params(params, winner, data_yaml)

    existing = models_dir / "best.pt"
    if existing.exists():  # keep the validation-selected model for comparison
        shutil.copy2(existing, models_dir / "best_sweep_winner.pt")
    print(
        f"refit: {winner['name']} on train+val for {p['train']['epochs']} epochs "
        f"(best val epoch was {winner['best_epoch']})"
    )
    result = train_run(p)
    result["refit_of"] = winner["name"]
    (Path(params["paths"]["metrics_dir"]) / "train.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--summary", default=None, help="sweep summary.json (default: sweeps/summary.json)")
    args = parser.parse_args()
    result = run(yaml.safe_load(open(args.params)), args.summary)
    print(f"refit done ({result['refit_of']}). Now evaluate ONCE on test:")
    print("  python -m src.evaluate && python -m src.export_onnx")


if __name__ == "__main__":
    main()
