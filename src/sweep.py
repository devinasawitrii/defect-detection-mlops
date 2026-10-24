"""Train several variants and pick the winner on the VALIDATION split (never the test split).

Each variant in ``params.yaml -> sweep`` overrides keys of the ``train`` section and gets its own folder under
``sweeps/``. All runs land in the same MLflow experiment so you can compare them side by side.
After choosing, ``--promote`` copies the winner to ``models/best.pt``; then run ``make evaluate export`` once to
get the official test numbers. Looking at test scores of every variant would leak the test set into model selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import yaml

from src.train import run as train_run


def variant_params(params: dict, variant: dict, root: Path) -> dict:
    p = copy.deepcopy(params)
    name = variant["name"]
    for key, value in variant.items():
        if key == "name":
            continue
        if key == "extra":
            p["train"].setdefault("extra", {}).update(value)
        else:
            p["train"][key] = value
    base = root / name
    p["paths"] = {
        **p["paths"],
        "models_dir": str(base / "models"),
        "metrics_dir": str(base / "metrics"),
        "reports_dir": str(base / "reports"),
        "runs_dir": str(base / "runs"),
    }
    p["mlflow"] = {**p["mlflow"], "run_name": f"sweep-{name}"}
    return p


def run(params: dict, only: list[str] | None = None, root: str | Path | None = None) -> dict:
    root = Path(root or params["paths"].get("sweeps_dir", "sweeps"))
    variants = [v for v in params.get("sweep", []) if not only or v["name"] in only]
    if not variants:
        raise SystemExit("No sweep variants selected. Define them under `sweep:` in params.yaml.")

    rows = []
    for v in variants:
        print(f"\n=== sweep variant: {v['name']} ===")
        vp = variant_params(params, v, root)
        result = train_run(vp)
        rows.append({
            "name": v["name"],
            "val_map50": result["val_map50"],
            "val_map50_95": result["val_map50_95"],
            "epochs_run": result["epochs_run"],
            "best_epoch": result["best_epoch"],
            "train_config": vp["train"],  # lets the train+val refit reproduce the winner exactly
            "model": result["model"],
            "imgsz": result["imgsz"],
            "checkpoint": str(root / v["name"] / "models" / "best.pt"),
        })

    # Merge with earlier sweeps so variants can be added incrementally (cheap first, expensive later).
    summary_file = root / "summary.json"
    if summary_file.exists():
        previous = json.loads(summary_file.read_text())["variants"]
        fresh = {r["name"] for r in rows}
        rows += [r for r in previous if r["name"] not in fresh and Path(r["checkpoint"]).exists()]

    rows.sort(key=lambda r: (r["val_map50_95"] or 0) + (r["val_map50"] or 0), reverse=True)
    summary = {"selected_on": "validation split (val mAP@50 + mAP@50-95)", "winner": rows[0]["name"], "variants": rows}
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def promote(summary_path: str | Path, models_dir: str | Path) -> Path:
    summary = json.loads(Path(summary_path).read_text())
    winner = summary["variants"][0]
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / "best.pt"
    shutil.copy2(winner["checkpoint"], dest)
    return dest


def print_table(summary: dict) -> None:
    print(f"\n{'variant':<14}{'model':<14}{'imgsz':>6}{'epochs':>8}{'val mAP50':>11}{'val mAP50-95':>14}")
    for r in summary["variants"]:
        print(f"{r['name']:<14}{r['model']:<14}{r['imgsz']:>6}{r['epochs_run']:>8}"
              f"{r['val_map50']:>11.4f}{r['val_map50_95']:>14.4f}")
    print(f"\nwinner (by validation): {summary['winner']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--only", nargs="*", help="run only these variant names")
    parser.add_argument("--promote", action="store_true", help="copy the winner to models/best.pt")
    args = parser.parse_args()
    params = yaml.safe_load(open(args.params))

    summary = run(params, args.only)
    print_table(summary)
    if args.promote:
        sweeps_root = Path(params["paths"].get("sweeps_dir", "sweeps"))
        dest = promote(sweeps_root / "summary.json", params["paths"]["models_dir"])
        print(f"promoted -> {dest}. Now run: python -m src.evaluate && python -m src.export_onnx")


if __name__ == "__main__":
    main()
