"""Fine-tune YOLO26 on NEU-DET (transfer learning from COCO weights) and track the run in MLflow."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def clean_key(key: str) -> str:
    """MLflow metric names cannot contain parentheses, e.g. 'metrics/mAP50(B)'."""
    return re.sub(r"[^\w\-\. /]", "", key.strip()).replace("//", "/")


def run(params: dict) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})  # we log to MLflow ourselves, with our own run name/params

    d, t, m, paths = params["data"], params["train"], params["mlflow"], params["paths"]
    device = 0 if torch.cuda.is_available() else "cpu"
    # `data.yaml` can be overridden (the train+val refit writes its own); default is the standard split
    data_yaml = Path(d["yaml"] if d.get("yaml") else Path(d["processed_dir"]) / "data.yaml").resolve()
    models_dir, metrics_dir = Path(paths["models_dir"]), Path(paths["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(t["model"])
    # '.pt' = COCO-pretrained weights (transfer learning); '.yaml' = random init (used by the tests)
    pretrained = str(t["model"]).endswith(".pt")

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment"])
    run_name = m.get("run_name") or f"{Path(t['model']).stem}-{git_sha()}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({f"data.{k}": v for k, v in d.items()})
        mlflow.log_params({f"train.{k}": v for k, v in t.items()})
        mlflow.set_tags({"git_sha": git_sha(), "device": str(device)})

        model.train(
            data=str(data_yaml),
            imgsz=t["imgsz"],
            epochs=t["epochs"],
            batch=t["batch"],
            patience=t["patience"],
            lr0=t["lr0"],
            optimizer=t["optimizer"],
            workers=t["workers"],
            amp=bool(t["amp"]),
            seed=t["seed"],
            device=device,
            nms=False,  # validate (and pick best.pt) with the NMS-free head, the same one we export and serve
            pretrained=pretrained,
            project=str(Path(paths["runs_dir"]).resolve()),
            name="train",
            exist_ok=True,
            plots=bool(t.get("plots", True)),
            verbose=False,
            **t.get("extra", {}),  # any other Ultralytics train arg: cos_lr, flipud, close_mosaic, cache, ...
        )

        trainer = model.trainer
        save_dir = Path(trainer.save_dir)
        # best.pt is chosen on validation fitness. In a train+val refit that choice is meaningless (val is in the
        # training data), so the refit takes the final epoch instead.
        shutil.copy2(trainer.last if t.get("checkpoint") == "last" else trainer.best, models_dir / "best.pt")

        results_csv = save_dir / "results.csv"
        epochs_run, best_epoch = 0, 0
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
            epochs_run = len(df)
            # Ultralytics' own fitness (what best.pt is picked on): 0.1 * mAP50 + 0.9 * mAP50-95
            fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
            best_epoch = int(fitness.values.argmax()) + 1
            for step, row in df.iterrows():
                mlflow.log_metrics(
                    {clean_key(k): float(v) for k, v in row.items() if k != "epoch" and pd.notna(v)},
                    step=int(step),
                )

        keep = ("mAP", "precision", "recall")
        final = {clean_key(k): float(v) for k, v in trainer.metrics.items() if any(s in k for s in keep)}
        mlflow.log_metrics({f"final/{k.replace('metrics/', '')}": v for k, v in final.items()})
        mlflow.log_artifact(str(models_dir / "best.pt"))

    result = {
        "val_map50": trainer.metrics.get("metrics/mAP50(B)"),
        "val_map50_95": trainer.metrics.get("metrics/mAP50-95(B)"),
        "val_precision": trainer.metrics.get("metrics/precision(B)"),
        "val_recall": trainer.metrics.get("metrics/recall(B)"),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "val_contaminated": bool(d.get("yaml")),  # True for a train+val refit: val metrics are not held-out
        "model": str(t["model"]),
        "imgsz": t["imgsz"],
    }
    (metrics_dir / "train.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(open(args.params))), indent=2))


if __name__ == "__main__":
    main()
