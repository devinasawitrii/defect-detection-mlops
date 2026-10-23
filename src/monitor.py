"""Data/prediction drift check: production logs vs. the validation-set reference.

Two-sample Kolmogorov-Smirnov test on input brightness, contrast, number of
detections per image and top detection confidence. Exit code 1 when drift is detected, so it can gate a scheduled job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import ks_2samp

FEATURES = ("brightness", "contrast", "n_detections", "max_confidence")


def load_log(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {f: [] for f in FEATURES}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        for f in FEATURES:
            cols[f].append(rec[f])
    return cols


def check(reference: dict, current: dict, alpha: float = 0.01, min_samples: int = 30) -> dict:
    n = len(current["brightness"])
    report: dict = {"n_current": n, "n_reference": reference["n"], "alpha": alpha, "features": {}}
    if n < min_samples:
        report["status"] = "insufficient_data"
        report["drift_detected"] = False
        return report
    drifted = False
    for f in FEATURES:
        stat, p = ks_2samp(reference[f], current[f])
        is_drift = bool(p < alpha)
        drifted = drifted or is_drift
        report["features"][f] = {
            "ks_stat": round(float(stat), 4),
            "p_value": float(p),
            "drift": is_drift,
            "ref_mean": round(sum(reference[f]) / len(reference[f]), 4),
            "cur_mean": round(sum(current[f]) / n, 4),
        }
    report["drift_detected"] = drifted
    report["status"] = "drift" if drifted else "ok"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="reports/reference_stats.json")
    parser.add_argument("--log", default="logs/predictions.jsonl")
    parser.add_argument("--out", default="reports/drift_report.json")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    reference = json.loads(Path(args.reference).read_text())
    log = Path(args.log)
    if not log.exists():
        sys.exit(f"No prediction log at {log}. Send some traffic first.")
    report = check(reference, load_log(log), args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["drift_detected"] else 0)


if __name__ == "__main__":
    main()
