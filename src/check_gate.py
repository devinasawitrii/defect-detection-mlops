"""Quality gate: fail CI/CD if the detector is not good enough to ship."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="metrics/eval.json")
    parser.add_argument("--min-map50", type=float, default=0.65)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    score = metrics["test_map50"]
    if score < args.min_map50:
        sys.exit(f"GATE FAILED: test mAP@50 {score:.4f} < {args.min_map50:.2f}")
    print(f"GATE PASSED: test mAP@50 {score:.4f} >= {args.min_map50:.2f}")


if __name__ == "__main__":
    main()
