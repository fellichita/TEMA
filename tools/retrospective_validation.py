"""Offline pilot: python -m tools.retrospective_validation --snapshot ... --output ..."""

import argparse
import json

from app.ml.corpus import read_snapshot
from app.ml.service import export_result
from app.ml.validation import run_retrospective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_retrospective(read_snapshot(args.snapshot))
    export_result(report, args.output, protected_paths=[args.snapshot])
    print(json.dumps({"status": report["status"], "train_fingerprint": report["train_fingerprint"],
                      "evaluation": report["evaluation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
