"""Reproducible desktop pipeline diagnostics, using the same service as the UI."""

import argparse
import json
from multiprocessing import freeze_support
from pathlib import Path
import time


def main() -> int:
    from app.pilot.service import PilotService
    from app.pilot.settings import load_settings, save_settings, PilotSettings
    from app.runtime.session import credentials

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("query")
    analyze.add_argument("--english-query")
    analyze.add_argument("--max-documents", type=int)
    analyze.add_argument("--no-history", action="store_true")
    analyze.add_argument("--output", type=Path)
    commands.add_parser("list")
    args = parser.parse_args()
    runtime = PilotService(args.data_dir, credentials(), model_dir=args.model_dir)
    try:
        if args.command == "list":
            for row in runtime.list_runs():
                print(json.dumps({key: row[key] for key in ("id", "state", "created_at", "error")}, ensure_ascii=False))
            return 0
        settings = load_settings(args.data_dir)
        if args.max_documents is not None or args.no_history:
            values = settings.model_dump()
            if args.max_documents is not None:
                values["discovery_documents"] = args.max_documents
            if args.no_history:
                values["history_enabled"] = False
            save_settings(args.data_dir, PilotSettings.model_validate(values))
        run_id = runtime.start(args.query, args.english_query)
        print("run_id=" + run_id, flush=True)
        previous = None
        while True:
            row = runtime.get(run_id)
            status = row["state"], row["stage"], row["completed"], row["message"]
            if status != previous:
                print(json.dumps(dict(state=row["state"], stage=row["stage"], completed=row["completed"],
                                      message=row["message"], error=row["error"]), ensure_ascii=False), flush=True)
                previous = status
            if row["state"] not in {"queued", "running"}:
                break
            time.sleep(0.5)
        if row["state"] != "succeeded":
            return 1
        result = runtime.result(run_id)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"cards": len(result["result"]["cards"]), "quality": result["result"]["quality"],
                          "budget": result["budget"], "discovery": result["discovery_summary"]}, ensure_ascii=False))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
