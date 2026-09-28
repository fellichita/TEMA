"""Run the local MVP: python -m app.ml --snapshot FILE --topic TOPIC --output RESULT.json."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

from app.backend.errors import BackendError
from app.ml.contracts import AnalysisInputError
from app.ml.service import export_result, inspect_snapshot, run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--history-id")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--topic", help="Для JSON по умолчанию используется тема снимка")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--relevance-mode", choices=("lexical", "semantic"), default="lexical",
                        help="lexical: прежние правила; semantic: установленная локальная модель (без скачивания)")
    parser.add_argument("--model-dir", type=Path,
                        help="Каталог установленной модели; только с --relevance-mode semantic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.model_dir is not None and args.relevance_mode != "semantic":
        parser.error("--model-dir допустим только с --relevance-mode semantic")
    try:
        if args.output.exists():
            raise AnalysisInputError("Выходной файл уже существует. Выберите новое имя.")
        context = nullcontext(None)
        if args.history_id:
            from app.backend.config import BackendSettings
            from app.backend.service import Backend
            context = Backend(BackendSettings(data_dir=args.data_dir) if args.data_dir else BackendSettings())
        with context as backend:
            topic = args.topic or (inspect_snapshot(args.snapshot)["topic"] if args.snapshot else
                                   backend.get_history(args.history_id).request.topic)
            result = run_analysis(backend, {"topic": topic, "start_year": args.start_year,
                                           "end_year": args.end_year, "top_k": args.top_k,
                                           "relevance_mode": args.relevance_mode},
                                  snapshot_path=args.snapshot, history_id=args.history_id,
                                  progress=lambda n, text: print(f"{n}% {text}", file=sys.stderr),
                                  **({"model_dir": args.model_dir} if args.relevance_mode == "semantic" else {}))
            export_result(result, args.output, protected_paths=[args.snapshot])
        print(f"Основной TOP: {len(result['candidates'])}; предварительных сигналов: "
              f"{len(result['preliminary_signals'])}; крупных тем: {len(result['established'])}; "
              f"отклонено: {len(result['excluded_off_direction'])}. Результат: {args.output}")
        return 0
    except ImportError:
        lock = "requirements/semantic.lock" if args.relevance_mode == "semantic" else "requirements/ml.lock"
        print(f"Установите зависимости: python -m pip install -r {lock}", file=sys.stderr)
        return 2
    except (BackendError, AnalysisInputError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Русские сообщения должны пережить конвейер при любой кодовой странице консоли.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
