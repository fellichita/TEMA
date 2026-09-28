"""Interface/CLI bridge and atomic result export."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.corpus import checkpoint, read_history, read_snapshot
from app.ml.engine import analyze


def inspect_snapshot(path, *, cancel=None):
    corpus = read_snapshot(path, cancel=cancel)
    years = [int(p["until_date"][:4]) for p in corpus["periods"]]
    end = min(max(years), datetime.now(timezone.utc).year - 1) if years else datetime.now(timezone.utc).year - 1
    start = max(min(int(p["from_date"][:4]) for p in corpus["periods"]), end - 5) if years else end - 5
    return {"topic": corpus["topic"], "source": corpus["source"], "start_year": start, "end_year": end,
            "occurrences": len(corpus["entries"])}


def run_analysis(backend, options, *, snapshot_path=None, history_id=None, model_dir=None,
                 cancel=None, progress=None):
    options = AnalysisOptions.model_validate(options)
    if model_dir is not None and options.relevance_mode != "semantic":
        raise AnalysisInputError("Папка модели применяется только в режиме смысловой проверки.")
    if bool(snapshot_path) == bool(history_id):
        raise AnalysisInputError("Выберите ровно один источник: JSON-снимок или сохранённую историю.")
    if progress:
        progress(2, "Чтение сохранённого корпуса")
    corpus = (read_snapshot(snapshot_path, options.max_documents, cancel) if snapshot_path else
              read_history(backend, history_id, options.max_documents, cancel))
    checkpoint(cancel)
    if options.relevance_mode == "semantic":
        from app.ml.semantic_contracts import validate_semantic_result
        result = analyze(corpus, options, model_dir=model_dir, cancel=cancel, progress=progress)
        validate_semantic_result(result)
        return result
    return analyze(corpus, options, cancel=cancel, progress=progress)


def export_result(result, path, *, protected_paths=(), overwrite=False):
    from app.ml.semantic_contracts import validate_semantic_result
    validate_semantic_result(result)
    path = Path(path)
    if path.suffix.casefold() != ".json":
        raise AnalysisInputError("Результат сохраняется в файл .json.")
    if path.resolve() in {Path(p).resolve() for p in protected_paths if p}:
        raise AnalysisInputError("Нельзя перезаписать исходный корпус результатом анализа.")
    if path.exists() and not overwrite:
        raise AnalysisInputError("Файл уже существует. Выберите другое имя.")
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish a complete file only if this name is still free. An
            # earlier exists() check cannot protect two concurrent exporters.
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise AnalysisInputError("Файл уже существует. Выберите другое имя.") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(path)
