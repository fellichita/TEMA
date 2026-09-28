"""Offline E5 installation smoke; synthetic examples are not scientific quality labels.

python -m scripts.check_local_model [--model-dir PATH] [--output NEW.json]
Uses installed local weights only. Never installs, downloads, or tunes a threshold.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter, process_time

from app.ml.contracts import AnalysisInputError
from app.ml.local_encoder import LocalEncoder
from app.ml.semantic import ABSTRACT_CHARACTER_LIMIT
from app.ml.service import export_result
from app.ml.text import _lexical_scope_check, clean

CASES_PATH = Path(__file__).resolve().parents[1] / "tests/fixtures/local_model_cases.json"
DIMENSIONS = 384
NORM_TOLERANCE = 0.0001
REPEAT_TOLERANCE = 0.000001


def passage_text(document):
    """Use the same text boundary as the application semantic retrieval policy."""
    return clean(document["title"]) + ". " + clean(document["abstract"])[:ABSTRACT_CHARACTER_LIMIT]


def _checked_vectors(values, count):
    import numpy as np

    try:
        vectors = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise AnalysisInputError("Проверка модели: некорректные эмбеддинги.") from error
    if vectors.shape != (count, DIMENSIONS) or not np.isfinite(vectors).all():
        raise AnalysisInputError("Проверка модели: неверная размерность или нечисловые эмбеддинги.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=NORM_TOLERANCE, rtol=0):
        raise AnalysisInputError("Проверка модели: эмбеддинги не нормализованы.")
    return vectors


def _peak_rss_bytes():
    """Process lifetime high-water mark; Unix ru_maxrss units differ by platform."""
    if sys.platform not in {"darwin", "linux"}:
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def check_local_model(model_dir=None, *, encoder=None):
    """Check numeric integrity, repeatability, and five elementary retrieval pairs.

    encoder injection supports offline software tests. A failed smoke comparison
    remains visible in the report. The unvalidated adversarial texts are scored
    and passed through lexical rules but never contribute to the pass/fail gate.
    """
    import numpy as np

    started, cpu_started = perf_counter(), process_time()
    fixtures = CASES_PATH.read_bytes()
    cases = json.loads(fixtures)["queries"]
    documents = [document for case in cases for document in case["documents"]]
    queries = [case["query"] for case in cases]
    passages = [passage_text(document) for document in documents]
    load_started = perf_counter()
    encoder = LocalEncoder(model_dir) if encoder is None else encoder
    load_seconds = perf_counter() - load_started

    def encode():
        return (_checked_vectors(encoder.encode(queries, kind="query"), len(queries)),
                _checked_vectors(encoder.encode(passages, kind="passage"), len(passages)))

    inference_started = perf_counter()
    query_vectors, passage_vectors = encode()
    inference_seconds = perf_counter() - inference_started
    repeat_started = perf_counter()
    repeated_queries, repeated_passages = encode()
    repeat_seconds = perf_counter() - repeat_started
    repeatable = bool(np.allclose(query_vectors, repeated_queries, atol=REPEAT_TOLERANCE, rtol=0)
                      and np.allclose(passage_vectors, repeated_passages, atol=REPEAT_TOLERANCE, rtol=0))
    rows, passed_pairs, offset = [], 0, 0
    for index, case in enumerate(cases):
        results = []
        for document in case["documents"]:
            similarity = float(np.clip(query_vectors[index] @ passage_vectors[offset], -1, 1))
            results.append({"id": document["id"], "kind": document["kind"], "title": document["title"],
                            "similarity": similarity,
                            "lexical_decision": _lexical_scope_check(
                                case["query"], document["title"], document["abstract"])})
            offset += 1
        scores = {document["id"]: document["similarity"] for document in results}
        pair = case["smoke_expected"]
        passed = scores[pair["higher"]] > scores[pair["lower"]]
        passed_pairs += passed
        rows.append({"id": case["id"], "query": case["query"], "documents": results,
                     "smoke_expected": {**pair, "passed": passed,
                                        "score_difference": scores[pair["higher"]] - scores[pair["lower"]]}})
    report = {
        "schema_version": 1, "kind": "local_model_synthetic_smoke",
        "ok": repeatable and passed_pairs == len(cases),
        "calibrated": False, "scientific_quality_evaluation": False,
        "fixtures_sha256": hashlib.sha256(fixtures).hexdigest(),
        "model": encoder.manifest(),
        "text_policy": {"title": "once", "abstract_characters": ABSTRACT_CHARACTER_LIMIT},
        "checks": {"finite_unit_vectors": True, "dimensions": DIMENSIONS,
                   "query_vectors": len(queries), "passage_vectors": len(passages),
                   "normalization_tolerance": NORM_TOLERANCE, "repeatability_tolerance": REPEAT_TOLERANCE,
                   "repeatable_embeddings": repeatable, "smoke_pairs_passed": passed_pairs,
                   "smoke_pairs_total": len(cases)},
        "queries": rows,
        "limitations": [
            "Synthetic hand-authored smoke_expected pairs test basic installation behaviour, not corpus accuracy.",
            "Adversarial_unvalidated scores have no reference labels and do not determine smoke pass/fail.",
            "Similarity is not a relevance probability; no threshold is selected or calibrated by this check.",
            "The check does not evaluate NMF, trend ranking, scientific claims, or generalization to new domains.",
            "Repeatability is checked twice in one process with the same model and input order.",
            "Timing includes this check only; process peak RSS includes lifetime allocations, not a model-only delta.",
        ],
    }
    report["report_fingerprint"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    report["measurements"] = {"model_load_wall_seconds": round(load_seconds, 6),
                              "inference_wall_seconds": round(inference_seconds, 6),
                              "repeat_inference_wall_seconds": round(repeat_seconds, 6),
                              "check_wall_seconds": round(perf_counter() - started, 6),
                              "process_cpu_seconds": round(process_time() - cpu_started, 6),
                              "process_peak_rss_bytes": _peak_rss_bytes(),
                              "platform": platform.system(), "architecture": platform.machine(),
                              "python": platform.python_version()}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Каталог уже установленной локальной модели")
    parser.add_argument("--output", type=Path, help="Новый JSON отчёт; по умолчанию печатается в stdout")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            if args.output.exists():
                raise AnalysisInputError("Отчёт уже существует; выберите новое имя.")
            if args.output.suffix.casefold() != ".json":
                raise AnalysisInputError("Отчёт сохраняется в файл .json.")
        report = check_local_model(model_dir=args.model_dir)
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            export_result(report, args.output, protected_paths=[CASES_PATH])
            print(json.dumps({"ok": report["ok"], "output": str(args.output),
                              "smoke_pairs_passed": report["checks"]["smoke_pairs_passed"],
                              "scientific_quality_evaluation": False}, ensure_ascii=False))
        return 0 if report["ok"] else 1
    except ImportError:
        print("Установите зависимости: python -m pip install -r requirements/semantic.lock", file=sys.stderr)
        return 2
    except (AnalysisInputError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
