"""Offline E5 throughput and FP32 comparison for a configured NVIDIA server.

Run only after installing the pinned local E5 model and ONNX Runtime GPU::

    python -m scripts.benchmark_e5_cuda --warmups 1 --repeats 3

No model downloads or network calls occur. The benchmark requires the actual
CUDAExecutionProvider and measures direct encode calls without the disk cache.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from statistics import median
import sys
from time import perf_counter
from typing import Any, Callable, Iterator

import numpy as np

from app.pilot.encoder import MultilingualEncoder, validate_vectors
from app.runtime.inference import BATCH_VARIABLE, PROVIDER_VARIABLE


BATCH_SIZES = (8, 16, 32)
_SEEDS = (
    "Optical neural networks perform matrix multiplication using integrated photonic circuits.",
    "A review compares solid-state batteries with liquid electrolytes across long-term cycling.",
    "Многоязычная модель сопоставляет научные статьи о квантовых датчиках и оптических схемах.",
    "Метод анализа публикаций группирует материалы по времени и проверяет источники данных.",
    "La investigación compara sensores ópticos para detectar contaminantes en agua.",
    "Des réseaux neuronaux photoniques calculent des produits matriciels dans un circuit intégré.",
    "Ein lokales Modell analysiert wissenschaftliche Arbeiten über Halbleiter und Energiespeicher.",
    "半導体の研究では光学センサーの性能と測定誤差を比較する。",
)


def fixed_texts() -> tuple[str, ...]:
    """A fixed mixed-length multilingual corpus; no external records or secrets."""
    return tuple(f"{_SEEDS[index % len(_SEEDS)]} Study {index:03d}: "
                 + ("measurement and comparison. " * (index % 5 + 1)) for index in range(128))


@contextmanager
def _cuda_batch(size: int) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in (PROVIDER_VARIABLE, BATCH_VARIABLE)}
    os.environ[PROVIDER_VARIABLE] = "cuda"
    os.environ[BATCH_VARIABLE] = str(size)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_size(size: int, texts: tuple[str, ...], warmups: int, repeats: int,
              model_factory: Callable[[], Any]) -> tuple[dict[str, Any], np.ndarray]:
    with _cuda_batch(size):
        started = perf_counter()
        encoder = model_factory()
        load_seconds = perf_counter() - started
        try:
            provider = getattr(encoder, "provider", None)
            if provider != "CUDAExecutionProvider":
                raise RuntimeError(f"E5 выбрал {provider}, требуется CUDAExecutionProvider.")
            for _ in range(warmups):
                encoder.encode(texts, kind="passage")
            times: list[float] = []
            baseline: np.ndarray | None = None
            repeat_max_abs_diff = 0.0
            for _ in range(repeats):
                started = perf_counter()
                vectors = encoder.encode(texts, kind="passage")
                times.append(perf_counter() - started)
                validate_vectors(vectors, len(texts))
                if baseline is None:
                    baseline = vectors.copy()
                else:
                    difference = np.abs(vectors.astype(np.float64) - baseline.astype(np.float64))
                    repeat_max_abs_diff = max(repeat_max_abs_diff, float(difference.max()))
            assert baseline is not None
            typical = median(times)
            return ({
                "batch_size": size,
                "provider": provider,
                "fingerprint": getattr(encoder, "fingerprint", None),
                "load_seconds": round(load_seconds, 6),
                "times_seconds": [round(value, 6) for value in times],
                "median_seconds": round(typical, 6),
                "texts_per_second": round(len(texts) / typical, 3) if typical > 0 else None,
                "max_abs_diff_within_repeats": repeat_max_abs_diff,
            }, baseline)
        finally:
            del encoder
            gc.collect()


def benchmark(*, warmups: int = 1, repeats: int = 3,
              model_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    if type(warmups) is not int or warmups < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("Число прогревов должно быть неотрицательным, повторов — положительным.")
    if model_factory is None:
        model_factory = MultilingualEncoder
    texts = fixed_texts()
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    rows = []
    baseline = None
    for size in BATCH_SIZES:
        row, vectors = _run_size(size, texts, warmups, repeats, model_factory)
        if baseline is None:
            baseline = vectors
        difference = np.abs(vectors.astype(np.float64) - baseline.astype(np.float64))
        current64 = vectors.astype(np.float64)
        baseline64 = baseline.astype(np.float64)
        cosine = np.sum(current64 * baseline64, axis=1) / (
            np.linalg.norm(current64, axis=1) * np.linalg.norm(baseline64, axis=1))
        row["max_abs_diff_vs_batch_8"] = float(difference.max())
        row["min_cosine_vs_batch_8"] = float(np.clip(cosine, -1.0, 1.0).min())
        rows.append(row)
    return {"benchmark": "e5-cuda-batch-v1", "texts": len(texts), "corpus_sha256": digest,
            "warmups": warmups, "repeats": repeats, "measurements": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Офлайн-замер E5 на NVIDIA CUDA: батчи 8, 16, 32.")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args(argv)
    try:
        report = benchmark(warmups=options.warmups, repeats=options.repeats)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Замер E5 не выполнен: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
