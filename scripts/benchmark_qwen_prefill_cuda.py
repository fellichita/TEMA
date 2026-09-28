"""Compare bounded Qwen prefill modes on the actual CUDA host.

Run after installing the pinned model and onnxruntime-gpu::

    python -m scripts.benchmark_qwen_prefill_cuda --compare-split --repeats 3 --warmups 1

Only a real CUDA provider is accepted.  The report records both timing and
the existing answer-quality gate; it never changes production configuration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

from app.pilot.local_llm import LocalInstructModel, load_spec
from app.runtime.qwen_cuda import (
    PREFILL_CHUNK_VARIABLE, SPLIT_PREFILL_VARIABLE, benchmarkable_prefill_chunks,
)
from scripts.benchmark_local_llm_cuda import run_provider, smoke_cases


@contextmanager
def _environment_override(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def compare_prefill_chunks(*, chunks: tuple[int, ...] | None = None,
                           repeats: int = 3, warmups: int = 1,
                           compare_split: bool = False,
                           model_factory: Callable[[], Any] = LocalInstructModel) -> dict[str, Any]:
    """Benchmark each CUDA mode with identical prompts and a strict answer gate.

    A mode is eligible for a future opt-in only when the two synthetic cases
    pass admission and their output text exactly matches 128/split0 (or the
    pinned chunk if that changes). This is stricter than valid JSON alone.
    """
    pinned = load_spec()["prefill_chunk"]
    allowed = benchmarkable_prefill_chunks(pinned)
    selected = allowed if chunks is None else chunks
    if (not selected or pinned not in selected or len(selected) != len(set(selected))
            or any(type(chunk) is not int or chunk not in allowed for chunk in selected)
            or type(repeats) is not int or not 1 <= repeats <= 10
            or type(warmups) is not int or not 0 <= warmups <= 3
            or type(compare_split) is not bool):
        raise ValueError("Invalid CUDA prefill comparison options")
    ordered = (pinned, *(chunk for chunk in selected if chunk != pinned))
    cases = smoke_cases()
    reports: dict[str, dict[str, Any]] = {}
    modes = tuple((chunk, split) for chunk in ordered for split in ((False, True) if compare_split else (False,)))
    def key(chunk: int, split: bool) -> str:
        return f"{chunk}/split{int(split)}" if compare_split else str(chunk)

    for chunk, split in modes:
        with _environment_override(PREFILL_CHUNK_VARIABLE, str(chunk)):
            with _environment_override(SPLIT_PREFILL_VARIABLE, str(int(split))):
                report = run_provider("cuda", cases, repeats=repeats, warmups=warmups,
                                      model_factory=model_factory)
        observed = {row.get("prefill_chunk_observed") for row in report["measurements"]}
        if observed != {chunk}:
            raise RuntimeError(
                f"Requested prefill chunk {chunk}, but observed {sorted(str(value) for value in observed)}. "
                "Check the production Qwen prefill hook before using this benchmark."
            )
        if split and any(row.get("prefill_logits_step_count") != 1
                         or row.get("prefill_logits_step_tokens") != 1
                         for row in report["measurements"]):
            raise RuntimeError(
                "Requested split prefill, but the final prefill logits pass did not contain exactly one token. "
                "Check the production Qwen split prefill hook before using this benchmark."
            )
        reports[key(chunk, split)] = report
    baseline_key = key(pinned, False)
    baseline = reports[baseline_key]
    baseline_hashes = [row["answer_sha256"] for row in baseline["measurements"]]
    baseline_cases = [row["case_id"] for row in baseline["measurements"]]
    comparisons = []
    for chunk, split in modes:
        run_key = key(chunk, split)
        report = reports[run_key]
        same_cases = [row["case_id"] for row in report["measurements"]] == baseline_cases
        same_answers = same_cases and [row["answer_sha256"] for row in report["measurements"]] == baseline_hashes
        comparisons.append({
            "run_key": run_key,
            "chunk": chunk,
            "split_prefill": split,
            "median_total_speedup_vs_pinned": round(
                baseline["median_total_seconds"] / report["median_total_seconds"], 3)
                if report["median_total_seconds"] > 0 else None,
            "median_prefill_speedup_vs_pinned": round(
                baseline["median_prefill_seconds"] / report["median_prefill_seconds"], 3)
                if report["median_prefill_seconds"] > 0 else None,
            "exact_answers_as_pinned": same_answers,
            "quality_pass": report["quality_pass"],
            "eligible_for_opt_in": same_answers and report["quality_pass"] and baseline["quality_pass"],
        })
    return {
        "benchmark": "qwen-cuda-prefill-synthetic-smoke-v1",
        "provider": "CUDAExecutionProvider",
        "pinned_chunk": pinned,
        "baseline_key": baseline_key,
        "candidate_chunks": list(ordered),
        "compare_split": compare_split,
        "limitations": "Synthetic answer checks only; measure on the target RTX and run a real corpus before enabling a chunk.",
        "runs": reports,
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", help="Comma-separated candidate sizes, including the pinned size")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--compare-split", action="store_true",
                        help="Compare each chunk with split prefill disabled and enabled")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args(argv)
    try:
        chunks = None if args.chunks is None else tuple(int(part.strip()) for part in args.chunks.split(","))
        report = compare_prefill_chunks(chunks=chunks, repeats=args.repeats, warmups=args.warmups,
                                        compare_split=args.compare_split)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if all(item["eligible_for_opt_in"] for item in report["comparisons"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
