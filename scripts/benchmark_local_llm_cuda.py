"""Measure the pinned local model on a CUDA server without weakening admission.

Run from the repository root after installing the local model and CUDA runtime::

    python -m scripts.benchmark_local_llm_cuda --provider cuda --compare-cpu

The bundled cases are synthetic smoke checks, not a scientific accuracy set.
They exercise the production local answer schema, exact source quotations, and
one deliberately misleading search result. The reported provider is the one
ONNX Runtime actually selected. A requested CUDA run refuses CPU fallback.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from typing import Any, Callable, Iterator

from pydantic import ValidationError

from app.pilot.evidence import LOCAL_LABEL_EXAMPLE, LocalLabelBatch, label_system_prompt
from app.pilot.llm import LlmError
from app.pilot.local_client import PROMPT_SUFFIX, answer_prefix, compact_schema, first_json_object
from app.pilot.local_llm import LocalCompletion, LocalInstructModel
from app.runtime.inference import PROVIDER_VARIABLE


BENCHMARK_ENGLISH_SCOPE = "photonic neuromorphic computing"


@dataclass(frozen=True)
class SourceDocument:
    revision_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    requested_scope: str
    documents: tuple[SourceDocument, ...]
    phrases: tuple[str, ...]
    expected_in_scope: bool

    @property
    def candidate_id(self) -> str:
        return "c0"


def _document(title: str, abstract: str) -> SourceDocument:
    revision_id = hashlib.sha256((title + "\n" + abstract).encode("utf-8")).hexdigest()
    return SourceDocument(revision_id, title, abstract)


def smoke_cases() -> tuple[BenchmarkCase, ...]:
    """A mechanism and a lexical decoy, each with one archived-style record."""
    return (
        BenchmarkCase(
            "photonic_computation",
            "Photonic neuromorphic computing: an optical circuit must perform a neural computation.",
            (_document(
                "Photonic neural networks for optical matrix computation",
                "For photonic neuromorphic computing, an integrated photonic neural network "
                "performs weighted matrix multiplication with optical signals. Measurements "
                "demonstrate the computing operation in a programmable optical circuit."),),
            ("photonic neural networks", "optical matrix computation"),
            True,
        ),
        BenchmarkCase(
            "photonic_interconnect_decoy",
            "Photonic neuromorphic computing: an optical circuit must perform a neural computation.",
            (_document(
                "Photonic interconnects for neural network accelerators",
                "The target application is photonic neuromorphic computing; this paper studies "
                "only optical data transport. An optical interconnect transfers activations "
                "between electronic processors in a neural network accelerator. All matrix "
                "multiplication and nonlinear activation are performed electronically; the "
                "photonic link transports the data but does not execute neural computation."),),
            ("photonic interconnects", "neural network accelerators"),
            False,
        ),
    )


def framed_prompt(case: BenchmarkCase) -> tuple[str, str, str]:
    """Use the production local instruction, schema framing and answer prefix."""
    system = label_system_prompt(local=True)
    material = {
        "requested_scope": case.requested_scope,
        "english_scope": BENCHMARK_ENGLISH_SCOPE,
        "excluded_scope": [],
        "clusters": [{
            "candidate_id": case.candidate_id,
            "keyword_label": "photonic neural network",
            "documents": [
                {"revision_id": document.revision_id, "title": document.title,
                 "abstract": document.abstract}
                for document in case.documents
            ],
            "total_studies": len(case.documents),
        }],
    }
    choices = "; ".join(f"{index}) {phrase}" for index, phrase in enumerate(case.phrases, 1))
    hint = LOCAL_LABEL_EXAMPLE + f"\nFor cluster c0, phrase_number chooses one of: {choices}"
    user = json.dumps(material, ensure_ascii=False) + "\n" + PROMPT_SUFFIX + compact_schema(LocalLabelBatch) + "\n" + hint
    return system, user, answer_prefix(LocalLabelBatch)


def assess_answer(case: BenchmarkCase, answer: str) -> tuple[bool, tuple[str, ...]]:
    """Apply schema, membership, title-choice and exact-quotation checks.

    These checks are a small deployment gate; the full analysis still performs
    its own frozen-candidate and scientific evidence verification.
    """
    try:
        parsed = json.loads(first_json_object(answer))
        batch = LocalLabelBatch.model_validate(parsed)
    except LlmError:
        return False, ("invalid_json",)
    except (ValueError, TypeError, ValidationError, RecursionError):
        return False, ("invalid_local_label_schema",)

    issues: list[str] = []
    if len(batch.candidates) != 1 or batch.candidates[0].candidate_id != case.candidate_id:
        return False, ("candidate_membership_changed",)
    proposal = batch.candidates[0]
    if proposal.phrase_number > len(case.phrases):
        issues.append("phrase_not_offered")
    if proposal.in_scope != case.expected_in_scope:
        issues.append("scope_misclassified")
    if case.expected_in_scope and proposal.specificity != "specific_technology":
        issues.append("specific_mechanism_missing")

    sources = {document.revision_id: document for document in case.documents}
    cited: set[str] = set()
    if len({support.revision_id for support in proposal.scope_support}) != len(proposal.scope_support):
        issues.append("duplicate_document_support")
    for support in proposal.scope_support:
        document = sources.get(support.revision_id)
        if document is None:
            issues.append("unknown_revision_id")
            continue
        if support.quote not in getattr(document, support.field):
            issues.append("quotation_not_in_claimed_field")
        else:
            cited.add(support.revision_id)
    if proposal.in_scope and cited != set(sources):
        issues.append("missing_document_support")
    return not issues, tuple(dict.fromkeys(issues))


@contextmanager
def _timed_steps(model: Any, prompt_length: int) -> Iterator[tuple[dict[str, float], dict[str, int], list[int], list[int]]]:
    """Account for the synchronous ONNX passes without changing their inputs."""
    original_prompt = model.prompt_tokens
    original_step = model._step
    phases = {"prompt_seconds": 0.0, "prefill_seconds": 0.0, "decode_seconds": 0.0}
    counts = {"decode_steps": 0, "prefill_tokens": 0}
    prefill_lengths: list[int] = []
    prefill_logits_lengths: list[int] = []

    def prompt(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter()
        try:
            return original_prompt(*args, **kwargs)
        finally:
            phases["prompt_seconds"] += perf_counter() - started

    def step(tokens: Any, offset: int, cache: Any, **kwargs: Any) -> Any:
        # A warmed model may reuse completed system-prefix KV blocks. The
        # absolute position still identifies the phase; counting only calls
        # made during this run would misclassify decode as prefill.
        phase = "prefill_seconds" if offset < prompt_length else "decode_seconds"
        if phase == "prefill_seconds":
            prefill_lengths.append(len(tokens))
            counts["prefill_tokens"] += len(tokens)
            if kwargs.get("need_logits", True):
                prefill_logits_lengths.append(len(tokens))
        started = perf_counter()
        try:
            return original_step(tokens, offset, cache, **kwargs)
        finally:
            phases[phase] += perf_counter() - started
            if phase == "decode_seconds":
                counts["decode_steps"] += 1

    model.prompt_tokens = prompt
    model._step = step
    try:
        yield phases, counts, prefill_lengths, prefill_logits_lengths
    finally:
        model.prompt_tokens = original_prompt
        model._step = original_step


def measure_case(model: Any, case: BenchmarkCase) -> dict[str, Any]:
    system, user, prefix = framed_prompt(case)
    expected_prompt_tokens = len(model.prompt_tokens(system, user, prefix))
    with _timed_steps(model, expected_prompt_tokens) as (phases, counts, prefill_lengths, prefill_logits_lengths):
        started = perf_counter()
        completion: LocalCompletion = model.generate(
            system=system, user=user, max_new_tokens=1024,
            stop_at_json=True, answer_prefix=prefix,
        )
        total_seconds = perf_counter() - started
    passed, issues = assess_answer(case, completion.text)
    # The first output token comes from prefill. Decode steps count actual
    # single-token ONNX passes, including the step that predicts an end token.
    decode_seconds = phases["decode_seconds"]
    prefill_seconds = phases["prefill_seconds"]
    return {
        "case_id": case.case_id,
        "provider": model.provider,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "stopped_at_limit": completion.stopped_at_limit,
        "total_seconds": round(total_seconds, 6),
        "prompt_seconds": round(phases["prompt_seconds"], 6),
        "prefill_seconds": round(prefill_seconds, 6),
        "decode_seconds": round(decode_seconds, 6),
        "decode_steps": counts["decode_steps"],
        "prefill_computed_tokens": counts["prefill_tokens"],
        "prefill_reused_tokens": completion.prompt_tokens - counts["prefill_tokens"],
        "prefill_chunk_observed": max(prefill_lengths, default=None),
        "prefill_steps": len(prefill_lengths),
        "prefill_logits_step_count": len(prefill_logits_lengths),
        "prefill_logits_step_tokens": prefill_logits_lengths[0] if len(prefill_logits_lengths) == 1 else None,
        "other_seconds": round(max(0.0, total_seconds - sum(phases.values())), 6),
        "prefill_tokens_per_second": round(counts["prefill_tokens"] / prefill_seconds, 3)
            if prefill_seconds > 0 else None,
        "decode_tokens_per_second": round(counts["decode_steps"] / decode_seconds, 3)
            if decode_seconds > 0 else None,
        "quality_pass": passed and not completion.stopped_at_limit,
        "quality_issues": list(issues) + (["output_token_limit"] if completion.stopped_at_limit else []),
        "failed_answer": completion.text if not passed or completion.stopped_at_limit else None,
        "answer_sha256": hashlib.sha256(completion.text.encode("utf-8")).hexdigest(),
    }


@contextmanager
def _requested_provider(name: str) -> Iterator[None]:
    previous = os.environ.get(PROVIDER_VARIABLE)
    os.environ[PROVIDER_VARIABLE] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PROVIDER_VARIABLE, None)
        else:
            os.environ[PROVIDER_VARIABLE] = previous


def run_provider(provider: str, cases: tuple[BenchmarkCase, ...], *, repeats: int,
                 warmups: int, model_factory: Callable[[], Any] = LocalInstructModel) -> dict[str, Any]:
    expected = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}[provider]
    with _requested_provider(provider):
        started = perf_counter()
        model = model_factory()
        load_seconds = perf_counter() - started
        try:
            if model.provider != expected:
                raise RuntimeError(
                    f"Requested {expected}, but ONNX Runtime selected {model.provider}. "
                    "Install/check the matching CUDA runtime before claiming a GPU benchmark."
                )
            for _ in range(warmups):
                for case in cases:
                    measure_case(model, case)
            measurements = [measure_case(model, case)
                            for _ in range(repeats) for case in cases]
        finally:
            model.close()
            del model
            gc.collect()
    return {
        "requested_provider": provider,
        "actual_provider": expected,
        "model_load_seconds": round(load_seconds, 6),
        "repeats": repeats,
        "warmups": warmups,
        "measurements": measurements,
        "median_total_seconds": round(median(row["total_seconds"] for row in measurements), 6),
        "median_prefill_seconds": round(median(row["prefill_seconds"] for row in measurements), 6),
        "median_decode_seconds": round(median(row["decode_seconds"] for row in measurements), 6),
        "quality_pass": all(row["quality_pass"] for row in measurements),
    }


def compare_runs(cuda: dict[str, Any], cpu: dict[str, Any]) -> dict[str, Any]:
    cuda_rows = cuda["measurements"]
    cpu_rows = cpu["measurements"]
    if [row["case_id"] for row in cuda_rows] != [row["case_id"] for row in cpu_rows]:
        raise ValueError("CPU and CUDA benchmarks must use cases in the same order.")
    return {
        "median_total_speedup_vs_cpu": round(
            cpu["median_total_seconds"] / cuda["median_total_seconds"], 3)
            if cuda["median_total_seconds"] > 0 else None,
        "same_answer_by_case": [
            {"case_id": gpu["case_id"], "identical_text": gpu["answer_sha256"] == host["answer_sha256"],
             "cuda_quality_pass": gpu["quality_pass"], "cpu_quality_pass": host["quality_pass"]}
            for gpu, host in zip(cuda_rows, cpu_rows, strict=True)
        ],
    }


def benchmark(*, provider: str = "cuda", repeats: int = 1, warmups: int = 0,
              compare_cpu: bool = False,
              model_factory: Callable[[], Any] = LocalInstructModel) -> dict[str, Any]:
    if provider not in {"cpu", "cuda"} or repeats < 1 or warmups < 0:
        raise ValueError("Invalid provider, repeats, or warmups.")
    if compare_cpu and provider != "cuda":
        raise ValueError("--compare-cpu requires --provider cuda.")
    cases = smoke_cases()
    selected = run_provider(provider, cases, repeats=repeats, warmups=warmups,
                            model_factory=model_factory)
    result: dict[str, Any] = {
        "case_set": "synthetic-local-label-smoke-v2",
        "model": "onnx-community/Qwen2.5-1.5B-Instruct pinned ONNX",
        "limitations": "Synthetic smoke checks; not a measured precision/recall estimate or full analysis runtime.",
        "selected": selected,
    }
    if compare_cpu:
        cpu = run_provider("cpu", cases, repeats=repeats, warmups=warmups,
                           model_factory=model_factory)
        result["cpu"] = cpu
        result["comparison"] = compare_runs(selected, cpu)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = benchmark(provider=args.provider, repeats=args.repeats,
                           warmups=args.warmups, compare_cpu=args.compare_cpu)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["selected"]["quality_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
