"""GPU prefill selection and benchmark gating without GPU hardware."""

from __future__ import annotations

import os

import pytest

from app.pilot.local_llm import LocalCompletion
from app.runtime.inference import PROVIDER_VARIABLE
from app.runtime.qwen_cuda import (
    PREFILL_CHUNK_VARIABLE, benchmarkable_prefill_chunks, qwen_prefill_chunk, qwen_split_prefill,
)
from scripts.benchmark_local_llm_cuda import smoke_cases
from scripts.benchmark_qwen_prefill_cuda import SPLIT_PREFILL_VARIABLE, compare_prefill_chunks
from tests.test_benchmark_local_llm_cuda import FakeModel, response
from tests.test_pilot_local_llm import build


class ChunkAwareFake(FakeModel):
    def prompt_tokens(self, system, user, prefix):
        return list(range(700))

    def generate(self, *, system, user, max_new_tokens, stop_at_json, answer_prefix):
        assert stop_at_json
        ids = self.prompt_tokens(system, user, answer_prefix)
        chunk = qwen_prefill_chunk(128, self.provider)
        split = qwen_split_prefill(self.provider)
        prefill_end = len(ids) - 1 if split else len(ids)
        for start in range(0, prefill_end, chunk):
            self._step(ids[start:min(start + chunk, prefill_end)], start, {},
                       need_logits=not split and start + chunk >= prefill_end)
        if split:
            self._step(ids[-1:], len(ids) - 1, {}, need_logits=True)
        self._step([1], len(ids), {})
        case_id = ("photonic_interconnect_decoy" if "Photonic interconnects" in user
                   else "photonic_computation")
        answer = response(case_id)
        return LocalCompletion(answer, len(ids), 40, False)


def test_cuda_prefill_defaults_to_256_and_cpu_keeps_pinned_chunk(monkeypatch):
    monkeypatch.delenv(PREFILL_CHUNK_VARIABLE, raising=False)
    monkeypatch.delenv(SPLIT_PREFILL_VARIABLE, raising=False)
    assert benchmarkable_prefill_chunks(128) == (128, 256, 512)
    assert qwen_prefill_chunk(128, "CUDAExecutionProvider") == 256
    assert qwen_split_prefill("CUDAExecutionProvider")
    assert not qwen_split_prefill("CPUExecutionProvider")
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "128")
    monkeypatch.setenv(SPLIT_PREFILL_VARIABLE, "0")
    assert qwen_prefill_chunk(128, "CUDAExecutionProvider") == 128
    assert not qwen_split_prefill("CUDAExecutionProvider")
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "512")
    assert qwen_prefill_chunk(128, "CUDAExecutionProvider") == 512
    assert qwen_prefill_chunk(128, "CPUExecutionProvider") == 128
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "2048")
    assert qwen_prefill_chunk(128, "CUDAExecutionProvider") == 128
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "broken")
    assert qwen_prefill_chunk(128, "CUDAExecutionProvider") == 128
    monkeypatch.setenv(SPLIT_PREFILL_VARIABLE, "broken")
    assert not qwen_split_prefill("CUDAExecutionProvider")


def test_production_decoder_uses_selected_chunk_on_cuda(monkeypatch):
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "256")
    model, session, _ = build([11])
    model.provider = "CUDAExecutionProvider"  # Simulate only provider selection, not CUDA arithmetic.
    model.spec["vocabulary_size"] = session.spec["vocabulary_size"] = 100
    model.spec["tokens"]["stop"] = [11]
    words = " ".join(f"w{index}" for index in range(300))
    model.generate(system="s", user=words, max_new_tokens=1)
    assert session.feeds[0]["input_ids"].shape[1] == 256


def test_comparison_checks_actual_chunk_exact_answers_and_restores_environment(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cpu")
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "custom")
    monkeypatch.setenv(SPLIT_PREFILL_VARIABLE, "1")
    report = compare_prefill_chunks(repeats=1, warmups=0, model_factory=ChunkAwareFake)
    assert report["candidate_chunks"] == [128, 256, 512]
    assert all(item["quality_pass"] and item["exact_answers_as_pinned"]
               and item["eligible_for_opt_in"] for item in report["comparisons"])
    assert [report["runs"][str(chunk)]["measurements"][0]["prefill_chunk_observed"]
            for chunk in (128, 256, 512)] == [128, 256, 512]
    assert report["runs"]["128"]["measurements"][0]["prefill_logits_step_tokens"] == 60
    assert os.environ[PROVIDER_VARIABLE] == "cpu"
    assert os.environ[PREFILL_CHUNK_VARIABLE] == "custom"
    assert os.environ[SPLIT_PREFILL_VARIABLE] == "1"


def test_split_comparison_checks_final_logits_pass_and_restores_environment(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cpu")
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "custom")
    monkeypatch.setenv(SPLIT_PREFILL_VARIABLE, "custom")
    report = compare_prefill_chunks(chunks=(128, 256), compare_split=True,
                                    repeats=1, warmups=0, model_factory=ChunkAwareFake)
    assert report["baseline_key"] == "128/split0"
    assert list(report["runs"]) == ["128/split0", "128/split1", "256/split0", "256/split1"]
    assert all(row["eligible_for_opt_in"] for row in report["comparisons"])
    for run_key in ("128/split1", "256/split1"):
        assert all(row["prefill_logits_step_count"] == 1
                   and row["prefill_logits_step_tokens"] == 1
                   for row in report["runs"][run_key]["measurements"])
    assert os.environ[PROVIDER_VARIABLE] == "cpu"
    assert os.environ[PREFILL_CHUNK_VARIABLE] == "custom"
    assert os.environ[SPLIT_PREFILL_VARIABLE] == "custom"


def test_split_comparison_fails_quality_gate_and_restores_environment(monkeypatch):
    class MisclassifiesSplit(ChunkAwareFake):
        def generate(self, *, system, user, max_new_tokens, stop_at_json, answer_prefix):
            completion = super().generate(system=system, user=user, max_new_tokens=max_new_tokens,
                                          stop_at_json=stop_at_json, answer_prefix=answer_prefix)
            if os.getenv(SPLIT_PREFILL_VARIABLE) == "1":
                case_id = ("photonic_interconnect_decoy" if "Photonic interconnects" in user
                           else "photonic_computation")
                case = next(item for item in smoke_cases()
                            if item.case_id == case_id)
                return LocalCompletion(response(case_id, in_scope=not case.expected_in_scope),
                                       completion.prompt_tokens, completion.completion_tokens, False)
            return completion

    monkeypatch.delenv(PREFILL_CHUNK_VARIABLE, raising=False)
    monkeypatch.delenv(SPLIT_PREFILL_VARIABLE, raising=False)
    report = compare_prefill_chunks(chunks=(128,), compare_split=True,
                                    repeats=1, warmups=0, model_factory=MisclassifiesSplit)
    baseline, candidate = report["comparisons"]
    assert baseline["quality_pass"] and baseline["eligible_for_opt_in"]
    assert not candidate["quality_pass"]
    assert not candidate["exact_answers_as_pinned"]
    assert not candidate["eligible_for_opt_in"]
    assert PREFILL_CHUNK_VARIABLE not in os.environ
    assert SPLIT_PREFILL_VARIABLE not in os.environ


def test_split_comparison_rejects_text_drift_even_if_schema_remains_valid():
    class ReformatsSplit(ChunkAwareFake):
        def generate(self, *, system, user, max_new_tokens, stop_at_json, answer_prefix):
            completion = super().generate(system=system, user=user, max_new_tokens=max_new_tokens,
                                          stop_at_json=stop_at_json, answer_prefix=answer_prefix)
            if os.getenv(SPLIT_PREFILL_VARIABLE) == "1":
                return LocalCompletion(completion.text + " ", completion.prompt_tokens,
                                       completion.completion_tokens, False)
            return completion

    report = compare_prefill_chunks(chunks=(128,), compare_split=True,
                                    repeats=1, warmups=0, model_factory=ReformatsSplit)
    candidate = report["comparisons"][1]
    assert candidate["quality_pass"]
    assert not candidate["exact_answers_as_pinned"]
    assert not candidate["eligible_for_opt_in"]


def test_split_comparison_restores_environment_when_model_fails(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cpu")
    monkeypatch.setenv(PREFILL_CHUNK_VARIABLE, "unchanged")
    monkeypatch.setenv(SPLIT_PREFILL_VARIABLE, "unchanged")

    def broken_factory():
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        compare_prefill_chunks(chunks=(128,), compare_split=True, repeats=1,
                               warmups=0, model_factory=broken_factory)
    assert os.environ[PROVIDER_VARIABLE] == "cpu"
    assert os.environ[PREFILL_CHUNK_VARIABLE] == "unchanged"
    assert os.environ[SPLIT_PREFILL_VARIABLE] == "unchanged"


def test_comparison_refuses_an_unwired_or_unsafe_chunk():
    with pytest.raises(RuntimeError, match="production Qwen prefill hook"):
        compare_prefill_chunks(chunks=(128,), repeats=1, warmups=0, model_factory=FakeModel)
    with pytest.raises(ValueError, match="Invalid CUDA prefill"):
        compare_prefill_chunks(chunks=(256,), repeats=1, warmups=0, model_factory=ChunkAwareFake)
    with pytest.raises(ValueError, match="Invalid CUDA prefill"):
        compare_prefill_chunks(chunks=(128, 1024), repeats=1, warmups=0, model_factory=ChunkAwareFake)
