"""The CUDA deployment check runs offline here with a deterministic model."""

from __future__ import annotations

import json
import hashlib
import os
from types import SimpleNamespace

from app.pilot.contracts import SCOPE_RULE_VERSION
from app.pilot.evidence import label_system_prompt, scope_is_anchored
from app.pilot.local_llm import LocalCompletion
from app.runtime.inference import PROVIDER_VARIABLE
from scripts.benchmark_local_llm_cuda import (
    BENCHMARK_ENGLISH_SCOPE, assess_answer, benchmark, compare_runs, framed_prompt, measure_case, run_provider,
    smoke_cases,
)


def response(case_id: str, *, quote: str | None = None, revision_id: str | None = None,
             in_scope: bool | None = None) -> str:
    case = next(item for item in smoke_cases() if item.case_id == case_id)
    document = case.documents[0]
    scope = case.expected_in_scope if in_scope is None else in_scope
    support = ([{"revision_id": revision_id or document.revision_id,
                 "field": "abstract", "quote": quote or document.abstract.split(". ")[0] + "."}]
               if scope else [])
    return json.dumps({"candidates": [{
        "candidate_id": case.candidate_id,
        "label": "Фотонный вычислительный контур",
        "definition": "Оптический механизм для нейронного вычисления.",
        "phrase_number": 1,
        "specificity": "specific_technology",
        "in_scope": scope,
        "scope_support": support,
        "scope_reason": "Проверена роль оптики в вычислении.",
    }]}, ensure_ascii=False)


class FakeModel:
    def __init__(self, *, corrupt_cuda: bool = False):
        requested = os.environ.get(PROVIDER_VARIABLE)
        self.provider = "CUDAExecutionProvider" if requested == "cuda" else "CPUExecutionProvider"
        self.corrupt_cuda = corrupt_cuda
        self.closed = False

    def prompt_tokens(self, system, user, prefix):
        return list(range(120))

    def _step(self, tokens, offset, cache, *, need_logits=True):
        return None, cache

    def generate(self, *, system, user, max_new_tokens, stop_at_json, answer_prefix):
        assert stop_at_json and answer_prefix.startswith('{"candidates":')
        ids = self.prompt_tokens(system, user, answer_prefix)
        self._step(ids[:100], 0, {}, need_logits=False)
        self._step(ids[100:], 100, {}, need_logits=True)
        self._step([1], len(ids), {})
        case_id = ("photonic_interconnect_decoy" if "Photonic interconnects" in user
                   else "photonic_computation")
        answer = response(case_id)
        if self.corrupt_cuda and self.provider == "CUDAExecutionProvider":
            case = next(item for item in smoke_cases() if item.case_id == case_id)
            answer = response(case_id, in_scope=not case.expected_in_scope)
        return LocalCompletion(answer, len(ids), 40, False)

    def close(self):
        self.closed = True


def test_both_smoke_documents_reach_the_real_scope_naming_stage():
    plan = SimpleNamespace(english_query=BENCHMARK_ENGLISH_SCOPE, synonyms=(), subdirections=())
    for case in smoke_cases():
        documents = tuple((None, SimpleNamespace(title=item.title, abstract=item.abstract))
                          for item in case.documents)
        assert scope_is_anchored(documents, plan, rule_version=SCOPE_RULE_VERSION)


def test_benchmark_uses_unchanged_production_label_instruction():
    expected = {
        True: (1762, "a59e1e6a36e6dd5c57c027782ce57a2461e8b9e0ef62bb9a37bebaf0dbc73cf1"),
        False: (2071, "a96981f14723a2116f30a830095693f93cd589347e1135c4c17055e86584bc4e"),
    }
    for local, (length, digest) in expected.items():
        prompt = label_system_prompt(local)
        assert len(prompt) == length
        assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == digest
    assert framed_prompt(smoke_cases()[0])[0] == label_system_prompt(local=True)


def test_quality_gate_rejects_false_scope_and_fabricated_evidence():
    positive, decoy = smoke_cases()
    assert assess_answer(positive, response(positive.case_id)) == (True, ())
    assert assess_answer(decoy, response(decoy.case_id)) == (True, ())
    passed, issues = assess_answer(decoy, response(decoy.case_id, in_scope=True))
    assert not passed and "scope_misclassified" in issues
    passed, issues = assess_answer(positive, response(positive.case_id, quote="not in source"))
    assert not passed and "quotation_not_in_claimed_field" in issues
    passed, issues = assess_answer(positive, response(positive.case_id, revision_id="f" * 64))
    assert not passed and "unknown_revision_id" in issues
    duplicated = json.loads(response(positive.case_id))
    duplicated["candidates"][0]["scope_support"] *= 2
    passed, issues = assess_answer(positive, json.dumps(duplicated))
    assert not passed and "duplicate_document_support" in issues


def test_quality_gate_rejects_bad_schema_and_changed_candidate_identity():
    case = smoke_cases()[0]
    assert assess_answer(case, "not a JSON object") == (False, ("invalid_json",))
    document = json.loads(response(case.case_id))
    document["candidates"][0]["candidate_id"] = "invented"
    assert assess_answer(case, json.dumps(document)) == (False, ("candidate_membership_changed",))
    del document["candidates"][0]["phrase_number"]
    assert assess_answer(case, json.dumps(document)) == (False, ("invalid_local_label_schema",))


def test_measurement_accounts_for_prompt_prefill_decode_and_checks_schema():
    row = measure_case(FakeModel(), smoke_cases()[0])
    assert row["quality_pass"]
    assert row["failed_answer"] is None
    assert row["prompt_tokens"] == 120 and row["completion_tokens"] == 40
    assert row["prefill_computed_tokens"] == 120
    assert row["prefill_reused_tokens"] == 0
    assert row["decode_steps"] == 1
    assert row["total_seconds"] >= row["prefill_seconds"]
    assert row["total_seconds"] >= row["decode_seconds"]
    assert len(row["answer_sha256"]) == 64


def test_measurement_accounts_for_reused_system_prefix_by_absolute_offset():
    class CachedPrefixModel(FakeModel):
        def generate(self, *, system, user, max_new_tokens, stop_at_json, answer_prefix):
            ids = self.prompt_tokens(system, user, answer_prefix)
            self._step(ids[40:100], 40, {}, need_logits=False)
            self._step(ids[100:], 100, {}, need_logits=True)
            self._step([1], len(ids), {})
            return LocalCompletion(response("photonic_computation"), len(ids), 40, False)

    row = measure_case(CachedPrefixModel(), smoke_cases()[0])
    assert row["quality_pass"]
    assert row["prefill_computed_tokens"] == 80
    assert row["prefill_reused_tokens"] == 40
    assert row["decode_steps"] == 1


def test_requested_cuda_must_actually_be_selected(monkeypatch):
    monkeypatch.setenv(PROVIDER_VARIABLE, "cpu")

    class FellBack(FakeModel):
        def __init__(self):
            super().__init__()
            self.provider = "CPUExecutionProvider"

    try:
        run_provider("cuda", smoke_cases(), repeats=1, warmups=0, model_factory=FellBack)
    except RuntimeError as error:
        assert "CUDAExecutionProvider" in str(error)
    else:
        raise AssertionError("CUDA benchmark silently accepted CPU fallback")
    assert os.environ[PROVIDER_VARIABLE] == "cpu"


def test_cpu_comparison_reports_exact_drift_and_quality_regression():
    report = benchmark(compare_cpu=True, model_factory=lambda: FakeModel(corrupt_cuda=True))
    assert report["selected"]["quality_pass"] is False
    assert report["selected"]["measurements"][0]["failed_answer"] is not None
    assert report["cpu"]["quality_pass"] is True
    comparison = report["comparison"]
    assert comparison["same_answer_by_case"][0]["identical_text"] is False
    assert comparison["median_total_speedup_vs_cpu"] is not None


def test_comparison_refuses_different_case_sequences():
    cuda = {"median_total_seconds": 1.0, "measurements": [{"case_id": "a"}]}
    cpu = {"median_total_seconds": 2.0, "measurements": [{"case_id": "b"}]}
    try:
        compare_runs(cuda, cpu)
    except ValueError as error:
        assert "same order" in str(error)
    else:
        raise AssertionError("Misaligned cases would produce an invalid speedup")
