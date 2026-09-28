"""Decoding, pinning and budget behaviour of the offline instruct model.

The weights are 1.8 GB and are not required here: a deterministic stand-in with
the same interface exercises the parts that can be wrong — cache plumbing,
prompt chunking, stop tokens, cancellation and accounting.
"""

import json
from pathlib import Path
from threading import Event
from typing import ClassVar

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.pilot.llm import LlmCancelled, LlmError
from app.pilot.local_client import (
    LocalLlmClient, LocalProviderConfig, _validation_summary, first_json_object,
)
from app.pilot.local_llm import (
    LocalInstructModel, LocalModelError, artifacts_present, load_spec, truncated_json_object, verify_artifacts,
)
from app.runtime.qwen_cuda import qwen_prefill_chunk
from app.runtime.budget import BudgetLimits, BudgetService
from app.sqlite_runtime import sqlite3


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=3)
    confident: bool = Field(strict=True)


class FakeTokenizer:
    """One token per word, and ids that stay stable across a test."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.words: dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self.vocabulary:
            token = 10 + len(self.vocabulary)
            self.vocabulary[word], self.words[token] = token, word
        return self.vocabulary[word]

    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": [self._id(word) for word in text.split() or [""]]})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words.get(token, "?") for token in ids)


class ScriptedSession:
    """Returns the scripted tokens one by one and records every feed it saw."""

    def __init__(self, tokens, spec, *, on_step=None):
        self.tokens = list(tokens)
        self.spec = spec
        self.feeds = []
        self.output_requests = []
        self.on_step = on_step
        self._emitted = 0

    def _names(self, names):
        return [type("Value", (), {"name": name, "type": "tensor(float)"})() for name in names]

    def get_inputs(self):
        past = [f"past_key_values.{index}.{kind}"
                for index in range(self.spec["layers"]) for kind in ("key", "value")]
        return self._names(["input_ids", "attention_mask", "position_ids", *past])

    def get_outputs(self):
        present = [f"present.{index}.{kind}"
                   for index in range(self.spec["layers"]) for kind in ("key", "value")]
        return self._names(["logits", *present])

    def run(self, names, feeds):
        self.feeds.append({key: value.copy() for key, value in feeds.items()})
        self.output_requests.append(tuple(names))
        if self.on_step is not None:
            self.on_step(len(self.feeds))
        length = feeds["input_ids"].shape[1]
        past = feeds["past_key_values.0.key"].shape[2]
        logits = np.zeros((1, length, self.spec["vocabulary_size"]), dtype=np.float32)
        token = self.tokens[min(self._emitted, len(self.tokens) - 1)]
        if "logits" in names:
            self._emitted += 1
        logits[0, -1, token] = 10.0
        present = np.zeros((1, self.spec["key_value_heads"], past + length, self.spec["head_dimension"]),
                           dtype=np.float32)
        return [logits if name == "logits" else present for name in names]


def build(tokens, *, on_step=None):
    spec = load_spec()
    tokenizer = FakeTokenizer()
    session = ScriptedSession(tokens, spec, on_step=on_step)
    return LocalInstructModel(session=session, tokenizer=tokenizer), session, tokenizer


def test_prompt_is_fed_in_bounded_chunks_and_every_step_carries_the_cache():
    spec = load_spec()
    words = " ".join(f"w{index}" for index in range(spec["prefill_chunk"] + 5))
    model, session, _ = build([spec["tokens"]["stop"][0]])
    model.generate(system="s", user=words, max_new_tokens=8)
    lengths = [feed["input_ids"].shape[1] for feed in session.feeds]
    assert lengths[0] == spec["prefill_chunk"], "длинный запрос идёт частями, а не одним куском"
    assert sum(lengths) == len(model.prompt_tokens("s", words))
    assert lengths == [spec["prefill_chunk"], len(model.prompt_tokens("s", words)) - spec["prefill_chunk"]]
    assert "logits" not in session.output_requests[0]
    assert "logits" in session.output_requests[-1]
    assert sum("logits" in request for request in session.output_requests) == 1
    offset = 0
    for feed in session.feeds:
        length = feed["input_ids"].shape[1]
        # Each pass sees exactly the positions that follow the cache it was given.
        assert feed["past_key_values.0.key"].shape[2] == offset
        assert feed["attention_mask"].shape[1] == offset + length
        assert feed["position_ids"].tolist() == [list(range(offset, offset + length))]
        offset += length


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
def test_repeated_system_instruction_is_tokenized_once_and_cache_is_bounded(provider):
    stop = load_spec()["tokens"]["stop"][0]
    model, _session, tokenizer = build([stop])
    model.provider = provider
    original_encode = tokenizer.encode
    seen = []

    def counted_encode(text, add_special_tokens=False):
        seen.append(text)
        return original_encode(text, add_special_tokens=add_special_tokens)

    tokenizer.encode = counted_encode
    system = "Check the scope and every cited study before naming a trend."
    model.generate(system=system, user="first paper", max_new_tokens=2)
    model.generate(system=system, user="second paper", max_new_tokens=2)
    assert seen.count(system) == 1

    # Callers receive a fresh list: changing it cannot corrupt a later prompt.
    tokens = model._system_prompt_tokens(system)
    tokens[0] = -1
    assert model._system_prompt_tokens(system)[0] == model.spec["tokens"]["start"]
    changed = system + " New instruction."
    model.generate(system=changed, user="third paper", max_new_tokens=2)
    assert seen.count(changed) == 1
    assert model._system_tokens[0] == changed
    model.close()
    assert model._system_tokens is None


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
def test_repeated_system_instruction_reuses_only_complete_identical_prefill_chunks(provider):
    spec = load_spec()
    system = " ".join(f"rule{index}" for index in range(2 * spec["prefill_chunk"] + 7))
    stop = spec["tokens"]["stop"][0]
    model, session, tokenizer = build([])
    model.provider = provider
    session.tokens = [tokenizer._id("ok"), tokenizer._id("done"), stop]
    model.generate(system=system, user="first material", max_new_tokens=4)
    first_passes = len(session.feeds)
    prefix = model._system_prompt_tokens(system)
    chunk = qwen_prefill_chunk(spec["prefill_chunk"], provider)
    reused = len(prefix) // chunk * chunk
    assert reused >= spec["prefill_chunk"]

    session._emitted = 0
    second = model.generate(system=system, user="different material", max_new_tokens=4)
    second_feeds = session.feeds[first_passes:]
    prompt = model.prompt_tokens(system, "different material")
    submitted = [token for feed in second_feeds for token in feed["input_ids"][0]]
    assert prefix[:reused] + submitted[:len(prompt) - reused] == prompt
    assert second_feeds[0]["past_key_values.0.key"].shape[2] == reused
    assert second_feeds[0]["position_ids"][0, 0] == reused
    assert second.text == "ok done" and second.completion_tokens == 2

    baseline_session = ScriptedSession(session.tokens, spec)
    baseline = LocalInstructModel(session=baseline_session, tokenizer=tokenizer)
    baseline.provider = provider
    assert baseline.generate(system=system, user="different material", max_new_tokens=4) == second

    prior = len(session.feeds)
    session._emitted = 0
    model.generate(system=system + " new rule", user="different material", max_new_tokens=4)
    assert session.feeds[prior]["past_key_values.0.key"].shape[2] == 0


@pytest.mark.parametrize("extra_words", [0, 1, 5, 128])
def test_cuda_default_projects_only_the_final_prompt_token_across_chunk_boundaries(
        monkeypatch, extra_words):
    monkeypatch.delenv("TRENDANALIZER_QWEN_PREFILL_CHUNK", raising=False)
    monkeypatch.delenv("TRENDANALIZER_QWEN_SPLIT_PREFILL", raising=False)
    spec = load_spec()
    words = " ".join(f"w{index}" for index in range(spec["prefill_chunk"] + extra_words))
    end = spec["tokens"]["stop"][0]
    model, session, _ = build([end])
    model.provider = "CUDAExecutionProvider"
    result = model.generate(system="s", user=words, max_new_tokens=8)
    assert result.completion_tokens == 0 and not result.stopped_at_limit
    lengths = [feed["input_ids"].shape[1] for feed in session.feeds]
    assert sum(lengths) == len(model.prompt_tokens("s", words))
    assert lengths[-1] == 1
    assert max(lengths) <= 256
    assert ["logits" in request for request in session.output_requests] == [False] * (len(lengths) - 1) + [True]


def test_split_prefill_is_the_cuda_default_and_is_ignored_by_cpu(monkeypatch):
    monkeypatch.delenv("TRENDANALIZER_QWEN_SPLIT_PREFILL", raising=False)
    words = " ".join(f"w{index}" for index in range(10))
    end = load_spec()["tokens"]["stop"][0]
    enabled, enabled_session, _ = build([end])
    enabled.provider = "CUDAExecutionProvider"
    enabled.generate(system="s", user=words, max_new_tokens=8)
    assert [feed["input_ids"].shape[1] for feed in enabled_session.feeds] == [
        len(enabled.prompt_tokens("s", words)) - 1, 1]

    monkeypatch.setenv("TRENDANALIZER_QWEN_SPLIT_PREFILL", "0")
    disabled, disabled_session, _ = build([end])
    disabled.provider = "CUDAExecutionProvider"
    disabled.generate(system="s", user=words, max_new_tokens=8)
    assert [feed["input_ids"].shape[1] for feed in disabled_session.feeds] == [
        len(disabled.prompt_tokens("s", words))]

    monkeypatch.setenv("TRENDANALIZER_QWEN_SPLIT_PREFILL", "1")
    cpu, cpu_session, _ = build([end])
    cpu.generate(system="s", user=words, max_new_tokens=8)
    assert [feed["input_ids"].shape[1] for feed in cpu_session.feeds] == [
        len(cpu.prompt_tokens("s", words))]


def test_cuda_default_preserves_the_full_greedy_answer(monkeypatch):
    monkeypatch.delenv("TRENDANALIZER_QWEN_PREFILL_CHUNK", raising=False)
    spec = load_spec()
    words = " ".join(f"w{index}" for index in range(spec["prefill_chunk"] + 5))
    end = spec["tokens"]["stop"][0]
    baseline, baseline_session, _ = build([31, 77, 78, end])
    optimized, optimized_session, _ = build([31, 77, 78, end])
    baseline.provider = optimized.provider = "CUDAExecutionProvider"
    monkeypatch.setenv("TRENDANALIZER_QWEN_PREFILL_CHUNK", "128")
    monkeypatch.setenv("TRENDANALIZER_QWEN_SPLIT_PREFILL", "0")
    expected = baseline.generate(system="s", user=words, max_new_tokens=8)
    monkeypatch.delenv("TRENDANALIZER_QWEN_PREFILL_CHUNK")
    monkeypatch.delenv("TRENDANALIZER_QWEN_SPLIT_PREFILL")
    actual = optimized.generate(system="s", user=words, max_new_tokens=8)
    assert actual == expected
    first_baseline_logits = next(index for index, request in enumerate(baseline_session.output_requests)
                                 if "logits" in request)
    first_optimized_logits = next(index for index, request in enumerate(optimized_session.output_requests)
                                  if "logits" in request)
    assert baseline_session.feeds[first_baseline_logits]["input_ids"].shape[1] > 1
    assert optimized_session.feeds[first_optimized_logits]["input_ids"].shape[1] == 1


def test_final_single_token_prefill_matches_full_chunk_greedy_logits_and_cache():
    tokens = list(range(40))
    full, full_session, _ = build([77])
    split, split_session, _ = build([77])
    full_logits, full_cache = full._step(tokens, 0, full._empty_cache())
    _, split_cache = split._step(tokens[:-1], 0, split._empty_cache(), need_logits=False)
    split_logits, split_cache = split._step(tokens[-1:], len(tokens) - 1, split_cache)
    assert np.array_equal(full_logits, split_logits)
    for name in full._past_names:
        assert np.array_equal(full_cache[name], split_cache[name])
    assert full_session.output_requests[0][0] == "logits"
    assert "logits" not in split_session.output_requests[0]
    assert split_session.feeds[1]["position_ids"].tolist() == [[len(tokens) - 1]]


def test_cancellation_between_prefill_chunks_still_stops_before_the_next_pass():
    spec = load_spec()
    cancel = Event()
    words = " ".join(f"w{index}" for index in range(spec["prefill_chunk"] + 5))
    model, session, _ = build([55], on_step=lambda count: cancel.set() if count == 1 else None)
    with pytest.raises(LocalModelError, match="отменён"):
        model.generate(system="s", user=words, max_new_tokens=8, cancel=cancel)
    assert len(session.feeds) == 1
    assert "logits" not in session.output_requests[0]


def test_cancellation_between_bulk_prefill_and_final_token_skips_logits_pass(monkeypatch):
    monkeypatch.delenv("TRENDANALIZER_QWEN_PREFILL_CHUNK", raising=False)
    monkeypatch.delenv("TRENDANALIZER_QWEN_SPLIT_PREFILL", raising=False)
    cancel = Event()
    model, session, _ = build([77], on_step=lambda count: cancel.set() if count == 1 else None)
    model.provider = "CUDAExecutionProvider"
    with pytest.raises(LocalModelError, match="отменён"):
        model.generate(system="s", user="short prompt", max_new_tokens=8, cancel=cancel)
    assert len(session.feeds) == 1
    assert len(session.feeds[0]["input_ids"][0]) == len(model.prompt_tokens("s", "short prompt")) - 1
    assert "logits" not in session.output_requests[0]


def test_cuda_binding_skips_unused_prefill_logits_without_copying_cache_to_cpu():
    spec = load_spec()
    tokenizer = FakeTokenizer()

    class Value:
        def __init__(self, array):
            self.array = array
            self.copies = 0

        def numpy(self):
            self.copies += 1
            return self.array

    class Binding:
        def __init__(self):
            self.inputs = {}
            self.outputs = []
            self.values = []

        def bind_cpu_input(self, name, value):
            self.inputs[name] = value

        def bind_ortvalue_input(self, name, value):
            self.inputs[name] = value

        def bind_output(self, name, device, device_id=0):
            self.outputs.append((name, device))

        def get_outputs(self):
            return self.values

    class BoundSession(ScriptedSession):
        def __init__(self):
            super().__init__([77], spec)
            self.bindings = []
            self.logits = Value(np.zeros((1, 3, spec["vocabulary_size"]), dtype=np.float32))
            self.logits.array[0, -1, 77] = 10.0
            self.present = Value(None)

        def io_binding(self):
            binding = Binding()
            self.bindings.append(binding)
            return binding

        def run_with_iobinding(self, binding):
            binding.values = [self.logits if name == "logits" else self.present
                              for name, _ in binding.outputs]

    session = BoundSession()
    model = LocalInstructModel(session=session, tokenizer=tokenizer)
    model._device = "cuda"
    cache = {name: Value(None) for name in model._past_names}

    logits, updated = model._bound_step([11, 12, 13], 0, cache, need_logits=False)
    assert logits is None
    assert all(value is session.present for value in updated.values())
    assert session.logits.copies == 0 and session.present.copies == 0
    assert [name for name, _ in session.bindings[0].outputs] == list(model._present_names)
    assert {device for _, device in session.bindings[0].outputs} == {"cuda"}

    logits, _ = model._bound_step([14], 3, updated, need_logits=True)
    assert int(np.argmax(logits)) == 77
    assert session.logits.copies == 1 and session.present.copies == 0
    assert session.bindings[1].outputs[0] == ("logits", "cpu")
    mask, positions = model._cuda_constant_inputs
    for binding, expected_mask, expected_positions in (
            (session.bindings[0], [[1, 1, 1]], [[0, 1, 2]]),
            (session.bindings[1], [[1, 1, 1, 1]], [[3]])):
        attention = binding.inputs["attention_mask"]
        position = binding.inputs["position_ids"]
        assert attention.tolist() == expected_mask
        assert position.tolist() == expected_positions
        assert attention.flags.c_contiguous and position.flags.c_contiguous
        assert np.shares_memory(attention, mask)
        assert np.shares_memory(position, positions)


def test_cuda_skips_only_the_unread_forward_pass_at_the_completion_limit():
    cuda, cuda_session, _ = build([55])
    cuda.provider = "CUDAExecutionProvider"
    cpu, cpu_session, _ = build([55])
    cpu.provider = "CPUExecutionProvider"
    cuda_result = cuda.generate(system="s", user="q", max_new_tokens=3)
    cpu_result = cpu.generate(system="s", user="q", max_new_tokens=3)
    assert cuda_result == cpu_result
    assert cuda_result.completion_tokens == 3 and cuda_result.stopped_at_limit
    # One logits pass fills the prompt; each later one prepares another token.
    # The CUDA path needs only two more passes for three emitted tokens.
    assert sum("logits" in names for names in cuda_session.output_requests) == 3
    assert sum("logits" in names for names in cpu_session.output_requests) == 4


def test_generation_stops_at_the_model_end_token_without_emitting_it():
    spec = load_spec()
    end = spec["tokens"]["stop"][0]
    model, _, tokenizer = build([tokenizer_id := 77, 78, end, 79])
    result = model.generate(system="s", user="q", max_new_tokens=10)
    assert result.completion_tokens == 2 and not result.stopped_at_limit
    assert result.text == tokenizer.decode([tokenizer_id, 78])


def test_token_limit_stops_a_model_that_never_ends():
    model, _, _ = build([55])
    result = model.generate(system="s", user="q", max_new_tokens=4)
    assert (result.completion_tokens, result.stopped_at_limit) == (4, True)


def test_cancellation_stops_between_tokens():
    cancel = Event()
    model, _, _ = build([55], on_step=lambda count: cancel.set() if count >= 2 else None)
    with pytest.raises(LocalModelError, match="отменён"):
        model.generate(system="s", user="q", max_new_tokens=50, cancel=cancel)


def test_prompt_framing_uses_the_models_own_control_tokens():
    spec = load_spec()
    model, _, _ = build([spec["tokens"]["stop"][0]])
    ids = model.prompt_tokens("rules", "material")
    assert ids.count(spec["tokens"]["start"]) == 3 and ids.count(spec["tokens"]["end"]) == 2
    assert ids[-3:][0] not in spec["tokens"]["stop"], "последним идёт открытый ход ассистента"


def test_oversized_prompt_is_refused_before_inference():
    model, session, _ = build([55])
    with pytest.raises(LocalModelError, match="размер"):
        model.generate(system="s", user="w " * 200_000, max_new_tokens=8)
    with pytest.raises(LocalModelError, match="размер"):
        model.generate(system="rule " * (model.spec["max_prompt_tokens"] + 1),
                       user="paper", max_new_tokens=8)
    assert model._system_tokens is None
    assert not session.feeds
    assert not session.feeds, "отказ происходит до первого прохода модели"


@pytest.mark.parametrize("value,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('Вот ответ: {"a": {"b": 2}} — готово', '{"a": {"b": 2}}'),
    ('{"a": "}"}', '{"a": "}"}'),
    ('{"a": "\\\\"}', '{"a": "\\\\"}'),
])
def test_json_is_recovered_from_ordinary_model_prose(value, expected):
    assert first_json_object(value) == expected


@pytest.mark.parametrize("value", ["no object here", '{"a": "unterminated', '{"a": [1}'])
def test_unusable_answers_are_reported_rather_than_guessed(value):
    with pytest.raises(LlmError) as caught:
        first_json_object(value)
    assert caught.value.code == "invalid_json"


def test_brackets_the_answer_left_open_are_closed_without_inventing_content():
    """The opening brace came from the prefix, so the model stops one short."""
    assert first_json_object('{"candidates": [{"label": "x"}]') == '{"candidates": [{"label": "x"}]}'
    assert first_json_object('{"a": [1, 2') == '{"a": [1, 2]}'
    # A value cut in the middle of a string is not something brackets can repair.
    with pytest.raises(LlmError):
        first_json_object('{"a": "half')


@pytest.mark.parametrize("value,expected", [
    # A looping list cut by the token limit keeps the items finished before the loop.
    ('{"q": "x", "items": ["a", "b", "b", "b', '{"q": "x", "items": ["a", "b", "b"]}'),
    ('{"q": "x", "items": ["a"], "next": "hal', '{"q": "x", "items": ["a"]}'),
    ('{"q": "x", "items": [', '{"q": "x", "items": []}'),
])
def test_truncated_answer_is_cut_back_to_its_last_complete_element(value, expected):
    assert truncated_json_object(value) == expected
    assert json.loads(expected)


def test_truncation_repair_is_used_only_by_schemas_that_opt_in():
    class Terms(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        q: str
        items: list[str] = []

    class RepairedTerms(Terms):
        LOCAL_TRUNCATION_REPAIR: ClassVar[bool] = True

    client = LocalLlmClient(LocalProviderConfig(), ledger(), model=StubModel(['{"q": "x"}']))
    text = '{"q": "x", "items": ["a", "b", "b", "b'
    with pytest.raises(LlmError) as caught:
        client._decode(Terms, text)
    assert caught.value.code == "invalid_json"
    assert client._decode(RepairedTerms, text).items == ["a", "b", "b"]


def ledger():
    connection = sqlite3.connect(":memory:")
    budget = BudgetService(connection)
    budget.create_scope("run/one", BudgetLimits(24, 200000, 30000, 0), currency="USD")
    return budget


class StubModel:
    """A model whose answers are decided by the test, not by weights."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.spec = load_spec()

    def generate(self, *, system, user, max_new_tokens, cancel=None, stop_at_json=False, answer_prefix=""):
        from app.pilot.local_llm import LocalCompletion

        assert stop_at_json, "клиенту нужен ответ, который заканчивается на закрытии объекта"
        assert answer_prefix.startswith('{"'), "ответ начинается внутри требуемого объекта"
        self.prompts.append((system, user))
        text = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        return LocalCompletion(text=text, prompt_tokens=120, completion_tokens=30, stopped_at_limit=False)

    def close(self):
        pass


def call(client, schema=Answer, **changes):
    values = dict(system_prompt="rules", user_content='{"cluster": 1}', prompt_version="test/1.0.0",
                  request_id="run/one/labels/1", scope_ids=("run/one",), cancel=Event(), max_output_tokens=200)
    return client.generate_json(schema, **(values | changes))


def test_valid_answer_is_returned_with_a_free_receipt_and_recorded_tokens():
    budget = ledger()
    model = StubModel(['```json\n{"label": "фотонное умножение", "confident": true}\n```'])
    client = LocalLlmClient(LocalProviderConfig(), budget, model=model)
    completion = call(client)
    assert completion.value.label == "фотонное умножение"
    receipt = completion.receipt
    assert (receipt.provider, receipt.cost_micro, receipt.usage.total_tokens) == ("local", 0, 150)
    used = budget.snapshot("run/one").used
    assert (used.calls, used.input_tokens, used.output_tokens, used.cost_micro) == (1, 120, 30, 0)


def test_a_rejected_answer_is_retried_once_with_the_rejected_field_named():
    budget = ledger()
    model = StubModel(['{"label": "x", "confident": "yes"}',
                       '{"label": "фотонное умножение", "confident": false}'])
    client = LocalLlmClient(LocalProviderConfig(), budget, model=model)
    completion = call(client)
    assert completion.value.confident is False
    assert len(model.prompts) == 2
    correction = model.prompts[1][1]
    assert "label" in correction and "confident" in correction
    assert '{"cluster": 1}' in correction, "исходный материал остаётся в повторе"
    # Both attempts are real work and both are in the ledger.
    assert budget.snapshot("run/one").used.calls == 2


def test_an_answer_that_stays_invalid_fails_without_inventing_one():
    budget = ledger()
    model = StubModel(['{"label": "x", "confident": "yes"}'])
    client = LocalLlmClient(LocalProviderConfig(), budget, model=model)
    with pytest.raises(LlmError) as caught:
        call(client)
    assert caught.value.code == "schema_failed"
    assert budget.snapshot("run/one").used.calls == 2


def test_cancellation_before_the_model_runs_is_a_cancellation_not_a_failure():
    budget = ledger()
    model = StubModel(['{"label": "okay", "confident": true}'])
    client = LocalLlmClient(LocalProviderConfig(), budget, model=model)
    cancel = Event()
    cancel.set()
    with pytest.raises(LlmCancelled):
        call(client, cancel=cancel)
    assert budget.snapshot("run/one").used.calls == 0 and not model.prompts


def test_a_budget_in_another_currency_is_refused_before_any_work():
    budget = ledger()
    model = StubModel(['{"label": "okay", "confident": true}'])
    client = LocalLlmClient(LocalProviderConfig(currency="RUB"), budget, model=model)
    with pytest.raises(LlmError) as caught:
        call(client)
    assert caught.value.code == "currency_mismatch" and not model.prompts


def test_the_fields_and_the_hint_follow_the_material_the_model_must_answer_about():
    budget = ledger()
    model = StubModel(['{"label": "okay", "confident": true}'])
    client = LocalLlmClient(LocalProviderConfig(), budget, model=model)
    call(client, local_hint="EXAMPLE-HINT")
    system, user = model.prompts[0]
    assert system == "rules", "роль модели остаётся в системном сообщении"
    # The contract is a field list, not generated JSON Schema, and it comes
    # after the material: measured, that order is what makes the answer usable.
    assert "confident: boolean" in user and "$defs" not in user
    assert user.index('{"cluster": 1}') < user.index("confident: boolean") < user.index("EXAMPLE-HINT")


def test_validation_summary_names_fields_without_echoing_their_values():
    try:
        Answer.model_validate({"label": "x", "confident": "yes"})
    except Exception as error:  # pydantic.ValidationError
        summary = _validation_summary(error)
    assert "label" in summary and "confident" in summary and "yes" not in summary


def test_pinned_files_are_verified_by_size_and_digest(tmp_path):
    spec = load_spec()
    smaller = json.loads(json.dumps(spec))
    for item in smaller["files"]:
        content = b"x" * 32
        item["bytes"], item["sha256"] = len(content), __import__("hashlib").sha256(content).hexdigest()
        (tmp_path / item["name"]).write_bytes(content)
    assert verify_artifacts(tmp_path, smaller) == tmp_path
    (tmp_path / smaller["files"][0]["name"]).write_bytes(b"y" * 32)
    with pytest.raises(LocalModelError, match="Контрольная сумма"):
        verify_artifacts(tmp_path, smaller)
    (tmp_path / smaller["files"][0]["name"]).unlink()
    with pytest.raises(LocalModelError, match="отсутствует"):
        verify_artifacts(tmp_path, smaller)


def test_shipped_specification_describes_the_pinned_qwen_export():
    spec = load_spec()
    assert spec["model_id"] == "onnx-community/Qwen2.5-1.5B-Instruct"
    assert (spec["layers"], spec["key_value_heads"], spec["head_dimension"]) == (28, 2, 128)
    assert spec["tokens"]["start"] == 151644 and 151645 in spec["tokens"]["stop"]


def small_spec(tmp_path, *, write=True):
    """The pinned contract with tiny files, so installation can be exercised whole."""
    import hashlib

    spec = json.loads(json.dumps(load_spec()))
    for index, item in enumerate(spec["files"]):
        content = bytes([index]) * (32 + index)
        item["bytes"], item["sha256"] = len(content), hashlib.sha256(content).hexdigest()
        if write:
            (tmp_path / item["name"]).write_bytes(content)
    return spec


def test_presence_check_reads_no_bytes_and_still_sees_a_truncated_file(tmp_path, monkeypatch):
    spec = small_spec(tmp_path)
    assert artifacts_present(tmp_path, spec) is True
    assert artifacts_present(tmp_path / "empty", spec) is False
    name = spec["files"][0]["name"]
    (tmp_path / name).write_bytes(b"x")
    assert artifacts_present(tmp_path, spec) is False

    def refuse(*_args, **_kwargs):
        raise AssertionError("Проверка наличия не должна читать веса")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert artifacts_present(tmp_path, spec) is False


def test_installation_reports_every_written_byte_of_the_whole_download(tmp_path, monkeypatch):
    import httpx

    from scripts import install_local_llm

    (tmp_path / "source").mkdir()
    spec = small_spec(tmp_path / "source")
    total = sum(item["bytes"] for item in spec["files"])
    bodies = {item["remote"].rsplit("/", 1)[-1]: (tmp_path / "source" / item["name"]).read_bytes()
              for item in spec["files"]}
    monkeypatch.setattr(install_local_llm.local_llm, "load_spec", lambda: spec)
    monkeypatch.setattr(install_local_llm, "install_from_staging", lambda *args: False)
    monkeypatch.setattr(install_local_llm, "CHUNK_BYTES", 16)

    def handle(request):
        return httpx.Response(200, stream=httpx.ByteStream(bodies[str(request.url).rsplit("/", 1)[-1]]))

    transport = httpx.MockTransport(handle)
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(**kwargs | {"transport": transport}))
    reported = []
    target = tmp_path / "installed"
    result = install_local_llm.install(target, progress=lambda done, whole: reported.append((done, whole)))

    assert result["installed"] is True
    assert artifacts_present(target, spec) is True
    assert {whole for _, whole in reported} == {total}
    assert [done for done, _ in reported] == sorted(done for done, _ in reported)
    assert reported[-1][0] == total
    assert len(reported) > len(spec["files"])  # Chunks, not one report per file.


def test_installed_weights_are_reported_complete_without_downloading_again(tmp_path, monkeypatch):
    from scripts import install_local_llm

    spec = small_spec(tmp_path)
    monkeypatch.setattr(install_local_llm.local_llm, "load_spec", lambda: spec)
    total = sum(item["bytes"] for item in spec["files"])
    reported = []
    result = install_local_llm.install(tmp_path, progress=lambda done, whole: reported.append((done, whole)))
    assert result["installed"] is False and reported == [(total, total)]


def test_cancelled_installation_leaves_no_half_written_model(tmp_path, monkeypatch):
    import httpx

    from scripts import install_local_llm

    (tmp_path / "source").mkdir()
    spec = small_spec(tmp_path / "source")
    bodies = {item["remote"].rsplit("/", 1)[-1]: (tmp_path / "source" / item["name"]).read_bytes()
              for item in spec["files"]}
    monkeypatch.setattr(install_local_llm.local_llm, "load_spec", lambda: spec)
    monkeypatch.setattr(install_local_llm, "install_from_staging", lambda *args: False)
    monkeypatch.setattr(install_local_llm, "CHUNK_BYTES", 8)
    cancel = Event()

    def handle(request):
        return httpx.Response(200, stream=httpx.ByteStream(bodies[str(request.url).rsplit("/", 1)[-1]]))

    transport = httpx.MockTransport(handle)
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(**kwargs | {"transport": transport}))
    target = tmp_path / "installed"
    with pytest.raises(LocalModelError, match="отменена"):
        install_local_llm.install(target, cancel=cancel, progress=lambda *_: cancel.set())
    assert not target.exists()
    assert not list((tmp_path).glob(".local-llm-*"))


def test_prepared_answers_are_handed_to_the_same_request_without_recomputing():
    from app.pilot.local_llm import GenerationRequest

    stop = load_spec()["tokens"]["stop"][0]
    model, session, _ = build([7, 8, stop, 9, stop])
    first = GenerationRequest(system="rules", user="alpha", max_new_tokens=8)
    second = GenerationRequest(system="rules", user="beta", max_new_tokens=8)
    # Без CUDA пакет считается по одному: результат тот же, что у прямых вызовов.
    model.prepare([first, second])
    calls = len(session.feeds)
    answer = model.generate(system="rules", user="alpha", max_new_tokens=8)
    assert len(session.feeds) == calls and answer.completion_tokens == 2
    # Готовый ответ выдаётся один раз; другой запрос считается заново.
    model.generate(system="rules", user="alpha", max_new_tokens=8)
    assert len(session.feeds) > calls


def test_client_prefetch_frames_requests_exactly_like_generate_json():
    prepared = []

    class PreparingModel(StubModel):
        def prepare(self, requests, *, cancel=None):
            prepared.extend(requests)

    model = PreparingModel(['{"label": "xyz", "confident": true}'])
    client = LocalLlmClient(LocalProviderConfig(), ledger(), model=model)
    requests = [{"system_prompt": "rules", "user_content": '{"cluster": 1}', "local_hint": "",
                 "max_output_tokens": 200},
                {"system_prompt": "rules", "user_content": '{"cluster": 2}', "local_hint": "hint",
                 "max_output_tokens": 200}]
    client.prefetch_json(Answer, requests, cancel=Event())
    call(client)
    # Первый запрос сформулирован для модели ровно так же, как его отправит generate_json.
    assert (prepared[0].system, prepared[0].user) == model.prompts[0]
    assert prepared[1].user.endswith("\nhint") and all(item.stop_at_json for item in prepared)
