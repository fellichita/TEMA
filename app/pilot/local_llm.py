"""Offline instruction model that names a mechanism and plans a search scope.

This is the local stand-in for the external AI provider. It performs exactly
the three jobs the paid path performs — proposing a scope, naming a document
cluster, and selecting quotations — and nothing else. Every check that decides
whether a candidate becomes a trend stays where it was: this module only writes
the proposal that those checks then verify against the archived documents.

Decoding is greedy and therefore reproducible: the same weights, prompt and
token limit produce the same text. It is also slow. A 1.5B model on a processor
answers one prompt in tens of seconds, so a run spends minutes here rather than
the seconds an online provider needs. That is the price of needing no key and
no network, and cancellation stops it between tokens.

The module never downloads anything and never reaches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol

from app.identity import default_data_dir
from app.runtime.model_resources import resolve_model

MODEL_KEY = "qwen2.5-1.5b-instruct"
MAX_PROMPT_CHARACTERS = 190_000
_MAX_CACHED_SYSTEM_TOKENS = 512
# Сколько независимых ответов декодируется одним батчем. Замер 28.09.2026 на
# RTX 3070: шаг батча 1 — 16,9 мс, батча 4 — 31 мс (7,8 мс на строку), батча 6 —
# 42 мс (7,0 мс). На восьми настоящих промптах именования батч 8 ускорил ответы
# в 2,44 раза, батч 4 — в 2,05, и все восемь ответов совпали с одиночными байт в байт.
BATCH_VARIABLE = "TRENDANALIZER_LLM_BATCH"
DEFAULT_GENERATION_BATCH = 8
MAX_GENERATION_BATCH = 8
# Заранее посчитанные ответы ждут своего вызова; больше одного окна не держим.
_MAX_PREPARED = 16


def generation_batch() -> int:
    """Размер пакета генерации: из окружения (панель владельца) или по умолчанию."""
    raw = os.environ.get(BATCH_VARIABLE, "").strip()
    if raw.isdigit() and 1 <= int(raw) <= MAX_GENERATION_BATCH:
        return int(raw)
    return DEFAULT_GENERATION_BATCH


class Cancellation(Protocol):
    def is_set(self) -> bool: ...


class LocalModelError(ValueError):
    """A safe, actionable failure that carries no prompt text or file path."""


@dataclass(frozen=True)
class GenerationRequest:
    """Один независимый запрос к модели — те же аргументы, что у `generate`."""
    system: str
    user: str
    max_new_tokens: int
    stop_at_json: bool = False
    answer_prefix: str = ""


@dataclass(frozen=True)
class LocalCompletion:
    """One generated answer with the counts a budget ledger records."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    stopped_at_limit: bool


def load_spec() -> dict[str, Any]:
    """Read the specification shipped with application code, never from a model directory."""
    spec = json.loads(Path(__file__).with_name("local-llm-spec.json").read_text(encoding="utf-8"))
    if (spec.get("schema_version") != 1 or not isinstance(spec.get("model_id"), str)
            or not re.fullmatch(r"[a-f0-9]{40}", spec.get("revision", ""))
            or spec.get("architecture") != "qwen2"
            or type(spec.get("layers")) is not int or not 1 <= spec["layers"] <= 128
            or type(spec.get("key_value_heads")) is not int or not 1 <= spec["key_value_heads"] <= 128
            or type(spec.get("head_dimension")) is not int or not 1 <= spec["head_dimension"] <= 1024
            or type(spec.get("vocabulary_size")) is not int or not 1 <= spec["vocabulary_size"] <= 10**6
            or not 1 <= spec.get("prefill_chunk", 0) <= 2048
            or not 1 <= spec.get("max_prompt_tokens", 0) <= 32768
            or not 1 <= spec.get("max_new_tokens", 0) <= 8192):
        raise LocalModelError("Повреждена встроенная спецификация локальной модели.")
    tokens = spec.get("tokens")
    if (not isinstance(tokens, dict)
            or any(type(tokens.get(name)) is not int for name in ("start", "end"))
            or not isinstance(tokens.get("stop"), list) or not tokens["stop"]
            or any(type(value) is not int for value in tokens["stop"])):
        raise LocalModelError("Повреждена встроенная спецификация локальной модели.")
    if {item["name"] for item in spec["files"]} != {"model.onnx", "tokenizer.json"}:
        raise LocalModelError("Неверный состав локальной модели.")
    for item in spec["files"]:
        if (not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= 4_000_000_000 or not isinstance(item["remote"], str)):
            raise LocalModelError("Повреждена встроенная спецификация локальной модели.")
    return spec


def model_directory(data_dir: Path | None = None) -> Path:
    return (default_data_dir() if data_dir is None else Path(data_dir)) / "models" / MODEL_KEY


def checkpoint(cancel: Cancellation | None) -> None:
    if cancel is not None and cancel.is_set():
        raise LocalModelError("Локальный AI-запрос отменён.")


def verify_artifacts(directory: Path, spec: dict[str, Any] | None = None,
                     cancel: Cancellation | None = None) -> Path:
    """Confirm the exact pinned bytes before any native code reads them."""
    spec = load_spec() if spec is None else spec
    directory = Path(directory)
    for item in spec["files"]:
        checkpoint(cancel)
        path = directory / item["name"]
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
                raise LocalModelError("Локальная AI-модель отсутствует или повреждена. "
                                      "Установите её командой python -m scripts.install_local_llm.")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(4 * 1024 * 1024):
                    checkpoint(cancel)
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise LocalModelError("Контрольная сумма локальной AI-модели не совпадает.")
        except OSError as error:
            raise LocalModelError("Не удалось прочитать локальную AI-модель.") from error
    return directory


def artifacts_present(directory: Path, spec: dict[str, Any] | None = None) -> bool:
    """Answer whether there is anything to load, without reading 1.8 GB.

    The interface asks this on every status refresh, so it compares sizes only.
    What decides that the weights may run is still `verify_artifacts`: the hashes
    are checked before any native code opens them.
    """
    spec = load_spec() if spec is None else spec
    for item in spec["files"]:
        path = Path(directory) / item["name"]
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
                return False
        except OSError:
            return False
    return True


def json_object_span(text: str) -> tuple[int, int] | None:
    """Offsets of the first complete JSON object, or None while it is unfinished.

    An instruction model wraps its object in a fence or a sentence and often
    keeps writing after it. Knowing exactly where the object closes is both how
    the answer is read and when generation can stop.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, index, in_string, escaped = 0, start, False, False
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
        index += 1
    return None


def completed_json_object(text: str) -> str | None:
    """The first JSON object, with the brackets the answer left open closed.

    The answer starts inside the object, so its opening brace was never written
    by the model and the model does not count it: measured on real material, it
    produced a complete, correct answer and stopped one brace short. Closing the
    brackets that are still open adds no content of its own — anything else
    missing, including an unterminated string, is still refused.
    """
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    index, in_string, escaped = start, False, False
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            stack.append("}" if character == "{" else "]")
        elif character in "}]":
            if not stack or stack[-1] != character:
                return None
            stack.pop()
            if not stack:
                return text[start:index + 1]
        index += 1
    if in_string or not stack:
        return None
    return text[start:] + "".join(reversed(stack))


def truncated_json_object(text: str) -> str | None:
    """The first JSON object cut back to its last complete element, then closed.

    A small model can loop on one list («data association», «data association»…)
    until the token limit ends it inside a string. Everything before the last
    completed element is its own words; only the unfinished tail is dropped.
    Callers opt in where a shorter list is still a correct answer.
    """
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    cut: tuple[int, tuple[str, ...]] | None = None
    index, in_string, escaped = start, False, False
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            stack.append("}" if character == "{" else "]")
            # An empty container is a complete value.
            cut = (index + 1, tuple(stack))
        elif character in "}]":
            if not stack or stack[-1] != character:
                return None
            stack.pop()
            if not stack:
                return text[start:index + 1]
        elif character == ",":
            # Before a separator the previous element or pair is complete.
            cut = (index, tuple(stack))
        index += 1
    if cut is None:
        return None
    end, open_brackets = cut
    return text[start:end] + "".join(reversed(open_brackets))


def spec_fingerprint(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class LocalInstructModel:
    """Greedy decoding over a pinned ONNX causal model, with a reused key/value cache.

    The cache is what makes this usable at all: without it every new token would
    recompute the whole prompt. The prompt is fed in bounded chunks; intermediate
    chunks request only the cache, because their full-vocabulary logits would be
    discarded without affecting the answer. CUDA can optionally run the final
    prompt token alone to avoid projecting an entire chunk onto the vocabulary.
    """

    def __init__(self, directory: Path | None = None, *, cancel: Cancellation | None = None,
                 session: Any = None, tokenizer: Any = None):
        """Load the pinned weights, or accept a prepared pair for testing.

        `session` and `tokenizer` exist so the decoding loop can be exercised
        against a small deterministic stand-in; supplying them skips the file
        verification, and application code never does.
        """
        import numpy as np

        self.spec = load_spec()
        self._np = np
        if session is not None and tokenizer is not None:
            self._session, self._tokenizer, self.provider = session, tokenizer, "InjectedSession"
        else:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from tokenizers import Tokenizer

            from app.runtime.inference import (
                PROVIDER_VARIABLE, RequestedCudaUnavailable, execution_providers, require_requested_provider,
                session_threads,
            )

            location = resolve_model(MODEL_KEY, self.spec["revision"], explicit_dir=directory,
                                     development_default=model_directory())
            directory = verify_artifacts(location.path, self.spec, cancel)
            self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = session_threads()
            options.log_severity_level = 3
            checkpoint(cancel)
            try:
                self._session = ort.InferenceSession(str(directory / "model.onnx"), sess_options=options,
                                                     providers=list(execution_providers()),
                                                     enable_fallback=(os.environ.get(PROVIDER_VARIABLE, "")
                                                                      .strip().lower() != "cuda"))
            except Exception as error:
                raise LocalModelError("ONNX Runtime не смог загрузить локальную AI-модель.") from error
            self.provider = self._session.get_providers()[0]
            try:
                require_requested_provider(self.provider)
            except RequestedCudaUnavailable as error:
                raise LocalModelError(str(error)) from None
        inputs = {item.name: item for item in self._session.get_inputs()}
        for name in ("input_ids", "attention_mask", "position_ids"):
            if name not in inputs:
                raise LocalModelError("Локальная AI-модель имеет неизвестный интерфейс.")
        self._past_names = tuple(f"past_key_values.{index}.{kind}"
                                 for index in range(self.spec["layers"]) for kind in ("key", "value"))
        if any(name not in inputs for name in self._past_names):
            raise LocalModelError("Локальная AI-модель имеет неизвестный интерфейс.")
        self._present_names = tuple(name.replace("past_key_values.", "present.") for name in self._past_names)
        outputs = {item.name for item in self._session.get_outputs()}
        if "logits" not in outputs or any(name not in outputs for name in self._present_names):
            raise LocalModelError("Локальная AI-модель имеет неизвестный интерфейс.")
        self._cache_dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}.get(
            inputs[self._past_names[0]].type)
        if self._cache_dtype is None:
            raise LocalModelError("Локальная AI-модель использует неподдерживаемый тип кеша.")
        # Where the attention cache lives between steps. Carrying it back to
        # host memory each token costs more than the step itself: at four
        # thousand tokens of context that is a hundred megabytes per token, and
        # measured decoding fell from 45 to 10 tokens per second because of it.
        self._device = "cuda" if self.provider == "CUDAExecutionProvider" else "cpu"
        self._binding = getattr(self._session, "io_binding", None)
        # Both arrays are constant throughout generation. Reuse their
        # contiguous views in CUDA I/O binding instead of allocating and
        # filling them for every generated token.
        self._cuda_constant_inputs: tuple[Any, Any] | None = None
        # Consecutive naming calls share the same system instruction. Keep only
        # its completed prefill chunks; user material and generated tokens never
        # enter this cache. One entry bounds device/host memory usage.
        self._system_prefill: tuple[str, str, int, int, dict[str, Any]] | None = None
        # The same instruction is also tokenized for every candidate batch.
        # Cache its immutable token ids separately from the provider-specific KV
        # cache so constructing the next prompt does not repeat BPE work.
        self._system_tokens: tuple[str, tuple[int, ...]] | None = None
        # Ответы, посчитанные батчем заранее (`prepare`); `generate` отдаёт их
        # тому же запросу вместо повторного вычисления.
        self._prepared: dict[GenerationRequest, LocalCompletion] = {}

    @property
    def fingerprint(self) -> str:
        return spec_fingerprint(self.spec | {"execution_provider": self.provider})

    @property
    def model_id(self) -> str:
        return self.spec["model_id"]

    def close(self) -> None:
        self._prepared = {}
        self._system_prefill = None
        self._system_tokens = None
        self._cuda_constant_inputs = None
        self._session = None

    def _system_prompt_tokens(self, system: str) -> list[int]:
        saved = self._system_tokens
        if saved is not None and saved[0] == system:
            return list(saved[1])
        start, end = self.spec["tokens"]["start"], self.spec["tokens"]["end"]
        newline = self._tokenizer.encode("\n", add_special_tokens=False).ids
        tokens = ([start] + self._tokenizer.encode("system", add_special_tokens=False).ids
                  + newline + self._tokenizer.encode(system, add_special_tokens=False).ids
                  + [end] + newline)
        # No valid complete prompt can include an oversized system prefix.
        if len(tokens) <= self.spec["max_prompt_tokens"]:
            self._system_tokens = (system, tuple(tokens))
        return tokens

    def prompt_tokens(self, system: str, user: str, answer_prefix: str = "") -> list[int]:
        """Build the model's own chat framing; the caller never supplies control tokens."""
        if not isinstance(system, str) or not isinstance(user, str) or not user.strip():
            raise LocalModelError("Требуется текст запроса.")
        if len(system) + len(user) > MAX_PROMPT_CHARACTERS:
            raise LocalModelError("Материал превышает безопасный размер одного AI-запроса.")
        start, end = self.spec["tokens"]["start"], self.spec["tokens"]["end"]
        newline = self._tokenizer.encode("\n", add_special_tokens=False).ids
        ids = self._system_prompt_tokens(system)
        ids.append(start)
        ids.extend(self._tokenizer.encode("user", add_special_tokens=False).ids)
        ids.extend(newline)
        ids.extend(self._tokenizer.encode(user, add_special_tokens=False).ids)
        ids.append(end)
        ids.extend(newline)
        ids.append(start)
        ids.extend(self._tokenizer.encode("assistant", add_special_tokens=False).ids)
        ids.extend(newline)
        if answer_prefix:
            # The answer starts inside the required object. A small model that
            # has already opened the right key cannot drift into retelling the
            # material it was given, which is its most expensive mistake.
            ids.extend(self._tokenizer.encode(answer_prefix, add_special_tokens=False).ids)
        if len(ids) > self.spec["max_prompt_tokens"]:
            raise LocalModelError("Материал превышает безопасный размер одного AI-запроса.")
        return ids

    def _empty_cache(self):
        np = self._np
        shape = (1, self.spec["key_value_heads"], 0, self.spec["head_dimension"])
        empty = np.zeros(shape, dtype=self._cache_dtype)
        if self._binding is None:
            return {name: empty for name in self._past_names}
        import onnxruntime as ort

        return {name: ort.OrtValue.ortvalue_from_numpy(empty, self._device, 0)
                for name in self._past_names}

    def _bound_step(self, tokens, offset: int, cache: dict[str, Any], *, need_logits: bool = True):
        """One pass whose cache never leaves the device it was computed on."""
        np = self._np
        length = len(tokens)
        binding = self._session.io_binding()
        binding.bind_cpu_input("input_ids", np.asarray([tokens], dtype=np.int64))
        maximum = self.spec["max_prompt_tokens"] + self.spec["max_new_tokens"]
        if self._device == "cuda" and offset + length <= maximum:
            if self._cuda_constant_inputs is None:
                self._cuda_constant_inputs = (np.ones((1, maximum), dtype=np.int64),
                                              np.arange(maximum, dtype=np.int64).reshape(1, maximum))
            mask, positions = self._cuda_constant_inputs
            attention_mask = mask[:, :offset + length]
            position_ids = positions[:, offset:offset + length]
        else:
            attention_mask = np.ones((1, offset + length), dtype=np.int64)
            position_ids = np.arange(offset, offset + length, dtype=np.int64).reshape(1, length)
        binding.bind_cpu_input("attention_mask", attention_mask)
        binding.bind_cpu_input("position_ids", position_ids)
        for name, value in cache.items():
            binding.bind_ortvalue_input(name, value)
        # A prefill chunk before the last one needs only its cache. Requesting
        # logits there projects every token onto the full vocabulary and copies
        # the resulting rows to the CPU, although none of them will be read.
        # Outputs come back in the order they are bound, not in graph order.
        if need_logits:
            binding.bind_output("logits", "cpu")
        for name in self._present_names:
            binding.bind_output(name, self._device, 0)
        try:
            self._session.run_with_iobinding(binding)
            results = binding.get_outputs()
        except Exception as error:
            raise LocalModelError("Ошибка вычисления локальной AI-модели.") from error
        names = (["logits"] if need_logits else []) + list(self._present_names)
        values = dict(zip(names, results, strict=True))
        if need_logits:
            logits = values["logits"].numpy()
            if logits.ndim != 3 or logits.shape[2] != self.spec["vocabulary_size"]:
                raise LocalModelError("Локальная AI-модель вернула неожиданный результат.")
            last_logits = np.asarray(logits[0, -1], dtype=np.float32)
        else:
            last_logits = None
        updated = {past: values[present]
                   for past, present in zip(self._past_names, self._present_names, strict=True)}
        return last_logits, updated

    def _step(self, tokens, offset: int, cache: dict[str, Any], *, need_logits: bool = True):
        """Run one forward pass and return its last logit row with the new cache."""
        if self._binding is not None:
            return self._bound_step(tokens, offset, cache, need_logits=need_logits)
        np = self._np
        length = len(tokens)
        feeds = {
            "input_ids": np.asarray([tokens], dtype=np.int64),
            "attention_mask": np.ones((1, offset + length), dtype=np.int64),
            "position_ids": np.arange(offset, offset + length, dtype=np.int64).reshape(1, length),
        }
        feeds.update(cache)
        try:
            names = (["logits"] if need_logits else []) + list(self._present_names)
            results = self._session.run(names, feeds)
        except Exception as error:
            raise LocalModelError("Ошибка вычисления локальной AI-модели.") from error
        if need_logits:
            logits = results[0]
            if logits.ndim != 3 or logits.shape[2] != self.spec["vocabulary_size"]:
                raise LocalModelError("Локальная AI-модель вернула неожиданный результат.")
            last_logits = np.asarray(logits[0, -1], dtype=np.float32)
            cache_results = results[1:]
        else:
            last_logits = None
            cache_results = results
        updated = dict(zip(self._past_names, cache_results, strict=True))
        return last_logits, updated

    def generate(self, *, system: str, user: str, max_new_tokens: int,
                 cancel: Cancellation | None = None, stop_at_json: bool = False,
                 answer_prefix: str = "") -> LocalCompletion:
        """Produce one greedy continuation, stopping at the model's own end token.

        With `stop_at_json` the answer also ends as soon as its first JSON object
        closes. The tail after it is never read, and generating it is the single
        most expensive habit this model has.
        """
        np = self._np
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.spec["max_new_tokens"]:
            raise LocalModelError("Недопустимый предел выходных токенов локальной модели.")
        if self._session is None:
            raise LocalModelError("Локальная AI-модель уже закрыта.")
        prepared = self._prepared.pop(GenerationRequest(system, user, max_new_tokens, stop_at_json, answer_prefix),
                                      None)
        if prepared is not None:
            checkpoint(cancel)
            return prepared
        tokens = self.prompt_tokens(system, user, answer_prefix)
        stop = set(self.spec["tokens"]["stop"])
        logits, cache, offset = self._prefill(system, tokens, cancel)
        generated: list[int] = []
        stopped_at_limit = True
        for step in range(max_new_tokens):
            checkpoint(cancel)
            token = int(np.argmax(logits))
            if token in stop:
                stopped_at_limit = False
                break
            generated.append(token)
            # Checking only after a brace keeps this off the hot path.
            if stop_at_json and "}" in self._tokenizer.decode([token], skip_special_tokens=True):
                if json_object_span(answer_prefix
                                    + self._tokenizer.decode(generated, skip_special_tokens=True)) is not None:
                    stopped_at_limit = False
                    break
            # The final forward pass would calculate logits and KV values that
            # no caller can read. Avoid it on CUDA, where it invokes the GPU.
            if self.provider == "CUDAExecutionProvider" and step + 1 == max_new_tokens:
                break
            logits, cache = self._step([token], offset, cache)
            offset += 1
        text = answer_prefix + self._tokenizer.decode(generated, skip_special_tokens=True)
        return LocalCompletion(text=text, prompt_tokens=len(tokens),
                               completion_tokens=len(generated), stopped_at_limit=stopped_at_limit)

    def _prefill(self, system: str, tokens: list[int], cancel: Cancellation | None):
        """Прочитать промпт: логиты его последнего токена, кеш внимания и длину."""
        from app.runtime.qwen_cuda import qwen_prefill_chunk, qwen_split_prefill

        chunk = qwen_prefill_chunk(self.spec["prefill_chunk"], self.provider)
        # Preserve the original chunk boundaries exactly. Reusing a partial
        # chunk would change ONNX floating-point work and could change a greedy
        # choice near a tie; completed system-only chunks are identical passes.
        system_end = len(self._system_prompt_tokens(system))
        reusable_end = (min(system_end, _MAX_CACHED_SYSTEM_TOKENS) // chunk) * chunk
        prefill_key = (system, self.provider, chunk, reusable_end)
        saved = self._system_prefill
        if saved is not None and saved[:4] == prefill_key:
            offset, cache = saved[3], saved[4]
        else:
            offset = 0
            cache = self._empty_cache()
        logits = None
        # On CUDA, move the final prompt token to its own pass. This projects
        # and transfers one vocabulary row instead of up to `chunk` rows, but
        # adds an inference call. CPU keeps its measured faster path.
        split_last = qwen_split_prefill(self.provider)
        prefill_end = len(tokens) - 1 if split_last else len(tokens)
        for start in range(offset, prefill_end, chunk):
            checkpoint(cancel)
            piece = tokens[start:min(start + chunk, prefill_end)]
            logits, cache = self._step(piece, offset, cache,
                                       need_logits=not split_last and start + chunk >= prefill_end)
            offset += len(piece)
            if reusable_end and offset == reusable_end:
                self._system_prefill = (system, self.provider, chunk, reusable_end, cache)
        if split_last:
            checkpoint(cancel)
            logits, cache = self._step(tokens[-1:], offset, cache, need_logits=True)
            offset += 1
        return logits, cache, offset

    def prepare(self, requests: list[GenerationRequest], *, cancel: Cancellation | None = None) -> None:
        """Посчитать ответы заранее одним батчем; `generate` с теми же аргументами возьмёт их."""
        for request, completion in zip(requests, self.generate_many(requests, cancel=cancel), strict=True):
            self._prepared[request] = completion
        while len(self._prepared) > _MAX_PREPARED:
            del self._prepared[next(iter(self._prepared))]

    def generate_many(self, requests: list[GenerationRequest], *,
                      cancel: Cancellation | None = None) -> list[LocalCompletion]:
        """Несколько независимых жадных ответов: промпты читаются по одному, ответы
        декодируются вместе.

        Модель упирается в пропускную способность памяти, поэтому шаг батча из
        четырёх строк почти вдвое дороже одиночного, а не вчетверо. Строки не
        видят друг друга: у каждой свои позиции, а чужие и пустые ячейки кеша
        закрыты маской внимания. Ответ тот же жадный, но численно батч считается
        иначе, чем одиночный проход, — у редкой строки может отличаться токен
        вблизи равенства. Без CUDA и для одного запроса — обычный `generate`.
        """
        np = self._np
        if self._session is None:
            raise LocalModelError("Локальная AI-модель уже закрыта.")
        if (len(requests) < 2 or self._device != "cuda" or self._binding is None
                or any(type(item.max_new_tokens) is not int or not 1 <= item.max_new_tokens <= self.spec["max_new_tokens"]
                       for item in requests)):
            return [self.generate(system=item.system, user=item.user, max_new_tokens=item.max_new_tokens,
                                  cancel=cancel, stop_at_json=item.stop_at_json, answer_prefix=item.answer_prefix)
                    for item in requests]
        import onnxruntime as ort

        stop = set(self.spec["tokens"]["stop"])
        prompts, rows, lengths = [], [], []
        for item in requests:
            tokens = self.prompt_tokens(item.system, item.user, item.answer_prefix)
            logits, prompt_cache, length = self._prefill(item.system, tokens, cancel)
            prompts.append(tokens)
            rows.append((logits, {name: value.numpy() for name, value in prompt_cache.items()}))
            lengths.append(length)
        count, longest = len(requests), max(lengths)
        # Кеш каждой строки дополняется нулями до самой длинной; эти ячейки маскируются.
        cache: dict[str, Any] = {}
        for name in self._past_names:
            first = rows[0][1][name]
            merged = np.zeros((count, first.shape[1], longest, first.shape[3]), dtype=first.dtype)
            for index, (_, values) in enumerate(rows):
                merged[index, :, :lengths[index]] = values[name][0]
            cache[name] = ort.OrtValue.ortvalue_from_numpy(merged, self._device, 0)
        prompt_mask = np.zeros((count, longest), dtype=np.int64)
        for index, length in enumerate(lengths):
            prompt_mask[index, :length] = 1
        logits = [row[0] for row in rows]
        del rows
        generated: list[list[int]] = [[] for _ in requests]
        finished = [False] * count
        at_limit = [True] * count
        step = 0
        while True:
            checkpoint(cancel)
            tokens = []
            for index, item in enumerate(requests):
                if finished[index]:
                    tokens.append(next(iter(stop)))
                    continue
                token = int(np.argmax(logits[index]))
                if token in stop:
                    finished[index], at_limit[index] = True, False
                    tokens.append(token)
                    continue
                generated[index].append(token)
                if item.stop_at_json and "}" in self._tokenizer.decode([token], skip_special_tokens=True):
                    if json_object_span(item.answer_prefix + self._tokenizer.decode(
                            generated[index], skip_special_tokens=True)) is not None:
                        finished[index], at_limit[index] = True, False
                if len(generated[index]) >= item.max_new_tokens:
                    finished[index] = True
                tokens.append(token)
            if all(finished):
                break
            binding = self._session.io_binding()
            binding.bind_cpu_input("input_ids", np.asarray(tokens, dtype=np.int64).reshape(count, 1))
            binding.bind_cpu_input("attention_mask",
                                   np.concatenate((prompt_mask, np.ones((count, step + 1), dtype=np.int64)), axis=1))
            binding.bind_cpu_input("position_ids",
                                   np.asarray([length + step for length in lengths], dtype=np.int64).reshape(count, 1))
            for name, value in cache.items():
                binding.bind_ortvalue_input(name, value)
            binding.bind_output("logits", "cpu")
            for name in self._present_names:
                binding.bind_output(name, self._device, 0)
            try:
                self._session.run_with_iobinding(binding)
                outputs = binding.get_outputs()
            except Exception as error:
                raise LocalModelError("Ошибка вычисления локальной AI-модели.") from error
            values = outputs[0].numpy()
            if values.ndim != 3 or values.shape[0] != count or values.shape[2] != self.spec["vocabulary_size"]:
                raise LocalModelError("Локальная AI-модель вернула неожиданный результат.")
            logits = [np.asarray(values[index, -1], dtype=np.float32) for index in range(count)]
            cache = dict(zip(self._past_names, outputs[1:], strict=True))
            step += 1
        return [LocalCompletion(text=item.answer_prefix + self._tokenizer.decode(generated[index], skip_special_tokens=True),
                                prompt_tokens=len(prompts[index]), completion_tokens=len(generated[index]),
                                stopped_at_limit=at_limit[index])
                for index, item in enumerate(requests)]
