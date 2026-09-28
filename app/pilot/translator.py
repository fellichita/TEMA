"""Offline Russian to English suggestion for the search formulation.

This is a small general-purpose machine translation model, not a terminology
authority. It exists to save typing: the panel shows what it produced and the
user confirms or corrects it before any search runs. Nothing here decides a
scientific question, and no output of this module reaches an evidence card.

Measured on the pinned build: a scientific direction of a few words translates
in well under a second, and about half of such phrases need a word corrected by
hand, because domain terms like "вычисления" come back as "calculations" rather
than "computing". Treat every result as a draft.

The module never downloads anything and never reaches the network.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Callable, Protocol

from app.identity import default_data_dir
from app.runtime.model_resources import resolve_model

MAX_SOURCE_CHARACTERS = 400
MODEL_KEY = "opus-mt-ru-en"
# The reverse direction drafts Russian versions of found English titles and
# abstracts for reading; like the forward one, it never feeds an evidence card.
ENGLISH_RUSSIAN_KEY = "opus-mt-en-ru"
# model key -> (specification file, pinned model id, source language, target language)
# The decoder that reuses its own keys and values between steps. Optional: the
# reading direction works without it, only about three times slower.
CACHED_DECODER = "decoder_with_past_model_quantized.onnx"
# Нормализатор модели языка, добавленного владельцем: используется как опубликован,
# а нулевой Precompiled заменяется на NFKC так же, как у встроенных моделей.
PUBLISHED_NORMALIZER = "published-or-nfkc/1.0.0"
SPECS = {MODEL_KEY: ("translator-spec.json", "Xenova/opus-mt-ru-en", "ru", "en"),
         ENGLISH_RUSSIAN_KEY: ("translator-en-ru-spec.json", "Xenova/opus-mt-en-ru", "en", "ru")}


class Cancellation(Protocol):
    def is_set(self) -> bool: ...


class TranslationError(ValueError):
    """A safe, actionable failure that never carries model paths or user text."""


def install_command(model_key: str = MODEL_KEY) -> str:
    return ("python -m scripts.install_translation_model" if model_key == MODEL_KEY
            else "python -m scripts.install_translation_model --direction en-ru")


def load_spec(model_key: str = MODEL_KEY) -> dict[str, Any]:
    """Read the specification shipped with application code, never from a model directory."""
    if model_key not in SPECS:
        raise TranslationError("Неизвестная модель перевода.")
    name, model_id, source, target = SPECS[model_key]
    spec = json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))
    return validate_spec(spec, model_id=model_id, source=source, target=target)


def validate_spec(spec: dict[str, Any], *, model_id: str, source: str, target: str) -> dict[str, Any]:
    """Проверить спецификацию модели: встроенную или закреплённую при установке языка."""
    if (not isinstance(spec, dict) or spec.get("schema_version") != 1 or spec.get("model_id") != model_id
            or not re.fullmatch(r"[a-f0-9]{40}", spec.get("revision", ""))
            or spec.get("source_language") != source or spec.get("target_language") != target
            or type(spec.get("vocabulary_size")) is not int
            or type(spec.get("decoder_start_token_id")) is not int
            or type(spec.get("eos_token_id")) is not int
            or not 1 <= spec.get("beam_width", 0) <= 8
            or not 1 <= spec.get("max_new_tokens", 0) <= 256):
        raise TranslationError("Повреждена встроенная спецификация модели перевода.")
    if len(spec["files"]) != 3 or {item["name"] for item in spec["files"]} != {
            "encoder_model_quantized.onnx", "decoder_model_quantized.onnx", "tokenizer.json"}:
        raise TranslationError("Неверный состав модели перевода.")
    optional = spec.get("optional_files", [])
    if not isinstance(optional, list) or {item["name"] for item in optional} - {CACHED_DECODER}:
        raise TranslationError("Неверный состав модели перевода.")
    for item in [*spec["files"], *optional]:
        if (not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= 200_000_000):
            raise TranslationError("Повреждена встроенная спецификация модели перевода.")
    return spec


def verify_optional(directory: Path, spec: dict[str, Any], name: str) -> Path | None:
    """An optional speed-up file, used only when it is present with its pinned bytes."""
    item = next((item for item in spec.get("optional_files", ()) if item["name"] == name), None)
    if item is None:
        return None
    path = Path(directory) / name
    digest = hashlib.sha256()
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
            return None
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return path if digest.hexdigest() == item["sha256"] else None


def model_directory(data_dir: Path | None = None, model_key: str = MODEL_KEY) -> Path:
    return (default_data_dir() if data_dir is None else Path(data_dir)) / "models" / model_key


def spec_key(spec: dict[str, Any]) -> str:
    return next((key for key, (_, model_id, _, _) in SPECS.items() if model_id == spec.get("model_id")), MODEL_KEY)


def checkpoint(cancel: Cancellation | None) -> None:
    if cancel is not None and cancel.is_set():
        raise TranslationError("Перевод отменён.")


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
                raise TranslationError("Модель перевода отсутствует или повреждена. "
                                       f"Установите её командой {install_command(spec_key(spec))}.")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    checkpoint(cancel)
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise TranslationError("Контрольная сумма модели перевода не совпадает.")
        except OSError as error:
            raise TranslationError("Не удалось прочитать модель перевода.") from error
    return directory


def readable_tokenizer(raw: bytes, spec: dict[str, Any]) -> str:
    """Return the pinned tokenizer with a normalizer this runtime can build.

    The published export stores ``{"type": "Precompiled", "precompiled_charsmap":
    null}`` because its own JavaScript runtime applies that step separately. The
    Rust tokenizers library refuses a null map, so the one unusable field is
    replaced by the NFKC normalization it stands for. The file on disk keeps its
    published bytes and its pinned checksum; only this in-memory copy differs,
    and the substitution is versioned by ``normalizer_version``.
    """
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise TranslationError("Токенизатор модели перевода повреждён.") from error
    normalizer = document.get("normalizer")
    if spec.get("normalizer_version") == PUBLISHED_NORMALIZER and not (
            isinstance(normalizer, dict) and normalizer.get("type") == "Precompiled"
            and normalizer.get("precompiled_charsmap") is None):
        # Модель добавленного языка: её нормализатор уже читается библиотекой как есть.
        return raw.decode("utf-8")
    if not isinstance(normalizer, dict) or normalizer.get("type") != "Precompiled" or \
            normalizer.get("precompiled_charsmap") is not None:
        raise TranslationError("Токенизатор модели перевода изменился; проверьте закреплённую ревизию.")
    if spec["normalizer_version"] not in {"marian-nfkc-normalizer/1.0.0", PUBLISHED_NORMALIZER}:
        raise TranslationError("Неизвестная версия нормализации токенизатора перевода.")
    document["normalizer"] = {"type": "NFKC"}
    return json.dumps(document, ensure_ascii=False)


class RussianEnglishTranslator:
    """CPU int8 Marian encoder/decoder with bounded beam search, no cache reuse."""

    model_key = MODEL_KEY
    # Standard beam rules: only candidates ranked into the beam may end it, and
    # search stops once no running beam can beat the kept endings. The query
    # direction keeps its measured legacy rules.
    ranked_endings = False

    def __init__(self, directory: Path | None = None, *, cancel: Cancellation | None = None,
                 spec: dict[str, Any] | None = None):
        import numpy as np
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer

        from app.runtime.inference import session_threads

        if spec is None:
            self.spec = load_spec(self.model_key)
            location = resolve_model(self.model_key, self.spec["revision"], explicit_dir=directory,
                                     development_default=model_directory(model_key=self.model_key))
            directory = verify_artifacts(location.path, self.spec, cancel)
        else:
            # Язык, добавленный владельцем: спецификация закреплена при установке.
            if directory is None:
                raise TranslationError("Для добавленного языка нужен каталог его модели.")
            self.spec = spec
            directory = verify_artifacts(Path(directory), self.spec, cancel)
        self.directory = directory
        self._np = np
        self._tokenizer = Tokenizer.from_str(
            readable_tokenizer((directory / "tokenizer.json").read_bytes(), self.spec))
        self._tokenizer.enable_truncation(max_length=self.spec["max_source_tokens"])
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = session_threads()
        options.log_severity_level = 3
        self._session_options = options
        checkpoint(cancel)
        try:
            self._encoder = ort.InferenceSession(str(directory / "encoder_model_quantized.onnx"),
                                                 sess_options=options, providers=["CPUExecutionProvider"])
            self._decoder = ort.InferenceSession(str(directory / "decoder_model_quantized.onnx"),
                                                 sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as error:
            raise TranslationError("ONNX Runtime не смог загрузить модель перевода.") from error

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True, ensure_ascii=False)
                              .encode("utf-8")).hexdigest()

    def _log_softmax(self, values):
        np = self._np
        values = values - values.max(axis=-1, keepdims=True)
        return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))

    def translate(self, text: str, *, cancel: Cancellation | None = None) -> str:
        """Return one English draft; an empty result means the model produced nothing."""
        np = self._np
        if not isinstance(text, str):
            raise TranslationError("Для перевода нужна строка.")
        text = " ".join(text.split())
        if not text or len(text) > MAX_SOURCE_CHARACTERS:
            raise TranslationError(f"Текст для перевода пуст или длиннее {MAX_SOURCE_CHARACTERS} символов.")
        checkpoint(cancel)
        source = np.asarray([self._tokenizer.encode(text).ids], dtype=np.int64)
        if not source.size:
            raise TranslationError("Текст для перевода не содержит известных модели символов.")
        mask = np.ones_like(source)
        hidden = self._encoder.run(["last_hidden_state"],
                                   {"input_ids": source, "attention_mask": mask})[0]
        start, end = self.spec["decoder_start_token_id"], self.spec["eos_token_id"]
        width = self.spec["beam_width"]
        beams: list[tuple[list[int], float]] = [([start], 0.0)]
        finished: list[tuple[list[int], float]] = []
        for _ in range(self.spec["max_new_tokens"]):
            checkpoint(cancel)
            rows = [sequence for sequence, _ in beams]
            logits = self._decoder.run(["logits"], {
                "input_ids": np.asarray(rows, dtype=np.int64),
                "encoder_hidden_states": np.repeat(hidden, len(rows), axis=0),
                "encoder_attention_mask": np.repeat(mask, len(rows), axis=0)})[0]
            scores = self._log_softmax(logits[:, -1, :].astype(np.float32))
            pool: list[tuple[list[int], float]] = []
            for index, (sequence, total) in enumerate(beams):
                for token in np.argpartition(-scores[index], width)[:width]:
                    pool.append((sequence + [int(token)], total + float(scores[index, token])))
            # Length normalisation keeps a short beam from winning by brevity alone.
            pool.sort(key=lambda item: -item[1] / len(item[0]))
            beams = []
            for sequence, total in pool:
                if sequence[-1] == end:
                    finished.append((sequence, total / len(sequence)))
                elif len(beams) < width:
                    beams.append((sequence, total))
                elif self.ranked_endings:
                    # An unlikely early end ranked below the beam would
                    # otherwise win by length normalisation and drop the tail.
                    break
            if not beams:
                break
            if len(finished) >= width:
                if not self.ranked_endings:
                    break
                kept = sorted((score for _, score in finished), reverse=True)[width - 1]
                if max(total / len(sequence) for sequence, total in beams) <= kept:
                    break
        best = max(finished, key=lambda item: item[1])[0] if finished else beams[0][0]
        body = [token for token in best if token not in (start, end)]
        return " ".join(self._tokenizer.decode(body, skip_special_tokens=True).split())


# A sentence ends at terminal punctuation followed by the start of the next one.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


QUOTED = re.compile(r'"\s*([^"]+?)\s*"')
# Curly and angle quotes of the source come back as doubled angle brackets.
ANGLE_QUOTED = re.compile(r"<\s*<\s*(.+?)\s*>\s*>")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([:;,.!?)»])")
# Links and DOIs pass through untouched: the model "translates" 10.5281 into 10,5281.
VERBATIM = re.compile(r"(https?://\S+|\b10\.\d{4,9}/\S+)")
SPACE_AFTER_OPENING = re.compile(r"([(«])\s+")
# Pieces decoded together; bounds the key/value cache to a few hundred megabytes.
BATCH_PIECES = 24


def reading_case(text: str) -> str:
    """An all-caps title reads to Marian as a string of acronyms; give it sentence case."""
    letters = [character for character in text if character.isalpha()]
    if len(letters) >= 8 and sum(character.isupper() for character in letters) / len(letters) > 0.8:
        lowered = text.lower()
        return lowered[:1].upper() + lowered[1:]
    return text


class EnglishRussianTranslator(RussianEnglishTranslator):
    """The same runtime in the reverse direction: a Russian reading draft of English text.

    With the cached decoder every piece of a whole batch is decoded together:
    one decoder call per step serves all of them, and each call reuses the keys
    and values of earlier steps instead of reading the whole prefix again. A
    call costs about the same for two rows or seventy, so the TOP translates in
    seconds instead of a minute.
    """

    model_key = ENGLISH_RUSSIAN_KEY
    ranked_endings = True

    def __init__(self, directory: Path | None = None, *, cancel: Cancellation | None = None,
                 spec: dict[str, Any] | None = None):
        super().__init__(directory, cancel=cancel, spec=spec)
        import onnxruntime as ort

        self._cached = None
        path = verify_optional(self.directory, self.spec, CACHED_DECODER)
        if path is not None:
            try:
                self._cached = ort.InferenceSession(str(path), sess_options=self._session_options,
                                                    providers=["CPUExecutionProvider"])
            except Exception:
                self._cached = None  # The full decoder still translates, only slower.
        if self._cached is not None:
            self._outputs = [output.name for output in self._decoder.get_outputs()]
            self._cached_outputs = [output.name for output in self._cached.get_outputs()]
            self._layers = sum(name.endswith(".decoder.key") for name in self._outputs)

    def translate_text(self, text: str, *, cancel: Cancellation | None = None) -> str:
        return self.translate_many([text], cancel=cancel)[0]

    def translate_many(self, texts: list[str], *, cancel: Cancellation | None = None,
                       done: Callable[[int], None] | None = None) -> list[str]:
        """Translate longer texts sentence by sentence, each piece within the model limit.

        Marian is trained on sentences: a whole abstract in one pass loses its
        tail, and a long sentence is split at a word boundary instead. The
        training data left HTML entities and spaced straight quotes in the
        output; they become characters and Russian guillemets, and a space
        before punctuation goes. Links and DOIs are kept as written. `done`
        hears the index of each text as soon as all its pieces are translated.
        """
        plans: list[list[str | int]] = []
        sources: list[str] = []
        owners: list[int] = []
        for number, text in enumerate(texts):
            plan: list[str | int] = []
            for sentence in SENTENCE_END.split(" ".join(text.split())):
                for part in VERBATIM.split(sentence):
                    part = part.strip()
                    if VERBATIM.fullmatch(part) or not any(character.isalpha() for character in part):
                        plan.append(part)
                        continue
                    # Per sentence: an all-caps label like "SHORT PAPER." opens many abstracts.
                    part = reading_case(part)
                    while part:
                        chunk = part[:MAX_SOURCE_CHARACTERS]
                        if len(part) > MAX_SOURCE_CHARACTERS and chunk.rfind(" ") > 0:
                            chunk = chunk[:chunk.rfind(" ")]
                        plan.append(len(sources))
                        sources.append(chunk)
                        owners.append(number)
                        part = part[len(chunk):].strip()
            plans.append(plan)
        remaining = [sum(isinstance(item, int) for item in plan) for plan in plans]

        def piece_done(index: int) -> None:
            remaining[owners[index]] -= 1
            if remaining[owners[index]] == 0 and done is not None:
                done(owners[index])

        if done is not None:
            for number, count in enumerate(remaining):
                if count == 0:
                    done(number)
        drafts = self._decode(sources, cancel, piece_done)
        return [" ".join(piece for piece in (readable(drafts[item]) if isinstance(item, int) else item
                                             for item in plan) if piece)
                for plan in plans]

    def _decode(self, sources: list[str], cancel: Cancellation | None,
                piece_done: Callable[[int], None]) -> list[str]:
        cached = getattr(self, "_cached", None)
        if cached is None:
            drafts = []
            for index, source in enumerate(sources):
                drafts.append(self.translate(source, cancel=cancel))
                piece_done(index)
            return drafts
        drafts = [""] * len(sources)
        # Similar lengths share a batch: less padding, and short pieces finish early.
        order = sorted(range(len(sources)), key=lambda index: len(sources[index]))
        for begin in range(0, len(order), BATCH_PIECES):
            chunk = order[begin:begin + BATCH_PIECES]

            def complete_chunk(position: int, indexes: list[int] = chunk) -> None:
                piece_done(indexes[position])

            outputs = self._batch([sources[index] for index in chunk], cancel, complete_chunk)
            for index, draft in zip(chunk, outputs, strict=True):
                drafts[index] = draft
        return drafts

    def _batch(self, sentences: list[str], cancel: Cancellation | None,
               finished: Callable[[int], None]) -> list[str]:
        """Beam search over a batch of sentences with the cached decoder.

        Every sentence keeps its own beams, endings and stopping rule — the
        same ranked rules as the single-sentence search; its rows leave the
        batch once it stops.
        """
        cached = self._cached
        assert cached is not None  # _decode selects this path only when the optional decoder loaded.
        np = self._np
        checkpoint(cancel)
        pad, start, end = (self.spec["pad_token_id"], self.spec["decoder_start_token_id"],
                           self.spec["eos_token_id"])
        width = self.spec["beam_width"]
        encoded = [self._tokenizer.encode(" ".join(sentence.split())).ids for sentence in sentences]
        if not all(encoded):
            raise TranslationError("Текст для перевода не содержит известных модели символов.")
        count, length = len(encoded), max(map(len, encoded))
        source = np.full((count, length), pad, dtype=np.int64)
        mask = np.zeros((count, length), dtype=np.int64)
        for row, ids in enumerate(encoded):
            source[row, :len(ids)], mask[row, :len(ids)] = ids, 1
        hidden = self._encoder.run(["last_hidden_state"], {"input_ids": source, "attention_mask": mask})[0]
        outputs = dict(zip(self._outputs, self._decoder.run(None, {
            "input_ids": np.full((count, 1), start, dtype=np.int64),
            "encoder_hidden_states": hidden, "encoder_attention_mask": mask}), strict=True))
        encoder_past = [(outputs[f"present.{layer}.encoder.key"], outputs[f"present.{layer}.encoder.value"])
                        for layer in range(self._layers)]
        decoder_past = [(outputs[f"present.{layer}.decoder.key"], outputs[f"present.{layer}.decoder.value"])
                        for layer in range(self._layers)]
        logits = outputs["logits"][:, -1, :]
        beams: list[list[tuple[list[int], float]]] = [[([start], 0.0)] for _ in range(count)]
        endings: list[list[tuple[list[int], float]]] = [[] for _ in range(count)]
        active = list(range(count))
        for _ in range(self.spec["max_new_tokens"]):
            checkpoint(cancel)
            scores = self._log_softmax(logits.astype(np.float32))
            row, parents, owners, still = 0, [], [], []
            for sentence in active:
                pool = []
                for offset, (sequence, total) in enumerate(beams[sentence]):
                    for token in np.argpartition(-scores[row + offset], width)[:width]:
                        pool.append((sequence + [int(token)], total + float(scores[row + offset, token]),
                                     row + offset))
                row += len(beams[sentence])
                pool.sort(key=lambda item: -item[1] / len(item[0]))
                kept: list[tuple[list[int], float, int]] = []
                for sequence, total, parent in pool:
                    if sequence[-1] == end:
                        endings[sentence].append((sequence, total / len(sequence)))
                    elif len(kept) < width:
                        kept.append((sequence, total, parent))
                    else:
                        break
                if kept:
                    beams[sentence] = [(sequence, total) for sequence, total, _ in kept]
                if not kept or len(endings[sentence]) >= width and (
                        max(total / len(sequence) for sequence, total, _ in kept)
                        <= sorted((score for _, score in endings[sentence]), reverse=True)[width - 1]):
                    finished(sentence)
                    continue
                still.append(sentence)
                parents += [parent for _, _, parent in kept]
                owners += [sentence] * len(kept)
            active = still
            if not active:
                break
            rows, sentence_rows = np.asarray(parents), np.asarray(owners)
            feeds = {"input_ids": np.asarray([[sequence[-1]] for sentence in active
                                              for sequence, _ in beams[sentence]], dtype=np.int64),
                     "encoder_attention_mask": mask[sentence_rows]}
            for layer in range(self._layers):
                feeds[f"past_key_values.{layer}.decoder.key"] = decoder_past[layer][0][rows]
                feeds[f"past_key_values.{layer}.decoder.value"] = decoder_past[layer][1][rows]
                feeds[f"past_key_values.{layer}.encoder.key"] = encoder_past[layer][0][sentence_rows]
                feeds[f"past_key_values.{layer}.encoder.value"] = encoder_past[layer][1][sentence_rows]
            outputs = dict(zip(self._cached_outputs, cached.run(None, feeds), strict=True))
            decoder_past = [(outputs[f"present.{layer}.decoder.key"], outputs[f"present.{layer}.decoder.value"])
                            for layer in range(self._layers)]
            logits = outputs["logits"][:, -1, :]
        for sentence in active:
            finished(sentence)  # The token limit ended these.
        drafts = []
        for sentence in range(count):
            best = (max(endings[sentence], key=lambda item: item[1])[0] if endings[sentence]
                    else beams[sentence][0][0])
            body = [token for token in best if token not in (start, end)]
            drafts.append(" ".join(self._tokenizer.decode(body, skip_special_tokens=True).split()))
        return drafts


def readable(draft: str) -> str:
    """Tidy one model output: entities, quotes and spaces before punctuation."""
    text = ANGLE_QUOTED.sub(r"«\1»", QUOTED.sub(r"«\1»", html.unescape(draft)))
    return SPACE_AFTER_OPENING.sub(r"\1", SPACE_BEFORE_PUNCTUATION.sub(r"\1", text))
