"""Локальный черновой перевод: закрепление, отказы и отсутствие тихой подмены."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.pilot import translator
from app.pilot.translator import TranslationError, load_spec, readable_tokenizer, verify_artifacts
from app.pilot.service import PilotService
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskFailure


PUBLISHED = {"normalizer": {"type": "Precompiled", "precompiled_charsmap": None},
             "model": {"type": "Unigram", "vocab": []}}


def test_specification_pins_three_files_with_exact_sizes_and_digests():
    spec = load_spec()
    assert spec["model_id"] == "Xenova/opus-mt-ru-en"
    assert len(spec["revision"]) == 40
    names = {item["name"] for item in spec["files"]}
    assert names == {"encoder_model_quantized.onnx", "decoder_model_quantized.onnx", "tokenizer.json"}
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in spec["files"])
    # A draft translator must not be able to claim more than one sentence of input.
    assert spec["max_new_tokens"] <= 256 and 1 <= spec["beam_width"] <= 8


def test_only_the_unusable_null_normalizer_is_replaced_in_memory():
    spec = load_spec()
    document = json.loads(readable_tokenizer(json.dumps(PUBLISHED).encode("utf-8"), spec))
    assert document["normalizer"] == {"type": "NFKC"}
    assert document["model"] == PUBLISHED["model"], "остальной токенизатор не трогаем"


@pytest.mark.parametrize("normalizer", [
    {"type": "Precompiled", "precompiled_charsmap": "AAAA"},  # карта появилась: ревизия другая
    {"type": "NFKC"},                                          # уже заменён кем-то ещё
    None,
])
def test_a_changed_tokenizer_is_refused_instead_of_silently_patched(normalizer):
    spec = load_spec()
    with pytest.raises(TranslationError, match="изменил|поврежд"):
        readable_tokenizer(json.dumps(PUBLISHED | {"normalizer": normalizer}).encode("utf-8"), spec)


def test_broken_tokenizer_bytes_are_a_safe_error():
    with pytest.raises(TranslationError, match="поврежд"):
        readable_tokenizer(b"{not json", load_spec())


def test_unknown_normalizer_version_is_refused():
    spec = load_spec() | {"normalizer_version": "marian-nfkc-normalizer/9.9.9"}
    with pytest.raises(TranslationError, match="версия нормализации"):
        readable_tokenizer(json.dumps(PUBLISHED).encode("utf-8"), spec)


def test_missing_and_damaged_files_are_named_without_leaking_paths(tmp_path):
    spec = load_spec()
    with pytest.raises(TranslationError, match="отсутствует или повреждена"):
        verify_artifacts(tmp_path, spec)
    for item in spec["files"]:
        (tmp_path / item["name"]).write_bytes(b"\0" * item["bytes"])
    with pytest.raises(TranslationError, match="Контрольная сумма"):
        verify_artifacts(tmp_path, spec)


def test_cancelled_verification_stops_before_reading_everything(tmp_path):
    spec = load_spec()
    for item in spec["files"]:
        (tmp_path / item["name"]).write_bytes(b"\0" * item["bytes"])

    class Cancelled:
        def is_set(self):
            return True

    with pytest.raises(TranslationError, match="отменён"):
        verify_artifacts(tmp_path, spec, Cancelled())


class StubTranslator:
    """Stands in for 117 MB of pinned weights; the service contract is what is tested."""
    created = 0

    def __init__(self, *_args, **_kwargs):
        type(self).created += 1
        self.spec = load_spec()

    def translate(self, text, *, cancel=None):
        return "" if text.strip() == "пусто" else "selective lithium membranes"


@pytest.fixture
def service(tmp_path, monkeypatch):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "get", lambda _name: None)
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda _path, **kwargs: None)
    StubTranslator.created = 0
    monkeypatch.setattr(translator, "RussianEnglishTranslator", StubTranslator)
    instance = PilotService(tmp_path / "data", credentials)
    try:
        yield instance
    finally:
        instance.close()


def test_draft_is_returned_for_review_and_never_marked_as_checked(service):
    draft = service.translate_query("Селективные мембраны для извлечения лития")
    assert draft["english_query"] == "selective lithium membranes"
    assert draft["reviewed"] is False, "машинный перевод не выдаётся за проверенный человеком"
    assert draft["model_id"] == "Xenova/opus-mt-ru-en" and len(draft["revision"]) == 40


def test_loaded_sessions_are_reused_across_requests(service):
    service.translate_query("Первый запрос")
    service.translate_query("Второй запрос")
    assert StubTranslator.created == 1


@pytest.mark.parametrize("text", ["", "   ", None, 5])
def test_empty_or_invalid_input_is_a_readable_refusal(service, text):
    with pytest.raises(TaskFailure, match="Введите направление"):
        service.translate_query(text)


def test_empty_model_output_asks_the_user_instead_of_searching_for_nothing(service):
    with pytest.raises(TaskFailure, match="Впишите формулировку сами"):
        service.translate_query("пусто")


def test_a_missing_model_is_an_actionable_failure_not_a_crash(service, monkeypatch):
    def absent(*_args, **_kwargs):
        raise TranslationError("Модель перевода отсутствует или повреждена. "
                               "Установите её командой python -m scripts.install_translation_model.")

    monkeypatch.setattr(translator, "RussianEnglishTranslator", absent)
    with pytest.raises(TaskFailure, match="install_translation_model"):
        service.translate_query("Селективные мембраны")


def test_specification_digest_is_stable_for_the_fingerprint():
    spec = load_spec()
    first = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    second = hashlib.sha256(json.dumps(load_spec(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert first == second


def test_reading_direction_is_pinned_separately_and_names_its_own_install_command(tmp_path):
    spec = load_spec(translator.ENGLISH_RUSSIAN_KEY)
    assert spec["model_id"] == "Xenova/opus-mt-en-ru"
    assert (spec["source_language"], spec["target_language"]) == ("en", "ru")
    assert spec["revision"] != load_spec()["revision"]
    assert translator.model_directory(tmp_path, translator.ENGLISH_RUSSIAN_KEY) == tmp_path / "models" / "opus-mt-en-ru"
    with pytest.raises(TranslationError, match="--direction en-ru"):
        verify_artifacts(tmp_path, spec)
    with pytest.raises(TranslationError, match="Неизвестная модель"):
        load_spec("opus-mt-xx-yy")


class Drafts(translator.EnglishRussianTranslator):
    """The text pipeline around the model, with the model replaced by canned output."""

    def __init__(self, outputs):
        self.outputs, self.seen = dict(outputs), []

    def translate(self, text, *, cancel=None):
        self.seen.append(text)
        return self.outputs.get(text, "")


def test_reading_translation_goes_sentence_by_sentence_and_cleans_the_output():
    drafts = Drafts({"Short paper.": "Краткая статья .",
                     "Analytics of &quot;big data&quot; grows.": 'Анализ " больших данных " растёт .',
                     "The course (“Data science”) runs.": "Курс ( < < Наука о данных > > ) идёт ."})
    text = "SHORT PAPER. Analytics of &quot;big data&quot; grows. The course (“Data science”) runs."
    assert drafts.translate_text(text) == ("Краткая статья. Анализ «больших данных» растёт. "
                                           "Курс («Наука о данных») идёт.")
    # An all-caps sentence reaches the model in sentence case; the others as written.
    assert drafts.seen[0] == "Short paper."


def test_a_long_sentence_is_split_at_a_word_boundary_within_the_model_limit():
    drafts = Drafts({})
    drafts.translate_text("word " * 200)
    assert len(drafts.seen) > 1
    assert all(len(piece) <= translator.MAX_SOURCE_CHARACTERS and not piece.endswith(" ")
               for piece in drafts.seen)
    assert " ".join(drafts.seen).split() == ["word"] * 200


def test_only_the_reading_direction_uses_ranked_beam_endings():
    assert translator.EnglishRussianTranslator.ranked_endings is True
    assert translator.RussianEnglishTranslator.ranked_endings is False


def test_links_and_dois_pass_through_the_reading_translation_untouched():
    drafts = Drafts({"Data is on": "Данные на", "and cited as": "и цитируются как", "today.": "сегодня."})
    text = "Data is on https://github.com/lab/data and cited as 10.5281/zenodo.123 today."
    assert drafts.translate_text(text) == ("Данные на https://github.com/lab/data и цитируются как "
                                           "10.5281/zenodo.123 сегодня.")
    assert drafts.seen == ["Data is on", "and cited as", "today."]


def test_the_cached_decoder_is_an_optional_file_used_only_with_its_pinned_bytes(tmp_path):
    spec = load_spec(translator.ENGLISH_RUSSIAN_KEY)
    assert [item["name"] for item in spec["optional_files"]] == [translator.CACHED_DECODER]
    assert "optional_files" not in load_spec()
    content = b"cached decoder"
    small = {**spec, "optional_files": [{"name": translator.CACHED_DECODER, "remote": "x",
                                         "sha256": hashlib.sha256(content).hexdigest(),
                                         "bytes": len(content)}]}
    assert translator.verify_optional(tmp_path, small, translator.CACHED_DECODER) is None
    (tmp_path / translator.CACHED_DECODER).write_bytes(b"cached decoded")
    assert translator.verify_optional(tmp_path, small, translator.CACHED_DECODER) is None
    (tmp_path / translator.CACHED_DECODER).write_bytes(content)
    assert translator.verify_optional(tmp_path, small, translator.CACHED_DECODER) == tmp_path / translator.CACHED_DECODER


def test_installer_adds_the_optional_file_without_touching_the_required_ones(tmp_path, monkeypatch):
    from scripts import install_translation_model as installer

    content = b"cached decoder"
    spec = {**load_spec(translator.ENGLISH_RUSSIAN_KEY),
            "optional_files": [{"name": translator.CACHED_DECODER, "remote": "x",
                                "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}]}
    (tmp_path / "encoder_model_quantized.onnx").write_bytes(b"required")
    fetched = []

    def download(_client, _spec, item, target):
        fetched.append(item["name"])
        target.write_bytes(content)

    monkeypatch.setattr(installer, "_download", download)
    assert installer._install_optional(tmp_path, spec) is True
    assert fetched == [translator.CACHED_DECODER]
    assert (tmp_path / translator.CACHED_DECODER).read_bytes() == content
    assert (tmp_path / "encoder_model_quantized.onnx").read_bytes() == b"required"
    assert not list(tmp_path.glob("*.part"))
    assert installer._install_optional(tmp_path, spec) is False  # Already there: nothing fetched.
    (tmp_path / translator.CACHED_DECODER).write_bytes(b"damaged bytes!")
    with pytest.raises(TranslationError, match="не совпадает"):
        installer._install_optional(tmp_path, spec)


def test_cached_batch_translates_every_text_and_reports_each(monkeypatch):
    # Exercise the real batching and cached beam loop without requiring model files.
    monkeypatch.setattr(translator, "BATCH_PIECES", 2)
    outputs = {
        "Swedish data platform": ["Шведская", "платформа", "данных"],
        "Machine learning improves battery design.": ["Машинное", "обучение", "улучшает", "батареи."],
        "It is fast.": ["Это", "быстро."],
        "See": ["См."],
        "for details.": ["подробнее."],
    }
    source_ids = {source: index + 10 for index, source in enumerate(outputs)}
    words = {word: index + 3 for index, word in enumerate(dict.fromkeys(
        word for sentence in outputs.values() for word in sentence))}
    decoded = {index: word for word, index in words.items()}
    sequences = {source_ids[source]: [words[word] for word in sentence]
                 for source, sentence in outputs.items()}
    eos = 1

    def logits(tokens):
        values = np.full((len(tokens), 1, max(words.values()) + 1), -20.0, dtype=np.float32)
        for row, token in enumerate(tokens):
            values[row, 0, token] = 20.0
        return values

    class Tokenizer:
        def encode(self, source):
            return SimpleNamespace(ids=[source_ids[source]])

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return " ".join(decoded[index] for index in ids)

    class Encoder:
        batch_sizes = []

        def run(self, names, feeds):
            assert names == ["last_hidden_state"]
            self.batch_sizes.append(len(feeds["input_ids"]))
            return [feeds["input_ids"][:, :1, None]]

    class Decoder:
        batch_sizes = []

        def run(self, names, feeds):
            assert names is None
            ids = feeds["encoder_hidden_states"][:, 0, 0]
            self.batch_sizes.append(len(ids))
            past = np.full((len(ids), 1), 1, dtype=np.int64)
            source = ids[:, None]
            return [logits([sequences[int(source_id)][0] for source_id in ids]),
                    source, source, past, past]

    class CachedDecoder:
        batch_sizes = []

        def run(self, names, feeds):
            assert names is None
            ids = feeds["past_key_values.0.encoder.key"][:, 0]
            steps = feeds["past_key_values.0.decoder.key"][:, 0]
            self.batch_sizes.append(len(ids))
            tokens = [sequences[int(source_id)][int(step)] if step < len(sequences[int(source_id)])
                      else eos for source_id, step in zip(ids, steps, strict=True)]
            next_steps = steps[:, None] + 1
            return [logits(tokens), next_steps, next_steps]

    reader = translator.EnglishRussianTranslator.__new__(translator.EnglishRussianTranslator)
    reader.spec = {"pad_token_id": 0, "decoder_start_token_id": 2, "eos_token_id": eos,
                   "beam_width": 1, "max_new_tokens": 8}
    reader._np = np
    reader._tokenizer = Tokenizer()
    reader._encoder = Encoder()
    reader._decoder = Decoder()
    reader._cached = CachedDecoder()
    reader._outputs = ["logits", "present.0.encoder.key", "present.0.encoder.value",
                       "present.0.decoder.key", "present.0.decoder.value"]
    reader._cached_outputs = ["logits", "present.0.decoder.key", "present.0.decoder.value"]
    reader._layers = 1

    finished = []
    drafts = reader.translate_many(["SWEDISH DATA PLATFORM",
                                    "Machine learning improves battery design. It is fast.",
                                    "See https://example.org/a for details."], done=finished.append)
    assert drafts == ["Шведская платформа данных",
                      "Машинное обучение улучшает батареи. Это быстро.",
                      "См. https://example.org/a подробнее."]
    assert finished == [2, 0, 1], "each text is reported when its last piece finishes"
    assert reader._encoder.batch_sizes == reader._decoder.batch_sizes == [2, 2, 1]
    assert reader._cached.batch_sizes == [2, 1, 2, 1, 1, 1, 1, 1, 1]
