"""Установка языков перевода и механика самообучения."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.pilot import languages
from app.pilot.translator import TranslationError
from app.relevance_learning import (
    MAX_SAMPLES_PER_RUN, RelevanceModel, Trainer, auc, discovery_decisions, run_samples, set_feedback,
)
from app.ui.api_client import ApiError, parse_translation

FILES = {"onnx/encoder_model_quantized.onnx": b"encoder", "onnx/decoder_model_quantized.onnx": b"decoder",
         "tokenizer.json": b"{}", "onnx/decoder_with_past_model_quantized.onnx": b"past"}


class FakeHub:
    def __init__(self, *, corrupt: str | None = None, model_type: str = "marian"):
        self.corrupt, self.model_type = corrupt, model_type
        self.requests: list[str] = []

    def json(self, url):
        self.requests.append(url)
        if "/api/models/" in url:
            return {"sha": "c" * 40, "siblings": [
                {"rfilename": name, "size": len(body),
                 "lfs": {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}}
                for name, body in FILES.items()] + [{"rfilename": "config.json"}]}
        return {"model_type": self.model_type, "decoder_start_token_id": 7, "eos_token_id": 0,
                "pad_token_id": 7, "vocab_size": 8}

    def download(self, url, target: Path, limit, progress):
        self.requests.append(url)
        remote = url.split("/resolve/" + "c" * 40 + "/", 1)[1]
        body = FILES[remote] + (b"!" if remote == self.corrupt else b"")
        target.write_bytes(body)
        progress(len(body))
        return hashlib.sha256(body).hexdigest()

    def close(self):
        pass


def test_language_install_pins_revision_and_checksums(tmp_path):
    language = languages.Language("de", "Немецкий", "Xenova/opus-mt-en-de")
    progress = []
    answer = languages.install_language(tmp_path, language, http=FakeHub(), progress=lambda *step: progress.append(step))
    assert answer["revision"] == "c" * 40 and progress[-1][0] == progress[-1][1]
    spec = json.loads(languages.spec_path(tmp_path, language).read_text(encoding="utf-8"))
    assert spec["target_language"] == "de" and spec["pad_token_id"] == 7
    assert {item["name"] for item in spec["files"]} == set(languages.REQUIRED)
    assert spec["optional_files"][0]["name"] == languages.CACHED_DECODER
    assert languages.installed_spec(tmp_path, language)["revision"] == "c" * 40
    assert languages.looks_installed(tmp_path, language)
    languages.remove_language(tmp_path, language)
    assert not languages.looks_installed(tmp_path, language)


def test_language_install_refuses_tampered_or_foreign_models(tmp_path):
    language = languages.Language("fr", "Французский", "Xenova/opus-mt-en-fr")
    with pytest.raises(TranslationError, match="контрольной"):
        languages.install_language(tmp_path, language, http=FakeHub(corrupt="tokenizer.json"))
    assert not languages.language_model_directory(tmp_path, language).exists()
    with pytest.raises(TranslationError, match="Marian"):
        languages.install_language(tmp_path, language, http=FakeHub(model_type="bert"))


def test_language_validation_and_settings(tmp_path):
    with pytest.raises(ValueError):
        languages.validate_language("EN", "x", "a/b")
    with pytest.raises(ValueError):
        languages.validate_language("de", "<b>", "a/b")
    with pytest.raises(ValueError):
        languages.validate_language("de", "Немецкий", "no-slash")
    manager = languages.LanguageManager(tmp_path)
    assert manager.default() == "ru" and manager.reading() == [{"code": "ru", "name": "Русский"}]
    with pytest.raises(ValueError):
        manager.act({"op": "default", "code": "de"})
    (tmp_path / languages.SETTINGS_FILE).write_text("{broken", encoding="utf-8")
    assert list(languages.load_settings(tmp_path)["languages"]) == ["ru"]


def test_site_parses_offered_languages_strictly():
    translation = parse_translation({"state": "ready", "texts": {"a": "b"}, "language": "de",
                                     "languages": [{"code": "ru", "name": "Русский"}, {"code": "de", "name": "Немецкий"}]})
    assert translation.language == "de" and translation.languages == (("ru", "Русский"), ("de", "Немецкий"))
    assert parse_translation({"state": "running"}).languages == ()
    with pytest.raises(ApiError):
        parse_translation({"state": "ready", "language": "DE"})
    with pytest.raises(ApiError):
        parse_translation({"state": "ready", "languages": [{"code": "de"}]})


def test_samples_are_balanced_and_labels_come_from_independent_signals():
    pool = []
    for index in range(700):
        pool.append({"publication_id": hashlib.sha256(str(index).encode()).hexdigest(), "title": f"Item {index}",
                     "source_id": "arxiv", "kind": "preprint", "study_ids": [f"s{index}"]})
    discovery = {f"s{index}": (0.9, "retained") if index < 100 else (0.7, "below_semantic_threshold")
                 for index in range(700)}
    record = run_samples("run-1", {"original_query": "q"}, pool, discovery=discovery)
    labels = [sample["label"] for sample in record["samples"]]
    assert len(labels) == MAX_SAMPLES_PER_RUN and labels.count(1) == 100
    trend = dict(pool[0], study_ids=[], trend={"confidence": "high"})
    ambiguous = dict(pool[1], study_ids=[])
    record = run_samples("run-2", {}, [trend, ambiguous], relevance={ambiguous["publication_id"]: {"cosine": 0.8}})
    assert [(sample["label"], sample["origin"]) for sample in record["samples"]] == [(1, "trend")]
    assert discovery_decisions([{"study_id": "a", "score": 0.8, "decision": "retained"}, {"study_id": 3}]) == {
        "a": (0.8, "retained")}


def test_auc_and_model_round_trip(tmp_path):
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert auc([0.1], [1]) is None
    model = RelevanceModel(weights={"bias": 0.5, "lexical": 2.0}, version="v1", blend=0.2,
                           sources={name: {"items": 30, "relevant": relevant, "precision": (relevant + 1) / 32}
                                    for name, relevant in (("arxiv", 25), ("habr", 3), ("github", 12))})
    assert RelevanceModel.from_json(model.to_json()).weights == model.weights
    weights = model.source_weights()
    assert weights["arxiv"] > 1 > weights["habr"] and all(0.4 <= value <= 1.6 for value in weights.values())
    assert RelevanceModel.from_json({"learning_version": "other"}).ready is False


def test_feedback_is_validated(tmp_path):
    item = {"publication_id": "a" * 64, "title": "T"}
    assert set_feedback(tmp_path, "run-1", item, 0)["marked"] == 1
    with pytest.raises(ValueError):
        set_feedback(tmp_path, "run 1", item, 0)
    with pytest.raises(ValueError):
        set_feedback(tmp_path, "run-1", item, 2)


def test_trainer_runs_one_training_at_a_time(tmp_path):
    trainer = Trainer(tmp_path)
    pending = []
    assert trainer.request(start=pending.append) is True
    assert trainer.request(start=pending.append) is False  # Второй запрос — ещё одно обучение после первого.
    pending[0]()
    assert trainer.running is False and trainer.last_model is not None
    assert trainer.last_model.metrics["state"] == "collecting"
