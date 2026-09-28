"""Semantic admission through the real preparation, NMF, selection and JSON flow."""

import json
from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.corpus import unpack_snapshot
from app.ml.engine import analyze, prepare_analysis
from app.ml.semantic_contracts import validate_semantic_result
from app.ml.service import export_result, run_analysis
from tests.mvp_fixture import snapshot
from tools.audit_ml_result import BUCKETS, audit_result
from tools.result_schema import ResultContract


class Encoder:
    score = .95

    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    def encode(self, texts, *, kind, cancel=None, progress=None):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1 if kind == "query" else self.score
        if kind != "query":
            vectors[:, 1] = np.sqrt(1 - self.score ** 2)
        if progress is not None:
            progress(len(texts), len(texts))
        return vectors

    def manifest(self):
        return {"model_id": "test/e5-shaped-encoder", "revision": "synthetic-fixture-v1"}


@pytest.fixture(autouse=True)
def local_encoder(monkeypatch):
    monkeypatch.setattr("app.ml.local_encoder.LocalEncoder", Encoder)


def generic_snapshot():
    data = snapshot()
    topic = "thermal energy storage"
    data["history"]["request"]["topic"] = topic
    titles = ["Phase change materials with encapsulated paraffin",
              "Molten salt reservoirs with ceramic insulation",
              "Thermochemical reactors using reversible hydration cycles"]
    for batch, period in zip(data["batches"], data["history"]["periods"], strict=True):
        period["job"]["request"]["topic"] = topic
        for entry in batch["documents"]:
            document = entry["document"]
            category = int(document["source_id"].split("-")[1])
            title = titles[category]
            document["title"] = title + ": experiment " + document["source_id"]
            document["abstract"] = (
                f"{title} enable load shifting between day and night. "
                "Existing systems lose heat through the container walls. "
                f"Here we demonstrate {title.casefold()} under repeated charging cycles. "
                "Improved insulation reduces heat loss and increases discharge efficiency. "
                "We measure retained heat after a twelve hour holding period and compare cycle durability.")
    return data


def options(corpus, mode="semantic"):
    return AnalysisOptions(topic=corpus["topic"], relevance_mode=mode)


@pytest.fixture
def generic_result():
    corpus = unpack_snapshot(generic_snapshot())
    return corpus, analyze(corpus, options(corpus))


def test_guarded_direction_preserves_groups_metrics_evidence_and_preparation():
    corpus = unpack_snapshot(snapshot())
    original = deepcopy(corpus)
    lexical = analyze(corpus, options(corpus, "lexical"))
    semantic = analyze(corpus, options(corpus))
    assert corpus == original
    for field in ("model", "preparation", "temporal_selection", "growth_comparability",
                  "direction_counts", "selection_summary"):
        assert semantic[field] == lexical[field]
    for bucket in BUCKETS:
        stripped = [{k: v for k, v in group.items() if k != "semantic_relevance"}
                    for group in semantic[bucket]]
        assert stripped == lexical[bucket]
    assert semantic["semantic_relevance"]["guarded_direction"]
    assert semantic["semantic_relevance"]["semantic_only_studies"] == 0
    assert semantic["fingerprint"] != lexical["fingerprint"]
    assert ResultContract.model_validate(semantic).model_dump() == semantic
    report = audit_result(corpus, semantic)
    assert report["ok"], report["errors"]


def test_generic_synonyms_enter_only_preliminary_groups_with_original_sources(generic_result):
    corpus, result = generic_result
    lexical = analyze(corpus, options(corpus, "lexical"))
    assert lexical["preparation"]["retained_studies"] == 0
    assert result["preparation"]["retained_studies"] >= 8
    assert result["preliminary_signals"]
    assert result["candidates"] == result["established"] == []
    assert result["semantic_relevance"]["semantic_only_studies"] == result["preparation"]["retained_studies"]
    originals = {entry["document"]["title"] for entry in corpus["entries"]}
    for group in result["preliminary_signals"]:
        assert group["semantic_relevance"]["requires_review"]
        assert group["selection"]["reasons"] == ["semantic_scope_requires_review"]
        assert all(source["title"] in originals for source in group["sources"])
    assert ResultContract.model_validate(result).model_dump() == result
    report = audit_result(corpus, result)
    assert report["ok"], report["errors"]


def test_low_similarity_does_not_fabricate_documents(monkeypatch):
    monkeypatch.setattr(Encoder, "score", .7)
    corpus = unpack_snapshot(generic_snapshot())
    result = analyze(corpus, options(corpus))
    assert result["status"] == "insufficient_data"
    assert result["preparation"]["retained_studies"] == 0
    assert all(not result[bucket] for bucket in BUCKETS)
    assert ResultContract.model_validate(result).model_dump() == result
    assert audit_result(corpus, result)["ok"]


def test_temporal_exclusions_happen_before_encoding(monkeypatch):
    corpus = unpack_snapshot(generic_snapshot())
    entries = deepcopy(corpus["entries"][:3])
    entries[0]["document"]["publication_year"] = 2026
    entries[1]["document"]["raw_metadata"]["is_retracted"] = True
    entries[2]["document"]["publication_year"] = None
    corpus["entries"] = entries

    def forbidden(*args, **kwargs):
        raise AssertionError("No usable dated documents: no inference should be performed")

    monkeypatch.setattr(Encoder, "encode", forbidden)
    result = analyze(corpus, options(corpus))
    assert result["preparation"]["retained_studies"] == 0
    assert result["temporal_selection"]["excluded_after_end_year"] == 1
    assert result["temporal_selection"]["excluded_retracted_occurrences"] == 1
    assert result["semantic_relevance"]["scored_texts"] == 0


def test_default_never_constructs_encoder_and_invalid_query_fails_before_model_load(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected model construction")

    monkeypatch.setattr("app.ml.local_encoder.LocalEncoder", forbidden)
    corpus = unpack_snapshot(snapshot())
    analyze(corpus, options(corpus, "lexical"))
    with pytest.raises(AnalysisInputError, match="английском"):
        prepare_analysis([], AnalysisOptions(topic="хранение тепла", relevance_mode="semantic"))


def test_progress_is_monotonic_and_fingerprint_reproducible():
    corpus = unpack_snapshot(generic_snapshot())
    progress = []
    result = analyze(corpus, options(corpus), progress=lambda n, message: progress.append(n))
    assert progress == sorted(progress)
    assert progress[-1] == 100
    assert analyze(corpus, options(corpus)) == result


def test_service_and_atomic_export_roundtrip(tmp_path):
    source, target = tmp_path / "corpus.json", tmp_path / "result.json"
    source.write_text(json.dumps(generic_snapshot()), encoding="utf-8")
    result = run_analysis(None, {"topic": "thermal energy storage", "relevance_mode": "semantic"},
                          snapshot_path=source, model_dir=tmp_path / "explicit-model")
    export_result(result, target, protected_paths=[source])
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert ResultContract.model_validate(saved).model_dump() == saved == result
    with pytest.raises(AnalysisInputError, match="только"):
        run_analysis(None, {"topic": "thermal energy storage"}, snapshot_path=source, model_dir=tmp_path)


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("semantic_relevance"),
    lambda r: r["semantic_relevance"].update(retained_studies=0),
    lambda r: r["semantic_relevance"]["study_decisions"][0].update(similarity=float("nan")),
    lambda r: r["semantic_relevance"]["study_decisions"][0].update(semantic_only=False),
    lambda r: r["preliminary_signals"][0]["semantic_relevance"].update(requires_review=False),
    lambda r: r["preliminary_signals"][0]["semantic_relevance"].update(corpus_requires_review=False),
    lambda r: r["candidates"].append(r["preliminary_signals"].pop()),
    lambda r: r["options"].update(relevance_mode="lexical"),
    lambda r: (r.pop("semantic_relevance"), r["options"].update(relevance_mode="lexical")),
    lambda r: r["semantic_relevance"]["model"].update(model_id=7),
    lambda r: r["semantic_relevance"]["model"].update(revision=""),
])
def test_semantic_contract_rejects_inconsistent_admission_and_buckets(generic_result, mutation, tmp_path):
    _, original = generic_result
    result = deepcopy(original)
    mutation(result)
    with pytest.raises((ValidationError, ValueError)):
        ResultContract.model_validate(result)
    with pytest.raises(ValueError):
        validate_semantic_result(result)
    with pytest.raises(ValueError):
        export_result(result, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("mutation,code", [
    (lambda r: r["semantic_relevance"]["model"].update(revision="tampered"), "semantic_scope_replay"),
    (lambda r: r["semantic_relevance"].update(threshold=.5), "semantic_scope_replay"),
    (lambda r: r["semantic_relevance"]["study_decisions"][0].update(similarity=.96), "semantic_scope_replay"),
    (lambda r: r["preliminary_signals"][0]["semantic_relevance"].update(requires_review=False),
     "semantic_group_replay"),
    (lambda r: r["preliminary_signals"][0].update(semantic_relevance=[]), "semantic_group_replay"),
    (lambda r: r["preliminary_signals"][0]["semantic_relevance"].update(study_decisions=None),
     "semantic_group_replay"),
])
def test_audit_recomputes_admission_and_detects_tampering_without_refitting(generic_result, monkeypatch,
                                                                         mutation, code):
    corpus, original = generic_result
    result = deepcopy(original)
    mutation(result)

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit must not refit NMF")

    monkeypatch.setattr("app.ml.engine.fit_topics", forbidden)
    report = audit_result(corpus, result)
    assert not report["ok"]
    assert code in {error["code"] for error in report["errors"]}


def test_audit_distinguishes_missing_weights_from_invalid_corpus(generic_result, monkeypatch):
    def missing(*args, **kwargs):
        raise AnalysisInputError("Установите модель: python -m scripts.install_ml_model")

    monkeypatch.setattr("app.ml.local_encoder.LocalEncoder", missing)
    report = audit_result(*generic_result)
    assert not report["ok"]
    assert report["errors"][0]["code"] == "semantic_replay_unavailable"
    assert "scripts.install_ml_model" in report["errors"][0]["message"]


def test_semantic_corpus_changes_keep_even_lexical_groups_under_review():
    data = generic_snapshot()
    for batch in data["batches"]:
        for entry in batch["documents"]:
            if "-0-" not in entry["document"]["source_id"]:
                entry["document"]["abstract"] += " Thermal storage systems retain heat for later use."
    corpus = unpack_snapshot(data)
    result = analyze(corpus, options(corpus))
    assert result["semantic_relevance"]["semantic_only_studies"] > 0
    assert result["semantic_relevance"]["lexical_studies"] > 0
    assert any(group["semantic_relevance"]["semantic_only_documents"] == 0
               for bucket in BUCKETS for group in result[bucket])
    assert result["candidates"] == result["established"] == []
    for group in result["preliminary_signals"]:
        assert group["semantic_relevance"]["corpus_requires_review"]
        assert group["semantic_relevance"]["requires_review"]
    assert ResultContract.model_validate(result).model_dump() == result
    report = audit_result(corpus, result)
    assert report["ok"], report["errors"]
