"""The optional local relevance adapter preserves preparation and lexical defaults."""

import json
from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event

import numpy as np
import pytest

from app.ml.contracts import AnalysisInputError, AnalysisOptions
from app.ml.engine import prepare
from app.ml.scope_context import current_semantic_policy
from app.ml.semantic import (
    build_semantic_policy,
    build_semantic_policy_from_studies,
    semantic_key,
    validate_query,
)
from app.ml.text import _lexical_scope_check, scope_check

TOPIC = "thermal energy storage"
TITLE = "Phase change reservoirs for reversible heat buffering"
ABSTRACT = ("We measure reversible melting in paraffin reservoirs across repeated cycles. "
            "Our apparatus retains heat reliably and releases it during cooling with stable performance.")
OPTIONS = AnalysisOptions(topic=TOPIC)


def entry(identifier="one", title=TITLE, abstract=ABSTRACT):
    return {"document_key": "doi:10.9999/" + identifier, "revision_id": identifier,
            "document": {"title": title, "abstract": abstract, "document_type": "article",
                         "publication_year": 2022, "url": "https://example.org/" + identifier,
                         "doi": "10.9999/" + identifier, "authors": ["Researcher " + identifier]}}


class Encoder:
    def __init__(self, score=0.9, revision="fixture-v1"):
        self.score, self.revision, self.calls = score, revision, []

    def encode(self, texts, kind="passage", cancel=None, progress=None):
        self.calls.append((kind, tuple(texts)))
        matrix = np.zeros((len(texts), 384))
        if kind == "query":
            matrix[:, 0] = 1
        else:
            matrix[:, 0] = self.score
            matrix[:, 1] = np.sqrt(1 - self.score ** 2)
        return matrix

    def manifest(self):
        return {"model": "test-local-encoder", "revision": self.revision}


def test_semantic_synonym_is_opt_in_and_never_changes_the_source():
    document = entry()
    before = deepcopy(document)
    assert scope_check(TOPIC, TITLE, ABSTRACT) == "scope_not_established"
    assert prepare([document], OPTIONS)[0] == []
    policy = build_semantic_policy([document], TOPIC, Encoder())
    with policy.context():
        studies, _ = prepare([document], OPTIONS)
    assert len(studies) == 1
    assert document == before
    assert scope_check(TOPIC, TITLE, ABSTRACT) == "scope_not_established"
    diagnostic = policy.summary(studies)
    assert diagnostic["semantic_only_studies"] == 1
    assert diagnostic["study_decisions"][0]["requires_review"]
    assert not diagnostic["calibrated"]
    assert diagnostic["similarity_kind"] == "cosine"
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("score,admitted", [(0.1, False), (0.8199, False), (0.82, True), (0.99, True)])
def test_semantic_threshold_is_an_explicit_inclusive_boundary(score, admitted):
    policy = build_semantic_policy([entry()], TOPIC, Encoder(score))
    with policy.context():
        assert (scope_check(TOPIC, TITLE, ABSTRACT) == "direct_lexical_signal") is admitted


def test_low_similarity_never_overrides_a_lexical_acceptance():
    document = entry(title="Thermal energy storage in paraffin reservoirs")
    policy = build_semantic_policy([document], TOPIC, Encoder(0.1))
    decision = policy.decision(TOPIC, document["document"]["title"], ABSTRACT)
    assert decision["admitted"] and decision["lexical_admitted"]
    assert decision["route"] == "lexical" and not decision["semantic_only"]


@pytest.mark.parametrize("topic,title,abstract", [
    ("photonic neuromorphic computing", "Optical device design with neural methods",
     "We use a neural network on a GPU to optimize an optical device. The optical chip does not perform neural inference."),
    ("quantum reservoir computing", "Classical reservoirs built from quantum dots",
     "We implement a classical reservoir network using quantum dots as materials. The processor uses classical electronic computation for the task."),
    ("dna data storage", "Symbolic DNA encryption for cloud data",
     "We implement cloud encryption using a symbolic DNA alphabet for data storage. The algorithm operates on ordinary digital computers using XOR operations."),
    ("photonic neural signal processing", "Optical device design with neural methods",
     "We use a neural network on a GPU to optimize an optical device. The optical chip does not perform neural inference."),
])
def test_high_similarity_cannot_bypass_profile_or_legacy_executor_guards(topic, title, abstract):
    assert _lexical_scope_check(topic, title, abstract) == "scope_not_established"
    policy = build_semantic_policy([entry(title=title, abstract=abstract)], topic, Encoder(0.99))
    assert policy.guarded
    with policy.context():
        assert scope_check(topic, title, abstract) == "scope_not_established"
    decision = policy.decision(topic, title, abstract)
    assert decision["scored"] and not decision["semantic_only"]


@pytest.mark.parametrize("change", ["url", "short", "oversized", "type", "year_conflict", "retracted"])
def test_semantic_scope_cannot_bypass_existing_preparation_filters(change):
    documents = [entry()]
    if change == "url":
        documents[0]["document"]["url"] = "javascript:alert(1)"
    elif change == "short":
        documents[0]["document"]["abstract"] = "Very short abstract."
    elif change == "oversized":
        documents[0]["document"]["abstract"] = "words " * 3001
    elif change == "type":
        documents[0]["document"]["document_type"] = "dataset"
    elif change == "year_conflict":
        other = deepcopy(documents[0])
        other["document"]["publication_year"] = 2021
        documents.append(other)
    else:
        documents[0]["document"]["raw_metadata"] = {"is_retracted": True}
    original = deepcopy(documents)
    policy = build_semantic_policy(documents, TOPIC, Encoder())
    with policy.context():
        assert prepare(documents, OPTIONS)[0] == []
    assert documents == original


def test_selected_usable_version_has_semantic_provenance_and_source_versions():
    usable, oversized = entry(), entry()
    oversized["revision_id"] = "oversized"
    oversized["document"]["abstract"] = "words " * 3001
    policy = build_semantic_policy([oversized, usable], TOPIC, Encoder())
    with policy.context():
        studies, _ = prepare([oversized, usable], OPTIONS)
    assert len(studies) == 1 and studies[0]["abstract"] == ABSTRACT
    assert {v["revision_id"] for v in studies[0]["versions"]} == {"one", "oversized"}
    summary = policy.summary(studies)
    assert summary["eligible_occurrences"] == 1 and summary["semantic_only_studies"] == 1


def test_batch_order_deduplication_and_cleaned_keys_are_reproducible():
    first, second = entry(), entry("two", title="Latent heat buffers using phase transition materials")
    duplicate = deepcopy(first)
    duplicate["document"]["title"] = "  " + TITLE + "  "
    left, right = Encoder(), Encoder()
    a = build_semantic_policy([first, second, duplicate], TOPIC, left)
    b = build_semantic_policy([duplicate, second, first], TOPIC, right)
    assert left.calls == right.calls
    assert len(left.calls) == 2 and len(left.calls[1][1]) == 2
    assert a.scores == b.scores
    assert a.eligible_occurrences == 3
    assert semantic_key(TOPIC, TITLE, ABSTRACT) == semantic_key(TOPIC, "  " + TITLE, ABSTRACT)


@pytest.mark.parametrize("failure", [RuntimeError("failed"), CancelledError("cancelled")])
def test_context_restores_previous_policy_even_on_failure(failure):
    outer = build_semantic_policy([entry()], TOPIC, Encoder(0.1))
    inner = build_semantic_policy([entry()], TOPIC, Encoder(0.9))
    with outer.context():
        with pytest.raises(type(failure)), inner.context():
            raise failure
        assert current_semantic_policy() is outer
    assert current_semantic_policy() is None


def test_parallel_analyses_do_not_share_a_semantic_policy():
    policy = build_semantic_policy([entry()], TOPIC, Encoder())
    barrier = Barrier(2)

    def semantic_worker():
        with policy.context():
            barrier.wait(timeout=5)
            return scope_check(TOPIC, TITLE, ABSTRACT)

    def lexical_worker():
        barrier.wait(timeout=5)
        return scope_check(TOPIC, TITLE, ABSTRACT)

    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic = executor.submit(semantic_worker)
        lexical = executor.submit(lexical_worker)
        assert semantic.result() == "direct_lexical_signal"
        assert lexical.result() == "scope_not_established"


@pytest.mark.parametrize("topic", ["хранение тепла", "quantum вычисления", "数字存储", "1234 !!!"])
def test_unknown_nonenglish_query_fails_before_loading_encoder(topic):
    encoder = Encoder()
    with pytest.raises(AnalysisInputError, match="английском"):
        build_semantic_policy([entry()], topic, encoder)
    assert encoder.calls == []


def test_known_russian_alias_resolves_to_its_english_profile():
    policy = build_semantic_policy([entry()], "фотонные нейросети", Encoder())
    assert policy.guarded and policy.topic == "photonic neuromorphic computing"


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), -0.1, 1.1, "0.82"])
def test_invalid_threshold_does_not_load_encoder(threshold):
    encoder = Encoder()
    with pytest.raises(AnalysisInputError, match="Порог"):
        build_semantic_policy([entry()], TOPIC, encoder, threshold)
    assert encoder.calls == []


@pytest.mark.parametrize("vectors", [np.zeros((1, 384)), np.ones((1, 3)), np.full((1, 384), np.nan)])
def test_invalid_encoder_output_fails_explicitly(vectors):
    encoder = Encoder()
    encoder.encode = lambda *args, **kwargs: vectors
    with pytest.raises(AnalysisInputError, match="encoder"):
        build_semantic_policy([entry()], TOPIC, encoder)


def test_encoder_failure_propagates_without_a_lexical_fallback():
    encoder = Encoder()

    def unavailable(*args, **kwargs):
        raise AnalysisInputError("Локальная модель отсутствует")

    encoder.encode = unavailable
    with pytest.raises(AnalysisInputError, match="отсутствует"):
        build_semantic_policy([entry()], TOPIC, encoder)
    assert current_semantic_policy() is None


def test_cancellation_stops_before_loading_encoder():
    cancel, encoder = Event(), Encoder()
    cancel.set()
    with pytest.raises(CancelledError):
        build_semantic_policy([entry()], TOPIC, encoder, cancel=cancel)
    assert encoder.calls == []


def test_prepared_study_scoring_preserves_baseline_and_manifest_is_not_shared():
    document = entry(title="Thermal energy storage in paraffin reservoirs")
    studies, _ = prepare([document], OPTIONS)
    original = deepcopy(studies)
    policy = build_semantic_policy_from_studies(studies, TOPIC, Encoder(revision="fixture-v2"))
    summary = policy.summary(studies)
    assert summary["model"]["revision"] == "fixture-v2"
    assert summary["lexical_studies"] == 1 and not summary["semantic_only_studies"]
    summary["model"]["revision"] = "changed"
    assert policy.summary(studies)["model"]["revision"] == "fixture-v2"
    assert studies == original


def test_unscored_text_and_another_topic_do_not_receive_a_semantic_acceptance():
    policy = build_semantic_policy([entry()], TOPIC, Encoder())
    assert not policy.decision(TOPIC, "Unseen document title", ABSTRACT)["scored"]
    with policy.context():
        assert scope_check("marine biology", TITLE, ABSTRACT) == "scope_not_established"


@pytest.mark.parametrize("language", ["ru", "fr", "de", "zh", "rus"])
def test_declared_nonenglish_text_cannot_gain_semantic_only_admission(language):
    document, encoder = entry(), Encoder(0.99)
    document["document"]["language"] = language
    policy = build_semantic_policy([document], TOPIC, encoder)
    with policy.context():
        assert prepare([document], OPTIONS)[0] == []
    assert encoder.calls == []
    assert policy.decision(TOPIC, TITLE, ABSTRACT)["unscored_reason"] == "declared_non_english_language"
    assert policy.summary([])["skipped_occurrences_by_reason"] == {"declared_non_english_language": 1}


def test_english_title_and_wrong_english_metadata_do_not_mask_a_nonlatin_abstract():
    abstract = ("Мы исследуем сохранение тепла при обратимом плавлении материалов и измеряем "
                "устойчивость устройства при повторных циклах нагрева и охлаждения образца.")
    document, encoder = entry(abstract=abstract), Encoder(0.99)
    document["document"]["language"] = "en"
    policy = build_semantic_policy([document], TOPIC, encoder)
    with policy.context():
        assert prepare([document], OPTIONS)[0] == []
    assert encoder.calls == []
    assert policy.decision(TOPIC, TITLE, abstract)["unscored_reason"] == "predominantly_non_latin_abstract"


@pytest.mark.parametrize("year", [None, "2022", True, 1899, 2026])
def test_ineligible_publication_year_is_filtered_before_embedding(year):
    document, encoder = entry(), Encoder()
    document["document"]["publication_year"] = year
    policy = build_semantic_policy([document], TOPIC, encoder, end_year=2025)
    assert encoder.calls == []
    assert policy.decision(TOPIC, TITLE, ABSTRACT)["unscored_reason"] == "unknown_or_future_year"


def test_query_embedding_completion_cannot_move_progress_backwards():
    class ProgressEncoder(Encoder):
        def encode(self, texts, kind="passage", cancel=None, progress=None):
            result = super().encode(texts, kind, cancel, progress)
            if progress is not None:
                for completed in range(1, len(texts) + 1):
                    progress(completed, len(texts))
            return result

    reports = []
    build_semantic_policy([entry(), entry("two", title="Different phase transition heat buffers")],
                          TOPIC, ProgressEncoder(), progress=lambda n, total: reports.append((n, total)))
    ratios = [completed / total for completed, total in reports]
    assert ratios == sorted(ratios)
    assert reports[-1] == (2, 2)


@pytest.mark.parametrize("language", [None, "", "und", "unknown", "en", "eng", "English", "en-US"])
def test_supported_or_unknown_language_does_not_claim_english_detection(language):
    document = entry()
    document["document"]["language"] = language
    policy = build_semantic_policy([document], TOPIC, Encoder())
    assert policy.decision(TOPIC, TITLE, ABSTRACT)["semantic_only"]
    assert policy.decision(TOPIC, TITLE, ABSTRACT)["unscored_reason"] is None
    language_policy = policy.summary([])["language_policy"]
    assert not language_policy["language_autodetection"]
    assert language_policy["unknown_latin_language"] == "not_verified"


def test_language_skip_keeps_legacy_lexical_admission_and_explains_unscored_study():
    document = entry(title="Thermal energy storage in paraffin reservoirs")
    document["document"]["language"] = "fr"
    baseline, _ = prepare([document], OPTIONS)
    encoder = Encoder()
    policy = build_semantic_policy([document], TOPIC, encoder)
    with policy.context():
        studies, _ = prepare([document], OPTIONS)
    assert studies == baseline and len(studies) == 1
    assert encoder.calls == []
    summary = policy.summary(studies)
    assert summary["unscored_studies"] == summary["lexical_studies"] == 1
    assert summary["study_decisions"][0]["unscored_reason"] == "declared_non_english_language"


def test_conflicting_language_metadata_for_identical_text_is_order_independent():
    english, french = entry(), entry("two")
    english["document"]["language"] = "en"
    french["document"]["language"] = "fr"
    left, right = Encoder(), Encoder()
    first = build_semantic_policy([english, french], TOPIC, left)
    second = build_semantic_policy([french, english], TOPIC, right)
    assert first.summary([]) == second.summary([])
    assert left.calls == right.calls == []
    assert first.decision(TOPIC, TITLE, ABSTRACT)["unscored_reason"] == "conflicting_language_metadata"
    summary = first.summary([])
    assert summary["eligible_occurrences"] == 0
    assert sum(summary["skipped_occurrences_by_reason"].values()) == 2
    assert summary["blocked_texts_by_reason"] == {"conflicting_language_metadata": 1}


def test_guarded_study_scoring_recovers_language_only_for_selected_source_text():
    topic = "photonic neuromorphic computing"
    document = entry(title="Optical neural processors for signal inference", abstract=(
        "We demonstrate an optical neural processor that performs inference. "
        "Our device achieves stable computations across repeated experiments with accurately measured responses."))
    document["document"]["language"] = "fr"
    studies, _ = prepare([document], AnalysisOptions(topic=topic))
    assert len(studies) == 1 and "language" not in studies[0]
    original = deepcopy(studies)
    encoder = Encoder()
    policy = build_semantic_policy_from_studies(studies, topic, encoder, source_entries=[document], end_year=2025)
    assert encoder.calls == []
    assert policy.summary(studies)["unscored_studies"] == 1
    assert policy.summary(studies)["study_decisions"][0]["unscored_reason"] == "declared_non_english_language"
    assert studies == original
    changed_version = deepcopy(document)
    changed_version["document"]["abstract"] += " Additional unselected text."
    accepted = build_semantic_policy_from_studies(studies, topic, Encoder(), source_entries=[changed_version])
    assert accepted.summary(studies)["unscored_studies"] == 0


def test_source_year_bound_is_preserved_when_scoring_prepared_studies():
    document = entry(title="Thermal energy storage in paraffin reservoirs")
    studies, _ = prepare([document], OPTIONS)
    encoder = Encoder()
    policy = build_semantic_policy_from_studies(studies, TOPIC, encoder, end_year=2021)
    assert encoder.calls == []
    assert policy.summary(studies)["study_decisions"][0]["unscored_reason"] == "unknown_or_future_year"
    assert policy.summary(studies)["publication_year_bounds"] == {"minimum": 1900, "maximum": 2021}


def test_invalid_version_year_does_not_hide_an_eligible_identical_text():
    valid, invalid = entry(), entry("invalid")
    invalid["document"]["publication_year"] = None
    policy = build_semantic_policy([invalid, valid], TOPIC, Encoder(), end_year=2025)
    assert policy.decision(TOPIC, TITLE, ABSTRACT)["semantic_only"]
    assert policy.summary([])["skipped_occurrences_by_reason"] == {"unknown_or_future_year": 1}
    assert policy.summary([])["blocked_texts_by_reason"] == {}


def test_query_validation_is_public_lightweight_and_has_no_photonic_substring_exception():
    assert validate_query("фотонные нейросети") == "photonic neuromorphic computing"
    assert validate_query(TOPIC) == TOPIC
    with pytest.raises(AnalysisInputError, match="английском"):
        validate_query("photonic neural вычисления")
