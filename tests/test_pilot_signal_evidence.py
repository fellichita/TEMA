"""Archived author novelty is a reproducible hypothesis, never independent review."""

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.evidence import EvidenceError, TITLE_ADMISSION_VERSION, admission_hash, archived_field
from app.pilot.contracts import verify_evidence_text
from app.pilot.signal_evidence import extract_signal_evidence, verify_signal_evidence
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot

NOVELTY = "Here we introduce novel lithium selective membranes with a tethered crown-ether transport mechanism."
EXPERIMENT = "We measured lithium selectivity in laboratory experiments using three brine compositions."


def prepared(tmp_path, *, abstract=None, count=1, **changes):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(number, abstract=abstract if abstract is not None else NOVELTY + " " + EXPERIMENT,
                          **changes) for number in range(1, count + 1))
    data = snapshot(docs, archive)
    item = candidate(data, specificity="specific_technology", synonyms=("lithium selective membranes",),
                     admission_rule_version=TITLE_ADMISSION_VERSION)
    item = item.model_copy(update={"admission_rule_hash": admission_hash(item)})
    return archive, docs, data, item, Context()


def test_single_concrete_author_assertion_is_archived_hypothesis_only(tmp_path):
    archive, docs, data, item, context = prepared(tmp_path)
    evidence = extract_signal_evidence(item, data, archive, context, query_plan=query_plan())
    assert len(evidence) == 1
    entry = evidence[0]
    assert entry.novelty.quote == NOVELTY and entry.experiment.quote == EXPERIMENT
    assert entry.novelty.study_id == item.discovery_study_ids[0]
    assert "не независимое подтверждение" in entry.limitation
    assert not hasattr(entry, "novelty_score") and not hasattr(entry, "verified_novelty")
    for quotation in (entry.novelty, entry.experiment):
        verify_evidence_text(quotation, archived_field(docs[0], quotation.text_field))
    assert verify_signal_evidence(evidence, item, data, archive, context, query_plan=query_plan()) is None


@pytest.mark.parametrize("abstract", [
    NOVELTY,
    "We measured lithium selective membranes in laboratory experiments with excellent efficiency.",
    "Previous researchers introduced novel lithium selective membranes. " + EXPERIMENT,
    "We review novel lithium selective membranes with experimental findings. " + EXPERIMENT,
    "We propose novel lithium selective membranes for future experiments.",
    NOVELTY + " Our simulations predict selectivity in laboratory experiments.",
    NOVELTY + " " + EXPERIMENT + " However, our results do not support the predicted mechanism.",
    "We introduce a new dataset for lithium selective membranes. " + EXPERIMENT,
    "We introduce a novel application of lithium selective membranes. " + EXPERIMENT,
    "We introduce novel sodium selective membranes. " + EXPERIMENT,
    NOVELTY + " Earlier researchers measured selectivity in laboratory experiments.",
    "Novel lithium selective membranes were developed by our colleagues. " + EXPERIMENT,
])
def test_review_hypothesis_context_negation_other_mechanism_or_benchmark_is_not_novelty(tmp_path, abstract):
    archive, _, data, item, context = prepared(tmp_path, abstract=abstract)
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan()) == ()


@pytest.mark.parametrize("changes", [
    {"publication_year": None, "date_precision": "unknown"}, {"publication_year": 2027}, {"document_type": "review"},
    {"raw_metadata": {"is_retracted": True}},
    {"title": "Lithium selective membranes: a comprehensive review"},
])
def test_ineligible_dated_record_is_not_an_experimental_signal(tmp_path, changes):
    archive, _, data, item, context = prepared(tmp_path, **changes)
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan()) == ()


def test_full_frozen_name_and_scope_and_specificity_are_required(tmp_path):
    archive, _, data, item, context = prepared(tmp_path)
    for changed in (
        item.model_copy(update={"specificity": "uncertain"}),
        item.model_copy(update={"admission_rule_hash": "a" * 64}),
        item.model_copy(update={"admission_rule_version": "title-phrase-admission/1.0.0"}),
    ):
        assert extract_signal_evidence(changed, data, archive, context, query_plan=query_plan()) == ()
    wrong_scope = query_plan().model_copy(update={"english_query": "quantum biological sensing", "subdirections": ()})
    with pytest.raises(EvidenceError, match="Область"):
        extract_signal_evidence(item, data, archive, context, query_plan=wrong_scope)
    wrong_data = data.model_copy(update={"plan_hash": wrong_scope.plan_hash})
    wrong_item = item.model_copy(update={"plan_hash": wrong_scope.plan_hash})
    assert extract_signal_evidence(wrong_item, wrong_data, archive, context, query_plan=wrong_scope) == ()


def test_tampering_and_silent_omission_are_rejected_by_exact_replay(tmp_path):
    archive, _, data, item, context = prepared(tmp_path)
    evidence = extract_signal_evidence(item, data, archive, context, query_plan=query_plan())
    corrupted = (evidence[0].model_copy(update={"publication_year": 2024}),)
    for changed in (corrupted, ()):
        with pytest.raises(EvidenceError, match="не воспроизводится"):
            verify_signal_evidence(changed, item, data, archive, context, query_plan=query_plan())
    with pytest.raises(ValueError, match="Duplicate"):
        verify_signal_evidence(evidence * 2, item, data, archive, context, query_plan=query_plan())


def test_archived_evidence_limit_is_deterministic_and_does_not_make_replicas_independent(tmp_path):
    archive, _, data, item, context = prepared(tmp_path, count=12)
    evidence = extract_signal_evidence(item, data, archive, context, query_plan=query_plan())
    assert len(evidence) == 10
    assert len({entry.novelty.study_id for entry in evidence}) == 10
    assert all("репликации" in entry.limitation for entry in evidence)
    assert verify_signal_evidence(evidence, item, data, archive, context, query_plan=query_plan()) is None
