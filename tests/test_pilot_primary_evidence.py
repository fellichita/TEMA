"""Primary units and contextual qualifiers: fewer false exclusions without accepting invented results."""

from datetime import date

import pytest

from app.pilot.completion import next_candidate_batch
from app.pilot.evidence import PRIMARY_GROUNDING_METHOD, QuoteSelection, _selection_supported, build_passport
from app.pilot.signal_evidence import (CONTEXTUAL_SIGNAL_EVIDENCE_VERSION, extract_signal_evidence,
                                      verify_signal_evidence)
from app.pilot.sources import exclusion_reason, primary_research_exclusion
from tests.test_pilot_evidence import candidate, document
from tests.test_pilot_signal_evidence import EXPERIMENT, NOVELTY, prepared


@pytest.mark.parametrize("qualification", [
    "This approach may need further optimisation for commercial deployment.",
    "Future work could improve device lifetime.",
    "We did not use an expensive purification step.",
    "An earlier review describes conventional membrane technology.",
])
def test_ordinary_contextual_caution_does_not_erase_positive_primary_novelty(tmp_path, qualification):
    archive, _, data, item, context = prepared(tmp_path, abstract=NOVELTY + " " + EXPERIMENT + " " + qualification)
    # Exact old extraction is intentionally unchanged for portable legacy packages.
    assert extract_signal_evidence(item, data, archive, context, query_plan=_plan()) == ()
    sources = extract_signal_evidence(item, data, archive, context, query_plan=_plan(),
                                     method_version=CONTEXTUAL_SIGNAL_EVIDENCE_VERSION)
    assert len(sources) == 1
    verify_signal_evidence(sources, item, data, archive, context, query_plan=_plan())
    card = build_passport(item, data, archive, context, methodology_version="3.3.0")
    assert any(claim.role == "case" and claim.support == "supported" for claim in card.claims)


def _plan():
    from tests.test_pilot_evidence import query_plan
    return query_plan()


@pytest.mark.parametrize("abstract", [
    NOVELTY + " Our laboratory experiments might validate the transport mechanism.",
    NOVELTY + " " + EXPERIMENT + " However, our results do not support the proposed mechanism.",
    "This paper surveys lithium selective membranes. " + NOVELTY + " " + EXPERIMENT,
    "We could introduce novel lithium selective membranes. " + EXPERIMENT,
    "We introduce a new dataset for lithium selective membranes. " + EXPERIMENT,
])
def test_contextual_novelty_does_not_admit_uncertain_results_reviews_or_new_datasets(tmp_path, abstract):
    archive, _, data, item, context = prepared(tmp_path, abstract=abstract)
    assert extract_signal_evidence(item, data, archive, context, query_plan=_plan(),
                                   method_version=CONTEXTUAL_SIGNAL_EVIDENCE_VERSION) == ()


@pytest.mark.parametrize("changes", [
    {"document_type": "dataset"}, {"raw_metadata": {"type": "dataset"}},
    {"document_type": "software"}, {"document_type": "component"},
    {"document_type": "review"},
    {"abstract": "This paper provides an overview of lithium selective membranes."},
    {"abstract": "This work reviews recent results for lithium selective membranes."},
])
def test_source_assets_and_literature_syntheses_are_not_primary_studies(changes):
    record = document(1, **changes)
    assert primary_research_exclusion(record) is not None
    assert exclusion_reason(record, date(2020, 1, 1), date(2025, 12, 31), primary_only=True)
    # Existing archives still use their original count admission by default.
    assert exclusion_reason(record, date(2020, 1, 1), date(2025, 12, 31)) is None
    selection = QuoteSelection(role="case", revision_id="test", field="title", quote=record.title)
    assert not _selection_supported(selection, record, method_version=PRIMARY_GROUNDING_METHOD)


def test_review_only_in_abstract_cannot_be_a_concrete_case_but_old_screen_replays():
    record = document(1, abstract="This paper reviews recent technological platforms.")
    selection = QuoteSelection(role="case", revision_id="test", field="title", quote=record.title)
    assert _selection_supported(selection, record)
    assert not _selection_supported(selection, record, method_version=PRIMARY_GROUNDING_METHOD)


def test_discovery_preserves_linked_dataset_as_support_without_counting_a_standalone_dataset():
    from app.pilot.discovery import _eligible_studies, deduplicate

    paper = document(10)
    dataset = document(11, document_type="dataset")
    supplement = document(12, document_type="dataset", raw_metadata={"relation": {
        "is-supplement-to": [{"id-type": "doi", "id": paper.doi}]}})
    docs = [paper, dataset, supplement]
    studies, excluded = _eligible_studies(deduplicate(docs), docs, _plan(), None)
    assert len(studies) == 1 and studies[0]["study_id"] == paper.document_key
    assert studies[0]["supplementary_study_ids"] == [supplement.document_key]
    assert any(item["study_id"] == dataset.document_key and item["reason"] == "supporting_asset_not_research"
               for item in excluded)


def test_refill_reaches_candidate_31_with_stable_balanced_order_without_duplicate_work(tmp_path):
    _, _, data, _, _ = prepared(tmp_path)
    group = candidate(data)
    groups = [group.model_copy(update={"candidate_id": f"group-{index}"}).model_dump(mode="json") for index in range(40)]
    rare = [group.model_copy(update={"candidate_id": f"single-{index}"}).model_dump(mode="json") for index in range(40)]
    # Distinct origin lane: groups can be large even though the test archive has one real record.
    for item in groups:
        item["discovery_study_ids"] = ("first", "second")
    discovered = {"candidates": groups, "review_queue": rare + [groups[0]]}
    first = next_candidate_batch(discovered, set())
    assert len(first) == 30 and first[0].candidate_id == "group-0" and first[1].candidate_id == "single-0"
    processed = {item.candidate_id for item in first}
    second = next_candidate_batch(discovered, processed)
    assert len(second) == 30 and second[0].candidate_id == "group-15" and second[1].candidate_id == "single-15"
    assert not processed.intersection(item.candidate_id for item in second)
    assert next_candidate_batch(discovered, processed) == second
    processed.update(item.candidate_id for item in second)
    third = next_candidate_batch(discovered, processed)
    assert len(third) == 20
    processed.update(item.candidate_id for item in third)
    assert next_candidate_batch(discovered, processed) == ()
