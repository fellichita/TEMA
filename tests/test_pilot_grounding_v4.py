"""New raw quotation rules, with immutable regression examples from the audit.

The three DOI-tagged negative quotations are verbatim passages in the 1,510
record corpus inspected on 2026-09-15. They are not generated paper summaries.
Other controls are deliberately synthetic, so polarity changes are isolated.
"""

from types import SimpleNamespace

import pytest

from app.pilot.contracts import verify_evidence_text
from app.pilot.evidence import (
    PRIMARY_GROUNDING_METHOD, RAW_GROUNDING_METHOD, RESEARCH_APPLICATION_METHOD,
    EvidenceError, LabelBatch, QuoteSelection, _role_supported, _selection_supported,
    automatic_research_application, build_passport, grounding_sentences, label_candidates,
    primary_result_sentence,
    raw_quote_evidence,
)
from app.pilot.signal_evidence import (
    CONTEXTUAL_SIGNAL_EVIDENCE_VERSION, PRIMARY_EVIDENCE_VERSION,
    RAW_PRIMARY_EVIDENCE_VERSION, RAW_SIGNAL_EVIDENCE_VERSION,
    extract_primary_observations, extract_signal_evidence,
    verify_primary_observations, verify_primary_sources, verify_signal_evidence, verify_signal_sources,
)
from tests.test_pilot_evidence import document, label_payload, query_plan, snapshot
from tests.test_pilot_signal_evidence import NOVELTY, prepared


AUDITED_NEGATIVES = (
    ("10.36227/techrxiv.23859162",
     "In many applications, such as aeromagnetic measurements, we need to increase the sampling rate, "
     "but at the same time we reduce the accuracy of the measurements."),
    ("10.1049/icp.2025.4202",
     "However, further improvements in bandwidth and dynamic range are required to detect high-frequency "
     "biomagnetic signals and overcome uncertain environmental magnetic field fluctuation."),
    ("10.1039/d3ay00676j",
     "biosensors is vital in clinical diagnostics, owing to their simple equipment, facile operation, high "
     "selectivity, economical, short diagnostic time, fast response, and easy miniaturization, but the need "
     "to improve sensitivity for protein detection is still a barrier limiting its wider practical applications."),
)
BENEFIT = "We measured a 43.94% reduction in energy consumption using lithium selective membranes in laboratory experiments."
RAW_BENEFIT = '<jats:p>We measured a 43.94% reduction in energy consumption, i.e. 2.75 kW,\r\nusing lithium selective membranes.</jats:p>'


@pytest.mark.parametrize("doi,quote", AUDITED_NEGATIVES, ids=[item[0] for item in AUDITED_NEGATIVES])
def test_actual_paper_aspirations_and_reduced_accuracy_are_not_advantages(tmp_path, doi, quote):
    archive, docs, data, item, context = prepared(tmp_path, abstract=quote, doi=doi)
    assert not _role_supported("advantage", quote, context=quote, method_version=RAW_GROUNDING_METHOD)
    assert not automatic_research_application(quote, docs[0])
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    assert not any(claim.role in {"advantage", "application"} and claim.support == "supported" for claim in card.claims)


def test_real_serf_paper_preserves_actual_experimental_result_after_motivation():
    # DOI 10.1049/icp.2025.4202: a limitation and a reported result have different roles.
    quote = "Experimental results show that the dynamic range of the closed-loop magnetometer is enhanced from 5 to 50 nT, limited by coil dimensions."
    abstract = AUDITED_NEGATIVES[1][1] + " " + (
        "Firstly, we theoretically analysis the signal transfer model in the single-beam SERF magnetometers, "
        "and implemented the core lock in amplifier (LIA) and PID module on SigmaDSP chip. "
        "Then, we experimentally investigated the dependence of the amplitude–frequency response of the "
        "lock-in amplifier on the filter order and cut-off frequency. ") + quote
    record = document(1, doi=AUDITED_NEGATIVES[1][0], title="A DSP-based closed loop serf atomic magnetometer", abstract=abstract)
    assert _role_supported("advantage", quote, context=abstract, method_version=RAW_GROUNDING_METHOD)
    assert automatic_research_application(quote, record)


@pytest.mark.parametrize("quote", [
    "We hope to improve membrane selectivity in future laboratory experiments.",
    "We measured no significant improvement in membrane selectivity in laboratory experiments.",
    "Our laboratory experiments did not improve membrane selectivity.",
    "We measured increased noise in laboratory experiments using lithium selective membranes.",
    "Our device reduced measurement accuracy in laboratory experiments.",
    "Earlier researchers improved lithium selectivity in laboratory experiments.",
    "Baseline membranes reduce energy consumption in laboratory experiments.",
    "We report that previous studies improved selectivity in laboratory experiments.",
    "Our simulations show improved selectivity in numerical experiments.",
    "Our first-principles calculations predict improved selectivity in laboratory experiments.",
    "Нами снижена точность измерения в лабораторном эксперименте.",
    "<jats:p>We did n<jats:italic>ot</jats:italic> improve membrane selectivity in laboratory experiments.</jats:p>",
])
def test_qualified_baseline_adverse_and_nonempirical_claims_are_not_empirical_advantages(quote):
    assert not _role_supported("advantage", quote, context=quote, method_version=RAW_GROUNDING_METHOD)
    assert not automatic_research_application(quote, document(1, abstract=quote))


@pytest.mark.parametrize("qualification", [
    "Future work may improve the membrane lifetime.",
    "We did not use an expensive purification step.",
    "Earlier researchers reported a conventional membrane design.",
])
def test_unrelated_caution_does_not_erase_an_observed_benefit(tmp_path, qualification):
    abstract = BENEFIT + " " + qualification
    archive, docs, data, item, context = prepared(tmp_path, abstract=abstract)
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    assert {claim.role for claim in card.claims if claim.support == "supported"} >= {"case", "advantage", "application"}
    application = next(claim for claim in card.claims if claim.role == "application")
    assert application.grounding_method == RESEARCH_APPLICATION_METHOD
    assert application.text == BENEFIT
    assert automatic_research_application(application.text, docs[0])


@pytest.mark.parametrize("conclusion", [
    "However, our results do not support the proposed mechanism.",
    "We were unable to reproduce the improvement.",
    "Our measurements failed to confirm the claimed effect.",
    "We found no significant improvement in the final analysis.",
])
def test_explicit_contrary_conclusion_rejects_cherry_picked_primary_result(conclusion):
    record = document(1, abstract=BENEFIT + " " + conclusion)
    assert primary_result_sentence(record, method_version=RAW_GROUNDING_METHOD) is None
    assert not automatic_research_application(BENEFIT, record)


def test_raw_jats_decimal_quote_offsets_and_hash_survive_passport_and_primary_replay(tmp_path):
    prefix = "Background describes lithium extraction.\t"
    raw = prefix + RAW_BENEFIT + "\nFuture work may improve membrane lifetime."
    archive, docs, data, item, context = prepared(tmp_path, abstract=raw)
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    quotations = {entry.quote: entry for entry in card.evidence}
    assert RAW_BENEFIT in quotations
    source = quotations[RAW_BENEFIT]
    assert source.start == len(prefix) and source.end == len(prefix) + len(RAW_BENEFIT)
    assert raw[source.start:source.end] == RAW_BENEFIT
    verify_evidence_text(source, docs[0].abstract)
    assert not any("94%" == entry.quote[:3] for entry in card.evidence)
    primary = extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    assert primary[0].result == source
    verify_primary_observations(primary, item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    verify_primary_sources(primary, item, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)


def test_legacy_decimal_and_context_rules_are_frozen():
    legacy = grounding_sentences(RAW_BENEFIT, method_version=PRIMARY_GROUNDING_METHOD)
    assert RAW_BENEFIT not in legacy and any("43." in sentence for sentence in legacy)
    assert grounding_sentences(RAW_BENEFIT, method_version=RAW_GROUNDING_METHOD) == (RAW_BENEFIT,)
    legacy_quote = "We measured reduced energy consumption in laboratory experiments."
    record = document(1, abstract=legacy_quote + " Future work may improve lifetime.")
    selection = QuoteSelection(role="advantage", revision_id="probe", field="abstract", quote=legacy_quote)
    assert not _selection_supported(selection, record)
    assert _selection_supported(selection, record, method_version=PRIMARY_GROUNDING_METHOD)
    assert _selection_supported(selection, record, method_version=RAW_GROUNDING_METHOD)


def test_repeated_quote_uses_actual_sentence_span_not_earlier_substring(tmp_path):
    prefix = "Previous researchers reported that " + BENEFIT + " "
    abstract = prefix + BENEFIT
    archive, docs, data, item, context = prepared(tmp_path, abstract=abstract)
    evidence = raw_quote_evidence(data.documents[0], docs[0], quote=BENEFIT)
    assert evidence.start == len(prefix)
    assert abstract.find(BENEFIT) < evidence.start
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    assert all(entry.start == len(prefix) for entry in card.evidence if entry.quote == BENEFIT)
    primary = extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    assert primary[0].result == evidence
    with pytest.raises(EvidenceError, match="целым предложением"):
        raw_quote_evidence(data.documents[0], docs[0], quote="43.94% reduction in energy consumption")


@pytest.mark.parametrize("abstract,kind", [
    ("We simulated lithium selective membranes and measured improved selectivity in numerical experiments.", "computational"),
    ("We derived an analytical transport relation for lithium selective membranes.", "theoretical"),
    ("We simulated lithium selective membranes using a numerical model. We measured an improved transport rate.", "computational"),
    ("We measured lithium selectivity in laboratory experiments using three brine compositions.", "experimental"),
])
def test_primary_result_type_does_not_turn_simulation_into_experimental_application(tmp_path, abstract, kind):
    archive, docs, data, item, context = prepared(tmp_path, abstract=abstract)
    primary = extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    assert len(primary) == 1 and primary[0].result_kind == kind
    if kind != "experimental":
        assert not automatic_research_application(primary[0].result.quote, docs[0])
        card = build_passport(item, data, archive, context, methodology_version="3.4.0")
        assert not any(claim.role in {"advantage", "application"} and claim.support == "supported" for claim in card.claims)


@pytest.mark.parametrize("changes", [
    {"specificity": "uncertain"},
    {"specificity": "broad_topic"},
    {"admission_rule_hash": "a" * 64},
])
def test_single_paper_benefit_is_not_broadcast_to_an_unverified_group(tmp_path, changes):
    archive, _, data, item, context = prepared(tmp_path, abstract=BENEFIT)
    card = build_passport(item.model_copy(update=changes), data, archive, context, methodology_version="3.4.0")
    advantage = next(claim for claim in card.claims if claim.role == "advantage")
    assert advantage.support == "unverified"
    assert not any(claim.role == "application" for claim in card.claims)


def test_one_off_scope_member_prevents_group_advantage_even_with_specific_flag(tmp_path):
    archive, docs, _, item, context = prepared(tmp_path, abstract=BENEFIT)
    data = snapshot((docs[0], document(2, title="Sodium selective membranes for batteries", abstract="A battery mechanism is described.")), archive)
    item = item.model_copy(update={"discovery_snapshot_id": data.snapshot_id,
                                  "discovery_study_ids": tuple(sorted(ref.study_id for ref in data.documents))})
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    assert not any(claim.role in {"advantage", "application"} and claim.support == "supported" for claim in card.claims)


def test_novelty_raw_sentence_replay_rejects_omission_or_wrong_method(tmp_path):
    abstract = NOVELTY + " We measured lithium selectivity of 43.94% in laboratory experiments, i.e. in three brines. Future work may improve lifetime."
    archive, _, data, item, context = prepared(tmp_path, abstract=abstract)
    evidence = extract_signal_evidence(item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION)
    assert len(evidence) == 1 and "43.94%" in evidence[0].experiment.quote and "i.e." in evidence[0].experiment.quote
    verify_signal_evidence(evidence, item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION)
    verify_signal_sources(evidence, item, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION)
    with pytest.raises(EvidenceError):
        verify_signal_evidence((), item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION)
    with pytest.raises(EvidenceError):
        verify_signal_sources(evidence, item, archive, context, query_plan=query_plan(), method_version=CONTEXTUAL_SIGNAL_EVIDENCE_VERSION)


def test_primary_raw_sentence_replay_rejects_omission_or_wrong_method(tmp_path):
    archive, _, data, item, context = prepared(tmp_path, abstract=BENEFIT)
    primary = extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    with pytest.raises(EvidenceError):
        verify_primary_observations((), item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    for verify, args in ((verify_primary_sources, (item, archive, context)),
                         (verify_primary_observations, (item, data, archive, context))):
        with pytest.raises(EvidenceError):
            verify(primary, *args, query_plan=query_plan(), method_version=PRIMARY_EVIDENCE_VERSION)


def test_retraction_notice_in_another_snapshot_revision_cannot_be_hidden_by_richer_abstract(tmp_path):
    archive, docs, _, item, context = prepared(tmp_path, abstract=NOVELTY + " " + BENEFIT)
    notice = document(2, doi="10.1234/notice", document_type="retraction", abstract=None,
                      raw_metadata={"update-to": [{"DOI": docs[0].doi, "type": "retraction"}]})
    data = snapshot((docs[0], notice), archive)
    item = item.model_copy(update={"discovery_snapshot_id": data.snapshot_id})
    assert extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION) == ()
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION) == ()
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    assert not any(claim.support == "supported" for claim in card.claims)
    # Old primary extraction uses only its frozen source-level status rule.
    assert extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=PRIMARY_EVIDENCE_VERSION)


@pytest.mark.parametrize("reverse", [False, True], ids=["supplement-to", "parent-to-supplement"])
def test_rich_unflagged_supplement_cannot_hide_status_in_another_revision(tmp_path, reverse):
    archive, docs, _, item, context = prepared(tmp_path, abstract=NOVELTY + " " + BENEFIT)
    parent = document(9, doi="10.1234/parent", title="Lithium selective membranes parent research")
    if reverse:
        status = parent.model_copy(update={"raw_metadata": {"relation": {
            "is-supplemented-by": [{"id-type": "doi", "id": docs[0].doi}]}}})
    else:
        status = document(2, source="crossref", doi=docs[0].doi, abstract=None, raw_metadata={"relation": {
            "is-supplement-to": [{"id-type": "doi", "id": parent.doi}]}})
    data = snapshot((docs[0], parent, status), archive)
    item = item.model_copy(update={"discovery_snapshot_id": data.snapshot_id})
    assert extract_primary_observations(item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION) == ()
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION) == ()
    card = build_passport(item, data, archive, context, methodology_version="3.4.0")
    assert not any(claim.role in {"case", "advantage", "application"} and claim.support == "supported" for claim in card.claims)


def test_new_novelty_does_not_admit_future_month_in_current_year(tmp_path):
    archive, _, data, item, context = prepared(tmp_path, publication_year=2026, publication_month=12, date_precision="month")
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION) == ()
    # Existing version 2 used only year admission; its replay remains unchanged.
    assert extract_signal_evidence(item, data, archive, context, query_plan=query_plan(), method_version=CONTEXTUAL_SIGNAL_EVIDENCE_VERSION)


@pytest.mark.parametrize("offset", [-1, 60, True, 0.5])
def test_label_offsets_reject_invalid_or_out_of_budget_indices(tmp_path, offset):
    archive, _, data, item, context = prepared(tmp_path)
    with pytest.raises(EvidenceError, match="смещение"):
        label_candidates((item,), data, archive, context, stage_offset=offset)


def test_continuation_labels_use_global_request_ids_and_record_rejected_input(tmp_path):
    archive, docs, data, item, context = prepared(tmp_path)
    calls = []

    def generate_json(*args, **kwargs):
        calls.append(kwargs["request_id"])
        value = LabelBatch.model_validate(label_payload(item, data, docs, in_scope=False, scope_reason="Mixed scope."))
        return SimpleNamespace(value=value)

    client = SimpleNamespace(last_receipt=None, generate_json=generate_json)
    assert label_candidates((item,), data, archive, context, client=client, query_plan=query_plan()) == ()
    assert label_candidates((item,), data, archive, context, client=client, query_plan=query_plan(), stage_offset=30) == ()
    assert calls == ["run-one:labels:0:attempt:1", "run-one:labels:30:attempt:1"]
    for index in (0, 30):
        checkpoint = context.checkpoints[f"labels_{index}"]
        assert checkpoint["input_candidate_ids"] == [item.candidate_id]
        assert checkpoint["rejected"][0]["reason_code"] == "off_scope_or_mixed"


def test_empty_new_extractions_replay_explicitly_without_falling_back_to_legacy(tmp_path):
    archive, _, data, item, context = prepared(tmp_path, abstract="Future work may investigate this membrane.")
    verify_primary_observations((), item, data, archive, context, query_plan=query_plan(), method_version=RAW_PRIMARY_EVIDENCE_VERSION)
    verify_signal_evidence((), item, data, archive, context, query_plan=query_plan(), method_version=RAW_SIGNAL_EVIDENCE_VERSION)
