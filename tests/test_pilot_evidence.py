"""Evidence is extracted from immutable real-shaped records, never model attestations."""

import json
from datetime import datetime, timezone
from threading import Event

import httpx
import pytest

from app.backend.contracts import DocumentRecord
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Candidate, CorpusSnapshot, Coverage, QueryPlan, SearchQuery, content_hash
from app.pilot.evidence import (
    TITLE_ADMISSION_VERSION, EvidenceError, admission_hash, build_passport, label_candidates,
    quote_evidence, title_matches, candidate_title_matches, coherent_aliases,
    representative_documents, scope_is_anchored, _role_supported,
)
from app.runtime.jobs import TaskCancelled
from tests import test_pilot_llm

clients = test_pilot_llm.clients
NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


class Context:
    def __init__(self):
        self.run_id = "run-one"
        self.attempt = 1
        self.cancel_event = Event()
        self.checkpoints = {}
        self.progress_events = []

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise TaskCancelled()

    def load_checkpoint(self, stage):
        self.check_cancelled()
        return self.checkpoints.get(stage)

    def checkpoint(self, stage, value):
        self.check_cancelled()
        self.checkpoints[stage] = json.loads(json.dumps(value))

    def progress(self, *args):
        self.check_cancelled()
        self.progress_events.append(args)


def query_plan():
    return QueryPlan(original_query="Извлечение лития", language="ru", definition="Direct lithium extraction from brines",
        english_query="direct lithium extraction", subdirections=("Selective membranes",),
        queries=(SearchQuery(source="openalex", text="lithium selective membranes"),),
        completed_years=tuple(range(2020, 2026)), as_of=NOW.date(), planner_version="test")


def document(number, year=2025, **changes):
    fields = dict(source="openalex", source_id=f"W{number}", doi=f"10.1234/study{number}",
        title=f"Lithium selective membranes for extraction experiment {number}",
        abstract="Existing extraction methods suffer from limited selectivity. "
                 "The lithium selective membrane reduces energy consumption in laboratory experiments.",
        publication_year=year, date_precision="year", authors=(f"Researcher {number}",),
        url=f"https://doi.org/10.1234/study{number}", fetched_at=NOW,
        raw_metadata={"is_retracted": False, "authorships": [{"author": {"id": f"https://openalex.org/A{number}"},
            "institutions": [{"id": f"https://openalex.org/I{number}"}]}]})
    return DocumentRecord(**(fields | changes))


def snapshot(documents, archive, *, purpose="discovery", coverage=None, snapshot_id=None):
    references = tuple(archive.put(item) for item in documents)
    plan = query_plan()
    years = plan.completed_years if purpose == "history" else (2023, 2024, 2025, 2026)
    observed_coverage = coverage or (Coverage(source="openalex", purpose=purpose, query_hash="a" * 64,
        state="complete", requested_years=years, completed_years=years, pagination_exhausted=True,
        comparable=purpose == "history", scanned_records=len(documents), accepted_records=len(documents)),)
    return CorpusSnapshot(snapshot_id=snapshot_id or content_hash({"purpose": purpose, "ids": [ref.revision_id for ref in references]}),
        plan_hash=plan.plan_hash, purpose=purpose, created_at=NOW, as_of=NOW.date(), documents=references,
        coverage=observed_coverage, normalizer_version="test", deduplication_version="doi-source-id-v1")


def candidate(data, **changes):
    fields = dict(candidate_id="candidate-one", plan_hash=data.plan_hash, label="lithium selective membranes",
        definition="Candidate requiring review", admission_rule_version="discovery-test", admission_rule_hash="b" * 64,
        discovery_snapshot_id=data.snapshot_id, discovery_study_ids=tuple(sorted({ref.study_id for ref in data.documents})),
        specificity="uncertain")
    return Candidate(**(fields | changes))


@pytest.fixture
def library(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1), document(2))
    data = snapshot(docs, archive)
    return archive, docs, data, candidate(data), Context()


def label_payload(item, data, docs, **changes):
    payload = dict(candidate_id=item.candidate_id, label="Литий-селективные мембраны",
        definition="Selective membrane mechanisms for lithium recovery from brines.",
        phrases=["lithium selective membranes"], exclusions=[], specificity="specific_technology", in_scope=True,
        scope_support=[dict(revision_id=ref.revision_id, field="title", quote=doc.title)
                       for ref, doc in zip(data.documents, docs, strict=True)], scope_reason="Названия описывают извлечение лития.")
    return {"candidates": [payload | changes]}


def test_local_pasport_has_only_exact_original_claims_and_no_invented_novelty(library):
    archive, docs, data, item, context = library
    card = build_passport(item, data, archive, context)
    assert {claim.role for claim in card.claims} == {"problem", "advantage", "case", "application"}
    evidence = {entry.evidence_id: entry for entry in card.evidence}
    for claim in card.claims:
        assert claim.support == "supported"
        assert claim.text == evidence[claim.evidence_ids[0]].quote
        assert any(claim.text in doc.title or claim.text in (doc.abstract or "") for doc in docs)
    assert card.category == "unassessed_cluster" and card.quality == "partial"
    assert next(claim for claim in card.claims if claim.role == "application").grounding_method == "verified-application/research"
    assert build_passport(item, data, archive, context) == card


def test_absent_motivation_and_negated_or_aspirational_benefits_stay_missing(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1, abstract="The mechanism does not reduce energy consumption. We hope to improve the result."),)
    data = snapshot(docs, archive)
    card = build_passport(candidate(data), data, archive, Context())
    assert {claim.role for claim in card.claims} == {"case"}
    assert any("преимущество" in value for value in card.limitations)


def test_quote_verification_rejects_rewording_and_discontinuous_excerpt(library):
    archive, docs, data, _, _ = library
    with pytest.raises(EvidenceError, match="отсутствует"):
        quote_evidence(data.documents[0], docs[0], text_field="abstract", quote="Our technology is now commercially proven.")
    with pytest.raises(EvidenceError):
        quote_evidence(data.documents[0], docs[0], text_field="abstract", quote="Existing extraction ... reduces energy")


def test_frozen_local_rule_has_no_false_specificity_and_preserves_membership(library):
    archive, _, data, item, context = library
    frozen, = label_candidates((item,), data, archive, context)
    assert frozen.discovery_study_ids == item.discovery_study_ids
    assert frozen.specificity == "uncertain"
    assert frozen.admission_rule_version == TITLE_ADMISSION_VERSION
    assert frozen.admission_rule_hash == admission_hash(frozen)
    assert title_matches("Lithium-selective membranes", frozen.synonyms, ())
    assert not title_matches("Lithium selective membrane's review", frozen.synonyms, ())
    assert not title_matches("Lithium selective membranes battery recycling", frozen.synonyms, ("battery recycling",))


def test_offline_label_fallback_handles_a_realistic_long_publication_title(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    title = ("A Comprehensive Study of Quantum Dots: Ranging From Synthesis to Applications in "
             "Electrochemical Biosensors in the Detection of Biomolecules, Gastrointestinal Diseases, "
             "and Electrophysiology")
    assert 180 < len(title) <= 200
    source = document(404, title=title)
    data = snapshot((source,), archive)
    item = candidate(data).model_copy(update={"label": title})
    checked, = label_candidates((item,), data, archive, Context(), query_plan=query_plan())
    assert checked.specificity == "uncertain"
    assert checked.synonyms and all(len(phrase) <= 180 for phrase in checked.synonyms)
    assert all(title_matches(title, (phrase,), ()) for phrase in checked.synonyms)


def test_ai_names_require_scope_evidence_for_every_representative_and_cache_the_receipt(library, clients):
    archive, docs, data, item, context = library
    payload = label_payload(item, data, docs)
    requests = []

    def handler(request):
        requests.append(request)
        assert json.loads(json.loads(request.content)["messages"][1]["content"])["requested_scope"] == query_plan().definition
        return httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload)))

    client, budget = clients(handler)
    first = label_candidates((item,), data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan())
    assert first[0].specificity == "specific_technology"
    assert context.checkpoints["labels_0"]["receipt"]["usage"]["total_tokens"] == 120
    assert label_candidates((item,), data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan()) == first
    assert len(requests) == 1 and budget.snapshot("run:one").used.calls == 1


def test_single_document_can_define_specificity_without_proving_a_trend(tmp_path, clients):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1),)
    data = snapshot(docs, archive)
    item = candidate(data)
    payload = label_payload(item, data, docs)
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    context = Context()
    frozen, = label_candidates((item,), data, archive, context, client=client,
                                scope_ids=("run:one",), query_plan=query_plan())
    assert frozen.specificity == "specific_technology"
    card = build_passport(frozen, data, archive, context)
    assert card.category == "insufficient_evidence"
    assert card.assessment_hash is None


def test_single_generic_word_does_not_define_a_specific_mechanism(tmp_path, clients):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1, title="Novel materials for lithium selective membranes"),)
    data = snapshot(docs, archive)
    item = candidate(data)
    payload = label_payload(item, data, docs, phrases=["novel materials"], label="Новые материалы")
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    frozen, = label_candidates((item,), data, archive, Context(), client=client,
                                scope_ids=("run:one",), query_plan=query_plan())
    assert frozen.specificity == "uncertain"


def test_valid_frozen_definition_survives_no_ai_recheck_and_checkpoint(library):
    archive, _, data, original, context = library
    item = original.model_copy(update={"specificity": "specific_technology",
        "synonyms": ("lithium selective membranes",), "admission_rule_version": TITLE_ADMISSION_VERSION})
    item = item.model_copy(update={"admission_rule_hash": admission_hash(item)})
    assert label_candidates((item,), data, archive, context, query_plan=query_plan()) == (item,)
    assert context.checkpoints["labels_0"]["label_status"] == "retained_frozen_definition"
    assert label_candidates((item,), data, archive, context, query_plan=query_plan()) == (item,)


def test_new_snapshot_copy_invalidates_label_and_passport_checkpoints(library):
    archive, _, data, item, _ = library
    extra_reference = archive.put(document(99))
    changed = data.model_copy(update={"documents": (*data.documents, extra_reference)})
    assert changed.snapshot_hash != data.snapshot_hash

    label_context = Context()
    label_candidates((item,), data, archive, label_context, query_plan=query_plan())
    with pytest.raises(EvidenceError, match="другой выборке"):
        label_candidates((item,), changed, archive, label_context, query_plan=query_plan())

    passport_context = Context()
    build_passport(item, data, archive, passport_context, methodology_version="3.4.0")
    with pytest.raises(EvidenceError, match="другой версии кандидата"):
        build_passport(item, changed, archive, passport_context, methodology_version="3.4.0")


@pytest.mark.parametrize("problem", ["stale_hash", "mixed_members", "broad_scope", "legacy", "unknown_scope"])
def test_frozen_definition_is_not_preserved_without_current_grounding(tmp_path, problem):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1), document(2, title="Unrelated optical computing") if problem == "mixed_members" else document(2))
    data = snapshot(docs, archive)
    item = candidate(data, specificity="specific_technology", synonyms=("lithium selective membranes",),
        admission_rule_version="title-phrase-admission/1.0.0" if problem == "legacy" else TITLE_ADMISSION_VERSION,
        label=query_plan().original_query if problem == "broad_scope" else "Литий-селективные мембраны")
    item = item.model_copy(update={"admission_rule_hash": "a" * 64 if problem == "stale_hash" else admission_hash(item)})
    checked, = label_candidates((item,), data, archive, Context(),
        query_plan=None if problem == "unknown_scope" else query_plan())
    assert checked.specificity != "specific_technology"


@pytest.mark.parametrize("changes", [dict(in_scope=False, scope_support=[], scope_reason="Неподходящая область"),
    dict(scope_support=[]), dict(scope_support=[dict(revision_id="c" * 64, field="title", quote="Invented title")])])
def test_offscope_and_unsupported_groups_are_rejected_before_history(library, clients, changes):
    archive, docs, data, item, context = library
    payload = label_payload(item, data, docs, **changes)
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    assert label_candidates((item,), data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan()) == ()
    assert context.checkpoints["labels_0"]["rejected"][0]["candidate_id"] == item.candidate_id
    assert label_candidates((item,), data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan()) == ()


def test_ai_cannot_invent_candidate_membership_or_unsupported_historical_synonyms(library, clients):
    archive, docs, data, item, context = library
    payload = label_payload(item, data, docs, phrases=["imaginary quantum lithium fusion"])
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    frozen, = label_candidates((item,), data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan())
    assert frozen.specificity == "uncertain"
    assert "imaginary quantum lithium fusion" not in frozen.synonyms
    assert frozen.discovery_study_ids == item.discovery_study_ids
    assert context.checkpoints["labels_0"]["label_status"] == "unverified_lexical_label"


def test_bad_ai_quote_falls_back_to_original_excerpts_without_repaying(library, clients):
    archive, _, data, item, context = library
    payload = {"selections": [{"role": "advantage", "revision_id": data.documents[0].revision_id,
                                "field": "abstract", "quote": "This eliminates every known technological limitation."}],
               "russian_interpretation": "Вымышленный успех"}
    client, budget = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    card = build_passport(item, data, archive, context, client=client, scope_ids=("run:one",))
    assert not any(claim.text == "Вымышленный успех" for claim in card.claims)
    assert any("не принято" in value for value in card.limitations)
    assert budget.snapshot("run:one").used.calls == 1
    assert build_passport(item, data, archive, context, client=client, scope_ids=("run:one",)) == card
    assert budget.snapshot("run:one").used.calls == 1


def test_russian_ai_interpretation_never_becomes_supported_scientific_claim(library, clients):
    archive, docs, data, item, context = library
    payload = {"selections": [{"role": "case", "revision_id": data.documents[0].revision_id,
                                "field": "title", "quote": docs[0].title}],
               "russian_interpretation": "Технология может помочь извлечению лития."}
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    card = build_passport(item, data, archive, context, client=client, scope_ids=("run:one",))
    assert next(claim for claim in card.claims if claim.role == "summary").support == "unverified"


def test_cancelled_work_cannot_publish_a_passport(library):
    archive, _, data, item, context = library
    context.cancel_event.set()
    with pytest.raises(TaskCancelled):
        build_passport(item, data, archive, context)
    assert not context.checkpoints


def test_sixteen_candidate_labels_use_bounded_evidence_batches_without_one_call_per_candidate(library, clients):
    archive, docs, data, item, context = library
    items = tuple(Candidate.model_validate(item.model_dump(mode="python") | {"candidate_id": f"candidate-{index}"})
                  for index in range(16))
    calls = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        names = []
        for cluster in payload["clusters"]:
            original = next(entry for entry in items if entry.candidate_id == cluster["candidate_id"])
            names.extend(label_payload(original, data, docs)["candidates"])
        return httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps({"candidates": names})))

    client, budget = clients(handler)
    result = label_candidates(items, data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan())
    assert len(result) == 16 and len(calls) == 4
    assert all(len(call["clusters"]) == 4 for call in calls)
    assert all(sum(len(item["documents"]) for item in call["clusters"]) <= 8 for call in calls)
    assert budget.snapshot("run:one").used.calls == 4


def test_broad_cluster_batch_bound_keeps_all_five_evidence_representatives(tmp_path, clients):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(index) for index in range(5))
    data = snapshot(docs, archive)
    items = tuple(candidate(data, candidate_id=f"candidate-{index}") for index in range(3))
    calls = []

    def handler(request):
        envelope = json.loads(request.content)
        payload = json.loads(envelope["messages"][1]["content"])
        calls.append(payload)
        assert envelope["max_tokens"] == 2156
        assert len(payload["clusters"]) == 1
        cluster = payload["clusters"][0]
        assert len(cluster["documents"]) == 5
        item = next(item for item in items if item.candidate_id == cluster["candidate_id"])
        return httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(label_payload(item, data, docs))))

    client, _ = clients(handler)
    context = Context()
    result = label_candidates(items, data, archive, context, client=client,
                              scope_ids=("run:one",), query_plan=query_plan())
    assert len(calls) == 3 and all(item.specificity == "specific_technology" for item in result)
    assert all(checkpoint["failure_code"] is None and not checkpoint["fallback_candidate_ids"]
               for checkpoint in context.checkpoints.values())


def test_scope_of_every_member_is_checked_even_when_sample_omits_an_off_scope_paper(
        tmp_path, clients, monkeypatch):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(index, title=f"Lithium membrane sensors for brine processing {index}",
                          abstract="Direct lithium extraction from brines with membrane sensors.")
                 for index in range(5)) + (
        document(6, title="Lithium membrane sensors for quantum battery separators",
                 abstract="A battery separator is evaluated during charge cycles."),)
    data = snapshot(docs, archive)
    item = candidate(data, label="lithium membrane sensors")
    context = Context()
    calls = []
    client, budget = clients(lambda request: calls.append(request))
    # The production review samples at most five papers. Force the boundary
    # case directly so the sixth member cannot be qualified by a cited sample.
    monkeypatch.setattr("app.pilot.evidence.representative_documents", lambda documents: documents[:5])

    named, = label_candidates((item,), data, archive, context, client=client,
                              scope_ids=("run:one",), query_plan=query_plan())

    assert named.specificity == "uncertain"
    assert context.checkpoints["labels_0"]["failure_code"] == "evidence_rejected"
    assert context.checkpoints["labels_0"]["label_status"] == "unverified_lexical_label"
    assert not calls and budget.snapshot("run:one").used.calls == 0


def test_eligible_member_is_reviewed_even_beside_an_unanchored_member(tmp_path, clients):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1), document(2, title="Quantum battery separator materials",
                             abstract="Cells were cycled at room temperature."))
    data = snapshot(docs, archive)
    relevant = candidate(data, candidate_id="relevant", discovery_study_ids=(data.documents[0].study_id,))
    unrelated = candidate(data, candidate_id="unrelated", discovery_study_ids=(data.documents[1].study_id,))
    seen = []

    def handler(request):
        clusters = json.loads(json.loads(request.content)["messages"][1]["content"])["clusters"]
        seen.append([cluster["candidate_id"] for cluster in clusters])
        payload = label_payload(relevant, data, docs)
        payload["candidates"][0]["scope_support"] = [dict(
            revision_id=data.documents[0].revision_id, field="title", quote=docs[0].title)]
        return httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload)))

    client, budget = clients(handler)
    context = Context()
    named = label_candidates((unrelated, relevant), data, archive, context, client=client,
                             scope_ids=("run:one",), query_plan=query_plan())

    assert seen == [["relevant"]]
    assert named[0].specificity == "uncertain"
    assert named[1].specificity == "specific_technology"
    assert context.checkpoints["labels_0"]["label_status"] == "mixed_model_and_lexical"
    assert budget.snapshot("run:one").used.calls == 1


@pytest.mark.parametrize("reason", ["read_timeout", "budget_exceeded"])
def test_label_fallback_records_safe_diagnostics_and_does_not_repay(library, clients, reason):
    archive, _, data, item, context = library
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout(test_pilot_llm.SECRET, request=request)

    kwargs = {"limits": test_pilot_llm.BudgetLimits(0, 0, 0, 0)} if reason == "budget_exceeded" else {}
    client, budget = clients(handler, **kwargs)
    first = label_candidates((item,), data, archive, context, client=client,
                               scope_ids=("run:one",), query_plan=query_plan())
    assert first[0].specificity == "uncertain"
    checkpoint = context.checkpoints["labels_0"]
    assert checkpoint["failure_code"] == reason
    assert checkpoint["fallback_candidate_ids"] == [item.candidate_id]
    assert test_pilot_llm.SECRET not in json.dumps(checkpoint)
    assert label_candidates((item,), data, archive, context, client=client,
                            scope_ids=("run:one",), query_plan=query_plan()) == first
    expected_calls = 0 if reason == "budget_exceeded" else 1
    assert len(calls) == expected_calls and budget.snapshot("run:one").used.calls == expected_calls


def test_budget_refusal_skips_later_labels_and_survives_resume(library, clients, monkeypatch):
    archive, docs, data, _item, context = library
    items = tuple(candidate(data, candidate_id=f"candidate-{index}") for index in range(16))
    requests = []

    def handler(request):
        clusters = json.loads(json.loads(request.content)["messages"][1]["content"])["clusters"]
        requests.append([cluster["candidate_id"] for cluster in clusters])
        names = [label_payload(next(item for item in items if item.candidate_id == cluster["candidate_id"]),
                               data, docs)["candidates"][0] for cluster in clusters]
        return httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps({"candidates": names})))

    client, budget = clients(handler, limits=test_pilot_llm.BudgetLimits(1, 200000, 30000, 1000000))
    reserve = budget.reserve
    admissions = []

    def counted_reserve(*args, **kwargs):
        admissions.append(args[0])
        return reserve(*args, **kwargs)

    monkeypatch.setattr(budget, "reserve", counted_reserve)
    first = label_candidates(items[:12], data, archive, context, client=client,
                             scope_ids=("run:one",), query_plan=query_plan(),
                             budget_unavailable=lambda: budget.snapshot("run:one").remaining.calls == 0)
    assert len(requests) == 1 and len(admissions) == 2
    assert all(item.specificity == "uncertain" for item in first[4:])
    assert context.checkpoints["labels_1"]["failure_code"] == "budget_exceeded"
    assert context.checkpoints["labels_2"]["failure_code"] == "budget_exceeded"

    second = label_candidates(items[12:], data, archive, context, client=client,
                              scope_ids=("run:one",), query_plan=query_plan(), stage_offset=12,
                              budget_unavailable=lambda: budget.snapshot("run:one").remaining.calls == 0)
    assert len(second) == 4 and all(item.specificity == "uncertain" for item in second)
    assert len(admissions) == 2
    assert context.checkpoints["labels_12"]["failure_code"] == "budget_exceeded"
    assert budget.snapshot("run:one").used.calls == 1


def test_large_label_batch_refusal_does_not_skip_a_smaller_feasible_batch(library, clients):
    archive, docs, data, _item, context = library
    items = tuple(candidate(data, candidate_id=f"candidate-{index}") for index in range(5))
    requests = []

    def handler(request):
        clusters = json.loads(json.loads(request.content)["messages"][1]["content"])["clusters"]
        requests.append([cluster["candidate_id"] for cluster in clusters])
        payload = label_payload(items[4], data, docs)
        return httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload)))

    client, budget = clients(handler, limits=test_pilot_llm.BudgetLimits(24, 200000, 1500, 1000000))
    named = label_candidates(items, data, archive, context, client=client,
                             scope_ids=("run:one",), query_plan=query_plan(),
                             budget_unavailable=lambda: budget.snapshot("run:one").remaining.calls == 0)

    assert all(item.specificity == "uncertain" for item in named[:4])
    assert named[4].specificity == "specific_technology"
    assert context.checkpoints["labels_0"]["failure_code"] == "budget_exceeded"
    assert context.checkpoints["labels_0"]["budget_stopped"] is False
    assert context.checkpoints["labels_1"]["failure_code"] is None
    assert requests == [["candidate-4"]]
    assert budget.snapshot("run:one").used.calls == 1


def test_label_progress_tracks_actual_batches_before_dispatch_and_cached_or_cancelled_work(library, clients):
    archive, _, data, item, context = library
    items = tuple(Candidate.model_validate(item.model_dump(mode="python") | {"candidate_id": f"candidate-{index}"})
                  for index in range(9))
    observed = []

    def handler(request):
        observed.append(context.progress_events[-1])
        raise httpx.ReadTimeout("Controlled response timeout", request=request)

    client, _budget = clients(handler)
    first = label_candidates(items, data, archive, context, client=client,
                             scope_ids=("run:one",), query_plan=query_plan())
    expected = [("labels", "Проверяем названия и границы кандидатов: 1–4 из 9", 0, 9),
                ("labels", "Проверяем названия и границы кандидатов: 5–8 из 9", 4, 9),
                ("labels", "Проверяем названия и границы кандидатов: 9 из 9", 8, 9)]
    assert observed == expected and all(candidate.specificity == "uncertain" for candidate in first)
    context.progress_events.clear()
    assert label_candidates(items, data, archive, context, client=client,
                            scope_ids=("run:one",), query_plan=query_plan()) == first
    assert context.progress_events == expected and observed == expected  # Cache reads do not dispatch again.
    context.cancel_event.set()
    context.progress_events.clear()
    with pytest.raises(TaskCancelled):
        label_candidates(items, data, archive, context, client=client, scope_ids=("run:one",), query_plan=query_plan())
    assert not context.progress_events and observed == expected


def test_imported_report_content_is_not_silently_sent_to_external_ai(tmp_path, clients):
    from app.pilot.reports import ReportRecord

    archive = DocumentArchive(tmp_path / "reports")
    base = document(1, source="report", source_id="a" * 64, doi=None, url="https://example.org/report")
    report = ReportRecord.model_validate(base.model_dump() | dict(document_type="report", full_text=base.abstract,
        pdf_sha256="a" * 64, pages_total=1, pages_with_text=1, extraction_status="text",
        page_spans=[{"page": 1, "start": 0, "end": len(base.abstract)}],
        public_license_allowed=True, license_note="Test public license"))
    data = snapshot((report,), archive)
    calls = []
    client, budget = clients(lambda request: calls.append(request))
    card = build_passport(candidate(data), data, archive, Context(), client=client, scope_ids=("run:one",))
    assert not calls and budget.snapshot("run:one").used.calls == 0
    assert all(not evidence.external_ai_allowed for evidence in card.evidence)
    assert any("только локально" in value for value in card.limitations)


def test_new_admission_rejects_or_of_unrelated_parents_but_replays_legacy(library):
    _, _, data, _, _ = library
    old = candidate(data, synonyms=("graphene", "quantum computing"),
                    admission_rule_version="title-phrase-admission/1.0.0")
    new = old.model_copy(update={"admission_rule_version": TITLE_ADMISSION_VERSION})
    assert candidate_title_matches("Graphene membranes for water filtering", old)
    assert candidate_title_matches("Trapped ion quantum computing", old)
    assert not candidate_title_matches("Graphene membranes for water filtering", new)
    assert not candidate_title_matches("Trapped ion quantum computing", new)
    assert coherent_aliases(("optically detected magnetic resonance", "ODMR"))
    assert not coherent_aliases(("lithium ion batteries", "sodium ion batteries"))
    membrane = new.model_copy(update={"synonyms": ("lithium selective membranes",)})
    assert candidate_title_matches("Lithium selective membrane for extraction", membrane)
    assert not candidate_title_matches("Lithium selective membrane for extraction", membrane.model_copy(
        update={"admission_rule_version": "title-phrase-admission/1.0.0"}))


def test_proposed_narrow_name_cannot_borrow_evidence_from_other_chemistry(tmp_path, clients):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1), document(2), document(3, title="Zinc ion cathodes for aqueous batteries"))
    data = snapshot(docs, archive)
    item = candidate(data, label="ion batteries / sibs / libs")
    payload = label_payload(item, data, docs)
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    frozen, = label_candidates((item,), data, archive, Context(), client=client,
                                scope_ids=("run:one",), query_plan=query_plan())
    assert frozen.specificity == "uncertain"
    assert frozen.label == item.label
    assert frozen.discovery_study_ids == item.discovery_study_ids


def test_representatives_include_temporal_and_thematic_edges_not_first_ids(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(i, year=2024) for i in range(1, 9)) + (
        document(90, year=2020, title="Initial lithium separation in mineral brines"),
        document(99, year=2026, title="Protein spin resonance in engineered living cells"))
    members = tuple((archive.put(doc), doc) for doc in docs)
    selected = representative_documents(members)
    assert len(selected) == 5
    assert {2020, 2026}.issubset({doc.publication_year for _, doc in selected})
    assert representative_documents(tuple(reversed(members))) == selected


def test_shared_material_word_or_medical_application_does_not_ground_measurement_scope(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    plan = query_plan().model_copy(update={"english_query": "quantum sensing of biological systems",
        "synonyms": ("quantum biosensing", "quantum sensors in biology", "NV center biosensing"),
        "subdirections": ("diamond magnetometry in biology",)})
    for title in (
        "Deep Learning Approaches for Brain Tumor Detection and Classification Using MRI Images",
        "Carbon Quantum Dots in Biomedical Applications: Advances, Challenges, and Future Prospects",
        "A high-selectivity NIR fluorescent probe for detection of nitric oxide in saliva samples and living cells imaging",
        "The state of quantum computing applications in health and medicine",
    ):
        doc = document(1, title=title, abstract=None)
        assert not scope_is_anchored(((archive.put(doc), doc),), plan)
    doc = document(2, title="NV center biosensing through optically detected magnetic resonance", abstract=None)
    assert scope_is_anchored(((archive.put(doc), doc),), plan)


def test_query_itself_is_not_a_specific_discovered_technology(library, clients):
    archive, docs, data, item, context = library
    payload = label_payload(item, data, docs, label=query_plan().original_query)
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    frozen, = label_candidates((item,), data, archive, context, client=client,
                                scope_ids=("run:one",), query_plan=query_plan())
    assert frozen.specificity == "broad_topic"


@pytest.mark.parametrize("text", [
    "Previous research hypothesized that European robin's cryptochrome 4a (ErCry4a) optimized intra-protein motion to minimize spin relaxation, enhancing magnetic sensing compared to the plant Arabidopsis thaliana's cryptochrome 1 (AtCry1).",
    "The previous device reduces power in laboratory measurements.",
    "Our simulations predict enhanced hydrogen storage capacity.",
    "The system could improve energy efficiency in future experiments.",
    "The thermal motion reduces sensitivity in laboratory measurements.",
    "The added control increases noise and enhances error in laboratory measurements.",
])
def test_hypothesis_comparator_and_simulation_are_not_demonstrated_benefits(text):
    assert not _role_supported("advantage", text, context=text)


def test_context_prevents_cherry_picked_benefit_and_opposing_conclusion(tmp_path, clients):
    archive = DocumentArchive(tmp_path / "revisions")
    positive = "The membrane reduces energy consumption in laboratory measurements."
    # The exact positive fragment is genuine, but the whole sentence is negative.
    negative = "The membrane does not reduce energy consumption in laboratory measurements."
    assert not _role_supported("advantage", "reduce energy consumption in laboratory measurements.", context=negative)
    assert not _role_supported("advantage", positive,
        context=positive + " However, this effect is likely negligible in realistic systems.")
    docs = (document(1, abstract=positive + " However, this effect is likely negligible in realistic systems."),)
    data = snapshot(docs, archive)
    payload = {"selections": [{"role": "advantage", "revision_id": data.documents[0].revision_id,
                               "field": "abstract", "quote": positive}]}
    client, _ = clients(lambda _: httpx.Response(200, json=test_pilot_llm.response_payload(json.dumps(payload))))
    card = build_passport(candidate(data), data, archive, Context(), client=client, scope_ids=("run:one",))
    assert not any(claim.role in {"advantage", "application"} and claim.support == "supported" for claim in card.claims)


def test_review_title_is_not_a_concrete_research_case_or_application(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = (document(1, title="A review of lithium selective membranes", abstract=None),)
    data = snapshot(docs, archive)
    card = build_passport(candidate(data), data, archive, Context())
    assert not any(claim.role in {"case", "application"} and claim.support == "supported" for claim in card.claims)


def test_repeated_status_question_reuses_one_scan_but_never_mixes_two_snapshots(tmp_path):
    """One snapshot is scanned once per run; a different corpus gets its own answer."""
    from app.pilot import evidence as module

    archive = DocumentArchive(tmp_path / "revisions")
    retracted = document(3, title="Retracted lithium selective membrane study",
                         raw_metadata={"is_retracted": True})
    clean = snapshot((document(1), document(2)), archive)
    withdrawn = snapshot((document(1), retracted), archive)
    module._SNAPSHOT_STATUS_CACHE.clear()
    reads = []
    original = DocumentArchive.get
    DocumentArchive.get = lambda self, revision_id: (reads.append(revision_id), original(self, revision_id))[1]
    try:
        context = Context()
        first = module.snapshot_primary_exclusions(clean, archive, context)
        after_first = len(reads)
        assert after_first == len(clean.documents)
        assert module.snapshot_primary_exclusions(clean, archive, context) == first
        assert len(reads) == after_first, "повторный вопрос о том же снимке не должен перечитывать архив"
        other = module.snapshot_primary_exclusions(withdrawn, archive, context)
        assert len(reads) == after_first + len(withdrawn.documents)
    finally:
        DocumentArchive.get = original
    assert other != first
    assert retracted.document_key in other and retracted.document_key not in first


def test_cached_snapshot_status_still_honours_cancellation(tmp_path):
    from app.pilot import evidence as module

    archive = DocumentArchive(tmp_path / "revisions")
    data = snapshot((document(1), document(2)), archive)
    module._SNAPSHOT_STATUS_CACHE.clear()
    context = Context()
    module.snapshot_primary_exclusions(data, archive, context)
    context.cancel_event.set()
    with pytest.raises(TaskCancelled):
        module.snapshot_primary_exclusions(data, archive, context)


def test_offered_phrases_lead_with_the_ones_that_can_actually_be_admitted():
    """A phrase that anchors every title is the only kind `_freeze` accepts.

    Measured on a real corpus: 11 of 28 multi-document clusters had such a
    phrase and the local model named one of them, because it was free to type
    a phrase instead of choosing. Ordering is the admission rule, computed.
    """
    from app.pilot.evidence import _anchored_title_matches, covering_title_phrases

    # Only the titles matter here; the reference side is never read.
    docs = tuple((None, document(index, title=title)) for index, title in enumerate((
        "Nitrogen vacancy magnetometry in bulk diamond",
        "Nitrogen vacancy magnetometry at ambient temperature",
        "Wide field nitrogen vacancy magnetometry of live cells")))
    offered = covering_title_phrases(docs, limit=6)
    assert offered, "the shared phrase must be offered at all"
    covering = tuple(phrase for phrase in offered
                     if all(_anchored_title_matches(record.title, (phrase,), ()) for _, record in docs))
    # Whatever else is offered, an admissible phrase comes first.
    assert offered[0] in covering


def test_a_chosen_number_becomes_that_exact_phrase_and_nothing_else():
    from app.pilot.evidence import LocalCandidateName, _chosen_name

    proposal = LocalCandidateName(candidate_id="c0", label="Магнитометрия на NV-центрах",
                                  definition="Азот-вакансионные центры в алмазе измеряют магнитное поле.",
                                  phrase_number=2, specificity="specific_technology", in_scope=True)
    assert _chosen_name(proposal, ("first phrase", "nitrogen vacancy magnetometry")).phrases == (
        "nitrogen vacancy magnetometry",)


def test_a_number_outside_the_offered_list_is_refused_not_guessed():
    from app.pilot.evidence import EvidenceError, LocalCandidateName, _chosen_name

    proposal = LocalCandidateName(candidate_id="c0", label="Что-то", definition="Описание.",
                                  phrase_number=3, specificity="uncertain", in_scope=True)
    with pytest.raises(EvidenceError, match="допустимых фраз"):
        _chosen_name(proposal, ("only one",))
    with pytest.raises(EvidenceError, match="допустимых фраз"):
        _chosen_name(proposal, ())
