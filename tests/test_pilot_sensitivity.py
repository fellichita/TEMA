"""Counterfactuals retain real archive provenance and never turn missing data into zero."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.backend.contracts import DocumentRecord
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Claim, Coverage, TrendCard
from app.pilot.evidence import EvidenceError, build_passport, quote_evidence
from app.pilot.methodology import NoveltyAssessment
from app.pilot.sensitivity import (
    SensitivityReport, SensitivityScenario, evaluate_sensitivity, load_sensitivity, save_sensitivity,
    verify_sensitivity,
)
from app.runtime.jobs import TaskCancelled
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot
from tests.test_pilot_history import frozen
from tests.platform_support import require_symlinks


@pytest.fixture
def library(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(1000 + year * 10 + index, year=year) for year, count in zip(
        range(2020, 2026), (1, 1, 1, 2, 4, 8), strict=True) for index in range(count))
    discovery = snapshot(docs[-4:], archive)
    historical = snapshot(docs, archive, purpose="history")
    item = frozen(candidate(discovery))
    context = Context()
    passport = build_passport(item, discovery, archive, context)
    return archive, docs, historical, item, context, passport


def evaluate(library, **scenario):
    archive, _, historical, item, context, passport = library
    return evaluate_sensitivity(item, query_plan(), historical, archive, context,
                                passport=passport, scenario=SensitivityScenario(**scenario))


def crossref_passport(library, *, novelty=False):
    archive, docs, _, _, _, passport = library
    alias = DocumentRecord.model_validate(docs[-1].model_dump(mode="python") | {
        "source": "crossref", "source_id": docs[-1].doi})
    ref = archive.put(alias)
    evidence = quote_evidence(ref, alias, text_field="title", quote=alias.title)
    claim = Claim(claim_id="crossref-claim", role="novelty" if novelty else "summary", text=evidence.quote,
        support="supported", evidence_ids=(evidence.evidence_id,),
        grounding_method="reviewed-novelty/reviewer1" if novelty else "exact-archived-quotation/1.0.0")
    card = TrendCard.model_validate(passport.model_dump(mode="python") | {
        "evidence": (*passport.evidence, evidence), "claims": (*passport.claims, claim)})
    return card, evidence, claim


def test_crossref_alias_removal_revokes_its_quote_but_keeps_the_openalex_work(library):
    card, evidence, claim = crossref_passport(library)
    report = evaluate((*library[:-1], card), excluded_sources=("crossref",))
    assert report.status == "evaluated"
    assert report.baseline.assessment.recent_studies == report.after.assessment.recent_studies == 14
    assert report.change.recent_studies_delta == report.change.smoothed_growth_delta == 0
    assert report.after.inputs.history.observations == report.baseline.inputs.history.observations
    assert evidence.evidence_id in report.removed_evidence_ids
    assert claim.claim_id in report.removed_claim_ids
    assert not report.excluded_study_ids
    assert all(item.source == "openalex" for item in report.after_card.evidence)
    assert report.after_card.candidate == report.baseline_card.candidate == library[3]


def test_removing_reference_catalogue_is_unknown_not_zero_or_a_stale_assessment(library):
    card, evidence, _ = crossref_passport(library)
    report = evaluate((*library[:-1], card), excluded_sources=("openalex",))
    assert report.status == "not_comparable"
    assert report.baseline.assessment.recent_studies == 14
    assert report.after is None and report.after_snapshot is None and report.change is None
    assert report.after_card.assessment_hash is None and report.after_card.historical_snapshot_id is None
    assert report.after_card.category == "insufficient_evidence" and report.after_card.quality == "partial"
    assert report.after_card.evidence == (evidence,)
    assert all(item.source != "openalex" for item in report.after_card.evidence)
    assert any("неизвестны" in item for item in report.limitations)


def test_excluding_a_whole_study_removes_all_source_versions_and_dependent_claims(library):
    archive, docs, _, item, context, passport = library
    # Two source IDs and distinct immutable revisions represent one reference work.
    last = next(doc for doc in docs if doc.document_key == passport.evidence[0].study_id)
    duplicate = DocumentRecord.model_validate(last.model_dump(mode="python") | {"source_id": "W999999"})
    historical = snapshot((*docs, duplicate), archive, purpose="history")
    report = evaluate((archive, docs, historical, item, context, passport), excluded_study_ids=(last.document_key,))
    assert report.baseline.assessment.recent_studies == 14 and report.after.assessment.recent_studies == 13
    assert report.change.recent_studies_delta == -1
    assert all(ref.study_id != last.document_key for ref in report.after_snapshot.documents)
    assert len(report.after_snapshot.documents) == len(historical.documents) - 2
    assert report.removed_claim_ids
    assert not {source.evidence_id for source in report.after_card.evidence} & set(report.removed_evidence_ids)
    assert report.after_snapshot.coverage == historical.coverage
    assert report.after_card.candidate.discovery_study_ids == item.discovery_study_ids


def test_largest_author_team_has_deterministic_identity_and_only_recent_members(library):
    archive, docs, _, item, context, passport = library
    grouped_ids = {doc.document_key for doc in docs[-5:]} | {docs[0].document_key}
    grouped = tuple(DocumentRecord.model_validate(doc.model_dump(mode="python") | {"raw_metadata": {
        "is_retracted": False, "authorships": [{"author": {"id": "https://openalex.org/A1" if doc.document_key in grouped_ids else f"https://openalex.org/A{10000 + index}"},
        "institutions": [{"id": "https://openalex.org/I1" if doc.document_key in grouped_ids else f"https://openalex.org/I{10000 + index}"}]}]}})
        for index, doc in enumerate(docs))
    historical = snapshot(grouped, archive, purpose="history")
    values = (archive, grouped, historical, item, context, passport)
    report = evaluate(values, exclude_largest_verified_group=True)
    assert report.status == "evaluated" and report.selected_group_id is not None
    assert set(report.excluded_study_ids) == {doc.document_key for doc in docs[-5:]}
    assert docs[0].document_key in report.after.inputs.history.observations[0].study_ids
    assert report.baseline.assessment.growth_confirmed and not report.after.assessment.growth_confirmed
    assert report.after.assessment.recent_studies == 9 and report.change.recent_studies_delta == -5
    assert report.after.assessment.baseline_studies == 3
    assert evaluate(values, exclude_largest_verified_group=True) == report
    assert any("более ранних" in item for item in report.limitations)


def test_tied_group_sizes_use_stable_group_id_and_explicit_exclusions_are_unioned(library):
    baseline = evaluate(library, excluded_sources=("crossref",)).baseline
    selected = min(baseline.inputs.independence.groups, key=lambda item: (-len(item.study_ids), item.group_id))
    report = evaluate(library, exclude_largest_verified_group=True, excluded_study_ids=(library[1][0].document_key,))
    assert report.selected_group_id == selected.group_id
    assert set(report.excluded_study_ids) == set(selected.study_ids) | {library[1][0].document_key}
    assert report.change.baseline_studies_delta == -1 and report.change.recent_studies_delta == -1


def test_unknown_author_identities_cannot_become_a_largest_verified_group(library):
    archive, docs, _, item, context, passport = library
    incomplete = DocumentRecord.model_validate(docs[-1].model_dump(mode="python") | {
        "raw_metadata": {"is_retracted": False, "author": "A plausible researcher name"}})
    historical = snapshot((*docs[:-1], incomplete), archive, purpose="history")
    report = evaluate((archive, docs, historical, item, context, passport), exclude_largest_verified_group=True)
    assert report.baseline.inputs.independence is None
    assert report.status == "unavailable"
    assert report.after is report.after_card is report.after_snapshot is report.change is None
    assert not report.excluded_study_ids and not report.removed_claim_ids


def test_partial_reference_does_not_claim_comparable_deltas_or_largest_global_group(library):
    archive, docs, historical, item, context, passport = library
    partial = Coverage.model_validate(historical.coverage[0].model_dump(mode="python") | {
        "state": "partial", "completed_years": (), "comparable": False,
        "pagination_exhausted": False, "limit_reached": True, "reasons": ("document_limit",)})
    changed = snapshot(docs, archive, purpose="history", coverage=(partial,))
    values = (archive, docs, changed, item, context, passport)
    report = evaluate(values, excluded_study_ids=(docs[-1].document_key,))
    assert report.status == "not_comparable" and report.change is None
    assert report.after.assessment.recent_studies == 13 and report.after_card.quality == "partial"
    assert not report.after.inputs.history.coverage.complete_history
    assert evaluate(values, exclude_largest_verified_group=True).status == "unavailable"


def test_explicit_removal_of_every_known_study_can_legitimately_result_in_zero(library):
    report = evaluate(library, excluded_study_ids=tuple(doc.document_key for doc in library[1]))
    assert report.status == "evaluated" and not report.after_snapshot.documents
    assert report.after.assessment.recent_studies == report.after.assessment.baseline_studies == 0
    assert report.after_card.quality == "insufficient_data"
    assert report.change.recent_studies_delta == -14
    assert report.after_card.candidate.discovery_study_ids == library[3].discovery_study_ids


def test_losing_novelty_evidence_revokes_previously_confirmed_category(library):
    card, _, claim = crossref_passport(library, novelty=True)
    novelty = NoveltyAssessment(kind="new_mechanism", claim_id=claim.claim_id,
        earlier_analogues_checked=True, terminology_changes_checked=True)
    archive, _, historical, item, context, _ = library
    report = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=card,
        scenario=SensitivityScenario(excluded_sources=("crossref",)), verified_novelty=novelty)
    assert report.baseline_card.category == "confirmed_trend"
    assert report.after_card.category == "insufficient_evidence" and report.after.inputs.novelty is None
    assert report.change.category_changed
    assert report.after.assessment.growth_confirmed
    assert claim.claim_id in report.removed_claim_ids
    with pytest.raises(EvidenceError, match="новизны"):
        evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=report.baseline_card,
                             scenario=SensitivityScenario(excluded_sources=("crossref",)))


def test_derived_numeric_claims_do_not_survive_with_old_metric_references(library):
    passport = library[-1]
    quoted = passport.claims[0]
    numeric = Claim.model_validate(quoted.model_dump(mode="python") | {
        "claim_id": "computed-count", "kind": "numeric", "metric_refs": ("original.history.recent_studies",)})
    card = TrendCard.model_validate(passport.model_dump(mode="python") | {"claims": (*passport.claims, numeric)})
    report = evaluate((*library[:-1], card), excluded_study_ids=(library[1][-1].document_key,))
    assert numeric.claim_id in report.removed_claim_ids
    assert all(item.kind != "numeric" for item in report.after_card.claims)


def test_sensitivity_is_offline_and_does_not_mutate_inputs_or_the_archive(library, monkeypatch):
    archive, _, historical, item, context, passport = library
    before = (historical.model_dump_json(), item.model_dump_json(), passport.model_dump_json())
    archive_bytes = {str(path): path.read_bytes() for path in archive.directory.rglob("*.json")}
    checkpoint_bytes = json.dumps(context.checkpoints, sort_keys=True)
    progress_events = list(context.progress_events)
    monkeypatch.setattr("httpx.Client.send", lambda *_args, **_kwargs: pytest.fail("Sensitivity must be offline"))
    monkeypatch.setattr("app.pilot.history.collect_snapshot", lambda *_args, **_kwargs: pytest.fail("No new collection"))
    monkeypatch.setattr(archive, "put", lambda *_args, **_kwargs: pytest.fail("No new archive revisions"))
    report = evaluate(library, exclude_largest_verified_group=True)
    assert before == (historical.model_dump_json(), item.model_dump_json(), passport.model_dump_json())
    assert archive_bytes == {str(path): path.read_bytes() for path in archive.directory.rglob("*.json")}
    assert json.dumps(context.checkpoints, sort_keys=True) == checkpoint_bytes
    assert context.progress_events == progress_events
    assert SensitivityReport.model_validate_json(report.model_dump_json()) == report


def test_unknown_study_id_and_invalid_or_duplicated_scenarios_fail_closed(library):
    with pytest.raises(EvidenceError, match="отсутствует"):
        evaluate(library, excluded_study_ids=("doi:unknown",))
    for values in ({}, {"excluded_sources": ("crossref", "crossref")}, {"excluded_sources": ("made-up",)},
                   {"excluded_study_ids": ("x", "x")}, {"exclude_largest_verified_group": 1}):
        with pytest.raises(ValidationError):
            SensitivityScenario(**values)


def test_cancelled_evaluation_stops_before_reading_archive_or_returning_success(library, monkeypatch):
    library[4].cancel_event.set()
    monkeypatch.setattr(library[0], "get", lambda *_: pytest.fail("Cancelled task must not load archive"))
    with pytest.raises(TaskCancelled):
        evaluate(library, excluded_sources=("crossref",))


def test_journal_is_separate_atomic_idempotent_and_detects_corruption(library, tmp_path):
    report = evaluate(library, excluded_sources=("crossref",))
    journal = tmp_path / "sensitivity"
    path = save_sensitivity(report, journal)
    assert path.name == f"sensitivity-{report.report_hash}.json"
    assert load_sensitivity(path) == report
    assert save_sensitivity(report, journal) == path
    assert len(list(journal.iterdir())) == 1
    path.write_bytes(b'{"broken":true}')
    with pytest.raises(EvidenceError, match="повреждена"):
        load_sensitivity(path)
    with pytest.raises(EvidenceError):
        save_sensitivity(report, journal)
    assert path.read_bytes() == b'{"broken":true}'
    assert len(list(journal.iterdir())) == 1


def test_journal_wont_write_legacy_data_or_publish_after_cancellation(library, tmp_path):
    report = evaluate(library, excluded_sources=("crossref",))
    with pytest.raises(ValueError, match="отдельные"):
        save_sensitivity(report, Path.home() / "Library" / "Application Support" / "Trendanalizer" / "sensitivity")
    context = Context()
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TaskCancelled()

    context.check_cancelled = check
    with pytest.raises(TaskCancelled):
        save_sensitivity(report, tmp_path / "journal", context=context)
    assert list((tmp_path / "journal").iterdir()) == []


def test_forged_report_cannot_change_membership_or_reattach_excluded_evidence(library):
    report = evaluate(library, excluded_sources=("openalex",))
    values = report.model_dump(mode="python")
    forged = report.after_card.model_dump(mode="python") | {"candidate": library[3].model_copy(update={"label": "different"})}
    with pytest.raises(ValidationError, match="candidate"):
        SensitivityReport.model_validate(values | {"after_card": forged})
    with pytest.raises(ValidationError):
        SensitivityReport.model_validate(values | {"status": "evaluated"})
    with pytest.raises(ValidationError):
        SensitivityReport.model_validate(values | {"removed_claim_ids": ()})


def test_loading_journal_rejects_unbounded_files_symlinks_and_noncanonical_payload(library, tmp_path, monkeypatch):
    require_symlinks()
    report = evaluate(library, excluded_sources=("crossref",))
    path = save_sensitivity(report, tmp_path / "journal")
    linked = tmp_path / path.name
    linked.symlink_to(path)
    with pytest.raises(EvidenceError, match="имя"):
        load_sensitivity(linked)
    monkeypatch.setattr("app.pilot.sensitivity.MAX_JOURNAL_BYTES", 100)
    with pytest.raises(EvidenceError, match="повреждена"):
        load_sensitivity(path)
    with pytest.raises(EvidenceError, match="размер"):
        save_sensitivity(report, tmp_path / "too-large")
    assert not (tmp_path / "too-large").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["method_version"] == report.method_version


def test_noncanonical_journal_cannot_reuse_a_different_content_identity(library, tmp_path):
    report = evaluate(library, excluded_sources=("crossref",))
    encoded = json.dumps(report.model_dump(mode="json"), indent=2).encode()
    path = tmp_path / ("sensitivity-" + hashlib.sha256(encoded).hexdigest() + ".json")
    path.write_bytes(encoded)
    with pytest.raises(EvidenceError, match="контракту"):
        load_sensitivity(path)


def test_concurrent_journal_appends_publish_exactly_one_complete_entry(library, tmp_path):
    report = evaluate(library, excluded_sources=("crossref",))
    directory = tmp_path / "journal"
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda _: save_sensitivity(report, directory), range(12)))
    assert len(set(paths)) == 1 and list(directory.iterdir()) == [paths[0]]
    assert load_sensitivity(paths[0]) == report


@pytest.mark.parametrize("exclusions", [{"excluded_sources": ("openalex",)},
    {"excluded_sources": ("crossref",)}, {"exclude_largest_verified_group": True}])
def test_archive_replay_verifies_both_assessments_and_scenario(exclusions, library):
    report = evaluate(library, **exclusions)
    verify_sensitivity(report, library[0], library[4])


def test_structurally_consistent_journal_still_requires_archive_replay(library):
    report = evaluate(library, excluded_sources=("crossref",))
    forged_original = report.original_snapshot.model_copy(update={"documents": report.original_snapshot.documents[:-1]})
    forged_after = report.after_snapshot.model_copy(update={"documents": report.after_snapshot.documents[:-1]})
    forged = SensitivityReport.model_validate(report.model_dump(mode="python") | {
        "original_snapshot": forged_original, "after_snapshot": forged_after})
    with pytest.raises(EvidenceError, match="воспроизводится"):
        verify_sensitivity(forged, library[0], library[4])


def test_cancellation_during_archive_reads_stops_sensitivity(library):
    context = library[4]
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 10:
            raise TaskCancelled()

    context.check_cancelled = check
    with pytest.raises(TaskCancelled):
        evaluate(library, exclude_largest_verified_group=True)
    assert calls == 10


@pytest.mark.parametrize("removed", ["doi:10.1234/study81", "doi:10.1234/study82"])
def test_excluding_one_manifestation_removes_its_entire_study_family(library, removed):
    archive, docs, _, item, context, passport = library
    preprint = document(81, year=2021, raw_metadata={"type": "preprint", "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study82"}]}})
    journal = document(82, year=2024)
    historical = snapshot((*docs, preprint, journal), archive, purpose="history")
    report = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=passport,
        scenario=SensitivityScenario(excluded_study_ids=(removed,)))
    assert report.method_version == "frozen-reference-exclusions/2.0.0"
    assert set(report.excluded_study_ids) == {preprint.document_key, journal.document_key}
    assert report.baseline.assessment.baseline_studies == 4
    assert report.baseline.assessment.recent_studies == 14
    assert report.after.assessment.baseline_studies == 3
    assert report.after.assessment.recent_studies == 14
    assert not set(report.excluded_study_ids) & {ref.study_id for ref in report.after_snapshot.documents}
    verify_sensitivity(report, archive, context)


def test_legacy_sensitivity_preserves_old_identity_exclusions_and_archive_replay(library):
    from app.pilot.contracts import Candidate
    from app.pilot.evidence import admission_hash
    archive, docs, _, item, context, passport = library
    values = item.model_dump(mode="python") | {"admission_rule_version": "title-phrase-admission/1.0.0"}
    candidate = Candidate.model_validate(values)
    candidate = Candidate.model_validate(values | {"admission_rule_hash": admission_hash(candidate)})
    passport = TrendCard.model_validate(passport.model_dump(mode="python") | {"candidate": candidate, "methodology_version": None})
    preprint = document(81, year=2021, raw_metadata={"type": "preprint", "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study82"}]}})
    journal = document(82, year=2024)
    historical = snapshot((*docs, preprint, journal), archive, purpose="history")
    report = evaluate_sensitivity(candidate, query_plan(), historical, archive, context, passport=passport,
        scenario=SensitivityScenario(excluded_study_ids=(preprint.document_key,)), methodology_version="3.0.0")
    assert report.method_version == "frozen-reference-exclusions/1.0.0"
    assert report.excluded_study_ids == (preprint.document_key,)
    assert journal.document_key in {ref.study_id for ref in report.after_snapshot.documents}
    assert report.after.assessment.recent_studies == 15
    assert "antecedents" not in report.model_dump(mode="json")
    assert "methodology_version" not in report.baseline_card.model_dump(mode="json")
    before = report.model_dump_json()
    verify_sensitivity(report, archive, context)
    assert report.model_dump_json() == before


def older_bundle(library, documents):
    from app.pilot.antecedents import collect_antecedents
    from app.runtime.credentials import CredentialStore
    from tests.test_pilot_antecedents import Provider
    archive, _, _, item, _, _ = library
    credentials = CredentialStore()
    try:
        return collect_antecedents(item, query_plan(), archive, credentials, Context(),
            provider_factory=lambda _: Provider(documents))
    finally:
        credentials.close()


def test_sensitivity_preserves_unchanged_antecedents_without_a_novelty_claim(library):
    archive, docs, historical, item, context, passport = library
    bundle = older_bundle(library, (document(81, year=2010),))
    report = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=passport,
        scenario=SensitivityScenario(excluded_study_ids=(docs[-1].document_key,)), antecedents=bundle)
    assert report.baseline.inputs.novelty is None
    assert report.baseline.assessment.first_observed_year == report.after.assessment.first_observed_year == 2010
    assert report.after.inputs.history.earlier_search_complete
    verify_sensitivity(report, archive, context)


def test_excluded_earliest_antecedent_cannot_reappear_and_absence_is_not_certified(library):
    archive, _, historical, item, context, passport = library
    earlier = document(81, year=2010)
    later = document(82, year=2012)
    bundle = older_bundle(library, (earlier, later))
    report = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=passport,
        scenario=SensitivityScenario(excluded_study_ids=(earlier.document_key,)), antecedents=bundle)
    assert report.baseline.assessment.first_observed_year == 2010
    assert report.after.assessment.first_observed_year == 2012
    assert not report.after.inputs.history.earlier_search_complete
    assert report.antecedents == bundle  # original source bundle is immutable
    verify_sensitivity(report, archive, context)
    empty = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=passport,
        scenario=SensitivityScenario(excluded_study_ids=(earlier.document_key, later.document_key)), antecedents=bundle)
    assert empty.after.assessment.first_observed_year == 2020
    assert not empty.after.inputs.history.earlier_search_complete
    assert empty.after_card.category == "insufficient_evidence"
    verify_sensitivity(empty, archive, context)


def test_study_family_exclusion_spans_antecedent_and_recent_journal_version(library):
    archive, docs, _, item, context, passport = library
    preprint = document(81, year=2010, raw_metadata={"type": "preprint", "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study82"}]}})
    journal = document(82, year=2024)
    historical = snapshot((*docs, journal), archive, purpose="history")
    bundle = older_bundle(library, (preprint,))
    report = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=passport,
        scenario=SensitivityScenario(excluded_study_ids=(preprint.document_key,)), antecedents=bundle)
    assert set(report.excluded_study_ids) == {preprint.document_key, journal.document_key}
    assert report.after.assessment.first_observed_year == 2020
    assert report.after.assessment.recent_studies == 14
    assert not report.after.inputs.history.earlier_search_complete
    verify_sensitivity(report, archive, context)


def test_changing_reviewed_antecedent_basis_revokes_old_novelty_even_when_its_quote_survives(library):
    card, _, claim = crossref_passport(library, novelty=True)
    archive, _, historical, item, context, _ = library
    older = document(81, year=2010)
    bundle = older_bundle(library, (older,))
    novelty = NoveltyAssessment(kind="new_mechanism", claim_id=claim.claim_id,
        earlier_analogues_checked=True, terminology_changes_checked=True)
    report = evaluate_sensitivity(item, query_plan(), historical, archive, context, passport=card,
        scenario=SensitivityScenario(excluded_study_ids=(older.document_key,)), antecedents=bundle,
        verified_novelty=novelty)
    assert report.baseline.inputs.novelty == novelty
    assert report.after.inputs.novelty is None
    assert claim.claim_id in report.removed_claim_ids
    assert report.after_card.category == "insufficient_evidence"
    verify_sensitivity(report, archive, context)
