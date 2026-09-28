"""Collected-page history, temporal bias, duplicate studies and truthful growth gates."""

import pytest

from app.backend.contracts import SourcePage
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Candidate, Claim, Coverage
from app.pilot.evidence import TITLE_ADMISSION_VERSION, EvidenceError, admission_hash, build_passport
from app.pilot.history import assess_history, assess_snapshot, verify_history_artifact
from app.pilot.methodology import NoveltyAssessment
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskFailure
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot


def frozen(item):
    values = item.model_dump(mode="python") | dict(synonyms=("lithium selective membranes",),
        definition="Lithium-selective membrane separation mechanisms", specificity="specific_technology",
        admission_rule_version=TITLE_ADMISSION_VERSION)
    provisional = Candidate.model_validate(values)
    return Candidate.model_validate(values | {"admission_rule_hash": admission_hash(provisional)})


@pytest.fixture
def scenario(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(1000 + year * 10 + index, year=year) for year, count in zip(
        range(2020, 2026), (1, 1, 1, 2, 4, 8), strict=True) for index in range(count))
    discovery = snapshot(docs[-4:], archive)
    historical = snapshot(docs, archive, purpose="history")
    item = frozen(candidate(discovery))
    context = Context()
    passport = build_passport(item, discovery, archive, context)
    return archive, docs, discovery, historical, item, context, passport


def test_complete_reference_confirms_observed_growth_but_does_not_invent_novelty(scenario):
    archive, _, _, historical, item, context, passport = scenario
    artifact, card, rejected = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
    result = artifact.assessment
    assert result.baseline_studies == 3 and result.recent_studies == 14 and result.smoothed_growth == 3.75
    assert result.growth_confirmed and result.category == "insufficient_evidence"
    assert card.category == "insufficient_evidence" and artifact.inputs.novelty is None
    assert result.first_observed_year == 2020 and not artifact.inputs.history.earlier_search_complete
    assert rejected == 0 and card.assessment_hash == result.assessment_hash
    assert artifact.inputs.independence.coverage_complete


def test_repeated_assessment_replays_result_and_detects_later_archive_damage(scenario):
    archive, _, _, historical, item, context, passport = scenario
    first = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
    assert assess_snapshot(item, query_plan(), historical, archive, context, passport=passport) == first

    path = archive.path(historical.documents[-1].revision_id)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(TaskFailure, match="повреждена"):
        assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)


def test_export_replay_accepts_assessed_card_and_rejects_changed_history(scenario):
    archive, docs, _, historical, item, context, passport = scenario
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context)
    changed = snapshot(docs[:-1], archive, purpose="history")
    with pytest.raises(EvidenceError, match="не воспроизводится"):
        verify_history_artifact(artifact, query_plan(), changed, archive, card, context)


def test_older_records_without_abstracts_have_the_same_temporal_membership(scenario):
    archive, docs, _, historical, item, context, passport = scenario
    originals = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)[0]
    without_abstracts = tuple(type(doc).model_validate(doc.model_dump(mode="python") | {"abstract": None}) for doc in docs)
    changed = snapshot(without_abstracts, archive, purpose="history")
    modified = assess_snapshot(item, query_plan(), changed, archive, context, passport=passport)[0]
    assert modified.assessment.smoothed_growth == originals.assessment.smoothed_growth
    assert modified.inputs.history.observations == originals.inputs.history.observations


def test_abstract_mentions_do_not_admit_irrelevant_titles_and_phrase_boundaries_are_frozen(scenario):
    archive, docs, _, _, item, context, passport = scenario
    unrelated = document(9, title="General battery economics", abstract="lithium selective membranes are cited in passing.")
    data = snapshot((*docs, unrelated), archive, purpose="history")
    artifact, _, rejected = assess_snapshot(item, query_plan(), data, archive, context, passport=passport)
    assert artifact.assessment.recent_studies == 14 and rejected == 1


def test_same_doi_from_multiple_openalex_records_is_one_study(scenario):
    archive, docs, _, _, item, context, passport = scenario
    duplicate = type(docs[-1]).model_validate(docs[-1].model_dump(mode="python") | {"source_id": "W999999"})
    data = snapshot((*docs, duplicate), archive, purpose="history")
    artifact, _, _ = assess_snapshot(item, query_plan(), data, archive, context, passport=passport)
    assert artifact.assessment.recent_studies == 14


def test_disagreeing_publication_years_are_unresolved_not_counted_twice_or_silently_chosen(scenario):
    archive, docs, _, _, item, context, passport = scenario
    conflict = type(docs[-1]).model_validate(docs[-1].model_dump(mode="python") | {"source_id": "W999999", "publication_year": 2024})
    data = snapshot((*docs, conflict), archive, purpose="history")
    artifact, card, _ = assess_snapshot(item, query_plan(), data, archive, context, passport=passport)
    assert not artifact.assessment.growth_confirmed and card.quality == "partial"
    assert artifact.inputs.history.date_conflicts == (docs[-1].document_key,)
    assert artifact.assessment.recent_studies == 13


def test_renamed_title_conflict_and_retracted_versions_do_not_manufacture_growth(scenario):
    archive, docs, _, _, item, context, passport = scenario
    changed_title = type(docs[-1]).model_validate(docs[-1].model_dump(mode="python") | {
        "source_id": "W999999", "title": "An older unrelated terminology"})
    conflicting = snapshot((*docs, changed_title), archive, purpose="history")
    assert not assess_snapshot(item, query_plan(), conflicting, archive, context, passport=passport)[0].assessment.growth_confirmed
    retracted = type(docs[-1]).model_validate(docs[-1].model_dump(mode="python") | {
        "source_id": "W999999", "raw_metadata": {"is_retracted": True}})
    revised = snapshot((*docs, retracted), archive, purpose="history")
    artifact, _, rejected = assess_snapshot(item, query_plan(), revised, archive, context, passport=passport)
    assert artifact.assessment.recent_studies == 13 and rejected == 1


def test_methodology_30_keeps_its_frozen_legacy_status_rule(scenario):
    archive, docs, _, _, item, context, passport = scenario
    old_status_record = document(999999, year=2025,
        title="[Retracted] Lithium selective membranes for extraction experiment",
        raw_metadata={"is_retracted": False})
    historical = snapshot((*docs, old_status_record), archive, purpose="history")

    artifact, _, rejected = assess_snapshot(item, query_plan(), historical, archive, context,
                                           passport=passport, methodology_version="3.0.0")
    assert old_status_record.document_key in artifact.inputs.history.observations[-1].study_ids
    assert rejected == 0


def test_single_completed_search_cannot_hide_capped_or_failed_synonym_query(scenario):
    archive, docs, _, historical, item, context, passport = scenario
    partial = Coverage(source="openalex", purpose="history", query_hash="c" * 64, state="partial",
        requested_years=query_plan().completed_years, pagination_exhausted=False, comparable=False,
        scanned_records=0, accepted_records=0, limit_reached=True, reasons=("document_limit",))
    data = snapshot(docs, archive, purpose="history", coverage=(*historical.coverage, partial))
    artifact, card, _ = assess_snapshot(item, query_plan(), data, archive, context, passport=passport)
    assert not artifact.assessment.growth_confirmed and card.quality == "partial"
    assert artifact.assessment.components[0].value is None


def test_names_without_real_author_and_institution_ids_never_become_independent_groups(scenario):
    archive, docs, _, _, item, context, passport = scenario
    altered = type(docs[-1]).model_validate(docs[-1].model_dump(mode="python") | {
        "raw_metadata": {"is_retracted": False, "authorships": [{"author": {"display_name": "Important Professor"}}]}})
    data = snapshot((*docs[:-1], altered), archive, purpose="history")
    artifact, _, _ = assess_snapshot(item, query_plan(), data, archive, context, passport=passport)
    assert artifact.inputs.independence is None
    assert artifact.assessment.components[3].value is None


def test_shared_institution_does_not_merge_distinct_author_groups_or_prove_replication(scenario):
    archive, docs, _, _, item, context, passport = scenario
    changed = tuple(type(doc).model_validate(doc.model_dump(mode="python") | {"raw_metadata": {
        "is_retracted": False, "authorships": [{"author": {"id": f"https://openalex.org/A{index + 1}"},
                                               "institutions": [{"id": "https://openalex.org/I1"}]}]}})
                    for index, doc in enumerate(docs))
    data = snapshot(changed, archive, purpose="history")
    artifact, _, _ = assess_snapshot(item, query_plan(), data, archive, context, passport=passport)
    assert len(artifact.inputs.independence.groups) == 14
    assert "team-diversity" in artifact.inputs.independence.method_version
    assert artifact.assessment.components[3].value is None


def test_novelty_requires_separate_review_with_traceable_evidence_not_the_word_novel(scenario):
    archive, _, _, historical, item, context, passport = scenario
    novelty = NoveltyAssessment(kind="new_mechanism", claim_id="novelty", earlier_analogues_checked=True,
                                terminology_changes_checked=True)
    with pytest.raises(EvidenceError, match="отдельной"):
        assess_snapshot(item, query_plan(), historical, archive, context, passport=passport, verified_novelty=novelty)
    source = passport.evidence[0]
    review_claim = Claim(claim_id="novelty", role="novelty", text="Reviewed evidence establishes renamed prior mechanism.",
        support="supported", evidence_ids=(source.evidence_id,), grounding_method="reviewed-novelty/reviewer-record-1")
    reviewed_passport = type(passport).model_validate(passport.model_dump(mode="python") | {"claims": (*passport.claims, review_claim)})
    old = NoveltyAssessment(kind="renamed", claim_id="novelty", earlier_analogues_checked=True,
                            terminology_changes_checked=True)
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=reviewed_passport,
                                      verified_novelty=old)
    assert artifact.assessment.category == card.category == "renewed_interest"


def test_supported_claim_cannot_swap_an_exact_quote_for_an_ai_paraphrase(scenario):
    archive, _, _, historical, item, context, passport = scenario
    claim = passport.claims[0]
    altered = type(claim).model_validate(claim.model_dump(mode="python") | {"text": "Manufactured scientific fact"})
    forged = type(passport).model_validate(passport.model_dump(mode="python") | {"claims": (altered, *passport.claims[1:])})
    with pytest.raises(EvidenceError, match="пересказом"):
        assess_snapshot(item, query_plan(), historical, archive, context, passport=forged)


def test_real_collection_path_checkpoint_and_restore_are_end_to_end_reproducible(scenario):
    archive, docs, discovery, _, item, context, passport = scenario
    requests = []

    class Provider:
        def iter_pages(self, request, cancel):
            requests.append(request)
            assert request.from_date.year == 2020 and request.until_date.year == 2025
            assert request.max_results == 3000
            yield SourcePage(documents=docs, scanned=len(docs), total_available=len(docs), exhausted=True)

        def close(self):
            pass

    credentials = CredentialStore()
    try:
        result = assess_history(item, query_plan(), discovery, archive, credentials, context, passport=passport,
                                provider_factory=lambda _: Provider())
        assert result.artifact.assessment.growth_confirmed
        assert result.new_document_count == len(docs) - len(discovery.documents)
        again = assess_history(item, query_plan(), discovery, archive, credentials, context, passport=passport,
                               remaining_new_documents=0, provider_factory=lambda _: pytest.fail("checkpoint must avoid HTTP"))
        assert again == result and len(requests) == 1
    finally:
        credentials.close()


def test_exhausted_new_document_budget_does_not_create_an_empty_complete_history(scenario):
    archive, _, discovery, _, item, context, passport = scenario
    credentials = CredentialStore()
    try:
        with pytest.raises(TaskFailure, match="Исчерпан"):
            assess_history(item, query_plan(), discovery, archive, credentials, context, passport=passport,
                           remaining_new_documents=0, provider_factory=lambda _: pytest.fail("no provider call"))
    finally:
        credentials.close()


def test_small_remaining_budget_caps_real_collection_and_preserves_partial_history(scenario):
    archive, docs, discovery, _, item, context, passport = scenario

    class Provider:
        def iter_pages(self, request, cancel):
            assert request.max_results == 2
            yield SourcePage(documents=docs[:2], scanned=2, total_available=len(docs), exhausted=False)

        def close(self):
            pass

    credentials = CredentialStore()
    try:
        result = assess_history(item, query_plan(), discovery, archive, credentials, context, passport=passport,
            remaining_new_documents=2, provider_factory=lambda _: Provider())
        assert result.new_document_count == 2
        assert not result.artifact.assessment.growth_confirmed
        assert result.artifact.inputs.history.coverage.limit_reached
        assert result.artifact.inputs.history.coverage.state == "partial"
    finally:
        credentials.close()


def test_verified_research_application_is_not_permanently_unknown(scenario):
    archive, _, _, historical, item, context, passport = scenario
    artifact, _, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
    assert artifact.inputs.application is not None
    assert artifact.inputs.application.kind == "research"
    assert artifact.assessment.components[4].value == 50
    assert artifact.assessment.priority_score is None  # novelty and replication remain unknown


def test_application_stage_cannot_be_guessed_from_a_case_title(scenario):
    from app.pilot.methodology import ApplicationAssessment
    archive, _, _, historical, item, context, passport = scenario
    with pytest.raises(EvidenceError, match="Стадия применения"):
        assess_snapshot(item, query_plan(), historical, archive, context, passport=passport,
            verified_application=ApplicationAssessment(kind="deployment", claim_id=passport.claims[0].claim_id))


def test_preprint_and_journal_are_counted_once_at_the_actual_earlier_identity(scenario):
    archive, docs, _, _, item, context, passport = scenario
    preprint = document(81, year=2021, raw_metadata={"type": "preprint", "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study82"}]}})
    journal = document(82, year=2024)
    historical = snapshot((*docs, preprint, journal), archive, purpose="history")
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
    assert artifact.assessment.baseline_studies == 4 and artifact.assessment.recent_studies == 14
    assert preprint.document_key in artifact.inputs.history.observations[1].study_ids
    assert journal.document_key not in artifact.inputs.history.observations[4].study_ids
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context)


def test_a_linked_clean_version_cannot_hide_retraction(scenario):
    archive, docs, _, _, item, context, passport = scenario
    preprint = document(81, year=2021, raw_metadata={"is_retracted": True, "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study82"}]}})
    journal = document(82, year=2024)
    historical = snapshot((*docs, preprint, journal), archive, purpose="history")
    artifact, _, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
    assert artifact.assessment.baseline_studies == 3 and artifact.assessment.recent_studies == 14


def test_older_verified_bundle_integrates_first_observation_without_inventing_novelty(scenario):
    from tests.test_pilot_antecedents import bundle_for
    archive, _, _, historical, item, context, passport = scenario
    bundle, _ = bundle_for(scenario, (document(81, year=2009),))
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport, antecedents=bundle)
    assert artifact.inputs.history.first_observed_year == 2009
    assert artifact.inputs.history.first_observed_study_id == "doi:10.1234/study81"
    assert artifact.inputs.history.earlier_search_complete
    assert artifact.inputs.novelty is None and card.category == "insufficient_evidence"
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context, antecedents=bundle)
    with pytest.raises(EvidenceError, match="не воспроизводится"):
        verify_history_artifact(artifact, query_plan(), historical, archive, card, context)


def test_legacy_methodology_replay_keeps_old_admission_and_labels(scenario):
    archive, _, _, historical, item, context, passport = scenario
    values = item.model_dump(mode="python") | {"admission_rule_version": "title-phrase-admission/1.0.0"}
    provisional = Candidate.model_validate(values)
    old = Candidate.model_validate(values | {"admission_rule_hash": admission_hash(provisional)})
    old_passport = type(passport).model_validate(passport.model_dump(mode="python") | {"candidate": old})
    artifact, card, _ = assess_snapshot(old, query_plan(), historical, archive, context, passport=old_passport,
                                      methodology_version="3.0.0")
    assert artifact.assessment.methodology_version == "3.0.0" and card.methodology_version is None
    assert card.category == "early_signal" and artifact.inputs.application is None
    assert artifact.inputs.independence.method_version == "openalex-author-institution-components/1.0.0"
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context)


def test_known_earlier_preprint_prevents_later_journal_from_counting_as_new_history(scenario):
    from tests.test_pilot_antecedents import bundle_for
    archive, docs, _, _, item, context, passport = scenario
    preprint = document(81, year=2010, raw_metadata={"type": "preprint", "relation": {
        "is-preprint-of": [{"id-type": "doi", "id": "10.1234/study82"}]}})
    journal = document(82, year=2024)
    historical = snapshot((*docs, journal), archive, purpose="history")
    bundle, _ = bundle_for(scenario, (preprint,))
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=passport, antecedents=bundle)
    assert artifact.assessment.first_observed_year == 2010
    assert artifact.assessment.baseline_studies == 3 and artifact.assessment.recent_studies == 14
    assert journal.document_key not in {study for year in artifact.inputs.history.observations for study in year.study_ids}
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context, antecedents=bundle)
    legacy, old_card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=passport, antecedents=bundle, methodology_version="3.0.0")
    assert legacy.assessment.recent_studies == 15 and legacy.assessment.first_observed_year == 2020
    verify_history_artifact(legacy, query_plan(), historical, archive, old_card, context, antecedents=bundle)


def test_conflicting_same_doi_dates_across_antecedent_and_history_are_unresolved(scenario):
    from tests.test_pilot_antecedents import bundle_for
    archive, docs, _, _, item, context, passport = scenario
    old = document(81, year=2010)
    redated = document(81, year=2024)
    historical = snapshot((*docs, redated), archive, purpose="history")
    bundle, _ = bundle_for(scenario, (old,))
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context,
        passport=passport, antecedents=bundle)
    assert artifact.inputs.history.date_conflicts == (old.document_key,)
    assert artifact.assessment.recent_studies == 14
    assert not artifact.assessment.growth_confirmed and card.quality == "partial"
