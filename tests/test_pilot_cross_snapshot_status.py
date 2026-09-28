"""Publication metadata anywhere in a current result cannot be ignored by its cards."""
from dataclasses import dataclass

import pytest

from app.pilot.contracts import AnalysisResult, content_hash
from app.pilot.antecedents import collect_antecedents
from app.pilot.evidence import EvidenceError
from app.pilot.export import export_result, read_result_package, verify_result
from app.pilot.history import assess_snapshot
from app.pilot.methodology import AssessmentArtifact
from app.pilot.selection import select_top
from app.pilot.service import _reconcile_publication_status
from app.runtime.backup import ArchiveError
from app.runtime.credentials import CredentialStore
from tests.test_pilot_antecedents import Provider
from tests.test_pilot_evidence import document, snapshot
from tests.test_pilot_methodology_v3_4 import prepared34
from tests.test_pilot_result_signals import automatic_result


@dataclass(frozen=True)
class StatusCase:
    name: str

    def record(self, source):
        if self.name == "publisher_retraction_notice":
            return document(999111, year=2025, title="Retraction notice for lithium membrane experiment",
                raw_metadata={"update-to": [{"type": "retraction", "DOI": source.doi}]})
        if self.name == "explicit_withdrawal_revision":
            return source.model_copy(update={"raw_metadata": {"is_retracted": True}})
        if self.name == "supplement_parent_relation":
            return document(999112, year=2025, raw_metadata={"relation": {
                "is-supplemented-by": [{"id-type": "doi", "id": source.doi}]}})
        return source.model_copy(update={"document_type": "dataset", "raw_metadata": {"type": "dataset"}})


STATUS_CASES = tuple(StatusCase(name) for name in (
    "publisher_retraction_notice", "explicit_withdrawal_revision", "supplement_parent_relation", "dataset_revision"))


@pytest.mark.parametrize("case", STATUS_CASES, ids=lambda item: item.name)
def test_portable_result_rejects_primary_claim_when_another_history_snapshot_contains_status(tmp_path, case):
    archive, source, discovery, _, checked, result = prepared34(tmp_path, year=2025)
    assert verify_result(result, archive, (checked.artifact,))[0] == result
    status_history = snapshot((case.record(source),), archive, purpose="history")
    altered = AnalysisResult.model_validate(result.model_dump(mode="python") | dict(
        snapshots=(*result.snapshots, status_history)))
    assert status_history.snapshot_id != discovery.snapshot_id
    with pytest.raises(ArchiveError):
        verify_result(altered, archive, (checked.artifact,))


@pytest.mark.parametrize("case", STATUS_CASES, ids=lambda item: item.name)
def test_history_cannot_keep_primary_and_application_after_learning_publication_status(tmp_path, case):
    archive, source, _, context, checked, result = prepared34(tmp_path, year=2025)
    history = snapshot((case.record(source),), archive, purpose="history")
    try:
        artifact, card, _ = assess_snapshot(checked.card.candidate, result.query_plan, history, archive,
            context, passport=checked.card, methodology_version="3.4.0",
            primary_observations=checked.artifact.inputs.primary_observations)
    except EvidenceError:
        return  # Explicit rejection is also safe; no positive result is published.
    assert artifact.inputs.primary_observations == ()
    assert artifact.inputs.application is None
    assert select_top((card,), (artifact,)) == ()
    assert not any(claim.role in {"advantage", "application", "case"} and claim.support == "supported"
                   for claim in card.claims)


def test_old_card_keeps_exact_historical_replay_in_new_container_with_newer_status_context(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    encoded, assessment_hash = result.cards[0].model_dump_json(), content_hash(artifact)
    source = archive.get(result.snapshots[0].documents[0].revision_id)
    status = snapshot((STATUS_CASES[0].record(source),), archive, purpose="history")
    mixed = AnalysisResult.model_validate(result.model_dump(mode="python") | dict(
        methodology_version="3.4.0", snapshots=(*result.snapshots, status)))
    verified, artifacts = verify_result(mixed, archive, (artifact,))
    assert verified.cards[0].model_dump_json() == encoded
    assert content_hash(artifacts[0]) == assessment_hash


@pytest.mark.parametrize("case", STATUS_CASES, ids=lambda item: item.name)
def test_late_sibling_status_is_reconciled_before_export_and_reopen(tmp_path, case):
    archive, source, _, context, checked, result = prepared34(tmp_path, year=2025)
    late = snapshot((case.record(source),), archive, purpose="history")
    cards = [checked.card]
    artifacts = [checked.artifact.model_dump(mode="json")]
    revised = _reconcile_publication_status(cards, artifacts, (*result.snapshots, late),
                                            result.query_plan, archive, context)
    artifact = AssessmentArtifact.model_validate(revised[0])
    assert artifact.inputs.primary_observations == ()
    assert artifact.inputs.application is None
    assert select_top(cards, (artifact,)) == ()
    assert not any(claim.support == "supported" and claim.role in {"case", "advantage", "application"}
                   for claim in cards[0].claims)
    portable = AnalysisResult.model_validate(result.model_dump(mode="python") | dict(
        snapshots=(*result.snapshots, late), cards=tuple(cards), quality=cards[0].quality,
        top_trend_ids=()))
    path = tmp_path / "reconciled.trendresult"
    export_result(path, portable, archive, (artifact,))
    with read_result_package(path) as opened:
        assert opened.result == portable
        assert opened.assessments == (artifact,)


def test_earliest_withdrawn_antecedent_uses_next_eligible_source_without_mutating_bundle(tmp_path):
    archive, _, _, context, checked, result = prepared34(tmp_path, year=2025)
    earliest, next_old = document(9001, year=2000), document(9002, year=2010)
    bundle = collect_antecedents(checked.card.candidate, result.query_plan, archive, CredentialStore(),
                                 context, provider_factory=lambda _: Provider((earliest, next_old)))
    original_bundle_hash = content_hash(bundle)
    assert bundle.earliest_observed_study_id == earliest.document_key
    notice = snapshot((earliest.model_copy(update={"raw_metadata": {"is_retracted": True}}),),
                      archive, purpose="history")
    artifact, _, _ = assess_snapshot(checked.card.candidate, result.query_plan, checked.snapshot,
        archive, context, passport=checked.card, methodology_version="3.4.0", antecedents=bundle,
        primary_observations=checked.artifact.inputs.primary_observations,
        publication_status_revisions=notice.documents)
    assert artifact.inputs.history.first_observed_study_id == next_old.document_key
    assert artifact.inputs.history.first_observed_year == 2010
    assert artifact.inputs.antecedents == bundle
    assert content_hash(bundle) == original_bundle_hash


def test_final_assembly_reports_its_own_counters_for_reconciliation_and_verification(tmp_path):
    archive, _, _, context, checked, result = prepared34(tmp_path, year=2025)
    cards = [checked.card]
    artifacts = [checked.artifact.model_dump(mode="json")]
    context.progress_events.clear()
    _reconcile_publication_status(cards, artifacts, result.snapshots, result.query_plan, archive, context)
    assert [event for event in context.progress_events if event[0] == "reconcile"] == [
        ("reconcile", "Сверяем статусы публикаций: 1 из 1", 0, 1),
        ("reconcile", "Статусы публикаций сверены", 1, 1)]
    context.progress_events.clear()
    verify_result(result, archive, (checked.artifact,), context=context)
    assert [event for event in context.progress_events if event[0] == "verify"] == [
        ("verify", "Проверяем доказательства карточек: 1 из 1", 0, 1),
        ("verify", "Доказательства карточек проверены", 1, 1)]


def test_verification_outside_a_run_has_nowhere_to_report_and_still_succeeds(tmp_path):
    archive, _, _, _, checked, result = prepared34(tmp_path, year=2025)
    assert verify_result(result, archive, (checked.artifact,))[0] == result
