"""Cross-component acceptance for versioned publication and quotation corrections."""

import json

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, Claim, content_hash
from app.pilot.evidence import build_passport, quote_evidence
from app.pilot.export import export_result, read_result_package, verify_result
from app.pilot.grounding import verify_supported_source_claim
from app.pilot.history import assess_history, assess_snapshot, verify_history_artifact
from app.pilot.methodology import AssessmentArtifact, evaluate_candidate
from app.pilot.selection import select_top
from app.pilot.sensitivity import SensitivityScenario, evaluate_sensitivity, verify_sensitivity
from app.pilot.signal_evidence import extract_primary_observations
from app.runtime.backup import ArchiveError
from app.runtime.credentials import CredentialStore
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot
from tests.test_pilot_history import frozen
from tests.test_pilot_methodology_v3_2 import scenario32


RESULT = "We measured lithium selective membranes in laboratory experiments and improved selectivity by 43.94 percent."
ABSTRACT = "Existing extraction methods suffer from limited selectivity. " + RESULT + " Future work may improve device lifetime."


def prepared34(tmp_path, *, abstract=ABSTRACT, year=2026):
    archive = DocumentArchive(tmp_path / "revisions")
    doc = document(404, year=year, abstract=abstract)
    discovery = snapshot((doc,), archive)
    item, context = frozen(candidate(discovery)), Context()
    passport = build_passport(item, discovery, archive, context, methodology_version="3.4.0")
    primary = extract_primary_observations(item, discovery, archive, context, query_plan=query_plan(),
                                           method_version="archived-primary-result/2.0.0")
    evidence = {entry.evidence_id: entry for entry in passport.evidence}
    claims = list(passport.claims)
    for entry in primary:
        evidence[entry.result.evidence_id] = entry.result
        if not any(claim.role == "case" and entry.result.evidence_id in claim.evidence_ids for claim in claims):
            claims.append(Claim(claim_id="primary-" + content_hash(entry)[:24], role="case", support="supported",
                grounding_method="exact-contextual-quotation/4.0.0", text=entry.result.quote,
                evidence_ids=(entry.result.evidence_id,)))
    passport = passport.model_copy(update={"claims": tuple(claims), "evidence": tuple(evidence.values())})
    checked = assess_history(item, query_plan(), discovery, archive, CredentialStore(), context,
        passport=passport, remaining_new_documents=0, methodology_version="3.4.0", primary_observations=primary)
    result = AnalysisResult(result_id="result34", run_id=context.run_id, query_plan=query_plan(),
        methodology_version="3.4.0", created_at=discovery.created_at, quality=checked.card.quality,
        snapshots=(discovery, checked.snapshot), cards=(checked.card,),
        top_limit=15, top_trend_ids=select_top((checked.card,), (checked.artifact,)),
        limitations=("История не собрана; это наблюдение, требующее проверки.",))
    return archive, doc, discovery, context, checked, result


def test_primary_application_with_separate_future_qualification_exports_and_reopens(tmp_path):
    archive, doc, discovery, context, checked, result = prepared34(tmp_path)
    assert checked.card.category == "weak_signal_candidate"
    assert checked.artifact.inputs.application.kind == "research"
    assert not checked.artifact.assessment.growth_confirmed
    assert checked.artifact.assessment.recent_studies == 0
    assert checked.artifact.assessment.signal_priority == 10
    advantage = next(claim for claim in checked.card.claims if claim.role == "advantage")
    application = next(claim for claim in checked.card.claims if claim.role == "application")
    assert advantage.text == application.text == RESULT
    assert application.grounding_method == "verified-application/research/2.0.0"
    source = next(entry for entry in checked.card.evidence if entry.evidence_id == advantage.evidence_ids[0])
    assert doc.abstract[source.start:source.end] == RESULT
    verify_history_artifact(checked.artifact, query_plan(), checked.snapshot, archive, checked.card,
                            context, methodology_version="3.4.0")
    target = tmp_path / "actual.trendresult"
    export_result(target, result, archive, (checked.artifact,))
    with read_result_package(target) as reopened:
        assert reopened.result == result
        assert reopened.assessments == (checked.artifact,)
        assert reopened.archive.get(discovery.documents[0].revision_id) == doc


def test_legacy_application_screen_remains_frozen_while_new_application_uses_its_own_rules(tmp_path):
    archive = DocumentArchive(tmp_path)
    text = "We measured lithium selective membranes in laboratory experiments and improved selectivity."
    doc = document(5, abstract=text + " Future work may improve device lifetime.")
    reference = archive.put(doc)
    evidence = quote_evidence(reference, doc, text_field="abstract", quote=text)
    claim = Claim(claim_id="research", role="application", support="supported", text=text,
        evidence_ids=(evidence.evidence_id,), grounding_method="verified-application/research")
    with pytest.raises(ValueError, match="Source quotation/context"):
        verify_supported_source_claim(claim, {evidence.evidence_id: evidence}, {reference.revision_id: doc},
                                       legacy=False, candidate_studies={doc.document_key})
    modern = claim.model_copy(update={"grounding_method": "verified-application/research/2.0.0"})
    verify_supported_source_claim(modern, {evidence.evidence_id: evidence}, {reference.revision_id: doc},
                                   legacy=False, candidate_studies={doc.document_key})


def test_primary_evidence_cannot_be_removed_by_recomputing_a_lower_category(tmp_path):
    archive, _, _, context, checked, result = prepared34(tmp_path)
    changed, card, _ = assess_snapshot(checked.card.candidate, query_plan(), checked.snapshot, archive,
        context, passport=checked.card, methodology_version="3.4.0", primary_observations=())
    assert changed.assessment.category == "insufficient_evidence"
    altered = result.model_copy(update={"cards": (card,), "quality": card.quality, "top_trend_ids": ()})
    with pytest.raises(ArchiveError, match="воспроизвод"):
        verify_result(altered, archive, (changed,))


def test_primary_method_cannot_be_downgraded_in_a_new_assessment(tmp_path):
    _, _, _, _, checked, _ = prepared34(tmp_path)
    old = checked.artifact.inputs.primary_observations[0].model_copy(update={"method_version": "archived-primary-result/1.0.0"})
    invalid = checked.artifact.inputs.model_copy(update={"primary_observations": (old,)})
    with pytest.raises(ValueError, match="extraction version"):
        evaluate_candidate(invalid, version="3.4.0")
    corrupt = json.loads(checked.artifact.model_dump_json())
    corrupt["assessment"]["signal_priority"] = 99
    with pytest.raises(ValueError, match="reproduce"):
        AssessmentArtifact.model_validate(corrupt)


def test_new_result_cannot_export_new_grounding_under_older_container(tmp_path):
    archive, _, _, _, checked, result = prepared34(tmp_path)
    downgraded = result.model_copy(update={"methodology_version": "3.3.0"})
    with pytest.raises(ArchiveError, match="контейнера"):
        verify_result(downgraded, archive, (checked.artifact,))


def test_portable_replay_checks_withdrawal_notice_in_another_snapshot_revision(tmp_path):
    archive, doc, discovery, _, checked, result = prepared34(tmp_path)
    notice = document(405, year=2026, raw_metadata={"update-to": [{"type": "retraction", "DOI": doc.doi}]})
    expanded = snapshot((doc, notice), archive, snapshot_id=discovery.snapshot_id)
    altered = result.model_copy(update={"snapshots": (expanded, checked.snapshot)})
    with pytest.raises(ArchiveError, match="отозванной работе"):
        verify_result(altered, archive, (checked.artifact,))


def test_excluding_primary_source_revokes_observation_application_and_advantage(tmp_path):
    archive, doc, _, context, checked, _ = prepared34(tmp_path)
    report = evaluate_sensitivity(checked.card.candidate, query_plan(), checked.snapshot, archive, context,
        passport=checked.card, methodology_version="3.4.0",
        primary_observations=checked.artifact.inputs.primary_observations,
        scenario=SensitivityScenario(excluded_study_ids=(doc.document_key,)))
    assert report.after.inputs.primary_observations == ()
    assert report.after.inputs.application is None
    assert report.after.assessment.category == "insufficient_evidence"
    assert not any(claim.support == "supported" for claim in report.after_card.claims)
    verify_sensitivity(report, archive, context)


def test_notice_metadata_excludes_target_from_new_history_without_changing_older_rules(tmp_path):
    archive, _, _, context, checked, _ = prepared34(tmp_path)
    target = document(10, year=2025)
    notice = document(11, year=2025, raw_metadata={"update-to": [{"type": "retraction", "DOI": target.doi}]})
    discussion = document(12, year=2025, abstract="We analyze notices saying that a different paper was retracted.")
    historical = snapshot((target, notice, discussion), archive, purpose="history")
    artifact, _, rejected = assess_snapshot(checked.card.candidate, query_plan(), historical, archive, context,
        passport=checked.card, methodology_version="3.4.0",
        primary_observations=checked.artifact.inputs.primary_observations)
    assert artifact.assessment.recent_studies == 1
    assert artifact.inputs.history.observations[-1].study_ids == (discussion.document_key,)
    assert rejected == 2
    legacy_passport = checked.card.model_copy(update={"claims": (), "evidence": ()})
    old, _, _ = assess_snapshot(checked.card.candidate, query_plan(), historical, archive, context,
        passport=legacy_passport, methodology_version="3.2.0")
    assert old.assessment.recent_studies == 3


@pytest.mark.parametrize("counts", [(0, 0, 0, 0, 1, 2), (0, 0, 0, 2, 4, 8), (1, 1, 1, 4, 8, 16)])
def test_no_formula_drift_without_new_evidence_inputs(tmp_path, counts):
    previous = scenario32(tmp_path, counts)[0]
    # Source novelty is not supplied in this comparison; its new extraction has
    # separate checks, while the arithmetic must be identical on identical input.
    inputs = previous.inputs.model_copy(update={"source_novelty": ()})
    old = evaluate_candidate(inputs, version="3.3.0")
    modern = evaluate_candidate(inputs, version="3.4.0")
    assert modern.model_dump(exclude={"methodology_version"}) == old.model_dump(exclude={"methodology_version"})
