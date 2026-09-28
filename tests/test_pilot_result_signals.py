"""Automatic signal provenance survives exports, manual review and exclusions."""

import pytest

from app.pilot.antecedents import collect_antecedents
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, Claim, content_hash
from app.pilot.evidence import build_passport
from app.pilot.export import export_result, read_result_package, verify_result
from app.pilot.history import assess_snapshot
from app.pilot.methodology import AssessmentArtifact, evaluate_candidate
from app.pilot.review import apply_novelty_review, record_review, verify_review_record
from app.pilot.selection import select_top
from app.pilot.sensitivity import SensitivityScenario, evaluate_sensitivity, verify_sensitivity
from app.pilot.signal_evidence import extract_signal_evidence
from app.runtime.backup import ArchiveError
from app.runtime.credentials import CredentialStore
from tests.test_pilot_antecedents import Provider
from tests.test_pilot_evidence import Context, NOW, candidate, document, query_plan, snapshot
from tests.test_pilot_field_history import field_exposure
from tests.test_pilot_history import frozen
from tests.test_pilot_review import decision_for
from tests.test_pilot_signal_evidence import NOVELTY, EXPERIMENT


def automatic_result(tmp_path):
    archive = DocumentArchive(tmp_path / "revisions")
    docs = tuple(document(year * 10 + index, year=year) for year, count in zip(
        range(2020, 2026), (0, 0, 0, 1, 2, 4), strict=True) for index in range(count))
    docs = (*docs[:-1], docs[-1].model_copy(update={"abstract":
        "Existing extraction methods suffer from limited selectivity. " + NOVELTY + " " + EXPERIMENT +
        " Our lithium selective membranes reduce energy consumption in laboratory experiments."}))
    discovery = snapshot(docs, archive)
    item = frozen(candidate(discovery))
    context = Context()
    passport = build_passport(item, discovery, archive, context)
    sources = extract_signal_evidence(item, discovery, archive, context, query_plan=query_plan())
    assert len(sources) == 1
    source = sources[0]
    claim = Claim(claim_id="source-novelty-" + content_hash(source)[:24], role="novelty", support="unverified",
        grounding_method=source.method_version, text=source.novelty.quote,
        evidence_ids=tuple(dict.fromkeys((source.novelty.evidence_id, source.experiment.evidence_id))))
    passport = passport.model_copy(update={"methodology_version": "3.2.0", "claims": (*passport.claims, claim),
        "evidence": tuple({item.evidence_id: item for item in
            (*passport.evidence, source.novelty, source.experiment)}.values())})
    historical = snapshot(docs, archive, purpose="history")
    bundle = collect_antecedents(item, query_plan(), archive, CredentialStore(), Context(),
        provider_factory=lambda _: Provider())
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport,
        methodology_version="3.2.0", antecedents=bundle, source_novelty=sources, field_exposure=field_exposure())
    result = AnalysisResult(result_id="automatic-signal", run_id="automatic-run", query_plan=query_plan(),
        methodology_version="3.2.0", created_at=NOW, quality=card.quality,
        snapshots=(discovery, historical, bundle.snapshot), cards=(card,),
        top_limit=15, top_trend_ids=select_top((card,), (artifact,)))
    assert card.category == "weak_signal_candidate"
    assert result.top_trend_ids == (item.candidate_id,)
    return result, archive, artifact


def test_automatic_signal_portable_roundtrip_replays_evidence_and_selection(tmp_path, monkeypatch):
    result, archive, artifact = automatic_result(tmp_path)
    import socket

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("Replay must be offline"))
    path = tmp_path / "signal.trendresult"
    export_result(path, result, archive, (artifact,))
    with read_result_package(path) as package:
        assert package.result == result
        assert package.assessments == (artifact,)
        assert package.result.cards[0].category == "weak_signal_candidate"
        assert package.assessments[0].inputs.novelty is None


def test_recomputed_top_cannot_hide_an_eligible_signal(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    with pytest.raises(ArchiveError):
        verify_result(result.model_copy(update={"top_trend_ids": ()}), archive, (artifact,))


def test_result_version_cannot_be_downgraded_to_bypass_signal_selection_replay(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    with pytest.raises(ArchiveError):
        verify_result(result.model_copy(update={"methodology_version": "3.1.0", "top_trend_ids": ()}), archive, (artifact,))


def test_automatic_prior_search_must_be_part_of_the_portable_archive(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    with pytest.raises(ArchiveError):
        verify_result(result.model_copy(update={"snapshots": result.snapshots[:2]}), archive, (artifact,))


def test_deleting_novelty_inputs_and_rehashing_does_not_hide_archived_assertion(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    claims = tuple(item for item in artifact.inputs.claims if item.role != "novelty")
    inputs = artifact.inputs.model_copy(update={"source_novelty": (), "claims": claims})
    changed = AssessmentArtifact(inputs=inputs, assessment=evaluate_candidate(inputs, version="3.2.0"))
    card = result.cards[0].model_copy(update={"claims": claims, "category": changed.assessment.category,
        "assessment_hash": changed.assessment.assessment_hash})
    forged = result.model_copy(update={"cards": (card,), "top_trend_ids": ()})
    with pytest.raises(ArchiveError):
        verify_result(forged, archive, (changed,))


def test_author_claim_cannot_be_upgraded_to_supported_or_rephrased(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    for change in ({"support": "supported"}, {"text": "Независимо подтверждена мировая новизна."}):
        claims = tuple(item.model_copy(update=change) if item.role == "novelty" else item for item in result.cards[0].claims)
        inputs = artifact.inputs.model_copy(update={"claims": claims})
        changed = AssessmentArtifact(inputs=inputs, assessment=evaluate_candidate(inputs, version="3.2.0"))
        card = result.cards[0].model_copy(update={"claims": claims, "assessment_hash": changed.assessment.assessment_hash})
        with pytest.raises(ArchiveError):
            verify_result(result.model_copy(update={"cards": (card,)}), archive, (changed,))


def test_manual_review_preserves_automatic_inputs_but_replaces_author_novelty_claim(tmp_path):
    result, archive, original = automatic_result(tmp_path)
    source_card = result.cards[0]
    bundle = original.inputs.antecedents
    decision = decision_for(bundle, source_card)
    expanded, novelty = apply_novelty_review(decision, bundle, source_card, archive, methodology_version="3.2.0")
    artifact, card, _ = assess_snapshot(source_card.candidate, result.query_plan, result.snapshots[1], archive,
        Context(), passport=expanded, verified_novelty=novelty, antecedents=bundle, methodology_version="3.2.0",
        field_exposure=original.inputs.field_exposure, source_novelty=original.inputs.source_novelty)
    assert card.category == "early_signal"
    record = record_review(tmp_path / "reviews", decision, bundle, source_card=source_card, reviewed_card=card,
        artifact=artifact, historical_snapshot=result.snapshots[1], archive=archive)
    assert verify_review_record(record, archive) is None
    reviewed = result.model_copy(update={"cards": (card,), "top_trend_ids": select_top((card,), (artifact,))})
    verify_result(reviewed, archive, (artifact,), reviews=(record,))
    export_result(tmp_path / "reviewed.trendresult", reviewed, archive, (artifact,), reviews=(record,))
    with read_result_package(tmp_path / "reviewed.trendresult") as package:
        assert package.assessments[0].inputs.source_novelty == original.inputs.source_novelty
        assert package.reviews == (record,)


def test_excluded_novelty_study_revokes_hypothesis_without_changing_field_exposure(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    report = evaluate_sensitivity(result.cards[0].candidate, result.query_plan, result.snapshots[1], archive, Context(),
        passport=result.cards[0], methodology_version="3.2.0", antecedents=artifact.inputs.antecedents,
        source_novelty=artifact.inputs.source_novelty, field_exposure=artifact.inputs.field_exposure,
        scenario=SensitivityScenario(excluded_study_ids=(artifact.inputs.source_novelty[0].novelty.study_id,)))
    assert report.baseline.assessment.category == "weak_signal_candidate"
    assert report.after.assessment.category == "insufficient_evidence"
    assert report.after.inputs.source_novelty == ()
    assert report.after.inputs.field_exposure == report.baseline.inputs.field_exposure
    assert any("автоматическая гипотеза отозвана" in item for item in report.limitations)
    verify_sensitivity(report, archive, Context())


def test_removing_historical_catalogue_does_not_leave_a_stale_automatic_signal(tmp_path):
    result, archive, artifact = automatic_result(tmp_path)
    report = evaluate_sensitivity(result.cards[0].candidate, result.query_plan, result.snapshots[1], archive, Context(),
        passport=result.cards[0], methodology_version="3.2.0", antecedents=artifact.inputs.antecedents,
        source_novelty=artifact.inputs.source_novelty, field_exposure=artifact.inputs.field_exposure,
        scenario=SensitivityScenario(excluded_sources=("openalex",)))
    assert report.after is None and report.after_card.assessment_hash is None
    assert report.after_card.category == "insufficient_evidence"
    assert not report.after_card.evidence
    verify_sensitivity(report, archive, Context())
