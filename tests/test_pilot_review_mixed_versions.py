"""Synthetic manual-review regression; no real independent expert labels."""

import pytest

from app.pilot.contracts import AnalysisResult, CandidateReviewState, Claim, TrendCard, content_hash
from app.pilot.evidence import build_passport
from app.pilot.export import export_result, read_result_package
from app.pilot.history import assess_snapshot
from app.pilot.methodology import AssessmentArtifact
from app.pilot.selection import select_top
from app.pilot.signal_evidence import extract_primary_observations, extract_signal_evidence
from tests.test_pilot_evidence import Context
from tests.test_pilot_result_signals import automatic_result
from tests.test_pilot_review import decision_for
from tests.test_pilot_service_signals import runtime_for


def mixed_result(tmp_path, version):
    result, archive, artifact = automatic_result(tmp_path)
    if version == "3.3.0":
        discovery = result.snapshots[0]
        item = result.cards[0].candidate
        context = Context()
        passport = build_passport(item, discovery, archive, context, methodology_version=version)
        sources = extract_signal_evidence(item, discovery, archive, context, query_plan=result.query_plan,
                                          method_version="archived-author-novelty/2.0.0")
        primary = extract_primary_observations(item, discovery, archive, context, query_plan=result.query_plan,
                                               method_version="archived-primary-result/1.0.0")
        evidence = {entry.evidence_id: entry for entry in passport.evidence}
        claims = list(passport.claims)
        for source in sources:
            evidence.update((entry.evidence_id, entry) for entry in (source.novelty, source.experiment))
            claims.append(Claim(claim_id="source-novelty-" + content_hash(source)[:24], role="novelty",
                support="unverified", grounding_method=source.method_version, text=source.novelty.quote,
                evidence_ids=tuple(dict.fromkeys((source.novelty.evidence_id, source.experiment.evidence_id)))))
        for observation in primary:
            evidence[observation.result.evidence_id] = observation.result
            if not any(claim.role == "case" and observation.result.evidence_id in claim.evidence_ids for claim in claims):
                claims.append(Claim(claim_id="primary-" + content_hash(observation)[:24], role="case", support="supported",
                    grounding_method="exact-contextual-quotation/3.0.0", text=observation.result.quote,
                    evidence_ids=(observation.result.evidence_id,)))
        passport = passport.model_copy(update={"claims": tuple(claims), "evidence": tuple(evidence.values())})
        artifact, card, _ = assess_snapshot(item, result.query_plan, result.snapshots[1], archive, context,
            passport=passport, methodology_version=version, source_novelty=sources, primary_observations=primary,
            antecedents=artifact.inputs.antecedents, field_exposure=artifact.inputs.field_exposure)
        result = result.model_copy(update={"methodology_version": version, "cards": (card,),
            "top_trend_ids": select_top((card,), (artifact,)), "quality": card.quality})
    target = result.cards[0].candidate
    sibling_candidate = target.model_copy(update={"candidate_id": "untouched-current-card"})
    sibling = TrendCard(candidate=sibling_candidate, methodology_version="3.4.0", category="unassessed_cluster",
        quality="partial", claims=(), evidence=(), limitations=("Synthetic retained sibling: still awaiting history.",))
    queue = (target.model_copy(update={"candidate_id": "untouched-queued-hypothesis"}),)
    state = CandidateReviewState(candidate_id=sibling_candidate.candidate_id, candidate_hash=content_hash(sibling_candidate),
        discovery_snapshot_id=sibling_candidate.discovery_snapshot_id, source_run_id="newer-refinement-run",
        attempt_ordinal=0, stage="candidate_batch_outcome_0", stage_hash="a" * 64, input_hash="b" * 64,
        outcome="passport_created")
    mixed = AnalysisResult.model_validate(result.model_dump(mode="python") | dict(methodology_version="3.4.0",
        cards=(*result.cards, sibling), candidate_queue=queue, candidate_review_states=(state,), quality="partial",
        limitations=("Mixed versions are retained without rewriting prior scientific results.",)))
    return mixed, archive, artifact


@pytest.mark.parametrize("version", ["3.2.0", "3.3.0"])
def test_actual_service_reviews_old_card_inside_current_container_without_upgrading_other_inputs(tmp_path, monkeypatch, version):
    from app.pilot.antecedents import AntecedentBundle

    original, archive, artifact = mixed_result(tmp_path / "original", version)
    target = original.cards[0]
    sibling_hash = content_hash(original.cards[1])
    old_artifact_hash = content_hash(artifact)
    package_path = tmp_path / "mixed-original.trendresult"
    export_result(package_path, original, archive, (artifact,))
    docs = tuple(archive.get(ref.revision_id) for ref in original.snapshots[0].documents)
    runtime, _, _ = runtime_for(tmp_path / "runtime", monkeypatch, docs)
    try:
        imported_id = runtime.import_result(str(package_path))["id"]
        before = runtime.result(imported_id)
        job = runtime.begin_review(imported_id, target.candidate.candidate_id)
        runtime.coordinator.wait(timeout=10)
        row = runtime.review_progress(job)
        assert row["state"] == "succeeded", row.get("error")
        prepared = row["review_data"]
        bundle = AntecedentBundle.model_validate(prepared["bundle"])
        card = TrendCard.model_validate(prepared["card"])
        values = decision_for(bundle, card).model_dump(mode="json")
        for key in ("schema_version", "version", "reviewed_at", "candidate_id", "admission_rule_hash", "bundle_hash"):
            values.pop(key, None)
        saved = runtime.apply_review(job, values)
        payload = runtime.result(saved["id"])
        reviewed_result = AnalysisResult.model_validate(payload["result"])
        reviewed = next(AssessmentArtifact.model_validate(raw) for raw in payload["assessments"]
                        if raw["assessment"]["candidate_id"] == target.candidate.candidate_id)
        assert reviewed.assessment.methodology_version == version
        assert next(item for item in reviewed_result.cards if item.candidate.candidate_id == target.candidate.candidate_id).methodology_version == version
        assert reviewed_result.methodology_version == "3.4.0"
        assert content_hash(next(item for item in reviewed_result.cards if item.candidate.candidate_id == "untouched-current-card")) == sibling_hash
        assert reviewed.inputs.source_novelty == artifact.inputs.source_novelty
        assert reviewed.inputs.primary_observations == artifact.inputs.primary_observations
        assert reviewed.inputs.field_exposure == artifact.inputs.field_exposure
        assert reviewed_result.candidate_queue == original.candidate_queue
        assert reviewed_result.candidate_review_states == original.candidate_review_states
        assert runtime.result(imported_id) == before
        assert content_hash(artifact) == old_artifact_hash
        destination = tmp_path / "mixed-reviewed.trendresult"
        runtime.export_result(saved["id"], str(destination))
        with read_result_package(destination) as reopened:
            assert reopened.result == reviewed_result
            assert reopened.assessments == (reviewed,)
    finally:
        runtime.close()
