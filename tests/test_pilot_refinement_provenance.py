"""A failed refinement stays attached to A when B is refined afterwards."""

import pytest

from app.pilot.contracts import AnalysisResult
from app.pilot.evidence import admission_hash, build_passport, label_candidates
from app.pilot.export import export_result, read_result_package
from app.pilot.history import assess_snapshot
from app.pilot.review import apply_novelty_review, record_review
from app.pilot.service import _REJECTED_REFINEMENT, _reviewed_refinement_notice
from tests.test_pilot_evidence import Context, query_plan
from tests.test_pilot_result_signals import automatic_result
from tests.test_pilot_review import decision_for
from tests.test_pilot_service_signals import runtime_for, wait_success


@pytest.mark.parametrize("reviewed", [False, True])
def test_reject_a_then_refine_b_preserves_warning_exact_assessment_and_export(tmp_path, monkeypatch, reviewed):
    original, archive, artifact = automatic_result(tmp_path / "original")
    source_a = original.cards[0]
    reviews = ()
    if reviewed:
        bundle = artifact.inputs.antecedents
        decision = decision_for(bundle, source_a)
        expanded, novelty = apply_novelty_review(decision, bundle, source_a, archive, methodology_version="3.2.0")
        assessed, card, _ = assess_snapshot(source_a.candidate, query_plan(), original.snapshots[1], archive,
            Context(), passport=expanded, verified_novelty=novelty, antecedents=bundle, methodology_version="3.2.0",
            field_exposure=artifact.inputs.field_exposure, source_novelty=artifact.inputs.source_novelty)
        reviews = (record_review(tmp_path / "reviews", decision, bundle, source_card=source_a, reviewed_card=card,
            artifact=assessed, historical_snapshot=original.snapshots[1], archive=archive),)
        artifact, source_a = assessed, card
    b = source_a.candidate.model_copy(update={"candidate_id": "candidate-b"})
    b = b.model_copy(update={"admission_rule_hash": admission_hash(b)})
    source_b = build_passport(b, original.snapshots[0], archive, Context()).model_copy(update={"methodology_version": "3.2.0"})
    original = AnalysisResult.model_validate(original.model_dump(mode="python") | {
        "cards": (source_a, source_b), "quality": "partial", "limitations": ("Initial incomplete candidate B",)})
    path = tmp_path / "original.trendresult"
    export_result(path, original, archive, (artifact,), reviews=reviews)
    docs = tuple(archive.get(item.revision_id) for item in original.snapshots[0].documents)
    runtime, _, _ = runtime_for(tmp_path / "runtime", monkeypatch, docs)
    reject = {source_a.candidate.candidate_id}

    def naming(candidates, *args, **kwargs):
        if candidates[0].candidate_id in reject:
            return ()
        return label_candidates(candidates, *args, **kwargs)

    monkeypatch.setattr("app.pilot.evidence.label_candidates", naming)
    try:
        initial_id = runtime.import_result(str(path))["id"]
        source_id = initial_id
        retained_a = source_a
        for _ in range(2):
            job = runtime.refine_candidate(source_id, source_a.candidate.candidate_id)
            result, assessments = wait_success(runtime, job)
            retained_a = next(card for card in result.cards if card.candidate.candidate_id == source_a.candidate.candidate_id)
            assert assessments == (artifact,) and len(result.cards) == 2
            if reviewed:
                assert retained_a == source_a
                assert result.limitations.count(_reviewed_refinement_notice(source_a)) == 1
            else:
                assert retained_a.limitations.count(_REJECTED_REFINEMENT) == 1
                assert retained_a.model_copy(update={"limitations": source_a.limitations}) == source_a
            source_id = job
        job_b = runtime.refine_candidate(source_id, b.candidate_id)
        final, assessments = wait_success(runtime, job_b)
        assert len(final.cards) == len(assessments) == 2
        assert next(card for card in final.cards if card.candidate.candidate_id == source_a.candidate.candidate_id) == retained_a
        assert next(item for item in assessments if item.inputs.candidate.candidate_id == source_a.candidate.candidate_id) == artifact
        if reviewed:
            assert final.limitations.count(_reviewed_refinement_notice(source_a)) == 1
        final_path = tmp_path / "chain.trendresult"
        runtime.export_result(job_b, str(final_path))
        with read_result_package(final_path) as packet:
            assert packet.result == final and packet.assessments == assessments and packet.reviews == reviews
        assert AnalysisResult.model_validate(runtime.result(initial_id)["result"]) == original
        # A successful replacement of A clears only A's stale refinement notice.
        reject.clear()
        job_a = runtime.refine_candidate(job_b, source_a.candidate.candidate_id)
        replaced, assessments = wait_success(runtime, job_a)
        assert len(replaced.cards) == len(assessments) == 2
        assert not any(_REJECTED_REFINEMENT in card.limitations for card in replaced.cards)
        assert _reviewed_refinement_notice(source_a) not in replaced.limitations
        assert AnalysisResult.model_validate(runtime.result(job_b)["result"]) == final
        runtime.export_result(job_a, str(tmp_path / "replaced.trendresult"))
    finally:
        runtime.close()
