"""One current-year finding is observable; it never fabricates scientific confirmation."""

import json

import pytest

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Claim, TrendCard
from app.pilot.evidence import build_passport
from app.pilot.methodology import AssessmentArtifact, AssessmentInput, NoveltyAssessment, evaluate_candidate
from app.pilot.selection import select_top
from app.pilot.signal_evidence import extract_primary_observations, verify_primary_observations, verify_primary_sources
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot
from tests.test_pilot_history import frozen
from tests.test_pilot_methodology import history
from tests.test_pilot_methodology_v3_2 import scenario32


PRIMARY_RESULT = "We measured lithium selective membranes in laboratory experiments and found lithium selectivity of 95 percent."
PRIMARY_ABSTRACT = "Existing extraction methods suffer from limited selectivity. " + PRIMARY_RESULT


def observation33(tmp_path, *, counts=(0, 0, 0, 0, 0, 0), year=2026, partial=True,
                  abstract=PRIMARY_ABSTRACT, novelty=None, **document_changes):
    archive = DocumentArchive(tmp_path / "revisions")
    doc = document(4242, year=year, abstract=abstract, **document_changes)
    discovery = snapshot((doc,), archive)
    item = frozen(candidate(discovery))
    context = Context()
    passport = build_passport(item, discovery, archive, context, methodology_version="3.3.0")
    primary = extract_primary_observations(item, discovery, archive, context, query_plan=query_plan())
    evidence = {entry.evidence_id: entry for entry in passport.evidence}
    for entry in primary:
        evidence[entry.result.evidence_id] = entry.result
    series = history(counts)
    series = series.model_copy(update={"candidate_id": item.candidate_id, "admission_rule_hash": item.admission_rule_hash})
    if partial:
        coverage = series.coverage.model_copy(update={"state": "partial", "pagination_exhausted": False,
            "comparable": False, "reasons": ("not_collected",)})
        series = series.model_copy(update={"coverage": coverage})
    claims = passport.claims
    review = None
    if novelty is not None:
        # Deliberate attributed review is stronger than an author's new-word assertion.
        ref = next(iter(evidence))
        claims += (Claim(claim_id="review-novelty", role="novelty", text="Reviewed novelty decision",
            support="supported", evidence_ids=(ref,), grounding_method="reviewed-novelty/test"),)
        review = NoveltyAssessment(kind=novelty, claim_id="review-novelty", earlier_analogues_checked=True,
                                  terminology_changes_checked=True)
    inputs = AssessmentInput(candidate=item, history=series, claims=claims, evidence_ids=tuple(evidence),
        primary_observations=primary, novelty=review)
    result = evaluate_candidate(inputs, version="3.3.0")
    artifact = AssessmentArtifact(inputs=inputs, assessment=result)
    card = TrendCard.model_validate(passport.model_dump(mode="python") | dict(claims=claims,
        evidence=tuple(evidence.values()), historical_snapshot_id=series.snapshot_id,
        assessment_hash=result.assessment_hash, category=result.category, quality=result.quality))
    return artifact, card, archive, discovery, context


def test_one_current_year_primary_result_enters_observation_top_without_filling_missing_history(tmp_path):
    artifact, card, archive, discovery, context = observation33(tmp_path)
    result = artifact.assessment
    assert card.category == "weak_signal_candidate" and result.first_observed_year == 2026
    assert result.recent_studies == result.baseline_studies == result.active_recent_years == 0
    assert result.relative_growth is None and not result.growth_confirmed
    assert result.confidence == "low" and result.quality == "partial" and result.signal_priority == 10
    assert artifact.inputs.novelty is None and artifact.inputs.source_novelty == ()
    assert "current_year_primary_result_has_no_comparable_full_year_growth" in result.limitations
    assert "novelty_and_independent_replication_require_separate_review" in result.limitations
    assert select_top((card,), (artifact,)) == (card.candidate.candidate_id,)
    assert all(year.year < 2026 for year in artifact.inputs.history.observations)
    verify_primary_observations(artifact.inputs.primary_observations, card.candidate, discovery, archive, context,
                                query_plan=query_plan())
    verify_primary_sources(artifact.inputs.primary_observations, card.candidate, archive, context, query_plan=query_plan())


@pytest.mark.parametrize("year,partial", [(2025, False), (2025, True), (2024, True), (2023, True)])
def test_observed_primary_result_can_precede_complete_growth_and_novelty_analysis(tmp_path, year, partial):
    artifact, _, _, _, _ = observation33(tmp_path, year=year, partial=partial)
    assert artifact.assessment.category == "weak_signal_candidate"
    assert not artifact.assessment.growth_confirmed and artifact.assessment.confidence == "low"
    assert artifact.assessment.signal_priority == 10
    # Absence of an advantage sentence is explicit and does not hide a concrete primary case.
    assert not any(claim.role == "advantage" for claim in artifact.inputs.claims)
    assert any(claim.role == "case" and claim.support == "supported" for claim in artifact.inputs.claims)


@pytest.mark.parametrize("changes", [
    {"document_type": "dataset"},
    {"raw_metadata": {"type": "dataset"}},
    {"document_type": "review"},
    {"abstract": "This paper reviews recent advances in lithium selective membranes. " + PRIMARY_RESULT},
    {"abstract": "We propose lithium selective membranes that might be tested in future experiments."},
    {"abstract": "A promising technology with a compelling title and no report of its own result."},
    {"abstract": PRIMARY_ABSTRACT + " Our results do not support the mechanism."},
    {"raw_metadata": {"is_retracted": True}},
    {"year": 2027},
    {"year": 2022},
])
def test_unsupported_units_future_work_and_known_old_work_do_not_become_observations(tmp_path, changes):
    artifact, card, _, _, _ = observation33(tmp_path, **changes)
    assert artifact.assessment.category == "insufficient_evidence"
    assert artifact.assessment.signal_priority is None and select_top((card,), (artifact,)) == ()


@pytest.mark.parametrize("novelty", ["established", "renamed", "new_application"])
def test_new_primary_result_cannot_override_reviewed_maturity_or_application_only_decision(tmp_path, novelty):
    artifact, _, _, _, _ = observation33(tmp_path, novelty=novelty)
    assert artifact.assessment.category not in {"weak_signal_candidate", "early_signal", "confirmed_trend"}
    assert artifact.assessment.signal_priority is None


@pytest.mark.parametrize("counts", [(1, 0, 0, 0, 0, 0), (0, 0, 0, 3, 2, 1), (0, 0, 0, 0, 0, 20)])
def test_existing_older_history_decline_and_large_burst_do_not_receive_sparse_label(tmp_path, counts):
    artifact, _, _, _, _ = observation33(tmp_path, counts=counts, partial=False)
    assert artifact.assessment.category not in {"weak_signal_candidate", "early_signal", "confirmed_trend"}
    assert artifact.assessment.signal_priority is None


def test_current_primary_result_survives_one_annual_dip_with_explicit_concern(tmp_path):
    artifact, _, _, _, _ = observation33(tmp_path, counts=(0, 0, 0, 0, 2, 1), partial=False)
    assert artifact.assessment.category == "weak_signal_candidate" and not artifact.assessment.growth_confirmed
    assert "completed_year_decline_remains_a_concern_despite_new_primary_observation" in artifact.assessment.limitations


def test_broad_scope_and_case_from_another_record_cannot_supply_primary_admission(tmp_path):
    artifact, _, _, _, _ = observation33(tmp_path)
    for inputs in (
        artifact.inputs.model_copy(update={"candidate": artifact.inputs.candidate.model_copy(update={"specificity": "broad_topic"})}),
        artifact.inputs.model_copy(update={"claims": tuple(claim for claim in artifact.inputs.claims if claim.role != "case")}),
    ):
        assert evaluate_candidate(inputs, version="3.3.0").category in {"unassessed_cluster", "insufficient_evidence"}


@pytest.mark.parametrize("sentence,kind", [
    ("We simulated lithium selective membranes and found selectivity of 95 percent in computational experiments.", "computational"),
    ("We derived an analytical model of lithium selective membranes and proved the transport bound.", "theoretical"),
])
def test_computational_and_theoretical_cases_keep_their_actual_evidence_stage(tmp_path, sentence, kind):
    artifact, _, _, _, _ = observation33(tmp_path, abstract=sentence)
    assert artifact.inputs.primary_observations[0].result_kind == kind
    assert artifact.assessment.category == "weak_signal_candidate"
    assert "computational_or_theoretical_result_is_not_experimental_demonstration" in artifact.assessment.limitations
    assert not artifact.assessment.growth_confirmed
    assert not any(claim.role == "application" for claim in artifact.inputs.claims)


def test_optional_primary_input_omits_itself_from_old_archives_and_new_archive_replays_exactly(tmp_path):
    legacy = scenario32(tmp_path / "legacy")[0]
    assert "primary_observations" not in legacy.inputs.model_dump(mode="json")
    assert AssessmentArtifact.model_validate_json(legacy.model_dump_json()) == legacy
    artifact, card, archive, _, context = observation33(tmp_path / "new")
    assert AssessmentArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    corrupt = json.loads(artifact.model_dump_json())
    corrupt["assessment"]["signal_priority"] = 100
    with pytest.raises(ValueError, match="reproduce"):
        AssessmentArtifact.model_validate(corrupt)
    primary = artifact.inputs.primary_observations[0].model_copy(update={"publication_year": 2024})
    with pytest.raises(Exception, match="не воспроизводится"):
        verify_primary_sources((primary,), card.candidate, archive, context, query_plan=query_plan())


@pytest.mark.parametrize("counts,novelty", [
    ((0, 0, 0, 0, 1, 2), None), ((0, 0, 0, 2, 4, 8), None),
    ((0, 0, 0, 2, 4, 8), "new_mechanism"), ((1, 1, 1, 4, 8, 16), "established"),
    ((1, 1, 1, 2, 100, 3), "new_mechanism"),
])
def test_existing_bibliometric_stages_growth_and_priority_remain_numerically_unchanged(tmp_path, counts, novelty):
    artifact = scenario32(tmp_path, counts, novelty=novelty)[0]
    result = evaluate_candidate(artifact.inputs, version="3.3.0")
    assert result.category == artifact.assessment.category
    assert result.growth_confirmed == artifact.assessment.growth_confirmed
    assert result.signal_priority == artifact.assessment.signal_priority
    assert result.relative_growth == artifact.assessment.relative_growth
    assert result.components == artifact.assessment.components
