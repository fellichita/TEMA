"""Known trajectories and falsification cases for the new historical methodology."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.pilot.contracts import Claim
from app.pilot.methodology import (
    APPLICATION_SCORES, NOVELTY_SCORES, WEIGHTS, ApplicationAssessment, AssessmentArtifact, AssessmentInput,
    HistoricalSeries, IndependenceAssessment, IndependentGroup, NoveltyAssessment,
    YearStudies, evaluate_candidate, rank_candidates,
)
from tests.test_pilot_contracts import HASH, NOW, candidate, coverage


def history(counts=(1, 1, 1, 2, 4, 8), **changes):
    years = tuple(range(2026 - len(counts), 2026))
    observations = tuple(YearStudies(year=year, study_ids=tuple(f"study-{year}-{i}" for i in range(count)))
                         for year, count in zip(years, counts, strict=True))
    values = dict(candidate_id="candidate-1", snapshot_id="history-1", admission_rule_hash=HASH,
                  as_of=NOW.date(), observations=observations,
                  coverage=coverage(requested_years=years, completed_years=years,
                                    scanned_records=sum(counts), accepted_records=sum(counts)))
    return HistoricalSeries(**(values | changes))


def assessment(counts=(1, 1, 1, 2, 4, 8), **changes):
    series = changes.pop("history", history(counts))
    claims = tuple(Claim(claim_id=role, role=role, text=f"Verified {role}", support="supported",
                         evidence_ids=("evidence",), grounding_method="human-review/1")
                   for role in ("problem", "advantage", "case", "novelty", "application"))
    recent = tuple(study for year in series.observations[-3:] for study in year.study_ids)
    groups = tuple(IndependentGroup(group_id=f"team-{index}", study_ids=recent[index::4],
                                   evidence_ids=("evidence",)) for index in range(min(4, len(recent))))
    values = dict(candidate=candidate(), history=series, claims=claims, evidence_ids=("evidence",),
                  novelty=NoveltyAssessment(kind="new_mechanism", claim_id="novelty",
                                            earlier_analogues_checked=True, terminology_changes_checked=True),
                  application=ApplicationAssessment(kind="research", claim_id="application"),
                  independence=IndependenceAssessment(groups=groups, method_version="verified-teams/1",
                                                      coverage_complete=True) if groups else None)
    return AssessmentInput(**(values | changes))


def test_known_complete_trajectory_has_exact_denominators_slope_scores_and_evidence_hash():
    inputs = assessment()
    result = evaluate_candidate(inputs)
    assert (result.baseline_studies, result.recent_studies) == (3, 14)
    assert result.smoothed_growth == 15 / 4
    assert result.raw_growth == 14 / 3
    assert result.recent_theil_sen_slope == 3
    assert result.priority_score == 91.10
    assert result.category == "confirmed_trend"
    assert result.confidence == "high"
    assert result.growth_confirmed and not result.gate_failures
    assert evaluate_candidate(AssessmentInput.model_validate_json(inputs.model_dump_json())) == result


def test_new_ranking_ties_do_not_reward_raw_volume_and_legacy_order_is_preserved():
    base = evaluate_candidate(assessment())
    small = base.model_copy(update={"candidate_id": "a-rare", "recent_studies": 14})
    large = base.model_copy(update={"candidate_id": "z-large", "recent_studies": 1400})
    assert rank_candidates((large, small)) == (small, large)
    old_small = small.model_copy(update={"methodology_version": "3.0.0"})
    old_large = large.model_copy(update={"methodology_version": "3.0.0"})
    assert rank_candidates((old_small, old_large)) == (old_large, old_small)


def test_first_observation_is_context_not_a_worldwide_invention_claim():
    result = evaluate_candidate(assessment((0, 0, 0, 0, 4, 16)))
    assert result.baseline_studies == 0 and result.raw_growth is None
    assert result.smoothed_growth == 21 and result.recent_theil_sen_slope == 8
    assert result.first_observed_year == 2024
    assert result.observation_label == "appearance_in_observed_corpus"
    assert "first_observation_is_not_worldwide_invention_date" in result.limitations


@pytest.mark.parametrize("counts,reason", [
    ((0, 0, 0, 0, 0, 20), "fewer_than_two_active_recent_years"),
    ((0, 0, 0, 0, 2, 3), "fewer_than_ten_recent_studies"),
    ((4, 4, 4, 4, 4, 4), "nonpositive_recent_slope"),
    ((0, 0, 0, 12, 6, 3), "nonpositive_recent_slope"),
    ((8, 8, 8, 2, 4, 8), "growth_below_1_5"),
])
def test_failure_of_any_gate_cannot_be_compensated_by_other_high_components(counts, reason):
    result = evaluate_candidate(assessment(counts))
    assert not result.growth_confirmed
    assert result.category == ("declining" if counts[-1] < counts[-2] else "insufficient_evidence")
    assert reason in result.gate_failures


def test_comparison_uses_immediately_preceding_three_years_not_all_old_years():
    six_year = evaluate_candidate(assessment())
    ten_year = evaluate_candidate(assessment((100, 100, 100, 100, 1, 1, 1, 2, 4, 8)))
    assert ten_year.baseline_years == (2020, 2021, 2022)
    assert ten_year.smoothed_growth == six_year.smoothed_growth
    assert ten_year.first_observed_year == 2016


@pytest.mark.parametrize("changes", [dict(purpose="discovery"), dict(source="crossref"),
    dict(comparable=False), dict(state="partial", pagination_exhausted=False, reasons=("cap",), limit_reached=True)])
def test_discovery_other_catalogues_and_capped_history_cannot_manufacture_confirmed_growth(changes):
    source_coverage = coverage(scanned_records=17, accepted_records=17, **changes)
    result = evaluate_candidate(assessment(history=history(coverage=source_coverage)))
    assert not result.growth_confirmed and result.quality == "partial"
    assert result.priority_score is None
    assert result.components[0].value is None and result.components[1].value is None
    assert result.smoothed_growth == 3.75  # Explicitly labelled as an observed, partial series.


def test_unknown_independence_is_null_and_an_interval_without_renormalizing_other_weights():
    result = evaluate_candidate(assessment(independence=None))
    assert result.category == "confirmed_trend" and result.confidence == "medium"
    assert result.components[3].value is None
    assert result.priority_score is None
    assert result.priority_lower_bound == 76.10
    assert result.priority_upper_bound == 91.10
    assert result.components[3].reason == "independent_groups_unknown_or_incomplete"


def test_growth_is_distinct_from_novelty_and_renaming_cannot_be_an_emerging_trend():
    unknown = evaluate_candidate(assessment(novelty=None))
    assert unknown.growth_confirmed and unknown.category == "insufficient_evidence"
    assert "novelty_not_verified" in unknown.gate_failures
    renamed = NoveltyAssessment(kind="renamed", claim_id="novelty", earlier_analogues_checked=True,
                                terminology_changes_checked=True)
    renewed = evaluate_candidate(assessment(novelty=renamed))
    assert renewed.category == "renewed_interest" and renewed.components[2].value == 0
    established = evaluate_candidate(assessment((4, 4, 4, 4, 4, 4), novelty=renamed))
    assert established.category == "established_topic"


def test_unchecked_earlier_analogues_and_unverified_card_claims_block_confirmation():
    novelty = NoveltyAssessment(kind="new_mechanism", claim_id="novelty", earlier_analogues_checked=False,
                                terminology_changes_checked=True)
    assert evaluate_candidate(assessment(novelty=novelty)).category == "insufficient_evidence"
    inputs = assessment()
    claims = tuple(claim for claim in inputs.claims if claim.role != "case")
    assert "unsupported_card_fields" in evaluate_candidate(assessment(claims=claims)).gate_failures
    assert "candidate_not_specific" in evaluate_candidate(assessment(candidate=candidate(specificity="broad_topic"))).gate_failures


def test_unknown_or_conflicting_dates_never_yield_high_confidence_or_a_false_first_year():
    result = evaluate_candidate(assessment(history=history(date_conflicts=("study-2025-1",))))
    assert result.confidence == "low" and not result.growth_confirmed
    with pytest.raises(ValidationError):
        history(first_observed_year=2025, first_observed_study_id="study-2025-0")
    with pytest.raises(ValidationError):
        history(first_observed_year=2020)


def test_duplicate_studies_across_years_source_count_mismatch_and_changed_admission_are_rejected():
    source = history()
    observations = source.observations[:-1] + (YearStudies(year=2025, study_ids=("study-2020-0",)),)
    with pytest.raises(ValidationError):
        history(observations=observations)
    with pytest.raises(ValidationError):
        history(coverage=coverage())
    with pytest.raises(ValidationError):
        assessment(history=history(admission_rule_hash="b" * 64))


def test_complete_independence_requires_dated_recent_members_and_grounded_evidence():
    group = IndependentGroup(group_id="group", study_ids=("study-2023-0",), evidence_ids=("evidence",))
    with pytest.raises(ValidationError):
        assessment(independence=IndependenceAssessment(groups=(group,), method_version="1", coverage_complete=True))
    partial = IndependenceAssessment(groups=(group,), method_version="1", coverage_complete=False)
    assert evaluate_candidate(assessment(independence=partial)).components[3].value is None
    with pytest.raises(ValidationError):
        IndependenceAssessment(groups=(group, group), method_version="1", coverage_complete=True)


def test_all_zero_history_is_insufficient_and_never_fills_the_top_with_placeholders():
    result = evaluate_candidate(assessment((0, 0, 0, 0, 0, 0)))
    assert result.quality == "insufficient_data" and result.first_observed_year is None
    assert result.observation_label == "no_observations"
    assert rank_candidates((result,)) == ()


def test_ranking_is_stable_bounded_and_rejects_duplicate_ids():
    def evaluated(identifier):
        return evaluate_candidate(assessment(candidate=candidate(candidate_id=identifier),
            history=history(candidate_id=identifier)))
    a, b = evaluated("a"), evaluated("b")
    assert rank_candidates((b, a)) == (a, b)
    assert rank_candidates((b, a), limit=1) == (a,)
    with pytest.raises(ValueError):
        rank_candidates((a, a))
    with pytest.raises(ValueError):
        rank_candidates((a,), limit=True)


def test_published_methodology_matches_executable_component_values():
    document = json.loads((Path(__file__).parents[1] / "app/pilot/methodology-v3.json").read_text(encoding="utf-8"))
    assert document["version"] == "3.0.0"
    assert tuple(document["weights"].values()) == WEIGHTS
    assert document["novelty_rubric"] == NOVELTY_SCORES
    assert document["application_rubric"] == APPLICATION_SCORES


def test_imported_assessment_recomputes_counts_ranking_and_classification():
    inputs = assessment()
    artifact = AssessmentArtifact(inputs=inputs, assessment=evaluate_candidate(inputs))
    assert AssessmentArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    for field, forged_value in (("recent_studies", 100), ("priority_score", 100), ("smoothed_growth", 100)):
        payload = artifact.model_dump(mode="json")
        payload["assessment"][field] = forged_value
        with pytest.raises(ValidationError, match="does not reproduce"):
            AssessmentArtifact.model_validate(payload)


def test_burst_collapse_cannot_confirm_even_with_positive_theil_sen_and_high_volume():
    inputs = assessment((1, 1, 1, 2, 100, 3))
    corrected = evaluate_candidate(inputs)
    assert corrected.recent_theil_sen_slope == 0.5  # Descriptive statistic is retained honestly.
    assert not corrected.growth_confirmed
    assert corrected.category == "transient_burst"
    assert corrected.components[1].value == 50
    assert "recent_decline_or_transient_burst" in corrected.gate_failures
    assert rank_candidates((corrected,)) == ()
    assert evaluate_candidate(assessment()).growth_confirmed


def test_legacy_burst_artifact_still_replays_without_rewriting_history():
    inputs = assessment((1, 1, 1, 2, 100, 3))
    legacy = evaluate_candidate(inputs, version="3.0.0")
    assert legacy.category == "confirmed_trend" and legacy.priority_score == 92.5
    artifact = AssessmentArtifact(inputs=inputs, assessment=legacy)
    assert AssessmentArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    changed = artifact.model_dump(mode="json")
    changed["assessment"]["methodology_version"] = "3.1.0"
    with pytest.raises(ValidationError, match="does not reproduce"):
        AssessmentArtifact.model_validate(changed)


def test_low_volume_watchlist_needs_reviewed_novelty_and_completed_antecedent_search():
    counts = (0, 0, 0, 0, 1, 2)
    ordinary = evaluate_candidate(assessment(counts))
    assert ordinary.category == "insufficient_evidence"
    reviewed = evaluate_candidate(assessment(counts, history=history(counts, earlier_search_complete=True)))
    assert reviewed.category == "early_signal" and not reviewed.growth_confirmed
    assert "reviewed_low_volume_hypothesis_not_confirmed_trend" in reviewed.limitations
    unknown = evaluate_candidate(assessment(counts, history=history(counts, earlier_search_complete=True), novelty=None))
    assert unknown.category == "insufficient_evidence"


def test_broad_group_and_metadata_team_diversity_are_not_scientific_confirmation():
    broad = evaluate_candidate(assessment(candidate=candidate(specificity="broad_topic")))
    assert broad.category == "unassessed_cluster"
    inputs = assessment()
    bibliometric = inputs.independence.model_copy(update={
        "method_version": "openalex-author-components-team-diversity/2.0.0"})
    result = evaluate_candidate(assessment(independence=bibliometric))
    assert result.components[3].value is None and result.priority_score is None
    assert "team_diversity_is_not_independent_replication" in result.limitations


def test_optional_card_version_preserves_legacy_canonical_bytes():
    from app.pilot.contracts import TrendCard
    from tests.test_pilot_contracts import card

    old = card()
    assert "methodology_version" not in old.model_dump(mode="json")
    assert TrendCard.model_validate_json(old.model_dump_json()).model_dump_json() == old.model_dump_json()
