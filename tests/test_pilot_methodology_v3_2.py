"""Candidate-to-passport replay for field-relative automatic and reviewed signals."""

import pytest

from app.pilot.antecedents import collect_antecedents
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Claim, TrendCard
from app.pilot.evidence import EvidenceError, build_passport
from app.pilot.history import assess_snapshot, verify_history_artifact
from app.pilot.methodology import AssessmentArtifact, AssessmentInput, NoveltyAssessment, evaluate_candidate
from app.pilot.signal_evidence import extract_signal_evidence
from app.runtime.credentials import CredentialStore
from tests.test_pilot_antecedents import Provider
from tests.test_pilot_evidence import Context, candidate, document, query_plan, snapshot
from tests.test_pilot_field_history import field_exposure
from tests.test_pilot_history import frozen
from tests.test_pilot_signal_evidence import NOVELTY, EXPERIMENT


def scenario32(tmp_path, counts=(0, 0, 0, 0, 1, 2), *, field_counts=(1000,) * 6,
               author=True, novelty=None, earlier=(), earlier_complete=True, with_field=True):
    archive = DocumentArchive(tmp_path / "revisions")
    abstract = document(1).abstract + (" " + NOVELTY + " " + EXPERIMENT if author else "")
    docs = tuple(document(10000 + year * 100 + index, year=year, abstract=abstract)
                 for year, count in zip(query_plan().completed_years, counts, strict=True) for index in range(count))
    discovery = snapshot(docs[-max(1, sum(counts[-3:])):], archive)
    historical = snapshot(docs, archive, purpose="history")
    item = frozen(candidate(discovery))
    context = Context()
    passport = build_passport(item, discovery, archive, context)
    source = extract_signal_evidence(item, discovery, archive, context, query_plan=query_plan())
    evidence = {entry.evidence_id: entry for entry in passport.evidence}
    for entry in source:
        evidence[entry.novelty.evidence_id] = entry.novelty
        evidence[entry.experiment.evidence_id] = entry.experiment
    claims = passport.claims
    verified_novelty = None
    if novelty is not None:
        reference = next(iter(evidence))
        claims += (Claim(claim_id="manual-novelty", role="novelty", text="Attributed expert novelty decision",
                         support="supported", evidence_ids=(reference,), grounding_method="reviewed-novelty/test"),)
        verified_novelty = NoveltyAssessment(kind=novelty, claim_id="manual-novelty", earlier_analogues_checked=True,
                                           terminology_changes_checked=True)
    passport = TrendCard.model_validate(passport.model_dump(mode="python") | {"claims": claims, "evidence": tuple(evidence.values())})
    bundle = collect_antecedents(item, query_plan(), archive, CredentialStore(), Context(),
                                provider_factory=lambda _: Provider(earlier, exhausted=earlier_complete))
    exposure = field_exposure(field_counts) if with_field else None
    artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport,
        antecedents=bundle, source_novelty=source, field_exposure=exposure, verified_novelty=verified_novelty,
        methodology_version="3.2.0")
    return artifact, card, archive, historical, context


def test_low_volume_author_claim_produces_labelled_weak_hypothesis_and_exact_archive_replay(tmp_path):
    artifact, card, archive, historical, context = scenario32(tmp_path)
    result = artifact.assessment
    assert card.category == result.category == "weak_signal_candidate"
    assert result.methodology_version == "3.2.0" and result.first_observed_year == 2024
    assert not result.growth_confirmed and result.confidence == "low"
    assert result.relative_growth.observed_growth and not result.relative_growth.excess_growth_supported
    assert result.relative_growth.ratio_lower_95 < 1
    assert artifact.inputs.novelty is None and result.components[2].value is None
    assert result.priority_score is None and result.signal_priority > 0
    assert not result.gate_failures
    assert AssessmentArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context)


def test_sustained_normalized_growth_with_author_claim_is_emerging_hypothesis_not_reviewed_confirmation(tmp_path):
    artifact, card, archive, historical, context = scenario32(tmp_path, (0, 0, 0, 2, 4, 8))
    assert card.category == "emerging_candidate" and artifact.assessment.growth_confirmed
    assert artifact.assessment.relative_growth.ratio_lower_95 > 1 and artifact.assessment.signal_priority > 0
    assert artifact.inputs.novelty is None and artifact.assessment.confidence == "low"
    verify_history_artifact(artifact, query_plan(), historical, archive, card, context)


@pytest.mark.parametrize("counts,expected", [((0, 0, 0, 0, 1, 2), "early_signal"), ((0, 0, 0, 2, 4, 8), "confirmed_trend")])
def test_reviewed_novelty_retains_separate_reviewed_stage_after_new_growth_gates(tmp_path, counts, expected):
    artifact, card, _, _, _ = scenario32(tmp_path, counts, author=False, novelty="new_mechanism")
    assert card.category == expected and artifact.inputs.source_novelty == ()
    assert artifact.assessment.signal_priority > 0


@pytest.mark.parametrize("counts,fields", [
    ((1, 1, 1, 10, 10, 10), (100, 100, 100, 1000, 1000, 1000)),
    ((0, 0, 0, 1, 2, 4), (100, 100, 100, 100, 200, 400)),
])
def test_growth_of_field_or_constant_relative_share_cannot_create_signal(tmp_path, counts, fields):
    artifact, card, _, _, _ = scenario32(tmp_path, counts, field_counts=fields)
    assert card.category == "insufficient_evidence" and artifact.assessment.signal_priority is None
    assert not artifact.assessment.growth_confirmed
    assert "no_observed_field_relative_growth" in artifact.assessment.gate_failures


@pytest.mark.parametrize("counts,expected", [((1, 1, 1, 2, 100, 3), "transient_burst"),
                                           ((1, 1, 1, 100, 20, 3), "declining")])
def test_decline_after_burst_never_enters_top_even_with_novelty_and_high_aggregate_score(tmp_path, counts, expected):
    artifact, card, _, _, _ = scenario32(tmp_path, counts, novelty="new_mechanism")
    assert card.category == expected and not artifact.assessment.growth_confirmed
    assert artifact.assessment.signal_priority is None


@pytest.mark.parametrize("changes,reason", [
    ({"with_field": False}, "field_exposure_unavailable"),
    ({"field_counts": (1000, None, 1000, 1000, 1000, 1000)}, "field_exposure_unavailable"),
    ({"earlier_complete": False}, "earlier_search_not_complete"),
    ({"author": False}, "no_reviewed_or_archived_novelty_evidence"),
    ({"earlier": (document(9090, year=2000),)}, "first_observation_not_recent"),
])
def test_missing_data_unknown_novelty_or_old_antecedent_cannot_be_weak_signal(tmp_path, changes, reason):
    artifact, card, _, _, _ = scenario32(tmp_path, **changes)
    assert card.category == "insufficient_evidence" and artifact.assessment.signal_priority is None
    assert reason in artifact.assessment.gate_failures


@pytest.mark.parametrize("novelty", ["established", "renamed", "new_application"])
def test_manual_earlier_maturity_decision_cannot_be_overridden_by_authors_word_novel(tmp_path, novelty):
    artifact, card, _, _, _ = scenario32(tmp_path, novelty=novelty)
    assert card.category == ("insufficient_evidence" if novelty == "new_application" else "established_topic")
    assert artifact.assessment.signal_priority is None


def test_established_fast_growing_technology_is_renewed_interest_with_no_signal_priority(tmp_path):
    artifact, card, _, _, _ = scenario32(tmp_path, (1, 1, 1, 4, 8, 16), novelty="established")
    assert card.category == "renewed_interest" and artifact.assessment.growth_confirmed
    assert artifact.assessment.signal_priority is None


def test_single_study_and_single_year_burst_cannot_become_positive_candidate(tmp_path):
    one, card, _, _, _ = scenario32(tmp_path / "one", (0, 0, 0, 0, 0, 1))
    assert card.category == "insufficient_evidence" and "fewer_than_two_recent_studies" in one.assessment.gate_failures
    burst, card, _, _, _ = scenario32(tmp_path / "burst", (0, 0, 0, 0, 0, 20))
    assert card.category == "insufficient_evidence" and not burst.assessment.growth_confirmed


def test_partial_history_broad_candidate_and_missing_card_fields_block_positive_classification(tmp_path):
    artifact, _, _, _, _ = scenario32(tmp_path)
    inputs = artifact.inputs
    partial = inputs.history.coverage.model_copy(update={"state": "partial", "reasons": ("cap",), "pagination_exhausted": False})
    tests = [
        inputs.model_copy(update={"history": inputs.history.model_copy(update={"coverage": partial})}),
        inputs.model_copy(update={"candidate": inputs.candidate.model_copy(update={"specificity": "broad_topic"})}),
        inputs.model_copy(update={"claims": tuple(item for item in inputs.claims if item.role != "advantage")}),
    ]
    for changed in tests:
        result = evaluate_candidate(AssessmentInput.model_validate(changed.model_dump(mode="python")), version="3.2.0")
        assert result.category in {"insufficient_evidence", "unassessed_cluster"} and result.signal_priority is None


def test_changed_source_year_or_quote_or_arithmetic_is_rejected_on_replay(tmp_path):
    artifact, card, archive, historical, context = scenario32(tmp_path)
    changed = artifact.model_dump(mode="python")
    changed["assessment"]["signal_priority"] += 1
    with pytest.raises(ValueError, match="reproduce"):
        AssessmentArtifact.model_validate(changed)
    inputs = artifact.inputs
    source = inputs.source_novelty[0].model_copy(update={"publication_year": 2020})
    with pytest.raises(EvidenceError):
        assess_snapshot(card.candidate, query_plan(), historical, archive, context, passport=card,
                        antecedents=inputs.antecedents, field_exposure=inputs.field_exposure,
                        source_novelty=(source,), methodology_version="3.2.0")


def test_deployment_bonus_does_not_change_signal_priority(tmp_path):
    from app.pilot.methodology import ApplicationAssessment

    artifact, _, _, _, _ = scenario32(tmp_path)
    data = artifact.inputs
    claim = next(item for item in data.claims if item.role == "application")
    deployment = data.model_copy(update={"application": ApplicationAssessment(kind="deployment", claim_id=claim.claim_id)})
    result = evaluate_candidate(deployment, version="3.2.0")
    assert result.signal_priority == artifact.assessment.signal_priority
    assert result.category == artifact.assessment.category


def test_old_versions_omit_new_optional_fields_and_keep_original_arithmetic(tmp_path):
    from tests.test_pilot_methodology import assessment

    legacy = assessment((1, 1, 1, 2, 100, 3))
    assert not {"field_exposure", "source_novelty", "antecedents"}.intersection(legacy.model_dump())
    old = evaluate_candidate(legacy, version="3.0.0")
    prior = evaluate_candidate(legacy, version="3.1.0")
    assert old.category == "confirmed_trend" and prior.category == "transient_burst"
    for result in (old, prior):
        assert not {"relative_growth", "signal_priority"}.intersection(result.model_dump())
        assert AssessmentArtifact.model_validate_json(AssessmentArtifact(inputs=legacy, assessment=result).model_dump_json()).assessment == result


def test_signal_ranking_cannot_be_boosted_by_high_legacy_score_of_ineligible_burst(tmp_path):
    from app.pilot.methodology import rank_candidates

    weak = scenario32(tmp_path / "weak")[0].assessment
    burst = scenario32(tmp_path / "burst", (1, 1, 1, 2, 100, 3), novelty="new_mechanism")[0].assessment
    burst = burst.model_copy(update={"candidate_id": "burst"})
    assert burst.priority_lower_bound > weak.priority_lower_bound
    assert burst.signal_priority is None and weak.signal_priority is not None
    assert rank_candidates((burst, weak), confirmed_only=False) == (weak, burst)
