"""Versioned deterministic emerging-signal methodology, including archive replay.

Discovery membership never supplies a historical growth denominator. Every
count below comes from unique, dated study identifiers in an immutable history.
Unknown components stay unknown; score intervals bound missing contributions
without treating those bounds as measured values. See methodology-v3.json.
"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from statistics import median
from typing import Literal, Self

from pydantic import Field, model_validator

from app.pilot.contracts import (
    METHODOLOGY_VERSION, Category, MethodologyVersion, Candidate, Claim, Contract, Coverage, Digest, Identifier,
    Quality, Year, DocumentRevisionRef, content_hash, require_unique, validate_years,
)
from app.pilot.antecedents import AntecedentBundle
from app.pilot.field_history import FieldExposure, RelativeGrowth, normalize_growth
from app.pilot.signal_evidence import SourceNoveltyEvidence, PrimaryStudyEvidence

NoveltyKind = Literal["new_mechanism", "new_combination", "new_application", "renamed", "established"]
ApplicationKind = Literal["research", "demonstrator", "deployment"]
ScoreName = Literal["growth", "persistence", "novelty", "independence", "application"]
WEIGHTS = (30, 20, 20, 15, 15)
NOVELTY_SCORES: dict[NoveltyKind, int] = {
    "new_mechanism": 100, "new_combination": 75, "new_application": 50, "renamed": 0, "established": 0,
}
APPLICATION_SCORES: dict[ApplicationKind, int] = {"research": 50, "demonstrator": 75, "deployment": 100}


class YearStudies(Contract):
    year: Year
    study_ids: tuple[Identifier, ...] = Field(default=(), max_length=3000)

    @model_validator(mode="after")
    def distinct_studies(self) -> Self:
        require_unique(self.study_ids, "yearly studies")
        return self

    @property
    def count(self) -> int:
        return len(self.study_ids)


class HistoricalSeries(Contract):
    candidate_id: Identifier
    snapshot_id: Identifier
    admission_rule_hash: Digest
    as_of: date
    observations: tuple[YearStudies, ...] = Field(min_length=6, max_length=10)
    coverage: Coverage
    first_observed_year: Year | None = None
    first_observed_study_id: Identifier | None = None
    earlier_search_complete: bool = Field(default=False, strict=True)
    date_conflicts: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def consistent_history(self) -> Self:
        years = tuple(observation.year for observation in self.observations)
        validate_years(years, self.as_of)
        if set(self.coverage.requested_years) != set(years):
            raise ValueError("Coverage and historical years must be identical")
        members = tuple(study for observation in self.observations for study in observation.study_ids)
        require_unique(members, "independent studies across years")
        if len(members) > 3000:
            raise ValueError("Historical candidate exceeds the documented 3000-study budget")
        if len(members) > self.coverage.accepted_records:
            raise ValueError("Historical studies exceed the accepted source records")
        known = [(observation.year, study) for observation in self.observations for study in observation.study_ids]
        if (self.first_observed_year is None) != (self.first_observed_study_id is None):
            raise ValueError("First observation requires both a year and a supporting study")
        if self.first_observed_year is not None:
            if self.first_observed_year > self.as_of.year - 1:
                raise ValueError("First historical observation is not a completed year")
            if known and self.first_observed_year > min(year for year, _ in known):
                raise ValueError("First observation is later than an observed historical study")
            if self.first_observed_year >= years[0]:
                if (self.first_observed_year, self.first_observed_study_id) not in known:
                    raise ValueError("First observed study is missing from its declared year")
        return self


class NoveltyAssessment(Contract):
    kind: NoveltyKind
    claim_id: Identifier
    earlier_analogues_checked: bool = Field(strict=True)
    terminology_changes_checked: bool = Field(strict=True)


class ApplicationAssessment(Contract):
    kind: ApplicationKind
    claim_id: Identifier


class IndependentGroup(Contract):
    """A verified research-team grouping, not a count of source catalogues."""
    group_id: Identifier
    study_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=3000)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        require_unique(self.study_ids, "group studies")
        require_unique(self.evidence_ids, "group evidence")
        return self


class IndependenceAssessment(Contract):
    groups: tuple[IndependentGroup, ...] = Field(min_length=1, max_length=3000)
    method_version: Identifier
    coverage_complete: bool = Field(strict=True)

    @model_validator(mode="after")
    def disjoint_groups(self) -> Self:
        require_unique(tuple(group.group_id for group in self.groups), "independent groups")
        # Connected/collaborating teams must be merged before this boundary.
        require_unique(tuple(study for group in self.groups for study in group.study_ids), "group memberships")
        return self


class AssessmentInput(Contract):
    candidate: Candidate
    history: HistoricalSeries
    claims: tuple[Claim, ...] = Field(default=(), max_length=40)
    evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=200)
    novelty: NoveltyAssessment | None = None
    independence: IndependenceAssessment | None = None
    application: ApplicationAssessment | None = None
    field_exposure: FieldExposure | None = Field(default=None, exclude_if=lambda value: value is None)
    source_novelty: tuple[SourceNoveltyEvidence, ...] = Field(default=(), max_length=10, exclude_if=lambda value: not value)
    primary_observations: tuple[PrimaryStudyEvidence, ...] = Field(default=(), max_length=10, exclude_if=lambda value: not value)
    antecedents: AntecedentBundle | None = Field(default=None, exclude_if=lambda value: value is None)
    publication_status_revisions: tuple[DocumentRevisionRef, ...] = Field(
        default=(), max_length=100000, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def assessment_provenance(self) -> Self:
        if (self.candidate.candidate_id != self.history.candidate_id
                or self.candidate.admission_rule_hash != self.history.admission_rule_hash):
            raise ValueError("History must use the frozen candidate and admission rule")
        require_unique(tuple(claim.claim_id for claim in self.claims), "assessment claims")
        require_unique(self.evidence_ids, "assessment evidence")
        require_unique(tuple(item.revision_id for item in self.publication_status_revisions),
                       "publication status revisions")
        evidence = set(self.evidence_ids)
        if any(not set(claim.evidence_ids).issubset(evidence) for claim in self.claims):
            raise ValueError("Assessment claims reference missing evidence")
        if self.field_exposure is not None:
            if (self.field_exposure.plan_hash != self.candidate.plan_hash
                    or tuple(item.year for item in self.field_exposure.years) != tuple(item.year for item in self.history.observations)):
                raise ValueError("Field exposure must belong to the candidate plan and historical years")
        if self.antecedents is not None:
            if (self.antecedents.candidate_id != self.candidate.candidate_id
                    or self.antecedents.admission_rule_hash != self.candidate.admission_rule_hash
                    or self.antecedents.query_plan.plan_hash != self.candidate.plan_hash
                    or self.history.earlier_search_complete != self.antecedents.search_complete):
                raise ValueError("Earlier search must match the candidate, plan and completeness declaration")
            if self.field_exposure is not None and self.field_exposure.fixed_query != self.antecedents.query_plan.english_query:
                raise ValueError("Field exposure must retain the exact frozen field query")
        require_unique(tuple(item.novelty.evidence_id for item in self.source_novelty), "source novelty entries")
        for item in self.source_novelty:
            if (item.candidate_id != self.candidate.candidate_id
                    or item.admission_rule_hash != self.candidate.admission_rule_hash
                    or item.discovery_snapshot_id != self.candidate.discovery_snapshot_id
                    or item.novelty.study_id not in self.candidate.discovery_study_ids
                    or item.publication_year > self.history.as_of.year
                    or not {item.novelty.evidence_id, item.experiment.evidence_id}.issubset(evidence)):
                raise ValueError("Author novelty must reference the exact dated candidate and archived evidence")
        require_unique(tuple(item.result.study_id for item in self.primary_observations), "primary observation studies")
        for primary_item in self.primary_observations:
            if (primary_item.candidate_id != self.candidate.candidate_id
                    or primary_item.admission_rule_hash != self.candidate.admission_rule_hash
                    or primary_item.discovery_snapshot_id != self.candidate.discovery_snapshot_id
                    or primary_item.result.study_id not in self.candidate.discovery_study_ids
                    or primary_item.publication_year > self.history.as_of.year
                    or primary_item.result.evidence_id not in evidence):
                raise ValueError("Primary observations must reference the exact dated candidate and archived evidence")
        claims = {claim.claim_id: claim for claim in self.claims}
        for assessment, role in ((self.novelty, "novelty"), (self.application, "application")):
            if assessment is not None:
                claim = claims.get(assessment.claim_id)
                if claim is None or claim.support != "supported" or claim.role != role:
                    raise ValueError(f"{role} assessment requires a supported {role} claim")
        if self.independence is not None:
            recent_studies = {study for observation in self.history.observations[-3:]
                              for study in observation.study_ids}
            grouped_studies = {study for group in self.independence.groups for study in group.study_ids}
            if not grouped_studies.issubset(recent_studies):
                raise ValueError("Independence groups must refer to recent historical studies")
            if any(not set(group.evidence_ids).issubset(evidence) for group in self.independence.groups):
                raise ValueError("Research groups reference missing evidence")
            if self.independence.coverage_complete and grouped_studies != recent_studies:
                raise ValueError("Complete independence coverage must account for every recent study")
        return self


class ScoreComponent(Contract):
    name: ScoreName
    weight: int = Field(ge=1, le=100, strict=True)
    value: float | None = Field(default=None, ge=0, le=100)
    reason: Identifier | None = None

    @model_validator(mode="after")
    def unknown_is_explained(self) -> Self:
        if self.value is None and self.reason is None:
            raise ValueError("Unknown score component requires a reason")
        return self


class CandidateAssessment(Contract):
    methodology_version: MethodologyVersion = METHODOLOGY_VERSION
    candidate_id: Identifier
    input_hash: Digest
    historical_snapshot_id: Identifier
    baseline_years: tuple[Year, ...] = Field(min_length=3, max_length=3)
    recent_years: tuple[Year, ...] = Field(min_length=3, max_length=3)
    baseline_studies: int = Field(ge=0, strict=True)
    recent_studies: int = Field(ge=0, strict=True)
    active_recent_years: int = Field(ge=0, le=3, strict=True)
    first_observed_year: Year | None
    observation_label: Literal["appearance_in_observed_corpus", "observed_growth_comparison", "no_observations"]
    smoothed_growth: float = Field(ge=0)
    raw_growth: float | None = Field(default=None, ge=0)
    recent_theil_sen_slope: float
    growth_confirmed: bool = Field(strict=True)
    category: Category
    quality: Quality
    confidence: Literal["high", "medium", "low"]
    components: tuple[ScoreComponent, ...] = Field(min_length=5, max_length=5)
    priority_score: float | None = Field(default=None, ge=0, le=100)
    relative_growth: RelativeGrowth | None = Field(default=None, exclude_if=lambda value: value is None)
    signal_priority: float | None = Field(default=None, ge=0, le=100, exclude_if=lambda value: value is None)
    priority_lower_bound: float = Field(ge=0, le=100)
    priority_upper_bound: float = Field(ge=0, le=100)
    gate_failures: tuple[Identifier, ...]
    limitations: tuple[Identifier, ...]

    @property
    def assessment_hash(self) -> str:
        return content_hash(self)


def _round(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _evaluate_v3_0(data: AssessmentInput) -> CandidateAssessment:
    """Frozen 3.0 arithmetic for existing archives; never use for new runs."""
    history = data.history
    baseline, recent = history.observations[-6:-3], history.observations[-3:]
    baseline_count = sum(item.count for item in baseline)
    recent_count = sum(item.count for item in recent)
    active = sum(item.count > 0 for item in recent)
    growth = (recent_count + 1) / (baseline_count + 1)
    slope = float(median((right.count - left.count) / (right.year - left.year)
                         for index, left in enumerate(recent) for right in recent[index + 1:]))
    complete = history.coverage.complete_history and not history.date_conflicts
    supported_roles = {claim.role for claim in data.claims if claim.support == "supported"}
    failures: list[str] = []
    checks = (
        (complete, "incomplete_or_incomparable_history"),
        (recent_count >= 10, "fewer_than_ten_recent_studies"),
        (active >= 2, "fewer_than_two_active_recent_years"),
        (growth >= 1.5, "growth_below_1_5"),
        (slope > 0, "nonpositive_recent_slope"),
        (data.candidate.specificity == "specific_technology", "candidate_not_specific"),
        ({"problem", "advantage", "case"}.issubset(supported_roles), "unsupported_card_fields"),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    growth_confirmed = not failures
    limitations: list[str] = []
    values: list[float | None] = [None] * 5
    reasons: list[str | None] = [None] * 5
    if complete:
        values[0] = min(100.0, max(0.0, 50 * math.log2(growth)))
        values[1] = 100 * active / 3
    else:
        reasons[0] = reasons[1] = "incomplete_or_incomparable_history"
        limitations.append("counts_and_growth_describe_only_observed_partial_history")
    novelty = data.novelty
    if novelty and novelty.earlier_analogues_checked and novelty.terminology_changes_checked:
        values[2] = float(NOVELTY_SCORES[novelty.kind])
    else:
        reasons[2] = "novelty_not_verified"
    independence = data.independence
    if independence and independence.coverage_complete:
        values[3] = min(100.0, 25.0 * len(independence.groups))
    else:
        reasons[3] = "independent_groups_unknown_or_incomplete"
    if data.application:
        values[4] = float(APPLICATION_SCORES[data.application.kind])
    else:
        reasons[4] = "application_evidence_not_verified"
    if values[2] is None:
        limitations.append("novelty_not_verified")
    if values[3] is None:
        limitations.append("independent_groups_unknown_or_incomplete")
    if values[4] is None:
        limitations.append("application_evidence_not_verified")
    if not history.earlier_search_complete:
        limitations.append("first_observation_is_not_worldwide_invention_date")
    if history.date_conflicts:
        limitations.append("unresolved_publication_date_conflicts")
    if growth_confirmed and values[2] is not None and values[2] >= 50:
        category: Category = "confirmed_trend"
    elif growth_confirmed and novelty and values[2] == 0:
        category = "renewed_interest"
    elif complete and novelty and values[2] == 0:
        category = "established_topic"
    else:
        category = "early_signal"
    if growth_confirmed and values[2] is None:
        failures.append("novelty_not_verified")
    known_sum = sum(weight * value / 100 for weight, value in zip(WEIGHTS, values, strict=True) if value is not None)
    unknown_weight = sum(weight for weight, value in zip(WEIGHTS, values, strict=True) if value is None)
    score = _round(known_sum) if unknown_weight == 0 else None
    names: tuple[ScoreName, ...] = ("growth", "persistence", "novelty", "independence", "application")
    components = tuple(ScoreComponent(name=name, weight=weight,
                                     value=_round(value) if value is not None else None, reason=reason)
                       for name, weight, value, reason in zip(names, WEIGHTS, values, reasons, strict=True))
    known_years = [item.year for item in history.observations if item.count]
    first_observed = history.first_observed_year if history.first_observed_year is not None else min(known_years, default=None)
    quality: Quality = ("insufficient_data" if recent_count == 0 else "complete" if complete else "partial")
    confidence: Literal["high", "medium", "low"] = (
        "high" if growth_confirmed and unknown_weight == 0 else "medium" if growth_confirmed else "low"
    )
    observation_label: Literal["appearance_in_observed_corpus", "observed_growth_comparison", "no_observations"] = (
        "no_observations" if not any(item.count for item in history.observations)
        else "appearance_in_observed_corpus" if baseline_count == 0
        else "observed_growth_comparison"
    )
    return CandidateAssessment(
        methodology_version="3.0.0",
        candidate_id=data.candidate.candidate_id, input_hash=content_hash(data),
        historical_snapshot_id=history.snapshot_id,
        baseline_years=tuple(item.year for item in baseline), recent_years=tuple(item.year for item in recent),
        baseline_studies=baseline_count, recent_studies=recent_count, active_recent_years=active,
        first_observed_year=first_observed, observation_label=observation_label,
        smoothed_growth=growth, raw_growth=recent_count / baseline_count if baseline_count else None,
        recent_theil_sen_slope=slope, growth_confirmed=growth_confirmed, category=category, quality=quality,
        confidence=confidence, components=components, priority_score=score,
        priority_lower_bound=_round(known_sum), priority_upper_bound=_round(known_sum + unknown_weight),
        gate_failures=tuple(failures), limitations=tuple(limitations),
    )


def _evaluate_v3_1(data: AssessmentInput, *, version: MethodologyVersion = METHODOLOGY_VERSION) -> CandidateAssessment:
    """Evaluate the requested version without silently reclassifying saved results.

    In 3.1, a positive three-point median slope is insufficient: the most recent
    observation must not fall, and neither transition may fall. Unknown evidence
    remains unknown; early_signal is a reviewed, recent, low-volume watchlist
    hypothesis, never the catch-all category. Scores are prioritization heuristics,
    not calibrated probabilities or proof of independent replication.
    """
    previous = _evaluate_v3_0(data)
    if version == "3.0.0":
        return previous
    if version != "3.1.0":
        raise ValueError("Unsupported methodology version")
    history = data.history
    recent = history.observations[-3:]
    transitions = tuple(right.count - left.count for left, right in zip(recent, recent[1:], strict=False))
    complete = history.coverage.complete_history and not history.date_conflicts
    roles = {claim.role for claim in data.claims if claim.support == "supported"}
    specific = data.candidate.specificity == "specific_technology"
    supported = {"problem", "advantage", "case"}.issubset(roles)
    failures = list(previous.gate_failures)
    limitations = list(previous.limitations)
    if any(change < 0 for change in transitions):
        failures.append("recent_decline_or_transient_burst")
    if not supported:
        limitations.append("card_claims_require_semantic_evidence_review")
    growth_confirmed = (previous.growth_confirmed and all(change >= 0 for change in transitions))
    verified_novelty = previous.components[2].value
    first = previous.first_observed_year
    observed_age = history.as_of.year - first if first is not None else None
    if not specific:
        category: Category = "unassessed_cluster"
    elif not complete or not supported or verified_novelty is None:
        category = "insufficient_evidence"
    elif verified_novelty == 0:
        category = "renewed_interest" if growth_confirmed else "established_topic"
    elif transitions[-1] < 0:
        category = "transient_burst" if transitions[0] > 0 else "declining"
    elif growth_confirmed:
        category = "confirmed_trend"
    elif (verified_novelty >= 75 and 2 <= previous.recent_studies < 10
          and previous.baseline_studies == 0 and observed_age is not None and observed_age <= 3
          and history.earlier_search_complete and transitions[-1] > 0
          and all(change >= 0 for change in transitions)):
        category = "early_signal"
        limitations.append("reviewed_low_volume_hypothesis_not_confirmed_trend")
    else:
        category = "insufficient_evidence"
    # Persistence measures both activity and direction. Merely being non-empty
    # during a burst does not make a time series persistent.
    components = list(previous.components)
    if complete:
        persistence = 100 * (previous.active_recent_years / 3) * sum(change >= 0 for change in transitions) / 2
        components[1] = ScoreComponent(name="persistence", weight=WEIGHTS[1], value=_round(persistence))
    if data.independence and data.independence.method_version == "openalex-author-components-team-diversity/2.0.0":
        components[3] = ScoreComponent(name="independence", weight=WEIGHTS[3], value=None,
                                       reason="team_diversity_is_not_independent_replication")
        limitations.append("team_diversity_is_not_independent_replication")
    known_sum = sum(item.weight * item.value / 100 for item in components if item.value is not None)
    unknown_weight = sum(item.weight for item in components if item.value is None)
    limitations.extend(("score_is_uncalibrated_priority_not_weak_signal_probability",
                        "growth_is_not_normalized_to_field_exposure",
                        "score_bounds_are_missing_component_bounds_not_confidence_interval"))
    confidence = ("high" if growth_confirmed and unknown_weight == 0 else
                  "medium" if growth_confirmed and verified_novelty is not None else "low")
    return CandidateAssessment.model_validate(previous.model_dump(mode="python") | {
        "methodology_version": version, "category": category, "growth_confirmed": growth_confirmed,
        "components": tuple(components), "priority_score": _round(known_sum) if unknown_weight == 0 else None,
        "priority_lower_bound": _round(known_sum), "priority_upper_bound": _round(known_sum + unknown_weight),
        "confidence": confidence, "gate_failures": tuple(dict.fromkeys(failures)),
        "limitations": tuple(dict.fromkeys(limitations)),
    })


def _evaluate_v3_2(data: AssessmentInput) -> CandidateAssessment:
    """Field-relative, age-gated hypotheses; author novelty never becomes expert review.

    Bibliometric excess growth and weak-signal hypotheses are different decisions.
    Small early candidates can enter a labelled hypothesis list even when their
    Poisson interval includes one, but never receive confirmed-growth status.
    """
    previous = _evaluate_v3_1(data, version="3.1.0")
    history = data.history
    counts = tuple(item.count for item in history.observations)
    recent = counts[-3:]
    transitions = tuple(right - left for left, right in zip(recent, recent[1:], strict=False))
    complete = history.coverage.complete_history and not history.date_conflicts
    specific = data.candidate.specificity == "specific_technology"
    supported = {"problem", "advantage", "case"}.issubset(
        claim.role for claim in data.claims if claim.support == "supported")
    relative = normalize_growth(counts, data.field_exposure, plan_hash=data.candidate.plan_hash,
        years=tuple(item.year for item in history.observations)) if data.field_exposure is not None else None
    exposure_available = relative is not None and relative.status == "available"
    observed_growth = bool(exposure_available and relative and relative.observed_growth
                           and all(change >= 0 for change in transitions))
    normalized_growth = bool(complete and specific and supported and previous.recent_studies >= 10
        and previous.active_recent_years >= 2 and relative and relative.excess_growth_supported
        and all(change >= 0 for change in transitions))
    verified_novelty = previous.components[2].value
    author_hypothesis = bool(data.source_novelty)
    novelty_available = verified_novelty > 0 if verified_novelty is not None else author_hypothesis
    first = previous.first_observed_year
    age = history.as_of.year - first if first is not None else None
    earlier_complete = history.earlier_search_complete and data.antecedents is not None
    eligible = complete and specific and supported and exposure_available and earlier_complete and novelty_available
    weak_age = age is not None and 1 <= age <= 3
    emerging_age = age is not None and 1 <= age <= 6
    weak_novelty = verified_novelty >= 75 if verified_novelty is not None else author_hypothesis
    weak = bool(eligible and weak_age and weak_novelty and 2 <= previous.recent_studies < 10
                and previous.baseline_studies == 0 and observed_growth)
    emerging = bool(eligible and emerging_age and normalized_growth)
    if not specific:
        category: Category = "unassessed_cluster"
    elif not complete or not supported:
        category = "insufficient_evidence"
    elif verified_novelty == 0:
        category = "renewed_interest" if normalized_growth else "established_topic"
    elif transitions[-1] < 0:
        category = "transient_burst" if transitions[0] > 0 else "declining"
    elif weak:
        category = "early_signal" if verified_novelty is not None and verified_novelty >= 75 else "weak_signal_candidate"
    elif emerging:
        category = "confirmed_trend" if verified_novelty is not None and verified_novelty >= 50 else "emerging_candidate"
    else:
        category = "insufficient_evidence"
    failures: list[str] = []
    checks = (
        (complete, "incomplete_or_incomparable_history"),
        (specific, "candidate_not_specific"),
        (supported, "unsupported_card_fields"),
        (exposure_available, "field_exposure_unavailable"),
        (earlier_complete, "earlier_search_not_complete"),
        (novelty_available, "no_reviewed_or_archived_novelty_evidence"),
        (emerging_age, "first_observation_not_recent"),
        (all(change >= 0 for change in transitions), "recent_decline_or_transient_burst"),
        (observed_growth, "no_observed_field_relative_growth"),
        (previous.recent_studies >= 2, "fewer_than_two_recent_studies"),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    if previous.recent_studies >= 10 and not normalized_growth:
        failures.append("field_relative_growth_not_statistically_supported")
    if 2 <= previous.recent_studies < 10 and not weak_age:
        failures.append("weak_signal_first_observation_older_than_three_years")
    if relative is not None and relative.reason:
        failures.append(relative.reason)
    limitations = [item for item in previous.limitations if item != "growth_is_not_normalized_to_field_exposure"]
    if relative is not None:
        limitations.extend(relative.limitations)
    else:
        limitations.append("field_exposure_not_collected")
    limitations.extend(("signal_priority_is_uncalibrated_heuristic_not_probability",
                        "legacy_priority_components_use_absolute_counts_signal_priority_uses_relative_growth",
                        "age_is_first_observation_in_frozen_queries_not_worldwide_invention_date"))
    if author_hypothesis:
        limitations.append("archived_author_novelty_assertion_is_hypothesis_not_independent_review")
    if weak:
        limitations.append("sparse_weak_signal_hypothesis_does_not_require_statistically_confirmed_growth")
    signal_priority = None
    if (weak or emerging) and relative is not None and age is not None:
        ratio = relative.raw_ratio if relative.raw_ratio is not None else relative.smoothed_ratio
        growth_component = min(1.0, max(0.0, math.log2(max(ratio or 0, 1.0)) / 3))
        persistence = previous.active_recent_years / 3 * sum(change >= 0 for change in transitions) / 2
        earlyness = min(1.0, max(0.0, (7 - age) / 6))
        # Specificity is an admission gate (1 here), never a compensating bonus.
        signal_priority = _round(100 * earlyness * (growth_component * persistence) ** (1 / 3))
    quality = previous.quality if exposure_available else "partial" if previous.recent_studies else "insufficient_data"
    return CandidateAssessment.model_validate(previous.model_dump(mode="python") | {
        "methodology_version": "3.2.0", "category": category, "growth_confirmed": normalized_growth,
        "relative_growth": relative, "signal_priority": signal_priority, "quality": quality,
        "confidence": "medium" if normalized_growth and verified_novelty is not None else "low",
        "gate_failures": tuple(dict.fromkeys(failures)), "limitations": tuple(dict.fromkeys(limitations)),
    })


def evaluate_candidate(data: AssessmentInput, *, version: MethodologyVersion = METHODOLOGY_VERSION) -> CandidateAssessment:
    """Version dispatch preserves the exact 3.0/3.1 scientific archive semantics."""
    from app.pilot.evidence_versions import novelty_method, primary_method

    if version != "3.4.0" and data.publication_status_revisions:
        raise ValueError("Publication status context requires methodology 3.4")
    if version == "3.4.0" and any(item.method_version != primary_method(version) for item in data.primary_observations):
        raise ValueError("Primary observation extraction version differs from the methodology")
    if version == "3.4.0" and any(item.method_version != novelty_method(version) for item in data.source_novelty):
        raise ValueError("Author novelty extraction version differs from the methodology")
    if version in {"3.0.0", "3.1.0"}:
        return _evaluate_v3_1(data, version=version)
    if version == "3.2.0":
        return _evaluate_v3_2(data)
    if version == "3.3.0":
        return _evaluate_v3_3(data)
    if version == "3.4.0":
        # Formulae and stage gates are unchanged; source admission, sentence
        # boundaries and claim grounding are separately versioned and replayed.
        previous = _evaluate_v3_3(data, history_resolves_antecedents=True)
        return CandidateAssessment.model_validate(previous.model_dump(mode="python")
                                                   | {"methodology_version": "3.4.0"})
    raise ValueError("Unsupported methodology version")


def _evaluate_v3_3(data: AssessmentInput, *, history_resolves_antecedents: bool = False) -> CandidateAssessment:
    """Observe a specific primary result before annual growth can be measured.

    Mature and bibliometrically confirmed stages retain the 3.2 calculations.
    One archived result may enter the explicit low-confidence hypothesis lane;
    it never fills a missing year, proves novelty, or confirms publication growth.
    """
    previous = _evaluate_v3_2(data)
    result = previous.model_dump(mode="python") | {"methodology_version": "3.3.0"}
    # Existing positive decisions and observed maturity/decline are not replaced
    # by a recent paper, a permissive rule or the absence of a field denominator.
    if previous.category not in {"insufficient_evidence", "transient_burst", "declining"}:
        return CandidateAssessment.model_validate(result)
    supported_cases = {identifier for claim in data.claims if claim.role == "case" and claim.support == "supported"
                       for identifier in claim.evidence_ids}
    primary = tuple(item for item in data.primary_observations
                    if item.result.evidence_id in supported_cases
                    and item.result.study_id not in data.history.date_conflicts)
    first_years = [item.publication_year for item in primary]
    if previous.first_observed_year is not None:
        first_years.append(previous.first_observed_year)
    # In 3.4 the history already incorporates every admissible older observation
    # against the complete saved status context. Its immutable search bundle may
    # still document an earlier source later withdrawn in another snapshot.
    if (not history_resolves_antecedents and data.antecedents is not None
            and data.antecedents.earliest_observed_year is not None):
        first_years.append(data.antecedents.earliest_observed_year)
    first = min(first_years, default=None)
    age = data.history.as_of.year - first if first is not None else None
    novelty = previous.components[2].value
    recent = data.history.observations[-3:]
    historical_ids = {identifier for year in recent for identifier in year.study_ids}
    # This is only a guard against calling an already large observed corpus a
    # sparse signal. The bounded primary packet is not reported as a total.
    observed_ids = historical_ids | {item.result.study_id for item in primary}
    current_primary = any(item.publication_year == data.history.as_of.year for item in primary)
    annual_dips = sum(right.count < left.count for left, right in zip(recent, recent[1:], strict=False))
    concerning_decline = previous.category in {"transient_burst", "declining"}
    weak = bool(data.candidate.specificity == "specific_technology" and primary
                and age is not None and 0 <= age <= 3 and len(observed_ids) < 10
                and (novelty is None or novelty >= 75)
                and not (data.history.coverage.complete_history and previous.baseline_studies > 0)
                and (not concerning_decline or current_primary and annual_dips == 1)
                and (not data.history.coverage.complete_history or annual_dips <= 1))
    if not weak:
        failures = list(previous.gate_failures)
        if not primary:
            failures.append("no_supported_primary_result_observation")
        if primary and (age is None or not 0 <= age <= 3):
            failures.append("primary_observation_has_known_older_antecedents")
        result["gate_failures"] = tuple(dict.fromkeys(failures))
        return CandidateAssessment.model_validate(result)
    limitations = list(previous.limitations) + list(previous.gate_failures)
    limitations.extend(("single_primary_result_is_observation_not_confirmed_weak_signal",
                        "primary_observations_do_not_change_completed_year_counts",
                        "primary_observation_priority_is_fixed_watchlist_order_not_growth_score",
                        "novelty_and_independent_replication_require_separate_review"))
    if any(item.publication_year == data.history.as_of.year for item in primary):
        limitations.append("current_year_primary_result_has_no_comparable_full_year_growth")
    if annual_dips:
        limitations.append("completed_year_decline_remains_a_concern_despite_new_primary_observation")
    if any(item.result_kind in {"theoretical", "computational"} for item in primary):
        limitations.append("computational_or_theoretical_result_is_not_experimental_demonstration")
    if not data.history.coverage.complete_history:
        limitations.append("primary_hypothesis_does_not_require_complete_history")
    result.update(category="weak_signal_candidate", confidence="low", growth_confirmed=False,
                  first_observed_year=first, signal_priority=10.0, quality="partial",
                  gate_failures=(), limitations=tuple(dict.fromkeys(limitations)))
    if previous.observation_label == "no_observations":
        result["observation_label"] = "appearance_in_observed_corpus"
    return CandidateAssessment.model_validate(result)


def rank_candidates(assessments: tuple[CandidateAssessment, ...], *, confirmed_only: bool = True,
                    limit: int = 15) -> tuple[CandidateAssessment, ...]:
    """Stable ranking by conservative score bound; absent evidence cannot boost rank."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 15:
        raise ValueError("Ranking limit must be an integer between 1 and 15")
    require_unique(tuple(item.candidate_id for item in assessments), "ranked candidates")
    candidates = [item for item in assessments if not confirmed_only or item.category == "confirmed_trend"]
    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return tuple(sorted(candidates, key=lambda item: (
        -(item.signal_priority if item.signal_priority is not None else -1.0) if item.methodology_version in {"3.2.0", "3.3.0", "3.4.0"} else -item.priority_lower_bound,
        confidence_order[item.confidence],
        -item.recent_studies if item.methodology_version == "3.0.0" else 0, item.candidate_id,
    ))[:limit])


class AssessmentArtifact(Contract):
    """Portable numerical proof: import recomputes metrics from saved membership.

The enclosing result references assessment.assessment_hash. The snapshot store
must additionally verify that these study IDs really belong to the declared
snapshot; self-contained arithmetic cannot authenticate external documents.
"""
    inputs: AssessmentInput
    assessment: CandidateAssessment

    @model_validator(mode="after")
    def recompute_before_accepting(self) -> Self:
        expected = evaluate_candidate(self.inputs, version=self.assessment.methodology_version)
        if expected != self.assessment:
            raise ValueError("Saved assessment does not reproduce from its frozen inputs")
        return self
