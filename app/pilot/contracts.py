"""Immutable v3 analysis contracts. Counts and quotations retain their provenance.

These models validate structure and internal consistency, not scientific truth.
`supported` requires a grounding check by the caller; an LLM's assertion alone is
not that check. Hashes identify content and do not authenticate its author.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.input_safety import is_safe_http_url

SCHEMA_VERSION: Literal[3] = 3
MethodologyVersion = Literal["3.0.0", "3.1.0", "3.2.0", "3.3.0", "3.4.0"]
# Low-level compatibility defaults remain 3.1; the current workflow explicitly
# selects 3.4, whose source/exposure inputs are absent from historical callers.
METHODOLOGY_VERSION: MethodologyVersion = "3.1.0"
SIGNAL_METHODOLOGY_VERSION: MethodologyVersion = "3.4.0"
ScopeRuleVersion = Literal["exact-scope-phrases/1", "all-query-concepts-local/2"]
LEGACY_SCOPE_RULE_VERSION: ScopeRuleVersion = "exact-scope-phrases/1"
SCOPE_RULE_VERSION: ScopeRuleVersion = "all-query-concepts-local/2"
Category = Literal["confirmed_trend", "early_signal", "renewed_interest", "established_topic",
                   "unassessed_cluster", "insufficient_evidence", "off_scope", "transient_burst", "declining",
                   "weak_signal_candidate", "emerging_candidate"]
Identifier = Annotated[str, Field(min_length=1, max_length=2048)]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Year = Annotated[int, Field(ge=1000, le=9999, strict=True)]
Count = Annotated[int, Field(ge=0, strict=True)]
Source = Literal["openalex", "crossref", "epo", "arxiv", "report"]
Quality = Literal["complete", "partial", "insufficient_data"]
RunState = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
StageName = Literal["plan", "discovery", "relevance", "candidates", "history", "evidence", "publish"]


def content_hash(value: BaseModel | Mapping[str, Any] | tuple[Any, ...] | list[Any]) -> str:
    """Hash canonical JSON; no unstable Python repr or locale-dependent encoding."""
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_unique(values: tuple[Any, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[3] = SCHEMA_VERSION

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Blank strings are not valid identifiers or content")
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Timestamp requires a timezone")
            return value.astimezone(timezone.utc)
        return value


class SearchQuery(Contract):
    source: Source
    text: str = Field(min_length=1, max_length=1000)
    purpose: Literal["discovery", "history", "enrichment"] = "discovery"


class QueryLimits(Contract):
    discovery_documents: int = Field(default=3000, ge=1, le=10000, strict=True)
    historical_documents_per_candidate: int = Field(default=3000, ge=1, le=3000, strict=True)
    new_historical_documents: int = Field(default=20000, ge=1, le=20000, strict=True)
    llm_calls: int = Field(default=24, ge=0, le=24, strict=True)
    input_tokens: int = Field(default=200000, ge=0, le=200000, strict=True)
    output_tokens: int = Field(default=30000, ge=0, le=30000, strict=True)


class QueryPlan(Contract):
    original_query: str = Field(min_length=1, max_length=500)
    language: Literal["ru", "en"]
    definition: str = Field(min_length=1, max_length=3000)
    english_query: str = Field(min_length=1, max_length=1000)
    subdirections: tuple[Identifier, ...] = Field(min_length=1, max_length=8)
    synonyms: tuple[Identifier, ...] = Field(default=(), max_length=40)
    exclusions: tuple[Identifier, ...] = Field(default=(), max_length=40)
    queries: tuple[SearchQuery, ...] = Field(min_length=1, max_length=32)
    completed_years: tuple[Year, ...] = Field(min_length=6, max_length=10)
    as_of: date
    planner_version: Identifier
    model_version: Identifier | None = None
    limits: QueryLimits = Field(default_factory=QueryLimits)

    @field_validator("original_query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        value = unicodedata.normalize("NFKC", value).strip()
        if any(unicodedata.category(character) in ("Cc", "Cf", "Cs") for character in value):
            raise ValueError("Query contains control characters")
        return value

    @model_validator(mode="after")
    def valid_plan(self) -> Self:
        validate_years(self.completed_years, self.as_of)
        for field in ("subdirections", "synonyms", "exclusions"):
            values = getattr(self, field)
            require_unique(tuple(value.strip().casefold() for value in values), field)
        require_unique(tuple((query.source, query.purpose, query.text) for query in self.queries), "queries")
        return self

    @property
    def plan_hash(self) -> str:
        return content_hash(self)


def validate_years(years: tuple[int, ...], as_of: date) -> None:
    if tuple(range(years[0], years[-1] + 1)) != years:
        raise ValueError("Years must be sorted, distinct and consecutive")
    if years[-1] >= as_of.year:
        raise ValueError("Historical comparisons cannot include the current or a future year")


class Coverage(Contract):
    """A completed request is not necessarily a complete/comparable history."""
    source: Source
    purpose: Literal["discovery", "history", "enrichment"]
    query_hash: Digest
    state: Literal["complete", "partial", "unavailable"]
    requested_years: tuple[Year, ...] = Field(min_length=1, max_length=1000)
    completed_years: tuple[Year, ...] = ()
    pagination_exhausted: bool = Field(strict=True)
    comparable: bool = Field(strict=True)
    scanned_records: Count = 0
    accepted_records: Count = 0
    rejected_records: Count = 0
    unresolved_records: Count = 0
    limit_reached: bool = Field(default=False, strict=True)
    reasons: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def consistent_coverage(self) -> Self:
        require_unique(self.requested_years, "requested years")
        require_unique(self.completed_years, "completed years")
        if not set(self.completed_years).issubset(self.requested_years):
            raise ValueError("Completed years must belong to the request")
        if self.scanned_records != self.accepted_records + self.rejected_records + self.unresolved_records:
            raise ValueError("Scanned record accounting is inconsistent")
        if self.state == "complete":
            if (not self.pagination_exhausted or self.limit_reached or self.unresolved_records
                    or set(self.completed_years) != set(self.requested_years)):
                raise ValueError("Complete coverage requires exhaustive, resolved pagination for every year")
        elif not self.reasons:
            raise ValueError("Incomplete coverage requires an explanation")
        return self

    @property
    def complete_history(self) -> bool:
        return (self.purpose == "history" and self.source == "openalex"
                and self.state == "complete" and self.comparable)


class DocumentRevisionRef(Contract):
    revision_id: Identifier
    study_id: Identifier
    source: Source
    source_id: Identifier
    text_hash: Digest
    observed_at: datetime
    publication_year: Year | None = None
    publicly_available_at: date | None = None


class CorpusSnapshot(Contract):
    snapshot_id: Identifier
    plan_hash: Digest
    purpose: Literal["discovery", "history", "enrichment"]
    created_at: datetime
    as_of: date
    documents: tuple[DocumentRevisionRef, ...] = Field(max_length=100000)
    coverage: tuple[Coverage, ...] = Field(min_length=1, max_length=1000)
    normalizer_version: Identifier
    deduplication_version: Identifier

    @model_validator(mode="after")
    def consistent_snapshot(self) -> Self:
        require_unique(tuple(document.revision_id for document in self.documents), "revision IDs")
        if any(coverage.purpose != self.purpose for coverage in self.coverage):
            raise ValueError("Snapshot and coverage purposes differ")
        if any(document.publicly_available_at and document.publicly_available_at > self.as_of
               for document in self.documents):
            raise ValueError("As-of snapshot contains a document not yet publicly available")
        return self

    @property
    def snapshot_hash(self) -> str:
        return content_hash(self)


class Candidate(Contract):
    candidate_id: Identifier
    plan_hash: Digest
    label: str = Field(min_length=1, max_length=200)
    definition: str = Field(min_length=1, max_length=3000)
    synonyms: tuple[Identifier, ...] = Field(default=(), max_length=40)
    exclusions: tuple[Identifier, ...] = Field(default=(), max_length=40)
    admission_rule_version: Identifier
    admission_rule_hash: Digest
    discovery_snapshot_id: Identifier
    discovery_study_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=10000)
    specificity: Literal["specific_technology", "broad_topic", "uncertain"]
    # Preserve the serialized bytes and hashes of existing candidates/results.
    # A new scope rule is explicit; loading a historical object never upgrades it.
    scope_rule_version: ScopeRuleVersion = Field(default=LEGACY_SCOPE_RULE_VERSION,
        exclude_if=lambda value: value == LEGACY_SCOPE_RULE_VERSION)

    @model_validator(mode="after")
    def distinct_members(self) -> Self:
        require_unique(self.discovery_study_ids, "candidate studies")
        return self


class Evidence(Contract):
    evidence_id: Identifier
    revision_id: Identifier
    study_id: Identifier
    source: Source
    source_url: str = Field(min_length=1, max_length=8192)
    retrieved_at: datetime
    text_hash: Digest
    text_field: Literal["title", "abstract", "full_text", "metadata"]
    start: Count
    end: int = Field(gt=0, strict=True)
    quote: str = Field(min_length=1, max_length=12000)
    external_ai_allowed: bool = Field(default=False, strict=True)

    @field_validator("source_url")
    @classmethod
    def safe_link(cls, value: str) -> str:
        # This validates a display link, not permission to fetch that host.
        if not is_safe_http_url(value):
            raise ValueError("Evidence requires an HTTP(S) URL without credentials")
        return value

    @model_validator(mode="after")
    def consistent_span(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("Quotation offsets must match its exact Unicode character length")
        return self


def verify_evidence_text(evidence: Evidence, revision_text: str) -> None:
    """Verify the exact archived field, never a translation or current web page."""
    digest = hashlib.sha256(revision_text.encode("utf-8")).hexdigest()
    if digest != evidence.text_hash or revision_text[evidence.start:evidence.end] != evidence.quote:
        raise ValueError(f"Evidence {evidence.evidence_id} does not match its archived text")


class Claim(Contract):
    claim_id: Identifier
    role: Literal["problem", "advantage", "case", "summary", "limitation", "novelty", "application"]
    text: str = Field(min_length=1, max_length=6000)
    support: Literal["supported", "unverified", "contradicted"]
    evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=50)
    kind: Literal["narrative", "numeric"] = "narrative"
    metric_refs: tuple[Identifier, ...] = Field(default=(), max_length=20)
    grounding_method: Identifier | None = None

    @model_validator(mode="after")
    def evidence_required(self) -> Self:
        require_unique(self.evidence_ids, "claim evidence")
        if self.support == "supported" and (not self.evidence_ids or not self.grounding_method):
            raise ValueError("Supported claims require evidence and a recorded grounding method")
        if self.kind == "numeric" and not self.metric_refs:
            raise ValueError("Numeric claims require reproducible metric references")
        return self


class AnalysisStage(Contract):
    run_id: Identifier
    stage: StageName
    attempt_id: Identifier
    state: RunState
    updated_at: datetime
    input_hash: Digest
    output_hash: Digest | None = None
    checkpoint_id: Identifier | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def stage_status(self) -> Self:
        if self.state == "succeeded" and self.output_hash is None:
            raise ValueError("Successful stages require a verified output hash")
        if self.state in ("failed", "cancelled", "interrupted") and not self.reason:
            raise ValueError("Non-successful terminal stages require a reason")
        return self


class AnalysisRun(Contract):
    run_id: Identifier
    plan_hash: Digest
    state: RunState
    created_at: datetime
    updated_at: datetime
    methodology_version: MethodologyVersion = METHODOLOGY_VERSION
    quality: Quality | None = None
    stages: tuple[AnalysisStage, ...] = Field(default=(), max_length=1000)
    result_id: Identifier | None = None
    cancellation_requested_at: datetime | None = None

    @model_validator(mode="after")
    def run_status(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("Run update predates creation")
        if any(stage.run_id != self.run_id for stage in self.stages):
            raise ValueError("Stage belongs to another run")
        require_unique(tuple((stage.stage, stage.attempt_id) for stage in self.stages), "stage attempts")
        if self.state == "succeeded" and (not self.result_id or self.quality is None):
            raise ValueError("Successful runs require a result and a separate quality assessment")
        if self.cancellation_requested_at and self.state == "succeeded":
            raise ValueError("A cancellation fence forbids publishing success")
        return self


class TrendCard(Contract):
    # Omit the absent field when serializing legacy cards: archived hashes and
    # review records must remain byte-for-byte reproducible.
    methodology_version: MethodologyVersion | None = Field(default=None, exclude_if=lambda value: value is None)
    candidate: Candidate
    category: Category
    quality: Quality
    claims: tuple[Claim, ...] = Field(max_length=40)
    evidence: tuple[Evidence, ...] = Field(max_length=200)
    historical_snapshot_id: Identifier | None = None
    assessment_hash: Digest | None = None
    limitations: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def card_references(self) -> Self:
        require_unique(tuple(item.evidence_id for item in self.evidence), "evidence IDs")
        require_unique(tuple(item.claim_id for item in self.claims), "claim IDs")
        evidence_ids = {item.evidence_id for item in self.evidence}
        if any(not set(claim.evidence_ids).issubset(evidence_ids) for claim in self.claims):
            raise ValueError("Claim references missing card evidence")
        if self.category == "confirmed_trend":
            supported_roles = {claim.role for claim in self.claims if claim.support == "supported"}
            if not {"problem", "advantage", "case"}.issubset(supported_roles):
                raise ValueError("Confirmed cards require supported problem, advantage and case")
            if (self.quality != "complete" or not self.historical_snapshot_id or not self.assessment_hash
                    or self.candidate.specificity != "specific_technology"):
                raise ValueError("Confirmed cards require a specific candidate and completed historical assessment")
        if self.quality != "complete" and not self.limitations:
            raise ValueError("Partial cards must explain their limitations")
        return self


class CandidateReviewState(Contract):
    """Last naming attempt, distinct from the scientific state of a passport.

    Stage hashes identify the owning run's immutable checkpoint; they are not
    proof of an external model's correctness or a replacement for quotations.
    """
    candidate_id: Identifier
    candidate_hash: Digest
    discovery_snapshot_id: Identifier
    source_run_id: Identifier
    attempt_ordinal: int = Field(ge=0, le=59, strict=True)
    stage: str = Field(pattern=r"^candidate_batch_outcome_[0-9]{1,2}$")
    stage_hash: Digest
    input_hash: Digest
    outcome: Literal["passport_created", "definition_rejected"]
    reason_code: Literal["off_scope_or_mixed", "scope_evidence_missing", "definition_rejected"] | None = None

    @model_validator(mode="after")
    def consistent_outcome(self) -> Self:
        if (self.outcome == "definition_rejected") != (self.reason_code is not None):
            raise ValueError("A rejected definition requires its safe rejection reason")
        if int(self.stage.rsplit("_", 1)[1]) > self.attempt_ordinal:
            raise ValueError("Candidate precedes its owning batch")
        return self


# One individual study per retained document of the largest corpus a plan may
# request, plus 2500 disjoint leaves of size >= 2.
QUEUE_LIMIT = 12_500


class AnalysisResult(Contract):
    result_id: Identifier
    run_id: Identifier
    query_plan: QueryPlan
    methodology_version: MethodologyVersion = METHODOLOGY_VERSION
    created_at: datetime
    quality: Quality
    # Up to 60 assessed mechanisms, each with discovery/history/antecedents/
    # optional patent provenance. Revisions and package bytes remain bounded.
    snapshots: tuple[CorpusSnapshot, ...] = Field(min_length=1, max_length=241)
    cards: tuple[TrendCard, ...] = Field(max_length=60)
    # Selection is distinct from scientific status: reviewing a sixteenth trend
    # must not erase a verified passport or demote its lifecycle to fit TOP-15.
    top_trend_ids: tuple[Identifier, ...] | None = Field(default=None, max_length=15,
                                                       exclude_if=lambda value: value is None)
    top_limit: int | None = Field(default=None, ge=1, le=15, strict=True,
                                  exclude_if=lambda value: value is None)
    # The earlier fixed 7500 assumed 5000 individual studies, which a full
    # 10 000-document corpus exceeds: a measured standard-profile run retained
    # 7 590 studies and queued 7 607 hypotheses, and the whole analysis was then
    # discarded at its last step, after sources, clustering and naming had
    # already been paid for. Archive bytes stay bounded independently.
    candidate_queue: tuple[Candidate, ...] = Field(default=(), max_length=QUEUE_LIMIT,
                                                  exclude_if=lambda value: not value)
    candidate_review_states: tuple[CandidateReviewState, ...] | None = Field(default=None, max_length=QUEUE_LIMIT,
        exclude_if=lambda value: value is None)
    limitations: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def result_references(self) -> Self:
        require_unique(tuple(snapshot.snapshot_id for snapshot in self.snapshots), "snapshot IDs")
        if self.methodology_version not in {"3.2.0", "3.3.0", "3.4.0"} and len(self.snapshots) > 100:
            raise ValueError("Legacy results may contain at most 100 snapshots")
        require_unique(tuple(card.candidate.candidate_id for card in self.cards), "result candidates")
        require_unique(tuple(card.candidate.candidate_id for card in self.cards)
                       + tuple(candidate.candidate_id for candidate in self.candidate_queue), "cards and review queue")
        if self.candidate_review_states is not None:
            if self.methodology_version != "3.4.0":
                raise ValueError("Naming-attempt provenance requires methodology 3.4")
            by_candidate = {item.candidate_id: item for item in self.candidate_queue}
            card_ids = {card.candidate.candidate_id for card in self.cards}
            by_candidate.update((card.candidate.candidate_id, card.candidate) for card in self.cards)
            require_unique(tuple(item.candidate_id for item in self.candidate_review_states), "candidate review states")
            require_unique(tuple((item.source_run_id, item.attempt_ordinal) for item in self.candidate_review_states),
                           "candidate attempt positions")
            for state in self.candidate_review_states:
                candidate = by_candidate.get(state.candidate_id)
                if (candidate is None or state.candidate_hash != content_hash(candidate)
                        or state.discovery_snapshot_id != candidate.discovery_snapshot_id
                        or state.outcome == "passport_created" and state.candidate_id not in card_ids):
                    raise ValueError("Candidate review state has no matching saved candidate")
        confirmed_ids = tuple(card.candidate.candidate_id for card in self.cards if card.category == "confirmed_trend")
        if len(confirmed_ids) > 15 and (self.methodology_version == "3.0.0" or self.top_trend_ids is None):
            raise ValueError("A result may contain at most 15 confirmed trends")
        if self.top_trend_ids is not None:
            require_unique(self.top_trend_ids, "TOP trend IDs")
            eligible_ids = (tuple(card.candidate.candidate_id for card in self.cards if card.category in
                            {"confirmed_trend", "early_signal", "weak_signal_candidate", "emerging_candidate"})
                            if self.methodology_version in {"3.2.0", "3.3.0", "3.4.0"} else confirmed_ids)
            if not set(self.top_trend_ids).issubset(eligible_ids):
                raise ValueError("TOP selection must refer to eligible cards")
        if self.methodology_version in {"3.2.0", "3.3.0", "3.4.0"}:
            if self.top_trend_ids is None or self.top_limit is None or len(self.top_trend_ids) > self.top_limit:
                raise ValueError("Signal results require an explicit bounded TOP selection")
        snapshots = {snapshot.snapshot_id: snapshot for snapshot in self.snapshots}
        if any(snapshot.plan_hash != self.query_plan.plan_hash for snapshot in self.snapshots):
            raise ValueError("Snapshot uses a different query plan")
        revision_definitions: dict[str, DocumentRevisionRef] = {}
        for snapshot in self.snapshots:
            for revision in snapshot.documents:
                if revision.revision_id in revision_definitions and revision_definitions[revision.revision_id] != revision:
                    raise ValueError("Conflicting definitions of the same immutable revision")
                revision_definitions[revision.revision_id] = revision
        revisions = {revision.revision_id: revision for snapshot in self.snapshots
                     for revision in snapshot.documents}
        for candidate in self.candidate_queue:
            discovery = snapshots.get(candidate.discovery_snapshot_id)
            if (candidate.plan_hash != self.query_plan.plan_hash or discovery is None
                    or discovery.purpose != "discovery"
                    or not set(candidate.discovery_study_ids).issubset(doc.study_id for doc in discovery.documents)):
                raise ValueError("Queued candidate has no matching discovery provenance")
        for card in self.cards:
            candidate = card.candidate
            if candidate.plan_hash != self.query_plan.plan_hash:
                raise ValueError("Candidate uses a different query plan")
            discovery = snapshots.get(candidate.discovery_snapshot_id)
            if discovery is None or discovery.purpose != "discovery":
                raise ValueError("Candidate has no discovery snapshot")
            if not set(candidate.discovery_study_ids).issubset(doc.study_id for doc in discovery.documents):
                raise ValueError("Candidate study is absent from discovery")
            if card.historical_snapshot_id:
                history = snapshots.get(card.historical_snapshot_id)
                if history is None or history.purpose != "history":
                    raise ValueError("Historical assessment has no historical snapshot")
                reference_coverage = tuple(item for item in history.coverage if item.source == "openalex")
                if card.category == "confirmed_trend":
                    if not reference_coverage or not all(item.complete_history for item in reference_coverage):
                        raise ValueError("Incomplete historical coverage cannot confirm a trend")
            for evidence in card.evidence:
                evidence_revision = revisions.get(evidence.revision_id)
                if (evidence_revision is None or evidence_revision.study_id != evidence.study_id
                        or evidence_revision.source != evidence.source):
                    raise ValueError("Evidence does not belong to a saved document revision")
        if self.quality == "complete" and any(card.quality != "complete" for card in self.cards):
            raise ValueError("A result containing partial cards must retain partial quality")
        if not self.cards and self.quality != "insufficient_data":
            raise ValueError("An empty result must report insufficient data")
        if self.quality != "complete" and not self.limitations:
            raise ValueError("Incomplete results require limitations")
        return self
