"""Strict v1 contracts for non-bibliographic evidence.

These models validate structure and arithmetic consistency. They do not certify
the truth of a source, its license, or an analyst's technology association.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.input_safety import is_safe_http_url
from app.pilot.contracts import Category

Digest = str
SourceKind = Literal["user", "scientific", "arxiv", "wordstat", "cordis", "investment_csv"]
SourceState = Literal["not_configured", "not_requested", "ready", "partial", "stale", "auth_error",
                      "rate_limited", "unavailable", "invalid_data"]
CoverageState = Literal["complete", "partial", "unknown"]
Retention = Literal["local_allowed", "extract_only", "unknown"]
ExportRight = Literal["share_allowed", "local_only", "unknown"]


def validate_import_export_right(export_right: ExportRight, license_ref: str | None) -> None:
    """Validate transfer choice before an adapter persists any source bytes."""
    if export_right == "share_allowed":
        if (not isinstance(license_ref, str) or not license_ref.strip() or len(license_ref) > 500
                or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in license_ref)):
            raise ValueError("Shared export needs an explicit rights reference")
    elif export_right != "local_only" or license_ref is not None:
        raise ValueError("Imported source rights must be explicit")
IdentityState = Literal["proposed", "confirmed", "ambiguous", "rejected"]
Relation = Literal["develops", "uses", "researches", "finances_specific_project", "mentioned"]
SearchValueState = Literal["observed", "missing", "suppressed", "invalid"]
SearchNormalization = Literal["usable", "unknown_unit", "quantized_zero", "inconsistent"]
SearchState = Literal["unavailable", "needs_review", "insufficient_comparison", "observed_zero",
                      "new_in_comparison", "low_volume_change", "sustained_growth", "declining",
                      "one_period_spike", "possible_seasonality", "flat_or_mixed"]
FundingState = Literal["unavailable", "needs_review", "multiple_units", "single_unit",
                       "none_in_observed_scope", "insufficient_coverage"]
PositiveInt = int


def _digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _decimal(value: str, *, maximum: Decimal | None = None, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,20}(?:\.[0-9]{1,20})?", value):
        raise ValueError("A bounded decimal string is required")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("Invalid decimal") from None
    if not number.is_finite() or number < 0 or (positive and not number) or maximum is not None and number > maximum:
        raise ValueError("Decimal is outside its allowed range")
    return number


def _signed_decimal(value: str) -> Decimal:
    if not isinstance(value, str) or not re.fullmatch(r"-?[0-9]{1,30}(?:\.[0-9]{1,20})?", value):
        raise ValueError("A bounded signed decimal string is required")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("Non-finite metric")
    return result


def _unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


class SignalContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    signal_schema_version: Literal[1] = 1

    @field_validator("*", mode="after")
    @classmethod
    def valid_scalar(cls, value: object) -> object:
        if isinstance(value, str):
            if not value.strip() or len(value) > 10_000 or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
                raise ValueError("Blank, oversized or control-bearing text")
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("A timezone-aware timestamp is required")
            return value.astimezone(timezone.utc)
        return value


class QueryTerm(SignalContract):
    text: str = Field(max_length=400)
    language: Literal["ru", "en", "other"]
    role: Literal["technology", "problem", "application", "practical_interest", "exclusion"]
    origin: Literal["user", "scientific", "wordstat", "arxiv", "cordis", "investment_csv", "ai_proposed"]
    status: IdentityState = "proposed"
    confirmed_at: datetime | None = None
    source_artifact_hash: Digest | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.status == "confirmed" and self.confirmed_at is None:
            raise ValueError("Confirmed query terms need a confirmation date")
        if self.source_artifact_hash is not None and not _digest(self.source_artifact_hash):
            raise ValueError("Invalid term source digest")
        return self


class QueryProfile(SignalContract):
    profile_id: UUID
    version: PositiveInt = Field(ge=1, strict=True)
    original_query: str = Field(max_length=500)
    definition: str = Field(max_length=3000)
    exclusions: tuple[str, ...] = Field(default=(), max_length=40)
    terms: tuple[QueryTerm, ...] = Field(default=(), max_length=50)
    primary_phrase: str | None = Field(default=None, max_length=400)
    region_ids: tuple[int, ...] = Field(default=(), max_length=100)
    devices: tuple[str, ...] = Field(default=(), max_length=3)
    scientific_plan_hash: Digest | None = None
    supersedes_hash: Digest | None = None
    confirmed_at: datetime | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        for name in ("scientific_plan_hash", "supersedes_hash"):
            value = getattr(self, name)
            if value is not None and not _digest(value):
                raise ValueError(f"Invalid {name}")
        if any(type(region) is not int or region < 0 for region in self.region_ids):
            raise ValueError("Invalid Wordstat regions")
        _unique(self.region_ids, "regions")
        _unique(self.devices, "devices")
        if any(device not in {"all", "desktop", "phone", "tablet"} for device in self.devices) or (
                "all" in self.devices and len(self.devices) > 1):
            raise ValueError("Invalid Wordstat device")
        _unique(tuple((term.language, term.role, term.text.casefold()) for term in self.terms), "query terms")
        if self.primary_phrase is not None and not self.primary_phrase.strip():
            raise ValueError("Blank primary phrase")
        if self.primary_phrase is not None:
            chosen = [term for term in self.terms if term.text.casefold() == self.primary_phrase.casefold()
                      and term.role in {"technology", "practical_interest"} and term.status == "confirmed"]
            if len(chosen) != 1 or self.confirmed_at is None or chosen[0].confirmed_at is None or chosen[0].confirmed_at > self.confirmed_at:
                raise ValueError("Primary phrase needs one confirmed term and profile confirmation")
        if any(not item.strip() or len(item) > 400 or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"}
                                                        for char in item) for item in self.exclusions):
            raise ValueError("Invalid exclusion")
        _unique(tuple(item.casefold() for item in self.exclusions), "exclusions")
        return self


class TechnologyConcept(SignalContract):
    concept_id: UUID
    label: str = Field(max_length=300)
    definition: str = Field(max_length=3000)
    problem_refs: tuple[UUID, ...] = Field(default=(), max_length=20)
    aliases: tuple[QueryTerm, ...] = Field(default=(), max_length=40)
    identity_status: IdentityState
    confirmed_at: datetime | None = None
    provenance_hashes: tuple[Digest, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.identity_status == "confirmed" and self.confirmed_at is None:
            raise ValueError("Confirmed concept identity needs a review date")
        _unique(self.problem_refs, "problem references")
        _unique(self.provenance_hashes, "provenance")
        if any(not _digest(value) for value in self.provenance_hashes):
            raise ValueError("Invalid concept provenance")
        return self


class SourceStatus(SignalContract):
    source: SourceKind
    state: SourceState
    reason_code: str = Field(max_length=80)
    safe_message: str = Field(max_length=500)
    attempted_at: datetime | None = None
    last_success_at: datetime | None = None


class SourceSnapshot(SignalContract):
    source: SourceKind
    adapter_version: str = Field(max_length=80)
    request_hash: Digest
    query_profile_hash: Digest
    observed_at: datetime
    available_at: datetime
    coverage: CoverageState
    comparable: bool = Field(strict=True)
    raw_hash: Digest | None = None
    extract_hash: Digest | None = None
    source_url: str | None = None
    retention: Retention = "unknown"
    export_right: ExportRight = "unknown"
    license_ref: str | None = Field(default=None, max_length=500)
    limitations: tuple[str, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        for name in ("request_hash", "query_profile_hash", "raw_hash", "extract_hash"):
            value = getattr(self, name)
            if value is not None and not _digest(value):
                raise ValueError(f"Invalid {name}")
        if self.available_at > self.observed_at:
            raise ValueError("Availability cannot follow observation")
        if self.source_url is not None and not is_safe_http_url(self.source_url):
            raise ValueError("Unsafe source URL")
        if self.retention == "extract_only" and self.raw_hash is not None:
            raise ValueError("Raw source is retained despite extract-only rights")
        if self.export_right == "share_allowed" and self.license_ref is None:
            raise ValueError("Shared export needs an explicit rights reference")
        if self.coverage != "complete" and self.comparable:
            raise ValueError("Incomplete coverage cannot be comparable")
        if self.raw_hash is None and self.extract_hash is None:
            raise ValueError("A snapshot needs raw or normalized source bytes")
        return self


class SearchObservation(SignalContract):
    series_id: Digest
    query_profile_hash: Digest
    snapshot_hash: Digest
    phrase: str = Field(max_length=400)
    phrase_role: Literal["technology", "problem", "practical_interest"]
    matching_mode: str = Field(max_length=80)
    region_ids: tuple[int, ...] = Field(default=(), max_length=100)
    devices: tuple[str, ...] = Field(default=(), max_length=3)
    period_start: date
    period_end: date
    calendar: Literal["provider_month"] = "provider_month"
    is_complete_period: bool = Field(strict=True)
    count: int | None = Field(default=None, ge=0, strict=True)
    value_status: SearchValueState
    share_raw: str | None = Field(default=None, max_length=80)
    share_unit: Literal["fraction", "percent", "unknown"] = "unknown"
    share_fraction: str | None = Field(default=None, max_length=80)
    share_precision: int | None = Field(default=None, ge=0, le=20, strict=True)
    normalization_status: SearchNormalization = "unknown_unit"
    observed_at: datetime
    available_at: datetime
    row_locator: str = Field(max_length=150)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(value) for value in (self.series_id, self.query_profile_hash, self.snapshot_hash)):
            raise ValueError("Invalid search reference")
        if self.period_start.day != 1 or self.period_end < self.period_start or self.period_end.month != self.period_start.month or self.period_end.year != self.period_start.year:
            raise ValueError("A monthly observation needs one calendar month")
        next_month = date(self.period_start.year + (self.period_start.month == 12), self.period_start.month % 12 + 1, 1)
        if self.is_complete_period and self.period_end.toordinal() + 1 != next_month.toordinal():
            raise ValueError("A complete monthly observation must end at month end")
        if self.available_at > self.observed_at:
            raise ValueError("Availability cannot follow observation")
        if self.value_status == "observed" and self.count is None:
            raise ValueError("Observed search volume needs a count")
        if self.value_status != "observed" and self.count is not None:
            raise ValueError("Missing, suppressed or invalid volume must not be zero-filled")
        if self.share_raw is not None:
            _decimal(self.share_raw)
            if self.share_unit == "percent" and _decimal(self.share_raw) > 100:
                raise ValueError("Search percentage exceeds 100")
        if self.share_fraction is not None:
            _decimal(self.share_fraction, maximum=Decimal(1))
            if self.share_raw is None:
                raise ValueError("Normalized search share needs its source value")
        if self.share_unit == "unknown" and self.share_fraction is not None:
            raise ValueError("Unknown share unit cannot yield a normalized fraction")
        if self.share_fraction is not None and self.share_raw is not None:
            expected = _decimal(self.share_raw) / (100 if self.share_unit == "percent" else 1)
            if expected != _decimal(self.share_fraction):
                raise ValueError("Normalized search share differs from its source")
        if self.normalization_status == "usable":
            if self.value_status != "observed" or self.share_fraction is None or self.share_unit == "unknown":
                raise ValueError("Usable normalized search needs a measured share")
            if self.count and not _decimal(self.share_fraction, maximum=Decimal(1)):
                raise ValueError("Positive count and rounded-zero share are not a usable rate")
        if self.normalization_status == "quantized_zero" and not (self.count and self.share_raw is not None and _decimal(self.share_raw) == 0):
            raise ValueError("Quantized zero requires positive count and observed zero share")
        if self.normalization_status == "quantized_zero" and self.share_fraction is not None:
            raise ValueError("A rounded-zero share is unavailable for normalized comparisons")
        if self.count == 0 and self.share_fraction is not None and _decimal(self.share_fraction) > 0:
            raise ValueError("Zero count contradicts positive share")
        _unique(self.region_ids, "regions")
        _unique(self.devices, "devices")
        if any(type(region) is not int or region < 0 for region in self.region_ids):
            raise ValueError("Invalid region")
        if any(device not in {"all", "desktop", "phone", "tablet"} for device in self.devices) or (
                "all" in self.devices and len(self.devices) > 1):
            raise ValueError("Invalid Wordstat device")
        return self


class WordstatImportReceipt(SignalContract):
    kind: Literal["dynamics", "top"]
    query_profile_hash: Digest
    snapshot_hash: Digest
    raw_hash: Digest
    mapping_hash: Digest
    row_count: int = Field(ge=1, le=10_000, strict=True)
    accepted_count: int = Field(ge=0, le=10_000, strict=True)
    rejected_count: int = Field(ge=0, le=10_000, strict=True)
    unselected_count: int = Field(ge=0, le=10_000, strict=True)
    observation_hashes: tuple[Digest, ...] = Field(default=(), max_length=10_000)
    term_hashes: tuple[Digest, ...] = Field(default=(), max_length=20)
    top_counts: tuple[int, ...] = Field(default=(), max_length=20)
    rejected_rows: tuple[str, ...] = Field(default=(), max_length=10_000)
    completed_at: datetime

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(digest) for digest in (self.query_profile_hash, self.snapshot_hash,
                                                  self.raw_hash, self.mapping_hash,
                                                  *self.observation_hashes, *self.term_hashes)):
            raise ValueError("Invalid Wordstat import reference")
        if self.row_count != self.accepted_count + self.rejected_count + self.unselected_count:
            raise ValueError("Wordstat row accounting does not close")
        if self.accepted_count != len(self.observation_hashes) + len(self.term_hashes):
            raise ValueError("Accepted rows do not match retained objects")
        if self.rejected_count != len(self.rejected_rows):
            raise ValueError("Rejections do not match their row numbers")
        if self.kind == "dynamics" and (self.term_hashes or self.unselected_count):
            raise ValueError("Dynamics cannot contain top-only terms")
        if self.kind == "top" and self.observation_hashes:
            raise ValueError("Top phrases are not time series")
        if len(self.top_counts) != len(self.term_hashes) or any(type(count) is not int or count < 0
                                                               for count in self.top_counts):
            raise ValueError("Top phrase counts are missing or invalid")
        _unique(self.observation_hashes, "observations")
        _unique(self.term_hashes, "top terms")
        return self


class ArxivVersion(SignalContract):
    arxiv_id: str = Field(max_length=200)
    version: int = Field(ge=1, le=9999, strict=True)
    revision_id: Digest
    raw_hash: Digest
    published_at: datetime
    updated_at: datetime
    status: Literal["preprint_unreviewed", "withdrawn_reported"]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not _digest(self.revision_id) or not _digest(self.raw_hash) or self.updated_at < self.published_at:
            raise ValueError("Invalid arXiv version provenance")
        return self


class ArxivImportReceipt(SignalContract):
    query_profile_hash: Digest
    snapshot_hash: Digest
    raw_hash: Digest
    as_of: date
    scanned: int = Field(ge=0, le=1000, strict=True)
    rejected: int = Field(ge=0, le=1000, strict=True)
    truncated: bool = Field(strict=True)
    version_hashes: tuple[Digest, ...] = Field(default=(), max_length=1000)
    selected_revision_ids: tuple[Digest, ...] = Field(default=(), max_length=1000)
    completed_at: datetime

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(value) for value in (self.query_profile_hash, self.snapshot_hash, self.raw_hash,
                                                *self.version_hashes, *self.selected_revision_ids)):
            raise ValueError("Invalid arXiv import reference")
        if self.scanned < self.rejected + len(self.version_hashes):
            raise ValueError("arXiv entry accounting is inconsistent")
        _unique(self.version_hashes, "arXiv versions")
        _unique(self.selected_revision_ids, "selected arXiv revisions")
        return self


class SearchMetric(SignalContract):
    series_id: Digest | None = None
    snapshot_hash: Digest | None = None
    policy_hash: Digest
    decision_at: datetime
    knowledge_cutoff: datetime
    state: SearchState
    recent_count: int | None = Field(default=None, ge=0, strict=True)
    base_count: int | None = Field(default=None, ge=0, strict=True)
    recent_share: str | None = None
    base_share: str | None = None
    yoy_share_change: str | None = None
    yoy_count_change: str | None = None
    persistence_numerator: int | None = Field(default=None, ge=0, le=3, strict=True)
    positive_recent_months: int | None = Field(default=None, ge=0, le=3, strict=True)
    low_volume: bool = Field(default=False, strict=True)
    used_observation_hashes: tuple[Digest, ...] = Field(default=(), max_length=100)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(value) for value in (self.series_id, self.snapshot_hash, self.policy_hash,
                                                *self.used_observation_hashes) if value is not None):
            raise ValueError("Invalid search metric reference")
        if self.decision_at > self.knowledge_cutoff:
            raise ValueError("Future knowledge cannot decide an earlier search metric")
        for value in (self.recent_share, self.base_share, self.yoy_share_change, self.yoy_count_change):
            if value is not None:
                _signed_decimal(value)
        if self.state == "sustained_growth" and (self.yoy_share_change is None or self.persistence_numerator is None):
            raise ValueError("Growth requires computed normalized evidence")
        _unique(self.used_observation_hashes, "search metric observations")
        return self


class FundingAmount(SignalContract):
    event_kind: Literal["equity_round", "grant_project"]
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    amount_scope: Literal["linked_events_not_attributed"] = "linked_events_not_attributed"
    disclosed_total: str
    largest_event_amount: str
    largest_event_share: str

    @model_validator(mode="after")
    def consistent(self) -> Self:
        total = _decimal(self.disclosed_total, positive=True)
        largest = _decimal(self.largest_event_amount, positive=True)
        share = _decimal(self.largest_event_share, maximum=Decimal(1))
        if largest > total or share != (largest / total).quantize(Decimal("0.000000000001")):
            raise ValueError("Funding amount concentration is inconsistent")
        return self


class FundingMetric(SignalContract):
    source: Literal["cordis", "investment_csv"]
    event_kind: Literal["equity_round", "grant_project"]
    snapshot_hash: Digest | None = None
    policy_hash: Digest
    decision_at: datetime
    knowledge_cutoff: datetime
    window_days: Literal[90, 365]
    state: FundingState
    unit_kind: Literal["companies", "projects"]
    unit_count: int = Field(ge=0, strict=True)
    event_hashes: tuple[Digest, ...] = Field(default=(), max_length=1000)
    proposed_count: int = Field(default=0, ge=0, strict=True)
    uncertain_date_count: int = Field(default=0, ge=0, strict=True)
    undisclosed_count: int = Field(default=0, ge=0, strict=True)
    latest_event_period: str | None = Field(default=None, pattern=r"^[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?$")
    amounts: tuple[FundingAmount, ...] = Field(default=(), max_length=30)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(value) for value in (self.snapshot_hash, self.policy_hash, *self.event_hashes)
               if value is not None):
            raise ValueError("Invalid funding metric reference")
        if self.decision_at > self.knowledge_cutoff:
            raise ValueError("Future knowledge cannot decide an earlier funding metric")
        if self.event_kind == "grant_project" and self.unit_kind != "projects" or (
                self.event_kind == "equity_round" and self.unit_kind != "companies"):
            raise ValueError("Funding units cannot be mixed")
        if self.state == "multiple_units" and self.unit_count < 2 or self.state == "single_unit" and self.unit_count != 1:
            raise ValueError("Funding state conflicts with unit count")
        _unique(self.event_hashes, "funding metric events")
        _unique(tuple((item.event_kind, item.currency) for item in self.amounts), "funding amount groups")
        return self


class CapitalDescription(SignalContract):
    source: Literal["cordis", "investment_csv"]
    source_event_id: str = Field(max_length=200)
    event_id: UUID
    source_hash: Digest
    title: str = Field(max_length=500)
    description: str | None = Field(default=None, max_length=5000)
    source_url: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not _digest(self.source_hash) or self.source_url is not None and not is_safe_http_url(self.source_url):
            raise ValueError("Invalid capital description provenance")
        return self


class CapitalImportReceipt(SignalContract):
    source: Literal["cordis", "investment_csv"]
    query_profile_hash: Digest
    snapshot_hash: Digest
    raw_hash: Digest
    participant_raw_hash: Digest | None = None
    mapping_hash: Digest
    row_count: int = Field(ge=1, le=10_000, strict=True)
    rejected_count: int = Field(ge=0, le=10_000, strict=True)
    event_hashes: tuple[Digest, ...] = Field(default=(), max_length=10_000)
    description_hashes: tuple[Digest, ...] = Field(default=(), max_length=10_000)
    rejected_rows: tuple[str, ...] = Field(default=(), max_length=10_000)
    completed_at: datetime

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(value) for value in (self.query_profile_hash, self.snapshot_hash, self.raw_hash,
                                                self.participant_raw_hash, self.mapping_hash,
                                                *self.event_hashes, *self.description_hashes) if value is not None):
            raise ValueError("Invalid capital import reference")
        if self.row_count != self.rejected_count + len(self.event_hashes) or (
                self.rejected_count != len(self.rejected_rows) or len(self.event_hashes) != len(self.description_hashes)):
            raise ValueError("Capital import row accounting does not close")
        _unique(self.event_hashes, "capital events")
        return self


class CapitalEvent(SignalContract):
    event_id: UUID
    source_event_id: str | None = Field(default=None, max_length=200)
    source: Literal["cordis", "investment_csv"]
    source_hash: Digest
    kind: Literal["equity_round", "grant_project", "debt", "acquisition", "other"]
    status: Literal["announced", "confirmed", "cancelled", "rumored", "unknown"]
    recipient_id: str | None = Field(default=None, max_length=200)
    recipient_name: str | None = Field(default=None, max_length=300)
    beneficiary_ids: tuple[str, ...] = Field(default=(), max_length=200)
    coordinator_id: str | None = Field(default=None, max_length=200)
    project_id: str | None = Field(default=None, max_length=200)
    announced_at: date | None = None
    agreement_at: date | None = None
    closed_at: date | None = None
    event_date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    event_month: str | None = Field(default=None, pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")
    event_year: int | None = Field(default=None, ge=1000, le=9999, strict=True)
    observed_at: datetime
    available_at: datetime
    amount: str | None = Field(default=None, max_length=80)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    amount_status: Literal["disclosed", "undisclosed", "estimated", "raw_zero_unverified"]
    amount_kind: Literal["round_amount", "eu_contribution", "debt_amount", "purchase_price"]
    funder_ids: tuple[str, ...] = Field(default=(), max_length=100)
    programme: str | None = Field(default=None, max_length=300)
    export_right: ExportRight = "unknown"
    license_ref: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not _digest(self.source_hash):
            raise ValueError("Invalid funding source")
        if self.available_at > self.observed_at:
            raise ValueError("Availability cannot follow observation")
        if self.export_right == "share_allowed" and self.license_ref is None:
            raise ValueError("Shared export needs an explicit rights reference")
        if self.kind == "grant_project":
            if self.source != "cordis" or not self.project_id or self.amount_kind != "eu_contribution":
                raise ValueError("CORDIS grants require one project and EU contribution units")
        elif self.kind == "equity_round":
            if not self.recipient_id or not self.recipient_name or self.amount_kind != "round_amount":
                raise ValueError("Equity rounds require a company and round amount units")
        if self.amount_status == "disclosed" and (self.amount is None or self.currency is None):
            raise ValueError("Disclosed amounts need a value and currency")
        if self.amount_status == "undisclosed" and self.amount is not None:
            raise ValueError("Undisclosed amount is unknown, not zero")
        if self.amount_status == "raw_zero_unverified" and self.amount != "0":
            raise ValueError("Unverified raw zero must remain a literal zero")
        if self.amount is not None:
            _decimal(self.amount)
        if self.amount is None and self.currency is not None and self.amount_status != "undisclosed":
            raise ValueError("Currency without an amount is ambiguous")
        exact = any((self.announced_at, self.agreement_at, self.closed_at))
        if self.event_date_precision == "day":
            if not exact or self.event_month is not None or self.event_year is not None:
                raise ValueError("Exact date precision requires a date only")
        elif self.event_date_precision == "month":
            if exact or self.event_month is None or self.event_year is not None:
                raise ValueError("Monthly precision must preserve the month without inventing a day")
        elif self.event_date_precision == "year":
            if exact or self.event_year is None or self.event_month is not None:
                raise ValueError("Yearly precision must preserve the year only")
        elif exact or self.event_month is not None or self.event_year is not None:
            raise ValueError("Unknown precision cannot carry a fabricated date")
        _unique(self.beneficiary_ids, "beneficiaries")
        _unique(self.funder_ids, "funders")
        return self


class TechnologyAssociation(SignalContract):
    concept_id: UUID
    subject_id: str = Field(max_length=200)
    subject_kind: Literal["event", "project", "organisation"]
    relation: Relation
    status: Literal["proposed", "confirmed", "rejected", "uncertain"]
    relation_at_event: Literal["supported", "unknown", "not_applicable"] = "unknown"
    evidence_hashes: tuple[Digest, ...] = Field(default=(), max_length=30)
    reviewer: str | None = Field(default=None, max_length=200)
    reviewed_at: datetime | None = None
    attributable_amount: str | None = Field(default=None, max_length=80)
    attribution_basis_hash: Digest | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if any(not _digest(value) for value in self.evidence_hashes):
            raise ValueError("Invalid association evidence")
        if self.status == "confirmed" and (not self.evidence_hashes or not self.reviewer or self.reviewed_at is None):
            raise ValueError("A confirmed relation needs evidence and a review")
        if self.attributable_amount is not None:
            _decimal(self.attributable_amount)
            if self.status != "confirmed" or self.attribution_basis_hash is None or not _digest(self.attribution_basis_hash):
                raise ValueError("Monetary attribution needs a confirmed basis")
        elif self.attribution_basis_hash is not None:
            raise ValueError("Orphan attribution basis")
        return self


class AssociationReview(SignalContract):
    association_hash: Digest
    decision: Literal["confirmed", "rejected", "uncertain"]
    reviewer: str = Field(max_length=200)
    reviewed_at: datetime
    evidence_hashes: tuple[Digest, ...] = Field(default=(), max_length=30)
    supersedes_hash: Digest | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        for digest in (self.association_hash, self.supersedes_hash, *self.evidence_hashes):
            if digest is not None and not _digest(digest):
                raise ValueError("Invalid review reference")
        if self.decision == "confirmed" and not self.evidence_hashes:
            raise ValueError("Confirmation requires evidence")
        return self


class SignalFinding(SignalContract):
    finding_id: UUID
    concept_id: UUID
    origins: tuple[SourceKind, ...] = Field(min_length=1, max_length=5)
    scientific_candidate_id: str | None = Field(default=None, max_length=2048)
    scientific_category: Category | None = None
    search_state: SearchState | None = None
    funding_state: FundingState | None = None
    observation_hashes: tuple[Digest, ...] = Field(default=(), max_length=1000)
    metric_hashes: tuple[Digest, ...] = Field(default=(), max_length=10)
    queue: Literal["attention", "deferred", "watch", "market_only", "insufficient_data", "rejected"]
    rule_id: str = Field(max_length=100)
    explanation: str = Field(max_length=2000)
    next_check: str = Field(max_length=1000)
    limitations: tuple[str, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        _unique(self.origins, "origins")
        _unique(self.observation_hashes, "observation references")
        _unique(self.metric_hashes, "metric references")
        if any(not _digest(value) for value in (*self.observation_hashes, *self.metric_hashes)):
            raise ValueError("Invalid observation reference")
        if self.scientific_category is not None and self.scientific_candidate_id is None:
            raise ValueError("A scientific category requires a real scientific candidate")
        if self.queue == "attention" and not self.observation_hashes and self.scientific_candidate_id is None:
            raise ValueError("An attention finding needs observed evidence")
        return self


class SignalProfile(SignalContract):
    profile_id: UUID
    policy_version: Literal["multisource/1.0.0"] = "multisource/1.0.0"
    policy_hash: Digest
    query_profile_hash: Digest
    import_receipt_hashes: tuple[Digest, ...] = Field(default=(), max_length=30)
    decision_at: datetime
    collection_finished_at: datetime
    knowledge_cutoff: datetime
    source_snapshot_hashes: tuple[Digest, ...] = Field(default=(), max_length=100)
    concept_artifact_hashes: tuple[Digest, ...] = Field(default=(), max_length=100)
    association_artifact_hashes: tuple[Digest, ...] = Field(default=(), max_length=500)
    metric_artifact_hashes: tuple[Digest, ...] = Field(default=(), max_length=300)
    findings: tuple[SignalFinding, ...] = Field(default=(), max_length=100)
    attention_ids: tuple[UUID, ...] = Field(default=(), max_length=15)
    watch_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    deferred_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    rejected_ids: tuple[UUID, ...] = Field(default=(), max_length=100)
    base_result_hash: Digest | None = None
    base_result_run_id: str | None = Field(default=None, max_length=80)
    limitations: tuple[str, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        for digest in (self.policy_hash, self.query_profile_hash, self.base_result_hash,
                       *self.import_receipt_hashes,
                       *self.source_snapshot_hashes, *self.concept_artifact_hashes,
                       *self.association_artifact_hashes, *self.metric_artifact_hashes):
            if digest is not None and not _digest(digest):
                raise ValueError("Invalid profile reference")
        if not self.decision_at <= self.collection_finished_at == self.knowledge_cutoff:
            raise ValueError("Profile cutoff must follow collection")
        ids = tuple(item.finding_id for item in self.findings)
        _unique(ids, "findings")
        _unique(self.source_snapshot_hashes, "source snapshots")
        _unique(self.import_receipt_hashes, "import receipts")
        _unique(self.concept_artifact_hashes, "concept artifacts")
        _unique(self.association_artifact_hashes, "association artifacts")
        _unique(self.metric_artifact_hashes, "metric artifacts")
        if set(self.metric_artifact_hashes) != {digest for finding in self.findings for digest in finding.metric_hashes}:
            raise ValueError("Profile metric closure is incomplete")
        if any(item.scientific_category is not None for item in self.findings) and self.base_result_hash is None:
            raise ValueError("Scientific assessment needs its immutable base result")
        if (self.base_result_hash is None) != (self.base_result_run_id is None):
            raise ValueError("Scientific result hash and run ID must travel together")
        queues = ((self.attention_ids, "attention"), (self.deferred_ids, "deferred"), (self.watch_ids, "watch"),
                  (self.rejected_ids, "rejected"))
        listed: list[UUID] = []
        by_id = {item.finding_id: item for item in self.findings}
        for selected, expected in queues:
            _unique(selected, expected)
            for identifier in selected:
                if identifier not in by_id or by_id[identifier].queue != expected:
                    raise ValueError("Finding queue and profile index disagree")
            listed.extend(selected)
        _unique(tuple(listed), "profile queues")
        if set(listed) != {item.finding_id for item in self.findings
                           if item.queue in {"attention", "deferred", "watch", "rejected"}}:
            raise ValueError("Published finding queues are incomplete")
        return self


class WatchRecord(SignalContract):
    """A user's dated watch choice; it never changes the frozen finding."""
    profile_hash: Digest
    concept_id: UUID
    action: Literal["watch", "unwatch"]
    recorded_at: datetime
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not _digest(self.profile_hash):
            raise ValueError("Invalid watched profile reference")
        return self


class SignalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["multisource/1.0.0"]
    growth_threshold: str
    min_recent_count: int = Field(ge=1, le=1000, strict=True)
    min_persistence_numerator: int = Field(ge=1, le=3, strict=True)
    seasonal_lower: str
    seasonal_upper: str
    spike_multiplier: str
    attention_limit: int = Field(ge=1, le=15, strict=True)
    concept_limit: int = Field(ge=1, le=100, strict=True)
    wordstat_seed_limit: int = Field(ge=1, le=3, strict=True)
    wordstat_suggestion_limit: int = Field(ge=1, le=20, strict=True)
    wordstat_phrase_limit: int = Field(ge=1, le=15, strict=True)
    wordstat_attempt_limit_per_run: int = Field(ge=1, le=40, strict=True)
    wordstat_attempt_limit_per_hour: int = Field(ge=1, le=80, strict=True)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        growth = _decimal(self.growth_threshold, maximum=Decimal(10), positive=True)
        try:
            lower = Decimal(self.seasonal_lower)
        except (InvalidOperation, TypeError):
            raise ValueError("Invalid seasonal lower bound") from None
        upper = _decimal(self.seasonal_upper, maximum=Decimal(10))
        spike = _decimal(self.spike_multiplier, maximum=Decimal(100), positive=True)
        if not lower.is_finite() or not -1 < lower < 0 or not lower < growth or upper < growth or spike <= 1:
            raise ValueError("Invalid signal policy thresholds")
        return self


def load_policy(path: Path | None = None) -> tuple[SignalPolicy, str]:
    """The exact validated policy bytes are part of every published profile."""
    target = path or Path(__file__).with_name("policy-v1.json")
    with target.open("rb") as handle:
        data = handle.read(8193)
    if len(data) > 8192:
        raise ValueError("Signal policy exceeds 8 KiB")
    policy = SignalPolicy.model_validate_json(data)
    return policy, hashlib.sha256(data).hexdigest()
