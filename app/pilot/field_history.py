"""Dated fixed-query OpenAlex exposure counts and conservative rate comparisons.

OpenAlex meta.count is an indexed-work exposure proxy, not the number of unique
research studies worldwide. Count uncertainty below is conditional Poisson
sampling uncertainty; it does not cover changing indexing, query recall, or the
work/study-family unit difference. No capped discovery corpus is a denominator.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, date, datetime
import hashlib
import json
import math
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.backend.contracts import SearchRequest, SourcePage
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.base import DocumentProvider
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.jobs import RunContext, TaskCancelled

if TYPE_CHECKING:
    from app.pilot.contracts import QueryPlan

EXPOSURE_METHOD: Literal["openalex-fixed-query-count/1.0.0"] = "openalex-fixed-query-count/1.0.0"
GROWTH_METHOD: Literal["conditional-poisson-exposure-ratio/1.0.0"] = "conditional-poisson-exposure-ratio/1.0.0"
LIMITATIONS = (
    "fixed_lexical_query_exposure_is_not_worldwide_field_coverage",
    "indexed_work_denominator_and_study_family_numerator_have_different_units",
    "interval_assumes_poisson_counts_and_does_not_cover_source_or_membership_bias",
    "current_index_counts_are_reconstructed_history_not_as_of_snapshots",
)
_EXPOSURE_WORKERS = 3


def _hash(value: BaseModel | dict) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def aware_timestamp(cls, value):
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Exposure timestamps require a timezone")
            return value.astimezone(UTC)
        return value


class FieldCountRequest(_Model):
    source: Literal["openalex"] = "openalex"
    query: str = Field(min_length=1, max_length=1000)
    from_date: date
    until_date: date
    max_results: Literal[1] = 1


class FieldCountPage(_Model):
    """Exact count metadata exposed by the normalized first provider page."""
    total_available: int | None = Field(default=None, ge=0, strict=True)
    scanned: int = Field(ge=0, le=1, strict=True)
    skipped: int = Field(ge=0, le=1, strict=True)
    exhausted: bool = Field(strict=True)
    source_ids: tuple[str, ...] = Field(default=(), max_length=1)

    @model_validator(mode="after")
    def valid_page(self) -> Self:
        if self.scanned != len(self.source_ids) + self.skipped:
            raise ValueError("Exposure page accounting mismatch")
        if self.total_available is not None and self.total_available < self.scanned:
            raise ValueError("Exposure total cannot be below scanned records")
        return self


class FieldYearExposure(_Model):
    year: int = Field(ge=1000, le=9999, strict=True)
    request: FieldCountRequest
    observed_at: datetime
    status: Literal["observed", "unavailable"]
    count: int | None = Field(default=None, ge=0, strict=True)
    page: FieldCountPage | None = None
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def consistent_observation(self) -> Self:
        if (self.request.from_date != date(self.year, 1, 1)
                or self.request.until_date != date(self.year, 12, 31)):
            raise ValueError("Exposure request must cover exactly one completed year")
        if self.year >= self.observed_at.year:
            raise ValueError("Exposure history cannot contain an unfinished year")
        if self.status == "observed":
            if (self.count is None or self.page is None or self.page.total_available != self.count
                    or self.reason is not None):
                raise ValueError("Observed count requires matching source total metadata")
        elif self.count is not None or self.reason is None:
            raise ValueError("Unavailable count must remain unknown with a reason")
        return self


class FieldExposure(_Model):
    method_version: Literal["openalex-fixed-query-count/1.0.0"] = EXPOSURE_METHOD
    source: Literal["openalex"] = "openalex"
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    fixed_query: str = Field(min_length=1, max_length=1000)
    observed_at: datetime
    years: tuple[FieldYearExposure, ...] = Field(min_length=6, max_length=10)
    provenance_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def consistent_snapshot(self) -> Self:
        years = tuple(item.year for item in self.years)
        if years != tuple(range(years[0], years[-1] + 1)):
            raise ValueError("Exposure years must be consecutive and unique")
        if any(item.request.query != self.fixed_query for item in self.years):
            raise ValueError("Every year must use the identical frozen field query")
        if any(item.observed_at > self.observed_at for item in self.years):
            raise ValueError("Exposure snapshot precedes a source observation")
        if self.provenance_hash != _hash(self.model_dump(mode="json", exclude={"provenance_hash"})):
            raise ValueError("Exposure provenance hash does not match source observations")
        return self

    @classmethod
    def create(cls, *, plan_hash: str, fixed_query: str, observed_at: datetime,
               years: tuple[FieldYearExposure, ...]) -> FieldExposure:
        # Canonical timestamps must be normalized before computing their hash.
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("Exposure timestamps require a timezone")
        observed_at = observed_at.astimezone(UTC)
        # Construct only to calculate the hash; the returned instance is fully validated.
        provisional = cls.model_construct(plan_hash=plan_hash, fixed_query=fixed_query,
                                           observed_at=observed_at, years=years, provenance_hash="0" * 64)
        values = provisional.model_dump(mode="json", exclude={"provenance_hash"})
        return cls.model_validate(values | {"provenance_hash": _hash(values)})


def verify_field_exposure(exposure: FieldExposure, plan: QueryPlan) -> FieldExposure:
    """Validate content hash and exact frozen plan, query and completed years."""
    verified = FieldExposure.model_validate(exposure.model_dump(mode="python"))
    if (verified.plan_hash != plan.plan_hash or verified.fixed_query != plan.english_query
            or tuple(item.year for item in verified.years) != plan.completed_years):
        raise ValueError("Field exposure belongs to a different query plan or historical window")
    return verified


def _collect_year_exposure(year: int, query: str, context: RunContext,
                           provider_factory: Callable[[str], DocumentProvider]) -> FieldYearExposure:
    """Collect one independent first-page count without mutating shared state."""
    context.check_cancelled()
    request_record = FieldCountRequest(query=query, from_date=date(year, 1, 1),
                                       until_date=date(year, 12, 31))
    provider = None
    pages = None
    page_record = None
    count = None
    reason = None
    try:
        request = SearchRequest(topic=query, source="openalex", from_date=date(year, 1, 1),
                                until_date=date(year, 12, 31), max_results=1)
        provider = provider_factory("openalex")
        pages = iter(provider.iter_pages(request, context.cancel_event))
        supplied = next(pages, None)
        context.check_cancelled()
        if supplied is None:
            raise BackendError("invalid_response", "Источник не сообщил метаданные объёма направления.")
        page = SourcePage.model_validate(supplied.model_dump() if isinstance(supplied, SourcePage) else supplied)
        if (page.scanned > 1 or not page.scanned and not page.exhausted
                or any(document.source != "openalex" for document in page.documents)):
            raise BackendError("invalid_response", "Нарушен источник или лимит запроса объёма направления.")
        page_record = FieldCountPage(total_available=page.total_available, scanned=page.scanned,
                                     skipped=page.skipped, exhausted=page.exhausted,
                                     source_ids=tuple(document.source_id for document in page.documents))
        if page.total_available is None:
            reason = "source_total_missing"
        elif not page.scanned and page.total_available > 0:
            reason = "empty_page_with_positive_source_total"
        else:
            count = page.total_available
    except CancelledError:
        raise TaskCancelled() from None
    except BackendError as error:
        reason = "source_" + error.code
    except CredentialUnavailable:
        reason = "credential_storage_unavailable"
    except (ValueError, TypeError):
        reason = "invalid_source_count_response"
    finally:
        if pages is not None and callable(close := getattr(pages, "close", None)):
            close()
        if provider is not None:
            provider.close()
    context.check_cancelled()
    return FieldYearExposure(year=year, request=request_record, observed_at=datetime.now(UTC),
        status="observed" if count is not None else "unavailable", count=count,
        page=page_record, reason=reason)


def collect_field_exposure(plan: QueryPlan, context: RunContext, credentials: CredentialStore, *,
                           provider_factory: Callable[[str], DocumentProvider] | None = None) -> FieldExposure:
    """Read only the first page's meta.count, once per completed year (at most 10).

    The search query and dates are frozen. A cap of one limits data transfer, not
    the source-reported total; missing totals and errors are recorded as unknown.
    No publication revisions or user library files are written here.
    """
    from app.pilot.sources import PublicationProviderSession

    context.check_cancelled()
    invalid_reason = ("query_outside_provider_request_limits"
                      if len(plan.english_query) > 500 or len(plan.english_query) < 2 else
                      "query_changes_under_provider_normalization"
                      if plan.english_query != plan.english_query.strip() else None)
    if invalid_reason is not None:
        observations = tuple(FieldYearExposure(year=year,
            request=FieldCountRequest(query=plan.english_query, from_date=date(year, 1, 1),
                                      until_date=date(year, 12, 31)),
            observed_at=datetime.now(UTC), status="unavailable", reason=invalid_reason)
            for year in plan.completed_years)
    else:
        factory: Callable[[str], DocumentProvider]
        if provider_factory is None:
            session = PublicationProviderSession(credentials)
            factory = session.provider
        else:
            session = None
            factory = provider_factory
        collected: list[FieldYearExposure] = []
        try:
            with ThreadPoolExecutor(max_workers=min(_EXPOSURE_WORKERS, len(plan.completed_years)),
                                    thread_name_prefix="field-exposure") as pool:
                for offset in range(0, len(plan.completed_years), _EXPOSURE_WORKERS):
                    context.check_cancelled()
                    batch = plan.completed_years[offset:offset + _EXPOSURE_WORKERS]
                    futures: list[Future[FieldYearExposure]] = [
                        pool.submit(_collect_year_exposure, year, plan.english_query, context, factory)
                        for year in batch
                    ]
                    try:
                        for future in futures:
                            completed = len(collected)
                            context.progress("history", "Собираем статистику направления по годам",
                                             completed, len(plan.completed_years))
                            collected.append(future.result())
                            context.progress("history", "Собираем статистику направления по годам",
                                             completed + 1, len(plan.completed_years))
                    except TaskCancelled:
                        context.cancel_event.set()
                        for future in futures:
                            future.cancel()
                        raise
        finally:
            if session is not None:
                session.close()
        observations = tuple(collected)
    return FieldExposure.create(plan_hash=plan.plan_hash, fixed_query=plan.english_query,
                                observed_at=datetime.now(UTC), years=observations)


class _CommonGrowth(TypedDict):
    exposure_hash: str
    baseline_studies: int
    recent_studies: int


class _BetaDistribution(Protocol):
    def ppf(self, q: float, a: int, b: int) -> float: ...


class RelativeGrowth(_Model):
    method_version: Literal["conditional-poisson-exposure-ratio/1.0.0"] = GROWTH_METHOD
    exposure_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["available", "unavailable"]
    reason: str | None = None
    baseline_studies: int = Field(ge=0, strict=True)
    recent_studies: int = Field(ge=0, strict=True)
    baseline_exposure: int | None = Field(default=None, ge=1, strict=True)
    recent_exposure: int | None = Field(default=None, ge=1, strict=True)
    raw_ratio: float | None = Field(default=None, ge=0)
    smoothed_ratio: float | None = Field(default=None, ge=0)
    ratio_lower_95: float | None = Field(default=None, ge=0)
    ratio_upper_95: float | None = Field(default=None, ge=0)
    recent_rates: tuple[float, ...] = Field(default=(), max_length=3)
    recent_share_non_decreasing: bool = False
    observed_growth: bool = False
    excess_growth_supported: bool = False
    limitations: tuple[str, ...] = LIMITATIONS


def normalize_growth(counts: tuple[int, ...], exposure: FieldExposure, *,
                     plan_hash: str, years: tuple[int, ...]) -> RelativeGrowth:
    """Compare adjacent final three-year study-family rates against fixed-query exposure.

    Exact conditional Poisson rate-ratio intervals avoid asymptotic certainty for
    sparse signals. They are descriptive uncertainty under stated assumptions,
    not a calibrated probability that a technology is a weak signal.
    """
    exposure = FieldExposure.model_validate(exposure.model_dump(mode="python"))
    if exposure.plan_hash != plan_hash or tuple(item.year for item in exposure.years) != years:
        raise ValueError("Exposure does not match the plan and historical years")
    if len(counts) != len(years) or any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Candidate counts must be nonnegative integers aligned with exposure years")
    baseline, recent = sum(counts[-6:-3]), sum(counts[-3:])
    common: _CommonGrowth = dict(exposure_hash=exposure.provenance_hash, baseline_studies=baseline, recent_studies=recent)
    if any(item.status != "observed" or item.count is None for item in exposure.years):
        return RelativeGrowth(**common, status="unavailable", reason="field_exposure_missing")
    field_counts = tuple(item.count if item.count is not None else 0 for item in exposure.years)
    if any(value == 0 for value in field_counts):
        return RelativeGrowth(**common, status="unavailable", reason="zero_field_exposure")
    if any(value > field for value, field in zip(counts, field_counts, strict=True)):
        return RelativeGrowth(**common, status="unavailable", reason="candidate_exceeds_reference_field_exposure")
    base_exposure, recent_exposure = sum(field_counts[-6:-3]), sum(field_counts[-3:])
    scale = base_exposure / recent_exposure
    rates = tuple(value / field for value, field in zip(counts[-3:], field_counts[-3:], strict=True))
    non_decreasing = all(right >= left or math.isclose(right, left, rel_tol=1e-12, abs_tol=0)
                         for left, right in zip(rates, rates[1:], strict=False))
    last_increase = rates[-1] > rates[-2] and not math.isclose(rates[-1], rates[-2], rel_tol=1e-12, abs_tol=0)
    beta = cast(_BetaDistribution, import_module("scipy.stats").beta)
    p_lower = float(beta.ppf(0.025, recent, baseline + 1)) if recent else 0.0
    # Use beta symmetry for the upper bound: subtracting a quantile very
    # close to one would amplify last-bit differences across platforms.
    upper_complement = float(beta.ppf(0.025, baseline, recent + 1)) if baseline else 0.0
    lower = scale * p_lower / (1 - p_lower) if p_lower < 1 else None
    upper = scale * (1 - upper_complement) / upper_complement if upper_complement > 0 else None
    # Portable replay records numerical evidence to 12 significant digits,
    # not implementation-specific final bits of SciPy's inverse beta function.
    lower = float(format(lower, ".12g")) if lower is not None else None
    upper = float(format(upper, ".12g")) if upper is not None else None
    # Zero observations offer no rate estimate even though smoothing has a value.
    if baseline + recent == 0:
        return RelativeGrowth(**common, status="unavailable", reason="no_candidate_observations",
                              baseline_exposure=base_exposure, recent_exposure=recent_exposure)
    observed_growth = bool((baseline == 0 or recent * scale / baseline >= 1.5)
                           and non_decreasing and last_increase and counts[-1] >= counts[-2])
    return RelativeGrowth(**common, status="available", baseline_exposure=base_exposure,
        recent_exposure=recent_exposure, raw_ratio=recent * scale / baseline if baseline else None,
        smoothed_ratio=(recent + 0.5) * scale / (baseline + 0.5), ratio_lower_95=lower, ratio_upper_95=upper,
        recent_rates=rates, recent_share_non_decreasing=non_decreasing,
        observed_growth=observed_growth,
        excess_growth_supported=bool(lower is not None and lower > 1 + 1e-10 and observed_growth))
