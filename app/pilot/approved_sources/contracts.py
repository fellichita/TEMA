"""Bounded observations from the approved, non-bibliographic source catalogue.

These records describe where a technology was mentioned. They do not certify
scientific novelty, independent corroboration, or rights to article text.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.input_safety import is_safe_http_url
from app.runtime.backup import ArchiveError, assert_no_credentials

SourceId = Literal[
    "arxiv", "biorxiv", "openreview", "zenodo", "nist_news",
    "mit_research_news", "horizon_magazine", "github", "hacker_news", "gdelt", "habr", "europe_pmc",
    "google_news", "semantic_scholar", "doaj", "cyberleninka", "hal", "osti", "nasa_ntrs", "dblp",
    "stack_exchange", "huggingface", "chemrxiv", "openaire", "jstage", "npm",
]
SOURCE_IDS: tuple[SourceId, ...] = (
    "arxiv", "biorxiv", "openreview", "zenodo", "nist_news",
    "mit_research_news", "horizon_magazine", "github", "hacker_news", "gdelt", "habr", "europe_pmc",
    # Источники без ключей API, добавленные к исходным двенадцати.
    "google_news", "semantic_scholar", "doaj", "cyberleninka", "hal", "osti", "nasa_ntrs", "dblp",
    "stack_exchange", "huggingface", "chemrxiv",
    # Добавлены 28.09.2026: граф исследований Евросоюза, японские журналы, пакеты npm.
    "openaire", "jstage", "npm",
)
# Сохранённые анализы прежних каталогов остаются читаемыми: они не заявляют
# охват источников, добавленных позже. Новый сбор всегда идёт по полному каталогу.
LEGACY_CATALOGUES: tuple[tuple[SourceId, ...], ...] = (SOURCE_IDS[:10], SOURCE_IDS[:12], SOURCE_IDS[:23])
ObservationKind = Literal[
    "preprint", "journal_article", "research_artifact", "institution_news", "news_aggregate", "repository", "community",
]
# Страна или регион источника: ISO 3166-1 alpha-2, EU — Евросоюз, INT — международный.
COUNTRY_CODE = r"^(?:[A-Z]{2}|INT)$"
CoverageState = Literal["complete", "partial", "unavailable"]


class SourceFetchError(RuntimeError):
    """A safe, source-independent failure code; never carry response content."""

    def __init__(self, code: str) -> None:
        self.code = code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else "source_error"
        super().__init__(self.code)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ExternalObservation(_Contract):
    source_id: SourceId
    item_id: str = Field(min_length=1, max_length=512)
    kind: ObservationKind
    title: str = Field(min_length=1, max_length=1000)
    url: str = Field(min_length=1, max_length=4096)
    published_at: date
    # GDELT's seendate is an index timestamp; an OpenReview note without odate
    # has only a reliable creation time, not a proven public-release date.
    # Year: the source states only the publication year (published_at is 1 January).
    date_basis: Literal["published", "indexed", "created", "year"] = "published"
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    summary: str | None = Field(default=None, max_length=1000)
    arxiv_primary_category: str | None = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)?$",
    )
    rights: Literal["local_only", "share_allowed"] = "local_only"
    license_ref: str | None = Field(default=None, max_length=500)
    country: str | None = Field(default=None, pattern=COUNTRY_CODE)

    @field_validator("url")
    @classmethod
    def safe_url(cls, value: str) -> str:
        if not is_safe_http_url(value):
            raise ValueError("Observation URL is unsafe")
        try:
            assert_no_credentials(value)
        except ArchiveError:
            raise ValueError("Observation URL contains access credentials") from None
        return value

    @model_validator(mode="after")
    def valid_observation(self) -> Self:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("Observation time requires a timezone")
        if self.source_id == "gdelt" and self.date_basis != "indexed":
            raise ValueError("GDELT indexing time must not be described as publication")
        if self.date_basis == "created" and self.source_id != "openreview":
            raise ValueError("Created-date observations are reserved for OpenReview")
        if self.date_basis == "year" and (self.published_at.month, self.published_at.day) != (1, 1):
            raise ValueError("A year-only date must be stored as 1 January")
        if self.arxiv_primary_category is not None and self.source_id != "arxiv":
            raise ValueError("arXiv categories are reserved for arXiv observations")
        if self.rights == "share_allowed" and not self.license_ref:
            raise ValueError("Sharing requires a rights reference")
        if self.rights == "local_only" and self.license_ref is not None:
            raise ValueError("Local-only observations do not claim transfer rights")
        return self


class ObservationPage(_Contract):
    observations: tuple[ExternalObservation, ...] = Field(default=(), max_length=1000)
    scanned: int = Field(ge=0, le=10000, strict=True)
    exhausted: bool = Field(strict=True)
    total_available: int | None = Field(default=None, ge=0, le=1_000_000_000, strict=True)

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if len(self.observations) > self.scanned:
            raise ValueError("Page observations exceed scanned items")
        return self


class SourceCoverage(_Contract):
    source_id: SourceId
    state: CoverageState
    requested_limit: int = Field(ge=0, le=10000, strict=True)
    scanned: int = Field(ge=0, le=10000, strict=True)
    accepted: int = Field(ge=0, le=10000, strict=True)
    rejected: int = Field(ge=0, le=10000, strict=True)
    duplicates: int = Field(ge=0, le=10000, strict=True)
    limit_reached: bool = Field(strict=True)
    reason_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    total_available: int | None = Field(default=None, ge=0, le=1_000_000_000, strict=True)

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if self.scanned != self.accepted + self.rejected + self.duplicates:
            raise ValueError("Source coverage counts do not reconcile")
        if self.scanned > self.requested_limit:
            raise ValueError("Source exceeded its request limit")
        if self.state == "complete" and (self.limit_reached or self.reason_code is not None):
            raise ValueError("Complete coverage cannot have a limiting reason")
        if self.state != "complete" and self.reason_code is None:
            raise ValueError("Incomplete coverage needs a reason")
        return self


class ArxivDomainMonth(_Contract):
    """Count of distinct observed arXiv articles, not an archive-wide total."""

    month: date
    domain: str = Field(min_length=1, max_length=64)
    primary_category: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)?$",
    )
    article_count: int = Field(ge=1, le=10000, strict=True)

    @model_validator(mode="after")
    def valid_category_month(self) -> Self:
        if self.month.day != 1 or self.domain != self.primary_category.split(".", 1)[0]:
            raise ValueError("Invalid arXiv domain month")
        return self


class SourceSnapshot(_Contract):
    query: str = Field(min_length=1, max_length=500)
    as_of: date
    collected_at: datetime
    observations: tuple[ExternalObservation, ...] = Field(default=(), max_length=10000)
    coverage: tuple[SourceCoverage, ...] = Field(min_length=1, max_length=50)
    normalizer_version: Literal["approved-observations/1"] = "approved-observations/1"

    @property
    def arxiv_domain_months(self) -> tuple[ArxivDomainMonth, ...]:
        """Observed first-publication counts by month and primary arXiv category.

        The arXiv adapter samples at most one recent page. These counts do not
        establish complete coverage of a month or category; see source coverage.
        """
        counts: dict[tuple[date, str], int] = {}
        seen_ids: set[str] = set()
        for item in self.observations:
            if item.source_id != "arxiv" or item.item_id in seen_ids:
                continue
            category = item.arxiv_primary_category
            if category is None:
                continue
            seen_ids.add(item.item_id)
            key = (item.published_at.replace(day=1), category)
            counts[key] = counts.get(key, 0) + 1
        return tuple(
            ArxivDomainMonth(month=month, domain=category.split(".", 1)[0],
                             primary_category=category, article_count=count)
            for (month, category), count in sorted(counts.items())
        )

    @model_validator(mode="after")
    def valid_snapshot(self) -> Self:
        if self.collected_at.tzinfo is None or self.collected_at.utcoffset() is None:
            raise ValueError("Collection time requires a timezone")
        recorded_sources = tuple(item.source_id for item in self.coverage)
        # Saved analyses from the ten- and twelve-source catalogues remain
        # readable after new sources were added. They do not claim coverage for
        # the newly added sources; a new collection always uses the full catalogue.
        if recorded_sources != SOURCE_IDS and recorded_sources not in LEGACY_CATALOGUES:
            raise ValueError("Coverage must follow a supported source catalogue")
        totals = {item.source_id: item.accepted for item in self.coverage}
        counts = dict.fromkeys(totals, 0)
        future_publication = missing_coverage = False
        for observation in self.observations:
            if observation.published_at > self.as_of:
                future_publication = True
            if observation.source_id in counts:
                counts[observation.source_id] += 1
            else:
                missing_coverage = True
        if future_publication:
            raise ValueError("Snapshot contains a future publication")
        if missing_coverage:
            raise ValueError("Snapshot observation has no source coverage")
        if any(counts[source] != accepted for source, accepted in totals.items()):
            raise ValueError("Snapshot observations and coverage disagree")
        return self
