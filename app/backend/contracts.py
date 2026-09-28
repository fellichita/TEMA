"""Версия 2 контракта backend. Модели не зависят от UI и ML."""

import re
from datetime import date, datetime, timezone
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.input_safety import MAX_ABSTRACT_CHARACTERS, MAX_TITLE_CHARACTERS, MAX_URL_CHARACTERS, is_safe_http_url

CONTRACT_VERSION = 2
SourceName = Literal["crossref", "openalex", "epo"]
DatePrecision = Literal["day", "month", "year", "unknown"]
SOURCE_NAMES = ("crossref", "openalex", "epo")
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def latest_today() -> date:
    """The later of the two dates a user may legitimately call "today".

    East of UTC the local date runs ahead of the UTC one for the first hours of
    every day. A request whose window ends on the local today is not a request
    about the future, so refusing it would make the application unusable at
    night in exactly those zones. The sources simply return nothing newer.
    `enrichment` and the arXiv import already read "today" this way.
    """
    return max(utc_now().date(), date.today())


def normalize_doi(value: str) -> str:
    value = value.strip()
    if value.lower().startswith(("https://doi.org/", "http://doi.org/", "https://dx.doi.org/")):
        value = unquote(urlsplit(value).path.lstrip("/"))
    elif value.lower().startswith("doi:"):
        value = value[4:].strip()
    value = value.casefold()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value) or len(value) > 2048:
        raise ValueError("Некорректный DOI")
    return value


def normalize_primary_topic_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical OpenAlex primary-topic selection, suitable for request provenance."""
    normalized = []
    for value in values:
        match = re.fullmatch(r"(?:https://openalex\.org/)?(T[1-9]\d{0,19})", value)
        if match is None:
            raise ValueError("Требуется ID темы OpenAlex вида T123 или https://openalex.org/T123")
        normalized.append("https://openalex.org/" + match.group(1))
    if len(normalized) != len(set(normalized)):
        raise ValueError("ID тем OpenAlex не должны повторяться")
    return tuple(sorted(normalized))


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SearchRequest(Contract):
    topic: str = Field(min_length=2, max_length=500)
    source: SourceName = "crossref"
    from_date: date | None = None
    until_date: date = Field(default_factory=lambda: utc_now().date())
    max_results: int = Field(default=200, ge=1, le=10_000, strict=True)
    primary_topic_ids: tuple[str, ...] = Field(default=(), max_length=100)

    @field_validator("primary_topic_ids")
    @classmethod
    def validate_primary_topics(cls, values):
        return normalize_primary_topic_ids(values)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("Направление не должно содержать управляющие символы")
        return value

    @model_validator(mode="after")
    def validate_dates(self):
        if self.primary_topic_ids and self.source != "openalex":
            raise ValueError("Отбор по primary_topic_ids доступен только для OpenAlex")
        if self.from_date and self.from_date > self.until_date:
            raise ValueError("Начало периода позже окончания")
        if self.until_date > latest_today():
            raise ValueError("Окончание периода не может быть в будущем")
        return self


class DocumentRecord(Contract):
    source: str = Field(min_length=1, max_length=50, pattern=r"^[a-z][a-z0-9_-]*$")
    source_id: str = Field(min_length=1, max_length=2048)
    doi: str | None = None
    patent_publication: str | None = None
    patent_family_id: str | None = Field(default=None, min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARACTERS)
    abstract: str | None = Field(default=None, max_length=MAX_ABSTRACT_CHARACTERS)
    publication_year: int | None = Field(default=None, ge=1000, le=9999, strict=True)
    publication_month: int | None = Field(default=None, ge=1, le=12, strict=True)
    publication_date: date | None = None
    date_precision: DatePrecision = "unknown"
    authors: tuple[str, ...] = Field(default=(), max_length=5000)
    url: str = Field(max_length=MAX_URL_CHARACTERS)
    language: str | None = Field(default=None, max_length=40)
    document_type: str = Field(default="publication", max_length=100)
    citation_count: int | None = Field(default=None, ge=0, strict=True)
    fetched_at: datetime = Field(default_factory=utc_now)
    raw_metadata: dict = Field(default_factory=dict)

    @field_validator("doi")
    @classmethod
    def valid_doi(cls, value):
        return normalize_doi(value) if value is not None else None

    @field_validator("patent_publication")
    @classmethod
    def valid_patent_number(cls, value):
        if value is None:
            return None
        normalized = re.sub(r"[\s.\-/]", "", value).upper()
        if not re.fullmatch(r"[A-Z]{2}[A-Z]?\d{1,15}[A-Z]\d{0,2}", normalized):
            raise ValueError("Требуется номер патентной публикации со страной и кодом вида, например EP1234567A1")
        return normalized

    @field_validator("url", mode="before")
    @classmethod
    def valid_url(cls, value):
        if not is_safe_http_url(value):
            raise ValueError("Ожидается корректная HTTP(S)-ссылка без учётных данных")
        return value

    @field_validator("fetched_at")
    @classmethod
    def aware_datetime(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Требуется время с часовым поясом")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def date_consistency(self):
        if self.patent_publication and (self.doi or self.document_type != "patent"):
            raise ValueError("Патентная публикация должна иметь тип patent без DOI статьи")
        if self.patent_family_id and not self.patent_publication:
            raise ValueError("Патентное семейство требует номера публикации")
        if self.date_precision == "unknown" and (self.publication_year or self.publication_date or self.publication_month):
            raise ValueError("Для известного года требуется точность даты")
        if self.date_precision != "unknown" and self.publication_year is None:
            raise ValueError("Требуется год публикации")
        if self.date_precision == "day" and self.publication_date is None:
            raise ValueError("Для точности day требуется полная дата")
        if self.date_precision == "month" and self.publication_month is None:
            raise ValueError("Для точности month требуется месяц")
        if self.date_precision == "year" and self.publication_month is not None:
            raise ValueError("Точность year не должна содержать месяц")
        if self.date_precision != "day" and self.publication_date is not None:
            raise ValueError("Неполную дату нельзя заменять выдуманным днём")
        if self.publication_date and self.publication_date.year != self.publication_year:
            raise ValueError("Год и дата не совпадают")
        if self.publication_date and self.publication_month not in (None, self.publication_date.month):
            raise ValueError("Месяц и дата не совпадают")
        return self

    @property
    def document_key(self) -> str:
        if self.patent_publication:
            return f"patent:{self.patent_publication}"
        return f"doi:{self.doi}" if self.doi else f"{self.source}:{self.source_id}"


class SourcePage(Contract):
    documents: tuple[DocumentRecord, ...] = ()
    scanned: int = Field(ge=0, strict=True)
    skipped: int = Field(default=0, ge=0, strict=True)
    total_available: int | None = Field(default=None, ge=0, strict=True)
    exhausted: bool = False

    @model_validator(mode="after")
    def valid_counts(self):
        if self.scanned != len(self.documents) + self.skipped:
            raise ValueError("Число обработанных записей должно совпадать с документами и пропусками")
        return self


class JobRecord(Contract):
    id: str
    request: SearchRequest
    state: JobState
    created_at: datetime
    updated_at: datetime
    scanned: int = 0
    stored: int = 0
    skipped: int = 0
    total_available: int | None = None
    source_exhausted: bool = False
    error_code: str | None = None
    error_message: str | None = None
    contract_version: int = CONTRACT_VERSION

    @property
    def coverage_complete(self) -> bool:
        # Reaching a provider's end marker alone does not reconcile a larger
        # advertised result count. Apply this to saved jobs as well as new ones.
        return (self.state == "succeeded" and self.source_exhausted and self.skipped == 0
                and (self.total_available is None or self.scanned >= self.total_available))


class DocumentSnapshot(Contract):
    revision_id: str
    document_key: str
    document: DocumentRecord
    sources: tuple[str, ...] = ()


class DocumentPage(Contract):
    items: tuple[DocumentSnapshot, ...]
    total: int
    limit: int
    offset: int
