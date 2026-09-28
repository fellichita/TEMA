"""Optional patent signals and coordinated arXiv metadata imports.

These sources never change the OpenAlex reference publication series. arXiv
network polling is deliberately absent: a locally supplied Atom export can be
produced by an organisation-wide coordinator respecting arXiv's global quota.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable
from datetime import date, datetime, UTC
from pathlib import Path
from threading import Event
from typing import Final, Literal, cast
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element, ParseError

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pydantic import Field

from app.backend.contracts import Contract, DocumentRecord, SearchRequest
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.base import DocumentProvider
from app.backend.providers.epo import EpoOpsProvider
from app.pilot.archive import DocumentArchive
from app.pilot.contracts import Candidate, Coverage, DocumentRevisionRef, QueryPlan, content_hash
from app.pilot.evidence import title_matches
from app.pilot.reports import open_local_regular
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.jobs import RunContext, TaskCancelled
from app.runtime.worker import Cancellation

MAX_ATOM_BYTES = 5_000_000
MAX_ARXIV_RECORDS = 1000
ARXIV_IMPORT_VERSION: Final = "coordinated-arxiv-atom/1.0.0"
PATENT_SIGNAL_VERSION: Final = "epo-publication-family-title-match/1.0.0"


class EnrichmentError(ValueError):
    pass


class PatentFamily(Contract):
    family_id: str = Field(min_length=1, max_length=100)
    publications: tuple[str, ...] = Field(min_length=1, max_length=200)
    revision_ids: tuple[str, ...] = Field(min_length=1, max_length=200)
    first_observed_publication_year: int = Field(ge=1000, le=9999, strict=True)
    first_observed_publication_date: date | None = None


class PatentSignal(Contract):
    version: Literal["epo-publication-family-title-match/1.0.0"] = PATENT_SIGNAL_VERSION
    candidate_id: str
    documents: tuple[DocumentRevisionRef, ...] = Field(max_length=200)
    families: tuple[PatentFamily, ...] = Field(max_length=200)
    unresolved_family_publications: int = Field(ge=0, le=200, strict=True)
    coverage: Coverage
    limitations: tuple[str, ...]


class _DeadlineEvent(Event):
    def __init__(self, parent: Event, deadline: float):
        super().__init__()
        self.parent = parent
        self.deadline = deadline

    def is_set(self) -> bool:
        return self.parent.is_set() or time.monotonic() >= self.deadline or super().is_set()

    def wait(self, timeout: float | None = None) -> bool:
        end = min(self.deadline, time.monotonic() + timeout) if timeout is not None else self.deadline
        while not self.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                return self.is_set()
            self.parent.wait(min(remaining, 0.05))
        return True


def collect_patent_signal(candidate: Candidate, plan: QueryPlan, archive: DocumentArchive,
                          credentials: CredentialStore, context: RunContext, *,
                          max_documents: int = 200, timeout_seconds: float = 90.0,
                          provider_factory: Callable[[], DocumentProvider] | None = None) -> PatentSignal:
    """Collect a bounded public EPO query union; family counts are lower bounds.

    The document budget applies to scanned records, including duplicates and
    rejected records, so noisy candidate queries cannot consume unbounded pages.
    Missing credentials return unavailable coverage rather than fabricated zero.
    """
    if type(max_documents) is not int or not 1 <= max_documents <= 200:
        raise ValueError("Патентный лимит должен быть от 1 до 200 записей.")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 180:
        raise ValueError("Патентное обогащение ограничено 180 секундами.")
    if candidate.plan_hash != plan.plan_hash or candidate.specificity != "specific_technology":
        raise EnrichmentError("Патентный поиск требует конкретного кандидата из текущего плана.")
    phrases = tuple(candidate.synonyms[:3])
    if not phrases or any(len(phrase) > 180 or '"' in phrase or "\\" in phrase for phrase in phrases):
        raise EnrichmentError("Некорректные зафиксированные патентные поисковые фразы.")
    context.check_cancelled()
    start = date(plan.completed_years[0], 1, 1)
    years = tuple(range(start.year, plan.as_of.year + 1))
    query_hash = content_hash({"candidate": candidate.admission_rule_hash, "phrases": phrases,
        "start": start.isoformat(), "end": plan.as_of.isoformat(), "method": PATENT_SIGNAL_VERSION})
    scanned = rejected = unresolved = 0
    exhaustive = True
    limited = False
    reasons: list[str] = []
    documents: dict[str, tuple[DocumentRevisionRef, DocumentRecord]] = {}
    provider = None
    cancel = _DeadlineEvent(context.cancel_event, time.monotonic() + timeout_seconds)
    try:
        if provider_factory:
            provider = provider_factory()
        else:
            key, secret = credentials.get("epo_ops_key"), credentials.get("epo_ops_secret")
            if not key or not secret:
                reasons.append("missing_epo_credentials")
            else:
                provider = cast(Callable[..., DocumentProvider], EpoOpsProvider)(consumer_key=key, consumer_secret=secret,
                    page_size=min(max_documents, 100), timeout_seconds=10, max_retries=0,
                    max_response_bytes=2_000_000, page_deadline_seconds=15)
        if provider is None:
            exhaustive = False
        else:
            for phrase in phrases:
                context.check_cancelled()
                if cancel.is_set():
                    raise TimeoutError()
                if scanned >= max_documents:
                    limited, exhaustive = True, False
                    reasons.append("patent_scan_budget")
                    break
                exhausted_query = False
                request = SearchRequest(topic=phrase, source="epo", from_date=start,
                    until_date=plan.as_of, max_results=max_documents - scanned)
                for page in provider.iter_pages(request, cancel):
                    context.check_cancelled()
                    scanned += page.scanned
                    unresolved += page.skipped
                    for document in page.documents:
                        if document.source != "epo" or document.document_type != "patent" or not document.patent_publication:
                            unresolved += 1
                            continue
                        year = document.publication_year
                        if year is None or (year == plan.as_of.year and document.publication_date is None):
                            unresolved += 1
                            continue
                        if (not start.year <= year <= plan.as_of.year
                                or (document.publication_date and not start <= document.publication_date <= plan.as_of)
                                or not title_matches(document.title, candidate.synonyms, candidate.exclusions)):
                            rejected += 1
                            continue
                        if document.document_key in documents:
                            rejected += 1
                            continue
                        if len(documents) >= max_documents:
                            unresolved += 1
                            limited = True
                            continue
                        reference = archive.put(document)
                        documents[document.document_key] = reference, document
                    exhausted_query = page.exhausted
                    context.progress("enrichment", f"Патентных публикаций с совпадением в названии: {len(documents)}",
                                     min(scanned, max_documents), max_documents)
                    if scanned >= max_documents:
                        if not page.exhausted:
                            limited = True
                        break
                exhaustive = exhaustive and exhausted_query
                if not exhausted_query:
                    reasons.append("patent_pagination_incomplete")
    except CredentialUnavailable:
        reasons.append("credential_store_unavailable")
        exhaustive = False
    except (CancelledError, TimeoutError):
        context.check_cancelled()
        reasons.append("patent_time_budget")
        exhaustive = False
        limited = True
    except BackendError as error:
        reasons.append("source_" + error.code)
        exhaustive = False
    finally:
        if provider is not None:
            provider.close()
    context.check_cancelled()
    if unresolved:
        reasons.append("unresolved_patent_records")
    if limited:
        reasons.append("patent_budget_reached")
    complete = exhaustive and not unresolved and not limited
    state: Literal["complete", "partial", "unavailable"] = "complete" if complete else "partial" if scanned else "unavailable"
    coverage = Coverage(source="epo", purpose="enrichment", query_hash=query_hash,
        state=state, requested_years=years, completed_years=years if complete else (),
        pagination_exhausted=exhaustive, comparable=False, scanned_records=scanned,
        accepted_records=len(documents), rejected_records=rejected, unresolved_records=unresolved,
        limit_reached=limited, reasons=tuple(dict.fromkeys(reasons)) if not complete else ())
    families: dict[str, list[tuple[DocumentRevisionRef, DocumentRecord]]] = {}
    missing_family = 0
    for pair in documents.values():
        if pair[1].patent_family_id:
            families.setdefault(pair[1].patent_family_id, []).append(pair)
        else:
            missing_family += 1
    groups = []
    for family, pairs in sorted(families.items()):
        first_year = min(document.publication_year for _, document in pairs if document.publication_year is not None)
        earliest = [document.publication_date for _, document in pairs if document.publication_year == first_year]
        first_day = min(day for day in earliest if day is not None) if all(earliest) else None
        groups.append(PatentFamily(family_id=family,
            publications=tuple(sorted(document.patent_publication for _, document in pairs if document.patent_publication)),
            revision_ids=tuple(sorted(reference.revision_id for reference, _ in pairs)),
            first_observed_publication_year=first_year, first_observed_publication_date=first_day))
    limitations = (
        "Патенты — отдельный дополнительный сигнал; они не увеличивают научные публикации и Emerging Trend Score.",
        "Учтены совпадения замороженных фраз в названиях опубликованных документов EPO; поиск не охватывает все патенты технологии.",
        "Семейства объединены только по полученному EPO family-id; отсутствие ID не означает новое независимое семейство.",
        "Показана первая публичная дата среди найденных членов семейства, а не дата изобретения или приоритета; возможна задержка публикации.",
        "Пустая или неполная выдача не доказывает отсутствие патентов либо интереса к технологии.",
    )
    return PatentSignal(candidate_id=candidate.candidate_id,
        documents=tuple(reference for reference, _ in sorted(documents.values(), key=lambda pair: pair[0].revision_id)),
        families=tuple(groups), unresolved_family_publications=missing_family, coverage=coverage, limitations=limitations)


class ArxivImport(Contract):
    version: Literal["coordinated-arxiv-atom/1.0.0"] = ARXIV_IMPORT_VERSION
    export_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    as_of: date
    documents: tuple[DocumentRecord, ...] = Field(max_length=MAX_ARXIV_RECORDS)
    scanned: int = Field(ge=0, strict=True)
    rejected: int = Field(ge=0, strict=True)
    truncated: bool = Field(strict=True)
    limitations: tuple[str, ...]


def _atom_text(node: Element, name: str, maximum: int) -> str:
    value = " ".join(node.findtext("{http://www.w3.org/2005/Atom}" + name, "").split())
    if not value or len(value) > maximum:
        raise ValueError("Missing or oversized Atom text")
    return value


def _arxiv_date(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Missing time zone")
    return result.astimezone(UTC)


def _arxiv_document(entry: Element, as_of: date, export_hash: str) -> tuple[DocumentRecord, int]:
    raw_id = _atom_text(entry, "id", 500)
    url = urlsplit(raw_id)
    if url.scheme not in {"http", "https"} or url.netloc not in {"arxiv.org", "export.arxiv.org"} or url.query or url.fragment:
        raise ValueError("Invalid arXiv ID URL")
    match = re.fullmatch(r"/abs/((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v([1-9]\d{0,3}))?", url.path)
    if match is None:
        raise ValueError("Invalid arXiv identifier")
    identity, raw_version = match.groups()
    version = int(raw_version or "1")
    published = _arxiv_date(_atom_text(entry, "published", 80))
    updated = _arxiv_date(_atom_text(entry, "updated", 80))
    if published.date() > as_of or updated.date() > as_of or updated < published or published.year < 1991:
        raise ValueError("Outside public as-of boundary")
    title = _atom_text(entry, "title", 10000)
    abstract = _atom_text(entry, "summary", 200000)
    authors = tuple(_atom_text(author, "name", 500) for author in entry.findall("{http://www.w3.org/2005/Atom}author"))
    if not authors or len(authors) > 5000:
        raise ValueError("Missing or oversized author list")
    doi = entry.findtext("{http://arxiv.org/schemas/atom}doi")
    return DocumentRecord(source="arxiv", source_id=identity, doi=doi.strip() if doi else None,
        title=title, abstract=abstract, publication_year=published.year, publication_month=published.month,
        publication_date=published.date(), date_precision="day", authors=authors,
        url="https://arxiv.org/abs/" + identity + "v" + str(version), document_type="preprint",
        raw_metadata={"arxiv_id": identity, "version": version, "published": published.isoformat(),
            "updated": updated.isoformat(), "coordinated_export_sha256": export_hash,
            "import_version": ARXIV_IMPORT_VERSION}), version


def read_arxiv_feed(path: Path, *, cancel: Cancellation) -> tuple[str, tuple[Element, ...]]:
    """Shared bounded XML boundary; consumers decide whether to retain all versions."""
    if cancel.is_set():
        raise TaskCancelled()
    try:
        with open_local_regular(path) as stream:
            data = stream.read(MAX_ATOM_BYTES + 1)
    except OSError:
        raise EnrichmentError("Не удалось прочитать локальный экспорт arXiv.") from None
    if len(data) > MAX_ATOM_BYTES:
        raise EnrichmentError("Экспорт arXiv превышает 5 МБ.")
    try:
        root = ElementTree.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (ParseError, DefusedXmlException, ValueError, RecursionError):
        raise EnrichmentError("Некорректный или небезопасный Atom XML; импорт отклонён.") from None
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise EnrichmentError("Требуется Atom feed из согласованного экспорта arXiv.")
    return hashlib.sha256(data).hexdigest(), tuple(root.findall("{http://www.w3.org/2005/Atom}entry"))


def import_arxiv_atom(path: Path, *, as_of: date, cancel: Cancellation,
                      max_records: int = MAX_ARXIV_RECORDS) -> ArxivImport:
    """Read a coordinated Atom export without any API request or AI processing."""
    if type(max_records) is not int or not 1 <= max_records <= MAX_ARXIV_RECORDS:
        raise ValueError("Допустимо от 1 до 1000 arXiv-записей.")
    if as_of > max(date.today(), datetime.now(UTC).date()) or as_of.year < 1991:
        raise ValueError("Некорректная дата среза arXiv.")
    export_hash, entries = read_arxiv_feed(path, cancel=cancel)
    documents: dict[str, tuple[DocumentRecord, int]] = {}
    conflicting_ids: set[str] = set()
    rejected = scanned = 0
    for entry in entries[:max_records]:
        if cancel.is_set():
            raise TaskCancelled()
        scanned += 1
        try:
            document, version = _arxiv_document(entry, as_of, export_hash)
        except (ValueError, TypeError, OverflowError):
            rejected += 1
            continue
        previous = documents.get(document.source_id)
        if document.source_id in conflicting_ids:
            rejected += 1
            continue
        if previous is not None:
            # Versions of one preprint are one record. Different publication
            # dates cannot silently invent an earlier first appearance.
            rejected += 1
            if previous[0].publication_date != document.publication_date:
                conflicting_ids.add(document.source_id)
                del documents[document.source_id]
                rejected += 1
                continue
            if previous[1] >= version:
                continue
        documents[document.source_id] = document, version
    return ArxivImport(export_sha256=export_hash, as_of=as_of,
        documents=tuple(item[0] for _, item in sorted(documents.items())), scanned=scanned, rejected=rejected,
        truncated=len(entries) > max_records, limitations=(
            "Это локальный экспорт метаданных, а не полная или текущая выдача arXiv.",
            "arXiv уже может присутствовать в OpenAlex; DOI следует объединять с научными работами, версии препринта не считать новыми исследованиями.",
            "Препринт не означает пройденное рецензирование; импорт не добавляет записи в reference-историю роста.",
            "Прямой API arXiv из каждого приложения отключён. Для общего сборщика требуется единая очередь всех экземпляров и соблюдение правил arXiv.",
        ))
