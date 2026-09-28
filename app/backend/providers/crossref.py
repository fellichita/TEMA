"""Bounded, cancellable access to the public Crossref works endpoint.

The provider retrieves bibliographic evidence, not trend rankings. A result limit
counts raw records, including invalid ones. Publication dates are never padded
with invented months or days; the original metadata remains available for audit.
"""

import math
from collections.abc import Iterator
from datetime import date
from html.parser import HTMLParser
from threading import Event
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from app.backend.contracts import DatePrecision, DocumentRecord, SearchRequest, SourcePage, normalize_doi
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.http_transport import BoundedHttpTransport
from app.runtime.backup import strip_link_credentials

_WORKS_URL = "https://api.crossref.org/works"
_MESSAGES = {
    "source_unavailable": "Источник публикаций временно недоступен. Повторите позже.",
    "rate_limited": "Источник ограничил частоту запросов. Повторите позже.",
    "invalid_response": "Источник вернул некорректные данные.",
    "response_too_large": "Ответ источника превышает допустимый размер.",
}


def _error(code: str) -> BackendError:
    return BackendError(code, _MESSAGES[code])


def _cancelled(cancel: Event) -> None:
    if cancel.is_set():
        raise CancelledError()


def _positive_number(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: требуется конечное число")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name}: недопустимое значение")
    return float(value)


class _PlainText(HTMLParser):
    """JATS/HTML text only; no XML parser, files, network, or entity resolver."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        if tag.split(":")[-1] in {"p", "br", "div", "title", "sec", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1
        if tag.split(":")[-1] in {"p", "div", "title", "sec", "li"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    # Titles and author names are usually plain text. Avoid constructing an
    # HTML parser unless there may be markup or an entity to decode.
    if "<" not in value and "&" not in value:
        return " ".join(value.split())
    parser = _PlainText()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def _publication_date(item: dict) -> tuple[int | None, int | None, date | None, DatePrecision]:
    # Crossref's publication dates are not its deposit/registration dates.
    for key in ("published", "published-online", "published-print", "issued"):
        field = item.get(key)
        if field is None:
            continue
        if not isinstance(field, dict):
            raise ValueError("Некорректная дата публикации")
        parts = field.get("date-parts")
        if parts is None:
            continue
        if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
            raise ValueError("Некорректная дата публикации")
        first = parts[0]
        if not 1 <= len(first) <= 3 or any(type(value) is not int for value in first):
            raise ValueError("Некорректная дата публикации")
        year = first[0]
        if not 1000 <= year <= 9999:
            raise ValueError("Некорректный год публикации")
        if len(first) == 1:
            return year, None, None, "year"
        if not 1 <= first[1] <= 12:
            raise ValueError("Некорректный месяц публикации")
        if len(first) == 2:
            return year, first[1], None, "month"
        return year, first[1], date(*first), "day"
    return None, None, None, "unknown"


def _authors(item: dict) -> tuple[str, ...]:
    authors = item.get("author")
    if authors is None:
        return ()
    if not isinstance(authors, list):
        raise ValueError("Некорректные авторы")
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        pieces = [author.get("given"), author.get("family")]
        name = " ".join(piece.strip() for piece in pieces if isinstance(piece, str))
        if not name and isinstance(author.get("name"), str):
            name = author["name"].strip()
        if name:
            names.append(_plain_text(name))
    return tuple(names)


def _normalize(item: object) -> DocumentRecord:
    if not isinstance(item, dict) or not isinstance(item.get("DOI"), str):
        raise ValueError("Требуется DOI")
    doi = normalize_doi(item["DOI"])
    titles = item.get("title")
    if not isinstance(titles, list):
        raise ValueError("Требуется заголовок")
    title = next((_plain_text(value) for value in titles if isinstance(value, str) and value.strip()), "")
    abstract = item.get("abstract")
    if abstract is not None:
        if not isinstance(abstract, str):
            raise ValueError("Некорректная аннотация")
        abstract = _plain_text(abstract) or None
    year, month, published, precision = _publication_date(item)
    return DocumentRecord(
        source="crossref",
        source_id=doi,
        doi=doi,
        title=title,
        abstract=abstract,
        publication_year=year,
        publication_month=month,
        publication_date=published,
        date_precision=precision,
        authors=_authors(item),
        url="https://doi.org/" + quote(doi, safe="/:;()@!$&'*,=+-._~"),
        language=item.get("language"),
        document_type=item.get("type") or "publication",
        citation_count=item.get("is-referenced-by-count"),
        # Retain only this source record, never request URLs, headers or credentials.
        raw_metadata=strip_link_credentials(item),
    )


def _reject_non_json_constant(value: str) -> None:
    raise ValueError("Недопустимая JSON-константа")


class CrossrefProvider(BoundedHttpTransport):
    def __init__(
        self, client: httpx.Client | None = None, page_size: int = 100,
        timeout_seconds: float = 15.0, max_retries: int = 2,
        max_response_bytes: int = 5_000_000, retry_delay: float = 1.0,
        *, page_delay: float = 1.0, page_deadline_seconds: float = 60.0,
    ) -> None:
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError("page_size должен быть целым числом от 1 до 1000")
        self.page_size = page_size
        self.page_delay = _positive_number("page_delay", page_delay, allow_zero=True)
        if self.page_delay > 30:
            raise ValueError("Пауза между запросами не должна превышать 30 секунд")
        super().__init__(
            _WORKS_URL, client, timeout_seconds, max_retries, max_response_bytes,
            retry_delay, page_deadline_seconds,
        )

    def _read_page(self, params: dict[str, str | int], cancel: Event) -> dict:
        payload = self.get_json(params, cancel)
        if payload.get("status", "ok") != "ok":
            raise _error("invalid_response")
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("items"), list):
            raise _error("invalid_response")
        total = message.get("total-results")
        if total is not None and (type(total) is not int or total < 0):
            raise _error("invalid_response")
        return message

    def iter_pages(self, request: SearchRequest, cancel: Event) -> Iterator[SourcePage]:
        cursor = "*"
        scanned_total = 0
        page_size = self.page_size
        filters = [f"until-pub-date:{request.until_date.isoformat()}"]
        if request.from_date:
            filters.insert(0, f"from-pub-date:{request.from_date.isoformat()}")
        while scanned_total < request.max_results:
            _cancelled(cancel)
            if scanned_total and cancel.wait(self.page_delay):
                raise CancelledError()
            rows = min(page_size, request.max_results - scanned_total)
            params: dict[str, str | int] = {"query": request.topic, "filter": ",".join(filters), "rows": rows,
                                            "cursor": cursor, "sort": "relevance", "order": "desc"}
            try:
                message = self._read_page(params, cancel)
            except BackendError as error:
                if rows <= 100 or error.code not in {"response_too_large", "source_unavailable"}:
                    raise
                # Retry the same cursor after a cancellable pause. Keep smaller
                # pages for this iterator so no cursor boundary is skipped.
                page_size = 100
                if cancel.wait(self.page_delay):
                    raise CancelledError() from None
                rows = min(page_size, request.max_results - scanned_total)
                params["rows"] = rows
                message = self._read_page(params, cancel)
            items = message["items"]
            # A source ignoring rows would otherwise silently lose records at a cursor boundary.
            if len(items) > rows:
                raise _error("invalid_response")
            documents = []
            skipped = 0
            for item in items:
                _cancelled(cancel)
                try:
                    documents.append(_normalize(item))
                except (ValueError, TypeError, AssertionError, ValidationError):
                    skipped += 1
            scanned_total += len(items)
            total = message.get("total-results")
            exhausted = len(items) < rows or (total is not None and scanned_total >= total)
            next_cursor = message.get("next-cursor")
            if not exhausted and scanned_total < request.max_results:
                if not isinstance(next_cursor, str) or not next_cursor or len(next_cursor) > 32_768:
                    raise _error("invalid_response")
                # Crossref legitimately repeats the same opaque cursor across pages.
                cursor = next_cursor
            _cancelled(cancel)
            yield SourcePage(
                documents=tuple(documents),
                scanned=len(items),
                skipped=skipped,
                total_available=total,
                exhausted=exhausted,
            )
            if exhausted or scanned_total >= request.max_results:
                return
