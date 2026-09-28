"""OpenAlex works: bounded bibliographic search and abstract reconstruction."""

import re
from datetime import date
from threading import Event
from urllib.parse import quote

import httpx

from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.crossref import _plain_text
from app.backend.providers.http_transport import BoundedHttpTransport, _cancelled, _positive_number
from app.runtime.backup import strip_link_credentials

ENDPOINT = "https://api.openalex.org/works"
# Сбой, переживший все повторы, анализ не переспрашивает две минуты.
OUTAGE_PAUSE_SECONDS = 120.0
_FIELDS = "id,doi,title,publication_date,publication_year,authorships,abstract_inverted_index,type,language,cited_by_count,primary_topic,is_retracted"


def _abstract(index) -> str | None:
    if index is None or index == {}:
        return None
    if not isinstance(index, dict) or len(index) > 50_000:
        raise ValueError("Некорректный индекс аннотации")
    positions = {}
    chars = 0
    for word, occurrences in index.items():
        if not isinstance(word, str) or not isinstance(occurrences, list):
            raise ValueError("Некорректный индекс аннотации")
        for position in occurrences:
            if type(position) is not int or not 0 <= position < 50_000 or position in positions:
                raise ValueError("Некорректная позиция в аннотации")
            positions[position] = word
            chars += len(word) + 1
            if chars > 200_000:
                raise ValueError("Слишком большая аннотация")
    if not positions:
        return None
    if max(positions) != len(positions) - 1:
        raise ValueError("Неполный индекс аннотации")
    return _plain_text(" ".join(positions[i] for i in range(len(positions)))) or None


def _normalize(item) -> DocumentRecord:
    if not isinstance(item, dict):
        raise ValueError("Некорректная запись")
    identifier = item.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"https://openalex\.org/W\d{1,20}", identifier):
        raise ValueError("Некорректный OpenAlex ID")
    title = item.get("title") or item.get("display_name")
    if not isinstance(title, str):
        raise ValueError("Отсутствует название")
    raw_date = item.get("publication_date")
    published = date.fromisoformat(raw_date) if isinstance(raw_date, str) else None
    if raw_date is not None and published is None:
        raise ValueError("Некорректная дата")
    year = item.get("publication_year")
    if year is None and published is not None:
        year = published.year
    authorships = item.get("authorships") or []
    if not isinstance(authorships, list):
        raise ValueError("Некорректные авторы")
    names = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict) and isinstance(author.get("display_name"), str):
            names.append(_plain_text(author["display_name"]))
    record = DocumentRecord(
        source="openalex", source_id=identifier.rsplit("/", 1)[1], doi=item.get("doi"),
        title=_plain_text(title), abstract=_abstract(item.get("abstract_inverted_index")),
        publication_year=year, publication_date=published,
        publication_month=published.month if published else None,
        date_precision="day" if published else "year" if year is not None else "unknown",
        authors=tuple(names), url=identifier, language=item.get("language"),
        document_type=item.get("type") or "publication", citation_count=item.get("cited_by_count"),
        raw_metadata=strip_link_credentials(item),
    )
    if record.doi:
        return record.model_copy(update={"url": "https://doi.org/" + quote(record.doi, safe="/")})
    return record


class OpenAlexProvider(BoundedHttpTransport):
    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None,
                 page_size=100, timeout_seconds=15.0, max_retries=2, max_response_bytes=5_000_000,
                 retry_delay=1.0, *, page_delay=1.0, page_deadline_seconds=60.0, max_retry_delay=30.0):
        # OpenAlex documents 100 as the supported maximum. The formerly
        # accepted 200-result page is deprecated and may disappear.
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("OpenAlex page_size должен быть 1–100")
        if api_key is not None and (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096
                                    or any(not 33 <= ord(char) <= 126 for char in api_key)):
            raise BackendError("invalid_credentials", "Некорректный формат ключа OpenAlex.")
        self.page_size = page_size
        self.page_delay = _positive_number("page_delay", page_delay, allow_zero=True)
        if self.page_delay > 30:
            raise ValueError("Недопустимая задержка")
        self._auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # 27.09.2026 OpenAlex минутами отвечал 503 «Anonymous search is paused while
        # the search cluster recovers»: каждый запрос истории проходил все повторы, и
        # анализ тратил около минуты на кандидата, не получив ни одного документа.
        super().__init__(ENDPOINT, client, timeout_seconds, max_retries, max_response_bytes,
                         retry_delay, page_deadline_seconds, max_retry_delay,
                         unavailable_pause=OUTAGE_PAUSE_SECONDS)

    def iter_pages(self, request: SearchRequest, cancel: Event):
        cursor, scanned = "*", 0
        topic_ids = set(request.primary_topic_ids)
        filters = [f"to_publication_date:{request.until_date.isoformat()}"]
        if request.from_date:
            filters.append(f"from_publication_date:{request.from_date.isoformat()}")
        if topic_ids:
            filters.append("primary_topic.id:" + "|".join(
                identifier.rsplit("/", 1)[1] for identifier in request.primary_topic_ids))
            selection = {"sort": "publication_date:asc"}
        else:
            selection = {"search": request.topic, "sort": "relevance_score:desc"}
        while scanned < request.max_results:
            _cancelled(cancel)
            if scanned and cancel.wait(self.page_delay):
                raise CancelledError()
            rows = min(self.page_size, request.max_results - scanned)
            payload = self.get_json({**selection, "filter": ",".join(filters),
                                     "per_page": rows, "cursor": cursor,
                                     "select": _FIELDS}, cancel, self._auth_headers)
            meta, items = payload.get("meta"), payload.get("results")
            if not isinstance(meta, dict) or not isinstance(items, list) or len(items) > rows:
                raise BackendError("invalid_response", "Некорректная выдача OpenAlex.")
            total = meta.get("count")
            if total is not None and (type(total) is not int or total < 0):
                raise BackendError("invalid_response", "Некорректное число результатов OpenAlex.")
            documents = []
            for item in items:
                _cancelled(cancel)
                if topic_ids:
                    primary_topic = item.get("primary_topic") if isinstance(item, dict) else None
                    if not isinstance(primary_topic, dict) or primary_topic.get("id") not in topic_ids:
                        raise BackendError("invalid_response", "OpenAlex вернул работу вне выбранных основных тем.")
                try:
                    documents.append(_normalize(item))
                except (ValueError, TypeError, OverflowError):
                    continue
            scanned += len(items)
            next_cursor = meta.get("next_cursor")
            exhausted = not items or (total is not None and scanned >= total)
            if not exhausted and scanned < request.max_results:
                if not isinstance(next_cursor, str) or not 1 <= len(next_cursor) <= 32768 or next_cursor == cursor:
                    raise BackendError("invalid_response", "OpenAlex не предоставил корректное продолжение выдачи.")
                cursor = next_cursor
            yield SourcePage(documents=tuple(documents), scanned=len(items), skipped=len(items) - len(documents),
                             total_available=total, exhausted=exhausted)
            if exhausted:
                return

    def close(self):
        self._auth_headers.clear()
        super().close()
