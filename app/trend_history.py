"""Помесячная история технологии из открытых источников для логики эксперта.

По каждому источнику — один запрос за весь период; записи раскладываются по
месяцам локально. Материал входит в историю, только если фраза технологии есть
в его названии: так поиск по релевантности не подмешивает соседние темы.
Отказ источника — потерянный охват, а не ноль материалов.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import calendar
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import re
from threading import Event, Lock
from time import monotonic

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
import httpx

from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError
from app.backend.providers.crossref import CrossrefProvider
from app.input_safety import work_key
from app.trend_confidence import DEFAULT_MONTHS, CurveAssessment, Material, assess_curve, month_window

MAX_RECORDS = 1000
MAX_RESPONSE_BYTES = 12_000_000
TIMEOUT_SECONDS = 30.0
USER_AGENT = "Trendanalyser/1.0 (weak-signal research; local)"
# Crossref ищет по релевантности без точной фразы. Если в последней пятой части
# выдачи фраза почти не встречается, подходящие записи исчерпаны.
SATURATION_SHARE = 0.02
# Правила arXiv API: не чаще одного запроса в 3 секунды со всего приложения.
ARXIV_INTERVAL_SECONDS = 3.0
_ARXIV_LOCK = Lock()
_ARXIV_LAST = [0.0]
_ATOM = "{http://www.w3.org/2005/Atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


class HistoryFetchError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class HistoryRecord:
    key: str
    source_id: str
    month: str
    title: str
    url: str
    published: str


@dataclass(frozen=True)
class SourceCoverage:
    source_id: str
    # complete — все подходящие записи получены; partial — упёрлись в лимит;
    # unavailable — источник не ответил.
    state: str
    scanned: int
    admitted: int
    reason: str | None = None


@dataclass(frozen=True)
class TechnologyHistory:
    aliases: tuple[str, ...]
    as_of: date
    months: tuple[str, ...]
    records: tuple[HistoryRecord, ...]
    coverage: tuple[SourceCoverage, ...]

    @property
    def coverage_complete(self) -> bool:
        return all(item.state == "complete" for item in self.coverage)

    def materials(self) -> list[Material]:
        return [Material(record.key, record.source_id, record.month) for record in self.records]


def phrase_pattern(aliases: Sequence[str]) -> re.Pattern[str]:
    """Фраза в названии: дефисы и пробелы равнозначны, допускается множественное число."""
    variants = []
    for alias in aliases:
        tokens = re.findall(r"[a-z0-9]+", alias.casefold())
        if not tokens:
            continue
        parts = [re.escape(token[:-1]) + "(?:y|ies)" if token.endswith("y") and len(token) > 3
                 else re.escape(token) + "(?:s|es)?" for token in tokens]
        variants.append(r"[\s\-‐‑–/]*".join(parts))
    if not variants:
        raise ValueError("Нужна хотя бы одна фраза технологии латиницей или цифрами.")
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(variants) + r")(?![a-z0-9])", re.IGNORECASE)


def _bounds(months: Sequence[str]) -> tuple[date, date]:
    first_year, first_month = map(int, months[0].split("-"))
    last_year, last_month = map(int, months[-1].split("-"))
    return (date(first_year, first_month, 1),
            date(last_year, last_month, calendar.monthrange(last_year, last_month)[1]))


def _get(client: httpx.Client, url: str, params: dict[str, str], cancel: Event) -> bytes:
    if cancel.is_set():
        raise HistoryFetchError("cancelled")
    try:
        with client.stream("GET", url, params=params, timeout=TIMEOUT_SECONDS, follow_redirects=False,
                           headers={"User-Agent": USER_AGENT}) as response:
            if response.status_code == 429:
                raise HistoryFetchError("rate_limited")
            if response.status_code != 200:
                raise HistoryFetchError(f"http_{response.status_code}")
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                if cancel.is_set():
                    raise HistoryFetchError("cancelled")
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise HistoryFetchError("response_too_large")
    except httpx.TimeoutException:
        raise HistoryFetchError("timeout") from None
    except httpx.HTTPError:
        raise HistoryFetchError("source_unavailable") from None
    return bytes(body)


def arxiv_get(client: httpx.Client, params: dict[str, str], cancel: Event) -> bytes:
    """Запрос к arXiv: одно соединение за раз, начала запросов не чаще раза в 3 секунды.

    Пауза отсчитывается от начала прошлого запроса, а не от его конца: правило
    arXiv ограничивает частоту запросов, и время загрузки ответа в паузу входит.
    """
    with _ARXIV_LOCK:
        wait = _ARXIV_LAST[0] + ARXIV_INTERVAL_SECONDS - monotonic()
        if wait > 0:
            if cancel.wait(wait):
                raise HistoryFetchError("cancelled")
        _ARXIV_LAST[0] = monotonic()
        return _get(client, "https://export.arxiv.org/api/query", params, cancel)


def _admit(source_id: str, title: str, url: str, published: date | None, months: set[str],
           pattern: re.Pattern[str]) -> HistoryRecord | None:
    title = " ".join(title.split())
    if not title or published is None or not pattern.search(title):
        return None
    month = f"{published.year:04d}-{published.month:02d}"
    if month not in months:
        return None
    return HistoryRecord(work_key(title), source_id, month, title[:500], url, published.isoformat())


def _quoted(aliases: Sequence[str], field: str) -> str:
    return " OR ".join(f'{field}"{alias.replace(chr(34), "")}"' for alias in aliases)


def fetch_arxiv(client: httpx.Client, aliases: Sequence[str], months: Sequence[str], pattern: re.Pattern[str],
                cancel: Event) -> tuple[list[HistoryRecord], SourceCoverage]:
    start, end = _bounds(months)
    query = f"({_quoted(aliases, 'ti:')}) AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    root = ElementTree.fromstring(arxiv_get(client, {
        "search_query": query, "start": "0", "max_results": str(MAX_RECORDS),
        "sortBy": "submittedDate", "sortOrder": "descending"}, cancel))
    total = int(root.findtext(_OPENSEARCH + "totalResults") or 0)
    records, scanned, allowed = [], 0, set(months)
    for entry in root.findall(_ATOM + "entry"):
        scanned += 1
        stamp = entry.findtext(_ATOM + "published") or ""
        published = date.fromisoformat(stamp[:10]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", stamp) else None
        link = (entry.findtext(_ATOM + "id") or "").replace("http://", "https://")
        record = _admit("arxiv", entry.findtext(_ATOM + "title") or "", link, published, allowed, pattern)
        if record:
            records.append(record)
    state = "partial" if total > scanned else "complete"
    return records, SourceCoverage("arxiv", state, scanned, len(records))


def fetch_europe_pmc(client: httpx.Client, aliases: Sequence[str], months: Sequence[str],
                     pattern: re.Pattern[str], cancel: Event) -> tuple[list[HistoryRecord], SourceCoverage]:
    start, end = _bounds(months)
    query = f"({_quoted(aliases, 'TITLE:')}) AND FIRST_PDATE:[{start.isoformat()} TO {end.isoformat()}]"
    payload = json.loads(_get(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search", {
        "query": query, "resultType": "lite", "pageSize": str(MAX_RECORDS), "format": "json",
        "cursorMark": "*"}, cancel))
    total = int(payload.get("hitCount") or 0)
    results = (payload.get("resultList") or {}).get("result") or []
    records, allowed = [], set(months)
    for item in results:
        stamp = str(item.get("firstPublicationDate") or "")
        published = date.fromisoformat(stamp) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp) else None
        doi = item.get("doi")
        link = (f"https://doi.org/{doi}" if doi
                else f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}")
        record = _admit("europe_pmc", str(item.get("title") or ""), link, published, allowed, pattern)
        if record:
            records.append(record)
    state = "partial" if total > len(results) else "complete"
    return records, SourceCoverage("europe_pmc", state, len(results), len(records))


def fetch_hacker_news(client: httpx.Client, aliases: Sequence[str], months: Sequence[str],
                      pattern: re.Pattern[str], cancel: Event) -> tuple[list[HistoryRecord], SourceCoverage]:
    start, end = _bounds(months)
    first = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp())
    last = int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC).timestamp())
    records, scanned, allowed = [], 0, set(months)
    for alias in aliases:
        payload = json.loads(_get(client, "https://hn.algolia.com/api/v1/search", {
            "query": f'"{alias}"', "tags": "story", "hitsPerPage": str(MAX_RECORDS),
            "numericFilters": f"created_at_i>={first},created_at_i<={last}",
            "attributesToRetrieve": "title,url,created_at_i"}, cancel))
        hits = payload.get("hits") or []
        scanned += len(hits)
        for hit in hits:
            stamp = hit.get("created_at_i")
            published = datetime.fromtimestamp(stamp, UTC).date() if isinstance(stamp, int) else None
            link = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
            record = _admit("hacker_news", str(hit.get("title") or ""), link, published, allowed, pattern)
            if record:
                records.append(record)
        if int(payload.get("nbHits") or 0) > len(hits):
            return records, SourceCoverage("hacker_news", "partial", scanned, len(records))
    return records, SourceCoverage("hacker_news", "complete", scanned, len(records))


def fetch_crossref(provider: CrossrefProvider, aliases: Sequence[str], months: Sequence[str],
                   pattern: re.Pattern[str], cancel: Event) -> tuple[list[HistoryRecord], SourceCoverage]:
    start, end = _bounds(months)
    allowed, records, flags = set(months), [], []
    request = SearchRequest(topic=" ".join(aliases[:1]), source="crossref", from_date=start, until_date=end,
                            max_results=MAX_RECORDS)
    exhausted = False
    try:
        for page in provider.iter_pages(request, cancel):
            for document in page.documents:
                # Рецензии и решения редакции повторяют название статьи — это не новые работы.
                if document.document_type == "peer-review":
                    continue
                published = document.publication_date or (
                    date(document.publication_year, document.publication_month, 1)
                    if document.publication_year and document.publication_month else None)
                record = _admit("crossref", document.title, document.url, published, allowed, pattern)
                flags.append(record is not None)
                if record:
                    records.append(record)
            exhausted = page.exhausted
    except BackendError as error:
        raise HistoryFetchError(error.code) from None
    tail = flags[-max(1, len(flags) // 5):]
    saturated = exhausted or (sum(tail) / len(tail) if tail else 0.0) < SATURATION_SHARE
    return records, SourceCoverage("crossref", "complete" if saturated else "partial", len(flags), len(records))


# Сколько технологий спрашивать у arXiv одним запросом. Замер 28.09.2026: ТОП
# технологий проверял 40 фраз по одной, и правило «запрос раз в 3 секунды» одно
# держало этап минуты (80 запросов, когда не ответил OpenAlex).
ARXIV_BATCH = 8
# Фраза с таким числом работ за окно истории почти наверняка массовая и за всё
# время: её первое упоминание спрашивается отдельным дешёвым запросом, а не
# переполняет общую пачку (иначе пачка дробится до одиночных запросов).
ARXIV_POPULAR = 40
# Когда нужны и история, и первое упоминание (OpenAlex не ответил), пачка
# спрашивается сразу за всё время: полный ответ содержит и записи окна истории.
# arXiv отдаёт за раз до 2000 записей.
ARXIV_JOINT_RECORDS = 2000


def _aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))[:3]


class ArxivBatcher:
    """Запросы к arXiv сразу по нескольким технологиям.

    Фразы регистрируются заранее; первая технология пачки, которой понадобился
    arXiv, получает ответ одним запросом «фраза А ИЛИ фраза Б…» на всю пачку, а
    записи раскладываются по технологиям той же проверкой «фраза в названии».
    Ответ, не вместивший всех записей, дробит пачку пополам, до одной фразы —
    тогда это ровно прежний запрос. Результаты те же, что по одной фразе, пока
    ответ полон; запросов — в разы меньше.
    """

    def __init__(self, client: httpx.Client, months: Sequence[str], cancel: Event, *, size: int = ARXIV_BATCH,
                 exact_first: bool = False):
        self._client, self._months, self._cancel, self._size = client, tuple(months), cancel, size
        # Первое упоминание по одной фразе — ровно как считает сам arXiv (со
        # словоформами). Пачка считает по точной фразе в названии: на выборке
        # «green hydrogen» 28.09.2026 ТОП совпал, а у 14 из 52 исключённых
        # разошлись счёт за всё время (на 1–30) и порой год появления.
        self._exact_first = exact_first
        self._lock = Lock()
        self.requests = 0
        # kind → (batches of keys, key → batch index, per-batch lock, key → answer)
        self._kinds: dict[str, tuple[list[list[tuple[str, ...]]], dict[tuple[str, ...], int],
                                     list[Lock], dict[tuple[str, ...], object]]] = {
            "history": ([], {}, [], {}), "first": ([], {}, [], {})}

    def register(self, kind: str, aliases: Sequence[str]) -> None:
        key = _aliases(aliases)
        if not key:
            return
        with self._lock:
            batches, index, locks, _ = self._kinds[kind]
            if key in index:
                return
            if not batches or len(batches[-1]) >= self._size:
                batches.append([])
                locks.append(Lock())
            batches[-1].append(key)
            index[key] = len(batches) - 1

    def _answer(self, kind: str, aliases: Sequence[str], fetch: Callable[[list[tuple[str, ...]]], None]) -> object:
        key = _aliases(aliases)
        with self._lock:
            batches, index, locks, answers = self._kinds[kind]
            position = index.get(key)
        if position is None:
            return None
        with locks[position]:
            if key not in answers:
                fetch(list(batches[position]))
        return answers.get(key)

    def history(self, aliases: Sequence[str]) -> tuple[list[HistoryRecord], SourceCoverage] | None:
        """История arXiv этой технологии за окно; None — фраза не зарегистрирована."""
        answer = self._answer("history", aliases, self._fetch_history)
        if isinstance(answer, HistoryFetchError):
            raise answer
        return answer  # type: ignore[return-value]

    def first(self, aliases: Sequence[str]) -> tuple[date | None, int] | None:
        """Самая ранняя дата и число работ arXiv с фразой в названии за всё время."""
        answer = self._answer("first", aliases, self._fetch_first)
        if isinstance(answer, HistoryFetchError):
            raise answer
        return answer  # type: ignore[return-value]

    def _request(self, keys: list[tuple[str, ...]], extra: str, sort: str) -> tuple[int, list]:
        aliases = [alias for key in keys for alias in key]
        root = ElementTree.fromstring(arxiv_get(self._client, {
            "search_query": f"({_quoted(aliases, 'ti:')}){extra}", "start": "0", "max_results": str(MAX_RECORDS),
            "sortBy": "submittedDate", "sortOrder": sort}, self._cancel))
        with self._lock:
            self.requests += 1
        return int(root.findtext(_OPENSEARCH + "totalResults") or 0), root.findall(_ATOM + "entry")

    def _fetch_history(self, keys: list[tuple[str, ...]]) -> None:
        answers = self._kinds["history"][3]
        first_index, first_answers = self._kinds["first"][1], self._kinds["first"][3]
        if (not self._exact_first and len(keys) > 1
                and all(key in first_index and key not in first_answers for key in keys)):
            self._fetch_joint(keys)
            return
        if len(keys) == 1:
            try:
                answers[keys[0]] = fetch_arxiv(self._client, keys[0], self._months, phrase_pattern(keys[0]),
                                               self._cancel)
            except HistoryFetchError as error:
                answers[keys[0]] = error
            return
        start, end = _bounds(self._months)
        try:
            total, entries = self._request(keys, f" AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]",
                                           "descending")
        except (HistoryFetchError, ValueError, ElementTree.ParseError, DefusedXmlException) as error:
            failure = error if isinstance(error, HistoryFetchError) else HistoryFetchError("invalid_response")
            answers.update((key, failure) for key in keys)
            return
        if total > len(entries):
            half = len(keys) // 2
            self._fetch_history(keys[:half])
            self._fetch_history(keys[half:])
            return
        allowed = set(self._months)
        parsed = [(entry.findtext(_ATOM + "title") or "", entry.findtext(_ATOM + "published") or "",
                   (entry.findtext(_ATOM + "id") or "").replace("http://", "https://")) for entry in entries]
        for key in keys:
            pattern, records = phrase_pattern(key), []
            for title, stamp, link in parsed:
                published = date.fromisoformat(stamp[:10]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", stamp) else None
                record = _admit("arxiv", title, link, published, allowed, pattern)
                if record:
                    records.append(record)
            answers[key] = (records, SourceCoverage("arxiv", "complete", len(entries), len(records)))

    def _fetch_joint(self, keys: list[tuple[str, ...]]) -> None:
        """История и первое упоминание одним запросом за всё время, пока ответ полон."""
        history, first = self._kinds["history"][3], self._kinds["first"][3]
        if len(keys) == 1:
            self._fetch_history(keys)
            self._fetch_first(keys)
            return
        aliases = [alias for key in keys for alias in key]
        try:
            root = ElementTree.fromstring(arxiv_get(self._client, {
                "search_query": _quoted(aliases, "ti:"), "start": "0", "max_results": str(ARXIV_JOINT_RECORDS),
                "sortBy": "submittedDate", "sortOrder": "ascending"}, self._cancel))
            with self._lock:
                self.requests += 1
            total = int(root.findtext(_OPENSEARCH + "totalResults") or 0)
            entries = root.findall(_ATOM + "entry")
        except (HistoryFetchError, ValueError, ElementTree.ParseError, DefusedXmlException) as error:
            failure = error if isinstance(error, HistoryFetchError) else HistoryFetchError("invalid_response")
            history.update((key, failure) for key in keys)
            first.update((key, failure) for key in keys)
            return
        if total > len(entries):
            half = len(keys) // 2
            self._fetch_joint(keys[:half])
            self._fetch_joint(keys[half:])
            return
        allowed = set(self._months)
        parsed = [(" ".join((entry.findtext(_ATOM + "title") or "").split()), entry.findtext(_ATOM + "published") or "",
                   (entry.findtext(_ATOM + "id") or "").replace("http://", "https://")) for entry in entries]
        in_window = sum(1 for _, stamp, _ in parsed if stamp[:7] in allowed)
        for key in keys:
            pattern, records, dates = phrase_pattern(key), [], []
            for title, stamp, link in parsed:
                published = date.fromisoformat(stamp[:10]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", stamp) else None
                if published is None or not pattern.search(title):
                    continue
                dates.append(published)
                record = _admit("arxiv", title, link, published, allowed, pattern)
                if record:
                    records.append(record)
            history[key] = (records, SourceCoverage("arxiv", "complete", in_window, len(records)))
            first[key] = (min(dates) if dates else None, len(dates))

    def _fetch_first(self, keys: list[tuple[str, ...]]) -> None:
        answers = self._kinds["first"][3]
        known = self._kinds["history"][3]
        if self._exact_first and len(keys) > 1:
            for key in keys:
                self._fetch_first([key])
            return
        popular = [key for key in keys if isinstance((answer := known.get(key)), tuple)
                   and len(answer[0]) >= ARXIV_POPULAR]
        if popular and len(keys) > 1:
            for key in popular:
                self._fetch_first([key])
            keys = [key for key in keys if key not in popular]
            if not keys:
                return
        if len(keys) == 1:
            try:
                answers[keys[0]] = self._single_first(keys[0])
            except (HistoryFetchError, ValueError, ElementTree.ParseError, DefusedXmlException) as error:
                answers[keys[0]] = error if isinstance(error, HistoryFetchError) else HistoryFetchError(
                    "invalid_response")
            return
        try:
            total, entries = self._request(keys, "", "ascending")
        except (HistoryFetchError, ValueError, ElementTree.ParseError, DefusedXmlException) as error:
            failure = error if isinstance(error, HistoryFetchError) else HistoryFetchError("invalid_response")
            answers.update((key, failure) for key in keys)
            return
        if total > len(entries):
            half = len(keys) // 2
            self._fetch_first(keys[:half])
            self._fetch_first(keys[half:])
            return
        parsed = [(" ".join((entry.findtext(_ATOM + "title") or "").split()),
                   entry.findtext(_ATOM + "published") or "") for entry in entries]
        for key in keys:
            pattern = phrase_pattern(key)
            dates = [date.fromisoformat(stamp[:10]) for title, stamp in parsed
                     if pattern.search(title) and re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", stamp)]
            # Счёт — записи с фразой в названии: в полном ответе это все работы с ней.
            answers[key] = (min(dates) if dates else None, len(dates))

    def _single_first(self, key: tuple[str, ...]) -> tuple[date | None, int]:
        """Одна фраза — прежний запрос: одна самая ранняя запись и счёт arXiv."""
        root = ElementTree.fromstring(arxiv_get(self._client, {
            "search_query": _quoted(key, "ti:"), "start": "0", "max_results": "1",
            "sortBy": "submittedDate", "sortOrder": "ascending"}, self._cancel))
        with self._lock:
            self.requests += 1
        stamp = root.findtext(f"{_ATOM}entry/{_ATOM}published") or ""
        earliest = date.fromisoformat(stamp[:10]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", stamp) else None
        return earliest, int(root.findtext(_OPENSEARCH + "totalResults") or 0)


Fetcher = Callable[..., tuple[list[HistoryRecord], SourceCoverage]]


def collect_history(aliases: Sequence[str], as_of: date, *, months: int = DEFAULT_MONTHS,
                    client: httpx.Client | None = None, crossref: CrossrefProvider | None = None,
                    cancel: Event | None = None, arxiv: ArxivBatcher | None = None) -> TechnologyHistory:
    """Параллельный опрос источников; каждый отвечает за себя.

    С `arxiv` история arXiv берётся из общего пакетного запроса по нескольким технологиям.
    """
    aliases = _aliases(aliases)
    pattern = phrase_pattern(aliases)
    window = month_window(as_of, months)
    cancel = cancel or Event()
    own_client = client is None
    client = client or httpx.Client(limits=httpx.Limits(max_connections=4))
    crossref = crossref or CrossrefProvider(page_size=MAX_RECORDS)
    jobs: dict[str, Callable[[], tuple[list[HistoryRecord], SourceCoverage]]] = {
        "crossref": lambda: fetch_crossref(crossref, aliases, window, pattern, cancel),
        "arxiv": lambda: ((arxiv.history(aliases) if arxiv is not None else None)
                          or fetch_arxiv(client, aliases, window, pattern, cancel)),
        "europe_pmc": lambda: fetch_europe_pmc(client, aliases, window, pattern, cancel),
        "hacker_news": lambda: fetch_hacker_news(client, aliases, window, pattern, cancel),
    }
    records: list[HistoryRecord] = []
    coverage: list[SourceCoverage] = []
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {source: pool.submit(job) for source, job in jobs.items()}
            for source, future in futures.items():
                try:
                    found, covered = future.result()
                except (HistoryFetchError, ValueError, KeyError, TypeError, ElementTree.ParseError,
                        DefusedXmlException) as error:
                    reason = error.reason if isinstance(error, HistoryFetchError) else "invalid_response"
                    coverage.append(SourceCoverage(source, "unavailable", 0, 0, reason))
                    continue
                records.extend(found)
                coverage.append(covered)
    finally:
        if own_client:
            client.close()
    # Одна и та же запись в выдаче одного источника не удваивается.
    unique = {(record.source_id, record.key): record for record in records}
    return TechnologyHistory(aliases, as_of, window, tuple(sorted(unique.values(), key=lambda item: item.published)),
                             tuple(coverage))


def assess_technology(aliases: Sequence[str], as_of: date, **options) -> tuple[TechnologyHistory, CurveAssessment]:
    history = collect_history(aliases, as_of, **options)
    return history, assess_curve(history.materials(), as_of, months=len(history.months),
                                 coverage_complete=history.coverage_complete)
