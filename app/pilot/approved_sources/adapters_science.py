"""Bounded public metadata adapters for research and institutional sources.

Each adapter owns at most one HTTP client and exposes only short, local-use
observations. A failed source is reported as lost coverage, never as zero hits.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
import json
import math
import re
from threading import Event
from time import monotonic
from typing import Literal
from urllib.parse import urlsplit

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
import httpx

from app.pilot.approved_sources._http_safety import read_bounded_response
from app.pilot.approved_sources.contracts import ExternalObservation, ObservationPage, SourceFetchError, SourceId
from app.runtime.jobs import TaskCancelled
from app.topic_relevance import TopicProfile, topic_match


_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_ATOM = "{http://arxiv.org/schemas/atom}"
_ARXIV_CATEGORY_SCHEME = "http://arxiv.org/schemas/atom"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_RSS_ONE = "{http://purl.org/rss/1.0/}"
_DUBLIN_CORE = "{http://purl.org/dc/elements/1.1/}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_MAX_RESPONSE_BYTES = 2_000_000
_MAX_TIMEOUT_SECONDS = 15.0
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v[1-9]\d{0,3})?")
_ARXIV_CATEGORY = re.compile(r"[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)?")
_DOI_ID = re.compile(r"10\.\d{4,9}/[A-Za-z0-9._;()/:-]+")
_EXACT_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_EUROPE_PMC_ID = re.compile(r"[A-Za-z0-9]{1,80}")


def _check(cancel: Event) -> None:
    if cancel.is_set():
        raise TaskCancelled()


def _budget(query: str, limit: int, timeout_seconds: float) -> str:
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise SourceFetchError("invalid_query")
    if type(limit) is not int or limit < 1:
        raise SourceFetchError("invalid_budget")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise SourceFetchError("invalid_budget")
    return query.strip()


def _terms(query: str) -> tuple[str, ...]:
    # Search syntax is never passed through from the user to a remote parser.
    return tuple(word[:50] for word in re.findall(r"[^\W_]+", query, re.UNICODE) if len(word) >= 2)[:8]


def _text(value: object, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = re.sub(r"<[^>]{0,500}>", " ", value)
    return " ".join(unescape(cleaned).split())[:limit]


def _iso_date(value: object) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _exact_date(value: object) -> date | None:
    """Do not assign a month to dates recorded with only year/month precision."""
    return _iso_date(value) if isinstance(value, str) and _EXACT_DATE.fullmatch(value) else None


def _count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdecimal() and len(value) <= 16:
        result = int(value)
    else:
        return None
    return result if result >= 0 else None


def _json_object(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (UnicodeError, ValueError) as exc:
        raise SourceFetchError("invalid_response") from exc
    if not isinstance(value, dict):
        raise SourceFetchError("invalid_response")
    return value


def _remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise SourceFetchError("timeout")
    return remaining


def _xml_root(body: bytes):
    try:
        return ElementTree.fromstring(body)
    except (DefusedXmlException, ElementTree.ParseError, UnicodeError) as exc:
        raise SourceFetchError("invalid_response") from exc


class _HttpAdapter:
    source_id: str

    def __init__(self, client: httpx.Client | None = None):
        self._own_client = client is None
        self._client = client or httpx.Client(
            follow_redirects=False, timeout=_MAX_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def _get(self, url: str, *, params: dict[str, str] | None, timeout_seconds: float,
             cancel: Event, accept: str) -> bytes:
        _check(cancel)
        timeout = min(float(timeout_seconds), _MAX_TIMEOUT_SECONDS)
        deadline = monotonic() + timeout
        try:
            with self._client.stream("GET", url, params=params, timeout=timeout,
                                     follow_redirects=False,
                                     headers={"Accept": accept, "Accept-Encoding": "gzip, deflate"}) as response:
                if response.status_code == 429:
                    raise SourceFetchError("rate_limited")
                if 300 <= response.status_code < 400:
                    raise SourceFetchError("unexpected_redirect")
                if 500 <= response.status_code:
                    raise SourceFetchError("source_unavailable")
                if response.status_code != 200:
                    raise SourceFetchError("source_http_error")
                body = read_bounded_response(response, max_bytes=_MAX_RESPONSE_BYTES,
                                             deadline=deadline, cancel=cancel)
        except httpx.TimeoutException as exc:
            _check(cancel)
            raise SourceFetchError("timeout") from exc
        except httpx.HTTPError as exc:
            _check(cancel)
            raise SourceFetchError("source_unavailable") from exc
        _check(cancel)
        if not body:
            raise SourceFetchError("empty_response")
        return body


class ArxivAdapter(_HttpAdapter):
    source_id: Literal["arxiv"] = "arxiv"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        query = _budget(query, limit, timeout_seconds)
        terms = _terms(query)
        if not terms:
            raise SourceFetchError("invalid_query")
        # arXiv's legacy API requires one connection and >=3 seconds between
        # requests across an organisation. One adapter run makes one request.
        cap = min(limit, 100)
        expression = " AND ".join(f"all:{term}" for term in terms[:4])
        expression += f" AND submittedDate:[199101010000 TO {as_of:%Y%m%d}2359]"
        body = self._get("https://export.arxiv.org/api/query", params={
            "search_query": expression, "start": "0", "max_results": str(cap),
            "sortBy": "submittedDate", "sortOrder": "descending",
        }, timeout_seconds=timeout_seconds, cancel=cancel, accept="application/atom+xml")
        root = _xml_root(body)
        if root.tag != _ATOM + "feed":
            raise SourceFetchError("invalid_response")
        entries = root.findall(_ATOM + "entry")
        total = _count(root.findtext(_OPENSEARCH + "totalResults"))
        observations: list[ExternalObservation] = []
        for entry in entries[:cap]:
            _check(cancel)
            raw_url = (entry.findtext(_ATOM + "id") or "").strip()
            try:
                parsed = urlsplit(raw_url)
            except ValueError:
                continue
            article_id = parsed.path.removeprefix("/abs/")
            published = _iso_date(entry.findtext(_ATOM + "published"))
            title = _text(entry.findtext(_ATOM + "title"), limit=500)
            if (parsed.hostname not in {"arxiv.org", "export.arxiv.org"} or parsed.path == article_id
                    or not _ARXIV_ID.fullmatch(article_id) or not published or published > as_of or not title):
                continue
            identity = re.sub(r"v[1-9]\d{0,3}$", "", article_id)
            summary = _text(entry.findtext(_ATOM + "summary"), limit=300)
            primary = entry.find(_ARXIV_ATOM + "primary_category")
            category = None
            if primary is not None and primary.get("scheme") == _ARXIV_CATEGORY_SCHEME:
                term = primary.get("term", "")
                if len(term) <= 64 and _ARXIV_CATEGORY.fullmatch(term):
                    category = term
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=identity, kind="preprint", title=title,
                url="https://arxiv.org/abs/" + article_id, published_at=published,
                summary=summary or None, arxiv_primary_category=category,
            ))
        yield ObservationPage(observations=tuple(observations), scanned=min(len(entries), cap),
                              exhausted=total is not None and total <= len(entries), total_available=total)
        if limit > cap and (total is None or total > len(entries)):
            raise SourceFetchError("source_page_cap")


class EuropePmcAdapter(_HttpAdapter):
    source_id: Literal["europe_pmc"] = "europe_pmc"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        query = _budget(query, limit, timeout_seconds)
        terms = _terms(query)
        if not terms:
            raise SourceFetchError("invalid_query")
        cap = min(limit, 50)
        # FIRST_PDATE can be algorithmically inferred from an incomplete date.
        # The core response's explicit electronic date must corroborate the
        # first-publication date before an item enters a monthly series.
        expression = " AND ".join(f'TITLE_ABS:"{term}"' for term in terms[:4])
        expression += f" AND FIRST_PDATE:[1900-01-01 TO {as_of.isoformat()}] sort_date:y"
        body = self._get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
            "query": expression, "format": "json", "resultType": "core", "pageSize": str(cap),
        }, timeout_seconds=timeout_seconds, cancel=cancel, accept="application/json")
        payload = _json_object(body)
        result_list = payload.get("resultList")
        if not isinstance(result_list, dict) or not isinstance(result_list.get("result"), list):
            raise SourceFetchError("invalid_response")
        records = result_list["result"][:cap]
        total = _count(payload.get("hitCount"))
        observations: list[ExternalObservation] = []
        for record in records:
            _check(cancel)
            if not isinstance(record, dict):
                continue
            source = record.get("source")
            identity = record.get("id")
            if source not in {"MED", "PMC", "PPR"} or not isinstance(identity, str):
                continue
            if not _EUROPE_PMC_ID.fullmatch(identity):
                continue
            first = _exact_date(record.get("firstPublicationDate"))
            electronic = _exact_date(record.get("electronicPublicationDate"))
            if first is None or electronic != first or first > as_of:
                continue
            title = _text(record.get("title"), limit=500)
            if not title:
                continue
            doi = record.get("doi")
            pmid = record.get("pmid")
            if isinstance(doi, str) and _DOI_ID.fullmatch(doi):
                work_id = doi.casefold()
            elif isinstance(pmid, str) and re.fullmatch(r"[1-9]\d{0,11}", pmid):
                work_id = "pmid:" + pmid
            else:
                continue
            summary = _text(record.get("abstractText"), limit=300)
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=work_id,
                kind="preprint" if source == "PPR" else "journal_article",
                title=title, url=f"https://europepmc.org/article/{source}/{identity}",
                published_at=first, summary=summary or None,
            ))
        scanned = len(records)
        yield ObservationPage(observations=tuple(observations), scanned=scanned,
                              exhausted=total is not None and total <= scanned, total_available=total)
        if limit > cap and (total is None or total > scanned):
            raise SourceFetchError("source_page_cap")


class BiorxivAdapter(_HttpAdapter):
    source_id: Literal["biorxiv"] = "biorxiv"

    def _recent_feed(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                     cancel: Event) -> Iterator[ObservationPage]:
        """Use bioRxiv's own latest-30 feed when its API returns no body."""
        body = self._get("https://connect.biorxiv.org/biorxiv_xml.php", params={"subject": "all"},
                         timeout_seconds=timeout_seconds, cancel=cancel, accept="application/xml")
        root = _xml_root(body)
        if root.tag != _RDF + "RDF":
            raise SourceFetchError("invalid_response")
        items = root.findall(_RSS_ONE + "item")
        if not items:
            raise SourceFetchError("empty_response")
        cap = min(limit, 30)
        profile = TopicProfile.build(query)
        observations: list[ExternalObservation] = []
        for item in items[:cap]:
            _check(cancel)
            raw_doi = (item.findtext(_DUBLIN_CORE + "identifier") or "").strip()
            doi = raw_doi.removeprefix("doi:")
            title = _text(item.findtext(_RSS_ONE + "title"), limit=500)
            full_summary = _text(item.findtext(_RSS_ONE + "description"), limit=10000)
            published = _iso_date(item.findtext(_DUBLIN_CORE + "date"))
            if (not _DOI_ID.fullmatch(doi) or not title or not published or published > as_of
                    or not topic_match(profile, title, full_summary)):
                continue
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=doi, kind="preprint", title=title,
                url="https://www.biorxiv.org/content/" + doi, published_at=published,
                summary=full_summary[:300] or None,
            ))
        yield ObservationPage(observations=tuple(observations), scanned=min(len(items), cap),
                              exhausted=False, total_available=None)
        raise SourceFetchError("feed_window_only")

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        query = _budget(query, limit, timeout_seconds)
        terms = tuple(term.casefold() for term in _terms(query) if len(term) >= 3)
        if not terms:
            raise SourceFetchError("invalid_query")
        profile = TopicProfile.build(query)
        # bioRxiv has no keyword search. Its numeric endpoint returns the N
        # *most recent* posts, avoiding the oldest-first bias of a wide date
        # interval. For a historical as-of date use a narrow date window.
        cap = min(limit, 120)
        recent = as_of >= datetime.now(UTC).date() - timedelta(days=1)
        start = as_of - timedelta(days=7)
        scanned = cursor = 0
        deadline = monotonic() + timeout_seconds
        for _ in range(4):
            _check(cancel)
            path = (f"https://api.biorxiv.org/details/biorxiv/{cap}/{cursor}" if recent else
                    f"https://api.biorxiv.org/details/biorxiv/{start.isoformat()}/{as_of.isoformat()}/{cursor}")
            try:
                body = self._get(
                    path, params=None, timeout_seconds=_remaining(deadline),
                    cancel=cancel, accept="application/json",
                )
            except SourceFetchError as error:
                if error.code == "empty_response" and scanned == 0 and recent:
                    yield from self._recent_feed(query, as_of=as_of, limit=limit,
                                                 timeout_seconds=_remaining(deadline), cancel=cancel)
                    return
                raise
            payload = _json_object(body)
            collection = payload.get("collection")
            messages = payload.get("messages")
            if not isinstance(collection, list) or not isinstance(messages, list):
                raise SourceFetchError("invalid_response")
            total = _count(messages[0].get("count")) if messages and isinstance(messages[0], dict) else None
            remaining = cap - scanned
            chunk = collection[:remaining]
            observations: list[ExternalObservation] = []
            for item in chunk:
                _check(cancel)
                if not isinstance(item, dict):
                    continue
                doi = item.get("doi")
                title = _text(item.get("title"), limit=500)
                full_summary = _text(item.get("abstract"), limit=10000)
                summary = full_summary[:300]
                published = _iso_date(item.get("date"))
                if (not isinstance(doi, str) or not _DOI_ID.fullmatch(doi) or not title
                        or not published or published > as_of):
                    continue
                if not topic_match(profile, title, full_summary):
                    continue
                observations.append(ExternalObservation(
                    source_id=self.source_id, item_id=doi, kind="preprint", title=title,
                    url="https://www.biorxiv.org/content/" + doi, published_at=published,
                    summary=summary or None,
                ))
            scanned += len(chunk)
            cursor += len(collection)
            sample_end = total is not None and cursor >= total and len(chunk) == len(collection)
            # A numeric N sample or seven-day interval never proves exhaustion
            # of the entire bioRxiv collection or topical search space.
            yield ObservationPage(observations=tuple(observations), scanned=len(chunk),
                                  exhausted=False, total_available=None)
            if sample_end or scanned >= cap or not collection or len(collection) < 30:
                break
        if scanned < limit:
            raise SourceFetchError("recent_window_only")


class OpenReviewAdapter(_HttpAdapter):
    source_id: Literal["openreview"] = "openreview"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        query = _budget(query, limit, timeout_seconds)
        terms = _terms(query)
        if not terms:
            raise SourceFetchError("invalid_query")
        cap = min(limit, 50)
        body = self._get("https://api2.openreview.net/notes/search", params={
            "term": " ".join(terms), "source": "forum", "sort": "cdate:desc", "limit": str(cap),
        }, timeout_seconds=timeout_seconds, cancel=cancel, accept="application/json")
        payload = _json_object(body)
        notes = payload.get("notes")
        if not isinstance(notes, list):
            raise SourceFetchError("invalid_response")
        total = _count(payload.get("count"))
        observations: list[ExternalObservation] = []
        for note in notes[:cap]:
            _check(cancel)
            if not isinstance(note, dict):
                continue
            identity = note.get("id")
            content = note.get("content")
            if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{5,100}", identity):
                continue
            if not isinstance(content, dict):
                continue
            title_value = content.get("title")
            abstract_value = content.get("abstract")
            title = _text(title_value.get("value") if isinstance(title_value, dict) else title_value, limit=500)
            summary = _text(abstract_value.get("value") if isinstance(abstract_value, dict) else abstract_value,
                            limit=300)
            # odate is when the submission first became public; pdate records
            # acceptance. Public search results often omit odate. For a
            # current-day scan, immutable tcdate is still a useful lead, but
            # it must be labelled as creation rather than publication. A
            # historical as-of scan cannot infer when it became public.
            public_timestamp = _count(note.get("odate"))
            created_timestamp = _count(note.get("tcdate"))
            use_creation = not public_timestamp and as_of >= datetime.now(UTC).date()
            timestamp = created_timestamp if use_creation else public_timestamp
            try:
                published = datetime.fromtimestamp(timestamp / 1000, UTC).date() if timestamp else None
            except (OverflowError, OSError, ValueError):
                published = None
            if not title or not summary or not published or published > as_of:
                continue
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=identity, kind="preprint", title=title,
                url="https://openreview.net/forum?id=" + identity, published_at=published,
                date_basis="created" if use_creation else "published",
                summary=summary or None,
            ))
        yield ObservationPage(observations=tuple(observations), scanned=min(len(notes), cap),
                              exhausted=total is not None and total <= len(notes), total_available=total)
        if limit > cap and (total is None or total > len(notes)):
            raise SourceFetchError("source_page_cap")


class ZenodoAdapter(_HttpAdapter):
    source_id: Literal["zenodo"] = "zenodo"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        query = _budget(query, limit, timeout_seconds)
        terms = _terms(query)
        if not terms:
            raise SourceFetchError("invalid_query")
        # Anonymous public-record search allows at most 25 records per page.
        page_size = min(limit, 25)
        scanned = 0
        exhausted = False
        deadline = monotonic() + timeout_seconds
        for page_number in range(1, 5):
            _check(cancel)
            cap = min(page_size, limit - scanned)
            if cap <= 0:
                break
            body = self._get("https://zenodo.org/api/records", params={
                "q": " ".join(terms), "sort": "mostrecent", "status": "published",
                "size": str(cap), "page": str(page_number),
            }, timeout_seconds=_remaining(deadline), cancel=cancel, accept="application/json")
            payload = _json_object(body)
            hits = payload.get("hits")
            if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
                raise SourceFetchError("invalid_response")
            records = hits["hits"][:cap]
            total_value = hits.get("total")
            total = _count(total_value.get("value")) if isinstance(total_value, dict) else _count(total_value)
            observations: list[ExternalObservation] = []
            for record in records:
                _check(cancel)
                if not isinstance(record, dict) or not isinstance(record.get("metadata"), dict):
                    continue
                metadata = record["metadata"]
                identity = _count(record.get("id"))
                title = _text(metadata.get("title") or record.get("title"), limit=500)
                summary = _text(metadata.get("description"), limit=300)
                published = _iso_date(metadata.get("publication_date"))
                if not identity or not title or not published or published > as_of:
                    continue
                observations.append(ExternalObservation(
                    source_id=self.source_id, item_id=str(identity), kind="research_artifact", title=title,
                    url=f"https://zenodo.org/records/{identity}", published_at=published,
                    summary=summary or None,
                ))
            scanned += len(records)
            exhausted = total is not None and scanned >= total
            yield ObservationPage(observations=tuple(observations), scanned=len(records),
                                  exhausted=exhausted, total_available=total)
            if exhausted or scanned >= limit or len(records) < cap:
                break
        if scanned < limit and not exhausted:
            raise SourceFetchError("source_page_cap")


class NistNewsAdapter(_HttpAdapter):
    source_id: Literal["nist_news"] = "nist_news"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        query = _budget(query, limit, timeout_seconds)
        terms = tuple(term.casefold() for term in _terms(query) if len(term) >= 3)
        if not terms:
            raise SourceFetchError("invalid_query")
        profile = TopicProfile.build(query)
        body = self._get("https://www.nist.gov/news-events/news/rss.xml", params=None,
                         timeout_seconds=timeout_seconds, cancel=cancel, accept="application/rss+xml")
        root = _xml_root(body)
        if root.tag != "rss":
            raise SourceFetchError("invalid_response")
        items = root.findall("./channel/item")
        cap = min(limit, 100)
        observations: list[ExternalObservation] = []
        for item in items[:cap]:
            _check(cancel)
            url = (item.findtext("link") or "").strip()
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            title = _text(item.findtext("title"), limit=500)
            full_summary = _text(item.findtext("description"), limit=5000)
            summary = full_summary[:300]
            raw_date = item.findtext("pubDate")
            try:
                moment = parsedate_to_datetime(raw_date) if raw_date else None
                published = moment.astimezone(UTC).date() if moment and moment.tzinfo else None
            except (TypeError, ValueError, OverflowError):
                published = None
            if (parsed.scheme != "https" or parsed.hostname not in {"nist.gov", "www.nist.gov"}
                    or not title or not published or published > as_of
                    or not topic_match(profile, title, full_summary)):
                continue
            try:
                observations.append(ExternalObservation(
                    source_id=self.source_id, item_id=sha256(url.encode("utf-8")).hexdigest(),
                    kind="institution_news", title=title, url=url, published_at=published,
                    summary=summary or None,
                ))
            except ValueError:
                # One malformed RSS link must not erase otherwise valid news.
                continue
        # RSS exposes only a moving latest-items window, never the full archive.
        yield ObservationPage(observations=tuple(observations), scanned=min(len(items), cap),
                              exhausted=False, total_available=None)
        if len(items) < limit:
            raise SourceFetchError("feed_window_only")


def make_science_adapters() -> dict[SourceId, _HttpAdapter]:
    """Create a fresh, independently owned session for each approved source."""
    return {adapter.source_id: adapter for adapter in (
        ArxivAdapter(), BiorxivAdapter(), OpenReviewAdapter(), ZenodoAdapter(), NistNewsAdapter(),
        EuropePmcAdapter(),
    )}
