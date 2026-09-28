"""Bounded public news, community, and software discovery adapters.

These sources supply leads for later verification.  In particular, a news
headline or a repository description is not treated as a research finding.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
from ipaddress import ip_address
import json
import re
import ssl
from threading import Event
from time import monotonic
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import certifi
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

import httpx

from app.input_safety import is_safe_http_url
from app.pilot.approved_sources._http_safety import read_bounded_response
from app.topic_relevance import TopicProfile, topic_match
from app.pilot.approved_sources.contracts import (
    ExternalObservation, ObservationKind, ObservationPage, SourceFetchError, SourceId,
)
from app.runtime.jobs import TaskCancelled
from app.runtime.backup import ArchiveError, assert_no_credentials, strip_link_credentials


_MAX_QUERY_CHARS = 200
_MAX_HTTP_SECONDS = 20.0
_MAX_RESPONSE_BYTES = 4_000_000
_TAG_RE = re.compile(r"<[^>]*>")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")
_GENERIC_TERMS = frozenset({"research", "science", "technology", "technologies", "news", "innovation"})
_PRIVATE_QUERY_KEYS = frozenset({
    "apikey", "accesstoken", "refreshtoken", "password", "secret", "authorization", "credential", "token",
})


def _check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise TaskCancelled()


def _query(query: str) -> str:
    cleaned = " ".join(query.split())
    if not cleaned:
        raise SourceFetchError("invalid_query")
    if len(cleaned) > _MAX_QUERY_CHARS:
        # Long generated plans still contribute their first complete terms.
        # Re-tokenizing avoids sending an unbalanced quote or parenthesis to
        # providers with query grammars (especially GDELT).
        words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", cleaned, flags=re.UNICODE)
        selected: list[str] = []
        for word in words:
            if len(" ".join([*selected, word])) > _MAX_QUERY_CHARS:
                break
            selected.append(word)
        cleaned = " ".join(selected)
    if not cleaned:
        raise SourceFetchError("invalid_query")
    return cleaned


def _terms(query: str) -> tuple[str, ...]:
    """Content words of the query (kept for callers that still need plain terms)."""
    terms = tuple(dict.fromkeys(
        word.casefold() for word in _WORD_RE.findall(query) if word.casefold() not in _GENERIC_TERMS
    ))
    return terms[:8]


def _profile(query: str, localized: Mapping[str, str] | None = None) -> TopicProfile:
    """Topic of a feed filter: the search formulation and its localized versions."""
    extra = [text for text in (localized or {}).values() if isinstance(text, str) and text.strip()]
    return TopicProfile.build(query, None, synonyms=extra)


def _matches(profile: TopicProfile, *parts: str) -> bool:
    """A feed item belongs to the topic only with its phrase or most of its terms.

    Accepting any single query word let «state» from «solid-state batteries»
    admit almost any story; the final relevance check could only hide it later.
    """
    title, *rest = (*parts, "")
    return topic_match(profile, title, " ".join(part for part in rest if part))


def _summary(value: object, *, max_chars: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    plain = unescape(_TAG_RE.sub(" ", value))
    plain = " ".join(plain.split())
    return plain[:max_chars] or None


def _date(value: object) -> date | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, tz=UTC).date()
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if re.fullmatch(r"\d{8}T\d{6}Z", raw):
            return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").date()
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return parsedate_to_datetime(raw).date()
        except (TypeError, ValueError, IndexError):
            return None


def _public_url(value: object, *, allowed_hosts: frozenset[str] | None = None) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password:
            return None
        if allowed_hosts is not None and host not in allowed_hosts:
            return None
        if host == "localhost" or host.endswith((".local", ".internal")):
            return None
        try:
            if not ip_address(host).is_global:
                return None
        except ValueError:
            pass
        # Source URLs can carry publisher-specific credentials or tracking.
        # Keep ordinary article identifiers while removing credential fields.
        safe_query = urlencode([
            (key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
            if re.sub(r"[^a-z0-9]", "", key.casefold()) not in _PRIVATE_QUERY_KEYS
        ])
        safe_url = strip_link_credentials(urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                                                       safe_query, "")))
        if not is_safe_http_url(safe_url):
            return None
        assert_no_credentials(safe_url)
        return safe_url
    except (ValueError, ArchiveError):
        return None


def _stable_id(url: str) -> str:
    return sha256(url.encode("utf-8")).hexdigest()[:24]


class _PublicAdapter:
    source_id: SourceId
    # Язык выдачи источника: русскоязычным отдаётся русская формулировка запроса.
    language = "en"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._own_client = client is None
        self._client = client or httpx.Client(follow_redirects=False, timeout=_MAX_HTTP_SECONDS)
        self._countries: tuple[str, ...] = ()
        self._editions: tuple[str, ...] = ()
        self._localized: dict[str, str] = {}

    def configure(self, *, countries: tuple[str, ...] = (), editions: tuple[str, ...] = (),
                  localized: Mapping[str, str] | None = None) -> None:
        """Страны владельца и формулировки запроса на языках источника."""
        self._countries = tuple(countries)
        self._editions = tuple(editions)
        self._localized = {key: value for key, value in (localized or {}).items()
                           if isinstance(key, str) and isinstance(value, str) and value.strip()}

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def _post_json(self, url: str, payload: Mapping[str, object], *, deadline: float, cancel: Event,
                   max_bytes: int = _MAX_RESPONSE_BYTES) -> object:
        _check_cancel(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise SourceFetchError("timeout")
        try:
            with self._client.stream(
                "POST", url, json=dict(payload), timeout=min(remaining, _MAX_HTTP_SECONDS), follow_redirects=False,
                headers={"User-Agent": "Trendanalyser/1.0 (public research metadata)",
                         "Accept": "application/json", "Accept-Encoding": "gzip, deflate"},
            ) as response:
                if response.status_code == 429:
                    raise SourceFetchError("rate_limited")
                if not 200 <= response.status_code < 300:
                    raise SourceFetchError("source_http_error")
                body = read_bounded_response(response, max_bytes=max_bytes, deadline=deadline, cancel=cancel)
        except SourceFetchError:
            raise
        except TaskCancelled:
            raise
        except (httpx.HTTPError, OSError):
            raise SourceFetchError("source_unavailable") from None
        try:
            return json.loads(body)
        except (UnicodeError, json.JSONDecodeError):
            raise SourceFetchError("invalid_response") from None

    def _get(self, url: str, *, params: Mapping[str, str | int] | None, deadline: float,
             cancel: Event, max_bytes: int = _MAX_RESPONSE_BYTES,
             max_request_seconds: float = _MAX_HTTP_SECONDS) -> bytes:
        _check_cancel(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise SourceFetchError("timeout")
        try:
            with self._client.stream(
                "GET", url, params=params, timeout=min(remaining, max_request_seconds, _MAX_HTTP_SECONDS),
                follow_redirects=False,
                headers={"User-Agent": "Trendanalyser/1.0 (public research metadata)",
                         "Accept": "application/json, application/rss+xml, application/xml",
                         "Accept-Encoding": "gzip, deflate"},
            ) as response:
                if response.status_code == 429:
                    raise SourceFetchError("rate_limited")
                if not 200 <= response.status_code < 300:
                    raise SourceFetchError("source_http_error")
                return read_bounded_response(response, max_bytes=max_bytes, deadline=deadline, cancel=cancel)
        except SourceFetchError:
            raise
        except TaskCancelled:
            raise
        except (httpx.HTTPError, OSError):
            # An exception may contain the original URL (including credentials).
            raise SourceFetchError("source_unavailable") from None

    def _json(self, url: str, *, params: Mapping[str, str | int] | None, deadline: float,
              cancel: Event, max_bytes: int = _MAX_RESPONSE_BYTES,
              max_request_seconds: float = _MAX_HTTP_SECONDS) -> object:
        try:
            return json.loads(self._get(url, params=params, deadline=deadline, cancel=cancel,
                                        max_bytes=max_bytes, max_request_seconds=max_request_seconds))
        except (UnicodeError, json.JSONDecodeError):
            raise SourceFetchError("invalid_response") from None


class _RssAdapter(_PublicAdapter):
    feed_url: str
    allowed_hosts: frozenset[str]
    kind: ObservationKind = "institution_news"
    country: str | None = None
    recent_only = False
    rights: Literal["local_only", "share_allowed"] = "local_only"
    license_ref: str | None = None

    def _feed_bytes(self, *, deadline: float, cancel: Event, query: str = "") -> bytes:
        return self._get(self.feed_url, params=None, deadline=deadline, cancel=cancel)

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        terms = _profile(_query(query), self._localized)
        cap = max(0, min(int(limit), 100))
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        deadline = monotonic() + max(0.1, min(float(timeout_seconds), _MAX_HTTP_SECONDS))
        try:
            root = ElementTree.fromstring(self._feed_bytes(deadline=deadline, cancel=cancel, query=query))
        except (ElementTree.ParseError, DefusedXmlException):
            raise SourceFetchError("invalid_response") from None
        items = root.findall("./channel/item")
        if not items:
            raise SourceFetchError("invalid_response")
        observed_at = datetime.now(UTC)
        observations: list[ExternalObservation] = []
        scanned = 0
        for item in items:
            _check_cancel(cancel)
            if scanned >= cap:
                break
            scanned += 1
            title = _summary(item.findtext("title"), max_chars=250)
            url = _public_url(item.findtext("link"), allowed_hosts=self.allowed_hosts)
            published = _date(item.findtext("pubDate"))
            full_description = _summary(item.findtext("description"), max_chars=10_000)
            description = (full_description or "")[:300] or None
            if not title or not url or not published or published > as_of:
                continue
            if not _matches(terms, title, full_description or ""):
                continue
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=_stable_id(url), kind=self.kind,
                title=title, url=url, published_at=published, observed_at=observed_at,
                summary=description, rights=self.rights, license_ref=self.license_ref,
                country=self.country,
            ))
        yield ObservationPage(observations=tuple(observations), scanned=scanned,
                              exhausted=not self.recent_only and scanned >= len(items),
                              total_available=None if self.recent_only else len(items))
        if self.recent_only and scanned < limit:
            raise SourceFetchError("recent_feed_only")


class MitResearchNewsAdapter(_RssAdapter):
    source_id = "mit_research_news"
    feed_url = "https://news.mit.edu/rss/research"
    allowed_hosts = frozenset({"news.mit.edu"})
    country = "US"

    def _feed_bytes(self, *, deadline: float, cancel: Event, query: str = "") -> bytes:
        if not self._own_client:
            return super()._feed_bytes(deadline=deadline, cancel=cancel)
        # MIT's Pantheon edge currently rejects httpx's transport with 403,
        # while the standard-library HTTPS client receives the public feed.
        # Keep the injected client path for deterministic offline tests.
        _check_cancel(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise SourceFetchError("timeout")

        class _NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, msg, headers, newurl):
                return None

        opener = build_opener(
            _NoRedirect(), HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
        )
        request = Request(self.feed_url, headers={"User-Agent": "Trendanalyser/1.0", "Accept": "application/rss+xml"})
        try:
            with opener.open(request, timeout=min(remaining, _MAX_HTTP_SECONDS)) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdecimal():
                    if len(content_length) > 20 or int(content_length) > _MAX_RESPONSE_BYTES:
                        raise SourceFetchError("response_too_large")
                content = bytearray()
                while True:
                    _check_cancel(cancel)
                    if monotonic() >= deadline:
                        raise SourceFetchError("timeout")
                    # read1 returns available bytes without waiting to fill the
                    # buffer, so a slow stream cannot hold a worker indefinitely.
                    chunk = response.read1(64_000)
                    if monotonic() >= deadline:
                        raise SourceFetchError("timeout")
                    if not chunk:
                        return bytes(content)
                    content.extend(chunk)
                    if len(content) > _MAX_RESPONSE_BYTES:
                        raise SourceFetchError("response_too_large")
        except HTTPError as error:
            raise SourceFetchError("rate_limited" if error.code == 429 else "source_http_error") from None
        except (URLError, OSError):
            raise SourceFetchError("source_unavailable") from None


class HorizonMagazineAdapter(_RssAdapter):
    source_id = "horizon_magazine"
    feed_url = "https://projects.research-and-innovation.ec.europa.eu/en/horizon-magazine/articles.xml"
    allowed_hosts = frozenset({"projects.research-and-innovation.ec.europa.eu"})
    rights = "local_only"
    country = "EU"


class HabrAdapter(_RssAdapter):
    """Russian technology discussions; never scientific confirmation.

    Хабр ищет по русской формулировке запроса через свою поисковую RSS; если
    поиск недоступен, остаётся прежняя общая лента свежих статей.
    """

    source_id = "habr"
    feed_url = "https://habr.com/ru/rss/all/all/?fl=ru"
    search_url = "https://habr.com/ru/rss/search/"
    allowed_hosts = frozenset({"habr.com"})
    kind = "community"
    recent_only = True
    language = "ru"
    country = "RU"

    def _feed_bytes(self, *, deadline: float, cancel: Event, query: str = "") -> bytes:
        search = self._localized.get("ru") or query
        if search.strip():
            try:
                return self._get(self.search_url, params={"q": _query(search), "target_type": "posts",
                                                          "order": "date", "fl": "ru"},
                                 deadline=deadline, cancel=cancel)
            except SourceFetchError as error:
                if error.code in {"timeout", "rate_limited"}:
                    raise
        return self._get(self.feed_url, params=None, deadline=deadline, cancel=cancel)


class GitHubRepositoryAdapter(_PublicAdapter):
    source_id = "github"
    _url = "https://api.github.com/search/repositories"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean_query = _query(query)
        cap = max(0, min(int(limit), 100))
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        deadline = monotonic() + max(0.1, min(float(timeout_seconds), _MAX_HTTP_SECONDS))
        data = self._json(self._url, params={"q": clean_query, "sort": "updated", "order": "desc", "per_page": cap},
                          deadline=deadline, cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise SourceFetchError("invalid_response")
        items = data["items"][:cap]
        total = data.get("total_count")
        if not isinstance(total, int) or total < 0:
            total = None
        observed_at = datetime.now(UTC)
        observations: list[ExternalObservation] = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            title = _summary(item.get("full_name"), max_chars=250)
            url = _public_url(item.get("html_url"), allowed_hosts=frozenset({"github.com"}))
            published = _date(item.get("created_at"))
            item_id = item.get("id")
            if not title or not url or not published or published > as_of or not isinstance(item_id, int):
                continue
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=str(item_id), kind="repository", title=title,
                url=url, published_at=published, observed_at=observed_at,
                summary=_summary(item.get("description")), rights="local_only",
            ))
        scanned = len(items)
        yield ObservationPage(observations=tuple(observations), scanned=scanned,
                              exhausted=total is not None and scanned >= total,
                              total_available=total)


class HackerNewsAdapter(_PublicAdapter):
    """Обсуждения Hacker News: поиск Algolia по запросу, запасной путь — свежая выборка.

    Прежде бралось 20 свежих и популярных историй и фильтровалось по любому слову
    запроса; почти все они были не по теме. Поиск Algolia (hn.algolia.com)
    открыт без ключа и ищет по всему архиву.
    """

    source_id = "hacker_news"
    _base = "https://hacker-news.firebaseio.com/v0"
    _search = "https://hn.algolia.com/api/v1/search"
    _search_cap = 50
    _new_cap = 14
    _top_cap = 6

    def _search_pages(self, query: str, *, as_of: date, limit: int, deadline: float,
                      cancel: Event, profile: TopicProfile) -> ObservationPage:
        cap = max(1, min(int(limit), self._search_cap))
        data = self._json(self._search, params={"query": query, "tags": "story", "hitsPerPage": cap},
                          deadline=deadline, cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
            raise SourceFetchError("invalid_response")
        hits = data["hits"][:cap]
        total = data.get("nbHits")
        observed_at = datetime.now(UTC)
        observations: list[ExternalObservation] = []
        for hit in hits:
            _check_cancel(cancel)
            if not isinstance(hit, dict):
                continue
            item_id = hit.get("objectID")
            title = _summary(hit.get("title"), max_chars=250)
            published = _date(hit.get("created_at_i")) or _date(hit.get("created_at"))
            text = _summary(hit.get("story_text"), max_chars=10_000)
            if (not isinstance(item_id, str) or not item_id.isdecimal() or len(item_id) > 12
                    or not title or not published or published > as_of):
                continue
            if not _matches(profile, title, text or ""):
                continue
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=item_id, kind="community", title=title,
                url=f"https://news.ycombinator.com/item?id={item_id}", published_at=published,
                observed_at=observed_at, summary=(text or "")[:300] or None, rights="local_only",
            ))
        return ObservationPage(observations=tuple(observations), scanned=len(hits),
                               exhausted=len(hits) < cap or isinstance(total, int) and total <= len(hits),
                               total_available=total if isinstance(total, int) and 0 <= total <= 1_000_000_000
                               else None)

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean_query = _query(query)
        terms = _profile(clean_query, self._localized)
        if not max(0, int(limit)):
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        deadline = monotonic() + max(0.1, min(float(timeout_seconds), _MAX_HTTP_SECONDS))
        try:
            page = self._search_pages(clean_query, as_of=as_of, limit=limit, deadline=deadline,
                                      cancel=cancel, profile=terms)
        except SourceFetchError as error:
            if error.code in {"timeout"}:
                raise
        else:
            yield page
            if not page.exhausted and page.scanned < limit:
                raise SourceFetchError("source_limit")
            return
        cap = max(0, min(int(limit), self._new_cap + self._top_cap))
        # One recent and one popular slice expose both early and established
        # discussion while keeping request volume strictly bounded.
        new_ids = self._json(f"{self._base}/newstories.json", params=None, deadline=deadline, cancel=cancel,
                             max_bytes=100_000)
        top_ids = self._json(f"{self._base}/topstories.json", params=None, deadline=deadline, cancel=cancel,
                             max_bytes=100_000)
        if not isinstance(new_ids, list) or not isinstance(top_ids, list):
            raise SourceFetchError("invalid_response")
        chosen: list[int] = []
        new_share = min(self._new_cap, max(1, cap - min(self._top_cap, cap // 4)))
        for value in [*new_ids[:new_share], *top_ids[:min(self._top_cap, cap - new_share)]]:
            if isinstance(value, int) and value > 0 and value not in chosen:
                chosen.append(value)
        observed_at = datetime.now(UTC)
        observations: list[ExternalObservation] = []
        scanned = 0
        failure: str | None = None
        for item_id in chosen[:cap]:
            _check_cancel(cancel)
            try:
                item = self._json(f"{self._base}/item/{item_id}.json", params=None, deadline=deadline,
                                  cancel=cancel, max_bytes=150_000, max_request_seconds=2.5)
            except SourceFetchError as error:
                failure = error.code
                break
            scanned += 1
            if isinstance(item, dict) and not item.get("deleted") and not item.get("dead") and item.get("type") == "story":
                title = _summary(item.get("title"), max_chars=250)
                published = _date(item.get("time"))
                full_text = _summary(item.get("text"), max_chars=10_000)
                if title and published and published <= as_of and _matches(terms, title, full_text or ""):
                    observations.append(ExternalObservation(
                        source_id=self.source_id, item_id=str(item_id), kind="community",
                        title=title, url=f"https://news.ycombinator.com/item?id={item_id}",
                        published_at=published, observed_at=observed_at,
                        summary=(full_text or "")[:300] or None, rights="local_only",
                    ))
            if scanned >= 5:
                yield ObservationPage(observations=tuple(observations), scanned=scanned, exhausted=False)
                observations.clear()
                scanned = 0
        if scanned:
            yield ObservationPage(observations=tuple(observations), scanned=scanned, exhausted=False)
        if failure:
            raise SourceFetchError(failure)
        # Both endpoints advertise up to 500 stories; a bounded slice cannot
        # be called a complete search, even if no matching item was found.
        if len(chosen) < cap:
            raise SourceFetchError("source_incomplete")
        if cap < limit:
            raise SourceFetchError("source_limit")


GDELT_COUNTRIES = {
    "RU": "russia", "US": "unitedstates", "GB": "unitedkingdom", "DE": "germany", "FR": "france",
    "CN": "china", "JP": "japan", "KR": "southkorea", "IN": "india", "CA": "canada", "AU": "australia",
    "BR": "brazil", "IL": "israel", "IT": "italy", "ES": "spain", "NL": "netherlands",
    "CH": "switzerland", "SG": "singapore", "AT": "austria", "BE": "belgium", "PL": "poland",
    "SE": "sweden", "FI": "finland", "DK": "denmark", "IE": "ireland", "PT": "portugal", "CZ": "czechrepublic",
}


_GDELT_CODES = {name: code for code, name in GDELT_COUNTRIES.items()}


class GdeltDocAdapter(_PublicAdapter):
    source_id = "gdelt"
    _url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean_query = _query(query)
        cap = max(0, min(int(limit), 250))
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        deadline = monotonic() + max(0.1, min(float(timeout_seconds), _MAX_HTTP_SECONDS))
        search = clean_query
        countries = [GDELT_COUNTRIES[code] for code in self._countries if code in GDELT_COUNTRIES][:8]
        if countries:
            # Фильтр стран владельца: GDELT называет страны без пробелов («unitedstates»).
            filters = " OR ".join(f"sourcecountry:{name}" for name in countries)
            search = f"{clean_query} ({filters})" if len(countries) > 1 else f"{clean_query} {filters}"
        data = self._json(self._url, params={
            "query": search, "mode": "artlist", "format": "json", "maxrecords": cap,
            "timespan": "3months", "sort": "datedesc",
        }, deadline=deadline, cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
            raise SourceFetchError("invalid_response")
        articles = data["articles"][:cap]
        observed_at = datetime.now(UTC)
        observations: list[ExternalObservation] = []
        for item in articles:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            title = _summary(item.get("title"), max_chars=250)
            url = _public_url(item.get("url"))
            # DOC ArtList calls `seendate` its publication date.  It is an
            # index timestamp, so later evidence checks must not infer that
            # the publisher itself supplied this date.
            published = _date(item.get("seendate"))
            if not title or not url or not published or published > as_of:
                continue
            country = item.get("sourcecountry")
            observations.append(ExternalObservation(
                source_id=self.source_id, item_id=_stable_id(url), kind="news_aggregate",
                title=title, url=url, published_at=published, observed_at=observed_at,
                summary=None, rights="local_only", date_basis="indexed",
                country=_GDELT_CODES.get(re.sub(r"[^a-z]", "", country.casefold()))
                if isinstance(country, str) else None,
            ))
        scanned = len(articles)
        yield ObservationPage(observations=tuple(observations), scanned=scanned, exhausted=scanned < cap)


def make_news_adapters() -> dict[str, object]:
    """Create independent adapter sessions, one per approved source."""
    adapters = (
        MitResearchNewsAdapter(), HorizonMagazineAdapter(), GitHubRepositoryAdapter(),
        HackerNewsAdapter(), GdeltDocAdapter(), HabrAdapter(),
    )
    return {adapter.source_id: adapter for adapter in adapters}


def news_adapters() -> dict[str, object]:
    """Compatibility alias for callers that use the shorter factory name."""
    return make_news_adapters()
