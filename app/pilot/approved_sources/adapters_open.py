"""Открытые источники без ключей API: программа сама собирает их выдачу.

Каждый адаптер делает один-два ограниченных запроса к публичному интерфейсу
(JSON, RSS или Atom), не требующему регистрации, и сохраняет только заголовок,
ссылку, дату и короткий фрагмент описания. Отказ или лимит источника
отражается в его охвате, а не превращается в «ничего не найдено».

* Google News — новости по странам: у каждой выбранной страны своё издание и
  язык запроса (русскому изданию — русская формулировка).
* Semantic Scholar, DOAJ, HAL, OSTI, NASA NTRS, dblp, ChemRxiv,
  КиберЛенинка — научные статьи, отчёты и препринты.
* Stack Overflow и Hugging Face — сообщество разработчиков и модели.

Лента, которая не умеет искать по запросу, проверяет тему сама — тем же
строгим правилом, что и остальные (`app.topic_relevance.topic_match`).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from threading import Event
from time import monotonic
from typing import Any, Literal
from urllib.parse import quote

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from app.pilot.approved_sources.adapters_news import (
    _MAX_HTTP_SECONDS, _check_cancel, _date, _profile, _public_url, _PublicAdapter, _query, _stable_id,
    _summary,
)
from app.pilot.approved_sources.contracts import (
    ExternalObservation, ObservationKind, ObservationPage, SourceFetchError,
)
from app.topic_relevance import topic_match

_MAX_ITEMS = 100


def _mapping(value: object) -> dict[str, Any]:
    """Словарь из разобранного JSON или пустой словарь."""
    return value if isinstance(value, dict) else {}


def _items(value: object) -> list[Any]:
    """Список из разобранного JSON или пустой список."""
    return value if isinstance(value, list) else []


def _deadline(timeout_seconds: float) -> float:
    return monotonic() + max(0.1, min(float(timeout_seconds), _MAX_HTTP_SECONDS))


def _cap(limit: int, maximum: int = _MAX_ITEMS) -> int:
    return max(0, min(int(limit), maximum))


def _year_date(value: object, as_of: date) -> date | None:
    """1 января года публикации; год из будущего или вне разумных границ отбрасывается."""
    if isinstance(value, bool):
        return None
    try:
        year = int(str(value).strip()[:4]) if value is not None else None
    except ValueError:
        return None
    if year is None or not 1900 <= year <= as_of.year:
        return None
    return date(year, 1, 1)


def _exact_or_year(value: object, as_of: date) -> tuple[date | None, Literal["published", "year"]]:
    """Полная дата «ГГГГ-ММ-ДД» — дата публикации; иначе только год."""
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        exact = _date(value[:10])
        if exact is not None:
            return exact, "published"
    return _year_date(value, as_of), "year"


def _doi_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    doi = value.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    if not doi.startswith("10.") or len(doi) > 300 or any(char.isspace() for char in doi):
        return None
    return _public_url("https://doi.org/" + quote(doi, safe="/:;()._-"))


def _first_text(value: object) -> str | None:
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item.strip()), None)
    return value if isinstance(value, str) else None


class _SearchAdapter(_PublicAdapter):
    """Общая часть поисковых адаптеров: одна страница выдачи и строгая проверка темы."""

    kind: ObservationKind = "journal_article"
    country: str | None = None
    # Лента без полнотекстового поиска: тема проверяется здесь.
    check_topic = False

    def _observation(self, *, item_id: str, title: str | None, url: str | None, published: date | None,
                     as_of: date, summary: str | None = None, basis: str = "published",
                     country: str | None = None, observed_at: datetime) -> ExternalObservation | None:
        if not title or not url or published is None or published > as_of or not item_id:
            return None
        try:
            return ExternalObservation(
                source_id=self.source_id, item_id=item_id[:512], kind=self.kind, title=title[:1000], url=url,
                published_at=published, observed_at=observed_at, summary=(summary or "")[:300] or None,
                rights="local_only", date_basis=basis,  # type: ignore[arg-type]
                country=country if country is not None else self.country)
        except ValueError:
            # Одна испорченная запись источника не отменяет остальные.
            return None

    def _page(self, observations: list[ExternalObservation | None], *, scanned: int, requested: int,
              total: object, query: str) -> ObservationPage:
        profile = _profile(query, self._localized)
        kept = tuple(item for item in observations if item is not None and (
            not self.check_topic or topic_match(profile, item.title, item.summary or "")))
        available = total if isinstance(total, int) and not isinstance(total, bool) and 0 <= total <= 1_000_000_000 \
            else None
        # Выдача короче запрошенного — источник отдал всё, что нашёл по запросу.
        exhausted = scanned < requested or available is not None and available <= scanned
        return ObservationPage(observations=kept, scanned=scanned, exhausted=exhausted,
                               total_available=available)


class GoogleNewsAdapter(_SearchAdapter):
    """Новости Google News по изданиям выбранных стран (RSS поиска, без ключа)."""

    source_id = "google_news"
    kind = "news_aggregate"
    _url = "https://news.google.com/rss/search"
    # страна -> (hl, gl, ceid, язык формулировки)
    EDITIONS: dict[str, tuple[str, str, str, str]] = {
        "US": ("en-US", "US", "US:en", "en"), "GB": ("en-GB", "GB", "GB:en", "en"),
        "RU": ("ru", "RU", "RU:ru", "ru"), "DE": ("de", "DE", "DE:de", "de"),
        "FR": ("fr", "FR", "FR:fr", "fr"), "IN": ("en-IN", "IN", "IN:en", "en"),
        "CA": ("en-CA", "CA", "CA:en", "en"), "AU": ("en-AU", "AU", "AU:en", "en"),
        "JP": ("ja", "JP", "JP:ja", "ja"), "CN": ("zh-CN", "CN", "CN:zh-Hans", "zh"),
        "KR": ("ko", "KR", "KR:ko", "ko"), "BR": ("pt-BR", "BR", "BR:pt-419", "pt"),
        "IL": ("en-IL", "IL", "IL:en", "en"), "IT": ("it", "IT", "IT:it", "it"),
        "ES": ("es", "ES", "ES:es", "es"), "NL": ("nl", "NL", "NL:nl", "nl"),
        "CH": ("de-CH", "CH", "CH:de", "de"), "SG": ("en-SG", "SG", "SG:en", "en"),
    }
    DEFAULT = ("US", "RU")

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        english = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        editions = [country for country in (self._editions or self.DEFAULT) if country in self.EDITIONS][:4]
        if not editions:
            raise SourceFetchError("country_filter")
        deadline = _deadline(timeout_seconds)
        share, extra = divmod(cap, len(editions))
        failure: str | None = None
        collected: list[ExternalObservation] = []
        scanned = 0
        full_edition = False
        for position, country in enumerate(editions):
            _check_cancel(cancel)
            edition_cap = share + (position < extra)
            if not edition_cap:
                continue
            hl, gl, ceid, language = self.EDITIONS[country]
            search = _query(self._localized.get(language) or english)
            try:
                body = self._get(self._url, params={"q": search, "hl": hl, "gl": gl, "ceid": ceid},
                                 deadline=deadline, cancel=cancel)
            except SourceFetchError as error:
                failure = error.code
                if error.code == "timeout":
                    break
                continue
            try:
                root = ElementTree.fromstring(body)
            except (ElementTree.ParseError, DefusedXmlException):
                failure = "invalid_response"
                continue
            items = root.findall("./channel/item")[:edition_cap]
            full_edition = full_edition or len(items) >= edition_cap
            scanned += len(items)
            observed_at = datetime.now(UTC)
            for item in items:
                _check_cancel(cancel)
                title = _summary(item.findtext("title"), max_chars=300)
                publisher = _summary(item.findtext("source"), max_chars=120)
                # Заголовок Google News оканчивается « - Издатель»: издатель уходит в описание.
                if title and publisher and title.endswith(" - " + publisher):
                    title = title[:-len(" - " + publisher)].strip()
                url = _public_url(item.findtext("link"), allowed_hosts=frozenset({"news.google.com"}))
                description = _summary(item.findtext("description"), max_chars=300)
                summary = publisher if publisher else None
                if description and title and description.casefold() != title.casefold() \
                        and not description.casefold().startswith(title.casefold()):
                    summary = description
                observation = self._observation(
                    item_id=_stable_id(url or ""), title=title, url=url, published=_date(item.findtext("pubDate")),
                    as_of=as_of, summary=summary, country=country, observed_at=observed_at)
                if observation is not None:
                    collected.append(observation)
        if scanned or failure is None:
            # Поиск Google News не сообщает полного объёма: неполное издание — это вся его выдача.
            yield ObservationPage(observations=tuple(collected), scanned=scanned,
                                  exhausted=not full_edition, total_available=None)
        if failure is not None:
            raise SourceFetchError(failure)


class SemanticScholarAdapter(_SearchAdapter):
    source_id = "semantic_scholar"
    _url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={
            "query": clean, "limit": cap, "fields": "title,abstract,url,publicationDate,year,externalIds"},
            deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("data", []), list):
            raise SourceFetchError("invalid_response")
        items = (data.get("data") or [])[:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            paper = item.get("paperId")
            published, basis = _exact_or_year(item.get("publicationDate"), as_of)
            if published is None:
                published, basis = _year_date(item.get("year"), as_of), "year"
            external = _mapping(item.get("externalIds"))
            url = _public_url(item.get("url"), allowed_hosts=frozenset({"www.semanticscholar.org"})) \
                or _doi_url(external.get("DOI"))
            observations.append(self._observation(
                item_id=paper if isinstance(paper, str) else "", title=_summary(item.get("title"), max_chars=500),
                url=url, published=published, basis=basis, as_of=as_of,
                summary=_summary(item.get("abstract")), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=data.get("total"), query=clean)


class DoajAdapter(_SearchAdapter):
    """Статьи журналов открытого доступа; страна записи — страна журнала."""

    source_id = "doaj"
    _url = "https://doaj.org/api/search/articles/"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        words = " ".join(part for part in clean.replace(":", " ").replace('"', " ").split() if part.isalnum()
                         or part.replace("-", "").isalnum())
        search = f"({words})" if words else clean
        countries = [code for code in self._countries if len(code) == 2][:30]
        if countries:
            search += " AND bibjson.journal.country:(" + " OR ".join(countries) + ")"
        data = self._json(self._url + quote(search, safe=""), params={"page": 1, "pageSize": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SourceFetchError("invalid_response")
        items = data["results"][:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            bib = item.get("bibjson") if isinstance(item, dict) else None
            if not isinstance(bib, dict):
                continue
            identifiers = _items(bib.get("identifier"))
            doi = next((entry.get("id") for entry in identifiers
                        if isinstance(entry, dict) and str(entry.get("type", "")).casefold() == "doi"), None)
            links = _items(bib.get("link"))
            fulltext = next((entry.get("url") for entry in links if isinstance(entry, dict)), None)
            identifier = item.get("id") if isinstance(item.get("id"), str) else ""
            url = _doi_url(doi) or _public_url(fulltext) or (
                _public_url(f"https://doaj.org/article/{identifier}") if identifier.isalnum() else None)
            journal = _mapping(bib.get("journal"))
            country = journal.get("country")
            observations.append(self._observation(
                item_id=identifier or (doi if isinstance(doi, str) else ""),
                title=_summary(bib.get("title"), max_chars=500), url=url,
                published=_year_date(bib.get("year"), as_of), basis="year", as_of=as_of,
                summary=_summary(bib.get("abstract")), observed_at=observed_at,
                country=country.upper() if isinstance(country, str) and len(country) == 2 and country.isalpha()
                else None))
        yield self._page(observations, scanned=len(items), requested=cap, total=data.get("total"), query=clean)


class CyberLeninkaAdapter(_SearchAdapter):
    """Российские научные журналы (поиск КиберЛенинки, русская формулировка)."""

    source_id = "cyberleninka"
    language = "ru"
    country = "RU"
    _url = "https://cyberleninka.ru/api/search"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        search = _query(self._localized.get("ru") or query)
        cap = _cap(limit, 50)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._post_json(self._url, {"mode": "articles", "q": search, "size": cap, "from": 0},
                               deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
            raise SourceFetchError("invalid_response")
        items = data["articles"][:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if not isinstance(link, str) or not link.startswith("/article/") or len(link) > 400:
                continue
            observations.append(self._observation(
                item_id=link.rsplit("/", 1)[-1], title=_summary(item.get("name"), max_chars=500),
                url=_public_url("https://cyberleninka.ru" + link, allowed_hosts=frozenset({"cyberleninka.ru"})),
                published=_year_date(item.get("year"), as_of), basis="year", as_of=as_of,
                summary=_summary(item.get("annotation")), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=data.get("found"), query=search)


class HalAdapter(_SearchAdapter):
    """Открытый архив французских исследований HAL (Solr, без ключа)."""

    source_id = "hal"
    country = "FR"
    _url = "https://api.archives-ouvertes.fr/search/"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        words = " ".join(word for word in clean.replace(":", " ").split() if word.replace("-", "").isalnum())
        data = self._json(self._url, params={
            "q": words or clean, "wt": "json", "rows": cap,
            "fl": "docid,title_s,abstract_s,uri_s,producedDate_s,doiId_s"},
            deadline=_deadline(timeout_seconds), cancel=cancel)
        response = data.get("response") if isinstance(data, dict) else None
        if not isinstance(response, dict) or not isinstance(response.get("docs"), list):
            raise SourceFetchError("invalid_response")
        items = response["docs"][:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            published, basis = _exact_or_year(item.get("producedDate_s"), as_of)
            docid = item.get("docid")
            observations.append(self._observation(
                item_id=str(docid) if isinstance(docid, (int, str)) else "",
                title=_summary(_first_text(item.get("title_s")), max_chars=500),
                url=_public_url(item.get("uri_s")) or _doi_url(item.get("doiId_s")),
                published=published, basis=basis, as_of=as_of,
                summary=_summary(_first_text(item.get("abstract_s"))), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=response.get("numFound"), query=clean)


class OstiAdapter(_SearchAdapter):
    """Отчёты и статьи Управления научной информации Минэнерго США."""

    source_id = "osti"
    country = "US"
    kind = "research_artifact"
    _url = "https://www.osti.gov/api/v1/records"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"q": clean, "rows": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, list):
            raise SourceFetchError("invalid_response")
        items = data[:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            osti_id = item.get("osti_id")
            identifier = str(osti_id) if isinstance(osti_id, (int, str)) and str(osti_id).isdecimal() else ""
            observations.append(self._observation(
                item_id=identifier, title=_summary(item.get("title"), max_chars=500),
                url=_public_url(f"https://www.osti.gov/biblio/{identifier}") if identifier else None,
                published=_date(item.get("publication_date")), as_of=as_of,
                summary=_summary(item.get("description")), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=None, query=clean)


class NasaNtrsAdapter(_SearchAdapter):
    """Технические отчёты и статьи NASA (NTRS)."""

    source_id = "nasa_ntrs"
    country = "US"
    kind = "research_artifact"
    _url = "https://ntrs.nasa.gov/api/citations/search"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"q": clean, "page.size": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SourceFetchError("invalid_response")
        items = data["results"][:cap]
        stats = _mapping(data.get("stats"))
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            identifier = str(item.get("id", ""))
            publications = _items(item.get("publications"))
            dated = next((entry.get("publicationDate") for entry in publications
                          if isinstance(entry, dict) and entry.get("publicationDate")), None)
            published, basis = (_date(dated), "published") if dated else (_date(item.get("submittedDate")),
                                                                          "indexed")
            observations.append(self._observation(
                item_id=identifier if identifier.isdecimal() else "", title=_summary(item.get("title"), max_chars=500),
                url=_public_url(f"https://ntrs.nasa.gov/citations/{identifier}") if identifier.isdecimal() else None,
                published=published, basis=basis, as_of=as_of,
                summary=_summary(item.get("abstract")), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=stats.get("total"), query=clean)


class DblpAdapter(_SearchAdapter):
    """Библиография информатики dblp: известен только год публикации."""

    source_id = "dblp"
    _url = "https://dblp.org/search/publ/api"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"q": clean, "format": "json", "h": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        result = data.get("result") if isinstance(data, dict) else None
        hits = result.get("hits") if isinstance(result, dict) else None
        if not isinstance(hits, dict):
            raise SourceFetchError("invalid_response")
        raw = hits.get("hit", [])
        if not isinstance(raw, list):
            raise SourceFetchError("invalid_response")
        items = raw[:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            info = item.get("info") if isinstance(item, dict) else None
            if not isinstance(info, dict):
                continue
            title = _summary(info.get("title"), max_chars=500)
            if title and title.endswith("."):
                title = title[:-1]
            electronic = info.get("ee")
            observations.append(self._observation(
                item_id=str(item.get("@id", "")), title=title,
                url=_public_url(_first_text(electronic)) or _public_url(info.get("url")),
                published=_year_date(info.get("year"), as_of), basis="year", as_of=as_of,
                summary=_summary(info.get("venue") if isinstance(info.get("venue"), str) else None),
                observed_at=observed_at))
        total = hits.get("@total")
        yield self._page(observations, scanned=len(items), requested=cap,
                         total=int(total) if isinstance(total, str) and total.isdecimal() else None, query=clean)


class StackExchangeAdapter(_SearchAdapter):
    """Вопросы разработчиков на Stack Overflow (API Stack Exchange без ключа)."""

    source_id = "stack_exchange"
    kind = "community"
    _url = "https://api.stackexchange.com/2.3/search/advanced"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"order": "desc", "sort": "relevance", "q": clean,
                                             "site": "stackoverflow", "pagesize": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise SourceFetchError("invalid_response")
        items = data["items"][:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            tags = _items(item.get("tags"))
            identifier = item.get("question_id")
            observations.append(self._observation(
                item_id=str(identifier) if isinstance(identifier, int) else "",
                title=_summary(item.get("title"), max_chars=300),
                url=_public_url(item.get("link"), allowed_hosts=frozenset({"stackoverflow.com"})),
                published=_date(item.get("creation_date")), as_of=as_of,
                summary=("Теги: " + ", ".join(str(tag) for tag in tags[:8])) if tags else None,
                observed_at=observed_at))
        page = self._page(observations, scanned=len(items), requested=cap, total=None, query=clean)
        yield ObservationPage(observations=page.observations, scanned=page.scanned,
                              exhausted=data.get("has_more") is False)


class HuggingFaceAdapter(_SearchAdapter):
    """Модели машинного обучения Hugging Face Hub, созданные под тему запроса."""

    source_id = "huggingface"
    kind = "repository"
    check_topic = True
    _url = "https://huggingface.co/api/models"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"search": clean, "limit": cap, "sort": "likes", "direction": -1},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, list):
            raise SourceFetchError("invalid_response")
        items = data[:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            if not isinstance(item, dict):
                continue
            model = item.get("id") or item.get("modelId")
            if not isinstance(model, str) or len(model) > 200 or model.count("/") > 1:
                continue
            details = [str(item["pipeline_tag"])] if isinstance(item.get("pipeline_tag"), str) else []
            for name, label in (("likes", "отметок"), ("downloads", "загрузок")):
                if isinstance(item.get(name), int):
                    details.append(f"{item[name]} {label}")
            observations.append(self._observation(
                item_id=model, title=model.replace("-", " ").replace("_", " "),
                url=_public_url(f"https://huggingface.co/{quote(model, safe='/')}",
                                allowed_hosts=frozenset({"huggingface.co"})),
                published=_date(item.get("createdAt")), as_of=as_of,
                summary="; ".join(details) or None, observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=None, query=clean)


class ChemRxivAdapter(_SearchAdapter):
    """Препринты химии ChemRxiv (открытый API Cambridge Open Engage)."""

    source_id = "chemrxiv"
    kind = "preprint"
    _url = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit, 50)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"term": clean, "limit": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("itemHits"), list):
            raise SourceFetchError("invalid_response")
        items = data["itemHits"][:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for hit in items:
            _check_cancel(cancel)
            item: Any = hit.get("item") if isinstance(hit, dict) else None
            if not isinstance(item, dict):
                continue
            identifier = str(item.get("id") or "")
            url = _doi_url(item.get("doi")) or (_public_url(
                f"https://chemrxiv.org/engage/chemrxiv/article-details/{identifier}") if identifier.isalnum() else None)
            observations.append(self._observation(
                item_id=identifier, title=_summary(item.get("title"), max_chars=500), url=url,
                published=_date(item.get("publishedDate")), as_of=as_of,
                summary=_summary(item.get("abstract")), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=data.get("totalCount"), query=clean)


def _listed(value: object) -> list[Any]:
    """OpenAIRE отдаёт один элемент словарём, несколько — списком."""
    return value if isinstance(value, list) else [value] if isinstance(value, dict) else []


def _dollar(value: object) -> str | None:
    """Текст узла OpenAIRE вида {"$": "..."}; у списка — первого узла."""
    for item in _listed(value) if not isinstance(value, str) else [value]:
        text = item.get("$") if isinstance(item, dict) else item
        if isinstance(text, (str, int)) and not isinstance(text, bool) and str(text).strip():
            return str(text)
    return None


class OpenAireAdapter(_SearchAdapter):
    """Европейский граф исследований OpenAIRE: статьи тысяч журналов и репозиториев.

    Выдача по релевантности, ограничена пятью последними годами и датой
    анализа. Слова запроса OpenAIRE ищет и в полном описании записи, поэтому
    тему по заголовку и аннотации проверяет строгое правило.
    """

    source_id = "openaire"
    kind = "journal_article"
    check_topic = True
    _url = "https://api.openaire.eu/search/publications"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={
            "keywords": clean, "format": "json", "size": cap, "page": 1,
            "fromDateAccepted": date(as_of.year - 5, 1, 1).isoformat(), "toDateAccepted": as_of.isoformat()},
            # Запись OpenAIRE подробная (авторы, организации, связи): 100 записей — до 10 МБ.
            deadline=_deadline(timeout_seconds), cancel=cancel, max_bytes=12_000_000)
        response = _mapping(_mapping(data).get("response"))
        header = _mapping(response.get("header"))
        if not header:
            raise SourceFetchError("invalid_response")
        # Без совпадений OpenAIRE отдаёт "results": null.
        items = _listed(_mapping(response.get("results")).get("result"))[:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            result = _mapping(_mapping(_mapping(_mapping(item).get("metadata")).get("oaf:entity")).get("oaf:result"))
            titles = _listed(result.get("title"))
            main = next((entry for entry in titles if entry.get("@classid") == "main title"), None)
            doi = next((_dollar(entry) for entry in _listed(result.get("pid"))
                        if entry.get("@classid") == "doi"), None)
            link = None
            for instance in _listed(_mapping(result.get("children")).get("instance")):
                link = link or next((_public_url(_dollar(resource.get("url")))
                                     for resource in _listed(instance.get("webresource"))), None)
            published, basis = _exact_or_year(_dollar(result.get("dateofacceptance")), as_of)
            identifier = _dollar(_mapping(_mapping(item).get("header")).get("dri:objIdentifier"))
            observations.append(self._observation(
                item_id=identifier or doi or "", title=_summary(_dollar(main or titles), max_chars=500),
                url=_doi_url(doi) or link, published=published, basis=basis, as_of=as_of,
                summary=_summary(_dollar(result.get("description"))), observed_at=observed_at))
        total = _dollar(header.get("total"))
        yield self._page(observations, scanned=len(items), requested=cap,
                         total=int(total) if total and total.isdecimal() else None, query=clean)


class JStageAdapter(_SearchAdapter):
    """Японские научные журналы J-STAGE (Atom-выдача поиска, без ключа)."""

    source_id = "jstage"
    kind = "journal_article"
    country = "JP"
    # Поиск J-STAGE находит и отдельные слова запроса: тема проверяется здесь.
    check_topic = True
    _url = "https://api.jstage.jst.go.jp/searchapi/do"
    _atom = "{http://www.w3.org/2005/Atom}"
    _prism = "{http://prismstandard.org/namespaces/basic/2.0/}"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        body = self._get(self._url, params={"service": 3, "text": clean, "count": cap,
                                            "pubyearfrom": as_of.year - 5, "pubyearto": as_of.year},
                         deadline=_deadline(timeout_seconds), cancel=cancel)
        try:
            root = ElementTree.fromstring(body)
        except (ElementTree.ParseError, DefusedXmlException):
            raise SourceFetchError("invalid_response") from None
        atom = self._atom
        status = (root.findtext(f"{atom}result/{atom}status") or "").strip()
        if status.startswith("ERR"):
            raise SourceFetchError("source_error")
        items = root.findall(f"{atom}entry")[:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for entry in items:
            _check_cancel(cancel)
            doi = entry.findtext(f"{self._prism}doi")
            link = entry.findtext(f"{atom}article_link/{atom}en")
            observations.append(self._observation(
                item_id=(doi or link or "").strip(),
                title=_summary(entry.findtext(f"{atom}article_title/{atom}en") or entry.findtext(f"{atom}title"),
                               max_chars=500),
                url=_doi_url(doi) or _public_url((link or "").strip(), allowed_hosts=frozenset({"www.jstage.jst.go.jp"})),
                published=_year_date(entry.findtext(f"{atom}pubyear"), as_of), basis="year", as_of=as_of,
                summary=_summary(entry.findtext(f"{atom}material_title/{atom}en"), max_chars=200),
                observed_at=observed_at))
        total = (root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") or "").strip()
        yield self._page(observations, scanned=len(items), requested=cap,
                         total=int(total) if total.isdecimal() else None, query=clean)


class NpmAdapter(_SearchAdapter):
    """Пакеты npm: что разработчики публикуют и обновляют по теме (поиск реестра без ключа)."""

    source_id = "npm"
    kind = "repository"
    # Поиск реестра нечёткий и находит пакеты по одному слову: тема проверяется здесь.
    check_topic = True
    _url = "https://registry.npmjs.org/-/v1/search"

    def iter_pages(self, query: str, *, as_of: date, limit: int, timeout_seconds: float,
                   cancel: Event) -> Iterator[ObservationPage]:
        clean = _query(query)
        cap = _cap(limit)
        if not cap:
            yield ObservationPage(observations=(), scanned=0, exhausted=False)
            return
        data = self._json(self._url, params={"text": clean, "size": cap},
                          deadline=_deadline(timeout_seconds), cancel=cancel)
        if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
            raise SourceFetchError("invalid_response")
        items = data["objects"][:cap]
        observed_at = datetime.now(UTC)
        observations = []
        for item in items:
            _check_cancel(cancel)
            package = _mapping(_mapping(item).get("package"))
            name = package.get("name")
            if not isinstance(name, str) or not 1 <= len(name) <= 214 or any(char.isspace() for char in name):
                continue
            description = _summary(package.get("description"), max_chars=300)
            details = ["npm " + name]
            monthly = _mapping(_mapping(item).get("downloads")).get("monthly")
            if isinstance(monthly, int) and not isinstance(monthly, bool):
                details.append(f"{monthly} загрузок в месяц")
            keywords = [str(word) for word in _items(package.get("keywords"))[:8] if isinstance(word, str)]
            if keywords:
                details.append("ключевые слова: " + ", ".join(keywords))
            observations.append(self._observation(
                item_id=name, title=f"{name}: {description}" if description else name,
                url=_public_url(f"https://www.npmjs.com/package/{quote(name, safe='@/')}",
                                allowed_hosts=frozenset({"www.npmjs.com"})),
                # Дата последней публикации пакета: признак того, что его развивают сейчас.
                published=_date(package.get("date")), as_of=as_of,
                summary="; ".join(details), observed_at=observed_at))
        yield self._page(observations, scanned=len(items), requested=cap, total=data.get("total"), query=clean)


def make_open_adapters() -> dict[str, object]:
    adapters = (
        GoogleNewsAdapter(), SemanticScholarAdapter(), DoajAdapter(), CyberLeninkaAdapter(), HalAdapter(),
        OstiAdapter(), NasaNtrsAdapter(), DblpAdapter(), StackExchangeAdapter(), HuggingFaceAdapter(),
        ChemRxivAdapter(), OpenAireAdapter(), JStageAdapter(), NpmAdapter(),
    )
    return {adapter.source_id: adapter for adapter in adapters}
