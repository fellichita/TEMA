"""Contract tests for public news and community adapters."""

from __future__ import annotations

from datetime import UTC, date, datetime
from threading import Event

import httpx
import pytest

from app.pilot.approved_sources import collect_approved_sources
from app.pilot.approved_sources.adapters_news import (
    GdeltDocAdapter, GitHubRepositoryAdapter, HabrAdapter, HackerNewsAdapter, HorizonMagazineAdapter,
    MitResearchNewsAdapter,
)
from app.pilot.approved_sources.contracts import SourceFetchError
from app.runtime.jobs import TaskCancelled


_AS_OF = date(2026, 9, 24)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _pages(adapter, query: str, limit: int = 10):
    return list(adapter.iter_pages(query, as_of=_AS_OF, limit=limit,
                                   timeout_seconds=5, cancel=Event()))


def test_mit_feed_scans_bounded_items_and_filters_locally():
    feed = b"""<rss><channel>
        <item><title>Quantum sensor works</title><link>https://news.mit.edu/2026/quantum-sensor</link>
          <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate><description>Prototype in a laboratory</description></item>
        <item><title>New campus building</title><link>https://news.mit.edu/2026/campus</link>
          <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate><description>Construction</description></item>
        <item><title>Quantum sensor tomorrow</title><link>https://news.mit.edu/2026/future</link>
          <pubDate>Fri, 25 Sep 2026 12:00:00 GMT</pubDate></item>
        </channel></rss>"""
    adapter = MitResearchNewsAdapter(_client(lambda request: httpx.Response(200, content=feed)))
    page, = _pages(adapter, "quantum sensor")
    assert page.scanned == 3 and page.exhausted is True
    assert len(page.observations) == 1
    item, = page.observations
    assert item.source_id == "mit_research_news" and item.kind == "institution_news"
    assert item.published_at == date(2026, 9, 23)
    assert item.rights == "local_only"


def test_horizon_feed_keeps_article_excerpt_local_and_reports_truncation():
    feed = b"""<rss><channel>
        <item><title>Robotic battery</title><link>https://projects.research-and-innovation.ec.europa.eu/en/horizon-magazine/robotic-battery</link>
          <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate></item>
        <item><title>Another technology</title><link>https://projects.research-and-innovation.ec.europa.eu/en/horizon-magazine/other</link>
          <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate></item>
        </channel></rss>"""
    adapter = HorizonMagazineAdapter(_client(lambda request: httpx.Response(200, content=feed)))
    page, = _pages(adapter, "technology", limit=1)
    assert page.scanned == 1 and page.exhausted is False
    assert page.observations[0].rights == "local_only"
    assert page.observations[0].license_ref is None


def test_habr_feed_is_recent_community_evidence_with_partial_coverage():
    feed = b"""<rss><channel>
        <item><title>Quantum sensor in Russia</title>
          <link>https://habr.com/ru/articles/123456/</link>
          <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate>
          <description>Engineering discussion, not a paper</description></item>
        <item><title>Unrelated story</title>
          <link>https://habr.com/ru/articles/123457/</link>
          <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate></item>
        </channel></rss>"""
    adapter = HabrAdapter(_client(lambda request: httpx.Response(200, content=feed)))
    pages = []
    with pytest.raises(SourceFetchError, match="recent_feed_only"):
        for page in adapter.iter_pages("quantum sensor", as_of=_AS_OF, limit=10,
                                       timeout_seconds=5, cancel=Event()):
            pages.append(page)
    assert len(pages) == 1
    assert pages[0].scanned == 2 and not pages[0].exhausted
    assert pages[0].total_available is None
    assert len(pages[0].observations) == 1
    assert pages[0].observations[0].kind == "community"
    assert pages[0].observations[0].rights == "local_only"
    snapshot = collect_approved_sources("quantum sensor", as_of=_AS_OF, cancel=Event(),
                                        adapters={"habr": HabrAdapter(_client(
                                            lambda request: httpx.Response(200, content=feed)))})
    coverage = next(item for item in snapshot.coverage if item.source_id == "habr")
    assert coverage.state == "partial" and coverage.reason_code == "recent_feed_only"


def test_github_uses_creation_date_and_never_claims_repository_license():
    payload = {"total_count": 2, "items": [
        {"id": 17, "full_name": "team/quantum-sensor", "html_url": "https://github.com/team/quantum-sensor",
         "created_at": "2026-09-20T00:00:00Z", "description": "A detector"},
        {"id": 18, "full_name": "team/future", "html_url": "https://github.com/team/future",
         "created_at": "2026-09-25T00:00:00Z", "description": "Not published yet"},
    ]}
    def handler(request):
        assert request.url.path == "/search/repositories"
        assert request.url.params["q"] == "quantum sensor"
        assert request.url.params["per_page"] == "10"
        return httpx.Response(200, json=payload)
    adapter = GitHubRepositoryAdapter(_client(handler))
    page, = _pages(adapter, "quantum sensor")
    assert page.scanned == 2 and page.total_available == 2 and page.exhausted
    assert len(page.observations) == 1
    item, = page.observations
    assert item.item_id == "17" and item.published_at == date(2026, 9, 20)
    assert item.rights == "local_only" and item.license_ref is None


def test_hacker_news_searches_algolia_and_keeps_only_topic_stories():
    def handler(request):
        assert request.url.host == "hn.algolia.com"
        assert request.url.params["query"] == "quantum sensor" and request.url.params["tags"] == "story"
        moment = int(datetime(2026, 9, 23, tzinfo=UTC).timestamp())
        return httpx.Response(200, json={"nbHits": 3, "hits": [
            {"objectID": "201", "title": "Quantum sensor startup raises seed", "created_at_i": moment},
            {"objectID": "202", "title": "State of the sensor market", "created_at_i": moment},
            {"objectID": "203", "title": "Quantum sensor from the future",
             "created_at_i": int(datetime(2026, 9, 30, tzinfo=UTC).timestamp())},
        ]})
    adapter = HackerNewsAdapter(_client(handler))
    page, = _pages(adapter, "quantum sensor", limit=10)
    assert page.scanned == 3 and page.exhausted and page.total_available == 3
    assert [item.url for item in page.observations] == ["https://news.ycombinator.com/item?id=201"]


def test_hacker_news_samples_recent_and_top_with_bounded_calls():
    calls: list[str] = []
    def handler(request):
        if request.url.host == "hn.algolia.com":
            return httpx.Response(503)  # Поиск недоступен: запасной путь — свежая выборка.
        path = request.url.path
        calls.append(path)
        if path.endswith("newstories.json"):
            return httpx.Response(200, json=[101, 102, 103])
        if path.endswith("topstories.json"):
            return httpx.Response(200, json=[104])
        item_id = int(path.rsplit("/", 1)[-1].split(".")[0])
        title = "Quantum sensor" if item_id == 101 else "Unrelated story"
        return httpx.Response(200, json={"id": item_id, "type": "story", "title": title,
                                          "time": int(datetime(2026, 9, 23, tzinfo=UTC).timestamp())})
    adapter = HackerNewsAdapter(_client(handler))
    page, = _pages(adapter, "quantum sensor", limit=4)
    assert page.scanned == 4 and not page.exhausted
    assert len(calls) == 6
    assert len(page.observations) == 1
    assert page.observations[0].url == "https://news.ycombinator.com/item?id=101"


def test_hacker_news_stops_at_twenty_items_and_preserves_partial_pages():
    calls = 0
    def handler(request):
        nonlocal calls
        if request.url.host == "hn.algolia.com":
            return httpx.Response(503)
        calls += 1
        if request.url.path.endswith("newstories.json"):
            return httpx.Response(200, json=list(range(101, 601)))
        if request.url.path.endswith("topstories.json"):
            return httpx.Response(200, json=list(range(601, 1101)))
        return httpx.Response(200, json={"type": "story", "title": "Quantum sensor",
                                          "time": int(datetime(2026, 9, 23, tzinfo=UTC).timestamp())})
    adapter = HackerNewsAdapter(_client(handler))
    pages = []
    with pytest.raises(SourceFetchError, match="source_limit"):
        for page in adapter.iter_pages("quantum sensor", as_of=_AS_OF, limit=50,
                                       timeout_seconds=5, cancel=Event()):
            pages.append(page)
    assert len(pages) == 4 and sum(page.scanned for page in pages) == 20
    assert sum(len(page.observations) for page in pages) == 20
    assert calls == 22


def test_gdelt_uses_indexed_date_and_removes_private_query_keys():
    payload = {"articles": [
        {"url": "https://publisher.example/story?id=42&api_key=hidden", "title": "Quantum sensor grows",
         "seendate": "20260923T120000Z"},
        {"url": "http://localhost/private", "title": "Private", "seendate": "20260923T120000Z"},
    ]}
    def handler(request):
        assert request.url.params["mode"] == "artlist"
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json=payload)
    adapter = GdeltDocAdapter(_client(handler))
    page, = _pages(adapter, "quantum sensor")
    assert page.scanned == 2 and page.exhausted
    assert len(page.observations) == 1
    item, = page.observations
    assert item.kind == "news_aggregate" and item.published_at == date(2026, 9, 23)
    assert item.date_basis == "indexed"
    assert item.url == "https://publisher.example/story?id=42"


def test_gdelt_skips_malformed_links_without_losing_valid_articles():
    payload = {"articles": [
        {"url": "https://@publisher.example/story", "title": "Invalid link",
         "seendate": "20260923T120000Z"},
        {"url": "https://publisher.example:999999/story", "title": "Invalid port",
         "seendate": "20260923T120000Z"},
        {"url": "https://publisher.example/story?id=42&key=private", "title": "Quantum sensor",
         "seendate": "20260923T120000Z"},
    ]}
    adapter = GdeltDocAdapter(_client(lambda _: httpx.Response(200, json=payload)))
    page, = _pages(adapter, "quantum sensor")
    assert page.scanned == 3
    assert [item.url for item in page.observations] == ["https://publisher.example/story?id=42"]


def test_long_generated_query_is_safely_shortened_at_word_boundary():
    observed = []
    def handler(request):
        observed.append(request.url.params["q"])
        return httpx.Response(200, json={"total_count": 0, "items": []})
    adapter = GitHubRepositoryAdapter(_client(handler))
    query = "quantum sensor " + "superconducting technology " * 15
    _pages(adapter, query)
    assert observed and len(observed[0]) <= 200
    assert observed[0].startswith("quantum sensor")
    assert observed[0].split()[-1] in {"superconducting", "technology"}


def test_transport_failures_are_safe_and_cancel_is_respected():
    adapter = MitResearchNewsAdapter(_client(lambda request: httpx.Response(429)))
    with pytest.raises(SourceFetchError, match="rate_limited"):
        _pages(adapter, "quantum sensor")
    cancelled = Event()
    cancelled.set()
    with pytest.raises(TaskCancelled):
        list(adapter.iter_pages("quantum sensor", as_of=_AS_OF, limit=1,
                                timeout_seconds=5, cancel=cancelled))
