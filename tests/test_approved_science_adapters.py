from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from threading import Event

import httpx
import pytest

from app.pilot.approved_sources.adapters_science import (
    ArxivAdapter, BiorxivAdapter, EuropePmcAdapter, NistNewsAdapter, OpenReviewAdapter, ZenodoAdapter,
)
from app.pilot.approved_sources.collector import collect_approved_sources
from app.pilot.approved_sources.contracts import SourceFetchError, SourceSnapshot
from app.runtime.jobs import TaskCancelled


AS_OF = date(2026, 9, 24)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_arxiv_search_is_single_bounded_request_with_real_publication_date() -> None:
    calls: list[httpx.Request] = []
    atom = b'''<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>3</opensearch:totalResults>
      <entry><id>http://arxiv.org/abs/2609.12345v2</id><title>Quantum sensor</title>
        <summary>A new detector.</summary><published>2026-09-20T10:00:00Z</published></entry>
      <entry><id>http://arxiv.org/abs/2609.99999v1</id><title>Future detector</title>
        <published>2026-09-25T10:00:00Z</published></entry>
    </feed>'''

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=atom)

    adapter = ArxivAdapter(_client(handler))
    pages = list(adapter.iter_pages("quantum sensor", as_of=AS_OF, limit=2,
                                    timeout_seconds=3, cancel=Event()))
    assert len(calls) == 1
    assert "all:quantum" in calls[0].url.params["search_query"]
    assert "202609242359" in calls[0].url.params["search_query"]
    assert pages[0].scanned == 2
    assert pages[0].exhausted is False
    assert [item.item_id for item in pages[0].observations] == ["2609.12345"]
    assert pages[0].observations[0].published_at == date(2026, 9, 20)


def test_arxiv_primary_domains_count_first_publications_once_per_article() -> None:
    atom = b'''<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>6</opensearch:totalResults>
      <entry><id>https://arxiv.org/abs/2608.12345v2</id><title>Quantum learning</title>
        <published>2026-08-28T10:00:00Z</published><updated>2026-09-20T10:00:00Z</updated>
        <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
        <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/></entry>
      <entry><id>https://arxiv.org/abs/2608.12345v1</id><title>Quantum learning</title>
        <published>2026-08-28T10:00:00Z</published>
        <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/></entry>
      <entry><id>https://arxiv.org/abs/2609.12346v1</id><title>Quantum networks</title>
        <published>2026-09-03T10:00:00Z</published>
        <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/></entry>
      <entry><id>https://arxiv.org/abs/2609.12347v1</id><title>Quantum classification</title>
        <published>2026-09-08T10:00:00Z</published>
        <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/></entry>
      <entry><id>https://arxiv.org/abs/2609.12348v1</id><title>Quantum sensors</title>
        <published>2026-09-09T10:00:00Z</published>
        <category term="physics.ins-det" scheme="http://arxiv.org/schemas/atom"/></entry>
      <entry><id>https://arxiv.org/abs/2609.12349v1</id><title>Quantum devices</title>
        <published>2026-09-10T10:00:00Z</published>
        <arxiv:primary_category term="bad category" scheme="http://arxiv.org/schemas/atom"/></entry>
    </feed>'''
    adapter = ArxivAdapter(_client(lambda _: httpx.Response(200, content=atom)))
    snapshot = collect_approved_sources("quantum", as_of=AS_OF, cancel=Event(),
                                        adapters={"arxiv": adapter}, per_source_cap=10,
                                        max_observations=110)
    assert [item.item_id for item in snapshot.observations] == [
        "2608.12345", "2609.12346", "2609.12347", "2609.12348", "2609.12349",
    ]
    assert snapshot.coverage[0].duplicates == 1
    assert snapshot.observations[0].arxiv_primary_category == "cs.AI"
    assert snapshot.observations[3].arxiv_primary_category is None
    assert [(item.month, item.domain, item.primary_category, item.article_count)
            for item in snapshot.arxiv_domain_months] == [
        (date(2026, 8, 1), "cs", "cs.AI", 1),
        (date(2026, 9, 1), "cs", "cs.LG", 2),
    ]
    assert SourceSnapshot.model_validate(snapshot.model_dump(mode="json")).arxiv_domain_months == (
        snapshot.arxiv_domain_months
    )


def test_europe_pmc_uses_only_corroborated_exact_first_publication_dates() -> None:
    requests: list[httpx.Request] = []
    payload = {"hitCount": 20, "resultList": {"result": [
        {"source": "MED", "id": "123", "pmid": "123", "doi": "10.1234/article.1",
         "title": "Quantum medicine", "abstractText": "A clinical study.",
         "firstPublicationDate": "2026-09-20", "electronicPublicationDate": "2026-09-20"},
        {"source": "PMC", "id": "PMC123", "pmid": "123", "doi": "10.1234/article.1",
         "title": "Quantum medicine duplicate", "firstPublicationDate": "2026-09-20",
         "electronicPublicationDate": "2026-09-20"},
        {"source": "MED", "id": "124", "pmid": "124", "title": "Inexact year",
         "firstPublicationDate": "2026-01-01", "electronicPublicationDate": "2026"},
        {"source": "PPR", "id": "PPR125", "doi": "10.1101/2026.09.20.123456",
         "title": "Quantum medicine preprint", "firstPublicationDate": "2026-09-21",
         "electronicPublicationDate": "2026-09-21"},
        {"source": "MED", "id": "126", "pmid": "126", "title": "Mismatched dates",
         "firstPublicationDate": "2026-09-20", "electronicPublicationDate": "2026-09-21"},
    ]}}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    adapter = EuropePmcAdapter(_client(handler))
    snapshot = collect_approved_sources("quantum medicine", as_of=AS_OF, cancel=Event(),
                                        adapters={"europe_pmc": adapter}, per_source_cap=5,
                                        max_observations=60)
    assert len(requests) == 1
    assert requests[0].url.params["pageSize"] == "5"
    assert requests[0].url.params["resultType"] == "core"
    assert "FIRST_PDATE:[1900-01-01 TO 2026-09-24]" in requests[0].url.params["query"]
    assert [(item.item_id, item.kind, item.published_at) for item in snapshot.observations] == [
        ("10.1234/article.1", "journal_article", date(2026, 9, 20)),
        ("10.1101/2026.09.20.123456", "preprint", date(2026, 9, 21)),
    ]
    coverage = next(item for item in snapshot.coverage if item.source_id == "europe_pmc")
    assert (coverage.scanned, coverage.accepted, coverage.rejected, coverage.duplicates) == (5, 2, 2, 1)
    assert coverage.state == "partial" and coverage.reason_code == "source_limit"
    assert coverage.total_available == 20


def test_biorxiv_filters_metadata_and_counts_all_scanned_records() -> None:
    payload = {"messages": [{"count": "2"}], "collection": [
        {"doi": "10.1101/2026.09.23.123456", "title": "Quantum cell sensor",
         "abstract": "Cell detection", "date": "2026-09-23"},
        {"doi": "10.1101/2026.09.23.123457", "title": "Protein atlas",
         "abstract": "No signal", "date": "2026-09-23"},
    ]}
    client = _client(lambda _: httpx.Response(200, json=payload))
    page = next(BiorxivAdapter(client).iter_pages("quantum sensor", as_of=AS_OF,
                                                 limit=2, timeout_seconds=3, cancel=Event()))
    assert page.scanned == 2
    assert page.exhausted is False  # The API only exposes a recent sample.
    assert [item.item_id for item in page.observations] == ["10.1101/2026.09.23.123456"]


def test_biorxiv_empty_200_is_explicit_loss_of_coverage() -> None:
    client = _client(lambda _: httpx.Response(200, content=b""))
    with pytest.raises(SourceFetchError, match="empty_response"):
        list(BiorxivAdapter(client).iter_pages("quantum", as_of=AS_OF, limit=2,
                                              timeout_seconds=3, cancel=Event()))


def test_biorxiv_uses_official_recent_feed_when_api_body_is_empty() -> None:
    current_day = datetime.now(UTC).date()
    feed = (f'''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
        <item><title>New quantum biosensor</title>
          <description>Sensor for protein imaging.</description>
          <dc:date>{current_day.isoformat()}</dc:date>
          <dc:identifier>doi:10.64898/2026.09.17.752434</dc:identifier></item>
        <item><title>Unrelated sample</title>
          <dc:date>{current_day.isoformat()}</dc:date>
          <dc:identifier>doi:10.64898/2026.09.17.752435</dc:identifier></item>
        </rdf:RDF>''').encode()
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return (httpx.Response(200, content=feed) if request.url.host == "connect.biorxiv.org"
                else httpx.Response(200, content=b""))

    adapter = BiorxivAdapter(_client(handler))
    pages = adapter.iter_pages("quantum biosensor", as_of=current_day, limit=5,
                               timeout_seconds=3, cancel=Event())
    page = next(pages)
    assert page.scanned == 2 and not page.exhausted
    assert [item.title for item in page.observations] == ["New quantum biosensor"]
    with pytest.raises(SourceFetchError, match="feed_window_only"):
        next(pages)
    assert hosts == ["api.biorxiv.org", "connect.biorxiv.org"]


def test_openreview_uses_public_visibility_timestamp_and_forum_not_review_reply() -> None:
    public_ms = int(datetime(2026, 9, 21, tzinfo=UTC).timestamp() * 1000)
    note = {"id": "AbcDef12", "odate": public_ms,
            "content": {"title": {"value": "Quantum network"},
                        "abstract": {"value": "A new switch."}}}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"count": 2, "notes": [note]})

    page = next(OpenReviewAdapter(_client(handler)).iter_pages("quantum", as_of=AS_OF,
                                                               limit=1, timeout_seconds=3, cancel=Event()))
    assert requests[0].url.params["source"] == "forum"
    assert page.scanned == 1
    assert page.observations[0].published_at == date(2026, 9, 21)
    assert page.observations[0].url == "https://openreview.net/forum?id=AbcDef12"


def test_openreview_current_public_result_without_odate_uses_labelled_creation_date() -> None:
    current_day = datetime.now(UTC).date()
    created_ms = int(datetime.combine(current_day, datetime.min.time(), UTC).timestamp() * 1000)
    note = {"id": "AbcDef12", "odate": None, "tcdate": created_ms,
            "content": {"title": {"value": "Quantum network"},
                        "abstract": {"value": "A new switch."}}}
    client = _client(lambda _: httpx.Response(200, json={"count": 1, "notes": [note]}))
    adapter = OpenReviewAdapter(client)
    page = next(adapter.iter_pages("quantum", as_of=current_day, limit=1,
                                   timeout_seconds=3, cancel=Event()))
    assert len(page.observations) == 1
    assert page.observations[0].date_basis == "created"
    assert page.observations[0].published_at == current_day
    older = current_day - timedelta(days=1)
    historical_page = next(adapter.iter_pages("quantum", as_of=older, limit=1,
                                              timeout_seconds=3, cancel=Event()))
    assert historical_page.observations == ()


def test_zenodo_anonymous_pagination_respects_scanned_limit() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = int(request.url.params["page"])
        first = 1 if page == 1 else 26
        count = 25 if page == 1 else 1
        records = [{"id": i, "metadata": {"title": f"Quantum artifact {i}",
                    "publication_date": "2026-09-20", "description": "<p>Research data</p>"}}
                   for i in range(first, first + count)]
        return httpx.Response(200, json={"hits": {"total": 26, "hits": records}})

    pages = list(ZenodoAdapter(_client(handler)).iter_pages("quantum", as_of=AS_OF,
                                                            limit=50, timeout_seconds=3, cancel=Event()))
    assert len(requests) == 2
    assert [page.scanned for page in pages] == [25, 1]
    assert pages[-1].exhausted is True
    assert sum(len(page.observations) for page in pages) == 26
    assert pages[0].observations[0].summary == "Research data"


def test_nist_rss_is_a_partial_recent_window_when_below_limit() -> None:
    rss = b'''<?xml version="1.0"?><rss version="2.0"><channel><item>
      <title>NIST Develops Quantum Sensor</title>
      <link>https://www.nist.gov/news-events/news/2026/09/new-sensor</link>
      <pubDate>Fri, 18 Sep 2026 12:00:00 +0000</pubDate>
      <description>A new measurement.</description>
    </item><item>
      <title>NIST Announces Education Grant</title>
      <link>https://www.nist.gov/news-events/news/2026/09/education-grant</link>
      <pubDate>Fri, 18 Sep 2026 12:00:00 +0000</pubDate>
      <description>New funding for schools.</description>
    </item></channel></rss>'''
    adapter = NistNewsAdapter(_client(lambda _: httpx.Response(200, content=rss)))
    snapshot = collect_approved_sources("quantum", as_of=AS_OF, cancel=Event(),
                                        adapters={"nist_news": adapter}, per_source_cap=3,
                                        max_observations=30, max_workers=1)
    coverage = next(item for item in snapshot.coverage if item.source_id == "nist_news")
    assert coverage.scanned == 2
    assert coverage.accepted == 1
    assert coverage.state == "partial"
    assert coverage.reason_code == "feed_window_only"
    assert snapshot.observations[0].published_at == date(2026, 9, 18)


def test_cancelled_adapter_makes_no_http_request() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        list(ZenodoAdapter(_client(handler)).iter_pages("quantum", as_of=AS_OF,
                                                        limit=1, timeout_seconds=3, cancel=cancel))
    assert calls == 0


def test_redirect_and_oversize_response_are_safe_source_errors() -> None:
    redirect = ArxivAdapter(_client(lambda _: httpx.Response(302, headers={"location": "https://other.test"})))
    with pytest.raises(SourceFetchError, match="unexpected_redirect"):
        list(redirect.iter_pages("quantum", as_of=AS_OF, limit=1,
                                 timeout_seconds=3, cancel=Event()))
    oversized = NistNewsAdapter(_client(lambda _: httpx.Response(200, content=b"x" * 2_000_001)))
    with pytest.raises(SourceFetchError, match="response_too_large"):
        list(oversized.iter_pages("quantum", as_of=AS_OF, limit=1,
                                  timeout_seconds=3, cancel=Event()))
