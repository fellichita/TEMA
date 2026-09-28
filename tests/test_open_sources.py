"""Бесключевые источники, фильтр стран владельца и строгая проверка темы лент."""

from __future__ import annotations

from datetime import UTC, date, datetime
import json
from threading import Event

import httpx
import pytest

from app.pilot.approved_sources import collect_approved_sources
from app.pilot.approved_sources.adapters_open import (
    ChemRxivAdapter, CyberLeninkaAdapter, DblpAdapter, DoajAdapter, GoogleNewsAdapter, HalAdapter,
    HuggingFaceAdapter, JStageAdapter, NasaNtrsAdapter, NpmAdapter, OpenAireAdapter, OstiAdapter,
    SemanticScholarAdapter, StackExchangeAdapter,
)
from app.pilot.approved_sources.catalog import (
    SOURCE_INFO, SourcePolicy, catalogue, load_policy, save_policy,
)
from app.pilot.approved_sources.collector import _caps
from app.pilot.approved_sources.contracts import (
    LEGACY_CATALOGUES, SOURCE_IDS, ExternalObservation, SourceFetchError, SourceSnapshot,
)
from app.pilot.approved_sources.adapters_news import GdeltDocAdapter, HabrAdapter

_AS_OF = date(2026, 9, 24)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _pages(adapter, query: str = "solid-state battery", limit: int = 10):
    return list(adapter.iter_pages(query, as_of=_AS_OF, limit=limit, timeout_seconds=5, cancel=Event()))


RSS = """<rss><channel>
<item><title>Solid-state battery plant opens - Example Times</title>
  <link>https://news.google.com/rss/articles/abc?oc=5</link>
  <pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate><source url="https://example.com">Example Times</source>
  <description>&lt;a href="x"&gt;Solid-state battery plant opens&lt;/a&gt; Example Times</description></item>
<item><title>Future story - Example</title><link>https://news.google.com/rss/articles/def</link>
  <pubDate>Wed, 30 Sep 2026 12:00:00 GMT</pubDate><source url="https://example.com">Example</source></item>
</channel></rss>"""


def test_google_news_asks_each_country_edition_in_its_language():
    seen = []

    def handler(request):
        seen.append((request.url.params["gl"], request.url.params["q"]))
        return httpx.Response(200, text=RSS)

    adapter = GoogleNewsAdapter(_client(handler))
    adapter.configure(editions=("RU", "DE"), localized={"ru": "твердотельные аккумуляторы"})
    page, = _pages(adapter, limit=10)
    assert seen == [("RU", "твердотельные аккумуляторы"), ("DE", "solid-state battery")]
    assert page.scanned == 4 and page.exhausted
    assert [(item.country, item.title) for item in page.observations] == [
        ("RU", "Solid-state battery plant opens"), ("DE", "Solid-state battery plant opens")]
    assert all(item.summary == "Example Times" and item.kind == "news_aggregate" for item in page.observations)


def test_google_news_reports_a_failed_edition_but_keeps_the_others():
    def handler(request):
        return httpx.Response(503) if request.url.params["gl"] == "US" else httpx.Response(200, text=RSS)

    adapter = GoogleNewsAdapter(_client(handler))
    adapter.configure(editions=("US", "RU"))
    pages = []
    with pytest.raises(SourceFetchError, match="source_http_error"):
        for page in adapter.iter_pages("solid-state battery", as_of=_AS_OF, limit=10,
                                       timeout_seconds=5, cancel=Event()):
            pages.append(page)
    assert len(pages) == 1 and len(pages[0].observations) == 1


def test_semantic_scholar_keeps_exact_dates_and_falls_back_to_year():
    payload = {"total": 2, "data": [
        {"paperId": "p1", "title": "Sulfide solid electrolytes", "abstract": "All-solid-state batteries",
         "url": "https://www.semanticscholar.org/paper/p1", "publicationDate": "2026-05-04", "year": 2026},
        {"paperId": "p2", "title": "Solid-state battery review", "url": "https://www.semanticscholar.org/paper/p2",
         "publicationDate": None, "year": 2025, "externalIds": {"DOI": "10.1000/xyz"}},
    ]}
    adapter = SemanticScholarAdapter(_client(lambda request: httpx.Response(200, json=payload)))
    page, = _pages(adapter)
    first, second = page.observations
    assert (first.published_at, first.date_basis) == (date(2026, 5, 4), "published")
    assert (second.published_at, second.date_basis) == (date(2025, 1, 1), "year")
    assert page.exhausted and page.total_available == 2


def test_doaj_filters_by_journal_country_and_records_it():
    requested = []

    def handler(request):
        requested.append(request.url.path)
        return httpx.Response(200, json={"total": 1, "results": [{"id": "abc123", "bibjson": {
            "title": "Solid-state battery electrolytes", "abstract": "Text", "year": "2026",
            "identifier": [{"type": "doi", "id": "10.1234/abc"}], "journal": {"country": "ru"}}}]})

    adapter = DoajAdapter(_client(handler))
    adapter.configure(countries=("RU", "DE"))
    page, = _pages(adapter)
    assert "bibjson.journal.country:(RU OR DE)" in requested[0]
    item, = page.observations
    assert item.country == "RU" and item.url == "https://doi.org/10.1234/abc" and item.date_basis == "year"


def test_cyberleninka_posts_the_russian_query():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"found": 1, "articles": [
            {"name": "<b>Твердотельные</b> аккумуляторы", "annotation": "Обзор", "link": "/article/n/tverdotelnye",
             "year": 2024}]})

    adapter = CyberLeninkaAdapter(_client(handler))
    adapter.configure(localized={"ru": "твердотельные аккумуляторы"})
    page, = _pages(adapter)
    assert bodies == [{"mode": "articles", "q": "твердотельные аккумуляторы", "size": 10, "from": 0}]
    item, = page.observations
    assert item.title == "Твердотельные аккумуляторы" and item.country == "RU"
    assert item.url == "https://cyberleninka.ru/article/n/tverdotelnye"


@pytest.mark.parametrize(("adapter_type", "payload", "url"), [
    (HalAdapter, {"response": {"numFound": 1, "docs": [{"docid": 7, "title_s": ["Solid-state battery"],
                                                         "uri_s": "https://hal.science/hal-07",
                                                         "producedDate_s": "2026-02"}]}},
     "https://hal.science/hal-07"),
    (OstiAdapter, [{"osti_id": "123", "title": "Solid-state battery report",
                    "publication_date": "2026-03-01T00:00:00Z"}], "https://www.osti.gov/biblio/123"),
    (NasaNtrsAdapter, {"stats": {"total": 1}, "results": [
        {"id": 20260001, "title": "Solid-state battery for aircraft",
         "publications": [{"publicationDate": "2026-01-10T00:00:00.0000000+00:00"}]}]},
     "https://ntrs.nasa.gov/citations/20260001"),
    (DblpAdapter, {"result": {"hits": {"@total": "1", "hit": [{"@id": "1", "info": {
        "title": "Solid-state battery models.", "year": "2025", "ee": "https://doi.org/10.1/x"}}]}}},
     "https://doi.org/10.1/x"),
    (ChemRxivAdapter, {"totalCount": 1, "itemHits": [{"item": {
        "id": "66aa", "title": "Solid-state battery chemistry", "publishedDate": "2026-04-01T10:00:00.000Z",
        "doi": "10.26434/chemrxiv-2026-66aa"}}]}, "https://doi.org/10.26434/chemrxiv-2026-66aa"),
])
def test_science_search_sources_parse_one_record(adapter_type, payload, url):
    adapter = adapter_type(_client(lambda request: httpx.Response(200, json=payload)))
    page, = _pages(adapter)
    item, = page.observations
    assert item.url == url and item.source_id == adapter.source_id
    assert page.exhausted
    if adapter_type in {HalAdapter, DblpAdapter}:
        assert item.date_basis == "year"


def test_openaire_reads_single_and_listed_nodes_and_bounds_the_period():
    seen = []
    payload = {"response": {"header": {"total": {"$": 2}}, "results": {"result": [
        {"header": {"dri:objIdentifier": {"$": "doi_________::a1"}},
         "metadata": {"oaf:entity": {"oaf:result": {
             "title": [{"@classid": "alternative title", "$": "Other"},
                       {"@classid": "main title", "$": "Sulfide solid-state battery electrolytes"}],
             "dateofacceptance": {"$": "2026-05-04"}, "description": {"$": "All-solid-state battery cells"},
             "pid": {"@classid": "doi", "$": "10.1000/ssb"}}}}},
        # Одна ссылка вместо DOI; тема не совпадает — строгая проверка её отбрасывает.
        {"header": {"dri:objIdentifier": {"$": "od::b2"}},
         "metadata": {"oaf:entity": {"oaf:result": {
             "title": {"@classid": "main title", "$": "Regional development agencies"},
             "dateofacceptance": {"$": "2025"},
             "children": {"instance": [{"webresource": {"url": {"$": "https://repo.example.org/b2"}}}]}}}}},
    ]}}}

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=payload)

    page, = _pages(OpenAireAdapter(_client(handler)))
    item, = page.observations
    assert (item.title, item.url, item.published_at) == (
        "Sulfide solid-state battery electrolytes", "https://doi.org/10.1000/ssb", date(2026, 5, 4))
    assert seen[0]["fromDateAccepted"] == "2021-01-01" and seen[0]["toDateAccepted"] == "2026-09-24"
    assert page.total_available == 2 and page.exhausted
    empty = {"response": {"header": {"total": {"$": 0}}, "results": None}}
    page, = _pages(OpenAireAdapter(_client(lambda request: httpx.Response(200, json=empty))))
    assert page.observations == () and page.exhausted


def test_jstage_parses_the_atom_feed_and_keeps_only_the_topic():
    feed = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <result><status>0</status></result><opensearch:totalResults>2</opensearch:totalResults>
      <entry><article_title><en>Composite electrodes for all-solid-state battery</en></article_title>
        <article_link><en>https://www.jstage.jst.go.jp/article/x/1/_article</en></article_link>
        <material_title><en>Journal of the Ceramic Society of Japan</en></material_title>
        <pubyear>2024</pubyear><prism:doi>10.2109/x.1</prism:doi></entry>
      <entry><article_title><en>Rice cultivation in Hokkaido</en></article_title>
        <article_link><en>https://www.jstage.jst.go.jp/article/y/2/_article</en></article_link>
        <pubyear>2025</pubyear></entry></feed>"""
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, text=feed)

    page, = _pages(JStageAdapter(_client(handler)))
    item, = page.observations
    assert item.url == "https://doi.org/10.2109/x.1" and item.country == "JP" and item.date_basis == "year"
    assert seen[0]["pubyearfrom"] == "2021" and page.exhausted
    failed = '<feed xmlns="http://www.w3.org/2005/Atom"><result><status>ERR_001</status></result></feed>'
    with pytest.raises(SourceFetchError):
        _pages(JStageAdapter(_client(lambda request: httpx.Response(200, text=failed))))


def test_npm_packages_are_community_leads_checked_against_the_topic():
    payload = {"total": 2, "objects": [
        {"downloads": {"monthly": 1200}, "package": {
            "name": "solid-state-battery-sim", "description": "Solid-state battery cell simulator",
            "date": "2026-09-01T10:00:00.000Z", "keywords": ["battery", "simulation"]}},
        {"package": {"name": "@ui/kit", "description": "Batteries included UI kit",
                     "date": "2026-09-01T10:00:00.000Z"}},
    ]}
    page, = _pages(NpmAdapter(_client(lambda request: httpx.Response(200, json=payload))))
    item, = page.observations
    assert item.url == "https://www.npmjs.com/package/solid-state-battery-sim" and item.kind == "repository"
    assert "1200 загрузок в месяц" in item.summary and item.title.startswith("solid-state-battery-sim: ")


def test_stack_overflow_and_hugging_face_are_community_leads():
    questions = {"has_more": False, "items": [{"question_id": 5, "title": "Solid-state battery simulation",
                                               "link": "https://stackoverflow.com/questions/5/x",
                                               "creation_date": int(datetime(2026, 9, 1, tzinfo=UTC).timestamp()),
                                               "tags": ["python", "battery"]}]}
    page, = _pages(StackExchangeAdapter(_client(lambda request: httpx.Response(200, json=questions))))
    assert page.observations[0].kind == "community" and page.exhausted
    models = [{"id": "lab/solid-state-battery-predictor", "createdAt": "2026-08-01T00:00:00.000Z",
               "likes": 3, "pipeline_tag": "tabular-regression"},
              {"id": "lab/unrelated-chatbot", "createdAt": "2026-08-01T00:00:00.000Z"}]
    page, = _pages(HuggingFaceAdapter(_client(lambda request: httpx.Response(200, json=models))))
    assert [item.item_id for item in page.observations] == ["lab/solid-state-battery-predictor"]


def test_habr_searches_with_the_russian_query_first():
    requested = []

    def handler(request):
        requested.append((request.url.path, request.url.params.get("q")))
        return httpx.Response(200, text="""<rss><channel><item><title>Твердотельные аккумуляторы Samsung</title>
            <link>https://habr.com/ru/articles/1/</link><pubDate>Wed, 23 Sep 2026 12:00:00 GMT</pubDate></item>
            </channel></rss>""")

    adapter = HabrAdapter(_client(handler))
    adapter.configure(localized={"ru": "твердотельные аккумуляторы"})
    with pytest.raises(SourceFetchError, match="recent_feed_only"):
        pages = []
        for page in adapter.iter_pages("solid-state battery", as_of=_AS_OF, limit=10,
                                       timeout_seconds=5, cancel=Event()):
            pages.append(page)
    assert requested == [("/ru/rss/search/", "твердотельные аккумуляторы")]
    assert pages[0].observations[0].country == "RU"


def test_gdelt_applies_the_country_filter():
    queries = []

    def handler(request):
        queries.append(request.url.params["query"])
        return httpx.Response(200, json={"articles": [{"url": "https://example.ru/a", "title": "Battery",
                                                       "seendate": "20260923T120000Z", "sourcecountry": "Russia"}]})

    adapter = GdeltDocAdapter(_client(handler))
    adapter.configure(countries=("RU", "US"))
    page, = _pages(adapter)
    assert queries == ["solid-state battery (sourcecountry:russia OR sourcecountry:unitedstates)"]
    assert page.observations[0].country == "RU"


def test_policy_keeps_catalogue_order_and_records_skipped_sources():
    policy = SourcePolicy(countries=("RU",), disabled=("habr",))
    assert not policy.allows("arxiv") and policy.skip_reason("arxiv") == "country_filter"
    assert policy.skip_reason("habr") == "disabled_by_owner"
    assert policy.allows("cyberleninka") and policy.allows("google_news")
    assert policy.editions(GoogleNewsAdapter.EDITIONS) == ("RU",)
    assert SourcePolicy(countries=("EU",)).allows("horizon_magazine")
    assert SourcePolicy(countries=("EU",)).allows("hal")  # Франция — страна Евросоюза.
    assert SourcePolicy().editions(GoogleNewsAdapter.EDITIONS) == ("US", "RU")

    snapshot = collect_approved_sources("solid-state battery", as_of=_AS_OF, cancel=Event(), policy=policy,
                                        adapters={"cyberleninka": CyberLeninkaAdapter(_client(
                                            lambda request: httpx.Response(200, json={"found": 0, "articles": []})))})
    assert tuple(item.source_id for item in snapshot.coverage) == SOURCE_IDS
    reasons = {item.source_id: item.reason_code for item in snapshot.coverage}
    assert reasons["arxiv"] == "country_filter" and reasons["habr"] == "disabled_by_owner"
    assert reasons["cyberleninka"] is None
    assert reasons["hal"] == "country_filter"


def test_policy_round_trips_and_rejects_unknown_values(tmp_path):
    policy = SourcePolicy(countries=("RU", "INT"), disabled=("gdelt",), weights={"arxiv": 1.4})
    save_policy(tmp_path, policy)
    assert load_policy(tmp_path) == policy
    with pytest.raises(ValueError):
        SourcePolicy(countries=("XX",))
    with pytest.raises(ValueError):
        SourcePolicy(weights={"arxiv": 9.0})
    (tmp_path / "web-source-policy.json").write_text("{broken", encoding="utf-8")
    assert load_policy(tmp_path) == SourcePolicy()
    rows = catalogue(policy)
    assert len(rows) == len(SOURCE_INFO)
    assert next(row for row in rows if row["source_id"] == "gdelt")["enabled"] is False


def test_learned_weights_share_the_same_budget():
    active = ("arxiv", "habr", "github")
    caps = _caps(50, 90, active, {"arxiv": 1.5, "habr": 0.5})
    assert sum(caps.values()) == 90
    assert caps["arxiv"] > caps["github"] > caps["habr"]
    assert all(caps[source] == 0 for source in SOURCE_IDS if source not in active)


def test_saved_twelve_source_snapshots_stay_readable():
    coverage = [{"source_id": source, "state": "complete", "requested_limit": 1, "scanned": 0, "accepted": 0,
                 "rejected": 0, "duplicates": 0, "limit_reached": False} for source in LEGACY_CATALOGUES[1]]
    snapshot = SourceSnapshot.model_validate({"query": "q", "as_of": "2026-09-24",
                                              "collected_at": "2026-09-24T00:00:00+00:00", "coverage": coverage})
    assert len(snapshot.coverage) == 12


def test_year_only_dates_must_be_the_first_of_january():
    with pytest.raises(ValueError):
        ExternalObservation(source_id="dblp", item_id="1", kind="journal_article", title="T",
                            url="https://dblp.org/rec/1", published_at=date(2026, 5, 1), date_basis="year")
