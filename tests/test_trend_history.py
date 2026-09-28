"""Сбор помесячной истории технологии на подменённых ответах источников."""

from datetime import date
import json

import httpx
import pytest

from app import trend_history
from app.backend.contracts import DocumentRecord, SourcePage
from app.trend_history import collect_history, phrase_pattern, work_key

AS_OF = date(2026, 9, 26)


@pytest.fixture(autouse=True)
def no_arxiv_pause(monkeypatch):
    # Пауза между запросами к arXiv нужна настоящему API, а не подменённому.
    monkeypatch.setattr(trend_history, "ARXIV_INTERVAL_SECONDS", 0.0)

ARXIV = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry><id>http://arxiv.org/abs/2607.00001v1</id><published>2026-07-03T10:00:00Z</published>
    <title>Optical Circuit Switches for AI clusters</title></entry>
  <entry><id>http://arxiv.org/abs/2601.00002v1</id><published>2026-01-10T10:00:00Z</published>
    <title>Packet switching revisited</title></entry>
</feed>"""


def handler(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host == "export.arxiv.org":
        assert 'ti:"optical circuit switch"' in request.url.params["search_query"]
        return httpx.Response(200, text=ARXIV)
    if host == "www.ebi.ac.uk":
        return httpx.Response(200, json={"hitCount": 1, "resultList": {"result": [
            {"id": "1", "source": "MED", "doi": "10.1234/x", "title": "Optical circuit switching in data centres",
             "firstPublicationDate": "2026-03-05"}]}})
    if host == "hn.algolia.com":
        return httpx.Response(200, json={"nbHits": 1, "hits": [
            {"objectID": "7", "title": "Google's optical circuit switches explained", "created_at_i": 1767225600}]})
    raise AssertionError(host)


class FakeCrossref:
    def iter_pages(self, request, cancel):
        assert request.from_date == date(2024, 9, 1) and request.until_date == date(2026, 8, 31)
        documents = (
            DocumentRecord(source="crossref", source_id="c1", title="Optical Circuit Switching in data centres",
                           publication_year=2026, publication_month=3, publication_date=date(2026, 3, 5),
                           date_precision="day", url="https://doi.org/10.1234/x", doi="10.1234/x"),
            DocumentRecord(source="crossref", source_id="c2", title="Unrelated photonics paper",
                           publication_year=2026, publication_month=4, date_precision="month",
                           url="https://doi.org/10.1234/y", doi="10.1234/y"),
        )
        yield SourcePage(documents=documents, scanned=2, skipped=0, total_available=2, exhausted=True)


def test_phrase_matches_plurals_and_hyphens_but_not_other_words():
    pattern = phrase_pattern(["optical circuit switch", "solid-state battery"])
    assert pattern.search("Optical Circuit Switches for AI")
    assert pattern.search("All-solid state batteries with sulfide electrolytes")
    assert not pattern.search("Optical circuits and switching fabrics")


def test_history_admits_title_matches_from_every_source_and_keeps_coverage():
    client = httpx.Client(transport=httpx.MockTransport(handler))
    history = collect_history(["optical circuit switch", "optical circuit switching"], AS_OF, client=client, crossref=FakeCrossref())
    by_source = {item.source_id: item for item in history.coverage}
    assert {source: item.state for source, item in by_source.items()} == {
        "crossref": "complete", "arxiv": "complete", "europe_pmc": "complete", "hacker_news": "complete"}
    assert by_source["arxiv"].admitted == 1 and by_source["crossref"].admitted == 1
    assert history.coverage_complete
    # Одна работа в Crossref и Europe PMC — один ключ: в кривой она посчитается один раз.
    keys = [record.key for record in history.records if record.month == "2026-03"]
    assert len(keys) == 2 and len(set(keys)) == 1
    assert {record.month for record in history.records} == {"2026-03", "2026-07", "2026-01"}


def test_failed_source_is_lost_coverage_not_zero():
    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hn.algolia.com":
            return httpx.Response(503)
        return handler(request)

    history = collect_history(["optical circuit switch"], AS_OF,
                              client=httpx.Client(transport=httpx.MockTransport(broken)), crossref=FakeCrossref())
    hacker_news = next(item for item in history.coverage if item.source_id == "hacker_news")
    assert hacker_news.state == "unavailable" and hacker_news.reason == "http_503"
    assert not history.coverage_complete


def test_work_key_joins_preprint_and_journal_titles():
    assert work_key("Optical Circuit Switching: A Survey") == work_key("optical circuit switching — a survey")
    assert json.dumps(work_key("x"))


def _feed(total: int, entries: list[tuple[str, str]]) -> str:
    items = "".join(f"<entry><id>http://arxiv.org/abs/{index}</id><published>{published}T10:00:00Z</published>"
                    f"<title>{title}</title></entry>" for index, (title, published) in enumerate(entries))
    return ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f"<opensearch:totalResults>{total}</opensearch:totalResults>{items}</feed>")


def test_arxiv_batcher_answers_several_technologies_with_one_request():
    from threading import Event

    from app.trend_history import ArxivBatcher, month_window

    queries = []

    def arxiv(request):
        queries.append(request.url.params["search_query"])
        return httpx.Response(200, text=_feed(3, [("Optical circuit switches at scale", "2026-07-03"),
                                                  ("Sulfide solid electrolytes for cells", "2026-02-01"),
                                                  ("Unrelated photonics", "2026-03-01")]))

    batcher = ArxivBatcher(httpx.Client(transport=httpx.MockTransport(arxiv)), month_window(AS_OF, 24), Event())
    for forms in (["optical circuit switch"], ["sulfide solid electrolyte"], ["halide electrolyte"]):
        batcher.register("history", forms)
    records, coverage = batcher.history(["optical circuit switch"])
    assert [record.title for record in records] == ["Optical circuit switches at scale"]
    assert (coverage.state, coverage.admitted) == ("complete", 1)
    assert batcher.history(["halide electrolyte"])[0] == []
    assert [record.month for record in batcher.history(["sulfide solid electrolyte"])[0]] == ["2026-02"]
    # Одна пачка — один запрос, со всеми фразами и тем же окном дат.
    assert len(queries) == 1 and batcher.requests == 1
    assert all(f'ti:"{phrase}"' in queries[0] for phrase in ("optical circuit switch", "halide electrolyte"))
    assert "submittedDate:[20240901" in queries[0]


def test_an_overflowing_batch_is_split_until_each_answer_is_complete():
    from threading import Event

    from app.trend_history import ArxivBatcher, month_window

    seen = []

    def arxiv(request):
        query = request.url.params["search_query"]
        seen.append(query.count('ti:"'))
        # Вместе фразы не помещаются в ответ, по отдельности — помещаются.
        if query.count('ti:"') > 1:
            return httpx.Response(200, text=_feed(5000, [("Popular topic", "2026-05-01")]))
        return httpx.Response(200, text=_feed(1, [("Quantum sensor networks", "2026-05-01")]))

    batcher = ArxivBatcher(httpx.Client(transport=httpx.MockTransport(arxiv)), month_window(AS_OF, 24), Event())
    for phrase in ("quantum sensor network", "atom interferometer"):
        batcher.register("history", [phrase])
    records, coverage = batcher.history(["quantum sensor network"])
    assert seen == [2, 1, 1] and coverage.state == "complete" and len(records) == 1


def test_first_mentions_come_from_one_request_while_the_answer_is_whole():
    from threading import Event

    from app.trend_history import ArxivBatcher, month_window

    queries = []

    def arxiv(request):
        queries.append(request.url.params)
        return httpx.Response(200, text=_feed(3, [("Spin qubits in silicon", "2012-04-02"),
                                                  ("Spin qubit arrays", "2019-06-01"),
                                                  ("Rydberg sensors", "2021-01-05")]))

    batcher = ArxivBatcher(httpx.Client(transport=httpx.MockTransport(arxiv)), month_window(AS_OF, 24), Event())
    batcher.register("first", ["spin qubit"])
    batcher.register("first", ["rydberg sensor"])
    assert batcher.first(["spin qubit"]) == (date(2012, 4, 2), 2)
    assert batcher.first(["rydberg sensor"]) == (date(2021, 1, 5), 1)
    assert len(queries) == 1 and queries[0]["sortOrder"] == "ascending" and "submittedDate" not in queries[0]["search_query"]


def test_history_and_first_mention_share_one_all_time_request():
    from threading import Event

    from app.trend_history import ArxivBatcher, month_window

    queries = []

    def arxiv(request):
        queries.append(dict(request.url.params))
        return httpx.Response(200, text=_feed(3, [("Spin qubits in silicon", "2012-04-02"),
                                                  ("Spin qubit arrays at scale", "2026-06-01"),
                                                  ("Rydberg sensors in the field", "2025-01-05")]))

    batcher = ArxivBatcher(httpx.Client(transport=httpx.MockTransport(arxiv)), month_window(AS_OF, 24), Event())
    for phrase in ("spin qubit", "rydberg sensor"):
        batcher.register("history", [phrase])
        batcher.register("first", [phrase])
    records, coverage = batcher.history(["spin qubit"])
    # Полный ответ за всё время: и окно истории, и самое раннее упоминание.
    assert [record.month for record in records] == ["2026-06"] and coverage.state == "complete"
    assert batcher.first(["spin qubit"]) == (date(2012, 4, 2), 2)
    assert batcher.first(["rydberg sensor"]) == (date(2025, 1, 5), 1)
    assert [record.month for record in batcher.history(["rydberg sensor"])[0]] == ["2025-01"]
    assert len(queries) == 1 and "submittedDate" not in queries[0]["search_query"]
    assert queries[0]["max_results"] == "2000"


def test_exact_first_mentions_ask_arxiv_one_phrase_at_a_time():
    from threading import Event

    from app.trend_history import ArxivBatcher, month_window

    queries = []

    def arxiv(request):
        queries.append(dict(request.url.params))
        return httpx.Response(200, text=_feed(430, [("Electric energy storage", "1999-05-01")]))

    batcher = ArxivBatcher(httpx.Client(transport=httpx.MockTransport(arxiv)), month_window(AS_OF, 24), Event(),
                           exact_first=True)
    for phrase in ("electrical energy", "public transportation"):
        batcher.register("history", [phrase])
        batcher.register("first", [phrase])
    # Счёт и самая ранняя дата — как их даёт сам arXiv, по одной фразе.
    assert batcher.first(["electrical energy"]) == (date(1999, 5, 1), 430)
    assert all(query["max_results"] == "1" for query in queries) and len(queries) == 2
