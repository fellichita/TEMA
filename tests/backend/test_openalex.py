from threading import Event

import httpx
import pytest

from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError
from app.backend.providers.openalex import OpenAlexProvider


def work(number=1, **changes):
    return {"id": f"https://openalex.org/W{number}", "doi": f"https://doi.org/10.1234/paper-{number}",
            "title": "A <i>new</i> technology", "publication_date": "2024-02-29", "publication_year": 2024,
            "abstract_inverted_index": {"New": [0], "technology": [1]},
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}], "type": "article", **changes}


def payload(items, count=None, cursor=None):
    return {"meta": {"count": len(items) if count is None else count, "next_cursor": cursor}, "results": items}


def provider(handler, **kwargs):
    return OpenAlexProvider(client=httpx.Client(transport=httpx.MockTransport(handler)),
                            page_delay=0, retry_delay=0, **kwargs)


def collect(source, **changes):
    return list(source.iter_pages(SearchRequest(topic="quantum sensors", source="openalex", **changes), Event()))


def test_normalization_and_query_preserve_evidence():
    seen = []
    item = work()

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload([item]))

    pages = collect(provider(handler), from_date="2024-01-01", until_date="2024-12-31")
    record = pages[0].documents[0]
    assert record.doi == "10.1234/paper-1" and record.source_id == "W1"
    assert record.abstract == "New technology" and record.authors == ("Ada Lovelace",)
    assert record.publication_month == 2 and record.title == "A new technology"
    assert record.raw_metadata == item
    assert seen[0].url.params["sort"] == "relevance_score:desc"
    assert seen[0].url.params["search"] == "quantum sensors"
    assert "from_publication_date:2024-01-01" in seen[0].url.params["filter"]
    assert "Authorization" not in seen[0].headers


def test_optional_key_is_header_only_and_not_saved():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer TEST_SECRET"
        assert "TEST_SECRET" not in str(request.url)
        return httpx.Response(200, json=payload([work()]))
    pages = collect(provider(handler, api_key="TEST_SECRET"))
    assert "TEST_SECRET" not in pages[0].model_dump_json()


def test_page_size_obeys_supported_openalex_limit():
    source = provider(lambda request: httpx.Response(200, json=payload([work()])))
    try:
        assert source.page_size == 100
        assert collect(source, max_results=150)[0].scanned == 1
    finally:
        source.close()
    with pytest.raises(ValueError, match="1–100"):
        OpenAlexProvider(page_size=200)


def test_no_doi_and_no_abstract_is_still_valid():
    pages = collect(provider(lambda request: httpx.Response(200, json=payload([
        work(doi=None, abstract_inverted_index=None, publication_date=None),
    ]))))
    record = pages[0].documents[0]
    assert record.document_key == "openalex:W1" and record.abstract is None
    assert record.date_precision == "year" and record.publication_date is None


@pytest.mark.parametrize("changes", [
    {"id": "https://evil.test/W1"}, {"title": " "}, {"publication_date": "2024-02-30"},
    {"publication_year": 2023}, {"doi": "invalid"},
    {"abstract_inverted_index": {"huge": [10**9]}},
    {"abstract_inverted_index": {"A": [0], "B": [0]}},
    {"abstract_inverted_index": {"A": [0], "B": [2]}},
    {"abstract_inverted_index": {"A": [True]}},
])
def test_invalid_records_skipped_without_large_allocation(changes):
    page = collect(provider(lambda request: httpx.Response(200, json=payload([work(**changes)]))))[0]
    assert page.skipped == 1 and not page.documents


def test_pagination_and_limit():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(200, json=payload([work(1), work(2)], 100, "next"))
        assert request.url.params["cursor"] == "next" and request.url.params["per_page"] == "1"
        return httpx.Response(200, json=payload([work(3)], 100, "third"))

    pages = collect(provider(handler, page_size=2), max_results=3)
    assert sum(p.scanned for p in pages) == 3 and not pages[-1].exhausted


def test_repeated_cursor_is_not_silently_complete():
    source = provider(lambda request: httpx.Response(200, json=payload([work()], 100, "*")), page_size=1)
    with pytest.raises(BackendError) as error:
        collect(source)
    assert error.value.code == "invalid_response"


def test_empty_result_is_successful_empty_page():
    pages = collect(provider(lambda request: httpx.Response(200, json=payload([]))))
    assert pages[0].exhausted and pages[0].scanned == 0


@pytest.mark.parametrize("status,code", [(401, "authentication_required"), (403, "authentication_required"),
                                        (429, "rate_limited"), (503, "source_unavailable")])
def test_http_errors_do_not_leak_credentials(status, code):
    source = provider(lambda request: httpx.Response(status, text="TEST_SECRET"),
                      api_key="TEST_SECRET", max_retries=0)
    with pytest.raises(BackendError) as error:
        collect(source)
    assert error.value.code == code and "TEST_SECRET" not in str(error.value)


def test_primary_topic_filter_replaces_text_search_on_every_page():
    seen = []

    def handler(request):
        seen.append(request)
        index = len(seen)
        return httpx.Response(200, json=payload([
            work(index, primary_topic={"id": f"https://openalex.org/T{index}"}),
        ], count=2, cursor="second" if index == 1 else None))

    pages = collect(provider(handler, page_size=1), max_results=2,
                    primary_topic_ids=("T2", "T1"), from_date="2024-01-01", until_date="2024-12-31")
    assert len(pages) == 2 and pages[-1].exhausted
    for request in seen:
        assert "search" not in request.url.params
        assert request.url.params["sort"] == "publication_date:asc"
        assert request.url.params["filter"] == (
            "to_publication_date:2024-12-31,from_publication_date:2024-01-01,primary_topic.id:T1|T2")


@pytest.mark.parametrize("primary_topic", [None, {}, {"id": "https://openalex.org/T3"}])
def test_primary_topic_filter_mismatch_fails_instead_of_silently_widening_scope(primary_topic):
    source = provider(lambda request: httpx.Response(200, json=payload([work(primary_topic=primary_topic)])))
    with pytest.raises(BackendError) as error:
        collect(source, primary_topic_ids=("T1", "T2"))
    assert error.value.code == "invalid_response"


def test_budget_refusal_answers_the_rest_of_the_operation_at_once():
    """Measured 2026-09-24: an exhausted anonymous budget made every later query
    of an analysis wait and fail; the same operation now fails immediately."""
    calls = []

    def refuse(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "37149"})

    source = provider(refuse, max_retries=3, max_retry_delay=60)
    operation = Event()
    request = SearchRequest(topic="quantum sensors", source="openalex")
    for _ in range(3):
        with pytest.raises(BackendError) as error:
            list(source.iter_pages(request, operation))
        assert error.value.code == "rate_limited"
    assert len(calls) == 1
    # Another analysis and a request with its own key still ask the source.
    with pytest.raises(BackendError):
        list(source.iter_pages(request, Event()))
    keyed = provider(refuse, api_key="own-budget-key", max_retries=0)
    with pytest.raises(BackendError):
        list(keyed.iter_pages(request, operation))
    assert len(calls) == 3


def test_an_outage_that_outlived_the_retries_is_not_asked_again_by_the_same_run(monkeypatch):
    """27.09.2026: OpenAlex answered 503 "Anonymous search is paused" for minutes and
    every history query of a run waited through its retries for nothing."""
    import app.backend.providers.http_transport as transport
    from app.backend.providers.openalex import OUTAGE_PAUSE_SECONDS

    calls = []

    def paused(request):
        calls.append(request)
        return httpx.Response(503, json={"error": "Search temporarily unavailable"})

    source = provider(paused, max_retries=2)
    operation = Event()
    request = SearchRequest(topic="quantum sensors", source="openalex")
    for _ in range(3):
        with pytest.raises(BackendError) as error:
            list(source.iter_pages(request, operation))
        assert error.value.code == "source_unavailable"
    assert len(calls) == 3  # One request with its two retries, then no more.
    with pytest.raises(BackendError):
        list(source.iter_pages(request, Event()))
    assert len(calls) == 6
    clock = transport.time.monotonic() + OUTAGE_PAUSE_SECONDS + 1
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock)
    with pytest.raises(BackendError):
        list(source.iter_pages(request, operation))
    assert len(calls) == 9


def test_retries_exhausted_by_short_pauses_pause_the_operation(monkeypatch):
    import app.backend.providers.http_transport as transport

    calls = []

    def throttle(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"})

    source = provider(throttle, max_retries=2)
    operation = Event()
    request = SearchRequest(topic="quantum sensors", source="openalex")
    with pytest.raises(BackendError):
        list(source.iter_pages(request, operation))
    assert len(calls) == 3
    with pytest.raises(BackendError):
        list(source.iter_pages(request, operation))
    assert len(calls) == 3
    # The pause is bounded: after it the source is asked again.
    clock = transport.time.monotonic() + transport._EXHAUSTED_RETRY_PAUSE + 1
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock)
    with pytest.raises(BackendError):
        list(source.iter_pages(request, operation))
    assert len(calls) == 6
