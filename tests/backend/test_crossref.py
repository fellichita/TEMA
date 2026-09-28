"""No network access: exercise the provider against controlled HTTP streams."""

import gzip
import json
from datetime import date
from threading import Event

import httpx
import pytest

from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.crossref import CrossrefProvider


def work(number=1, **changes):
    item = {
        "DOI": f"10.1234/PAPER-{number}",
        "title": ["A <i>new</i> technology"],
        "published": {"date-parts": [[2024, 2, 29]]},
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "type": "journal-article",
        "is-referenced-by-count": 7,
    }
    item.update(changes)
    return item


def result(items, total=None, cursor="opaque-cursor"):
    message = {"items": items, "next-cursor": cursor}
    if total is not None:
        message["total-results"] = total
    return {"status": "ok", "message-type": "work-list", "message": message}


def provider(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return CrossrefProvider(client, page_delay=0, retry_delay=0, **kwargs)


def collect(source, **kwargs):
    return list(source.iter_pages(SearchRequest(topic="quantum sensors", **kwargs), Event()))


def test_request_filters_and_normalized_evidence():
    requests = []
    item = work(
        DOI="https://doi.org/10.1234/Example",
        abstract="<jats:p>First &amp; second.</jats:p><jats:p>A result.</jats:p>",
        publisher="Example publisher",
        unknown_field={"preserved": True},
    )

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=result([item], total=1))

    source = provider(handler, page_size=25)
    pages = collect(source, from_date=date(2020, 1, 1), until_date=date(2025, 1, 1), max_results=40)
    assert len(requests) == 1
    assert str(requests[0].url).startswith("https://api.crossref.org/works?")
    assert requests[0].url.params["query"] == "quantum sensors"
    assert requests[0].url.params["filter"] == "from-pub-date:2020-01-01,until-pub-date:2025-01-01"
    assert requests[0].url.params["rows"] == "25"
    assert requests[0].url.params["cursor"] == "*"
    assert requests[0].url.params["sort"] == "relevance"
    assert requests[0].url.params["order"] == "desc"
    document = pages[0].documents[0]
    assert document.doi == "10.1234/example"
    assert document.source_id == document.doi
    assert document.url == "https://doi.org/10.1234/example"
    assert document.title == "A new technology"
    assert document.abstract == "First & second. A result."
    assert document.authors == ("Ada Lovelace",)
    assert document.publication_date == date(2024, 2, 29)
    assert document.date_precision == "day"
    assert document.citation_count == 7
    assert document.raw_metadata == item
    assert "query" not in document.raw_metadata
    assert pages[0].exhausted


@pytest.mark.parametrize(
    ("parts", "year", "precision", "full_date"),
    [([2024], 2024, "year", None), ([2024, 6], 2024, "month", None), ([2024, 6, 7], 2024, "day", date(2024, 6, 7))],
)
def test_partial_dates_are_not_fabricated(parts, year, precision, full_date):
    source = provider(lambda _: httpx.Response(200, json=result([work(published={"date-parts": [parts]})])))
    record = collect(source)[0].documents[0]
    assert record.publication_year == year
    assert record.publication_month == (parts[1] if len(parts) >= 2 else None)
    assert record.publication_date == full_date
    assert record.date_precision == precision
    assert record.raw_metadata["published"]["date-parts"] == [parts]


def test_missing_abstract_and_publication_date_stay_unknown():
    item = work(published=None, created={"date-parts": [[2024, 5, 1]]})
    source = provider(lambda _: httpx.Response(200, json=result([item])))
    record = collect(source)[0].documents[0]
    assert record.abstract is None
    assert record.publication_year is None
    assert record.publication_date is None
    assert record.date_precision == "unknown"


def test_issued_date_and_group_authors():
    source = provider(lambda _: httpx.Response(200, json=result([work(
        published=None,
        issued={"date-parts": [[2023]]},
        author=[{"name": "Research Consortium"}, None, {"family": "Curie"}],
    )])))
    record = collect(source)[0].documents[0]
    assert record.publication_year == 2023
    assert record.authors == ("Research Consortium", "Curie")


@pytest.mark.parametrize("bad_item", [
    None,
    "not a record",
    {},
    work(DOI="not-a-doi"),
    work(title=[]),
    work(title=["   "]),
    work(abstract=42),
    work(published={"date-parts": [[2023, 2, 29]]}),
    work(published={"date-parts": [[2024, 13]]}),
    work(published={"date-parts": [[True]]}),
    work(published={"date-parts": [[2024, 1, 1, 0]]}),
    work(published={"date-parts": []}),
    work(published={"date-parts": [["2024"]]}),
    work(**{"is-referenced-by-count": -1}),
])
def test_invalid_records_are_counted_as_skipped(bad_item):
    source = provider(lambda _: httpx.Response(200, json=result([bad_item, work(2)], total=2)))
    page = collect(source)[0]
    assert page.scanned == 2
    assert page.skipped == 1
    assert len(page.documents) == 1


def test_limit_counts_raw_records_and_shortens_last_request():
    requests = []

    def handler(request):
        requests.append(request)
        items = [None, work(2)] if len(requests) == 1 else [work(3)]
        return httpx.Response(200, json=result(items, total=10))

    pages = collect(provider(handler, page_size=2), max_results=3)
    assert [request.url.params["rows"] for request in requests] == ["2", "1"]
    assert sum(page.scanned for page in pages) == 3
    assert sum(len(page.documents) for page in pages) == 2
    assert not pages[-1].exhausted


@pytest.mark.parametrize("page_size", [200, 400])
def test_large_page_falls_back_without_skipping_cursor_or_records(page_size):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, headers={"Content-Length": "6000000"}, content=b"{}")
        first = 1 if len(requests) == 2 else 101
        return httpx.Response(200, json=result(
            [work(number) for number in range(first, first + 100)],
            total=200,
            cursor=f"cursor-{len(requests)}",
        ))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cancel = RecordingEvent()
    pages = list(CrossrefProvider(client, page_size=page_size).iter_pages(
        SearchRequest(topic="quantum", max_results=400), cancel,
    ))

    assert [request.url.params["rows"] for request in requests] == [str(page_size), "100", "100"]
    assert [request.url.params["cursor"] for request in requests] == ["*", "*", "cursor-2"]
    assert cancel.waits == [1.0, 1.0]
    assert [page.scanned for page in pages] == [100, 100]
    assert [doc.doi for page in pages for doc in page.documents] == [
        f"10.1234/paper-{number}" for number in range(1, 201)
    ]
    assert pages[-1].exhausted


def test_unavailable_large_page_falls_back_to_100():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=result([work()], total=1))

    source = provider(handler, page_size=200, max_retries=0)
    assert collect(source, max_results=200)[0].scanned == 1
    assert [request.url.params["rows"] for request in requests] == ["200", "100"]


def test_cancel_during_large_page_fallback_stops_before_retry():
    class CancelOnWait(Event):
        def wait(self, timeout=None):
            self.set()
            return True

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    source = provider(handler, page_size=200, max_retries=0)
    with pytest.raises(CancelledError):
        list(source.iter_pages(SearchRequest(topic="quantum", max_results=200), CancelOnWait()))
    assert len(calls) == 1


@pytest.mark.parametrize(("response", "code"), [
    (httpx.Response(429), "rate_limited"),
    (httpx.Response(200, json={"message": []}), "invalid_response"),
])
def test_large_page_does_not_fall_back_for_unrelated_errors(response, code):
    requests = []

    def handler(request):
        requests.append(request)
        return response

    with pytest.raises(BackendError) as error:
        collect(provider(handler, page_size=200, max_retries=0), max_results=200)
    assert error.value.code == code
    assert len(requests) == 1


def test_repeating_cursor_is_not_end_of_results():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=result([work(len(requests))], total=3, cursor="same"))

    pages = collect(provider(handler, page_size=1), max_results=10)
    assert [request.url.params["cursor"] for request in requests] == ["*", "same", "same"]
    assert len(pages) == 3
    assert pages[-1].exhausted
    assert len({page.documents[0].doi for page in pages}) == 3


def test_empty_result_yields_exhausted_page():
    page = collect(provider(lambda _: httpx.Response(200, json=result([], total=0))))[0]
    assert page.scanned == 0 and page.exhausted and page.documents == ()


def test_short_page_marks_source_exhausted():
    page = collect(provider(lambda _: httpx.Response(200, json=result([work()], total=100))))[0]
    assert page.exhausted


def test_missing_cursor_is_error_if_another_page_is_required():
    source = provider(lambda _: httpx.Response(200, json=result([work()], total=10, cursor=None)), page_size=1)
    with pytest.raises(BackendError) as error:
        collect(source)
    assert error.value.code == "invalid_response"


def test_missing_cursor_allowed_when_request_limit_reached():
    source = provider(lambda _: httpx.Response(200, json=result([work()], total=10, cursor=None)), page_size=1)
    assert len(collect(source, max_results=1)) == 1


@pytest.mark.parametrize("payload", [[], {"message": []}, {"message": {}}, {"message": {"items": [], "total-results": True}}, {"status": "error", "message": {"items": []}}])
def test_invalid_response_shape_is_sanitized(payload):
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, json=payload)))
    assert error.value.code == "invalid_response"
    assert "quantum" not in str(error.value)


def test_invalid_json_does_not_leak_body():
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, content=b"secret body and token")))
    assert error.value.code == "invalid_response"
    assert "secret" not in str(error.value)


def test_transient_errors_retry_then_succeed():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) < 3 else 200, json=result([work()]))

    assert collect(provider(handler))[0].scanned == 1
    assert len(calls) == 3


def test_transport_error_retries_and_does_not_leak_request():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("secret url?token=private", request=request)

    with pytest.raises(BackendError) as error:
        collect(provider(handler, max_retries=1))
    assert error.value.code == "source_unavailable"
    assert "private" not in str(error.value)
    assert len(calls) == 2


class RecordingEvent(Event):
    def __init__(self):
        super().__init__()
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()


def test_rate_limit_respects_retry_after():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json=result([]))

    cancel = RecordingEvent()
    list(provider(handler).iter_pages(SearchRequest(topic="quantum"), cancel))
    assert cancel.waits == [2.0]
    assert len(calls) == 2


@pytest.mark.parametrize("delay", ["31", "inf", "NaN"])
def test_excessive_or_nonfinite_retry_after_fails_without_retry(delay):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": delay})

    with pytest.raises(BackendError) as error:
        collect(provider(handler))
    assert error.value.code == "rate_limited"
    assert len(calls) == 1


def test_rate_limit_exhaustion_has_distinct_code():
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(429), max_retries=0))
    assert error.value.code == "rate_limited"


def test_redirects_never_follow_even_with_redirect_enabled_client():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    with pytest.raises(BackendError):
        collect(CrossrefProvider(client, page_delay=0))
    assert len(calls) == 1


def test_cancel_before_network():
    cancel = Event()
    cancel.set()
    source = provider(lambda _: pytest.fail("Network request after cancellation"))
    with pytest.raises(CancelledError):
        list(source.iter_pages(SearchRequest(topic="quantum"), cancel))


def test_cancel_during_retry_wait():
    class CancelOnWait(RecordingEvent):
        def wait(self, timeout=None):
            self.set()
            return super().wait(timeout)

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    with pytest.raises(CancelledError):
        list(provider(handler).iter_pages(SearchRequest(topic="quantum"), CancelOnWait()))
    assert len(calls) == 1


def test_successful_pages_use_cancellable_pause():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=result([work(len(calls))], total=2))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cancel = RecordingEvent()
    list(CrossrefProvider(client, page_size=1).iter_pages(SearchRequest(topic="quantum"), cancel))
    assert cancel.waits == [1.0]


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks, after_chunk=None):
        self.chunks = chunks
        self.after_chunk = after_chunk
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if self.after_chunk:
                self.after_chunk()
            yield chunk

    def close(self):
        self.closed = True


def test_stream_size_limit_and_response_closed():
    stream = Chunks([b" " * 100, b" " * 100])
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, stream=stream), max_response_bytes=150))
    assert error.value.code == "response_too_large"
    assert stream.closed


def test_content_length_size_limit():
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, headers={"Content-Length": "999"}, content=b"{}"), max_response_bytes=100))
    assert error.value.code == "response_too_large"


def test_decompressed_payload_size_is_limited():
    encoded = gzip.compress(json.dumps(result([work(abstract="a" * 10_000)])).encode())
    assert len(encoded) < 1000
    stream = Chunks([encoded])
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream), max_response_bytes=1000))
    assert error.value.code == "response_too_large"
    assert stream.closed


def test_cancel_during_stream_closes_response():
    cancel = Event()
    stream = Chunks([b" " * 100], after_chunk=cancel.set)
    source = provider(lambda _: httpx.Response(200, stream=stream))
    with pytest.raises(CancelledError):
        list(source.iter_pages(SearchRequest(topic="quantum"), cancel))
    assert stream.closed


def test_slow_stream_hits_total_page_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.backend.providers.http_transport.time.monotonic", lambda: clock[0])
    stream = Chunks([b" "], after_chunk=lambda: clock.__setitem__(0, 61.0))
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, stream=stream), page_deadline_seconds=60))
    assert error.value.code == "source_unavailable"
    assert stream.closed


def test_deadline_checks_compressed_chunks_even_without_decoded_output(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.backend.providers.http_transport.time.monotonic", lambda: clock[0])
    stream = Chunks([b"\x1f"], after_chunk=lambda: clock.__setitem__(0, 61.0))
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)))
    assert error.value.code == "source_unavailable"
    assert stream.closed


def test_valid_compressed_stream_is_decoded():
    encoded = gzip.compress(json.dumps(result([work()])).encode())
    stream = Chunks([encoded[:10], encoded[10:]])
    page = collect(provider(lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)))[0]
    assert page.scanned == 1
    assert page.documents[0].doi == "10.1234/paper-1"


def test_incomplete_compressed_stream_is_invalid():
    stream = Chunks([b"\x1f"])
    with pytest.raises(BackendError) as error:
        collect(provider(lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)))
    assert error.value.code == "invalid_response"


def test_external_xml_entities_do_not_resolve():
    xml = '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///private">]><jats:p>&xxe;</jats:p>'
    page = collect(provider(lambda _: httpx.Response(200, json=result([work(abstract=xml)]))))[0]
    assert page.scanned == 1
    if page.documents:
        assert "private" not in (page.documents[0].abstract or "")


def test_reserved_doi_characters_are_url_encoded():
    source = provider(lambda _: httpx.Response(200, json=result([work(DOI="10.1234/thing?part#one")])))
    assert collect(source)[0].documents[0].url == "https://doi.org/10.1234/thing%3Fpart%23one"


def test_injected_client_stays_owned_by_caller():
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result([]))))
    CrossrefProvider(client).close()
    assert not client.is_closed
    client.close()


@pytest.mark.parametrize("kwargs", [{"page_size": 0}, {"page_size": True}, {"page_size": 1001}, {"timeout_seconds": float("inf")}, {"max_retries": -1}, {"max_response_bytes": 0}, {"retry_delay": -1}, {"page_deadline_seconds": 0}])
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        CrossrefProvider(**kwargs)


def test_publisher_access_tokens_in_full_text_links_do_not_reach_the_archive():
    """Measured 24.09.2026: one Crossref PDF link with ?token= refused a finished analysis."""
    from app.runtime.backup import assert_no_credentials

    links = [{"URL": "https://www.whxb.pku.edu.cn/CN/PDF/10.3866/PKU.WHXB202305040?token=bd08d653e62c41e4a5769e6116f6cfc8",
              "content-type": "application/pdf"},
             {"URL": "https://example.org/fulltext?lang=en&signature=abc&format=pdf"},
             {"URL": "https://kiss.kstudy.com/Detail/Ar?key=4012345"}]
    pages = collect(provider(lambda _: httpx.Response(200, json=result([work(link=links)], total=1))))
    record = pages[0].documents[0]
    assert [item["URL"] for item in record.raw_metadata["link"]] == [
        "https://www.whxb.pku.edu.cn/CN/PDF/10.3866/PKU.WHXB202305040",
        "https://example.org/fulltext?lang=en&format=pdf",
        "https://kiss.kstudy.com/Detail/Ar?key=4012345"]
    assert_no_credentials(record.model_dump(mode="json"))


def test_the_final_credential_check_is_unchanged_for_links_it_did_not_receive_from_a_source():
    from app.runtime.backup import ArchiveError, assert_no_credentials

    for url in ("https://api.example.org/data?api_key=secret", "https://user:pass@example.org/data"):
        with pytest.raises(ArchiveError):
            assert_no_credentials({"link": url})
