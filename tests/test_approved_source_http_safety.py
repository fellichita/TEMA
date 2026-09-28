"""Hostile source responses must not hold workers or inflate memory."""

from __future__ import annotations

import gzip
import zlib
from datetime import date
from threading import Event
from time import monotonic

import httpx
import pytest

from app.pilot.approved_sources.adapters_news import GitHubRepositoryAdapter, MitResearchNewsAdapter
from app.pilot.approved_sources.adapters_science import ArxivAdapter
from app.pilot.approved_sources._http_safety import read_bounded_response
from app.pilot.approved_sources.contracts import SourceFetchError


class _Chunks(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], after_chunk=None) -> None:
        self.chunks = chunks
        self.after_chunk = after_chunk
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if self.after_chunk is not None:
                self.after_chunk()
            yield chunk

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("adapter_type", [GitHubRepositoryAdapter, ArxivAdapter])
def test_single_gzip_chunk_cannot_expand_past_response_limit(adapter_type):
    compressed = gzip.compress(b"x" * 20_000_000)
    assert len(compressed) < 25_000
    stream = _Chunks([compressed])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})))
    adapter = adapter_type(client)
    with pytest.raises(SourceFetchError, match="response_too_large"):
        list(adapter.iter_pages("quantum", as_of=date(2026, 9, 24), limit=1,
                                timeout_seconds=3, cancel=Event()))
    assert stream.closed


@pytest.mark.parametrize("adapter_type", [GitHubRepositoryAdapter, ArxivAdapter])
def test_normal_gzip_responses_keep_working(adapter_type):
    if adapter_type is GitHubRepositoryAdapter:
        body = b'{"total_count":0,"items":[]}'
    else:
        body = b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
    stream = _Chunks([gzip.compress(body)])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})))
    adapter = adapter_type(client)
    pages = list(adapter.iter_pages("quantum", as_of=date(2026, 9, 24), limit=1,
                                    timeout_seconds=3, cancel=Event()))
    assert len(pages) == 1 and pages[0].scanned == 0
    assert stream.closed


@pytest.mark.parametrize("adapter_type", [GitHubRepositoryAdapter, ArxivAdapter])
@pytest.mark.parametrize("raw", [False, True])
def test_wrapped_and_raw_deflate_responses_keep_working(adapter_type, raw):
    if adapter_type is GitHubRepositoryAdapter:
        body = b'{"total_count":0,"items":[]}'
    else:
        body = b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
    if raw:
        encoder = zlib.compressobj(wbits=-zlib.MAX_WBITS)
        encoded = encoder.compress(body) + encoder.flush()
    else:
        encoded = zlib.compress(body)
    stream = _Chunks([encoded])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream, headers={"Content-Encoding": "deflate"})))
    adapter = adapter_type(client)
    pages = list(adapter.iter_pages("quantum", as_of=date(2026, 9, 24), limit=1,
                                    timeout_seconds=3, cancel=Event()))
    assert len(pages) == 1 and pages[0].scanned == 0
    assert stream.closed


def test_raw_deflate_expansion_is_bounded():
    encoder = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    encoded = encoder.compress(b"x" * 20_000_000) + encoder.flush()
    stream = _Chunks([encoded])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream, headers={"Content-Encoding": "deflate"})))
    with client.stream("GET", "https://example.org") as response:
        assert not response.is_stream_consumed  # Real requests use the raw-stream path.
        with pytest.raises(SourceFetchError, match="response_too_large"):
            read_bounded_response(response, max_bytes=2_000_000,
                                  deadline=monotonic() + 3, cancel=Event())
    assert stream.closed


def test_already_consumed_injected_response_still_checks_size():
    response = httpx.Response(200, content=b"x" * 11)
    assert response.is_stream_consumed
    with pytest.raises(SourceFetchError, match="response_too_large"):
        read_bounded_response(response, max_bytes=10,
                              deadline=monotonic() + 3, cancel=Event())


def test_valid_compressed_body_can_exceed_decoded_limit_on_wire():
    body = bytes(range(64))
    encoded = gzip.compress(body)
    assert len(encoded) > len(body)
    stream = _Chunks([encoded])
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream, headers={
            "Content-Encoding": "gzip", "Content-Length": str(len(encoded))})))
    with client.stream("GET", "https://example.org") as response:
        assert read_bounded_response(response, max_bytes=len(body),
                                     deadline=monotonic() + 3, cancel=Event()) == body
    assert stream.closed


@pytest.mark.parametrize("adapter_type,module", [
    (GitHubRepositoryAdapter, "app.pilot.approved_sources.adapters_news"),
    (ArxivAdapter, "app.pilot.approved_sources.adapters_science"),
])
def test_slow_stream_hits_total_deadline_and_closes_response(monkeypatch, adapter_type, module):
    clock = [0.0]
    monkeypatch.setattr(module + ".monotonic", lambda: clock[0])
    monkeypatch.setattr("app.pilot.approved_sources._http_safety.monotonic", lambda: clock[0])
    stream = _Chunks([b"a", b"b"], after_chunk=lambda: clock.__setitem__(0, clock[0] + 2.0))
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)))
    adapter = adapter_type(client)
    with pytest.raises(SourceFetchError, match="timeout"):
        list(adapter.iter_pages("quantum", as_of=date(2026, 9, 24), limit=1,
                                timeout_seconds=3, cancel=Event()))
    assert stream.closed


def test_mit_standard_library_feed_stops_after_total_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.pilot.approved_sources.adapters_news.monotonic", lambda: clock[0])

    class _Response:
        headers: dict[str, str] = {}
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def read1(self, _):
            clock[0] = 4.0
            return b"<rss>"

    response = _Response()

    class _Opener:
        def open(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr("app.pilot.approved_sources.adapters_news.build_opener", lambda *_: _Opener())
    adapter = MitResearchNewsAdapter()
    try:
        with pytest.raises(SourceFetchError, match="timeout"):
            adapter._feed_bytes(deadline=3.0, cancel=Event())
    finally:
        adapter.close()
    assert response.closed
