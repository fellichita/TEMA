"""Read untrusted source responses without unbounded decompression or waits."""

from __future__ import annotations

import zlib
from threading import Event
from time import monotonic

import httpx

from app.pilot.approved_sources.contracts import SourceFetchError
from app.runtime.jobs import TaskCancelled


def read_bounded_response(response: httpx.Response, *, max_bytes: int,
                          deadline: float, cancel: Event) -> bytes:
    """Bound both wire and decoded bytes, including a single compressed chunk."""
    # A valid gzip/deflate response can be slightly larger than its decoded
    # body when the content is incompressible. Keep that transport overhead
    # without relaxing the existing decoded payload limit.
    max_wire_bytes = max_bytes + 65_536
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdecimal():
        if len(content_length) > 20 or int(content_length) > max_wire_bytes:
            raise SourceFetchError("response_too_large")
    if cancel.is_set():
        raise TaskCancelled()
    if monotonic() >= deadline:
        raise SourceFetchError("timeout")

    # An injected test client can hand us an already decoded response. Real
    # network streams go through iter_raw so httpx never expands gzip first.
    if response.is_stream_consumed:
        if len(response.content) > max_bytes:
            raise SourceFetchError("response_too_large")
        return response.content

    encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding not in {"", "identity", "gzip", "deflate"}:
        raise SourceFetchError("invalid_response")
    decoder = (zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
               if encoding in {"gzip", "deflate"} else None)
    allow_raw_deflate = encoding == "deflate"
    body = bytearray()
    wire_bytes = 0
    for raw in response.iter_raw():
        if cancel.is_set():
            raise TaskCancelled()
        if monotonic() >= deadline:
            raise SourceFetchError("timeout")
        wire_bytes += len(raw)
        if wire_bytes > max_wire_bytes:
            raise SourceFetchError("response_too_large")
        if decoder is None:
            chunk = raw
        else:
            try:
                chunk = decoder.decompress(raw, max_bytes - len(body) + 1)
            except zlib.error:
                if not allow_raw_deflate:
                    raise SourceFetchError("invalid_response") from None
                # Some servers label raw DEFLATE as "deflate". Match httpx's
                # first-chunk fallback while retaining the decoded-size bound.
                decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                try:
                    chunk = decoder.decompress(raw, max_bytes - len(body) + 1)
                except zlib.error:
                    raise SourceFetchError("invalid_response") from None
            allow_raw_deflate = False
            if decoder.unconsumed_tail:
                raise SourceFetchError("response_too_large")
            if decoder.unused_data:
                raise SourceFetchError("invalid_response")
        if len(body) + len(chunk) > max_bytes:
            raise SourceFetchError("response_too_large")
        body.extend(chunk)
    if cancel.is_set():
        raise TaskCancelled()
    if monotonic() >= deadline:
        raise SourceFetchError("timeout")
    if decoder is not None:
        if not decoder.eof:
            raise SourceFetchError("invalid_response")
        try:
            tail = decoder.flush(max_bytes - len(body) + 1)
        except zlib.error:
            raise SourceFetchError("invalid_response") from None
        if len(body) + len(tail) > max_bytes:
            raise SourceFetchError("response_too_large")
        body.extend(tail)
    return bytes(body)
