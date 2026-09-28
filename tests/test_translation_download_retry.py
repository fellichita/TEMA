"""A broken model transfer must resume safely and remain pinned to its digest."""

import hashlib

import httpx
import pytest

from app.pilot.translator import TranslationError
from scripts import install_translation_model as installer


PAYLOAD = b"abcdefghij"
SPEC = {"model_id": "example/model", "revision": "fixed-revision"}
ITEM = {"name": "weights.bin", "remote": "weights.bin", "bytes": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest()}


class InterruptedStream(httpx.SyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks
        raise httpx.RemoteProtocolError("peer closed during response body")


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    # HTTPX coalesces stream chunks up to chunk_size before yielding them.
    monkeypatch.setattr(installer, "CHUNK_BYTES", 2)
    monkeypatch.setattr(installer.time, "sleep", lambda _seconds: None)


def _partial_response():
    return httpx.Response(200, headers={"content-length": str(len(PAYLOAD))},
                          stream=InterruptedStream(PAYLOAD[:4]))


def test_interrupted_body_resumes_from_received_byte_and_verifies_file(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return _partial_response()
        return httpx.Response(206, headers={"content-range": "bytes 4-9/10",
                                             "content-length": "6"},
                              stream=httpx.ByteStream(PAYLOAD[4:]))

    target = tmp_path / "weights.bin"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        installer._download(client, SPEC, ITEM, target)

    assert [request.headers.get("range") for request in requests] == [None, "bytes=4-"]
    assert all(str(request.url).endswith("/example/model/resolve/fixed-revision/weights.bin")
               for request in requests)
    assert target.read_bytes() == PAYLOAD


def test_full_response_to_range_request_restarts_without_duplicating_prefix(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return _partial_response()
        return httpx.Response(200, headers={"content-length": str(len(PAYLOAD))},
                              stream=httpx.ByteStream(PAYLOAD))

    target = tmp_path / "weights.bin"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        installer._download(client, SPEC, ITEM, target)

    assert [request.headers.get("range") for request in requests] == [None, "bytes=4-"]
    assert target.read_bytes() == PAYLOAD


@pytest.mark.parametrize("suffix", [b"xxxxxx", b"efg"])
def test_resumed_file_still_requires_full_digest_and_size(tmp_path, suffix):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return _partial_response()
        return httpx.Response(206, headers={"content-range": "bytes 4-9/10"},
                              stream=httpx.ByteStream(suffix))

    target = tmp_path / "weights.bin"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(TranslationError):
            installer._download(client, SPEC, ITEM, target)

    assert requests[1].headers.get("range") == "bytes=4-"
    assert target.read_bytes() != PAYLOAD


@pytest.mark.parametrize("content_range", [
    "bytes 0-5/10",  # The response overlaps bytes already on disk.
    "bytes 4-9/11",  # The total disagrees with the pinned size.
    "bytes four-nine/10",  # A malformed range must not be guessed.
])
def test_invalid_content_range_is_rejected(tmp_path, content_range):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return _partial_response()
        return httpx.Response(206, headers={"content-range": content_range},
                              stream=httpx.ByteStream(PAYLOAD[4:]))

    target = tmp_path / "weights.bin"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(TranslationError):
            installer._download(client, SPEC, ITEM, target)

    assert requests[1].headers.get("range") == "bytes=4-"
    assert not target.exists() or target.read_bytes() != PAYLOAD


def test_repeated_transport_failures_have_a_finite_retry_limit(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) > 8:
            pytest.fail("download did not stop after repeated transport failures")
        if len(requests) == 1:
            return _partial_response()
        return httpx.Response(206, headers={"content-range": "bytes 4-9/10",
                                             "content-length": "6"},
                              stream=InterruptedStream())

    target = tmp_path / "weights.bin"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises((httpx.HTTPError, TranslationError)):
            installer._download(client, SPEC, ITEM, target)

    assert 2 <= len(requests) <= 8
    assert all(request.headers.get("range") == "bytes=4-" for request in requests[1:])
    assert not target.exists() or target.read_bytes() != PAYLOAD
