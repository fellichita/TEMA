"""Shared bounded HTTP transport for fixed, trusted provider endpoints.

Provider adapters own response schemas. This module owns network boundaries,
decompression limits, cancellation and safe errors; it never logs requests,
responses, credentials or query strings.
"""

import json
import math
import time
import zlib
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Event, Lock
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

import httpx

from app.backend.errors import BackendError, CancelledError

# A refusal for rate or budget describes the next minutes, not one request.
# Measured on 2026-09-24: once OpenAlex's anonymous daily budget ran out, every
# history and antecedent query of an analysis waited up to three Retry-After
# pauses of about 36 seconds and failed anyway — some two minutes per candidate
# without a single document. The operation that was refused (known by its
# cancel event: one per analysis run or collection job) now gets the same answer
# at once until the pause the source named has passed. Other operations, and
# requests carrying credentials with their own budget, are not affected.
_REFUSALS: WeakKeyDictionary[Event, dict[tuple, float]] = WeakKeyDictionary()
_REFUSALS_LOCK = Lock()
_REFUSAL_CEILING = 3600.0
# A refusal that outlasted the retries is a throttle measured in minutes (the
# OpenAlex search cluster "under elevated load"), so the rest of a typical run
# does not ask again.
_EXHAUSTED_RETRY_PAUSE = 600.0


def _refused(cancel: Event, key: tuple) -> bool:
    with _REFUSALS_LOCK:
        until = _REFUSALS.get(cancel, {}).get(key)
    return until is not None and time.monotonic() < until


def _remember_refusal(cancel: Event, key: tuple, pause: float) -> None:
    with _REFUSALS_LOCK:
        _REFUSALS.setdefault(cancel, {})[key] = time.monotonic() + min(pause, _REFUSAL_CEILING)


_MESSAGES = {
    "source_unavailable": "Источник данных временно недоступен. Повторите позже.",
    "rate_limited": "Источник ограничил частоту запросов. Повторите позже.",
    "invalid_response": "Источник вернул некорректные данные.",
    "response_too_large": "Ответ источника превышает допустимый размер.",
    "authentication_required": "Источник требует действующие учётные данные.",
}


def _error(code: str) -> BackendError:
    return BackendError(code, _MESSAGES[code])


def _cancelled(cancel: Event) -> None:
    if cancel.is_set():
        raise CancelledError()


def _positive_number(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: требуется конечное число")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name}: недопустимое значение")
    return float(value)


def _validate_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or any(ord(char) <= 32 or ord(char) == 127 for char in endpoint):
        raise ValueError("endpoint: требуется фиксированный HTTPS-адрес без учётных данных")
    try:
        parts = urlsplit(endpoint)
        valid = (
            parts.scheme == "https"
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
            and "\\" not in endpoint
        )
        # Accessing .port also rejects malformed port numbers without echoing URLs.
        _ = parts.port
    except (ValueError, TypeError):
        raise ValueError("endpoint: некорректный HTTPS-адрес") from None
    if not valid:
        raise ValueError("endpoint: требуется фиксированный HTTPS-адрес без учётных данных")
    return endpoint


def _reject_non_json_constant(value: str) -> None:
    raise ValueError("Недопустимая JSON-константа")


class BoundedHttpTransport:
    def __init__(
        self,
        endpoint: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        max_response_bytes: int = 5_000_000,
        retry_delay: float = 1.0,
        page_deadline_seconds: float = 60.0,
        max_retry_delay: float = 30.0,
        *,
        client_error_handler: Callable[[int, httpx.Headers, bytes], BackendError | None] | None = None,
        unavailable_pause: float | None = None,
    ) -> None:
        self._endpoint = _validate_endpoint(endpoint)
        # Сколько операция не спрашивает источник, чей 5xx пережил все повторы.
        # Только для источников, у которых такой отказ означает сбой на минуты,
        # а не просьбу уменьшить страницу (как 503 у Crossref).
        self.unavailable_pause = (None if unavailable_pause is None
                                  else _positive_number("unavailable_pause", unavailable_pause))
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise ValueError("max_retries должен быть целым числом от 0 до 10")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes должен быть положительным целым числом")
        self.timeout_seconds = _positive_number("timeout_seconds", timeout_seconds)
        self.max_retries = max_retries
        self.max_response_bytes = max_response_bytes
        self.retry_delay = _positive_number("retry_delay", retry_delay, allow_zero=True)
        self.page_deadline_seconds = _positive_number("page_deadline_seconds", page_deadline_seconds)
        # How long a source may ask us to wait before we treat its refusal as
        # final. A source that names a longer pause is telling us to come back
        # later, not to keep the connection waiting.
        self.max_retry_delay = _positive_number("max_retry_delay", max_retry_delay)
        if self.max_retry_delay > 120:
            raise ValueError("Ожидание по просьбе источника не должно превышать 120 секунд")
        self._client_error_handler = client_error_handler
        if self.retry_delay > 30:
            raise ValueError("Пауза между запросами не должна превышать 30 секунд")
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(
            follow_redirects=False,
            timeout=self.timeout_seconds,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def close(self) -> None:
        """Only close owned clients; injected clients stay caller-owned."""
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _check_deadline(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _error("source_unavailable")
        return remaining

    def _wait_retry(self, cancel: Event, delay: float, deadline: float) -> None:
        remaining = self._check_deadline(deadline)
        if cancel.wait(min(delay, remaining)):
            raise CancelledError()
        self._check_deadline(deadline)

    @staticmethod
    def _requested_pause(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            try:
                timestamp = parsedate_to_datetime(value)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return (timestamp - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None

    def _retry_after(self, value: str | None, attempt: int, code: str) -> float:
        delay = min(30.0, self.retry_delay * (2 ** attempt))
        requested = self._requested_pause(value)
        if requested is not None:
            if not math.isfinite(requested) or requested > self.max_retry_delay:
                raise _error(code)
            delay = max(0.0, requested)
        return delay

    def _read_body(self, response: httpx.Response, cancel: Event, deadline: float,
                   *, max_bytes: int | None = None) -> bytes:
        limit = min(self.max_response_bytes, max_bytes) if max_bytes is not None else self.max_response_bytes
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdecimal():
            if len(content_length) > 20 or int(content_length) > limit:
                raise _error("response_too_large")
        # Mock/injected clients may supply already-decoded content. Real responses
        # take the raw branch, checking the deadline on every network chunk.
        if response.is_stream_consumed:
            if len(response.content) > limit:
                raise _error("response_too_large")
            return response.content
        encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding not in {"", "identity", "gzip", "deflate"}:
            raise _error("invalid_response")
        decoder = None
        if encoding in {"gzip", "deflate"}:
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
        body = bytearray()
        received = 0
        for raw in response.iter_raw():
            _cancelled(cancel)
            self._check_deadline(deadline)
            received += len(raw)
            if received > limit:
                raise _error("response_too_large")
            if decoder is None:
                chunk = raw
            else:
                try:
                    # Bound expansion before allocating decompressed output, even
                    # when one compressed network chunk contains a decompression bomb.
                    chunk = decoder.decompress(raw, limit - len(body) + 1)
                except zlib.error:
                    raise _error("invalid_response") from None
                if decoder.unconsumed_tail:
                    raise _error("response_too_large")
                if decoder.unused_data:
                    raise _error("invalid_response")
            if len(body) + len(chunk) > limit:
                raise _error("response_too_large")
            body.extend(chunk)
        if decoder is not None and not decoder.eof:
            raise _error("invalid_response")
        return bytes(body)

    def request_bytes(
        self,
        cancel: Event,
        *,
        method: str = "GET",
        params: dict | None = None,
        headers: dict | None = None,
        data: dict | None = None,
    ) -> bytes:
        """GET source data or POST an authentication form to this fixed endpoint.

        POST retries are intended for token retrieval, not arbitrary mutations.
        Providers keep credentials in memory and supply explicit auth headers.
        """
        if not isinstance(method, str) or method.upper() not in {"GET", "POST"}:
            raise ValueError("method: поддерживаются только GET и POST")
        request_headers = httpx.Headers({
            "Accept": "application/json",
            "User-Agent": "Trendanalyser/0.1 (local research application)",
        })
        if headers:
            request_headers.update(headers)
        # The bounded decoder intentionally supports only these representations.
        request_headers["Accept-Encoding"] = "gzip, deflate"
        refusal_key = (self._endpoint, "Authorization" in request_headers)
        outage_key = (*refusal_key, "unavailable")
        if _refused(cancel, refusal_key):
            raise _error("rate_limited")
        if _refused(cancel, outage_key):
            raise _error("source_unavailable")
        deadline = time.monotonic() + self.page_deadline_seconds
        for attempt in range(self.max_retries + 1):
            _cancelled(cancel)
            timeout = min(self.timeout_seconds, self._check_deadline(deadline))
            try:
                with self._client.stream(
                    method.upper(),
                    self._endpoint,
                    params=params,
                    headers=request_headers,
                    data=data,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    _cancelled(cancel)
                    self._check_deadline(deadline)
                    if (self._client_error_handler is not None
                            and 400 <= response.status_code < 500 and response.status_code != 429):
                        # The adapter sees bounded response data, never a request
                        # object. Only its fixed, safe public error may escape.
                        body = self._read_body(response, cancel, deadline, max_bytes=64_000)
                        _cancelled(cancel)
                        self._check_deadline(deadline)
                        error = self._client_error_handler(response.status_code, response.headers, body)
                        if error is not None:
                            raise error
                    if response.status_code in {401, 403}:
                        raise _error("authentication_required")
                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        code = "rate_limited" if response.status_code == 429 else "source_unavailable"
                        if code == "rate_limited":
                            pause = self._requested_pause(response.headers.get("Retry-After"))
                            if pause is not None and not math.isfinite(pause):
                                pause = _REFUSAL_CEILING
                            if pause is not None and pause > self.max_retry_delay:
                                _remember_refusal(cancel, refusal_key, pause)
                            elif attempt == self.max_retries:
                                _remember_refusal(cancel, refusal_key, max(pause or 0.0, _EXHAUSTED_RETRY_PAUSE))
                        elif attempt == self.max_retries and self.unavailable_pause is not None:
                            _remember_refusal(cancel, outage_key, self.unavailable_pause)
                        if attempt == self.max_retries:
                            raise _error(code)
                        delay = self._retry_after(response.headers.get("Retry-After"), attempt, code)
                    elif not 200 <= response.status_code < 300:
                        # Never follow redirects, even when an injected client would.
                        raise _error("source_unavailable")
                    else:
                        body = self._read_body(response, cancel, deadline)
                        _cancelled(cancel)
                        self._check_deadline(deadline)
                        return body
            except httpx.DecodingError:
                raise _error("invalid_response") from None
            except httpx.RequestError:
                _cancelled(cancel)
                if attempt == self.max_retries:
                    raise _error("source_unavailable") from None
                delay = min(30.0, self.retry_delay * (2 ** attempt))
            # Close the previous response before entering a cancellable backoff.
            self._wait_retry(cancel, delay, deadline)
        raise _error("source_unavailable")

    def get_json(self, params: dict, cancel: Event, headers: dict | None = None) -> dict:
        """Return an unmodified JSON object; provider-specific validation is separate."""
        body = self.request_bytes(cancel, params=params, headers=headers)
        _cancelled(cancel)
        try:
            payload = json.loads(body, parse_constant=_reject_non_json_constant)
        except (ValueError, UnicodeError, RecursionError):
            raise _error("invalid_response") from None
        if not isinstance(payload, dict):
            raise _error("invalid_response")
        _cancelled(cancel)
        return payload
