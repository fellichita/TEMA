"""EPO OPS error protocol: classify fixed meanings without echoing responses."""

import gzip
from threading import Event

import httpx
import pytest

from app.backend.config import BackendSettings
from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.epo import AUTH_ENDPOINT, EpoOpsProvider
from app.backend.service import Backend

TOKEN = "PRIVATE_ACCESS_TOKEN"
KEY = "PRIVATE_CONSUMER_KEY"
SECRET = "PRIVATE_CONSUMER_SECRET"
REMOTE_DETAIL = "PRIVATE_REMOTE_ERROR_DESCRIPTION"
FAIR_USE = "This request has been rejected due to the violation of Fair Use policy"
EMPTY_RESULT = '<root><biblio-search total-result-count="0"/></root>'


def error_xml(message):
    return f'<error><message>{message}</message><description>{REMOTE_DETAIL}</description></error>'


def assert_safe(error):
    for marker in (TOKEN, KEY, SECRET, REMOTE_DETAIL):
        assert marker not in str(error)
    assert not hasattr(error, "response") and not hasattr(error, "request")


class OpsSession:
    def __init__(self, search_response, auth_response=None):
        self.search_response = search_response
        self.auth_response = auth_response
        self.calls = {"auth": 0, "search": 0}

    def __call__(self, request):
        if str(request.url) == AUTH_ENDPOINT:
            self.calls["auth"] += 1
            return (self.auth_response() if self.auth_response else
                    httpx.Response(200, json={"access_token": TOKEN, "expires_in": 1200}))
        self.calls["search"] += 1
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        return self.search_response(self.calls["search"])


def source(client):
    return EpoOpsProvider(KEY, SECRET, client=client, page_delay=0, retry_delay=0)


def collect(provider, cancel=None):
    return list(provider.iter_pages(SearchRequest(topic="patent audit", source="epo"), cancel or Event()))


@pytest.mark.parametrize("message,headers", [
    (FAIR_USE, {}),
    (FAIR_USE, {"X-Rejection-Reason": "Individual quota exceeded", "Retry-After": "60"}),
    ("", {"X-Rejection-Reason": "IndividualQuotaPerHour"}),
    ("", {"X-Rejection-Reason": "RegisteredQuotaPerWeek"}),
    ("", {"X-Rejection-Reason": "RegisteredPayingQuotaPerWeek"}),
])
def test_quota_does_not_refresh_a_valid_token_or_retry_the_search(message, headers):
    session = OpsSession(lambda count: httpx.Response(403, text=error_xml(message), headers=headers))
    with (httpx.Client(transport=httpx.MockTransport(session)) as client,
          pytest.raises(BackendError) as error):
        collect(source(client))
    assert error.value.code == "rate_limited"
    assert_safe(error.value)
    assert session.calls == {"auth": 1, "search": 1}


def test_user_retry_after_quota_reuses_the_valid_token():
    session = OpsSession(lambda count: (httpx.Response(403, text=error_xml(FAIR_USE)) if count == 1
                                       else httpx.Response(200, text=EMPTY_RESULT)))
    with httpx.Client(transport=httpx.MockTransport(session)) as client:
        provider = source(client)
        with pytest.raises(BackendError, match="квот"):
            collect(provider)
        assert collect(provider)[0].exhausted
    assert session.calls == {"auth": 1, "search": 2}


@pytest.mark.parametrize("status", [400, 401])
@pytest.mark.parametrize("representation", ["xml", "json"])
def test_invalid_client_identifies_credentials_without_search(status, representation):
    def response():
        if representation == "xml":
            return httpx.Response(status, text=error_xml("invalid_client"))
        return httpx.Response(status, json={"error": "invalid_client", "error_description": REMOTE_DETAIL})

    session = OpsSession(lambda count: pytest.fail("Search must not run"), response)
    with (httpx.Client(transport=httpx.MockTransport(session)) as client,
          pytest.raises(BackendError) as error):
        collect(source(client))
    assert error.value.code == "invalid_credentials"
    assert_safe(error.value)
    assert session.calls == {"auth": 1, "search": 0}


@pytest.mark.parametrize("status", [400, 401, 403])
@pytest.mark.parametrize("repeat_failure", [False, True])
def test_explicit_invalid_token_refreshes_once(status, repeat_failure):
    def search(count):
        return (httpx.Response(status, text=error_xml("invalid_access_token"))
                if count == 1 or repeat_failure else httpx.Response(200, text=EMPTY_RESULT))

    session = OpsSession(search)
    with httpx.Client(transport=httpx.MockTransport(session)) as client:
        if repeat_failure:
            with pytest.raises(BackendError) as error:
                collect(source(client))
            assert error.value.code == "authentication_required"
            assert_safe(error.value)
        else:
            assert collect(source(client))[0].exhausted
    assert session.calls == {"auth": 2, "search": 2}


@pytest.mark.parametrize("body", [
    error_xml("This request has been rejected"), error_xml("Developer account is blocked"),
    error_xml(REMOTE_DETAIL), REMOTE_DETAIL,
    ('<!DOCTYPE error [<!ENTITY secret SYSTEM "file:///private/secret">]>'
     '<error><message>invalid_access_token</message></error>'),
])
def test_denied_unknown_or_unsafe_403_never_retries_credentials(body):
    session = OpsSession(lambda count: httpx.Response(403, text=body))
    with (httpx.Client(transport=httpx.MockTransport(session)) as client,
          pytest.raises(BackendError) as error):
        collect(source(client))
    assert error.value.code == "access_denied"
    assert_safe(error.value)
    assert session.calls == {"auth": 1, "search": 1}


@pytest.mark.parametrize("message,code", [
    ("invalid_client", "invalid_credentials"), ("invalid_request", "invalid_response"),
    ("unsupported_grant_type", "invalid_response"), ("Developer account is blocked", "access_denied"),
])
def test_known_non_token_error_overrides_generic_search_401(message, code):
    session = OpsSession(lambda count: httpx.Response(401, text=error_xml(message)))
    with (httpx.Client(transport=httpx.MockTransport(session)) as client,
          pytest.raises(BackendError) as error):
        collect(source(client))
    assert error.value.code == code
    assert_safe(error.value)
    assert session.calls == {"auth": 1, "search": 1}


class RawBody(httpx.SyncByteStream):
    def __init__(self, body):
        self.body = body

    def __iter__(self):
        yield self.body


@pytest.mark.parametrize("compressed", [False, True])
def test_error_body_and_decompressed_size_are_bounded(compressed):
    body = b"x" * 70_000
    headers = {"Content-Encoding": "gzip"} if compressed else {}
    payload = gzip.compress(body) if compressed else body
    session = OpsSession(lambda count: httpx.Response(403, headers=headers, stream=RawBody(payload)))
    with (httpx.Client(transport=httpx.MockTransport(session)) as client,
          pytest.raises(BackendError) as error):
        collect(source(client))
    assert error.value.code == "response_too_large"
    assert session.calls == {"auth": 1, "search": 1}


def test_cancelled_rejection_does_not_start_token_refresh():
    cancel = Event()

    def search(count):
        cancel.set()
        return httpx.Response(401, text=error_xml("invalid_access_token"))

    session = OpsSession(search)
    with httpx.Client(transport=httpx.MockTransport(session)) as client, pytest.raises(CancelledError):
        collect(source(client), cancel)
    assert session.calls == {"auth": 1, "search": 1}


def test_backend_persists_only_safe_classified_quota_error(tmp_path):
    session = OpsSession(lambda count: httpx.Response(403, text=error_xml(FAIR_USE)))
    with (httpx.Client(transport=httpx.MockTransport(session)) as client,
          Backend(BackendSettings(data_dir=tmp_path), lambda: source(client)) as backend):
        job = backend.collect(SearchRequest(topic="patent audit", source="epo"))
        assert job.state == "failed" and job.error_code == "rate_limited"
        for marker in (TOKEN, KEY, SECRET, REMOTE_DETAIL):
            assert marker not in job.model_dump_json()
        assert backend.list_documents(job_id=job.id).total == 0
    assert session.calls == {"auth": 1, "search": 1}
