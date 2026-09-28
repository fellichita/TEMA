from threading import Event

import httpx
import pytest

from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError, CancelledError
from app.backend.providers.epo import AUTH_ENDPOINT, SEARCH_ENDPOINT, EpoOpsProvider


def patent(number="1234567", kind="A1", day="20240229"):
    return f'''<exchange-document country="EP" doc-number="{number}" kind="{kind}" family-id="123">
      <bibliographic-data><publication-reference><document-id document-id-type="docdb">
      <country>EP</country><doc-number>{number}</doc-number><kind>{kind}</kind><date>{day}</date>
      </document-id></publication-reference>
      <application-reference><document-id><date>19990101</date></document-id></application-reference>
      <invention-title lang="de">Neuronale Systeme</invention-title>
      <invention-title lang="en">Neuromorphic systems</invention-title>
      <parties><inventors><inventor><inventor-name><name>Ada Example</name></inventor-name></inventor></inventors>
      <applicants><applicant><applicant-name><name>Example Corp</name></applicant-name></applicant></applicants></parties>
      </bibliographic-data><abstract lang="en"><p>Low energy computing.</p></abstract></exchange-document>'''


def envelope(items, total=None):
    total = len(items) if total is None else total
    return (f'<ops:world-patent-data xmlns:ops="http://ops.epo.org" xmlns="http://www.epo.org/exchange">'
            f'<ops:biblio-search total-result-count="{total}"><ops:search-result><exchange-documents>'
            + "".join(items) + '</exchange-documents></ops:search-result></ops:biblio-search></ops:world-patent-data>')


def provider(search_handler, **kwargs):
    def handler(request):
        if str(request.url) == AUTH_ENDPOINT:
            assert request.method == "POST"
            assert request.headers["Authorization"].startswith("Basic ")
            assert request.content == b"grant_type=client_credentials"
            return httpx.Response(200, json={"access_token": "TEST_ACCESS_TOKEN", "expires_in": "1199"})
        assert str(request.url).startswith(SEARCH_ENDPOINT)
        assert request.headers["Authorization"] == "Bearer TEST_ACCESS_TOKEN"
        assert "TEST_ACCESS_TOKEN" not in str(request.url)
        return search_handler(request)
    return EpoOpsProvider("TEST_KEY", "TEST_SECRET", httpx.Client(transport=httpx.MockTransport(handler)),
                          page_delay=0, retry_delay=0, **kwargs)


def collect(source, **kwargs):
    return list(source.iter_pages(SearchRequest(topic="neuromorphic computing", source="epo", **kwargs), Event()))


def test_oauth_search_and_publication_metadata():
    def search(request):
        assert request.headers["X-OPS-Range"] == "1-20"
        assert request.url.params["q"] == 'ta all "neuromorphic computing" and pd within "20240101 20241231"'
        return httpx.Response(200, text=envelope([patent()]))
    record = collect(provider(search), max_results=20, from_date="2024-01-01", until_date="2024-12-31")[0].documents[0]
    assert record.patent_publication == "EP1234567A1" and record.document_key == "patent:EP1234567A1"
    assert record.patent_family_id == "123" and record.doi is None
    assert record.title == "Neuromorphic systems" and record.authors == ("Ada Example",)
    assert record.publication_year == 2024 and record.publication_month == 2
    assert record.abstract == "Low energy computing." and "Example Corp" in record.raw_metadata["xml"]
    assert "TEST_" not in record.model_dump_json()


def test_missing_keys_fail_before_http():
    def forbidden(request):
        raise AssertionError("Network must not be used")
    source = EpoOpsProvider(client=httpx.Client(transport=httpx.MockTransport(forbidden)))
    with pytest.raises(BackendError) as error:
        collect(source)
    assert error.value.code == "credentials_required"


@pytest.mark.parametrize("key,secret", [("a\nb", "s"), ("a:b", "s"), ("k", "bad\rsecret"), ("k", " ")])
def test_invalid_credentials_fail_without_echo(key, secret):
    with pytest.raises(BackendError) as error:
        EpoOpsProvider(key, secret)
    assert error.value.code == "invalid_credentials"


def test_publication_kind_is_part_of_identity():
    records = collect(provider(lambda req: httpx.Response(200, text=envelope([patent(kind="A1"), patent(kind="B1")]))))[0].documents
    assert records[0].document_key != records[1].document_key


@pytest.mark.parametrize("bad_xml", ["invalid", "<error>secret server text</error>",
    '<!DOCTYPE x [<!ENTITY y SYSTEM "file:///etc/passwd">]><x>&y;</x>'])
def test_invalid_or_unsafe_xml_rejected(bad_xml):
    with pytest.raises(BackendError) as error:
        collect(provider(lambda req: httpx.Response(200, text=bad_xml)))
    assert error.value.code == "invalid_response" and "secret server" not in str(error.value)


def test_bad_publication_date_is_counted_as_skipped():
    page = collect(provider(lambda req: httpx.Response(200, text=envelope([patent(day="20240230")]))))[0]
    assert page.skipped == 1 and not page.documents


def test_404_is_not_faked_as_empty_result():
    with pytest.raises(BackendError) as error:
        collect(provider(lambda req: httpx.Response(404, text="SERVER.EntityNotFound")))
    assert error.value.code == "source_unavailable"


def test_zero_result_response_is_empty():
    page = collect(provider(lambda req: httpx.Response(200, text=envelope([]))))[0]
    assert page.exhausted and page.scanned == 0


def test_pagination_and_limit():
    ranges = []
    def search(request):
        ranges.append(request.headers["X-OPS-Range"])
        return httpx.Response(200, text=envelope([patent(number=str(len(ranges)))], total=5))
    pages = collect(provider(search, page_size=1), max_results=2)
    assert ranges == ["1-1", "2-2"] and not pages[-1].exhausted


def test_2000_record_cap_does_not_claim_complete_coverage():
    def search(request):
        start, end = map(int, request.headers["X-OPS-Range"].split("-"))
        return httpx.Response(200, text=envelope([patent(number=str(i)) for i in range(start, end + 1)], 5000))
    pages = collect(provider(search), max_results=2100)
    assert sum(page.scanned for page in pages) == 2000 and not pages[-1].exhausted


def test_cancel_before_authentication():
    event = Event()
    event.set()
    with pytest.raises(CancelledError):
        list(EpoOpsProvider().iter_pages(SearchRequest(topic="patents", source="epo"), event))


def test_expired_token_is_refreshed_once():
    calls = {"auth": 0, "search": 0}
    def handler(request):
        if str(request.url) == AUTH_ENDPOINT:
            calls["auth"] += 1
            return httpx.Response(200, json={"access_token": f"TOKEN{calls['auth']}", "expires_in": 1200})
        calls["search"] += 1
        if calls["search"] == 1:
            return httpx.Response(401)
        assert request.headers["Authorization"] == "Bearer TOKEN2"
        return httpx.Response(200, text=envelope([patent()]))
    source = EpoOpsProvider("key", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert collect(source)[0].scanned == 1
    assert calls == {"auth": 2, "search": 2}


def test_forbidden_access_stops_without_refreshing_or_leaking_token():
    calls = []
    def search(request):
        calls.append(request)
        return httpx.Response(403, text="TEST_ACCESS_TOKEN")
    with pytest.raises(BackendError) as error:
        collect(provider(search))
    assert error.value.code == "access_denied" and len(calls) == 1
    assert "TEST_ACCESS_TOKEN" not in str(error.value)


def test_adjacent_abstract_paragraphs_are_separated():
    xml = patent().replace("<p>Low energy computing.</p>", "<p>First.</p><p>Second.</p>")
    doc = collect(provider(lambda req: httpx.Response(200, text=envelope([xml]))))[0].documents[0]
    assert doc.abstract == "First. Second."
