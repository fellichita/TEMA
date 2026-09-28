"""Backend admission, ML sources and browser opening share one URL boundary."""

import webbrowser
from unittest.mock import Mock

import pytest

from app.backend.contracts import DocumentRecord
from app.input_safety import is_safe_http_url
from app.ml.text import safe_url
from app.ui.controller import Controller
from tests.mvp_fixture import records

INVALID_URLS = [
    None, 42, "", "javascript:alert(1)", "file:///tmp/study", "//example.org/study",
    "https://example.org:invalid/study", "https://example.org:999999/study",
    "https://example.org:-1/study", "https://example.org:/study",
    "https://bad host/study", "https://example.org/a b", "https://example.org/\tstudy",
    "\nhttps://example.org/study", "https://example.org/\x00study", "https://example.org/\x7fstudy",
    "https://user:pass@example.org/study", "https://@example.org/study",
    "https://user@example.org/study", "https://example.org\\@attacker.example/study",
    "https://example.org../study", "https://example..org/study", "https://-example.org/study",
    "https://example_.org/study", "https://%65xample.org/study", "https://[invalid]/study",
    "https://[::1/study", "https://[::1%25en0]/study", "https://999.999.999.999/study",
    "https://127.0.0.999/study", "https://" + "a" * 64 + ".org/study",
    "https://example.org/\ud800", "https://example.org/study?query=\udfff",
    "https://example.org/" + "a" * 4096,
]
VALID_URLS = [
    "https://doi.org/10.1021/acssynbio.5c00175.s001",
    "http://example.org:8080/study?part=a%20b#section", "https://example.org./study",
    "https://doi.org/10.1000/test%28part%29", "https://пример.рф/статья",
    "https://xn--e1afmkfd.xn--p1ai/study", "http://localhost:8000/study",
    "http://127.0.0.1/study", "http://192.168.1.2:80/study",
    "https://[2001:db8::1]:443/study", "http://[::1]:8080/study",
    "HTTPS://EXAMPLE.ORG/study", "https://example.org:65535/study",
]


@pytest.mark.parametrize("url", INVALID_URLS)
def test_invalid_url_is_rejected_before_browser_open(monkeypatch, url):
    assert not is_safe_http_url(url)
    assert not safe_url(url)
    document = next(records(2020)).model_dump()
    with pytest.raises(ValueError):
        DocumentRecord.model_validate({**document, "url": url})
    browser = Mock(return_value=True)
    monkeypatch.setattr(webbrowser, "open", browser)
    with pytest.raises(ValueError, match="ссылка"):
        Controller.__new__(Controller)._invoke("open_url", (url,), {})
    browser.assert_not_called()


@pytest.mark.parametrize("url", VALID_URLS)
def test_valid_url_is_preserved_across_all_boundaries(monkeypatch, url):
    assert is_safe_http_url(url)
    assert safe_url(url)
    document = next(records(2020)).model_dump()
    assert DocumentRecord.model_validate({**document, "url": url}).url == url
    browser = Mock(return_value=True)
    monkeypatch.setattr(webbrowser, "open", browser)
    Controller.__new__(Controller)._invoke("open_url", (url,), {})
    browser.assert_called_once_with(url)
