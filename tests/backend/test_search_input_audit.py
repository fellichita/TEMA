"""Audit A12: a malformed search must never widen a real SQLite query."""

import pytest

from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError
from app.backend.history import HistoryRequest
from app.backend.service import Backend


class SearchFixtureProvider:
    def iter_pages(self, request, cancel):
        titles = ("ВТОРОЙ ДОКУМЕНТ", "Скидка 100%_test \\ literal")
        if request.topic == "outside scope":
            titles = ("Outside collection",)
        documents = tuple(
            DocumentRecord(
                source=request.source,
                source_id=title,
                title=title,
                url="https://example.org/document",
            )
            for title in titles
        )
        yield SourcePage(
            documents=documents,
            scanned=len(documents),
            total_available=len(documents),
            exhausted=True,
        )

    def close(self):
        pass


@pytest.fixture
def stored_search(tmp_path):
    settings = BackendSettings(data_dir=tmp_path, history_period_delay_seconds=0)
    with Backend(settings, SearchFixtureProvider) as backend:
        report = backend.collect_history(HistoryRequest(
            topic="search audit",
            from_date="2024-01-01",
            until_date="2024-12-31",
            period="year",
            sources=("crossref",),
        ))
        assert report.coverage_complete
        job_id = report.periods[0].job.id
        backend.collect(SearchRequest(topic="outside scope"))
        yield backend, {"library": {}, "job": {"job_id": job_id},
                        "history": {"history_id": report.id}}


@pytest.mark.parametrize("scope", ["library", "job", "history"])
@pytest.mark.parametrize("query", ["\x00", "\x00UNTRUSTED", "UN\x00TRUSTED", "UNTRUSTED\x00"])
def test_nul_query_is_rejected_without_widening_scope(stored_search, scope, query):
    backend, scopes = stored_search
    with pytest.raises(BackendError) as error:
        backend.list_documents(query=query, **scopes[scope])
    assert error.value.code == "invalid_query"
    assert "UNTRUSTED" not in str(error.value) and "\x00" not in str(error.value)
    # Rejection leaves the store and each selection usable.
    assert backend.list_documents(**scopes[scope]).total == (3 if scope == "library" else 2)


@pytest.mark.parametrize("scope", ["library", "job", "history"])
@pytest.mark.parametrize("query, matches", [
    ("второй", 1), ("%_", 1), ("\\", 1), ("%' OR 1=1 --", 0), ("missing", 0),
])
def test_valid_search_keeps_unicode_and_literal_semantics(stored_search, scope, query, matches):
    backend, scopes = stored_search
    assert backend.list_documents(query=query, **scopes[scope]).total == matches


@pytest.mark.parametrize("scope", ["library", "job", "history"])
@pytest.mark.parametrize("query", [None, "", "\n"])
def test_existing_empty_and_newline_searches_remain_supported(stored_search, scope, query):
    backend, scopes = stored_search
    assert backend.list_documents(query=query, **scopes[scope]).total == (3 if scope == "library" else 2)
