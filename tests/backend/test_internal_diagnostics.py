"""Internal diagnostics preserve job lifecycle and never include exception data."""

import logging

import pytest

from app import diagnostics
from app.backend.config import BackendSettings
from app.backend.contracts import SearchRequest, SourcePage
from app.backend.errors import BackendError, CancelledError
from app.backend.history import HistoryRequest
from app.backend.service import Backend

PRIVATE = "PRIVATE_TOKEN_QUERY_AND_PATH"


class Provider:
    def iter_pages(self, request, cancel):
        yield SourcePage(scanned=0, exhausted=True, total_available=0)

    def close(self):
        pass


def unexpected():
    try:
        raise ValueError(PRIVATE + "/cause")
    except ValueError as cause:
        raise RuntimeError(PRIVATE + "/message") from cause


class FailingProvider(Provider):
    def iter_pages(self, request, cancel):
        unexpected()


def assert_diagnostic(caplog, event, path):
    records = [record for record in caplog.records if record.name == "app.diagnostics"]
    assert len(records) == 1
    record = records[0]
    assert f"event={event}" in record.getMessage()
    assert "exception=RuntimeError" in record.getMessage()
    assert "app/backend/" in record.getMessage() and "stack=" in record.getMessage()
    assert PRIVATE not in caplog.text and str(path) not in caplog.text
    assert record.exc_info is None and record.stack_info is None


def test_unexpected_collection_failure_keeps_safe_persisted_job(tmp_path, caplog):
    with Backend(BackendSettings(data_dir=tmp_path), FailingProvider) as backend:
        job = backend.collect(SearchRequest(topic="diagnostics audit"))
        assert job.state == "failed" and job.error_code == "internal_error"
        assert PRIVATE not in job.model_dump_json()
        assert backend.get_job(job.id) == job
    assert_diagnostic(caplog, "backend.collection_failed", tmp_path)


def test_unexpected_history_failure_keeps_safe_report(tmp_path, caplog, monkeypatch):
    with Backend(BackendSettings(data_dir=tmp_path), Provider) as backend:
        monkeypatch.setattr(backend.history, "begin_attempt", lambda *args: unexpected())
        report = backend.collect_history(HistoryRequest(
            topic="diagnostics audit", from_date="2024-01-01", until_date="2024-12-31",
            period="year", sources=("crossref",),
        ))
        assert report.state == "failed" and report.error_code == "internal_error"
        assert PRIVATE not in report.model_dump_json()
        assert backend.get_history(report.id) == report
    assert_diagnostic(caplog, "backend.history_failed", tmp_path)


def test_provider_close_failure_does_not_replace_success(tmp_path, caplog):
    class CloseFailure(Provider):
        def close(self):
            unexpected()

    with Backend(BackendSettings(data_dir=tmp_path), CloseFailure) as backend:
        job = backend.collect(SearchRequest(topic="diagnostics audit"))
        assert job.state == "succeeded" and job.coverage_complete
        assert job.error_code is None
    assert_diagnostic(caplog, "backend.provider_close_failed", tmp_path)


@pytest.mark.parametrize("kind", ["public", "cancelled", "validation"])
def test_expected_collection_errors_do_not_become_crash_logs(tmp_path, caplog, kind):
    class ExpectedFailure(Provider):
        def iter_pages(self, request, cancel):
            if kind == "public":
                raise BackendError("rate_limited", "Источник ограничил доступ.")
            if kind == "cancelled":
                raise CancelledError()
            SearchRequest(topic=PRIVATE, max_results=0)

    with Backend(BackendSettings(data_dir=tmp_path), ExpectedFailure) as backend:
        job = backend.collect(SearchRequest(topic="diagnostics audit"))
        assert job.state == ("cancelled" if kind == "cancelled" else "failed")
    assert not [record for record in caplog.records if record.name == "app.diagnostics"]


def test_broken_logging_handler_cannot_break_collection_completion(tmp_path, monkeypatch):
    calls = []

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            calls.append(record)
            raise OSError(PRIVATE)

    monkeypatch.setattr(diagnostics._LOGGER, "handlers", [BrokenHandler()])
    with Backend(BackendSettings(data_dir=tmp_path), FailingProvider) as backend:
        job = backend.collect(SearchRequest(topic="diagnostics audit"))
        assert job.state == "failed" and job.error_code == "internal_error"
        assert backend.get_job(job.id) == job
    assert len(calls) == 1
