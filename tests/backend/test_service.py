from threading import Event

import pytest

from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError
from app.backend.repository import Repository
from app.backend.service import Backend


def page(exhausted=True):
    return SourcePage(documents=(DocumentRecord(
        source="crossref", source_id="10.1234/test", doi="10.1234/test",
        title="Тестовая публикация", url="https://doi.org/10.1234/test",
    ),), scanned=1, total_available=1, exhausted=exhausted)


class StubProvider:
    def __init__(self, failure=None):
        self.failure = failure
        self.closed = False

    def iter_pages(self, request, cancel):
        yield page(exhausted=self.failure is None)
        if self.failure:
            raise self.failure

    def close(self):
        self.closed = True


def test_collection_and_reopen_work_without_network(tmp_path):
    settings = BackendSettings(data_dir=tmp_path)
    provider = StubProvider()
    with Backend(settings, lambda: provider) as backend:
        job = backend.collect(SearchRequest(topic="ИИ"))
        assert job.state == "succeeded" and job.stored == 1
        assert job.coverage_complete
        job_id = job.id
        assert backend.list_documents(job_id=job_id).total == 1
    assert provider.closed
    with Backend(settings) as backend:
        assert backend.get_job(job_id).state == "succeeded"
        assert backend.list_documents(query="ПУБЛИКАЦИЯ").total == 1


def test_partial_failure_keeps_already_saved_documents(tmp_path):
    with Backend(BackendSettings(data_dir=tmp_path),
                 lambda: StubProvider(BackendError("source_unavailable", "Источник недоступен."))) as backend:
        job = backend.collect(SearchRequest(topic="ИИ"))
        assert job.state == "failed" and job.stored == 1
        assert job.error_code == "source_unavailable"
        assert not job.coverage_complete
        assert backend.list_documents(job_id=job.id).total == 1


def test_internal_exception_does_not_expose_sensitive_details(tmp_path):
    with Backend(BackendSettings(data_dir=tmp_path),
                 lambda: StubProvider(RuntimeError("token=TOPSECRET"))) as backend:
        job = backend.collect(SearchRequest(topic="ИИ"))
        assert job.error_code == "internal_error"
        assert "TOPSECRET" not in job.model_dump_json()


def test_cancel_running_job(tmp_path):
    entered = Event()

    class BlockingProvider(StubProvider):
        def iter_pages(self, request, cancel):
            yield page(exhausted=False)
            entered.set()
            if not cancel.wait(3):
                raise AssertionError("Отмена не доставлена")

    with Backend(BackendSettings(data_dir=tmp_path), BlockingProvider) as backend:
        job_id = backend.submit_collection(SearchRequest(topic="ИИ"))
        assert entered.wait(2)
        assert backend.cancel(job_id)
        result = backend.wait(job_id, timeout=3)
        assert result.state == "cancelled" and result.stored == 1
        assert not backend.cancel(job_id)


def test_instance_lock_prevents_false_recovery_of_live_jobs(tmp_path):
    settings = BackendSettings(data_dir=tmp_path)
    with Backend(settings, StubProvider):
        with pytest.raises(BackendError, match="уже используется"):
            Backend(settings, StubProvider)
    with Backend(settings, StubProvider) as reopened:
        assert reopened.list_jobs() == ()


def test_recover_abandoned_job_on_startup(tmp_path):
    settings = BackendSettings(data_dir=tmp_path)
    repository = Repository(settings.database_path)
    job = repository.create_job(SearchRequest(topic="ИИ"))
    repository.start_job(job.id)
    with Backend(settings, StubProvider) as backend:
        assert backend.get_job(job.id).state == "interrupted"


def test_queue_is_bounded_and_close_cancels_queued_jobs(tmp_path):
    entered = Event()

    class BlockingProvider(StubProvider):
        def iter_pages(self, request, cancel):
            entered.set()
            assert cancel.wait(3)
            yield SourcePage(scanned=0, exhausted=True)

    settings = BackendSettings(data_dir=tmp_path, max_pending_jobs=2)
    with Backend(settings, BlockingProvider) as backend:
        first = backend.submit_collection(SearchRequest(topic="ИИ"))
        assert entered.wait(2)
        second = backend.submit_collection(SearchRequest(topic="ИИ"))
        with pytest.raises(BackendError) as error:
            backend.submit_collection(SearchRequest(topic="ИИ"))
        assert error.value.code == "queue_full"
    with Backend(settings, StubProvider) as reopened:
        assert reopened.get_job(first).state == "cancelled"
        assert reopened.get_job(second).state == "cancelled"


def test_provider_cannot_exceed_request_limit(tmp_path):
    class BadProvider(StubProvider):
        def iter_pages(self, request, cancel):
            yield page()
            yield page()

    with Backend(BackendSettings(data_dir=tmp_path), BadProvider) as backend:
        job = backend.collect(SearchRequest(topic="ИИ", max_results=1))
        assert job.state == "failed" and job.error_code == "invalid_response"
        assert job.scanned == 1


def test_cancel_after_terminal_commit_does_not_claim_success(tmp_path):
    closing, release = Event(), Event()

    class SlowClose(StubProvider):
        def close(self):
            closing.set()
            assert release.wait(3)

    with Backend(BackendSettings(data_dir=tmp_path), SlowClose) as backend:
        job_id = backend.submit_collection(SearchRequest(topic="ИИ"))
        try:
            assert closing.wait(2)
            assert backend.get_job(job_id).state == "succeeded"
            assert backend.cancel(job_id) is False
        finally:
            release.set()
        assert backend.wait(job_id, timeout=3).state == "succeeded"


def test_interrupted_close_keeps_lock_until_shutdown_finishes(tmp_path, monkeypatch):
    settings = BackendSettings(data_dir=tmp_path)
    backend = Backend(settings, StubProvider)
    original = backend._executor.shutdown

    def interrupted_shutdown(**kwargs):
        raise KeyboardInterrupt()

    try:
        monkeypatch.setattr(backend._executor, "shutdown", interrupted_shutdown)
        with pytest.raises(KeyboardInterrupt):
            backend.close()
        with pytest.raises(BackendError) as error:
            Backend(settings, StubProvider)
        assert error.value.code == "backend_busy"
    finally:
        monkeypatch.setattr(backend._executor, "shutdown", original)
        backend.close()
    with Backend(settings, StubProvider):
        pass
