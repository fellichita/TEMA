"""Публичное ядро: фоновый сбор, отмена, сохранённые документы и история."""

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from threading import Event, RLock
from typing import Any, TypedDict

from pydantic import ValidationError

from app.backend.config import BackendSettings
from app.backend.contracts import SOURCE_NAMES, TERMINAL_STATES, DocumentPage, JobRecord, SearchRequest, SourcePage
from app.backend.errors import BackendError, CancelledError
from app.backend.history import HistoryReport, HistoryStore
from app.backend.history_service import HistoryBackendMixin
from app.backend.locking import InstanceLock
from app.backend.providers.base import DocumentProvider
from app.backend.providers.crossref import CrossrefProvider
from app.backend.providers.openalex import OpenAlexProvider
from app.backend.providers.epo import EpoOpsProvider
from app.backend.repository import Repository
from app.backend.validation import collection_requests
from app.diagnostics import log_internal_error
from app.runtime.credentials import CredentialStore, CredentialUnavailable
from app.runtime.session import credentials as session_credentials


class _ProviderOptions(TypedDict):
    page_size: int
    timeout_seconds: float
    max_retries: int
    max_response_bytes: int


class Backend(HistoryBackendMixin):
    """Использовать как context manager. Независимые задания имеют ограниченный параллелизм."""

    def __init__(
        self,
        settings: BackendSettings | None = None,
        provider_factory: Callable[[], DocumentProvider] | None = None,
        *, provider_factories: Mapping[str, Callable[[], DocumentProvider]] | None = None,
        credentials: CredentialStore | None = None,
    ):
        if provider_factories and any(source not in SOURCE_NAMES for source in provider_factories):
            raise BackendError("invalid_source", "Неизвестный источник.")
        self.settings = settings or BackendSettings()
        self.credentials = credentials if credentials is not None else session_credentials()
        self.credentials.import_legacy_environment()
        self._lock = InstanceLock(self.settings.data_dir / "backend.lock")
        self._lock.acquire()
        try:
            self.repository = Repository(self.settings.database_path)
            self.repository.recover_interrupted()
            self.history = HistoryStore(self.repository)
            self.history.recover()
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-collector")
        except BaseException:
            self._lock.release()
            raise
        self._provider_factory = provider_factory
        self._provider_factories = dict(provider_factories or {})
        self._mutex = RLock()
        self._pending: dict[str, tuple[Event, Future[Any]]] = {}
        self._history_pending: dict[str, tuple[Event, Future[HistoryReport]]] = {}
        self._closed = False
        self._closing = False

    def _default_provider(self, source: str) -> DocumentProvider:
        options: _ProviderOptions = dict(
            page_size=min(self.settings.page_size, 100),
            timeout_seconds=self.settings.timeout_seconds,
            max_retries=self.settings.max_retries,
            max_response_bytes=self.settings.max_response_bytes,
        )
        if source == "crossref":
            return CrossrefProvider(**options)
        if source == "openalex":
            return OpenAlexProvider(api_key=self.credentials.get("openalex_api_key"), **options)
        if source == "epo":
            return EpoOpsProvider(consumer_key=self.credentials.get("epo_ops_key"),
                                  consumer_secret=self.credentials.get("epo_ops_secret"), **options)
        raise BackendError("invalid_source", "Неизвестный источник.")

    @staticmethod
    def sources(credentials: CredentialStore | None = None) -> tuple[dict, ...]:
        store = credentials if credentials is not None else session_credentials()
        def present(name):
            try:
                return bool(store.get(name))
            except CredentialUnavailable:
                return None
        openalex, epo_key, epo_secret = (present(name) for name in ("openalex_api_key", "epo_ops_key", "epo_ops_secret"))
        return (
            {"id": "crossref", "kind": "publications", "credentials_required": False},
            {"id": "openalex", "kind": "publications", "credentials_required": False,
             "key_configured": openalex,
             "credential_state": "unavailable" if openalex is None else "available"},
            {"id": "epo", "kind": "patents", "credentials_required": True,
             "credentials_configured": bool(epo_key and epo_secret),
             "credential_state": "unavailable" if epo_key is None or epo_secret is None else "available",
             "max_results_per_query": 2000},
        )

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise BackendError("backend_closed", "Backend уже закрыт.")

    def submit_collection(self, request: SearchRequest) -> str:
        request = SearchRequest.model_validate(request)
        with self._mutex:
            self._ensure_open()
            if len(self._pending) + len(self._history_pending) >= self.settings.max_pending_jobs:
                raise BackendError("queue_full", "Очередь заполнена. Дождитесь завершения текущих заданий.")
            job = self.repository.create_job(request)
            cancel = Event()
            try:
                future = self._executor.submit(self._collect, job.id, request, cancel)
            except RuntimeError:
                self.repository.finish_job(job.id, "failed", "backend_closed", "Не удалось запустить сбор.")
                raise BackendError("backend_closed", "Не удалось запустить сбор.") from None
            self._pending[job.id] = (cancel, future)
            future.add_done_callback(lambda completed: self._forget(job.id))
            return job.id

    def _forget(self, job_id: str) -> None:
        with self._mutex:
            self._pending.pop(job_id, None)

    def _forget_group(self, job_id: str, _completed: Future[tuple[JobRecord, ...]]) -> None:
        self._forget(job_id)

    def _collect(self, job_id: str, request: SearchRequest, cancel: Event) -> JobRecord:
        provider = None
        try:
            self.repository.start_job(job_id)
            if cancel.is_set():
                raise CancelledError()
            factory = self._provider_factories.get(request.source) or self._provider_factory
            provider = factory() if factory else self._default_provider(request.source)
            pages_seen = False
            scanned = 0
            for page in provider.iter_pages(request, cancel):
                if cancel.is_set():
                    raise CancelledError()
                pages_seen = True
                scanned += page.scanned
                if scanned > request.max_results or any(doc.source != request.source for doc in page.documents):
                    raise BackendError("invalid_response", "Источник нарушил формат или лимит выдачи.")
                self.repository.ingest_page(job_id, page)
            if not pages_seen:
                raise BackendError("invalid_response", "Источник не сообщил результат загрузки.")
            # Принятая отмена и финальный успешный commit не обгоняют друг друга.
            with self._mutex:
                if cancel.is_set():
                    raise CancelledError()
                return self.repository.finish_job(job_id, "succeeded")
        except CancelledError as error:
            return self.repository.finish_job(job_id, "cancelled", error.code, error.message)
        except BackendError as error:
            return self.repository.finish_job(job_id, "failed", error.code, error.message)
        except Exception as error:
            # Исходная ошибка может содержать тело ответа, запрос или локальные пути.
            if not isinstance(error, ValidationError):
                log_internal_error("backend.collection_failed", error)
            return self.repository.finish_job(job_id, "failed", "internal_error", "Внутренняя ошибка сбора документов.")
        finally:
            if provider is not None:
                try:
                    provider.close()
                except Exception as error:
                    # Ошибка освобождения клиента не меняет сохранённый результат.
                    if not isinstance(error, (BackendError, ValidationError)):
                        log_internal_error("backend.provider_close_failed", error)

    def _fetch_pages(self, request: SearchRequest,
                     cancel: Event) -> tuple[tuple[SourcePage, ...], Exception | None]:
        """Fetch one fixed request without touching shared storage.

        Grouped source requests may overlap on the network, but their pages are
        committed later in request order.  Retaining pages before an error also
        preserves the existing partial-collection behaviour.
        """
        provider = None
        pages: list[SourcePage] = []
        scanned = 0
        try:
            if cancel.is_set():
                raise CancelledError()
            factory = self._provider_factories.get(request.source) or self._provider_factory
            provider = factory() if factory else self._default_provider(request.source)
            for page in provider.iter_pages(request, cancel):
                if cancel.is_set():
                    raise CancelledError()
                page = SourcePage.model_validate(page.model_dump() if isinstance(page, SourcePage) else page)
                scanned += page.scanned
                if scanned > request.max_results or any(doc.source != request.source for doc in page.documents):
                    raise BackendError("invalid_response", "Источник нарушил формат или лимит выдачи.")
                pages.append(page)
            if not pages:
                raise BackendError("invalid_response", "Источник не сообщил результат загрузки.")
            return tuple(pages), None
        except Exception as error:
            return tuple(pages), error
        finally:
            if provider is not None:
                try:
                    provider.close()
                except Exception as error:
                    if not isinstance(error, (BackendError, ValidationError)):
                        log_internal_error("backend.provider_close_failed", error)

    def _commit_fetched(self, job_id: str, cancel: Event, pages: tuple[SourcePage, ...],
                        error: Exception | None) -> JobRecord:
        """Commit prefetched pages deterministically after network work finishes."""
        try:
            for page in pages:
                self.repository.ingest_page(job_id, page)
            if cancel.is_set() or isinstance(error, CancelledError):
                raise CancelledError()
            if isinstance(error, BackendError):
                raise error
            if error is not None:
                if not isinstance(error, ValidationError):
                    log_internal_error("backend.collection_failed", error)
                return self.repository.finish_job(
                    job_id, "failed", "internal_error", "Внутренняя ошибка сбора документов.")
            return self.repository.finish_job(job_id, "succeeded")
        except CancelledError as failure:
            return self.repository.finish_job(job_id, "cancelled", failure.code, failure.message)
        except BackendError as failure:
            return self.repository.finish_job(job_id, "failed", failure.code, failure.message)
        except Exception as failure:
            if not isinstance(failure, ValidationError):
                log_internal_error("backend.collection_failed", failure)
            return self.repository.finish_job(
                job_id, "failed", "internal_error", "Внутренняя ошибка сбора документов.")

    def _collect_group(self, items: tuple[tuple[str, SearchRequest, Event], ...]) -> tuple[JobRecord, ...]:
        """Overlap fixed source requests, then preserve the former commit order."""
        for job_id, _request, _cancel in items:
            self.repository.start_job(job_id)
        with ThreadPoolExecutor(max_workers=min(self.settings.collection_workers, len(items)),
                                thread_name_prefix="source-fetch") as pool:
            futures = [pool.submit(self._fetch_pages, request, cancel)
                       for _job_id, request, cancel in items]
            fetched = [future.result() for future in futures]
        return tuple(self._commit_fetched(job_id, cancel, pages, error)
                     for (job_id, _request, cancel), (pages, error) in zip(items, fetched, strict=True))

    def cancel(self, job_id: str) -> bool:
        with self._mutex:
            self._ensure_open()
            job = self.repository.get_job(job_id)
            item = self._pending.get(job_id)
            if job.state in TERMINAL_STATES or item is None or item[1].done():
                return False
            item[0].set()
            return True

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord:
        with self._mutex:
            self._ensure_open()
            item = self._pending.get(job_id)
        if item is not None:
            item[1].result(timeout=timeout)
        return self.repository.get_job(job_id)

    def collect(self, request: SearchRequest) -> JobRecord:
        """Синхронный вариант для скриптов; UI использует submit_collection."""
        return self.wait(self.submit_collection(request))

    def submit_collections(self, request: SearchRequest, sources=("crossref", "openalex")) -> tuple[str, ...]:
        requests = collection_requests(request, sources)
        with self._mutex:
            self._ensure_open()
            if len(self._pending) + len(self._history_pending) + len(requests) > self.settings.max_pending_jobs:
                raise BackendError("queue_full", "Недостаточно места в очереди для выбранных источников.")
            jobs = tuple(self.repository.create_job(item) for item in requests)
            items = tuple((job.id, item, Event()) for job, item in zip(jobs, requests, strict=True))
            try:
                future = self._executor.submit(self._collect_group, items)
            except RuntimeError:
                for job_id, _item, _cancel in items:
                    self.repository.finish_job(job_id, "failed", "backend_closed", "Не удалось запустить сбор.")
                raise BackendError("backend_closed", "Не удалось запустить сбор.") from None
            for job_id, _item, cancel in items:
                self._pending[job_id] = (cancel, future)
                future.add_done_callback(partial(self._forget_group, job_id))
            return tuple(job.id for job in jobs)

    def collect_many(self, request: SearchRequest, sources=("crossref", "openalex")) -> tuple[JobRecord, ...]:
        """Лимит применяется к каждому источнику. Ошибка одного не останавливает остальные."""
        return tuple(self.wait(job_id) for job_id in self.submit_collections(request, sources))

    def list_document_versions(self, document_key: str, limit=100, offset=0) -> DocumentPage:
        self._ensure_open()
        return self.repository.list_document_versions(document_key, limit=limit, offset=offset)

    def get_job(self, job_id: str) -> JobRecord:
        self._ensure_open()
        return self.repository.get_job(job_id)

    def list_jobs(self, limit: int = 50) -> tuple[JobRecord, ...]:
        self._ensure_open()
        return self.repository.list_jobs(limit=limit)

    def has_active_work(self) -> bool:
        with self._mutex:
            return bool(self._pending or self._history_pending)

    def list_documents(
        self, *, job_id: str | None = None, query: str | None = None,
        limit: int = 100, offset: int = 0, history_id: str | None = None,
        sort_by: str = "default", descending: bool = False,
    ) -> DocumentPage:
        self._ensure_open()
        return self.repository.list_documents(job_id=job_id, query=query, limit=limit, offset=offset, history_id=history_id,
                                              sort_by=sort_by, descending=descending)

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closing = True
            for cancel, _ in self._pending.values():
                cancel.set()
            for cancel, _ in self._history_pending.values():
                cancel.set()
        # При KeyboardInterrupt оставляем lock за живыми workers; close можно повторить.
        self._executor.shutdown(wait=True)
        self._lock.release()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
