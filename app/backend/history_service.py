"""History orchestration on the same bounded worker as ordinary collections."""

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, RLock
from collections import deque

from pydantic import ValidationError

from app.backend.config import BackendSettings
from app.backend.contracts import JobRecord, SearchRequest
from app.backend.errors import BackendError, CancelledError
from app.backend.history import HistoryReport, HistoryRequest, HistoryStore
from app.backend.history_progress import HistoryProgress
from app.diagnostics import log_internal_error


class HistoryBackendMixin(ABC):
    """The host owns these resources; both workflows share its bounded pool and lock."""

    settings: BackendSettings
    history: HistoryStore
    _mutex: RLock
    _executor: ThreadPoolExecutor
    _pending: dict[str, tuple[Event, Future]]
    _history_pending: dict[str, tuple[Event, Future[HistoryReport]]]

    @abstractmethod
    def _ensure_open(self) -> None:
        """Reject new work after the host starts closing."""

    @abstractmethod
    def _collect(self, job_id: str, request: SearchRequest, cancel: Event) -> JobRecord:
        """Run one collection on the host's worker and preserve its result."""

    def submit_history(self, request: HistoryRequest) -> str:
        request = HistoryRequest.model_validate(request)
        with self._mutex:
            self._ensure_open()
            self._check_history_capacity()
            run_id = self.history.create(request)
            self._schedule_history(run_id, False)
            return run_id

    def _check_history_capacity(self) -> None:
        if len(self._pending) + len(self._history_pending) >= self.settings.max_pending_jobs:
            raise BackendError("queue_full", "Очередь заполнена.")

    def resume_history(self, run_id: str, *, retry_incomplete: bool = False) -> str:
        if type(retry_incomplete) is not bool:
            raise BackendError("invalid_query", "retry_incomplete должен быть bool.")
        with self._mutex:
            self._ensure_open()
            report = self.history.report(run_id)
            if run_id in self._history_pending or report.state in {"queued", "running"}:
                raise BackendError("history_busy", "Исторический сбор уже запущен.")
            if report.coverage_complete:
                return run_id
            self._check_history_capacity()
            self.history.prepare_resume(run_id)
            self._schedule_history(run_id, retry_incomplete)
            return run_id

    def _schedule_history(self, run_id: str, retry_incomplete: bool) -> None:
        cancel = Event()
        try:
            future = self._executor.submit(self._collect_history, run_id, cancel, retry_incomplete)
        except RuntimeError:
            self.history.state(run_id, "failed", "backend_closed", "Не удалось запустить сбор.")
            raise BackendError("backend_closed", "Не удалось запустить сбор.") from None
        self._history_pending[run_id] = (cancel, future)
        future.add_done_callback(lambda completed: self._forget_history(run_id))

    def _forget_history(self, run_id: str) -> None:
        with self._mutex:
            self._history_pending.pop(run_id, None)

    def _collect_history(self, run_id: str, cancel: Event, retry_incomplete: bool) -> HistoryReport:
        try:
            self.history.state(run_id, "running")
            # Snapshot the retry set once: a failed period is not retried forever.
            periods = deque(p for p in self.history.report(run_id).periods if p.state != "split")
            attempted = False
            while periods:
                period = periods.popleft()
                if cancel.is_set():
                    raise CancelledError()
                children = self.history.split_overflow(run_id, period.id)
                if children:
                    child_ids = set(children)
                    periods.extend(p for p in self.history.report(run_id).periods if p.id in child_ids)
                    continue
                if period.state == "complete" or (period.state == "partial" and not retry_incomplete):
                    continue
                if attempted and cancel.wait(self.settings.history_period_delay_seconds):
                    raise CancelledError()
                job = self.history.begin_attempt(run_id, period.id)
                self._collect(job.id, job.request, cancel)
                attempted = True
                if cancel.is_set():
                    raise CancelledError()
                children = self.history.split_overflow(run_id, period.id)
                if children:
                    child_ids = set(children)
                    periods.extend(p for p in self.history.report(run_id).periods if p.id in child_ids)
            with self._mutex:
                if cancel.is_set():
                    raise CancelledError()
                report = self.history.report(run_id)
                self.history.state(run_id, "succeeded" if report.completed_periods == report.total_periods else "partial")
        except CancelledError as error:
            self.history.state(run_id, "cancelled", error.code, error.message)
        except BackendError as error:
            self.history.state(run_id, "failed", error.code, error.message)
        except Exception as error:
            if not isinstance(error, ValidationError):
                log_internal_error("backend.history_failed", error)
            self.history.state(run_id, "failed", "internal_error", "Внутренняя ошибка исторического сбора.")
        return self.history.report(run_id)

    def get_history(self, run_id: str) -> HistoryReport:
        self._ensure_open()
        return self.history.report(run_id)

    def get_history_progress(self, run_id: str) -> HistoryProgress:
        from app.backend.history_progress import history_progress
        return history_progress(self.get_history(run_id))

    def list_history(self, limit: int = 50) -> tuple[dict, ...]:
        self._ensure_open()
        return self.history.list_runs(limit)

    def wait_history(self, run_id: str, timeout: float | None = None) -> HistoryReport:
        with self._mutex:
            self._ensure_open()
            item = self._history_pending.get(run_id)
        if item:
            item[1].result(timeout=timeout)
        return self.history.report(run_id)

    def cancel_history(self, run_id: str) -> bool:
        with self._mutex:
            self._ensure_open()
            report = self.history.report(run_id)
            item = self._history_pending.get(run_id)
            if item is None or item[1].done() or report.state not in {"queued", "running"}:
                return False
            item[0].set()
            return True

    def collect_history(self, request: HistoryRequest) -> HistoryReport:
        return self.wait_history(self.submit_history(request))
