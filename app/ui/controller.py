"""Owned control, document-read and compute lanes; callbacks run on Tk's timer."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import RLock

from app.diagnostics import close_private_logging, configure_private_logging, log_internal_error

AUXILIARY_METHODS = {"ml_analyze", "ml_inspect", "ml_export", "pilot_history_view", "pilot_install_model", "pilot_install_local_llm", "pilot_import_report",
                     "pilot_start", "pilot_refine_candidate", "pilot_result", "pilot_documents", "pilot_list_runs", "pilot_begin_review",
                     "pilot_import_arxiv", "pilot_import_result", "pilot_export_result", "pilot_sensitivity", "pilot_supplemental_matches", "pilot_apply_review",
                     "pilot_supplemental_list", "pilot_create_signal_query", "pilot_preview_signal_csv",
                     "pilot_import_signal_csv", "pilot_import_signal_atom", "pilot_fetch_signal_wordstat", "pilot_start_signals",
                     "pilot_list_signal_runs", "pilot_signal_result", "pilot_export_signal", "pilot_import_signal", "pilot_signal_associations",
                     "pilot_confirm_signal_grant", "pilot_signal_arxiv_candidates", "pilot_link_signal_arxiv",
                     "pilot_signal_scientific_runs", "pilot_signal_scientific_cards",
                     "pilot_signal_finding_evidence", "pilot_signal_scenario", "pilot_signal_compare",
                     "pilot_signal_largest_event_options", "pilot_signal_watch_state", "pilot_set_signal_watch",
                     "pilot_translate_query", "pilot_prepare_local_models"}
DOCUMENT_READ_METHODS = {"list_documents", "list_document_versions"}

def _diagnose_unexpected(event: str, error: Exception) -> None:
    from concurrent.futures import CancelledError

    from pydantic import ValidationError

    from app.backend.errors import BackendError
    from app.ml.contracts import AnalysisInputError

    if not isinstance(error, (BackendError, AnalysisInputError, ValidationError, CancelledError)):
        log_internal_error(event, error)


def create_backend(data_dir=None):
    from app.backend.config import BackendSettings
    from app.backend.service import Backend
    from app.identity import default_data_dir
    from app.profiles import resolve_profile

    return Backend(BackendSettings(data_dir=resolve_profile(data_dir if data_dir is not None else default_data_dir())))


class Controller:
    def __init__(self, scheduler, factory=create_backend):
        self.scheduler, self.factory = scheduler, factory
        self.backend = None
        self.profile_anchor = None
        self.pilot = None
        self.pilot_lock = RLock()
        self.auxiliary_lock = RLock()
        self.read_lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-backend")
        self.ml_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-local-ml")
        self.read_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-document-read")
        self.pending = {}
        self._pending_epochs = {}
        self.profile_epoch = 0
        self.profile_transition = False
        self.on_profile_transition = None
        self.closing = self.stopped = False
        self.close_future = None
        self._poll_timer = None
        self._poll_running = False
        self._schedule_poll(500)

    def _schedule_poll(self, delay):
        if self._poll_timer is not None:
            cancel = getattr(self.scheduler, "after_cancel", None)
            if cancel is not None:
                cancel(self._poll_timer)
        self._poll_timer = self.scheduler.after(delay, self._poll)

    def call(self, key, method, success, failure, *args, **kwargs):
        if self.closing or self.stopped or self.profile_transition or key in self.pending:
            return False
        idle = not self.pending and self.close_future is None
        epoch = self.profile_epoch
        restoring = method == "pilot_restore"
        if restoring:
            self._profile_transition(True)

        def deliver(callback, value):
            if restoring:
                # Keep admission closed until Tk is ready to reset the old UI.
                # Discard obsolete delivery slots before the new UI requests
                # pages with the same keys. Running readers retain their locks.
                for old_key, old_epoch in tuple(self._pending_epochs.items()):
                    if old_epoch != self.profile_epoch:
                        self.pending.pop(old_key, None)
                        self._pending_epochs.pop(old_key, None)
                self._profile_transition(False)
            if restoring or epoch == self.profile_epoch:
                callback(value)

        if method in DOCUMENT_READ_METHODS:
            executor = self.read_executor
        else:
            executor = self.ml_executor if method in AUXILIARY_METHODS else self.executor
        try:
            future = executor.submit(self._invoke, method, args, kwargs, epoch)
        except Exception:
            if restoring:
                self._profile_transition(False)
            raise
        self.pending[key] = (future, lambda value: deliver(success, value),
                             lambda error: deliver(failure, error))
        self._pending_epochs[key] = epoch
        if idle and not self._poll_running:
            self._schedule_poll(50)
        return True

    def _profile_transition(self, active):
        if self.profile_transition == active:
            return
        self.profile_transition = active
        if self.on_profile_transition is not None:
            self.on_profile_transition(active)

    def _invoke(self, method, args, kwargs, epoch=None):
        try:
            if (getattr(self, "closing", False) or getattr(self, "stopped", False)
                    or epoch is not None and epoch != self.profile_epoch):
                from concurrent.futures import CancelledError
                raise CancelledError()
            lock = (self.read_lock if method in DOCUMENT_READ_METHODS else
                    self.auxiliary_lock if method in AUXILIARY_METHODS else None)
            if lock is not None:
                with lock:
                    if self.closing or self.stopped or epoch is not None and epoch != self.profile_epoch:
                        from concurrent.futures import CancelledError
                        raise CancelledError()
                    return self._dispatch(method, args, kwargs)
            return self._dispatch(method, args, kwargs)
        except Exception as error:
            _diagnose_unexpected("ui.invoke_failed", error)
            raise

    def _dispatch(self, method, args, kwargs):
        if method == "open":
            self.backend = self.factory()
            configure_private_logging(self.backend.settings.data_dir)
            return str(self.backend.settings.data_dir), self.backend.sources()
        if method == "open_url":
            import webbrowser

            from app.input_safety import is_safe_http_url
            from app.ml.contracts import AnalysisInputError
            if not is_safe_http_url(args[0]):
                raise AnalysisInputError("Некорректная HTTP(S)-ссылка источника.")
            if not webbrowser.open(args[0]):
                raise OSError("Browser unavailable")
            return
        if self.backend is None:
            raise RuntimeError("Backend is not open")
        if method == "pilot_backup":
            return self._backup(*args, **kwargs)
        if method == "pilot_restore":
            return self._restore(*args, **kwargs)
        if method.startswith("pilot_"):
            from app.pilot.service import PilotService

            operations = {"status", "configure", "install_model", "install_local_llm", "start", "get", "list_runs", "history_view", "cancel",
                          "resume", "result", "documents", "export_result", "import_result", "sensitivity",
                          "supplemental_list", "import_report", "import_arxiv", "supplemental_matches",
                          "begin_review", "refine_candidate", "review_progress", "apply_review", "budget_status", "budget_reconcile",
                          "budget_acknowledge", "delete_credential", "translate_query",
                          "prepare_local_models",
                          "create_signal_query", "preview_signal_csv",
                          "import_signal_csv", "import_signal_atom", "fetch_signal_wordstat",
                          "start_signals", "list_signal_runs", "signal_result", "export_signal", "import_signal", "signal_associations",
                          "confirm_signal_grant", "signal_arxiv_candidates", "link_signal_arxiv",
                          "signal_scientific_runs", "signal_scientific_cards", "signal_finding_evidence",
                          "signal_scenario", "signal_compare", "signal_largest_event_options",
                          "signal_watch_state", "set_signal_watch"}
            operation = method.removeprefix("pilot_")
            if operation not in operations:
                raise ValueError("Unknown analysis operation")
            with self.pilot_lock:
                if self.closing or self.stopped:
                    from concurrent.futures import CancelledError
                    raise CancelledError()
                if self.pilot is None:
                    self.pilot = PilotService(self.backend.settings.data_dir, self.backend.credentials)
                # Shutdown can be requested while the first service opens its
                # storage. Publish it for the closing worker, but do not start
                # the requested operation after that shutdown request.
                if self.closing or self.stopped:
                    from concurrent.futures import CancelledError
                    raise CancelledError()
            if operation == "history_view":
                from app.ui.analysis_history import history_view
                return history_view(self.pilot, *args, **kwargs)
            return getattr(self.pilot, operation)(*args, **kwargs)
        if method in {"ml_analyze", "ml_inspect", "ml_export"}:
            from app.ml import service
            if method == "ml_analyze":
                return service.run_analysis(self.backend, *args, **kwargs)
            function = service.inspect_snapshot if method == "ml_inspect" else service.export_result
            return function(*args, **kwargs)
        if method == "ml_collect":
            from app.backend.history import HistoryRequest
            return self.backend.submit_history(HistoryRequest.model_validate(args[0]))
        if method == "start_collection":
            from app.backend.contracts import SearchRequest
            values, sources = args
            return self.backend.submit_collections(SearchRequest.model_validate(values), sources)
        return getattr(self.backend, method)(*args, **kwargs)

    @contextmanager
    def _idle_bulk_work(self, message: str) -> Iterator[None]:
        """Fence both kinds of profile readers without queuing maintenance behind them."""
        from app.runtime.jobs import TaskFailure

        acquired = []
        try:
            for lock in (self.auxiliary_lock, self.read_lock):
                if not lock.acquire(blocking=False):
                    raise TaskFailure(message)
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    def _backup(self, destination):
        from pathlib import Path
        from app.runtime.backup import BackupSession, create_backup
        from app.runtime.jobs import TaskFailure
        from app.pilot.settings import load_settings

        with self._idle_bulk_work("Дождитесь завершения чтения документов, импорта, экспорта или локального анализа перед резервным копированием."):
            if self.backend is None:
                raise RuntimeError("Backend is not open")
            directory = self.backend.settings.data_dir
            if self.backend.has_active_work():
                raise TaskFailure("Завершите сбор документов и исторические задачи перед резервным копированием.")
            settings = load_settings(directory).model_dump(mode="json")
            if self.pilot is None:
                from app.pilot.service import PilotService
                self.pilot = PilotService(directory, self.backend.credentials)
            if self.pilot.coordinator.has_active_work():
                raise TaskFailure("Перед резервным копированием завершите или отмените текущий анализ.")
            self.pilot.close()
            self.pilot = None
            self.backend.close()
            self.backend = None
            try:
                with BackupSession(directory) as session:
                    saved = create_backup(session, Path(destination), settings=settings, keep=3)
                return {"path": str(saved.path), "files": saved.files, "rotation_warning": saved.rotation_warning}
            finally:
                self.backend = self.factory()
                configure_private_logging(self.backend.settings.data_dir)

    def _restore(self, package, parent_directory):
        from datetime import datetime, UTC
        from pathlib import Path
        from uuid import uuid4
        from app.pilot.service import PilotService
        from app.profiles import activate_profile
        from app.runtime.backup import restore_backup
        from app.runtime.jobs import TaskFailure

        previous_factory = self.factory
        with self._idle_bulk_work("Дождитесь завершения чтения документов и локальных операций перед восстановлением."):
            if self.backend is None:
                raise RuntimeError("Backend is not open")
            if self.backend.has_active_work():
                raise TaskFailure("Завершите текущий сбор данных перед восстановлением.")
            if self.pilot is not None and self.pilot.coordinator.has_active_work():
                raise TaskFailure("Завершите текущий анализ перед восстановлением.")
            anchor = getattr(self, "profile_anchor", None) or self.backend.settings.data_dir
            target = Path(parent_directory) / ("Trendanalyser-восстановлено-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
            restored = restore_backup(Path(package), target)
            if self.pilot is not None:
                self.pilot.close()
                self.pilot = None
            self.backend.close()
            self.backend = None
            try:
                self.backend = create_backend(restored)
                self.pilot = PilotService(restored, self.backend.credentials)
                sources, status = self.backend.sources(), self.pilot.status()
                durability_warning = activate_profile(anchor, restored)
                configure_private_logging(self.backend.settings.data_dir)
                self.factory = lambda: create_backend(anchor)
                self.profile_epoch += 1
                return {"path": str(restored), "sources": sources, "status": status,
                        "durability_warning": durability_warning}
            except Exception:
                if self.pilot is not None:
                    self.pilot.close()
                    self.pilot = None
                if self.backend is not None:
                    self.backend.close()
                self.backend = previous_factory()
                configure_private_logging(self.backend.settings.data_dir)
                raise

    def close(self, success, failure):
        if self.stopped or self.close_future is not None:
            return
        self.closing = True
        pilot = self.pilot
        if pilot is not None:
            pilot.model_cancel.set()
            pilot.view_cancel.set()
        self.close_success, self.close_failure = success, failure
        self.close_future = self.executor.submit(self._close_backend)
        if not self._poll_running:
            self._schedule_poll(50)

    def _close_backend(self):
        try:
            self.ml_executor.shutdown(wait=False, cancel_futures=True)
            self.read_executor.shutdown(wait=False, cancel_futures=True)
            # An auxiliary request may still be constructing the first pilot
            # service. Serialize with publication so it cannot outlive close.
            with self.pilot_lock:
                if self.pilot is not None:
                    self.pilot.close()
            self.ml_executor.shutdown(wait=True, cancel_futures=True)
            # Legacy reads use their own SQLite connections. Keep the backend
            # and its profile ownership alive until those readers have exited.
            self.read_executor.shutdown(wait=True, cancel_futures=True)
            if self.backend is not None:
                self.backend.close()
            close_private_logging()
        except Exception as error:
            _diagnose_unexpected("ui.close_failed", error)
            raise

    def _callback_error(self, error):
        reporter = getattr(self.scheduler, "report_callback_exception", None)
        if reporter is None:
            log_internal_error("ui.callback_failed", error)
        else:
            try:
                reporter(type(error), error, error.__traceback__)
            except Exception:
                log_internal_error("ui.callback_failed", error)

    def _deliver_failure(self, failure, error):
        try:
            failure(error)
        except Exception as callback_error:
            self._callback_error(callback_error)

    def _poll(self):
        self._poll_timer = None
        self._poll_running = True
        try:
            for key, entry in list(self.pending.items()):
                future, success, failure = entry
                if self.pending.get(key) is not entry:
                    continue
                if not future.done():
                    continue
                del self.pending[key]
                self._pending_epochs.pop(key, None)
                if self.closing:
                    continue
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - worker errors must reach their UI failure callback
                    self._deliver_failure(failure, error)
                else:
                    try:
                        success(result)
                    except Exception as error:
                        self._callback_error(error)
                        self._deliver_failure(failure, error)
            if self.close_future is not None and self.close_future.done():
                future, self.close_future = self.close_future, None
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - shutdown failure keeps the UI available for retry
                    self.close_failure(error)
                else:
                    self.stopped = True
                    self.executor.shutdown(wait=False)
                    self.close_success()
        finally:
            self._poll_running = False
            if not self.stopped:
                self._schedule_poll(50 if self.pending or self.close_future is not None else 500)
