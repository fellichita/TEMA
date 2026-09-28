"""Document reads, compute and control have independent owned execution lanes."""

from concurrent.futures import CancelledError
from threading import Event, RLock
from types import SimpleNamespace

import pytest

from app.runtime.jobs import TaskFailure
from app.ui.controller import Controller


class Scheduler:
    def after(self, *_):
        pass


@pytest.fixture
def controller():
    value = Controller(Scheduler())
    yield value
    value.executor.shutdown(wait=True, cancel_futures=True)
    value.ml_executor.shutdown(wait=True, cancel_futures=True)
    reader = getattr(value, "read_executor", None)
    if reader is not None:
        reader.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("method", ["list_documents", "list_document_versions"])
def test_document_reads_and_cancel_complete_while_ml_is_blocked(controller, method):
    entered, release = Event(), Event()

    def dispatch(operation, args, kwargs):
        if operation == "ml_analyze":
            entered.set()
            assert release.wait(3), "test did not release ML"
            return "analysis"
        return operation

    controller._dispatch = dispatch
    try:
        controller.call("ml", "ml_analyze", lambda _: None, lambda _: None)
        assert entered.wait(1)
        controller.call("read", method, lambda _: None, lambda _: None)
        controller.call("cancel", "pilot_cancel", lambda _: None, lambda _: None)
        assert controller.pending["cancel"][0].result(1) == "pilot_cancel"
        assert controller.pending["read"][0].result(1) == method
        assert not controller.pending["ml"][0].done()
    finally:
        release.set()


def test_blocked_read_serializes_other_reads_without_blocking_ml_or_cancel(controller):
    entered, release = Event(), Event()
    seen = []

    def dispatch(operation, args, kwargs):
        seen.append(operation)
        if operation == "list_documents":
            entered.set()
            assert release.wait(3), "test did not release document read"
        return operation

    controller._dispatch = dispatch
    try:
        controller.call("first", "list_documents", lambda _: None, lambda _: None)
        assert entered.wait(1)
        controller.call("second", "list_document_versions", lambda _: None, lambda _: None)
        controller.call("ml", "ml_analyze", lambda _: None, lambda _: None)
        controller.call("cancel", "pilot_cancel", lambda _: None, lambda _: None)
        assert controller.pending["cancel"][0].result(1) == "pilot_cancel"
        assert controller.pending["ml"][0].result(1) == "ml_analyze"
        assert not controller.pending["second"][0].done()
        assert "list_document_versions" not in seen
        release.set()
        assert controller.pending["second"][0].result(1) == "list_document_versions"
    finally:
        release.set()


def test_close_waits_for_owned_read_and_cancels_queued_reads(controller, monkeypatch):
    entered, release, completed, waiting, backend_closed = (Event() for _ in range(5))
    operations = []

    def read():
        operations.append("read")
        entered.set()
        assert release.wait(3), "test did not release read during close"
        completed.set()
        return []

    def close_backend():
        assert completed.is_set(), "backend closed while a read still used it"
        backend_closed.set()

    controller.backend = SimpleNamespace(list_documents=read, list_document_versions=lambda: operations.append("queued"),
                                         close=close_backend)
    shutdown = controller.read_executor.shutdown

    def observed_shutdown(*, wait, cancel_futures=False):
        if wait:
            waiting.set()
        return shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(controller.read_executor, "shutdown", observed_shutdown)
    try:
        controller.call("read", "list_documents", lambda _: None, lambda _: None)
        assert entered.wait(1)
        controller.call("queued", "list_document_versions", lambda _: None, lambda _: None)
        controller.close(lambda: None, lambda _: None)
        assert waiting.wait(1), "close did not join the owned read executor"
        assert not backend_closed.is_set()
        assert not controller.close_future.done()
        assert controller.pending["queued"][0].cancelled()
        release.set()
        controller.close_future.result(1)
        assert backend_closed.is_set()
        assert operations == ["read"]
    finally:
        release.set()


@pytest.mark.parametrize("method,args", [("pilot_backup", ("unused-backups",)),
                                         ("pilot_restore", ("unused.zip", "unused-profile"))])
@pytest.mark.parametrize("busy_method,busy_lock,free_lock", [
    ("list_documents", "read_lock", "auxiliary_lock"),
    ("ml_analyze", "auxiliary_lock", "read_lock"),
])
def test_backup_restore_reject_active_bulk_lane_and_release_partial_locks(
        controller, method, args, busy_method, busy_lock, free_lock):
    entered, release = Event(), Event()
    original_dispatch = controller._dispatch

    def dispatch(operation, values, options):
        if operation == busy_method:
            entered.set()
            assert release.wait(3), "test did not release busy lane"
            return None
        return original_dispatch(operation, values, options)

    controller._dispatch = dispatch
    controller.backend = SimpleNamespace(has_active_work=lambda: False)
    try:
        controller.call("busy", busy_method, lambda _: None, lambda _: None)
        assert entered.wait(1)
        controller.call("maintenance", method, lambda _: None, lambda _: None, *args)
        with pytest.raises(TaskFailure, match="Дождитесь"):
            controller.pending["maintenance"][0].result(1)
        assert not getattr(controller, busy_lock).acquire(blocking=False)
        available = getattr(controller, free_lock)
        assert available.acquire(blocking=False), "maintenance leaked a partially acquired lock"
        available.release()
        controller._poll()
        assert not controller.profile_transition
    finally:
        release.set()


@pytest.mark.parametrize("method", ["list_documents", "list_document_versions"])
def test_read_rechecks_profile_epoch_after_waiting_for_read_lock(controller, method):
    waiting = Event()
    lock = RLock()
    lock.acquire()

    class ObservedLock:
        def __enter__(self):
            waiting.set()
            lock.acquire()

        def __exit__(self, *_):
            lock.release()

    controller.read_lock = ObservedLock()
    dispatched = []
    controller._dispatch = lambda *args: dispatched.append(args)
    try:
        controller.call("read", method, lambda _: None, lambda _: None)
        assert waiting.wait(1), "read bypassed its profile ownership lock"
        controller.profile_epoch += 1
    finally:
        lock.release()
    with pytest.raises(CancelledError):
        controller.pending["read"][0].result(1)
    assert dispatched == []
