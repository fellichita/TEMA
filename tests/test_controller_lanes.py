"""Bulk reads/startup must not queue cancellation behind filesystem work."""

from concurrent.futures import CancelledError
from threading import Event
from types import SimpleNamespace

import pytest

from app.ui.controller import Controller, DOCUMENT_READ_METHODS


class Scheduler:
    def after(self, delay, callback):
        pass


@pytest.mark.parametrize('method', ['pilot_start', 'pilot_result', 'pilot_documents', 'pilot_list_runs', 'pilot_begin_review',
                                    'pilot_supplemental_list', 'list_documents', 'list_document_versions'])
def test_control_completes_while_bulk_operation_is_blocked(method):
    controller = Controller(Scheduler())
    entered, release, cancelled = Event(), Event(), Event()

    def dispatch(operation, args, kwargs):
        if operation == method:
            entered.set()
            assert release.wait(3), 'test did not release bulk operation'
        elif operation == 'pilot_cancel':
            cancelled.set()
            return True

    controller._dispatch = dispatch
    try:
        assert controller.call('bulk', method, lambda _: None, lambda _: None)
        assert entered.wait(1)
        assert controller.call('cancel', 'pilot_cancel', lambda _: None, lambda _: None)
        assert cancelled.wait(1), 'cancellation queued behind bulk work'
        assert controller.pending['cancel'][0].result(1) is True
        assert not controller.pending['bulk'][0].done()
        # The same lock protects profile restore/backup against in-flight reads.
        lock = controller.read_lock if method in DOCUMENT_READ_METHODS else controller.auxiliary_lock
        assert not lock.acquire(blocking=False)
    finally:
        release.set()
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)


def test_close_signals_bulk_work_and_discards_queued_operations():
    controller = Controller(Scheduler())
    entered, stopped = Event(), Event()
    controller.pilot = SimpleNamespace(model_cancel=Event(), view_cancel=Event(), close=lambda: None)
    controller.backend = SimpleNamespace(close=stopped.set)
    calls = []

    def dispatch(operation, args, kwargs):
        calls.append(operation)
        entered.set()
        assert controller.pilot.model_cancel.wait(3)
        raise CancelledError()

    controller._dispatch = dispatch
    try:
        controller.call('start', 'pilot_start', lambda _: None, lambda _: None)
        assert entered.wait(1)
        controller.call('read', 'pilot_result', lambda _: None, lambda _: None)
        controller.close(lambda: None, lambda _: None)
        controller.close_future.result(2)
        assert stopped.is_set()
        assert controller.pilot.view_cancel.is_set()
        assert calls == ['pilot_start']
        assert not controller.call('late', 'pilot_start', lambda _: None, lambda _: None)
    finally:
        controller.pilot.model_cancel.set()
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)


def test_close_waits_for_initial_service_publication_and_never_starts_its_operation(monkeypatch):
    from app.pilot import service

    constructing, release, service_closed, backend_closed = Event(), Event(), Event(), Event()
    operations = []

    class DelayedPilot:
        def __init__(self, *_):
            self.model_cancel, self.view_cancel = Event(), Event()
            constructing.set()
            assert release.wait(3), 'test did not release service construction'

        def list_runs(self):
            operations.append('list_runs')
            return []

        def close(self):
            self.model_cancel.set()
            self.view_cancel.set()
            service_closed.set()

    monkeypatch.setattr(service, 'PilotService', DelayedPilot)
    controller = Controller(Scheduler())
    controller.backend = SimpleNamespace(settings=SimpleNamespace(data_dir='unused-test-directory'),
                                         credentials=None, close=backend_closed.set)
    try:
        assert controller.call('initial-read', 'pilot_list_runs', lambda _: None, lambda _: None)
        assert constructing.wait(1)
        assert controller.pilot is None
        controller.close(lambda: None, lambda _: None)
        release.set()
        controller.close_future.result(2)
        with pytest.raises(CancelledError):
            controller.pending['initial-read'][0].result(1)
        assert operations == []
        assert service_closed.is_set()
        assert backend_closed.is_set()
        assert controller.pilot.model_cancel.is_set()
        assert controller.pilot.view_cancel.is_set()
    finally:
        release.set()
        controller.executor.shutdown(wait=True, cancel_futures=True)
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)
