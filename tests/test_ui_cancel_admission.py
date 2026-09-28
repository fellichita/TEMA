"""A user cancellation is owned by its start attempt or its collection job."""

from types import SimpleNamespace
from unittest.mock import Mock

from app.ui.window import Application
from tests.test_pilot_operation_ownership import DeferredController, panel as panel


def test_cancel_before_start_ack_signals_the_worker_token(panel):
    panel.start()
    method, _, options = panel.app.controller.calls[0]
    assert method == "pilot_start"
    token = options["cancel"]
    assert not token.is_set()
    panel.cancel()
    assert token.is_set()
    assert panel.pending_cancel
    assert panel.active


def test_new_start_gets_a_fresh_cancel_token(panel):
    panel.start()
    original = panel.app.controller.calls[0][2]["cancel"]
    panel.cancel()
    panel.app.controller.complete("pilot_start", error=RuntimeError("cancelled preflight"))
    panel.start()
    current = panel.app.controller.calls[-1][2]["cancel"]
    assert original.is_set()
    assert current is not original and not current.is_set()


def test_cancellation_of_a_second_collection_is_not_dropped():
    app = object.__new__(Application)
    app.closing = False
    app.controller = DeferredController()
    app.cancel_button = Mock()
    app.job_tree = SimpleNamespace(selection=lambda: ("first",))
    app.cancel_requested = set()
    app._message = app._operation_error = app._select_job = app._request_jobs = Mock()
    app.cancel_job()
    app.job_tree.selection = lambda: ("second",)
    app.cancel_job()
    assert app.controller.arguments("cancel") == [("first",), ("second",)]
