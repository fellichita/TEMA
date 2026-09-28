"""Profile changes fence queued commands and callbacks from the previous UI."""

from concurrent.futures import CancelledError, Future
from threading import Event

import pytest

from app.ui.controller import Controller
from tests.test_controller_lanes import Scheduler


@pytest.fixture
def controller():
    value = Controller(Scheduler())
    yield value
    value.executor.shutdown(wait=True, cancel_futures=True)
    value.ml_executor.shutdown(wait=True, cancel_futures=True)
    value.read_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("fails", [False, True])
def test_restore_rejects_mutations_until_its_ui_callback(controller, fails):
    entered, release = Event(), Event()
    transitions, dispatched, results, errors = [], [], [], []
    controller.on_profile_transition = transitions.append

    def dispatch(method, args, kwargs):
        dispatched.append(method)
        if method == "pilot_restore":
            entered.set()
            assert release.wait(3)
            if fails:
                raise ValueError("invalid archive")
            controller.profile_epoch += 1
            return "new profile"

    controller._dispatch = dispatch
    try:
        assert controller.call("restore", "pilot_restore", results.append, errors.append)
        assert entered.wait(1)
        assert not controller.call("settings", "pilot_configure", results.append, errors.append)
        assert not controller.call("collect", "start_collection", results.append, errors.append)
        release.set()
        future = controller.pending["restore"][0]
        if fails:
            with pytest.raises(ValueError):
                future.result(2)
        else:
            future.result(2)
        assert not controller.call("before-ui-reset", "pilot_start", results.append, errors.append)
        controller._poll()
        assert transitions == [True, False]
        assert not controller.profile_transition
        assert controller.call("after-ui-reset", "pilot_status", results.append, errors.append)
        controller.pending["after-ui-reset"][0].result(2)
        assert dispatched == ["pilot_restore", "pilot_status"]
        assert bool(errors) is fails
    finally:
        release.set()


def test_stale_completed_callback_cannot_publish_into_new_profile(controller):
    published = []
    controller._dispatch = lambda *_: "old page"
    assert controller.call("page", "pilot_documents", published.append, published.append)
    controller.pending["page"][0].result(2)
    controller.profile_epoch += 1
    controller._poll()
    assert published == []


def test_auxiliary_command_rechecks_profile_after_acquiring_its_lock(controller):
    dispatched = []
    controller._dispatch = lambda *args: dispatched.append(args)
    with controller.auxiliary_lock:
        assert controller.call("queued-page", "pilot_documents", lambda _: None, lambda _: None)
        controller.profile_epoch += 1
    with pytest.raises(CancelledError):
        controller.pending["queued-page"][0].result(2)
    assert dispatched == []


def test_restore_releases_old_pending_keys_before_new_profile_refresh(controller, monkeypatch):
    old_page, restored, new_page = Future(), Future(), Future()
    monkeypatch.setattr(controller.ml_executor, "submit", lambda *_: old_page)
    monkeypatch.setattr(controller.executor, "submit", lambda *_: restored)
    published, admitted = [], []
    assert controller.call("page", "pilot_documents", published.append, published.append)

    def refresh(_):
        monkeypatch.setattr(controller.ml_executor, "submit", lambda *_: new_page)
        admitted.append(controller.call("page", "pilot_documents", published.append, published.append))

    assert controller.call("restore", "pilot_restore", refresh, published.append)
    controller.profile_epoch += 1
    restored.set_result("new profile")
    controller._poll()
    assert admitted == [True]
    old_page.set_result("stale data")
    controller._poll()
    assert published == []
    assert controller.pending["page"][0] is new_page
    new_page.set_result("new data")
    controller._poll()
    assert published == ["new data"]
