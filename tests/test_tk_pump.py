"""Exercise Tk wait lifecycle with deterministic events, without a display."""

import tkinter as tk
from types import SimpleNamespace

import pytest

from tests.ui import test_desktop as desktop


class Scheduler:
    def __init__(self):
        self.now = 0
        self.callbacks = {}
        self.sequence = 0
        self.stopped = False
        self.destroyed = False
        self.on_quit = lambda: None

    def after(self, delay, callback):
        self.sequence += 1
        identifier = f"after#{self.sequence}"
        self.callbacks[identifier] = (self.now + delay, self.sequence, callback)
        return identifier

    def after_cancel(self, identifier):
        self.callbacks.pop(identifier, None)
        if self.destroyed:
            raise tk.TclError("Application destroyed")

    def mainloop(self):
        self.stopped = False
        while self.callbacks and not self.stopped and not self.destroyed:
            identifier = min(self.callbacks, key=lambda key: self.callbacks[key][:2])
            self.now, _, callback = self.callbacks.pop(identifier)
            callback()

    def quit(self):
        self.stopped = True
        self.on_quit()

    def destroy(self):
        self.destroyed = True


@pytest.fixture
def case(monkeypatch):
    result = desktop.TkCase()
    result.root = Scheduler()
    result.callback_errors = []
    result.controller = None
    monkeypatch.setattr(desktop.time, "monotonic", lambda: result.root.now / 1000)
    return result


def test_success_is_latched_before_other_callbacks_mutate_pending_during_quit(case):
    pending = {}
    observations = []

    def ready():
        observations.append(not pending)
        return not pending

    case.root.on_quit = lambda: pending.update(jobs=object())
    case.pump(ready)
    assert observations == [True]
    assert "jobs" in pending
    assert case.root.callbacks == {}


def test_early_exit_cancels_tick_and_cannot_quit_the_next_pump(case):
    observations = []
    case.root.after(1, case.root.quit)
    with pytest.raises(AssertionError, match="event loop exited before UI state"):
        case.pump(lambda: observations.append(case.root.now) or False)
    assert case.root.now == 1
    assert case.root.callbacks == {}
    assert not case._pump_active
    previous_observations = list(observations)
    done = []
    case.root.after(20, lambda: done.append(True))
    case.pump(lambda: bool(done))
    assert done == [True]
    assert observations == previous_observations
    assert case.root.callbacks == {}


def test_root_destroyed_on_shutdown_can_finish_without_another_tick(case):
    stopped = []

    def close():
        stopped.append(True)
        case.root.destroy()

    case.root.after(1, close)
    case.pump(lambda: bool(stopped))
    assert case.root.destroyed
    assert case.root.now == 1
    assert case.root.callbacks == {}


def test_deadline_failure_has_state_and_is_not_rechecked_after_quit(case):
    condition = []
    case.app = SimpleNamespace(ready=True, loading=True, closing=False)
    case.controller = SimpleNamespace(pending={"documents": object()})
    case.root.on_quit = lambda: condition.append(True)
    with pytest.raises(AssertionError, match="Timed out waiting for UI state") as caught:
        case.pump(lambda: bool(condition), timeout=.02)
    assert case.root.now == 20
    assert "pending=['documents']" in str(caught.value)
    assert "'loading': True" in str(caught.value)
    assert case.root.callbacks == {}


def test_predicate_error_preserves_cause_and_stops_polling(case):
    error = ValueError("broken predicate")

    def broken():
        raise error

    with pytest.raises(ValueError) as caught:
        case.pump(broken)
    assert caught.value is error
    assert case.root.callbacks == {}
    assert not case._pump_active


def test_recursive_pump_is_rejected_without_leaving_callbacks(case):
    with pytest.raises(AssertionError, match="Nested TkCase.pump"):
        case.pump(lambda: case.pump(lambda: True))
    assert case.root.callbacks == {}
    assert not case._pump_active


def test_mainloop_exception_cleans_up_scheduled_tick(case, monkeypatch):
    error = RuntimeError("scheduler interrupted")

    def broken():
        raise error

    monkeypatch.setattr(case.root, "mainloop", broken)
    with pytest.raises(RuntimeError) as caught:
        case.pump(lambda: False)
    assert caught.value is error
    assert case.root.callbacks == {}
    assert not case._pump_active


def test_already_queued_callback_from_old_pump_is_inert_during_next_wait(case, monkeypatch):
    observations = []
    case.root.after(1, case.root.quit)

    def unavailable(_identifier):
        raise tk.TclError("Callback is no longer cancellable")

    with monkeypatch.context() as patch:
        patch.setattr(case.root, "after_cancel", unavailable)
        with pytest.raises(AssertionError, match="event loop exited before UI state"):
            case.pump(lambda: observations.append(case.root.now) or False)
    assert len(case.root.callbacks) == 1
    previous_observations = list(observations)
    done = []
    case.root.after(10, lambda: done.append(True))
    case.pump(lambda: bool(done))
    assert done == [True]
    assert observations == previous_observations
    assert case.root.callbacks == {}


def test_callback_error_is_reported_instead_of_mislabelled_as_timeout(case):
    case.root.after(1, lambda: case.callback_errors.append((RuntimeError, "callback failed", None)))
    with pytest.raises(AssertionError, match="callback failed"):
        case.pump(lambda: False)
    assert case.root.now == 5
    assert case.root.callbacks == {}


class Widget:
    def __init__(self, name, master=None, *, viewable=True, width=100, height=30):
        self.name, self.master = name, master
        self.viewable, self.width, self.height = viewable, width, height

    def __str__(self):
        return self.name

    def winfo_viewable(self):
        return self.viewable

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height


def test_mapping_waits_for_target_size_and_visible_ancestor(case):
    parent = Widget(".dialog", viewable=False)
    child = Widget(".dialog.button", parent, width=1)
    case.root.after(2, lambda: setattr(parent, "viewable", True))
    case.root.after(7, lambda: setattr(child, "width", 100))
    case.wait_mapped([child])
    assert case.root.now == 10
    assert case.root.callbacks == {}


def test_mapping_timeout_identifies_hidden_ancestor(case):
    parent = Widget(".hidden", viewable=False)
    child = Widget(".hidden.button", parent)
    with pytest.raises(AssertionError, match="widget/ancestor geometry") as caught:
        case.wait_mapped([child], timeout=.01)
    assert "'widget': '.hidden', 'viewable': False" in str(caught.value)
    assert case.root.callbacks == {}


def test_functional_wait_allows_slow_dispatch_and_records_actual_latency(case):
    ready = []
    case.root.after(5000, lambda: ready.append(True))
    case.pump(lambda: bool(ready))
    assert case._ui_waits[-1]["timeout"] == 10
    assert case._ui_waits[-1]["matched_at"] == 5
    assert case._ui_waits[-1]["elapsed"] == 5
    assert case._ui_waits[-1]["matched"] is True
    assert case._ui_waits[-1]["predicate"].startswith("test_tk_pump.py:")


def test_wait_measurement_distinguishes_matched_state_from_slow_loop_unwind(case):
    case.root.on_quit = lambda: setattr(case.root, "now", 1000)
    case.pump(lambda: True)
    assert case._ui_waits[-1]["matched_at"] == 0
    assert case._ui_waits[-1]["elapsed"] == 1


def test_wait_records_are_bounded(case):
    for _ in range(300):
        case.pump(lambda: True)
    assert len(case._ui_waits) == 256


def test_explicit_mapping_deadline_remains_a_strict_limit(case):
    child = Widget(".not-ready", viewable=False)
    case.root.after(300, lambda: setattr(child, "viewable", True))
    with pytest.raises(AssertionError, match="Timed out"):
        case.wait_mapped([child], timeout=.1)
    assert case._ui_waits[-1]["timeout"] == .1
    assert case._ui_waits[-1]["timed_out"] is True
    assert case.root.now == 100
