"""The real-data smoke uses the same event-loop evidence contract as GUI tests."""

import pytest

from scripts import pilot_gui_smoke as smoke
from tests.test_tk_pump import Scheduler


@pytest.fixture
def root(monkeypatch):
    scheduler = Scheduler()
    scheduler.winfo_exists = lambda: not scheduler.destroyed
    monkeypatch.setattr(smoke.time, "monotonic", lambda: scheduler.now / 1000)
    return scheduler


def test_smoke_latches_success_before_native_exit_changes_state(root):
    state = [True]
    root.on_quit = lambda: state.clear()
    smoke.wait_for_gui(root, lambda: bool(state), "ready", [])
    assert state == []
    assert root.callbacks == {}


def test_smoke_deadline_cannot_be_turned_into_success_after_quit(root):
    state = []
    root.on_quit = lambda: state.append(True)
    with pytest.raises(RuntimeError, match="deadline"):
        smoke.wait_for_gui(root, lambda: bool(state), "ready", [], timeout=.02)
    assert root.callbacks == {}


def test_smoke_cancels_stale_callback_after_unexpected_exit(root):
    root.after(1, root.quit)
    with pytest.raises(RuntimeError, match="event loop exited"):
        smoke.wait_for_gui(root, lambda: False, "ready", [])
    assert root.callbacks == {}


def test_smoke_accepts_shutdown_that_destroys_tk_before_the_next_tick(root):
    root.after(1, root.destroy)
    smoke.wait_for_gui(root, lambda: root.destroyed, "shutdown", [])
    assert root.callbacks == {}
