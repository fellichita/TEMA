"""History polling must preserve the user's actionable cancellation state."""

from threading import Event
from types import SimpleNamespace

import pytest

from app.ui.trends_panel import TrendsPanel


def panel(*, requested=True, pending=False, accepted=False):
    value = TrendsPanel.__new__(TrendsPanel)
    value.app = SimpleNamespace(closing=False)
    value.collecting_id = "current-history"
    value.cancel_event = Event()
    if requested:
        value.cancel_event.set()
    value._cancel_pending, value._collection_cancelled = pending, accepted
    messages = []
    value.notice = SimpleNamespace(set=messages.append)
    return value, messages


def test_later_poll_failure_retains_retry_after_failed_cancellation():
    value, messages = panel()
    value.background_failed(OSError("private cancellation detail"), "Не удалось отменить сбор. Повторите отмену.")
    value.history_poll_failed(OSError("private database detail"), "current-history")
    assert "Повторите отмену" in messages[-1]
    assert "Проверка повторится автоматически" in messages[-1]
    assert "private" not in messages[-1]
    assert value.collecting_id == "current-history"


@pytest.mark.parametrize("state", [{"requested": False}, {"pending": True}, {"accepted": True}])
def test_poll_failure_does_not_offer_duplicate_or_unrequested_cancellation(state):
    value, messages = panel(**state)
    value.history_poll_failed(OSError("private detail"), "current-history")
    assert "Повторите отмену" not in messages[-1]
    assert "Проверка повторится автоматически" in messages[-1]


def test_old_collection_poll_cannot_change_current_cancellation_notice():
    value, messages = panel()
    value.history_poll_failed(OSError("private detail"), "previous-history")
    assert messages == []
