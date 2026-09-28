"""Darwin's exiting groups must not hide real signal-permission failures."""

import errno
from types import SimpleNamespace

import pytest

from tools import run_checks


class _Worker:
    pid = 12345  # A double: no real signal is sent to this or any discovered PID.

    def __init__(self, returncode=-15):
        self.returncode = returncode
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.returncode


def _clock(monkeypatch):
    # Supply POSIX signal constants even when these doubles run on Windows.
    monkeypatch.setattr(run_checks, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    values = {"now": 0.0, "sleeps": []}

    def sleep(seconds):
        assert seconds > 0
        values["sleeps"].append(seconds)
        values["now"] += seconds

    monkeypatch.setattr(run_checks, "time", SimpleNamespace(monotonic=lambda: values["now"], sleep=sleep))
    return values


@pytest.mark.parametrize("action", [15, 9])
@pytest.mark.parametrize("settled", ["delivered", "gone"])
def test_exited_darwin_group_must_eventually_deliver_signal_or_disappear(monkeypatch, action, settled):
    worker = _Worker()
    clock = _clock(monkeypatch)
    monkeypatch.setattr(run_checks, "sys", SimpleNamespace(platform="darwin"))
    calls = []
    attempts = 0

    def killpg(pid, observed):
        nonlocal attempts
        assert pid == worker.pid
        calls.append(observed)
        if observed == action:
            attempts += 1
            if attempts <= 2:
                raise PermissionError(errno.EPERM, "Zombie-only group still exists")
            if settled == "gone":
                raise ProcessLookupError(errno.ESRCH, "Owned group was reaped")

    monkeypatch.setattr(run_checks, "os", SimpleNamespace(killpg=killpg))
    run_checks._stop(worker)
    assert attempts == 3
    assert calls == ([action] * 3 + [9] if action == 15
                     else [15] + [action] * 3)
    assert len(clock["sleeps"]) == 2
    assert worker.waits == [2, None]


@pytest.mark.parametrize("action", [15, 9])
def test_persistent_darwin_permission_failure_remains_failure_with_bounded_retry(monkeypatch, action):
    worker = _Worker()
    clock = _clock(monkeypatch)
    monkeypatch.setattr(run_checks, "sys", SimpleNamespace(platform="darwin"))
    denial = PermissionError(errno.EPERM, "Real denial must not be ignored")
    attempts = 0

    def killpg(pid, observed):
        nonlocal attempts
        assert pid == worker.pid
        if observed == action:
            attempts += 1
            raise denial

    monkeypatch.setattr(run_checks, "os", SimpleNamespace(killpg=killpg))
    with pytest.raises(PermissionError) as captured:
        run_checks._stop(worker)
    assert captured.value is denial
    assert attempts > 1
    assert clock["now"] == pytest.approx(2.0)
    assert max(clock["sleeps"]) <= .01
    assert None not in worker.waits


def test_retry_does_not_restart_grace_after_waiting_for_leader(monkeypatch):
    worker = _Worker(None)
    clock = _clock(monkeypatch)
    monkeypatch.setattr(run_checks, "sys", SimpleNamespace(platform="darwin"))

    def wait(timeout=None):
        assert timeout == 2
        clock["now"] += 1.75
        worker.returncode = -15
        return worker.returncode

    def killpg(pid, action):
        assert pid == worker.pid
        if action == 9:
            raise PermissionError(errno.EPERM, "Exiting group still present")

    worker.wait = wait
    monkeypatch.setattr(run_checks, "os", SimpleNamespace(killpg=killpg))
    with pytest.raises(PermissionError):
        run_checks._stop(worker)
    assert sum(clock["sleeps"]) == pytest.approx(.25)
    assert clock["now"] == pytest.approx(2.0)


@pytest.mark.parametrize("platform,returncode,error_number", [
    ("darwin", None, errno.EPERM), ("linux", -15, errno.EPERM), ("darwin", -15, errno.EACCES),
])
def test_unrelated_permission_failures_are_never_retried(monkeypatch, platform, returncode, error_number):
    worker = _Worker(returncode)
    clock = _clock(monkeypatch)
    monkeypatch.setattr(run_checks, "sys", SimpleNamespace(platform=platform))
    denial = PermissionError(error_number, "Signal refused")
    calls = []

    def killpg(pid, action):
        calls.append((pid, action))
        raise denial

    monkeypatch.setattr(run_checks, "os", SimpleNamespace(killpg=killpg))
    with pytest.raises(PermissionError) as captured:
        run_checks._stop(worker)
    assert captured.value is denial
    assert calls == [(worker.pid, 15)]
    assert clock["sleeps"] == []
    assert worker.waits == []


def test_windows_cleanup_keeps_its_owned_job_path(monkeypatch):
    worker = _Worker()
    calls = []
    worker.kill = lambda: calls.append("kill")
    monkeypatch.setattr(run_checks, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(run_checks, "os", SimpleNamespace())
    run_checks._stop(worker)
    assert calls == ["kill"] and worker.waits == [5]
