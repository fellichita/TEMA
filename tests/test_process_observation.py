"""Process liveness observations must not revive a reaped fixture process."""

from types import SimpleNamespace

import pytest

from tests import test_runtime_worker as worker_tests


@pytest.fixture
def process_observer(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    process = proc / "42"
    process.mkdir()
    stat = process / "stat"
    calls = []

    def observe(pid, signal):
        calls.append((pid, signal))

    def proc_path(value):
        assert value == "/proc" or value.startswith("/proc/")
        return proc / value.removeprefix("/proc").lstrip("/")

    operating_system = SimpleNamespace(name="posix", kill=observe)
    monkeypatch.setattr(worker_tests, "os", operating_system)
    monkeypatch.setattr(worker_tests, "Path", proc_path)
    return proc, stat, operating_system, calls


def test_process_observer_does_not_revive_a_zombie_reaped_between_probes(process_observer):
    _, stat, operating_system, calls = process_observer
    stat.write_text("42 (python) Z 1 42 42\n")

    def observe_then_reap(pid, signal):
        calls.append((pid, signal))
        if len(calls) == 2:
            # kill(pid, 0) succeeded, but the kernel reaped the zombie before
            # its stat file could be examined by the second observation.
            stat.unlink()

    operating_system.kill = observe_then_reap
    assert not worker_tests._pid_is_running(42)
    assert not worker_tests._pid_is_running(42)
    assert calls == [(42, 0), (42, 0)]


@pytest.mark.parametrize("comm", ["python", "owned) python"])
@pytest.mark.parametrize("state, expected", [("R", True), ("S", True), ("Z", False)])
def test_process_observer_reads_state_after_the_complete_command_name(process_observer, comm, state, expected):
    _, stat, _, _ = process_observer
    stat.write_text(f"42 ({comm}) {state} 1 42 42\n")
    assert worker_tests._pid_is_running(42) is expected


def test_process_observer_uses_signal_probe_on_systems_without_proc(process_observer):
    proc, stat, _, calls = process_observer
    stat.parent.rmdir()
    proc.rmdir()
    assert worker_tests._pid_is_running(42)
    assert calls == [(42, 0)]


def test_process_observer_rejects_missing_pid_before_reading_proc(process_observer):
    _, _, operating_system, _ = process_observer

    def missing(pid, signal):
        raise ProcessLookupError(pid)

    operating_system.kill = missing
    assert not worker_tests._pid_is_running(42)
