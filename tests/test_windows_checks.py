"""Portable startup-fence tests; native Job lifecycle runs through the runner on Windows."""

import subprocess
import sys

import pytest

from tools import windows_checks
from tools.run_checks import REPO


def marker_command(path):
    return [sys.executable, "-c", f"from pathlib import Path; Path({str(path)!r}).write_text('started')"]


@pytest.mark.parametrize("gate", [b"", b"X", b"G"])
def test_bootstrap_cannot_launch_command_before_valid_parent_permission(tmp_path, gate):
    marker = tmp_path / "started"
    process = subprocess.run([sys.executable, "-m", "tools.windows_checks", *marker_command(marker)],
                             cwd=REPO, input=gate, capture_output=True, timeout=5)
    assert process.returncode == (0 if gate == b"G" else 72)
    assert marker.exists() == (gate == b"G")


def test_owner_assigns_job_before_releasing_start_and_closes_it_after_completion(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    events = []

    class Job:
        def __init__(self, pid):
            assert pid > 0
            assert not marker.exists()
            events.append("assigned")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(windows_checks, "_job", Job)
    with windows_checks.owned_process(marker_command(marker), cwd=REPO, env=None, stdout=subprocess.DEVNULL) as process:
        assert process.wait(timeout=5) == 0
        assert marker.read_text(encoding="utf-8") == "started"
        assert events == ["assigned"]
    assert events == ["assigned", "closed"]


def test_failed_job_assignment_never_releases_command_and_reaps_bootstrap(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    processes = []
    popen = subprocess.Popen

    def track_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def failed_job(pid):
        raise OSError("Native protection unavailable")

    monkeypatch.setattr(windows_checks.subprocess, "Popen", track_process)
    monkeypatch.setattr(windows_checks, "_job", failed_job)
    with pytest.raises(OSError, match="Native protection unavailable"):
        with windows_checks.owned_process(marker_command(marker), cwd=REPO, env=None, stdout=subprocess.DEVNULL):
            raise AssertionError("The child must never be released")
    assert len(processes) == 1 and processes[0].poll() is not None
    assert not marker.exists()
