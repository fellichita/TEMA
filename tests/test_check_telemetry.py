"""Check diagnostics remain bounded and optional on non-Linux platforms."""

from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from tests import _checks_plugin as checks


def test_absent_cgroup_files_do_not_skip_or_fail_checks(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(Path, "read_text", missing)
    assert checks.CheckReport.cpu_limits() == {}


def test_only_fixed_scheduler_paths_are_read_and_values_are_bounded(monkeypatch):
    paths = []

    def contents(path, **kwargs):
        paths.append(path.as_posix())
        return "1" * 5000

    monkeypatch.setattr(Path, "read_text", contents)
    result = checks.CheckReport.cpu_limits()
    assert paths == ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat"]
    assert {key: len(value) for key, value in result.items()} == {"cpu.max": 4096, "cpu.stat": 4096}


def test_report_wait_records_are_bounded_and_only_attached_to_call_phase():
    plugin = checks.CheckReport.__new__(checks.CheckReport)
    item = SimpleNamespace(instance=SimpleNamespace(_ui_waits=[{"elapsed": .1}] * 300))
    for phase in ("setup", "call", "teardown"):
        report = SimpleNamespace(when=phase)
        hook = plugin.pytest_runtest_makereport(item, None)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(SimpleNamespace(get_result=lambda report=report: report))
        assert len(getattr(report, "ui_waits", ())) == (256 if phase == "call" else 0)


def test_report_clocks_are_not_replaced_by_application_time_patches(monkeypatch):
    monotonic, process_time = checks.monotonic, checks.process_time
    monkeypatch.setattr(time, "monotonic", lambda: -1)
    monkeypatch.setattr(time, "process_time", lambda: -1)
    assert checks.monotonic is monotonic
    assert checks.process_time is process_time
    assert checks.monotonic() >= 0
    assert checks.process_time() >= 0


def test_subtest_report_does_not_duplicate_parent_wait_observations():
    plugin = checks.CheckReport.__new__(checks.CheckReport)
    plugin.reports = []
    plugin.event = lambda *args, **kwargs: None
    report = SimpleNamespace(nodeid="case", when="call", outcome="passed", duration=.1,
                             ui_waits=[{"elapsed": .1}], context=object())
    plugin.pytest_runtest_logreport(report)
    del report.context
    plugin.pytest_runtest_logreport(report)
    assert "ui_waits" not in plugin.reports[0]
    assert plugin.reports[1]["ui_waits"] == [{"elapsed": .1}]
