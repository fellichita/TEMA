"""Machine-readable pytest reports and explicit GUI isolation for run_checks."""

import faulthandler
import json
from pathlib import Path
from time import monotonic, process_time
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("project checks")
    group.addoption("--checks-mode", choices=("collect", "non-gui", "gui"))
    group.addoption("--checks-output")
    group.addoption("--checks-deadline", type=float)
    group.addoption("--checks-exclude", action="append", default=[])


def pytest_configure(config):
    config.addinivalue_line("markers", "gui: requires a Tk display and an isolated process")
    config.pluginmanager.register(CheckReport(config), "project-check-report")


class CheckReport:
    def __init__(self, config):
        self.config = config
        self.mode = config.getoption("--checks-mode")
        self.manifest = []
        self.selected = []
        self.reports = []
        self.warnings = []
        self.collection_errors = []
        self.collection_skips = []
        self.guard_violations = []
        self.restore_tk = None
        self.started = monotonic()
        self.cpu_started = process_time()
        destination = config.getoption("--checks-output")
        self.events = (Path(destination).with_suffix(".events.jsonl").open("w", encoding="utf-8")
                       if destination else None)
        self.stacks = (Path(destination).with_suffix(".stacks.log").open("w", encoding="utf-8")
                       if destination else None)

    def arm_stack_dump(self):
        deadline = self.config.getoption("--checks-deadline")
        if deadline is not None and self.stacks is not None:
            faulthandler.dump_traceback_later(max(.01, deadline - monotonic() - .25), file=self.stacks)

    def event(self, kind, **details):
        if self.events is not None:
            self.events.write(json.dumps({"event": kind, "monotonic": monotonic(), **details},
                                         ensure_ascii=False) + "\n")
            self.events.flush()

    @staticmethod
    def cpu_limits():
        """Optional Linux scheduler evidence; no environment values or paths."""
        values = {}
        for name in ("cpu.max", "cpu.stat"):
            try:
                values[name] = Path("/sys/fs/cgroup", name).read_text(encoding="ascii")[:4096]
            except (OSError, UnicodeError):
                pass
        return values

    def pytest_sessionstart(self, session):
        self.event("session_start", mode=self.mode, cpu_limits=self.cpu_limits())
        self.arm_stack_dump()
        if self.mode not in {"collect", "non-gui"}:
            return
        import tkinter as tk

        original = tk.Tk.__init__

        def guarded(instance, *args, **kwargs):
            use_tk = kwargs.get("useTk", args[3] if len(args) > 3 else True)
            if use_tk:
                message = "Graphical Tk created outside an isolated GUI test; add pytest.mark.gui."
                self.guard_violations.append(message)
                raise RuntimeError(message)
            original(instance, *args, **kwargs)

        patcher = patch.object(tk.Tk, "__init__", guarded)
        patcher.start()
        self.restore_tk = patcher.stop

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        self.manifest = [{"nodeid": item.nodeid, "gui": item.get_closest_marker("gui") is not None}
                         for item in items]
        if self.mode in {"non-gui", "gui"}:
            wanted = self.mode == "gui"
            excluded = set(self.config.getoption("--checks-exclude"))
            kept = [item for item, row in zip(items, self.manifest, strict=True)
                    if row["gui"] == wanted and item.nodeid not in excluded]
            dropped = [item for item, row in zip(items, self.manifest, strict=True)
                       if row["gui"] != wanted or item.nodeid in excluded]
            items[:] = kept
            self.config.hook.pytest_deselected(items=dropped)
        self.selected = [item.nodeid for item in items]
        self.event("collected", selected=self.selected)

    def pytest_runtest_logstart(self, nodeid, location):
        self.event("test_start", nodeid=nodeid)
        self.arm_stack_dump()

    def pytest_runtest_logfinish(self, nodeid, location):
        self.event("test_finish", nodeid=nodeid)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(str(report.longrepr))
        elif report.skipped:
            self.collection_skips.append(str(report.longrepr))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when == "call":
            report.ui_waits = list(getattr(getattr(item, "instance", None), "_ui_waits", ()))[:256]

    def pytest_runtest_logreport(self, report):
        row = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
                             "subtest": hasattr(report, "context"),
                             "wasxfail": getattr(report, "wasxfail", None),
                             "duration": report.duration}
        waits = getattr(report, "ui_waits", None)
        if waits and not row["subtest"]:
            row["ui_waits"] = waits
        self.reports.append(row)
        self.event("report", **row)

    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        self.warnings.append({"message": str(warning_message.message), "when": when, "nodeid": nodeid})

    def pytest_sessionfinish(self, session, exitstatus):
        self.event("session_finish", exitstatus=int(exitstatus),
                   elapsed=monotonic() - self.started, process_cpu=process_time() - self.cpu_started,
                   cpu_limits=self.cpu_limits())
        destination = self.config.getoption("--checks-output")
        if destination:
            Path(destination).write_text(json.dumps({
                "schema_version": 1, "mode": self.mode, "exitstatus": int(exitstatus),
                "manifest": self.manifest, "selected": self.selected, "reports": self.reports,
                "warnings": self.warnings, "collection_errors": self.collection_errors,
                "collection_skips": self.collection_skips, "guard_violations": self.guard_violations,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pytest_unconfigure(self, config):
        if self.config.getoption("--checks-deadline") is not None:
            faulthandler.cancel_dump_traceback_later()
        if self.stacks is not None:
            self.stacks.close()
        if self.events is not None:
            self.events.close()
        if self.restore_tk is not None:
            self.restore_tk()
