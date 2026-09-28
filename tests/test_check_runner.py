"""Exercise runner failures with real pytest subprocesses, without creating Tk."""

import json
import signal
import subprocess
import sys
import time

import pytest

from tools.run_checks import REPO, run_process, validate_report


def run_suite(tmp_path, source, *options):
    suite = tmp_path / "test_sample.py"
    suite.write_text(source, encoding="utf-8")
    output = tmp_path / "reports"
    child = subprocess.run([sys.executable, "-m", "tools.run_checks", "--mode", "non-gui",
                            "--output-dir", str(output), *options, str(suite)],
                           cwd=REPO, capture_output=True, text=True, timeout=30)
    reports = list(output.glob("*/summary.json"))
    assert len(reports) == 1, child.stdout + child.stderr
    return child, json.loads(reports[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", [
    "def test_pass(): assert True\n",
    "import unittest\nclass Cases(unittest.TestCase):\n def test_pass(self):\n"
    "  with self.subTest(value=1): self.assertEqual(1, 1)\n",
])
def test_complete_success_has_zero_exit_and_no_missing_tests(tmp_path, source):
    child, report = run_suite(tmp_path, source)
    assert child.returncode == 0, child.stdout + child.stderr
    assert report["exitstatus"] == 0 and report["status"] == "passed"
    assert report["not_completed"] == []
    assert report["counts"]["passed"] == 1


@pytest.mark.parametrize("source", [
    "def test_failure(): assert False\n",
    "def test_broken(:\n",
    "import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError('setup')\n"
    "def test_failure(broken): pass\n",
    "import pytest\n@pytest.fixture\ndef broken():\n yield\n raise RuntimeError('teardown')\n"
    "def test_failure(broken): pass\n",
    "import unittest\nclass Cases(unittest.TestCase):\n def test_failure(self):\n"
    "  with self.subTest(value=1): self.assertEqual(1, 2)\n",
    "import os\ndef test_abort(): os._exit(0)\n",
])
def test_failure_including_zero_exit_without_report_is_not_green(tmp_path, source):
    child, report = run_suite(tmp_path, source)
    assert child.returncode != 0
    assert report["exitstatus"] != 0 and report["status"] == "failed"
    assert any(not row["ok"] for row in report["records"])


def test_skip_is_reported_and_strict_skip_gate_fails(tmp_path):
    source = "import pytest\ndef test_optional(): pytest.skip('optional dependency')\n"
    child, report = run_suite(tmp_path, source)
    assert child.returncode == 0
    assert report["counts"]["skipped"] == 1
    strict = tmp_path / "strict"
    strict.mkdir()
    child, report = run_suite(strict, source, "--fail-on-skip")
    assert child.returncode != 0
    assert report["not_completed"] == []


@pytest.mark.parametrize("source", [
    "import pytest\n@pytest.mark.skip(reason='skipped setup')\ndef test_skip(): pass\n",
    "import pytest\n@pytest.mark.xfail(reason='known failure')\ndef test_xfail(): assert False\n",
    "import pytest\n@pytest.mark.xfail(reason='unexpected pass')\ndef test_xpass(): pass\n",
])
def test_strict_gate_rejects_setup_skip_xfail_and_xpass(tmp_path, source):
    child, report = run_suite(tmp_path, source, "--fail-on-skip")
    assert child.returncode != 0
    assert "strict run" in " ".join(report["records"][-1]["problems"])


def test_collection_skip_is_counted_once_and_guard_allows_headless_tcl(tmp_path):
    skipped = tmp_path / "test_optional.py"
    skipped.write_text("import pytest\npytest.skip('module optional', allow_module_level=True)\n")
    child, report = run_suite(tmp_path, "import tkinter as tk\ndef test_tcl():\n"
                              " assert tk.Tcl().eval('expr 1 + 1') == '2'\n", str(skipped))
    assert child.returncode == 0, child.stdout + child.stderr
    assert report["counts"]["collection_skipped"] == 1


def test_non_gui_guard_catches_missing_marker_before_display_access(tmp_path):
    # Even catching the RuntimeError cannot hide the isolation violation.
    child, report = run_suite(tmp_path, "import tkinter as tk\ndef test_unmarked():\n"
                              " try: tk.Tk()\n except RuntimeError: pass\n")
    assert child.returncode != 0
    assert "GUI isolation violation" in " ".join(report["records"][-1]["problems"])


def test_gui_marker_is_inherited_outside_ui_directory(tmp_path):
    suite = tmp_path / "test_other_module.py"
    suite.write_text("import pytest, unittest\n@pytest.mark.gui\nclass TkCase(unittest.TestCase): pass\n"
                     "class OutsideUI(TkCase):\n def test_gui(self): pass\n"
                     "def test_pure(): pass\n")
    output = tmp_path / "collection.json"
    child = subprocess.run([sys.executable, "-m", "pytest", "-p", "tests._checks_plugin",
                            "--checks-mode", "collect", "--checks-output", str(output),
                            "--collect-only", "-q", str(suite)], cwd=REPO,
                           capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stdout + child.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))["manifest"]
    assert {row["nodeid"].split("::")[-1]: row["gui"] for row in manifest} == {
        "test_gui": True, "test_pure": False}


def test_timeout_is_nonzero_and_reported(tmp_path):
    child, report = run_suite(tmp_path, "import time\ndef test_slow(): time.sleep(30)\n",
                              "--suite-timeout", "0.4")
    assert child.returncode != 0
    assert report["records"][-1]["timed_out"]
    assert report["not_completed"]


def test_timeout_preserves_active_node_and_thread_stack(tmp_path):
    child, report = run_suite(tmp_path, "import time\ndef test_blocked(): time.sleep(30)\n",
                              "--suite-timeout", "2")
    record = report["records"][-1]
    assert record["timed_out"]
    assert record["active_test"].endswith("::test_blocked")
    assert "INTERRUPTED" in child.stdout
    from pathlib import Path
    log = Path(record["stack_log"]).read_text(encoding="utf-8")
    assert "Timeout" in log and "test_blocked" in log
    events = [json.loads(line) for line in Path(record["report"]).with_suffix(".events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "test_start" and row["nodeid"].endswith("::test_blocked") for row in events)


def test_failure_summary_names_subtests_and_source_environment(tmp_path):
    child, report = run_suite(tmp_path, "import unittest\nclass Cases(unittest.TestCase):\n"
        " def test_parent(self):\n  with self.subTest(value=1): self.assertEqual(1, 2)\n")
    assert child.returncode == 1
    assert len(report["failed_tests"]) == 1
    assert report["failed_tests"][0].endswith("::Cases::test_parent")
    assert "FAILED" in child.stdout
    assert report["environment"]["python"]
    assert report["environment"]["dependencies"]["pytest"]
    assert "subprocess exited with code 1" in report["records"][-1]["problems"]
    assert not report["records"][-1]["timed_out"]


def test_report_clock_is_independent_of_application_clock_patches(tmp_path):
    child, report = run_suite(tmp_path, "import time\ndef test_clock(monkeypatch):\n"
        " values = iter([1])\n monkeypatch.setattr(time, 'monotonic', lambda: next(values))\n"
        " assert time.monotonic() == 1\n")
    assert child.returncode == 0, child.stdout + child.stderr
    assert report['counts']['passed'] == 1
    assert report['not_completed'] == []


@pytest.mark.parametrize("finish", ["normal", "timeout"])
def test_completion_stops_started_descendant_process_tree(tmp_path, finish):
    from tests.test_runtime_worker import _pid_is_running

    started = tmp_path / "descendant.pid"
    observed_alive = tmp_path / "observed-alive"
    stop = tmp_path / "stop-descendant"
    script = tmp_path / "spawn.py"
    descendant = ("import os, time\nfrom pathlib import Path\n"
                  f"marker = Path({str(started)!r})\n"
                  "temporary = marker.with_name(marker.name + '.tmp')\n"
                  "temporary.write_text(str(os.getpid()), encoding='ascii')\n"
                  "temporary.replace(marker)\n"
                  "deadline = time.monotonic() + 20\n"
                  f"while not Path({str(stop)!r}).exists() and time.monotonic() < deadline:\n"
                  " time.sleep(.01)\n")
    script.write_text("import subprocess, sys, time\n"
                      "from pathlib import Path\n"
                      f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
                      "deadline = time.monotonic() + 5\n"
                      f"while not Path({str(started)!r}).exists() and time.monotonic() < deadline:\n"
                      " time.sleep(.01)\n"
                      f"assert Path({str(started)!r}).exists() and child.poll() is None\n"
                      f"Path({str(observed_alive)!r}).write_text('alive')\n"
                      + ("time.sleep(30)\n" if finish == "timeout" else ""))
    try:
        result = run_process([sys.executable, str(script)], tmp_path / "child.log", 2)
        assert observed_alive.exists(), "The descendant must start before testing cleanup"
        assert result["timed_out"] == (finish == "timeout")
        if finish == "normal":
            assert result["returncode"] == 0
        pid = int(started.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 5
        while _pid_is_running(pid) and time.monotonic() < deadline:
            time.sleep(.025)
        assert not _pid_is_running(pid), f"A live descendant survived {finish} completion"
    finally:
        # Release the fixture itself if the containment assertion regresses.
        stop.touch()


def test_incomplete_zero_exit_report_is_rejected(tmp_path):
    output = tmp_path / "incomplete.json"
    output.write_text(json.dumps({"schema_version": 1, "exitstatus": 0,
        "manifest": [{"nodeid": "test_one", "gui": False}], "selected": ["test_one"],
        "reports": [], "warnings": [], "collection_errors": [], "collection_skips": [], "guard_violations": []}))
    record = {"returncode": 0, "timed_out": False, "report": str(output)}
    validate_report(record, ["test_one"])
    assert not record["ok"]
    assert "complete reports" in " ".join(record["problems"])


def test_each_run_uses_fresh_reports_in_same_output_parent(tmp_path):
    suite = tmp_path / "test_sample.py"
    suite.write_text("def test_pass(): pass\n")
    output = tmp_path / "reports"
    command = [sys.executable, "-m", "tools.run_checks", "--mode", "non-gui", "--output-dir", str(output), str(suite)]
    first = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=15)
    suite.write_text("import os\ndef test_pass(): os._exit(0)\n")
    second = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=15)
    assert first.returncode == 0
    assert second.returncode != 0
    assert len(list(output.glob("*/summary.json"))) == 2


def test_explicit_platform_exclusion_is_reported_and_does_not_weaken_strict_skip_gate(tmp_path):
    from pathlib import Path

    source = "def test_kept(): pass\ndef test_platform_only(): raise AssertionError('not applicable')\n"
    suite = tmp_path / "test_sample.py"
    suite.write_text(source)
    catalog = tmp_path / "catalog.json"
    collected = subprocess.run([sys.executable, "-m", "pytest", "-c", str(REPO / "pyproject.toml"),
                                "--rootdir", str(REPO), "-p", "tests._checks_plugin",
                                "--checks-mode", "collect", "--checks-output", str(catalog),
                                "--collect-only", "-q", str(suite)], cwd=REPO,
                               capture_output=True, text=True, timeout=15)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    excluded = next(row["nodeid"] for row in json.loads(catalog.read_text(encoding="utf-8"))["manifest"]
                    if row["nodeid"].endswith("::test_platform_only"))
    reason = "Native platform primitive is not available"
    child, report = run_suite(tmp_path, source, "--fail-on-skip", "--exclude-test", excluded, reason)
    assert child.returncode == 0, child.stdout + child.stderr
    assert report["excluded_tests"] == [{"nodeid": excluded, "reason": reason}]
    assert report["collected_count"] == 2
    assert report["counts"]["passed"] == 1
    assert report["not_completed"] == []
    assert len(report["expected"]) == 1 and report["expected"][0].endswith("::test_kept")
    manifest = json.loads(Path(report["records"][0]["report"]).read_text(encoding="utf-8"))["manifest"]
    assert excluded in {row["nodeid"] for row in manifest}


@pytest.mark.parametrize("reason", ["Platform-inapplicable primitive", ""])
def test_unknown_exclusion_cannot_silently_reduce_test_coverage(tmp_path, reason):
    child, report = run_suite(tmp_path, "def test_kept(): pass\n", "--exclude-test", "unknown.py::missing", reason)
    assert child.returncode != 0
    assert report["status"] == "failed"
    assert "Exclusions require" in report["error"]
    assert report["counts"].get("passed", 0) == 0


def test_interrupt_preserves_failure_and_stops_worker(tmp_path):
    suite = tmp_path / "test_signal.py"
    started = tmp_path / "started"
    suite.write_text(f"import os, time\nfrom pathlib import Path\ndef test_wait():\n"
                     f" marker = Path({str(started)!r})\n"
                     " temporary = marker.with_name(marker.name + '.tmp')\n"
                     " temporary.write_text(str(os.getpid()), encoding='ascii')\n"
                     " temporary.replace(marker)\n time.sleep(30)\n")
    output = tmp_path / "reports"
    child = subprocess.Popen([sys.executable, "-m", "tools.run_checks", "--mode", "non-gui",
                              "--output-dir", str(output), str(suite)], cwd=REPO,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert started.exists()
        if sys.platform == "win32":
            from tests.test_runtime_worker import _pid_is_running

            pid = int(started.read_text(encoding="utf-8"))
            assert _pid_is_running(pid)
            child.kill()  # Windows SIGTERM is termination; no Python handler runs.
            assert child.wait(timeout=8) != 0
            deadline = time.monotonic() + 5
            while _pid_is_running(pid) and time.monotonic() < deadline:
                time.sleep(.025)
            assert not _pid_is_running(pid), "The Job survived abrupt loss of its owning runner"
            report = json.loads(next(output.glob("*/summary.json")).read_text(encoding="utf-8"))
            assert report["status"] != "passed"
        else:
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=8) == 130
            report = json.loads(next(output.glob("*/summary.json")).read_text(encoding="utf-8"))
            assert "Interrupted" in report["error"]
            assert report["not_completed"]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
