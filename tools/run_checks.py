"""Run all selected tests, isolating each GUI case in a bounded subprocess.

Usage: python -m tools.run_checks [--mode non-gui|gui|collect] [test paths/node IDs]
On Linux headless hosts, wrap the command with xvfb-run; macOS needs a GUI session.
"""

import argparse
import hashlib
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
import errno
import json
from importlib import metadata
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[1]
PLUGIN = "tests._checks_plugin"


def _timeout(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return number


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stop(process):
    """Reap the worker and its process group, including ML subprocesses."""
    if sys.platform == "win32":
        process.kill()
        process.wait(timeout=5)
        return  # The owning context closes the Job and all its descendants.
    deadline = time.monotonic() + 2
    for action in (signal.SIGTERM, signal.SIGKILL):
        while True:
            try:
                os.killpg(process.pid, action)
            except ProcessLookupError:
                break
            except PermissionError as error:
                # Darwin may still find a group after all its members become
                # zombies, then return EPERM because none can receive a signal.
                # Retry only an exited owned worker within the existing grace;
                # a live worker or persistent denial must remain a hard failure.
                if (sys.platform != "darwin" or error.errno != errno.EPERM
                        or process.poll() is None):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(.01, remaining))
                continue
            break
        if action == signal.SIGTERM:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        else:
            process.wait()


def run_process(command, log, timeout, env=None):
    started = time.monotonic()
    timed_out = False
    with log.open("w", encoding="utf-8") as handle, ExitStack() as owners:
        if sys.platform == "win32":
            from tools.windows_checks import owned_process

            process = owners.enter_context(owned_process(command, cwd=REPO, env=env, stdout=handle))
        else:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(process)
            returncode = process.returncode
        except BaseException:
            _stop(process)
            raise
        else:
            if sys.platform != "win32":
                # A finished test may leave live subprocesses behind. Match the
                # Windows Job's close-on-completion isolation on POSIX too.
                _stop(process)
    return {"command": command, "returncode": returncode, "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log)}


def validate_report(record, expected=None, *, catalog=None, collect=False, fail_on_skip=False):
    """A successful exit alone cannot prove the requested tests actually ran."""
    problems = []
    if record["timed_out"]:
        problems.append("subprocess exceeded its deadline")
    elif record["returncode"] != 0:
        problems.append(f"subprocess exited with code {record['returncode']}")
    try:
        report = json.loads(Path(record["report"]).read_text(encoding="utf-8"))
        if report["schema_version"] != 1:
            raise ValueError("unsupported report version")
        manifest = {row["nodeid"]: row["gui"] for row in report["manifest"]}
        selected = report["selected"]
        if len(manifest) != len(report["manifest"]) or len(selected) != len(set(selected)):
            problems.append("duplicate test node IDs")
        if expected is not None and set(selected) != set(expected):
            problems.append("selected tests differ from the planned set")
        if catalog is not None and manifest != catalog:
            problems.append("collection changed between subprocesses")
        if report["exitstatus"] != 0 or report["collection_errors"] or report["guard_violations"]:
            problems.append("pytest reported an error or GUI isolation violation")
        reports = report["reports"]
        if not collect:
            terminal = {row["nodeid"] for row in reports if row["when"] == "teardown"}
            called = {row["nodeid"] for row in reports if row["when"] == "call" or (
                row["when"] == "setup" and row["outcome"] in {"failed", "skipped"})}
            if terminal != set(selected) or called != set(selected):
                problems.append("not all planned tests produced complete reports")
        if any(row["outcome"] == "failed" for row in reports):
            problems.append("test or subtest failed")
        record["failed_tests"] = sorted({row["nodeid"] for row in reports if row["outcome"] == "failed"})
        counts = Counter(row["outcome"] for row in reports if row["when"] == "call" and not row["subtest"])
        counts["setup_skipped"] = sum(row["when"] == "setup" and row["outcome"] == "skipped" for row in reports)
        counts["collection_skipped"] = len(report["collection_skips"])
        counts["xfailed"] = sum(bool(row["wasxfail"]) and row["outcome"] == "skipped" for row in reports)
        counts["xpassed"] = sum(bool(row["wasxfail"]) and row["outcome"] == "passed" for row in reports)
        if fail_on_skip and (report["collection_skips"] or any(
                row["outcome"] == "skipped" or row["wasxfail"] for row in reports)):
            problems.append("strict run contains skipped or expected-failure tests")
        record.update(counts=dict(counts), warnings_count=len(report["warnings"]))
    except (OSError, ValueError, TypeError, KeyError) as error:
        problems.append(f"missing or invalid pytest report ({type(error).__name__})")
        report = None
    record["problems"] = problems
    record["ok"] = not problems
    return report


def source_fingerprint(repo: Path | None = None) -> str:
    """Hash runtime, launchers, resources and check inputs without Git metadata."""
    repo = REPO if repo is None else repo
    paths = [repo / 'pyproject.toml']
    attributes = repo / '.gitattributes'
    if attributes.is_file():
        paths.append(attributes)
    for name in ('app', 'scripts', 'tools', 'tests', 'requirements', 'resources', 'launchers', '.github'):
        paths.extend(path for path in (repo / name).rglob('*')
                     if '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}
                     and path.is_file())
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda path: path.relative_to(repo).as_posix()):
        digest.update(path.relative_to(repo).as_posix().encode('utf-8') + b'\0')
        if path.is_symlink():
            digest.update(b'symlink\0' + os.readlink(path).encode('utf-8'))
        else:
            with path.open('rb') as stream:
                digest.update(hashlib.file_digest(stream, 'sha256').digest())
    return digest.hexdigest()


def source_environment():
    """Record reproducible identities, never environment variables or credentials."""
    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=REPO, text=True,
                                    capture_output=True, timeout=5, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "working_tree_dirty": None if status is None else bool(status),
            "source_tree_sha256": source_fingerprint(),
            "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
            "dependencies": dict(sorted((distribution.metadata["Name"], distribution.version)
                                         for distribution in metadata.distributions() if distribution.metadata["Name"]))}


def interrupted_test(destination):
    """The flushed journal survives os._exit, native crashes and forced termination."""
    active = None
    try:
        with destination.with_suffix(".events.jsonl").open(encoding="utf-8") as events:
            for line in events:
                try:
                    event = json.loads(line)
                except ValueError:
                    break  # A hard kill may interrupt the final append.
                if event["event"] == "test_start":
                    active = event["nodeid"]
                elif event["event"] == "test_finish":
                    active = None
    except OSError:
        pass
    return active


class Runner:
    def __init__(self, args, directory):
        self.args, self.directory = args, directory
        self.records = []
        self.expected = []
        self.executed = set()
        self.excluded = [{"nodeid": nodeid, "reason": reason} for nodeid, reason in args.exclude_test]
        self.collected_count = 0
        self.error = None
        self.environment = source_environment()
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(REPO) + os.pathsep + self.env.get("PYTHONPATH", "")
        self.env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        self.env.pop("PYTEST_ADDOPTS", None)
        self.env.pop("PYTEST_PLUGINS", None)

    def pytest(self, label, mode, paths, *, expected=None, catalog=None):
        destination = self.directory / f"{label}.json"
        command = [sys.executable, "-m", "pytest", "-c", str(REPO / "pyproject.toml"),
                   "--rootdir", str(REPO), "-p", PLUGIN, "--checks-mode", mode,
                   "--checks-output", str(destination), "--strict-markers", "-q", *paths]
        for row in self.excluded:
            command.extend(["--checks-exclude", row["nodeid"]])
        if mode == "collect":
            command.append("--collect-only")
        timeout = 120 if mode == "collect" else (
            self.args.gui_timeout if mode == "gui" else self.args.suite_timeout)
        command.extend(["--checks-deadline", str(time.monotonic() + timeout)])
        print(f"{label}: starting {', '.join(paths)}", flush=True)
        record = run_process(command, self.directory / f"{label}.log", timeout, self.env)
        record.update(label=label, report=str(destination), stack_log=str(destination.with_suffix(".stacks.log")))
        report = validate_report(record, expected, catalog=catalog, collect=mode == "collect",
                                 fail_on_skip=self.args.fail_on_skip)
        self.records.append(record)
        if not record["ok"]:
            record["active_test"] = interrupted_test(destination)
            for problem in record["problems"]:
                print(f"{label}: {problem}", flush=True)
            for nodeid in record.get("failed_tests", []):
                print(f"FAILED {nodeid}", flush=True)
            if record["active_test"]:
                print(f"INTERRUPTED {record['active_test']}", flush=True)
        if mode != "collect" and report is not None:
            self.executed.update(row["nodeid"] for row in report["reports"] if row["when"] == "teardown")
        self.save()
        print(f"{label}: {'OK' if record['ok'] else 'FAIL'} ({record['elapsed_seconds']:.2f}s)", flush=True)
        return report, record["ok"]

    def save(self, error=None, exitstatus=None):
        if error is not None:
            self.error = error
        counts: Counter[str] = Counter()
        for record in self.records:
            if record["label"] != "collection":
                counts.update(record.get("counts", {}))
        if self.records:
            counts["collection_skipped"] = self.records[0].get("counts", {}).get("collection_skipped", 0)
        summary = {"mode": self.args.mode, "exitstatus": exitstatus,
                   "status": "running" if exitstatus is None else "passed" if exitstatus == 0 else "failed",
                   "expected": self.expected,
                   "not_completed": sorted(set(self.expected) - self.executed),
                   "counts": dict(counts), "records": self.records, "error": self.error,
                   "environment": self.environment,
                   "collected_count": self.collected_count, "excluded_tests": self.excluded,
                   "failed_tests": sorted({nodeid for row in self.records for nodeid in row.get("failed_tests", [])})}
        _write(self.directory / "summary.json", summary)

    def run(self):
        collected, ok = self.pytest("collection", "collect", self.args.paths)
        if not ok or collected is None or not collected["manifest"]:
            return 1
        catalog = {row["nodeid"]: row["gui"] for row in collected["manifest"]}
        self.collected_count = len(catalog)
        excluded = {row["nodeid"] for row in self.excluded}
        if (excluded - catalog.keys() or len(excluded) != len(self.excluded)
                or any(not row["reason"].strip() for row in self.excluded)):
            self.save("Exclusions require unique collected node IDs and a nonempty reason")
            return 1
        if self.args.mode == "collect":
            return 0
        self.expected = [nodeid for nodeid, gui in catalog.items() if nodeid not in excluded and (
            self.args.mode == "all" or gui == (self.args.mode == "gui"))]
        if not self.expected:
            self.save("No tests selected for this mode")
            return 1
        non_gui = [nodeid for nodeid in self.expected if not catalog[nodeid]]
        gui = [nodeid for nodeid in self.expected if catalog[nodeid]]
        if non_gui:
            self.pytest("non-gui", "non-gui", self.args.paths, expected=non_gui, catalog=catalog)
        if gui:
            preflight = run_process([sys.executable, "-c", (
                "import tkinter as tk, sys; r=tk.Tk(); r.withdraw(); "
                "print(sys.version); print(r.tk.call('info','patchlevel'), "
                "r.tk.call('tk','windowingsystem'), r.winfo_screenwidth(), r.winfo_screenheight()); r.destroy()"
            )], self.directory / "tk-preflight.log", 15, self.env)
            preflight.update(label="tk-preflight", ok=preflight["returncode"] == 0 and not preflight["timed_out"])
            self.records.append(preflight)
            self.save()
            if not preflight["ok"]:
                print("GUI preflight failed: use a desktop session or Linux xvfb-run; see tk-preflight.log.")
                return 1
            for index, nodeid in enumerate(gui, 1):
                self.pytest(f"gui-{index:03d}", "gui", [nodeid], expected=[nodeid], catalog={nodeid: True})
        self.save()
        return 0 if self.executed == set(self.expected) and all(row["ok"] for row in self.records) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "non-gui", "gui", "collect"), default="all")
    parser.add_argument("--output-dir", type=Path, default=REPO / "build/checks")
    parser.add_argument("--gui-timeout", type=_timeout, default=60)
    parser.add_argument("--suite-timeout", type=_timeout, default=900)
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--exclude-test", nargs=2, action="append", default=[], metavar=("NODEID", "REASON"),
                        help="Explicit platform-inapplicable test; exact ID and reason are retained in the report")
    parser.add_argument("paths", nargs="*", default=["tests"])
    args = parser.parse_args(argv)
    if os.name not in {"posix", "nt"}:
        parser.error("This isolated test runner supports Linux, macOS and Windows.")
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = args.output_dir.resolve() / run_id
    directory.mkdir(parents=True)
    print(f"Check reports: {directory}", flush=True)
    runner = Runner(args, directory)
    try:
        status = runner.run()
        runner.save(exitstatus=status)
        return status
    except KeyboardInterrupt:
        runner.save("Interrupted; the active subprocess group was stopped", exitstatus=130)
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        runner.save(f"Runner failed: {error}", exitstatus=2)
        print(f"Runner failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    def _interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)
    sys.exit(main())
