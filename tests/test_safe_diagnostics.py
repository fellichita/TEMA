"""Internal errors stay diagnosable without publishing their private context."""

import logging
import json
import os
import zipfile
from concurrent.futures import CancelledError
from threading import Event, RLock
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app import diagnostics
from app.backend.contracts import SearchRequest
from app.backend.errors import BackendError
from app.backend.errors import CancelledError as BackendCancelledError
from app.ml.contracts import AnalysisInputError
from app.ui.controller import Controller
from tests.platform_support import require_symlinks

SECRET = "private-token-ABCD-do-not-log"


def captured_private_error():
    private_url = f"https://example.org/?token={SECRET}"
    try:
        raise ValueError(private_url)
    except ValueError as cause:
        try:
            raise RuntimeError(SECRET) from cause
        except RuntimeError as error:
            return error


def assert_private_data_absent(records):
    assert records
    for record in records:
        assert SECRET not in repr(record.__dict__)
        assert "https://" not in record.getMessage()
        assert record.exc_info is None and record.exc_text is None and record.stack_info is None
        assert record.args == ()
        assert "/Users/" not in record.getMessage()
        assert "raise RuntimeError" not in record.getMessage()


def test_diagnostic_contains_only_event_type_and_safe_stack(caplog):
    error = captured_private_error()
    assert error.__cause__ is not None and SECRET in str(error.__cause__)
    assert SECRET in error.__traceback__.tb_frame.f_locals["private_url"]
    with caplog.at_level(logging.ERROR, logger="app.diagnostics"):
        diagnostics.log_internal_error("ui.invoke_failed", error)
    assert_private_data_absent(caplog.records)
    message = caplog.records[0].getMessage()
    assert "event=ui.invoke_failed exception=RuntimeError" in message
    assert "test_safe_diagnostics.py:captured_private_error:" in message


def test_unknown_event_is_never_interpolated(caplog):
    diagnostics.log_internal_error(SECRET, RuntimeError(SECRET))
    assert_private_data_absent(caplog.records)
    assert "event=internal.error" in caplog.records[0].getMessage()


def test_diagnostics_never_calls_exception_str(caplog):
    class UnprintableError(Exception):
        def __str__(self):
            raise AssertionError("Exception text must not be inspected")

    diagnostics.log_internal_error("ui.invoke_failed", UnprintableError(SECRET))
    assert_private_data_absent(caplog.records)
    assert "exception=UnprintableError" in caplog.records[0].getMessage()


def test_compiled_traceback_does_not_expose_full_filename_or_source(caplog):
    namespace = {"SECRET": SECRET}
    try:
        exec(compile("raise RuntimeError(SECRET)", f"/private/{SECRET}/worker.py", "exec"), namespace)  # noqa: S102 - fixed fixture code
    except RuntimeError as error:
        diagnostics.log_internal_error("ui.invoke_failed", error)
    assert_private_data_absent(caplog.records)
    assert "worker.py:<module>:1" in caplog.records[0].getMessage()


def test_stack_size_is_bounded(caplog):
    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        raise RuntimeError(SECRET)

    try:
        recurse(100)
    except RuntimeError as error:
        diagnostics.log_internal_error("ui.invoke_failed", error)
    assert_private_data_absent(caplog.records)
    message = caplog.records[0].getMessage()
    assert message.endswith(" > truncated")
    assert message.count(" > ") == 64


def test_invalid_source_url_is_an_expected_user_error(caplog):
    controller = Controller.__new__(Controller)
    with pytest.raises(AnalysisInputError):
        controller._invoke("open_url", (f"https://user:{SECRET}@example.org",), {})
    assert caplog.records == []


def validation_error():
    try:
        SearchRequest.model_validate({"unknown": SECRET})
    except ValidationError as error:
        return error
    raise AssertionError("Invalid request unexpectedly passed validation")


@pytest.mark.parametrize("error", [
    BackendError("public", SECRET), BackendCancelledError(), CancelledError(SECRET),
    AnalysisInputError(SECRET), validation_error(),
])
@pytest.mark.parametrize("operation", ["invoke", "close"])
def test_expected_errors_are_preserved_without_internal_crash_record(caplog, error, operation):
    def fail(*args, **kwargs):
        raise error

    controller = Controller.__new__(Controller)
    controller.backend = SimpleNamespace(execute=fail, close=fail)
    controller.pilot, controller.pilot_lock = None, RLock()
    controller.ml_executor = SimpleNamespace(shutdown=lambda **kwargs: None)
    controller.read_executor = SimpleNamespace(shutdown=lambda **kwargs: None)
    with pytest.raises(type(error)) as caught:
        if operation == "invoke":
            controller._invoke("execute", (), {})
        else:
            controller._close_backend()
    assert caught.value is error
    assert caplog.records == []


@pytest.mark.parametrize("operation", ["invoke", "close"])
def test_broken_logging_cannot_replace_the_operation_error(monkeypatch, operation):
    original = captured_private_error()

    def fail(*args, **kwargs):
        raise original

    def broken_handler(*args, **kwargs):
        raise OSError("Broken logging handler " + SECRET)

    monkeypatch.setattr(diagnostics._LOGGER, "error", broken_handler)
    controller = Controller.__new__(Controller)
    controller.backend = SimpleNamespace(execute=fail, close=fail)
    controller.pilot, controller.pilot_lock = None, RLock()
    controller.ml_executor = SimpleNamespace(shutdown=lambda **kwargs: None)
    controller.read_executor = SimpleNamespace(shutdown=lambda **kwargs: None)
    with pytest.raises(RuntimeError) as caught:
        if operation == "invoke":
            controller._invoke("execute", (), {})
        else:
            controller._close_backend()
    assert caught.value is original


class Scheduler:
    def __init__(self):
        self.scheduled = []

    def after(self, milliseconds, callback):
        self.scheduled.append((milliseconds, callback))


def test_worker_reports_failure_even_when_closing_skips_the_ui_callback(caplog):
    entered, release = Event(), Event()
    original = captured_private_error()

    def fail():
        entered.set()
        assert release.wait(5)
        raise original

    controller = Controller(Scheduler())
    controller.backend = SimpleNamespace(execute=fail, close=lambda: None)
    callbacks = []
    try:
        assert controller.call("work", "execute", callbacks.append, callbacks.append)
        assert entered.wait(5)
        future = controller.pending["work"][0]
        controller.close(lambda: callbacks.append("closed"), callbacks.append)
        release.set()
        with pytest.raises(RuntimeError) as caught:
            future.result(timeout=5)
        assert caught.value is original
        controller.close_future.result(timeout=5)
        assert len(caplog.records) == 1
        assert caplog.records[0].threadName.startswith("ui-backend")
        assert "app/ui/controller.py:_invoke:" in caplog.records[0].getMessage()
        assert_private_data_absent(caplog.records)
        controller._poll()
        assert callbacks == ["closed"]
        assert controller.stopped and controller.pending == {}
        assert len(caplog.records) == 1  # polling cannot duplicate worker logs
    finally:
        release.set()
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)
        controller.executor.shutdown(wait=True, cancel_futures=True)


def test_close_worker_reports_error_before_poll_and_preserves_failure_callback(caplog):
    original = captured_private_error()

    def fail():
        raise original

    controller = Controller(Scheduler())
    controller.backend = SimpleNamespace(close=fail)
    callbacks = []
    try:
        controller.close(lambda: callbacks.append("closed"), callbacks.append)
        with pytest.raises(RuntimeError):
            controller.close_future.result(timeout=5)
        assert len(caplog.records) == 1
        assert "event=ui.close_failed" in caplog.records[0].getMessage()
        assert caplog.records[0].threadName.startswith("ui-backend")
        assert_private_data_absent(caplog.records)
        controller._poll()
        assert callbacks == [original]
        assert not controller.stopped
        assert len(caplog.records) == 1
    finally:
        controller.ml_executor.shutdown(wait=True, cancel_futures=True)
        controller.read_executor.shutdown(wait=True, cancel_futures=True)
        controller.executor.shutdown(wait=True, cancel_futures=True)


def test_worker_wire_details_exclude_private_context_and_preserve_task_line(caplog):
    error = captured_private_error()
    details = diagnostics.worker_error_details(error, captured_private_error)
    assert details == {"exception": "RuntimeError", "line": error.__traceback__.tb_lineno}
    wire = json.dumps({"status": "error", "code": "task_failed", "diagnostic": details}).encode("ascii")
    assert len(wire) < 1024
    assert SECRET.encode() not in wire
    assert str(__file__).encode() not in wire
    correlation = "abcdef0123456789" * 2
    diagnostics.log_worker_error("task_failed", correlation, details, captured_private_error)
    assert_private_data_absent(caplog.records)
    assert "diagnostic_id=" + correlation in caplog.records[0].getMessage()
    assert "test_safe_diagnostics.py:captured_private_error:" in caplog.records[0].getMessage()


def test_worker_unknown_exception_name_is_replaced_by_its_builtin_base():
    exception_type = type("private_token_do_not_log", (ValueError,), {})
    details = diagnostics.worker_error_details(exception_type(SECRET), captured_private_error)
    assert details == {"exception": "ValueError", "line": None}
    assert "private_token" not in json.dumps(details)


@pytest.mark.parametrize("details", [
    {"exception": SECRET, "line": None},
    {"exception": "RuntimeError", "line": True},
    {"exception": "RuntimeError", "line": 1_000_000},
    {"exception": "RuntimeError", "line": "1"},
    {"exception": "RuntimeError", "line": None, "filename": "/private/" + SECRET},
    {"exception": "RuntimeError", "line": None, "message": SECRET},
    {"exception": "RuntimeError", "line": [SECRET]},
    [SECRET],
])
def test_parent_rejects_worker_details_outside_its_own_code_and_type_tables(details, caplog):
    with pytest.raises(ValueError):
        diagnostics.validate_worker_details(details, captured_private_error)
    diagnostics.log_worker_error("task_failed", "0" * 32, details, captured_private_error)
    assert caplog.records == []


@pytest.mark.parametrize("code,correlation", [(SECRET, "0" * 32), ("task_failed", SECRET),
                                            ("task_failed", "x" * 32), ("task_failed", "0" * 33)])
def test_worker_events_do_not_accept_free_form_code_or_correlation(code, correlation, caplog):
    details = {"exception": "ValueError", "line": None}
    diagnostics.log_worker_error(code, correlation, details, captured_private_error)
    assert caplog.records == []


@pytest.fixture
def private_logs(tmp_path):
    diagnostics.close_private_logging()
    try:
        yield tmp_path
    finally:
        diagnostics.close_private_logging()


def test_private_file_sink_only_records_safe_diagnostics_and_uses_private_permissions(private_logs):
    assert diagnostics.configure_private_logging(private_logs)
    diagnostics.log_internal_error("ui.invoke_failed", captured_private_error())
    logging.getLogger("third_party.transport").error(SECRET)
    diagnostics._LOGGER.error(SECRET)
    path = private_logs / "logs" / "diagnostics.log"
    content = path.read_text(encoding="utf-8")
    assert "event=ui.invoke_failed exception=RuntimeError" in content
    assert SECRET not in content
    assert "https://" not in content
    assert str(private_logs) not in content
    assert len(content.splitlines()) == 1
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_private_file_rotation_bounds_disk_use_and_preserves_permissions(private_logs, monkeypatch):
    monkeypatch.setattr(diagnostics, "_LOG_BYTES", 700)
    monkeypatch.setattr(diagnostics, "_LOG_BACKUPS", 2)
    assert diagnostics.configure_private_logging(private_logs)
    for _ in range(20):
        diagnostics.log_internal_error("ui.invoke_failed", RuntimeError(SECRET))
    paths = sorted((private_logs / "logs").iterdir())
    assert [path.name for path in paths] == ["diagnostics.log", "diagnostics.log.1", "diagnostics.log.2"]
    assert all(0 < path.stat().st_size <= 700 for path in paths)
    assert all(SECRET not in path.read_text(encoding="utf-8") for path in paths)
    if os.name != "nt":
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)


def test_profile_switch_closes_previous_sink_and_does_not_append_to_previous_library(private_logs):
    original, restored = private_logs / "original", private_logs / "restored"
    original.mkdir()
    restored.mkdir()
    assert diagnostics.configure_private_logging(original)
    previous = diagnostics._PRIVATE_HANDLER
    diagnostics.log_internal_error("ui.invoke_failed", RuntimeError(SECRET))
    before = (original / "logs" / "diagnostics.log").read_bytes()
    assert diagnostics.configure_private_logging(restored)
    assert previous.stream is None
    diagnostics.log_internal_error("ui.close_failed", ValueError(SECRET))
    assert (original / "logs" / "diagnostics.log").read_bytes() == before
    assert "event=ui.close_failed exception=ValueError" in (restored / "logs" / "diagnostics.log").read_text(encoding="utf-8")
    current = diagnostics._PRIVATE_HANDLER
    assert diagnostics.configure_private_logging(restored)
    assert diagnostics._PRIVATE_HANDLER is current
    diagnostics.close_private_logging()
    assert current.stream is None
    assert diagnostics._PRIVATE_HANDLER is None


def test_broken_new_sink_is_nonfatal_and_does_not_keep_writing_to_old_profile(private_logs):
    original, broken = private_logs / "original", private_logs / "broken"
    original.mkdir()
    broken.mkdir()
    (broken / "logs").write_text("not a directory")
    assert diagnostics.configure_private_logging(original)
    previous = diagnostics._PRIVATE_HANDLER
    assert not diagnostics.configure_private_logging(broken)
    assert previous.stream is None
    assert diagnostics._PRIVATE_HANDLER is None
    diagnostics.log_internal_error("ui.invoke_failed", RuntimeError(SECRET))
    assert (original / "logs" / "diagnostics.log").read_text(encoding="utf-8") == ""


def test_private_sink_write_failure_does_not_print_exception_or_change_operation(private_logs, monkeypatch, capsys):
    assert diagnostics.configure_private_logging(private_logs)

    def broken_rollover(_record):
        raise OSError(SECRET)

    monkeypatch.setattr(diagnostics._PRIVATE_HANDLER, "shouldRollover", broken_rollover)
    diagnostics.log_internal_error("ui.invoke_failed", captured_private_error())
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert "Logging error" not in captured.err
    assert (private_logs / "logs" / "diagnostics.log").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("target", ["directory", "file"])
def test_private_sink_does_not_follow_links(private_logs, target, monkeypatch):
    require_symlinks()
    outside = private_logs / "outside"
    outside.mkdir()
    outside_file = outside / "keep.log"
    outside_file.write_text("existing unrelated content")
    library = private_logs / "library"
    library.mkdir()
    logs = library / "logs"
    if os.name == "nt":
        # Creating Windows links needs privileges; exercise the same policy
        # boundary with an explicit link observation, without claiming creation.
        logs.mkdir()
        original = type(logs).is_symlink
        linked = logs if target == "directory" else logs / "diagnostics.log"
        monkeypatch.setattr(type(logs), "is_symlink", lambda path: path == linked or original(path))
    elif target == "directory":
        logs.symlink_to(outside, target_is_directory=True)
    else:
        logs.mkdir()
        (logs / "diagnostics.log").symlink_to(outside_file)
    assert not diagnostics.configure_private_logging(library)
    assert diagnostics._PRIVATE_HANDLER is None
    assert outside_file.read_text(encoding="utf-8") == "existing unrelated content"
    assert list(outside.iterdir()) == [outside_file]


def test_diagnostics_are_not_included_in_result_export_or_library_backup(private_logs):
    from app.backend.repository import Repository
    from app.pilot.export import export_result
    from app.runtime.backup import BackupSession, create_backup
    from app.runtime.jobs import Coordinator
    from tests.test_pilot_export import make_result

    Repository(private_logs / "documents.sqlite3")
    coordinator = Coordinator(private_logs, lambda context, payload: None)
    coordinator.close()
    result, archive, artifacts = make_result(private_logs)
    assert diagnostics.configure_private_logging(private_logs)
    diagnostics.log_internal_error("ui.invoke_failed", captured_private_error())
    diagnostics.close_private_logging()
    exported = export_result(private_logs / "result.trendresult", result, archive, artifacts)
    with BackupSession(private_logs) as session:
        backup = create_backup(session, private_logs / "backups")
    for path in (exported.path, backup.path):
        with zipfile.ZipFile(path) as package:
            assert not any(name.startswith("logs/") for name in package.namelist())
            assert b"event=ui.invoke_failed" not in b"".join(package.read(name) for name in package.namelist())
