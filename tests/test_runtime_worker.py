"""Real spawn processes exercise result integrity, failure and parent loss."""

import ctypes
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from app.runtime.credentials import LEGACY_ENVIRONMENT, CredentialStore
from app.runtime import worker
from app.runtime.worker import WorkerCancelled, WorkerError, WorkerTimeout, run_in_process


# These importable top-level functions are trusted test fixtures, never product
# implementations or names supplied by an input document.
def _success_task(input_path, output_path, cancel):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.write_text(json.dumps({"answer": data["value"] * 2}), encoding="utf-8")


def _error_task(input_path, output_path, cancel):
    print("must-not-escape-private-token")
    print("must-not-escape-private-token", file=sys.stderr)
    raise RuntimeError("must-not-escape-private-token")


def _crash_task(input_path, output_path, cancel):
    os._exit(73)


def _teardown_crash_task(input_path, output_path, cancel):
    import atexit

    # CUDA and ONNX Runtime can crash while unloading at interpreter exit,
    # after the manifest is validated and "ok" has already been sent.
    atexit.register(os._exit, 3)
    _success_task(input_path, output_path, cancel)


def _busy_task(input_path, output_path, cancel):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if "started" in data:
        marker = Path(data["started"])
        temporary = marker.with_name(marker.name + ".tmp")
        temporary.write_text(str(os.getpid()), encoding="ascii")
        temporary.replace(marker)  # exists() means the entire PID is readable.
    # Native work releases the GIL as ONNX does. The pipe watcher must still
    # terminate this process on EOF; the task deliberately ignores cancellation.
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32")
        kernel.Sleep(120_000)
    else:
        library = ctypes.CDLL(None)
        library.sleep.argtypes = [ctypes.c_uint]
        library.sleep(120)


def _progress_task(input_path, output_path, cancel):
    from app.runtime.worker import report_progress

    total = json.loads(input_path.read_text(encoding="utf-8"))["value"]
    for completed in range(total + 1):
        # Throttling keeps intermediate reports optional; the last one is not.
        time.sleep(0.05)
        report_progress(completed, total)
    report_progress(-1, total)  # Rejected in the child, never sent.
    output_path.write_text(json.dumps({"answer": total}), encoding="utf-8")


def _invalid_task(input_path, output_path, cancel):
    output_path.write_text(json.loads(input_path.read_text(encoding="utf-8"))["output"], encoding="utf-8")


def _returned_object_task(input_path, output_path, cancel):
    output_path.write_text('{"ok": true}', encoding="utf-8")
    return {"array": [1, 2, 3]}


def _no_manifest_task(input_path, output_path, cancel):
    pass


def _clean_boundary_task(input_path, output_path, cancel):
    assert not any(name in os.environ for name in LEGACY_ENVIRONMENT)
    assert not any("child-secret-marker-" + "982" in value for value in sys.argv)
    assert "tkinter" not in sys.modules
    assert "app.ui.window" not in sys.modules
    assert isinstance(input_path, Path) and isinstance(output_path, Path)
    assert isinstance(cancel.is_set(), bool)
    output_path.write_text('{"clean": true}', encoding="utf-8")


def _symlink_manifest_task(input_path, output_path, cancel):
    output_path.symlink_to(input_path)


def _shutdown_busy_task(input_path, output_path, cancel):
    # A library may leave a non-daemon thread behind after returning its result.
    threading.Thread(target=_busy_task, args=(input_path, output_path, cancel), daemon=False).start()
    output_path.write_text('{"ready":true}', encoding="utf-8")


def _parent_process_fixture(input_path, output_path):
    store = CredentialStore()
    store.import_legacy_environment()
    run_in_process(
        _busy_task, Path(input_path), Path(output_path), threading.Event(), credentials=store,
        timeout_seconds=60,
    )


def _parent_shutdown_fixture(input_path, output_path):
    store = CredentialStore()
    store.import_legacy_environment()
    run_in_process(
        _shutdown_busy_task, Path(input_path), Path(output_path), threading.Event(), credentials=store,
        timeout_seconds=60,
    )


@pytest.fixture
def runtime(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text('{"value": 21}', encoding="utf-8")
    store = CredentialStore()
    try:
        yield input_path, tmp_path / "output.json", threading.Event(), store
    finally:
        store.close()


def _run(task, runtime, **options):
    input_path, output_path, cancel, store = runtime
    return run_in_process(task, input_path, output_path, cancel, credentials=store, **options)


def test_busy_fixture_pid_is_complete_before_readiness_marker_becomes_visible(tmp_path, monkeypatch):
    marker = tmp_path / "started.pid"
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"started": str(marker)}), encoding="utf-8")
    observed = []

    def interleaved_write(path, value, *, encoding=None):
        with path.open("w", encoding=encoding) as stream:
            # Deterministically inspect exactly the two states in which another
            # process could observe a created-but-empty or partially written PID.
            assert not marker.exists(), "Opening the staging file must not publish readiness"
            observed.append("empty staging")
            stream.write(value[:1])
            stream.flush()
            assert not marker.exists(), "A partial PID must not publish readiness"
            observed.append("partial staging")
            stream.write(value[1:])
        return len(value)

    class NativeSleep:
        def __call__(self, _duration):
            observed.append("native work reached")

    class NativeLibrary:
        sleep = NativeSleep()
        Sleep = NativeSleep()

    monkeypatch.setattr(Path, "write_text", interleaved_write)
    monkeypatch.setattr(ctypes, "WinDLL" if os.name == "nt" else "CDLL", lambda *_a: NativeLibrary())
    _busy_task(input_path, tmp_path / "unused.json", threading.Event())
    assert observed == ["empty staging", "partial staging", "native work reached"]
    assert marker.read_text(encoding="ascii") == str(os.getpid())
    assert not list(tmp_path.glob("*.tmp"))


def test_success_publishes_json_and_reaps_child(runtime):
    original_children = {process.pid for process in multiprocessing.active_children()}
    result = _run(_success_task, runtime)
    assert json.loads(result.output_path.read_text(encoding="utf-8")) == {"answer": 42}
    assert result.output_bytes == result.output_path.stat().st_size
    assert len(result.sha256) == 64
    assert result.elapsed_seconds > 0
    if os.name != "nt":
        assert result.output_path.stat().st_mode & 0o777 == 0o600
    assert {process.pid for process in multiprocessing.active_children()} == original_children
    assert not list(result.output_path.parent.glob(".cpu-task-*"))


def test_task_failure_is_structured_and_never_exposes_exception_or_prints(runtime, capfd, caplog):
    runtime[1].write_text('{"previous":true}', encoding="utf-8")
    with pytest.raises(WorkerError) as caught:
        _run(_error_task, runtime)
    assert caught.value.code == "task_failed"
    assert caught.value.diagnostic_id is not None and len(caught.value.diagnostic_id) == 32
    assert "private-token" not in str(caught.value)
    assert runtime[1].read_text(encoding="utf-8") == '{"previous":true}'
    captured = capfd.readouterr()
    assert "private-token" not in captured.out + captured.err
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "event=worker.failed code=task_failed" in record.getMessage()
    assert "diagnostic_id=" + caught.value.diagnostic_id in record.getMessage()
    assert "exception=RuntimeError" in record.getMessage()
    assert "test_runtime_worker.py:_error_task:" in record.getMessage()
    assert "must-not-escape-private-token" not in repr(record.__dict__)
    assert str(runtime[0]) not in record.getMessage()
    assert record.exc_info is None and record.stack_info is None and record.args == ()


def test_forged_child_diagnostic_is_rejected_without_logging_or_publication(runtime, monkeypatch, caplog):
    runtime[1].write_text('{"previous":true}', encoding="utf-8")
    monkeypatch.setattr(worker, "_status_message", lambda connection: {
        "status": "error", "code": "task_failed", "diagnostic": {
            "exception": "RuntimeError", "line": 1, "path": "/private/injected-token.py"}})
    with pytest.raises(WorkerError) as caught:
        _run(_success_task, runtime)
    assert caught.value.code == "protocol_error"
    assert caught.value.diagnostic_id is None
    assert caplog.records == []
    assert runtime[1].read_text(encoding="utf-8") == '{"previous":true}'


class StatusPipe:
    def __init__(self, data):
        self.data = data
        self.maximum = None

    def recv_bytes(self, maxlength=None):
        self.maximum = maxlength
        if len(self.data) > maxlength:
            raise OSError("Oversized frame")
        return self.data


@pytest.mark.parametrize("value", [
    b'{"status":"cancelled","private":"injected-token"}',
    b'{"status":"cancelled","status":"ok"}',
    b'{"status":"error","code":"task_failed","diagnostic":NaN}',
    b'{"status":"error","code":"task_failed"}',
    b'{"status":"progress","completed":2}',
    b'{"status":"progress","completed":3,"total":2}',
    b'{"status":"progress","completed":-1,"total":2}',
    b'{"status":"progress","completed":true,"total":2}',
    b'"' + b"x" * 1024 + b'"',
])
def test_status_protocol_rejects_unknown_duplicate_or_oversized_fields(value):
    connection = StatusPipe(value)
    with pytest.raises(WorkerError) as caught:
        worker._status_message(connection)
    assert caught.value.code == "protocol_error"
    assert connection.maximum == 1024


def test_task_progress_reaches_the_parent_and_never_replaces_the_result(runtime):
    observed = []
    result = _run(_progress_task, runtime, on_progress=lambda completed, total: observed.append((completed, total)))
    assert json.loads(result.output_path.read_text(encoding="utf-8")) == {"answer": 21}
    assert observed and observed[-1] == (21, 21)
    assert observed == sorted(observed) and all(0 <= completed <= 21 for completed, _ in observed)


def test_failing_progress_callback_never_fails_the_computation(runtime):
    def broken(completed, total):
        raise RuntimeError("must-not-escape-callback")

    result = _run(_progress_task, runtime, on_progress=broken)
    assert json.loads(result.output_path.read_text(encoding="utf-8")) == {"answer": 21}


def test_progress_is_silent_outside_a_worker_process():
    assert worker._PROGRESS is None
    worker.report_progress(1, 2)  # A no-op in the coordinator's own process.


def test_unexpected_process_exit_is_detected_and_never_published(runtime, caplog):
    with pytest.raises(WorkerError) as caught:
        _run(_crash_task, runtime)
    assert caught.value.code == "crashed"
    assert not runtime[1].exists()
    # The exit code is the only trace of a silenced child; nothing else is logged.
    assert [record.getMessage() for record in caplog.records] == ["event=worker.crashed exitcode=73"]


def test_crash_in_exit_teardown_keeps_a_verified_result(runtime, caplog):
    result = _run(_teardown_crash_task, runtime)
    assert json.loads(result.output_path.read_text(encoding="utf-8")) == {"answer": 42}
    assert [record.getMessage() for record in caplog.records] == ["event=worker.crashed exitcode=3"]
    assert not list(result.output_path.parent.glob(".cpu-task-*"))


def test_timeout_terminates_a_noncooperative_native_task(runtime):
    existing = {process.pid for process in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(WorkerTimeout):
        _run(_busy_task, runtime, timeout_seconds=0.5, stop_grace_seconds=0.05)
    assert time.monotonic() - started < 4
    assert not runtime[1].exists()
    assert {process.pid for process in multiprocessing.active_children()} == existing


def test_cancel_before_start_does_not_launch_process_or_overwrite_saved_result(runtime):
    existing = {process.pid for process in multiprocessing.active_children()}
    runtime[1].write_text('{"previous":true}', encoding="utf-8")
    runtime[2].set()
    with pytest.raises(WorkerCancelled):
        _run(_success_task, runtime)
    assert runtime[1].read_text(encoding="utf-8") == '{"previous":true}'
    assert {process.pid for process in multiprocessing.active_children()} == existing


def test_cancel_while_running_terminates_noncooperative_task(runtime):
    started_marker = runtime[0].parent / "started.pid"
    runtime[0].write_text(json.dumps({"started": str(started_marker)}), encoding="utf-8")

    def cancel_when_started():
        deadline = time.monotonic() + 5
        while not started_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        runtime[2].set()

    canceller = threading.Thread(target=cancel_when_started)
    canceller.start()
    try:
        with pytest.raises(WorkerCancelled):
            _run(_busy_task, runtime, stop_grace_seconds=0.05)
        assert started_marker.exists()
        assert not runtime[1].exists()
    finally:
        canceller.join(6)


def test_cancellation_at_publication_fence_keeps_previous_result(runtime, monkeypatch):
    runtime[1].write_text('{"previous":true}', encoding="utf-8")
    validate = worker._manifest_digest

    def cancel_after_validation(path, maximum):
        result = validate(path, maximum)
        runtime[2].set()
        return result

    monkeypatch.setattr(worker, "_manifest_digest", cancel_after_validation)
    with pytest.raises(WorkerCancelled):
        _run(_success_task, runtime)
    assert runtime[1].read_text(encoding="utf-8") == '{"previous":true}'


def test_manifest_changed_after_child_validation_is_not_published(runtime, monkeypatch):
    runtime[1].write_text('{"previous":true}', encoding="utf-8")
    digest = worker._manifest_digest

    def change_manifest(path, maximum):
        path.write_text('{"tampered":true}', encoding="utf-8")
        return digest(path, maximum)

    monkeypatch.setattr(worker, "_manifest_digest", change_manifest)
    with pytest.raises(WorkerError) as caught:
        _run(_success_task, runtime)
    assert caught.value.code == "publish_failed"
    assert runtime[1].read_text(encoding="utf-8") == '{"previous":true}'


@pytest.mark.parametrize("output", [
    "not json", "[]", '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e9999}',
    '{"text":"' + "a" * 2_000 + '"}',
])
def test_invalid_or_oversized_manifest_is_rejected(runtime, output):
    runtime[0].write_text(json.dumps({"output": output}), encoding="utf-8")
    with pytest.raises(WorkerError) as caught:
        _run(_invalid_task, runtime, max_output_bytes=1_000)
    assert caught.value.code == "invalid_output"
    assert not runtime[1].exists()


@pytest.mark.parametrize("task", [_returned_object_task, _no_manifest_task])
def test_tasks_cannot_return_arrays_over_ipc_or_omit_manifest(runtime, task):
    with pytest.raises(WorkerError) as caught:
        _run(task, runtime)
    assert caught.value.code == "invalid_output"


def test_manifest_symlink_is_not_published(runtime):
    # Creation of symlinks is an OS privilege on Windows; this boundary is still
    # covered by the existing-output path check on every platform.
    if os.name == "nt":
        with pytest.raises(WorkerError):
            _run(_symlink_manifest_task, runtime)
    else:
        with pytest.raises(WorkerError) as caught:
            _run(_symlink_manifest_task, runtime)
        assert caught.value.code == "invalid_output"
    assert not runtime[1].exists()
    assert json.loads(runtime[0].read_text(encoding="utf-8")) == {"value": 21}


def test_spawn_has_no_legacy_credentials_or_ui_import(runtime, monkeypatch):
    for name in LEGACY_ENVIRONMENT:
        monkeypatch.setenv(name, "child-secret-marker-982")
    result = _run(_clean_boundary_task, runtime)
    assert json.loads(result.output_path.read_text(encoding="utf-8")) == {"clean": True}
    assert not any(name in os.environ for name in LEGACY_ENVIRONMENT)
    assert runtime[3].get("deepseek_api_key") == "child-secret-marker-982"


@pytest.mark.parametrize("task", ["app.main.run", lambda a, b, c: None, {"module": "os", "function": "system"}])
def test_user_strings_or_serialized_descriptors_cannot_select_a_callable(runtime, task):
    with pytest.raises(TypeError):
        _run(task, runtime)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadlines_rejected_before_spawn(runtime, timeout):
    with pytest.raises(ValueError):
        _run(_success_task, runtime, timeout_seconds=timeout)


def test_input_size_and_input_output_collision_are_rejected(runtime):
    with pytest.raises(WorkerError) as caught:
        _run(_success_task, runtime, max_input_bytes=1)
    assert caught.value.code == "invalid_input"
    input_path, _, cancel, store = runtime
    with pytest.raises(WorkerError):
        run_in_process(_success_task, input_path, input_path, cancel, credentials=store)
    assert json.loads(input_path.read_text(encoding="utf-8")) == {"value": 21}


def _pid_is_running(pid):
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # Observe only; never kill a process using a discovered PID.
    except ProcessLookupError:
        return False
    if Path("/proc").is_dir():
        # Container PID 1 may delay adopting/reaping zombies; they cannot work.
        # A successful signal probe can race with reaping. Once /proc exists,
        # a missing per-process stat means gone, not a platform without /proc.
        try:
            return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()[0] != "Z"
        except (FileNotFoundError, ProcessLookupError):
            return False
    return True


@pytest.mark.parametrize("parent_target", [_parent_process_fixture, _parent_shutdown_fixture])
def test_abrupt_parent_loss_stops_cpu_child_during_native_work_and_exit_hooks(tmp_path, parent_target):
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    marker = tmp_path / "cpu.pid"
    input_path.write_text(json.dumps({"started": str(marker)}), encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    parent = context.Process(target=parent_target, args=(str(input_path), str(output_path)))
    parent.start()
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline and parent.is_alive():
            time.sleep(0.025)
        assert marker.exists(), "CPU process did not reach the native task"
        pid = int(marker.read_text(encoding="ascii"))
        assert _pid_is_running(pid)
        parent.kill()  # Own Process object; deliberately bypass all parent cleanup.
        parent.join(3)
        deadline = time.monotonic() + 5
        while _pid_is_running(pid) and time.monotonic() < deadline:
            time.sleep(0.025)
        assert not _pid_is_running(pid), "CPU process survived loss of its parent"
        assert not output_path.exists()
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join(3)
        parent.close()
