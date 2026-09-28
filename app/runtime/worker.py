"""Bounded CPU execution with explicit files and a parent-owned publication fence.

Only trusted application code supplies the callable; user manifests never select
a module or function. The task receives an input path, a private output path and
a multiprocessing cancellation event. It writes a JSON object and returns None.
Neither credentials, connections nor result arrays cross the process boundary.

This is lifecycle isolation, not a filesystem or operating-system sandbox. A
parent-loss watcher terminates the child even during native work that releases
the GIL (including ONNX Runtime). Windows additionally uses a kernel Job Object.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

from app.diagnostics import log_worker_crash, log_worker_error, validate_worker_details, worker_error_details
from app.runtime.credentials import CredentialStore, LEGACY_ENVIRONMENT
from app.runtime.files import open_staged_regular


class Cancellation(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


class PipeChannel(Protocol):
    def send_bytes(self, buf: bytes) -> None: ...

    def recv_bytes(self, maxlength: int | None = None) -> bytes: ...

    def poll(self, timeout: float = 0) -> bool: ...

    def close(self) -> None: ...


CpuTask = Callable[[Path, Path, Cancellation], object]
MAX_PROGRESS = 2**31 - 1


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str, *, diagnostic_id: str | None = None) -> None:
        self.code = code
        self.diagnostic_id = diagnostic_id
        super().__init__(message)


class WorkerCancelled(WorkerError):
    def __init__(self) -> None:
        super().__init__("cancelled", "Вычисление отменено. Незавершённый результат не опубликован.")


class WorkerTimeout(WorkerError):
    def __init__(self) -> None:
        super().__init__("timeout", "Вычисление превысило допустимое время и остановлено.")


@dataclass(frozen=True)
class ProcessResult:
    output_path: Path
    output_bytes: int
    sha256: str
    elapsed_seconds: float


_MESSAGES = {
    "task_failed": "Ошибка вычисления. Предыдущие сохранённые результаты доступны.",
    "invalid_output": "Вычисление вернуло некорректный или слишком большой результат.",
    "crashed": "Вычислительный процесс неожиданно завершился. Результат не опубликован.",
    "start_failed": "Не удалось запустить изолированный вычислительный процесс.",
    "protection_failed": "Не удалось включить защиту от оставшихся фоновых процессов.",
    "protocol_error": "Вычислительный процесс вернул некорректное сообщение состояния.",
    "publish_failed": "Не удалось надёжно сохранить результат вычисления.",
    "invalid_input": "Входной файл вычисления отсутствует, недоступен или превышает ограничение.",
    "stop_failed": "Вычислительный процесс не подтвердил остановку после принудительного завершения.",
}


def _safe_error(code: str, *, diagnostic_id: str | None = None) -> WorkerError:
    return WorkerError(code, _MESSAGES[code], diagnostic_id=diagnostic_id)


def _crashed(process: BaseProcess) -> WorkerError:
    log_worker_crash(process.exitcode)
    return _safe_error("crashed")


def _silence_child_output() -> None:
    # Native libraries can write directly to stderr, bypassing Python logging.
    try:
        descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(descriptor, 1)
            os.dup2(descriptor, 2)
        finally:
            if descriptor not in (1, 2):
                os.close(descriptor)
    except OSError:
        pass


def _watch_parent(connection: PipeChannel) -> None:
    try:
        # The parent never sends data; EOF means its sole write handle closed.
        connection.recv_bytes(maxlength=1)
    except (EOFError, OSError):
        pass
    os._exit(72)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


def _reject_constant(value: str) -> object:
    raise ValueError("Non-finite JSON constant")


def _validate_output(path: Path, maximum: int) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing JSON object")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        raise ValueError("JSON size limit")
    with open_staged_regular(path) as stream:
        payload = stream.read(maximum + 1)
        os.fsync(stream.fileno())
    if not 1 <= len(payload) <= maximum:
        raise ValueError("JSON size limit")
    data = json.loads(
        payload.decode("utf-8"), object_pairs_hook=_json_object,
        parse_constant=_reject_constant, parse_float=_finite_float,
    )
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object")
    if os.name != "nt":
        # Preserve the private staging-directory policy after atomic publication
        # into a user-selected directory with a potentially broader umask.
        os.chmod(path, 0o600)
    # Hash the exact bytes that passed validation, not a second read that could
    # see a different file. Parent hashes only, avoiding another large JSON parse.
    return len(payload), hashlib.sha256(payload).hexdigest()


def _manifest_digest(path: Path, maximum: int) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= maximum:
        raise ValueError("Manifest size or type changed")
    digest = hashlib.sha256()
    size = 0
    with open_staged_regular(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            if size > maximum:
                raise ValueError("Manifest size changed")
            digest.update(chunk)
        os.fsync(stream.fileno())
    return size, digest.hexdigest()


def _send_status(connection: PipeChannel, status: dict[str, object]) -> None:
    try:
        connection.send_bytes(json.dumps(status, separators=(",", ":")).encode("ascii"))
    except (BrokenPipeError, EOFError, OSError):
        # The parent will not publish anything if delivery fails.
        pass


class _ProgressChannel:
    """Child-side rate limiter over the one-way status pipe.

    Progress is advisory: a dropped or throttled message never changes the
    result, so delivery failures stay silent exactly like a terminal status.
    """

    def __init__(self, status: PipeChannel, interval: float = 0.25) -> None:
        self._status = status
        self._interval = interval
        self._lock = threading.Lock()
        self._sent = 0.0
        self._last: tuple[int, int] | None = None

    def report(self, completed: int, total: int) -> None:
        if type(completed) is not int or type(total) is not int or not 0 <= completed <= total <= MAX_PROGRESS:
            return
        now = time.monotonic()
        with self._lock:
            if (completed, total) == self._last:
                return
            if completed < total and now - self._sent < self._interval:
                return
            self._sent, self._last = now, (completed, total)
            # Framing stays inside the lock: a task may report from its own
            # threads, and two interleaved frames would break the parent's parse.
            _send_status(self._status, {"status": "progress", "completed": completed, "total": total})


_PROGRESS: _ProgressChannel | None = None


def report_progress(completed: int, total: int) -> None:
    """Report a task's own position to the parent's on_progress callback.

    Call from inside a worker task. Outside one this is a no-op, so the same
    computation runs unchanged in tests and in the coordinator's own process.
    """
    channel = _PROGRESS
    if channel is not None:
        channel.report(completed, total)


def _child_main(
    task_module: str, task_name: str, input_path: Path, output_path: Path, cancel: Cancellation,
    start_gate: Cancellation, status: PipeChannel, parent_liveness: PipeChannel, maximum: int,
) -> None:
    _silence_child_output()
    # Defense in depth. Bootstrap removes these BEFORE spawning as well.
    for name in LEGACY_ENVIRONMENT:
        os.environ.pop(name, None)
    watcher = threading.Thread(
        target=_watch_parent, args=(parent_liveness,), name="cpu-parent-watch", daemon=True,
    )
    watcher.start()
    try:
        while not start_gate.is_set():
            if cancel.is_set():
                _send_status(status, {"status": "cancelled"})
                return
            time.sleep(0.02)
        if cancel.is_set():
            _send_status(status, {"status": "cancelled"})
            return
        task = None
        try:
            # The descriptor is derived from a validated function in parent code,
            # never read from a manifest. Import heavy dependencies only AFTER the
            # watchdog and Windows job/start fence are active.
            task = getattr(importlib.import_module(task_module), task_name)
            _validate_task(task)
            globals()["_PROGRESS"] = _ProgressChannel(status)
            returned = task(input_path, output_path, cancel)
        except BaseException as error:
            _send_status(status, {"status": "error", "code": "task_failed",
                                  "diagnostic": worker_error_details(error, task)})
            return
        if cancel.is_set():
            _send_status(status, {"status": "cancelled"})
            return
        try:
            if returned is not None:
                raise ValueError("Tasks communicate through a manifest")
            size, digest = _validate_output(output_path, maximum)
        except Exception as error:
            _send_status(status, {"status": "error", "code": "invalid_output",
                                  "diagnostic": worker_error_details(error, _validate_output)})
            return
        _send_status(status, {"status": "ok", "bytes": size, "sha256": digest})
    finally:
        # No task progress may follow the terminal status on this pipe.
        globals()["_PROGRESS"] = None
        status.close()
        # Keep the liveness pipe/watchdog active through native-library and Python
        # exit hooks. The daemon and its descriptor disappear with this process.


class _WindowsJob:
    """An uninheritable parent handle whose closure kills the assigned child.

    This branch is exercised by native Windows packaging tests; importing the
    worker on other platforms does not import or initialize Win32 APIs.
    """

    _kernel: Any  # ctypes configures Win32 signatures dynamically below.
    _handle: int | None

    def __init__(self, pid: int) -> None:
        if sys.platform != "win32":
            raise _safe_error("protection_failed")
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._kernel.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel.OpenProcess.restype = wintypes.HANDLE
        self._kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel.CreateJobObjectW(None, None)
        if not self._handle:
            raise _safe_error("protection_failed")
        process_handle = None
        try:
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self._kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise _safe_error("protection_failed")
            process_handle = self._kernel.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process_handle or not self._kernel.AssignProcessToJobObject(self._handle, process_handle):
                raise _safe_error("protection_failed")
        except BaseException:
            self.close()
            raise
        finally:
            if process_handle:
                self._kernel.CloseHandle(process_handle)

    def close(self) -> None:
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None


def _stop_owned_process(process: BaseProcess, cancel: Cancellation, grace: float) -> None:
    cancel.set()
    process.join(grace)
    if process.is_alive():
        process.terminate()
        process.join(grace)
    if process.is_alive():
        process.kill()
        process.join(grace)
    if process.is_alive():
        raise _safe_error("stop_failed")


def _validate_task(task: CpuTask) -> None:
    if (
        not inspect.isfunction(task) or task.__name__ == "<lambda>" or "<locals>" in task.__qualname__
        or task.__qualname__ != task.__name__ or task.__closure__ is not None
        or task.__module__ in {"__main__", "__mp_main__"}
    ):
        raise TypeError("Нужна импортируемая функция верхнего уровня из кода приложения.")
    module = sys.modules.get(task.__module__)
    if module is None or getattr(module, task.__name__, None) is not task:
        raise TypeError("Вычислительная функция должна принадлежать импортируемому модулю приложения.")


def _status_message(connection: PipeChannel) -> dict[str, object]:
    try:
        value = json.loads(connection.recv_bytes(maxlength=1024).decode("ascii"), object_pairs_hook=_json_object,
                           parse_constant=_reject_constant)
    except (ValueError, UnicodeError, OSError, EOFError):
        raise _safe_error("protocol_error") from None
    if (not isinstance(value, dict) or not isinstance(value.get("status"), str)
            or value["status"] not in {"ok", "error", "cancelled", "progress"}):
        raise _safe_error("protocol_error")
    expected = {"ok": {"status", "bytes", "sha256"}, "error": {"status", "code", "diagnostic"},
                "cancelled": {"status"}, "progress": {"status", "completed", "total"}}
    if set(value) != expected[value["status"]]:
        raise _safe_error("protocol_error")
    if value["status"] == "progress":
        completed, total = value["completed"], value["total"]
        if (type(completed) is not int or type(total) is not int
                or not 0 <= completed <= total <= MAX_PROGRESS):
            raise _safe_error("protocol_error")
    return value


def _notify_progress(on_progress: Callable[[int, int], None] | None, message: dict[str, object]) -> None:
    """Deliver advisory progress without ever failing the computation."""
    if on_progress is None:
        return
    try:
        on_progress(int(cast(int, message["completed"])), int(cast(int, message["total"])))
    except Exception:
        # A reporting callback owns its own errors; cancellation and timeouts
        # are decided by the loop, not by a display side effect.
        pass


def run_in_process(
    task: CpuTask, input_path: Path, output_path: Path, cancel: Cancellation, *,
    credentials: CredentialStore, timeout_seconds: float = 900,
    max_input_bytes: int = 256 * 1024 * 1024, max_output_bytes: int = 64 * 1024 * 1024,
    stop_grace_seconds: float = 0.5, on_progress: Callable[[int, int], None] | None = None,
) -> ProcessResult:
    """Run one trusted task, returning only after bounded cleanup and publication.

    Invoke from the coordinator, never Tk's thread. Import legacy credentials
    into the shared store during bootstrap before any other workers are started;
    this method scrubs again before spawning and never passes the store to a child.
    The caller owns run IDs/checkpoints and must fence its later database commit.
    """
    _validate_task(task)
    if (
        isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86_400
        or isinstance(stop_grace_seconds, bool) or not math.isfinite(stop_grace_seconds)
        or not 0.05 <= stop_grace_seconds <= 5
    ):
        raise ValueError("Некорректное ограничение времени вычисления.")
    if any(type(value) is not int or not 1 <= value <= 1024**3 for value in (max_input_bytes, max_output_bytes)):
        raise ValueError("Некорректное ограничение размера файла вычисления.")
    if cancel.is_set():
        raise WorkerCancelled()
    started = time.monotonic()
    try:
        input_path = Path(input_path).resolve(strict=True)
        raw_output = Path(output_path).absolute()
        if raw_output.is_symlink():
            raise ValueError("Output symlink")
        output_path = raw_output.resolve()
        if not input_path.is_file() or not 1 <= input_path.stat().st_size <= max_input_bytes:
            raise ValueError("Input size")
        if input_path == output_path or (output_path.exists() and not output_path.is_file()):
            raise ValueError("Output path")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, RuntimeError):
        raise _safe_error("invalid_input") from None
    credentials.import_legacy_environment()
    context = multiprocessing.get_context("spawn")
    child_cancel = context.Event()
    start_gate = context.Event()
    parent_status, child_status = context.Pipe(duplex=False)
    child_liveness, parent_liveness = context.Pipe(duplex=False)
    staging_directory: Path | None = None
    process: BaseProcess | None = None
    windows_job: _WindowsJob | None = None
    try:
        staging_directory = Path(tempfile.mkdtemp(prefix=".cpu-task-", dir=output_path.parent))
        staging_path = staging_directory / "result.json"
        process = context.Process(
            target=_child_main,
            args=(task.__module__, task.__name__, input_path, staging_path, child_cancel,
                  start_gate, child_status, child_liveness, max_output_bytes),
            name="trendanalyser-cpu", daemon=False,
        )
        try:
            process.start()
        except Exception:
            raise _safe_error("start_failed") from None
        child_status.close()
        child_liveness.close()
        if sys.platform == "win32":
            if process.pid is None:
                raise _safe_error("start_failed")
            windows_job = _WindowsJob(process.pid)
        start_gate.set()
        status: dict[str, object] | None = None
        while True:
            if cancel.is_set():
                raise WorkerCancelled()
            if time.monotonic() - started >= timeout_seconds:
                raise WorkerTimeout()
            if status is None and parent_status.poll(0.025):
                try:
                    message = _status_message(parent_status)
                except WorkerError:
                    process.join(0.05)
                    if process.exitcode is not None and process.exitcode != 0:
                        raise _crashed(process) from None
                    raise
                if message["status"] == "progress":
                    _notify_progress(on_progress, message)
                else:
                    status = message
            if not process.is_alive():
                process.join(0)
                try:
                    while status is None and parent_status.poll(0):
                        # Progress queued just before exit must not hide the terminal status.
                        message = _status_message(parent_status)
                        if message["status"] == "progress":
                            _notify_progress(on_progress, message)
                        else:
                            status = message
                except WorkerError:
                    if process.exitcode != 0:
                        raise _crashed(process) from None
                    raise
                if process.exitcode != 0:
                    error = _crashed(process)
                    # "ok" is sent only after the child validated its manifest,
                    # and the digest check below rejects any later change. A
                    # crash in native teardown after that (CUDA/ONNX unloading)
                    # must not discard a finished result.
                    if status is None or status["status"] != "ok":
                        raise error
                elif status is None:
                    raise _crashed(process)
                break
            if status is not None:
                time.sleep(0.025)
        if status["status"] == "cancelled":
            raise WorkerCancelled()
        if status["status"] == "error":
            code = status.get("code")
            if not isinstance(code, str) or code not in {"task_failed", "invalid_output"}:
                raise _safe_error("protocol_error")
            diagnostic_task = task if code == "task_failed" else _validate_output
            try:
                validate_worker_details(status["diagnostic"], diagnostic_task)
            except (ValueError, TypeError, AttributeError):
                raise _safe_error("protocol_error") from None
            diagnostic_id = uuid4().hex
            log_worker_error(code, diagnostic_id, status["diagnostic"], diagnostic_task)
            raise _safe_error(code, diagnostic_id=diagnostic_id)
        size, digest = status.get("bytes"), status.get("sha256")
        if type(size) is not int or not isinstance(digest, str) or len(digest) != 64:
            raise _safe_error("protocol_error")
        try:
            actual_size, actual_digest = _manifest_digest(staging_path, max_output_bytes)
            if size != actual_size or digest != actual_digest:
                raise ValueError("Manifest changed")
            if cancel.is_set():
                raise WorkerCancelled()
            if time.monotonic() - started >= timeout_seconds:
                raise WorkerTimeout()
            os.replace(staging_path, output_path)
            if os.name != "nt":
                directory_fd = os.open(output_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except WorkerError:
            raise
        except (OSError, ValueError, RuntimeError):
            raise _safe_error("publish_failed") from None
        return ProcessResult(output_path, actual_size, actual_digest, time.monotonic() - started)
    except WorkerError:
        raise
    except Exception:
        raise _safe_error("start_failed") from None
    finally:
        try:
            if process is not None and process.pid is not None:
                _stop_owned_process(process, child_cancel, stop_grace_seconds)
                process.close()
        finally:
            if windows_job is not None:
                windows_job.close()
            for connection in (parent_liveness, child_liveness, parent_status, child_status):
                connection.close()
            if staging_directory is not None:
                try:
                    shutil.rmtree(staging_directory)
                except OSError:
                    # A startup cleanup can remove an unreferenced staging dir.
                    # Never mask the safe original status with a filesystem path.
                    pass
