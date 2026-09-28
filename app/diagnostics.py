"""Best-effort internal diagnostics without exception text or source data.

Callers classify expected/public errors before calling this helper. The optional
private file sink accepts only these fixed records, never third-party logging.
"""

import builtins
from io import TextIOWrapper
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import stat
from threading import RLock
from types import CodeType, FrameType
from typing import Callable

_LOGGER = logging.getLogger(__name__)
_APP_ROOT = Path(__file__).parent
_EVENTS = frozenset({
    "ui.invoke_failed", "ui.close_failed", "ui.callback_failed", "backend.collection_failed",
    "backend.history_failed", "backend.provider_close_failed",
})
_MAX_FRAMES = 64
_WORKER_EXCEPTION_NAMES = frozenset(name for name, value in vars(builtins).items()
                                   if isinstance(value, type) and issubclass(value, BaseException))
_CONFIG_LOCK = RLock()
_RECORD_MARKER = object()
_PRIVATE_HANDLER: RotatingFileHandler | None = None
_LOG_BYTES = 256 * 1024
_LOG_BACKUPS = 3


class _PrivateHandler(RotatingFileHandler):
    def _open(self) -> TextIOWrapper:
        if Path(self.baseFilename).is_symlink():
            raise OSError("A private diagnostic file is required")
        descriptor = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND
                             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError("A regular diagnostic file is required")
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            return os.fdopen(descriptor, "a", encoding="utf-8")
        except Exception:
            os.close(descriptor)
            raise

    def filter(self, record: logging.LogRecord) -> bool:
        return (record.name == __name__ and getattr(record, "_safe_diagnostic", None) is _RECORD_MARKER
                and record.exc_info is None and record.stack_info is None and not record.args)

    def handleError(self, record: logging.LogRecord) -> None:
        # Diagnostics must not write handler exceptions or filesystem paths to stderr.
        return


def close_private_logging() -> None:
    """Release the selected profile's diagnostic file; safe during shutdown."""
    global _PRIVATE_HANDLER
    with _CONFIG_LOCK:
        handler, _PRIVATE_HANDLER = _PRIVATE_HANDLER, None
        if handler is not None:
            _LOGGER.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def configure_private_logging(data_dir: Path) -> bool:
    """Select a bounded private sink for this active library, never third parties.

    A failed switch closes the old sink so a newly selected profile cannot leak
    records into its predecessor. The caller supplies the opened backend's path.
    """
    global _PRIVATE_HANDLER
    with _CONFIG_LOCK:
        try:
            from app.identity import validate_data_dir

            directory = validate_data_dir(Path(data_dir))
            if not directory.is_dir():
                raise OSError("The active library must already exist")
            directory = directory / "logs"
            if directory.is_symlink():
                raise OSError("A private diagnostic directory is required")
            directory.mkdir(mode=0o700, exist_ok=True)
            if os.name != "nt":
                directory.chmod(0o700)
            path = directory / "diagnostics.log"
            if _PRIVATE_HANDLER is not None and Path(_PRIVATE_HANDLER.baseFilename) == path:
                return True
            handler = _PrivateHandler(path, maxBytes=_LOG_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8")
            handler.setLevel(logging.ERROR)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        except Exception:
            close_private_logging()
            return False
        close_private_logging()
        _PRIVATE_HANDLER = handler
        _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.ERROR)
        return True


def _write(message: str) -> None:
    with _CONFIG_LOCK:
        _LOGGER.error(message, extra={"_safe_diagnostic": _RECORD_MARKER})


def _safe_identifier(value: object) -> str:
    if isinstance(value, str) and len(value) <= 100 and value.isascii() and value.isidentifier():
        return value
    return "unknown"


def _code_location(code: CodeType, line: int | None) -> str:
    filename = Path(code.co_filename)
    try:
        relative = filename.relative_to(_APP_ROOT)
        parts = relative.parts
        if not parts or any(part == ".." or not part.replace(".", "_").isidentifier() or not part.isascii()
                            for part in parts):
            raise ValueError
        location = "app/" + relative.as_posix()
        if len(location) > 240:
            location = "app/unknown"
    except ValueError:
        name = filename.name
        location = name if name.endswith(".py") and _safe_identifier(name[:-3]) != "unknown" else "external"
    function = "<module>" if code.co_name == "<module>" else _safe_identifier(code.co_name)
    return f"{location}:{function}:{line if line is not None else 'unavailable'}"


def _frame_location(frame: FrameType, line: int) -> str:
    return _code_location(frame.f_code, line)


def worker_error_details(error: BaseException, task: Callable | None) -> dict[str, object]:
    """Encode no free-form child strings: a builtin exception category and line.

    Custom exceptions use their nearest builtin base; no exception text, class
    name supplied by user data, frame filename, source or locals crosses IPC.
    """
    category = "BaseException"
    for candidate in type(error).__mro__:
        if candidate.__name__ in _WORKER_EXCEPTION_NAMES and getattr(builtins, candidate.__name__) is candidate:
            category = candidate.__name__
            break
    code = getattr(task, "__code__", None)
    line = None
    traceback = error.__traceback__
    for _ in range(_MAX_FRAMES):
        if traceback is None:
            break
        if traceback.tb_frame.f_code is code:
            line = traceback.tb_lineno
        traceback = traceback.tb_next
    return {"exception": category, "line": line}


def validate_worker_details(value: object, task: Callable) -> tuple[str, int | None]:
    """Validate untrusted IPC values against parent-owned exception/code tables."""
    if not isinstance(value, dict) or set(value) != {"exception", "line"}:
        raise ValueError("Invalid worker diagnostic")
    exception_type, line = value["exception"], value["line"]
    if not isinstance(exception_type, str) or exception_type not in _WORKER_EXCEPTION_NAMES:
        raise ValueError("Invalid worker exception category")
    if line is not None and (type(line) is not int or line not in {item[2] for item in task.__code__.co_lines()}):
        raise ValueError("Invalid worker source position")
    return exception_type, line


def log_worker_error(code: str, diagnostic_id: str, details: object, task: Callable) -> None:
    """Rebuild the location from trusted parent code, never a child's path/name."""
    try:
        if code not in {"task_failed", "invalid_output"}:
            return
        if len(diagnostic_id) != 32 or any(char not in "0123456789abcdef" for char in diagnostic_id):
            return
        exception_type, line = validate_worker_details(details, task)
        location = _code_location(task.__code__, line)
        _write(f"event=worker.failed code={code} diagnostic_id={diagnostic_id} exception={exception_type} stack={location}")
    except Exception:
        return


def log_worker_crash(exitcode: object) -> None:
    """Record how a silenced worker ended without a status: only its exit code.

    The child's stdout/stderr go to devnull, so without this line a native
    crash, an abort and a parent-loss exit all look the same afterwards.
    """
    try:
        if type(exitcode) is not int:
            value = "unavailable"
        elif 0 <= exitcode <= 255:
            value = str(exitcode)
        else:
            # Windows reports NTSTATUS codes (0xC0000005 and the like) as large integers.
            value = f"{exitcode}/0x{exitcode & 0xFFFFFFFF:08X}"
        _write(f"event=worker.crashed exitcode={value}")
    except Exception:
        return


def log_internal_error(event: str, error: BaseException) -> None:
    """Log a fixed event, class name and bounded stack; never inspect messages.

    No exc_info/stack_info, traceback formatting, source lines, locals, exception
    arguments or chained exceptions enter the record. Logging is best-effort and
    cannot replace the error the caller is already handling.
    """
    try:
        safe_event = event if event in _EVENTS else "internal.error"
        exception_type = _safe_identifier(type(error).__name__)
        frames: list[str] = []
        traceback = error.__traceback__
        while traceback is not None and len(frames) < _MAX_FRAMES:
            frames.append(_frame_location(traceback.tb_frame, traceback.tb_lineno))
            traceback = traceback.tb_next
        if traceback is not None:
            frames.append("truncated")
        stack = " > ".join(frames) or "unavailable"
        _write(f"event={safe_event} exception={exception_type} stack={stack}")
    except Exception:  # noqa: BLE001 - diagnostic handlers must never replace an operation error
        # A broken handler must not alter operation completion or shutdown.
        return
