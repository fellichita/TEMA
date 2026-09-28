"""Один владелец заданий на каталог данных; блокировку снимает и сама ОС при аварии."""

import os
import stat
import sys
from pathlib import Path
from typing import BinaryIO

from app.backend.errors import BackendError


class InstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self._file: BinaryIO | None = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        try:
            # Data directories are resolved by the service, but the lock is
            # also used directly. Refuse a directory alias before creating a
            # file in it, including a symlink in an ancestor component.
            if self.path.parent.absolute() != self.path.parent.resolve():
                raise OSError("Symbolic lock directory")
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.path.parent.absolute() != self.path.parent.resolve():
                raise OSError("Symbolic lock directory")
            if os.name != "nt":
                os.chmod(self.path.parent, 0o700)
            # A lock name inside the data directory must never be allowed to
            # append to a symlink or a hard link pointing at another file.
            if self.path.is_symlink():
                raise OSError("Symbolic lock file")
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                opened = os.fstat(descriptor)
                named = self.path.stat(follow_symlinks=False)
                if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode)
                        or not os.path.samestat(opened, named) or opened.st_nlink != 1):
                    raise OSError("Unsafe lock file")
                if os.name != "nt":
                    os.fchmod(descriptor, 0o600)
                handle = os.fdopen(descriptor, "r+b")
            except BaseException:
                os.close(descriptor)
                raise
        except (OSError, RuntimeError):
            raise BackendError("backend_busy", "Этот каталог данных уже используется другим backend.") from None
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise BackendError("backend_busy", "Этот каталог данных уже используется другим backend.") from None
        self._file = handle

    def release(self) -> None:
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
