"""Regular-file handles for private, application-owned publication staging."""

import os
from pathlib import Path
import stat
from typing import BinaryIO


def open_staged_regular(path: Path) -> BinaryIO:
    """Open an existing staged file without truncation, with access for fsync.

    Windows FlushFileBuffers requires write access even when validation only
    reads bytes. The descriptor must identify the same regular file as the path.
    Callers own the private staging directory and the returned handle.
    """
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise OSError("A staged regular file is required")
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        after = path.stat(follow_symlinks=False)
        if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(after.st_mode)
                or not os.path.samestat(before, opened) or not os.path.samestat(before, after)):
            raise OSError("Staged file changed while opening")
        return os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise
