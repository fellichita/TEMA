"""Capabilities the running machine actually grants this test process.

A test that proves something about symbolic links has to create one. Windows
grants that only with the SeCreateSymbolicLink privilege (developer mode or an
elevated process), so without it the test cannot run at all. Skipping names the
reason instead of leaving a permanent red failure that hides real regressions.
"""

from functools import cache
from pathlib import Path
import tempfile

import pytest

_REASON = ("Символьные ссылки недоступны: нет права SeCreateSymbolicLink "
           "(включите режим разработчика Windows или запустите с повышенными правами).")


@cache
def symlinks_available() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "target").write_text("probe", encoding="utf-8")
        try:
            (root / "link").symlink_to(root / "target")
        except (OSError, NotImplementedError):
            return False
    return True


def require_symlinks() -> None:
    if not symlinks_available():
        pytest.skip(_REASON)
