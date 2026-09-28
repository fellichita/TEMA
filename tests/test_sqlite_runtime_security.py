"""Fail closed when neither installed SQLite driver contains the FTS5 fixes."""

from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


MODULE = Path(__file__).resolve().parents[1] / "app" / "sqlite_runtime.py"


def driver(name, version):
    module = ModuleType(name)
    module.sqlite_version_info = version
    return module


def test_secure_stdlib_needs_no_private_driver(monkeypatch):
    secure = driver("sqlite3", (3, 53, 2))
    monkeypatch.setitem(sys.modules, "sqlite3", secure)
    monkeypatch.setitem(sys.modules, "pysqlite3", None)
    assert runpy.run_path(str(MODULE))["sqlite3"] is secure


def test_old_stdlib_selects_private_security_release(monkeypatch):
    secure = driver("pysqlite3", (3, 53, 4))
    monkeypatch.setitem(sys.modules, "sqlite3", driver("sqlite3", (3, 50, 4)))
    monkeypatch.setitem(sys.modules, "pysqlite3", secure)
    assert runpy.run_path(str(MODULE))["sqlite3"] is secure


@pytest.mark.parametrize("private", [None, driver("pysqlite3", (3, 53, 1))])
def test_unavailable_or_wal_only_private_driver_fails_closed(monkeypatch, private):
    monkeypatch.setitem(sys.modules, "sqlite3", driver("sqlite3", (3, 50, 4)))
    monkeypatch.setitem(sys.modules, "pysqlite3", private)
    with pytest.raises(RuntimeError, match="3.53.2"):
        runpy.run_path(str(MODULE))
