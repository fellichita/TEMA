"""Windows-compatible durable publication retains real writable file handles."""

import hashlib
import os
from pathlib import Path
import stat

import pytest

from app.runtime import worker
from app.pilot.archive import DocumentArchive
from app.runtime.backup import BackupSession, create_backup, restore_backup
from app.runtime.files import open_staged_regular
from tests.test_runtime_backup import profile as profile


@pytest.fixture
def windows_fsync(monkeypatch):
    original_open = Path.open
    original_fdopen = os.fdopen
    original_fsync = os.fsync
    writable = {}
    flushed = []

    def remember(handle):
        writable[handle.fileno()] = handle.writable()
        return handle

    def opened(path, *args, **kwargs):
        return remember(original_open(path, *args, **kwargs))

    def fdopened(*args, **kwargs):
        return remember(original_fdopen(*args, **kwargs))

    def fsync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            if not writable.get(descriptor, False):
                raise OSError("Windows FlushFileBuffers requires write access")
            flushed.append(descriptor)
        return original_fsync(descriptor)

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(os, "fdopen", fdopened)
    monkeypatch.setattr(os, "fsync", fsync)
    return flushed


@pytest.mark.parametrize("validate", [worker._validate_output, worker._manifest_digest])
def test_worker_manifest_flush_uses_writable_handle_without_changing_bytes(tmp_path, windows_fsync, validate):
    path = tmp_path / "output.json"
    data = '{\r\n  "result": "готово"\n}\r\n'.encode("utf-8")
    path.write_bytes(data)
    assert validate(path, 1000) == (len(data), hashlib.sha256(data).hexdigest())
    assert len(windows_fsync) == 1
    assert path.read_bytes() == data


def test_restore_flushes_owned_files_with_writable_handles(profile, tmp_path, windows_fsync):
    directory, reference, document = profile
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    windows_fsync.clear()
    restored = restore_backup(saved.path, tmp_path / "restored")
    files = [path for path in restored.rglob("*") if path.is_file()]
    assert DocumentArchive(restored / "revisions").get(reference.revision_id) == document
    assert len(windows_fsync) == len(files)
    assert not list(tmp_path.glob(".trendanalyser-restore-*"))


def test_staged_open_rejects_nonregular_path_before_open(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "open", lambda *args: pytest.fail("Nonregular path must not be opened"))
    with pytest.raises(OSError, match="regular"):
        open_staged_regular(tmp_path)


def test_staged_open_rejects_descriptor_replacement_and_closes_handle(tmp_path, monkeypatch):
    path = tmp_path / "staged.json"
    external = tmp_path / "other.json"
    path.write_bytes(b'{"result":1}')
    external.write_bytes(b'{"result":2}')
    original_open = os.open
    descriptors = []

    def replaced(target, flags):
        descriptor = original_open(external, flags)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", replaced)
    with pytest.raises(OSError, match="changed"):
        open_staged_regular(path)
    assert external.read_bytes() == b'{"result":2}'
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_staged_open_rejects_path_replacement_after_descriptor_open(tmp_path, monkeypatch):
    path = tmp_path / "staged.json"
    external = tmp_path / "other.json"
    path.write_bytes(b'{"result":1}')
    external.write_bytes(b'{"result":2}')
    original_stat = Path.stat
    calls = 0

    def replaced(target, *args, **kwargs):
        nonlocal calls
        if target == path:
            calls += 1
            if calls == 2:
                return original_stat(external, *args, **kwargs)
        return original_stat(target, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", replaced)
    with pytest.raises(OSError, match="changed"):
        open_staged_regular(path)
    assert path.read_bytes() == b'{"result":1}'


def test_failed_restore_flush_does_not_publish_or_leave_staging(profile, tmp_path, monkeypatch):
    directory, _, _ = profile
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    before = saved.path.read_bytes()

    def fail_flush(descriptor):
        raise OSError("Synthetic storage flush failure")

    monkeypatch.setattr(os, "fsync", fail_flush)
    from app.runtime.backup import ArchiveError

    with pytest.raises(ArchiveError):
        restore_backup(saved.path, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".trendanalyser-restore-*"))
    assert saved.path.read_bytes() == before
