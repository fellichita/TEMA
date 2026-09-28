"""A forged lock entry must not let startup write outside its data directory."""

import os
import stat

import pytest

from app.backend.errors import BackendError
from app.backend.locking import InstanceLock
from tests.platform_support import require_symlinks


def test_lock_refuses_symlink_without_modifying_its_target(tmp_path):
    require_symlinks()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pilot.lock").symlink_to(outside)

    with pytest.raises(BackendError) as error:
        InstanceLock(data_dir / "pilot.lock").acquire()

    assert error.value.code == "backend_busy"
    assert outside.read_bytes() == b""


def test_lock_refuses_hard_link_without_modifying_its_target(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        os.link(outside, data_dir / "pilot.lock")
    except OSError:
        pytest.skip("Hard links are unavailable on this filesystem")

    with pytest.raises(BackendError) as error:
        InstanceLock(data_dir / "pilot.lock").acquire()

    assert error.value.code == "backend_busy"
    assert outside.read_bytes() == b""


def test_lock_refuses_symlinked_data_directory(tmp_path):
    require_symlinks()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "profile"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BackendError) as error:
        InstanceLock(linked / "pilot.lock").acquire()

    assert error.value.code == "backend_busy"
    assert not (outside / "pilot.lock").exists()


def test_lock_still_opens_existing_regular_file(tmp_path):
    path = tmp_path / "pilot.lock"
    path.write_bytes(b"\0")
    lock = InstanceLock(path)
    try:
        lock.acquire()
    finally:
        lock.release()
    # Windows byte-range locks deny a second handle's read until release.
    assert path.read_bytes() == b"\0"


@pytest.mark.skipif(os.name == "nt", reason="Windows uses profile ACLs rather than POSIX file modes")
def test_lock_closes_existing_permissive_profile_and_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_dir.chmod(0o755)
    path = data_dir / "backend.lock"
    path.write_bytes(b"\0")
    path.chmod(0o644)
    lock = InstanceLock(path)
    try:
        lock.acquire()
        assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        lock.release()
