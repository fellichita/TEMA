"""The published dataset bundle must restore exact bytes without overwriting local work."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.restore_storage import restore


def bundle(path, name="storage/corpus.json", content=b'{"topic":"photonic"}', symlink=False):
    path.mkdir()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        if symlink:
            info.type, info.linkname = tarfile.SYMTYPE, "/tmp/unrelated"
            archive.addfile(info)
        else:
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    payload = buffer.getvalue()
    middle = len(payload) // 2
    parts = []
    for index, chunk in enumerate((payload[:middle], payload[middle:])):
        part = f"part{index}"
        (path / part).write_bytes(chunk)
        parts.append({"name": part, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    manifest = {"schema_version": 1, "parts": parts,
                "files": [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_restore_roundtrip_and_repeat_preserve_existing_bytes(tmp_path):
    source, destination = tmp_path / "bundle", tmp_path / "project"
    bundle(source)
    assert restore(source, destination) == {"restored": 1, "already_present": 0, "verified": 1}
    assert (destination / "storage/corpus.json").read_bytes() == b'{"topic":"photonic"}'
    assert restore(source, destination) == {"restored": 0, "already_present": 1, "verified": 1}


def test_different_local_data_is_not_overwritten(tmp_path):
    source, destination = tmp_path / "bundle", tmp_path / "project"
    bundle(source)
    target = destination / "storage/corpus.json"
    target.parent.mkdir(parents=True)
    target.write_text("my local data")
    with pytest.raises(ValueError, match="Локальный файл отличается"):
        restore(source, destination)
    assert target.read_text(encoding="utf-8") == "my local data"


def test_corrupt_archive_part_is_rejected_before_restoring_files(tmp_path):
    source, destination = tmp_path / "bundle", tmp_path / "project"
    bundle(source)
    (source / "part0").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Повреждена часть"):
        restore(source, destination)
    assert not (destination / "storage/corpus.json").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "storage/../outside"])
def test_archive_cannot_write_outside_storage(tmp_path, name):
    source, destination = tmp_path / "bundle", tmp_path / "project"
    bundle(source, name=name)
    with pytest.raises(ValueError, match="Некорректный путь"):
        restore(source, destination)


def test_archive_link_is_not_extracted(tmp_path):
    source, destination = tmp_path / "bundle", tmp_path / "project"
    bundle(source, symlink=True)
    with pytest.raises(ValueError, match="Неожиданный файл"):
        restore(source, destination)
    assert not destination.exists()
