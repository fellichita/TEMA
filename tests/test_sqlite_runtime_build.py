"""Build isolation and source-integrity failures; native proof is a separate smoke."""

import hashlib
import io
import tarfile
import zipfile

import pytest

from scripts import build_sqlite_runtime as builder


def sources(tmp_path, monkeypatch):
    directory = tmp_path / "isolated" / "runtime-sources"
    directory.mkdir(parents=True)
    driver = directory / "pysqlite3-0.6.0.tar.gz"
    with tarfile.open(driver, "w:gz") as archive:
        payload = b'[project]\nname = "pysqlite3"\nversion = "0.6.0"\n'
        info = tarfile.TarInfo("pysqlite3-0.6.0/pyproject.toml")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    sqlite = directory / "sqlite-amalgamation-3530400.zip"
    code = b'#define SQLITE_VERSION "3.53.4"\n'
    with zipfile.ZipFile(sqlite, "w") as archive:
        archive.writestr("sqlite-amalgamation-3530400/sqlite3.c", code)
        archive.writestr("sqlite-amalgamation-3530400/sqlite3.h", b"test header")
    monkeypatch.setattr(builder, "SOURCES", tuple(
        (path.name, "https://invalid.example/never-requested", hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (driver, sqlite)))
    monkeypatch.setattr(builder, "SQLITE_ARCHIVE_SHA3_256", hashlib.sha3_256(sqlite.read_bytes()).hexdigest())
    monkeypatch.setattr(builder, "SQLITE_C_SHA3_256", hashlib.sha3_256(code).hexdigest())
    monkeypatch.setattr(builder.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("Offline build requested network"))
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected compiler invocation"))
    return directory.parent, driver, sqlite


def test_corrupt_cached_source_fails_before_compiling_or_requesting_network(tmp_path, monkeypatch):
    directory, driver, _ = sources(tmp_path, monkeypatch)
    driver.write_bytes(driver.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build(offline=True, build_root=directory)


@pytest.mark.parametrize("hash_name", ["SQLITE_ARCHIVE_SHA3_256", "SQLITE_C_SHA3_256"])
def test_published_upstream_hash_mismatch_prevents_compilation(tmp_path, monkeypatch, hash_name):
    directory, _, _ = sources(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, hash_name, "0" * 64)
    with pytest.raises(ValueError, match="officially published SHA3-256"):
        builder.build(offline=True, build_root=directory)


def test_isolated_build_directory_and_private_version_are_forwarded_to_native_builder(tmp_path, monkeypatch):
    directory, _, _ = sources(tmp_path, monkeypatch)
    original_project = tmp_path / "untouched-project"
    original_project.mkdir()
    monkeypatch.setattr(builder, "ROOT", original_project)
    outputs = []

    def compile_wheel(command, *, check, timeout):
        source = directory / command[-1]
        assert source.is_relative_to(directory)
        assert f'version = "{builder.DRIVER_VERSION}"' in (source / "pyproject.toml").read_text(encoding="utf-8")
        output = directory / command[command.index("--wheel-dir") + 1]
        assert output == directory / "wheels"
        assert "--no-deps" in command and "--no-build-isolation" in command
        assert check and timeout == 300
        wheel = output / f"pysqlite3-{builder.DRIVER_VERSION}-cp313-test.whl"
        wheel.write_bytes(b"stub native output; not a real wheel")
        outputs.append(wheel)

    monkeypatch.setattr(builder.subprocess, "run", compile_wheel)
    assert builder.build(offline=True, build_root=directory) == outputs
    assert list(original_project.iterdir()) == []
