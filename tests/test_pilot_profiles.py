"""Saved library pointers must fail closed without blocking first-run creation."""

import json
import os
from unittest.mock import patch

import pytest

from app.profiles import activate_profile, resolve_profile
from app.sqlite_runtime import sqlite3
from tests.test_pilot_restore_workflow import controller as controller
from tests.platform_support import require_symlinks


def library(path):
    path.mkdir(parents=True)
    for name in ("documents.sqlite3", "pilot.sqlite3"):
        with sqlite3.connect(path / name) as connection:
            connection.execute("CREATE TABLE fixture_identity (id INTEGER PRIMARY KEY)")
    return path


def pointer(anchor, target):
    anchor.mkdir(exist_ok=True)
    (anchor / "active-library.json").write_text(json.dumps({"version": 1, "path": str(target)}))


def test_new_anchor_without_pointer_is_allowed_without_creating_files(tmp_path):
    anchor = tmp_path / "fresh-main2"
    assert resolve_profile(anchor) == anchor
    assert not anchor.exists()


def test_valid_selected_library_and_one_further_restore_are_followed(tmp_path):
    anchor = tmp_path / "anchor"
    first, second = library(tmp_path / "one"), library(tmp_path / "two")
    activate_profile(anchor, first)
    activate_profile(first, second)
    assert resolve_profile(anchor) == second


@pytest.mark.parametrize("missing", ["documents.sqlite3", "pilot.sqlite3"])
def test_missing_selected_database_is_never_recreated(tmp_path, missing):
    selected = library(tmp_path / "selected")
    (selected / missing).unlink()
    anchor = tmp_path / "anchor"
    pointer(anchor, selected)
    with pytest.raises(ValueError, match="Не удалось открыть"):
        resolve_profile(anchor)
    assert not (selected / missing).exists()


@pytest.mark.parametrize("value", [b"", b"not a database", b"[" * 1200])
def test_empty_corrupt_or_pathological_selected_database_is_rejected(tmp_path, value):
    selected = library(tmp_path / "selected")
    (selected / "pilot.sqlite3").write_bytes(value)
    anchor = tmp_path / "anchor"
    pointer(anchor, selected)
    with pytest.raises(ValueError, match="Не удалось открыть"):
        resolve_profile(anchor)
    assert (selected / "pilot.sqlite3").read_bytes() == value


@pytest.mark.parametrize("file_name", ["active-library.json", "documents.sqlite3", "pilot.sqlite3"])
def test_symlinks_cannot_redirect_profile_or_database_file_reads(tmp_path, file_name):
    require_symlinks()
    selected = library(tmp_path / "selected")
    anchor = tmp_path / "anchor"
    pointer(anchor, selected)
    file = anchor / file_name if file_name == "active-library.json" else selected / file_name
    external = tmp_path / "unrelated-file"
    external.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(external)
    with pytest.raises(ValueError, match="Не удалось открыть"):
        resolve_profile(anchor)
    assert file.is_symlink()


@pytest.mark.parametrize("data", [b"[" * 1200 + b"]" * 1200, b" " * 10001 + b"{}", b'{"version":1,"path":null}'])
def test_bounded_pointer_parsing_rejects_deep_oversized_and_invalid_data(tmp_path, data):
    (tmp_path / "active-library.json").write_bytes(data)
    with pytest.raises(ValueError, match="Не удалось открыть"):
        resolve_profile(tmp_path)
    assert (tmp_path / "active-library.json").read_bytes() == data


def test_activation_rejects_uninitialized_database_without_writing_pointer(tmp_path):
    selected = tmp_path / "empty-library"
    selected.mkdir()
    (selected / "documents.sqlite3").touch()
    (selected / "pilot.sqlite3").touch()
    with pytest.raises(ValueError, match="обеих проверенных баз"):
        activate_profile(tmp_path / "anchor", selected)
    assert not (tmp_path / "anchor").exists()


def test_sync_failure_before_pointer_commit_preserves_previous_selection(tmp_path, monkeypatch):
    anchor = tmp_path / "anchor"
    first, second = library(tmp_path / "one"), library(tmp_path / "two")
    activate_profile(anchor, first)
    def fail(_descriptor):
        raise OSError("Test-only full device")
    monkeypatch.setattr("app.profiles.os.fsync", fail)
    with pytest.raises(OSError):
        activate_profile(anchor, second)
    assert resolve_profile(anchor) == first
    assert list(anchor.iterdir()) == [anchor / "active-library.json"]


@pytest.mark.skipif(os.name == "nt", reason="Directory fsync is a POSIX durability mechanism")
def test_directory_sync_failure_after_commit_keeps_new_selection_with_warning(tmp_path, monkeypatch):
    anchor = tmp_path / "anchor"
    first, second = library(tmp_path / "one"), library(tmp_path / "two")
    activate_profile(anchor, first)
    original = os.fsync
    calls = []
    def fail_directory(descriptor):
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError("Test-only directory sync failure")
        return original(descriptor)
    monkeypatch.setattr("app.profiles.os.fsync", fail_directory)
    warning = activate_profile(anchor, second)
    assert warning and "не подтвердила" in warning
    assert resolve_profile(anchor) == second
    assert list(anchor.iterdir()) == [anchor / "active-library.json"]


@pytest.mark.parametrize("operation", ["app.pilot.service.PilotService.status", "app.backend.service.Backend.sources"])
def test_restore_status_or_sources_failure_precedes_pointer_commit(controller, tmp_path, operation):
    from app.runtime.jobs import TaskFailure

    previous = controller.backend.settings.data_dir
    saved = controller._invoke("pilot_backup", (str(tmp_path / "backups"),), {})
    with patch(operation, side_effect=TaskFailure("Test-only restored view unavailable")):
        with pytest.raises(TaskFailure):
            controller._invoke("pilot_restore", (saved["path"], str(tmp_path / "restored")), {})
    assert controller.backend.settings.data_dir == previous
    assert resolve_profile(previous) == previous
    assert not (previous / "active-library.json").exists()
    assert controller._invoke("pilot_status", (), {})["settings"]["schema_version"] == 1


@pytest.mark.skipif(os.name == "nt", reason="Directory fsync is a POSIX durability mechanism")
def test_restore_postcommit_sync_failure_returns_new_backend_and_explicit_warning(controller, tmp_path):
    original = controller.backend.settings.data_dir
    saved = controller._invoke("pilot_backup", (str(tmp_path / "backups"),), {})
    real_sync = os.fsync
    def activation_with_directory_failure(anchor, target):
        calls = []
        def fail_directory(descriptor):
            calls.append(descriptor)
            if len(calls) == 2:
                raise OSError("Test-only directory sync failure")
            return real_sync(descriptor)
        with patch("app.profiles.os.fsync", side_effect=fail_directory):
            return activate_profile(anchor, target)
    with patch("app.profiles.activate_profile", side_effect=activation_with_directory_failure):
        restored = controller._invoke("pilot_restore", (saved["path"], str(tmp_path / "restored")), {})
    assert restored["durability_warning"]
    assert str(controller.backend.settings.data_dir) == restored["path"]
    assert controller.backend.settings.data_dir == resolve_profile(original)
    assert controller.backend.settings.data_dir != original
    assert controller._invoke("pilot_status", (), {})["settings"]["schema_version"] == 1
