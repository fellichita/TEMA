"""Real controller backup/restore preserves the original library and changes only main2."""

import json
from pathlib import Path

import pytest

from app.profiles import activate_profile, resolve_profile
from app.runtime.backup import ArchiveError
from app.ui.controller import Controller, create_backend


class Scheduler:
    def after(self, *_):
        return None


@pytest.fixture
def controller(tmp_path, monkeypatch):
    from app.runtime.credentials import CredentialStore
    monkeypatch.setattr(CredentialStore, "get", lambda *_: None)
    directory = tmp_path / "main2-profile"
    instance = Controller(Scheduler(), factory=lambda: create_backend(directory))
    instance.profile_anchor = directory
    instance._invoke("open", (), {})
    instance._invoke("pilot_status", (), {})
    yield instance
    instance._close_backend()
    instance.executor.shutdown(wait=True, cancel_futures=True)


def test_real_backup_restore_opens_new_profile_and_persists_selection(controller, tmp_path):
    original = controller.backend.settings.data_dir
    saved = controller._invoke("pilot_backup", (str(tmp_path / "backups"),), {})
    assert controller.backend.settings.data_dir == original
    # A fresh profile starts on the local model; the restore must carry whatever
    # the profile had, not a provider this test invented.
    assert controller._invoke("pilot_status", (), {})["settings"]["provider"] == "local"
    restored = controller._invoke("pilot_restore", (saved["path"], str(tmp_path / "restored")), {})
    target = Path(restored["path"])
    assert target != original
    assert original.joinpath("documents.sqlite3").is_file()
    assert original.joinpath("pilot.sqlite3").is_file()
    assert resolve_profile(original) == target
    assert controller.backend.settings.data_dir == target
    assert controller._invoke("pilot_budget_status", (), {})["restore_pending"] is True
    controller._invoke("pilot_budget_acknowledge", (None, 0, True), {})
    assert controller._invoke("pilot_budget_status", (), {})["restore_pending"] is False
    # A backup made after switching uses the selected library and reopens it.
    second = controller._invoke("pilot_backup", (str(tmp_path / "backups"),), {})
    assert Path(second["path"]).is_file()
    assert controller.backend.settings.data_dir == target


def test_invalid_restore_leaves_running_library_and_settings_available(controller, tmp_path):
    before = controller.backend.settings.data_dir
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"Not a package")
    with pytest.raises(ArchiveError):
        controller._invoke("pilot_restore", (str(invalid), str(tmp_path / "restore")), {})
    assert controller.backend.settings.data_dir == before
    assert controller._invoke("pilot_status", (), {})["settings"]["schema_version"] == 1
    assert not (before / "active-library.json").exists()


def test_profile_cannot_target_protected_original_or_loop(tmp_path):
    anchor = tmp_path / "profile"
    anchor.mkdir()
    from app.identity import APP_NAME
    assert "main2" in APP_NAME
    protected = Path.home() / "Library" / "Application Support" / "Trendanalizer"
    with pytest.raises(ValueError, match="отдельные данные"):
        activate_profile(anchor, protected)
    (anchor / "active-library.json").write_text(json.dumps({"version": 1, "path": str(anchor)}))
    with pytest.raises(ValueError, match="Циклическая"):
        resolve_profile(anchor)


@pytest.mark.parametrize("payload", [[], {"version": True, "path": "/tmp"}, {"version": 1, "path": "relative"},
                                   {"version": 1, "path": "/tmp", "api_key": "should-not-exist"}])
def test_malformed_profile_pointer_never_opens_an_arbitrary_database(tmp_path, payload):
    (tmp_path / "active-library.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        resolve_profile(tmp_path)
