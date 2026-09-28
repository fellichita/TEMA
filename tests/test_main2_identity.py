from pathlib import Path
import subprocess
import sys

import pytest

from app.backend.config import BackendSettings
from app import identity
from app.identity import APP_NAME, DATA_DIR_NAME, default_data_dir, validate_data_dir
from tests.platform_support import require_symlinks


def test_default_storage_isolated_and_independent_of_cwd(tmp_path, monkeypatch):
    initial = default_data_dir()
    monkeypatch.chdir(tmp_path)
    assert default_data_dir() == initial
    assert initial.name == DATA_DIR_NAME
    assert APP_NAME == "Trendanalyser Pilot main2"
    assert initial.is_absolute()


@pytest.mark.parametrize("checkout_name", [
    "Trendanalyser-main2", "Trendanalizer-main2", "Trendanalizer-main2-production", "downloaded-source",
])
def test_legacy_storage_rejected_including_symlink(tmp_path, monkeypatch, checkout_name):
    require_symlinks()
    checkout = tmp_path / checkout_name
    monkeypatch.setattr(identity, "__file__", str(checkout / "app" / "identity.py"))
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    original = tmp_path / "Trendanalizer"
    original.mkdir()
    with pytest.raises(ValueError, match="отдельные данные"):
        BackendSettings(data_dir=original / "storage" / "desktop")
    link = tmp_path / "old"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="отдельные данные"):
        validate_data_dir(link / "storage")


@pytest.mark.parametrize("frozen", [False, True])
def test_standard_legacy_storage_rejected_for_any_checkout_name(tmp_path, monkeypatch, frozen):
    require_symlinks()
    monkeypatch.setattr(identity, "__file__", str(tmp_path / "downloaded-source" / "app" / "identity.py"))
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local-app-data"))
    legacy_locations = (
        Path.home() / ".local" / "share" / "Trendanalizer",
        Path.home() / "Library" / "Application Support" / "Trendanalizer",
        tmp_path / "local-app-data" / "Trendanalizer",
    )
    for index, location in enumerate(legacy_locations):
        location.mkdir(parents=True)
        with pytest.raises(ValueError, match="отдельные данные"):
            validate_data_dir(location / "storage")
        link = tmp_path / f"legacy-{index}"
        link.symlink_to(location, target_is_directory=True)
        with pytest.raises(ValueError, match="отдельные данные"):
            validate_data_dir(link / "storage")
    assert validate_data_dir(default_data_dir()) == default_data_dir()


def test_checkout_with_repository_name_can_use_its_own_storage(tmp_path, monkeypatch):
    checkout = tmp_path / "Trendanalizer"
    monkeypatch.setattr(identity, "__file__", str(checkout / "app" / "identity.py"))
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert validate_data_dir(checkout / "storage") == checkout / "storage"


def test_custom_isolated_path_does_not_create_directories(tmp_path):
    target = tmp_path / "Новые данные"
    assert BackendSettings(data_dir=target).data_dir == target
    assert not target.exists()


def test_import_entrypoint_does_not_import_tk_or_start_workers():
    result = subprocess.run([sys.executable, "-c", "import app.main, sys; "
                             "assert 'tkinter' not in sys.modules; "
                             "assert 'app.ui.window' not in sys.modules"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
