"""Application identity and data isolation, shared by GUI, CLI and packaging."""

import os
import sys
from pathlib import Path

APP_ID = "org.trendanalizer.pilot.main2"
APP_NAME = "Trendanalyser Pilot main2"
# Keep the installed data directory stable so existing libraries remain available.
DATA_DIR_NAME = "Trendanalizer Pilot main2"
KEYRING_NAMESPACE = "Trendanalizer.Pilot.main2"


def default_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        if not base.is_absolute():
            base = Path.home() / ".local" / "share"
    return base / DATA_DIR_NAME


def validate_data_dir(value: Path) -> Path:
    """Reject legacy storage even when reached through a symbolic link.

    Packaged builds protect the old application's standard locations. Source
    builds additionally protect the sibling working checkout saved for main2.
    No directories are created during validation.
    """
    value = value.expanduser().resolve()
    legacy = [Path.home() / ".local" / "share" / "Trendanalizer",
              Path.home() / "Library" / "Application Support" / "Trendanalizer"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        legacy.append(Path(local) / "Trendanalizer")
    if not getattr(sys, "frozen", False):
        checkout = Path(__file__).resolve().parents[1]
        # Worktrees and downloaded copies can have different names. Protect
        # the sibling legacy checkout unless it is this source tree itself.
        if checkout.name != "Trendanalizer":
            legacy.append(checkout.parent / "Trendanalizer")
    if any(value.is_relative_to(path.resolve()) for path in legacy):
        raise ValueError("main2 использует отдельные данные. Выберите новый каталог, не хранилище прежней версии.")
    if sys.platform == "win32" and str(value).startswith("\\\\"):
        raise ValueError("Для базы нужен локальный каталог: сетевые папки не поддерживаются.")
    return value
