"""Remember an explicitly selected main2 library without moving existing data."""

import json
import os
from pathlib import Path
import stat
from uuid import uuid4

from app.identity import validate_data_dir


def _read_regular(path: Path, limit: int) -> bytes:
    """Reject links/devices and bound reads before following a saved selection."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise OSError("A regular profile file is required")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        raise OSError("Profile file disappeared during opening") from None
    descriptor_open = True
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError("Profile file changed during opening")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor_open = False
            return handle.read(limit)
    finally:
        if descriptor_open:
            os.close(descriptor)


def _validate_selected_library(directory: Path) -> None:
    # Only explicitly selected libraries require both databases. A fresh anchor
    # remains valid and is initialized by the backend in the ordinary way.
    for name in ("documents.sqlite3", "pilot.sqlite3"):
        if _read_regular(directory / name, 16) != b"SQLite format 3\x00":
            raise ValueError("Selected library database is missing or damaged")


def resolve_profile(anchor: Path) -> Path:

    current = validate_data_dir(anchor)
    seen: set[Path] = set()
    for _ in range(8):
        if current in seen:
            raise ValueError("Циклическая ссылка на библиотеку main2.")
        try:
            if seen:
                _validate_selected_library(current)
            seen.add(current)
            try:
                data = _read_regular(current / "active-library.json", 10001)
            except FileNotFoundError:
                return current
            value = json.loads(data)
            if (len(data) > 10000 or not isinstance(value, dict) or set(value) != {"version", "path"}
                    or type(value["version"]) is not int or value["version"] != 1 or not isinstance(value["path"], str)):
                raise ValueError("Invalid pointer")
            target = Path(value["path"])
            if not target.is_absolute():
                raise ValueError("Absolute profile path required")
            current = validate_data_dir(target)
        except (OSError, ValueError, TypeError, RecursionError):
            raise ValueError("Не удалось открыть выбранную библиотеку main2. Укажите существующий каталог данных при запуске.") from None
    raise ValueError("Слишком длинная цепочка выбранных библиотек main2.")


def activate_profile(anchor: Path, target: Path) -> str | None:
    """Switch atomically; a post-commit sync failure reports uncertain durability.

    Once replacement succeeds, callers must keep the selected library active.
    Raising after this point would leave UI rollback inconsistent with disk.
    """
    anchor, target = validate_data_dir(anchor), validate_data_dir(target)
    if anchor == target:
        return None
    if resolve_profile(target) == anchor:
        raise ValueError("Библиотека не может ссылаться на себя через другой каталог.")
    try:
        _validate_selected_library(target)
    except (OSError, ValueError):
        raise ValueError("В каталоге нет обеих проверенных баз main2.") from None
    anchor.mkdir(parents=True, exist_ok=True)
    temporary = anchor / (".active-library-" + uuid4().hex + ".tmp")
    committed = False
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump({"version": 1, "path": str(target)}, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(anchor / "active-library.json")
        committed = True
        if os.name != "nt":
            try:
                descriptor = os.open(anchor, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                return ("Библиотека открыта, но система не подтвердила сохранение выбора каталога. "
                        "После перезапуска проверьте путь хранилища.")
        return None
    finally:
        if not committed:
            temporary.unlink(missing_ok=True)
