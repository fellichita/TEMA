"""Verified local model copies kept inside the project, so a second install never downloads.

The project stages every pinned model under `storage/`. Once a model is staged, any further installation — a new user
profile, a reinstall, a second checkout that keeps this folder — copies those
bytes instead of fetching them again. Every copy is verified by the model's own
specification before and after, so a staged folder is never trusted on its name.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

PROJECT = Path(__file__).resolve().parents[1]
LOCK = PROJECT / "resources/models/registry.json"


def staging_directory(key: str) -> Path:
    """Where this project keeps its verified copy of one pinned model."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for item in lock.get("models", ()):
        if item.get("key") == key:
            source = Path(item["source"])
            # A POSIX-rooted string is not "absolute" on Windows, yet joining it
            # still escapes to the drive root, so the root and drive are checked too.
            if (source.is_absolute() or source.root or source.drive
                    or any(part in {"", ".", ".."} for part in source.parts)):
                raise ValueError("Invalid staged model source")
            return PROJECT / source
    raise ValueError(f"Unknown model key: {key}")


def verified_staged_copy(key: str, spec: dict[str, Any],
                         verify: Callable[[Path, dict[str, Any]], Any]) -> Path | None:
    """Return the staged directory only when it holds the exact pinned bytes."""
    source = staging_directory(key)
    if source.is_symlink() or not source.is_dir():
        return None
    try:
        verify(source, spec)
    except Exception:
        # A damaged or outdated stage is not an error here: the caller downloads.
        return None
    return source


def install_from_staging(key: str, directory: Path, spec: dict[str, Any],
                         verify: Callable[[Path, dict[str, Any]], Any]) -> bool:
    """Publish a staged model into `directory` atomically; False means nothing staged.

    The destination appears only after the copy verifies, so an interrupted copy
    never leaves a directory that looks installed.
    """
    source = verified_staged_copy(key, spec, verify)
    if source is None:
        return False
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staged-model-", dir=directory.parent) as temporary:
        pending = Path(temporary) / "model"
        pending.mkdir()
        for item in spec["files"]:
            shutil.copyfile(source / item["name"], pending / item["name"])
        verify(pending, spec)
        if directory.exists():
            if any(directory.iterdir()):
                return False  # A non-empty destination is the caller's to judge.
            directory.rmdir()
        pending.rename(directory)
    return True


def stage_from(key: str, source: Path, spec: dict[str, Any],
               verify: Callable[[Path, dict[str, Any]], Any]) -> bool:
    """Fill this project's stage from an already verified directory elsewhere."""
    target = staging_directory(key)
    if verified_staged_copy(key, spec, verify) is not None:
        return False
    verify(Path(source), spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-model-", dir=target.parent) as temporary:
        pending = Path(temporary) / "model"
        pending.mkdir()
        for item in spec["files"]:
            shutil.copyfile(Path(source) / item["name"], pending / item["name"])
        verify(pending, spec)
        if target.exists() and not any(target.iterdir()):
            target.rmdir()
        if target.exists():
            return False
        pending.rename(target)
    return True
