"""Locate read-only model resources without importing inference or network code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Literal


ModelOrigin = Literal["bundled", "explicit", "development"]
MODEL_KEYS = frozenset({"multilingual-e5-small", "e5-small-v2", "opus-mt-ru-en", "opus-mt-en-ru",
                        "qwen2.5-1.5b-instruct"})


@dataclass(frozen=True)
class ModelLocation:
    path: Path
    origin: ModelOrigin
    model_key: str
    revision: str


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundled_root() -> Path:
    """PyInstaller places data paths relative to its immutable resource root."""
    if not frozen():
        raise ValueError("Bundled model resources exist only in a frozen application")
    root = getattr(sys, "_MEIPASS", None)
    if not isinstance(root, str) or not root:
        raise ValueError("Frozen application resource root is unavailable")
    return Path(root) / "app" / "bundled_models"


def _inside_current_application(path: Path) -> bool:
    resource_root = bundled_root().parents[1].resolve(strict=False)
    # macOS stores datas in Contents/Resources and imports from a symlink in
    # Contents/Frameworks. Both locations are part of the same signed app.
    allowed = resource_root.parent if sys.platform == "darwin" else resource_root
    return path.resolve(strict=False).is_relative_to(allowed)


def resolve_model(model_key: str, revision: str, *, explicit_dir: Path | None = None,
                  development_default: Path | None = None) -> ModelLocation:
    """Choose a path; encoders still verify all files before inference."""
    if model_key not in MODEL_KEYS or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Unknown model identity")
    if explicit_dir is not None:
        return ModelLocation(Path(explicit_dir), "explicit", model_key, revision)
    if frozen():
        path = bundled_root() / model_key / revision
        if not _inside_current_application(path):
            raise ValueError("Bundled model resource escapes the application")
        return ModelLocation(path, "bundled", model_key, revision)
    if development_default is None:
        raise ValueError("Source model directory is not configured")
    return ModelLocation(Path(development_default), "development", model_key, revision)
