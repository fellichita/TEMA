"""Frozen defaults must use the app's own resources, independent of cwd or caches."""

from pathlib import Path

import pytest

from app.runtime import model_resources as resources
from tests.platform_support import require_symlinks


REVISION = "a" * 40


def test_source_default_and_explicit_directory(monkeypatch, tmp_path):
    monkeypatch.delattr(resources.sys, "frozen", raising=False)
    source = tmp_path / "developer model"
    assert resources.resolve_model("e5-small-v2", REVISION,
                                   development_default=source).path == source
    explicit = tmp_path / "test model"
    location = resources.resolve_model("e5-small-v2", REVISION, explicit_dir=explicit,
                                       development_default=source)
    assert (location.path, location.origin) == (explicit, "explicit")


def test_frozen_default_never_uses_user_model_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(tmp_path / "Unicode модель"), raising=False)
    previous = tmp_path / "user" / "models"
    previous.mkdir(parents=True)
    result = resources.resolve_model("multilingual-e5-small", REVISION, development_default=previous)
    assert result.origin == "bundled"
    assert result.path == tmp_path / "Unicode модель" / "app/bundled_models/multilingual-e5-small" / REVISION
    assert result.path != previous


def test_invalid_identity_or_missing_frozen_root_is_rejected(monkeypatch):
    for key, revision in (("unknown", REVISION), ("e5-small-v2", "../" + REVISION)):
        with pytest.raises(ValueError, match="Unknown"):
            resources.resolve_model(key, revision, development_default=Path("model"))
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
    monkeypatch.delattr(resources.sys, "_MEIPASS", raising=False)
    with pytest.raises(ValueError, match="root"):
        resources.resolve_model("e5-small-v2", REVISION, development_default=Path("model"))


def test_frozen_model_directory_cannot_escape_through_parent_symlink(monkeypatch, tmp_path):
    require_symlinks()
    root = tmp_path / "application" / "_internal"
    root.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (root / "app").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(root), raising=False)
    with pytest.raises(ValueError, match="escapes"):
        resources.resolve_model("e5-small-v2", REVISION)
