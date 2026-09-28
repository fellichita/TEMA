"""Disposable cache writes must remain inside their owning profile."""

import pytest

from app.pilot.service import PilotService
from tests.platform_support import require_symlinks


def _service(data_dir):
    service = object.__new__(PilotService)
    service.data_dir = data_dir
    return service


@pytest.mark.parametrize("alias", ["cache", "analyses", "key"])
def test_analysis_cache_symlink_cannot_redirect_write(tmp_path, alias):
    require_symlinks()
    outside = tmp_path / "outside"
    outside.mkdir()
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    cache = data_dir / "cache"
    key = "a" * 64
    if alias != "cache":
        cache.mkdir()
    if alias == "key":
        (cache / "analyses").mkdir()
    redirect = {"cache": cache, "analyses": cache / "analyses",
                "key": cache / "analyses" / key}[alias]
    redirect.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="symbolic link"):
        _service(data_dir)._cache_write(key, "plan", {"value": 1})
    assert list(outside.iterdir()) == []


def test_shared_cache_symlink_failure_is_disposable(tmp_path):
    require_symlinks()
    outside = tmp_path / "outside"
    outside.mkdir()
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    (data_dir / "cache").symlink_to(outside, target_is_directory=True)

    _service(data_dir)._shared_cache_write("discovery", "a" * 64, {"value": 1})
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("key,name", [("../escape", "plan"), ("a" * 64, "../escape")])
def test_analysis_cache_rejects_path_components(tmp_path, key, name):
    data_dir = tmp_path / "profile"
    data_dir.mkdir()
    with pytest.raises(ValueError, match="Invalid cache key"):
        _service(data_dir)._cache_write(key, name, {"value": 1})
