"""A frozen web launch keeps its mutable data outside the application bundle."""

from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest

from app.pilot.settings import PilotSettings, load_settings, save_settings
from app.runtime.model_resources import ModelLocation
from scripts import frozen_web_setup
from scripts import install_local_llm as local_installer


def _ready_setup(monkeypatch, tmp_path):
    user = tmp_path / "user-data"
    bundle = tmp_path / "Trendanalyser.app"
    bundle.mkdir()
    monkeypatch.setattr(frozen_web_setup, "frozen", lambda: True)
    monkeypatch.setattr(frozen_web_setup, "default_data_dir", lambda: user)
    monkeypatch.setattr(frozen_web_setup, "_bundle_roots", lambda: (bundle,))
    monkeypatch.setattr(frozen_web_setup, "_verify_bundled_models", lambda _: None)
    monkeypatch.setattr(frozen_web_setup, "_prepare_local_ai", lambda *_: None)
    return user, bundle


def test_first_frozen_web_launch_uses_user_data_and_local_settings(monkeypatch, tmp_path):
    user, bundle = _ready_setup(monkeypatch, tmp_path)
    notices = []
    profile = frozen_web_setup.prepare_frozen_web_profile(progress=notices.append)

    assert profile == user / "web-local-profile"
    assert profile.is_dir()
    assert load_settings(profile).provider == "local"
    assert any("встроенные модели" in message for message in notices)
    assert list(bundle.iterdir()) == []


def test_existing_web_settings_switch_to_local_without_touching_desktop(monkeypatch, tmp_path):
    user, _bundle = _ready_setup(monkeypatch, tmp_path)
    profile = user / "web-local-profile"
    save_settings(user, PilotSettings.for_provider("deepseek"))
    desktop_bytes = (user / "settings.json").read_bytes()
    save_settings(profile, PilotSettings.for_provider("deepseek"))

    assert frozen_web_setup.prepare_frozen_web_profile() == profile
    assert load_settings(profile).provider == "local"
    assert (user / "settings.json").read_bytes() == desktop_bytes


def test_frozen_web_rejects_profile_inside_installation_before_writes(monkeypatch, tmp_path):
    _user, bundle = _ready_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(frozen_web_setup, "default_data_dir", lambda: bundle / "private-data")
    with pytest.raises(frozen_web_setup.WebSetupError, match="внутри установленной программы"):
        frozen_web_setup.prepare_frozen_web_profile()
    assert not (bundle / "private-data").exists()


def test_frozen_web_rejects_redirected_profile_without_touching_target(monkeypatch, tmp_path):
    user, _bundle = _ready_setup(monkeypatch, tmp_path)
    user.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    marker = other / "sentinel"
    marker.write_text("keep", encoding="utf-8")
    if sys.platform == "win32":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(user / "web-local-profile"), str(other)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
    else:
        (user / "web-local-profile").symlink_to(other, target_is_directory=True)
    with pytest.raises(frozen_web_setup.WebSetupError, match="перенаправлен"):
        frozen_web_setup.prepare_frozen_web_profile()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_frozen_web_rejects_library_pointer(monkeypatch, tmp_path):
    user, _bundle = _ready_setup(monkeypatch, tmp_path)
    profile = user / "web-local-profile"
    profile.mkdir(parents=True)
    pointer = profile / "active-library.json"
    pointer.write_text('{"version":1,"path":"/somewhere"}', encoding="utf-8")
    before = pointer.read_bytes()
    with pytest.raises(frozen_web_setup.WebSetupError, match="перенаправлен на другую библиотеку"):
        frozen_web_setup.prepare_frozen_web_profile()
    assert pointer.read_bytes() == before


def test_frozen_web_checks_bundled_models_without_copying_them(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    checked = []
    specs = {"multilingual-e5-small": {"revision": "a" * 40},
             "opus-mt-ru-en": {"revision": "b" * 40}}
    monkeypatch.setattr(frozen_web_setup.encoder, "load_spec", lambda: specs["multilingual-e5-small"])
    monkeypatch.setattr(frozen_web_setup.translator, "load_spec", lambda: specs["opus-mt-ru-en"])

    def resolve(key, revision):
        assert specs[key]["revision"] == revision
        return ModelLocation(bundle / key, "bundled", key, revision)

    monkeypatch.setattr(frozen_web_setup, "resolve_model", resolve)
    monkeypatch.setattr(frozen_web_setup.encoder, "verify_artifacts",
                        lambda directory, spec, cancel: checked.append((directory, spec)))
    monkeypatch.setattr(frozen_web_setup.translator, "verify_artifacts",
                        lambda directory, spec, cancel: checked.append((directory, spec)))
    frozen_web_setup._verify_bundled_models(None)
    assert checked == [(bundle / key, specs[key]) for key in specs]
    assert not bundle.exists()


def test_frozen_web_reuses_only_verified_desktop_ai(monkeypatch, tmp_path):
    user = tmp_path / "user"
    profile = user / "web-local-profile"
    spec = {"files": [{"bytes": 10}]}
    monkeypatch.setattr(frozen_web_setup, "default_data_dir", lambda: user)
    monkeypatch.setattr(frozen_web_setup.local_llm, "load_spec", lambda: spec)
    copied = []
    monkeypatch.setattr(frozen_web_setup, "_copy_verified",
                        lambda source, target, model: copied.append((source, target, model.key)) or True)
    monkeypatch.setattr(frozen_web_setup, "install_local_llm",
                        lambda *_args, **_kwargs: pytest.fail("Verified desktop model was downloaded"))
    messages = []
    frozen_web_setup._prepare_local_ai(profile, messages.append, None)
    assert copied == [(user / "models" / frozen_web_setup.local_llm.MODEL_KEY,
                       profile / "models" / frozen_web_setup.local_llm.MODEL_KEY,
                       frozen_web_setup.local_llm.MODEL_KEY)]
    assert any("скопирована" in message for message in messages)


def test_frozen_web_ai_download_reports_progress_and_honours_cancel(monkeypatch, tmp_path):
    user = tmp_path / "user"
    profile = user / "web-local-profile"
    cancel = Event()
    monkeypatch.setattr(frozen_web_setup, "default_data_dir", lambda: user)
    monkeypatch.setattr(frozen_web_setup.local_llm, "load_spec", lambda: {"files": [{"bytes": 100}]})
    monkeypatch.setattr(frozen_web_setup, "_copy_verified", lambda *_: False)
    seen = []

    def install(target: Path, *, cancel: Event, progress):
        seen.append((target, cancel))
        progress(25, 100)
        progress(31, 100)
        progress(100, 100)

    monkeypatch.setattr(frozen_web_setup, "install_local_llm", install)
    messages = []
    frozen_web_setup._prepare_local_ai(profile, messages.append, cancel)
    assert seen == [(profile / "models" / frozen_web_setup.local_llm.MODEL_KEY, cancel)]
    assert [message.rsplit("(", 1)[-1] for message in messages] == ["20%).", "30%).", "100%)."]

    cancel.set()
    with pytest.raises(frozen_web_setup.WebSetupCancelled, match="прервана"):
        frozen_web_setup._prepare_local_ai(profile, messages.append, cancel)


def test_frozen_ai_installer_does_not_read_unbundled_source_lock(monkeypatch, tmp_path):
    target = tmp_path / "user" / "models" / "local-ai"
    spec = {"revision": "a" * 40, "files": [{"name": "model.onnx", "bytes": 4, "sha256": "unused"}]}
    monkeypatch.setattr(local_installer, "frozen", lambda: True)
    monkeypatch.setattr(local_installer.local_llm, "load_spec", lambda: spec)
    monkeypatch.setattr(local_installer, "install_from_staging",
                        lambda *_args: pytest.fail("Frozen installer tried to read source staging"))

    def verify(directory, _spec, cancel=None):
        assert (directory / "model.onnx").read_bytes() == b"test"

    def download(_client, _spec, _item, output, cancel=None, progress=None):
        output.write_bytes(b"test")
        if progress is not None:
            progress(4, 4)

    monkeypatch.setattr(local_installer.local_llm, "verify_artifacts", verify)
    monkeypatch.setattr(local_installer, "_download", download)
    result = local_installer.install(target)
    assert result["source"] == "download"
    assert (target / "model.onnx").read_bytes() == b"test"
