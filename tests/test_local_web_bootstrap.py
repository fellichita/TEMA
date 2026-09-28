"""A local web launch prepares the actual analysis runtime before showing the page."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from app.identity import default_data_dir
from app.pilot.settings import PilotSettings, save_settings
from scripts import local_web_bootstrap, run_web_demo, setup_models


def test_local_web_launcher_is_an_executable_shell_file():
    launcher = local_web_bootstrap.ROOT / "launchers/macos/web-local.command"
    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)
    if sys.platform != "win32":
        result = subprocess.run(["/bin/bash", "-n", str(launcher)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_bootstrap_prepares_environment_profile_and_models_before_server(monkeypatch):
    events = []
    monkeypatch.setattr(local_web_bootstrap.mac_bootstrap, "main",
                        lambda args: events.append(("base", args)) or 0)
    monkeypatch.setattr(local_web_bootstrap.mac_bootstrap, "installation_lock", lambda _: nullcontext())
    monkeypatch.setattr(local_web_bootstrap, "ensure_web_runtime", lambda _: events.append("web"))
    monkeypatch.setattr(local_web_bootstrap, "prepare_web_profile",
                        lambda _, profile: events.append(("profile", profile)))

    def call(command, **_kwargs):
        events.append(command)
        return 0

    monkeypatch.setattr(local_web_bootstrap.subprocess, "call", call)
    assert local_web_bootstrap.main([]) == 0
    assert events[:3] == [
        ("base", ["--install-only"]), "web", ("profile", local_web_bootstrap.WEB_PROFILE),
    ]
    model_command, server_command = events[3:]
    assert model_command[-3:] == ["--web-analysis", "--profile", str(local_web_bootstrap.WEB_PROFILE)]
    assert server_command[-5:] == ["--local-only", "--open-browser", "--no-auth", "--data-dir",
                                   str(local_web_bootstrap.WEB_PROFILE)]


def test_bootstrap_stops_before_server_if_models_fail(monkeypatch):
    monkeypatch.setattr(local_web_bootstrap.mac_bootstrap, "main", lambda _: 0)
    monkeypatch.setattr(local_web_bootstrap.mac_bootstrap, "installation_lock", lambda _: nullcontext())
    monkeypatch.setattr(local_web_bootstrap, "ensure_web_runtime", lambda _: None)
    monkeypatch.setattr(local_web_bootstrap, "prepare_web_profile", lambda *_: None)
    calls = []
    monkeypatch.setattr(local_web_bootstrap.subprocess, "call",
                        lambda command, **_: calls.append(command) or 1)
    assert local_web_bootstrap.main([]) == 1
    assert len(calls) == 1
    assert "scripts.setup_models" in calls[0]


def test_install_only_does_not_start_server_and_ready_web_skips_install(monkeypatch, tmp_path):
    monkeypatch.setattr(local_web_bootstrap, "web_runtime_status", lambda _: (True, "ready"))
    monkeypatch.setattr(local_web_bootstrap, "install_web",
                        lambda _: pytest.fail("Already ready web environment was reinstalled"))
    local_web_bootstrap.ensure_web_runtime(tmp_path)

    monkeypatch.setattr(local_web_bootstrap.mac_bootstrap, "main", lambda _: 0)
    monkeypatch.setattr(local_web_bootstrap.mac_bootstrap, "installation_lock", lambda _: nullcontext())
    monkeypatch.setattr(local_web_bootstrap, "prepare_web_profile", lambda *_: None)
    calls = []
    monkeypatch.setattr(local_web_bootstrap.subprocess, "call",
                        lambda command, **_: calls.append(command) or 0)
    assert local_web_bootstrap.main(["--install-only"]) == 0
    assert len(calls) == 1 and "scripts.setup_models" in calls[0]


def test_web_profile_settings_do_not_change_desktop_profile(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    desktop = default_data_dir()
    assert desktop.is_relative_to(home)
    desktop.mkdir(parents=True)
    save_settings(desktop, PilotSettings.for_provider("deepseek"))
    desktop_settings = (desktop / "settings.json").read_bytes()
    project = tmp_path / "checkout"
    web = project / "storage" / "web-local-profile"
    save_settings(web, PilotSettings.for_provider("deepseek"))

    result = subprocess.run([sys.executable, "-X", "utf8", "-E", "-s", "-c",
                             local_web_bootstrap.PROFILE_SETUP,
                             str(web), str(project)],
                            cwd=local_web_bootstrap.ROOT, env=os.environ.copy(),
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert json.loads((web / "settings.json").read_text(encoding="utf-8"))["provider"] == "local"
    assert (desktop / "settings.json").read_bytes() == desktop_settings


def test_local_server_ignores_inherited_auth_and_passes_isolated_profile(monkeypatch, tmp_path, capsys):
    api_python = tmp_path / "api-python"
    web_python = tmp_path / "web-python"
    api_python.touch()
    web_python.touch()
    monkeypatch.setattr(run_web_demo, "API_PYTHON", api_python)
    monkeypatch.setattr(run_web_demo, "WEB_PYTHON", web_python)
    monkeypatch.setenv("TREND_WEB_REQUIRE_AUTH", "1")
    monkeypatch.setenv("TREND_WEB_ACCESS_PASSWORD", "stale-password-123")
    monkeypatch.setenv("TREND_API_SEPARATE_VISITORS", "1")
    monkeypatch.setattr(run_web_demo, "cloudflared", lambda: pytest.fail("Local launch requested tunnel"))
    monkeypatch.setattr(run_web_demo, "_busy_local_ports", lambda *_: ())
    monkeypatch.setattr(run_web_demo, "_wait_health", lambda *_, **__: None)
    monkeypatch.setattr(run_web_demo, "_wait_web", lambda *_, **__: None)
    monkeypatch.setattr(run_web_demo, "keep_awake", lambda _: None)
    monkeypatch.setattr(run_web_demo.signal, "signal", lambda *_: None)
    monkeypatch.setattr(run_web_demo, "start_admin_panel", lambda **_: None)
    browser = []
    monkeypatch.setattr(run_web_demo.webbrowser, "open", lambda url: browser.append(url))
    launched = []

    class Process:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return Process()

    monkeypatch.setattr(run_web_demo.subprocess, "Popen", popen)

    class EndRun(Exception):
        pass

    monkeypatch.setattr(run_web_demo.time, "sleep", lambda _: (_ for _ in ()).throw(EndRun()))
    profile = tmp_path / "web data"
    with pytest.raises(EndRun):
        run_web_demo.main(["--local-only", "--no-auth", "--open-browser", "--data-dir", str(profile)])
    assert len(launched) == 2
    api_command, api_options = launched[0]
    assert api_command[-2:] == ["--data-dir", str(profile)]
    assert "TREND_WEB_REQUIRE_AUTH" not in api_options["env"]
    assert "TREND_WEB_ACCESS_PASSWORD" not in api_options["env"]
    # Without a password there are no visitors to keep apart, whatever the parent set.
    assert "TREND_API_SEPARATE_VISITORS" not in api_options["env"]
    assert launched[1][1]["env"]["TREND_WEB_REQUIRE_AUTH"] == "0"
    assert browser == ["http://127.0.0.1:8501"]
    assert "Пароль:" not in capsys.readouterr().out


def test_no_auth_is_rejected_for_public_tunnel():
    with pytest.raises(SystemExit) as error:
        run_web_demo.main(["--no-auth"])
    assert error.value.code == 2


def test_web_analysis_selects_the_models_needed_to_finish_a_search(tmp_path):
    models = setup_models._models(tmp_path / "profile", web_analysis=True)

    # The English→Russian model translates the web TOP; without it cards stay in the original.
    assert [model.key for model in models] == [
        "multilingual-e5-small", "opus-mt-ru-en", "opus-mt-en-ru", "qwen2.5-1.5b-instruct",
    ]
    assert all(model.runtime.is_relative_to(tmp_path / "profile") for model in models)
    assert models[-1].staged is False
    # The build's model lock has no reading model: it is copied profile to profile.
    assert models[2].staged is False and models[2].reuse_source is not None


def test_web_reading_model_accepts_setup_progress(tmp_path, monkeypatch):
    from scripts import install_translation_model

    calls = []
    monkeypatch.setattr(install_translation_model, "install",
                        lambda directory, model_key: calls.append((directory, model_key)))
    reading = setup_models._models(tmp_path / "profile", web_analysis=True)[2]

    reading.download(reading.runtime, progress=lambda *_: None)

    assert calls == [(reading.runtime, "opus-mt-en-ru")]


def test_web_ai_model_is_downloaded_once_and_verified_on_reuse(tmp_path):
    target = tmp_path / "profile" / "models" / "local-ai"
    downloads = []

    def verify(directory: Path, _spec):
        if (directory / "model.onnx").read_bytes() != b"pinned":
            raise ValueError("wrong weights")

    def download(directory: Path):
        downloads.append(directory)
        directory.mkdir(parents=True)
        (directory / "model.onnx").write_bytes(b"pinned")

    model = setup_models.Model(
        "local-ai", "local AI", {"files": [{"bytes": 6}]}, verify,
        target, download, staged=False,
    )
    assert setup_models.prepare(model, allow_download=True)["state"] == "ready"
    assert setup_models.prepare(model, allow_download=True)["state"] == "ready"
    assert downloads == [target]

    (target / "model.onnx").write_bytes(b"broken")
    assert setup_models.prepare(model, allow_download=False)["state"] == "missing"
    assert downloads == [target], "Offline checking must not fetch damaged weights"


def test_web_ai_model_is_not_reported_ready_after_a_bad_download(tmp_path):
    target = tmp_path / "profile" / "models" / "local-ai"

    def verify(directory: Path, _spec):
        if (directory / "model.onnx").read_bytes() != b"pinned":
            raise ValueError("wrong weights")

    def download(directory: Path):
        directory.mkdir(parents=True)
        (directory / "model.onnx").write_bytes(b"broken")

    model = setup_models.Model(
        "local-ai", "local AI", {"files": [{"bytes": 6}]}, verify,
        target, download, staged=False,
    )
    assert setup_models.prepare(model, allow_download=True)["state"] != "ready"


def test_web_profile_reuses_only_verified_ai_weights_from_desktop(tmp_path):
    source = tmp_path / "desktop" / "models" / "local-ai"
    source.mkdir(parents=True)
    (source / "model.onnx").write_bytes(b"pinned")
    target = tmp_path / "web" / "models" / "local-ai"

    def verify(directory: Path, _spec):
        if (directory / "model.onnx").read_bytes() != b"pinned":
            raise ValueError("wrong weights")

    model = setup_models.Model(
        "local-ai", "local AI", {"files": [{"name": "model.onnx", "bytes": 6}]}, verify,
        target, lambda _: pytest.fail("Verified local weights should avoid the network"),
        staged=False, reuse_source=source,
    )
    assert setup_models.prepare(model, allow_download=True)["state"] == "ready"
    assert (target / "model.onnx").read_bytes() == b"pinned"
    assert (source / "model.onnx").read_bytes() == b"pinned"

    (source / "model.onnx").write_bytes(b"broken")
    (target / "model.onnx").unlink()
    target.rmdir()
    assert setup_models.prepare(model, allow_download=False)["state"] == "missing"


def test_web_model_setup_shows_download_progress_and_keeps_json_clean(tmp_path, monkeypatch, capsys):
    target = tmp_path / "profile" / "models" / "local-ai"

    def verify(directory: Path, _spec):
        if (directory / "model.onnx").read_bytes() != b"pinned":
            raise ValueError("wrong weights")

    def download(directory: Path, *, progress=None):
        assert progress is not None
        directory.mkdir(parents=True)
        progress(3, 10)
        progress(10, 10)
        (directory / "model.onnx").write_bytes(b"pinned")

    model = setup_models.Model(
        "local-ai", "local AI", {"files": [{"name": "model.onnx", "bytes": 6}]},
        verify, target, download, staged=False,
    )
    monkeypatch.setattr(setup_models, "_models", lambda *_args, **_kwargs: [model])
    assert setup_models.main(["--web-analysis", "--profile", str(tmp_path / "profile")]) == 0
    output = capsys.readouterr().out
    assert "30%" in output and "100%" in output

    assert setup_models.main(["--web-analysis", "--json", "--profile", str(tmp_path / "profile")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["models"][0]["state"] == "ready"
