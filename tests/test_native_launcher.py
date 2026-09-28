"""The packaged start window must isolate services and clean up failed launches."""

from queue import Queue

from app import launcher
from scripts import frozen_web_setup


def test_child_environments_keep_provider_secrets_out_of_browser_process(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-provider-secret")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/modules")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/python")

    api, web = launcher._child_environments(12345, "test-instance")

    assert api["TREND_API_INSTANCE_ID"] == "test-instance"
    assert web["API_URL"] == "http://127.0.0.1:12345"
    assert api["TREND_API_TOKEN"] == web["TREND_API_TOKEN"]
    assert "DEEPSEEK_API_KEY" not in web
    assert "PYTHONPATH" not in api and "PYTHONPATH" not in web
    assert "PYTHONHOME" not in api and "PYTHONHOME" not in web
    assert api["PYINSTALLER_RESET_ENVIRONMENT"] == web["PYINSTALLER_RESET_ENVIRONMENT"] == "1"


def test_service_ports_are_distinct_loopback_ephemeral_ports():
    api, web = launcher._free_loopback_ports()
    assert 1024 <= api <= 65535
    assert 1024 <= web <= 65535
    assert api != web


def test_failed_web_ui_stops_both_bundled_children(tmp_path, monkeypatch):
    profile = tmp_path / "web-profile"
    profile.mkdir()
    monkeypatch.setattr(frozen_web_setup, "prepare_frozen_web_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(launcher, "_free_loopback_ports", lambda: (18000, 18500))
    monkeypatch.setattr(launcher, "_child_environments", lambda *_args: ({}, {}))
    monkeypatch.setattr(launcher, "_child_commands", lambda *_args: (["api"], ["web"]))

    started = []

    class Child:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.stopped = False
            started.append(self)

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout=None):
            assert self.stopped
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Child)

    def health(process, *_args, **_kwargs):
        if process.command == ["web"]:
            raise launcher.LauncherError("Веб не запустился")

    monkeypatch.setattr(launcher, "_wait_for_service", health)
    session = launcher._WebSession()
    events: Queue[tuple[launcher._WebSession, str, str]] = Queue()

    launcher._start_web(session, events)

    assert len(started) == 2
    assert all(child.stopped for child in started)
    assert events.get_nowait() == (session, "status", "Запускаем локальные сервисы…")
    assert events.get_nowait() == (session, "error", "Веб не запустился")
    assert session.cancel.is_set()


def test_child_command_uses_only_the_installed_executable(tmp_path):
    binary = tmp_path / "Trendanalyser"
    profile = tmp_path / "profile"
    api, web = launcher._child_commands(binary, profile, 18000, 18500)
    assert api == [str(binary), "--web-api", "--port", "18000", "--data-dir", str(profile)]
    assert web == [str(binary), "--web-ui", "--port", "18500"]
    assert all(".venv" not in part for part in (*api, *web))
