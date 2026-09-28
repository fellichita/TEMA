"""Startup should not advertise a Cloudflare URL before it is connected."""

import subprocess
import sys

import pytest

import scripts.run_web_demo as launcher
from scripts.run_web_demo import (LaunchError, _child_environments, _tunnel_environment,
                                  _access_password, _wait_tunnel, _web_command)


def test_public_launcher_isolates_secrets_and_bounds_streamlit_payloads(monkeypatch):
    monkeypatch.setenv("TREND_WEB_ACCESS_PASSWORD", "inherited-secret")
    monkeypatch.setenv("TREND_API_TOKEN", "inherited-token")
    monkeypatch.setenv("UNRELATED_SERVER_SECRET", "private-value")
    api, web = _child_environments("strong-demo-password", "fresh-token-" + "x" * 32)
    assert "TREND_WEB_ACCESS_PASSWORD" not in api
    assert api["TREND_API_TOKEN"] == web["TREND_API_TOKEN"] == "fresh-token-" + "x" * 32
    assert api["UNRELATED_SERVER_SECRET"] == "private-value"
    assert "UNRELATED_SERVER_SECRET" not in web
    assert web["TREND_WEB_ACCESS_PASSWORD"] == "strong-demo-password"
    assert len(web["STREAMLIT_SERVER_COOKIE_SECRET"]) >= 32
    tunnel = _tunnel_environment()
    assert not {"TREND_WEB_ACCESS_PASSWORD", "TREND_WEB_REQUIRE_AUTH", "TREND_API_TOKEN"} & tunnel.keys()
    assert "UNRELATED_SERVER_SECRET" not in tunnel

    command = _web_command()
    settings = dict(zip(command[7::2], command[8::2], strict=True))
    assert settings["--server.address"] == "127.0.0.1"
    assert settings["--server.maxMessageSize"] == "8"
    assert settings["--server.maxUploadSize"] == "1"
    assert settings["--server.enableXsrfProtection"] == "true"
    assert settings["--server.enableCORS"] == "true"
    assert settings["--server.enableStaticServing"] == "false"
    assert command[-4:] == ["--server.allowedHosts", "localhost", "--server.allowedHosts", "127.0.0.1"]
    assert _web_command(public=True)[-2:] == ["--server.allowedHosts", "*.trycloudflare.com"]
    custom = _web_command(public=True, port=18501)
    assert custom[custom.index("--server.port") + 1] == "18501"
    assert _child_environments("strong-demo-password", "x" * 32, api_port=18000)[1]["API_URL"] == (
        "http://127.0.0.1:18000")


@pytest.mark.skipif(sys.platform != "win32", reason="имена переменных без учёта регистра только в Windows")
def test_windows_children_keep_system_root_whatever_its_spelling():
    import os

    # Python на Windows отдаёт «SYSTEMROOT»; без неё сокеты в Streamlit не создаются.
    assert "SYSTEMROOT" in os.environ
    _, web = _child_environments(None, "x" * 32)
    assert web["SYSTEMROOT"] == os.environ["SYSTEMROOT"]
    assert "SYSTEMROOT" in _tunnel_environment()


def test_public_access_uses_fresh_random_password(monkeypatch):
    monkeypatch.setenv("TREND_WEB_ACCESS_PASSWORD", "weak-operator-password")
    first = _access_password(local_only=False, no_auth=False)
    second = _access_password(local_only=False, no_auth=False)
    assert first != second
    assert first != "weak-operator-password"
    assert len(first) >= 32
    assert _access_password(local_only=True, no_auth=False) == "weak-operator-password"


def test_launcher_waits_for_its_own_api_instance(monkeypatch):
    class Response:
        status = 200
        headers = {"X-Trend-Instance": "old-instance-0123456789abcdef"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class Process:
        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 1

    monkeypatch.setattr(launcher._LOCAL_OPENER, "open", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(launcher.time, "sleep", lambda *_: None)
    with pytest.raises(LaunchError, match="не запустился"):
        launcher._wait_health(Process(), "new-instance-0123456789abcdef", timeout=1)


def test_public_link_keeps_visitors_apart_in_their_own_profile(monkeypatch, tmp_path, capsys):
    api_python, web_python = tmp_path / "api-python", tmp_path / "web-python"
    api_python.touch()
    web_python.touch()
    monkeypatch.setattr(launcher, "API_PYTHON", api_python)
    monkeypatch.setattr(launcher, "WEB_PYTHON", web_python)
    monkeypatch.setattr(launcher, "fxtunnel", lambda: tmp_path / "fxtunnel.exe")
    monkeypatch.setattr(launcher, "cloudflared", lambda: pytest.fail("Russian visitors cannot reach Cloudflare"))
    monkeypatch.setattr(launcher, "_busy_local_ports", lambda *_: ())
    monkeypatch.setattr(launcher, "_wait_health", lambda *_, **__: None)
    monkeypatch.setattr(launcher, "_wait_web", lambda *_, **__: None)
    monkeypatch.setattr(launcher, "_wait_tunnel", lambda *_args, **_kwargs: "https://trendanalizer.fxtun.ru")
    monkeypatch.setattr(launcher, "keep_awake", lambda _: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    monkeypatch.setattr(launcher, "start_admin_panel", lambda **_: None)
    prepared, launched, access = [], [], []
    monkeypatch.setattr(launcher, "prepare_public_profile", prepared.append)
    access_file = tmp_path / "access.txt"
    monkeypatch.setattr(launcher, "ACCESS_FILE", access_file)
    # The processes stop first; the access file is read at that moment, then removed.
    monkeypatch.setattr(launcher, "_stop", lambda _processes: access.append(access_file.read_text(encoding="utf-8")))

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return Process()

    class EndRun(Exception):
        pass

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: (_ for _ in ()).throw(EndRun()))
    with pytest.raises(EndRun):
        launcher.main([])
    # The owner's history, library and settings stay out of the public site.
    assert prepared == [launcher.PUBLIC_PROFILE]
    api_command, api_options = launched[0]
    assert api_command[-2:] == ["--data-dir", str(launcher.PUBLIC_PROFILE)]
    assert api_options["env"]["TREND_API_SEPARATE_VISITORS"] == "1"
    web_command, web_options = launched[1]
    assert web_options["env"]["TREND_WEB_REQUIRE_AUTH"] == "1"
    assert web_command[-2:] == ["--server.allowedHosts", "*.fxtun.ru"]
    tunnel_command, tunnel_options = launched[2]
    # A permanent Russian address; the local traffic inspector would record visitors' passwords.
    assert tunnel_command == [str(tmp_path / "fxtunnel.exe"), "http", "8501", "--no-inspect",
                              "--log-level", "info", "--domain", "trendanalizer"]
    assert tunnel_options["encoding"] == "utf-8" and tunnel_options["env"]["NO_COLOR"] == "1"
    output = capsys.readouterr().out
    assert "https://trendanalizer.fxtun.ru" in output and "Пароль:" in output
    assert "каждый вошедший видит только свои анализы" in output and "Адрес постоянный" in output
    assert access[0].startswith("Ссылка: https://trendanalizer.fxtun.ru\nПароль: ")
    assert not access_file.exists()


def test_cloudflare_remains_available_over_tcp():
    tunnel = launcher.cloudflare_tunnel(launcher.Path("cloudflared"), 8501)
    # TCP gets through proxies and VPNs where QUIC does not.
    assert tunnel.command[-2:] == ("--protocol", "http2") and tunnel.allowed_hosts == "*.trycloudflare.com"
    assert not tunnel.permanent


def test_fxtunnel_address_is_read_through_colours_and_bad_tokens_are_explained():
    tunnel = launcher.fxtunnel_tunnel(launcher.Path("fxtunnel"), 8501, "trendanalizer")
    process = _process(["\x1b[90mConnecting to fxtunnel server...\x1b[0m",
                        "  \u2192 \x1b[1mhttps://trendanalizer.fxtun.ru\x1b[0m"], stay_alive=True)
    try:
        assert _wait_tunnel(process, timeout=5, tunnel=tunnel) == "https://trendanalizer.fxtun.ru"
    finally:
        process.kill()
    refused = _process(["  Failed to connect: authenticate: authentication failed: invalid token"],
                       stay_alive=False)
    with pytest.raises(LaunchError, match="fxtunnel login"):
        _wait_tunnel(refused, timeout=5, tunnel=tunnel)


def test_windows_finds_cloudflared_installed_after_the_window_opened(monkeypatch, tmp_path):
    installed = tmp_path / "cloudflared.exe"
    installed.touch()
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    monkeypatch.setattr(launcher, "CLOUDFLARED_FALLBACKS", (tmp_path / "missing", installed))
    assert launcher.cloudflared() == installed


def test_local_launcher_explains_busy_ports_before_spawning(monkeypatch, tmp_path, capsys):
    api_python = tmp_path / "api-python"
    web_python = tmp_path / "web-python"
    api_python.touch()
    web_python.touch()
    monkeypatch.setattr(launcher, "API_PYTHON", api_python)
    monkeypatch.setattr(launcher, "WEB_PYTHON", web_python)
    monkeypatch.setattr(launcher, "_busy_local_ports", lambda *_: (8000, 8501))
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    monkeypatch.setattr(launcher, "start_admin_panel", lambda **_: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs:
                        pytest.fail("A second server must not be started"))
    assert launcher.main(["--local-only", "--no-auth"]) == 1
    message = capsys.readouterr().err
    assert "8000, 8501" in message
    assert "http://127.0.0.1:8501" in message
    assert "Ctrl+C" in message


def _process(lines: list[str], *, stay_alive: bool) -> subprocess.Popen:
    source = "import sys, time\n"
    source += "\n".join(f"print({line!r}, flush=True)" for line in lines)
    if stay_alive:
        source += "\ntime.sleep(10)\n"
    return subprocess.Popen([sys.executable, "-u", "-c", source], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def test_wait_tunnel_requires_registered_connection():
    process = _process(["https://example-words.trycloudflare.com"], stay_alive=True)
    try:
        with pytest.raises(LaunchError, match="не подключился"):
            _wait_tunnel(process, timeout=.3)
    finally:
        process.terminate()
        process.wait()


def test_wait_tunnel_accepts_connected_url():
    process = _process(["https://example-words.trycloudflare.com",
                        "2026-09-23 INF Registered tunnel connection"], stay_alive=True)
    try:
        assert _wait_tunnel(process, timeout=2) == "https://example-words.trycloudflare.com"
    finally:
        process.terminate()
        process.wait()


def test_wait_tunnel_reports_early_exit():
    process = _process(["2026-09-23 ERR error=network_unavailable"], stay_alive=False)
    with pytest.raises(LaunchError, match="завершился до подключения"):
        _wait_tunnel(process, timeout=2)
    process.wait()


def test_tunnel_errors_do_not_print_proxy_credentials():
    secret = "https://operator:private-password@proxy.example.test"
    generic = str(launcher._tunnel_failure("Туннель остановился.", f"error={secret}"))
    dns = str(launcher._tunnel_failure("Туннель остановился.",
                                      f"Could not lookup srv records via {secret}"))
    assert secret not in generic and secret not in dns
    assert "DNS не находит" in dns


def test_panel_turns_the_server_off_and_on_keeping_link_and_password(monkeypatch, tmp_path, capsys):
    api_python, web_python = tmp_path / "api-python", tmp_path / "web-python"
    api_python.touch()
    web_python.touch()
    monkeypatch.setattr(launcher, "API_PYTHON", api_python)
    monkeypatch.setattr(launcher, "WEB_PYTHON", web_python)
    monkeypatch.setattr(launcher, "_busy_local_ports", lambda *_: ())
    monkeypatch.setattr(launcher, "_wait_health", lambda *_, **__: None)
    monkeypatch.setattr(launcher, "_wait_web", lambda *_, **__: None)
    monkeypatch.setattr(launcher, "keep_awake", lambda _: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    panel = {}

    class Panel:
        url = "http://127.0.0.1:8502"

        def close(self):
            panel["closed"] = True

    monkeypatch.setattr(launcher, "start_admin_panel", lambda **options: panel.update(options) or Panel())
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
        process = Process()
        launched.append((command, kwargs, process))
        return process

    class EndRun(Exception):
        pass

    ticks = []

    def tick(_seconds):
        ticks.append(panel["launcher_state"]()["server"]["state"])
        if len(ticks) == 1:
            assert panel["control"]("stop") == (True, "Выключаем сервер…")
            assert panel["control"]("stop")[0] is False  # Команда уже ждёт исполнения.
        elif len(ticks) == 2:
            assert all(process.returncode == 0 for *_, process in launched)
            assert panel["control"]("stop") == (False, "Сервер уже выключен.")
            assert panel["control"]("password") == (False, "Сначала включите сервер.")
            assert panel["control"]("start") == (True, "Запускаем сервер…")
        elif len(ticks) == 3:
            assert panel["control"]("password")[0] is True
        else:
            raise EndRun

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.time, "sleep", tick)
    with pytest.raises(EndRun):
        launcher.main(["--local-only"])
    assert ticks == ["running", "stopped", "running", "running"]
    api_starts = [kwargs["env"] for command, kwargs, _ in launched if "app.web_api" in command]
    web_starts = [kwargs["env"] for command, kwargs, _ in launched if "streamlit" in command]
    assert len(api_starts) == 2 and len(web_starts) == 3
    # Пароль входа живёт весь запуск окна; внутренний токен API — свой у каждого включения.
    assert web_starts[0]["TREND_WEB_ACCESS_PASSWORD"] == web_starts[1]["TREND_WEB_ACCESS_PASSWORD"]
    assert api_starts[0]["TREND_API_TOKEN"] != api_starts[1]["TREND_API_TOKEN"]
    # Смена пароля перезапускает только сайт: модели и API остаются, токен к ним тот же.
    assert web_starts[2]["TREND_WEB_ACCESS_PASSWORD"] != web_starts[1]["TREND_WEB_ACCESS_PASSWORD"]
    assert web_starts[2]["TREND_API_TOKEN"] == api_starts[1]["TREND_API_TOKEN"]
    assert api_starts[0]["TREND_API_BLOCKLIST"] == str(launcher.BLOCKLIST_FILE)
    # Токен панели есть только у API, у сайта его нет.
    assert api_starts[0]["TREND_API_ADMIN_TOKEN"] == api_starts[1]["TREND_API_ADMIN_TOKEN"] == panel["admin_token"]
    assert "TREND_API_ADMIN_TOKEN" not in web_starts[0]
    assert panel["closed"] and all(process.returncode == 0 for *_, process in launched)
    assert "Сервер выключен из панели управления" in capsys.readouterr().out


def test_without_a_panel_a_crashed_service_still_ends_the_launch(monkeypatch, tmp_path, capsys):
    api_python, web_python = tmp_path / "api-python", tmp_path / "web-python"
    api_python.touch()
    web_python.touch()
    monkeypatch.setattr(launcher, "API_PYTHON", api_python)
    monkeypatch.setattr(launcher, "WEB_PYTHON", web_python)
    monkeypatch.setattr(launcher, "_busy_local_ports", lambda *_: ())
    monkeypatch.setattr(launcher, "_wait_health", lambda *_, **__: None)
    monkeypatch.setattr(launcher, "_wait_web", lambda *_, **__: None)
    monkeypatch.setattr(launcher, "keep_awake", lambda _: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *_: None)
    monkeypatch.setattr(launcher, "start_admin_panel", lambda **_: pytest.fail("--no-panel started a panel"))

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    processes = []

    def popen(command, **kwargs):
        processes.append(Process())
        assert "TREND_API_ADMIN_TOKEN" not in kwargs["env"]
        return processes[-1]

    def crash(_seconds):
        processes[0].returncode = 1

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.time, "sleep", crash)
    assert launcher.main(["--local-only", "--no-auth", "--no-panel"]) == 1
    assert "неожиданно остановился" in capsys.readouterr().err


def test_admin_panel_answers_only_its_own_host_and_obeys_only_its_own_page(tmp_path):
    from http.client import HTTPConnection
    import json

    from scripts.web_admin_panel import AdminPanel

    commands = []
    state = {"server": {"state": "stopped"}, "processes": []}
    panel = AdminPanel(port=0, api_port=1, admin_token="t" * 40, disk=tmp_path,
                       launcher_state=lambda: dict(state),
                       control=lambda action: (commands.append(action) or True, "ok"))
    panel.start()

    def request(method, path, headers=None, payload=None):
        connection = HTTPConnection("127.0.0.1", panel.port, timeout=5)
        body = json.dumps(payload).encode() if payload is not None else None
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    own = {"Host": f"127.0.0.1:{panel.port}"}
    command = {**own, "Content-Type": "application/json", "X-Panel": "1"}
    try:
        status, page, headers = request("GET", "/", own)
        nonce = headers["Content-Security-Policy"].split("'nonce-", 1)[1].split("'", 1)[0]
        assert status == 200 and f'<script nonce="{nonce}">'.encode() in page and b"{{nonce}}" not in page
        assert headers["X-Frame-Options"] == "DENY"
        # Подмена DNS: чужое имя хоста до панели не доходит.
        assert request("GET", "/", {"Host": "evil.example"})[0] == 403
        assert request("GET", "/overview?after=0", {"Host": f"attacker.test:{panel.port}"})[0] == 403
        status, body, _ = request("GET", "/overview?after=0", own)
        # Выключенный сервер не опрашивается: сводка API пуста без ошибки.
        assert status == 200 and json.loads(body)["api"] is None and json.loads(body)["api_error"] is None
        for headers in ({**own, "Content-Type": "application/json"},
                        {**command, "Origin": "https://evil.example"},
                        {**command, "Sec-Fetch-Site": "cross-site"},
                        {**command, "Content-Type": "text/plain"}):
            assert request("POST", "/server", headers, {"action": "start"})[0] == 403
        assert request("POST", "/server", command, {"action": "reboot"})[0] == 422
        for bad in ({"action": "block"}, {"action": "message", "person": "0123456789ab", "text": ""},
                    {"action": "drop_database"}, {"action": "unblock", "id": "../../x"}):
            assert request("POST", "/action", command, bad)[0] == 422
        # Верная команда уходит в API; здесь его нет — панель честно говорит об этом.
        status, body, _ = request("POST", "/action", command, {"action": "pause"})
        assert status == 502 and json.loads(body)["error"] == "API не отвечает."
        assert request("POST", "/server", {**command, "Origin": f"http://127.0.0.1:{panel.port}",
                                           "Sec-Fetch-Site": "same-origin"}, {"action": "start"})[0] == 202
        assert commands == ["start"]
    finally:
        panel.close()


def test_tunnel_log_lines_for_the_panel_hide_proxy_credentials():
    from scripts.web_admin_panel import mask_credentials

    assert (mask_credentials("proxy http://user:secret@127.0.0.1:7897 failed")
            == "proxy http://***@127.0.0.1:7897 failed")
