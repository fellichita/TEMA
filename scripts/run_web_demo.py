"""Start the loopback API and web UI, optionally with a public demo tunnel."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
from queue import Empty, Queue
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, TYPE_CHECKING
import webbrowser
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
if TYPE_CHECKING:
    from scripts.web_admin_panel import AdminPanel, mask_credentials
else:
    try:
        from scripts.web_admin_panel import AdminPanel, mask_credentials
    except ImportError:  # Файл запущен напрямую: пакета scripts на пути импорта нет.
        from web_admin_panel import AdminPanel, mask_credentials
# The same two environments under either layout: POSIX venvs keep their
# interpreter in bin/, Windows ones in Scripts/ with an extension.
API_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
WEB_PYTHON = ROOT / ".venv-web" / ("Scripts/python.exe" if WINDOWS else "bin/python")
# Homebrew installs outside the default PATH of a double-clicked launcher, so
# the known location is tried after the ordinary lookup rather than instead.
CLOUDFLARED_FALLBACKS = (Path("/opt/homebrew/bin/cloudflared"), Path("/usr/local/bin/cloudflared"),
                         # winget ставит сюда; окно, открытое до установки, PATH ещё не видит.
                         Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
                         / "cloudflared" / "cloudflared.exe",
                         Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "cloudflared" / "cloudflared.exe")
# Публичная ссылка по умолчанию работает в отдельном профиле: история, библиотека
# и настройки владельца посетителям не видны.
PUBLIC_PROFILE = ROOT / "storage" / "web-public-profile"
ACCESS_FILE = ROOT / "storage" / "web-public-access.txt"
# Заблокированные в панели браузеры и адреса: переживают перезапуски сервера.
BLOCKLIST_FILE = ROOT / "storage" / "web-blocklist.json"
# Ресурсы из панели: сколько анализов одновременно, пакет модели и прочее.
RESOURCES_FILE = ROOT / "storage" / "web-resources.json"
# fxTunnel — российский туннель: Cloudflare из России с июня 2025 года режется
# провайдерами (проходят первые ~16 КБ ответа), и сайт по trycloudflare.com не
# открывается. Установщик fxTunnel кладёт клиент сюда.
FXTUNNEL_FALLBACKS = (Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
                      / "fxTunnel" / "fxtunnel.exe",
                      Path.home() / ".local" / "bin" / "fxtunnel", Path("/usr/local/bin/fxtunnel"))
# Постоянный адрес https://trendanalizer.fxtun.ru; поддомен закрепляется за аккаунтом fxTunnel.
DEFAULT_FXTUNNEL_DOMAIN = "trendanalizer"
PUBLIC_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
FXTUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.fxtun\.(?:ru|dev)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MAX_WEB_MESSAGE_MB = 8
MAX_WEB_UPLOAD_MB = 1
# Панель управления владельца: только этот компьютер, туннель ведёт на порт сайта.
DEFAULT_PANEL_PORT = 8502
TUNNEL_LOG_LINES = 200
_LOCAL_OPENER = build_opener(ProxyHandler({}))
_CHILD_ENVIRONMENT_KEYS = frozenset({
    "PATH", "PATHEXT", "HOME", "USER", "LOGNAME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SystemRoot", "WINDIR",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "XDG_DATA_HOME", "SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
})


def _minimal_environment(*, proxy: bool = False) -> dict[str, str]:
    """Keep unrelated credentials out of the UI and public tunnel processes."""
    allowed = set(_CHILD_ENVIRONMENT_KEYS)
    if proxy:
        allowed.update({"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"})
    # На Windows os.environ отдаёт имена заглавными («SYSTEMROOT»). Без
    # SystemRoot дочерний Python не поднимает сокеты (WinError 10106).
    fold = str.upper if WINDOWS else str
    allowed = {fold(name) for name in allowed}
    return {name: value for name, value in os.environ.items()
            if fold(name) in allowed or name.startswith("LC_")}


def _child_environments(password: str | None, api_token: str, *, api_port: int = 8000) -> tuple[dict[str, str], dict[str, str]]:
    """Keep the browser password out of the model/API process."""
    api_environment = os.environ.copy()
    api_environment["TREND_API_TOKEN"] = api_token
    api_environment.pop("TREND_WEB_ACCESS_PASSWORD", None)
    api_environment.pop("TREND_WEB_REQUIRE_AUTH", None)
    # Токен панели владельца выдаёт этот запуск, если панель включена.
    api_environment.pop("TREND_API_ADMIN_TOKEN", None)
    # Разделение посетителей решает этот запуск, а не унаследованное окружение.
    api_environment.pop("TREND_API_SEPARATE_VISITORS", None)
    web_environment = _minimal_environment()
    web_environment.update(API_URL=f"http://127.0.0.1:{api_port}", TREND_API_TOKEN=api_token,
                           STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    web_environment["TREND_WEB_REQUIRE_AUTH"] = "0" if password is None else "1"
    web_environment["STREAMLIT_SERVER_COOKIE_SECRET"] = secrets.token_urlsafe(32)
    if password is None:
        web_environment.pop("TREND_WEB_ACCESS_PASSWORD", None)
    else:
        web_environment["TREND_WEB_ACCESS_PASSWORD"] = password
    return api_environment, web_environment


def _web_command(*, public: bool = False, port: int = 8501,
                 allowed_hosts: str = "*.trycloudflare.com") -> list[str]:
    """Limit untrusted WebSocket and upload payloads before Streamlit parses them."""
    command = [str(WEB_PYTHON), "-E", "-s", "-m", "streamlit", "run", "app/ui/web.py",
            "--server.address", "127.0.0.1", "--server.port", str(port),
            "--server.headless", "true", "--server.fileWatcherType", "none",
            "--server.maxMessageSize", str(MAX_WEB_MESSAGE_MB),
            "--server.maxUploadSize", str(MAX_WEB_UPLOAD_MB),
            "--server.enableCORS", "true", "--server.enableXsrfProtection", "true",
            "--server.enableStaticServing", "false"]
    # Streamlit otherwise accepts arbitrary Host headers on its WebSocket,
    # allowing DNS rebinding to the loopback UI and its private API client.
    command.extend(("--server.allowedHosts", "localhost", "--server.allowedHosts", "127.0.0.1"))
    if public:
        command.extend(("--server.allowedHosts", allowed_hosts))
    return command


def _tunnel_environment() -> dict[str, str]:
    """The network tunnel needs no application password or API capability."""
    environment = _minimal_environment(proxy=True)
    # Токен fxTunnel обычно в системном хранилище (fxtunnel login); на машине без
    # него — в переменной окружения. Цвета в журнале туннеля не нужны.
    if os.environ.get("FXTUNNEL_TOKEN"):
        environment["FXTUNNEL_TOKEN"] = os.environ["FXTUNNEL_TOKEN"]
    environment["NO_COLOR"] = "1"
    return environment


@dataclass(frozen=True)
class Tunnel:
    """How to start one public tunnel and read its address from its log."""
    name: str
    command: tuple[str, ...]
    url: re.Pattern[str]
    # Строка журнала о готовом подключении; None — готовность означает сам адрес.
    connected: str | None
    allowed_hosts: str
    permanent: bool = False


def cloudflare_tunnel(binary: Path, port: int) -> Tunnel:
    # HTTP/2 идёт по TCP: QUIC (UDP) часто не проходит через прокси, VPN и
    # корпоративные сети — 27.09.2026 туннель за прокси так и не подключился.
    return Tunnel("Cloudflare Tunnel",
                  (str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}",
                   "--no-autoupdate", "--loglevel", "info", "--protocol", "http2"),
                  PUBLIC_URL, "Registered tunnel connection", "*.trycloudflare.com")


def fxtunnel_tunnel(binary: Path, port: int, domain: str | None) -> Tunnel:
    # --no-inspect: локальный инспектор fxTunnel записывал бы все запросы
    # посетителей, включая ввод пароля.
    command = [str(binary), "http", str(port), "--no-inspect", "--log-level", "info"]
    if domain:
        command.extend(("--domain", domain))
    return Tunnel("fxTunnel", tuple(command), FXTUNNEL_URL, None, "*.fxtun.ru", permanent=bool(domain))


def _access_password(*, local_only: bool, no_auth: bool) -> str | None:
    if no_auth:
        return None
    # Public quick tunnels have a new URL each launch. A new random secret also
    # avoids accidentally exposing a reused or weak operator-supplied password.
    if not local_only:
        return secrets.token_urlsafe(32)
    return os.environ.get("TREND_WEB_ACCESS_PASSWORD") or secrets.token_urlsafe(15)


def cloudflared() -> Path | None:
    found = shutil.which("cloudflared")
    if found is not None:
        return Path(found)
    return next((path for path in CLOUDFLARED_FALLBACKS if path.is_file()), None)


def fxtunnel() -> Path | None:
    found = shutil.which("fxtunnel")
    if found is not None:
        return Path(found)
    return next((path for path in FXTUNNEL_FALLBACKS if path.is_file()), None)


def keep_awake(processes: list[subprocess.Popen]) -> None:
    """Hold sleep off for as long as this launcher lives, where the OS allows it.

    An analysis runs for minutes without input, and a machine that suspends in
    the middle of one drops the web analysis. macOS is asked through caffeinate;
    Windows through its own execution-state flag, which needs no process. Any
    other system keeps its own power policy, which is not an error.
    """
    if sys.platform == "win32":
        import ctypes

        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: the system stays up until this
        # process exits or clears the flag; the display may still switch off.
        with suppress(AttributeError, OSError):
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        return
    if sys.platform == "darwin" and Path("/usr/bin/caffeinate").is_file():
        processes.append(subprocess.Popen(["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())]))


class LaunchError(RuntimeError):
    pass


def _busy_local_ports(*ports: int) -> tuple[int, ...]:
    """Find listeners before starting a second copy of the local services."""
    busy = []
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.3):
                busy.append(port)
        except OSError:
            continue
    return tuple(busy)


def _tunnel_failure(message: str, last_error: str | None) -> LaunchError:
    if last_error and "invalid token" in last_error:
        return LaunchError(f"{message} fxTunnel не принял токен. Войдите один раз: fxtunnel login "
                           "(токен — в личном кабинете fxtun.ru) и запустите демо снова.")
    if last_error and ("domain" in last_error or "subdomain" in last_error):
        return LaunchError(f"{message} Поддомен занят или не закреплён за вашим аккаунтом. "
                           "Закрепите его: fxtunnel domains add <имя> — или запустите с другим --domain.")
    if last_error and "Could not lookup srv records" in last_error:
        return LaunchError(f"{message} DNS не находит серверы Cloudflare Tunnel. "
                           "Проверьте DNS, VPN или прокси и запустите демо снова.")
    # The tunnel inherits proxy settings. A connection error may include the
    # proxy URL with credentials, so never echo a child log line to the user.
    return LaunchError(message + (" Проверьте подключение туннеля." if last_error else ""))


def _wait_health(process: subprocess.Popen, expected_instance_id: str, *, port: int = 8000,
                 timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaunchError(f"Внутренний сервис не запустился. Проверьте сообщение выше и свободен ли порт {port}.")
        try:
            with _LOCAL_OPENER.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if (response.status == 200
                        and response.headers.get("X-Trend-Instance") == expected_instance_id
                        and process.poll() is None):
                    return
        except OSError:
            pass
        time.sleep(.25)
    raise LaunchError("Внутренний сервис не ответил за 30 секунд.")


def _wait_web(process: subprocess.Popen, *, port: int = 8501, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaunchError(f"Веб-интерфейс не запустился. Проверьте, свободен ли порт {port}.")
        try:
            with _LOCAL_OPENER.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return
        except OSError:
            pass
        time.sleep(.25)
    raise LaunchError("Веб-интерфейс не ответил за 30 секунд.")


def _wait_tunnel(process: subprocess.Popen, timeout: float = 45, tunnel: Tunnel | None = None,
                 log: deque[str] | None = None) -> str:
    """Адрес туннеля из его журнала; журнал и дальше копится в `log` для панели."""
    events: Queue[tuple[str, str]] = Queue()
    pattern = tunnel.url if tunnel is not None else PUBLIC_URL
    marker = tunnel.connected if tunnel is not None else "Registered tunnel connection"
    name = tunnel.name if tunnel is not None else "Cloudflare Tunnel"

    def read() -> None:
        if process.stdout is None:
            return
        for raw in process.stdout:
            line = ANSI.sub("", raw)
            if log is not None and line.strip():
                log.append(mask_credentials(line.rstrip())[:500])
            match = pattern.search(line)
            if match:
                events.put(("url", match.group(0)))
                if marker is None:
                    events.put(("connected", ""))
            if marker is not None and marker in line:
                events.put(("connected", ""))
            if " ERR " in line or " error=" in line or "Failed" in line:
                events.put(("error", line.strip()))

    thread = threading.Thread(target=read, name="tunnel-output", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    url = None
    connected = False
    last_error = None
    while time.monotonic() < deadline:
        try:
            kind, value = events.get(timeout=min(.1, max(0, deadline - time.monotonic())))
        except Empty:
            if process.poll() is not None:
                thread.join(timeout=.2)
                while not events.empty():
                    pending_kind, pending_value = events.get_nowait()
                    if pending_kind == "error":
                        last_error = pending_value
                raise _tunnel_failure(f"{name} завершился до подключения.", last_error) from None
            continue
        if kind == "url":
            url = value
        elif kind == "connected":
            connected = True
        elif kind == "error":
            last_error = value
        if url and connected:
            if process.poll() is None:
                return url
            raise _tunnel_failure(f"{name} завершился до подключения.", last_error)
    raise _tunnel_failure(f"{name} не подключился за {timeout:g} секунд.", last_error)


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 8
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()


def prepare_public_profile(profile: Path) -> None:
    """Profile for visitors: local AI selected, models copied from the owner's profile once."""
    profile.mkdir(parents=True, exist_ok=True)
    # Скрипт настройки берётся в дочернем процессе из корня проекта: этот файл
    # можно запустить и напрямую, без пакета scripts на пути импорта.
    # -X utf8: с -E дочерний Python иначе пишет в канал кодировкой консоли (cp1251).
    setup = subprocess.run([str(API_PYTHON), "-E", "-s", "-X", "utf8", "-c",
                            "from scripts.local_web_bootstrap import PROFILE_SETUP; exec(PROFILE_SETUP)",
                            str(profile), str(ROOT)],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=90)
    if setup.returncode:
        detail = (setup.stderr.strip().splitlines() or ["Причина не указана."])[-1]
        raise LaunchError("Не удалось подготовить профиль посетителей: " + detail[:500])
    print("Проверяем модели профиля посетителей; недостающие копируются из основного профиля…", flush=True)
    if subprocess.call([str(API_PYTHON), "-E", "-s", "-m", "scripts.setup_models", "--web-analysis",
                        "--profile", str(profile)], cwd=ROOT):
        raise LaunchError("Модели профиля посетителей не готовы. Проверьте сообщение выше и повторите запуск.")


def start_admin_panel(**options: Any) -> AdminPanel | None:
    """Панель не обязательна для сайта: занятый порт — предупреждение, не отказ."""
    panel = AdminPanel(**options)
    try:
        panel.start()
    except OSError:
        print(f"Панель управления не запущена: порт {panel.port} занят. "
              "Запустите с другим --panel-port.", file=sys.stderr, flush=True)
        return None
    return panel


def _process_state(name: str, process: subprocess.Popen | None, port: int | None = None) -> dict[str, Any]:
    alive = process is not None and process.poll() is None
    return {"name": name, "alive": alive, "started": process is not None,
            "pid": getattr(process, "pid", None) if alive else None,
            "exit_code": process.poll() if process is not None and not alive else None, "port": port}


class WebStack:
    """Процессы сайта одного окна запуска: API с моделями, Streamlit и туннель.

    Панель управления выключает и снова включает их, не закрывая окно: ссылка
    и пароль этого запуска сохраняются, меняются только внутренние токены. Всё
    запускает и останавливает главный поток лаунчера; панель лишь оставляет
    команду и читает состояние.
    """

    def __init__(self, *, args: argparse.Namespace, password: str | None, tunnel: Tunnel | None,
                 data_dir: Path | None, admin_token: str | None):
        self.args = args
        self.password = password
        self.tunnel = tunnel
        self.data_dir = data_dir
        self.admin_token = admin_token
        self.launched_at = time.time()
        self.state = "stopped"  # starting · running · stopping · stopped · failed
        self.step: str | None = None
        self.error: str | None = None
        self.url: str | None = None
        self.started_at: float | None = None
        self.starts = 0
        self.processes: list[subprocess.Popen] = []
        self.children: dict[str, subprocess.Popen] = {}
        self.tunnel_log: deque[str] = deque(maxlen=TUNNEL_LOG_LINES)
        self.access_written = False
        self._profile_ready = False
        self._api_token: str | None = None
        self._command: str | None = None
        self._lock = threading.Lock()

    def _set(self, **fields: Any) -> None:
        with self._lock:
            for name, value in fields.items():
                setattr(self, name, value)

    def request(self, action: str) -> tuple[bool, str]:
        """Команда из панели: принять к исполнению или объяснить отказ."""
        with self._lock:
            if self._command is not None or self.state in {"starting", "stopping"}:
                return False, "Сервер уже запускается или останавливается. Подождите."
            if action == "start" and self.state in {"stopped", "failed"}:
                self._command = "start"
                return True, "Запускаем сервер…"
            if action == "stop" and self.state == "running":
                self._command = "stop"
                return True, "Выключаем сервер…"
            if action == "password":
                if self.password is None:
                    return False, "Сайт запущен без пароля — менять нечего."
                if self.state != "running":
                    return False, "Сначала включите сервер."
                self._command = "password"
                return True, "Меняем пароль: сайт перезапустится, все войдут заново…"
            return False, "Сервер уже включён." if action == "start" else "Сервер уже выключен."

    def take_command(self) -> str | None:
        with self._lock:
            command, self._command = self._command, None
            return command

    def _spawn(self, name: str, command: list[str], **options: Any) -> subprocess.Popen:
        process = subprocess.Popen(command, cwd=ROOT, **options)
        self.processes.append(process)
        with self._lock:
            self.children[name] = process
        return process

    def start(self) -> str:
        """Поднять API, сайт и туннель; вернуть адрес сайта."""
        args = self.args
        self._set(state="starting", step="Проверяем порты", error=None)
        busy = _busy_local_ports(args.api_port, args.web_port)
        if busy:
            ports = ", ".join(map(str, busy))
            hint = (f" Если это прежняя копия локального веба, откройте http://127.0.0.1:{args.web_port} "
                    "или остановите её в старом окне запуска (Ctrl+C), затем запустите файл снова "
                    "для обновления программы.") if args.local_only else ""
            raise LaunchError(f"Не удалось запустить новую копию: локальные порты {ports} уже заняты.{hint}")
        if not args.local_only:
            # Прежний запуск могли закрыть, не дав ему убрать файл: старая ссылка уже не работает.
            ACCESS_FILE.unlink(missing_ok=True)
        if self.data_dir == PUBLIC_PROFILE and not self._profile_ready:
            self._set(step="Готовим профиль посетителей")
            prepare_public_profile(self.data_dir)
            self._profile_ready = True
        self._api_token = secrets.token_urlsafe(32)
        api_environment, web_environment = _child_environments(self.password, self._api_token,
                                                               api_port=args.api_port)
        api_environment["TREND_API_BLOCKLIST"] = str(BLOCKLIST_FILE)
        api_environment["TREND_API_RESOURCES"] = str(RESOURCES_FILE)
        instance_id = secrets.token_urlsafe(24)
        api_environment["TREND_API_INSTANCE_ID"] = instance_id
        if self.password is not None:
            # С паролем каждый браузер видит только свои анализы; API проверяет это сам.
            api_environment["TREND_API_SEPARATE_VISITORS"] = "1"
        if self.admin_token is not None:
            api_environment["TREND_API_ADMIN_TOKEN"] = self.admin_token
        api_command = [str(API_PYTHON), "-E", "-s", "-m", "app.web_api", "--port", str(args.api_port)]
        if self.data_dir is not None:
            api_command.extend(("--data-dir", str(self.data_dir)))
        self._set(step="Загружаем модели и API")
        api = self._spawn("api", api_command, env=api_environment, stdout=subprocess.DEVNULL)
        _wait_health(api, instance_id, port=args.api_port)
        self._set(step="Запускаем сайт")
        self._start_web(web_environment)
        keep_awake(self.processes)
        if self.tunnel is None:
            url = f"http://127.0.0.1:{args.web_port}"
        else:
            self._set(step="Подключаем публичную ссылку")
            # Журнал туннеля в UTF-8 (в нём бывают имена сетевых адаптеров).
            tunnel_process = self._spawn("tunnel", list(self.tunnel.command), env=_tunnel_environment(),
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1)
            url = _wait_tunnel(tunnel_process, tunnel=self.tunnel, log=self.tunnel_log)
        if not args.local_only and self.password is not None:
            # Владелец копирует ссылку и пароль отсюда; файл живёт, пока работает ссылка.
            ACCESS_FILE.write_text(f"Ссылка: {url}\nПароль: {self.password}\n", encoding="utf-8")
            self.access_written = True
        self._set(state="running", step=None, url=url, started_at=time.time(), starts=self.starts + 1)
        return url

    def _start_web(self, web_environment: dict[str, str]) -> None:
        args = self.args
        web = self._spawn("web", _web_command(public=not args.local_only, port=args.web_port,
                                              allowed_hosts=self.tunnel.allowed_hosts if self.tunnel else ""),
                          env=web_environment, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _wait_web(web, port=args.web_port)

    def change_password(self) -> str:
        """Новый пароль входа: перезапускается только сайт, модели остаются в памяти.

        Прежние cookie подписаны старым паролем и больше не пускают — все
        посетители, в том числе выгнанные, входят заново.
        """
        self._set(state="starting", step="Меняем пароль: перезапускаем сайт")
        web = self.children.get("web")
        if web is not None:
            _stop([web])
            with suppress(ValueError):
                self.processes.remove(web)
        password = secrets.token_urlsafe(32 if not self.args.local_only else 15)
        _, web_environment = _child_environments(password, self._api_token or secrets.token_urlsafe(32),
                                                 api_port=self.args.api_port)
        self._set(password=password)
        self._start_web(web_environment)
        if self.access_written and self.url:
            ACCESS_FILE.write_text(f"Ссылка: {self.url}\nПароль: {password}\n", encoding="utf-8")
        self._set(state="running", step=None)
        return password

    def stop(self, *, failure: str | None = None) -> None:
        """Остановить все процессы; ссылка перестаёт открывать сайт."""
        self._set(state="stopping", step="Останавливаем процессы")
        _stop(self.processes)
        self.processes.clear()
        if self.access_written:
            ACCESS_FILE.unlink(missing_ok=True)
            self.access_written = False
        self._set(state="failed" if failure else "stopped", step=None, error=failure, started_at=None)

    def problem(self) -> str | None:
        """Что упало у работающего сервера, если упало."""
        if self.state != "running":
            return None
        tunnel_process = self.children.get("tunnel")
        if self.tunnel is not None and tunnel_process is not None and tunnel_process.poll() is not None:
            return f"{self.tunnel.name} отключился, и ссылка больше не открывает сайт."
        if any(self.children[name].poll() is not None for name in ("api", "web") if name in self.children):
            return "Один из локальных сервисов неожиданно остановился."
        return None

    def snapshot(self) -> dict[str, Any]:
        """Что знает только лаунчер — для панели управления."""
        with self._lock:
            children = dict(self.children)
            state: dict[str, Any] = {
                "launched_at": datetime.fromtimestamp(self.launched_at, UTC).isoformat(),
                "server": {"state": self.state, "step": self.step, "error": self.error, "starts": self.starts,
                           "started_at": datetime.fromtimestamp(self.started_at, UTC).isoformat()
                           if self.started_at is not None else None},
                "mode": ("public" if not self.args.local_only
                         else "local" if self.password is not None else "local-open"),
                "url": self.url, "local_url": f"http://127.0.0.1:{self.args.web_port}",
                "password": self.password, "data_dir": str(self.data_dir) if self.data_dir is not None else None,
                "tunnel_log": list(self.tunnel_log)[-80:], "tunnel": None}
        running = state["server"]["state"] == "running"
        state["processes"] = [
            _process_state("Анализ и модели (API)", children.get("api") if running else None, self.args.api_port),
            _process_state("Сайт (Streamlit)", children.get("web") if running else None, self.args.web_port)]
        if self.tunnel is not None:
            tunnel_process = children.get("tunnel") if running else None
            state["processes"].append(_process_state(self.tunnel.name, tunnel_process))
            state["tunnel"] = {"name": self.tunnel.name, "permanent": self.tunnel.permanent,
                               "state": "connected" if tunnel_process is not None and tunnel_process.poll() is None
                               else "off"}
        return state


def _announce(stack: WebStack, url: str, panel: AdminPanel | None) -> None:
    args, password, tunnel = stack.args, stack.password, stack.tunnel
    title = "Trendanalyser локальная веб-версия готова" if args.no_auth else "Trendanalyser Web Demo готов"
    print(f"\n{title}", flush=True)
    print(f"Откройте в браузере:  {url}", flush=True)
    if stack.data_dir == PUBLIC_PROFILE:
        print(f"Профиль посетителей: {stack.data_dir}. Ваша история и настройки им не видны; "
              "каждый вошедший видит только свои анализы. Вычисления идут на этом компьютере.",
              flush=True)
    if password is not None:
        print(f"Пароль:  {password}", flush=True)
    if stack.access_written:
        print(f"Ссылка и пароль записаны в {ACCESS_FILE}; файл удалится при остановке.", flush=True)
    if panel is not None:
        print(f"Панель управления (только на этом компьютере):  {panel.url}", flush=True)
    if tunnel is not None and tunnel.permanent:
        print("\nАдрес постоянный и сохранится при следующих запусках; пароль меняется "
              "при каждом запуске. Сайт работает, пока открыто это окно.", flush=True)
    elif not args.local_only:
        print("\nПубличная ссылка временная: она действует только пока открыто это окно. "
              "При следующем запуске используйте новую ссылку.", flush=True)
    if password is not None:
        print("Не публикуйте пароль рядом со ссылкой.", flush=True)
    print("Для остановки нажмите Ctrl+C или закройте окно.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="Не открывать публичный туннель")
    parser.add_argument("--no-auth", action="store_true", help="Без пароля, только для локального адреса")
    parser.add_argument("--open-browser", action="store_true", help="Открыть готовый сайт в браузере")
    parser.add_argument("--data-dir", type=Path, help="Отдельный каталог данных для API")
    parser.add_argument("--tunnel", choices=("fxtunnel", "cloudflare"), default="fxtunnel",
                        help="Сервис публичной ссылки: fxtunnel (серверы в России) или cloudflare")
    parser.add_argument("--domain", default=DEFAULT_FXTUNNEL_DOMAIN,
                        help="Постоянный поддомен fxTunnel; пустая строка — случайный адрес")
    parser.add_argument("--api-port", type=int, default=8000, help="Локальный порт API (по умолчанию 8000)")
    parser.add_argument("--web-port", type=int, default=8501, help="Локальный порт сайта (по умолчанию 8501)")
    parser.add_argument("--panel-port", type=int, default=DEFAULT_PANEL_PORT,
                        help="Порт панели управления на этом компьютере (по умолчанию 8502)")
    parser.add_argument("--no-panel", action="store_true", help="Не запускать панель управления")
    parser.add_argument("--open-panel", action="store_true", help="Открыть панель управления в браузере")
    args = parser.parse_args(argv)
    if args.no_auth and not args.local_only:
        parser.error("--no-auth допускается только вместе с --local-only")
    ports = (args.api_port, args.web_port) + (() if args.no_panel else (args.panel_port,))
    if not all(1024 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        parser.error("Укажите разные свободные порты от 1024 до 65535.")
    for path, message in ((API_PYTHON, "Не готово основное окружение .venv."),
                          (WEB_PYTHON, "Не готово веб-окружение .venv-web.")):
        if not path.is_file():
            parser.error(message)
    tunnel_binary = None if args.local_only else fxtunnel() if args.tunnel == "fxtunnel" else cloudflared()
    if not args.local_only and tunnel_binary is None:
        install = ("fxTunnel: irm https://fxtun.ru/install.ps1 | iex (Windows) — затем fxtunnel login"
                   if args.tunnel == "fxtunnel" else "cloudflared: winget install --id Cloudflare.cloudflared")
        parser.error(f"Не найден клиент туннеля. Установите {install} и повторите, "
                     "либо запустите с --local-only без публичной ссылки.")
    if args.domain and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", args.domain) is None:
        parser.error("--domain: латинские строчные буквы, цифры и дефис, до 63 символов.")
    tunnel = (None if tunnel_binary is None else
              fxtunnel_tunnel(tunnel_binary, args.web_port, args.domain or None) if args.tunnel == "fxtunnel"
              else cloudflare_tunnel(tunnel_binary, args.web_port))

    password = _access_password(local_only=args.local_only, no_auth=args.no_auth)
    if password is not None and not 16 <= len(password) <= 128:
        parser.error("TREND_WEB_ACCESS_PASSWORD должен содержать от 16 до 128 символов.")
    data_dir = args.data_dir if args.data_dir is not None or args.local_only else PUBLIC_PROFILE
    stack = WebStack(args=args, password=password, tunnel=tunnel, data_dir=data_dir,
                     admin_token=None if args.no_panel else secrets.token_urlsafe(32))
    panel: AdminPanel | None = None

    def shutdown(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, shutdown)
    try:
        if stack.admin_token is not None:
            panel = start_admin_panel(port=args.panel_port, api_port=args.api_port, admin_token=stack.admin_token,
                                      disk=data_dir if data_dir is not None else ROOT / "storage",
                                      launcher_state=stack.snapshot, control=stack.request)
        url = stack.start()
        _announce(stack, url, panel)
        if args.open_browser:
            webbrowser.open(url)
        if args.open_panel and panel is not None:
            webbrowser.open(panel.url)
        while True:
            command = stack.take_command()
            if command == "stop":
                stack.stop()
                print("\nСервер выключен из панели управления; окно и панель остаются открытыми. "
                      "Включить снова можно там же.", flush=True)
            elif command == "password":
                try:
                    password = stack.change_password()
                except LaunchError as error:
                    stack.stop(failure=str(error))
                    print(f"Сайт не перезапустился после смены пароля: {error}", file=sys.stderr, flush=True)
                else:
                    print(f"\nПароль сменён из панели управления. Новый пароль:  {password}", flush=True)
            elif command == "start":
                print("\nВключаем сервер из панели управления…", flush=True)
                try:
                    url = stack.start()
                except LaunchError as error:
                    stack.stop(failure=str(error))
                    print(f"Сервер не включился: {error}", file=sys.stderr, flush=True)
                else:
                    _announce(stack, url, panel)
            problem = stack.problem()
            if problem is not None:
                # Без панели включить сервер обратно негде — окно завершается, как раньше.
                if panel is None:
                    raise LaunchError(problem + " Перезапустите веб-демо.")
                stack.stop(failure=problem)
                print(f"\n{problem} Включите сервер снова в панели управления: {panel.url}",
                      file=sys.stderr, flush=True)
            time.sleep(1)
    except LaunchError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1
    finally:
        if panel is not None:
            panel.close()
        stack.stop()


if __name__ == "__main__":
    raise SystemExit(main())
