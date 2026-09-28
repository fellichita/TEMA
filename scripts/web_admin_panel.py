"""Панель управления владельца: живая сводка веб-сервиса на 127.0.0.1.

Панель поднимает лаунчер веба (scripts/run_web_demo.py). Туннель к ней не
ведёт: она слушает только петлевой адрес, отвечает только на свой Host и берёт
сводку у API по отдельному токену, которого нет у процесса сайта. Работает на
стандартной библиотеке — лаунчер запускается без веб-окружения.
"""

from __future__ import annotations

import ctypes
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from threading import Lock, Thread
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "ui" / "admin_panel.html"
MAX_BODY_BYTES = 4_096
API_TIMEOUT_SECONDS = 8
GPU_CACHE_SECONDS = 2.0
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
PERSON = re.compile(r"[0-9a-f]{12}\Z")
BLOCK_ID = re.compile(r"[A-Za-z0-9_-]{8,32}\Z")
MAX_MESSAGE_CHARACTERS = 500
# Действия владельца над посетителями и приёмом анализов (исполняет API).
VISITOR_ACTIONS = frozenset({"block", "unblock", "sign_out", "message", "pause", "resume", "resources",
                             "unload_llm"})
# Вкладки «Источники и языки», «Анализы» и «Обучение»: значения строго проверяет API.
OWNER_ACTIONS = {"sources": {"values"}, "languages": {"values"}, "learning": {"values"}, "feedback": {"values"},
                 "retrain": {"values"}, "reset_learning": {"values"}}
# Разделы панели, которые она читает у API, и допустимые параметры запроса.
VIEWS = {"/sources": set(), "/languages": set(), "/learning": set(),
         "/analyses": {"q", "state", "mode", "from", "to", "visitor", "with_signals", "with_technologies",
                       "country", "source", "sort", "order", "offset", "limit"},
         "/compare": {"ids"}, "/publications": {"run_id", "decision", "offset", "limit"}}
MAX_VIEW_QUERY = 1024
# Ресурсы сервиса и их пределы — те же, что проверяет API.
RESOURCE_LIMITS = {"slots": (1, 4), "llm_batch": (1, 8), "queue_limit": (0, 30)}
RESOURCE_FLAGS = frozenset({"keep_llm_loaded", "deep_allowed", "radar_exact"})
SERVER_ACTIONS = frozenset({"start", "stop", "password"})
# Строки журнала туннеля могут нести адрес прокси с логином и паролем.
CREDENTIALS = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@")
_OPENER = build_opener(ProxyHandler({}))
WINDOWS = sys.platform == "win32"


def _kernel32() -> Any:
    # `windll` есть в заглушках типов только под Windows: проверку платформы
    # mypy понимает сам, и файл проходит типы на всех трёх системах.
    if sys.platform != "win32":
        raise OSError("kernel32 есть только в Windows")
    return ctypes.windll.kernel32


def mask_credentials(line: str) -> str:
    return CREDENTIALS.sub(r"\1***@", line)


def iso(moment: float | None) -> str | None:
    return None if moment is None else datetime.fromtimestamp(moment, UTC).isoformat()


class SystemProbe:
    """Процессор, память, диск и видеокарта этого компьютера без сторонних пакетов."""

    def __init__(self, disk: Path):
        self._disk = disk
        self._lock = Lock()
        self._cpu_last: tuple[int, int] | None = None
        self._gpu: tuple[float, list[dict[str, Any]] | None] = (0.0, None)
        self._nvidia = shutil.which("nvidia-smi")

    def _cpu_times(self) -> tuple[int, int] | None:
        """(занято, всего) в тиках системы с её запуска."""
        if WINDOWS:
            class FileTime(ctypes.Structure):
                _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

            idle, kernel, user = FileTime(), FileTime(), FileTime()
            if not _kernel32().GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            value = lambda moment: moment.high << 32 | moment.low  # noqa: E731
            # Время ядра включает простой.
            total = value(kernel) + value(user)
            return total - value(idle), total
        try:
            fields = [int(part) for part in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        except (OSError, ValueError):
            return None
        total = sum(fields)
        return total - fields[3] - (fields[4] if len(fields) > 4 else 0), total

    def cpu_percent(self) -> float | None:
        with self._lock:
            times = self._cpu_times()
            previous, self._cpu_last = self._cpu_last, times
        if times is None or previous is None or times[1] <= previous[1]:
            return None
        return round(100 * (times[0] - previous[0]) / (times[1] - previous[1]), 1)

    @staticmethod
    def memory() -> dict[str, int] | None:
        if WINDOWS:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                            ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                            ("extended", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return {"total": status.total, "used": status.total - status.available}
        try:
            values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
            total = int(values["MemTotal"].split()[0]) * 1024
            available = int(values["MemAvailable"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            return None
        return {"total": total, "used": total - available}

    @staticmethod
    def _descendants(pid: int) -> list[int]:
        """Процесс и все его потомки: python.exe из venv на Windows — лишь переходник,
        настоящий интерпретатор (и рабочие процессы API) — его дети."""

        class Entry(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("usage", ctypes.c_ulong), ("pid", ctypes.c_ulong),
                        ("heap", ctypes.c_size_t), ("module", ctypes.c_ulong), ("threads", ctypes.c_ulong),
                        ("parent", ctypes.c_ulong), ("priority", ctypes.c_long), ("flags", ctypes.c_ulong),
                        ("name", ctypes.c_wchar * 260)]

        kernel32 = _kernel32()
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.Process32FirstW.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        kernel32.Process32NextW.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            return [pid]
        children: dict[int, list[int]] = {}
        try:
            entry = Entry()
            entry.size = ctypes.sizeof(Entry)
            found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while found:
                if entry.pid != entry.parent:
                    children.setdefault(entry.parent, []).append(entry.pid)
                found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        tree: list[int] = []
        queue = [pid]
        while queue and len(tree) < 64:
            current = queue.pop()
            tree.append(current)
            queue.extend(child for child in children.get(current, ()) if child not in tree)
        return tree

    @classmethod
    def process_memory(cls, pid: int | None) -> int | None:
        """Рабочий набор процесса вместе с потомками, в байтах (только Windows:
        без psutil больше неоткуда)."""
        if pid is None or not WINDOWS:
            return None
        sizes = [cls._working_set(member) for member in cls._descendants(pid)]
        return sum(size for size in sizes if size) or None

    @staticmethod
    def _working_set(pid: int) -> int | None:
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                        ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                        ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]

        kernel32 = _kernel32()
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.K32GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.working_set)
        finally:
            kernel32.CloseHandle(handle)

    def gpu(self) -> list[dict[str, Any]] | None:
        """Загрузка, видеопамять и температура каждой NVIDIA-карты; кэш на пару секунд."""
        if self._nvidia is None:
            return None
        with self._lock:
            moment, cached = self._gpu
            if time.monotonic() - moment < GPU_CACHE_SECONDS:
                return cached
        try:
            completed = subprocess.run(
                [self._nvidia, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=3, creationflags=0x08000000 if WINDOWS else 0)  # CREATE_NO_WINDOW
            cards = []
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 6:
                    continue
                number = lambda text: float(text) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text) else None  # noqa: E731
                cards.append({"name": parts[0][:80], "utilization": number(parts[1]),
                              "memory_used_mb": number(parts[2]), "memory_total_mb": number(parts[3]),
                              "temperature": number(parts[4]), "power_w": number(parts[5])})
            result = cards if completed.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            result = None
        with self._lock:
            self._gpu = (time.monotonic(), result)
        return result

    def snapshot(self) -> dict[str, Any]:
        try:
            usage = shutil.disk_usage(self._disk)
            disk = {"path": str(self._disk), "total": usage.total, "free": usage.free}
        except OSError:
            disk = None
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        return {"cpu_percent": self.cpu_percent(), "cpu_count": os.cpu_count(), "load": load,
                "memory": self.memory(), "disk": disk, "gpu": self.gpu()}


class AdminPanel:
    """Сервер панели: страница, сводка раз в секунду-полторы и команды владельца.

    Панель живёт дольше сайта: выключенный сервер включается отсюда же.
    """

    def __init__(self, *, port: int, api_port: int, admin_token: str, disk: Path,
                 launcher_state: Callable[[], dict[str, Any]], control: Callable[[str], tuple[bool, str]]):
        self.port = port
        self.api_port = api_port
        self._admin_token = admin_token
        self._launcher_state = launcher_state
        self._control = control
        self.probe = SystemProbe(disk)
        self._server: PanelServer | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        server = PanelServer(("127.0.0.1", self.port), PanelHandler)
        server.panel = self
        self._server = server
        self.port = server.server_port  # Порт 0 в тестах: система выбрала свободный.
        # Первый замер процессора — точка отсчёта для следующего.
        self.probe.cpu_percent()
        Thread(target=server.serve_forever, kwargs={"poll_interval": 0.25}, daemon=True,
               name="admin-panel").start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def _api(self, method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(f"http://127.0.0.1:{self.api_port}{path}", data=body, method=method,
                          headers={"X-Trend-Admin-Token": self._admin_token,
                                   **({"Content-Type": "application/json"} if body is not None else {})})
        try:
            with _OPENER.open(request, timeout=API_TIMEOUT_SECONDS) as response:
                return response.status, json.loads(response.read(8_000_000))
        except HTTPError as error:
            try:
                detail = json.loads(error.read(10_000)).get("error")
            except (ValueError, AttributeError):
                detail = None
            return error.code, {"error": detail if isinstance(detail, str) else "API отказал в запросе."}
        except (URLError, OSError, ValueError):
            return HTTPStatus.BAD_GATEWAY, {"error": "API не отвечает."}

    def overview(self, after: int, person: str | None = None) -> dict[str, Any]:
        launcher = self._launcher_state()
        for process in launcher.get("processes", []):
            process["memory"] = self.probe.process_memory(process.get("pid")) if process.get("alive") else None
        # Выключенный или ещё не поднявшийся API не спрашиваем: ответ известен.
        if launcher.get("server", {}).get("state") != "running":
            return {"now": iso(time.time()), "launcher": launcher, "system": self.probe.snapshot(),
                    "api": None, "api_error": None}
        status, api = self._api("GET", f"/admin/overview?after={after}" + (f"&person={person}" if person else ""))
        return {"now": iso(time.time()), "launcher": launcher, "system": self.probe.snapshot(),
                "api": api if status == HTTPStatus.OK else None,
                "api_error": None if status == HTTPStatus.OK else api.get("error")}

    def cancel(self, run_id: str) -> tuple[int, dict[str, Any]]:
        return self._api("POST", "/admin/cancel", {"id": run_id})

    def act(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return self._api("POST", "/admin/action", payload)

    def run(self, run_id: str) -> tuple[int, dict[str, Any]]:
        return self._api("GET", f"/admin/run?run_id={run_id}")

    def view(self, route: str, query: str) -> tuple[int, dict[str, Any]]:
        """Раздел вкладки панели у API; строка запроса уже проверена обработчиком."""
        return self._api("GET", "/admin" + route + (f"?{query}" if query else ""))

    def control(self, action: str) -> tuple[bool, str]:
        return self._control(action)


def valid_action(payload: object) -> bool:
    """Команда владельца над посетителем в том виде, в каком её шлёт страница панели."""
    if isinstance(payload, dict) and payload.get("action") in OWNER_ACTIONS:
        values = payload.get("values", {})
        return (set(payload) <= {"action", "values"} and isinstance(values, dict) and len(values) <= 10
                and len(json.dumps(values)) <= 2_000)
    if not isinstance(payload, dict) or payload.get("action") not in VISITOR_ACTIONS:
        return False
    action = payload["action"]
    needs = {"block": {"person"}, "sign_out": {"person"}, "message": {"person", "text"},
             "unblock": {"id"}, "pause": set(), "resume": set(), "resources": {"values"},
             "unload_llm": set()}[action]
    if set(payload) != {"action", *needs}:
        return False
    if action == "resources":
        # Известные пределы проверяются здесь; новые настройки строго проверяет
        # API, чтобы ради нового переключателя не перезапускать окно запуска.
        values = payload["values"]
        return (isinstance(values, dict) and 0 < len(values) <= 10 and all(
            isinstance(name, str) and re.fullmatch(r"[a-z_]{1,32}", name) is not None
            and (type(value) is bool or type(value) is int and (
                name not in RESOURCE_LIMITS or RESOURCE_LIMITS[name][0] <= value <= RESOURCE_LIMITS[name][1]))
            and (name not in RESOURCE_FLAGS or type(value) is bool)
            for name, value in values.items()))
    return (("person" not in needs or isinstance(payload["person"], str) and PERSON.fullmatch(payload["person"]))
            and ("id" not in needs or isinstance(payload["id"], str) and BLOCK_ID.fullmatch(payload["id"]))
            and ("text" not in needs or isinstance(payload["text"], str)
                 and 0 < len(payload["text"].strip()) and len(payload["text"]) <= MAX_MESSAGE_CHARACTERS)) is True


class PanelServer(ThreadingHTTPServer):
    daemon_threads = True
    panel: AdminPanel


class PanelHandler(BaseHTTPRequestHandler):
    server: PanelServer
    protocol_version = "HTTP/1.1"
    server_version = "Trendanalyser-panel"
    sys_version = ""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _origins(self) -> set[str]:
        port = self.server.panel.port
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _host_allowed(self) -> bool:
        # Петлевой адрес не спасает от подмены DNS чужим сайтом: отвечаем только своему Host.
        hosts = self.headers.get_all("Host", [])
        return len(hosts) == 1 and hosts[0].lower() in self._origins()

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        target = urlsplit(self.path)
        if target.path == "/":
            nonce = secrets.token_urlsafe(18)
            page = PAGE.read_text(encoding="utf-8").replace("{{nonce}}", nonce).encode("utf-8")
            self._send(HTTPStatus.OK, page, "text/html; charset=utf-8", {
                "Content-Security-Policy": (f"default-src 'none'; script-src 'nonce-{nonce}'; "
                                            "style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; "
                                            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
                "X-Frame-Options": "DENY"})
        elif target.path == "/overview":
            try:
                parameters = dict(parse_qsl(target.query))
                after, person = parameters.get("after", "0"), parameters.get("person") or None
                if (re.fullmatch(r"[0-9]{1,9}", after) is None
                        or person is not None and PERSON.fullmatch(person) is None):
                    raise ValueError
            except ValueError:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
                return
            self._json(HTTPStatus.OK, self.server.panel.overview(int(after), person))
        elif target.path == "/run":
            run_id = dict(parse_qsl(target.query)).get("run_id", "")
            if RUN_ID.fullmatch(run_id) is None:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
                return
            status, answer = self.server.panel.run(run_id)
            self._json(status, answer)
        elif target.path in VIEWS:
            try:
                pairs = parse_qsl(target.query, keep_blank_values=True, strict_parsing=bool(target.query))
            except ValueError:
                pairs = None
            if (pairs is None or len(target.query) > MAX_VIEW_QUERY
                    or not {name for name, _ in pairs} <= VIEWS[target.path] or len(pairs) > len(VIEWS[target.path])):
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
                return
            status, answer = self.server.panel.view(target.path, urlencode(pairs))
            self._json(status, answer)
        elif target.path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "text/plain")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        # Команды — только со страницы самой панели: чужая страница не пошлёт
        # JSON со своим заголовком без предварительного запроса, а его мы не разрешаем.
        origins = self.headers.get_all("Origin", [])
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if (not self._host_allowed() or self.headers.get("X-Panel") != "1"
                or any(urlsplit(origin).netloc.lower() not in self._origins() for origin in origins)
                or fetch_site not in {None, "same-origin"}
                or self.headers.get_content_type() != "application/json"):
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        length = self.headers.get("Content-Length", "")
        if not length.isdigit() or not 0 < int(length) <= MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
            return
        try:
            payload = json.loads(self.rfile.read(int(length)).decode("utf-8"))
        except (ValueError, UnicodeError):
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
            return
        if self.path == "/cancel":
            if (not isinstance(payload, dict) or set(payload) != {"id"} or not isinstance(payload["id"], str)
                    or RUN_ID.fullmatch(payload["id"]) is None):
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
                return
            status, answer = self.server.panel.cancel(payload["id"])
            self._json(status, answer)
        elif self.path == "/action":
            if not valid_action(payload):
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
                return
            status, answer = self.server.panel.act(payload)
            self._json(status, answer)
        elif self.path == "/server":
            if not isinstance(payload, dict) or set(payload) != {"action"} or payload["action"] not in SERVER_ACTIONS:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid_request"})
                return
            accepted, message = self.server.panel.control(payload["action"])
            self._json(HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT,
                       {"message": message} if accepted else {"error": message})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
