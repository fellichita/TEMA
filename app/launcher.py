"""Small native start window for the bundled desktop and loopback web modes."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from queue import Empty, Queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser


_OPENER = build_opener(ProxyHandler({}))


class LauncherError(RuntimeError):
    """An actionable failure shown in the launcher window."""


def _free_loopback_ports() -> tuple[int, int]:
    """Ask the OS for two distinct ephemeral ports, never a fixed shared pair."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as first:
        first.bind(("127.0.0.1", 0))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second:
            second.bind(("127.0.0.1", 0))
            return first.getsockname()[1], second.getsockname()[1]


def _wait_for_service(process: subprocess.Popen, url: str, *, cancel: threading.Event,
                      instance_id: str | None = None, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancel.is_set():
            raise LauncherError("Запуск отменён.")
        if process.poll() is not None:
            raise LauncherError("Веб-сервис не запустился. Повторите запуск; если ошибка сохранится, проверьте пакет.")
        try:
            with _OPENER.open(url, timeout=1) as response:
                valid = (response.status == 200 and (instance_id is None or
                         response.headers.get("X-Trend-Instance") == instance_id))
                if valid and process.poll() is None:
                    return
        except OSError:
            pass
        cancel.wait(.25)
    raise LauncherError("Веб-сервис не ответил вовремя. Повторите запуск.")


def _child_commands(executable: Path, profile: Path, api_port: int,
                    web_port: int) -> tuple[list[str], list[str]]:
    return ([str(executable), "--web-api", "--port", str(api_port), "--data-dir", str(profile)],
            [str(executable), "--web-ui", "--port", str(web_port)])


def _child_environments(api_port: int, instance_id: str) -> tuple[dict[str, str], dict[str, str]]:
    # The web process receives only the private API capability. The API keeps
    # the provider environment used by the existing desktop application.
    from scripts.run_web_demo import _child_environments as demo_environments

    api, web = demo_environments(None, secrets.token_urlsafe(32), api_port=api_port)
    api["TREND_API_INSTANCE_ID"] = instance_id
    for environment in (api, web):
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        # A child started from a PyInstaller executable must get its own
        # bootloader state, independent of the already-running start window.
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return api, web


class _WebSession:
    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.processes: list[subprocess.Popen] = []
        self.lock = threading.Lock()
        self.url: str | None = None

    def add(self, process: subprocess.Popen) -> None:
        with self.lock:
            self.processes.append(process)
        if self.cancel.is_set():
            self.stop()

    def alive(self) -> bool:
        with self.lock:
            return bool(self.processes) and all(process.poll() is None for process in self.processes)

    def stop(self) -> None:
        self.cancel.set()
        with self.lock:
            processes = tuple(reversed(self.processes))
        for process in processes:
            if process.poll() is None:
                with suppress(OSError):
                    process.terminate()
        deadline = time.monotonic() + 8
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        process.kill()


def _start_web(session: _WebSession, events: Queue[tuple[_WebSession, str, str]]) -> None:
    try:
        from scripts.frozen_web_setup import prepare_frozen_web_profile

        profile = prepare_frozen_web_profile(progress=lambda value: events.put((session, "status", value)),
                                             cancel=session.cancel)
        if session.cancel.is_set():
            return
        events.put((session, "status", "Запускаем локальные сервисы…"))
        api_port, web_port = _free_loopback_ports()
        instance_id = secrets.token_urlsafe(24)
        api_env, web_env = _child_environments(api_port, instance_id)
        api_command, web_command = _child_commands(Path(sys.executable), profile, api_port, web_port)
        api = subprocess.Popen(api_command, cwd=profile, env=api_env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT)
        session.add(api)
        _wait_for_service(api, f"http://127.0.0.1:{api_port}/health", cancel=session.cancel,
                          instance_id=instance_id)
        if session.cancel.is_set():
            return
        web = subprocess.Popen(web_command, cwd=profile, env=web_env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT)
        session.add(web)
        _wait_for_service(web, f"http://127.0.0.1:{web_port}/_stcore/health", cancel=session.cancel)
        if session.cancel.is_set():
            return
        session.url = f"http://127.0.0.1:{web_port}"
        events.put((session, "ready", session.url))
    except Exception as error:  # noqa: BLE001 - the user needs a recoverable launcher state
        if not session.cancel.is_set():
            events.put((session, "error", str(error)[:600] or type(error).__name__))
    finally:
        if session.url is None or session.cancel.is_set():
            session.stop()


def run_launcher() -> int:
    """Show one installed app icon with desktop and local web choices."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Trendanalyser")
    root.geometry("580x425")
    root.minsize(500, 400)
    root.resizable(True, True)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame = ttk.Frame(root, padding=28)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(0, weight=1)
    ttk.Label(frame, text="Trendanalyser", font=("TkDefaultFont", 22, "bold")).grid(
        row=0, column=0, sticky="w")
    ttk.Label(frame, text="Выберите, как открыть программу.", font=("TkDefaultFont", 12)).grid(
        row=1, column=0, sticky="w", pady=(8, 20))
    status = tk.StringVar(value="Готово к запуску")
    events: Queue[tuple[_WebSession, str, str]] = Queue()
    session: _WebSession | None = None
    worker: threading.Thread | None = None
    desktop_selected = False
    closing = False

    def open_desktop() -> None:
        nonlocal desktop_selected
        desktop_selected = True
        root.destroy()

    def start_web() -> None:
        nonlocal session, worker
        if session is not None:
            return
        session = _WebSession()
        desktop_button.state(["disabled"])
        web_button.state(["disabled"])
        progress.grid()
        progress.start(12)
        status.set("Готовим веб-версию. При первом запуске загрузится модель около 1,8 ГБ…")
        worker = threading.Thread(target=_start_web, args=(session, events), name="web-launch", daemon=True)
        worker.start()

    def stop_web() -> None:
        nonlocal session
        if session is None:
            return
        old = session
        session = None
        old.cancel.set()
        threading.Thread(target=old.stop, name="web-stop", daemon=True).start()
        progress.stop()
        progress.grid_remove()
        browser_button.grid_remove()
        stop_button.grid_remove()
        desktop_button.state(["!disabled"])
        web_button.state(["!disabled"])
        status.set("Веб-версия остановлена. Можно запустить её снова.")

    def close() -> None:
        nonlocal closing
        closing = True
        if session is not None:
            session.cancel.set()
        root.destroy()

    desktop_button = ttk.Button(frame, text="Открыть настольную версию", command=open_desktop)
    desktop_button.grid(row=2, column=0, sticky="ew", ipady=10)
    web_button = ttk.Button(frame, text="Открыть веб-версию", command=start_web)
    web_button.grid(row=3, column=0, sticky="ew", ipady=10, pady=(10, 0))
    ttk.Label(frame, text="Веб открывается только на этом компьютере, в вашем браузере.",
              wraplength=500).grid(row=4, column=0, sticky="w", pady=(14, 8))
    progress = ttk.Progressbar(frame, mode="indeterminate")
    progress.grid(row=5, column=0, sticky="ew", pady=(0, 6))
    progress.grid_remove()
    ttk.Label(frame, textvariable=status, wraplength=510).grid(row=6, column=0, sticky="w")
    browser_button = ttk.Button(frame, text="Открыть в браузере",
                                command=lambda: webbrowser.open(session.url) if session and session.url else None)
    browser_button.grid(row=7, column=0, sticky="ew", pady=(10, 0))
    browser_button.grid_remove()
    stop_button = ttk.Button(frame, text="Остановить веб-версию", command=stop_web)
    stop_button.grid(row=8, column=0, sticky="ew", pady=(8, 0))
    stop_button.grid_remove()

    def poll() -> None:
        nonlocal session
        if closing or desktop_selected:
            return
        while True:
            try:
                source, kind, value = events.get_nowait()
            except Empty:
                break
            if source is not session:
                continue
            if kind == "status" and session is not None:
                status.set(value)
            elif kind == "ready" and session is not None and session.url == value:
                progress.stop()
                progress.grid_remove()
                status.set("Локальная веб-версия работает. Закрытие этого окна остановит её.")
                browser_button.grid()
                stop_button.grid()
                webbrowser.open(value)
            elif kind == "error" and session is not None:
                session = None
                progress.stop()
                progress.grid_remove()
                desktop_button.state(["!disabled"])
                web_button.state(["!disabled"])
                status.set("Не удалось запустить веб: " + value)
        if session is not None and session.url is not None and not session.alive():
            stop_web()
            status.set("Веб-сервис остановился. Запустите его снова.")
        root.after(150, poll)

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<Escape>", lambda _event: close())
    desktop_button.focus_set()
    root.after(150, poll)
    root.mainloop()
    if session is not None:
        session.stop()
    if worker is not None and worker.is_alive():
        worker.join(timeout=10)
    if desktop_selected:
        from app.runtime.session import credentials
        from app.ui.window import run_app

        credentials()
        sys.argv = [sys.argv[0]]
        run_app()
    return 0
