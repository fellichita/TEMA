"""Create a local, locked macOS environment and start the desktop application.

The standard library is sufficient to start this script. Nothing is installed
system-wide. A second launch validates the local runtime without network calls.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
PIP_VERSION = "26.2.1"

# This probe runs in the target environment, where packaging must be installed.
# Import checks exercise native libraries before opening the user's database.
RUNTIME_PROBE = r'''
import importlib.metadata
import os
import pathlib
import sys
import tkinter
from packaging.requirements import Requirement

root = pathlib.Path(sys.argv[1])
assert sys.version_info[:2] == (3, 13), "Нужен Python 3.13"
assert pathlib.Path(sys.prefix).resolve() == (root / ".venv").resolve(), "Чужое окружение"
seen = set()
def validate(path):
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r "):
            validate(path.parent / line[3:].strip())
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            installed = importlib.metadata.version(requirement.name)
            assert installed in requirement.specifier, f"Нужен {requirement}; установлен {installed}"
validate(root / "requirements/pilot.lock")
# ORT's native telemetry starts on import; disable it before loading the module.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import numpy
import scipy
import sklearn
import onnxruntime
onnxruntime.disable_telemetry_events()
import tokenizers
import keyring
import pypdf
from app.sqlite_runtime import sqlite3
from app.pilot.encoder import load_spec
from app.ui.window import Application
load_spec()
assert tkinter.TkVersion >= 8.6, "Нужен Tk 8.6 или новее"
print(f"Python {sys.version.split()[0]}, Tk {tkinter.TkVersion}, "
      f"SQLite {sqlite3.sqlite_version}, ONNX Runtime {onnxruntime.__version__}")
'''


class SetupError(RuntimeError):
    """An actionable source-launch failure."""


def validate_platform() -> None:
    if sys.platform != "darwin":
        raise SetupError("Этот запускатель предназначен для macOS. Другие платформы: README.md.")
    if sys.version_info[:2] != (3, 13):
        raise SetupError("Нужен обычный CPython 3.13 с Tkinter, установленный с python.org.")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise SetupError("Нужен обычный Python 3.13; экспериментальная free-threaded сборка не поддерживается.")
    if platform.machine() != "arm64":
        raise SetupError("Эта версия локального ONNX Runtime требует Apple Silicon (M1 или новее). "
                         "Intel Mac пока не поддерживается. На Apple Silicon используйте нативный "
                         "Python arm64 и выключите запуск Terminal через Rosetta.")
    if int(platform.mac_ver()[0].split(".")[0]) < 14:
        raise SetupError("Закреплённому ONNX Runtime нужна macOS 14 или новее.")
    try:
        import tkinter

        if tkinter.TkVersion < 8.6:
            raise ImportError
    except ImportError:
        raise SetupError("В Python отсутствует Tkinter 8.6+. Установите Python 3.13 macOS "
                         "universal2 installer с python.org; для Homebrew нужен python-tk@3.13.") from None


def child_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PIP_") and key not in {
               "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "TCL_LIBRARY", "TK_LIBRARY"}}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1",
               PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", ORT_DISABLE_TELEMETRY="1")
    return env


def run(command: list[str], root: Path, *, capture: bool = False, timeout: int = 1800):
    return subprocess.run(command, cwd=root, env=child_environment(), check=False,
                          timeout=timeout, capture_output=capture, text=True)


def runtime_status(root: Path) -> tuple[bool, str]:
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        return False, "Локальное окружение ещё не создано."
    try:
        result = run([str(python), "-E", "-s", "-c", RUNTIME_PROBE, str(root)], root,
                     capture=True, timeout=90)
        if result.returncode:
            detail = result.stderr.strip().splitlines()[-1:] or ["Причина не указана."]
            return False, "Окружение не прошло проверку: " + detail[0][:500]
        checked = run([str(python), "-m", "pip", "check"], root, capture=True, timeout=90)
        if checked.returncode:
            return False, "В окружении конфликт зависимостей."
        return True, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False, "Не удалось запустить проверку локального окружения."


@contextmanager
def installation_lock(root: Path):
    if sys.platform == "win32":
        raise SetupError("Блокировка macOS-запускателя недоступна на Windows.")
    import fcntl

    # A global temporary lock prevents two launchers writing the same new venv.
    # It is intentionally independent of the presence of the build directory.
    import tempfile

    key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]
    path = Path(tempfile.gettempdir()) / f"trendanalizer-setup-{os.getuid()}-{key}.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError("Установка в эту папку уже идёт в другом окне. Дождитесь её завершения.") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def install(root: Path) -> None:
    directory = root / ".venv"
    if directory.is_symlink():
        raise SetupError(".venv является ссылкой на другое окружение. Используйте отдельную локальную .venv; "
                         "связанный каталог автоматически не изменяется.")
    compiler = run(["/usr/bin/xcrun", "--find", "clang"], root, capture=True, timeout=30)
    if compiler.returncode:
        raise SetupError("Нужны Xcode Command Line Tools для локальной сборки SQLite. "
                         "В Terminal выполните xcode-select --install, закончите установку "
                         "и снова откройте запускатель.")
    python = directory / "bin" / "python"
    if directory.exists():
        if not python.is_file():
            raise SetupError("Папка .venv уже существует, но не содержит работающий Python. "
                             "Переименуйте только .venv и запустите повторно; данные приложения вне неё.")
        result = run([str(python), "-E", "-s", "-c",
                      "import sys; from pathlib import Path; "
                      "raise SystemExit(not (sys.version_info[:2] == (3, 13) "
                      "and sys.prefix != sys.base_prefix "
                      "and Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()))", str(directory)],
                     root, capture=True)
        if result.returncode:
            raise SetupError("Существующее .venv не является отдельным окружением Python 3.13. Переименуйте .venv "
                             "и повторите запуск; старое окружение автоматически не удаляется.")
    else:
        result = run([sys.executable, "-E", "-s", "-m", "venv", str(directory)], root)
        if result.returncode:
            raise SetupError("Не удалось создать .venv. Проверьте доступ на запись в папку проекта.")
    pip = [str(python), "-m", "pip", "install", "--index-url", "https://pypi.org/simple"]
    commands = [
        ("Подготавливаем установщик", [*pip, "--upgrade", f"pip=={PIP_VERSION}"]),
        ("Устанавливаем библиотеки приложения", [*pip, "--only-binary=:all:", "-r",
         "requirements/semantic.lock", "-r", "requirements/native-build.lock"]),
        ("Собираем проверенную версию SQLite", [str(python), "-m", "scripts.build_sqlite_runtime"]),
        ("Устанавливаем SQLite и оставшиеся библиотеки", [*pip, "--only-binary=:all:",
         "-r", "requirements/pilot.lock"]),
    ]
    for title, command in commands:
        print(f"\n{title}…", flush=True)
        if run(command, root).returncode:
            raise SetupError("Установка остановлена на этапе: " + title + ". Причина показана выше. "
                             "Проверьте интернет и повторите запуск: выполненные загрузки сохраняются в кеше.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Проверка без установки, GUI и сетевых запросов")
    mode.add_argument("--install-only", action="store_true", help="Подготовить окружение без запуска GUI")
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="Аргументы приложения после --")
    args = parser.parse_args(argv)
    try:
        validate_platform()
        with installation_lock(ROOT):
            ready, message = runtime_status(ROOT)
            if not ready and args.check:
                raise SetupError(message + " Запустите без --check для установки.")
            if not ready:
                print("Первый запуск требует интернета для библиотек. Ключи и модель настраиваются в приложении.",
                      flush=True)
                install(ROOT)
                ready, message = runtime_status(ROOT)
                if not ready:
                    raise SetupError(message + " Подробная проверка: .venv/bin/python -m pip check.")
            print("Окружение готово: " + message, flush=True)
        if args.check or args.install_only:
            return 0
        app_args = args.app_args[1:] if args.app_args[:1] == ["--"] else args.app_args
        return subprocess.call([str(ROOT / ".venv" / "bin" / "python"), "-E", "-s", "-m", "app.main",
                                *app_args], cwd=ROOT, env=child_environment())
    except (SetupError, OSError, subprocess.TimeoutExpired) as error:
        print(f"\n{error}\nИнструкция: {ROOT / 'docs' / 'guides' / 'macos-setup.md'}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nЗапуск прерван. Можно безопасно запустить снова.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
