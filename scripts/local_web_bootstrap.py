"""Prepare an isolated local web analysis and start its loopback services."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from scripts import mac_bootstrap

ROOT = Path(__file__).resolve().parents[1]
WEB_PROFILE = ROOT / "storage" / "web-local-profile"
API_PYTHON = ROOT / ".venv" / "bin" / "python"

WEB_RUNTIME_PROBE = r'''
import importlib.metadata
from pathlib import Path
import sys
from packaging.requirements import Requirement

root = Path(sys.argv[1])
assert sys.version_info[:2] == (3, 13), "Нужен Python 3.13"
assert Path(sys.prefix).resolve() == (root / ".venv-web").resolve(), "Чужое веб-окружение"
for raw in (root / "requirements/web.txt").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    if requirement.marker is None or requirement.marker.evaluate():
        installed = importlib.metadata.version(requirement.name)
        assert installed in requirement.specifier, f"Нужен {requirement}; установлен {installed}"
import streamlit
import requests
'''

PROFILE_SETUP = r'''
from pathlib import Path
import sys
from app.identity import validate_data_dir
from app.pilot.settings import PilotSettings, load_settings, save_settings

profile = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
storage = root / "storage"
if storage.is_symlink() or profile.is_symlink():
    raise ValueError("Каталог storage или веб-профиль является ссылкой на другую папку.")
if not profile.resolve().is_relative_to(storage.resolve()) or profile.parent.resolve() != storage.resolve():
    raise ValueError("Веб-профиль должен находиться внутри storage этого проекта.")
validate_data_dir(profile)
if (profile / "active-library.json").exists():
    raise ValueError("Веб-профиль перенаправлен на другую библиотеку. Уберите ссылку в active-library.json.")
original = load_settings(profile)
settings = original
if settings.provider != "local":
    # This dedicated web profile is always offline; desktop preferences live
    # elsewhere and are never read or rewritten by this launcher.
    settings = PilotSettings.for_provider("local")
if not (profile / "settings.json").exists() or settings.provider != original.provider:
    save_settings(profile, settings)
print("Отдельный веб-профиль готов: локальная модель выбрана.", flush=True)
'''


class WebSetupError(RuntimeError):
    """A local web setup failure with an actionable message."""


def web_runtime_status(root: Path) -> tuple[bool, str]:
    python = root / ".venv-web" / "bin" / "python"
    if not python.is_file():
        return False, "Веб-окружение ещё не создано."
    try:
        result = mac_bootstrap.run([str(python), "-E", "-s", "-c", WEB_RUNTIME_PROBE, str(root)],
                                   root, capture=True, timeout=90)
        if result.returncode:
            detail = result.stderr.strip().splitlines()[-1:] or ["Причина не указана."]
            return False, "Веб-окружение не прошло проверку: " + detail[0][:500]
        checked = mac_bootstrap.run([str(python), "-m", "pip", "check"], root, capture=True, timeout=90)
        if checked.returncode:
            return False, "В веб-окружении конфликт зависимостей."
        return True, "Streamlit и веб-библиотеки установлены."
    except (OSError, subprocess.TimeoutExpired):
        return False, "Не удалось проверить веб-окружение."


def install_web(root: Path) -> None:
    directory = root / ".venv-web"
    python = directory / "bin" / "python"
    if directory.is_symlink():
        raise WebSetupError(".venv-web является ссылкой. Переименуйте её и повторите запуск.")
    if directory.exists():
        if not python.is_file():
            raise WebSetupError(".venv-web уже есть, но в ней нет Python. Переименуйте папку и повторите запуск.")
        existing = mac_bootstrap.run([str(python), "-E", "-s", "-c",
                                      "import sys; from pathlib import Path; "
                                      "raise SystemExit(not (sys.version_info[:2] == (3, 13) "
                                      "and sys.prefix != sys.base_prefix "
                                      "and Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()))",
                                      str(directory)], root, capture=True, timeout=90)
        if existing.returncode:
            raise WebSetupError("Существующая .venv-web не является окружением Python 3.13. "
                                "Переименуйте её и повторите запуск.")
    else:
        print("Создаём веб-окружение…", flush=True)
        created = mac_bootstrap.run([sys.executable, "-E", "-s", "-m", "venv", str(directory)],
                                    root, timeout=180)
        if created.returncode:
            raise WebSetupError("Не удалось создать .venv-web. Проверьте доступ на запись в папку проекта.")
    print("Устанавливаем веб-библиотеки…", flush=True)
    installed = mac_bootstrap.run([str(python), "-m", "pip", "install", "--index-url",
                                   "https://pypi.org/simple", "--only-binary=:all:", "-r",
                                   "requirements/web.txt"], root, timeout=1800)
    if installed.returncode:
        raise WebSetupError("Не удалось установить веб-библиотеки. Проверьте интернет и повторите запуск.")


def ensure_web_runtime(root: Path) -> None:
    ready, message = web_runtime_status(root)
    if not ready:
        print(message, flush=True)
        install_web(root)
        ready, message = web_runtime_status(root)
        if not ready:
            raise WebSetupError(message)
    print(message, flush=True)


def prepare_web_profile(root: Path, profile: Path) -> None:
    result = mac_bootstrap.run([str(root / ".venv" / "bin" / "python"), "-E", "-s", "-c",
                                PROFILE_SETUP, str(profile), str(root)], root, capture=True, timeout=90)
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1:] or ["Причина не указана."]
        raise WebSetupError("Не удалось подготовить отдельный веб-профиль: " + detail[0][:500])
    print(result.stdout.strip(), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-only", action="store_true", help="Подготовить всё без запуска сайта")
    args = parser.parse_args(argv)
    try:
        if mac_bootstrap.main(["--install-only"]):
            return 1
        with mac_bootstrap.installation_lock(ROOT):
            ensure_web_runtime(ROOT)
            prepare_web_profile(ROOT, WEB_PROFILE)
            print("Проверяем и при необходимости скачиваем модели анализа…", flush=True)
            model_command = [str(API_PYTHON), "-E", "-s", "-m", "scripts.setup_models",
                             "--web-analysis", "--profile", str(WEB_PROFILE)]
            if subprocess.call(model_command, cwd=ROOT, env=mac_bootstrap.child_environment()):
                raise WebSetupError("Модели анализа не готовы. Проверьте сообщение выше и повторите запуск.")
        if args.install_only:
            print("Локальная веб-версия готова. Откройте .command ещё раз для запуска.", flush=True)
            return 0
        print("Запускаем локальную веб-версию…", flush=True)
        return subprocess.call([str(API_PYTHON), "-E", "-s", "-m", "scripts.run_web_demo",
                                "--local-only", "--open-browser", "--no-auth",
                                "--data-dir", str(WEB_PROFILE)],
                               cwd=ROOT, env=mac_bootstrap.child_environment())
    except (WebSetupError, mac_bootstrap.SetupError, OSError, subprocess.TimeoutExpired) as error:
        print(f"\n{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nЗапуск прерван. Выполненные загрузки сохраняются.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
