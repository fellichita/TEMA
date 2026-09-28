"""Prepare the local web profile used by a frozen, read-only application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from threading import Event

from app.backend.errors import BackendError
from app.backend.locking import InstanceLock
from app.identity import default_data_dir, validate_data_dir
from app.pilot import encoder, local_llm, translator
from app.pilot.settings import PilotSettings, load_settings, save_settings
from app.runtime.jobs import TaskFailure
from app.runtime.model_resources import frozen, resolve_model
from scripts.install_local_llm import install as install_local_llm
from scripts.setup_models import Model, _copy_verified

WEB_PROFILE_NAME = "web-local-profile"


class WebSetupError(RuntimeError):
    """A short, user-facing reason the local web could not be prepared."""


class WebSetupCancelled(WebSetupError):
    """The launcher was closed while preparing the web profile."""


def _check_cancel(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise WebSetupCancelled("Подготовка локального веба прервана.")


def _bundle_roots() -> tuple[Path, ...]:
    """Reject writes anywhere in the installed app, not only in its model tree."""
    executable = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        application = next((parent for parent in executable.parents if parent.suffix == ".app"), None)
        install_root = application or executable.parent
    else:
        install_root = executable.parent
    resource = getattr(sys, "_MEIPASS", None)
    return (install_root, Path(resource).resolve()) if isinstance(resource, str) and resource else (install_root,)


def _web_profile() -> Path:
    base = default_data_dir().expanduser().absolute()
    profile = base / WEB_PROFILE_NAME
    # Neither an app bundle nor a symlink/junction may become the writable
    # profile. The lock below repeats the ancestor check before creating files.
    try:
        redirected = any(path.is_symlink() or (path.exists() and not path.is_dir())
                         or path.absolute() != path.resolve(strict=False) for path in (base, profile))
    except (OSError, RuntimeError) as error:
        raise WebSetupError("Не удалось проверить пользовательский каталог данных веба.") from error
    if redirected:
        raise WebSetupError("Каталог данных веба перенаправлен. Выберите обычный пользовательский каталог.")
    try:
        resolved = validate_data_dir(profile)
    except (OSError, ValueError) as error:
        raise WebSetupError("Пользовательский каталог данных веба недоступен.") from error
    if any(resolved.is_relative_to(root) for root in _bundle_roots()):
        raise WebSetupError("Данные веба нельзя размещать внутри установленной программы.")
    if (profile / "active-library.json").exists() or (profile / "active-library.json").is_symlink():
        raise WebSetupError("Веб-профиль перенаправлен на другую библиотеку. Уберите выбор другой библиотеки.")
    return resolved


def _verify_bundled_models(cancel: Event | None) -> None:
    try:
        for key, spec, verify in (
            ("multilingual-e5-small", encoder.load_spec(), encoder.verify_artifacts),
            ("opus-mt-ru-en", translator.load_spec(), translator.verify_artifacts),
        ):
            _check_cancel(cancel)
            location = resolve_model(key, spec["revision"])
            if location.origin != "bundled":
                raise ValueError("Model is not bundled")
            verify(location.path, spec, cancel)
    except WebSetupCancelled:
        raise
    except Exception as error:
        if cancel is not None and cancel.is_set():
            raise WebSetupCancelled("Подготовка локального веба прервана.") from None
        raise WebSetupError("Встроенная модель анализа или перевода повреждена. Переустановите программу.") from error


def _prepare_settings(profile: Path) -> None:
    if (profile / "active-library.json").exists() or (profile / "active-library.json").is_symlink():
        raise WebSetupError("Веб-профиль перенаправлен на другую библиотеку. Уберите выбор другой библиотеки.")
    try:
        original = load_settings(profile)
        if original.provider != "local" or not (profile / "settings.json").exists():
            save_settings(profile, PilotSettings.for_provider("local"))
    except (TaskFailure, OSError, ValueError) as error:
        raise WebSetupError("Не удалось подготовить отдельные настройки локального веба.") from error


def _prepare_local_ai(profile: Path, progress: Callable[[str], None] | None,
                      cancel: Event | None) -> None:
    try:
        spec = local_llm.load_spec()
    except (OSError, ValueError) as error:
        raise WebSetupError("Встроенная спецификация AI-модели повреждена. Переустановите программу.") from error
    target = local_llm.model_directory(profile)
    source = local_llm.model_directory(default_data_dir())
    _check_cancel(cancel)
    if not target.exists() and source != target:
        # Reuse only a fully verified desktop copy. This preserves an offline
        # first launch for users who already installed the AI model there.
        model = Model(local_llm.MODEL_KEY, "локальный AI-анализ", spec,
                      local_llm.verify_artifacts, target, install_local_llm, staged=False)
        try:
            if _copy_verified(source, target, model):
                if progress is not None:
                    progress("Локальная AI-модель скопирована из настольного приложения.")
                return
        except (OSError, local_llm.LocalModelError) as error:
            raise WebSetupError("Не удалось скопировать локальную AI-модель в веб-профиль.") from error
    _check_cancel(cancel)
    total = sum(item["bytes"] for item in spec["files"])
    last_decile = -1

    def on_bytes(received: int, _expected: int) -> None:
        nonlocal last_decile
        _check_cancel(cancel)
        decile = min(10, max(0, received * 10 // total))
        if progress is not None and decile > last_decile:
            last_decile = decile
            progress(f"Локальная AI-модель: {received / 1e6:.0f} из {total / 1e6:.0f} МБ ({decile * 10}%).")

    try:
        install_local_llm(target, cancel=cancel, progress=on_bytes)
    except Exception as error:
        if cancel is not None and cancel.is_set():
            raise WebSetupCancelled("Загрузка локальной AI-модели прервана. Можно запустить снова.") from None
        if isinstance(error, local_llm.LocalModelError) and target.exists():
            raise WebSetupError("Локальная AI-модель в веб-профиле повреждена. "
                                "Переместите её папку из пользовательских данных и запустите снова.") from error
        raise WebSetupError("Не удалось загрузить локальную AI-модель. "
                            "Проверьте интернет и место на диске, затем запустите снова.") from error


def prepare_frozen_web_profile(*, progress: Callable[[str], None] | None = None,
                               cancel: Event | None = None) -> Path:
    """Return a ready isolated web profile; never write into the frozen bundle.

    The scientific encoder and translator remain in the signed application.
    The larger instruction model is downloaded once into the user's profile.
    """
    if not frozen():
        raise WebSetupError("Этот способ запуска предназначен для установленной программы.")
    _check_cancel(cancel)
    profile = _web_profile()
    if progress is not None:
        progress("Проверяем встроенные модели анализа и перевода…")
    _verify_bundled_models(cancel)
    _check_cancel(cancel)
    lock = InstanceLock(profile / ".web-setup.lock")
    try:
        lock.acquire()
    except BackendError as error:
        raise WebSetupError("Подготовка локального веба уже идёт или каталог данных недоступен.") from error
    try:
        _check_cancel(cancel)
        _prepare_settings(profile)
        if progress is not None:
            progress("Проверяем локальную AI-модель…")
        _prepare_local_ai(profile, progress, cancel)
        _check_cancel(cancel)
        return profile
    finally:
        lock.release()
