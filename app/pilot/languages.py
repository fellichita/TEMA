"""Языки перевода результатов: встроенный русский и языки, добавленные владельцем.

Перевод ТОПа — машинный черновик для чтения: английские заголовки и аннотации
переводятся локальной моделью Marian (семейство Helsinki-NLP opus-mt в
экспорте ONNX). Русский встроен и закреплён в коде. Другой язык владелец
добавляет в панели управления: выбирает его из каталога или указывает код
языка и модель Hugging Face сам.

При добавлении программа один раз обращается к Hugging Face: узнаёт текущую
ревизию модели и контрольные суммы файлов, скачивает их, сверяет каждый байт
и записывает спецификацию в профиль. Дальше модель загружается только по этой
спецификации и только с диска — как и встроенные модели, сеть при переводе не
нужна. Совпадение с опубликованной ревизией подтверждает целостность, но не
качество перевода.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import Event, Lock, Thread
import time
from typing import Any

from app.pilot.translator import (
    CACHED_DECODER, ENGLISH_RUSSIAN_KEY, PUBLISHED_NORMALIZER, TranslationError, load_spec, model_directory,
    validate_spec,
)

SETTINGS_FILE = "web-languages.json"
SPECS_DIRECTORY = "translation-specs"
DEFAULT_LANGUAGE = "ru"
CODE = re.compile(r"[a-z]{2,3}\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
MAX_LANGUAGES = 20
MAX_FILE_BYTES = 200_000_000
REQUIRED = {"encoder_model_quantized.onnx": "onnx/encoder_model_quantized.onnx",
            "decoder_model_quantized.onnx": "onnx/decoder_model_quantized.onnx",
            "tokenizer.json": "tokenizer.json"}
OPTIONAL = {CACHED_DECODER: "onnx/" + CACHED_DECODER}

# Языки, модели которых известны заранее (английский → язык). Любой другой
# владелец добавит сам, указав модель.
CATALOGUE: dict[str, tuple[str, str]] = {
    "ru": ("Русский", "Xenova/opus-mt-en-ru"),
    "de": ("Немецкий", "Xenova/opus-mt-en-de"),
    "fr": ("Французский", "Xenova/opus-mt-en-fr"),
    "es": ("Испанский", "Xenova/opus-mt-en-es"),
    "it": ("Итальянский", "Xenova/opus-mt-en-it"),
    "zh": ("Китайский", "Xenova/opus-mt-en-zh"),
    "ja": ("Японский", "Xenova/opus-mt-en-jap"),
    "uk": ("Украинский", "Xenova/opus-mt-en-uk"),
    "ar": ("Арабский", "Xenova/opus-mt-en-ar"),
    "hi": ("Хинди", "Xenova/opus-mt-en-hi"),
    "nl": ("Нидерландский", "Xenova/opus-mt-en-nl"),
    "sv": ("Шведский", "Xenova/opus-mt-en-sv"),
}


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    model_id: str

    @property
    def bundled(self) -> bool:
        return self.code == DEFAULT_LANGUAGE


def _mapping(value: object) -> dict[str, Any]:
    """Словарь из разобранного JSON или пустой словарь."""
    return value if isinstance(value, dict) else {}


def _items(value: object) -> list[Any]:
    """Список из разобранного JSON или пустой список."""
    return value if isinstance(value, list) else []


def _safe(model_id: str) -> str:
    return model_id.replace("/", "--")


def language_model_directory(data_dir: Path, language: Language) -> Path:
    if language.bundled:
        return model_directory(data_dir, ENGLISH_RUSSIAN_KEY)
    return Path(data_dir) / "models" / ("translation-" + _safe(language.model_id))


def spec_path(data_dir: Path, language: Language) -> Path:
    return Path(data_dir) / "models" / SPECS_DIRECTORY / (_safe(language.model_id) + ".json")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def validate_language(code: object, name: object, model_id: object) -> Language:
    if not isinstance(code, str) or CODE.fullmatch(code) is None or code == "en":
        raise ValueError("Код языка — две-три строчные латинские буквы (например, de), кроме en.")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40 or any(char in name for char in "<>\"'\n"):
        raise ValueError("Название языка — до 40 символов без кавычек и угловых скобок.")
    if not isinstance(model_id, str) or MODEL_ID.fullmatch(model_id) is None:
        raise ValueError("Модель — идентификатор Hugging Face вида владелец/название.")
    if code == DEFAULT_LANGUAGE and model_id != CATALOGUE[DEFAULT_LANGUAGE][1]:
        raise ValueError("Русский встроен в программу, его модель не меняется.")
    return Language(code, name.strip(), model_id)


def load_settings(data_dir: Path) -> dict[str, Any]:
    """Добавленные языки и язык, на который сайт переводит ТОП по умолчанию."""
    languages = {DEFAULT_LANGUAGE: Language(DEFAULT_LANGUAGE, *CATALOGUE[DEFAULT_LANGUAGE])}
    default = DEFAULT_LANGUAGE
    try:
        raw = json.loads((Path(data_dir) / SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if isinstance(raw, dict):
        for entry in raw.get("languages", []) if isinstance(raw.get("languages"), list) else []:
            if not isinstance(entry, dict):
                continue
            try:
                language = validate_language(entry.get("code"), entry.get("name"), entry.get("model_id"))
            except ValueError:
                continue
            languages.setdefault(language.code, language)
        if raw.get("default") in languages:
            default = raw["default"]
    return {"languages": languages, "default": default}


def save_settings(data_dir: Path, languages: dict[str, Language], default: str) -> None:
    _write_json(Path(data_dir) / SETTINGS_FILE, {
        "version": 1, "default": default,
        "languages": [{"code": item.code, "name": item.name, "model_id": item.model_id}
                      for item in languages.values() if not item.bundled]})


def installed_spec(data_dir: Path, language: Language) -> dict[str, Any] | None:
    """Закреплённая при установке спецификация (у русского — встроенная)."""
    if language.bundled:
        return load_spec(ENGLISH_RUSSIAN_KEY)
    try:
        spec = json.loads(spec_path(data_dir, language).read_text(encoding="utf-8"))
        return validate_spec(spec, model_id=language.model_id, source="en", target=language.code)
    except (OSError, ValueError, TranslationError, KeyError, TypeError):
        return None


def looks_installed(data_dir: Path, language: Language) -> bool:
    """Быстрая проверка по размерам файлов; полную сверку делает загрузка модели."""
    spec = installed_spec(data_dir, language)
    if spec is None:
        return False
    directory = language_model_directory(data_dir, language)
    try:
        return all((directory / item["name"]).stat().st_size == item["bytes"] for item in spec["files"])
    except OSError:
        return False


def reading_translator(data_dir: Path, code: str, cancel: Any = None) -> Any:
    """Переводчик с английского на язык `code` для черновика ТОПа."""
    from app.pilot.translator import EnglishRussianTranslator

    language = load_settings(data_dir)["languages"].get(code)
    if language is None:
        raise TranslationError("Этот язык не добавлен в панели управления.")
    if language.bundled:
        return EnglishRussianTranslator(model_directory(data_dir, ENGLISH_RUSSIAN_KEY), cancel=cancel)
    spec = installed_spec(data_dir, language)
    if spec is None:
        raise TranslationError(f"Модель языка «{language.name}» не установлена. Установите её в панели управления.")
    try:
        return EnglishRussianTranslator(language_model_directory(data_dir, language), cancel=cancel, spec=spec)
    except TranslationError as error:
        raise TranslationError(f"Модель языка «{language.name}» повреждена или неполна: переустановите её "
                               "в панели управления.") from error


# --- Установка ----------------------------------------------------------------------------------


class _Http:
    """Минимальный клиент Hugging Face; в тестах подменяется."""

    def __init__(self) -> None:
        import httpx

        self._client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=15),
                                    headers={"User-Agent": "Trendanalyser/1.0"})

    def json(self, url: str) -> Any:
        response = self._client.get(url)
        if response.status_code == 404:
            raise TranslationError("Модель не найдена на Hugging Face. Проверьте идентификатор.")
        response.raise_for_status()
        return response.json()

    def download(self, url: str, target: Path, limit: int, progress: Callable[[int], None]) -> str:
        digest, received = hashlib.sha256(), 0
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("xb") as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    received += len(chunk)
                    if received > limit:
                        raise TranslationError("Файл модели больше допустимого размера.")
                    digest.update(chunk)
                    output.write(chunk)
                    progress(len(chunk))
                output.flush()
                os.fsync(output.fileno())
        return digest.hexdigest()

    def close(self) -> None:
        self._client.close()


def _file_entries(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        raise TranslationError("Hugging Face вернул неожиданное описание модели.")
    entries = {}
    for sibling in siblings:
        if isinstance(sibling, dict) and isinstance(sibling.get("rfilename"), str):
            entries[sibling["rfilename"]] = sibling
    return entries


def _pinned(entry: dict[str, Any]) -> tuple[str | None, int | None]:
    lfs = _mapping(entry.get("lfs"))
    sha = lfs.get("sha256") if isinstance(lfs.get("sha256"), str) else None
    size = lfs.get("size") if isinstance(lfs.get("size"), int) else entry.get("size")
    return (sha if sha and re.fullmatch(r"[a-f0-9]{64}", sha) else None,
            size if isinstance(size, int) and 0 < size <= MAX_FILE_BYTES else None)


def install_language(data_dir: Path, language: Language, *, http: Any = None,
                     progress: Callable[[int, int], None] | None = None, cancel: Event | None = None) -> dict[str, Any]:
    """Скачать и проверить модель языка, закрепить её ревизию и контрольные суммы."""
    if language.bundled:
        raise TranslationError("Русский встроен: его модель ставит python -m scripts.install_translation_model "
                               "--direction en-ru.")
    own = http is None
    http = http or _Http()
    try:
        info = http.json(f"https://huggingface.co/api/models/{language.model_id}/revision/main?blobs=true")
        revision = info.get("sha") if isinstance(info, dict) else None
        if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
            raise TranslationError("Hugging Face не сообщил ревизию модели.")
        entries = _file_entries(info)
        missing = [remote for remote in REQUIRED.values() if remote not in entries]
        if missing or "config.json" not in entries:
            raise TranslationError("В модели нет файлов ONNX для перевода (нужен экспорт Xenova/opus-mt).")
        config = http.json(f"https://huggingface.co/{language.model_id}/resolve/{revision}/config.json")
        if not isinstance(config, dict) or config.get("model_type") != "marian":
            raise TranslationError("Это не модель перевода Marian (opus-mt).")
        numbers = {name: config.get(name) for name in ("decoder_start_token_id", "eos_token_id", "pad_token_id",
                                                        "vocab_size")}
        if any(type(value) is not int or value < 0 for value in numbers.values()):
            raise TranslationError("В описании модели нет служебных номеров токенов.")
        wanted = {name: remote for name, remote in REQUIRED.items()}
        optional = {name: remote for name, remote in OPTIONAL.items() if remote in entries}
        total = sum((_pinned(entries[remote])[1] or 0) for remote in [*wanted.values(), *optional.values()])
        done = [0]

        def advance(size: int) -> None:
            done[0] += size
            if progress is not None:
                progress(done[0], max(total, done[0]))

        directory = language_model_directory(data_dir, language)
        directory.parent.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        extras: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix=".language-", dir=directory.parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            for group, target in ((wanted, files), (optional, extras)):
                for name, remote in group.items():
                    if cancel is not None and cancel.is_set():
                        raise TranslationError("Установка языка отменена.")
                    expected_sha, expected_size = _pinned(entries[remote])
                    digest = http.download(f"https://huggingface.co/{language.model_id}/resolve/{revision}/{remote}",
                                           staging / name, expected_size or MAX_FILE_BYTES, advance)
                    size = (staging / name).stat().st_size
                    if expected_sha is not None and digest != expected_sha or \
                            expected_size is not None and size != expected_size:
                        raise TranslationError(f"Файл {name} не совпадает с опубликованной контрольной суммой.")
                    target.append({"name": name, "remote": remote, "sha256": digest, "bytes": size})
            spec = {"schema_version": 1, "model_id": language.model_id, "revision": revision,
                    "source_language": "en", "target_language": language.code, "quantization": "int8",
                    "decoder_start_token_id": numbers["decoder_start_token_id"],
                    "eos_token_id": numbers["eos_token_id"], "pad_token_id": numbers["pad_token_id"],
                    "vocabulary_size": numbers["vocab_size"], "max_source_tokens": 512, "max_new_tokens": 200,
                    "beam_width": 2, "normalizer_version": PUBLISHED_NORMALIZER, "files": files,
                    "optional_files": extras, "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            validate_spec(spec, model_id=language.model_id, source="en", target=language.code)
            if directory.exists():
                shutil.rmtree(directory)
            staging.rename(directory)
        _write_json(spec_path(data_dir, language), spec)
        return {"state": "installed", "model_id": language.model_id, "revision": revision,
                "bytes": sum(item["bytes"] for item in [*files, *extras])}
    finally:
        if own:
            http.close()


def remove_language(data_dir: Path, language: Language) -> None:
    if language.bundled:
        raise TranslationError("Русский встроен в программу и не удаляется.")
    shutil.rmtree(language_model_directory(data_dir, language), ignore_errors=True)
    spec_path(data_dir, language).unlink(missing_ok=True)


class LanguageManager:
    """Языки сервиса: список для панели, установка в фоне, язык по умолчанию."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._lock = Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def overview(self) -> dict[str, Any]:
        settings = load_settings(self.data_dir)
        with self._lock:
            jobs = {code: dict(job) for code, job in self._jobs.items()}
        rows = []
        for code, language in settings["languages"].items():
            job = jobs.get(code, {})
            state = job.get("state") if job.get("state") == "installing" else (
                "ready" if looks_installed(self.data_dir, language) else job.get("state") or "not_installed")
            rows.append({"code": code, "name": language.name, "model_id": language.model_id,
                         "bundled": language.bundled, "state": state, "default": code == settings["default"],
                         "progress": job.get("progress"), "message": job.get("message")})
        offered = [{"code": code, "name": name, "model_id": model} for code, (name, model) in CATALOGUE.items()
                   if code not in settings["languages"]]
        return {"languages": rows, "catalogue": offered, "default": settings["default"]}

    def reading(self) -> list[dict[str, str]]:
        """Языки, на которые сайт может перевести ТОП прямо сейчас."""
        settings = load_settings(self.data_dir)
        return [{"code": code, "name": language.name} for code, language in settings["languages"].items()
                if language.bundled or looks_installed(self.data_dir, language)]

    def default(self) -> str:
        return load_settings(self.data_dir)["default"]

    def name(self, code: str) -> str:
        language = load_settings(self.data_dir)["languages"].get(code)
        return language.name if language is not None else code

    def act(self, values: dict[str, Any], *, start: Callable[[Callable[[], None]], Any] | None = None) -> str:
        operation = values.get("op")
        settings = load_settings(self.data_dir)
        languages: dict[str, Language] = settings["languages"]
        if operation == "add":
            code = values.get("code")
            name, model_id = values.get("name"), values.get("model_id")
            if isinstance(code, str) and code in CATALOGUE and (name is None or model_id is None):
                name, model_id = name or CATALOGUE[code][0], model_id or CATALOGUE[code][1]
            language = validate_language(code, name, model_id)
            if language.code in languages and languages[language.code] != language:
                raise ValueError("Этот язык уже добавлен с другой моделью: сначала удалите его.")
            if len(languages) >= MAX_LANGUAGES and language.code not in languages:
                raise ValueError("Добавлено максимальное число языков.")
            languages[language.code] = language
            save_settings(self.data_dir, languages, settings["default"])
            if not language.bundled and not looks_installed(self.data_dir, language):
                self._start_install(language, start)
                return f"Язык «{language.name}» добавлен, модель скачивается."
            return f"Язык «{language.name}» добавлен."
        code = values.get("code")
        if not isinstance(code, str) or code not in languages:
            raise ValueError("Такого языка нет в списке.")
        language = languages[code]
        if operation == "install":
            self._start_install(language, start)
            return f"Скачиваем модель языка «{language.name}»."
        if operation == "default":
            if not (language.bundled or looks_installed(self.data_dir, language)):
                raise ValueError("Сначала дождитесь установки модели этого языка.")
            save_settings(self.data_dir, languages, code)
            return f"Сайт переводит ТОП на язык «{language.name}» по умолчанию."
        if operation == "remove":
            with self._lock:
                if self._jobs.get(code, {}).get("state") == "installing":
                    raise ValueError("Модель ещё скачивается.")
                self._jobs.pop(code, None)
            remove_language(self.data_dir, language)
            del languages[code]
            save_settings(self.data_dir, languages, settings["default"] if settings["default"] != code
                          else DEFAULT_LANGUAGE)
            return f"Язык «{language.name}» удалён."
        raise ValueError("Неизвестное действие с языком.")

    def _start_install(self, language: Language, start: Callable[[Callable[[], None]], Any] | None) -> None:
        with self._lock:
            if self._jobs.get(language.code, {}).get("state") == "installing":
                raise ValueError("Модель этого языка уже скачивается.")
            job: dict[str, Any] = {"state": "installing", "progress": [0, 0], "message": None}
            self._jobs[language.code] = job

        def run() -> None:
            def step(done: int, total: int) -> None:
                with self._lock:
                    job["progress"] = [done, total]
            try:
                install_language(self.data_dir, language, progress=step)
                state, message = "ready", None
            except TranslationError as error:
                state, message = "failed", str(error)[:300]
            except Exception:
                state, message = "failed", "Не удалось скачать модель: проверьте сеть и повторите."
            with self._lock:
                job.update(state=state, message=message)

        (start or (lambda target: Thread(target=target, daemon=True, name="language-install").start()))(run)
