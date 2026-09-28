"""Explicit online installation: python -m scripts.install_translation_model [--direction en-ru] [--model-dir DIR].

Downloads only the pinned artifacts of one direction — Russian to English for
the search formulation by default, English to Russian for reading the found
TOP — verifies every byte and publishes a verified directory atomically.
Nothing else in the application ever reaches the network for a model.
"""

import argparse
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

from app.pilot import translator
from scripts.model_staging import install_from_staging

CHUNK_BYTES = 1024 * 1024
MAX_DOWNLOAD_ATTEMPTS = 5


def _existing(directory: Path, spec: dict) -> bool:
    if not directory.exists():
        return False
    if directory.is_dir() and not any(directory.iterdir()):
        return False
    try:
        translator.verify_artifacts(directory, spec)
    except translator.TranslationError as error:
        raise translator.TranslationError("Непустой каталог модели не перезаписывается. "
                                          "Выберите новый пустой каталог через --model-dir.") from error
    return True


def _download(client, spec: dict, item: dict, target: Path) -> None:
    import httpx

    # Neither model directories nor user arguments can supply remote URLs.
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{item['remote']}"
    digest, received = hashlib.sha256(), 0
    with target.open("xb") as output:
        for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
            headers = {"Accept-Encoding": "identity"}
            if received:
                headers["Range"] = f"bytes={received}-"
            try:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise translator.TranslationError(f"Неожиданное сжатие загрузки {item['name']}.")
                    if received and response.status_code == 200:
                        # The server ignored Range. A full response can still be used safely.
                        output.seek(0)
                        output.truncate(0)
                        digest, received = hashlib.sha256(), 0
                    elif received:
                        match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                                             response.headers.get("content-range", ""))
                        if response.status_code != 206 or match is None:
                            raise translator.TranslationError(f"Некорректный ответ на продолжение {item['name']}.")
                        start, end, total = (int(value) for value in match.groups())
                        if start != received or not start <= end < total or total != item["bytes"]:
                            raise translator.TranslationError(f"Некорректный диапазон загрузки {item['name']}.")
                    elif response.status_code != 200:
                        raise translator.TranslationError(f"Некорректный ответ на загрузку {item['name']}.")
                    length = response.headers.get("content-length")
                    if length is not None:
                        if re.fullmatch(r"[0-9]+", length) is None:
                            raise translator.TranslationError(f"Некорректный размер загрузки {item['name']}.")
                        expected_length = end - start + 1 if response.status_code == 206 else item["bytes"]
                        if int(length) > expected_length or (response.status_code == 206
                                                              and int(length) != expected_length):
                            raise translator.TranslationError(
                                f"Размер загрузки {item['name']} превышает закреплённый лимит.")
                    for chunk in response.iter_raw(chunk_size=CHUNK_BYTES):
                        if received + len(chunk) > item["bytes"]:
                            raise translator.TranslationError(
                                f"Размер загрузки {item['name']} превышает закреплённый лимит.")
                        output.write(chunk)
                        received += len(chunk)
                        digest.update(chunk)
            except httpx.TransportError:
                if attempt + 1 == MAX_DOWNLOAD_ATTEMPTS and received != item["bytes"]:
                    raise
            if received == item["bytes"]:
                break
            if attempt + 1 == MAX_DOWNLOAD_ATTEMPTS:
                raise translator.TranslationError(f"Загрузка {item['name']} прервана до получения всех байтов.")
            time.sleep(0.2 * 2**attempt)
        output.flush()
        os.fsync(output.fileno())
    if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise translator.TranslationError(f"Загрузка {item['name']} не соответствует закреплённому размеру или SHA256.")


DIRECTIONS = {"ru-en": translator.MODEL_KEY, "en-ru": translator.ENGLISH_RUSSIAN_KEY}


def _install_optional(directory: Path, spec: dict) -> bool:
    """Add missing optional speed-up files to a verified model directory.

    Each file is downloaded next to its place, verified, then renamed in; the
    required files are never touched. A damaged file under that name is not
    replaced silently.
    """
    import httpx

    missing = [item for item in spec.get("optional_files", ())
               if translator.verify_optional(directory, spec, item["name"]) is None]
    for item in missing:
        if (directory / item["name"]).exists():
            raise translator.TranslationError(f"Файл {item['name']} в каталоге модели не совпадает с закреплённым; "
                                              "удалите его и повторите установку.")
    if not missing:
        return False
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=15)) as client:
        for item in missing:
            partial = directory / f".{item['name']}.{os.getpid()}.part"
            try:
                _download(client, spec, item, partial)
                partial.replace(directory / item["name"])
            finally:
                partial.unlink(missing_ok=True)
    return True


def install(model_dir=None, model_key: str = translator.MODEL_KEY) -> dict:
    import httpx

    directory = Path(translator.model_directory(model_key=model_key) if model_dir is None
                     else model_dir).expanduser().resolve()
    spec = translator.load_spec(model_key)
    try:
        state = _install_required(directory, spec, model_key)
        if _install_optional(directory, spec) and state == "already_present":
            state = "updated"
    except (OSError, httpx.HTTPError) as error:
        raise translator.TranslationError("Не удалось установить модель перевода. Проверьте сеть и права записи; "
                                          "исходные локальные файлы не заменены.") from error
    return {"state": state, "model_id": spec["model_id"], "revision": spec["revision"]}


def _install_required(directory: Path, spec: dict, model_key: str) -> str:
    """Publish the required files as one verified directory; its state name."""
    import httpx

    if _existing(directory, spec):
        return "already_present"
    # A verified copy staged in the project makes a second install a local copy.
    # Only packaged models have a staging place; the reading direction is not packaged.
    try:
        staged = install_from_staging(model_key, directory, spec, translator.verify_artifacts)
    except ValueError:
        staged = False
    if staged:
        return "copied_from_project"
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{model_key}-install-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        # Redirects are required by Hugging Face's official blob storage.
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=15)) as client:
            for item in spec["files"]:
                _download(client, spec, item, staging / item["name"])
        translator.verify_artifacts(staging, spec)
        if _existing(directory, spec):
            return "already_present"
        if directory.exists():
            directory.rmdir()  # Only an empty directory may be removed.
        staging.rename(directory)
    return "installed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), default="ru-en")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = install(args.model_dir, DIRECTIONS[args.direction])
    except (translator.TranslationError, ImportError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Модель перевода проверена: {result['model_id']} @ {result['revision']} ({result['state']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
