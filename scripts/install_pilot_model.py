"""Explicit, bounded installation of the official multilingual model into main2."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit

from app.identity import validate_data_dir
from app.pilot.encoder import EncoderError, checkpoint, load_spec, model_directory, verify_artifacts
from scripts.model_staging import install_from_staging

MAX_REDIRECTS = 5
MAX_DOWNLOAD_SECONDS = 1200


def _allowed_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        return (parsed.scheme == "https" and parsed.port in (None, 443) and not parsed.username
                and not parsed.password and (host == "huggingface.co" or host.endswith(".huggingface.co")
                                            or host.endswith(".hf.co") or host == "hf.co"))
    except ValueError:
        return False


def _existing(directory: Path, spec: dict) -> bool:
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        raise EncoderError("Каталог модели должен быть отдельной локальной папкой.")
    if not any(directory.iterdir()):
        return False
    try:
        verify_artifacts(directory, spec)
    except EncoderError as error:
        raise EncoderError("Непустой каталог модели не перезаписывается. Выберите новую папку.") from error
    return True


def _download(client, spec: dict, item: dict, target: Path, cancel=None, progress=None) -> None:
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{item['remote_path']}"
    deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
    for redirect in range(MAX_REDIRECTS + 1):
        checkpoint(cancel)
        if not _allowed_url(url):
            raise EncoderError("Загрузка модели перенаправлена на недопустимый сервер.")
        with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
            if response.is_redirect:
                if redirect == MAX_REDIRECTS or "location" not in response.headers:
                    raise EncoderError("Слишком много перенаправлений при загрузке модели.")
                url = str(response.url.join(response.headers["location"]))
                continue
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise EncoderError("Неожиданное сжатие файла модели.")
            length = response.headers.get("content-length")
            if length is not None and (not length.isdigit() or int(length) != item["bytes"]):
                raise EncoderError("Размер загружаемой модели не соответствует спецификации.")
            digest, received = hashlib.sha256(), 0
            with target.open("xb") as output:
                for chunk in response.iter_raw(chunk_size=1024 * 1024):
                    checkpoint(cancel)
                    if time.monotonic() > deadline:
                        raise EncoderError("Превышено время загрузки модели.")
                    received += len(chunk)
                    if received > item["bytes"]:
                        raise EncoderError("Модель превышает закреплённый размер.")
                    digest.update(chunk)
                    output.write(chunk)
                    if progress is not None:
                        progress(received, item["bytes"])
                output.flush()
                os.fsync(output.fileno())
            if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise EncoderError("Контрольная сумма скачанной модели не совпадает.")
            return


def install(directory: Path | None = None, *, cancel=None, progress=None) -> dict:
    import httpx

    directory = validate_data_dir(model_directory() if directory is None else Path(directory))
    spec = load_spec()
    total = sum(item["bytes"] for item in spec["files"])
    checkpoint(cancel)
    if _existing(directory, spec):
        if progress is not None:
            progress(total, total)
        return {"state": "already_present", "model_id": spec["model_id"], "revision": spec["revision"]}
    # A verified copy staged in the project makes a second install a local copy.
    if install_from_staging("multilingual-e5-small", directory, spec, verify_artifacts):
        if progress is not None:
            progress(total, total)
        return {"state": "copied_from_project", "model_id": spec["model_id"], "revision": spec["revision"]}
    checkpoint(cancel)
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".multilingual-install-", dir=directory.parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            with httpx.Client(follow_redirects=False, trust_env=False,
                              timeout=httpx.Timeout(60, connect=15)) as client:
                completed = 0
                for item in spec["files"]:
                    def downloaded(received, expected, completed=completed):
                        if progress is not None:
                            progress(completed + received, total)

                    _download(client, spec, item, staging / item["name"], cancel, downloaded)
                    completed += item["bytes"]
            verify_artifacts(staging, spec, cancel)
            checkpoint(cancel)
            if not _existing(directory, spec):
                if directory.exists():
                    directory.rmdir()
                staging.rename(directory)
            else:
                return {"state": "already_present", "model_id": spec["model_id"], "revision": spec["revision"]}
    except (OSError, httpx.HTTPError) as error:
        raise EncoderError("Не удалось установить модель. Проверьте сеть, место на диске и права папки.") from error
    return {"state": "installed", "model_id": spec["model_id"], "revision": spec["revision"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_directory())
    args = parser.parse_args(argv)
    try:
        result = install(args.model_dir)
    except (EncoderError, OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Модель проверена: {result['model_id']} @ {result['revision']} ({result['state']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
