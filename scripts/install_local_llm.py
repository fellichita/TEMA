"""Explicit online installation: python -m scripts.install_local_llm [--model-dir DIR].

Downloads only the pinned instruct model, verifies every byte and publishes a
verified directory atomically. This is the one place that reaches the network
for these weights; the analysis itself never does.

The download is about 1.8 GB, so it is a deliberate request rather than part of
opening the application: either this command, or the button in the analysis
settings, which passes its own `progress` to draw a bar.
"""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from app.pilot import local_llm
from app.runtime.model_resources import frozen
from scripts.model_staging import install_from_staging

CHUNK_BYTES = 4 * 1024 * 1024


def _existing(directory: Path, spec: dict) -> bool:
    if not directory.exists():
        return False
    if directory.is_dir() and not any(directory.iterdir()):
        return False
    try:
        local_llm.verify_artifacts(directory, spec)
    except local_llm.LocalModelError as error:
        raise local_llm.LocalModelError("Непустой каталог модели не перезаписывается. "
                                        "Выберите новый пустой каталог через --model-dir.") from error
    return True


def _checkpoint(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise local_llm.LocalModelError("Загрузка локальной AI-модели отменена.")


def _download(client, spec: dict, item: dict, target: Path, cancel=None, progress=None) -> None:
    # Neither model directories nor user arguments can supply remote URLs.
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{item['remote']}"
    digest, received = hashlib.sha256(), 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length is not None:
            try:
                length = int(length)
            except ValueError as error:
                raise local_llm.LocalModelError(f"Некорректный размер загрузки {item['name']}.") from error
            if length < 0 or length > item["bytes"]:
                raise local_llm.LocalModelError(f"Размер загрузки {item['name']} превышает закреплённый лимит.")
        with target.open("xb") as output:
            for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                _checkpoint(cancel)
                received += len(chunk)
                if received > item["bytes"]:
                    raise local_llm.LocalModelError(f"Размер загрузки {item['name']} превышает закреплённый лимит.")
                digest.update(chunk)
                output.write(chunk)
                if progress is not None:
                    progress(received, item["bytes"])
            output.flush()
            os.fsync(output.fileno())
    if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise local_llm.LocalModelError(f"Загрузка {item['name']} не соответствует закреплённому размеру или SHA256.")


def install(model_dir=None, *, cancel=None, progress=None) -> dict:
    """Return the published directory; an already verified one is left untouched.

    `progress(received, total)` is called for every written chunk with the bytes
    of the whole installation, not of one file, because that is what a caller
    draws. A path that downloads nothing reports the total once, so a bar that
    started never stays empty.
    """
    import httpx

    spec = local_llm.load_spec()
    total = sum(item["bytes"] for item in spec["files"])

    def done() -> None:
        if progress is not None:
            progress(total, total)

    directory = Path(model_dir) if model_dir else local_llm.model_directory()
    _checkpoint(cancel)
    if _existing(directory, spec):
        done()
        return {"path": str(directory), "installed": False, "revision": spec["revision"]}
    if not frozen():
        try:
            if install_from_staging(local_llm.MODEL_KEY, directory, spec, local_llm.verify_artifacts):
                done()
                return {"path": str(directory), "installed": True, "revision": spec["revision"], "source": "staging"}
        except ValueError:
            # These weights are deliberately not part of the native package: 1.8 GB
            # would dominate it, and only a run without a key needs them. With no
            # staged copy the download below is the only source.
            pass
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".local-llm-", dir=directory.parent))
    try:
        with httpx.Client(follow_redirects=True, trust_env=False, timeout=180,
                          limits=httpx.Limits(max_connections=2, max_keepalive_connections=1)) as client:
            written = 0
            for item in spec["files"]:
                def received(count, expected, written=written):
                    if progress is not None:
                        progress(written + count, total)

                _download(client, spec, item, staging / item["name"], cancel, received)
                written += item["bytes"]
        local_llm.verify_artifacts(staging, spec, cancel)
        os.replace(staging, directory)
    except BaseException:
        for path in sorted(staging.glob("*")):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return {"path": str(directory), "installed": True, "revision": spec["revision"], "source": "download"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Каталог установки; по умолчанию каталог профиля")
    arguments = parser.parse_args(argv)
    try:
        result = install(arguments.model_dir)
    except local_llm.LocalModelError as error:
        print(str(error))
        return 2
    print(("Установлена локальная AI-модель: " if result["installed"] else "Модель уже установлена: ")
          + result["path"])
    return 0


if __name__ == "__main__":
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
