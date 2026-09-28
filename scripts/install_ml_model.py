"""Explicit online installation: python -m scripts.install_ml_model [--model-dir DIR]."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from app.ml import local_encoder
from app.ml.contracts import AnalysisInputError
from scripts.model_staging import install_from_staging

CHUNK_BYTES = 1024 * 1024


def _existing(directory, spec):
    if not directory.exists():
        return False
    if directory.is_dir() and not any(directory.iterdir()):
        return False
    try:
        local_encoder.verify_artifacts(directory, spec)
    except AnalysisInputError as error:
        raise AnalysisInputError("Непустой каталог модели не перезаписывается. "
                                 "Выберите новый пустой каталог через --model-dir.") from error
    return True


def _download(client, spec, item, target):
    # Neither model directories nor user arguments can supply remote URLs.
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{item['name']}"
    digest, received = hashlib.sha256(), 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length is not None:
            try:
                length = int(length)
            except ValueError as error:
                raise AnalysisInputError(f"Некорректный размер загрузки {item['name']}.") from error
            if length < 0 or length > item["bytes"]:
                raise AnalysisInputError(f"Размер загрузки {item['name']} превышает закреплённый лимит.")
        with target.open("xb") as output:
            for chunk in response.iter_bytes(chunk_size=CHUNK_BYTES):
                received += len(chunk)
                if received > item["bytes"]:
                    raise AnalysisInputError(f"Размер загрузки {item['name']} превышает закреплённый лимит.")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise AnalysisInputError(f"Загрузка {item['name']} не соответствует закреплённому размеру или SHA256.")


def install(model_dir=None):
    """Download only pinned artifacts, then atomically publish a verified directory."""
    import httpx
    directory = Path(local_encoder.DEFAULT_MODEL_DIR if model_dir is None else model_dir).expanduser().resolve()
    spec = local_encoder.load_spec()
    try:
        if _existing(directory, spec):
            return {"state": "already_present", "model_id": spec["model_id"], "revision": spec["revision"]}
        # A verified copy staged in the project makes a second install a local copy.
        if install_from_staging("e5-small-v2", directory, spec, local_encoder.verify_artifacts):
            return {"state": "copied_from_project", "model_id": spec["model_id"], "revision": spec["revision"]}
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".e5-install-", dir=directory.parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            # Redirects are required by Hugging Face's official blob storage.
            # Read timeout bounds an idle connection; byte limits bound each download.
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=15)) as client:
                for item in spec["files"]:
                    _download(client, spec, item, staging / item["name"])
            local_encoder.verify_artifacts(staging, spec)
            if _existing(directory, spec):
                return {"state": "already_present", "model_id": spec["model_id"], "revision": spec["revision"]}
            if directory.exists():
                directory.rmdir()  # Only an empty directory may be removed.
            staging.rename(directory)
    except (OSError, httpx.HTTPError) as error:
        raise AnalysisInputError("Не удалось установить локальную модель. Проверьте сеть и права записи; "
                                 "исходные локальные файлы не заменены.") from error
    return {"state": "installed", "model_id": spec["model_id"], "revision": spec["revision"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=local_encoder.DEFAULT_MODEL_DIR)
    args = parser.parse_args(argv)
    try:
        result = install(args.model_dir)
    except (AnalysisInputError, ImportError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Локальная модель проверена: {result['model_id']} @ {result['revision']} ({result['state']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
