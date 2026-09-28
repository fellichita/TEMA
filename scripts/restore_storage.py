"""Restore the published offline corpora and audit artifacts without overwriting local data."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def restore(archive_dir, destination):
    archive_dir, destination = Path(archive_dir), Path(destination).resolve()
    manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Неизвестная версия архива данных.")
    expected = {item["path"]: item for item in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("В манифесте повторены пути.")
    for name, item in expected.items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or parts.parts[0] != "storage":
            raise ValueError("Некорректный путь в манифесте.")
        target = destination.joinpath(*parts.parts)
        if not target.resolve().is_relative_to(destination / "storage"):
            raise ValueError("Путь восстановления выходит за пределы storage.")
        if target.exists() and (not target.is_file() or digest(target) != item["sha256"]):
            raise ValueError(f"Локальный файл отличается от архива: {name}. Выберите пустой каталог.")
    restored, preserved = 0, 0
    with tempfile.TemporaryFile() as combined:
        for part in manifest["parts"]:
            if Path(part["name"]).name != part["name"]:
                raise ValueError("Некорректное имя части архива.")
            source = archive_dir / part["name"]
            if source.stat().st_size != part["bytes"] or digest(source) != part["sha256"]:
                raise ValueError(f"Повреждена часть архива: {part['name']}")
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, combined, CHUNK)
        combined.seek(0)
        seen = set()
        with tarfile.open(fileobj=combined, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError("Неожиданный файл в архиве.")
                item = expected[member.name]
                if member.size != item["bytes"]:
                    raise ValueError(f"Неверный размер файла: {member.name}")
                target = destination.joinpath(*PurePosixPath(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with archive.extractfile(member) as source, tempfile.NamedTemporaryFile(
                            dir=target.parent, delete=False) as output:
                        temporary = Path(output.name)
                        shutil.copyfileobj(source, output, CHUNK)
                    if digest(temporary) != item["sha256"]:
                        raise ValueError(f"Нарушена целостность файла: {member.name}")
                    if target.exists():
                        if digest(target) != item["sha256"]:
                            raise ValueError(f"Файл изменён во время восстановления: {member.name}")
                        preserved += 1
                    else:
                        temporary.replace(target)
                        restored += 1
                    seen.add(member.name)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        if seen != set(expected):
            raise ValueError("Архив содержит не все файлы манифеста.")
    return {"restored": restored, "already_present": preserved, "verified": len(seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "data/offline-storage")
    parser.add_argument("--destination", type=Path, default=ROOT,
                        help="Корень проекта или отдельный каталог для проверки восстановления.")
    args = parser.parse_args()
    try:
        result = restore(args.archive_dir, args.destination)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(2, f"Не удалось восстановить данные: {error}\n")
    print(f"Проверено файлов: {result['verified']}; восстановлено: {result['restored']}; "
          f"уже присутствуют: {result['already_present']}.")


if __name__ == "__main__":
    main()
