"""Bounded, content-addressed files for immutable multisource inputs.

Publication of a completed profile is intentionally a separate workflow step:
merely writing a CAS object or raw file never makes a finding visible to users.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import Event
from typing import TypeVar
from uuid import uuid4

from app.pilot.library import read_artifact
from app.pilot.multisource.contracts import SignalContract, WatchRecord
from app.pilot.reports import open_local_regular
from app.runtime.jobs import TaskCancelled, TaskFailure

MAX_RAW_BYTES = 250 * 1024 * 1024
MAX_OBJECT_BYTES = 25_000_000
CHUNK_BYTES = 1024 * 1024
_SUFFIXES = frozenset({"csv", "xml", "json"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_T = TypeVar("_T", bound=SignalContract)


def _check(cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()


def _directory(path: Path) -> Path:
    """Reject an existing indirection in the application-owned storage path."""
    if path.is_symlink() or path.parent.is_symlink():
        raise TaskFailure("Каталог сигналов содержит недопустимую ссылку.")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise TaskFailure("Каталог сигналов повреждён.")
    return path


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def object_bytes(value: SignalContract) -> bytes:
    """Canonical CAS envelope used by references before and after publication."""
    if type(value) is SignalContract:
        raise TypeError("A concrete signal contract is required")
    validated = type(value).model_validate_json(value.model_dump_json())
    payload = {"kind": type(value).__name__, "value": validated.model_dump(mode="json")}
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_OBJECT_BYTES:
        raise TaskFailure("Сохранённые данные превышают ограничение 25 МБ.")
    return data


def object_digest(value: SignalContract) -> str:
    return hashlib.sha256(object_bytes(value)).hexdigest()


class SignalStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "signals"
        self.objects = self.root / "objects"
        self.raw = self.root / "raw"
        self.watch = self.root / "watch"

    def put_watch_record(self, record: WatchRecord) -> str:
        """Publish an immutable user choice after its CAS object is durable."""
        digest = self.put_object(record)
        directory = _directory(self.watch)
        target = directory / (digest + ".json")
        try:
            os.link(self.objects / (digest + ".json"), target)
        except FileExistsError:
            pass  # A repeated identical choice is idempotent after verification below.
        except OSError:
            raise TaskFailure("Не удалось сохранить выбор наблюдения.") from None
        if target.is_symlink() or read_artifact(directory, digest) != json.loads(object_bytes(record)):
            raise TaskFailure("Запись наблюдения повреждена.")
        _fsync_directory(directory)
        return digest

    def watch_records(self) -> tuple[tuple[str, WatchRecord], ...]:
        if not self.watch.exists():
            return ()
        if self.root.is_symlink() or self.watch.is_symlink() or not self.watch.is_dir():
            raise TaskFailure("Каталог наблюдения повреждён.")
        paths = sorted(self.watch.iterdir())
        if len(paths) > 10_000:
            raise TaskFailure("Слишком много записей наблюдения.")
        records = []
        for path in paths:
            if path.suffix != ".json" or not _HASH.fullmatch(path.stem) or path.is_symlink():
                raise TaskFailure("Каталог наблюдения содержит неизвестный файл.")
            envelope = read_artifact(self.watch, path.stem)
            record = self.get_object(path.stem, WatchRecord)
            if envelope != json.loads(object_bytes(record)):
                raise TaskFailure("Запись наблюдения повреждена.")
            records.append((path.stem, record))
        return tuple(sorted(records, key=lambda pair: (pair[1].recorded_at, pair[0])))

    def put_object(self, value: SignalContract, *, cancel: Event | None = None) -> str:
        _check(cancel)
        data = object_bytes(value)
        digest = hashlib.sha256(data).hexdigest()
        directory = _directory(self.objects)
        temporary = directory / (uuid4().hex + ".tmp")
        target = directory / (digest + ".json")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _check(cancel)
            try:
                os.link(temporary, target)
            except FileExistsError:
                # Immutable CAS: never overwrite a different or corrupt file.
                if self.get_object(digest, type(value)) != value:
                    raise TaskFailure("Сохранённый сигнал повреждён.") from None
            _fsync_directory(directory)
            _check(cancel)
            return digest
        except OSError:
            raise TaskFailure("Не удалось надёжно сохранить сигнал.") from None
        finally:
            temporary.unlink(missing_ok=True)

    def get_object(self, digest: str, expected: type[_T]) -> _T:
        if not isinstance(expected, type) or not issubclass(expected, SignalContract) or expected is SignalContract:
            raise TypeError("A concrete signal contract is required")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise TaskFailure("Некорректный идентификатор сигнала.")
        if self.root.is_symlink() or self.objects.is_symlink():
            raise TaskFailure("Каталог сигналов содержит недопустимую ссылку.")
        if (self.objects / (digest + ".json")).is_symlink():
            raise TaskFailure("Сохранённый сигнал заменён ссылкой.")
        stored = read_artifact(self.objects, digest)
        if set(stored) != {"kind", "value"} or stored["kind"] != expected.__name__:
            raise TaskFailure("Тип сохранённого сигнала не соответствует ожидаемому.")
        try:
            return expected.model_validate(stored["value"])
        except (TypeError, ValueError):
            raise TaskFailure("Сохранённый сигнал повреждён.") from None

    def put_raw(self, path: Path, suffix: str, *, retention: str, cancel: Event | None = None) -> str:
        """Stream a permitted source exactly once, with atomic no-overwrite publish."""
        if suffix not in _SUFFIXES or retention != "local_allowed":
            raise TaskFailure("Этот формат или право локального хранения не подтверждены.")
        _check(cancel)
        directory = _directory(self.raw)
        temporary = directory / (uuid4().hex + ".tmp")
        digest = hashlib.sha256()
        size = 0
        try:
            with open_local_regular(path) as source, temporary.open("xb") as destination:
                original = os.fstat(source.fileno())
                if original.st_size > MAX_RAW_BYTES:
                    raise TaskFailure("Исходный файл превышает ограничение импорта.")
                while True:
                    _check(cancel)
                    block = source.read(CHUNK_BYTES)
                    if not block:
                        break
                    size += len(block)
                    if size > MAX_RAW_BYTES:
                        raise TaskFailure("Исходный файл превышает ограничение импорта.")
                    digest.update(block)
                    destination.write(block)
                if (size != original.st_size or os.fstat(source.fileno()).st_mtime_ns != original.st_mtime_ns
                        or os.fstat(source.fileno()).st_size != original.st_size):
                    raise TaskFailure("Исходный файл изменился во время импорта.")
                destination.flush()
                os.fsync(destination.fileno())
            _check(cancel)
            target = directory / (digest.hexdigest() + "." + suffix)
            try:
                os.link(temporary, target)
            except FileExistsError:
                self.verify_raw(digest.hexdigest(), suffix)
            _fsync_directory(directory)
            _check(cancel)
            return digest.hexdigest()
        except (OSError, ValueError):
            raise TaskFailure("Не удалось надёжно сохранить исходный файл.") from None
        finally:
            temporary.unlink(missing_ok=True)

    def verify_raw(self, digest: str, suffix: str, *, cancel: Event | None = None) -> Path:
        if not isinstance(digest, str) or not _HASH.fullmatch(digest) or suffix not in _SUFFIXES:
            raise TaskFailure("Некорректный идентификатор исходного файла.")
        if self.root.is_symlink() or self.raw.is_symlink():
            raise TaskFailure("Каталог сигналов содержит недопустимую ссылку.")
        path = self.raw / (digest + "." + suffix)
        if path.is_symlink():
            raise TaskFailure("Исходный файл заменён ссылкой.")
        observed = hashlib.sha256()
        size = 0
        try:
            with open_local_regular(path) as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_RAW_BYTES:
                    raise TaskFailure("Исходный файл превышает ограничение импорта.")
                while True:
                    _check(cancel)
                    block = source.read(CHUNK_BYTES)
                    if not block:
                        break
                    observed.update(block)
                    size += len(block)
                    if size > MAX_RAW_BYTES:
                        raise TaskFailure("Исходный файл превышает ограничение импорта.")
                if size != opened.st_size or observed.hexdigest() != digest:
                    raise TaskFailure("Исходный файл повреждён.")
                current = path.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino, current.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
                    raise TaskFailure("Исходный файл заменён во время проверки.")
        except OSError:
            raise TaskFailure("Исходный файл отсутствует или повреждён.") from None
        return path
