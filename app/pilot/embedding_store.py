"""Bounded, disposable per-text vectors; never part of scientific storage.

One process lock covers inference and publication. SQLite uses a fixed schema,
raw float32 blobs and per-key hashes, without text, pickle, extensions or WAL.
The quota reserves database growth AND rollback-journal space before writing;
freelist pages count as physical bytes and are reused within max_page_count.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import time

from app.backend.errors import BackendError
from app.backend.locking import InstanceLock
from app.pilot.encoder import Cancellation, EncoderError, checkpoint, validate_vectors
from app.sqlite_runtime import sqlite3

DEFAULT_CACHE_BYTES = 512 * 1024 * 1024
_PAGE_BYTES = 4096
_DB_NAME = "text-vectors-v1.sqlite3"
_AGGREGATE = re.compile(r"[a-f0-9]{64}\.(?:npy|json)\Z")
_TABLE_SQL = "CREATE TABLE vectors(key TEXT PRIMARY KEY,payload BLOB NOT NULL,sha256 TEXT NOT NULL,created INTEGER NOT NULL)"
_INDEX_SQL = "CREATE INDEX vector_age ON vectors(created,key)"
_SCHEMA = {("table", "vectors", "vectors", _TABLE_SQL), ("index", "vector_age", "vectors", _INDEX_SQL),
           ("index", "sqlite_autoindex_vectors_1", "vectors", None)}


def _identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("Cache requires regular files")
    return info.st_dev, info.st_ino


class DerivedCache:
    def __init__(self, directory: Path, max_bytes: int, cancel: Cancellation | None):
        if type(max_bytes) is not int or not 128 * 1024 <= max_bytes <= 4 * 1024**3:
            raise EncoderError("Лимит кеша должен быть от 128 КиБ до 4 ГиБ.")
        self.directory, self.max_bytes, self.cancel = directory, max_bytes, cancel
        self.db_limit = (max_bytes // 4 // _PAGE_BYTES) * _PAGE_BYTES
        # Conservative allowance includes the blob, both indexes and fragmentation.
        self.max_rows = max(1, (self.db_limit - 4 * _PAGE_BYTES) // _PAGE_BYTES)
        self.path = directory / _DB_NAME
        self.lock = InstanceLock(directory / ".embedding.lock")

    def __enter__(self):
        checkpoint(self.cancel)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (self.lock.path.is_symlink() or (self.lock.path.exists()
                and not stat.S_ISREG(self.lock.path.stat(follow_symlinks=False).st_mode))):
            raise EncoderError("Повреждена блокировка кеша эмбеддингов.")
        deadline = time.monotonic() + 5
        while True:
            checkpoint(self.cancel)
            try:
                self.lock.acquire()
                break
            except BackendError:
                if time.monotonic() >= deadline:
                    raise EncoderError("Кеш эмбеддингов занят другим расчётом. Повторите после его завершения.") from None
                time.sleep(0.05)
        try:
            if os.name != "nt":
                os.chmod(self.lock.path, 0o600)
            self._remove_staging()
            if (self.path.is_symlink() or (self.path.exists() and self.path.stat().st_size > self.db_limit)
                    or self._has_sidecar()):
                self._discard()
            self.reserve(0)
        except BaseException:
            self.lock.release()
            raise
        return self

    def __exit__(self, *args):
        self.lock.release()

    def _remove_staging(self):
        for path in self.directory.iterdir():
            checkpoint(self.cancel)
            if path.name.startswith(".embedding-") and path.is_dir() and not path.is_symlink():
                for name in ("vectors.npy", "manifest.json"):
                    (path / name).unlink(missing_ok=True)
                # Never recurse into unexpected user data or follow a symlink.
                try:
                    path.rmdir()
                except OSError:
                    raise EncoderError("Не удалось очистить временный кеш эмбеддингов.") from None

    def reserve(self, additional: int):
        """Evict old aggregates before allocating; never count only live DB rows."""
        total = 0
        candidates = []
        for path in self.directory.iterdir():
            checkpoint(self.cancel)
            info = path.stat(follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                if _AGGREGATE.fullmatch(path.name):
                    candidates.append((info.st_mtime_ns, path.name, path, info.st_size))
        for _, _, path, size in sorted(candidates):
            if total + additional <= self.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                # In particular, Windows may retain a caller's read-only mmap.
                continue
            total -= size
        if total + additional > self.max_bytes:
            raise EncoderError("Кеш достиг лимита места. Закройте открытые результаты и повторите расчёт.")

    @property
    def store_exists(self) -> bool:
        return self.path.is_file() and not self.path.is_symlink()

    def _discard(self):
        for suffix in ("-wal", "-shm", "-journal", ""):
            try:
                self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
            except OSError:
                raise EncoderError("Не удалось заменить повреждённый производный кеш.") from None

    def _has_sidecar(self) -> bool:
        return any(path.exists() or path.is_symlink() for path in (
            self.path.with_name(self.path.name + suffix) for suffix in ("-wal", "-shm", "-journal")))

    def _connect(self):
        checkpoint(self.cancel)
        if (self.path.is_symlink() or (self.path.exists() and self.path.stat().st_size > self.db_limit)
                or self._has_sidecar()):
            self._discard()
        current = self.path.stat().st_size if self.path.exists() else 0
        # Only pre-existing pages can enter the rollback journal. Also reserve
        # initialization pages for a new DB, and all possible DB growth. This
        # allows a large verified legacy mmap to seed an initially empty store.
        self.reserve(self.db_limit - current + 2 * max(current, 4 * _PAGE_BYTES))
        connection = None
        try:
            existing = self.path.exists()
            if existing:
                previous = _identity(self.path)
            else:
                # SQLite must never create through a raced symlink. Establish
                # an exclusive regular file, then open it only in mode=rw.
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
                try:
                    info = os.fstat(descriptor)
                    previous = info.st_dev, info.st_ino
                finally:
                    os.close(descriptor)
            connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=0.2, isolation_level=None)
            if _identity(self.path) != previous:
                raise ValueError("Replaced file before opening")
            if os.name != "nt":
                os.chmod(self.path, 0o600)
            if hasattr(connection, "setlimit"):
                connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
                connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 4096)
            deadline = time.monotonic() + 30
            connection.set_progress_handler(
                lambda: int((self.cancel is not None and self.cancel.is_set()) or time.monotonic() > deadline), 1000)
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA cell_size_check=ON")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA cache_size=-2048")
            if existing:
                schema = set(connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master LIMIT 4"))
                if (schema != _SCHEMA or connection.execute("PRAGMA user_version").fetchone()[0] != 1
                        or connection.execute("PRAGMA page_size").fetchone()[0] != _PAGE_BYTES
                        or _identity(self.path) != previous):
                    raise ValueError("Unknown derived schema or replaced file")
            else:
                connection.execute(f"PRAGMA page_size={_PAGE_BYTES}")
                connection.execute(_TABLE_SQL)
                connection.execute(_INDEX_SQL)
                connection.execute("PRAGMA user_version=1")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA journal_size_limit=0")
            connection.execute(f"PRAGMA max_page_count={self.db_limit // _PAGE_BYTES}")
            return connection, _identity(self.path)
        except (sqlite3.Error, OSError, ValueError):
            if connection is not None:
                connection.close()
            checkpoint(self.cancel)
            self._discard()
            return None, None

    def read(self, keys: list[str]) -> dict:
        import numpy as np

        connection, identity = self._connect()
        if connection is None:
            return {}
        result = {}
        try:
            for offset in range(0, len(keys), 256):
                checkpoint(self.cancel)
                selected = keys[offset:offset + 256]
                rows = connection.execute("SELECT key,payload,sha256 FROM vectors WHERE key IN (" +
                                          ",".join("?" for _ in selected) + ") AND typeof(payload)='blob' "
                                          "AND length(payload)=1536 AND typeof(sha256)='text' AND length(sha256)=64", selected)
                for key, data, digest in rows:
                    if (not isinstance(data, bytes) or len(data) != 384 * 4
                            or hashlib.sha256(key.encode("ascii") + data).hexdigest() != digest):
                        continue
                    vector = np.frombuffer(data, dtype="<f4").reshape(1, 384)
                    try:
                        validate_vectors(vector, 1, cancel=self.cancel)
                    except EncoderError:
                        continue
                    result[key] = vector[0]
            if _identity(self.path) != identity:
                raise ValueError("Replaced cache file")
            return result
        except (sqlite3.Error, OSError, ValueError, TypeError, UnicodeError):
            checkpoint(self.cancel)
            return {}
        finally:
            connection.close()

    def write(self, values: list[tuple[str, bytes]]) -> None:
        if not values:
            return
        connection, identity = self._connect()
        if connection is None:
            connection, identity = self._connect()
        if connection is None:
            return  # The aggregate can still be published after recomputation.
        try:
            values = values[-self.max_rows:]
            count = connection.execute("SELECT count(*) FROM vectors").fetchone()[0]
            connection.execute("BEGIN IMMEDIATE")
            remove = max(0, count + len(values) - self.max_rows)
            connection.execute("DELETE FROM vectors WHERE key IN (SELECT key FROM vectors ORDER BY created,key LIMIT ?)", (remove,))
            created = time.time_ns()
            for offset in range(0, len(values), 256):
                checkpoint(self.cancel)
                connection.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)", [
                    (key, data, hashlib.sha256(key.encode("ascii") + data).hexdigest(), created)
                    for key, data in values[offset:offset + 256]])
            checkpoint(self.cancel)
            if _identity(self.path) != identity:
                raise ValueError("Replaced cache file")
            connection.commit()
        except (sqlite3.Error, OSError, ValueError):
            connection.rollback()
            checkpoint(self.cancel)
            connection.close()
            connection = None
            self._discard()
        finally:
            if connection is not None:
                connection.close()
