"""Verified local backup and bounded ZIP transport, never a copy of a hot DB.

BackupSession must be entered after both application services have quiesced. It
owns their OS locks until both SQLite snapshots and referenced immutable files
have been captured. SHA-256 detects changed content; it does not authenticate
the archive's author. Restore always targets a new/empty directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import threading
import time
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Literal, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.backend.locking import InstanceLock
from app.identity import validate_data_dir
from app.runtime.budget import BudgetService
from app.runtime.credentials import CREDENTIAL_NAMES, LEGACY_ENVIRONMENT
from app.runtime.files import open_staged_regular
from app.sqlite_runtime import sqlite3

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
# A result package carries one file per archived revision, and a full
# ten-thousand-study corpus arrives as more revisions than studies.
MAX_FILES = 25_000
INTEGRITY_NOTICE: Literal["SHA-256 verifies content integrity, not authorship or scientific truth."] = "SHA-256 verifies content integrity, not authorship or scientific truth."
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,239}\Z")
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_SECRET_KEYS = frozenset({"apikey", "accesstoken", "refreshtoken", "authorization", "password", "secret",
                          "consumersecret", "consumerkey", "iamtoken", "credentials", "credentialstore"}
                         | {re.sub(r"[^a-z0-9]", "", name.lower())
                            for name in (*CREDENTIAL_NAMES, *LEGACY_ENVIRONMENT)})


class ArchiveError(RuntimeError):
    """Safe archive failure: no private paths, payloads or credentials in messages."""


class ArchiveCancellation(Protocol):
    def is_set(self) -> bool: ...


def _check_cancel(cancel: ArchiveCancellation | None) -> None:
    if cancel is not None and cancel.is_set():
        from app.runtime.jobs import TaskCancelled
        raise TaskCancelled()


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1, max_length=240)
    size: int = Field(ge=0, le=MAX_MEMBER_BYTES, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["trendanalizer-backup", "trendanalizer-result", "trendanalizer-signals"]
    version: Literal[1] = 1
    application_id: Literal["org.trendanalizer.pilot.main2"] = "org.trendanalizer.pilot.main2"
    created_at: str = Field(max_length=64)
    integrity_notice: Literal["SHA-256 verifies content integrity, not authorship or scientific truth."] = INTEGRITY_NOTICE
    files: tuple[FileEntry, ...] = Field(min_length=1, max_length=MAX_FILES)


@dataclass(frozen=True)
class BackupResult:
    path: Path
    sha256: str
    files: int
    uncompressed_bytes: int
    rotated: int = 0
    rotation_warning: bool = False


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("Non-finite JSON")


def strict_json(data: bytes) -> Any:
    result = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    # Also reject overflowed finite-looking literals such as 1e9999.
    json.dumps(result, allow_nan=False)
    return result


# Query parameters that carry a credential in a link: the final check refuses
# them, and source metadata is cleaned of exactly the same set on ingestion.
_URL_CREDENTIAL_KEYS = _SECRET_KEYS | {"key", "token", "signature", "xamzsignature", "xamzcredential"}
# A user:password prefix in a string that no URL parser accepts.
_UNPARSED_USERINFO = re.compile(r"^https?://[^/?#@\s]*:[^/?#@\s]*@", re.IGNORECASE)


def _credential_parameter(name: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", name.lower()) in _URL_CREDENTIAL_KEYS


def _public_kiss_article(parsed) -> bool:
    # KISS uses a short decimal `key` as the public article identifier
    # in this one URL shape. Crossref preserves that publisher URL in
    # bibliographic metadata; treating it as an API credential prevents
    # an otherwise complete analysis from passing final verification.
    return (parsed.scheme == "https" and parsed.netloc == "kiss.kstudy.com"
            and parsed.path == "/Detail/Ar" and not parsed.fragment
            and re.fullmatch(r"key=[0-9]{1,12}", parsed.query) is not None)


def strip_link_credentials(value: Any) -> Any:
    """Drop credential-like query parameters from links in a source's own metadata.

    Crossref records carry publishers' full-text links, and some embed an access
    token in the query (measured 24.09.2026: Acta Physico-Chimica Sinica PDF links
    ``?token=…``). No part of the application follows those links, yet one of them
    made the final credential check refuse a whole finished analysis of 10 000
    documents. Removing exactly the parameters that check refuses keeps the
    archive free of third-party tokens while the check itself stays unchanged
    for everything else, including links with user information.
    """
    if isinstance(value, dict):
        return {key: strip_link_credentials(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_link_credentials(item) for item in value]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        if _public_kiss_article(parsed):
            return value
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        kept = [(name, item) for name, item in pairs if not (_credential_parameter(name) and item)]
        if len(kept) != len(pairs):
            return urlunsplit(parsed._replace(query=urlencode(kept)))
    return value


def assert_no_credentials(value: Any) -> None:
    """Reject structured credential fields, not arbitrary words in article text."""
    stack = [value]
    examined = 0
    while stack:
        item = stack.pop()
        examined += 1
        if examined > 2_000_000:
            raise ArchiveError("Структура архива превышает допустимую сложность.")
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                positions = isinstance(nested, list) and all(type(position) is int for position in nested)
                if normalized in _SECRET_KEYS and not positions:
                    raise ArchiveError("Архив содержит поле для ключей доступа и не может быть сохранён или открыт.")
                stack.append(nested)
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str) and item.startswith(("https://", "http://")):
            try:
                parsed = urlsplit(item)
            except ValueError:
                # Source metadata contains strings that only begin like a link,
                # such as an address with a trailing note in the same field. They
                # are not addresses any client can use, so they carry no working
                # credential, and one of them must not destroy a finished
                # analysis. Plain user information is still refused.
                if _UNPARSED_USERINFO.match(item):
                    raise ArchiveError("Архив содержит ссылку с ключом доступа.") from None
                continue
            if _public_kiss_article(parsed):
                continue
            if (parsed.username or parsed.password or any(
                _credential_parameter(key) and value
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            )):
                raise ArchiveError("Архив содержит ссылку с ключом доступа.")


def _safe_name(name: str) -> str:
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{number}" for number in range(1, 10)} | {f"LPT{number}" for number in range(1, 10)}
    if (not _NAME.fullmatch(name) or "\\" in name
            or any(part in {"", ".", ".."} or part.endswith(".") or part.split(".", 1)[0].upper() in reserved
                   for part in name.split("/"))
            or PurePosixPath(name).is_absolute()):
        raise ArchiveError("Архив содержит недопустимый путь.")
    return name


def _digest_file(path: Path, maximum: int = MAX_MEMBER_BYTES, *, cancel: ArchiveCancellation | None = None) -> tuple[int, str]:
    from app.pilot.reports import open_local_regular

    _check_cancel(cancel)
    if path.is_symlink() or not path.is_file():
        raise ArchiveError("Файл архива отсутствует или является ссылкой.")
    digest = hashlib.sha256()
    size = 0
    with open_local_regular(path) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            _check_cancel(cancel)
            size += len(block)
            if size > maximum:
                raise ArchiveError("Архив превышает ограничение размера.")
            digest.update(block)
    _check_cancel(cancel)
    return size, digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _inspect_zip_header(path: Path) -> None:
    """Bound central-directory allocation BEFORE ZipFile builds its entry list."""
    from app.pilot.reports import open_local_regular

    with open_local_regular(path) as handle:
        _inspect_zip_stream(handle)


def _inspect_zip_stream(handle: BinaryIO) -> None:
    size = os.fstat(handle.fileno()).st_size
    handle.seek(0)
    if not 22 <= size <= MAX_ARCHIVE_BYTES or handle.read(4) != b"PK\x03\x04":
        raise ArchiveError("Файл не является допустимым архивом приложения.")
    handle.seek(max(0, size - 65_557))
    tail = handle.read(65_557)
    index = tail.rfind(b"PK\x05\x06")
    if index < 0 or len(tail) - index < 22:
        raise ArchiveError("Архив повреждён.")
    _, disk, central_disk, disk_count, total, central_size, central_offset, comment_size = struct.unpack(
        "<4s4H2LH", tail[index:index + 22]
    )
    eocd_offset = size - len(tail) + index
    if (disk or central_disk or disk_count != total or not 2 <= total <= MAX_FILES + 1
            or central_size > 4 * 1024 * 1024 or central_offset + central_size != eocd_offset
            or index + 22 + comment_size != len(tail)):
        raise ArchiveError("Многотомные, ZIP64 или чрезмерно большие архивы не поддерживаются.")


def write_package(path: Path, files: Mapping[str, Path], *, kind: Literal["trendanalizer-backup", "trendanalizer-result", "trendanalizer-signals"],
                  cancel: ArchiveCancellation | None = None) -> BackupResult:
    """Publish a complete archive exclusively; never overwrite a pre-existing file."""
    from app.pilot.reports import open_local_regular

    _check_cancel(cancel)
    path = Path(path).absolute()
    if path.exists() or path.is_symlink() or not 1 <= len(files) <= MAX_FILES:
        raise ArchiveError("Выберите новое имя файла для архива.")
    entries: list[FileEntry] = []
    names: set[str] = set()
    total = 0
    for name, source in sorted(files.items()):
        _check_cancel(cancel)
        _safe_name(name)
        if name == "manifest.json" or name.casefold() in names:
            raise ArchiveError("Повторяющийся путь в архиве.")
        names.add(name.casefold())
        size, digest = _digest_file(source, cancel=cancel)
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ArchiveError("Архив превышает общий предел распакованных данных.")
        entries.append(FileEntry(path=name, size=size, sha256=digest))
    manifest = PackageManifest(format=kind, created_at=datetime.now(UTC).isoformat(), files=tuple(entries))
    _check_cancel(cancel)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + uuid4().hex + ".zip.tmp")
    try:
        with temporary.open("xb") as raw:
            # The archive contains private checkpoints and reports. Its bytes
            # must stay private while compression is still writing them, not
            # only after the finished ZIP has passed validation.
            if os.name != "nt":
                os.fchmod(raw.fileno(), 0o600)
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=False) as output:
                def put(name: str, data: bytes) -> None:
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    output.writestr(info, data)

                put("manifest.json", manifest.model_dump_json().encode("utf-8"))
                for entry in entries:
                    _check_cancel(cancel)
                    info = zipfile.ZipInfo(entry.path)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    size, stream_digest = 0, hashlib.sha256()
                    with open_local_regular(files[entry.path]) as stream, output.open(info, "w") as target:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            _check_cancel(cancel)
                            size += len(block)
                            if size > entry.size:
                                raise ArchiveError("Файл изменился во время создания архива.")
                            stream_digest.update(block)
                            target.write(block)
                    if size != entry.size or stream_digest.hexdigest() != entry.sha256:
                        raise ArchiveError("Файл изменился во время создания архива.")
            raw.flush()
            os.fsync(raw.fileno())
        _check_cancel(cancel)
        _inspect_zip_header(temporary)
        # Validate the same limits readers enforce, including compression ratios.
        with unpack_package(temporary, expected_kind=kind, cancel=cancel):
            pass
        _, archive_digest = _digest_file(temporary, MAX_ARCHIVE_BYTES, cancel=cancel)
        os.chmod(temporary, 0o600)
        _check_cancel(cancel)
        os.link(temporary, path)  # Atomic no-overwrite publication on APFS/NTFS/local filesystems.
        _fsync_directory(path.parent)
        return BackupResult(path, archive_digest, len(entries), total)
    except ArchiveError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise ArchiveError("Не удалось надёжно сохранить архив. Проверьте место, права и новое имя файла.") from None
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def unpack_package(path: Path, *, expected_kind: Literal["trendanalizer-backup", "trendanalizer-result", "trendanalizer-signals"],
                   cancel: ArchiveCancellation | None = None) -> Iterator[tuple[Path, PackageManifest]]:
    """Extract only verified regular files into a private, disposable quarantine."""
    from app.pilot.reports import open_local_regular

    _check_cancel(cancel)
    temporary = tempfile.TemporaryDirectory(prefix="trendanalyser-quarantine-")
    root = Path(temporary.name)
    deadline = time.monotonic() + 120
    try:
        # Header bounds and ZIP parsing use the same open regular file, so replacing
        # the selected path cannot bypass the central-directory allocation bound.
        with open_local_regular(Path(path)) as raw:
            _inspect_zip_stream(raw)
            _check_cancel(cancel)
            with zipfile.ZipFile(raw, "r") as archive:
                yield from _extract_package(archive, root, expected_kind, deadline, cancel)
    except ArchiveError:
        raise
    except (OSError, ValueError, RuntimeError, RecursionError, zipfile.BadZipFile, struct.error):
        raise ArchiveError("Архив повреждён или имеет неподдерживаемый формат.") from None
    finally:
        temporary.cleanup()


def _extract_package(archive: zipfile.ZipFile, root: Path,
                     expected_kind: Literal["trendanalizer-backup", "trendanalizer-result", "trendanalizer-signals"],
                     deadline: float, cancel: ArchiveCancellation | None) -> Iterator[tuple[Path, PackageManifest]]:
    infos = archive.infolist()
    if len(infos) > MAX_FILES + 1:
        raise ArchiveError("Архив содержит слишком много файлов.")
    names: dict[str, zipfile.ZipInfo] = {}
    normalized: set[str] = set()
    total = 0
    for info in infos:
        _check_cancel(cancel)
        name = _safe_name(info.filename)
        mode = info.external_attr >> 16
        if (info.orig_filename != name or name.casefold() in normalized or info.is_dir()
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or mode & 0o111
                or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or info.file_size > MAX_MEMBER_BYTES
                or (info.file_size > 1024 * 1024 and info.file_size > max(info.compress_size, 1) * 200)):
            raise ArchiveError("Архив содержит ссылки, повторы, исполняемые или чрезмерно сжатые файлы.")
        normalized.add(name.casefold())
        names[name] = info
        total += info.file_size
    if total > MAX_TOTAL_BYTES or "manifest.json" not in names or names["manifest.json"].file_size > 4 * 1024 * 1024:
        raise ArchiveError("Архив превышает ограничения или не содержит описания.")
    _check_cancel(cancel)
    manifest = PackageManifest.model_validate(strict_json(archive.read(names["manifest.json"])))
    _check_cancel(cancel)
    if manifest.format != expected_kind:
        raise ArchiveError("Выбран другой тип архива приложения.")
    expected = {entry.path: entry for entry in manifest.files}
    if len(expected) != len(manifest.files) or set(names) != set(expected) | {"manifest.json"}:
        raise ArchiveError("Состав архива не соответствует его описанию.")
    for name, entry in expected.items():
        _check_cancel(cancel)
        _safe_name(name)
        if names[name].file_size != entry.size:
            raise ArchiveError("Размер файла не соответствует описанию архива.")
        destination = root.joinpath(*name.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        with archive.open(names[name]) as source, destination.open("xb") as target:
            os.chmod(destination, 0o600)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                _check_cancel(cancel)
                count += len(block)
                if count > entry.size or time.monotonic() > deadline:
                    raise ArchiveError("Распаковка превысила допустимый размер или время.")
                digest.update(block)
                target.write(block)
        if count != entry.size or digest.hexdigest() != entry.sha256:
            raise ArchiveError("Контрольная сумма архива не совпадает.")
    _check_cancel(cancel)
    yield root, manifest


class BackupSession:
    """Caller-owned proof of quiescence: both service locks remain held."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = validate_data_dir(data_dir)
        self._locks = [InstanceLock(self.data_dir / name) for name in ("backend.lock", "pilot.lock")]
        self._active = False
        self._owner: tuple[int, int] | None = None

    def __enter__(self) -> BackupSession:
        if self._active or not self.data_dir.is_dir():
            raise ArchiveError("Для резервирования требуется существующий остановленный профиль приложения.")
        try:
            for lock in self._locks:
                lock.acquire()
        except Exception:
            for lock in reversed(self._locks):
                lock.release()
            raise ArchiveError("Остановите сбор и анализ и закройте оба сервиса перед резервированием.") from None
        self._active = True
        self._owner = os.getpid(), threading.get_ident()
        return self

    def check(self) -> None:
        if not self._active or self._owner != (os.getpid(), threading.get_ident()):
            raise ArchiveError("Сеанс резервирования закрыт или принадлежит другому потоку.")

    def __exit__(self, *args: object) -> None:
        self._active = False
        for lock in reversed(self._locks):
            lock.release()


_TABLES = {
    "documents.sqlite3": {"documents", "revisions", "aliases", "jobs", "job_documents", "history_runs", "history_periods", "history_attempts"},
    "pilot.sqlite3": {"analysis_runs", "analysis_checkpoints", "pilot_budget_metadata", "pilot_budget_scopes", "pilot_budget_requests", "pilot_budget_allocations", "pilot_budget_events"},
}

# On-disk layouts for documents v4 / pilot v1. A matching user_version alone
# cannot establish that a restored database supports the application's queries.
_COLUMN_LAYOUTS = {
    "aliases": "alias:TEXT document_key:TEXT",
    "documents": "document_key:TEXT latest_revision:TEXT search_text:TEXT updated_at:TEXT",
    "history_attempts": "period_id:TEXT job_id:TEXT attempt:INTEGER",
    "history_periods": "id:TEXT run_id:TEXT ordinal:INTEGER source:TEXT from_date:TEXT until_date:TEXT full_calendar_period:INTEGER latest_job_id:TEXT parent_id:TEXT granularity:TEXT is_split:INTEGER split_block_reason:TEXT",
    "history_runs": "id:TEXT request_json:TEXT state:TEXT created_at:TEXT updated_at:TEXT error_code:TEXT error_message:TEXT",
    "job_documents": "job_id:TEXT document_key:TEXT revision_id:TEXT observed_at:TEXT search_text:TEXT",
    "jobs": "id:TEXT request_json:TEXT state:TEXT created_at:TEXT updated_at:TEXT scanned:INTEGER stored:INTEGER skipped:INTEGER total_available:INTEGER source_exhausted:INTEGER error_code:TEXT error_message:TEXT contract_version:INTEGER",
    "revisions": "revision_id:TEXT document_key:TEXT payload:TEXT created_at:TEXT",
    "analysis_checkpoints": "run_id:TEXT stage:TEXT attempt:INTEGER digest:TEXT",
    "analysis_runs": "id:TEXT attempt:INTEGER state:TEXT input_json:TEXT stage:TEXT message:TEXT completed:INTEGER total:INTEGER created_at:TEXT updated_at:TEXT error:TEXT",
    "pilot_budget_allocations": "scope_id:TEXT request_id:TEXT",
    "pilot_budget_events": "id:INTEGER subject_id:TEXT action:TEXT created_at:TEXT",
    "pilot_budget_metadata": "singleton:INTEGER schema_version:INTEGER restore_pending:INTEGER",
    "pilot_budget_requests": "request_id:TEXT currency:TEXT state:TEXT reserved_input:INTEGER reserved_output:INTEGER reserved_cost:INTEGER charged_input:INTEGER charged_output:INTEGER charged_cost:INTEGER overrun:INTEGER created_at:TEXT updated_at:TEXT",
    "pilot_budget_scopes": "scope_id:TEXT currency:TEXT max_calls:INTEGER max_input:INTEGER max_output:INTEGER max_cost:INTEGER reconcile:INTEGER",
}


@contextmanager
def _database(path: Path, *, writable: bool = False) -> Iterator[Any]:
    if path.is_symlink() or not path.is_file():
        raise ArchiveError("Одна из баз резервной копии отсутствует.")
    connection = sqlite3.connect(path.as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True, timeout=5, isolation_level=None)
    deadline = time.monotonic() + 60
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    connection.create_collation("UNICODE_NOCASE", lambda a, b: (a.casefold() > b.casefold()) - (a.casefold() < b.casefold()))
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
    finally:
        connection.close()


def _validate_database(path: Path) -> None:
    from app.runtime.jobs import ANTECEDENTS_INDEX_SQL

    expected_version = {"documents.sqlite3": 4, "pilot.sqlite3": 1}[path.name]
    with _database(path) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] != expected_version:
            raise ArchiveError("Версия базы данных резервной копии не поддерживается.")
        schema = connection.execute("SELECT name,type,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        if ({row[0] for row in schema if row[1] == "table"} != _TABLES[path.name]
                or any(row[1] not in {"table", "index"} or "VIRTUAL TABLE" in (row[2] or "").upper() for row in schema)):
            raise ArchiveError("Структура базы данных резервной копии не поддерживается.")
        allowed_schema_calls = {name.upper() for name in _TABLES[path.name]} | {"CHECK", "IN", "KEY", "UNIQUE"}
        # SQLite omits IF NOT EXISTS in sqlite_master. Permit only this exact
        # application-owned expression index, never arbitrary JSON expressions.
        antecedents_index = " ".join(ANTECEDENTS_INDEX_SQL.replace("IF NOT EXISTS ", "", 1).split())
        for name, kind, sql in schema:
            if (path.name == "pilot.sqlite3" and name == "ix_analysis_antecedents_lookup" and kind == "index"
                    and sql and " ".join(sql.split()) == antecedents_index):
                continue
            if sql and (len(sql) > 100_000 or any(
                name.upper() not in allowed_schema_calls for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", sql)
            )):
                raise ArchiveError("База содержит неподдерживаемые выражения в схеме.")
        for table in _TABLES[path.name]:
            columns_info = connection.execute(f"PRAGMA table_xinfo({table})").fetchall()
            if (" ".join(f"{column[1]}:{column[2]}" for column in columns_info) != _COLUMN_LAYOUTS[table]
                    or any(column[6] for column in columns_info)):
                raise ArchiveError("Столбцы базы данных не соответствуют поддерживаемой версии.")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)] or connection.execute("PRAGMA foreign_key_check").fetchone():
            raise ArchiveError("Проверка целостности базы данных не пройдена.")
        columns: tuple[tuple[str, str], ...]
        if path.name == "pilot.sqlite3":
            if connection.execute("SELECT schema_version FROM pilot_budget_metadata WHERE singleton=1").fetchone() != (1,):
                raise ArchiveError("Версия журнала расходов не поддерживается.")
            columns = (("analysis_runs", "input_json"),)
        else:
            columns = (("revisions", "payload"), ("jobs", "request_json"), ("history_runs", "request_json"))
        for table, column in columns:
            for row in connection.execute(f"SELECT {column} FROM {table}"):
                assert_no_credentials(strict_json(row[0].encode("utf-8")))


def _references(value: Any) -> set[str]:
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            revision = item.get("revision_id")
            if isinstance(revision, str) and _DIGEST.fullmatch(revision) and {"source", "text_hash"}.issubset(item):
                found.add(revision)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def _referenced_files(directory: Path) -> dict[str, Path]:
    from app.pilot.archive import DocumentArchive, document_text
    from app.pilot.contracts import DocumentRevisionRef, content_hash
    from app.runtime.jobs import MAX_CHECKPOINT_BYTES
    from app.pilot.multisource.contracts import ArxivImportReceipt, ArxivVersion, SignalContract, SignalProfile
    from app.pilot.multisource.store import SignalStore

    files: dict[str, Path] = {}
    revisions: set[str] = set()
    imported: list[Path] = []
    sensitivity_paths: list[Path] = []
    review_paths: list[Path] = []
    supplemental_refs: list[DocumentRevisionRef] = []
    deadline = time.monotonic() + 120
    indexed_bytes = sum((directory / name).stat().st_size for name in _TABLES)

    class ReplayContext:
        def check_cancelled(self) -> None:
            if time.monotonic() > deadline:
                raise ArchiveError("Проверка сохранённых результатов превысила допустимое время.")
    with _database(directory / "pilot.sqlite3") as connection:
        digests = {row[0] for row in connection.execute("SELECT digest FROM analysis_checkpoints")}
        result_rows = connection.execute("SELECT digest,run_id FROM analysis_checkpoints WHERE stage='result'").fetchall()
    for digest in digests:
        if time.monotonic() > deadline or len(files) >= MAX_FILES:
            raise ArchiveError("Слишком много сохранённых этапов для одной резервной копии.")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ArchiveError("Указатель сохранённого этапа повреждён.")
        path = directory / "checkpoints" / (digest + ".json")
        size, observed = _digest_file(path, MAX_CHECKPOINT_BYTES)
        indexed_bytes += size
        if indexed_bytes > MAX_TOTAL_BYTES:
            raise ArchiveError("Архив превышает ограничение размера.")
        if observed != digest:
            raise ArchiveError("Сохранённый этап повреждён.")
        data = strict_json(path.read_bytes())
        assert_no_credentials(data)
        revisions.update(_references(data))
        files[f"checkpoints/{digest}.json"] = path
    # These append-only catalogues own artifacts that need not belong to a run.
    # Never walk arbitrary directories or silently omit malformed saved entries.
    for catalogue in ("imported-results", "supplemental", "reviews"):
        folder = directory / catalogue
        if not folder.exists():
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise ArchiveError("Каталог сохранённых материалов повреждён.")
        for path in folder.iterdir():
            if time.monotonic() > deadline or len(files) >= MAX_FILES:
                raise ArchiveError("Слишком много материалов для одной резервной копии.")
            if not _DIGEST.fullmatch(path.stem) or path.suffix != ".json":
                raise ArchiveError("Каталог содержит неизвестный формат материала.")
            size, observed = _digest_file(path, MAX_CHECKPOINT_BYTES)
            indexed_bytes += size
            if indexed_bytes > MAX_TOTAL_BYTES:
                raise ArchiveError("Архив превышает ограничение размера.")
            value = strict_json(path.read_bytes())
            if not isinstance(value, dict) or observed != path.stem or content_hash(value) != path.stem:
                raise ArchiveError("Сохранённый материал повреждён.")
            assert_no_credentials(value)
            if catalogue == "imported-results":
                if (set(value) not in ({"result", "assessments", "imported"}, {"result", "assessments", "imported", "reviews"})
                        or value["imported"] is not True):
                    raise ArchiveError("Формат импортированного результата не поддерживается.")
                imported.append(path)
            elif catalogue == "supplemental":
                expected_fields = {"version", "kind", "created_at", "documents", "limitations"}
                if (set(value) not in (expected_fields, expected_fields | {"schema_version"})
                        or value.get("schema_version", 3) != 3
                        or type(value["version"]) is not int or value["version"] != 1
                        or value["kind"] not in {"arxiv", "report"}
                        or not isinstance(value["created_at"], str)
                        or not isinstance(value["documents"], list) or len(value["documents"]) > 1000
                        or not isinstance(value["limitations"], list) or len(value["limitations"]) > 100
                        or any(not isinstance(item, str) or len(item) > 4000 for item in value["limitations"])):
                    raise ArchiveError("Формат дополнительных источников не поддерживается.")
                created = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
                offset = created.utcoffset()
                if offset is None or offset.total_seconds() != 0:
                    raise ArchiveError("Дата дополнительных материалов должна быть указана в UTC.")
                references = [DocumentRevisionRef.model_validate(item) for item in value["documents"]]
                if (len({item.revision_id for item in references}) != len(references)
                        or any(item.source != value["kind"] for item in references)):
                    raise ArchiveError("Дополнительные источники не соответствуют сохранённому материалу.")
                supplemental_refs.extend(references)
            else:
                from app.pilot.review import read_review

                read_review(folder, path.stem)
                review_paths.append(path)
            revisions.update(_references(value))
            files[f"{catalogue}/{path.name}"] = path
    sensitivity_folder = directory / "sensitivity"
    if sensitivity_folder.exists():
        from app.pilot.sensitivity import MAX_JOURNAL_BYTES, load_sensitivity

        if sensitivity_folder.is_symlink() or not sensitivity_folder.is_dir():
            raise ArchiveError("Каталог проверок устойчивости повреждён.")
        for path in sensitivity_folder.iterdir():
            if time.monotonic() > deadline or len(files) >= MAX_FILES:
                raise ArchiveError("Слишком много материалов для одной резервной копии.")
            size, _ = _digest_file(path, MAX_JOURNAL_BYTES)
            indexed_bytes += size
            if indexed_bytes > MAX_TOTAL_BYTES:
                raise ArchiveError("Архив превышает ограничение размера.")
            report = load_sensitivity(path)
            value = report.model_dump(mode="json")
            assert_no_credentials(value)
            revisions.update(_references(value))
            sensitivity_paths.append(path)
            files[f"sensitivity/{path.name}"] = path
    signal_store = SignalStore(directory)
    for folder, suffixes in ((signal_store.objects, {"json"}), (signal_store.raw, {"csv", "xml", "json"})):
        if not folder.exists():
            continue
        if signal_store.root.is_symlink() or folder.is_symlink() or not folder.is_dir():
            raise ArchiveError("Каталог источников сигналов повреждён.")
        for path in folder.iterdir():
            if time.monotonic() > deadline or len(files) >= MAX_FILES:
                raise ArchiveError("Слишком много данных сигналов для резервной копии.")
            if path.suffix == ".tmp":
                continue  # An interrupted, unpublished write is not an artifact.
            if not _DIGEST.fullmatch(path.stem) or path.suffix.removeprefix(".") not in suffixes:
                raise ArchiveError("Каталог сигналов содержит неизвестный файл.")
            size, digest = _digest_file(path)
            indexed_bytes += size
            if indexed_bytes > MAX_TOTAL_BYTES or digest != path.stem:
                raise ArchiveError("Источник сигналов повреждён или превышен размер копии.")
            if folder == signal_store.objects:
                envelope = strict_json(path.read_bytes())
                assert_no_credentials(envelope)
                if not isinstance(envelope, dict) or set(envelope) != {"kind", "value"}:
                    raise ArchiveError("Объект сигнала повреждён.")
                kind = next((candidate for candidate in SignalContract.__subclasses__()
                             if candidate.__name__ == envelope["kind"]), None)
                if kind is None:
                    raise ArchiveError("Неизвестный тип объекта сигнала.")
                value = signal_store.get_object(path.stem, kind)
                if isinstance(value, ArxivImportReceipt):
                    revisions.update(signal_store.get_object(version_hash, ArxivVersion).revision_id
                                     for version_hash in value.version_hashes)
            else:
                signal_store.verify_raw(path.stem, path.suffix.removeprefix("."))
            files[path.relative_to(directory).as_posix()] = path
    for digest, watch_record in signal_store.watch_records():
        if time.monotonic() > deadline or len(files) >= MAX_FILES:
            raise ArchiveError("Слишком много записей наблюдения для резервной копии.")
        profile = signal_store.get_object(watch_record.profile_hash, SignalProfile)
        if (watch_record.recorded_at < profile.knowledge_cutoff or
                watch_record.concept_id not in {item.concept_id for item in profile.findings}):
            raise ArchiveError("Запись наблюдения относится к другой технологии или дате.")
        assert_no_credentials(watch_record.model_dump(mode="json"))
        path = signal_store.watch / (digest + ".json")
        size, observed = _digest_file(path)
        indexed_bytes += size
        if observed != digest or indexed_bytes > MAX_TOTAL_BYTES:
            raise ArchiveError("Запись наблюдения повреждена или архив слишком велик.")
        files[f"signals/watch/{digest}.json"] = path
    archive = DocumentArchive(directory / "revisions")
    for revision in revisions:
        if time.monotonic() > deadline or len(files) >= MAX_FILES:
            raise ArchiveError("Слишком много материалов для одной резервной копии.")
        document = archive.get(revision)
        assert_no_credentials(document.model_dump(mode="json"))
        size, digest = _digest_file(archive.path(revision))
        indexed_bytes += size
        if digest != revision or indexed_bytes > MAX_TOTAL_BYTES:
            raise ArchiveError("Ревизия повреждена или архив превышает ограничение размера.")
        files[f"revisions/{revision[:2]}/{revision}.json"] = archive.path(revision)
    for reference in supplemental_refs:
        ReplayContext().check_cancelled()
        document = archive.get(reference.revision_id)
        if (reference.source != document.source or reference.source_id != document.source_id
                or reference.observed_at != document.fetched_at
                or reference.publication_year != document.publication_year
                or reference.publicly_available_at != document.publication_date
                or reference.text_hash != hashlib.sha256(document_text(document).encode("utf-8")).hexdigest()):
            raise ArchiveError("Ссылка на дополнительный источник не совпадает с его ревизией.")
    # Backups preserve a usable result, not just self-consistent file hashes.
    # The coordinator also supports non-analysis tasks, so replay only envelopes
    # that explicitly identify the v3 analysis format.
    from app.pilot.contracts import AnalysisResult
    from app.pilot.export import verify_result
    from app.pilot.methodology import AssessmentArtifact

    envelopes: list[tuple[Path, str | None]] = [
        (directory / "checkpoints" / (digest + ".json"), run_id) for digest, run_id in result_rows]
    envelopes.extend((path, None) for path in imported)
    for path, run_id in envelopes:
        if time.monotonic() > deadline:
            raise ArchiveError("Проверка сохранённых результатов превысила допустимое время.")
        envelope = strict_json(path.read_bytes())
        if isinstance(envelope, dict) and "result" in envelope:
            payload = envelope["result"]
            raw_artifacts = envelope.get("assessments", [])
            raw_reviews = envelope.get("reviews", [])
            if not isinstance(payload, dict) or payload.get("schema_version") != 3 or not isinstance(raw_artifacts, list) or len(raw_artifacts) > 60:
                raise ArchiveError("Формат сохранённого результата не поддерживается.")
            if not isinstance(raw_reviews, list) or len(raw_reviews) > 60:
                raise ArchiveError("Формат экспертных решений результата не поддерживается.")
            analysis_result = AnalysisResult.model_validate(payload)
            if run_id is not None and analysis_result.run_id != run_id:
                raise ArchiveError("Сохранённый результат относится к другому анализу.")
            arguments: dict[str, Any] = {}
            if raw_reviews:
                from app.pilot.review import ReviewRecord

                arguments["reviews"] = tuple(ReviewRecord.model_validate(item) for item in raw_reviews)
            verify_result(analysis_result, archive,
                          tuple(AssessmentArtifact.model_validate(item) for item in raw_artifacts), context=ReplayContext(), **arguments)
    from app.pilot.multisource.workflow import verify_signal_profile

    def read_base(run_id: str) -> dict:
        if run_id.startswith("import-"):
            from app.pilot.library import ResultLibrary

            return ResultLibrary(directory, archive).read(run_id)
        with _database(directory / "pilot.sqlite3") as connection:
            row = connection.execute("SELECT c.digest FROM analysis_checkpoints c JOIN analysis_runs r "
                "ON r.id=c.run_id WHERE r.id=? AND r.state='succeeded' AND c.stage='result' "
                "AND c.attempt=r.attempt", (run_id,)).fetchone()
        if row is None:
            raise ArchiveError("Связанный научный результат отсутствует в резервной копии.")
        return strict_json((directory / "checkpoints" / (row[0] + ".json")).read_bytes())

    for digest, _ in result_rows:
        envelope = strict_json((directory / "checkpoints" / (digest + ".json")).read_bytes())
        if isinstance(envelope, dict) and envelope.get("kind") == "signals":
            profile = verify_signal_profile(signal_store, archive, envelope["profile_hash"], read_base)
            if envelope.get("profile") != profile.model_dump(mode="json"):
                raise ArchiveError("Профиль сигналов отличается от сохранённого результата.")
    imported_signals = signal_store.root / "imported"
    if imported_signals.exists():
        from app.pilot.library import ResultLibrary
        from app.pilot.multisource.export import read_imported_signal

        if imported_signals.is_symlink() or not imported_signals.is_dir():
            raise ArchiveError("Каталог импортированных сигналов повреждён.")
        for path in imported_signals.iterdir():
            if time.monotonic() > deadline or len(files) >= MAX_FILES:
                raise ArchiveError("Слишком много импортированных сигналов для резервной копии.")
            if path.suffix != ".json" or not _DIGEST.fullmatch(path.stem):
                raise ArchiveError("Каталог импортированных сигналов содержит неизвестный файл.")
            size, observed = _digest_file(path, MAX_CHECKPOINT_BYTES)
            indexed_bytes += size
            if indexed_bytes > MAX_TOTAL_BYTES or observed != path.stem:
                raise ArchiveError("Импортированный профиль сигналов повреждён или слишком велик.")
            value = strict_json(path.read_bytes())
            assert_no_credentials(value)
            profile_hash, _, base, science_id = read_imported_signal(directory, "signal-import-" + path.stem)
            if science_id is not None:
                saved = ResultLibrary(directory, archive).read(science_id)
                if base is None or any(saved.get(key) != base.get(key) for key in ("result", "assessments", "reviews")
                                       if key in base):
                    raise ArchiveError("Научный результат импортированного профиля не совпадает с каталогом.")
            if value.get("profile_hash") != profile_hash:
                raise ArchiveError("Импортированный профиль сигналов изменился.")
            files[f"signals/imported/{path.name}"] = path
    if review_paths:
        from app.pilot.review import read_review

        for path in review_paths:
            if time.monotonic() > deadline:
                raise ArchiveError("Проверка экспертных решений превысила допустимое время.")
            record = read_review(path.parent, path.stem, archive=archive, context=ReplayContext())
            prior = record.decision.supersedes_review_id
            if prior is not None:
                predecessor = read_review(path.parent, prior)
                if predecessor.decision.candidate_id != record.decision.candidate_id:
                    raise ArchiveError("Цепочка экспертных решений содержит другого кандидата.")
    if sensitivity_paths:
        from app.pilot.sensitivity import load_sensitivity, verify_sensitivity

        for path in sensitivity_paths:
            verify_sensitivity(load_sensitivity(path), archive, ReplayContext())
    return files


def _settings_data(settings: Mapping[str, Any] | None, data_dir: Path) -> bytes:
    from app.pilot.settings import PilotSettings, load_settings

    value = PilotSettings.model_validate(dict(settings)) if settings is not None else load_settings(data_dir)
    return value.model_dump_json().encode("utf-8")


def create_backup(session: BackupSession, destination: Path, *, settings: Mapping[str, Any] | None = None, keep: int = 3) -> BackupResult:
    session.check()
    if type(keep) is not int or not 1 <= keep <= 30:
        raise ValueError("Число ротационных копий должно быть от 1 до 30.")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="trendanalyser-backup-") as temporary:
            staged = Path(temporary)
            for name in _TABLES:
                source = session.data_dir / name
                _validate_database(source)
                with _database(source) as live:
                    page_size = live.execute("PRAGMA page_size").fetchone()[0]
                    copy = sqlite3.connect(staged / name, isolation_level=None)
                    try:
                        deadline = time.monotonic() + 60

                        def progress(status: int, remaining: int, total: int, *, until: float = deadline, page_bytes: int = page_size) -> None:
                            if time.monotonic() > until or total * page_bytes > MAX_MEMBER_BYTES:
                                raise ArchiveError("Резервирование базы превысило допустимое время или размер.")

                        live.backup(copy, pages=128, progress=progress, sleep=0.01)
                        def interrupted(until: float = deadline) -> int:
                            return int(time.monotonic() > until)

                        copy.set_progress_handler(interrupted, 10000)
                        copy.execute("PRAGMA journal_mode=DELETE")
                        copy.execute("VACUUM")  # Do not carry deleted/private freelist bytes into a backup.
                    finally:
                        copy.close()
                _validate_database(staged / name)
            # Copy only database-referenced immutable payloads while locks are held.
            for relative, source in _referenced_files(session.data_dir).items():
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            _referenced_files(staged)
            (staged / "settings.json").write_bytes(_settings_data(settings, session.data_dir))
            files = {path.relative_to(staged).as_posix(): path for path in staged.rglob("*") if path.is_file()}
            name = "trendanalyser-backup-main2-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8] + ".zip"
            result = write_package(destination / name, files, kind="trendanalizer-backup")
        try:
            rotated = _rotate(destination, result, keep)
        except (OSError, ArchiveError):
            return BackupResult(result.path, result.sha256, result.files, result.uncompressed_bytes, rotation_warning=True)
        return BackupResult(result.path, result.sha256, result.files, result.uncompressed_bytes, rotated)
    except ArchiveError:
        raise
    except Exception:
        raise ArchiveError("Резервная копия не создана: не пройдена проверка данных или доступа.") from None


def _rotate(directory: Path, created: BackupResult, keep: int) -> int:
    """Only delete exact previously registered bytes created by this feature."""
    registry = directory / ".trendanalizer-backups-main2.json"
    records: list[dict[str, str]] = []
    if registry.is_file() and not registry.is_symlink() and registry.stat().st_size <= 128_000:
        try:
            old = strict_json(registry.read_bytes())
            if isinstance(old, list):
                for item in old[-100:]:
                    if (isinstance(item, dict) and set(item) == {"name", "sha256"}
                            and re.fullmatch(
                                r"(?:trendanalizer|trendanalyser)-backup-main2-\d{8}T\d{6}-[a-f0-9]{8}\.zip",
                                item["name"],
                            )
                            and _DIGEST.fullmatch(item["sha256"])):
                        records.append(item)
        except (ValueError, OSError, TypeError):
            pass
    records.append({"name": created.path.name, "sha256": created.sha256})
    removed = 0
    for item in records[:-keep]:
        path = directory / item["name"]
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_ARCHIVE_BYTES:
            if _digest_file(path, MAX_ARCHIVE_BYTES)[1] == item["sha256"]:
                path.unlink()
                removed += 1
    temporary = registry.with_name("." + uuid4().hex + ".registry.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(json.dumps(records[-keep:]).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, registry)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return removed


def restore_backup(path: Path, new_data_dir: Path) -> Path:
    destination = validate_data_dir(new_data_dir)
    raw_destination = Path(new_data_dir).absolute()
    if raw_destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ArchiveError("Восстановление разрешено только в новый пустой каталог данных.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with unpack_package(path, expected_kind="trendanalizer-backup") as (quarantine, manifest):
            names = {entry.path for entry in manifest.files}
            for name in _TABLES:
                _validate_database(quarantine / name)
            expected = set(_TABLES) | {"settings.json"} | set(_referenced_files(quarantine))
            if names != expected:
                raise ArchiveError("Резервная копия содержит лишние или отсутствующие файлы.")
            _settings_data(strict_json((quarantine / "settings.json").read_bytes()), quarantine)
            staged = Path(tempfile.mkdtemp(prefix=".trendanalyser-restore-", dir=destination.parent))
            for name in names:
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(quarantine / name, target)
                os.chmod(target, 0o600)
            with _database(staged / "pilot.sqlite3", writable=True) as connection:
                connection.execute("PRAGMA synchronous=FULL")
                BudgetService(connection).mark_restored()
                connection.execute("PRAGMA journal_mode=DELETE")
            for file in staged.rglob("*"):
                if file.is_file():
                    with open_staged_regular(file) as handle:
                        os.fsync(handle.fileno())
            _fsync_directory(staged)
            # POSIX rename replaces an empty directory only; a concurrent app
            # start creates locks/DB files and makes this atomic operation fail.
            if destination.exists():
                if any(destination.iterdir()):
                    raise ArchiveError("Каталог назначения уже используется.")
                if os.name == "nt":
                    destination.rmdir()
            os.rename(staged, destination)
            staged = None
            _fsync_directory(destination.parent)
            return destination
    except ArchiveError:
        raise
    except Exception:
        raise ArchiveError("Не удалось подтвердить восстановление: проверьте архив и новый каталог данных.") from None
    finally:
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
