"""Content-addressed local catalog of validated, offline result packages."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from threading import RLock
from uuid import uuid4

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, content_hash
from app.pilot.export import _package_revision_ids, read_result_package, verify_result
from app.pilot.methodology import AssessmentArtifact
from app.runtime.jobs import TaskFailure
from app.pilot.reports import open_local_regular

MAX_IMPORTS = 1000
_PUBLISH_LOCK = RLock()
FileIdentity = tuple[int, int, int, int, int, int]
# A ten-thousand-study result carries a revision reference per document,
# so a saved artifact is several times larger than a 1500-study one.
_MAX_ARTIFACT_BYTES = 50_000_000


def _identity_metadata(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _file_identity(path: Path, cancel=None) -> FileIdentity:
    """Verify bytes on every reuse: filesystem change timestamps may collide."""
    try:
        _check_cancelled(cancel)
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_ARTIFACT_BYTES:
            raise OSError("A regular file is required")
        metadata = _identity_metadata(info)
        digest = hashlib.sha256()
        total = 0
        with open_local_regular(path) as handle:
            descriptor_metadata = _identity_metadata(os.fstat(handle.fileno()))
            # Windows path stat reports creation time as ctime on Python 3.13,
            # while fstat reports ChangeTime. Compare ctime only within each API.
            if descriptor_metadata[:5] != metadata[:5]:
                raise OSError("File changed before opening")
            while True:
                _check_cancelled(cancel)
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_ARTIFACT_BYTES:
                    raise OSError("File exceeds artifact limit")
                digest.update(chunk)
            if total != info.st_size or _identity_metadata(os.fstat(handle.fileno())) != descriptor_metadata:
                raise OSError("File changed during inspection")
        if _identity_metadata(path.stat(follow_symlinks=False)) != metadata:
            raise OSError("File changed during inspection")
        _check_cancelled(cancel)
        return (*metadata[:5], int.from_bytes(digest.digest(), "big"))
    except OSError:
        raise TaskFailure("Сохранённый результат повреждён или отсутствует.") from None


def _check_cancelled(cancel) -> None:
    if cancel is not None and cancel.is_set():
        from app.runtime.jobs import TaskCancelled
        raise TaskCancelled()


@dataclass(frozen=True)
class _VerifiedView:
    digest: str
    identity: FileIdentity
    revisions: tuple[tuple[str, FileIdentity], ...]
    payload: dict
    result: AnalysisResult


def catalogue_paths(directory: Path) -> list[Path]:
    """Bound filesystem enumeration before loading any potentially large JSON."""
    paths: list[Path] = []
    for path in directory.glob("*.json"):
        paths.append(path)
        if len(paths) > MAX_IMPORTS:
            raise TaskFailure("Библиотека превышает ограничение пилота: 1000 импортов.")
    return paths


def _artifact_bytes(payload: dict) -> bytes:
    if not isinstance(payload, dict):
        raise ValueError("Сохраняемые данные должны быть JSON-объектом.")
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > 25_000_000:
        raise TaskFailure("Сохранённые данные превышают ограничение 25 МБ.")
    return data


def _write_artifact(directory: Path, data: bytes, *, cancel=None) -> str:
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise TaskFailure("Каталог сохранённых результатов не должен быть символической ссылкой.")
    target = directory / (digest + ".json")
    temporary = directory / (uuid4().hex + ".tmp")
    try:
        if cancel is not None and cancel.is_set():
            from app.runtime.jobs import TaskCancelled
            raise TaskCancelled()
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if cancel is not None and cancel.is_set():
            from app.runtime.jobs import TaskCancelled
            raise TaskCancelled()
        temporary.replace(target)
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def write_artifact(directory: Path, payload: dict, *, cancel=None) -> str:
    """Publish exactly the bytes hashed, without a caller-mutation window."""
    return _write_artifact(directory, _artifact_bytes(payload), cancel=cancel)


def read_artifact(directory: Path, digest: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise TaskFailure("Некорректный идентификатор сохранённых данных.")
    try:
        with open_local_regular(directory / (digest + ".json")) as handle:
            data = handle.read(25_000_001)
        if len(data) > 25_000_000 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Size limit")
        payload = json.loads(data)
        if not isinstance(payload, dict) or content_hash(payload) != digest:
            raise ValueError("Digest mismatch")
        return payload
    except (OSError, ValueError, RecursionError):
        raise TaskFailure("Сохранённый результат повреждён или отсутствует.") from None


def publish_catalogue_artifact(directory: Path, payload: dict, *, cancel=None) -> str:
    """Serialize the final capacity check and publication across local callers."""
    data = _artifact_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    with _PUBLISH_LOCK:
        paths = catalogue_paths(directory)
        if len(paths) >= MAX_IMPORTS and not any(path.name == digest + ".json" for path in paths):
            raise TaskFailure("Достигнут лимит 1000 импортов. Новый результат не сохранён.")
        return _write_artifact(directory, data, cancel=cancel)


class ResultLibrary:
    def __init__(self, data_dir: Path, archive: DocumentArchive):
        self.directory = data_dir / "imported-results"
        self.archive = archive
        self._cache_lock = RLock()
        # At most one parsed, already bounded (25 MB JSON) result is retained.
        # Summaries contain only query/date/status and are capped by the catalogue.
        self._view: _VerifiedView | None = None
        self._summaries: OrderedDict[str, tuple[FileIdentity, dict]] = OrderedDict()

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._view = None
            self._summaries.clear()

    def import_file(self, path: Path, *, cancel=None) -> dict:
        with read_result_package(path, cancel=cancel) as package:
            # The verified package may also contain older evidence owned by its
            # review records rather than the result's displayed snapshots.
            for digest in package.revision_ids:
                if cancel is not None and cancel.is_set():
                    from app.runtime.jobs import TaskCancelled
                    raise TaskCancelled()
                copied = self.archive.put(package.archive.get(digest))
                if copied.revision_id != digest:
                    raise TaskFailure("Перенесённая ревизия не соответствует исходному результату.")
            verify_result(package.result, self.archive, package.assessments, reviews=package.reviews, cancel=cancel)
            payload = {"result": package.result.model_dump(mode="json"),
                       "assessments": [item.model_dump(mode="json") for item in package.assessments], "imported": True}
            if package.reviews:
                payload["reviews"] = [item.model_dump(mode="json") for item in package.reviews]
            digest = publish_catalogue_artifact(self.directory, payload, cancel=cancel)
        return {"id": "import-" + digest, "payload": payload}

    def save_result(self, result, artifacts, reviews=(), *, cancel=None):
        verify_result(result, self.archive, artifacts, reviews=reviews, cancel=cancel)
        payload = {"result": result.model_dump(mode="json"),
                   "assessments": [item.model_dump(mode="json") for item in artifacts], "imported": True}
        if reviews:
            payload["reviews"] = [item.model_dump(mode="json") for item in reviews]
        digest = publish_catalogue_artifact(self.directory, payload, cancel=cancel)
        return {"id": "import-" + digest, "payload": payload | {"view_id": "import-" + digest}}

    def count(self) -> int:
        return len(catalogue_paths(self.directory))

    def _unchanged_revisions(self, revisions: tuple[tuple[str, FileIdentity], ...], cancel) -> bool:
        for revision_id, identity in revisions:
            _check_cancelled(cancel)
            if _file_identity(self.archive.path(revision_id), cancel) != identity:
                return False
        return True

    def _verified_view(self, run_id: str, *, cancel=None) -> _VerifiedView:
        _check_cancelled(cancel)
        if not run_id.startswith("import-"):
            raise TaskFailure("Некорректный идентификатор импортированного результата.")
        digest = run_id.removeprefix("import-")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise TaskFailure("Некорректный идентификатор импортированного результата.")
        path = self.directory / (digest + ".json")
        identity = _file_identity(path, cancel)
        with self._cache_lock:
            cached = self._view
        if (cached is not None and cached.digest == digest and cached.identity == identity
                and self._unchanged_revisions(cached.revisions, cancel)
                and _file_identity(path, cancel) == identity):
            return cached
        with self._cache_lock:
            self._view = None
        payload = read_artifact(self.directory, digest)
        result = AnalysisResult.model_validate(payload["result"])
        artifacts = tuple(AssessmentArtifact.model_validate(item) for item in payload["assessments"])
        from app.pilot.review import ReviewRecord

        reviews = tuple(ReviewRecord.model_validate(item) for item in payload.get("reviews", []))
        dependencies = []
        # Review-only revisions outside displayed snapshots are part of the
        # verification dependency set too; changing one invalidates the view.
        for revision_id in sorted(_package_revision_ids(result, reviews)):
            _check_cancelled(cancel)
            dependencies.append((revision_id, _file_identity(self.archive.path(revision_id), cancel)))
        revisions = tuple(dependencies)
        verify_result(result, self.archive, artifacts, reviews=reviews, cancel=cancel)
        if _file_identity(path, cancel) != identity or not self._unchanged_revisions(revisions, cancel):
            raise TaskFailure("Файлы результата изменились во время проверки. Откройте результат заново.")
        _check_cancelled(cancel)
        verified = _VerifiedView(digest, identity, revisions, payload, result)
        with self._cache_lock:
            self._view = verified
        return verified

    def read(self, run_id: str, *, cancel=None) -> dict:
        # Callers may add view_id or edit exported data. Never expose cached dicts.
        return deepcopy(self._verified_view(run_id, cancel=cancel).payload)

    def documents(self, run_id: str, offset: int = 0, limit: int = 50, *, cancel=None) -> dict:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid pagination")
        view = self._verified_view(run_id, cancel=cancel)
        snapshot = next((item for item in view.result.snapshots if item.purpose == "discovery"), None)
        if snapshot is None:
            return {"items": [], "total": 0}
        items = []
        for reference in snapshot.documents[offset:offset + limit]:
            _check_cancelled(cancel)
            items.append(self.archive.get(reference.revision_id).model_dump(mode="json"))
        return {"items": items, "total": len(snapshot.documents)}

    def _summary(self, path: Path, identity: FileIdentity) -> dict:
        with self._cache_lock:
            cached = self._summaries.get(path.stem)
            if cached is not None and cached[0] == identity and _file_identity(path) == identity:
                self._summaries.move_to_end(path.stem)
                return dict(cached[1])
        payload = read_artifact(self.directory, path.stem)
        result = AnalysisResult.model_validate(payload["result"])
        if _file_identity(path) != identity:
            raise TaskFailure("Файл результата изменился во время чтения.")
        row = dict(id="import-" + path.stem, state="succeeded", created_at=result.created_at.isoformat(),
                   input_json=json.dumps({"query": result.query_plan.original_query}, ensure_ascii=False),
                   error=None, imported=True)
        with self._cache_lock:
            self._summaries[path.stem] = identity, row
            self._summaries.move_to_end(path.stem)
            while len(self._summaries) > MAX_IMPORTS:
                self._summaries.popitem(last=False)
        return dict(row)

    def list_rows(self, offset: int = 0, limit: int = 50) -> list[dict]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Некорректная страница импортированных результатов.")
        indexed = []
        for path in catalogue_paths(self.directory):
            try:
                info = path.stat(follow_symlinks=False)
                modified = info.st_mtime_ns if stat.S_ISREG(info.st_mode) else 0
            except OSError:
                modified = 0
            # Metadata controls ordering only. Verify bytes for selected rows,
            # without hashing up to 1000 unselected artifacts on every page.
            indexed.append((modified, path.name, path))
        indexed.sort(reverse=True)
        rows = []
        for _, _, path in indexed[offset:offset + limit]:
            identifier = "import-" + path.stem
            try:
                identity = _file_identity(path)
                rows.append(self._summary(path, identity))
            except (TaskFailure, ValueError, KeyError):
                # Keep corruption visible; one damaged import must not hide the library.
                rows.append(dict(id=identifier, state="failed", created_at="", input_json='{"query":"Повреждённый импорт"}',
                                 error="Пакет повреждён. Импортируйте исходный файл заново.", imported=True))
        return rows
