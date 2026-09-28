"""Content-addressed public document revisions; no mutable evidence references."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Protocol
from uuid import uuid4

from app.backend.contracts import DocumentRecord
from app.pilot.contracts import DocumentRevisionRef
from app.runtime.jobs import TaskFailure

MAX_REVISION_BYTES = 3_000_000
# A single verification can touch thousands of revisions. A count alone is
# insufficient: licensed reports may each contain nearly a megabyte of text.
MAX_READING_CACHE_BYTES = 64 * 1024 * 1024


class RevisionSource(Protocol):
    """Read-only access to exact archived revisions of one archive.

    A verification pass needs nothing but this. Accepting the protocol lets
    one operation hand its helpers an ``ArchiveReading`` so a revision is read
    and verified once instead of once per helper, without any of them gaining
    the ability to write.
    """

    @property
    def directory(self) -> Path: ...

    def path(self, revision_id: str) -> Path: ...

    def get(self, revision_id: str) -> DocumentRecord: ...


def document_text(document: DocumentRecord) -> str:
    text = document.title + "\n" + (document.abstract or "")
    if document.source == "report":
        from app.pilot.reports import ReportRecord

        if not isinstance(document, ReportRecord):
            raise TaskFailure("Отчёт должен содержать проверенные данные извлечения текста.")
        text += "\n" + document.full_text
    return text


class DocumentArchive:
    def __init__(self, directory: Path):
        if directory.is_symlink():
            raise TaskFailure("Каталог архива документов не должен быть символической ссылкой.")
        self.directory = directory.resolve()

    def path(self, revision_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", revision_id):
            raise TaskFailure("Некорректный идентификатор ревизии документа.")
        return self.directory / revision_id[:2] / (revision_id + ".json")

    def put(self, document: DocumentRecord, *, study_id: str | None = None) -> DocumentRevisionRef:
        data = json.dumps(document.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(data) > MAX_REVISION_BYTES:
            raise TaskFailure("Документ превышает допустимый размер архива.")
        digest = hashlib.sha256(data).hexdigest()
        path = self.path(digest)
        if self.directory.is_symlink():
            raise TaskFailure("Каталог архива документов не должен быть символической ссылкой.")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise TaskFailure("Каталог архива документов не должен быть символической ссылкой.")
        if os.name != "nt":
            os.chmod(self.directory, 0o700)
            os.chmod(path.parent, 0o700)
        if path.exists():
            self._verify_stored_bytes(path, data)
        else:
            temporary = path.with_suffix("." + uuid4().hex + ".tmp")
            try:
                with temporary.open("xb") as handle:
                    if os.name != "nt":
                        os.fchmod(handle.fileno(), 0o600)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
                if os.name != "nt":
                    descriptor = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
        return DocumentRevisionRef.model_validate(dict(
            revision_id=digest, study_id=study_id or document.document_key,
            source=document.source, source_id=document.source_id,
            text_hash=hashlib.sha256(document_text(document).encode("utf-8")).hexdigest(),
            observed_at=document.fetched_at, publication_year=document.publication_year,
            publicly_available_at=document.publication_date,
        ))

    def _verify_stored_bytes(self, path: Path, data: bytes) -> None:
        """Confirm an already archived revision still holds exactly these bytes.

        The caller has just serialised the document and derived the revision id
        from ``data``, so comparing the stored bytes proves both integrity and
        identity. Reparsing the file through ``get`` would repeat a JSON decode
        and a full model validation for a value already in memory.
        """
        from app.pilot.reports import open_local_regular

        try:
            with open_local_regular(path) as handle:
                stored = handle.read(MAX_REVISION_BYTES + 1)
        except (OSError, ValueError):
            raise TaskFailure("Ревизия документа повреждена или отсутствует.") from None
        if stored != data:
            raise TaskFailure("Ревизия документа повреждена или отсутствует.")

    def get(self, revision_id: str) -> DocumentRecord:
        from app.pilot.reports import open_local_regular

        try:
            path = self.path(revision_id)
            with open_local_regular(path) as handle:
                data = handle.read(MAX_REVISION_BYTES + 1)
            if len(data) > MAX_REVISION_BYTES or hashlib.sha256(data).hexdigest() != revision_id:
                raise ValueError("Invalid revision")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("Invalid revision structure")
            if value.get("source") == "report":
                from app.pilot.reports import ReportRecord

                return ReportRecord.model_validate(value)
            return DocumentRecord.model_validate(value)
        except (OSError, ValueError):
            raise TaskFailure("Ревизия документа повреждена или отсутствует.") from None


class ArchiveReading:
    """One bounded read-through view of an archive, owned by a single operation.

    A revision id is the SHA-256 of that revision's exact bytes, so the first
    read verifies and parses a revision and every later question about the same
    id inside the same operation answers from that verified record. Verification
    passes used to reread the whole corpus several times over: once directly and
    again through publication status.

    The view exists only while its operation runs, so a later analysis never
    trusts a file it did not read itself. ``DocumentArchive.get`` keeps no state
    and still rereads and reverifies on every call, which is what detects a
    replaced file between operations.
    """

    def __init__(self, archive: DocumentArchive, limit: int = 20_000):
        if type(limit) is not int or not 1 <= limit <= 100_000:
            raise TaskFailure("Некорректный предел чтения архива.")
        self._archive = archive
        self._limit = limit
        self._documents: dict[str, DocumentRecord] = {}
        self._cached_bytes = 0

    @property
    def directory(self) -> Path:
        return self._archive.directory

    def path(self, revision_id: str) -> Path:
        return self._archive.path(revision_id)

    def get(self, revision_id: str) -> DocumentRecord:
        document = self._documents.get(revision_id)
        if document is None:
            document = self._archive.get(revision_id)
            # Use the size of the verified on-disk revision as a cheap cache
            # budget estimate. It avoids serializing every parsed record again
            # and keeps large reports from multiplying into gigabytes of RAM.
            if len(self._documents) < self._limit and self._cached_bytes < MAX_READING_CACHE_BYTES:
                try:
                    size = self.path(revision_id).stat().st_size
                except OSError:
                    size = MAX_READING_CACHE_BYTES + 1
                if 0 < size <= MAX_READING_CACHE_BYTES - self._cached_bytes:
                    self._documents[revision_id] = document
                    self._cached_bytes += size
        return document
