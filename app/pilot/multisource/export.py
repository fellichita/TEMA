"""Rights-gated, self-contained transport for a verified signal profile.

The package is a projection of one immutable root, never a dump of the local
signals directory. SHA-256 establishes integrity, not authorship or a license.
"""

from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult
from app.pilot.export import _package_revision_ids
from app.pilot.library import ResultLibrary, publish_catalogue_artifact, read_artifact
from app.pilot.methodology import AssessmentArtifact
from app.pilot.multisource.contracts import (ArxivImportReceipt, ArxivVersion, CapitalDescription,
    CapitalEvent, CapitalImportReceipt, FundingMetric, QueryProfile, QueryTerm, SearchMetric,
    SearchObservation, SignalContract, SignalProfile, SourceSnapshot, TechnologyAssociation,
    TechnologyConcept, WordstatImportReceipt)
from app.pilot.multisource.store import SignalStore
from app.pilot.multisource.workflow import verify_signal_profile
from app.pilot.review import ReviewRecord
from app.runtime.backup import (ArchiveError, BackupResult, assert_no_credentials, strict_json,
                                unpack_package, write_package)
from app.runtime.jobs import TaskFailure

_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_URL = re.compile(r"https?://[^\s<>\"'\\]+", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
                                r"authorization|password|client[_-]?secret)\s*[:=]")
_MAX_SCAN = 32 * 1024 * 1024
_T = TypeVar("_T", bound=SignalContract)


@dataclass(frozen=True)
class SignalPackage:
    profile_hash: str
    profile: SignalProfile
    store: SignalStore
    archive: DocumentArchive
    base_payload: dict[str, Any] | None
    files: Mapping[str, Path]


def _base_payload(profile: SignalProfile, read_base: Callable[[str], dict]) -> dict[str, Any] | None:
    if profile.base_result_run_id is None:
        return None
    value = read_base(profile.base_result_run_id)
    payload = {key: value[key] for key in ("result", "assessments")}
    if value.get("reviews"):
        payload["reviews"] = value["reviews"]
    assert_no_credentials(payload)
    return payload


def _scan_raw(path: Path, suffix: str) -> None:
    """Reject obvious credentials in source exports before bytes become portable."""
    if path.stat().st_size > _MAX_SCAN:
        raise ArchiveError("Исходный файл слишком велик для проверки безопасной передачи.")
    raw = path.read_bytes()
    if suffix == "json":
        assert_no_credentials(strict_json(raw))
        return
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        if suffix != "csv":
            raise ArchiveError("Текст источника не читается для безопасной передачи.") from None
        text = raw.decode("cp1251")
    if suffix == "csv":
        for delimiter in (";", ",", "\t"):
            headers = next(csv.reader(io.StringIO(text), delimiter=delimiter), ())
            assert_no_credentials({str(header).strip(): "" for header in headers})
    if _SECRET_ASSIGNMENT.search(text):
        raise ArchiveError("Исходная выгрузка содержит поле, похожее на ключ доступа.")
    for url in _URL.findall(text):
        assert_no_credentials({"url": html.unescape(url.rstrip(".,;)"))})


def _receipt(store: SignalStore, digest: str) -> WordstatImportReceipt | ArxivImportReceipt | CapitalImportReceipt:
    for kind in (WordstatImportReceipt, ArxivImportReceipt, CapitalImportReceipt):
        try:
            return cast(WordstatImportReceipt | ArxivImportReceipt | CapitalImportReceipt,
                        store.get_object(digest, kind))
        except TaskFailure:
            continue
    raise ArchiveError("Тип импорта сигналов не распознан.")


def _closure(store: SignalStore, archive: DocumentArchive, profile_hash: str,
             profile: SignalProfile, base_payload: dict[str, Any] | None) -> dict[str, Path]:
    """Construct exact transitive membership while enforcing every source right."""
    files: dict[str, Path] = {}

    def object_file(digest: str, kind: type[_T]) -> _T:
        item = store.get_object(digest, kind)
        assert_no_credentials(item.model_dump(mode="json"))
        files[f"signals/objects/{digest}.json"] = store.objects / (digest + ".json")
        return item

    def raw_file(digest: str, suffix: str) -> None:
        path = store.verify_raw(digest, suffix)
        _scan_raw(path, suffix)
        files[f"signals/raw/{digest}.{suffix}"] = path

    def revision_file(digest: str) -> None:
        archive.get(digest)
        files[f"revisions/{digest[:2]}/{digest}.json"] = archive.path(digest)

    object_file(profile_hash, SignalProfile)
    object_file(profile.query_profile_hash, QueryProfile)
    for digest in profile.import_receipt_hashes:
        receipt = _receipt(store, digest)
        object_file(digest, type(receipt))
        snapshot = object_file(receipt.snapshot_hash, SourceSnapshot)
        assert isinstance(snapshot, SourceSnapshot)
        if snapshot.export_right != "share_allowed" or snapshot.license_ref is None:
            raise ArchiveError("Источник разрешён только для локального использования; перенос профиля запрещён.")
        suffix = ("xml" if isinstance(receipt, ArxivImportReceipt) else
                  "json" if isinstance(receipt, WordstatImportReceipt)
                  and snapshot.adapter_version == "wordstat-api-v2/1" else "csv")
        raw_file(receipt.raw_hash, suffix)
        if isinstance(receipt, WordstatImportReceipt):
            for child in receipt.observation_hashes:
                object_file(child, SearchObservation)
            for child in receipt.term_hashes:
                object_file(child, QueryTerm)
        elif isinstance(receipt, ArxivImportReceipt):
            for child in receipt.version_hashes:
                version = object_file(child, ArxivVersion)
                assert isinstance(version, ArxivVersion)
                revision_file(version.revision_id)
        else:
            if receipt.participant_raw_hash is not None:
                raw_file(receipt.participant_raw_hash, "csv")
            for child in receipt.event_hashes:
                event = object_file(child, CapitalEvent)
                assert isinstance(event, CapitalEvent)
                if event.export_right != "share_allowed" or event.license_ref is None:
                    raise ArchiveError("Финансовое событие разрешено только для локального использования.")
            for child in receipt.description_hashes:
                object_file(child, CapitalDescription)
    for digest in profile.concept_artifact_hashes:
        object_file(digest, TechnologyConcept)
    for digest in profile.association_artifact_hashes:
        object_file(digest, TechnologyAssociation)
    for digest in profile.metric_artifact_hashes:
        try:
            object_file(digest, SearchMetric)
        except TaskFailure:
            object_file(digest, FundingMetric)
    if base_payload is not None:
        result = AnalysisResult.model_validate(base_payload["result"])
        reviews = tuple(ReviewRecord.model_validate(item) for item in base_payload.get("reviews", ()))
        for digest in _package_revision_ids(result, reviews):
            revision_file(digest)
    return files


def export_signal_package(path: Path, store: SignalStore, archive: DocumentArchive,
                          profile_hash: str, read_base: Callable[[str], dict]) -> BackupResult:
    """Save one published and redistributable profile with no unrelated local data."""
    try:
        profile = verify_signal_profile(store, archive, profile_hash, read_base)
        base = _base_payload(profile, read_base)
        with tempfile.TemporaryDirectory(prefix="trendanalyser-signals-write-") as temporary:
            directory = Path(temporary)
            files = _closure(store, archive, profile_hash, profile, base)
            metadata = directory / "metadata.json"
            metadata.write_text(json.dumps({"version": 1, "profile_hash": profile_hash}, sort_keys=True), encoding="utf-8")
            files["metadata.json"] = metadata
            if base is not None:
                base_file = directory / "base.json"
                base_file.write_text(json.dumps(base, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
                files["base.json"] = base_file
            return write_package(path, files, kind="trendanalizer-signals")
    except (ArchiveError, TaskFailure):
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise ArchiveError("Не удалось проверить и сохранить переносимый профиль сигналов.") from None


@contextmanager
def read_signal_package(path: Path) -> Iterator[SignalPackage]:
    """Verify exact membership, source rights, base science and every CAS edge."""
    package_context = unpack_package(path, expected_kind="trendanalizer-signals")
    directory, manifest = package_context.__enter__()
    try:
        try:
            metadata_path = directory / "metadata.json"
            if metadata_path.stat().st_size > 1024:
                raise ValueError("Oversized signals metadata")
            metadata = strict_json(metadata_path.read_bytes())
            if (not isinstance(metadata, dict) or set(metadata) != {"version", "profile_hash"}
                    or type(metadata["version"]) is not int or metadata["version"] != 1
                    or not isinstance(metadata["profile_hash"], str)
                    or not _DIGEST.fullmatch(metadata["profile_hash"])):
                raise ValueError("Invalid signals metadata")
            store = SignalStore(directory)
            archive = DocumentArchive(directory / "revisions")
            raw_base = None
            if (directory / "base.json").exists():
                base_path = directory / "base.json"
                if base_path.stat().st_size > 25_000_000:
                    raise ValueError("Oversized scientific base")
                raw_base = strict_json(base_path.read_bytes())
                if not isinstance(raw_base, dict) or set(raw_base) not in ({"result", "assessments"},
                                                                            {"result", "assessments", "reviews"}):
                    raise ValueError("Invalid scientific base")
                assert_no_credentials(raw_base)
            profile_hash = metadata["profile_hash"]
            profile = store.get_object(profile_hash, SignalProfile)
            if (raw_base is None) != (profile.base_result_run_id is None):
                raise ValueError("Scientific base and profile disagree")

            def read_base(run_id: str) -> dict:
                if raw_base is None or run_id != profile.base_result_run_id:
                    raise TaskFailure("Связанный научный результат отсутствует в пакете.")
                return raw_base

            profile = verify_signal_profile(store, archive, profile_hash, read_base)
            files = _closure(store, archive, profile_hash, profile, raw_base)
            expected = set(files) | {"metadata.json"}
            if raw_base is not None:
                expected.add("base.json")
            if expected != {entry.path for entry in manifest.files}:
                raise ValueError("Signals package membership is not exact")
            package = SignalPackage(profile_hash, profile, store, archive, raw_base, files)
        except ArchiveError:
            raise
        except (TaskFailure, OSError, ValueError, TypeError, KeyError):
            raise ArchiveError("Пакет сигналов не прошёл проверку состава или источников.") from None
        yield package
    finally:
        # Closing with no injected exception prevents unpack_package from
        # translating an error raised by the caller into a corrupt-archive error.
        package_context.__exit__(None, None, None)


def _copy_immutable(source: Path, target: Path) -> None:
    """Publish one already verified member without replacing existing evidence."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return  # The full local transitive verifier checks its bytes below.
    temporary = target.with_name("." + uuid4().hex + ".tmp")
    try:
        with source.open("rb") as readable, temporary.open("xb") as writable:
            shutil.copyfileobj(readable, writable, length=1024 * 1024)
            writable.flush()
            os.fsync(writable.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        if os.name != "nt":
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def import_signal_package(path: Path, data_dir: Path, library: ResultLibrary) -> dict[str, Any]:
    """Verify in quarantine, copy CAS, verify again, then publish one catalogue row."""
    with read_signal_package(path) as package:
        try:
            for name, source in sorted(package.files.items()):
                _copy_immutable(source, data_dir / name)
            store = SignalStore(data_dir)
            archive = DocumentArchive(data_dir / "revisions")

            def read_base(run_id: str) -> dict:
                if package.base_payload is None or run_id != package.profile.base_result_run_id:
                    raise TaskFailure("Связанный научный результат отсутствует в пакете.")
                return package.base_payload

            profile = verify_signal_profile(store, archive, package.profile_hash, read_base)
            if profile != package.profile:
                raise ArchiveError("Перенесённый профиль отличается от проверенного пакета.")
            science_id = None
            if package.base_payload is not None:
                science = package.base_payload
                saved = library.save_result(AnalysisResult.model_validate(science["result"]),
                                            tuple(AssessmentArtifact.model_validate(item)
                                                  for item in science["assessments"]),
                                            reviews=tuple(ReviewRecord.model_validate(item)
                                                          for item in science.get("reviews", ())))
                science_id = saved["id"]
            record = {"profile_hash": package.profile_hash, "base": package.base_payload,
                      "science_id": science_id}
            digest = publish_catalogue_artifact(data_dir / "signals" / "imported", record)
            return {"id": "signal-import-" + digest, "profile_hash": package.profile_hash,
                    "profile": profile.model_dump(mode="json")}
        except (ArchiveError, TaskFailure):
            raise
        except (OSError, ValueError):
            raise ArchiveError("Не удалось перенести и проверить профиль сигналов.") from None


def read_imported_signal(data_dir: Path, run_id: str) -> tuple[str, SignalProfile, dict[str, Any] | None, str | None]:
    if not run_id.startswith("signal-import-"):
        raise TaskFailure("Некорректный идентификатор импортированного профиля.")
    digest = run_id.removeprefix("signal-import-")
    record = read_artifact(data_dir / "signals" / "imported", digest)
    if set(record) != {"profile_hash", "base", "science_id"} or not isinstance(record["profile_hash"], str):
        raise TaskFailure("Каталог импортированных сигналов повреждён.")
    science_id = record["science_id"]
    if science_id is not None and (not isinstance(science_id, str) or not science_id.startswith("import-")):
        raise TaskFailure("Ссылка на научный результат повреждена.")
    base = record["base"]
    if base is not None and (not isinstance(base, dict) or set(base) not in (
            {"result", "assessments"}, {"result", "assessments", "reviews"})):
        raise TaskFailure("Связанный научный результат повреждён.")
    if base is not None:
        assert_no_credentials(base)

    def read_base(base_id: str) -> dict:
        root = store.get_object(record["profile_hash"], SignalProfile)
        if base is None or base_id != root.base_result_run_id:
            raise TaskFailure("Связанный научный результат отсутствует.")
        return base

    store = SignalStore(data_dir)
    archive = DocumentArchive(data_dir / "revisions")
    profile = verify_signal_profile(store, archive, record["profile_hash"], read_base)
    if (base is None) != (science_id is None):
        raise TaskFailure("Каталог научной связи неполон.")
    if science_id is not None and base is not None:
        saved = ResultLibrary(data_dir, archive).read(science_id)
        if (saved.get("result") != base.get("result") or saved.get("assessments") != base.get("assessments")
                or saved.get("reviews", []) != base.get("reviews", [])):
            raise TaskFailure("Научная карточка для повторного использования изменилась.")
    return record["profile_hash"], profile, base, science_id
