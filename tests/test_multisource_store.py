"""The signal CAS must fail closed on altered bytes and never publish a run."""

from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from app.pilot.multisource.contracts import QueryProfile, SourceSnapshot
from app.pilot.multisource.store import SignalStore, object_digest
from app.runtime.jobs import TaskCancelled, TaskFailure
from tests.platform_support import require_symlinks


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
HASH = "a" * 64


def _profile() -> QueryProfile:
    return QueryProfile(profile_id=uuid4(), version=1, original_query="хранение данных в ДНК",
                        definition="Запись цифровой информации в молекулы ДНК")


def test_typed_roundtrip_is_idempotent_and_has_no_published_catalogue(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    profile = _profile()
    digest = store.put_object(profile)
    assert digest == object_digest(profile)
    assert store.put_object(profile) == digest
    assert store.get_object(digest, QueryProfile) == profile
    with pytest.raises(TaskFailure):
        store.get_object(digest, SourceSnapshot)
    assert sorted(path.name for path in store.objects.iterdir()) == [digest + ".json"]
    assert not (store.root / "catalogue.json").exists()


def test_tampered_object_and_symlink_are_rejected(tmp_path: Path) -> None:
    require_symlinks()
    store = SignalStore(tmp_path)
    profile = _profile()
    digest = store.put_object(profile)
    object_path = store.objects / (digest + ".json")
    object_path.write_text("{}", encoding="utf-8")
    with pytest.raises(TaskFailure):
        store.get_object(digest, QueryProfile)
    with pytest.raises(TaskFailure):
        store.put_object(profile)
    object_path.unlink()
    object_path.symlink_to(tmp_path / "missing")
    with pytest.raises(TaskFailure):
        store.get_object(digest, QueryProfile)


def test_raw_import_verifies_bytes_and_rights(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    source = tmp_path / "source.csv"
    source.write_bytes(b"month;count\n2026-08;12\n")
    with pytest.raises(TaskFailure):
        store.put_raw(source, "csv", retention="unknown")
    digest = store.put_raw(source, "csv", retention="local_allowed")
    assert store.put_raw(source, "csv", retention="local_allowed") == digest
    assert store.verify_raw(digest, "csv").read_bytes() == source.read_bytes()
    assert not (store.root / "catalogue.json").exists()
    with pytest.raises(TaskFailure):
        store.verify_raw(digest, "json")


def test_corrupt_existing_raw_is_not_silently_reused(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    source = tmp_path / "source.csv"
    source.write_bytes(b"2026-08;12\n")
    digest = store.put_raw(source, "csv", retention="local_allowed")
    (store.raw / (digest + ".csv")).write_bytes(b"altered")
    with pytest.raises(TaskFailure):
        store.put_raw(source, "csv", retention="local_allowed")


def test_cancel_does_not_create_a_published_profile(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "app")
    source = tmp_path / "source.csv"
    source.write_bytes(b"2026-08;12\n")
    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        store.put_raw(source, "csv", retention="local_allowed", cancel=cancel)
    assert not store.root.exists()


def test_storage_directory_symlink_is_rejected(tmp_path: Path) -> None:
    require_symlinks()
    outside = tmp_path / "outside"
    outside.mkdir()
    data = tmp_path / "app"
    data.mkdir()
    (data / "signals").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TaskFailure):
        SignalStore(data).put_object(_profile())


def test_snapshot_contract_can_be_stored_without_exposing_raw_data(tmp_path: Path) -> None:
    store = SignalStore(tmp_path)
    snapshot = SourceSnapshot(source="wordstat", adapter_version="wordstat-csv/1",
                              request_hash=HASH, query_profile_hash=HASH, observed_at=NOW,
                              available_at=NOW, coverage="partial", comparable=False,
                              extract_hash=HASH, retention="extract_only")
    digest = store.put_object(snapshot)
    assert store.get_object(digest, SourceSnapshot) == snapshot
    assert not store.raw.exists()
