"""A portable signal profile has exact, rights-checked transitive membership."""

import shutil
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.backend.repository import Repository
from app.pilot.multisource.export import export_signal_package, read_signal_package
from app.pilot.multisource.contracts import TechnologyConcept
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import SignalStore
from app.pilot.multisource.wordstat import DynamicsMapping, import_wordstat_csv
from app.pilot.export import export_result, read_result_package
from app.pilot.service import PilotService
from app.runtime.backup import ArchiveError, BackupSession, create_backup, restore_backup
from app.runtime.credentials import CredentialStore
from tests.test_multisource_service import _profile
from tests.test_pilot_export import make_result


def _published(tmp_path: Path, *, share: bool, secret_header: bool = False) -> tuple[PilotService, str, Path]:
    data_dir = tmp_path / "app"
    store = SignalStore(data_dir)
    query_hash, concept_hash, _ = _profile(store)
    path = tmp_path / "source.csv"
    extra = ";api_key" if secret_header else ""
    path.write_text(f"Месяц;Запросов;Доля{extra}\n07.2026;100;0,10%{';token' if secret_header else ''}\n"
                    f"08.2026;120;0,12%{';token' if secret_header else ''}\n", encoding="utf-8")
    mapping = DynamicsMapping(date_column="Месяц", count_column="Запросов", share_column="Доля",
                              date_format="MM.YYYY", share_unit="percent", phrase="ДНК память",
                              expected_from=date(2026, 7, 1), expected_to=date(2026, 8, 1))
    receipt = import_wordstat_csv(store, path, query_hash, kind="dynamics", mapping=mapping,
                                  encoding="utf-8-sig", delimiter=";", retention="local_allowed",
                                  export_right="share_allowed" if share else "local_only",
                                  license_ref="own generated test data" if share else None)
    service = PilotService(data_dir, CredentialStore())
    run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt)
    service.coordinator.wait()
    assert service.get(run)["state"] == "succeeded"
    return service, run, data_dir


def test_restricted_source_cannot_be_exported(tmp_path: Path) -> None:
    service, run, _ = _published(tmp_path, share=False)
    try:
        path = tmp_path / "private.trendsignals"
        with pytest.raises(ArchiveError, match="локального"):
            export_signal_package(path, SignalStore(service.data_dir), service.archive,
                                  service.signal_result(run)["profile_hash"], service.result)
        assert not path.exists()
    finally:
        service.close()


def test_portable_export_rejects_unused_secret_column(tmp_path: Path) -> None:
    service, run, _ = _published(tmp_path, share=True, secret_header=True)
    try:
        with pytest.raises(ArchiveError, match="ключ"):
            service.export_signal(run, str(tmp_path / "unsafe.trendsignals"))
    finally:
        service.close()


def test_portable_profile_verifies_after_origin_is_removed(tmp_path: Path) -> None:
    service, run, data_dir = _published(tmp_path, share=True)
    try:
        profile_hash = service.signal_result(run)["profile_hash"]
        path = tmp_path / "shared.trendsignals"
        export_signal_package(path, SignalStore(data_dir), service.archive, profile_hash, service.result)
    finally:
        service.close()
    shutil.rmtree(data_dir)
    with read_signal_package(path) as package:
        assert package.profile_hash == profile_hash
        assert package.profile.findings
        assert all(name.startswith(("signals/", "revisions/")) for name in package.files)
        assert not any("watch" in name for name in package.files)
    recipient = PilotService(tmp_path / "recipient", CredentialStore())
    try:
        imported = recipient.import_signal(str(path))
        assert recipient.signal_result(imported["id"])["profile"] == imported["profile"]
        assert recipient.signal_finding_evidence(imported["id"], imported["profile"]["findings"][0]["finding_id"])
        assert imported["id"] in {row["id"] for row in recipient.list_signal_runs()}
        assert recipient.import_signal(str(path))["id"] == imported["id"]
    finally:
        recipient.close()


def test_signal_package_preserves_consumer_validation_error(tmp_path: Path) -> None:
    service, run, _ = _published(tmp_path, share=True)
    try:
        path = tmp_path / "valid.trendsignals"
        service.export_signal(run, str(path))
    finally:
        service.close()
    with pytest.raises(ValueError, match="consumer rejected cutoff"):
        with read_signal_package(path):
            raise ValueError("consumer rejected cutoff")


def test_extra_member_and_corrupt_child_are_rejected(tmp_path: Path) -> None:
    service, run, data_dir = _published(tmp_path, share=True)
    try:
        path = tmp_path / "valid.trendsignals"
        export_signal_package(path, SignalStore(data_dir), service.archive,
                              service.signal_result(run)["profile_hash"], service.result)
    finally:
        service.close()
    with zipfile.ZipFile(path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    extra = tmp_path / "extra.trendsignals"
    with zipfile.ZipFile(extra, "w") as output:
        for name, data in members.items():
            output.writestr(name, data)
        output.writestr("unrelated.txt", "private")
    with pytest.raises(ArchiveError):
        with read_signal_package(extra):
            pass
    child = next(name for name in members if name.startswith("signals/objects/"))
    damaged = tmp_path / "damaged.trendsignals"
    with zipfile.ZipFile(damaged, "w") as output:
        for name, data in members.items():
            output.writestr(name, b"tampered" if name == child else data)
    with pytest.raises(ArchiveError):
        with read_signal_package(damaged):
            pass


def test_scientific_base_survives_import_and_reexport(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    result, archive, assessments = make_result(data_dir, historical=True)
    original = export_result(tmp_path / "science.trendresult", result, archive, assessments)
    service = PilotService(data_dir, CredentialStore())
    try:
        science_id = service.import_result(str(original.path))["id"]
        label = result.cards[0].candidate.label
        query = build_manual_profile(label, result.cards[0].candidate.definition,
                                     seed_terms=(label,), confirmed_at=datetime.now(timezone.utc))
        store = SignalStore(data_dir)
        query_hash = store.put_object(query)
        concept = TechnologyConcept(concept_id=uuid4(), label=label, definition=query.definition,
                                    identity_status="confirmed", confirmed_at=datetime.now(timezone.utc),
                                    provenance_hashes=(query_hash,))
        concept_hash = store.put_object(concept)
        run = service.start_signals(query_hash, (concept_hash,), base_result_run_id=science_id,
                                    scientific_links=({"concept_id": str(concept.concept_id),
                                                       "candidate_id": result.cards[0].candidate.candidate_id},))
        service.coordinator.wait()
        assert service.get(run)["state"] == "succeeded"
        path = tmp_path / "linked.trendsignals"
        service.export_signal(run, str(path))
    finally:
        service.close()
    shutil.rmtree(data_dir)
    receiver = PilotService(tmp_path / "receiver", CredentialStore())
    try:
        imported = receiver.import_signal(str(path))
        view = receiver.signal_result(imported["id"])
        assert view["profile"]["base_result_hash"] is not None
        assert view["scientific_reuse_run_id"].startswith("import-")
        assert receiver.result(view["scientific_reuse_run_id"])["result"] == result.model_dump(mode="json")
        assert receiver.signal_scenario(imported["id"], "exclude_science")["changes"]
        second = tmp_path / "linked-again.trendsignals"
        receiver.export_signal(imported["id"], str(second))
        with read_signal_package(second) as package:
            assert package.profile_hash == imported["profile_hash"]
        with pytest.raises(ArchiveError):
            with read_result_package(second):
                pass
    finally:
        receiver.close()
    Repository(tmp_path / "receiver" / "documents.sqlite3")
    with BackupSession(tmp_path / "receiver") as session:
        backup = create_backup(session, tmp_path / "private-backups")
    restored = restore_backup(backup.path, tmp_path / "restored")
    reopened = PilotService(restored, CredentialStore())
    try:
        assert reopened.signal_result(imported["id"])["profile"] == view["profile"]
        assert reopened.result(view["scientific_reuse_run_id"])["result"] == result.model_dump(mode="json")
    finally:
        reopened.close()
