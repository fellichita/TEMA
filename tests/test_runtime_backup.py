"""Actual SQLite backups plus malicious archive and overwrite boundaries."""

import json
import os
import stat
import time
import zipfile

import pytest

from app.backend.contracts import DocumentRecord, SearchRequest, SourcePage
from app.backend.locking import InstanceLock
from app.backend.repository import Repository
from app.pilot.archive import DocumentArchive
from app.runtime.backup import (
    ArchiveError, BackupSession, assert_no_credentials, create_backup, restore_backup, unpack_package, write_package,
)
from app.runtime.budget import BudgetLimits, BudgetService, BudgetStateError, RequestAllowance
from app.runtime.credentials import CREDENTIAL_NAMES, LEGACY_ENVIRONMENT
from app.runtime.jobs import ANTECEDENTS_INDEX_SQL, Coordinator
from app.sqlite_runtime import sqlite3


@pytest.mark.parametrize("name", sorted(CREDENTIAL_NAMES | LEGACY_ENVIRONMENT.keys()))
@pytest.mark.parametrize("location", ["nested_field", "query_parameter"])
def test_archive_rejects_every_supported_credential_name_and_environment_alias(name, location):
    private = "synthetic-archive-marker-not-a-real-key"
    if location == "nested_field":
        payload = {"metadata": [{name: private}]}
    else:
        payload = {"metadata": {"link": f"https://example.com/article?{name}={private}"}}
    with pytest.raises(ArchiveError) as caught:
        assert_no_credentials(payload)
    assert private not in str(caught.value)


@pytest.mark.parametrize("link", [
    "https://products.beyondgravity.com [retrieved",   # настоящая запись источника
    "https://example.org/a[b]c",
    "http://[unclosed",
])
def test_a_link_no_parser_accepts_does_not_destroy_the_archive(link):
    """One malformed address in third-party metadata must not fail a whole result."""
    assert_no_credentials({"metadata": {"note": link}})


@pytest.mark.parametrize("link", [
    "https://user:secret@example.org/a",
    "https://user:secret@[unclosed",
])
def test_user_information_is_refused_even_when_the_link_cannot_be_parsed(link):
    with pytest.raises(ArchiveError):
        assert_no_credentials({"metadata": {"note": link}})


def test_credential_filter_preserves_public_prose_and_source_word_positions():
    payload = {"title": "How Yandex and OpenAI protect API keys", "source": "openai",
               "abstract": "Do not share a yandex_api_key or OPENAI_API_KEY.",
               "abstract_inverted_index": {name: [index] for index, name in enumerate(
                   sorted(CREDENTIAL_NAMES | LEGACY_ENVIRONMENT.keys()))}}
    assert_no_credentials(payload)


@pytest.mark.parametrize("article_id", ["1234567", "123456789012"])
def test_public_kiss_article_identifier_is_not_an_access_key(article_id):
    assert_no_credentials({"raw_metadata": {"resource": {"primary": {
        "URL": f"https://kiss.kstudy.com/Detail/Ar?key={article_id}"
    }}}})


@pytest.mark.parametrize("url", [
    "https://kiss.kstudy.com/Detail/Ar?key=private-secret",
    "https://other.example/Detail/Ar?key=1234567",
    "https://kiss.kstudy.com/Detail/Other?key=1234567",
    "https://kiss.kstudy.com/Detail/Ar?key=1234567&token=private-secret",
    "https://kiss.kstudy.com/Detail/Ar?key=1234567&api_key=private-secret",
    "https://user:private-secret@kiss.kstudy.com/Detail/Ar?key=1234567",
    "http://kiss.kstudy.com/Detail/Ar?key=1234567",
    "https://kiss.kstudy.com:8443/Detail/Ar?key=1234567",
    "https://kiss.kstudy.com/Detail/Ar?key=1234567#private",
    "https://kiss.kstudy.com/Detail/Ar?key=1234567890123",
])
def test_public_kiss_article_exception_remains_narrow(url):
    with pytest.raises(ArchiveError):
        assert_no_credentials({"raw_metadata": {"resource": {"primary": {"URL": url}}}})


@pytest.fixture
def profile(tmp_path):
    directory = tmp_path / "profile"
    repository = Repository(directory / "documents.sqlite3")
    document = DocumentRecord(source="openalex", source_id="W1", title="Adaptive robot control",
        abstract="Independent research demonstrates faster robot control.", publication_year=2025,
        date_precision="year", url="https://openalex.org/W1")
    job = repository.create_job(SearchRequest(topic="robotics", source="openalex"))
    repository.start_job(job.id)
    repository.ingest_page(job.id, SourcePage(documents=(document,), scanned=1, exhausted=True))
    repository.finish_job(job.id, "succeeded")
    archive = DocumentArchive(directory / "revisions")
    reference = archive.put(document)

    def process(context, payload):
        context.checkpoint("retrieval", {"references": [reference.model_dump(mode="json")]})
        return {"retained": reference.model_dump(mode="json")}

    coordinator = Coordinator(directory, process)
    run_id = coordinator.submit({"query": "robotics"})
    try:
        deadline = time.monotonic() + 5
        while coordinator.get(run_id)["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert coordinator.get(run_id)["state"] == "succeeded"
    finally:
        coordinator.close()
    connection = sqlite3.connect(directory / "pilot.sqlite3", isolation_level=None)
    try:
        budget = BudgetService(connection)
        budget.create_scope("day:one", BudgetLimits(24, 200_000, 30_000, 300_000), currency="USD")
        budget.reserve("req:one", ["day:one"], RequestAllowance(100, 10, 10_000))
        budget.mark_sent("req:one")
    finally:
        connection.close()
    return directory, reference, document


def test_sqlite_backup_restores_documents_checkpoints_and_blocks_old_budget(profile, tmp_path):
    directory, reference, document = profile
    (directory / "models").mkdir()
    (directory / "models" / "large.onnx").write_bytes(b"excluded")
    (directory / "keys.json").write_text('{"api_key":"must-not-be-exported"}')
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    with zipfile.ZipFile(saved.path) as archive:
        assert not any("lock" in name or "models" in name or "keys.json" in name for name in archive.namelist())
        assert b"must-not-be-exported" not in b"".join(archive.read(name) for name in archive.namelist())
    target = restore_backup(saved.path, tmp_path / "restored")
    assert DocumentArchive(target / "revisions").get(reference.revision_id) == document
    connection = sqlite3.connect(target / "documents.sqlite3")
    try:
        assert connection.execute("SELECT count(*) FROM revisions").fetchone()[0] == 1
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()
    connection = sqlite3.connect(target / "pilot.sqlite3", isolation_level=None)
    try:
        budget = BudgetService(connection)
        assert budget.snapshot("day:one").requires_reconciliation
        assert budget.reservation("req:one").state == "unknown"
        with pytest.raises(BudgetStateError):
            budget.reserve("req:two", ["day:one"], RequestAllowance(1, 1, 1))
    finally:
        connection.close()


def test_backup_restores_published_multisource_profile_and_raw_evidence(profile, tmp_path):
    from app.pilot.service import PilotService
    from app.pilot.multisource.store import SignalStore
    from app.runtime.credentials import CredentialStore
    from tests.test_multisource_service import _profile, _wordstat

    directory = profile[0]
    store = SignalStore(directory)
    query_hash, concept_hash, _ = _profile(store)
    receipt_hash = _wordstat(tmp_path, store, query_hash)
    service = PilotService(directory, CredentialStore())
    try:
        run = service.start_signals(query_hash, (concept_hash,), wordstat_receipt_hash=receipt_hash)
        service.coordinator.wait()
        expected = service.signal_result(run)["profile"]
    finally:
        service.close()
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "signal-backups")
    restored = restore_backup(saved.path, tmp_path / "signal-restored")
    opened = PilotService(restored, CredentialStore())
    try:
        assert opened.signal_result(run)["profile"] == expected
        assert opened.list_signal_runs()[0]["id"] == run
    finally:
        opened.close()


def test_backup_retains_every_arxiv_version_for_later_portable_export(profile, tmp_path):
    from datetime import date, datetime, timezone

    from app.pilot.multisource.arxiv import import_arxiv_discovery
    from app.pilot.multisource.contracts import ArxivImportReceipt, ArxivVersion
    from app.pilot.multisource.queries import build_manual_profile
    from app.pilot.multisource.store import SignalStore
    from tests.test_multisource_arxiv import SCOPE, _entry, _feed

    directory = profile[0]
    store = SignalStore(directory)
    query_hash = store.put_object(build_manual_profile(SCOPE, SCOPE, seed_terms=(SCOPE,),
                                                        primary_phrase=SCOPE, confirmed_at=datetime.now(timezone.utc)))
    path = _feed(tmp_path / "versions.atom", _entry(1), _entry(2, updated="2025-02-01T10:00:00Z"))
    receipt_hash = import_arxiv_discovery(store, DocumentArchive(directory / "revisions"), path, query_hash,
                                          as_of=date.today(), retention="local_allowed",
                                          export_right="share_allowed", license_ref="public test metadata")
    receipt = store.get_object(receipt_hash, ArxivImportReceipt)
    revision_ids = [store.get_object(digest, ArxivVersion).revision_id for digest in receipt.version_hashes]
    assert len(revision_ids) == 2 and len(receipt.selected_revision_ids) == 1
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "arxiv-backups")
    restored = restore_backup(saved.path, tmp_path / "arxiv-restored")
    archive = DocumentArchive(restored / "revisions")
    assert all(archive.get(digest).source == "arxiv" for digest in revision_ids)


def test_backup_replays_real_v3_history_envelope_before_and_after_restore(profile, tmp_path):
    from tests.test_pilot_export import make_result

    directory = profile[0]
    output, _, artifacts = make_result(directory, historical=True)

    def process(context, payload):
        result = output.model_copy(update={"run_id": context.run_id})
        return {"result": result.model_dump(mode="json"), "assessments": [item.model_dump(mode="json") for item in artifacts]}

    coordinator = Coordinator(directory, process)
    run_id = coordinator.submit({"query": "lithium extraction"})
    try:
        deadline = time.monotonic() + 5
        while coordinator.get(run_id)["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert coordinator.get(run_id)["state"] == "succeeded"
    finally:
        coordinator.close()
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    restored = restore_backup(saved.path, tmp_path / "restored")
    assert (restored / "pilot.sqlite3").is_file()


def _catalogue(directory, name, payload):
    from app.pilot.contracts import content_hash

    path = directory / name / (content_hash(payload) + ".json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def test_backup_keeps_imported_analysis_without_a_coordinator_run(profile, tmp_path):
    from app.pilot.export import verify_result
    from tests.test_pilot_export import make_result

    directory = profile[0]
    output, _, artifacts = make_result(directory, historical=True)
    payload = {"result": output.model_dump(mode="json"), "assessments": [item.model_dump(mode="json") for item in artifacts],
               "imported": True}
    path = _catalogue(directory, "imported-results", payload)
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    restored = restore_backup(saved.path, tmp_path / "restored")
    assert (restored / "imported-results" / path.name).read_bytes() == path.read_bytes()
    verify_result(output, DocumentArchive(restored / "revisions"), artifacts)


def test_backup_preserves_and_replays_sensitivity_journal(profile, tmp_path):
    from app.pilot.sensitivity import SensitivityScenario, evaluate_sensitivity, load_sensitivity, save_sensitivity, verify_sensitivity
    from tests.test_pilot_evidence import Context
    from tests.test_pilot_export import make_result

    directory = profile[0]
    result, archive, artifacts = make_result(directory, historical=True)
    card = result.cards[0]
    report = evaluate_sensitivity(card.candidate, result.query_plan, result.snapshots[1], archive, Context(),
        passport=card, scenario=SensitivityScenario(exclude_largest_verified_group=True))
    path = save_sensitivity(report, directory / "sensitivity")
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    restored = restore_backup(saved.path, tmp_path / "restored")
    restored_report = load_sensitivity(restored / "sensitivity" / path.name)
    assert restored_report == report
    verify_sensitivity(restored_report, DocumentArchive(restored / "revisions"), Context())


def test_backup_preserves_expert_review_chain_and_rejects_missing_predecessor(profile, tmp_path):
    from app.pilot.history import assess_snapshot
    from app.pilot.review import apply_novelty_review, read_review, record_review
    from tests.test_pilot_antecedents import bundle_for
    from tests.test_pilot_evidence import Context, document
    from tests.test_pilot_export import make_result
    from tests.test_pilot_review import decision_for

    directory = profile[0]
    result, archive, _ = make_result(directory, historical=True)
    passport = result.cards[0]
    scenario = (archive, (), *result.snapshots, passport.candidate, Context(), passport)
    bundle, _ = bundle_for(scenario, (document(81, year=2010),))
    previous = None
    records = []
    for _ in range(2):
        decision = decision_for(bundle, passport, supersedes_review_id=previous)
        expanded, novelty = apply_novelty_review(decision, bundle, passport, archive)
        artifact, card, _ = assess_snapshot(passport.candidate, result.query_plan, result.snapshots[1], archive, Context(),
            passport=expanded, verified_novelty=novelty, antecedents=bundle)
        record = record_review(directory / "reviews", decision, bundle, source_card=passport, reviewed_card=card,
            artifact=artifact, historical_snapshot=result.snapshots[1], archive=archive)
        records.append(record)
        previous = record.review_id
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    restored = restore_backup(saved.path, tmp_path / "restored")
    for record in records:
        assert read_review(restored / "reviews", record.review_id, archive=DocumentArchive(restored / "revisions")) == record
    (directory / "reviews" / (records[0].review_id + ".json")).unlink()
    with BackupSession(directory) as session, pytest.raises(ArchiveError):
        create_backup(session, tmp_path / "new-backups")


def test_backup_preserves_supplemental_arxiv_and_licensed_report_text(profile, tmp_path):
    from app.pilot.reports import ReportPage, ReportRecord
    from app.pilot.supplemental import SupplementalLibrary
    from threading import Event

    directory = profile[0]
    archive = DocumentArchive(directory / "revisions")
    documents = [DocumentRecord(source="arxiv", source_id="2501.01234", title="Membrane technology",
                               publication_year=2025, date_precision="year", url="https://arxiv.org/abs/2501.01234"),
        ReportRecord(source_id="a" * 64, title="Published industry report", publication_year=2025, date_precision="year",
            url="https://example.org/report.pdf",
            full_text="Licensed report text.", page_spans=(ReportPage(page=1, start=0, end=21),),
            pages_total=1, pages_with_text=1, extraction_status="text", pdf_sha256="a" * 64,
            public_license_allowed=True, license_note="CC BY 4.0")]
    entries = []
    library = SupplementalLibrary(directory, archive)
    for document in documents:
        reference = archive.put(document)
        saved_entry = library._save(document.source, (document,), cancel=Event())
        path = directory / "supplemental" / (saved_entry["id"] + ".json")
        entries.append((path, reference, document))
    with BackupSession(directory) as session:
        saved = create_backup(session, tmp_path / "backups")
    restored = restore_backup(saved.path, tmp_path / "restored")
    for path, reference, document in entries:
        assert (restored / "supplemental" / path.name).read_bytes() == path.read_bytes()
        assert DocumentArchive(restored / "revisions").get(reference.revision_id) == document


@pytest.mark.parametrize("tampering", ["filename", "reference", "secret"])
def test_backup_rejects_damaged_supplemental_catalogue(profile, tmp_path, tampering):
    directory = profile[0]
    document = DocumentRecord(source="arxiv", source_id="2501.01234", title="Membrane technology", url="https://arxiv.org/abs/2501.01234")
    reference = DocumentArchive(directory / "revisions").put(document).model_dump(mode="json")
    if tampering == "reference":
        reference["text_hash"] = "f" * 64
    payload = {"version": 1, "kind": "arxiv", "created_at": "2026-09-10T10:00:00+00:00",
               "documents": [reference], "limitations": []}
    if tampering == "secret":
        payload["api_key"] = "private-key-must-not-escape"
    path = _catalogue(directory, "supplemental", payload)
    if tampering == "filename":
        path.rename(path.with_name("f" * 64 + ".json"))
    with BackupSession(directory) as session, pytest.raises(ArchiveError) as error:
        create_backup(session, tmp_path / "backups")
    assert "private-key" not in str(error.value)
    assert not list((tmp_path / "backups").glob("*.zip"))


@pytest.mark.parametrize("name", ["backend.lock", "pilot.lock"])
def test_backup_cannot_run_while_either_service_owns_profile(profile, name):
    directory, _, _ = profile
    lock = InstanceLock(directory / name)
    lock.acquire()
    try:
        with pytest.raises(ArchiveError):
            with BackupSession(directory):
                pass
    finally:
        lock.release()


def test_closed_session_is_not_a_quiescence_bypass(profile, tmp_path):
    with BackupSession(profile[0]) as session:
        pass
    with pytest.raises(ArchiveError):
        create_backup(session, tmp_path / "backups")


def test_default_rotation_keeps_three_own_backups_and_never_deletes_foreign_archive(profile, tmp_path):
    destination = tmp_path / "backups"
    destination.mkdir()
    foreign = destination / "my-important.zip"
    foreign.write_bytes(b"user archive")
    with BackupSession(profile[0]) as session:
        created = [create_backup(session, destination) for _ in range(4)]
    assert all(item.path.name.startswith("trendanalyser-backup-main2-") for item in created)
    assert not created[0].path.exists()
    assert all(item.path.exists() for item in created[1:])
    assert created[-1].rotated == 1
    assert foreign.read_bytes() == b"user archive"


def test_rotation_accepts_pre_rename_backup_names(profile, tmp_path):
    destination = tmp_path / "backups"
    with BackupSession(profile[0]) as session:
        old = create_backup(session, destination, keep=1)
        previous_name = old.path.name.replace("trendanalyser-", "trendanalizer-", 1)
        previous_path = old.path.rename(destination / previous_name)
        registry = destination / ".trendanalizer-backups-main2.json"
        registry.write_text(registry.read_text(encoding="utf-8").replace(old.path.name, previous_name), encoding="utf-8")
        current = create_backup(session, destination, keep=1)
    assert current.path.exists()
    assert not previous_path.exists()
    assert current.rotated == 1


def test_rotation_preserves_user_replacement_of_previously_owned_filename(profile, tmp_path):
    with BackupSession(profile[0]) as session:
        old = create_backup(session, tmp_path / "backups", keep=1)
        old.path.write_bytes(b"replacement created by user")
        create_backup(session, tmp_path / "backups", keep=1)
    assert old.path.read_bytes() == b"replacement created by user"


def test_restore_rejects_active_or_nonempty_destination_without_changing_files(profile, tmp_path):
    with BackupSession(profile[0]) as session:
        saved = create_backup(session, tmp_path / "backups")
    target = tmp_path / "existing"
    target.mkdir()
    original = target / "important.txt"
    original.write_text("keep me")
    with pytest.raises(ArchiveError):
        restore_backup(saved.path, target)
    assert original.read_text(encoding="utf-8") == "keep me"


def test_missing_referenced_revision_aborts_backup_without_partial_archive(profile, tmp_path):
    directory, reference, _ = profile
    DocumentArchive(directory / "revisions").path(reference.revision_id).unlink()
    with BackupSession(directory) as session:
        with pytest.raises(ArchiveError):
            create_backup(session, tmp_path / "backups")
    assert not list((tmp_path / "backups").glob("*.zip"))


def test_structured_key_in_checkpoint_is_rejected(profile, tmp_path):
    directory = profile[0]
    connection = sqlite3.connect(directory / "pilot.sqlite3", isolation_level=None)
    try:
        connection.execute("UPDATE analysis_runs SET input_json=?", (json.dumps({"api_key": "private"}),))
    finally:
        connection.close()
    with BackupSession(directory) as session:
        with pytest.raises(ArchiveError):
            create_backup(session, tmp_path / "backups")


def test_future_database_version_is_refused(profile, tmp_path):
    connection = sqlite3.connect(profile[0] / "pilot.sqlite3", isolation_level=None)
    connection.execute("PRAGMA user_version=99")
    connection.close()
    with BackupSession(profile[0]) as session:
        with pytest.raises(ArchiveError):
            create_backup(session, tmp_path / "backups")


def test_schema_expressions_cannot_invoke_unexpected_functions(profile, tmp_path):
    connection = sqlite3.connect(profile[0] / "documents.sqlite3", isolation_level=None)
    connection.execute("CREATE INDEX unexpected ON jobs(length(error_message))")
    connection.close()
    with BackupSession(profile[0]) as session:
        with pytest.raises(ArchiveError):
            create_backup(session, tmp_path / "backups")


@pytest.mark.parametrize("expression", ["json_extract(input_json, '$.payload.query')", "length(input_json)"])
def test_schema_allowlist_rejects_modified_application_expression_index(profile, tmp_path, expression):
    connection = sqlite3.connect(profile[0] / "pilot.sqlite3", isolation_level=None)
    try:
        connection.execute("DROP INDEX ix_analysis_antecedents_lookup")
        connection.execute(ANTECEDENTS_INDEX_SQL.replace("json_extract(input_json, '$.payload.candidate_id')", expression))
    finally:
        connection.close()
    with BackupSession(profile[0]) as session:
        with pytest.raises(ArchiveError, match="неподдерживаемые выражения"):
            create_backup(session, tmp_path / "backups")


def test_backup_accepts_older_pilot_database_without_additive_review_index(profile, tmp_path):
    connection = sqlite3.connect(profile[0] / "pilot.sqlite3", isolation_level=None)
    try:
        connection.execute("DROP INDEX ix_analysis_antecedents_lookup")
    finally:
        connection.close()
    with BackupSession(profile[0]) as session:
        saved = create_backup(session, tmp_path / "backups")
    restored = restore_backup(saved.path, tmp_path / "restored")
    coordinator = Coordinator(restored, lambda context, payload: {})
    coordinator.close()
    connection = sqlite3.connect(restored / "pilot.sqlite3")
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT 1 FROM sqlite_master WHERE name='ix_analysis_antecedents_lookup'").fetchone() == (1,)
    finally:
        connection.close()


def test_foreign_key_violation_is_refused_before_restoring(profile, tmp_path):
    connection = sqlite3.connect(profile[0] / "pilot.sqlite3", isolation_level=None)
    connection.execute("INSERT INTO analysis_checkpoints VALUES ('missing-run','result',1,?)", ("a" * 64,))
    connection.close()
    with BackupSession(profile[0]) as session:
        with pytest.raises(ArchiveError):
            create_backup(session, tmp_path / "backups")


def test_backup_uses_sqlite_api_for_committed_wal_bytes(profile, tmp_path):
    # Service locks remain free, but a read-capable connection keeps WAL open.
    directory = profile[0]
    connection = sqlite3.connect(directory / "documents.sqlite3", isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("UPDATE jobs SET error_message='committed in WAL'")
    try:
        with BackupSession(directory) as session:
            saved = create_backup(session, tmp_path / "backups")
        restored = restore_backup(saved.path, tmp_path / "restored")
    finally:
        connection.close()
    checked = sqlite3.connect(restored / "documents.sqlite3")
    try:
        assert checked.execute("SELECT error_message FROM jobs").fetchone()[0] == "committed in WAL"
    finally:
        checked.close()


def _malicious_zip(path, name, *, mode=None, content=b"x"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", "{}")
        info = zipfile.ZipInfo(name)
        info.compress_type = zipfile.ZIP_DEFLATED
        if mode is not None:
            info.external_attr = mode << 16
        archive.writestr(info, content)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/windows", "dir\\escape", "a/../b", "a//b", "directory/", "CON", "nested/NUL.txt", "data."])
def test_archive_paths_cannot_escape_quarantine(tmp_path, name):
    path = tmp_path / "malicious.zip"
    _malicious_zip(path, name)
    with pytest.raises(ArchiveError):
        with unpack_package(path, expected_kind="trendanalizer-result"):
            pass
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFREG | 0o755, stat.S_IFIFO | 0o600])
def test_links_executables_and_special_files_are_rejected(tmp_path, mode):
    path = tmp_path / "malicious.zip"
    _malicious_zip(path, "payload", mode=mode)
    with pytest.raises(ArchiveError):
        with unpack_package(path, expected_kind="trendanalizer-result"):
            pass


def test_high_compression_bomb_is_rejected(tmp_path):
    path = tmp_path / "bomb.zip"
    _malicious_zip(path, "payload", content=b"0" * (2 * 1024 * 1024))
    with pytest.raises(ArchiveError):
        with unpack_package(path, expected_kind="trendanalizer-result"):
            pass


def test_duplicate_archive_members_are_rejected(tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("payload", "one")
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("payload", "two")
    with pytest.raises(ArchiveError):
        with unpack_package(path, expected_kind="trendanalizer-result"):
            pass


def test_verified_zip_detects_changed_bytes_even_with_recomputed_crc(tmp_path):
    payload = tmp_path / "value.json"
    payload.write_text('{"value":1}')
    saved = write_package(tmp_path / "valid.zip", {"value.json": payload}, kind="trendanalizer-result")
    altered = tmp_path / "altered.zip"
    with zipfile.ZipFile(saved.path) as source, zipfile.ZipFile(altered, "w") as target:
        for info in source.infolist():
            target.writestr(info, b'{"value":2}' if info.filename == "value.json" else source.read(info))
    with pytest.raises(ArchiveError):
        with unpack_package(altered, expected_kind="trendanalizer-result"):
            pass


def test_package_writer_never_overwrites_existing_archive(tmp_path):
    payload, target = tmp_path / "payload", tmp_path / "existing.zip"
    payload.write_text("data")
    target.write_bytes(b"keep")
    with pytest.raises(ArchiveError):
        write_package(target, {"payload": payload}, kind="trendanalizer-result")
    assert target.read_bytes() == b"keep"


@pytest.mark.skipif(os.name == "nt", reason="Windows uses profile ACLs rather than POSIX file modes")
def test_package_is_private_while_compression_is_writing(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.write_text("private report")
    original_zip = zipfile.ZipFile
    observed = []

    def inspect_open(file, mode="r", *args, **kwargs):
        if mode == "w":
            observed.append(stat.S_IMODE(os.fstat(file.fileno()).st_mode))
        return original_zip(file, mode, *args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", inspect_open)
    previous_umask = os.umask(0o022)
    try:
        saved = write_package(tmp_path / "private.zip", {"payload": payload}, kind="trendanalizer-result")
    finally:
        os.umask(previous_umask)
    assert observed == [0o600]
    assert stat.S_IMODE(saved.path.stat().st_mode) == 0o600


def test_precancelled_archive_actions_never_open_or_publish(tmp_path):
    from threading import Event
    from app.runtime.jobs import TaskCancelled

    cancel = Event()
    cancel.set()
    with pytest.raises(TaskCancelled):
        with unpack_package(tmp_path / "missing.zip", expected_kind="trendanalizer-result", cancel=cancel):
            pytest.fail("Cancelled import must not yield")
    with pytest.raises(TaskCancelled):
        write_package(tmp_path / "output.zip", {"missing": tmp_path / "missing"},
                      kind="trendanalizer-result", cancel=cancel)
    assert not list(tmp_path.iterdir())


def test_cancel_during_extraction_stops_at_first_chunk_and_removes_quarantine(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from threading import Event
    import app.runtime.backup as backup
    from app.runtime.jobs import TaskCancelled

    payload = tmp_path / "data.bin"
    payload.write_bytes(os.urandom(3 * 1024 * 1024))
    saved = write_package(tmp_path / "result.zip", {"data.bin": payload}, kind="trendanalizer-result")
    cancel = Event()
    original_read = zipfile.ZipExtFile.read
    extracted = []
    quarantines = []
    original_temporary = backup.tempfile.TemporaryDirectory

    def temporary(*args, **kwargs):
        directory = original_temporary(*args, **kwargs)
        quarantines.append(Path(directory.name))
        return directory

    def read(stream, size=-1):
        block = original_read(stream, size)
        if stream.name == "data.bin":
            extracted.append(len(block))
            cancel.set()
        return block

    monkeypatch.setattr(backup.tempfile, "TemporaryDirectory", temporary)
    monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    with pytest.raises(TaskCancelled):
        with unpack_package(saved.path, expected_kind="trendanalizer-result", cancel=cancel):
            pytest.fail("Cancelled extraction must not yield")
    assert sum(extracted) == 1024 * 1024
    assert quarantines and all(not path.exists() for path in quarantines)


def test_cancel_during_zip_writing_removes_partial_file(tmp_path, monkeypatch):
    import os
    from threading import Event
    from app.runtime.jobs import TaskCancelled

    payload = tmp_path / "data.bin"
    payload.write_bytes(os.urandom(3 * 1024 * 1024))
    cancel = Event()
    original = zipfile._ZipWriteFile.write
    written = []

    def write(stream, data):
        result = original(stream, data)
        if len(data) == 1024 * 1024:
            written.append(len(data))
            cancel.set()
        return result

    monkeypatch.setattr(zipfile._ZipWriteFile, "write", write)
    with pytest.raises(TaskCancelled):
        write_package(tmp_path / "result.zip", {"data.bin": payload}, kind="trendanalizer-result", cancel=cancel)
    assert written == [1024 * 1024]
    assert set(tmp_path.iterdir()) == {payload}


def test_cancel_at_final_publish_fence_leaves_no_archive(tmp_path, monkeypatch):
    from threading import Event
    import app.runtime.backup as backup
    from app.runtime.jobs import TaskCancelled

    payload = tmp_path / "data.bin"
    payload.write_bytes(b"actual document")
    cancel = Event()
    original = backup.os.chmod

    def chmod(path, mode):
        original(path, mode)
        if str(path).endswith(".zip.tmp"):
            cancel.set()

    monkeypatch.setattr(backup.os, "chmod", chmod)
    with pytest.raises(TaskCancelled):
        write_package(tmp_path / "result.zip", {"data.bin": payload}, kind="trendanalizer-result", cancel=cancel)
    assert set(tmp_path.iterdir()) == {payload}


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="Unix special-file boundary")
def test_importing_local_fifo_is_rejected_without_blocking(tmp_path):
    import os

    selected = tmp_path / "named.pipe"
    os.mkfifo(selected)
    started = time.monotonic()
    with pytest.raises(ArchiveError):
        with unpack_package(selected, expected_kind="trendanalizer-result"):
            pass
    assert time.monotonic() - started < 1


def test_backup_enforces_checkpoint_count_before_unbounded_payload_loading(profile, monkeypatch):
    import app.runtime.backup as backup

    directory = profile[0]
    monkeypatch.setattr(backup, "MAX_FILES", 1)
    with pytest.raises(ArchiveError, match="Слишком много сохранённых этапов"):
        backup._referenced_files(directory)


def test_backup_history_replay_uses_remaining_shared_deadline(profile, monkeypatch):
    import app.runtime.backup as backup
    import app.pilot.export as portable
    from tests.test_pilot_export import make_result

    directory = profile[0]
    result, _, artifacts = make_result(directory, historical=True)
    _catalogue(directory, "imported-results", {
        "result": result.model_dump(mode="json"), "assessments": [item.model_dump(mode="json") for item in artifacts],
        "imported": True})
    instant = [0.0]
    monkeypatch.setattr(backup.time, "monotonic", lambda: instant[0])
    original = portable.verify_result

    def slow_verify(*args, **kwargs):
        # Archival indexing used the entire shared budget before replay began.
        instant[0] = 121.0
        return original(*args, **kwargs)

    monkeypatch.setattr(portable, "verify_result", slow_verify)
    with pytest.raises(ArchiveError, match="превысила допустимое время"):
        backup._referenced_files(directory)


@pytest.mark.parametrize("statement", [
    "ALTER TABLE documents RENAME COLUMN updated_at TO missing_updated_at",
    "ALTER TABLE documents ADD COLUMN unexpected TEXT",
])
def test_supported_version_with_incompatible_columns_cannot_be_backed_up(profile, tmp_path, statement):
    directory = profile[0]
    connection = sqlite3.connect(directory / "documents.sqlite3", isolation_level=None)
    try:
        connection.execute(statement)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()
    with BackupSession(directory) as session:
        with pytest.raises(ArchiveError, match="Столбцы базы"):
            create_backup(session, tmp_path / "backup")
    assert not list((tmp_path / "backup").glob("*.zip"))
