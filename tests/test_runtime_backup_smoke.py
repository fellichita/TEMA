"""Runtime checks detect durable data loss across backup, restore and reopen."""

import hashlib
import json
import socket

import pytest

from tools import runtime_smoke


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.runtime import credentials, session

    def forbidden(*_args, **_kwargs):
        pytest.fail("Backup smoke must not use network or user credentials")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr(session, "credentials", forbidden)
    monkeypatch.setattr(credentials.CredentialStore, "get", forbidden)
    monkeypatch.setattr(runtime_smoke.tempfile, "tempdir", str(tmp_path))
    return tmp_path


def test_backup_smoke_reopens_exact_durable_records_and_removes_owned_data(isolated):
    proof = runtime_smoke._backup_restore_check(isolated)
    assert proof["documents"] == 1
    assert proof["document_revisions"] == 2
    assert proof["collection_jobs"] == 2
    assert proof["history_periods"] == 1
    assert proof["analysis_runs"] == 1
    assert proof["reopen_count"] == 2
    assert proof["application_id"] == "org.trendanalizer.pilot.main2"
    assert proof["profile_selection_reopened"] is True
    assert proof["temporary_data_removed"] is True
    assert proof["imported_result_verified"] is False
    assert proof["no_network_or_credentials_required"] is True
    assert len(proof["backup_sha256"]) == 64
    assert list(isolated.iterdir()) == []


@pytest.mark.parametrize("damage", ["document", "versions", "job_revision", "collection_history", "analysis_history",
                                   "checkpoint", "revision_file", "settings", "profile_selection"])
def test_backup_smoke_rejects_missing_or_changed_restored_data_and_cleans_up(isolated, monkeypatch, damage):
    from app.runtime import backup
    from app.sqlite_runtime import sqlite3

    restore = backup.restore_backup
    restored = []

    def damaged_restore(*args, **kwargs):
        target = restore(*args, **kwargs)
        restored.append(target)
        database = target / ("pilot.sqlite3" if damage in {"analysis_history", "checkpoint"} else "documents.sqlite3")
        statements = {
            "document": "UPDATE revisions SET payload=replace(payload, 'robot control', 'altered prose')",
            "versions": "DELETE FROM revisions WHERE revision_id NOT IN (SELECT latest_revision FROM documents)",
            "job_revision": "UPDATE job_documents SET revision_id=(SELECT latest_revision FROM documents)",
            "collection_history": "DELETE FROM history_periods",
            "analysis_history": "UPDATE analysis_runs SET message='unexpected restored state'",
            "checkpoint": "DELETE FROM analysis_checkpoints WHERE stage='retrieval'",
        }
        if damage in statements:
            connection = sqlite3.connect(database)
            try:
                connection.execute(statements[damage])
                connection.commit()
            finally:
                connection.close()
        elif damage == "revision_file":
            next((target / "revisions").rglob("*.json")).unlink()
        elif damage == "settings":
            (target / "settings.json").write_text("{}", encoding="utf-8")
        else:
            # A valid but wrong selected profile must not silently be accepted.
            (target / "active-library.json").write_text(json.dumps({
                "version": 1, "path": str(target.parent / "исходный профиль")}), encoding="utf-8")
        return target

    monkeypatch.setattr(backup, "restore_backup", damaged_restore)
    with pytest.raises(RuntimeError):
        runtime_smoke._backup_restore_check(isolated)
    assert len(restored) == 1
    assert list(isolated.iterdir()) == []


def test_backup_restore_is_mandatory_and_failure_blocks_runtime_success(monkeypatch):
    for name in ("_resources_check", "_native_check", "_spawn_check"):
        monkeypatch.setattr(runtime_smoke, name, lambda: {})
    monkeypatch.setattr(runtime_smoke, "_sqlite_check", lambda _: {"wal_reset_fix": True})
    called = []

    def fail(*_args):
        called.append(True)
        raise RuntimeError("private profile detail must not enter the report")

    monkeypatch.setattr(runtime_smoke, "_backup_restore_check", fail)
    proof = runtime_smoke.run_checks()
    assert called == [True]
    assert "backup_restore" in runtime_smoke.REQUIRED_RUNTIME_CHECKS
    assert proof["checks"]["backup_restore"]["status"] == "failed"
    assert proof["checks"]["backup_restore"]["error_type"] == "RuntimeError"
    assert not proof["runtime_passed"] and not proof["all_requested_checks_passed"]
    assert "private profile detail" not in json.dumps(proof)


def test_supplied_result_survives_backup_restore_without_changing_original(tmp_path, monkeypatch):
    from app.pilot.export import export_result
    from tests.test_pilot_export import make_result

    result, archive, artifacts = make_result(tmp_path / "fixture", historical=True)
    original = tmp_path / "public.trendresult"
    export_result(original, result, archive, artifacts)
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr("httpx.Client.send", lambda *_a, **_k: pytest.fail("No network"))
    proof = runtime_smoke._backup_restore_check(scratch, original)
    assert proof["imported_result_verified"] is True
    assert proof["input_package_sha256"] == before
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert list(scratch.iterdir()) == []
