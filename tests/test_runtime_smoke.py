"""Runtime capabilities, process isolation and failure reporting."""

import json
import multiprocessing
import socket

import pytest

from tools import runtime_smoke
from tools.runtime_environment import diagnostic_environment


def test_offline_smoke_guard_blocks_connects_and_restores_socket_methods():
    original = socket.socket.connect
    with runtime_smoke._deny_external_network():
        assert runtime_smoke._network_denial_verified() is True
    assert socket.socket.connect is original


@pytest.mark.parametrize(("version", "patched"), [
    ("3.50.4", False), ("3.51.2", False), ("3.51.3", False), ("3.53.1", False),
    ("3.50.7", False), ("3.44.6", False), ("3.44.5", False), ("3.50.6", False),
    ("3.53.2", True), ("3.53.4", True), ("3.54.0", True),
    ("invalid", False), ("3.51", False),
])
def test_security_gate_rejects_wal_only_fixes_and_accepts_fts5_fixes(version, patched):
    assert runtime_smoke.sqlite_is_patched(version) is patched


def test_sqlite_smoke_uses_unicode_paths_and_real_fts5(tmp_path):
    report = runtime_smoke._sqlite_check(tmp_path)
    assert report["fts5"] is True
    assert report["integrity"] == "ok"
    assert report["synchronous"] == "FULL"
    assert report["security_fixes"] is True
    assert report["minimum_secure_version"] == "3.53.2"
    assert (tmp_path / "проверка базы.sqlite3").is_file()


def test_application_resources_are_actually_readable():
    report = runtime_smoke._resources_check()
    assert report["fonts"] == 4
    assert report["font_license"] is True
    assert report["legacy_direction_resources"] is True


def test_spawn_runs_native_dependencies_without_importing_tk():
    report = runtime_smoke._spawn_check()
    assert report["separate_process"] is True
    assert report["tk_imported_in_worker"] is False
    assert report["exitcode"] == 0


def test_timed_out_spawn_is_reaped_without_touching_other_processes():
    existing = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(TimeoutError):
        runtime_smoke._spawn_check(timeout=0)
    assert {process.pid for process in multiprocessing.active_children()} == existing


def test_report_does_not_turn_missing_checks_or_failed_runtime_into_release_acceptance(monkeypatch):
    monkeypatch.setattr(runtime_smoke, "_sqlite_check", lambda directory: {"security_fixes": False})
    monkeypatch.setattr(runtime_smoke, "_resources_check", lambda: {})
    monkeypatch.setattr(runtime_smoke, "_native_check", lambda: {})
    monkeypatch.setattr(runtime_smoke, "_spawn_check", lambda: {})
    result = runtime_smoke.run_checks()
    assert result["runtime_passed"] is True
    assert result["sqlite_security_gate_passed"] is False
    assert result["all_requested_checks_passed"] is False
    assert result["release_acceptance"] is False
    assert result["checks"]["gui"]["status"] == "not_run"
    assert result["checks"]["snapshot"]["status"] == "not_run"
    assert result["checks"]["model"]["status"] == "not_run"
    assert result["checks"]["pilot_model_spawn"]["status"] == "not_run"
    assert result["checks"]["pilot_result"]["status"] == "not_run"
    assert result["checks"]["pilot_discovery_spawn"]["status"] == "not_run"


def test_old_wal_only_report_cannot_pass_current_security_gate(monkeypatch):
    for name in ("_resources_check", "_native_check", "_spawn_check"):
        monkeypatch.setattr(runtime_smoke, name, lambda: {})
    monkeypatch.setattr(runtime_smoke, "_backup_restore_check", lambda *_: {})
    monkeypatch.setattr(runtime_smoke, "_sqlite_check", lambda _: {"wal_reset_fix": True, "version": "3.53.1"})
    result = runtime_smoke.run_checks()
    assert result["runtime_passed"] is True
    assert result["sqlite_security_gate_passed"] is False
    assert result["all_requested_checks_passed"] is False


def test_failed_check_records_exception_type_without_sensitive_exception_message(monkeypatch):
    def fail():
        raise RuntimeError("secret-value-from-provider")

    monkeypatch.setattr(runtime_smoke, "_sqlite_check", lambda directory: {"security_fixes": True})
    monkeypatch.setattr(runtime_smoke, "_resources_check", fail)
    monkeypatch.setattr(runtime_smoke, "_native_check", lambda: {})
    monkeypatch.setattr(runtime_smoke, "_spawn_check", lambda: {})
    result = runtime_smoke.run_checks()
    assert result["all_requested_checks_passed"] is False
    assert result["checks"]["resources"]["error_type"] == "RuntimeError"
    assert "secret-value" not in json.dumps(result)


def test_gui_smoke_uses_unique_empty_keychain_namespace_and_restores_it(monkeypatch):
    from app.runtime import credentials, session

    monkeypatch.setattr(session, "_credentials", None)
    original = credentials.KEYRING_NAMESPACE
    with runtime_smoke._isolated_gui_credentials():
        first = credentials.KEYRING_NAMESPACE
        assert first.startswith(original + ".runtime-smoke.")
        store = session.credentials()
        assert store is not None
    assert credentials.KEYRING_NAMESPACE == original
    assert session._credentials is None
    with pytest.raises(credentials.CredentialUnavailable):
        _ = store.storage_status
    with runtime_smoke._isolated_gui_credentials():
        assert credentials.KEYRING_NAMESPACE != first


def test_gui_smoke_cannot_reuse_or_close_a_live_application_credential_session(monkeypatch):
    from app.runtime import credentials, session

    active = credentials.CredentialStore()
    monkeypatch.setattr(session, "_credentials", active)
    original = credentials.KEYRING_NAMESPACE
    with pytest.raises(runtime_smoke.GuiSmokeFailure) as caught:
        with runtime_smoke._isolated_gui_credentials():
            pytest.fail("A live application session must not enter smoke isolation")
    assert caught.value.stage == "credential_isolation"
    assert credentials.KEYRING_NAMESPACE == original
    assert session._credentials is active
    assert active._closed is False
    active.close()


def test_gui_failure_diagnostic_never_records_callback_text(monkeypatch):
    for name in ("_resources_check", "_native_check", "_spawn_check"):
        monkeypatch.setattr(runtime_smoke, name, lambda: {})
    monkeypatch.setattr(runtime_smoke, "_sqlite_check", lambda _: {"security_fixes": True})
    def fail(*args):
        raise runtime_smoke.GuiSmokeFailure("pilot_status", "secret-value-from-provider")
    monkeypatch.setattr(runtime_smoke, "_gui_check", fail)
    result = runtime_smoke.run_checks(gui=True)
    assert result["checks"]["gui"]["stage"] == "pilot_status"
    assert "secret-value" not in json.dumps(result)


def test_atomic_report_replaces_valid_json_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "отчёт.json"
    path.write_text("old", encoding="utf-8")
    runtime_smoke.write_report(path, {"status": "проверено"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "проверено"}
    assert list(tmp_path.iterdir()) == [path]


def test_diagnostic_tools_receive_no_api_secrets_or_developer_pythonpath(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "do-not-copy")
    monkeypatch.setenv("EPO_OPS_SECRET", "do-not-copy")
    monkeypatch.setenv("EPO_OPS_KEY", "do-not-copy")
    monkeypatch.setenv("PYTHONPATH", "/protected-original-checkout")
    monkeypatch.setenv("PATH", "/system/bin")
    environment = diagnostic_environment()
    assert "DEEPSEEK_API_KEY" not in environment
    assert "EPO_OPS_SECRET" not in environment
    assert "EPO_OPS_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PATH"] == "/system/bin"
    assert environment["ORT_DISABLE_TELEMETRY"] == "1"


def test_pilot_result_smoke_replays_real_contracts_quotes_metrics_and_library(tmp_path, monkeypatch):
    from app.pilot.export import export_result
    from tests.test_pilot_export import make_result

    result, archive, artifacts = make_result(tmp_path / "fixture", historical=True)
    path = tmp_path / "saved.trendresult"
    export_result(path, result, archive, artifacts)
    monkeypatch.setattr("httpx.Client.send", lambda *_args, **_kwargs: pytest.fail("No network in reopening"))
    proof = runtime_smoke._pilot_result_check(path, tmp_path / "reopen")
    assert proof["schema_version"] == 3 and proof["assessments_replayed"] == 1
    assert proof["cards"] == 1 and proof["archived_revisions"] == 17
    assert proof["exact_quotations_verified"] >= 3
    assert proof["local_library_reopened"] and proof["duplicate_import_idempotent"]
    assert proof["read_only_source"]


def test_pilot_result_smoke_rejects_corrupt_package(tmp_path):
    from app.runtime.backup import ArchiveError

    path = tmp_path / "broken.trendresult"
    path.write_bytes(b"not a package")
    with pytest.raises(ArchiveError):
        runtime_smoke._pilot_result_check(path, tmp_path / "reopen")


def test_requested_pilot_check_failure_stays_failed_without_exposing_provider_details(tmp_path, monkeypatch):
    for name in ("_resources_check", "_native_check", "_spawn_check"):
        monkeypatch.setattr(runtime_smoke, name, lambda: {})
    monkeypatch.setattr(runtime_smoke, "_sqlite_check", lambda _: {"security_fixes": True})

    def fail(*_args):
        raise RuntimeError("secret-model-or-provider-details")

    monkeypatch.setattr(runtime_smoke, "_pilot_model_check", fail)
    result = runtime_smoke.run_checks(pilot_model_dir=tmp_path)
    assert result["checks"]["pilot_model_spawn"]["status"] == "failed"
    assert result["checks"]["pilot_result"]["status"] == "not_run"
    assert not result["all_requested_checks_passed"]
    assert "secret-model-or-provider" not in json.dumps(result)
