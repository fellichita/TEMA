"""Admission races use Events and isolated storage, never providers or real keys."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest

from app.pilot.service import PilotService, WORKFLOW_VERSION
from app.pilot.settings import PilotSettings, load_settings, save_settings
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import Coordinator, TaskCancelled, TaskFailure
from app.sqlite_runtime import sqlite3


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    credentials = CredentialStore()
    monkeypatch.setattr(credentials, "_load_backend", lambda: None)
    credentials.set("deepseek_api_key", "fixture-original")
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda *_a, **_k: None)
    runtime = PilotService(tmp_path / "data", credentials)
    save_settings(runtime.data_dir, PilotSettings())
    monkeypatch.setattr(runtime.coordinator, "_processor", lambda _context, payload: payload)
    try:
        yield runtime
    finally:
        runtime.close()
        credentials.close()


def seed_old_cancelled_run(pilot):
    payload = json.dumps({"workflow_version": WORKFLOW_VERSION, "payload": {"fixture": True}})

    def seed():
        # Fixture setup is one durable batch; the runtime's FULL-sync writer
        # otherwise commits every row separately in autocommit mode.
        with pilot.coordinator._connection as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT INTO analysis_runs(id,attempt,state,input_json,created_at,updated_at) "
                "VALUES (?,1,'cancelled',?,?,?)",
                [("old", payload, "2020-01-01", "2020-01-01"),
                 *[(f"later-{index:03d}", payload, "2026-01-01", "2026-01-01") for index in range(101)]],
            )

    pilot.coordinator._executor.submit(seed).result(timeout=2)
    assert "old" not in {row["id"] for row in pilot.coordinator.list_runs()}


def test_seed_history_uses_one_durable_transaction_without_changing_runtime_policy(pilot):
    statements = []

    def trace():
        connection = pilot.coordinator._connection
        assert connection.isolation_level is None
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        connection.set_trace_callback(lambda sql: statements.append((sql.split()[0], connection.in_transaction)))

    pilot.coordinator._executor.submit(trace).result(timeout=2)
    try:
        seed_old_cancelled_run(pilot)
    finally:
        pilot.coordinator._executor.submit(lambda: pilot.coordinator._connection.set_trace_callback(None)).result(timeout=2)
    writes = [in_transaction for statement, in_transaction in statements if statement == "INSERT"]
    assert len(writes) == 102
    assert all(writes), "Fixture inserts must share one transaction instead of 102 FULL-sync autocommits"
    assert [statement for statement, _ in statements if statement in {"BEGIN", "COMMIT", "ROLLBACK"}] == ["BEGIN", "COMMIT"]
    assert len(pilot.coordinator.list_runs(limit=100)) == 100
    assert pilot.coordinator.get("old")["state"] == "cancelled"
    assert pilot.coordinator._executor.submit(lambda: pilot.coordinator._connection.in_transaction).result(timeout=2) is False


def test_failed_history_seed_rolls_back_partial_rows_and_leaves_writer_usable(pilot):
    def conflict():
        pilot.coordinator._connection.execute(
            "INSERT INTO analysis_runs(id,attempt,state,input_json,created_at,updated_at) "
            "VALUES ('later-000',1,'cancelled','{}','2026-01-01','2026-01-01')")

    pilot.coordinator._executor.submit(conflict).result(timeout=2)
    with pytest.raises(sqlite3.IntegrityError):
        seed_old_cancelled_run(pilot)
    assert [row["id"] for row in pilot.coordinator.list_runs()] == ["later-000"]
    assert pilot.coordinator._executor.submit(lambda: pilot.coordinator._connection.in_transaction).result(timeout=2) is False


@pytest.mark.parametrize("operation", ["configure", "delete"])
def test_old_resumed_active_run_blocks_credential_mutations(pilot, monkeypatch, operation):
    seed_old_cancelled_run(pilot)
    entered = Event()

    def process(context, _payload):
        entered.set()
        assert context.cancel_event.wait(5)
        context.check_cancelled()

    monkeypatch.setattr(pilot.coordinator, "_processor", process)
    pilot.resume("old")
    assert entered.wait(2)
    before = load_settings(pilot.data_dir)
    try:
        with pytest.raises(TaskFailure):
            if operation == "configure":
                pilot.configure(before.model_dump(mode="json"), {"deepseek_api_key": "fixture-replacement"})
            else:
                pilot.delete_credential("deepseek_api_key")
        assert pilot.credentials.get("deepseek_api_key") == "fixture-original"
        assert load_settings(pilot.data_dir) == before
    finally:
        pilot.cancel("old")
        pilot.coordinator.wait(timeout=2)


@pytest.mark.parametrize("operation", ["configure", "delete", "submit", "resume"])
def test_preflight_reservation_blocks_other_admission_and_mutations(pilot, monkeypatch, operation):
    seed_old_cancelled_run(pilot)
    entered, release = Event(), Event()

    def verify(*_a, **_k):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", verify)
    with ThreadPoolExecutor(max_workers=1) as executor:
        start = executor.submit(pilot.start, "fixture materials", "fixture materials")
        assert entered.wait(2)
        try:
            with pytest.raises(TaskFailure):
                if operation == "configure":
                    pilot.configure(PilotSettings().model_dump(mode="json"), {"deepseek_api_key": "fixture-replacement"})
                elif operation == "delete":
                    pilot.delete_credential("deepseek_api_key")
                elif operation == "submit":
                    pilot.coordinator.submit({})
                else:
                    pilot.resume("old")
            assert pilot.credentials.get("deepseek_api_key") == "fixture-original"
        finally:
            release.set()
            start.result(timeout=2)
            pilot.coordinator.wait(timeout=2)


def test_start_cancellation_during_model_check_creates_no_run(pilot, monkeypatch):
    requested = Event()

    def verify(_directory, *, cancel):
        requested.set()
        assert cancel.is_set()

    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", verify)
    with pytest.raises(TaskCancelled):
        pilot.start("fixture materials", cancel=requested)
    assert pilot.coordinator.list_runs() == []
    assert not pilot.model_cancel.is_set()
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda *_a, **_k: None)
    run_id = pilot.start("fixture materials", "fixture materials")
    pilot.coordinator.wait(timeout=2)
    assert pilot.coordinator.get(run_id)["state"] == "succeeded"


def test_cancel_between_record_creation_and_worker_dispatch_never_calls_processor(pilot, monkeypatch):
    requested = Event()
    create = pilot.coordinator._create
    calls = []

    def create_then_cancel(*args):
        create(*args)
        requested.set()

    monkeypatch.setattr(pilot.coordinator, "_create", create_then_cancel)
    monkeypatch.setattr(pilot.coordinator, "_processor", lambda *_: calls.append("processor") or {})
    run_id = pilot.start("fixture materials", "fixture materials", cancel=requested)
    pilot.coordinator.wait(timeout=2)
    assert calls == []
    assert pilot.coordinator.get(run_id)["state"] == "cancelled"


def test_pre_cancelled_start_skips_model_io_and_admission(pilot, monkeypatch):
    requested = Event()
    requested.set()
    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", lambda *_a, **_k: pytest.fail("Cancelled model IO"))
    with pytest.raises(TaskCancelled):
        pilot.start("fixture materials", cancel=requested)
    assert pilot.coordinator.list_runs() == []


def test_start_token_remains_the_worker_cancellation_after_admission(pilot, monkeypatch):
    requested, entered = Event(), Event()

    def process(context, _payload):
        assert context.cancel_event is requested
        entered.set()
        assert context.cancel_event.wait(5)
        context.check_cancelled()

    monkeypatch.setattr(pilot.coordinator, "_processor", process)
    run_id = pilot.start("fixture materials", "fixture materials", cancel=requested)
    try:
        assert entered.wait(2)
    finally:
        requested.set()
        pilot.coordinator.wait(timeout=2)
    assert pilot.coordinator.get(run_id)["state"] == "cancelled"


def test_cancelled_resume_cannot_increment_attempt_or_call_processor(pilot):
    seed_old_cancelled_run(pilot)
    requested = Event()
    requested.set()
    with pytest.raises(TaskCancelled):
        pilot.resume("old", cancel=requested)
    assert pilot.coordinator.get("old")["attempt"] == 1
    assert pilot.coordinator.get("old")["state"] == "cancelled"


def test_configuration_reservation_does_not_hold_control_gate_during_keychain_io(pilot, monkeypatch):
    entered, release = Event(), Event()
    original = pilot.credentials.set

    def slow_set(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(pilot.credentials, "set", slow_set)
    with ThreadPoolExecutor(max_workers=2) as executor:
        configuration = executor.submit(pilot.configure, PilotSettings().model_dump(mode="json"),
                                        {"deepseek_api_key": "fixture-replacement"})
        assert entered.wait(2)
        try:
            assert executor.submit(pilot.coordinator.has_active_work).result(timeout=1)
            assert executor.submit(pilot.cancel, "no-run").result(timeout=1) is False
            assert executor.submit(pilot.coordinator.list_runs).result(timeout=1) == []
            with pytest.raises(TaskFailure):
                executor.submit(pilot.start, "fixture materials", "fixture materials").result(timeout=1)
        finally:
            release.set()
            configuration.result(timeout=2)
    assert not pilot.coordinator.has_active_work()


def test_failed_preflight_releases_admission_reservation(pilot, monkeypatch):
    def invalid_model(*_a, **_k):
        raise ValueError("fixture damaged model")

    monkeypatch.setattr("app.pilot.encoder.verify_artifacts", invalid_model)
    with pytest.raises(ValueError, match="fixture damaged"):
        pilot.start("fixture materials", "fixture materials")
    assert not pilot.coordinator.has_active_work()
    pilot.configure(PilotSettings().model_dump(mode="json"), {"deepseek_api_key": "fixture-replacement"})
    assert pilot.credentials.get("deepseek_api_key") == "fixture-replacement"


def test_close_timeout_retains_profile_lock_until_reserved_io_exits(pilot):
    entered, release = Event(), Event()

    def operation():
        with pilot.coordinator.idle_operation():
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(operation)
        assert entered.wait(2)
        try:
            with pytest.raises(TimeoutError):
                pilot.coordinator.close(timeout=0.01)
            with pytest.raises(Exception, match="используется|уже"):
                Coordinator(pilot.data_dir, lambda *_: {})
        finally:
            release.set()
            pending.result(timeout=2)
    pilot.close()
    reopened = Coordinator(pilot.data_dir, lambda *_: {})
    reopened.close()
