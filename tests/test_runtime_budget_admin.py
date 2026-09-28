"""Owner reconciliation never infers current provider credit from an old backup."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event

import pytest

from app.runtime.budget import BudgetLimits, BudgetService, BudgetStateError, RequestAllowance
from app.runtime.budget_admin import BudgetAdmin, budget_status
from app.runtime.jobs import Coordinator, TaskFailure
from app.sqlite_runtime import sqlite3


@pytest.fixture
def restored(tmp_path):
    def initialize(context, _):
        budget = BudgetService(context.connection)
        for scope in ("day:one", "run:one"):
            budget.create_scope(scope, BudgetLimits(20, 10_000, 5_000, 300_000), currency="USD")
        budget.reserve("unknown:one", ["day:one", "run:one"], RequestAllowance(200, 30, 20_000))
        budget.mark_sent("unknown:one")
        budget.mark_restored()
        return {}

    coordinator = Coordinator(tmp_path, initialize)
    coordinator.submit({})
    coordinator.wait()
    try:
        yield coordinator, BudgetAdmin(coordinator)
    finally:
        coordinator.close()


def test_status_is_bounded_and_preserves_unknown_holds_and_currency(restored):
    coordinator, admin = restored
    status = admin.status(limit=1)
    assert status["restore_pending"] and status["reconciliation_required"] == 2
    assert status["scope_total"] == 2 and status["unknown_total"] == 1
    assert len(status["scopes"]) == 1 and status["scopes"][0]["currency"] == "USD"
    assert status["unknown_requests"][0]["charged"]["cost_micro"] == 20_000
    assert status["unknown_requests"][0]["state"] == "unknown"
    assert set(status["unknown_requests"][0]["scope_ids"]) == {"day:one", "run:one"}
    second = admin.status(scope_after=status["next_scope_after"], limit=1)
    assert second["scopes"][0]["scope_id"] != status["scopes"][0]["scope_id"]
    assert second["next_scope_after"] is None
    assert "не подтверждает текущий баланс" in status["notice"]
    assert json.loads(json.dumps(status))  # Status is a small UI-safe JSON response.
    with pytest.raises(sqlite3.ProgrammingError):
        coordinator._connection.execute("SELECT 1")


@pytest.mark.parametrize("confirmation", [False, 1, "true", None])
def test_no_actual_or_restoration_change_without_explicit_boolean_confirmation(restored, confirmation):
    _, admin = restored
    with pytest.raises(BudgetStateError, match="подтверждения"):
        admin.reconcile_unknown("unknown:one", input_tokens=0, output_tokens=0, cost_micro=0, confirmed=confirmation)
    with pytest.raises(BudgetStateError, match="подтверждения"):
        admin.acknowledge_restore("day:one", confirmed=confirmation)
    assert admin.status()["unknown_requests"][0]["charged"]["cost_micro"] == 20_000
    assert admin.status()["reconciliation_required"] == 2


def test_verified_actual_usage_is_audited_and_cannot_be_replayed(restored):
    coordinator, admin = restored
    settled = admin.reconcile_unknown("unknown:one", input_tokens=120, output_tokens=12, cost_micro=15_000, confirmed=True)
    assert settled["state"] == "settled" and settled["charged"]["cost_micro"] == 15_000
    assert admin.status()["restore_pending"] and admin.status()["unknown_total"] == 0
    with pytest.raises(BudgetStateError):
        admin.reconcile_unknown("unknown:one", input_tokens=0, output_tokens=0, cost_micro=0, confirmed=True)
    connection = sqlite3.connect(coordinator.path)
    try:
        event = connection.execute("SELECT action FROM pilot_budget_events WHERE action LIKE 'reconciled_actual:%'").fetchone()[0]
        assert json.loads(event.split(":", 1)[1]) == dict(input_tokens=120, output_tokens=12, cost_micro=15_000, currency="USD")
    finally:
        connection.close()


def test_actual_zero_requires_explicit_input_and_confirmation(restored):
    _, admin = restored
    with pytest.raises(TypeError):
        admin.reconcile_unknown("unknown:one", confirmed=True)
    assert admin.status()["unknown_total"] == 1
    response = admin.reconcile_unknown("unknown:one", input_tokens=0, output_tokens=0, cost_micro=0, confirmed=True)
    assert response["charged"] == dict(input_tokens=0, output_tokens=0, cost_micro=0)
    assert admin.status()["restore_pending"]


def test_conservative_acknowledgement_grants_no_new_spending_and_never_drops_unknown_hold(restored):
    coordinator, admin = restored
    first = admin.acknowledge_restore("day:one", confirmed=True)
    assert first["restore_pending"] and first["reconciliation_required"] == 1
    day = next(item for item in first["scopes"] if item["scope_id"] == "day:one")
    assert day["limits"] == day["used"] and day["remaining"]["cost_micro"] == 0
    with pytest.raises(BudgetStateError, match="каждое"):
        admin.acknowledge_restore(None, confirmed=True)
    final = admin.acknowledge_restore("run:one", confirmed=True)
    assert not final["restore_pending"] and final["reconciliation_required"] == 0
    assert final["unknown_requests"][0]["charged"]["cost_micro"] == 20_000
    assert final["unknown_requests"][0]["state"] == "unknown"
    assert all(scope["remaining"] == dict(calls=0, input_tokens=0, output_tokens=0, cost_micro=0) for scope in final["scopes"])
    connection = sqlite3.connect(coordinator.path)
    try:
        assert connection.execute("SELECT count(*) FROM pilot_budget_events WHERE action LIKE 'reconciled_headroom:%'").fetchone()[0] == 2
    finally:
        connection.close()


def test_explicit_new_headroom_is_added_above_all_charges_not_old_backup_limit(restored):
    _, admin = restored
    status = admin.acknowledge_restore("day:one", additional_allowance=BudgetLimits(2, 100, 50, 1_000), confirmed=True)
    day = next(item for item in status["scopes"] if item["scope_id"] == "day:one")
    assert day["limits"] == dict(calls=3, input_tokens=300, output_tokens=80, cost_micro=21_000)
    assert day["remaining"] == dict(calls=2, input_tokens=100, output_tokens=50, cost_micro=1_000)
    assert day["used"]["cost_micro"] == 20_000


def test_empty_restored_database_still_requires_explicit_owner_acknowledgement(tmp_path):
    def initialize(context, _):
        BudgetService(context.connection).mark_restored()
        return {}

    coordinator = Coordinator(tmp_path, initialize)
    try:
        coordinator.submit({})
        coordinator.wait()
        admin = BudgetAdmin(coordinator)
        assert admin.status()["restore_pending"]
        with pytest.raises(BudgetStateError):
            admin.acknowledge_restore(None)
        assert not admin.acknowledge_restore(None, confirmed=True)["restore_pending"]
    finally:
        coordinator.close()


def test_concurrent_reconciliation_cannot_grant_headroom_twice(restored):
    _, admin = restored

    def acknowledge():
        try:
            admin.acknowledge_restore("day:one", additional_allowance=BudgetLimits(1, 100, 10, 1_000), confirmed=True)
            return True
        except BudgetStateError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: acknowledge(), range(2)))
    assert sorted(results) == [False, True]
    day = next(item for item in admin.status()["scopes"] if item["scope_id"] == "day:one")
    assert day["limits"]["cost_micro"] == 21_000


def test_maintenance_refuses_active_analysis_and_user_selected_code(tmp_path):
    entered, proceed = Event(), Event()

    def processor(*_):
        entered.set()
        assert proceed.wait(5)
        return {}

    coordinator = Coordinator(tmp_path, processor)
    try:
        for operation in ("budget_status", lambda connection: {}, {"module": "app.runtime.budget_admin"}):
            with pytest.raises(ValueError, match="доверенная"):
                coordinator.maintenance(operation)
        coordinator.submit({})
        assert entered.wait(5)
        with pytest.raises(TaskFailure, match="Дождитесь"):
            BudgetAdmin(coordinator).status()
        proceed.set()
        coordinator.wait()
        assert coordinator.maintenance(budget_status)["scope_total"] == 0
    finally:
        proceed.set()
        coordinator.close()


def test_reconciliation_audit_failure_rolls_back_actual_charge(restored):
    coordinator, admin = restored
    coordinator._executor.submit(lambda: coordinator._connection.execute("""
        CREATE TRIGGER reject_audit BEFORE INSERT ON pilot_budget_events
        WHEN NEW.action LIKE 'reconciled_actual:%' BEGIN SELECT RAISE(ABORT,'audit unavailable'); END
    """)).result()
    with pytest.raises(sqlite3.IntegrityError):
        admin.reconcile_unknown("unknown:one", input_tokens=100, output_tokens=10, cost_micro=1_000, confirmed=True)
    status = admin.status()
    assert status["unknown_total"] == 1
    assert status["unknown_requests"][0]["charged"]["cost_micro"] == 20_000
