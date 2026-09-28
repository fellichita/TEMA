"""Durable admission, multi-scope quotas and crash/restore spending fences."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.sqlite_runtime import sqlite3

from app.runtime.budget import (
    BudgetExceeded, BudgetLimits, BudgetService, BudgetStateError, BudgetUsage, RequestAllowance, quote_microcurrency,
)


@pytest.fixture
def budget(tmp_path):
    connection = sqlite3.connect(tmp_path / "budget.sqlite3", isolation_level=None)
    service = BudgetService(connection)
    service.create_scope("run:one", BudgetLimits(3, 1_000, 200, 300_000), currency="USD")
    try:
        yield service
    finally:
        connection.close()


def test_reservation_is_durable_and_holds_call_tokens_and_money(budget, tmp_path):
    allowance = RequestAllowance(300, 50, 100_000)
    budget.reserve("request:1", ["run:one"], allowance)
    connection = sqlite3.connect(tmp_path / "budget.sqlite3")
    try:
        reopened = BudgetService(connection)
        assert reopened.reservation("request:1").state == "reserved"
        assert reopened.snapshot("run:one").used == BudgetUsage(1, 300, 50, 100_000)
        assert reopened.snapshot("run:one").remaining == BudgetLimits(2, 700, 150, 200_000)
    finally:
        connection.close()


@pytest.mark.parametrize("allowance", [
    RequestAllowance(1_001, 0, 0), RequestAllowance(0, 201, 0), RequestAllowance(0, 0, 300_001),
])
def test_insufficient_quota_reserves_nothing(budget, allowance):
    with pytest.raises(BudgetExceeded):
        budget.reserve("request:1", ["run:one"], allowance)
    assert budget.snapshot("run:one").used == BudgetUsage(0, 0, 0, 0)
    with pytest.raises(BudgetStateError):
        budget.reservation("request:1")


def test_call_limit_applies_to_zero_price_requests(budget):
    for number in range(3):
        budget.reserve(f"request:{number}", ["run:one"], RequestAllowance(0, 0, 0))
    with pytest.raises(BudgetExceeded):
        budget.reserve("request:4", ["run:one"], RequestAllowance(0, 0, 0))


def test_scope_creation_is_idempotent_but_does_not_reset_existing_quota(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(300, 20, 10_000))
    budget.create_scope("run:one", BudgetLimits(3, 1_000, 200, 300_000), currency="USD")
    assert budget.snapshot("run:one").used.calls == 1
    with pytest.raises(BudgetStateError):
        budget.create_scope("run:one", BudgetLimits(4, 1_000, 200, 300_000), currency="USD")


def test_atomic_reservation_across_run_and_daily_limits(budget):
    budget.create_scope("day:2026-09-10", BudgetLimits(24, 10_000, 2_000, 1), currency="USD")
    with pytest.raises(BudgetExceeded):
        budget.reserve("request:1", ["run:one", "day:2026-09-10"], RequestAllowance(1, 1, 2))
    assert budget.snapshot("run:one").used.calls == 0
    assert budget.snapshot("day:2026-09-10").used.calls == 0


def test_mixed_currencies_are_not_summed(budget):
    budget.create_scope("ruble:one", BudgetLimits(10, 100, 100, 10_000), currency="RUB")
    with pytest.raises(BudgetStateError):
        budget.reserve("request:1", ["run:one", "ruble:one"], RequestAllowance(1, 1, 2))
    assert budget.snapshot("run:one").used.calls == 0


def test_dispatch_is_one_time_even_after_process_restart(budget, tmp_path):
    budget.reserve("request:1", ["run:one"], RequestAllowance(100, 20, 10_000))
    budget.mark_sent("request:1")
    connection = sqlite3.connect(tmp_path / "budget.sqlite3")
    try:
        reopened = BudgetService(connection)
        with pytest.raises(BudgetStateError):
            reopened.mark_sent("request:1")
        with pytest.raises(BudgetStateError):
            reopened.reserve("request:1", ["run:one"], RequestAllowance(1, 1, 1))
    finally:
        connection.close()


def test_cancel_before_send_refunds_and_cannot_reuse_request_identity(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(500, 100, 100_000))
    assert budget.cancel("request:1").state == "released"
    assert budget.cancel("request:1").state == "released"
    assert budget.snapshot("run:one").used == BudgetUsage(0, 0, 0, 0)
    with pytest.raises(BudgetStateError):
        budget.mark_sent("request:1")
    with pytest.raises(BudgetStateError):
        budget.reserve("request:1", ["run:one"], RequestAllowance(500, 100, 100_000))


def test_cancel_after_send_and_timeout_hold_full_reservation(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(500, 100, 100_000))
    budget.mark_sent("request:1")
    assert budget.cancel("request:1").state == "unknown"
    assert budget.mark_unknown("request:1").state == "unknown"
    assert budget.snapshot("run:one").used == BudgetUsage(1, 500, 100, 100_000)
    with pytest.raises(BudgetStateError):
        budget.settle("request:1", RequestAllowance(0, 0, 0))
    budget.reconcile_unknown("request:1", RequestAllowance(200, 50, 30_000))
    assert budget.snapshot("run:one").used == BudgetUsage(1, 200, 50, 30_000)


def test_interrupted_sent_requests_become_unknown_without_refund(budget):
    budget.reserve("sent:1", ["run:one"], RequestAllowance(500, 100, 100_000))
    budget.reserve("unsent:2", ["run:one"], RequestAllowance(100, 10, 10_000))
    budget.mark_sent("sent:1")
    assert budget.recover_interrupted() == 1
    assert budget.recover_interrupted() == 0
    assert budget.reservation("sent:1").state == "unknown"
    assert budget.reservation("unsent:2").state == "reserved"
    assert budget.snapshot("run:one").used.cost_micro == 110_000


def test_valid_settlement_releases_unused_funds_but_keeps_one_call(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(500, 100, 100_000))
    budget.mark_sent("request:1")
    settled = budget.settle("request:1", RequestAllowance(201, 37, 51_200))
    assert settled.state == "settled"
    assert not settled.exceeded_reservation
    assert budget.snapshot("run:one").used == BudgetUsage(1, 201, 37, 51_200)
    assert budget.cancel("request:1").state == "settled"
    assert budget.snapshot("run:one").used.cost_micro == 51_200
    with pytest.raises(BudgetStateError):
        budget.settle("request:1", RequestAllowance(201, 37, 51_200))


@pytest.mark.parametrize("actual", [
    RequestAllowance(101, 20, 10_000), RequestAllowance(100, 21, 10_000), RequestAllowance(100, 20, 10_001),
])
def test_real_provider_overrun_is_recorded_then_further_spending_blocked(budget, actual):
    budget.reserve("request:1", ["run:one"], RequestAllowance(100, 20, 10_000))
    budget.mark_sent("request:1")
    result = budget.settle("request:1", actual)
    assert result.exceeded_reservation
    assert result.charged == actual
    assert budget.snapshot("run:one").requires_reconciliation
    with pytest.raises(BudgetStateError):
        budget.reserve("request:2", ["run:one"], RequestAllowance(1, 1, 1))


def test_backup_restore_blocks_existing_and_new_periods_until_explicit_reconciliation(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(100, 20, 10_000))
    budget.mark_restored()
    assert budget.reservation("request:1").state == "unknown"
    assert budget.snapshot("run:one").requires_reconciliation
    with pytest.raises(BudgetStateError):
        budget.reserve("request:2", ["run:one"], RequestAllowance(1, 1, 1))
    budget.create_scope("day:tomorrow", BudgetLimits(24, 100_000, 18_000, 500_000), currency="USD")
    with pytest.raises(BudgetStateError):
        budget.reserve("request:3", ["day:tomorrow"], RequestAllowance(1, 1, 1))
    with pytest.raises(BudgetStateError):
        budget.finish_restore_reconciliation()
    budget.reconcile_scope("run:one", additional_allowance=BudgetLimits(1, 200, 30, 20_000))
    assert budget.snapshot("run:one").remaining == BudgetLimits(1, 200, 30, 20_000)
    budget.reconcile_scope("day:tomorrow", additional_allowance=BudgetLimits(0, 0, 0, 0))
    budget.finish_restore_reconciliation()
    budget.reserve("request:4", ["run:one"], RequestAllowance(200, 30, 20_000))
    with pytest.raises(BudgetExceeded):
        budget.reserve("request:5", ["run:one"], RequestAllowance(1, 1, 1))
    assert budget.reservation("request:1").state == "unknown"


def test_unreconciled_scope_cannot_dispatch_previously_reserved_request(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(100, 20, 10_000))
    budget.reserve("request:2", ["run:one"], RequestAllowance(100, 20, 10_000))
    budget.mark_sent("request:1")
    budget.settle("request:1", RequestAllowance(100, 20, 20_000))
    with pytest.raises(BudgetStateError):
        budget.mark_sent("request:2")
    assert budget.cancel("request:2").state == "released"


def test_parallel_connections_cannot_both_spend_last_available_amount(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    initial = sqlite3.connect(path, isolation_level=None)
    BudgetService(initial).create_scope("day:one", BudgetLimits(10, 100, 100, 10), currency="USD")
    initial.close()
    barrier = threading.Barrier(2)

    def reserve(number):
        connection = sqlite3.connect(path, timeout=10, isolation_level=None)
        try:
            service = BudgetService(connection)
            barrier.wait(timeout=5)
            try:
                service.reserve(f"request:{number}", ["day:one"], RequestAllowance(1, 1, 10))
                return "reserved"
            except BudgetExceeded:
                return "refused"
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(reserve, (1, 2))) == ["refused", "reserved"]
    verify = sqlite3.connect(path)
    try:
        assert BudgetService(verify).snapshot("day:one").used.cost_micro == 10
    finally:
        verify.close()


def test_existing_caller_transaction_is_never_committed_by_budget(budget):
    budget.connection.execute("BEGIN")
    with pytest.raises(BudgetStateError):
        budget.reserve("request:1", ["run:one"], RequestAllowance(1, 1, 1))
    assert budget.connection.in_transaction
    budget.connection.rollback()


def test_future_schema_rejected_and_transaction_rolled_back(budget):
    budget.connection.execute("UPDATE pilot_budget_metadata SET schema_version=99")
    with pytest.raises(BudgetStateError):
        BudgetService(budget.connection)
    assert not budget.connection.in_transaction


@pytest.mark.parametrize("bad", [-1, True, 1.2, float("nan"), 10**12 + 1])
def test_resources_require_bounded_integers(bad):
    with pytest.raises(ValueError):
        RequestAllowance(bad, 0, 0)
    with pytest.raises(ValueError):
        BudgetLimits(0, 0, 0, bad)


@pytest.mark.parametrize("scopes", [[], "run:one", ["run:one", "run:one"], ["x"] * 9])
def test_scope_list_must_be_bounded_and_unique(budget, scopes):
    with pytest.raises(ValueError):
        budget.reserve("request:1", scopes, RequestAllowance(1, 1, 1))


def test_price_quote_rounds_up_without_float_currency():
    assert quote_microcurrency(100_000, 18_000, input_per_million=440_000, output_per_million=1_320_000) == 67_760
    assert quote_microcurrency(1, 1, input_per_million=1, output_per_million=1) == 2
    assert quote_microcurrency(0, 0, input_per_million=440_000, output_per_million=1_320_000) == 0


def test_warning_is_triggered_at_eighty_percent(budget):
    assert not budget.snapshot("run:one").warning
    budget.reserve("request:1", ["run:one"], RequestAllowance(800, 0, 0))
    assert budget.snapshot("run:one").warning


def test_limit_update_preserves_settled_and_unknown_usage_and_currency(budget):
    budget.reserve("paid:1", ["run:one"], RequestAllowance(200, 30, 20_000))
    budget.mark_sent("paid:1")
    budget.settle("paid:1", RequestAllowance(100, 20, 10_000))
    budget.reserve("unknown:2", ["run:one"], RequestAllowance(200, 30, 20_000))
    budget.mark_sent("unknown:2")
    budget.mark_unknown("unknown:2")
    updated = budget.update_limits("run:one", BudgetLimits(10, 2_000, 500, 500_000))
    assert updated.currency == "USD"
    assert updated.used == BudgetUsage(2, 300, 50, 30_000)
    assert updated.remaining == BudgetLimits(8, 1_700, 450, 470_000)
    assert budget.reservation("unknown:2").state == "unknown"


def test_lowered_limit_blocks_new_reservations_without_erasing_holds(budget):
    budget.reserve("held:1", ["run:one"], RequestAllowance(200, 30, 20_000))
    changed = budget.update_limits("run:one", BudgetLimits(0, 100, 10, 1))
    assert changed.used == BudgetUsage(1, 200, 30, 20_000)
    assert changed.remaining == BudgetLimits(0, 0, 0, 0)
    with pytest.raises(BudgetExceeded):
        budget.reserve("new:2", ["run:one"], RequestAllowance(0, 0, 0))
    assert budget.reservation("held:1").state == "reserved"


def test_changing_settings_cannot_clear_restoration_reconciliation(budget):
    budget.reserve("request:1", ["run:one"], RequestAllowance(10, 5, 100))
    budget.mark_restored()
    changed = budget.update_limits("run:one", BudgetLimits(100, 100_000, 10_000, 1_000_000))
    assert changed.requires_reconciliation
    assert budget.reservation("request:1").state == "unknown"
    with pytest.raises(BudgetStateError):
        budget.reserve("new:2", ["run:one"], RequestAllowance(1, 1, 1))


def test_limit_update_requires_existing_scope(budget):
    with pytest.raises(BudgetStateError):
        budget.update_limits("missing", BudgetLimits(1, 1, 1, 1))


def test_aggregate_actual_overruns_remain_readable_and_conservative_reconciliation_possible(budget):
    for number in range(2):
        request = f"overrun:{number}"
        budget.reserve(request, ["run:one"], RequestAllowance(1, 1, 1))
        budget.mark_sent(request)
    for number in range(2):
        request = f"overrun:{number}"
        budget.mark_unknown(request)
        budget.reconcile_unknown(request, RequestAllowance(1, 1, 10**12))
    snapshot = budget.snapshot("run:one")
    assert snapshot.used.cost_micro == 2 * 10**12
    assert snapshot.remaining.cost_micro == 0
    assert snapshot.requires_reconciliation
    assert snapshot.warning
    budget.mark_restored()
    before = budget.connection.execute("SELECT count(*) FROM pilot_budget_events").fetchone()[0]
    with pytest.raises(BudgetStateError, match="без новых расходов"):
        budget.reconcile_scope("run:one", additional_allowance=BudgetLimits(0, 0, 0, 1))
    assert budget.snapshot("run:one").requires_reconciliation
    assert budget.connection.execute("SELECT count(*) FROM pilot_budget_events").fetchone()[0] == before
    reconciled = budget.reconcile_scope("run:one", additional_allowance=BudgetLimits(0, 0, 0, 0))
    assert reconciled.used == BudgetUsage(2, 2, 2, 2 * 10**12)
    assert reconciled.remaining == BudgetLimits(0, 0, 0, 0)
    budget.finish_restore_reconciliation()
    with pytest.raises(BudgetExceeded):
        budget.reserve("extra", ["run:one"], RequestAllowance(0, 0, 0))
    # Restoring the ability to open a future period never rewrites historical use.
    budget.create_scope("day:new", BudgetLimits(1, 1, 1, 1), currency="USD")
    budget.reserve("future", ["day:new"], RequestAllowance(1, 1, 1))
    assert budget.snapshot("run:one").used.cost_micro == 2 * 10**12
