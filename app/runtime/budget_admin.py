"""Explicit local budget-owner reconciliation through the single writer.

An old backup cannot establish the provider's current balance or spending on
other devices. Unknown requests retain their full reservation until the owner
enters verified actual usage. Reconciliation grants only explicitly authorized
NEW headroom above all local charges/holds, never a reconstructed old balance.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.runtime.budget import BudgetLimits, BudgetService, BudgetStateError, RequestAllowance
from app.runtime.jobs import Coordinator
from app.sqlite_runtime import sqlite3

RECONCILIATION_NOTICE = (
    "Резервная копия не подтверждает текущий баланс провайдера и расходы на других устройствах. "
    "Неизвестные запросы полностью удержаны. Введите фактическое списание только после проверки у провайдера. "
    "Новый запас разрешает дополнительные расходы сверх всех сохранённых списаний и удержаний."
)


def _confirm(confirmed: bool) -> None:
    if confirmed is not True:
        raise BudgetStateError("Сверка требует явного подтверждения владельца бюджета.")


def budget_status(connection: sqlite3.Connection, *, scope_after: str = "", request_after: str = "", limit: int = 100) -> dict:
    """Bounded keyset pages; balances are local guards, never provider balances."""
    if (type(limit) is not int or not 1 <= limit <= 100 or not isinstance(scope_after, str)
            or not isinstance(request_after, str) or len(scope_after) > 160 or len(request_after) > 160):
        raise ValueError("Некорректная страница журнала бюджета.")
    budget = BudgetService(connection)
    scope_ids = connection.execute("SELECT scope_id FROM pilot_budget_scopes WHERE scope_id>? ORDER BY scope_id LIMIT ?",
                                   (scope_after, limit + 1)).fetchall()
    unknown_ids = connection.execute(
        "SELECT request_id FROM pilot_budget_requests WHERE state='unknown' AND request_id>? ORDER BY request_id LIMIT ?",
        (request_after, limit + 1)).fetchall()
    scopes: list[dict[str, Any]] = []
    for row in scope_ids[:limit]:
        snapshot = budget.snapshot(row[0])
        scopes.append(asdict(snapshot) | {"remaining": asdict(snapshot.remaining), "warning": snapshot.warning})
    unknown = []
    for row in unknown_ids[:limit]:
        reservation = budget.reservation(row[0])
        scope_refs = tuple(item[0] for item in connection.execute(
            "SELECT scope_id FROM pilot_budget_allocations WHERE request_id=? ORDER BY scope_id", (row[0],)))
        unknown.append(asdict(reservation) | {"scope_ids": scope_refs})
    return dict(restore_pending=bool(connection.execute(
        "SELECT restore_pending FROM pilot_budget_metadata WHERE singleton=1").fetchone()[0]),
        reconciliation_required=connection.execute("SELECT count(*) FROM pilot_budget_scopes WHERE reconcile=1").fetchone()[0],
        unknown_total=connection.execute("SELECT count(*) FROM pilot_budget_requests WHERE state='unknown'").fetchone()[0],
        scope_total=connection.execute("SELECT count(*) FROM pilot_budget_scopes").fetchone()[0],
        scopes=scopes, unknown_requests=unknown, notice=RECONCILIATION_NOTICE, money_unit="millionth_of_scope_currency",
        next_scope_after=scope_ids[limit - 1][0] if len(scope_ids) > limit else None,
        next_request_after=unknown_ids[limit - 1][0] if len(unknown_ids) > limit else None)


def reconcile_request(connection: sqlite3.Connection, request_id: str, *, input_tokens: int, output_tokens: int,
                      cost_micro: int, confirmed: bool = False) -> dict:
    _confirm(confirmed)
    actual = RequestAllowance(input_tokens, output_tokens, cost_micro)
    reservation = BudgetService(connection).reconcile_unknown(request_id, actual)
    return asdict(reservation)


def acknowledge_budget_restore(connection: sqlite3.Connection, scope_id: str | None, *,
                               additional_allowance: BudgetLimits | None = None, confirmed: bool = False) -> dict:
    _confirm(confirmed)
    if additional_allowance is not None and not isinstance(additional_allowance, BudgetLimits):
        raise ValueError("Новый запас бюджета должен содержать четыре целых ограничения.")
    budget = BudgetService(connection)
    pending = bool(connection.execute("SELECT restore_pending FROM pilot_budget_metadata WHERE singleton=1").fetchone()[0])
    if scope_id is not None:
        if not budget.snapshot(scope_id).requires_reconciliation:
            raise BudgetStateError("Это ограничение уже сверено. Изменение обычного лимита доступно в настройках.")
        budget.reconcile_scope(scope_id, additional_allowance=additional_allowance or BudgetLimits(0, 0, 0, 0))
    elif additional_allowance is not None:
        raise ValueError("Нельзя назначить новый запас без конкретного бюджетного ограничения.")
    elif not pending:
        raise BudgetStateError("Восстановление бюджета уже завершено.")
    remaining = connection.execute("SELECT 1 FROM pilot_budget_scopes WHERE reconcile=1 LIMIT 1").fetchone()
    if pending and remaining is None:
        budget.finish_restore_reconciliation()
    elif scope_id is None:
        raise BudgetStateError("Сначала необходимо сверить каждое восстановленное ограничение бюджета.")
    return budget_status(connection)


class BudgetAdmin:
    """UI facade: operations are fixed Python callables, not user-selected code."""

    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator

    def status(self, *, scope_after: str = "", request_after: str = "", limit: int = 100) -> dict:
        return self.coordinator.maintenance(budget_status, scope_after=scope_after, request_after=request_after, limit=limit)

    def reconcile_unknown(self, request_id: str, *, input_tokens: int, output_tokens: int, cost_micro: int,
                          confirmed: bool = False) -> dict:
        return self.coordinator.maintenance(reconcile_request, request_id, input_tokens=input_tokens,
            output_tokens=output_tokens, cost_micro=cost_micro, confirmed=confirmed)

    def acknowledge_restore(self, scope_id: str | None, *, additional_allowance: BudgetLimits | None = None,
                            confirmed: bool = False) -> dict:
        return self.coordinator.maintenance(acknowledge_budget_restore, scope_id,
            additional_allowance=additional_allowance, confirmed=confirmed)
