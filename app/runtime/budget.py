"""Durable, conservative admission control for provider requests.

The coordinator owns the supplied SQLite connection. Every operation uses a
short transaction and rejects an already-open caller transaction. No HTTP call
belongs inside those transactions. ``mark_sent`` must succeed immediately before
network dispatch; a crash or timeout after that point keeps the reservation.

Money is integer millionths of the scope currency, never floating point. This is
a local spending guard, not a provider billing authority or cross-device quota.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from app.sqlite_runtime import sqlite3

_MAX_AMOUNT = 10**12
_IDENTIFIER = re.compile(r"[A-Za-z0-9_./:-]{1,160}\Z")
_STATES = frozenset({"reserved", "sent", "unknown", "settled", "released"})


class BudgetError(RuntimeError):
    """A safe budget refusal without credentials or provider payloads."""


class BudgetExceeded(BudgetError):
    pass


class BudgetStateError(BudgetError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    calls: int
    input_tokens: int
    output_tokens: int
    cost_micro: int

    def __post_init__(self) -> None:
        for value in (self.calls, self.input_tokens, self.output_tokens, self.cost_micro):
            _amount(value)


@dataclass(frozen=True)
class RequestAllowance:
    input_tokens: int
    output_tokens: int
    cost_micro: int

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens, self.cost_micro):
            _amount(value)


@dataclass(frozen=True)
class BudgetUsage:
    """Ledger totals may exceed any single configured limit or request charge."""

    calls: int
    input_tokens: int
    output_tokens: int
    cost_micro: int

    def __post_init__(self) -> None:
        for value in (self.calls, self.input_tokens, self.output_tokens, self.cost_micro):
            if type(value) is not int or value < 0:
                raise ValueError("Итог журнала должен быть целым неотрицательным числом.")


@dataclass(frozen=True)
class BudgetSnapshot:
    scope_id: str
    currency: str
    limits: BudgetLimits
    used: BudgetUsage
    requires_reconciliation: bool

    @property
    def remaining(self) -> BudgetLimits:
        return BudgetLimits(*(max(0, limit - used) for limit, used in zip(
            _values(self.limits), _values(self.used), strict=True
        )))

    @property
    def warning(self) -> bool:
        return any(limit > 0 and used * 5 >= limit * 4 for limit, used in zip(
            _values(self.limits), _values(self.used), strict=True
        ))


@dataclass(frozen=True)
class BudgetReservation:
    request_id: str
    currency: str
    state: str
    reserved: RequestAllowance
    charged: RequestAllowance
    exceeded_reservation: bool


def _amount(value: int) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_AMOUNT:
        raise ValueError("Лимит должен быть целым неотрицательным числом не более 10¹².")


def _identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("Некорректный идентификатор бюджетной операции.")


def _values(limits: BudgetLimits | BudgetUsage) -> tuple[int, int, int, int]:
    return limits.calls, limits.input_tokens, limits.output_tokens, limits.cost_micro


def quote_microcurrency(
    input_tokens: int, output_tokens: int, *, input_per_million: int, output_per_million: int,
) -> int:
    """Round each independently billable token component upward to a micro-unit.

    Rates themselves are microcurrency per million tokens. Cache discounts are
    intentionally ignored when reserving: use the maximum applicable tariff.
    """
    for value in (input_tokens, output_tokens, input_per_million, output_per_million):
        _amount(value)
    price = (input_tokens * input_per_million + 999_999) // 1_000_000
    price += (output_tokens * output_per_million + 999_999) // 1_000_000
    _amount(price)
    return price


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS pilot_budget_metadata (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), schema_version INTEGER NOT NULL,
        restore_pending INTEGER NOT NULL DEFAULT 0 CHECK(restore_pending IN (0,1)))""",
    """CREATE TABLE IF NOT EXISTS pilot_budget_scopes (
        scope_id TEXT PRIMARY KEY, currency TEXT NOT NULL,
        max_calls INTEGER NOT NULL CHECK(max_calls>=0),
        max_input INTEGER NOT NULL CHECK(max_input>=0),
        max_output INTEGER NOT NULL CHECK(max_output>=0),
        max_cost INTEGER NOT NULL CHECK(max_cost>=0),
        reconcile INTEGER NOT NULL DEFAULT 0 CHECK(reconcile IN (0,1)))""",
    """CREATE TABLE IF NOT EXISTS pilot_budget_requests (
        request_id TEXT PRIMARY KEY, currency TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('reserved','sent','unknown','settled','released')),
        reserved_input INTEGER NOT NULL CHECK(reserved_input>=0),
        reserved_output INTEGER NOT NULL CHECK(reserved_output>=0),
        reserved_cost INTEGER NOT NULL CHECK(reserved_cost>=0),
        charged_input INTEGER NOT NULL CHECK(charged_input>=0),
        charged_output INTEGER NOT NULL CHECK(charged_output>=0),
        charged_cost INTEGER NOT NULL CHECK(charged_cost>=0),
        overrun INTEGER NOT NULL DEFAULT 0 CHECK(overrun IN (0,1)),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS pilot_budget_allocations (
        scope_id TEXT NOT NULL REFERENCES pilot_budget_scopes(scope_id),
        request_id TEXT NOT NULL REFERENCES pilot_budget_requests(request_id),
        PRIMARY KEY(scope_id,request_id))""",
    """CREATE TABLE IF NOT EXISTS pilot_budget_events (
        id INTEGER PRIMARY KEY, subject_id TEXT NOT NULL,
        action TEXT NOT NULL, created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_pilot_budget_requests_state ON pilot_budget_requests(state)",
    "CREATE INDEX IF NOT EXISTS ix_pilot_budget_allocation_request ON pilot_budget_allocations(request_id)",
)


class BudgetService:
    """Use a connection on its coordinator thread, with foreign keys enabled.

    The connection remains owned by the caller and is never closed here. Separate
    connections/processes are serialized by SQLite's BEGIN IMMEDIATE, so checking
    remaining funds and reserving them is one atomic operation. Request identifiers
    must be stable operation IDs; they cannot be reused, including after cancellation.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        if self.connection.in_transaction:
            raise BudgetStateError("Бюджет нельзя инициализировать внутри открытой транзакции.")
        self.connection.execute("PRAGMA foreign_keys=ON")
        with self._transaction():
            for statement in _SCHEMA:
                self.connection.execute(statement)
            self.connection.execute(
                "INSERT OR IGNORE INTO pilot_budget_metadata VALUES (1,1,0)"
            )
            version = self.connection.execute(
                "SELECT schema_version FROM pilot_budget_metadata WHERE singleton=1"
            ).fetchone()[0]
            if version != 1:
                raise BudgetStateError("Версия журнала бюджета не поддерживается.")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        if self.connection.in_transaction:
            raise BudgetStateError("Операция бюджета требует отдельной короткой транзакции.")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _event(self, subject: str, action: str) -> None:
        self.connection.execute(
            "INSERT INTO pilot_budget_events(subject_id,action,created_at) VALUES (?,?,?)",
            (subject, action, datetime.now(UTC).isoformat()),
        )

    def create_scope(self, scope_id: str, limits: BudgetLimits, *, currency: str) -> None:
        _identifier(scope_id)
        if not isinstance(currency, str) or re.fullmatch(r"[A-Z]{3}", currency) is None:
            raise ValueError("Требуется трёхбуквенный код валюты.")
        with self._transaction():
            existing = self.connection.execute(
                "SELECT currency,max_calls,max_input,max_output,max_cost FROM pilot_budget_scopes WHERE scope_id=?",
                (scope_id,),
            ).fetchone()
            expected = (currency, *_values(limits))
            if existing is not None:
                if tuple(existing) != expected:
                    raise BudgetStateError("Бюджет уже существует с другими параметрами.")
                return
            restored = self.connection.execute(
                "SELECT restore_pending FROM pilot_budget_metadata WHERE singleton=1"
            ).fetchone()[0]
            self.connection.execute(
                "INSERT INTO pilot_budget_scopes VALUES (?,?,?,?,?,?,?)", (scope_id, *expected, restored),
            )
            self._event(scope_id, "scope_created")

    def _snapshot(self, scope_id: str) -> BudgetSnapshot:
        row = self.connection.execute(
            "SELECT currency,max_calls,max_input,max_output,max_cost,reconcile "
            "FROM pilot_budget_scopes WHERE scope_id=?", (scope_id,),
        ).fetchone()
        if row is None:
            raise BudgetStateError("Бюджетный период не найден.")
        used = self.connection.execute(
            """SELECT COALESCE(SUM(CASE WHEN r.state='released' THEN 0 ELSE 1 END),0),
                COALESCE(SUM(r.charged_input),0),COALESCE(SUM(r.charged_output),0),
                COALESCE(SUM(r.charged_cost),0)
                FROM pilot_budget_requests r JOIN pilot_budget_allocations a USING(request_id)
                WHERE a.scope_id=?""", (scope_id,),
        ).fetchone()
        return BudgetSnapshot(scope_id, row[0], BudgetLimits(*row[1:5]), BudgetUsage(*used), bool(row[5]))

    def update_limits(self, scope_id: str, limits: BudgetLimits) -> BudgetSnapshot:
        """Apply explicit configuration changes without refunding earlier spending.

        Currency, sent/unknown reservations and reconciliation fences are retained.
        Lowering a cap below current use is allowed: subsequent reservations fail
        until sufficient headroom is explicitly configured. A running analysis's
        frozen scope is not updated by the settings UI; only future admissions
        should adopt changed configuration.
        """
        _identifier(scope_id)
        with self._transaction():
            self._snapshot(scope_id)  # Require an existing scope; never invent a currency.
            self.connection.execute(
                "UPDATE pilot_budget_scopes SET max_calls=?,max_input=?,max_output=?,max_cost=? WHERE scope_id=?",
                (*_values(limits), scope_id),
            )
            self._event(scope_id, "limits_updated")
            return self._snapshot(scope_id)

    def snapshot(self, scope_id: str) -> BudgetSnapshot:
        _identifier(scope_id)
        # Read both the limits and ledger in one consistent snapshot.
        with self._transaction():
            return self._snapshot(scope_id)

    def _reservation(self, request_id: str) -> BudgetReservation:
        row = self.connection.execute(
            "SELECT currency,state,reserved_input,reserved_output,reserved_cost,"
            "charged_input,charged_output,charged_cost,overrun "
            "FROM pilot_budget_requests WHERE request_id=?", (request_id,),
        ).fetchone()
        if row is None:
            raise BudgetStateError("Резерв запроса не найден.")
        if row[1] not in _STATES:
            raise BudgetStateError("Некорректное состояние журнала бюджета.")
        return BudgetReservation(
            request_id, row[0], row[1], RequestAllowance(*row[2:5]), RequestAllowance(*row[5:8]), bool(row[8]),
        )

    def reservation(self, request_id: str) -> BudgetReservation:
        _identifier(request_id)
        return self._reservation(request_id)

    def reserve(
        self, request_id: str, scope_ids: Sequence[str], allowance: RequestAllowance,
    ) -> BudgetReservation:
        _identifier(request_id)
        if isinstance(scope_ids, (str, bytes)) or not 1 <= len(scope_ids) <= 8:
            raise ValueError("Запрос должен иметь от одного до восьми бюджетных ограничений.")
        if len(set(scope_ids)) != len(scope_ids):
            raise ValueError("Бюджетные ограничения не должны повторяться.")
        for scope_id in scope_ids:
            _identifier(scope_id)
        with self._transaction():
            if self.connection.execute(
                "SELECT 1 FROM pilot_budget_requests WHERE request_id=?", (request_id,),
            ).fetchone() is not None:
                raise BudgetStateError("Этот запрос уже зарегистрирован. Автоматическая повторная отправка запрещена.")
            currency: str | None = None
            requested = (1, allowance.input_tokens, allowance.output_tokens, allowance.cost_micro)
            for scope_id in scope_ids:
                snapshot = self._snapshot(scope_id)
                if snapshot.requires_reconciliation:
                    raise BudgetStateError("Бюджет требует сверки перед новыми запросами.")
                if currency is not None and snapshot.currency != currency:
                    raise BudgetStateError("Ограничения одного запроса должны использовать одну валюту.")
                currency = snapshot.currency
                if any(used + extra > limit for used, extra, limit in zip(
                    _values(snapshot.used), requested, _values(snapshot.limits), strict=True,
                )):
                    raise BudgetExceeded("Лимит запросов, токенов или стоимости исчерпан.")
            now = datetime.now(UTC).isoformat()
            values = (allowance.input_tokens, allowance.output_tokens, allowance.cost_micro)
            self.connection.execute(
                "INSERT INTO pilot_budget_requests VALUES (?,?, 'reserved',?,?,?,?,?,?,0,?,?)",
                (request_id, currency, *values, *values, now, now),
            )
            self.connection.executemany(
                "INSERT INTO pilot_budget_allocations VALUES (?,?)", ((scope, request_id) for scope in scope_ids),
            )
            self._event(request_id, "reserved")
            return self._reservation(request_id)

    def _transition(self, request_id: str, state: str) -> None:
        self.connection.execute(
            "UPDATE pilot_budget_requests SET state=?,updated_at=? WHERE request_id=?",
            (state, datetime.now(UTC).isoformat(), request_id),
        )
        self._event(request_id, state)

    def mark_sent(self, request_id: str) -> None:
        """Acquire the one-time dispatch fence BEFORE sending anything externally."""
        _identifier(request_id)
        with self._transaction():
            if self._reservation(request_id).state != "reserved":
                raise BudgetStateError("Запрос уже отправлен или завершён; повторная отправка запрещена.")
            scopes = self.connection.execute(
                "SELECT s.reconcile FROM pilot_budget_scopes s "
                "JOIN pilot_budget_allocations a USING(scope_id) WHERE a.request_id=?", (request_id,),
            ).fetchall()
            if any(row[0] for row in scopes):
                raise BudgetStateError("Отправка запрещена до сверки бюджета.")
            self._transition(request_id, "sent")

    def cancel(self, request_id: str) -> BudgetReservation:
        """Refund only a proven-unsent request; a sent request becomes unknown."""
        _identifier(request_id)
        with self._transaction():
            state = self._reservation(request_id).state
            if state == "reserved":
                self.connection.execute(
                    "UPDATE pilot_budget_requests SET charged_input=0,charged_output=0,charged_cost=0 "
                    "WHERE request_id=?", (request_id,),
                )
                self._transition(request_id, "released")
            elif state == "sent":
                self._transition(request_id, "unknown")
            return self._reservation(request_id)

    def mark_unknown(self, request_id: str) -> BudgetReservation:
        _identifier(request_id)
        with self._transaction():
            state = self._reservation(request_id).state
            if state not in {"sent", "unknown"}:
                raise BudgetStateError("Неопределённый расход допустим только после отправки.")
            if state == "sent":
                self._transition(request_id, "unknown")
            return self._reservation(request_id)

    def _settle(self, request_id: str, actual: RequestAllowance, *, expected_state: str) -> BudgetReservation:
        _identifier(request_id)
        with self._transaction():
            reservation = self._reservation(request_id)
            if reservation.state != expected_state:
                raise BudgetStateError("Списание не соответствует текущему состоянию запроса.")
            overrun = any(actual_value > reserved for actual_value, reserved in zip(
                (actual.input_tokens, actual.output_tokens, actual.cost_micro),
                (reservation.reserved.input_tokens, reservation.reserved.output_tokens, reservation.reserved.cost_micro),
                strict=True,
            ))
            self.connection.execute(
                "UPDATE pilot_budget_requests SET charged_input=?,charged_output=?,charged_cost=?,overrun=? "
                "WHERE request_id=?",
                (actual.input_tokens, actual.output_tokens, actual.cost_micro, int(overrun), request_id),
            )
            if overrun:
                # Never conceal a real provider overrun by refusing to record it.
                self.connection.execute(
                    "UPDATE pilot_budget_scopes SET reconcile=1 WHERE scope_id IN "
                    "(SELECT scope_id FROM pilot_budget_allocations WHERE request_id=?)", (request_id,),
                )
            self._transition(request_id, "settled")
            if expected_state == "unknown":
                self._event(request_id, "unknown_reconciled")
                self._event(request_id, "reconciled_actual:" + json.dumps({
                    "input_tokens": actual.input_tokens, "output_tokens": actual.output_tokens,
                    "cost_micro": actual.cost_micro, "currency": reservation.currency,
                }, sort_keys=True, separators=(",", ":")))
            return self._reservation(request_id)

    def settle(self, request_id: str, actual: RequestAllowance) -> BudgetReservation:
        return self._settle(request_id, actual, expected_state="sent")

    def reconcile_unknown(self, request_id: str, actual: RequestAllowance) -> BudgetReservation:
        """Explicit provider/budget-owner reconciliation, never an automatic retry."""
        return self._settle(request_id, actual, expected_state="unknown")

    def recover_interrupted(self) -> int:
        """After a process crash, all dispatched unfinished requests retain their caps."""
        with self._transaction():
            requests = self.connection.execute(
                "SELECT request_id FROM pilot_budget_requests WHERE state='sent'"
            ).fetchall()
            for row in requests:
                self._transition(row[0], "unknown")
            return len(requests)

    def mark_restored(self) -> None:
        """Mandatory after restoring a backup, before permitting any paid operation."""
        with self._transaction():
            self.connection.execute("UPDATE pilot_budget_metadata SET restore_pending=1 WHERE singleton=1")
            self.connection.execute("UPDATE pilot_budget_scopes SET reconcile=1")
            for row in self.connection.execute(
                "SELECT request_id FROM pilot_budget_requests WHERE state IN ('reserved','sent')"
            ).fetchall():
                self._transition(row[0], "unknown")
            self._event("all", "backup_restored_reconciliation_required")

    def reconcile_scope(self, scope_id: str, *, additional_allowance: BudgetLimits) -> BudgetSnapshot:
        """Explicitly authorize conservative NEW spending after reconciliation.

        ``additional_allowance`` is headroom above everything currently charged or
        held locally. In particular, it must not include amounts already held for
        unknown requests. It is supplied by a user/provider budget reconciliation,
        never inferred from an old backup's balance.
        """
        _identifier(scope_id)
        with self._transaction():
            snapshot = self._snapshot(scope_id)
            # Real overrun totals must remain visible even beyond the configured
            # cap bound. Zero headroom can acknowledge such an exhausted scope;
            # positive headroom beyond that bound must never silently be granted.
            reconciled_limits = []
            for used, extra in zip(_values(snapshot.used), _values(additional_allowance), strict=True):
                if extra and used + extra > _MAX_AMOUNT:
                    raise BudgetStateError("Дополнительный бюджет превышает допустимый предел. Можно подтвердить период без новых расходов.")
                reconciled_limits.append(min(used + extra, _MAX_AMOUNT))
            limits = BudgetLimits(*reconciled_limits)
            self.connection.execute(
                "UPDATE pilot_budget_scopes SET max_calls=?,max_input=?,max_output=?,max_cost=?,reconcile=0 "
                "WHERE scope_id=?", (*_values(limits), scope_id),
            )
            self._event(scope_id, "scope_reconciled")
            self._event(scope_id, "reconciled_headroom:" + json.dumps({
                "used": _values(snapshot.used), "additional_allowance": _values(additional_allowance),
                "currency": snapshot.currency,
            }, sort_keys=True, separators=(",", ":")))
            return self._snapshot(scope_id)

    def finish_restore_reconciliation(self) -> None:
        """Allow future periods only after every restored scope has been reconciled."""
        with self._transaction():
            if self.connection.execute(
                "SELECT 1 FROM pilot_budget_scopes WHERE reconcile=1 LIMIT 1"
            ).fetchone() is not None:
                raise BudgetStateError("Не все восстановленные ограничения бюджета прошли сверку.")
            self.connection.execute("UPDATE pilot_budget_metadata SET restore_pending=0 WHERE singleton=1")
            self._event("all", "restore_reconciliation_completed")
