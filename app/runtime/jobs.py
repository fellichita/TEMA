"""Durable analysis coordinator: immutable checkpoints, cancellation fences.

The document library remains owned by the legacy backend. New analysis runs and
the spending ledger share pilot.sqlite3. Admission, maintenance and shutdown go
through the coordinator's own writer thread; each admitted run executes on its
own thread with its own connection (WAL serialises the short writes), so the
web service can run several analyses at once. Workers receive immutable input;
they never receive a connection.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
from threading import Event, RLock, get_ident
from time import monotonic
from typing import Callable, Any, Concatenate, Iterator, ParamSpec, TypeVar
from uuid import uuid4

from app.backend.locking import InstanceLock
from app.diagnostics import log_internal_error
from app.identity import validate_data_dir
from app.runtime.budget import BudgetService
from app.sqlite_runtime import sqlite3

# A finished discovery stage is the largest checkpoint a run saves, so this
# stays in step with app.pilot.discovery.MAX_OUTPUT_BYTES.
MAX_CHECKPOINT_BYTES = 50_000_000
ANTECEDENTS_INDEX_SQL = """CREATE INDEX IF NOT EXISTS ix_analysis_antecedents_lookup ON analysis_runs(
    json_extract(input_json, '$.payload.source_run_id'),
    json_extract(input_json, '$.payload.candidate_id'), created_at DESC,id DESC)
    WHERE json_extract(input_json, '$.payload.operation')='antecedents'"""
_STAGE = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
_PERSISTENCE_ERROR = (
    "Не удалось надёжно сохранить завершение анализа. Результат не опубликован. "
    "Проверьте свободное место и перезапустите приложение; продолжение доступно вручную."
)
_P = ParamSpec("_P")
_T = TypeVar("_T")
# Сколько анализов может идти одновременно. Настольное приложение держит один;
# веб-сервис поднимает число из панели владельца.
MAX_RUN_SLOTS = 4


class TaskCancelled(Exception):
    pass


class TaskFailure(RuntimeError):
    """An intentionally public, credential-free task failure."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _encode(value: dict) -> bytes:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    if len(data) > MAX_CHECKPOINT_BYTES:
        raise TaskFailure("Превышен размер сохраняемого этапа анализа.")
    return data


@dataclass
class RunContext:
    run_id: str
    attempt: int
    cancel_event: Event
    _owner: Coordinator = field(repr=False)
    _database: sqlite3.Connection | None = field(default=None, repr=False)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TaskCancelled()

    def progress(self, stage: str, message: str, completed: int = 0, total: int = 0) -> None:
        self.check_cancelled()
        if not _STAGE.fullmatch(stage) or not 0 <= completed <= total:
            raise ValueError("Invalid progress")
        self._owner._progress(self, stage, message, completed, total)

    def checkpoint(self, stage: str, value: dict) -> None:
        self.check_cancelled()
        self._owner._checkpoint(self, stage, value)

    def load_checkpoint(self, stage: str) -> dict | None:
        self.check_cancelled()
        return self._owner._load_checkpoint(self.run_id, stage, cancel=self.cancel_event, database=self._database)

    @property
    def connection(self) -> sqlite3.Connection:
        """This run's own connection: pass it to BudgetService, never to a CPU worker."""
        return self._database if self._database is not None else self._owner._connection


class Coordinator:
    def __init__(self, data_dir: Path, processor: Callable[[RunContext, dict], dict], *, workflow_version: str = "1",
                 slots: int = 1):
        if not re.fullmatch(r"[a-zA-Z0-9._-]{1,80}", workflow_version):
            raise ValueError("Invalid workflow version")
        if type(slots) is not int or not 1 <= slots <= MAX_RUN_SLOTS:
            raise ValueError("Invalid run slots")
        self.workflow_version = workflow_version
        self.data_dir = validate_data_dir(data_dir)
        self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.data_dir, 0o700)
        self.path = self.data_dir / "pilot.sqlite3"
        self._processor = processor
        self._gate = RLock()
        self._slots = slots
        # run_id → (cancellation, future, attempt) of admitted runs; a finished
        # one stays until the next admission has checked its terminal row.
        self._runs: dict[str, tuple[Event, Future, int]] = {}
        self._storage_failure: dict[str, Any] | None = None
        self._maintenance_future: Future | None = None
        self._idle_operation: tuple[int, Event] | None = None
        self._read_shutdown = Event()
        self._reading = 0
        self._reads_done = Event()
        self._reads_done.set()
        self._closed = False
        self._closing = False
        self._lock = InstanceLock(self.data_dir / "pilot.lock")
        self._lock.acquire()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pilot-coordinator")
        self._run_executor = ThreadPoolExecutor(max_workers=MAX_RUN_SLOTS, thread_name_prefix="pilot-run")
        try:
            self._executor.submit(self._open).result(timeout=15)
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._run_executor.shutdown(wait=True, cancel_futures=True)
            self._lock.release()
            raise

    @property
    def slots(self) -> int:
        return self._slots

    def set_slots(self, slots: int) -> None:
        """Сколько анализов допускать одновременно; идущие не прерываются."""
        if type(slots) is not int or not 1 <= slots <= MAX_RUN_SLOTS:
            raise ValueError("Invalid run slots")
        with self._gate:
            self._slots = slots

    def running(self) -> list[str]:
        """Идентификаторы анализов, которые сейчас выполняются."""
        with self._gate:
            return [run_id for run_id, (_, future, _) in self._runs.items() if not future.done()]

    def _open(self) -> None:
        # sqlite3.connect follows links before any schema check. Refuse a
        # preexisting alias so a forged profile cannot initialise an unrelated
        # file as the analysis database. The profile lock is already held.
        try:
            info = self.path.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise TaskFailure("Файл базы анализов должен быть обычным файлом каталога данных.")
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        self._connection = connection
        connection.row_factory = sqlite3.Row
        try:
            if os.name != "nt":
                os.chmod(self.path, 0o600)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise TaskFailure("Анализы созданы более новой версией приложения.")
            if version == 0 and connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone():
                raise TaskFailure("Неизвестный формат базы анализов.")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(f"""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS analysis_runs (
                  id TEXT PRIMARY KEY, attempt INTEGER NOT NULL, state TEXT NOT NULL
                    CHECK(state IN ('queued','running','succeeded','failed','cancelled','interrupted')),
                  input_json TEXT NOT NULL, stage TEXT, message TEXT NOT NULL DEFAULT '',
                  completed INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT);
                CREATE TABLE IF NOT EXISTS analysis_checkpoints (
                  run_id TEXT NOT NULL REFERENCES analysis_runs(id), stage TEXT NOT NULL,
                  attempt INTEGER NOT NULL, digest TEXT NOT NULL,
                  PRIMARY KEY(run_id, stage));
                CREATE INDEX IF NOT EXISTS ix_analysis_runs_recent ON analysis_runs(created_at DESC,id DESC);
                {ANTECEDENTS_INDEX_SQL};
                PRAGMA user_version=1;
                UPDATE analysis_runs SET state='interrupted',
                    error='Предыдущий запуск прерван. Продолжение запускается вручную.'
                    WHERE state IN ('running','queued');
                COMMIT;
            """)
            BudgetService(connection).recover_interrupted()
        except BaseException:
            connection.close()
            raise

    @contextmanager
    def idle_operation(self, *, for_run: bool = False) -> Iterator[None]:
        """Reserve admission for preflight/settings without locking slow IO.

        Only this thread may submit or maintain while it owns the reservation.
        Readers and cancellation still acquire the gate normally; close waits
        for the reservation outside the gate before releasing storage ownership.
        """
        with self._gate:
            # Подготовка нового анализа допустима рядом с идущими, пока есть место.
            self._available(run=for_run)
            if self._idle_operation is not None:
                raise TaskFailure("Дождитесь текущей операции с настройками или подготовкой анализа.")
            reservation = get_ident(), Event()
            self._idle_operation = reservation
        try:
            yield
        finally:
            with self._gate:
                self._idle_operation = None
                reservation[1].set()

    def has_active_work(self) -> bool:
        """Actual owned work, including preparation; independent of history pages."""
        with self._gate:
            return (self._idle_operation is not None
                    or self._reading > 0
                    or any(not future.done() for _, future, _ in self._runs.values())
                    or self._maintenance_future is not None and not self._maintenance_future.done())

    def submit(self, payload: dict, *, cancel: Event | None = None) -> str:
        # JSON round-trip prevents a caller changing input after admission.
        encoded = _encode({"workflow_version": self.workflow_version, "payload": payload}).decode("utf-8")
        with self._gate:
            self._available(run=True)
            cancel = cancel if cancel is not None else Event()
            if cancel.is_set():
                raise TaskCancelled()
            run_id = uuid4().hex
            self._executor.submit(self._create, run_id, encoded).result(timeout=10)
            future = self._run_executor.submit(self._run, run_id, 1, json.loads(encoded)["payload"], cancel)
            self._runs[run_id] = cancel, future, 1
            return run_id

    def _available(self, *, run: bool = False) -> None:
        """Admission check: a run needs a free slot; other work needs no run at all."""
        if self._closed or self._closing:
            raise TaskFailure("Приложение завершает работу.")
        if self._idle_operation is not None and self._idle_operation[0] != get_ident():
            raise TaskFailure("Дождитесь текущей операции с настройками или подготовкой анализа.")
        busy = sum(not future.done() for _, future, _ in self._runs.values())
        if busy and (not run or self._slots == 1):
            raise TaskFailure("Дождитесь текущего анализа или отмените его.")
        if busy >= self._slots:
            raise TaskFailure("Все места для одновременных анализов заняты. Дождитесь завершения одного из них.")
        if self._maintenance_future is not None and not self._maintenance_future.done():
            raise TaskFailure("Дождитесь завершения операции с журналом приложения.")
        for run_id, (_, future, _) in list(self._runs.items()):
            if future.done():
                if self._storage_failure is None:
                    # Also covers an unexpected executor-level exit outside _run's guard.
                    self.get(run_id)
                del self._runs[run_id]
        if self._storage_failure is not None:
            raise TaskFailure(_PERSISTENCE_ERROR)

    def maintenance(self, operation: Callable[Concatenate[sqlite3.Connection, _P], _T],
                    *args: _P.args, **kwargs: _P.kwargs) -> _T:
        """Run a short, trusted application operation on the sole writer thread.

        This accepts a callable selected by Python application code, never an
        operation/module name deserialized from the UI. No worker serialization
        occurs. A timed-out operation remains tracked and blocks new admission;
        close retains the profile lock until queued writer work actually exits.
        """
        if (not inspect.isfunction(operation) or not operation.__module__.startswith("app.")
                or operation.__qualname__ != operation.__name__ or operation.__code__.co_freevars
                or getattr(importlib.import_module(operation.__module__), operation.__name__, None) is not operation):
            raise ValueError("Требуется доверенная функция приложения для обслуживания журнала.")
        with self._gate:
            self._available()
            future = self._executor.submit(self._maintain, operation, args, kwargs)
            self._maintenance_future = future
            return future.result(timeout=10)

    def _maintain(self, operation: Callable[..., _T], args: tuple, kwargs: dict) -> _T:
        if self._connection.in_transaction:
            self._connection.rollback()
            raise TaskFailure("Предыдущая операция оставила незавершённую транзакцию.")
        try:
            result = operation(self._connection, *args, **kwargs)
            if self._connection.in_transaction:
                raise TaskFailure("Операция обслуживания не подтвердила сохранение изменений.")
            return result
        finally:
            if self._connection.in_transaction:
                self._connection.rollback()

    def _create(self, run_id: str, encoded: str) -> None:
        now = _now()
        self._connection.execute("INSERT INTO analysis_runs "
                                 "(id,attempt,state,input_json,created_at,updated_at) VALUES (?,1,'queued',?,?,?)",
                                 (run_id, encoded, now, now))

    def resume(self, run_id: str, *, cancel: Event | None = None) -> str:
        with self._gate:
            self._available(run=True)
            cancel = cancel if cancel is not None else Event()
            if cancel.is_set():
                raise TaskCancelled()
            row = self.get(run_id)
            if row["state"] not in {"interrupted", "failed", "cancelled"}:
                raise TaskFailure("Этот анализ нельзя продолжить.")
            saved = json.loads(row["input_json"])
            if saved.get("workflow_version", "1") != self.workflow_version:
                raise TaskFailure("Метод анализа изменился. Создайте новый анализ: старые результаты доступны в истории.")
            attempt = row["attempt"] + 1
            self._executor.submit(self._resume, run_id, attempt).result(timeout=10)
            self._runs[run_id] = cancel, self._run_executor.submit(
                self._run, run_id, attempt, saved.get("payload", saved), cancel), attempt
            return run_id

    def _resume(self, run_id: str, attempt: int) -> None:
        self._connection.execute("UPDATE analysis_runs SET attempt=?,state='queued',error=NULL,updated_at=? "
                                 "WHERE id=?", (attempt, _now(), run_id))

    def cancel(self, run_id: str) -> bool:
        with self._gate:
            active = self._runs.get(run_id)
            if active is not None and not active[1].done():
                active[0].set()
                return True
            return False

    def _run_connection(self) -> sqlite3.Connection:
        """A run's own connection, used only on its own thread."""
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _run(self, run_id: str, attempt: int, payload: dict, cancel: Event) -> None:
        try:
            connection = self._run_connection()
        except BaseException as error:
            log_internal_error("pilot.run_connection_failed", error)
            with self._gate:
                self._mark_storage_failure({"id": run_id, "attempt": attempt, "state": "running",
                                            "input_json": "{}", "stage": None, "message": "", "completed": 0,
                                            "total": 0, "created_at": _now(), "updated_at": _now(), "error": None})
            return
        try:
            self._execute(RunContext(run_id, attempt, cancel, self, connection), payload)
        finally:
            connection.close()

    def _execute(self, context: RunContext, payload: dict) -> None:
        cancel, run_id, attempt = context.cancel_event, context.run_id, context.attempt
        try:
            changed = context.connection.execute(
                "UPDATE analysis_runs SET state='running',updated_at=? WHERE id=? AND attempt=? AND state='queued'",
                (_now(), run_id, attempt))
            if changed.rowcount != 1:
                raise TaskFailure("Состояние анализа изменилось; создайте новый запуск.")
            context.check_cancelled()
            result = self._processor(context, payload)
            with self._gate:
                context.check_cancelled()
                self._checkpoint(context, "result", result)
                self._finish(context, "succeeded", None, payload)
        except TaskCancelled:
            self._finish(context, "cancelled", None, payload)
        except BaseException as error:
            log_internal_error("pilot.run_failed", error)
            message = str(error) if isinstance(error, TaskFailure) else (
                "Анализ не завершён. Данные этапов сохранены; проверьте настройки и повторите запуск.")
            self._finish(context, "cancelled" if cancel.is_set() else "failed", message, payload)

    def _finish(self, context: RunContext, state: str, error: str | None, payload: dict) -> None:
        """Retry only an idempotent state write, never processor/provider calls.

        Permanent storage failure poisons admission for this process. A volatile
        failure view makes polling terminate, but never pretends the DB commit
        succeeded. Startup's existing recovery marks unfinished rows interrupted.
        """
        with self._gate:
            database = context.connection
            for _ in range(2):
                try:
                    if database.in_transaction:
                        # A processor must not leave a transaction open: otherwise
                        # an apparently successful terminal write is not durable.
                        database.rollback()
                        state, error = "failed", _PERSISTENCE_ERROR
                    self._terminal(context, state, error)
                    return
                except BaseException as failure:
                    log_internal_error("pilot.terminal_failed", failure)
            now = _now()
            row: dict[str, Any] = dict(id=context.run_id, attempt=context.attempt, state="running",
                input_json=_encode({"workflow_version": self.workflow_version, "payload": payload}).decode("utf-8"),
                stage=None, message="", completed=0, total=0, created_at=now, updated_at=now, error=None)
            try:
                saved = database.execute("SELECT * FROM analysis_runs WHERE id=? AND attempt=?",
                                         (context.run_id, context.attempt)).fetchone()
                if saved is not None:
                    row = dict(saved)
            except BaseException as failure:
                log_internal_error("pilot.terminal_read_failed", failure)
            self._mark_storage_failure(row)

    def _mark_storage_failure(self, row: dict[str, Any]) -> dict[str, Any]:
        # Called under _gate, after the processor has exited. Keep the storage
        # lock until normal close confirms the owned executor has really stopped.
        self._storage_failure = row | dict(state="failed", error=_PERSISTENCE_ERROR,
                                           updated_at=_now(), persistence_error=True)
        return dict(self._storage_failure)

    def _terminal(self, context: RunContext, state: str, error: str | None) -> None:
        database = context.connection
        if state not in {"succeeded", "failed", "cancelled"} or database.in_transaction:
            raise TaskFailure("Невозможно подтвердить завершение анализа.")
        changed = database.execute("UPDATE analysis_runs SET state=?,error=?,updated_at=? "
            "WHERE id=? AND attempt=? AND state IN ('queued','running')",
            (state, error, _now(), context.run_id, context.attempt))
        if changed.rowcount != 1:
            saved = database.execute("SELECT state,error FROM analysis_runs WHERE id=? AND attempt=?",
                                     (context.run_id, context.attempt)).fetchone()
            if saved is None or saved["state"] != state or saved["error"] != error:
                raise TaskFailure("Состояние завершения анализа не подтверждено.")

    def _progress(self, context: RunContext, stage: str, message: str, completed: int, total: int) -> None:
        # Bounded row update; no unbounded queue of progress callbacks or prompts.
        context.connection.execute("UPDATE analysis_runs SET stage=?,message=?,completed=?,total=?,updated_at=? "
                                 "WHERE id=? AND attempt=? AND state='running'",
                                 (stage, message[:1000], completed, total, _now(), context.run_id, context.attempt))

    def _checkpoint_path(self, digest: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise TaskFailure("Повреждён указатель сохранённого этапа.")
        return self.data_dir / "checkpoints" / (digest + ".json")

    def _checkpoint(self, context: RunContext, stage: str, value: dict) -> None:
        if not _STAGE.fullmatch(stage):
            raise ValueError("Invalid checkpoint stage")
        data = _encode(value)
        digest = hashlib.sha256(data).hexdigest()
        path = self._checkpoint_path(digest)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise TaskFailure("Каталог сохранённых этапов не должен быть символической ссылкой.")
        if os.name != "nt":
            os.chmod(path.parent, 0o700)
        temporary = path.with_suffix("." + uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as handle:
                if os.name != "nt":
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            with self._gate:
                context.check_cancelled()
                temporary.replace(path)
                if os.name != "nt":
                    descriptor = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                context.connection.execute("INSERT INTO analysis_checkpoints VALUES (?,?,?,?) "
                                         "ON CONFLICT(run_id,stage) DO UPDATE SET attempt=excluded.attempt, "
                                         "digest=excluded.digest",
                                         (context.run_id, stage, context.attempt, digest))
        finally:
            temporary.unlink(missing_ok=True)

    def _load_checkpoint(self, run_id: str, stage: str, *, cancel: Event | None = None,
                         database: sqlite3.Connection | None = None) -> dict | None:
        row = (database or self._connection).execute(
            "SELECT digest FROM analysis_checkpoints WHERE run_id=? AND stage=?", (run_id, stage)).fetchone()
        return self._read_checkpoint(row[0], cancel=cancel) if row else None

    def _check_read_cancelled(self, cancel: Event | None) -> None:
        if self._read_shutdown.is_set() or cancel is not None and cancel.is_set():
            raise TaskCancelled()

    @contextmanager
    def _checkpoint_reader(self, cancel: Event | None) -> Iterator[None]:
        with self._gate:
            self._check_read_cancelled(cancel)
            self._reading += 1
            self._reads_done.clear()
        try:
            yield
        finally:
            with self._gate:
                self._reading -= 1
                if self._reading == 0:
                    self._reads_done.set()

    def _read_checkpoint(self, digest: str, *, cancel: Event | None = None) -> dict:
        from app.pilot.reports import open_local_regular

        try:
            self._check_read_cancelled(cancel)
            path = self._checkpoint_path(digest)
            data, checksum = bytearray(), hashlib.sha256()
            with open_local_regular(path) as handle:
                while True:
                    self._check_read_cancelled(cancel)
                    chunk = handle.read(min(64 * 1024, MAX_CHECKPOINT_BYTES + 1 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > MAX_CHECKPOINT_BYTES:
                        raise ValueError("Oversized checkpoint")
                    checksum.update(chunk)
            self._check_read_cancelled(cancel)
            if checksum.hexdigest() != digest:
                raise ValueError("Invalid checkpoint bytes")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("Invalid checkpoint object")
            self._check_read_cancelled(cancel)
            return value
        except (OSError, ValueError, RecursionError):
            raise TaskFailure("Сохранённый этап повреждён. Создайте новый анализ или восстановите резервную копию.") from None

    def _reader(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _visible_row(self, row: dict[str, Any]) -> dict[str, Any]:
        with self._gate:
            failure = self._storage_failure
            if failure is not None and (failure["id"], failure["attempt"]) == (row["id"], row["attempt"]):
                return dict(failure)
            active = self._runs.get(row["id"])
            if (active is not None and active[2] == row["attempt"]
                    and active[1].done() and row["state"] in {"queued", "running"}):
                return self._mark_storage_failure(row)
            return row

    def get(self, run_id: str) -> dict[str, Any]:
        # Keep the SQL snapshot and Future inspection under the same gate as
        # terminal publication: a stale running row must not poison a successful
        # run whose Future completed while a reader was waiting for this gate.
        with self._gate:
            return self._get(run_id)

    def _get(self, run_id: str) -> dict[str, Any]:
        with self._gate:
            if self._storage_failure is not None and self._storage_failure["id"] == run_id:
                return dict(self._storage_failure)
        connection = None
        try:
            connection = self._reader()
            row = connection.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise TaskFailure("Анализ не найден.")
            return self._visible_row(dict(row))
        except sqlite3.Error:
            with self._gate:
                if self._storage_failure is not None and self._storage_failure["id"] == run_id:
                    return dict(self._storage_failure)
            raise TaskFailure("Не удалось прочитать журнал анализов. Проверьте диск и перезапустите приложение.") from None
        finally:
            if connection is not None:
                connection.close()

    def list_runs(self, limit: int = 100, offset: int = 0, *, analyses_only: bool = False) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0 or type(analyses_only) is not bool:
            raise ValueError("Invalid page size")
        with self._gate:
            return self._list_runs(limit, offset, analyses_only=analyses_only)

    def list_operation_runs(self, operation: str, *, limit: int = 50, offset: int = 0) -> list[dict]:
        if operation not in {"signals", "antecedents"} or type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
            raise ValueError("Invalid operation page")
        with self._gate:
            connection = None
            try:
                connection = self._reader()
                return [self._visible_row(dict(row)) for row in connection.execute(
                    "SELECT * FROM analysis_runs WHERE json_extract(input_json, '$.payload.operation')=? "
                    "ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?", (operation, limit, offset))]
            except sqlite3.Error:
                raise TaskFailure("Не удалось прочитать историю сигналов.") from None
            finally:
                if connection is not None:
                    connection.close()

    def find_antecedents_run(self, source_run_id: str, candidate_id: str) -> dict[str, Any] | None:
        """Find the latest review for this pair independently of history pages."""
        if any(not isinstance(value, str) or not 1 <= len(value) <= 2048
               for value in (source_run_id, candidate_id)):
            raise ValueError("Invalid antecedents identifiers")
        with self._gate:
            connection = None
            try:
                connection = self._reader()
                row = connection.execute(
                    "SELECT * FROM analysis_runs WHERE json_extract(input_json, '$.payload.operation')='antecedents' "
                    "AND json_extract(input_json, '$.payload.source_run_id')=? "
                    "AND json_extract(input_json, '$.payload.candidate_id')=? ORDER BY created_at DESC,id DESC LIMIT 1",
                    (source_run_id, candidate_id)).fetchone()
                return self._visible_row(dict(row)) if row is not None else None
            except sqlite3.Error:
                raise TaskFailure("Не удалось прочитать журнал исторических проверок. Проверьте диск и повторите.") from None
            finally:
                if connection is not None:
                    connection.close()

    def _list_runs(self, limit: int, offset: int, *, analyses_only: bool = False) -> list[dict]:
        connection = None
        try:
            connection = self._reader()
            clause = (" WHERE coalesce(json_extract(input_json, '$.payload.operation'), '') NOT IN ('antecedents','signals')"
                      if analyses_only else "")
            return [self._visible_row(dict(row)) for row in connection.execute(
                "SELECT * FROM analysis_runs" + clause + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?", (limit, offset))]
        except sqlite3.Error:
            with self._gate:
                if self._storage_failure is not None:
                    return [dict(self._storage_failure)]
            raise TaskFailure("Не удалось прочитать журнал анализов. Проверьте диск и перезапустите приложение.") from None
        finally:
            if connection is not None:
                connection.close()

    def result(self, run_id: str, *, cancel: Event | None = None) -> dict:
        with self._checkpoint_reader(cancel):
            with self._gate:
                if self._storage_failure is not None and self._storage_failure["id"] == run_id:
                    raise TaskFailure(_PERSISTENCE_ERROR)
                connection = None
                try:
                    connection = self._reader()
                    row = connection.execute("SELECT c.digest FROM analysis_checkpoints c JOIN analysis_runs r "
                                             "ON r.id=c.run_id WHERE r.id=? AND r.state='succeeded' "
                                             "AND c.stage='result' AND c.attempt=r.attempt", (run_id,)).fetchone()
                    if row is None:
                        raise TaskFailure("Готовый результат отсутствует.")
                    digest = row[0]
                except sqlite3.Error:
                    raise TaskFailure("Не удалось прочитать журнал анализов. Проверьте диск и перезапустите приложение.") from None
                finally:
                    if connection is not None:
                        connection.close()
            value = self._read_checkpoint(digest, cancel=cancel)
            with self._gate:
                self._check_read_cancelled(cancel)
                if self._storage_failure is not None and self._storage_failure["id"] == run_id:
                    raise TaskFailure(_PERSISTENCE_ERROR)
            return value

    def checkpoint_value(self, run_id: str, stage: str, *, cancel: Event | None = None) -> dict | None:
        """Read a verified completed stage while the coordinator is occupied."""
        if not _STAGE.fullmatch(stage):
            raise ValueError("Invalid checkpoint stage")
        with self._checkpoint_reader(cancel):
            with self._gate:
                connection = self._reader()
                try:
                    row = connection.execute("SELECT digest FROM analysis_checkpoints WHERE run_id=? AND stage=?",
                                             (run_id, stage)).fetchone()
                    digest = row[0] if row else None
                finally:
                    connection.close()
            value = self._read_checkpoint(digest, cancel=cancel) if digest is not None else None
            with self._gate:
                self._check_read_cancelled(cancel)
            return value

    def wait(self, timeout: float = 60) -> None:
        deadline = monotonic() + timeout
        with self._gate:
            futures = [future for _, future, _ in self._runs.values()]
        for future in futures:
            future.result(timeout=max(0.0, deadline - monotonic()))

    def close(self, timeout: float = 30) -> None:
        deadline = monotonic() + timeout

        def remaining() -> float:
            return max(0.0, deadline - monotonic())

        with self._gate:
            if self._closed:
                return
            operation = self._idle_operation
            if operation is not None and operation[0] == get_ident():
                raise TaskFailure("Завершите текущую локальную операцию перед закрытием.")
            self._closing = True
            self._read_shutdown.set()
            for cancel, _, _ in self._runs.values():
                cancel.set()
        if operation is not None and not operation[1].wait(remaining()):
            raise TimeoutError("Локальная операция ещё не завершена; блокировка данных сохранена.")
        if not self._reads_done.wait(remaining()):
            raise TimeoutError("Чтение сохранённых данных ещё не завершено; блокировка данных сохранена.")
        try:
            self.wait(remaining())
        except BaseException as error:
            with self._gate:
                if any(not future.done() for _, future, _ in self._runs.values()):
                    # In particular, a shutdown timeout never releases the lock
                    # while the processor or its owned CPU child may still run.
                    raise
            log_internal_error("pilot.close_after_failure", error)
        self._run_executor.shutdown(wait=True)
        self._executor.submit(self._connection.close).result(timeout=remaining())
        self._executor.shutdown(wait=True)
        self._lock.release()
        self._closed = True
