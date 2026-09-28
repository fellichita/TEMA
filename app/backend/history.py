"""Durable calendar-period plans. Checkpoint boundary is a period, not an API cursor."""

import calendar
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from app.backend.contracts import Contract, SearchRequest, SourceName, JobRecord, utc_now, normalize_primary_topic_ids
from app.backend.errors import BackendError
from app.backend.validation import validate_pagination

if TYPE_CHECKING:
    from app.backend.repository import Repository

HistoryPeriodState = Literal[
    "pending", "queued", "running", "complete", "partial", "failed", "cancelled", "interrupted", "split",
]


HISTORY_SCHEMA = (
    """CREATE TABLE history_runs (
        id TEXT PRIMARY KEY, request_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','partial','cancelled','interrupted','failed')),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        error_code TEXT, error_message TEXT)""",
    """CREATE TABLE history_periods (
        id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES history_runs(id),
        ordinal INTEGER NOT NULL, source TEXT NOT NULL, from_date TEXT NOT NULL,
        until_date TEXT NOT NULL, full_calendar_period INTEGER NOT NULL,
        latest_job_id TEXT REFERENCES jobs(id), UNIQUE(run_id,ordinal))""",
    """CREATE TABLE history_attempts (
        period_id TEXT NOT NULL REFERENCES history_periods(id),
        job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), attempt INTEGER NOT NULL,
        PRIMARY KEY(period_id,attempt))""",
    "CREATE INDEX ix_history_created ON history_runs(created_at DESC,id)",
    "CREATE INDEX ix_aliases_document ON aliases(document_key)",
)


class HistoryRequest(Contract):
    topic: str
    from_date: date
    until_date: date = Field(default_factory=lambda: utc_now().date())
    sources: tuple[SourceName, ...] = ("crossref", "openalex")
    period: Literal["year", "month"] = "month"
    max_results_per_period: int = Field(default=1000, ge=1, le=10_000, strict=True)
    auto_split: bool = Field(default=True, strict=True)
    max_periods: int = Field(default=1200, ge=1, le=10_000, strict=True)
    primary_topic_ids: tuple[str, ...] = Field(default=(), max_length=100)

    @field_validator("primary_topic_ids")
    @classmethod
    def validate_primary_topics(cls, values):
        return normalize_primary_topic_ids(values)

    @model_validator(mode="after")
    def valid_plan(self):
        SearchRequest(topic=self.topic, from_date=self.from_date, until_date=self.until_date,
                      max_results=self.max_results_per_period)
        if not self.sources or len(self.sources) != len(set(self.sources)):
            raise ValueError("Требуется непустой список разных источников")
        if self.primary_topic_ids and self.sources != ("openalex",):
            raise ValueError("Отбор по primary_topic_ids требует единственного источника OpenAlex")
        if self.from_date.year < 1000 or self.until_date.year - self.from_date.year > 100:
            raise ValueError("Допустим период до 100 лет, начиная с 1000 года")
        if sum(1 for _ in self.intervals()) * len(self.sources) > min(1200, self.max_periods):
            raise ValueError("Начальный план превышает лимит периодов (не более 1200)")
        return self

    def intervals(self):
        current = self.from_date
        while current <= self.until_date:
            start = date(current.year, current.month, 1) if self.period == "month" else date(current.year, 1, 1)
            end = (date(current.year, current.month, calendar.monthrange(current.year, current.month)[1])
                   if self.period == "month" else date(current.year, 12, 31))
            clipped_end = min(end, self.until_date)
            yield current, clipped_end, current == start and clipped_end == end
            if clipped_end == self.until_date:
                break
            current = clipped_end + timedelta(days=1)


class HistoryPeriod(Contract):
    id: str
    source: SourceName
    from_date: date
    until_date: date
    full_calendar_period: bool
    state: HistoryPeriodState
    parent_id: str | None = None
    granularity: Literal["year", "month", "day"]
    attempts: tuple[str, ...] = ()
    job: JobRecord | None = None
    incomplete_reason: str | None = None


class HistoryReport(Contract):
    id: str
    request: HistoryRequest
    state: Literal["queued", "running", "succeeded", "partial", "cancelled", "interrupted", "failed"]
    created_at: datetime
    updated_at: datetime
    error_code: str | None = None
    error_message: str | None = None
    periods: tuple[HistoryPeriod, ...]
    total_periods: int
    split_periods: int
    total_plan_periods: int
    completed_periods: int
    partial_periods: int
    failed_periods: int
    processed_periods: int
    partial_calendar_years: tuple[int, ...]
    coverage_complete: bool
    contract_version: int = 2


class HistoryStore:
    def __init__(self, repository: "Repository") -> None:
        self.repo = repository

    @staticmethod
    def _run(connection, run_id):
        row = connection.execute("SELECT * FROM history_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise BackendError("history_not_found", "Исторический сбор не найден.")
        return row

    def create(self, request):
        request = HistoryRequest.model_validate(request)
        run_id, now = str(uuid4()), utc_now().isoformat()
        with self.repo._transaction() as connection:
            connection.execute("INSERT INTO history_runs VALUES(?,?,'queued',?,?,NULL,NULL)",
                               (run_id, request.model_dump_json(), now, now))
            ordinal = 0
            for start, end, full in request.intervals():
                for source in request.sources:
                    connection.execute("""INSERT INTO history_periods
                        (id,run_id,ordinal,source,from_date,until_date,full_calendar_period,granularity)
                        VALUES(?,?,?,?,?,?,?,?)""",
                        (str(uuid4()), run_id, ordinal, source, start.isoformat(), end.isoformat(), int(full), request.period))
                    ordinal += 1
        return run_id

    def state(self, run_id, state, code=None, message=None):
        with self.repo._transaction() as connection:
            self._run(connection, run_id)
            connection.execute("UPDATE history_runs SET state=?,updated_at=?,error_code=?,error_message=? WHERE id=?",
                               (state, utc_now().isoformat(), code, message, run_id))

    def begin_attempt(self, run_id, period_id):
        # The job and its checkpoint link must survive together or not at all.
        with self.repo._transaction() as connection:
            run = self._run(connection, run_id)
            if run["state"] != "running":
                raise BackendError("invalid_state", "Исторический сбор не запущен.")
            period = connection.execute("SELECT * FROM history_periods WHERE id=? AND run_id=?", (period_id, run_id)).fetchone()
            if period is None:
                raise BackendError("history_not_found", "Период не найден.")
            if period["is_split"]:
                raise BackendError("invalid_state", "Разделённый период заменён дочерними периодами.")
            if period["latest_job_id"]:
                previous = self.repo._job(connection, period["latest_job_id"])
                if previous.state in {"queued", "running"} or previous.coverage_complete:
                    raise BackendError("invalid_state", "Период уже активен или полностью собран.")
            request = HistoryRequest.model_validate_json(run["request_json"])
            job = self.repo._create_job(connection, SearchRequest(
                topic=request.topic, source=period["source"], from_date=period["from_date"],
                until_date=period["until_date"], max_results=request.max_results_per_period,
                primary_topic_ids=request.primary_topic_ids,
            ))
            attempt = connection.execute("SELECT COUNT(*) FROM history_attempts WHERE period_id=?", (period_id,)).fetchone()[0] + 1
            connection.execute("INSERT INTO history_attempts VALUES(?,?,?)", (period_id, job.id, attempt))
            connection.execute("UPDATE history_periods SET latest_job_id=?,split_block_reason=NULL WHERE id=?", (job.id, period_id))
            connection.execute("UPDATE history_runs SET updated_at=? WHERE id=?", (utc_now().isoformat(), run_id))
            return job

    def split_overflow(self, run_id, period_id):
        """Atomically replace an overflowing leaf with calendar children; idempotent."""
        with self.repo._transaction() as connection:
            run = self._run(connection, run_id)
            if run["state"] != "running":
                raise BackendError("invalid_state", "Исторический сбор не запущен.")
            request = HistoryRequest.model_validate_json(run["request_json"])
            row = connection.execute("SELECT * FROM history_periods WHERE id=? AND run_id=?", (period_id, run_id)).fetchone()
            if row is None:
                raise BackendError("history_not_found", "Период не найден.")
            if row["is_split"] or not request.auto_split or not row["latest_job_id"]:
                return ()
            job = self.repo._job(connection, row["latest_job_id"])
            limit = min(request.max_results_per_period, 2000) if row["source"] == "epo" else request.max_results_per_period
            if job.state != "succeeded" or job.source_exhausted or job.scanned < limit:
                return ()
            start, end = date.fromisoformat(row["from_date"]), date.fromisoformat(row["until_date"])
            if start == end or row["granularity"] == "day":
                connection.execute("UPDATE history_periods SET split_block_reason='daily_limit_reached' WHERE id=?", (period_id,))
                return ()
            granularity = "month" if row["granularity"] == "year" else "day"
            children = []
            current = start
            while current <= end:
                boundary = (date(current.year, current.month, calendar.monthrange(current.year, current.month)[1])
                            if granularity == "month" else current)
                last = min(end, boundary)
                full = granularity == "day" or (current.day == 1 and last == boundary)
                children.append((current, last, full))
                if last == end:
                    break
                current = last + timedelta(days=1)
            count, ordinal = connection.execute("SELECT COUNT(*),MAX(ordinal) FROM history_periods WHERE run_id=?", (run_id,)).fetchone()
            if count + len(children) > request.max_periods:
                connection.execute("UPDATE history_periods SET split_block_reason='period_budget_exceeded' WHERE id=?", (period_id,))
                return ()
            ids = []
            for index, (first, last, full) in enumerate(children, start=1):
                child_id = str(uuid4())
                connection.execute("""INSERT INTO history_periods
                    (id,run_id,ordinal,source,from_date,until_date,full_calendar_period,parent_id,granularity)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (child_id, run_id, ordinal + index, row["source"], first.isoformat(), last.isoformat(),
                     int(full), period_id, granularity))
                ids.append(child_id)
            connection.execute("UPDATE history_periods SET is_split=1,split_block_reason=NULL WHERE id=?", (period_id,))
            connection.execute("UPDATE history_runs SET updated_at=? WHERE id=?", (utc_now().isoformat(), run_id))
            return tuple(ids)

    def prepare_resume(self, run_id):
        # Called under the backend mutex after excluding a live history worker.
        with self.repo._transaction() as connection:
            run = self._run(connection, run_id)
            if run["state"] in {"queued", "running"}:
                raise BackendError("history_busy", "Исторический сбор уже запущен.")
            now = utc_now().isoformat()
            connection.execute("""UPDATE jobs SET state='interrupted',updated_at=?,error_code='interrupted',
                error_message='Попытка не завершилась; создана новая попытка.'
                WHERE state IN ('queued','running') AND id IN
                (SELECT latest_job_id FROM history_periods WHERE run_id=?)""", (now, run_id))
            connection.execute("UPDATE history_runs SET state='queued',updated_at=?,error_code=NULL,error_message=NULL WHERE id=?", (now, run_id))

    def report(self, run_id: str) -> HistoryReport:
        with self.repo._connection() as connection:
            connection.execute("BEGIN")
            run = dict(self._run(connection, run_id))
            request = HistoryRequest.model_validate_json(run.pop("request_json"))
            attempts: dict[str, list[str]] = {}
            for row in connection.execute(
                "SELECT a.period_id,a.job_id FROM history_attempts a JOIN history_periods p ON p.id=a.period_id WHERE p.run_id=? ORDER BY a.attempt",
                (run_id,),
            ):
                attempts.setdefault(row[0], []).append(row[1])
            # Read all latest attempts in this same snapshot, without issuing a
            # separate query on each UI poll for every one of up to 10,000 nodes.
            jobs = {row['id']: self.repo._job_record(row) for row in connection.execute(
                "SELECT j.* FROM history_periods p JOIN jobs j ON j.id=p.latest_job_id WHERE p.run_id=?",
                (run_id,),
            )}
            periods = []
            for row in connection.execute("SELECT * FROM history_periods WHERE run_id=? ORDER BY ordinal", (run_id,)):
                job = jobs.get(row["latest_job_id"])
                if row["latest_job_id"] and job is None:
                    raise BackendError("job_not_found", "Задание не найдено.")
                state: HistoryPeriodState = "pending"
                reason = None
                if job and job.state == "succeeded":
                    state = "complete" if job.coverage_complete else "partial"
                    if not job.source_exhausted:
                        reason = "source_not_exhausted"
                    elif job.skipped:
                        reason = "invalid_records_skipped"
                    elif not job.coverage_complete:
                        reason = "inconsistent_total"
                elif job and job.state != "succeeded":
                    state = job.state
                    reason = job.error_code
                if row["is_split"]:
                    state, reason = "split", None
                elif state == "partial" and row["split_block_reason"]:
                    reason = row["split_block_reason"]
                periods.append(HistoryPeriod(
                    id=row["id"], source=row["source"], from_date=row["from_date"], until_date=row["until_date"],
                    full_calendar_period=bool(row["full_calendar_period"]), state=state,
                    parent_id=row["parent_id"], granularity=row["granularity"],
                    attempts=tuple(attempts.get(row["id"], ())), job=job, incomplete_reason=reason,
                ))
            completed = sum(p.state == "complete" for p in periods)
            partial = sum(p.state == "partial" for p in periods)
            failed = sum(p.state == "failed" for p in periods)
            partial_years = set()
            if request.from_date != date(request.from_date.year, 1, 1):
                partial_years.add(request.from_date.year)
            if request.until_date != date(request.until_date.year, 12, 31):
                partial_years.add(request.until_date.year)
            split_count = sum(p.state == "split" for p in periods)
            leaves = len(periods) - split_count
            return HistoryReport(**run, request=request, periods=tuple(periods), total_periods=leaves,
                                 split_periods=split_count, total_plan_periods=len(periods),
                                 completed_periods=completed, partial_periods=partial, failed_periods=failed,
                                 processed_periods=completed + partial + failed,
                                 partial_calendar_years=tuple(sorted(partial_years)),
                                 coverage_complete=run["state"] == "succeeded" and completed == leaves)

    def list_runs(self, limit=50):
        validate_pagination(limit)
        with self.repo._connection() as connection:
            rows = connection.execute("SELECT id,state,created_at,updated_at,request_json FROM history_runs ORDER BY created_at DESC,id LIMIT ?", (limit,)).fetchall()
            return tuple({key: row[key] for key in ("id", "state", "created_at", "updated_at")} |
                         {"request": HistoryRequest.model_validate_json(row["request_json"]).model_dump(mode="json")}
                         for row in rows)

    def recover(self):
        with self.repo._transaction() as connection:
            connection.execute("UPDATE history_runs SET state='interrupted',updated_at=?,error_code='interrupted',error_message='Процесс остановился; сбор можно продолжить.' WHERE state IN ('queued','running')", (utc_now().isoformat(),))
