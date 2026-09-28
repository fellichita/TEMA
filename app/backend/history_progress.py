"""Compact progress contract and plain terminal output, independent of any GUI."""

from datetime import date, datetime
from typing import Literal

from app.backend.contracts import Contract
from app.backend.history import HistoryReport

STATES = {
    "queued": "В очереди", "pending": "Не начат", "running": "Загрузка",
    "complete": "Полная выдача", "succeeded": "Завершено", "partial": "Неполная выдача",
    "failed": "Ошибка", "cancelled": "Отменён", "interrupted": "Прерван", "split": "Разделён",
}
REASONS = {
    "source_not_exhausted": "выдача источника не исчерпана",
    "invalid_records_skipped": "есть пропущенные невалидные записи",
    "inconsistent_total": "счётчики источника противоречат друг другу",
    "daily_limit_reached": "достигнут лимит выдачи за день",
    "period_budget_exceeded": "исчерпан бюджет дробления периодов",
    "source_unavailable": "источник недоступен",
    "rate_limited": "источник ограничил частоту запросов",
    "credentials_required": "не настроены ключи доступа",
    "authentication_required": "источник отклонил авторизацию",
    "cancelled": "сбор отменён", "interrupted": "процесс был прерван",
}
COVERAGE = {"complete": "полная", "incomplete": "неполная", "unknown": "пока неизвестна", "replaced": "по дочерним периодам"}
CoverageState = Literal["complete", "incomplete", "unknown", "replaced"]


class PeriodProgress(Contract):
    id: str
    parent_id: str | None
    source: str
    from_date: date
    until_date: date
    granularity: str
    state: str
    coverage: CoverageState
    full_calendar_period: bool
    attempt_count: int
    job_id: str | None
    updated_at: datetime | None
    scanned: int | None
    stored: int | None
    skipped: int | None
    total_available: int | None
    effective_limit: int
    incomplete_reason: str | None


class HistoryProgress(Contract):
    id: str
    topic: str
    state: str
    total_periods: int
    processed_periods: int
    execution_percent: float
    completed_periods: int
    partial_periods: int
    failed_periods: int
    pending_periods: int
    active_periods: int
    cancelled_periods: int
    interrupted_periods: int
    split_periods: int
    total_plan_periods: int
    coverage_complete: bool
    partial_calendar_years: tuple[int, ...]
    error_code: str | None
    periods: tuple[PeriodProgress, ...]
    contract_version: int = 1


def history_progress(report: HistoryReport) -> HistoryProgress:
    periods = []
    for period in report.periods:
        job = period.job
        coverage: CoverageState = ("replaced" if period.state == "split" else "complete" if period.state == "complete"
                                   else "unknown" if period.state in {"pending", "queued", "running"} else "incomplete")
        periods.append(PeriodProgress(
            id=period.id, parent_id=period.parent_id, source=period.source,
            from_date=period.from_date, until_date=period.until_date, granularity=period.granularity,
            state=period.state, coverage=coverage, full_calendar_period=period.full_calendar_period,
            attempt_count=len(period.attempts), job_id=job.id if job else None,
            updated_at=job.updated_at if job else None,
            scanned=job.scanned if job else None, stored=job.stored if job else None,
            skipped=job.skipped if job else None, total_available=job.total_available if job else None,
            effective_limit=min(report.request.max_results_per_period, 2000) if period.source == "epo"
                            else report.request.max_results_per_period,
            incomplete_reason=period.incomplete_reason,
        ))
    counts = {state: sum(p.state == state for p in periods) for state in STATES}
    return HistoryProgress(
        id=report.id, topic=report.request.topic, state=report.state,
        total_periods=report.total_periods, processed_periods=report.processed_periods,
        execution_percent=round(100 * report.processed_periods / report.total_periods, 1) if report.total_periods else 0,
        completed_periods=report.completed_periods, partial_periods=report.partial_periods,
        failed_periods=report.failed_periods, pending_periods=counts["pending"],
        active_periods=counts["queued"] + counts["running"], cancelled_periods=counts["cancelled"],
        interrupted_periods=counts["interrupted"], split_periods=report.split_periods,
        total_plan_periods=report.total_plan_periods, coverage_complete=report.coverage_complete,
        partial_calendar_years=report.partial_calendar_years, error_code=report.error_code, periods=tuple(periods),
    )


def _safe(value):
    return "".join(char if char.isprintable() else " " for char in str(value))


def progress_summary(progress):
    return (f"{STATES.get(progress.state, progress.state)}. Обработано конечных периодов: "
            f"{progress.processed_periods}/{progress.total_periods} ({progress.execution_percent:.1f}% выполнения); "
            f"полных: {progress.completed_periods}, неполных: {progress.partial_periods}, ошибок: {progress.failed_periods}. "
            f"Полнота всего запроса: {'полная' if progress.coverage_complete else 'не подтверждена'}.")


def period_line(period):
    def number(value):
        return str(value) if value is not None else "?"
    text = (f"[{period.id[:8]}] {period.source} {period.from_date}..{period.until_date} ({period.granularity}) | "
            f"{STATES.get(period.state, period.state)} | просмотрено/сохранено/пропущено: "
            f"{number(period.scanned)}/{number(period.stored)}/{number(period.skipped)} | "
            f"всего у источника: {number(period.total_available)} | лимит: {period.effective_limit} | "
            f"полнота: {COVERAGE[period.coverage]} | попыток: {period.attempt_count}")
    if period.incomplete_reason:
        text += f" | {REASONS.get(period.incomplete_reason, period.incomplete_reason)}"
    if not period.full_calendar_period:
        text += " | неполный календарный период"
    return _safe(text)


def render_history_progress(progress):
    lines = [f"Исторический сбор: {_safe(progress.id)} — {_safe(progress.topic)}", progress_summary(progress),
             "Процент выполнения не означает полноту данных. При дроблении число периодов увеличивается.",
             f"Всего узлов: {progress.total_plan_periods}; разделённых родителей: {progress.split_periods}."]
    if progress.partial_calendar_years:
        lines.append("Годы, охваченные не целиком: " + ", ".join(map(str, progress.partial_calendar_years)))
    if progress.error_code:
        lines.append("Ошибка исследования: " + _safe(progress.error_code))
    return "\n".join(lines + [period_line(period) for period in progress.periods])


class ProgressPrinter:
    """Emit only changed states; final fast jobs are included even without a timeout."""
    def __init__(self, stream):
        self.stream = stream
        self.previous = {}
        self.summary = None

    def update(self, report):
        progress = history_progress(report)
        summary = progress_summary(progress)
        if summary != self.summary:
            print(summary, file=self.stream, flush=True)
            self.summary = summary
        for period in progress.periods:
            if period.state != "pending" and self.previous.get(period.id) != period:
                print(period_line(period), file=self.stream, flush=True)
            self.previous[period.id] = period
