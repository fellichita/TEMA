"""Клиент веб-интерфейса: один запрос без повторов и без загрузки моделей."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
import json
import os
import re
from threading import BoundedSemaphore
from urllib.parse import urlencode, urlsplit

import requests

from app.input_safety import is_safe_http_url
from app.ui.radar_client import RadarFormatError, RadarStatus, parse_radar

MAX_QUERY_LENGTH = 2000
MAX_RESPONSE_BYTES = 2_000_000
PUBLICATION_PAGE_SIZE = 100
MAX_PUBLICATION_OFFSET = 1_000_000
ANALYSIS_MODES = frozenset({"fast", "deep"})
SIGNAL_CATEGORIES = frozenset({"confirmed_trend", "early_signal", "weak_signal_candidate",
                               "emerging_candidate"})
PROGRESS_STAGE = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
MAX_RUN_SECONDS = 7 * 24 * 3600
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
TREND_FEATURES = ("growth", "persistence", "novelty", "independence", "application")
MONTH_PATTERN = re.compile(r"[0-9]{4}-(?:0[1-9]|1[0-2])\Z")
RESULT_VERSION = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
LANGUAGE = re.compile(r"[a-z]{2,3}\Z")
HISTORY_PAGE_SIZE = 20
VISITOR = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
RUN_STATES = frozenset({"queued", "running", "succeeded", "failed", "cancelled", "interrupted"})


class ApiError(Exception):
    """Безопасное сообщение, которое можно показать пользователю."""


class AccessRefused(ApiError):
    """Владелец закрыл этому посетителю вход: `blocked` — блокировка, `signed_out` — выгнал."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__("Доступ к сервису закрыт владельцем." if reason == "blocked"
                         else "Владелец завершил ваш сеанс. Войдите снова.")


@dataclass(frozen=True)
class Signal:
    title: str
    summary: str
    category: str
    source_urls: tuple[str, ...]
    # Оценка методики, а не вероятность; у старых результатов её нет.
    confidence: str | None = None
    growth_confirmed: bool = False
    checked_features: tuple[str, ...] = ()
    unchecked_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trend:
    """Оценённая методикой тема, в которую входит публикация."""
    title: str
    confidence: str
    growth_confirmed: bool = False
    checked_features: tuple[str, ...] = ()
    unchecked_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfidence:
    """Порядковая оценка моделью соответствия публикации запросу, не вероятность."""
    score: int
    reason: str
    evidence_quote: str
    basis: str


@dataclass(frozen=True)
class SimilarMonth:
    month: str
    count: int
    # Сколько материалов любой темы выборка собрала за месяц.
    collected: int


@dataclass(frozen=True)
class SimilarActivity:
    """Похожие материалы собранной выборки по месяцам, а не по всему миру."""
    months: tuple[SimilarMonth, ...]
    total: int


@dataclass(frozen=True)
class Publication:
    publication_id: str
    source_id: str
    kind: str
    title: str
    url: str
    published_at: date | None
    publication_year: int | None
    date_basis: str
    summary: str | None
    # Нет у публикации вне оценённых тем и у старых результатов.
    trend: Trend | None = None
    # Есть только у публикаций, для которых модель вернула проверяемую оценку.
    model_confidence: ModelConfidence | None = None
    # Есть только у публикаций ТОПа.
    similar: SimilarActivity | None = None


@dataclass(frozen=True)
class SourceCoverage:
    source_id: str
    state: str
    scanned: int
    accepted: int
    limit_reached: bool
    reason_code: str | None


@dataclass(frozen=True)
class ArxivDomainMonth:
    month: date
    domain: str
    primary_category: str
    article_count: int


@dataclass(frozen=True)
class ArxivDomains:
    coverage_state: str
    scanned: int
    months: tuple[ArxivDomainMonth, ...]


@dataclass(frozen=True)
class FundingAward:
    application_id: int
    title: str
    award_notice_date: date
    award_amount_usd: str
    funding_type: str
    detail_url: str


@dataclass(frozen=True)
class FundingEvidence:
    source_id: str
    from_date: date
    to_date: date
    awards: tuple[FundingAward, ...]
    total_available: int | None
    partial_coverage: bool
    coverage_reason: str
    topic: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    signals: tuple[Signal, ...]
    omitted: int = 0
    incomplete_coverage: bool = False
    publications: tuple[Publication, ...] = ()
    top_publications: tuple[Publication, ...] = ()
    publication_total: int = 0
    source_coverage: tuple[SourceCoverage, ...] = ()
    openalex_rate_limited: bool = False
    arxiv_domains: ArxivDomains | None = None
    funding_evidence: FundingEvidence | None = None


@dataclass(frozen=True)
class Translation:
    """Машинный перевод текстов ТОПа: черновик для чтения, не оригинал.

    `language` — язык перевода (по умолчанию русский), `languages` — языки,
    которые владелец установил и на которые можно переключиться: (код, название).
    """
    state: str
    completed: int = 0
    total: int = 0
    texts: dict[str, str] = field(default_factory=dict)
    message: str | None = None
    language: str | None = None
    languages: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AnalysisStatus:
    state: str
    id: str | None = None
    query: str = ""
    mode: str = "fast"
    result: AnalysisResult | None = None
    error: str | None = None
    stage: str | None = None
    message: str = ""
    completed: int = 0
    total: int = 0
    elapsed_seconds: int | None = None
    expected_seconds: int | None = None
    translation: Translation | None = None
    radar: RadarStatus | None = None
    # Время запуска; сервис сообщает его только для анализа из истории.
    created_at: datetime | None = None
    # Сервис занят чужим анализом (при разделении посетителей); свой пока не запустить.
    service_busy: bool = False
    # Владелец временно приостановил новые анализы; готовые результаты доступны.
    paused: bool = False
    # Все места заняты: анализ ждёт в очереди под этим номером (1 — следующий).
    queue_position: int | None = None
    # Сколько анализов уже ждёт в очереди: новый встанет за ними.
    queue_length: int = 0
    # Глубокий режим владелец временно закрыл для гостей.
    deep_disabled: bool = False
    # Все места для анализа заняты: новый анализ встанет в очередь.
    slots_full: bool = False


@dataclass(frozen=True)
class HistoryEntry:
    """Сохранённый анализ профиля: только то, что нужно строке списка."""
    id: str
    query: str
    # Нет у анализов, начатых без выбора режима (например, в настольном приложении).
    mode: str | None
    state: str
    created_at: datetime


@dataclass(frozen=True)
class HistoryPage:
    entries: tuple[HistoryEntry, ...]
    offset: int
    has_more: bool
    # Анализ, который сейчас показывают все веб-сессии.
    current_id: str | None = None


@dataclass(frozen=True)
class ModeEstimate:
    expected_seconds: int
    basis: str
    runs: int
    documents: int
    top: int
    patents: bool


def _seconds(payload: dict, key: str, *, minimum: int) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int or not minimum <= value <= MAX_RUN_SECONDS:
        raise ApiError("Сервис вернул некорректное время анализа.")
    return value


def parse_estimates(payload: object) -> dict[str, ModeEstimate]:
    modes = payload.get("modes") if isinstance(payload, dict) else None
    if not isinstance(modes, dict) or set(modes) != ANALYSIS_MODES:
        raise ApiError("Сервис вернул прогноз неизвестного формата.")
    estimates = {}
    for mode, item in modes.items():
        if not isinstance(item, dict):
            raise ApiError("Сервис вернул некорректный прогноз режима.")
        expected = _seconds(item, "expected_seconds", minimum=1)
        runs, documents, top = item.get("runs"), item.get("documents"), item.get("top")
        if (expected is None or item.get("basis") not in {"history", "default"}
                or type(runs) is not int or not 0 <= runs <= 100
                or type(documents) is not int or not 1 <= documents <= 100_000
                or type(top) is not int or not 1 <= top <= 100
                or type(item.get("patents")) is not bool):
            raise ApiError("Сервис вернул некорректный прогноз режима.")
        estimates[mode] = ModeEstimate(expected, item["basis"], runs, documents, top, item["patents"])
    return estimates


def parse_history(payload: object, *, offset: int, limit: int) -> HistoryPage:
    """Страница истории: чужая страница, дубли и лишние строки отвергаются целиком."""
    error = ApiError("Сервис вернул некорректную историю анализов.")
    if (not isinstance(payload, dict) or type(payload.get("offset")) is not int
            or payload["offset"] != offset or type(payload.get("has_more")) is not bool
            or not isinstance(payload.get("runs"), list) or len(payload["runs"]) > limit):
        raise error
    current = payload.get("current_id")
    if current is not None and (not isinstance(current, str) or RUN_ID.fullmatch(current) is None):
        raise error
    entries = []
    for item in payload["runs"]:
        if not isinstance(item, dict):
            raise error
        identifier, query, mode, state, created = (
            item.get(key) for key in ("id", "query", "mode", "state", "created_at"))
        if (not isinstance(identifier, str) or RUN_ID.fullmatch(identifier) is None
                or not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH
                or mode is not None and mode not in ANALYSIS_MODES
                or state not in RUN_STATES or created is None):
            raise error
        created_at = _moment(created, str(error))
        assert created_at is not None
        entries.append(HistoryEntry(identifier, query.strip(), mode, state, created_at))
    if len({entry.id for entry in entries}) != len(entries):
        raise error
    return HistoryPage(tuple(entries), offset, payload["has_more"], current)


def parse_translation(payload: object) -> Translation | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("state") not in {"running", "ready", "unavailable"}:
        raise ApiError("Сервис вернул некорректный перевод ТОПа.")
    completed, total = payload.get("completed", 0), payload.get("total", 0)
    texts, message = payload.get("texts", {}), payload.get("message")
    if (type(completed) is not int or type(total) is not int or not 0 <= completed <= total <= 500
            or not isinstance(texts, dict) or len(texts) > 500
            or any(not isinstance(key, str) or not isinstance(value, str)
                   or len(key) > 20_000 or len(value) > 20_000 for key, value in texts.items())
            or message is not None and (not isinstance(message, str) or len(message) > 500)):
        raise ApiError("Сервис вернул некорректный перевод ТОПа.")
    language, raw_languages = payload.get("language"), payload.get("languages", [])
    if (language is not None and (not isinstance(language, str) or LANGUAGE.fullmatch(language) is None)
            or not isinstance(raw_languages, list) or len(raw_languages) > 20):
        raise ApiError("Сервис вернул некорректный перевод ТОПа.")
    languages = []
    for entry in raw_languages:
        if (not isinstance(entry, dict) or not isinstance(entry.get("code"), str)
                or LANGUAGE.fullmatch(entry["code"]) is None or not isinstance(entry.get("name"), str)
                or not 1 <= len(entry["name"]) <= 40):
            raise ApiError("Сервис вернул некорректный перевод ТОПа.")
        languages.append((entry["code"], entry["name"]))
    return Translation(payload["state"], completed, total, dict(texts), message, language, tuple(languages))


def parse_status(payload: object, *, cached_result: AnalysisResult | None = None) -> AnalysisStatus:
    if not isinstance(payload, dict):
        raise ApiError("Сервис вернул некорректное состояние анализа.")
    state = payload.get("state")
    busy, paused = payload.get("service_busy", False), payload.get("paused", False)
    deep_disabled, queue_length = payload.get("deep_disabled", False), payload.get("queue_length", 0)
    slots_full = payload.get("slots_full", False)
    if (type(busy) is not bool or type(paused) is not bool or type(deep_disabled) is not bool
            or type(slots_full) is not bool
            or type(queue_length) is not int or not 0 <= queue_length <= 1_000):
        raise ApiError("Сервис вернул некорректное состояние анализа.")
    if state == "idle":
        return AnalysisStatus(state="idle", service_busy=busy, paused=paused, queue_length=queue_length,
                              deep_disabled=deep_disabled, slots_full=slots_full)
    position = payload.get("queue_position")
    if state == "waiting" and (type(position) is not int or not 1 <= position <= 1_000):
        raise ApiError("Сервис вернул некорректное место в очереди.")
    if state not in {"waiting", "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"}:
        raise ApiError("Сервис вернул неизвестное состояние анализа.")
    identifier, query, mode = payload.get("id"), payload.get("query"), payload.get("mode")
    if (not isinstance(identifier, str) or not identifier or
            not isinstance(query, str) or len(query) > MAX_QUERY_LENGTH or
            not isinstance(mode, str) or mode not in ANALYSIS_MODES):
        raise ApiError("Сервис вернул неполное состояние анализа.")
    if state == "succeeded" and "result" not in payload and cached_result is None:
        raise ApiError("Сервис не вернул результат завершённого анализа.")
    result = None
    if state == "succeeded":
        result = parse_result(payload["result"]) if "result" in payload else cached_result
    error = payload.get("error")
    progress_fields = {"stage", "message", "completed", "total"}
    if progress_fields & payload.keys():
        if not progress_fields <= payload.keys():
            raise ApiError("Сервис вернул неполный прогресс анализа.")
        stage, message = payload["stage"], payload["message"]
        completed, total = payload["completed"], payload["total"]
        if (not isinstance(stage, str) or PROGRESS_STAGE.fullmatch(stage) is None
                or not isinstance(message, str) or len(message) > 1_000
                or type(completed) is not int or type(total) is not int
                or not 0 <= completed <= total):
            raise ApiError("Сервис вернул некорректный прогресс анализа.")
    else:
        stage, message, completed, total = None, "", 0, 0
    try:
        radar = parse_radar(payload.get("radar")) if state == "succeeded" else None
    except RadarFormatError:
        raise ApiError("Сервис вернул некорректный ТОП технологий.") from None
    return AnalysisStatus(state, identifier, query, mode, result,
                          error if isinstance(error, str) else None,
                          stage, message, completed, total,
                          _seconds(payload, "elapsed_seconds", minimum=0),
                          _seconds(payload, "expected_seconds", minimum=1),
                          parse_translation(payload.get("translation")) if state == "succeeded" else None,
                          radar, _moment(payload.get("created_at"), "Сервис вернул некорректное время анализа."),
                          busy, paused, position if state == "waiting" else None, queue_length, deep_disabled,
                          slots_full)


def _moment(value: object, message: str) -> datetime | None:
    """ISO time from the service; a time without a zone is UTC."""
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 40:
        raise ApiError(message)
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise ApiError(message) from None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def parse_confidence(item: dict) -> tuple[str | None, bool, tuple[str, ...], tuple[str, ...]]:
    """Уверенность в тренде у сигнала или темы публикации: уровень и проверенные признаки."""
    confidence = item.get("confidence")
    checked, unchecked = item.get("checked_features", []), item.get("unchecked_features", [])
    if (confidence is not None and confidence not in CONFIDENCE_LEVELS
            or type(item.get("growth_confirmed", False)) is not bool
            or not isinstance(checked, list) or not isinstance(unchecked, list)
            or any(feature not in TREND_FEATURES for feature in [*checked, *unchecked])
            or len(set(checked) | set(unchecked)) != len(checked) + len(unchecked)):
        raise ApiError("Сервис вернул некорректную уверенность в тренде.")
    return confidence, item.get("growth_confirmed", False), tuple(checked), tuple(unchecked)


def parse_trend(item: object) -> Trend | None:
    if item is None:
        return None
    title = item.get("title") if isinstance(item, dict) else None
    if not isinstance(title, str) or not title.strip() or len(title) > 500:
        raise ApiError("Сервис вернул некорректную тему публикации.")
    assert isinstance(item, dict)
    confidence = parse_confidence(item)
    level = confidence[0]
    if level is None:
        raise ApiError("Сервис вернул некорректную уверенность в тренде.")
    return Trend(title.strip(), level, *confidence[1:])


def parse_model_confidence(item: object) -> ModelConfidence | None:
    if item is None:
        return None
    if not isinstance(item, dict):
        raise ApiError("Сервис вернул некорректную оценку модели для публикации.")
    score, reason, evidence_quote, basis = (
        item.get(key) for key in ("score", "reason", "evidence_quote", "basis"))
    if (type(score) is not int or not 0 <= score <= 100
            or not isinstance(reason, str) or not reason.strip() or len(reason) > 500
            or not isinstance(evidence_quote, str) or not evidence_quote.strip()
            or len(evidence_quote) > 300 or not isinstance(basis, str)
            or basis not in {"title", "title_and_summary"}):
        raise ApiError("Сервис вернул некорректную оценку модели для публикации.")
    return ModelConfidence(score, reason.strip(), evidence_quote.strip(), basis)


def parse_similar(item: object) -> SimilarActivity | None:
    if item is None:
        return None
    error = ApiError("Сервис вернул некорректный график похожих материалов.")
    months = item.get("months") if isinstance(item, dict) else None
    total = item.get("total") if isinstance(item, dict) else None
    if not isinstance(months, list) or not 1 <= len(months) <= 36 or type(total) is not int:
        raise error
    rows = []
    for row in months:
        month, count, collected = ((row.get("month"), row.get("count"), row.get("collected"))
                                   if isinstance(row, dict) else (None, None, None))
        if (not isinstance(month, str) or MONTH_PATTERN.fullmatch(month) is None
                or type(count) is not int or type(collected) is not int or not 0 <= count <= collected):
            raise error
        rows.append(SimilarMonth(month, count, collected))
    if [row.month for row in rows] != sorted({row.month for row in rows}) or total != sum(row.count for row in rows):
        raise error
    return SimilarActivity(tuple(rows), total)


def parse_arxiv_domains(value: object) -> ArxivDomains:
    if not isinstance(value, dict) or value.get("coverage_state") not in {
            "complete", "partial", "unavailable"}:
        raise ApiError("Сервис вернул некорректную сводку направлений arXiv.")
    scanned, rows = value.get("scanned"), value.get("months")
    if type(scanned) is not int or not 0 <= scanned <= 10000 or not isinstance(rows, list) or len(rows) > 1000:
        raise ApiError("Сервис вернул некорректную сводку направлений arXiv.")
    months = []
    seen: set[tuple[date, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ApiError("Сервис вернул некорректную сводку направлений arXiv.")
        month, domain, category, count = (row.get(key) for key in
                                           ("month", "domain", "primary_category", "article_count"))
        if (not isinstance(month, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-01", month) is None
                or not isinstance(category, str)
                or len(category) > 64
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)?", category) is None
                or domain != category.split(".", 1)[0]
                or type(count) is not int or not 1 <= count <= scanned):
            raise ApiError("Сервис вернул некорректную сводку направлений arXiv.")
        try:
            parsed = date.fromisoformat(month)
        except ValueError:
            raise ApiError("Сервис вернул некорректную сводку направлений arXiv.") from None
        key = (parsed, category)
        if key in seen:
            raise ApiError("Сервис вернул некорректную сводку направлений arXiv.")
        seen.add(key)
        months.append(ArxivDomainMonth(parsed, domain, category, count))
    if sum(row.article_count for row in months) > scanned:
        raise ApiError("Сервис вернул некорректную сводку направлений arXiv.")
    return ArxivDomains(value["coverage_state"], scanned, tuple(months))


def parse_funding_evidence(value: object) -> FundingEvidence:
    if not isinstance(value, dict) or value.get("source_id") != "nih_reporter":
        raise ApiError("Сервис вернул некорректную сводку финансирования.")
    reason = value.get("coverage_reason")
    topic = value.get("topic")
    partial = value.get("partial_coverage")
    total = value.get("total_available")
    rows = value.get("awards")
    if (topic is not None and (not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 180)
            or reason not in {"complete", "page_cap", "invalid_records", "unknown_total", "source_unavailable"}
            or type(partial) is not bool or partial != (reason != "complete")
            or total is not None and (type(total) is not int or total < 0)
            or not isinstance(rows, list) or len(rows) > 50):
        raise ApiError("Сервис вернул некорректную сводку финансирования.")
    start_text, end_text = value.get("from_date"), value.get("to_date")
    if (not isinstance(start_text, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", start_text) is None
            or not isinstance(end_text, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", end_text) is None):
        raise ApiError("Сервис вернул некорректную сводку финансирования.")
    try:
        first = date.fromisoformat(start_text)
        last = date.fromisoformat(end_text)
    except (KeyError, TypeError, ValueError):
        raise ApiError("Сервис вернул некорректную сводку финансирования.") from None
    if first > last:
        raise ApiError("Сервис вернул некорректную сводку финансирования.")
    awards = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ApiError("Сервис вернул некорректную сводку финансирования.")
        identity, title, date_text, amount, kind, url = (row.get(key) for key in
            ("application_id", "title", "award_notice_date", "award_amount_usd", "funding_type", "detail_url"))
        if (type(identity) is not int or identity <= 0 or identity in seen
                or not isinstance(title, str) or not 1 <= len(title.strip()) <= 500
                or not isinstance(amount, str) or len(amount) > 128
                or re.fullmatch(r"[0-9]{1,13}(?:\.[0-9]+)?", amount) is None
                or kind not in {"grant_or_cooperative", "contract", "intramural", "other_or_unknown"}
                or url != f"https://reporter.nih.gov/project-details/{identity}"):
            raise ApiError("Сервис вернул некорректную сводку финансирования.")
        if (not isinstance(date_text, str)
                or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_text) is None
                or Decimal(amount) > 1_000_000_000_000):
            raise ApiError("Сервис вернул некорректную сводку финансирования.")
        try:
            award_date = date.fromisoformat(date_text)
        except (TypeError, ValueError):
            raise ApiError("Сервис вернул некорректную сводку финансирования.") from None
        if not first <= award_date <= last:
            raise ApiError("Сервис вернул некорректную сводку финансирования.")
        seen.add(identity)
        awards.append(FundingAward(identity, title.strip(), award_date, amount, kind, url))
    if total is not None and total < len(awards):
        raise ApiError("Сервис вернул некорректную сводку финансирования.")
    return FundingEvidence("nih_reporter", first, last, tuple(awards), total, partial, reason,
                           topic.strip() if isinstance(topic, str) else None)


def parse_publication(item: object) -> Publication:
    """Validate a publication shared by the initial result and later pages."""
    if not isinstance(item, dict):
        raise ApiError("Сервис вернул некорректную публикацию.")
    publication_id, source_id, kind, title = (
        item.get(key) for key in ("publication_id", "source_id", "kind", "title"))
    url, published_at, summary = (item.get(key) for key in ("url", "published_at", "summary"))
    publication_year = item.get("publication_year")
    date_basis = item.get("date_basis", "unknown")
    if (not isinstance(publication_id, str) or re.fullmatch(r"[0-9a-f]{64}", publication_id) is None
            or not isinstance(source_id, str) or PROGRESS_STAGE.fullmatch(source_id) is None
            or not isinstance(kind, str) or not kind.strip() or len(kind) > 100
            or not isinstance(title, str) or not title.strip() or len(title) > 1000
            or not isinstance(url, str) or len(url) > 4096 or not is_safe_http_url(url)
            or published_at is not None and (not isinstance(published_at, str)
                                             or len(published_at) != 10)
            or publication_year is not None and (type(publication_year) is not int
                                                 or not 1 <= publication_year <= 9999)
            or date_basis not in {"published", "indexed", "created", "unknown"}
            or summary is not None and (not isinstance(summary, str) or len(summary) > 1500)):
        raise ApiError("Сервис вернул некорректную публикацию.")
    try:
        published_date = date.fromisoformat(published_at) if published_at else None
    except ValueError:
        raise ApiError("Сервис вернул некорректную дату публикации.") from None
    return Publication(publication_id, source_id, kind, title.strip(), url, published_date,
                       publication_year, date_basis, summary, parse_trend(item.get("trend")),
                       parse_model_confidence(item.get("model_confidence")),
                       parse_similar(item.get("similar")))


def parse_publication_page(payload: object, *, run_id: str, total: int, offset: int,
                           limit: int, known_ids: frozenset[str] = frozenset()) -> tuple[Publication, ...]:
    """Reject stale, duplicated, oversized, or partial pages before showing them."""
    if (not isinstance(payload, dict) or payload.get("run_id") != run_id
            or type(payload.get("total")) is not int or payload["total"] != total
            or type(payload.get("offset")) is not int or payload["offset"] != offset):
        raise ApiError("Список публикаций относится к другому анализу. Обновите страницу.")
    raw = payload.get("publications")
    if (not isinstance(raw, list) or len(raw) != min(limit, total - offset)):
        raise ApiError("Сервис вернул неполную или слишком большую страницу публикаций.")
    publications = tuple(parse_publication(item) for item in raw)
    identifiers = {item.publication_id for item in publications}
    if len(identifiers) != len(publications) or identifiers & known_ids:
        raise ApiError("Сервис вернул повторяющиеся публикации.")
    return publications


def parse_result(payload: object) -> AnalysisResult:
    """Проверяем формат; достоверность источников обязан проверять пайплайн."""
    if not isinstance(payload, dict) or not isinstance(payload.get("signals"), list):
        raise ApiError("Сервис вернул ответ неизвестного формата. Проверьте версию API.")
    if len(payload["signals"]) > 100:
        raise ApiError("Сервис вернул слишком много сигналов для одного запроса.")
    incomplete_coverage = payload.get("incomplete_coverage", False)
    openalex_rate_limited = payload.get("openalex_rate_limited", False)
    if type(incomplete_coverage) is not bool or type(openalex_rate_limited) is not bool:
        raise ApiError("Сервис вернул некорректную оценку охвата источников.")
    signals = []
    omitted = 0
    for item in payload["signals"]:
        if not isinstance(item, dict):
            raise ApiError("Сервис вернул некорректную карточку результата.")
        urls = item.get("source_urls", [])
        if not isinstance(urls, list) or len(urls) > 100:
            raise ApiError("Сервис вернул некорректный список источников.")
        urls = tuple(dict.fromkeys(url for url in urls if is_safe_http_url(url)))
        if not urls:
            omitted += 1
            continue
        for key, limit in (("title", 500), ("summary", 10000)):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ApiError("Сервис вернул неполную карточку результата.")
        if item.get("category") not in SIGNAL_CATEGORIES:
            raise ApiError("Сервис вернул неизвестный статус сигнала.")
        signals.append(Signal(item["title"].strip(), item["summary"].strip(),
                              item["category"], urls, *parse_confidence(item)))
    publications_payload = payload.get("publications", [])
    top_publications_payload = payload.get("top_publications", publications_payload[:15]
                                   if isinstance(publications_payload, list) else None)
    publication_total = payload.get("publication_total", len(publications_payload)
                                    if isinstance(publications_payload, list) else None)
    coverage_payload = payload.get("source_coverage", [])
    if (not isinstance(publications_payload, list) or len(publications_payload) > 200
            or not isinstance(top_publications_payload, list) or len(top_publications_payload) > 15
            or type(publication_total) is not int or publication_total < len(publications_payload)
            or not isinstance(coverage_payload, list) or len(coverage_payload) > 50):
        raise ApiError("Сервис вернул некорректный список публикаций или источников.")

    publications = tuple(parse_publication(item) for item in publications_payload)
    top_publications = tuple(parse_publication(item) for item in top_publications_payload)
    publication_ids = {item.publication_id for item in publications}
    if (len(publication_ids) != len(publications)
            or top_publications != publications[:15]):
        raise ApiError("Сервис вернул несовместимые списки публикаций.")
    coverage = []
    seen_sources: set[str] = set()
    for item in coverage_payload:
        if not isinstance(item, dict):
            raise ApiError("Сервис вернул некорректный статус источника.")
        source_id, state = item.get("source_id"), item.get("state")
        scanned, accepted = item.get("scanned"), item.get("accepted")
        limit_reached, reason_code = item.get("limit_reached"), item.get("reason_code")
        if (not isinstance(source_id, str) or PROGRESS_STAGE.fullmatch(source_id) is None
                or source_id in seen_sources or state not in {"complete", "partial", "unavailable"}
                or type(scanned) is not int or scanned < 0
                or type(accepted) is not int or not 0 <= accepted <= scanned
                or type(limit_reached) is not bool
                or reason_code is not None and (not isinstance(reason_code, str)
                                                or PROGRESS_STAGE.fullmatch(reason_code) is None)):
            raise ApiError("Сервис вернул некорректный статус источника.")
        seen_sources.add(source_id)
        coverage.append(SourceCoverage(source_id, state, scanned, accepted,
                                       limit_reached, reason_code))
    if coverage and any(item.source_id not in seen_sources for item in publications):
        raise ApiError("Сервис вернул публикации без статуса источника.")
    arxiv_domains = parse_arxiv_domains(payload["arxiv_domains"]) if "arxiv_domains" in payload else None
    funding_evidence = (parse_funding_evidence(payload["funding_evidence"])
                        if "funding_evidence" in payload else None)
    return AnalysisResult(tuple(signals), omitted, incomplete_coverage,
                          publications, top_publications, publication_total,
                          tuple(coverage), openalex_rate_limited,
                          arxiv_domains, funding_evidence)


def _refusal(response) -> str | None:
    try:
        error = json.loads(response.content[:1_000]).get("error")
    except (ValueError, AttributeError):
        return None
    return error if error in {"blocked", "signed_out"} else None


class ApiClient:
    """Один рабочий поток; повторная отправка не создаёт очередь анализов."""

    def __init__(self, api_url: str, visitor: str | None = None):
        if not is_safe_http_url(api_url) or urlsplit(api_url).query or urlsplit(api_url).fragment:
            raise ApiError("Не настроен адрес сервиса анализа. Укажите API_URL и перезапустите интерфейс.")
        if visitor is not None and (not isinstance(visitor, str) or VISITOR.fullmatch(visitor) is None):
            raise ApiError("Сеанс посетителя повреждён. Войдите заново.")
        token = os.environ.get("TREND_API_TOKEN", "")
        if token and (urlsplit(api_url).hostname != "127.0.0.1" or urlsplit(api_url).scheme != "http"
                      or not 32 <= len(token) <= 256 or not token.isascii()
                      or not token.isprintable()):
            raise ApiError("Небезопасное соединение с сервисом анализа. Проверьте настройку сервера.")
        self.endpoint = api_url.rstrip("/") + "/analyze"
        self.base_url = api_url.rstrip("/")
        self._headers = {"X-Trend-API-Token": token} if token else {}
        if visitor is not None:
            # Один клиент — один посетитель: API отдаёт ему только его анализы.
            self._headers["X-Trend-Visitor"] = visitor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-api")
        self._available = BoundedSemaphore(1)
        self._result_cache: tuple[str, str, AnalysisResult] | None = None
        # run_id → (версия, результат) открытых из истории анализов.
        self._saved_cache: dict[str, tuple[str, AnalysisResult]] = {}

    def _json_request(self, method: str, path: str, payload: dict | None = None,
                      *, extra_headers: dict[str, str] | None = None,
                      messages: dict[int, str] | None = None) -> object:
        try:
            # The first completed status includes the assembled publication list;
            # reading archived discovery records can take longer than a progress poll.
            read_timeout = (30 if method == "GET" and path.startswith(("/analyses/current", "/analyses/saved"))
                            else 10)
            with requests.Session() as session:
                session.trust_env = False
                headers = self._headers | extra_headers if extra_headers else self._headers
                with session.request(method, self.base_url + path, json=payload, timeout=(5, read_timeout),
                                     headers=headers, allow_redirects=False, stream=True) as response:
                    if response.status_code == 403:
                        # Отказ владельца отличается от прочих: странице надо показать не ошибку, а вход.
                        reason = _refusal(response)
                        if reason is not None:
                            raise AccessRefused(reason)
                    if response.status_code not in {200, 202}:
                        messages = {
                            409: "Состояние анализа изменилось. Обновите страницу.",
                            429: "Сервис уже выполняет анализ. Обновите страницу для просмотра состояния.",
                            422: "Сервис не принял запрос. Проверьте тему и режим анализа.",
                            423: "Владелец сервиса временно приостановил новые анализы. Попробуйте позже.",
                            503: "Сервис анализа ещё не готов. Попробуйте немного позже.",
                        } | (messages or {})
                        raise ApiError(messages.get(response.status_code,
                            "Не удалось получить состояние анализа. Повторите попытку."))
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ApiError("Ответ сервиса слишком большой.")
                        chunks.append(chunk)
                    return json.loads(b"".join(chunks))
        except requests.RequestException:
            raise ApiError("Нет связи с сервисом анализа. Проверьте, что веб-демо запущено.") from None
        except (ValueError, UnicodeError):
            raise ApiError("Не удалось прочитать состояние анализа.") from None

    def _status_request(self, method: str, path: str, payload: dict | None = None) -> AnalysisStatus:
        response = self._json_request(method, path, payload)
        try:
            return parse_status(response)
        except (ValueError, UnicodeError):
            raise ApiError("Не удалось прочитать состояние анализа.") from None

    def current_analysis(self, language: str | None = None) -> AnalysisStatus:
        cached = self._result_cache
        headers = {"X-Trend-Result-Version": cached[1]} if cached else {}
        if language is not None and LANGUAGE.fullmatch(language):
            headers["X-Trend-Language"] = language
        response = self._json_request("GET", "/analyses/current", extra_headers=headers or None)
        version = response.get("result_version") if isinstance(response, dict) else None
        if version is not None and (not isinstance(version, str)
                                    or RESULT_VERSION.fullmatch(version) is None):
            raise ApiError("Сервис вернул некорректную версию результата.")
        reusable = (cached[2] if cached is not None and isinstance(response, dict)
                    and response.get("id") == cached[0] and version == cached[1] else None)
        status = parse_status(response, cached_result=reusable)
        if status.state == "succeeded" and status.id is not None and status.result is not None:
            if version is not None:
                self._result_cache = (status.id, version, status.result)
        else:
            self._result_cache = None
        return status

    def publication_page(self, run_id: str, *, offset: int, limit: int = PUBLICATION_PAGE_SIZE,
                         expected_total: int, known_ids: frozenset[str] = frozenset()
                         ) -> tuple[Publication, ...]:
        """Load only the next bounded page of the current completed analysis."""
        if (not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None
                or type(offset) is not int or not 0 <= offset <= MAX_PUBLICATION_OFFSET
                or type(limit) is not int or not 1 <= limit <= PUBLICATION_PAGE_SIZE
                or type(expected_total) is not int or not offset < expected_total):
            raise ApiError("Не удалось запросить страницу публикаций.")
        query = urlencode({"run_id": run_id, "offset": offset, "limit": limit})
        response = self._json_request("GET", "/analyses/current/publications?" + query)
        return parse_publication_page(response, run_id=run_id, total=expected_total,
                                      offset=offset, limit=limit, known_ids=known_ids)

    def history(self, offset: int = 0, limit: int = HISTORY_PAGE_SIZE) -> HistoryPage:
        """Сохранённые анализы профиля, новые сверху; без результатов."""
        if (type(offset) is not int or not 0 <= offset <= MAX_PUBLICATION_OFFSET
                or type(limit) is not int or not 1 <= limit <= 30):
            raise ApiError("Не удалось запросить страницу истории.")
        response = self._json_request("GET", "/analyses/history?" + urlencode({"offset": offset, "limit": limit}),
                                      messages={503: "Не удалось прочитать историю анализов. Повторите попытку."})
        return parse_history(response, offset=offset, limit=limit)

    def saved_analysis(self, run_id: str, language: str | None = None) -> AnalysisStatus:
        """Готовый анализ из истории; текущий анализ сервиса при этом не меняется."""
        if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
            raise ApiError("Ссылка на анализ повреждена. Откройте его заново из истории.")
        cached = self._saved_cache.get(run_id)
        headers = {"X-Trend-Result-Version": cached[0]} if cached else {}
        if language is not None and LANGUAGE.fullmatch(language):
            headers["X-Trend-Language"] = language
        response = self._json_request(
            "GET", "/analyses/saved?" + urlencode({"run_id": run_id}),
            extra_headers=headers or None, messages={
                404: "Анализ не найден в истории.",
                409: "У этого анализа нет готового результата.",
                503: "Не удалось открыть сохранённый анализ. Повторите попытку.",
            })
        version = response.get("result_version") if isinstance(response, dict) else None
        if version is not None and (not isinstance(version, str) or RESULT_VERSION.fullmatch(version) is None):
            raise ApiError("Сервис вернул некорректную версию результата.")
        if not isinstance(response, dict) or response.get("id") != run_id or response.get("state") != "succeeded":
            raise ApiError("Сервис вернул другой анализ. Откройте его заново из истории.")
        status = parse_status(response, cached_result=cached[1] if cached and version == cached[0] else None)
        if version is not None and status.result is not None:
            # Несколько последних: страницы сохранённых анализов опрашиваются по очереди.
            self._saved_cache.pop(run_id, None)
            self._saved_cache[run_id] = (version, status.result)
            while len(self._saved_cache) > 4:
                del self._saved_cache[next(iter(self._saved_cache))]
        return status

    def mode_estimates(self) -> dict[str, ModeEstimate]:
        response = self._json_request("GET", "/analyses/estimates")
        try:
            return parse_estimates(response)
        except (ValueError, UnicodeError):
            raise ApiError("Не удалось прочитать прогноз времени.") from None

    def start_analysis(self, query: str, mode: str = "fast") -> AnalysisStatus:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_LENGTH:
            raise ApiError("Введите запрос длиной от 1 до 2000 символов.")
        if mode not in ANALYSIS_MODES:
            raise ApiError("Выберите доступный режим анализа.")
        return self._status_request("POST", "/analyses", {"query": query, "mode": mode})

    def cancel_analysis(self, run_id: str) -> AnalysisStatus:
        if not isinstance(run_id, str) or not run_id:
            raise ApiError("Не удалось определить анализ для отмены. Обновите страницу.")
        return self._status_request("POST", "/analyses/cancel", {"id": run_id})

    def _owner_request(self, path: str, payload: dict) -> dict:
        """Короткий запрос для панели владельца; сбой — пустой ответ, странице он не мешает."""
        try:
            with requests.Session() as http:
                http.trust_env = False
                with http.post(self.base_url + path, json=payload, timeout=(1, 2), headers=self._headers,
                               allow_redirects=False) as response:
                    answer = response.json() if response.status_code == 200 and len(response.content) < 20_000 else {}
        except (requests.RequestException, ValueError):
            return {}
        return answer if isinstance(answer, dict) else {}

    def check_access(self, ip: str | None) -> str | None:
        """`blocked`, `signed_out` или ничего — до того, как страница что-то покажет."""
        access = self._owner_request("/access", {"ip": ip[:64] if ip else None}).get("access")
        return access if access in {"blocked", "signed_out"} else None

    def report_presence(self, session: str, page: str, *, event: str | None = None, agent: str | None = None,
                        ip: str | None = None, view: str | None = None, detail: str | None = None) -> dict:
        """Отметить вкладку для панели владельца. В ответе — закрыт ли вход и
        сообщения владельца этому посетителю."""
        answer = self._owner_request("/presence", {
            "session": session, "page": page, "event": event, "agent": agent[:400] if agent else None,
            "ip": ip[:64] if ip else None, "view": view, "detail": detail[:160] if detail else None})
        access = answer.get("access")
        messages = answer.get("messages")
        return {"access": access if access in {"blocked", "signed_out"} else None,
                "messages": [text[:500] for text in messages if isinstance(text, str) and text.strip()][:5]
                if isinstance(messages, list) else []}

    def submit(self, query: str, mode: str = "fast") -> Future:
        if not query.strip() or len(query) > MAX_QUERY_LENGTH:
            raise ApiError("Введите запрос длиной от 1 до 2000 символов.")
        if mode not in ANALYSIS_MODES:
            raise ApiError("Выберите доступный режим анализа.")
        if not self._available.acquire(blocking=False):
            raise ApiError("Сервис уже обрабатывает запрос из другого окна. Дождитесь его завершения.")
        try:
            future = self._executor.submit(self.analyze, query.strip(), mode)
        except Exception:
            self._available.release()
            raise
        future.add_done_callback(lambda _: self._available.release())
        return future

    def analyze(self, query: str, mode: str = "fast") -> AnalysisResult:
        if mode not in ANALYSIS_MODES:
            raise ApiError("Выберите доступный режим анализа.")
        try:
            # Сессия принадлежит одному запросу; POST не повторяется автоматически.
            with requests.Session() as session:
                session.trust_env = False
                with session.post(self.endpoint, json={"query": query, "mode": mode}, timeout=(5, 1800),
                                  headers=self._headers, allow_redirects=False, stream=True) as response:
                    if response.status_code != 200:
                        messages = {
                            422: "Сервис не принял запрос. Уточните технологическое направление.",
                            429: "Сервис занят. Попробуйте после завершения текущего анализа.",
                            503: "Сервис анализа ещё не готов. Попробуйте немного позже.",
                        }
                        raise ApiError(messages.get(response.status_code,
                            "Не удалось выполнить анализ. Проверьте работу сервиса и повторите запрос."))
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise ApiError("Ответ слишком большой. Попробуйте сузить тему запроса.")
                        chunks.append(chunk)
                    import json

                    return parse_result(json.loads(b"".join(chunks)))
        except requests.Timeout:
            raise ApiError("Ответ не получен за 30 минут. Анализ на сервере мог продолжиться; "
                           "не отправляйте запрос повторно сразу.") from None
        except requests.RequestException:
            raise ApiError("Нет связи с сервисом анализа. Убедитесь, что API запущен и API_URL указан верно.") from None
        except (ValueError, UnicodeError):
            raise ApiError("Не удалось прочитать ответ сервиса. Проверьте работу API.") from None
