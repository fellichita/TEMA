"""CSV Wordstat dynamics and top phrases, with explicit units and provenance."""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Event
from typing import Literal

from app.pilot.contracts import content_hash
from app.pilot.multisource.contracts import (CoverageState, ExportRight, QueryProfile, QueryTerm, SearchNormalization, SearchObservation,
                                             SourceSnapshot, WordstatImportReceipt, validate_import_export_right)
from app.pilot.multisource.imports import CsvDelimiter, CsvDocument, CsvEncoding, read_csv_document
from app.pilot.multisource.queries import normalize_term
from app.pilot.multisource.store import SignalStore, object_digest
from app.runtime.jobs import TaskFailure

DateFormat = Literal["YYYY-MM", "MM.YYYY", "YYYY-MM-DD", "DD.MM.YYYY"]
ShareUnit = Literal["fraction", "percent", "unknown"]
_COUNT = re.compile(r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:[ \u00a0\u202f]\d{3})+)")
_SHARE = re.compile(r"\d+(?:[,.]\d+)?")


@dataclass(frozen=True)
class DynamicsMapping:
    date_column: str
    count_column: str
    share_column: str | None
    date_format: DateFormat
    share_unit: ShareUnit
    phrase: str
    region_ids: tuple[int, ...] = ()
    devices: tuple[str, ...] = ()
    matching_mode: str = "wordstat-csv-monthly-v1"
    expected_from: date | None = None
    expected_to: date | None = None


@dataclass(frozen=True)
class TopMapping:
    phrase_column: str
    count_column: str


@dataclass(frozen=True)
class RejectedRow:
    line: int
    reason: str


@dataclass(frozen=True)
class DynamicsParse:
    observations: tuple[SearchObservation, ...]
    rejected: tuple[RejectedRow, ...]


@dataclass(frozen=True)
class TopParse:
    terms: tuple[QueryTerm, ...]
    counts: tuple[int, ...]
    rejected: tuple[RejectedRow, ...]


def _count(value: str) -> int | None:
    if not value:
        return None
    if not _COUNT.fullmatch(value):
        raise ValueError("invalid_count")
    return int(value.replace(" ", "").replace("\u00a0", "").replace("\u202f", ""))


def _share(value: str, unit: ShareUnit) -> tuple[str | None, str | None, int | None]:
    if not value:
        return None, None, None
    text = value.strip()
    has_percent = text.endswith("%")
    if has_percent:
        text = text[:-1].strip()
        if unit != "percent":
            raise ValueError("share_unit_conflict")
    text = text.replace(",", ".")
    if not _SHARE.fullmatch(text):
        raise ValueError("invalid_share")
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        raise ValueError("invalid_share") from None
    if not numeric.is_finite() or numeric < 0 or unit == "fraction" and numeric > 1 or unit == "percent" and numeric > 100:
        raise ValueError("invalid_share_range")
    precision = len(text.partition(".")[2])
    raw = format(numeric, "f")
    if unit == "unknown":
        return raw, None, precision
    fraction = numeric / (100 if unit == "percent" else 1)
    return raw, format(fraction, "f"), precision


def _month(value: str, fmt: DateFormat) -> date:
    if fmt == "YYYY-MM" and re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (int(item) for item in value.split("-"))
        return date(year, month, 1)
    if fmt == "MM.YYYY" and re.fullmatch(r"\d{2}\.\d{4}", value):
        month, year = (int(item) for item in value.split("."))
        return date(year, month, 1)
    if fmt == "YYYY-MM-DD" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        result = date.fromisoformat(value)
        if result.day == 1:
            return result
    if fmt == "DD.MM.YYYY" and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value):
        day, month, year = (int(item) for item in value.split("."))
        if day == 1:
            return date(year, month, day)
    raise ValueError("invalid_month")


def parse_dynamics(document: CsvDocument, mapping: DynamicsMapping, profile: QueryProfile, *,
                   snapshot_hash: str, observed_at: datetime, available_at: datetime) -> DynamicsParse:
    if mapping.date_format not in {"YYYY-MM", "MM.YYYY", "YYYY-MM-DD", "DD.MM.YYYY"} or mapping.share_unit not in {
            "fraction", "percent", "unknown"}:
        raise TaskFailure("Параметры даты или доли Wordstat не подтверждены.")
    phrase = normalize_term(mapping.phrase)
    if any(character in '!+-"[]()|' for character in phrase):
        raise TaskFailure("Для месячной истории выберите фразу без операторов Wordstat.")
    if not any(term.status == "confirmed" and term.text.casefold() == phrase.casefold()
               for term in profile.terms):
        raise TaskFailure("Фраза файла должна соответствовать подтверждённому термину профиля.")
    if len(mapping.region_ids) != len(set(mapping.region_ids)) or len(mapping.devices) != len(set(mapping.devices)):
        raise TaskFailure("Повторяющиеся регионы или типы устройств.")
    date_idx, count_idx = document.column(mapping.date_column), document.column(mapping.count_column)
    if len(document.rows) > 120:
        raise TaskFailure("Месячная выгрузка превышает десять лет; выберите меньший период.")
    share_idx = document.column(mapping.share_column) if mapping.share_column is not None else None
    if len({date_idx, count_idx, share_idx} - {None}) != 2 + (share_idx is not None):
        raise TaskFailure("Дата, число запросов и доля должны быть в разных колонках.")
    if share_idx is None and mapping.share_unit != "unknown":
        raise TaskFailure("Единицу доли нельзя подтвердить без колонки доли.")
    series_id = content_hash({"source": "wordstat", "phrase": phrase, "matching_mode": mapping.matching_mode,
                              "regions": mapping.region_ids, "devices": mapping.devices,
                              "granularity": "month", "semantics": 1})
    profile_hash = object_digest(profile)
    observations = []
    rejected = []
    months: set[date] = set()
    for row in document.rows:
        try:
            month = _month(row.cells[date_idx], mapping.date_format)
            if month in months:
                raise TaskFailure(f"Месяц {month.isoformat()} повторяется в CSV; выберите непротиворечивую выгрузку.")
            months.add(month)
            if month > observed_at.date():
                raise ValueError("future_month")
            count = _count(row.cells[count_idx])
            raw, fraction, precision = _share(row.cells[share_idx] if share_idx is not None else "", mapping.share_unit)
            if count is None and raw is not None:
                raise ValueError("share_without_count")
            if count is not None and fraction is not None and count == 0 and Decimal(fraction) > 0:
                raise ValueError("count_share_conflict")
            end = date(month.year, month.month, calendar.monthrange(month.year, month.month)[1])
            complete = end < observed_at.date()
            normalized: SearchNormalization = ("quantized_zero" if count is not None and count > 0
                                               and fraction is not None and Decimal(fraction) == 0
                                               else "usable" if count is not None and fraction is not None
                                               else "unknown_unit")
            observations.append(SearchObservation(
                series_id=series_id, query_profile_hash=profile_hash, snapshot_hash=snapshot_hash,
                phrase=phrase, phrase_role="technology", matching_mode=mapping.matching_mode,
                region_ids=mapping.region_ids, devices=mapping.devices, period_start=month,
                period_end=end if complete else min(end, observed_at.date()), is_complete_period=complete,
                count=count, value_status="observed" if count is not None else "missing",
                share_raw=raw, share_unit=mapping.share_unit, share_fraction=None if normalized == "quantized_zero" else fraction,
                share_precision=precision, normalization_status=normalized,
                observed_at=observed_at, available_at=available_at, row_locator=f"line:{row.line}"))
        except TaskFailure:
            raise
        except (ValueError, OverflowError) as error:
            rejected.append(RejectedRow(row.line, str(error)[:80]))
    return DynamicsParse(tuple(observations), tuple(rejected))


def parse_top(document: CsvDocument, mapping: TopMapping, *, snapshot_hash: str,
              limit: int = 20) -> TopParse:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise TaskFailure("Некорректный лимит предложений Wordstat.")
    phrase_idx, count_idx = document.column(mapping.phrase_column), document.column(mapping.count_column)
    if phrase_idx == count_idx:
        raise TaskFailure("Фраза и частота должны быть в разных колонках.")
    candidates: list[tuple[int, str]] = []
    rejected = []
    seen = set()
    for row in document.rows:
        try:
            phrase = normalize_term(row.cells[phrase_idx])
            count = _count(row.cells[count_idx])
            if count is None:
                raise ValueError("missing_count")
            if phrase.casefold() in seen:
                raise ValueError("duplicate_phrase")
            seen.add(phrase.casefold())
            candidates.append((count, phrase))
        except (ValueError, OverflowError) as error:
            rejected.append(RejectedRow(row.line, str(error)[:80]))
    candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
    terms = [QueryTerm(text=phrase, language="ru" if re.search(r"[А-Яа-яЁё]", phrase) else "en",
                       role="technology", origin="wordstat", status="proposed",
                       source_artifact_hash=snapshot_hash) for _, phrase in candidates[:limit]]
    return TopParse(tuple(terms), tuple(count for count, _ in candidates[:limit]), tuple(rejected))


def _expected_months(mapping: DynamicsMapping) -> set[date] | None:
    if mapping.expected_from is None and mapping.expected_to is None:
        return None
    if (mapping.expected_from is None or mapping.expected_to is None or mapping.expected_from.day != 1
            or mapping.expected_to.day != 1 or mapping.expected_to < mapping.expected_from):
        raise TaskFailure("Укажите первый и последний ожидаемые месяцы вместе.")
    months = set()
    year, month = mapping.expected_from.year, mapping.expected_from.month
    while (year, month) <= (mapping.expected_to.year, mapping.expected_to.month):
        months.add(date(year, month, 1))
        if len(months) > 120:
            raise TaskFailure("Одна месячная выгрузка не должна превышать десять лет.")
        year, month = year + (month == 12), month % 12 + 1
    return months


def _mapping_payload(mapping: DynamicsMapping | TopMapping) -> dict:
    result = asdict(mapping)
    if isinstance(mapping, DynamicsMapping):
        result["expected_from"] = mapping.expected_from.isoformat() if mapping.expected_from else None
        result["expected_to"] = mapping.expected_to.isoformat() if mapping.expected_to else None
    return result


def import_wordstat_csv(store: SignalStore, path: Path, profile_hash: str, *,
                        kind: Literal["dynamics", "top"], mapping: DynamicsMapping | TopMapping,
                        encoding: CsvEncoding, delimiter: CsvDelimiter, retention: str,
                        observed_at: datetime | None = None, cancel: Event | None = None,
                        export_right: ExportRight = "local_only", license_ref: str | None = None) -> str:
    """Import owned/retained CSV into CAS; return receipt, never a published finding."""
    if (kind == "dynamics" and not isinstance(mapping, DynamicsMapping)) or (
            kind == "top" and not isinstance(mapping, TopMapping)):
        raise TaskFailure("Выбранная карта колонок не подходит к типу отчёта Wordstat.")
    validate_import_export_right(export_right, license_ref)
    profile = store.get_object(profile_hash, QueryProfile)
    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise TaskFailure("Дата импорта должна включать часовой пояс.")
    raw_hash = store.put_raw(path, "csv", retention=retention, cancel=cancel)
    document = read_csv_document(store.verify_raw(raw_hash, "csv", cancel=cancel), encoding=encoding,
                                 delimiter=delimiter, cancel=cancel)
    dummy_hash = "0" * 64
    expected = None
    if isinstance(mapping, DynamicsMapping):
        expected = _expected_months(mapping)
        provisional = parse_dynamics(document, mapping, profile, snapshot_hash=dummy_hash,
                                     observed_at=timestamp, available_at=timestamp)
        actual = {item.period_start for item in provisional.observations}
        coverage: CoverageState = ("unknown" if expected is None else "complete" if not provisional.rejected
                                   and actual == expected and all(item.is_complete_period for item in provisional.observations)
                                   else "partial")
        comparable = coverage == "complete" and mapping.share_unit != "unknown"
    else:
        provisional_top = parse_top(document, mapping, snapshot_hash=dummy_hash)
        coverage = "complete" if not provisional_top.rejected else "partial"
        comparable = False
    mapping_hash = content_hash({"mapping": _mapping_payload(mapping), "encoding": encoding, "delimiter": delimiter})
    request_hash = content_hash({"source": "wordstat-csv", "kind": kind, "mapping_hash": mapping_hash,
                                 "query_profile_hash": profile_hash})
    snapshot = SourceSnapshot(source="wordstat", adapter_version="wordstat-csv/1", request_hash=request_hash,
                              query_profile_hash=profile_hash, observed_at=timestamp, available_at=timestamp,
                              coverage=coverage, comparable=comparable, raw_hash=raw_hash,
                              retention="local_allowed", export_right=export_right, license_ref=license_ref,
                              limitations=(("Ожидаемый диапазон месяцев не задан.",) if coverage == "unknown" else
                                           ("Часть строк или месяцев отсутствует/отклонена.",) if coverage == "partial" else ()))
    snapshot_hash = store.put_object(snapshot, cancel=cancel)
    if isinstance(mapping, DynamicsMapping):
        parsed = parse_dynamics(document, mapping, profile, snapshot_hash=snapshot_hash,
                                observed_at=timestamp, available_at=timestamp)
        if (len(parsed.observations), len(parsed.rejected)) != (len(provisional.observations), len(provisional.rejected)):
            raise TaskFailure("Данные импорта изменились во время проверки.")
        observations = tuple(store.put_object(item, cancel=cancel) for item in parsed.observations)
        terms: tuple[str, ...] = ()
        top_counts: tuple[int, ...] = ()
        rejected = parsed.rejected
    else:
        parsed_top = parse_top(document, mapping, snapshot_hash=snapshot_hash)
        if (len(parsed_top.terms), len(parsed_top.rejected)) != (len(provisional_top.terms), len(provisional_top.rejected)):
            raise TaskFailure("Данные импорта изменились во время проверки.")
        observations = ()
        terms = tuple(store.put_object(item, cancel=cancel) for item in parsed_top.terms)
        top_counts = parsed_top.counts
        rejected = parsed_top.rejected
    store.verify_raw(raw_hash, "csv", cancel=cancel)
    accepted = len(observations) + len(terms)
    receipt = WordstatImportReceipt(kind=kind, query_profile_hash=profile_hash, snapshot_hash=snapshot_hash,
                                    raw_hash=raw_hash, mapping_hash=mapping_hash, row_count=len(document.rows),
                                    accepted_count=accepted, rejected_count=len(rejected),
                                    unselected_count=len(document.rows) - accepted - len(rejected),
                                    observation_hashes=observations, term_hashes=terms, top_counts=top_counts,
                                    rejected_rows=tuple(f"line:{item.line}:{item.reason}" for item in rejected),
                                    completed_at=timestamp)
    return store.put_object(receipt, cancel=cancel)
