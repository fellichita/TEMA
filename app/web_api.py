"""Loopback-only HTTP bridge from the web demo to the desktop pilot service."""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
import hmac
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import statistics
from threading import BoundedSemaphore, Event, Lock, Thread
import time
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit

from app.backend.contracts import normalize_doi
from app.identity import default_data_dir
from app.input_safety import is_safe_http_url, url_key, work_key
from app.pilot.archive import DocumentArchive
from app.pilot.selection import SIGNAL_CATEGORIES
from app.pilot.settings import COLLECTION_PROFILES, CollectionProfile
from app.runtime.jobs import TaskCancelled, TaskFailure
from app.web_admin import OwnerTools
from app.web_monitor import MAX_MESSAGE_CHARACTERS, Blocklist, Monitor, SilentMonitor, stage_label

MAX_REQUEST_BYTES = 16_000
MAX_QUERY_CHARACTERS = 2_000
MIN_API_TOKEN_CHARACTERS = 32
MAX_API_TOKEN_CHARACTERS = 256
MAX_API_CONNECTIONS = 16
API_SOCKET_TIMEOUT_SECONDS = 10
HEALTH_CACHE_SECONDS = 1.0
MAX_PUBLICATION_PAGE = 100
MAX_PUBLICATION_OFFSET = 1_000_000
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
ANALYSIS_MODES = frozenset({"fast", "deep"})
PROGRESS_STAGE = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
RADAR_POLICY_VERSION = re.compile(r"radar/[0-9]{1,6}\.[0-9]{1,6}\.[0-9]{1,6}\Z")
# Ориентир, пока на этом компьютере нет собственных прогонов режима. Замер
# 24.09.2026 после ускорения, локальная модель на RTX 3070, «solid-state
# batteries»: быстрый — 2,2 мин, глубокий — 14,3 мин.
# OpenAlex в тот день отказывал по дневному лимиту; с ключом история и ранние
# аналоги собираются по-настоящему, поэтому ориентир взят с запасом.
# Первый завершённый анализ режима заменяет ориентир своей длительностью.
DEFAULT_MODE_SECONDS = {"fast": 4 * 60, "deep": 16 * 60}
ESTIMATE_SAMPLES = 7
ESTIMATE_HISTORY_PAGES = 4
MAX_RUN_SECONDS = 7 * 24 * 3600
RUN_STATES = TERMINAL_STATES | {"queued", "running"}
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
# Посетитель публичной ссылки: случайный идентификатор, подтверждённый паролем в веб-интерфейсе.
VISITOR = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
# Страница истории: +1 строка в запросе к базе показывает, есть ли следующая.
MAX_HISTORY_PAGE = 30
# Перевод и ТОП технологий ведутся отдельно для каждого показанного анализа;
# в памяти — несколько последних, и переход по истории их не пересчитывает.
FINISHED_JOB_CACHE = 8
# Незаконченное задание анализа, который никто не опрашивает дольше этого,
# отменяется, когда начинается новое: сеть и переводчик нужнее открытому.
JOB_IDLE_SECONDS = 15.0
# Собранные результаты (с полным списком публикаций) нескольких анализов.
RENDERED_RUNS = 4
# Панель владельца: как часто сверяются этапы идущего анализа, сколько
# последних анализов в сводке и как долго переиспользуется состояние модели.
WATCH_SECONDS = 1.0
ADMIN_RUNS = 12
ADMIN_STATUS_SECONDS = 10.0
# Очередь анализов: сколько ждут одновременно и как часто диспетчер смотрит,
# не освободилось ли место. Ждущий анализ имеет свой номер вместо run_id.
DISPATCH_SECONDS = 1.0
TICKET_PREFIX = "wait-"
# Ресурсы сервиса, которыми управляет владелец из панели; файл переживает
# перезапуски. Значения по умолчанию — для RTX 3070 (8 ГБ) и одного процесса API.
RESOURCE_DEFAULTS: dict[str, Any] = {"slots": 2, "llm_batch": 8, "keep_llm_loaded": True,
                                     "deep_allowed": True, "queue_limit": 10, "radar_exact": False}
RESOURCE_LIMITS = {"slots": (1, 4), "llm_batch": (1, 8), "queue_limit": (0, 30)}


def valid_resources(values: object) -> dict[str, Any]:
    """Только известные настройки ресурсов с допустимыми значениями."""
    if not isinstance(values, dict):
        raise ValueError("resources")
    clean: dict[str, Any] = {}
    for name, value in values.items():
        if name in RESOURCE_LIMITS:
            low, high = RESOURCE_LIMITS[name]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(name)
        elif name in {"keep_llm_loaded", "deep_allowed", "radar_exact"}:
            if type(value) is not bool:
                raise ValueError(name)
        else:
            raise ValueError(name)
        clean[name] = value
    return clean


class WebApiError(RuntimeError):
    """A bounded, user-safe API failure."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous request objects before applying API field validation."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _summary(card: dict[str, Any]) -> str:
    claims = card.get("claims")
    if isinstance(claims, list):
        for support in ("supported", "unverified"):
            for role in ("summary", "advantage", "case", "problem"):
                for claim in claims:
                    if (isinstance(claim, dict) and claim.get("role") == role
                            and claim.get("support") == support):
                        text = claim.get("text")
                        if isinstance(text, str) and text.strip():
                            return text.strip()[:10_000]
    candidate = card.get("candidate")
    definition = candidate.get("definition") if isinstance(candidate, dict) else None
    if isinstance(definition, str) and definition.strip():
        return definition.strip()[:10_000]
    raise WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                      "Сохранённый результат не содержит описания сигнала.")


def _publication_aliases(url: str, doi: str | None = None) -> set[str]:
    """Recognize the same publication across the scientific and approved feeds."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    # URL paths may be case-sensitive; only the host and DOI are case-folded.
    aliases = {f"url:{host}{path}?{parsed.query}"}
    if doi:
        aliases.add("doi:" + doi.casefold())
    if host in {"doi.org", "dx.doi.org"}:
        try:
            aliases.add("doi:" + normalize_doi(url))
        except ValueError:
            pass
    return aliases


def _publication_date(item: dict[str, Any]) -> date | None:
    value = item.get("published_at")
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    month = item.get("publication_month")
    if isinstance(month, str):
        try:
            return date.fromisoformat(month + "-01")
        except ValueError:
            pass
    year = item.get("publication_year")
    return date(year, 1, 1) if type(year) is int and 1 <= year <= 9999 else None


def _evidence_as_of(result: dict[str, Any], source_snapshot: Any) -> date | None:
    """Use the run's own cutoff, not the day an archived result is reopened."""
    plan = result.get("query_plan")
    value = plan.get("as_of") if isinstance(plan, dict) else None
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    if source_snapshot is not None:
        return source_snapshot.as_of
    snapshots = result.get("snapshots")
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or snapshot.get("purpose") != "discovery":
                continue
            value = snapshot.get("as_of")
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value)
                except ValueError:
                    continue
    return None


def _publication_pool(result: dict[str, Any], source_snapshot: Any, archive: DocumentArchive | None,
                      trends: dict[str, dict[str, Any]] | None = None, *,
                      keep_studies: bool = False) -> list[dict[str, Any]]:
    """Build the full deduplicated pool before the browser's display limit.

    `trends` maps a discovery study id to the assessed candidate that
    contains it. This metadata is displayed separately and never changes
    which source may enter the preliminary TOP. `keep_studies` adds the
    discovery study ids of each publication for the relevance assessment.
    """
    trends = trends or {}
    entries: list[tuple[dict[str, Any], set[str], set[str]]] = []
    snapshots = result.get("snapshots")
    if archive is not None and isinstance(snapshots, list):
        # Discovery is the search result. History and enrichment are supporting
        # material for candidate assessment, not publications found by this search.
        references: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or snapshot.get("purpose") != "discovery":
                continue
            documents = snapshot.get("documents")
            if not isinstance(documents, list):
                continue
            for reference in documents:
                if not isinstance(reference, dict):
                    continue
                study_id, revision_id = reference.get("study_id"), reference.get("revision_id")
                if isinstance(study_id, str) and isinstance(revision_id, str):
                    # Several scientific providers can archive revisions of
                    # the same study. Preserve each provider for provenance;
                    # DOI/URL grouping below still counts the study once.
                    references.setdefault(revision_id, reference)
        for reference in references.values():
            try:
                document = archive.get(reference["revision_id"])
            except TaskFailure:
                raise WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                                  "Не удалось прочитать сохранённую публикацию.") from None
            if not is_safe_http_url(document.url):
                continue
            item = {"title": document.title[:1000],
                    "summary": document.abstract[:1000] if document.abstract else None,
                    "url": document.url, "source_id": document.source,
                    "kind": document.document_type, "published_at":
                    document.publication_date.isoformat() if document.publication_date else None,
                    "publication_year": document.publication_year,
                    "publication_month": (
                        f"{document.publication_year:04d}-{document.publication_month:02d}"
                        if document.publication_year and document.publication_month else
                        document.publication_date.strftime("%Y-%m") if document.publication_date else None),
                    "date_precision": document.date_precision,
                    "date_basis": "published" if document.publication_year else "unknown"}
            aliases = _publication_aliases(document.url, document.doi)
            study_id = reference["study_id"]
            if study_id.startswith("doi:"):
                aliases.add(study_id.casefold())
            entries.append((item, aliases, {study_id}))
    if source_snapshot is not None:
        for observation in source_snapshot.observations:
            if not is_safe_http_url(observation.url):
                continue
            published_at = observation.published_at.isoformat()
            item = {"title": observation.title, "summary": observation.summary,
                    "url": observation.url, "source_id": observation.source_id,
                    "kind": observation.kind, "published_at": published_at,
                    "publication_year": observation.published_at.year,
                    "publication_month": (published_at[:7]
                                          if observation.date_basis == "published" else None),
                    "date_precision": "day" if observation.date_basis == "published" else "unknown",
                    "date_basis": observation.date_basis}
            if observation.date_basis == "year":
                # Источник знает только год: это не дата дня и не месяц.
                item.update(published_at=None, publication_month=None, date_precision="year",
                            date_basis="published")
            if observation.country is not None:
                item["country"] = observation.country
            doi = (observation.item_id if observation.source_id in {"biorxiv", "europe_pmc"}
                   and observation.item_id.startswith("10.") else None)
            entries.append((item, _publication_aliases(observation.url, doi), set()))

    # Union aliases before taking the TOP: a DOI and a landing-page URL can
    # connect records from different feeds even when their titles differ.
    parents = list(range(len(entries)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    by_alias: dict[str, int] = {}
    for index, (_, aliases, _) in enumerate(entries):
        for alias in aliases:
            previous = by_alias.setdefault(alias, index)
            current_root, previous_root = root(index), root(previous)
            if current_root != previous_root:
                parents[current_root] = previous_root
    grouped: dict[int, list[tuple[dict[str, Any], set[str], set[str]]]] = {}
    for index, entry in enumerate(entries):
        grouped.setdefault(root(index), []).append(entry)
    publications = []
    for group in grouped.values():
        # Prefer the fuller record within a duplicate group; this preference
        # depends on metadata, never on the identity of its source.
        chosen = max((item for item, _, _ in group),
                     key=lambda item: (bool(item["summary"]),
                                       len(item["summary"] or ""),
                                       bool(item["published_at"]),
                                       item["url"], item["title"]))
        # The fuller text can come from an index or news feed. Publication
        # timing instead comes from the most precise original-date record.
        dated = [item for item, _, _ in group if item["date_basis"] == "published"]
        if dated:
            timing = max(dated, key=lambda item: ({"day": 3, "month": 2, "year": 1,
                                                 "unknown": 0}[item["date_precision"]],
                                                  item["published_at"] or "", item["url"]))
            chosen = {**chosen, **{key: timing[key] for key in
                                   ("published_at", "publication_year", "date_basis", "date_precision")}}
        months = {item["publication_month"] for item in dated if item["publication_month"] is not None}
        # Contradictory dates cannot be assigned to a monthly bucket.
        chosen["publication_month"] = next(iter(months)) if len(months) == 1 else None
        if len(months) > 1:
            years = {item["publication_year"] for item in dated if item["publication_year"] is not None}
            chosen.update(published_at=None, publication_year=next(iter(years)) if len(years) == 1 else None,
                          date_precision="unknown", date_basis="published")
        aliases = set().union(*(item_aliases for _, item_aliases, _ in group))
        identity = min((alias for alias in aliases if alias.startswith("doi:")), default=None)
        if identity is None:
            identity = min(aliases)
        publication = {"publication_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(), **chosen,
                       "source_ids": sorted({item["source_id"] for item, _, _ in group})}
        studies = sorted(set().union(*(item_studies for _, _, item_studies in group)))
        trend = max((trends[study] for study in studies if study in trends), key=_trend_rank, default=None)
        if trend is not None:
            publication["trend"] = trend
        if keep_studies:
            publication["study_ids"] = studies
        publications.append(publication)
    publications.sort(key=lambda item: (-(_publication_date(item) or date.min).toordinal(),
                                        item["url"], item["title"]))
    return publications


# В ТОП-15 не больше стольких публикаций одного источника, если есть другие по теме.
TOP_SOURCE_CAP = 6
TOP_SIZE = 15
RELEVANCE_ORDER = {"relevant": 0, "weak": 1}


def relevance_ranking(publications: list[dict[str, Any]], relevance: dict[str, tuple[float, str, float, float | None]],
                      as_of: date | None) -> tuple[list[dict[str, Any]], int]:
    """Публикации по теме: сначала ТОП-15 по релевантности, затем остальные по дате.

    Внутри ТОПа порядок задаёт оценка темы с небольшой поправкой на свежесть
    (до +0,1 за материал последнего месяца, ноль — старше трёх лет); один
    источник занимает не больше шести мест, пока есть другие материалы по теме.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for publication in publications:
        entry = relevance.get(publication["publication_id"])
        if entry is not None and entry[1] == "off_topic":
            removed += 1
            continue
        if entry is not None:
            publication["relevance"] = {"score": round(entry[0], 3), "decision": entry[1]}
        kept.append(publication)

    def freshness(publication: dict[str, Any]) -> float:
        moment = _publication_date(publication)
        if moment is None or as_of is None:
            return 0.0
        return max(0.0, 1 - max(0, (as_of - moment).days) / 1095)

    def rank(publication: dict[str, Any]) -> tuple[int, float, int, str]:
        entry = relevance.get(publication["publication_id"])
        order = RELEVANCE_ORDER.get(entry[1], 1) if entry is not None else 1
        score = (entry[0] if entry is not None else 0.0) + 0.1 * freshness(publication)
        return (order, -score, -(_publication_date(publication) or date.min).toordinal(), publication["url"])

    ranked = sorted(kept, key=rank)
    top: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    deferred: list[dict[str, Any]] = []
    for publication in ranked:
        if len(top) == TOP_SIZE:
            break
        source = publication["source_id"]
        if per_source.get(source, 0) >= TOP_SOURCE_CAP:
            deferred.append(publication)
            continue
        per_source[source] = per_source.get(source, 0) + 1
        top.append(publication)
    # Мало других источников: свободные места ТОПа занимают отложенные.
    top.extend(deferred[:TOP_SIZE - len(top)])
    chosen = {id(publication) for publication in top}
    return top + [publication for publication in kept if id(publication) not in chosen], removed


def _publications(result: dict[str, Any], source_snapshot: Any, archive: DocumentArchive | None,
                  trends: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], int]:
    """Rank one deduplicated pool by available date, without a source or trend bonus."""
    publications = _publication_pool(result, source_snapshot, archive, trends)
    return publications[:200], len(publications)


SKIPPED_BY_POLICY = frozenset({"country_filter", "disabled_by_owner"})


def _source_coverage(result: dict[str, Any], source_snapshot: Any) -> list[dict[str, Any]]:
    """Expose every searched source under one accounting contract."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    snapshots = result.get("snapshots")
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or snapshot.get("purpose") != "discovery":
                continue
            for coverage in snapshot.get("coverage", []):
                if not isinstance(coverage, dict):
                    continue
                source_id = coverage.get("source")
                if not isinstance(source_id, str):
                    continue
                reasons = coverage.get("reasons")
                reason_code = (reasons[0] if isinstance(reasons, list) and reasons
                               and isinstance(reasons[0], str) and PROGRESS_STAGE.fullmatch(reasons[0])
                               else None)
                grouped.setdefault(source_id, []).append({
                    "state": coverage.get("state"), "scanned": coverage.get("scanned_records", 0),
                    "accepted": coverage.get("accepted_records", 0),
                    "limit_reached": coverage.get("limit_reached", False),
                    "reason_code": reason_code})
    if source_snapshot is not None:
        for coverage in source_snapshot.coverage:
            # Источник, выключенный владельцем или фильтром стран, не опрашивался
            # намеренно: это не пробел в охвате, и в список он не входит.
            if coverage.reason_code in SKIPPED_BY_POLICY:
                continue
            grouped.setdefault(coverage.source_id, []).append({
                "state": coverage.state, "scanned": coverage.scanned,
                "accepted": coverage.accepted, "limit_reached": coverage.limit_reached,
                "reason_code": coverage.reason_code if coverage.reason_code is not None
                and PROGRESS_STAGE.fullmatch(coverage.reason_code) else None})
    response = []
    for source_id, rows in grouped.items():
        states = {row["state"] for row in rows}
        state = ("complete" if states == {"complete"} else "unavailable"
                 if states == {"unavailable"} else "partial")
        scanned = sum(row["scanned"] for row in rows
                      if type(row["scanned"]) is int and row["scanned"] >= 0)
        accepted = sum(row["accepted"] for row in rows
                       if type(row["accepted"]) is int and row["accepted"] >= 0)
        reasons = {row["reason_code"] for row in rows if row["reason_code"] is not None}
        reason_code = (None if state == "complete" else next(iter(reasons))
                       if len(reasons) == 1 and len(rows) == 1 else
                       "source_unavailable" if state == "unavailable" else "partial_coverage")
        response.append({"source_id": source_id, "state": state,
                         "scanned": max(scanned, accepted), "accepted": accepted,
                         "limit_reached": any(row["limit_reached"] is True for row in rows),
                         "reason_code": reason_code})
    return response


def _funding_evidence(value: object) -> dict[str, Any] | None:
    """Bound saved NIH money facts before exposing them to the browser."""
    if value is None:
        return None
    error = WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Не удалось прочитать данные финансирования.")
    if not isinstance(value, dict) or value.get("source_id") != "nih_reporter":
        raise error
    if (value.get("date_basis") != "award_notice_date"
            or value.get("amount_basis") != "reported_fiscal_year_award_usd"
            or value.get("coverage_reason") not in {"complete", "page_cap", "invalid_records",
                                                 "unknown_total", "source_unavailable"}):
        raise error
    topic, start, end = (value.get(key) for key in ("topic", "from_date", "to_date"))
    returned, rejected, total = (value.get(key) for key in
                                 ("records_returned", "rejected_records", "total_available"))
    awards = value.get("awards")
    if (not isinstance(topic, str) or not 1 <= len(topic) <= 180
            or not isinstance(start, str) or not isinstance(end, str)
            or type(returned) is not int or not 0 <= returned <= 50
            or type(rejected) is not int or not 0 <= rejected <= returned
            or total is not None and (type(total) is not int or total < returned)
            or not isinstance(awards, list) or len(awards) != returned - rejected
            or type(value.get("partial_coverage")) is not bool
            or value["partial_coverage"] != (value["coverage_reason"] != "complete")):
        raise error
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except (TypeError, ValueError):
        raise error from None
    if first > last:
        raise error
    clean_awards = []
    seen_ids: set[int] = set()
    for award in awards:
        if not isinstance(award, dict):
            raise error
        identity = award.get("application_id")
        title, date_text, amount_text = (award.get(key) for key in
                                         ("title", "award_notice_date", "award_amount_usd"))
        kind = award.get("funding_type")
        project = award.get("project_number")
        mechanism = award.get("funding_mechanism")
        if (type(identity) is not int or identity <= 0 or identity in seen_ids
                or not isinstance(title, str) or not 1 <= len(title.strip()) <= 500
                or not isinstance(amount_text, str) or len(amount_text) > 32
                or kind not in {"grant_or_cooperative", "contract", "intramural", "other_or_unknown"}
                or project is not None and (not isinstance(project, str) or len(project) > 100)
                or mechanism is not None and (not isinstance(mechanism, str) or len(mechanism) > 100)
                or award.get("detail_url") != f"https://reporter.nih.gov/project-details/{identity}"):
            raise error
        if not isinstance(date_text, str):
            raise error
        try:
            award_date = date.fromisoformat(date_text)
            amount = Decimal(amount_text)
        except (TypeError, ValueError, InvalidOperation):
            raise error from None
        if not first <= award_date <= last or not amount.is_finite() or not 0 <= amount <= 1_000_000_000_000:
            raise error
        seen_ids.add(identity)
        clean_awards.append({"application_id": identity, "title": title.strip(),
                             "award_notice_date": award_date.isoformat(),
                             "award_amount_usd": format(amount, "f"), "funding_type": kind,
                             "detail_url": award["detail_url"]})
    return {"source_id": "nih_reporter", "topic": topic,
            "from_date": first.isoformat(), "to_date": last.isoformat(),
            "date_basis": "award_notice_date",
            "amount_basis": "reported_fiscal_year_award_usd",
            "awards": clean_awards, "records_returned": returned,
            "total_available": total, "rejected_records": rejected,
            "partial_coverage": value["partial_coverage"],
            "coverage_reason": value["coverage_reason"]}


TREND_FEATURES = ("growth", "persistence", "novelty", "independence", "application")
CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}


def _trend_rank(trend: dict[str, Any] | None) -> tuple[int, int, int]:
    """Stronger methodology verdict first: level, confirmed growth, measured features."""
    if trend is None:
        return 0, 0, 0
    return (CONFIDENCE_RANK[trend["confidence"]], int(trend["growth_confirmed"]),
            len(trend["checked_features"]))


def trend_confidence(assessment: dict[str, Any] | None) -> dict[str, Any]:
    """Confidence fields of one TOP signal, or nothing for an unassessed result.

    `confidence` is the methodology's category (high only with confirmed growth
    and every feature measured); a feature counts as checked when its score
    component has a value. Neither is a probability that the trend is real.
    """
    if not isinstance(assessment, dict) or assessment.get("confidence") not in {"low", "medium", "high"}:
        return {}
    components = assessment.get("components")
    measured = {component.get("name"): component.get("value") is not None
                for component in components if isinstance(component, dict)} if isinstance(components, list) else {}
    return {"confidence": assessment["confidence"],
            "growth_confirmed": assessment.get("growth_confirmed") is True,
            "checked_features": [name for name in TREND_FEATURES if measured.get(name)],
            "unchecked_features": [name for name in TREND_FEATURES if not measured.get(name)]}


def _source_snapshot(payload: dict[str, Any]) -> Any:
    source_payload = payload.get("approved_sources")
    if source_payload is None:
        return None
    from app.pilot.approved_sources import SourceSnapshot

    try:
        return SourceSnapshot.model_validate(source_payload)
    except ValueError:
        raise WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                          "Не удалось прочитать найденные публикации.") from None


def radar_input(payload: dict[str, Any], *, archive: DocumentArchive | None = None
                ) -> tuple[list[dict[str, Any]], date | None, list[str]]:
    """Выдача запуска по теме, дата среза и английские термины запроса для радара технологий.

    Материалы, признанные «не по теме», в радар не идут: из их заголовков
    вырастали кандидаты в технологии совсем другой области.
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        raise WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Сервис не смог прочитать сохранённый результат.")
    source_snapshot = _source_snapshot(payload)
    query_plan = result.get("query_plan")
    plan = query_plan if isinstance(query_plan, dict) else {}
    synonyms = plan.get("synonyms")
    candidates = [plan.get("english_query")]
    if isinstance(synonyms, (list, tuple)):
        candidates.extend(synonyms)
    terms = [value for value in candidates
             if isinstance(value, str) and value.strip()]
    pool = _publication_pool(result, source_snapshot, archive)
    from app.pilot.publication_relevance import relevance_items

    relevance = relevance_items(payload.get("publication_relevance"))
    if relevance is not None:
        pool = [item for item in pool if relevance.get(item["publication_id"], (0, "weak"))[1] != "off_topic"]
    return (pool, _evidence_as_of(result, source_snapshot), terms)


def web_result(payload: dict[str, Any], *, archive: DocumentArchive | None = None,
               publication_pool_out: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Expose searched publications and the separately assessed signal TOP."""
    result = payload.get("result")
    if not isinstance(result, dict):
        raise WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                          "Сервис не смог прочитать сохранённый результат.")
    cards = result.get("cards")
    top_ids = result.get("top_trend_ids")
    if not isinstance(cards, list) or not isinstance(top_ids, list):
        raise WebApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                          "Сервис вернул результат неизвестной версии.")
    by_id = {}
    for card in cards:
        candidate = card.get("candidate") if isinstance(card, dict) else None
        identifier = candidate.get("candidate_id") if isinstance(candidate, dict) else None
        if isinstance(identifier, str):
            by_id[identifier] = card
    # The methodology's own verdict on each candidate: a categorical confidence
    # and which of the five trend features (growth, persistence, novelty,
    # independence, application) could be measured at all.
    assessments: dict[str, dict[str, Any]] = {}
    raw_assessments = payload.get("assessments")
    for artifact in raw_assessments if isinstance(raw_assessments, list) else ():
        assessment = artifact.get("assessment") if isinstance(artifact, dict) else None
        if isinstance(assessment, dict) and isinstance(assessment.get("candidate_id"), str):
            assessments[assessment["candidate_id"]] = assessment
    # Every found publication inherits the verdict of the assessed candidate it
    # belongs to, TOP signal or not; with several, the strongest verdict wins.
    trends: dict[str, dict[str, Any]] = {}
    for identifier, card in by_id.items():
        confidence = trend_confidence(assessments.get(identifier))
        candidate = card["candidate"]
        label, studies = candidate.get("label"), candidate.get("discovery_study_ids")
        if not confidence or not isinstance(label, str) or not label.strip() or not isinstance(studies, list):
            continue
        trend = {"title": label.strip()[:500], **confidence}
        for study in studies:
            if isinstance(study, str) and _trend_rank(trend) > _trend_rank(trends.get(study)):
                trends[study] = trend
    signals = []
    seen: set[str] = set()
    for identifier in top_ids[:15]:
        if not isinstance(identifier, str) or identifier in seen:
            continue
        seen.add(identifier)
        card = by_id.get(identifier)
        if not isinstance(card, dict) or card.get("category") not in SIGNAL_CATEGORIES:
            continue
        candidate = card.get("candidate")
        title = candidate.get("label") if isinstance(candidate, dict) else None
        if not isinstance(title, str) or not title.strip():
            continue
        urls = []
        seen_urls: set[str] = set()
        for evidence in card.get("evidence", []):
            url = evidence.get("source_url") if isinstance(evidence, dict) else None
            if isinstance(url, str) and url not in seen_urls and is_safe_http_url(url):
                urls.append(url)
                seen_urls.add(url)
                if len(urls) == 100:
                    break
        signal = {"title": title.strip(), "summary": _summary(card),
                  "category": card["category"], "source_urls": urls}
        signal.update(trend_confidence(assessments.get(identifier)))
        signals.append(signal)
    # A source request can finish but still have partial pagination or hit its
    # record limit. The UI must disclose that limitation without interpreting
    # exploratory discovery quality as a coverage measure.
    snapshots = result.get("snapshots")
    if not isinstance(snapshots, list):
        snapshots = []
    incomplete_coverage = any(
        isinstance(snapshot, dict) and snapshot.get("purpose") in {"discovery", "history"}
        and isinstance(snapshot.get("coverage"), list)
        and any(isinstance(coverage, dict) and coverage.get("state") != "complete"
                for coverage in snapshot["coverage"])
        for snapshot in snapshots
    )
    response: dict[str, Any] = {"signals": signals, "incomplete_coverage": incomplete_coverage}
    # Without its own key OpenAlex shares one daily budget of 1000 requests per
    # IP address. Once it is spent every history query is refused, no trend can
    # be confirmed and the TOP stays empty — the user has to know why.
    if any(isinstance(snapshot, dict) and isinstance(snapshot.get("coverage"), list)
           and any(isinstance(coverage, dict) and coverage.get("source") == "openalex"
                   and "source_rate_limited" in (coverage.get("reasons") or ())
                   for coverage in snapshot["coverage"])
           for snapshot in snapshots):
        response["openalex_rate_limited"] = True
    source_snapshot = _source_snapshot(payload)
    # Keep the complete deduplicated pool for similar-material charts. The
    # browser's 200-item limit is only a display limit and must not change them.
    publications = _publication_pool(result, source_snapshot, archive, trends)
    off_topic_total = 0
    from app.pilot.publication_relevance import relevance_items

    relevance = relevance_items(payload.get("publication_relevance"))
    if relevance is not None:
        # Новый анализ оценил тему у каждой публикации: чужие убираются из выдачи,
        # а ТОП берётся по релевантности, а не по свежести.
        publications, off_topic_total = relevance_ranking(publications, relevance,
                                                          _evidence_as_of(result, source_snapshot))
        response["off_topic_total"] = off_topic_total
    publication_total = len(publications)
    # Model relevance belongs to one publication, unlike the methodology's
    # categorical confidence in a candidate trend. The local model assessed
    # only the preliminary TOP; all sources were eligible for those places.
    saved_confidences = payload.get("publication_confidences")
    if isinstance(saved_confidences, dict):
        from app.pilot.publication_confidence import validate_confidence

        for publication in publications[:15]:
            publication_confidence = validate_confidence(
                saved_confidences.get(publication["publication_id"]),
                publication["title"], publication.get("summary"))
            if publication_confidence is not None:
                publication["model_confidence"] = publication_confidence
        # Keep the original evidence/date order for ties and missing scores.
        # An unavailable model never fabricates a numeric rank.
        publications[:15] = sorted(publications[:15],
                                   key=lambda item: -item["model_confidence"]["score"]
                                   if "model_confidence" in item else 1)
    response["publications"] = publications[:15]
    response["publication_total"] = publication_total
    response["top_publications"] = publications[:15]
    response["source_coverage"] = _source_coverage(result, source_snapshot)
    funding_evidence = _funding_evidence(payload.get("funding_sources"))
    if funding_evidence is not None:
        response["funding_evidence"] = funding_evidence
    if source_snapshot is not None:
        arxiv_coverage = next((item for item in source_snapshot.coverage
                               if item.source_id == "arxiv"), None)
        if arxiv_coverage is not None:
            response["arxiv_domains"] = {
                "coverage_state": arxiv_coverage.state,
                "scanned": arxiv_coverage.scanned,
                "months": [item.model_dump(mode="json")
                           for item in source_snapshot.arxiv_domain_months],
            }
    response["incomplete_coverage"] = incomplete_coverage or any(
        source["state"] != "complete" for source in response["source_coverage"])
    as_of = _evidence_as_of(result, source_snapshot)
    if as_of is not None and publications:
        from app.web_similar import similar_activity

        # Похожие ищутся по всей выборке, а не по 200 записям, ушедшим в браузер.
        activity = similar_activity(publications[:15], publications, as_of)
        for publication in publications[:15]:
            publication["similar"] = activity[publication["publication_id"]]
    # The browser refuses responses above 2 MB. Keep a margin for the outer
    # status envelope while preserving the complete TOP and the true total.
    size = len(json.dumps(response, ensure_ascii=False, separators=(",", ":"))
               .encode("utf-8", "backslashreplace"))
    for publication in publications[15:]:
        if len(response["publications"]) >= 200:
            break
        item_size = len(json.dumps(publication, ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8", "backslashreplace")) + 1
        if size + item_size > 1_800_000:
            break
        response["publications"].append(publication)
        size += item_size
    if publication_pool_out is not None:
        publication_pool_out.extend(publications)
    return response


# A TOP card shows about three lines of an abstract; translating the whole
# abstract took two thirds of the time and nobody saw the rest.
EXCERPT_CHARACTERS = 280


def reading_excerpt(text: str, limit: int = EXCERPT_CHARACTERS) -> tuple[str, bool]:
    """Leading whole sentences a card can show, and whether anything was cut."""
    from app.pilot.translator import SENTENCE_END

    text = " ".join(text.split())
    excerpt = ""
    for sentence in SENTENCE_END.split(text):
        if excerpt and len(excerpt) + 1 + len(sentence) > limit:
            break
        excerpt = f"{excerpt} {sentence}".strip()
    return excerpt, len(excerpt) < len(text)


def english_text(text: str) -> bool:
    """Is the text mostly Latin letters, the only input the reading translators understand?"""
    letters = [char for char in text if char.isalpha()]
    latin = sum("a" <= char.casefold() <= "z" for char in letters)
    return bool(letters) and latin >= 0.6 * len(letters)


def top_texts(result: dict[str, Any]) -> list[str]:
    """Translate text actually visible in the ranked publication TOP cards."""
    titles, descriptions = top_text_groups(result)
    return titles + descriptions


def top_text_groups(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Titles and topic names, then descriptions: the short ones reach the reader first."""
    shown = result.get("top_publications", [])
    if not isinstance(shown, list):
        return [], []
    attached: dict[int, dict[str, Any]] = {}
    for signal in result.get("signals", []):
        if not isinstance(signal, dict) or not isinstance(signal.get("source_urls"), list):
            continue
        for url in signal["source_urls"]:
            if not isinstance(url, str):
                continue
            matched = next((index for index, publication in enumerate(shown)
                            if index not in attached and isinstance(publication, dict)
                            and isinstance(publication.get("url"), str)
                            and _publication_aliases(publication["url"]) & _publication_aliases(url)), None)
            if matched is not None:
                attached[matched] = signal
                break
    titles, descriptions = [], []
    for index, publication in enumerate(shown):
        if not isinstance(publication, dict):
            continue
        signal = attached.get(index)
        titles.append(publication.get("title"))
        if signal is not None:
            titles.append(signal.get("title"))
        elif isinstance(publication.get("trend"), dict):
            titles.append(publication["trend"].get("title"))
        descriptions.append(publication.get("summary") or
                            (signal.get("summary") if signal is not None else None))
    filtered_titles = list(dict.fromkeys(text for text in titles
                                         if isinstance(text, str) and text.strip()))
    known = set(filtered_titles)
    return filtered_titles, [text for text in dict.fromkeys(descriptions)
                    if isinstance(text, str) and text.strip() and text not in known]


def web_progress(row: dict[str, Any]) -> dict[str, str | int]:
    """Expose only bounded progress from the current run's saved row."""
    stage, message = row.get("stage"), row.get("message")
    completed, total = row.get("completed"), row.get("total")
    if (not isinstance(stage, str) or PROGRESS_STAGE.fullmatch(stage) is None
            or not isinstance(message, str) or len(message) > 1_000
            or type(completed) is not int or type(total) is not int
            or not 0 <= completed <= total):
        return {}
    return {"stage": stage, "message": message, "completed": completed, "total": total}


def _seconds_between(start: object, end: object) -> int | None:
    try:
        seconds = (datetime.fromisoformat(cast(str, end))
                   - datetime.fromisoformat(cast(str, start))).total_seconds()
    except (TypeError, ValueError):
        return None
    return round(seconds) if 0 <= seconds <= MAX_RUN_SECONDS else None


def run_duration_seconds(row: dict[str, Any]) -> int | None:
    """Wall time of one uninterrupted successful attempt.

    A resumed run's timestamps span the pause before resumption, so only the
    first attempt describes how long the mode actually takes.
    """
    if row.get("state") != "succeeded" or row.get("attempt") != 1:
        return None
    seconds = _seconds_between(row.get("created_at"), row.get("updated_at"))
    return seconds if seconds else None


def mode_estimates(rows: list[dict[str, Any]], provider: object) -> dict[str, dict[str, Any]]:
    """Forecast each mode from this machine's own finished runs with the same controls.

    Rows arrive newest first. A run from another provider or with the mode's
    former limits describes a different workload and is not counted.
    """
    samples: dict[str, list[int]] = {mode: [] for mode in COLLECTION_PROFILES}
    for row in rows:
        duration = run_duration_seconds(row)
        if duration is None:
            continue
        try:
            payload = json.loads(row["input_json"])["payload"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("operation") is not None:
            continue
        mode, settings = payload.get("collection_profile"), payload.get("settings")
        if (mode not in samples or not isinstance(settings, dict)
                or settings.get("provider") != provider
                or any(settings.get(key) != value for key, value in COLLECTION_PROFILES[mode].items())):
            continue
        if len(samples[mode]) < ESTIMATE_SAMPLES:
            samples[mode].append(duration)
    estimates = {}
    for mode, controls in COLLECTION_PROFILES.items():
        measured = samples[mode]
        estimates[mode] = {
            "expected_seconds": round(statistics.median(measured)) if measured else DEFAULT_MODE_SECONDS[mode],
            "basis": "history" if measured else "default", "runs": len(measured),
            "documents": controls["discovery_documents"], "top": controls["candidate_limit"],
            "patents": controls["patents_enabled"]}
    return estimates


ANALYST_ITEMS = 15


def _openalex_closed() -> float:
    """Сколько ещё OpenAlex закрыт для ТОПа технологий (исчерпан бюджет без ключа)."""
    from app.radar.evidence import openalex_closed_seconds

    return openalex_closed_seconds()


def analyst_summary(rendered: dict[str, Any], payload: dict[str, Any], translations: dict[str, str]) -> dict[str, Any]:
    """Тренды и технологии с принадлежащими им публикациями для панели аналитика."""
    def ru(text: object) -> str | None:
        return translations.get(text) if isinstance(text, str) else None

    raw_radar = payload.get("radar")
    radar_payload: dict[str, Any] = raw_radar if isinstance(raw_radar, dict) else {}
    raw_result = radar_payload.get("result")
    radar: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
    raw_policy_version = radar.get("policy_version")
    policy_version = (raw_policy_version if isinstance(raw_policy_version, str)
                      and RADAR_POLICY_VERSION.fullmatch(raw_policy_version) else None)
    raw_translation = radar.get("translation")
    radar_ru: dict[str, Any] = raw_translation if isinstance(raw_translation, dict) else {}
    top_publications = [item for item in rendered.get("top_publications", []) if isinstance(item, dict)]
    # A duplicate normalized URL is ambiguous: do not let TOP order decide
    # which publication's score gets attached to the radar source.
    publications_by_url: dict[str, dict[str, Any] | None] = {}
    for publication in top_publications:
        url = publication.get("url")
        if not isinstance(url, str):
            continue
        key = url_key(url)
        if key in publications_by_url:
            publications_by_url[key] = None
        else:
            publications_by_url[key] = publication

    def model_score(item: dict[str, Any] | None) -> int | None:
        confidence = item.get("model_confidence") if item else None
        score = confidence.get("score") if isinstance(confidence, dict) else None
        return score if type(score) is int and 0 <= score <= 100 else None

    # A preprint and journal version can have different URLs. Match by title
    # only when the rendered TOP has exactly one work with that title.
    publications_by_work: dict[str, dict[str, Any]] = {}
    ambiguous_works: set[str] = set()
    for publication in top_publications:
        title = publication.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        key = work_key(title)
        if key in publications_by_work:
            ambiguous_works.add(key)
        else:
            publications_by_work[key] = publication
    for key in ambiguous_works:
        publications_by_work.pop(key, None)

    def technology(item: dict[str, Any]) -> dict[str, Any]:
        raw_reasons = item.get("reasons")
        reasons: list[Any] = raw_reasons if isinstance(raw_reasons, list) else []
        description = item.get("description") if isinstance(item.get("description"), str) else None
        raw_translation = radar_ru.get(description) if description is not None else None
        translated_description = raw_translation if isinstance(raw_translation, str) else None
        sources = []
        seen_urls: set[str] = set()
        for source in item.get("sources", []) if isinstance(item.get("sources"), list) else []:
            if not isinstance(source, dict) or not is_safe_http_url(source.get("url")):
                continue
            url = source["url"]
            key = url_key(url)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            entry: dict[str, Any] = {"title": str(source.get("title") or "")[:300], "url": url,
                                     "published": str(source.get("published") or "")[:40],
                                     "source": str(source.get("source") or "")[:100],
                                     "type": str(source.get("source_type") or "")[:100]}
            # URL identity wins even when that publication has no model score.
            # Title matching is only a fallback for a different-version URL.
            publication = (publications_by_url[key] if key in publications_by_url else
                           publications_by_work.get(work_key(entry["title"])) if entry["title"] else None)
            score = model_score(publication)
            if score is not None:
                entry["model_confidence"] = score
            sources.append(entry)
            if len(sources) == 30:
                break
        return {"title": item.get("title"), "probability": item.get("probability"), "is_signal": item.get("is_signal"),
                "rule_excluded": item.get("rule_excluded"), "documents": item.get("pool_documents"),
                "description": (translated_description or description or "")[:400] or None,
                "reasons": [str(reason)[:200] for reason in reasons[:3]], "sources": sources}

    technologies = [item for item in radar.get("technologies", []) if isinstance(item, dict)]
    excluded = [item for item in radar.get("excluded", []) if isinstance(item, dict)]
    coverage = [item for item in rendered.get("source_coverage", []) if isinstance(item, dict)]
    warnings = []
    if rendered.get("openalex_rate_limited"):
        warnings.append("OpenAlex отказал из-за лимита бесплатных запросов: история публикаций и ТОП технологий "
                        "неполные. Поставьте ключ OpenAlex.")
    partial = [item["source_id"] for item in coverage if item.get("state") not in {"complete", None}]
    if rendered.get("incomplete_coverage") and partial:
        warnings.append("Источники собраны не полностью: " + ", ".join(sorted(set(partial))[:8]) + ".")
    if radar_payload.get("state") == "unavailable":
        warnings.append(str(radar_payload.get("message") or "ТОП технологий не собран."))
    elif radar_payload.get("state") == "pending":
        warnings.append("ТОП технологий ещё досчитывался, когда анализ закончился.")
    elif radar and not technologies:
        warnings.append("ТОП технологий пуст: ни один кандидат не прошёл отбор (см. исключённые).")
    funding = rendered.get("funding_evidence") if isinstance(rendered.get("funding_evidence"), dict) else None
    awards = funding.get("awards") if funding and isinstance(funding.get("awards"), list) else []
    total_usd = 0
    for award in awards:
        try:
            total_usd += int(Decimal(str(award.get("award_amount_usd") or 0)))
        except (InvalidOperation, ValueError, TypeError):
            continue
    return {
        "radar_policy_version": policy_version,
        "numbers": {"publications": rendered.get("publication_total"), "signals": len(rendered.get("signals", [])),
                    "technologies": len(technologies), "technology_candidates": len(technologies) + len(excluded),
                    "sources": sum(1 for item in coverage if (item.get("accepted") or 0) > 0),
                    "grants": len(awards), "grants_usd": total_usd},
        "warnings": warnings,
        "signals": [{"title": item.get("title"), "title_ru": ru(item.get("title")),
                     "summary": ru(item.get("summary")) or item.get("summary"), "category": item.get("category"),
                     "confidence": item.get("confidence"), "growth_confirmed": item.get("growth_confirmed"),
                     "checked": item.get("checked_features", []), "sources": len(item.get("source_urls", [])),
                     "url": (item.get("source_urls") or [None])[0]}
                    for item in rendered.get("signals", [])[:ANALYST_ITEMS] if isinstance(item, dict)],
        "technologies": [technology(item) for item in technologies[:ANALYST_ITEMS]],
        "excluded": [technology(item) for item in excluded[:8]],
        "top_publications": [{"title": item.get("title"), "title_ru": ru(item.get("title")),
                              "published_at": item.get("published_at"), "source": item.get("source_id"),
                              "kind": item.get("kind"), "url": item.get("url"),
                              "model_confidence": model_score(item),
                              "trend": (item.get("trend") or {}).get("title") if isinstance(item.get("trend"), dict)
                              else None}
                             for item in top_publications[:ANALYST_ITEMS]],
        "sources": [{"source_id": item.get("source_id"), "state": item.get("state"), "accepted": item.get("accepted")}
                    for item in coverage],
    }


def history_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    """Public fields of one saved analysis, or nothing for a service operation.

    Режим есть только у запусков с выбранным профилем сбора: у анализов,
    начатых в настольном приложении без него, `mode` пустой.
    """
    identifier, state = row.get("id"), row.get("state")
    if not isinstance(identifier, str) or RUN_ID.fullmatch(identifier) is None or state not in RUN_STATES:
        return None
    try:
        saved = json.loads(row["input_json"])
        created_at = datetime.fromisoformat(row["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    payload = saved.get("payload", saved) if isinstance(saved, dict) else None
    if not isinstance(payload, dict) or payload.get("operation") is not None:
        return None
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    mode = payload.get("collection_profile")
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return {"id": identifier, "query": query.strip()[:MAX_QUERY_CHARACTERS],
            "mode": mode if mode in ANALYSIS_MODES else None, "state": state,
            "created_at": created_at.isoformat()}


def _job(jobs: dict[str, dict[str, Any]], run_id: str, start: Any) -> dict[str, Any]:
    """The background job of one run: reused while it is watched or already done.

    `jobs` is ordered from the least to the most recently polled run.
    """
    now = time.monotonic()
    job = jobs.pop(run_id, None)
    if job is None:
        for other in list(jobs.values()):
            if other["state"] == "running" and now - other["seen"] > JOB_IDLE_SECONDS:
                other["cancel"].set()
                del jobs[other["run_id"]]
        job = start()
    job["seen"] = now
    jobs[run_id] = job
    while len(jobs) > FINISHED_JOB_CACHE:
        jobs.pop(next(iter(jobs)))["cancel"].set()
    return job


class WebAnalysisService(OwnerTools):
    """Own the same single-user profile as the desktop app."""

    def __init__(self, data_dir: Path | None = None, *, poll_seconds: float = 0.7,
                 timeout_seconds: float = 1_800):
        from app.backend.config import BackendSettings
        from app.backend.service import Backend
        from app.pilot.service import PilotService
        from app.profiles import resolve_profile

        directory = resolve_profile(data_dir if data_dir is not None else default_data_dir())
        self.backend = Backend(BackendSettings(data_dir=directory))
        try:
            self.pilot = PilotService(self.backend.settings.data_dir, self.backend.credentials)
        except BaseException:
            self.backend.close()
            raise
        self.pilot.radar_translator = self._radar_translation
        self.pilot.radar_handoff = True
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self._admission = Lock()
        self._state_lock = Lock()
        self._rendered_lock = Lock()
        # run_id → (result for the browser, its version, the complete publication pool).
        self._rendered: dict[str, tuple[dict[str, Any], str, tuple[dict[str, Any], ...]]] = {}
        # Russian reading drafts of each shown run's TOP, made in the background.
        self._translation_lock = Lock()
        self._translations: dict[str, dict[str, Any]] = {}
        self._reader: Any = None
        # Один переводчик на процесс: перевод ТОПа и радар не должны декодировать одновременно.
        self._reader_lock = Lock()
        self._radar_lock = Lock()
        self._radars: dict[str, dict[str, Any]] = {}
        self._current: dict[str, Any] | None = None
        self._cancel_requested_id: str | None = None
        # С паролем у каждого браузера свои анализы: API помнит, кто что запустил.
        # Связь живёт в памяти — после перезапуска меняются и ссылка, и пароль.
        self.separate_visitors = os.environ.get("TREND_API_SEPARATE_VISITORS") == "1"
        self._runs: dict[str, dict[str, Any]] = {}
        self._owners: dict[str, str] = {}
        self._latest: dict[str, str] = {}
        # Живая сводка для панели управления владельца (только в памяти процесса).
        # Блокировки владельца переживают перезапуск: лаунчер передаёт файл списка.
        blocklist = os.environ.get("TREND_API_BLOCKLIST")
        self.monitor = Monitor(anonymous="Не вошёл" if self.separate_visitors else "Локальный пользователь",
                               blocklist=Blocklist(Path(blocklist) if blocklist else None))
        # Владелец может приостановить новые анализы, не выключая сайт.
        self.paused = False
        self._closing = Event()
        self._admin_status: tuple[float, dict[str, Any]] | None = None
        # Несколько анализов одновременно и очередь сверх свободных мест.
        self._queue: list[str] = []
        self._cancelled: set[str] = set()
        self._dispatching = False
        self.resources = dict(RESOURCE_DEFAULTS)
        path = os.environ.get("TREND_API_RESOURCES")
        self._resources_path = Path(path) if path else None
        if self._resources_path is not None:
            try:
                self.resources.update(valid_resources(json.loads(self._resources_path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                pass
        self._apply_resources()

    def _apply_resources(self) -> None:
        from app.pilot.local_llm import BATCH_VARIABLE

        resources = self.__dict__.get("resources", RESOURCE_DEFAULTS)
        coordinator = getattr(self.pilot, "coordinator", None)
        if coordinator is not None and hasattr(coordinator, "set_slots"):
            coordinator.set_slots(resources["slots"])
        os.environ[BATCH_VARIABLE] = str(resources["llm_batch"])
        from app.radar.pipeline import EXACT_VARIABLE

        os.environ[EXACT_VARIABLE] = "1" if resources.get("radar_exact") else "0"
        self.pilot.keep_llm_loaded = resources["keep_llm_loaded"]

    def set_resources(self, values: dict[str, Any]) -> dict[str, Any]:
        """Изменить ресурсы сервиса и сохранить их; идущие анализы не прерываются."""
        changes = valid_resources(values)
        self.resources.update(changes)
        self._apply_resources()
        if not self.resources["keep_llm_loaded"] and not self._running_ids():
            getattr(self.pilot, "unload_llm", lambda: False)()
        if self._resources_path is not None:
            temporary = self._resources_path.with_name(self._resources_path.name + ".tmp")
            try:
                temporary.write_text(json.dumps(self.resources, indent=1), encoding="utf-8")
                os.replace(temporary, self._resources_path)
            except OSError:
                temporary.unlink(missing_ok=True)
        self._kick_dispatch()
        return dict(self.resources)
    def _monitor(self) -> Monitor:
        # Подделки сервиса в тестах создаются без инициализатора и без сводки.
        return cast(Monitor, self.__dict__.get("monitor") or SilentMonitor())

    def _watch_run(self, run_id: str) -> None:
        """Следить за этапами запуска, даже когда его страницу никто не смотрит."""
        monitor = self._monitor()
        closing = self.__dict__.get("_closing") or Event()
        failures = 0
        while not closing.wait(WATCH_SECONDS):
            try:
                row = self.pilot.get(run_id)
            except Exception:
                failures += 1
                if failures == 5:
                    monitor.event("service", "Не удаётся прочитать состояние идущего анализа.", level="error")
                continue
            failures = 0
            monitor.observe(run_id, row)
            if row.get("state") in TERMINAL_STATES:
                # Освободилось место: следующий из очереди стартует, пока этот доделывается.
                self._kick_dispatch()
                if row.get("state") == "succeeded":
                    self._warm_finished(run_id)
                    # Каждый завершённый анализ учит модель темы для следующих.
                    self.learn_after_run(run_id)
                return

    def _warm_finished(self, run_id: str) -> None:
        """Собрать готовый результат и начать переводы сразу по окончании анализа.

        Раньше это делал первый опрос страницы, и посетитель ждал сборки
        (замер 28.09.2026: 6,7 с на «quantum sensors») и затем перевода ТОПа.
        """
        try:
            with self._state_lock:
                current = self._runs.get(run_id) or (self._current if self._current and self._current["id"] == run_id
                                                     else None)
            query = current["query"] if current else ""
            rendered = self._web_result_for_run(run_id)
            self._translation_for_run(run_id, rendered)
            if query:
                self._radar_for_run(run_id, query)
        except Exception:
            pass  # Страница соберёт то же сама при первом опросе.

    @staticmethod
    def _validate_request(query: str, mode: str) -> str:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARACTERS:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY,
                              "Введите запрос длиной от 1 до 2000 символов.")
        try:
            query.encode("utf-8")
        except UnicodeEncodeError:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY,
                              "Запрос содержит некорректные символы.") from None
        if mode not in ANALYSIS_MODES:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY,
                              "Выберите доступный режим анализа.")
        return query

    def _web_result_for_run(self, run_id: str) -> dict[str, Any]:
        """Archive reads are done once for a completed run, not every UI poll."""
        lock = getattr(self, "_rendered_lock", None)
        if lock is None:
            # Small service fakes in integration tests construct this class
            # without calling its initializer.
            return web_result(self.pilot.result(run_id), archive=getattr(self.pilot, "archive", None))
        return self._rendered_run(run_id)[0]

    def _rendered_run(self, run_id: str) -> tuple[dict[str, Any], str, tuple[dict[str, Any], ...]]:
        """Result, version and complete pool of one finished run, assembled once."""
        with self._rendered_lock:
            cached = self._rendered.pop(run_id, None)
            if cached is None:
                pool: list[dict[str, Any]] = []
                rendered = web_result(self.pilot.result(run_id), archive=getattr(self.pilot, "archive", None),
                                      publication_pool_out=pool)
                # A version binds the immutable result to this run. The browser
                # polls translation and radar separately; it need not receive and
                # parse the complete publication list on every one of those polls.
                encoded = json.dumps(rendered, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8", "backslashreplace")
                version = hashlib.sha256(run_id.encode("utf-8") + b"\0" + encoded).hexdigest()
                cached = (rendered, version, tuple(pool))
            self._rendered[run_id] = cached
            while len(self._rendered) > RENDERED_RUNS:
                del self._rendered[next(iter(self._rendered))]
            return cached

    def _finished_run(self, run_id: str) -> dict[str, Any]:
        """Saved row of a finished analysis, or a bounded refusal."""
        try:
            row = self.pilot.get(run_id)
        except TaskFailure:
            raise WebApiError(HTTPStatus.NOT_FOUND, "Анализ не найден.") from None
        if row.get("state") != "succeeded":
            raise WebApiError(HTTPStatus.CONFLICT, "Анализ ещё не завершён.")
        return row

    def _check_owner(self, run_id: str, visitor: str | None) -> None:
        """A visitor reaches only the analyses it started; others look absent."""
        if visitor is not None:
            with self._state_lock:
                owner = self.__dict__.get("_owners", {}).get(run_id)
            if owner != visitor:
                raise WebApiError(HTTPStatus.NOT_FOUND, "Анализ не найден.")

    def publications_page(self, run_id: str, offset: int, limit: int, *,
                          visitor: str | None = None) -> dict[str, Any]:
        """Return a bounded slice of a finished run's already assembled pool."""
        if (not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None
                or type(offset) is not int or not 0 <= offset <= MAX_PUBLICATION_OFFSET
                or type(limit) is not int or not 1 <= limit <= MAX_PUBLICATION_PAGE):
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректная страница публикаций.")
        self._check_owner(run_id, visitor)
        self._finished_run(run_id)
        rendered, _, pool = self._rendered_run(run_id)
        if len(pool) != rendered["publication_total"]:
            raise WebApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Не удалось прочитать список публикаций.")
        if offset > len(pool):
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректная страница публикаций.")
        self._monitor().visitor_action(visitor, f"листает публикации: {min(offset + limit, len(pool))} "
                                       f"из {len(pool)}", repeat="publications:" + run_id)
        return {"run_id": run_id, "total": len(pool), "offset": offset,
                "publications": list(pool[offset:offset + limit])}

    def _finished_fields(self, run_id: str, query: str, language: str | None = None) -> dict[str, Any]:
        """Result of a finished run with its version, translation and technology TOP."""
        fields: dict[str, Any] = {"result": self._web_result_for_run(run_id)}
        cached = getattr(self, "_rendered", {}).get(run_id)
        if cached is not None:
            fields["result_version"] = cached[1]
        translation = (self._translation_for_run(run_id, fields["result"], language) if language is not None
                       else self._translation_for_run(run_id, fields["result"]))
        if translation is not None:
            fields["translation"] = translation
        radar = self._radar_for_run(run_id, query)
        if radar is not None:
            fields["radar"] = radar
        return fields

    def saved_analysis(self, run_id: str, *, visitor: str | None = None,
                       language: str | None = None) -> dict[str, Any]:
        """A finished analysis from the history for one browser page.

        The shared current analysis stays as it is: a saved result can be read
        while another analysis runs, by any number of sessions at once.
        """
        if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректный анализ.")
        self._check_owner(run_id, visitor)
        entry = history_entry({**self._finished_run(run_id), "id": run_id})
        if entry is None:
            raise WebApiError(HTTPStatus.NOT_FOUND, "Анализ не найден.")
        self._monitor().visitor_action(visitor, f"открыл анализ из истории «{entry['query'][:160]}»",
                                       repeat="saved:" + run_id)
        return {"id": run_id, "state": "succeeded", "query": entry["query"], "mode": entry["mode"] or "fast",
                "created_at": entry["created_at"], **self._finished_fields(run_id, entry["query"], language)}

    def status(self) -> dict[str, Any]:
        status = self.pilot.status()
        settings = status.get("settings", {})
        provider = settings.get("provider", "deepseek") if isinstance(settings, dict) else "deepseek"
        key_name = "yandex_api_key" if provider == "yandex" else "deepseek_api_key"
        return {"ready": bool(status.get("model_installed")), "model_state": status.get("model_state"),
                "provider": provider, "provider_key_configured": status.get("keys", {}).get(key_name)}

    def estimates(self) -> dict[str, Any]:
        """Expected duration and scope of each mode for the provider now configured."""
        settings = self.pilot.status().get("settings")
        provider = settings.get("provider") if isinstance(settings, dict) else None
        rows: list[dict[str, Any]] = []
        for page in range(ESTIMATE_HISTORY_PAGES):
            batch = self.pilot.list_runs(offset=page * 50, limit=50, source="local")
            rows.extend(batch)
            if len(batch) < 50:
                break
        return {"modes": mode_estimates(rows, provider)}

    def _expected_seconds(self, mode: str) -> int | None:
        # Прогноз только сопровождает запуск: сбой его расчёта не мешает анализу.
        try:
            return self.estimates()["modes"][mode]["expected_seconds"]
        except Exception:
            return None

    def analyze(self, query: str, mode: str = "fast") -> dict[str, Any]:
        query = self._validate_request(query, mode)
        if not self._admission.acquire(blocking=False):
            raise WebApiError(HTTPStatus.TOO_MANY_REQUESTS,
                              "Сервис уже выполняет анализ. Дождитесь его завершения.")
        try:
            status = self.status()
            if not status["ready"]:
                raise WebApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                                  "Локальная модель пока не готова.")
            try:
                run_id = self.pilot.start(query, collection_profile=cast(CollectionProfile, mode))
            except TaskCancelled:
                raise WebApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                                  "Запуск анализа был отменён.") from None
            except TaskFailure as error:
                raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from None
            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                row = self.pilot.get(run_id)
                state = row.get("state")
                if state == "succeeded":
                    return self._web_result_for_run(run_id)
                if state in TERMINAL_STATES:
                    message = row.get("error")
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY,
                                      message if isinstance(message, str) and message else
                                      "Анализ завершился без результата.")
                time.sleep(self.poll_seconds)
            self.pilot.cancel(run_id)
            raise WebApiError(HTTPStatus.GATEWAY_TIMEOUT,
                              "Анализ превысил лимит времени и был отменён.")
        finally:
            self._admission.release()

    def _queue_list(self) -> list[str]:
        return self.__dict__.setdefault("_queue", [])

    def _running_ids(self) -> list[str]:
        coordinator = getattr(self.pilot, "coordinator", None)
        running = getattr(coordinator, "running", None)
        return list(running()) if callable(running) else []

    def _free_slots(self) -> int:
        """Свободные места для анализа сейчас."""
        coordinator = getattr(self.pilot, "coordinator", None)
        if coordinator is None or not hasattr(coordinator, "running"):
            # Подделки сервиса в тестах: одно место, как было до очереди.
            with self._state_lock:
                current = dict(self._current) if self._current is not None else None
            return 0 if current is not None and self._is_active(current) else 1
        return max(0, coordinator.slots - len(coordinator.running()))

    def _is_active(self, record: dict[str, Any]) -> bool:
        """Анализ ещё не закончен: ждёт в очереди или считается."""
        if record.get("ticket"):
            if record.get("state") == "started" and record.get("run_id"):
                return self.pilot.get(record["run_id"]).get("state") not in TERMINAL_STATES
            return record.get("state") == "waiting"
        return self.pilot.get(record["id"]).get("state") not in TERMINAL_STATES

    def start_analysis(self, query: str, mode: str = "fast", *, visitor: str | None = None) -> dict[str, Any]:
        """Admit one web-owned run and return without waiting for its result.

        Свободное место — анализ стартует сразу; все места заняты — встаёт в
        очередь и стартует сам, как только одно освободится. У посетителя (и у
        владельца без пароля) один анализ за раз.
        """
        query = self._validate_request(query, mode)
        if self.__dict__.get("paused"):
            raise WebApiError(HTTPStatus.LOCKED, "Владелец временно приостановил новые анализы.")
        resources = self.__dict__.get("resources", RESOURCE_DEFAULTS)
        if mode == "deep" and visitor is not None and not resources["deep_allowed"]:
            raise WebApiError(HTTPStatus.LOCKED, "Глубокий режим временно выключен владельцем.")
        # Повторное нажатие того же посетителя отклоняется сразу; разные
        # посетители проходят допуск по очереди, не мешая друг другу.
        key = visitor or ""
        with self._state_lock:
            admitting = self.__dict__.setdefault("_admitting", set())
            if key in admitting:
                raise WebApiError(HTTPStatus.TOO_MANY_REQUESTS,
                                  "Сервис уже выполняет анализ. Дождитесь его завершения.")
            admitting.add(key)
        try:
            return self._admit(query, mode, visitor)
        finally:
            with self._state_lock:
                self.__dict__["_admitting"].discard(key)

    def _admit(self, query: str, mode: str, visitor: str | None) -> dict[str, Any]:
        if not self._admission.acquire(timeout=10):
            raise WebApiError(HTTPStatus.TOO_MANY_REQUESTS,
                              "Сервис перегружен запросами. Повторите через несколько секунд.")
        try:
            with self._state_lock:
                if visitor is None:
                    previous = dict(self._current) if self._current is not None else None
                else:
                    latest = self._latest.get(visitor)
                    previous = dict(self._runs[latest]) if latest in self._runs else None
            if previous is not None and self._is_active(previous):
                raise WebApiError(HTTPStatus.TOO_MANY_REQUESTS,
                                  "Сервис уже выполняет анализ. Дождитесь его завершения." if visitor is None
                                  else "Ваш анализ ещё идёт. Дождитесь его завершения или отмените его.")
            if not self.status()["ready"]:
                raise WebApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                                  "Локальная модель пока не готова.")
            # Прогноз фиксируется до запуска: сам запуск ещё не завершён и не
            # должен попасть в собственную оценку.
            expected = self._expected_seconds(mode)
            record: dict[str, Any] = {"query": query, "mode": mode}
            if expected is not None:
                record["expected_seconds"] = expected
            if self._free_slots() > 0 and not self._queue_list():
                return {**self._launch(record, visitor), "state": "queued"}
            return self._enqueue(record, visitor)
        finally:
            self._admission.release()

    def _launch(self, record: dict[str, Any], visitor: str | None) -> dict[str, Any]:
        """Start one admitted analysis; the caller holds the admission lock."""
        try:
            # Правило источников владельца (страны, выключенные, обученные доли) — в запуск.
            options = self.start_options() if "monitor" in self.__dict__ else {}
            run_id = self.pilot.start(record["query"], collection_profile=cast(CollectionProfile, record["mode"]),
                                      **options)
        except TaskCancelled:
            raise WebApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                              "Запуск анализа был отменён.") from None
        except TaskFailure as error:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from None
        current: dict[str, Any] = {"id": run_id, "query": record["query"], "mode": record["mode"]}
        if "expected_seconds" in record:
            current["expected_seconds"] = record["expected_seconds"]
        with self._state_lock:
            self._current = current
            self._cancel_requested_id = None
            if visitor is not None:
                self._runs[run_id] = current
                self._owners[run_id] = visitor
                self._latest[visitor] = run_id
        self._monitor().run_started(run_id, record["query"], record["mode"], visitor)
        if "monitor" in self.__dict__:
            Thread(target=self._watch_run, args=(run_id,), daemon=True, name="run-monitor").start()
        return current

    def _enqueue(self, record: dict[str, Any], visitor: str | None) -> dict[str, Any]:
        """Все места заняты: анализ ждёт своей очереди под номером-заявкой."""
        resources = self.__dict__.get("resources", RESOURCE_DEFAULTS)
        with self._state_lock:
            queue = self._queue_list()
            if len(queue) >= resources["queue_limit"]:
                raise WebApiError(HTTPStatus.TOO_MANY_REQUESTS,
                                  "Все места заняты, и очередь анализов заполнена. Попробуйте через несколько минут.")
            ticket = TICKET_PREFIX + secrets.token_hex(10)
            waiting = {**record, "id": ticket, "ticket": True, "state": "waiting", "visitor": visitor,
                       "created": time.time()}
            self._runs[ticket] = waiting
            queue.append(ticket)
            position = len(queue)
            if visitor is not None:
                self._owners[ticket] = visitor
                self._latest[visitor] = ticket
            else:
                self._current = waiting
        monitor = self._monitor()
        monitor.event("run", f"{monitor.label(visitor)} встал в очередь с анализом «{record['query'][:120]}»"
                             f" · перед ним {position - 1}")
        self._kick_dispatch()
        return {**self._public_record(waiting), "state": "waiting", "queue_position": position}

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in ("id", "query", "mode", "expected_seconds") if key in record}

    def _kick_dispatch(self) -> None:
        """Разбудить диспетчер очереди, если есть кого запускать."""
        with self._state_lock:
            if self.__dict__.get("_dispatching") or not self._queue_list():
                return
            self._dispatching = True
        Thread(target=self._dispatch_loop, daemon=True, name="analysis-queue").start()

    def _dispatch_loop(self) -> None:
        closing = self.__dict__.get("_closing") or Event()
        try:
            while not closing.is_set():
                with self._state_lock:
                    if not self._queue_list():
                        return
                if self.__dict__.get("paused") or self._free_slots() <= 0 or not self._dispatch_one():
                    closing.wait(DISPATCH_SECONDS)
        finally:
            with self._state_lock:
                self._dispatching = False
                again = bool(self._queue_list()) and not closing.is_set()
            if again:
                self._kick_dispatch()

    def _dispatch_one(self) -> bool:
        """Запустить первый ждущий анализ; False — место ещё не освободилось."""
        if not self._admission.acquire(timeout=5):
            return False
        try:
            with self._state_lock:
                queue = self._queue_list()
                if not queue:
                    return True
                ticket = queue[0]
                record = dict(self._runs[ticket])
            try:
                current = self._launch(record, record.get("visitor"))
            except WebApiError as error:
                if "места" in error.message:
                    return False  # Прежний анализ ещё закрывается: место освободится вот-вот.
                with self._state_lock:
                    if ticket in queue:
                        queue.remove(ticket)
                    self._runs[ticket].update(state="failed", error=error.message)
                return True
            with self._state_lock:
                if ticket in queue:
                    queue.remove(ticket)
                self._runs[ticket].update(state="started", run_id=current["id"])
            return True
        finally:
            self._admission.release()

    def history(self, offset: int, limit: int, *, visitor: str | None = None) -> dict[str, Any]:
        """Saved analyses of this profile, newest first, without their results.

        Вспомогательные операции (сигналы, ранние аналоги) в список не входят,
        поэтому страница может оказаться короче лимита.
        """
        if (type(offset) is not int or not 0 <= offset <= MAX_PUBLICATION_OFFSET
                or type(limit) is not int or not 1 <= limit <= MAX_HISTORY_PAGE):
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректная страница истории.")
        # Открытие истории видно по странице посетителя; API добавляет листание.
        if offset:
            self._monitor().visitor_action(visitor, f"листает историю анализов: с {offset + 1}-го", repeat="history")
        if visitor is not None:
            return self._visitor_history(offset, limit, visitor)
        rows = self.pilot.list_runs(offset=offset, limit=limit + 1, source="local")
        with self._state_lock:
            current = self._current["id"] if self._current is not None else None
        return {"offset": offset, "has_more": len(rows) > limit, "current_id": current,
                "runs": [entry for entry in map(history_entry, rows[:limit]) if entry is not None]}

    def _visitor_history(self, offset: int, limit: int, visitor: str) -> dict[str, Any]:
        """The visitor's own analyses, newest first."""
        with self._state_lock:
            owned = [run_id for run_id, owner in self._owners.items() if owner == visitor][::-1]
            latest = self._latest.get(visitor)
        rows = []
        for run_id in owned[offset:offset + limit]:
            try:
                rows.append({**self.pilot.get(run_id), "id": run_id})
            except TaskFailure:
                continue
        return {"offset": offset, "has_more": len(owned) > offset + limit, "current_id": latest,
                "runs": [entry for entry in map(history_entry, rows) if entry is not None]}

    def current_analysis(self, *, visitor: str | None = None, language: str | None = None) -> dict[str, Any]:
        """The run a page follows: the shared current one, or the visitor's latest.

        Посетитель не видит чужой анализ; пока тот идёт, он узнаёт только, что
        сервис занят (`service_busy`), — без текста запроса.
        """
        with self._state_lock:
            active = dict(self._current) if self._current is not None else None
            cancelled = set(self.__dict__.get("_cancelled", ())) | {self._cancel_requested_id} - {None}
            queue_length = len(self.__dict__.get("_queue", ()))
            if visitor is None:
                current = active
            else:
                latest = self.__dict__.get("_latest", {}).get(visitor)
                current = dict(self._runs[latest]) if latest is not None and latest in self._runs else None
        response = self._run_status(current, cancelled, language)
        resources = self.__dict__.get("resources", RESOURCE_DEFAULTS)
        if "coordinator" in getattr(self.pilot, "__dict__", {}) or hasattr(getattr(self.pilot, "coordinator", None),
                                                                          "running"):
            free = self._free_slots()
            if free <= 0 or queue_length:
                # Новый анализ встанет в очередь; стартовать нельзя, только если и она полна.
                response["queue_length"] = queue_length
                response["slots_full"] = True
                if queue_length >= resources["queue_limit"] and response.get("state") not in {"waiting", "queued",
                                                                                              "running", "cancelling"}:
                    response["service_busy"] = True
        elif (visitor is not None and active is not None and (current is None or active["id"] != current["id"])
                and self.pilot.get(active["id"]).get("state") in {"queued", "running"}):
            response["service_busy"] = True
        if self.__dict__.get("paused"):
            response["paused"] = True
        if not resources["deep_allowed"] and visitor is not None:
            response["deep_disabled"] = True
        return response

    def _run_status(self, current: dict[str, Any] | None, cancelled: Any,
                    language: str | None = None) -> dict[str, Any]:
        if current is None:
            return {"state": "idle"}
        if current.get("ticket"):
            state = current.get("state")
            if state == "started" and current.get("run_id"):
                with self._state_lock:
                    started = self._runs.get(current["run_id"]) or {"id": current["run_id"], "query": current["query"],
                                                                    "mode": current["mode"]}
                return self._run_status(dict(started), cancelled, language)
            response: dict[str, Any] = {**self._public_record(current), "state": state}
            if state == "waiting":
                with self._state_lock:
                    queue = self._queue_list()
                    response["queue_position"] = queue.index(current["id"]) + 1 if current["id"] in queue else 1
                response["elapsed_seconds"] = max(0, round(time.time() - current.get("created", time.time())))
            elif state == "failed":
                response["error"] = current.get("error") or "Анализ не удалось запустить."
            return response
        row = self.pilot.get(current["id"])
        state = row.get("state")
        if state not in TERMINAL_STATES | {"queued", "running"}:
            raise WebApiError(HTTPStatus.SERVICE_UNAVAILABLE,
                              "Не удалось прочитать состояние анализа.")
        self._monitor().observe(current["id"], row)
        is_cancelled = (current["id"] in cancelled if isinstance(cancelled, (set, frozenset))
                        else cancelled == current["id"])
        response = {**self._public_record(current),
                    "state": "cancelling" if state in {"queued", "running"} and is_cancelled else state}
        response.update(web_progress(row))
        if state in {"queued", "running"}:
            elapsed = _seconds_between(row.get("created_at"), datetime.now(UTC).isoformat())
            if elapsed is not None:
                response["elapsed_seconds"] = elapsed
        if state == "succeeded":
            response.update(self._finished_fields(current["id"], current["query"], language))
        elif state in {"failed", "interrupted"}:
            error = row.get("error")
            response["error"] = error[:2_000] if isinstance(error, str) and error else "Анализ не завершён."
        return response

    def _languages(self) -> Any:
        """Языки перевода профиля; у подделок сервиса в тестах их нет."""
        try:
            return self._language_manager()
        except Exception:
            return None

    def _reading_language(self, requested: str | None) -> str:
        """Язык черновика ТОПа: выбранный посетителем, если он установлен, иначе язык сайта."""
        manager = self._languages()
        if manager is None:
            return "ru"
        try:
            available = {entry["code"] for entry in manager.reading()}
            default = manager.default()
        except Exception:
            return "ru"
        if requested in available:
            return str(requested)
        return default if default in available else "ru"

    def _translation_for_run(self, run_id: str, rendered: dict[str, Any],
                             language: str | None = None) -> dict[str, Any] | None:
        """Start the TOP translation of a finished run once and report how far it got.

        Texts arrive in two whole steps: titles and topics first, descriptions
        after, so cards never change language one by one. Каждый язык — своё
        задание: переход на другой язык не выбрасывает уже готовый перевод.
        """
        lock = getattr(self, "_translation_lock", None)
        if lock is None:
            return None  # Service fakes in tests skip the initializer.
        code = self._reading_language(language)
        key = run_id if code == "ru" else f"{run_id}.{code}"

        def start() -> dict[str, Any]:
            groups = top_text_groups(rendered)
            job = {"run_id": key, "state": "running", "texts": {}, "published": {}, "done": 0,
                   "total": sum(map(len, groups)), "message": None, "cancel": Event(), "language": code}
            Thread(target=self._translate_top, args=(job, *groups), daemon=True,
                   name="top-translation").start()
            return job

        with lock:
            job = _job(self.__dict__.setdefault("_translations", {}), key, start)
            public = {"state": job["state"], "completed": job["done"], "total": job["total"]}
            if job["published"]:
                public["texts"] = dict(job["published"])
            if job["message"]:
                public["message"] = job["message"]
        manager = self._languages()
        if manager is not None and code != "ru" or manager is not None and len(manager.reading()) > 1:
            public["language"] = code
            public["languages"] = manager.reading()
        return public

    def _translate_top(self, job: dict[str, Any], titles: list[str], descriptions: list[str]) -> None:
        from app.pilot.translator import TranslationError

        def counted(_index: int) -> None:
            with self._translation_lock:
                job["done"] += 1

        language = job.get("language", "ru")
        options = {"language": language} if language != "ru" else {}
        try:
            # One batch per step: the model decodes a whole batch almost as fast as one text.
            for group, excerpt in ((titles, False), (descriptions, True)):
                # Переводчики читают английский: русская новость или статья
                # КиберЛенинки выходила из них бессмыслицей («В»: «В» (В), 0,001…),
                # поэтому такой текст остаётся в оригинале.
                skipped = sum(not english_text(text) for text in group)
                group = [text for text in group if english_text(text)]
                if skipped:
                    with self._translation_lock:
                        job["done"] += skipped
                sources = [reading_excerpt(text) if excerpt else (text, False) for text in group]
                drafts = self._translate([source for source, _ in sources], cancel=job["cancel"], done=counted,
                                         **options)
                with self._translation_lock:
                    for text, (_, clipped), russian in zip(group, sources, drafts, strict=True):
                        if russian:
                            job["texts"][text] = (russian + (" …" if clipped else ""))[:20_000]
                    job["published"] = dict(job["texts"])
        except TranslationError as error:
            if job["cancel"].is_set():
                return  # Nobody watched this run any more; the job was dropped.
            with self._translation_lock:
                job.update(state="unavailable", message=str(error)[:500])
            return
        except Exception:
            with self._translation_lock:
                job.update(state="unavailable", message="Не удалось перевести ТОП.")
            return
        with self._translation_lock:
            job["state"] = "ready"

    def _translate(self, texts: list[str], *, cancel: Event, done: Any = None, language: str = "ru") -> list[str]:
        from app.pilot.translator import ENGLISH_RUSSIAN_KEY, EnglishRussianTranslator, model_directory

        with self._reader_lock:
            if language != "ru":
                # Переводчики добавленных языков грузятся по первому запросу и живут в процессе.
                from app.pilot.languages import reading_translator

                readers = self.__dict__.setdefault("_readers", {})
                if language not in readers:
                    readers[language] = reading_translator(self.pilot.data_dir, language, cancel)
                return readers[language].translate_many(texts, cancel=cancel, done=done)
            if self._reader is None:
                self._reader = EnglishRussianTranslator(
                    model_directory(getattr(self.pilot, "data_dir", None), ENGLISH_RUSSIAN_KEY), cancel=cancel)
            return self._reader.translate_many(texts, cancel=cancel, done=done)

    def _radar_for_run(self, run_id: str, query: str) -> dict[str, Any] | None:
        """ТОП-15 технологий показанного анализа: из результата или, у старых, расчётом в фоне."""
        lock = getattr(self, "_radar_lock", None)
        if lock is None:
            return None  # Service fakes in tests skip the initializer.
        def start() -> dict[str, Any]:
            job = {"run_id": run_id, "state": "running", "done": 0, "total": 0, "result": None,
                   "message": None, "cancel": Event()}
            Thread(target=self._build_radar, args=(job, run_id, query), daemon=True,
                   name="technology-radar").start()
            return job

        with lock:
            job = _job(self.__dict__.setdefault("_radars", {}), run_id, start)
            public: dict[str, Any] = {"state": job["state"], "completed": job["done"], "total": job["total"]}
            if job["result"] is not None:
                public["result"] = job["result"]
            if job["message"]:
                public["message"] = job["message"]
            return public

    def _build_radar(self, job: dict[str, Any], run_id: str, query: str) -> None:
        from app.radar.pipeline import SHARED_CACHE, build_radar

        def progress(done: int, total: int) -> None:
            with self._radar_lock:
                job.update(done=done, total=total)

        try:
            payload = self.pilot.result(run_id)
            # Веб-анализ считает ТОП технологий сам и сохраняет его в результат;
            # заново он собирается только для анализов, где его нет.
            stored = payload.get("radar")
            if isinstance(stored, dict) and stored.get("state") == "unavailable":
                message = stored.get("message")
                with self._radar_lock:
                    job.update(state="unavailable", message=message[:500] if isinstance(message, str) and message
                               else "Не удалось собрать ТОП технологий.")
                return
            handed = (getattr(self.pilot, "take_radar", lambda _run_id: None)(run_id)
                      if isinstance(stored, dict) and stored.get("state") == "pending" else None)
            if handed is not None:
                # Анализ закончился раньше своего ТОПа технологий и передал его досчёт.
                future, steps, radar_cancel = handed
                while not future.done():
                    if job["cancel"].wait(0.5):
                        # Страницу закрыли: досчёт не бросается, а ждёт следующего показа.
                        self.pilot._hand_off_radar(run_id, future, steps, radar_cancel)
                        return
                    progress(*steps)
                stored = future.result()
                if not isinstance(stored, dict) or stored.get("state") != "ready":
                    message = stored.get("message") if isinstance(stored, dict) else None
                    with self._radar_lock:
                        job.update(state="unavailable", message=message[:500] if isinstance(message, str) and message
                                   else "Не удалось собрать ТОП технологий.")
                    return
            if isinstance(stored, dict) and stored.get("state") == "ready" and isinstance(stored.get("result"), dict):
                radar = dict(stored["result"])
            else:
                pool, as_of, terms = radar_input(payload, archive=getattr(self.pilot, "archive", None))
                if as_of is None or not pool:
                    with self._radar_lock:
                        job.update(state="unavailable", message="Для ТОПа технологий нет найденных материалов.")
                    return
                credentials = getattr(getattr(self, "backend", None), "credentials", None)
                radar = build_radar(pool, query=query, query_terms=terms or [query], as_of=as_of,
                                    cancel=job["cancel"], progress=progress, cache=SHARED_CACHE,
                                    openalex_key=credentials.get("openalex_api_key") if credentials else None)
            # Веб-анализ переводит ТОП технологий ещё во время анализа; здесь —
            # только старые анализы и те, где переводчик тогда не справился.
            if not radar.get("translation"):
                with self._radar_lock:
                    job.update(message="Переводим описания технологий на русский…")
                radar["translation"] = self._radar_translation(radar, job["cancel"])
        except Exception:
            if job["cancel"].is_set():
                return
            with self._radar_lock:
                job.update(state="unavailable", message="Не удалось собрать ТОП технологий.")
            return
        with self._radar_lock:
            if not job["cancel"].is_set():
                job.update(state="ready", result=radar, message=None)

    def _radar_translation(self, radar: dict[str, Any], cancel: Event) -> dict[str, str]:
        """Русские названия и выжимки; отказ переводчика оставляет оригинал."""
        texts = list(dict.fromkeys(
            text for entry in (*radar["technologies"], *radar["excluded"])
            # Короткие термины переводчик искажает («prompt injection» → «быстрый впрыск»):
            # названия технологий остаются в оригинале, переводятся только выжимки.
            for text in (entry.get("description"), entry.get("advantage"), entry.get("case"))
            if isinstance(text, str) and text.strip()))
        if not texts:
            return {}
        try:
            drafts = self._translate(texts, cancel=cancel)
        except Exception:
            return {}
        return {text: russian[:5_000] for text, russian in zip(texts, drafts, strict=True) if russian}

    def _withdraw(self, ticket: str) -> bool:
        """Убрать ждущий анализ из очереди; False — он уже стартовал или закончен."""
        with self._state_lock:
            queue = self._queue_list()
            record = self._runs.get(ticket)
            if record is None or record.get("state") != "waiting" or ticket not in queue:
                return False
            queue.remove(ticket)
            record["state"] = "cancelled"
            return True

    def cancel_analysis(self, expected_id: str, *, visitor: str | None = None) -> dict[str, Any]:
        with self._state_lock:
            if visitor is None:
                current = dict(self._current) if self._current is not None else None
            else:
                current = dict(self._runs[expected_id]) if expected_id in self._runs else None
            owner = self.__dict__.get("_owners", {}).get(expected_id)
        if current is None or current["id"] != expected_id or visitor is not None and owner != visitor:
            raise WebApiError(HTTPStatus.CONFLICT,
                              "Состояние анализа изменилось. Обновите страницу.")
        if current.get("ticket"):
            if current.get("state") == "started" and current.get("run_id"):
                return self.cancel_analysis(current["run_id"], visitor=visitor)
            if not self._withdraw(expected_id):
                raise WebApiError(HTTPStatus.CONFLICT, "Состояние анализа изменилось. Обновите страницу.")
            monitor = self._monitor()
            monitor.event("run", f"{monitor.label(visitor)} убрал свой анализ из очереди", level="warning")
            return self.current_analysis(visitor=visitor) if visitor is not None else self.current_analysis()
        if self.pilot.cancel(current["id"]):
            with self._state_lock:
                self.__dict__.setdefault("_cancelled", set()).add(current["id"])
                if self._current is not None and self._current["id"] == current["id"]:
                    self._cancel_requested_id = current["id"]
            monitor = self._monitor()
            monitor.cancel_requested(current["id"], monitor.label(visitor), visitor=visitor)
        status = self.current_analysis(visitor=visitor) if visitor is not None else self.current_analysis()
        if status.get("id") != expected_id:
            raise WebApiError(HTTPStatus.CONFLICT,
                              "Состояние анализа изменилось. Обновите страницу.")
        return status

    def presence(self, session: str, page: str, event: str | None, agent: object, ip: object, *,
                 view: str | None = None, detail: object = None, visitor: str | None = None) -> dict[str, Any]:
        """Отметка вкладки сайта для панели владельца; в ответ — закрыт ли вход и сообщения."""
        try:
            answer = self._monitor().presence(session, visitor, page, agent=agent, ip=ip, event=event,
                                              view=view, detail=detail)
        except ValueError:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request") from None
        return answer or {"access": None, "messages": []}

    def access(self, ip: object, *, visitor: str | None = None) -> dict[str, Any]:
        """Пускать ли этот браузер и адрес на сайт — до того, как страница что-то покажет."""
        return {"access": self._monitor().access(visitor, ip)}

    def visitor_refusal(self, visitor: str) -> str | None:
        """Заблокированному или выгнанному гостю API не отвечает ничем, кроме отказа."""
        return self._monitor().blocked_visitor(visitor)

    def _admin_service(self) -> dict[str, Any]:
        """Готовность модели и ключи: проверка ключей дорогая, поэтому не на каждый опрос."""
        cached = self.__dict__.get("_admin_status")
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
        try:
            status = self.pilot.status()
        except Exception:
            status = {}
        settings = status.get("settings") if isinstance(status.get("settings"), dict) else {}
        keys = status.get("keys") if isinstance(status.get("keys"), dict) else {}
        service = {"ready": bool(status.get("model_installed")), "model_state": status.get("model_state"),
                   "provider": settings.get("provider"), "local_llm_installed": status.get("local_llm_installed"),
                   "keys": {name: keys.get(name) for name in ("deepseek_api_key", "yandex_api_key",
                                                                "openalex_api_key", "epo_ops_key")},
                   "data_dir": str(getattr(self.pilot, "data_dir", "") or "") or None}
        self._admin_status = (time.monotonic() + ADMIN_STATUS_SECONDS, service)
        return service

    def _admin_current(self, current: dict[str, Any], row: dict[str, Any], cancelled: str | None,
                       owner: str | None) -> dict[str, Any]:
        monitor = self._monitor()
        state = row.get("state")
        live: dict[str, Any] = {
            "id": current["id"], "query": current["query"], "mode": current["mode"],
            "state": "cancelling" if state in {"queued", "running"} and cancelled == current["id"] else state,
            "visitor": monitor.label(owner) if owner is not None else None,
            "expected_seconds": current.get("expected_seconds"), "started_at": row.get("created_at"),
            **web_progress(row)}
        live["stage_label"] = stage_label(live.get("stage"))
        end = row.get("updated_at") if state in TERMINAL_STATES else datetime.now(UTC).isoformat()
        live["elapsed_seconds"] = _seconds_between(row.get("created_at"), end)
        live["stages"] = monitor.timeline(current["id"])
        error = row.get("error")
        if state in {"failed", "interrupted"} and isinstance(error, str) and error:
            live["error"] = error[:2_000]
        return live

    def _admin_runs(self, owners: dict[str, str]) -> list[dict[str, Any]]:
        monitor = self._monitor()
        runs = []
        for row in self.pilot.list_runs(offset=0, limit=ADMIN_RUNS, source="local"):
            entry = history_entry(row)
            if entry is None:
                continue
            if entry["state"] in TERMINAL_STATES:
                entry["duration_seconds"] = _seconds_between(row.get("created_at"), row.get("updated_at"))
            else:
                entry["stage_label"] = stage_label(row.get("stage"))
            owner = owners.get(entry["id"])
            entry["visitor"] = monitor.label(owner) if owner is not None else None
            error = row.get("error")
            if entry["state"] in {"failed", "interrupted"} and isinstance(error, str) and error:
                entry["error"] = error[:500]
            runs.append(entry)
        return runs

    def _admin_jobs(self) -> list[dict[str, Any]]:
        """Фоновые переводы ТОПа и ТОПы технологий показанных анализов."""
        monitor, now, jobs = self._monitor(), time.monotonic(), []
        for kind, lock_name, table in (("Перевод ТОПа", "_translation_lock", "_translations"),
                                       ("ТОП-15 технологий", "_radar_lock", "_radars")):
            lock = self.__dict__.get(lock_name)
            if lock is None:
                continue
            with lock:
                for key, job in self.__dict__.get(table, {}).items():
                    run_id, _, language = key.partition(".")
                    record = monitor.run_record(run_id) or {}
                    jobs.append({"kind": kind + (f" ({language})" if language else ""), "run_id": run_id,
                                 "query": record.get("query"),
                                 "state": job["state"], "completed": job["done"], "total": job["total"],
                                 "message": job["message"], "idle_seconds": round(now - job["seen"])})
        return jobs

    def admin_overview(self, after: int = 0, person: str | None = None) -> dict[str, Any]:
        """Всё, что видит панель управления владельца, одним ответом."""
        monitor = self._monitor()
        with self._state_lock:
            current = dict(self._current) if self._current is not None else None
            cancelled = self._cancel_requested_id
            owners = dict(self.__dict__.get("_owners", {}))
        live = None
        if current is not None and not current.get("ticket"):
            row = self.pilot.get(current["id"])
            monitor.observe(current["id"], row)
            live = self._admin_current(current, row, cancelled, owners.get(current["id"]))
        # Все идущие анализы (их может быть несколько) и очередь.
        active = []
        with self._state_lock:
            records = {run_id: dict(record) for run_id, record in self._runs.items()}
            queue = list(self.__dict__.get("_queue", ()))
            cancelled_all = set(self.__dict__.get("_cancelled", ())) | ({cancelled} if cancelled else set())
        if current is not None:
            records.setdefault(current["id"], current)
        for run_id in self._running_ids():
            record = records.get(run_id) or {"id": run_id, "query": "(анализ настольного приложения)", "mode": "fast"}
            try:
                row = self.pilot.get(run_id)
            except TaskFailure:
                continue
            monitor.observe(run_id, row)
            view = self._admin_current(record, row, None, owners.get(run_id))
            if run_id in cancelled_all and view["state"] in {"queued", "running"}:
                view["state"] = "cancelling"
            active.append(view)
        waiting = []
        for position, ticket in enumerate(queue, start=1):
            record = records.get(ticket, {})
            visitor = record.get("visitor")
            waiting.append({"id": ticket, "position": position, "query": record.get("query"), "mode": record.get("mode"),
                            "visitor": monitor.label(visitor) if visitor is not None else None,
                            "waiting_seconds": max(0, round(time.time() - record.get("created", time.time())))})
        try:
            runs: list[dict[str, Any]] | None = self._admin_runs(owners)
        except Exception:
            runs = None
        return {"generated_at": datetime.now(UTC).isoformat(),
                "service": {**self._admin_service(), "separate_visitors": self.separate_visitors,
                            "paused": bool(self.__dict__.get("paused")),
                            "openalex_closed_seconds": round(_openalex_closed()),
                            "llm_loaded": bool(getattr(self.pilot, "llm_loaded", lambda: False)())},
                "resources": dict(self.__dict__.get("resources", RESOURCE_DEFAULTS)),
                "active": active, "queue": waiting,
                "current": live, "runs": runs, "jobs": self._admin_jobs(), **monitor.snapshot(after, person)}

    def admin_run(self, run_id: str) -> dict[str, Any]:
        """Результат одного анализа глазами аналитика."""
        if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректный анализ.")
        try:
            row = self.pilot.get(run_id)
        except TaskFailure:
            raise WebApiError(HTTPStatus.NOT_FOUND, "Анализ не найден.") from None
        entry = history_entry({**row, "id": run_id})
        if entry is None:
            raise WebApiError(HTTPStatus.NOT_FOUND, "Это служебная операция, а не анализ.")
        with self._state_lock:
            owner = self.__dict__.get("_owners", {}).get(run_id)
        monitor = self._monitor()
        summary: dict[str, Any] = {**entry, "visitor": monitor.label(owner) if owner is not None else None,
                                   "duration_seconds": _seconds_between(row.get("created_at"), row.get("updated_at"))
                                   if entry["state"] in TERMINAL_STATES else None,
                                   "stages": monitor.timeline(run_id)}
        if entry["state"] in {"failed", "interrupted"}:
            summary["error"] = (row.get("error") or "Анализ не завершён.")[:2_000]
        if entry["state"] != "succeeded":
            summary.update(web_progress(row))
            return summary
        rendered = self._web_result_for_run(run_id)
        translations: dict[str, str] = {}
        lock = self.__dict__.get("_translation_lock")
        if lock is not None:
            with lock:
                job = self.__dict__.get("_translations", {}).get(run_id)
                if job is not None:
                    translations = dict(job["published"])
        payload = self.pilot.result(run_id)
        radar_lock = self.__dict__.get("_radar_lock")
        if radar_lock is not None:
            # Досчитанный после анализа ТОП технологий живёт у сайта, не в сохранённом результате.
            with radar_lock:
                job = self.__dict__.get("_radars", {}).get(run_id)
                if job is not None and job["state"] == "ready" and isinstance(job["result"], dict):
                    payload = {**payload, "radar": {"state": "ready", "result": job["result"]}}
                elif job is not None and job["state"] == "running":
                    payload = {**payload, "radar": {"state": "pending"}}
        summary.update(analyst_summary(rendered, payload, translations))
        return summary

    def admin_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Рычаги владельца над посетителями и приёмом анализов."""
        monitor = self._monitor()
        action = payload.get("action")
        if action == "sources":
            return self.set_sources(payload.get("values"))
        if action == "languages":
            return self.set_languages(payload.get("values"))
        if action in {"learning", "retrain", "reset_learning", "feedback"}:
            return self.learning_action(str(action), payload.get("values"))
        try:
            if action == "pause":
                self.paused = True
                monitor.event("owner", "Владелец приостановил новые анализы", level="warning")
                return {"message": "Новые анализы приостановлены. Идущий анализ продолжается."}
            if action == "resume":
                self.paused = False
                monitor.event("owner", "Владелец снова открыл запуск анализов", level="success")
                return {"message": "Запуск анализов снова открыт."}
            if action == "resources":
                values = payload.get("values")
                try:
                    applied = self.set_resources(values if isinstance(values, dict) else {})
                except ValueError:
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Недопустимое значение ресурса.") from None
                monitor.event("owner", "Владелец изменил ресурсы: " + ", ".join(
                    f"{name} = {applied[name]}" for name in cast(dict[str, Any], values)), level="info")
                return {"message": "Ресурсы обновлены.", "resources": applied}
            if action == "unload_llm":
                if self._running_ids():
                    raise WebApiError(HTTPStatus.CONFLICT, "Сейчас идёт анализ: модель выгрузится, когда он закончится. "
                                      "Выключите «держать модель в памяти».")
                unloaded = getattr(self.pilot, "unload_llm", lambda: False)()
                monitor.event("owner", "Владелец выгрузил локальную модель из памяти")
                return {"message": "Модель выгружена из видеопамяти." if unloaded else "Модель и так не загружена."}
            if action == "unblock":
                return {"message": monitor.unblock(payload["id"])}
            if action == "message":
                return {"message": monitor.message(payload["person"], payload["text"])}
            if action == "sign_out":
                return {"message": monitor.sign_out(payload["person"])}
            if action == "block":
                blocked = monitor.block(payload["person"])
                message = blocked["message"]
                # Заблокированный не должен и дальше занимать сервис своим анализом.
                with self._state_lock:
                    current = dict(self._current) if self._current is not None else None
                    owner = self.__dict__.get("_owners", {}).get(current["id"]) if current else None
                if (current is not None and blocked["visitor"] is not None and owner == blocked["visitor"]
                        and self.pilot.get(current["id"]).get("state") in {"queued", "running"}):
                    self.admin_cancel(current["id"])
                    message += " Его анализ остановлен."
                return {"message": message}
        except KeyError:
            raise WebApiError(HTTPStatus.NOT_FOUND, "Посетитель или блокировка уже не найдены. Обновите панель.") from None
        except (LookupError, ValueError) as error:
            raise WebApiError(HTTPStatus.CONFLICT, str(error) or "Действие недоступно.") from None
        raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")

    def admin_cancel(self, expected_id: str) -> dict[str, Any]:
        """Владелец останавливает идущий анализ из панели, чей бы он ни был."""
        if expected_id.startswith(TICKET_PREFIX):
            if not self._withdraw(expected_id):
                raise WebApiError(HTTPStatus.CONFLICT, "Этот анализ уже не ждёт в очереди.")
            self._monitor().event("run", "Владелец (из панели) убрал анализ из очереди", level="warning")
            return {"id": expected_id, "state": "cancelled"}
        with self._state_lock:
            current = dict(self._current) if self._current is not None else None
            known = expected_id in self._runs or current is not None and current["id"] == expected_id
        if not known and expected_id not in self._running_ids():
            raise WebApiError(HTTPStatus.CONFLICT, "Этот анализ уже не идёт.")
        if self.pilot.get(expected_id).get("state") in TERMINAL_STATES:
            raise WebApiError(HTTPStatus.CONFLICT, "Этот анализ уже завершён.")
        if self.pilot.cancel(expected_id):
            with self._state_lock:
                self.__dict__.setdefault("_cancelled", set()).add(expected_id)
                if self._current is not None and self._current["id"] == expected_id:
                    self._cancel_requested_id = expected_id
            self._monitor().cancel_requested(expected_id, "Владелец (из панели)")
        return {"id": expected_id, "state": "cancelling"}

    def close(self) -> None:
        closing = self.__dict__.get("_closing")
        if closing is not None:
            closing.set()
        with suppress(Exception):
            self.pilot.close()
        self.backend.close()


class ApiServer(ThreadingHTTPServer):
    analysis: WebAnalysisService
    daemon_threads = True
    request_queue_size = MAX_API_CONNECTIONS

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler],
                 bind_and_activate: bool = True):
        if server_address[0] != "127.0.0.1":
            raise ValueError("Внутренний API можно слушать только на 127.0.0.1.")
        token = os.environ.get("TREND_API_TOKEN")
        if token is None and os.environ.get("TREND_API_ALLOW_UNAUTHENTICATED_LOCAL") != "1":
            raise ValueError("Для API нужен TREND_API_TOKEN. Только для ручной локальной разработки "
                             "можно задать TREND_API_ALLOW_UNAUTHENTICATED_LOCAL=1.")
        if token is not None and (not MIN_API_TOKEN_CHARACTERS <= len(token) <= MAX_API_TOKEN_CHARACTERS
                                  or not token.isascii() or not token.isprintable()):
            raise ValueError("TREND_API_TOKEN должен содержать от 32 до 256 печатных ASCII-символов.")
        self.api_token = token
        # Панель владельца ходит со своим токеном: у процесса сайта его нет, и
        # сводка обо всех посетителях до сайта не доходит, даже если он ошибётся.
        admin_token = os.environ.get("TREND_API_ADMIN_TOKEN")
        if admin_token is not None and (not MIN_API_TOKEN_CHARACTERS <= len(admin_token) <= MAX_API_TOKEN_CHARACTERS
                                        or not admin_token.isascii() or not admin_token.isprintable()
                                        or admin_token == token):
            raise ValueError("TREND_API_ADMIN_TOKEN должен быть отдельной строкой из 32–256 печатных ASCII-символов.")
        self.admin_token = admin_token
        instance_id = os.environ.get("TREND_API_INSTANCE_ID")
        if instance_id is not None and (not 24 <= len(instance_id) <= 128
                                        or re.fullmatch(r"[A-Za-z0-9_-]+", instance_id) is None):
            raise ValueError("TREND_API_INSTANCE_ID должен быть безопасным идентификатором запуска.")
        self.instance_id = instance_id
        self._client_slots = BoundedSemaphore(MAX_API_CONNECTIONS)
        self._health_lock = Lock()
        self._health_cache: tuple[float, tuple[int, dict[str, Any]]] | None = None
        super().__init__(server_address, handler_class, bind_and_activate)

    def health(self) -> tuple[int, dict[str, Any]]:
        """Coalesce unauthenticated probes before they reach model and keyring checks."""
        with self._health_lock:
            cached = self._health_cache
            if cached is not None and time.monotonic() < cached[0]:
                return cached[1]
            try:
                status = self.analysis.status()
                # Startup probes need readiness, never provider configuration or
                # credential presence. Health remains reachable without a token.
                health: dict[str, Any] = {key: status[key] for key in ("ready", "model_state")
                                          if key in status}
                response: tuple[int, dict[str, Any]] = (HTTPStatus.OK, health)
            except Exception:
                response = (HTTPStatus.SERVICE_UNAVAILABLE, {"ready": False})
            self._health_cache = (time.monotonic() + HEALTH_CACHE_SECONDS, response)
            return response

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(API_SOCKET_TIMEOUT_SECONDS)
        return request, client_address

    def process_request(self, request, client_address):
        if not self._client_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._client_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._client_slots.release()


# Вкладки панели владельца и допустимые параметры их запросов.
ADMIN_VIEWS: dict[str, set[str]] = {
    "/admin/sources": set(), "/admin/languages": set(), "/admin/learning": set(),
    "/admin/analyses": {"q", "state", "mode", "from", "to", "visitor", "with_signals", "with_technologies",
                        "country", "source", "sort", "order", "offset", "limit"},
    "/admin/compare": {"ids"},
    "/admin/publications": {"run_id", "decision", "offset", "limit"},
}
# Действия владельца, которые исполняют инструменты панели (значения — в `values`).
OWNER_ACTIONS = frozenset({"sources", "languages", "learning", "retrain", "reset_learning", "feedback"})


class ApiHandler(BaseHTTPRequestHandler):
    server: ApiServer
    protocol_version = "HTTP/1.1"
    server_version = "Trendanalyser"
    sys_version = ""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: int, payload: object, *, instance_id: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8", "backslashreplace")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if instance_id is not None:
            self.send_header("X-Trend-Instance", instance_id)
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def _local_request(self) -> bool:
        # Loopback binding alone does not protect against DNS rebinding from a
        # browser visiting another site. The UI's server-side client uses this
        # exact local host; a foreign Origin cannot submit JSON mutations.
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            return False
        host = hosts[0].lower()
        if host not in {f"127.0.0.1:{self.server.server_port}",
                        f"localhost:{self.server.server_port}"}:
            return False
        # Streamlit calls the API from its server process. Browsers mark even
        # image/navigation GETs with Fetch Metadata although they may omit
        # Origin; those requests must not trigger private API work.
        return not (self.headers.get_all("Origin", [])
                    or self.headers.get_all("Sec-Fetch-Site", []))

    def _authorized(self) -> bool:
        token = self.server.api_token
        if token is None:
            return True
        supplied = self.headers.get_all("X-Trend-API-Token", [])
        return (len(supplied) == 1 and len(supplied[0]) <= MAX_API_TOKEN_CHARACTERS
                and supplied[0].isascii() and hmac.compare_digest(supplied[0], token))

    def _admin(self) -> bool:
        token = getattr(self.server, "admin_token", None)
        supplied = self.headers.get_all("X-Trend-Admin-Token", [])
        return (token is not None and len(supplied) == 1 and len(supplied[0]) <= MAX_API_TOKEN_CHARACTERS
                and supplied[0].isascii() and hmac.compare_digest(supplied[0], token))

    def _optional_visitor(self) -> str | None:
        """Посетитель отметки присутствия: на экране входа его ещё нет."""
        values = self.headers.get_all("X-Trend-Visitor", [])
        if not values:
            return None
        if len(values) != 1 or VISITOR.fullmatch(values[0]) is None:
            raise WebApiError(HTTPStatus.FORBIDDEN, "forbidden")
        return values[0]

    def _admin_get(self) -> None:
        if not self._admin():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        try:
            names = {"after", "person"} if "person=" in self.path else {"after"}
            parameters = self._page_parameters(names, "invalid_request")
            if (re.fullmatch(r"[0-9]{1,9}", parameters["after"]) is None
                    or re.fullmatch(r"[0-9a-f]{12}", parameters.get("person", "0" * 12)) is None):
                raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
            self._json(HTTPStatus.OK, self.server.analysis.admin_overview(int(parameters["after"]),
                                                                          parameters.get("person")))
        except WebApiError as error:
            self._json(error.status, {"error": error.message})
        except Exception:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Не удалось собрать сводку сервиса."})

    def _admin_view(self) -> None:
        """Вкладки панели владельца: источники, языки, обучение, анализы, сравнение, публикации."""
        if not self._admin():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        route = self.path.split("?", 1)[0]
        analysis = self.server.analysis
        try:
            target = urlsplit(self.path)
            if len(self.path) > 2048 or target.fragment:
                raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
            try:
                pairs = parse_qsl(target.query, keep_blank_values=True, strict_parsing=bool(target.query),
                                  max_num_fields=20)
            except ValueError:
                raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request") from None
            parameters = dict(pairs)
            if len(parameters) != len(pairs) or not set(parameters) <= ADMIN_VIEWS[route]:
                raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
            if route == "/admin/sources":
                answer = analysis.admin_sources()
            elif route == "/admin/languages":
                answer = analysis.admin_languages()
            elif route == "/admin/learning":
                answer = analysis.admin_learning()
            elif route == "/admin/analyses":
                answer = analysis.admin_analyses(parameters)
            elif route == "/admin/compare":
                answer = analysis.admin_compare([item for item in parameters.get("ids", "").split(",") if item])
            else:
                answer = analysis.admin_publications(parameters)
            self._json(HTTPStatus.OK, answer)
        except WebApiError as error:
            self._json(error.status, {"error": error.message})
        except Exception:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Не удалось собрать данные панели."})

    def _request_json(self) -> object:
        if (self.headers.get_all("Transfer-Encoding")
                or len(self.headers.get_all("Content-Type", [])) != 1
                or self.headers.get_content_type() != "application/json"):
            raise WebApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "invalid_content_type")
        lengths = self.headers.get_all("Content-Length", [])
        raw_length = lengths[0] if len(lengths) == 1 else ""
        if not raw_length.isdigit() or len(raw_length) > 6 or not 0 < int(raw_length) <= MAX_REQUEST_BYTES:
            raise WebApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "invalid_size")
        try:
            raw = self.rfile.read(int(raw_length))
            return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object,
                              parse_constant=_invalid_json_constant)
        except TimeoutError:
            raise WebApiError(HTTPStatus.REQUEST_TIMEOUT, "request_timeout") from None
        except (ValueError, UnicodeError, RecursionError):
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request") from None

    def _page_parameters(self, names: set[str], message: str) -> dict[str, str]:
        """Exactly the named query parameters, each once, or a bounded refusal."""
        target = urlsplit(self.path)
        if len(self.path) > 512 or target.fragment:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, message)
        try:
            pairs = parse_qsl(target.query, keep_blank_values=True, strict_parsing=True,
                              max_num_fields=len(names))
        except ValueError:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, message) from None
        parameters = dict(pairs)
        if len(pairs) != len(names) or set(parameters) != names:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, message)
        return parameters

    def _as_visitor(self, method: Any, *args: Any) -> Any:
        """Call the service for this request's visitor when visitors are kept apart.

        Посетителя называет веб-интерфейс, проверивший его пароль; без этого
        заголовка разделённый API не отвечает ничем.
        """
        if not getattr(self.server.analysis, "separate_visitors", False):
            return method(*args)
        values = self.headers.get_all("X-Trend-Visitor", [])
        if len(values) != 1 or VISITOR.fullmatch(values[0]) is None:
            raise WebApiError(HTTPStatus.FORBIDDEN, "forbidden")
        refusal = getattr(self.server.analysis, "visitor_refusal", lambda _visitor: None)(values[0])
        if refusal is not None:
            raise WebApiError(HTTPStatus.FORBIDDEN, refusal)
        return method(*args, visitor=values[0])

    def _language_bound(self, method: Any) -> Any:
        """Язык перевода ТОПа, выбранный посетителем (передаётся, только если выбран)."""
        values = self.headers.get_all("X-Trend-Language", [])
        if not values:
            return method
        if len(values) != 1 or re.fullmatch(r"[a-z]{2,3}", values[0]) is None:
            raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
        language = values[0]
        return lambda *args, **kwargs: method(*args, language=language, **kwargs)

    def _status_json(self, status: dict[str, Any]) -> None:
        """Send an analysis state; a result the page already holds is not sent again."""
        versions = self.headers.get_all("X-Trend-Result-Version", [])
        if (status.get("state") == "succeeded" and len(versions) == 1
                and len(versions[0]) == 64 and versions[0] == status.get("result_version")):
            status = {key: value for key, value in status.items() if key != "result"}
        self._json(HTTPStatus.OK, status)

    def do_GET(self) -> None:  # noqa: N802
        if not self._local_request():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        if self.path.split("?", 1)[0] == "/admin/overview":
            self._admin_get()
            return
        if self.path.split("?", 1)[0] in ADMIN_VIEWS:
            self._admin_view()
            return
        if self.path.split("?", 1)[0] == "/admin/run":
            if not self._admin():
                self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
                return
            try:
                parameters = self._page_parameters({"run_id"}, "Некорректный анализ.")
                self._json(HTTPStatus.OK, self.server.analysis.admin_run(parameters["run_id"]))
            except WebApiError as error:
                self._json(error.status, {"error": error.message})
            except Exception:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Не удалось собрать результат анализа."})
            return
        if self.path != "/health" and not self._authorized():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        publication_route = self.path.split("?", 1)[0] == "/analyses/current/publications"
        history_route = self.path.split("?", 1)[0] == "/analyses/history"
        saved_route = self.path.split("?", 1)[0] == "/analyses/saved"
        if (self.path not in {"/health", "/analyses/current", "/analyses/estimates"}
                and not publication_route and not history_route and not saved_route):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            if self.path == "/health":
                health_status, health = self.server.health()
                self._json(health_status, health,
                           instance_id=self.server.instance_id if health_status == HTTPStatus.OK else None)
            elif self.path == "/analyses/estimates":
                self._json(HTTPStatus.OK, self.server.analysis.estimates())
            elif publication_route:
                parameters = self._page_parameters({"run_id", "offset", "limit"},
                                                   "Некорректная страница публикаций.")
                if (re.fullmatch(r"[A-Za-z0-9_-]{1,128}", parameters["run_id"]) is None
                        or re.fullmatch(r"[0-9]{1,7}", parameters["offset"]) is None
                        or re.fullmatch(r"[0-9]{1,3}", parameters["limit"]) is None):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректная страница публикаций.")
                self._json(HTTPStatus.OK, self._as_visitor(
                    self.server.analysis.publications_page,
                    parameters["run_id"], int(parameters["offset"]), int(parameters["limit"])))
            elif history_route:
                parameters = self._page_parameters({"offset", "limit"}, "Некорректная страница истории.")
                if (re.fullmatch(r"[0-9]{1,7}", parameters["offset"]) is None
                        or re.fullmatch(r"[0-9]{1,2}", parameters["limit"]) is None):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректная страница истории.")
                self._json(HTTPStatus.OK, self._as_visitor(
                    self.server.analysis.history, int(parameters["offset"]), int(parameters["limit"])))
            elif saved_route:
                parameters = self._page_parameters({"run_id"}, "Некорректный анализ.")
                if RUN_ID.fullmatch(parameters["run_id"]) is None:
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Некорректный анализ.")
                self._status_json(self._as_visitor(self._language_bound(self.server.analysis.saved_analysis),
                                                   parameters["run_id"]))
            else:
                self._status_json(self._as_visitor(self._language_bound(self.server.analysis.current_analysis)))
        except WebApiError as error:
            self._json(error.status, {"error": error.message})
        except Exception:
            if self.path == "/health":
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"ready": False})
            elif self.path == "/analyses/estimates":
                self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                           {"error": "Не удалось рассчитать прогноз времени."})
            elif publication_route:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                           {"error": "Не удалось прочитать страницу публикаций."})
            elif history_route:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                           {"error": "Не удалось прочитать историю анализов."})
            elif saved_route:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                           {"error": "Не удалось открыть сохранённый анализ."})
            else:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                           {"error": "Не удалось прочитать состояние анализа."})

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_request():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        admin_route = self.path in {"/admin/cancel", "/admin/action"}
        if not (self._admin() if admin_route else self._authorized()):
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        if self.path not in {"/analyze", "/analyses", "/analyses/cancel", "/presence", "/access",
                             "/admin/cancel", "/admin/action"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            payload = self._request_json()
            if self.path == "/presence":
                optional = ("event", "agent", "ip", "view", "detail")
                if (not isinstance(payload, dict) or not {"session", "page"} <= set(payload)
                        or not set(payload) <= {"session", "page", *optional}
                        or not all(isinstance(payload[name], str) for name in ("session", "page"))
                        or not all(isinstance(payload.get(name), str | None) for name in optional)):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
                self._json(HTTPStatus.OK, self.server.analysis.presence(
                    payload["session"], payload["page"], payload.get("event"), payload.get("agent"),
                    payload.get("ip"), view=payload.get("view"), detail=payload.get("detail"),
                    visitor=self._optional_visitor()))
                return
            if self.path == "/access":
                if not isinstance(payload, dict) or set(payload) != {"ip"} or not isinstance(payload["ip"], str | None):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
                self._json(HTTPStatus.OK, self.server.analysis.access(payload["ip"], visitor=self._optional_visitor()))
                return
            if self.path == "/admin/action":
                person, text, block = (payload.get(name) if isinstance(payload, dict) else None
                                       for name in ("person", "text", "id"))
                if (not isinstance(payload, dict) or not isinstance(payload.get("action"), str)
                        or not set(payload) <= {"action", "person", "text", "id", "values"}
                        or payload["action"] in {"resources", "sources", "languages", "learning", "feedback"}
                        and not isinstance(payload.get("values"), dict)
                        or payload.get("values") is not None and not isinstance(payload["values"], dict)
                        or person is not None and (not isinstance(person, str) or re.fullmatch(r"[0-9a-f]{12}", person) is None)
                        or block is not None and (not isinstance(block, str) or re.fullmatch(r"[A-Za-z0-9_-]{8,32}", block) is None)
                        or text is not None and (not isinstance(text, str) or len(text) > MAX_MESSAGE_CHARACTERS)
                        or payload["action"] in {"block", "sign_out", "message"} and person is None
                        or payload["action"] == "message" and not (text or "").strip()
                        or payload["action"] == "unblock" and block is None):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
                self._json(HTTPStatus.OK, self.server.analysis.admin_action(payload))
                return
            if self.path == "/admin/cancel":
                if (not isinstance(payload, dict) or set(payload) != {"id"} or not isinstance(payload["id"], str)
                        or RUN_ID.fullmatch(payload["id"]) is None):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
                self._json(HTTPStatus.OK, self.server.analysis.admin_cancel(payload["id"]))
                return
            if self.path == "/analyses/cancel":
                if (not isinstance(payload, dict) or set(payload) != {"id"}
                        or not isinstance(payload["id"], str)
                        or not 0 < len(payload["id"]) <= 128):
                    raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
                self._json(HTTPStatus.OK, self._as_visitor(self.server.analysis.cancel_analysis, payload["id"]))
                return
            if (not isinstance(payload, dict) or not set(payload) <= {"query", "mode"}
                    or "query" not in payload or not isinstance(payload["query"], str)
                    or not isinstance(payload.get("mode", "fast"), str)):
                raise WebApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request")
            if self.path == "/analyses":
                started = self._as_visitor(self.server.analysis.start_analysis,
                                           payload["query"], payload.get("mode", "fast"))
                self._json(HTTPStatus.ACCEPTED, started)
            else:
                finished = self.server.analysis.analyze(payload["query"], payload.get("mode", "fast"))
                self._json(HTTPStatus.OK, finished)
        except WebApiError as error:
            self._json(error.status, {"error": error.message})
        except Exception:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": "Не удалось обработать анализ. Проверьте локальный журнал."})


def serve(data_dir: Path | None = None, *, port: int = 8000) -> None:
    server = ApiServer(("127.0.0.1", port), ApiHandler)
    try:
        analysis = WebAnalysisService(data_dir)
    except BaseException:
        server.server_close()
        raise
    server.analysis = analysis
    try:
        print(f"Внутренний API готов: http://127.0.0.1:{port}", flush=True)
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        analysis.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--port", type=int, default=8000, choices=range(1024, 65536), metavar="PORT")
    args = parser.parse_args(argv)
    serve(args.data_dir, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
