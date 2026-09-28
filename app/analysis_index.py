"""Анализ анализов: сводка каждого совершённого анализа для вкладки панели владельца.

Сводка собирается из сохранённого результата без чтения архива документов:
числа (публикации, отсеянные как не по теме, тренды, технологии, источники),
главные тренды и технологии, страны материалов, правило источников и версия
обученной модели. Сводки кешируются в профиле (`analysis-index.json`) и
пересчитываются, только когда у анализа меняется время обновления.

Поверх сводок — фильтры, сортировка, общая статистика (что чаще всего ищут,
какие технологии повторяются между анализами, какие источники дают материал)
и сравнение нескольких анализов.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
import json
import os
from pathlib import Path
import re
from threading import Lock
from time import monotonic
from typing import Any

INDEX_FILE = "analysis-index.json"
SUMMARY_SECONDS = 3.0
INDEX_VERSION = 2
MAX_RUNS = 5000
SIGNAL_CATEGORIES = frozenset({"confirmed_trend", "early_signal", "weak_signal_candidate", "emerging_candidate"})
SORTS = {"created_at", "duration", "publications", "signals", "technologies", "off_topic_share", "query"}
STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted", "queued", "running"})


def _mapping(value: object) -> dict[str, Any]:
    """Словарь из разобранного JSON или пустой словарь."""
    return value if isinstance(value, dict) else {}


def _items(value: object) -> list[Any]:
    """Список из разобранного JSON или пустой список."""
    return value if isinstance(value, list) else []


def _number(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def result_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Короткая сводка сохранённого результата анализа (без архива документов)."""
    result = _mapping(payload.get("result"))
    cards = _items(result.get("cards"))
    by_id = {}
    for card in cards:
        candidate = card.get("candidate") if isinstance(card, dict) else None
        if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str):
            by_id[candidate["candidate_id"]] = card
    signals = []
    for identifier in result.get("top_trend_ids") or []:
        card = by_id.get(identifier)
        if not isinstance(card, dict) or card.get("category") not in SIGNAL_CATEGORIES:
            continue
        label = card["candidate"].get("label")
        if isinstance(label, str) and label.strip():
            signals.append({"title": label.strip()[:200], "category": card["category"]})
    radar_payload = _mapping(payload.get("radar"))
    radar = _mapping(radar_payload.get("result"))
    technologies = [{"title": str(item.get("title"))[:200], "probability": item.get("probability")}
                    for item in radar.get("technologies", []) if isinstance(item, dict) and item.get("title")]
    sources: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    approved = _mapping(payload.get("approved_sources"))
    for observation in approved.get("observations") or []:
        if isinstance(observation, dict):
            sources[str(observation.get("source_id"))] += 1
            if isinstance(observation.get("country"), str):
                countries[observation["country"]] += 1
    skipped = [item.get("source_id") for item in approved.get("coverage") or []
               if isinstance(item, dict) and item.get("reason_code") in {"country_filter", "disabled_by_owner"}]
    discovery = _mapping(payload.get("discovery_summary"))
    relevance = _mapping(payload.get("publication_relevance"))
    counts = _mapping(relevance.get("counts"))
    on_topic = (_number(counts.get("relevant")) or 0) + (_number(counts.get("weak")) or 0)
    off_topic = _number(counts.get("off_topic"))
    publications = on_topic if counts else ((_number(discovery.get("unique_studies")) or 0) + sum(sources.values()))
    total = publications + (off_topic or 0)
    plan = _mapping(result.get("query_plan"))
    policy = payload.get("source_policy") if isinstance(payload.get("source_policy"), dict) else None
    return {
        "english_query": plan.get("english_query") if isinstance(plan.get("english_query"), str) else None,
        "publications": publications, "off_topic": off_topic,
        "off_topic_share": round((off_topic or 0) / total, 4) if total and off_topic is not None else None,
        "relevant": _number(counts.get("relevant")), "signals": len(signals), "signal_titles": signals[:10],
        "technologies": len(technologies), "technology_titles": technologies[:15],
        "sources": dict(sources.most_common()), "countries": dict(countries.most_common()),
        "skipped_sources": [item for item in skipped if isinstance(item, str)],
        "source_countries": list(policy.get("countries") or []) if policy else [],
        "relevance_model": relevance.get("model_version") if isinstance(relevance.get("model_version"), str) else None,
        "semantic_relevance": bool(relevance.get("semantic")) if relevance else None,
        "quality": result.get("quality") if isinstance(result.get("quality"), str) else None,
    }


def _seconds(start: object, end: object) -> int | None:
    try:
        value = (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds()
    except ValueError:
        return None
    return round(value) if 0 <= value <= 7 * 24 * 3600 else None


def _normal_query(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))[:200]


class AnalysisIndex:
    """Сводки анализов профиля с кешем на диске; один построитель за раз."""

    def __init__(self, data_dir: Path | None):
        self.path = Path(data_dir) / INDEX_FILE if data_dir is not None else None
        self._lock = Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict) and raw.get("version") == INDEX_VERSION and isinstance(raw.get("entries"), dict):
            self._entries = {key: value for key, value in raw["entries"].items()
                             if isinstance(key, str) and isinstance(value, dict)}

    def _save(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps({"version": INDEX_VERSION, "entries": self._entries},
                                            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def refresh(self, rows: Iterable[Mapping[str, Any]], entry: Callable[[Mapping[str, Any]], dict[str, Any] | None],
                summary: Callable[[str], dict[str, Any] | None], *, budget: int = 40,
                seconds: float = SUMMARY_SECONDS) -> dict[str, Any]:
        """Обновить сводки по строкам истории.

        Тяжёлых чтений результатов — не больше `budget` и не дольше `seconds` за
        раз: панель ждёт ответа несколько секунд и дочитывает остальное следующим
        запросом.
        """
        deadline = monotonic() + seconds
        with self._lock:
            self._load()
            current: dict[str, dict[str, Any]] = {}
            pending = 0
            changed = False
            for row in rows:
                base = entry(row)
                if base is None:
                    continue
                run_id = base["id"]
                known = self._entries.get(run_id)
                stamp = str(row.get("updated_at") or "")
                record = {**base, "updated_at": stamp,
                          "duration_seconds": _seconds(row.get("created_at"), row.get("updated_at"))
                          if base["state"] not in {"queued", "running"} else None}
                if isinstance(row.get("error"), str) and base["state"] in {"failed", "interrupted"}:
                    record["error"] = row["error"][:300]
                if known is not None and known.get("updated_at") == stamp and "summary" in known:
                    record["summary"] = known["summary"]
                elif base["state"] == "succeeded":
                    if budget > 0 and monotonic() < deadline:
                        budget -= 1
                        try:
                            record["summary"] = summary(run_id)
                        except Exception:
                            record["summary"] = None
                        changed = True
                    else:
                        pending += 1
                if known != record:
                    changed = True
                current[run_id] = record
            if set(current) != set(self._entries):
                changed = True
            self._entries = current
            if changed:
                self._save()
            return {"indexed": sum(1 for item in current.values() if "summary" in item), "pending": pending,
                    "total": len(current)}

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(item) for item in self._entries.values()]


def _matches(item: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    summary = item.get("summary") or {}
    text = filters.get("q")
    if text:
        haystack = " ".join(str(value) for value in (
            item.get("query"), summary.get("english_query"),
            *(entry.get("title") for entry in summary.get("signal_titles", [])),
            *(entry.get("title") for entry in summary.get("technology_titles", [])))).casefold()
        if not all(part in haystack for part in str(text).casefold().split()):
            return False
    for name in ("state", "mode"):
        if filters.get(name) and item.get(name) != filters[name]:
            return False
    created = str(item.get("created_at") or "")[:10]
    if filters.get("from") and created < filters["from"]:
        return False
    if filters.get("to") and created > filters["to"]:
        return False
    if filters.get("visitor") and item.get("visitor") != filters["visitor"]:
        return False
    if filters.get("with_signals") and not summary.get("signals"):
        return False
    if filters.get("with_technologies") and not summary.get("technologies"):
        return False
    country = filters.get("country")
    if country and country not in (summary.get("countries") or {}) and country not in (
            summary.get("source_countries") or []):
        return False
    source = filters.get("source")
    if source and source not in (summary.get("sources") or {}):
        return False
    return True


def _sort_key(name: str) -> Callable[[Mapping[str, Any]], Any]:
    def key(item: Mapping[str, Any]) -> Any:
        summary = item.get("summary") or {}
        if name == "created_at":
            return str(item.get("created_at") or "")
        if name == "duration":
            return item.get("duration_seconds") or 0
        if name == "query":
            return str(item.get("query") or "").casefold()
        value = summary.get(name)
        return value if isinstance(value, (int, float)) else -1
    return key


def query_entries(entries: Sequence[Mapping[str, Any]], *, filters: Mapping[str, Any], sort: str = "created_at",
                  descending: bool = True, offset: int = 0, limit: int = 50) -> dict[str, Any]:
    if sort not in SORTS:
        raise ValueError("sort")
    matched = [dict(item) for item in entries if _matches(item, filters)]
    matched.sort(key=_sort_key(sort), reverse=descending)
    return {"total": len(matched), "offset": offset, "runs": matched[offset:offset + limit],
            "statistics": statistics(matched)}


def statistics(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Общая картина по выбранным анализам."""
    states = Counter(str(item.get("state")) for item in entries)
    modes = Counter(str(item.get("mode") or "—") for item in entries)
    durations: dict[str, list[int]] = {}
    queries: Counter[str] = Counter()
    labels: dict[str, str] = {}
    technologies: Counter[str] = Counter()
    signals: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    countries: Counter[str] = Counter()
    days: Counter[str] = Counter()
    off_topic: list[float] = []
    publications = 0
    for item in entries:
        summary = item.get("summary") or {}
        if item.get("state") == "succeeded" and isinstance(item.get("duration_seconds"), int):
            durations.setdefault(str(item.get("mode") or "—"), []).append(item["duration_seconds"])
        key = _normal_query(str(item.get("query") or ""))
        if key:
            queries[key] += 1
            labels.setdefault(key, str(item.get("query"))[:200])
        technologies.update({entry["title"] for entry in summary.get("technology_titles", [])
                             if isinstance(entry, dict) and entry.get("title")})
        signals.update({entry["title"] for entry in summary.get("signal_titles", [])
                        if isinstance(entry, dict) and entry.get("title")})
        sources.update({name: count for name, count in (summary.get("sources") or {}).items()
                        if isinstance(count, int)})
        countries.update({name: count for name, count in (summary.get("countries") or {}).items()
                          if isinstance(count, int)})
        if isinstance(summary.get("off_topic_share"), (int, float)):
            off_topic.append(float(summary["off_topic_share"]))
        publications += summary.get("publications") or 0
        day = str(item.get("created_at") or "")[:10]
        if day:
            days[day] += 1
    today = datetime.now(UTC).date()
    activity = [{"day": (today - timedelta(days=offset)).isoformat(),
                 "runs": days.get((today - timedelta(days=offset)).isoformat(), 0)} for offset in range(29, -1, -1)]
    return {
        "runs": len(entries), "states": dict(states), "modes": dict(modes), "publications": publications,
        "median_seconds": {mode: sorted(values)[len(values) // 2] for mode, values in durations.items() if values},
        "average_off_topic_share": round(sum(off_topic) / len(off_topic), 4) if off_topic else None,
        "top_queries": [{"query": labels[key], "runs": count} for key, count in queries.most_common(10)],
        "recurring_technologies": [{"title": title, "runs": count} for title, count in technologies.most_common(15)
                                   if count > 1],
        "recurring_signals": [{"title": title, "runs": count} for title, count in signals.most_common(15)
                              if count > 1],
        "sources": dict(sources.most_common()), "countries": dict(countries.most_common()),
        "activity": activity,
    }


def compare(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Сравнение 2–6 анализов: числа рядом, общие и собственные тренды и технологии."""
    columns = []
    technology_sets, signal_sets = [], []
    for item in entries:
        summary = item.get("summary") or {}
        technologies = {entry["title"] for entry in summary.get("technology_titles", []) if isinstance(entry, dict)}
        signals = {entry["title"] for entry in summary.get("signal_titles", []) if isinstance(entry, dict)}
        technology_sets.append(technologies)
        signal_sets.append(signals)
        columns.append({"id": item.get("id"), "query": item.get("query"), "mode": item.get("mode"),
                        "state": item.get("state"), "created_at": item.get("created_at"),
                        "duration_seconds": item.get("duration_seconds"),
                        "numbers": {name: summary.get(name) for name in (
                            "publications", "off_topic", "off_topic_share", "signals", "technologies")},
                        "sources": summary.get("sources") or {}, "countries": summary.get("countries") or {}})
    common_technologies = set.intersection(*technology_sets) if technology_sets else set()
    common_signals = set.intersection(*signal_sets) if signal_sets else set()
    for column, technologies, signals in zip(columns, technology_sets, signal_sets, strict=True):
        column["own_technologies"] = sorted(technologies - common_technologies)[:15]
        column["own_signals"] = sorted(signals - common_signals)[:15]
    return {"analyses": columns, "common_technologies": sorted(common_technologies),
            "common_signals": sorted(common_signals)}


def valid_filters(parameters: Mapping[str, str]) -> dict[str, Any]:
    """Фильтры вкладки «Анализы» из строки запроса; недопустимое значение — ошибка."""
    filters: dict[str, Any] = {}
    text = parameters.get("q", "")
    if len(text) > 200:
        raise ValueError("q")
    if text.strip():
        filters["q"] = text.strip()
    state = parameters.get("state", "")
    if state:
        if state not in STATES:
            raise ValueError("state")
        filters["state"] = state
    mode = parameters.get("mode", "")
    if mode:
        if mode not in {"fast", "deep"}:
            raise ValueError("mode")
        filters["mode"] = mode
    for name in ("from", "to"):
        value = parameters.get(name, "")
        if value:
            date.fromisoformat(value)
            filters[name] = value
    for name in ("with_signals", "with_technologies"):
        if parameters.get(name) == "1":
            filters[name] = True
    for name, pattern in (("country", r"(?:[A-Z]{2}|INT)"), ("source", r"[a-z][a-z0-9_]{0,39}")):
        value = parameters.get(name, "")
        if value:
            if re.fullmatch(pattern, value) is None:
                raise ValueError(name)
            filters[name] = value
    visitor = parameters.get("visitor", "")
    if visitor:
        if len(visitor) > 80:
            raise ValueError("visitor")
        filters["visitor"] = visitor
    return filters
