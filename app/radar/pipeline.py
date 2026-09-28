"""Радар технологий: от собранной выдачи к ТОП-15 и журналу исключений.

1. Кандидаты — устойчивые фразы из названий найденных материалов.
2. Для каждого — помесячная история и кривая эксперта, первое упоминание и
   объём за всё время, сведения из выдачи.
3. Паспорт в схеме датасета оценивается моделью этапа 1.
4. ТОП-15 — подтверждённые слабые сигналы, затем проверенные молодые
   технологии с честно показанной более низкой вероятностью модели.
   Зрелые темы, спад и материалы без достаточной истории исключаются.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import os
from datetime import date
from pathlib import Path
from threading import Event, Lock
from time import monotonic
from typing import Any, TypeVar

import httpx

from app.radar.evidence import FirstMention, first_mention, openalex_years, pool_evidence, source_view
from app.radar.passport import (TOP_PROBABILITY, build_passport, decide, excluded_by_rule, exclusion_reasons,
                                is_weak_signal, quick_exclusion, yearly_growth)
from app.radar.phrases import Candidate, mine_candidates, variants
from app.signal_model.model import HIGH_CONFIDENCE, SignalModel
from app.trend_confidence import SOURCE_RULES, assess_curve
from app.trend_history import ArxivBatcher, HistoryFetchError, TechnologyHistory, collect_history, month_window

POLICY_VERSION = "radar/1.3.0"
# Владелец может потребовать первое упоминание в arXiv по одной фразе, как считает
# сам arXiv, — медленнее, но без приближения, когда OpenAlex не ответил.
EXACT_VARIABLE = "TRENDANALIZER_RADAR_EXACT"
TOP_SIZE = 15
# Воронка: быстрый отсев по годам OpenAlex для многих кандидатов,
# помесячная история — только для самых растущих.
MAX_CANDIDATES = 60
MAX_DEEP = 40
WORKERS = 5
TOP_SOURCE_LIMIT = 30
EXCLUDED_SOURCE_LIMIT = 10
NOT_CHECKED = "Не проверен подробно: рост в OpenAlex ниже, чем у проверенных кандидатов"
BELOW_TOP = f"Не вошёл в первые {TOP_SIZE} по итоговой оценке"
MODEL_PATH = Path(__file__).resolve().parents[1] / "signal_model/model.json"

Progress = Callable[[int, int], None]
T = TypeVar("T")


class PhraseCache:
    """Ответы источников по фразе на дату среза, общие для анализов одного процесса.

    Повторный и соседний запросы проверяют во многом те же фразы; их годовые
    подсчёты и помесячная история за несколько часов не меняются. Неполный
    ответ (отказ, отмена) не запоминается: следующий анализ спросит заново.
    """

    def __init__(self, seconds: float = 6 * 3600, limit: int = 2000):
        self.seconds, self.limit = seconds, limit
        self._items: dict[Hashable, tuple[float, Any]] = {}
        self._lock = Lock()

    def peek(self, key: Hashable) -> Any:
        """Сохранённый ответ без вычисления; None — нет или устарел."""
        with self._lock:
            hit = self._items.get(key)
            return hit[1] if hit is not None and monotonic() - hit[0] < self.seconds else None

    def get(self, key: Hashable, compute: Callable[[], T], keep: Callable[[T], bool]) -> T:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and monotonic() - hit[0] < self.seconds:
                return hit[1]
        value = compute()
        if keep(value):
            with self._lock:
                self._items.pop(key, None)
                self._items[key] = (monotonic(), value)
                while len(self._items) > self.limit:
                    del self._items[next(iter(self._items))]
        return value


# Кэш процесса; радар пользуется им, только когда его передали явно.
SHARED_CACHE = PhraseCache()


def _history(forms: Sequence[str], as_of: date, months: int, client: httpx.Client, cancel: Event,
             cache: PhraseCache | None, arxiv: ArxivBatcher | None = None) -> TechnologyHistory:
    def fetch() -> TechnologyHistory:
        options: dict[str, Any] = {"arxiv": arxiv} if arxiv is not None else {}
        return collect_history(forms, as_of, months=months, client=client, cancel=cancel, **options)

    if cache is None:
        return fetch()
    return cache.get(("history", tuple(forms), as_of, months), fetch,
                     lambda history: all(item.state != "unavailable" for item in history.coverage))


def _years(client: httpx.Client, phrase: str, cancel: Event, openalex_key: str | None, as_of: date,
           cache: PhraseCache | None) -> tuple[tuple[int, int], ...] | None:
    def fetch() -> tuple[tuple[int, int], ...] | None:
        return openalex_years(client, phrase, cancel, openalex_key)

    if cache is None:
        return fetch()
    return cache.get(("years", phrase, as_of), fetch, lambda years: years is not None)


def _evaluate(candidate: Candidate, pool: Sequence[Mapping[str, Any]], area: str, as_of: date,
              model: SignalModel, client: httpx.Client, cancel: Event, months: int,
              openalex_key: str | None = None, years: object = None, *, deep: bool = True,
              forced_reason: str | None = None, cache: PhraseCache | None = None,
              arxiv: ArxivBatcher | None = None) -> dict[str, Any]:
    """Оценка кандидата; без `deep` — без помесячной истории, для журнала исключений."""
    forms = variants(candidate.surface)
    if deep:
        history = _history(forms, as_of, months, client, cancel, cache, arxiv)
    else:
        history = TechnologyHistory(forms, as_of, month_window(as_of, months), (), ())
    curve = assess_curve(history.materials(), as_of, months=months, coverage_complete=history.coverage_complete)
    batched = {"arxiv": arxiv} if arxiv is not None and years is None else {}
    mention = (first_mention(client, forms, cancel, openalex_key, years, **batched) if years is not None or deep
               else FirstMention(None, None, None))
    evidence = pool_evidence(candidate.surface, pool, source_limit=TOP_SOURCE_LIMIT)
    earliest = min((date.fromisoformat(record.published) for record in history.records), default=None)
    passport = build_passport(candidate.surface, area, as_of, curve, mention, evidence, earliest)
    prediction = decide(model, passport)
    signal = forced_reason is None and is_weak_signal(passport, prediction, curve if deep else None)
    rule = forced_reason is not None or excluded_by_rule(passport, curve if deep else None)
    recent = sorted(history.records,
                    key=lambda record: (SOURCE_RULES[record.source_id].coefficient, record.published),
                    reverse=True)[:TOP_SOURCE_LIMIT]
    sources = list(evidence.items) + [source_view(record.source_id, record.title, record.url, record.published)
                                      for record in recent]
    # A preprint and its journal publication can have different URLs. Keep the
    # strongest, newest view of a work, then bound every technology's response.
    from app.trend_history import work_key

    sources.sort(key=lambda item: (item.trust == "высокий", item.published, item.url), reverse=True)
    seen_urls: set[str] = set()
    seen_works: set[str] = set()
    unique_sources = []
    for item in sources:
        work = work_key(item.title)
        if item.url in seen_urls or work in seen_works:
            continue
        seen_urls.add(item.url)
        seen_works.add(work)
        unique_sources.append(item)
        if len(unique_sources) == TOP_SOURCE_LIMIT:
            break
    reasons = [] if signal else _reasons(forced_reason, exclusion_reasons(passport, curve, prediction), deep)
    if not signal and not rule and prediction.probability <= TOP_PROBABILITY:
        reasons.insert(0, f"Уверенность модели не превышает порог слабого сигнала: "
                          f"{round(prediction.probability * 100)}% (нужно больше 60%).")
    return {
        "phrase": candidate.phrase, "title": candidate.surface, "probability": prediction.probability,
        "score": rank_score(prediction.probability, curve.confidence),
        "is_signal": signal, "rule_excluded": rule, "pool_documents": len(candidate.documents),
        "curve": {"confidence": curve.confidence, "trend": curve.trend, "materials": curve.materials,
                  "quadratic_r2": curve.quadratic_r2, "source_classes": list(curve.source_classes),
                  "coverage_complete": curve.coverage_complete,
                  "months": [asdict(point) for point in curve.months],
                  "checks": [asdict(check) for check in curve.checks]},
        "passport": {"stage": passport.technology.stage, "trend": passport.technology.trend,
                     "rationale": passport.technology.rationale, "companies": passport.technology.companies,
                     "first_year": passport.first_year, "all_time": passport.all_time, "mature": passport.mature,
                     "volume_basis": passport.volume_basis,
                     "years": [list(item) for item in passport.years if item[0] >= as_of.year - 15]},
        "predictors": [{"label": part.label, "value": part.value, "weight": part.weight}
                       for part in prediction.contributions[:6]],
        "sources": [asdict(item) for item in unique_sources],
        "description": evidence.description, "advantage": evidence.advantage, "case": evidence.case,
        "coverage": [asdict(item) for item in history.coverage],
        "reasons": reasons,
    }


def _reasons(forced: str | None, measured: Sequence[str], deep: bool) -> list[str]:
    """Причина быстрого отсева не повторяется выводом паспорта; без истории нечего говорить о её объёме."""
    skipped = ("Зрелая тема", "Давно известное") if forced else ()
    kept = [reason for reason in measured if not reason.startswith(skipped)
            and (deep or not reason.startswith("Мало работ"))]
    return list(dict.fromkeys(([forced] if forced else []) + kept))


def rank_score(probability: float, curve_confidence: int | None) -> float:
    """Порядок ТОПа: вероятность модели, уточнённая уверенностью по кривой эксперта.

    Кривая без оценки (мало данных) считается нейтральной, а не нулевой.
    """
    curve = 0.5 if curve_confidence is None else curve_confidence / 100
    return round(probability * (0.6 + 0.4 * curve), 4)


def build_radar(pool: Sequence[Mapping[str, Any]], *, query: str, query_terms: Sequence[str], as_of: date,
                model: SignalModel | None = None, cancel: Event | None = None, progress: Progress | None = None,
                months: int = 24, max_candidates: int = MAX_CANDIDATES,
                client: httpx.Client | None = None, openalex_key: str | None = None,
                cache: PhraseCache | None = None) -> dict[str, Any]:
    cancel = cancel or Event()
    model = model or SignalModel.load(MODEL_PATH)
    candidates = mine_candidates(pool, query_terms=query_terms, limit=max_candidates)
    done, lock = [0], Lock()
    own_client = client is None
    client = client or httpx.Client(limits=httpx.Limits(max_connections=12))
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # 1. Быстрый отсев: один запрос OpenAlex по годам на кандидата.
            yearly = dict(zip(candidates, executor.map(
                lambda candidate: _years(client, candidate.surface, cancel, openalex_key, as_of, cache),
                candidates), strict=True))
            quick: list[tuple[Candidate, str]] = []
            deep: list[Candidate] = []
            for candidate in candidates:
                years = yearly[candidate]
                reason = quick_exclusion(FirstMention(None, None, None, years), as_of) if years is not None else None
                if reason:
                    quick.append((candidate, reason))
                else:
                    deep.append(candidate)
            deep.sort(key=lambda candidate: (-(yearly_growth(yearly[candidate] or (), as_of) or 1.0),
                                             -len(candidate.documents), candidate.phrase))
            checked, unchecked = deep[:MAX_DEEP], deep[MAX_DEEP:]
            if progress:
                progress(0, len(checked))
            # arXiv разрешает один запрос в 3 секунды на всё приложение: фразы
            # подробной проверки спрашиваются у него пачками, а не по одной.
            batcher = ArxivBatcher(client, month_window(as_of, months), cancel,
                                   exact_first=os.environ.get(EXACT_VARIABLE) == "1")
            for candidate in checked:
                forms = variants(candidate.surface)
                if cache is None or cache.peek(("history", tuple(forms), as_of, months)) is None:
                    batcher.register("history", forms)
                if yearly[candidate] is None:
                    batcher.register("first", forms)
            # Отсеянные и непроверенные попадают в журнал без помесячной истории.
            for candidate, reason in [*quick, *((candidate, NOT_CHECKED) for candidate in unchecked)]:
                results.append(_evaluate(candidate, pool, query, as_of, model, client, cancel, months,
                                         openalex_key, yearly[candidate], deep=False, forced_reason=reason))
            # 2. Помесячная история и кривая эксперта для самых растущих.
            futures = {executor.submit(_evaluate, candidate, pool, query, as_of, model, client, cancel, months,
                                       openalex_key, yearly[candidate], cache=cache, arxiv=batcher): candidate
                       for candidate in checked}
            for future in as_completed(futures):
                if cancel.is_set():
                    break
                try:
                    results.append(future.result())
                except HistoryFetchError:
                    failed.append(futures[future].surface)
                with lock:
                    done[0] += 1
                    if progress:
                        progress(done[0], len(checked))
    finally:
        if own_client:
            client.close()
    signals = sorted((item for item in results if item["is_signal"]),
                     key=lambda item: (-item["score"], item["title"]))
    # A probability below the weak-signal threshold is not a confirmed signal.
    # It can still be a useful, evidence-backed technology candidate when
    # enough monthly work exists and no maturity or decline rule excludes it.
    lower_confidence = sorted((item for item in results if not item["is_signal"]
                               and not item["rule_excluded"] and item["curve"]["trend"] != "снижается"),
                              key=lambda item: (-item["score"], item["title"]))
    top = signals[:TOP_SIZE] + lower_confidence[:max(0, TOP_SIZE - len(signals))]
    for item in signals[TOP_SIZE:]:
        item["reasons"] = [BELOW_TOP]
    for item in lower_confidence[max(0, TOP_SIZE - len(signals)):]:
        item["reasons"] = [BELOW_TOP, *item["reasons"]]
    selected = {id(item) for item in top}
    excluded = sorted((item for item in results if id(item) not in selected),
                      key=lambda item: (-item["probability"], item["title"]))
    # Expanded cards need more linked papers; exclusion reports stay compact.
    for item in excluded:
        item["sources"] = item["sources"][:EXCLUDED_SOURCE_LIMIT]
    return {
        "policy_version": POLICY_VERSION, "query": query, "as_of": as_of.isoformat(), "months": months,
        "top_probability": TOP_PROBABILITY,
        "candidates_total": len(candidates), "evaluated": len(results), "failed": failed,
        "high_confidence": sum(item["probability"] > HIGH_CONFIDENCE for item in top),
        "sources_processed": len(pool) + sum(item["curve"]["materials"] for item in results),
        "technologies": top, "excluded": excluded,
    }
