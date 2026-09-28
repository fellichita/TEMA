"""Релевантность каждой найденной публикации теме анализа — до выбора ТОПа.

Раньше ТОП-15 веба брал самые свежие записи выдачи, а смысл проверял уже у
них, поэтому в ТОП попадали свежие, но чужие материалы. Теперь оценка
делается для всего пула:

* научный документ получает оценку этапа обнаружения — косинус E5 по всем
  фрагментам его аннотации (тот же порог 0,80 и та же шкала);
* материал дополнительных источников оценивается той же моделью по заголовку
  и описанию относительно исходной, английской формулировок и синонимов;
* к смыслу добавляется лексическое покрытие темы и, если программа уже
  обучилась на своих анализах, оценка обученной модели.

Материал «не по теме» не показывается в выдаче и не попадает в ТОП; его число
сообщается отдельно. Оценка — упорядочивающая эвристика, а не вероятность.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
import math
from threading import Event, Timer
from typing import Any

from app.relevance_learning import RelevanceModel, features, profile_terms
from app.topic_relevance import RELEVANCE_VERSION, TopicProfile, combine, lexical_match

MAX_ENCODED = 3000
ENCODING_TIMEOUT_SECONDS = 180
CODES = {"relevant": "r", "weak": "w", "off_topic": "o"}
DECISIONS = {code: decision for decision, code in CODES.items()}


class _Deadline:
    def __init__(self, cancel: Any, expired: Event):
        self._cancel, self._expired = cancel, expired

    def is_set(self) -> bool:
        return bool(self._cancel is not None and self._cancel.is_set()) or self._expired.is_set()


def _plan_value(plan: Mapping[str, Any] | Any, name: str) -> Any:
    return plan.get(name) if isinstance(plan, Mapping) else getattr(plan, name, None)


def _semantic(encoder: Any, plan: Mapping[str, Any] | Any, texts: Sequence[str], cancel: Any) -> list[float | None]:
    """Косинус каждого текста с темой по правилу этапа обнаружения: 0,8·якорь + 0,2·направление."""
    if encoder is None or not texts:
        return [None] * len(texts)
    anchors = [text for text in dict.fromkeys((_plan_value(plan, "original_query"), _plan_value(plan, "english_query")))
               if isinstance(text, str) and text.strip()]
    directions = [text for name in ("synonyms", "subdirections") for text in (_plan_value(plan, name) or ())
                  if isinstance(text, str) and text.strip()]
    queries = list(dict.fromkeys([*anchors, *directions]))[:40]
    if not anchors:
        return [None] * len(texts)
    expired = Event()
    timer = Timer(ENCODING_TIMEOUT_SECONDS, expired.set)
    timer.daemon = True
    timer.start()
    guard = _Deadline(cancel, expired)
    try:
        query_vectors = encoder.encode([text[:2000] for text in queries], kind="query", cancel=guard)
        vectors = encoder.encode([text[:2000] for text in texts], kind="passage", cancel=guard)
    except Exception:
        if cancel is not None and cancel.is_set():
            raise
        return [None] * len(texts)
    finally:
        timer.cancel()
    import numpy as np

    anchor_count = len(anchors)
    matrix = np.asarray(vectors, dtype=np.float32) @ np.asarray(query_vectors, dtype=np.float32).T
    scores: list[float | None] = []
    for similarities in matrix:
        if not similarities.size or not np.isfinite(similarities).all():
            scores.append(None)
            continue
        anchor = float(similarities[:anchor_count].max())
        scores.append(round(max(-1.0, min(1.0, 0.8 * anchor + 0.2 * float(similarities.max()))), 4))
    return scores


def assess_pool(pool: Sequence[Mapping[str, Any]], plan: Mapping[str, Any] | Any, *,
                discovery: Mapping[str, tuple[float, str]] | None = None, encoder: Any = None,
                cancel: Any = None, model: RelevanceModel | None = None,
                progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Оценка темы для всех публикаций пула; словарь сохраняется в результат анализа."""
    discovery = discovery or {}
    profile = TopicProfile.from_plan(plan)
    terms = profile_terms(profile)
    as_of: date | None = None
    raw_as_of = _plan_value(plan, "as_of")
    if isinstance(raw_as_of, date):
        as_of = raw_as_of
    elif isinstance(raw_as_of, str):
        try:
            as_of = date.fromisoformat(raw_as_of)
        except ValueError:
            as_of = None
    model = model if model is not None and model.ready else None
    cosines: dict[str, float | None] = {}
    excluded: set[str] = set()
    pending: list[tuple[str, str]] = []
    for item in pool:
        publication_id = item.get("publication_id")
        if not isinstance(publication_id, str):
            continue
        studies = [discovery[study] for study in item.get("study_ids") or () if study in discovery]
        if studies:
            cosines[publication_id] = max(score for score, _ in studies)
            if {decision for _, decision in studies} == {"closer_to_excluded_scope"}:
                excluded.add(publication_id)
        elif len(pending) < MAX_ENCODED:
            title = str(item.get("title") or "").strip()
            summary = item.get("summary") if isinstance(item.get("summary"), str) else ""
            pending.append((publication_id, (title + (". " + summary if summary else ""))[:2000]))
    if progress is not None:
        progress(0, len(pending))
    for (publication_id, _), cosine in zip(pending, _semantic(encoder, plan, [text for _, text in pending], cancel),
                                           strict=True):
        cosines[publication_id] = cosine
    if progress is not None:
        progress(len(pending), len(pending))
    items: dict[str, list[Any]] = {}
    counts = dict.fromkeys(CODES, 0)
    for item in pool:
        publication_id = item.get("publication_id")
        if not isinstance(publication_id, str) or publication_id in items:
            continue
        match = lexical_match(profile, str(item.get("title") or ""),
                              item.get("summary") if isinstance(item.get("summary"), str) else None)
        learned = model.predict(features(item, match, terms, as_of)) if model is not None else None
        cosine = cosines.get(publication_id)
        score, decision = combine(match.score, cosine, learned, model.blend if model is not None else 0.0)
        if publication_id in excluded:
            decision = "off_topic"
        counts[decision] += 1
        items[publication_id] = [score, CODES[decision], match.score, cosine,
                                 None if learned is None else round(learned, 4)]
    return {"version": RELEVANCE_VERSION, "model_version": model.version if model is not None else None,
            "blend": model.blend if model is not None else 0.0, "semantic": encoder is not None,
            "counts": counts, "items": items}


def relevance_items(value: object) -> dict[str, tuple[float, str, float, float | None]] | None:
    """Сохранённая оценка темы: publication_id → (оценка, решение, лексика, косинус)."""
    if not isinstance(value, dict) or value.get("version") != RELEVANCE_VERSION or not isinstance(value.get("items"),
                                                                                                   dict):
        return None
    items = {}
    for publication_id, entry in value["items"].items():
        if (not isinstance(publication_id, str) or not isinstance(entry, list) or len(entry) < 4
                or not isinstance(entry[0], (int, float)) or entry[1] not in DECISIONS
                or not isinstance(entry[2], (int, float))):
            continue
        cosine = entry[3] if isinstance(entry[3], (int, float)) and math.isfinite(entry[3]) else None
        items[publication_id] = (float(entry[0]), DECISIONS[entry[1]], float(entry[2]), cosine)
    return items
