"""Похожие материалы по месяцам для каждой публикации ТОПа.

Сходство — косинус TF-IDF по словам названия и аннотации внутри одной
собранной выборки. Это не поиск по всему миру: пустой месяц значит лишь,
что в собранной выборке похожих материалов за него нет. Поэтому рядом с числом
похожих передаётся, сколько материалов выборка вообще собрала за месяц, а
месяцы до начала собранных данных не показываются.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
import math
import re
from typing import Any

WINDOW_MONTHS = 12
# Подобрано на выборке «solid-state batteries»: от 0,15 соседи по той же
# подтеме, ниже 0,12 общим остаётся одно распространённое слово.
MIN_SIMILARITY = 0.15
# Выше — та же работа: рецензии и решения редакции Crossref повторяют название.
SAME_WORK_SIMILARITY = 0.8
# Название короче аннотации, но точнее описывает тему.
TITLE_WEIGHT = 2

_WORD = re.compile(r"[a-zа-яё][a-zа-яё0-9]*(?:-[a-zа-яё0-9]+)*")
_STOPWORDS = frozenset("""
about above across after again against also among and another any are around based
because been before being below between both but can could does done during each
either for from further had has have having here how however into its itself just
may might more most much must new not now off once one only onto other our out over
own paper per present presents propose proposed results same several shall should
show shows shown since some such than that the their them then there these they this
those through thus too two under until upon use used uses using very via was were
what when where whether which while who whom whose why will with within without
would yet you your
без более будет было были для его если есть еще ещё или как когда кото которая
которые который которых между может можно над них она они при про своих так также
такие только том что чтобы это этого этой эти этот
""".split())


def _tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [word for word in _WORD.findall(text.casefold())
            if len(word) >= 3 and word not in _STOPWORDS]


def _terms(publication: Mapping[str, Any]) -> Counter[str]:
    terms = Counter(_tokens(publication.get("summary")))
    for word in _tokens(publication.get("title")):
        terms[word] += TITLE_WEIGHT
    return terms


def _vector(terms: Counter[str], idf: Mapping[str, float]) -> dict[str, float]:
    weights = {word: (1 + math.log(count)) * idf[word] for word, count in terms.items()}
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    return {word: weight / norm for word, weight in weights.items()} if norm else {}


def _month(publication: Mapping[str, Any]) -> str | None:
    """Месяц выхода; у записей с противоречивыми датами его нет."""
    month = publication.get("publication_month")
    if isinstance(month, str):
        return month
    exact = publication.get("published_at")
    return exact[:7] if isinstance(exact, str) else None


def _window(as_of: date) -> tuple[str, ...]:
    last = as_of.year * 12 + as_of.month - 1
    return tuple(f"{index // 12:04d}-{index % 12 + 1:02d}"
                 for index in range(last - WINDOW_MONTHS + 1, last + 1))


def similar_activity(top: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]],
                     as_of: date) -> dict[str, dict[str, Any]]:
    """Для каждой публикации ТОПа — число похожих материалов выборки по месяцам.

    Сама публикация и её копии в подсчёт не входят. Ключ ответа — `publication_id`.
    """
    terms = [_terms(item) for item in pool]
    frequency = Counter(word for item in terms for word in item)
    total = len(pool)
    idf = {word: math.log((1 + total) / (1 + count)) + 1 for word, count in frequency.items()}
    vectors = [_vector(item, idf) for item in terms]
    months = _window(as_of)
    publication_months = [_month(item) for item in pool]
    collected = Counter(publication_months)
    # График начинается с первого месяца, за который выборка что-то собрала.
    first = next((index for index, month in enumerate(months) if collected[month]), len(months) - 1)
    months = months[first:]
    position = {month: index for index, month in enumerate(months)}
    month_positions = [position.get(month) if month is not None else None
                       for month in publication_months]
    index_of = {item["publication_id"]: index for index, item in enumerate(pool)}
    targets = []
    for publication in top:
        own = index_of.get(publication["publication_id"])
        vector = vectors[own] if own is not None else _vector(_terms(publication), idf)
        targets.append((publication, own, vector))
    relevant_words = {word for _, _, vector in targets for word in vector}
    # A publication can only be similar if it shares a term. Index eligible
    # vectors for the TOP's terms once, instead of scanning every collected
    # paper for every TOP card or retaining unrelated terms in the index.
    postings: dict[str, list[int]] = {}
    for index, (vector, month_position) in enumerate(zip(vectors, month_positions, strict=True)):
        if month_position is None:
            continue
        for word in vector:
            if word in relevant_words:
                postings.setdefault(word, []).append(index)
    activity: dict[str, dict[str, Any]] = {}
    for publication, own, vector in targets:
        counts = [0] * len(months)
        if vector:
            scores = [0.0] * len(pool)
            touched: list[int] = []
            for word, weight in vector.items():
                for index in postings.get(word, ()):
                    if scores[index] == 0.0:
                        touched.append(index)
                    scores[index] += weight * vectors[index][word]
            for index in touched:
                if index == own:
                    continue
                candidate = vectors[index]
                similarity = scores[index]
                if len(candidate) < len(vector):
                    # Preserve the original summation order when the pool
                    # vector is shorter, including values near the thresholds.
                    similarity = sum(weight * vector.get(word, 0.0)
                                     for word, weight in candidate.items())
                if MIN_SIMILARITY <= similarity < SAME_WORK_SIMILARITY:
                    month_position = month_positions[index]
                    assert month_position is not None
                    counts[month_position] += 1
        activity[publication["publication_id"]] = {
            "months": [{"month": month, "count": count, "collected": collected[month]}
                       for month, count in zip(months, counts, strict=True)],
            "total": sum(counts),
        }
    return activity
