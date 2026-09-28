"""Уверенность в тренде технологии по помесячной кривой (логика эксперта).

1. Берётся период (по умолчанию 24 завершённых месяца).
2. Каждому источнику назначен коэффициент достоверности; наибольший — у
   научных каталогов.
3. Одна работа, найденная в нескольких источниках, считается один раз — с
   наибольшим коэффициентом («не повторяющиеся источники»).
4. По месяцам суммируются коэффициенты, ряд сглаживается скользящим средним.
5. Сглаженный ряд приближается параболой. Ветви вверх, рост к концу периода,
   хорошее совпадение с параболой и независимые типы источников дают 100.

Коэффициенты — редакционная шкала, а не измеренная точность источников.
Уверенность описывает форму кривой найденных материалов, а не вероятность
того, что технология станет успешной.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np

POLICY_VERSION = "trend-confidence/1.0.0"
DEFAULT_MONTHS = 24
SMOOTHING_WINDOW = 3
# Ниже этих порогов форму кривой не оценить: параболу можно провести через что угодно.
MIN_MATERIALS = 10
MIN_ACTIVE_MONTHS = 6
MIN_SOURCE_CLASSES = 2
FULL_CONFIDENCE_R2 = 0.9
# Один тип источников или неполная выдача не дают полной уверенности.
SINGLE_CLASS_CAP = 60
PARTIAL_COVERAGE_CAP = 80
FAST_GROWTH_RATIO = 2.0


@dataclass(frozen=True)
class SourceRule:
    coefficient: float
    source_class: str
    label: str


SOURCE_RULES: dict[str, SourceRule] = {
    "crossref": SourceRule(1.0, "scientific_index", "Crossref"),
    "openalex": SourceRule(1.0, "scientific_index", "OpenAlex"),
    "europe_pmc": SourceRule(1.0, "scientific_index", "Europe PMC"),
    "epo": SourceRule(0.9, "patent", "Патенты EPO"),
    "arxiv": SourceRule(0.85, "preprint", "arXiv"),
    "biorxiv": SourceRule(0.85, "preprint", "bioRxiv"),
    "openreview": SourceRule(0.8, "preprint", "OpenReview"),
    "zenodo": SourceRule(0.7, "research_artifact", "Zenodo"),
    "report": SourceRule(0.6, "analytical_report", "Аналитический отчёт"),
    "nist_news": SourceRule(0.7, "institutional_news", "NIST News"),
    "mit_research_news": SourceRule(0.65, "institutional_news", "MIT Research News"),
    "horizon_magazine": SourceRule(0.6, "institutional_news", "Horizon Magazine"),
    "github": SourceRule(0.5, "repository", "GitHub"),
    "gdelt": SourceRule(0.35, "news_aggregate", "GDELT"),
    "hacker_news": SourceRule(0.25, "community", "Hacker News"),
    "habr": SourceRule(0.25, "community", "Хабр"),
    # Бесключевые источники, которые программа собирает сама.
    "semantic_scholar": SourceRule(0.95, "scientific_index", "Semantic Scholar"),
    "doaj": SourceRule(0.9, "scientific_index", "DOAJ"),
    "cyberleninka": SourceRule(0.85, "scientific_index", "КиберЛенинка"),
    "hal": SourceRule(0.85, "scientific_index", "HAL"),
    "dblp": SourceRule(0.85, "scientific_index", "dblp"),
    "chemrxiv": SourceRule(0.8, "preprint", "ChemRxiv"),
    "osti": SourceRule(0.75, "research_artifact", "OSTI"),
    "nasa_ntrs": SourceRule(0.75, "research_artifact", "NASA NTRS"),
    "google_news": SourceRule(0.35, "news_aggregate", "Google News"),
    "huggingface": SourceRule(0.45, "repository", "Hugging Face"),
    "stack_exchange": SourceRule(0.25, "community", "Stack Overflow"),
    "openaire": SourceRule(0.9, "scientific_index", "OpenAIRE"),
    "jstage": SourceRule(0.85, "scientific_index", "J-STAGE"),
    "npm": SourceRule(0.35, "repository", "npm"),
}
SOURCE_CLASS_LABELS = {
    "scientific_index": "научный каталог", "patent": "патенты", "preprint": "препринты",
    "research_artifact": "исследовательские материалы", "analytical_report": "аналитические отчёты",
    "institutional_news": "новости организаций", "repository": "репозитории",
    "news_aggregate": "агрегатор новостей", "community": "сообщество",
}


def trust_level(source_id: str) -> str:
    """Уровень доверенности для карточки источника (требование ТЗ)."""
    rule = SOURCE_RULES.get(source_id)
    if rule is None:
        return "не определён"
    if rule.coefficient >= 0.85:
        return "высокий"
    if rule.coefficient >= 0.5:
        return "средний"
    # Соцсети, агрегаторы и сообщества — только первичный индикатор.
    return "пониженный"


@dataclass(frozen=True)
class Material:
    """Одна работа: `key` склеивает копии из разных источников (DOI или ссылка)."""
    key: str
    source_id: str
    month: str


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    label: str


@dataclass(frozen=True)
class MonthPoint:
    month: str
    materials: int
    weighted: float
    smoothed: float
    fitted: float | None


@dataclass(frozen=True)
class CurveAssessment:
    policy_version: str
    months: tuple[MonthPoint, ...]
    # None — данных недостаточно для оценки формы; это не ноль.
    confidence: int | None
    trend: str
    checks: tuple[Check, ...]
    materials: int
    source_classes: tuple[str, ...]
    sources: dict[str, int]
    quadratic_r2: float | None
    linear_r2: float | None
    curvature: float | None
    coverage_complete: bool


def month_window(as_of: date, months: int = DEFAULT_MONTHS) -> tuple[str, ...]:
    """Завершённые месяцы перед месяцем `as_of`: текущий месяц ещё не закончился."""
    last = as_of.year * 12 + as_of.month - 2
    return tuple(f"{index // 12:04d}-{index % 12 + 1:02d}" for index in range(last - months + 1, last + 1))


def smooth(values: list[float], window: int = SMOOTHING_WINDOW) -> list[float]:
    """Центрированное скользящее среднее; на краях — по доступным соседям."""
    half = window // 2
    return [sum(values[max(0, index - half):index + half + 1])
            / len(values[max(0, index - half):index + half + 1]) for index in range(len(values))]


def _r2(values: np.ndarray, fitted: np.ndarray) -> float:
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1 - float(np.sum((values - fitted) ** 2)) / total))


def _trend(raw: list[float], curvature: float | None) -> str:
    """Последняя треть периода против первой; выход роста на плато — не спад."""
    if not any(raw):
        return "нет данных"
    third = max(1, len(raw) // 3)
    early, late = sum(raw[:third]), sum(raw[-third:])
    if late < early * 0.8:
        return "снижается"
    if late >= max(early, 1e-9) * FAST_GROWTH_RATIO and (curvature or 0) > 0:
        return "растёт быстро"
    if late > early * 1.2:
        return "растёт"
    return "стабильный"


def assess_curve(materials: Iterable[Material], as_of: date, *, months: int = DEFAULT_MONTHS,
                 coverage_complete: bool = True) -> CurveAssessment:
    """Помесячная кривая технологии и уверенность 0–100 с объяснением по шагам.

    `coverage_complete=False` — хотя бы один источник упёрся в лимит записей,
    и часть материалов за период могла не попасть в выборку.
    """
    window = month_window(as_of, months)
    position = {month: index for index, month in enumerate(window)}
    strongest: dict[str, tuple[float, str, str]] = {}
    sources: dict[str, int] = {}
    for material in materials:
        rule = SOURCE_RULES.get(material.source_id)
        if rule is None or material.month not in position:
            continue
        sources[material.source_id] = sources.get(material.source_id, 0) + 1
        current = strongest.get(material.key)
        if current is None or rule.coefficient > current[0]:
            strongest[material.key] = (rule.coefficient, rule.source_class, material.month)
    counts = [0] * len(window)
    weighted = [0.0] * len(window)
    classes: set[str] = set()
    for coefficient, source_class, month in strongest.values():
        counts[position[month]] += 1
        weighted[position[month]] += coefficient
        classes.add(source_class)
    smoothed = smooth(weighted)
    total = len(strongest)
    active = sum(1 for count in counts if count)

    fitted_values: list[float | None] = [None] * len(window)
    quadratic_r2 = linear_r2 = curvature = end_slope = None
    if total and any(smoothed):
        x = np.arange(len(window), dtype=float)
        y = np.array(smoothed)
        a, b, c = np.polyfit(x, y, 2)
        fitted = a * x * x + b * x + c
        quadratic_r2 = _r2(y, fitted)
        linear_r2 = _r2(y, np.polyval(np.polyfit(x, y, 1), x))
        curvature, end_slope = float(a), float(2 * a * x[-1] + b)
        fitted_values = [round(max(0.0, float(value)), 4) for value in fitted]

    enough = total >= MIN_MATERIALS and active >= MIN_ACTIVE_MONTHS
    accelerating = curvature is not None and curvature > 0 and end_slope is not None and end_slope > 0
    independent = len(classes) >= MIN_SOURCE_CLASSES
    shape_fit = quadratic_r2 is not None and quadratic_r2 >= FULL_CONFIDENCE_R2
    checks = (
        Check("enough_data", enough, f"Материалов {total}, месяцев с материалами {active} из {len(window)} "
                                     f"(нужно не меньше {MIN_MATERIALS} и {MIN_ACTIVE_MONTHS})"),
        Check("accelerating", accelerating, "Парабола ветвями вверх, к концу периода рост"
              if accelerating else "Нет ускоряющегося роста: ветви параболы вниз или спад в конце периода"),
        Check("parabola_fit", shape_fit, "Совпадение с параболой R² = "
              + (f"{quadratic_r2:.2f}".replace(".", ",") if quadratic_r2 is not None else "—")
              + f" (для 100 нужно ≥ {FULL_CONFIDENCE_R2:.1f})".replace(".", ",")),
        Check("independent_sources", independent, f"Независимых типов источников {len(classes)} "
                                                  f"(нужно не меньше {MIN_SOURCE_CLASSES})"),
        Check("coverage_complete", coverage_complete, "Выдача источников полная" if coverage_complete
              else "Источник упёрся в лимит записей: часть материалов могла не попасть"),
    )
    if not enough:
        confidence = None
    elif not accelerating:
        confidence = 0
    else:
        confidence = 100 if shape_fit else round(100 * (quadratic_r2 or 0.0))
        if not independent:
            confidence = min(confidence, SINGLE_CLASS_CAP)
        if not coverage_complete:
            confidence = min(confidence, PARTIAL_COVERAGE_CAP)
    points = tuple(MonthPoint(month, counts[index], round(weighted[index], 4), round(smoothed[index], 4),
                              fitted_values[index]) for index, month in enumerate(window))
    return CurveAssessment(
        POLICY_VERSION, points, confidence, _trend(weighted, curvature), checks, total,
        tuple(sorted(classes)), dict(sorted(sources.items())),
        round(quadratic_r2, 4) if quadratic_r2 is not None else None,
        round(linear_r2, 4) if linear_r2 is not None else None,
        round(curvature, 6) if curvature is not None else None, coverage_complete)
