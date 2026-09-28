"""Паспорт кандидата в схеме датасета и решение модели этапа 1.

Измеренные факты (объём за всё время, первое упоминание, кривая эксперта,
типы источников, слова стадии в найденных текстах) переводятся в поля
датасета — стадию, тренд, компании и обоснование — фиксированными правилами.
К паспорту применяется та же модель, что обучена на датасете организаторов,
поэтому объяснение решения на открытом поиске и на датасете одно и то же.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.radar.evidence import FirstMention, PoolEvidence
from app.signal_model.dataset import Technology
from app.signal_model.model import Prediction, SignalModel
from app.trend_confidence import SOURCE_CLASS_LABELS, SOURCE_RULES, CurveAssessment

# Столько работ с фразой в названии за всё время — уже массовая тема. OpenAlex
# покрывает всю науку, arXiv и Europe PMC — её часть, поэтому пороги разные.
MATURE_VOLUME = {"openalex": 5000, "arxiv_europe_pmc": 1000}
OLD_TOPIC_VOLUME = {"openalex": 1000, "arxiv_europe_pmc": 300}
OLD_TOPIC_YEARS = 10
# Давно известное понятие без ускорения роста — не зарождающаяся технология.
OLD_CONCEPT_YEARS = 15
OLD_CONCEPT_CURVE = 50
# Для быстрого отсева: рост за два года меньше полутора раз — не ускорение.
OLD_CONCEPT_GROWTH = 1.5
NICHE_VOLUME = 100
LOW_TRUST_CLASSES = frozenset({"community", "news_aggregate", "repository"})
TRENDS = {"растёт быстро": "Растёт быстро", "растёт": "Растёт", "стабильный": "Стабильный",
          "снижается": "Снижается"}


@dataclass(frozen=True)
class Passport:
    technology: Technology
    first_year: int | None
    all_time: int | None
    mature: bool
    reasons: tuple[str, ...]
    # Откуда объём за всё время: вся наука (OpenAlex) или arXiv с Europe PMC.
    volume_basis: str = "arxiv_europe_pmc"
    old_concept: bool = False
    years: tuple[tuple[int, int], ...] = ()


def plural(number: int, one: str, few: str, many: str) -> str:
    """Русское согласование: 1 работа, 3 работы, 5 работ."""
    tail = number % 100
    if 11 <= tail <= 14:
        return many
    return one if tail % 10 == 1 else few if 2 <= tail % 10 <= 4 else many


def yearly_growth(years: tuple[tuple[int, int], ...], as_of: date) -> float | None:
    """Два последних полных года против двух предыдущих; None — работ нет."""
    counts = dict(years)
    recent = counts.get(as_of.year - 1, 0) + counts.get(as_of.year - 2, 0)
    earlier = counts.get(as_of.year - 3, 0) + counts.get(as_of.year - 4, 0)
    if not recent and not earlier:
        return None
    return recent / max(earlier, 1)


def quick_exclusion(mention: FirstMention, as_of: date) -> str | None:
    """Быстрый отсев по годам OpenAlex до дорогой помесячной истории."""
    total, first_year = mention.openalex_total, mention.openalex_first_year
    if mention.openalex_years is None or total is None:
        return None
    age = as_of.year - first_year if first_year is not None else 0
    growth = yearly_growth(mention.openalex_years, as_of)
    if total >= MATURE_VOLUME["openalex"] or (age >= OLD_TOPIC_YEARS and total >= OLD_TOPIC_VOLUME["openalex"]):
        return (f"Зрелая тема: {total} {plural(total, 'работа', 'работы', 'работ')} во всей науке"
                + (f", первое упоминание в {first_year} году" if first_year else ""))
    if age >= OLD_CONCEPT_YEARS and (growth is None or growth < OLD_CONCEPT_GROWTH):
        return f"Давно известное понятие (с {first_year} года) без ускорения роста"
    return None


def _class_counts(curve: CurveAssessment) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source, count in curve.sources.items():
        source_class = SOURCE_RULES[source].source_class
        counts[source_class] = counts.get(source_class, 0) + count
    return counts


def build_passport(phrase: str, area: str, as_of: date, curve: CurveAssessment, mention: FirstMention,
                   evidence: PoolEvidence, earliest_record: date | None) -> Passport:
    if mention.openalex_years is not None:
        basis, all_time = "openalex", mention.openalex_total
        first_year = mention.openalex_first_year or (earliest_record.year if earliest_record else None)
    else:
        basis = "arxiv_europe_pmc"
        totals = [value for value in (mention.arxiv_total, mention.europe_pmc_total) if value is not None]
        all_time = sum(totals) if totals else None
        years = [value.year for value in (mention.earliest, earliest_record) if value is not None]
        first_year = min(years) if years else None
    age = as_of.year - first_year if first_year is not None else None
    old = age is not None and age >= OLD_TOPIC_YEARS
    volume_mature = all_time is not None and (
        all_time >= MATURE_VOLUME[basis] or (old and all_time >= OLD_TOPIC_VOLUME[basis]))
    old_concept = (age is not None and age >= OLD_CONCEPT_YEARS
                   and (curve.confidence or 0) < OLD_CONCEPT_CURVE)
    mature = volume_mature or old_concept
    stage = "Массовое внедрение" if mature else evidence.stage or "Исследование"
    classes = _class_counts(curve)
    low_trust = sum(count for name, count in classes.items() if name in LOW_TRUST_CLASSES)
    science = sum(count for name, count in classes.items() if name in {"scientific_index", "preprint", "patent"})
    facts: list[str] = []
    if first_year is not None:
        facts.append(f"Фраза впервые встречается в названиях в {first_year} году; последние материалы — "
                     f"{as_of.year} год.")
    breakdown = ", ".join(f"{SOURCE_CLASS_LABELS[name]} — {count}" for name, count in sorted(classes.items()))
    months = len(curve.months)
    facts.append(f"За {months} {plural(months, 'месяц', 'месяца', 'месяцев')} найдено работ: {curve.materials}"
                 + (f" ({breakdown})." if breakdown else "."))
    if classes.get("preprint"):
        facts.append("Идут препринты и исследовательские статьи.")
    if mature and all_time is not None:
        facts.append(f"Массовая тема: {all_time} {plural(all_time, 'работа', 'работы', 'работ')} с этой фразой "
                     "в названии за всё время"
                     + ("; развивается больше десятилетия." if old else "."))
    elif all_time is not None and all_time < NICHE_VOLUME:
        facts.append(f"Тема нишевая: {all_time} {plural(all_time, 'работа', 'работы', 'работ')} за всё время.")
    if evidence.funding_mentions:
        facts.append(f"В найденных материалах упоминаются стартапы и раунды финансирования "
                     f"({evidence.funding_mentions}).")
    if low_trust and not science:
        facts.append("Упоминания держатся в медиа и сообществах без научных подтверждений.")
    if curve.trend == "снижается":
        facts.append("Интерес снижается.")
    if evidence.standard_mentions >= 2:
        facts.append("В материалах упоминаются стандарты.")
    technology = Technology(
        title=phrase, area=area, companies=", ".join(evidence.incumbents), rationale=" ".join(facts),
        stage=stage, trend=TRENDS.get(curve.trend, ""), label=None)
    return Passport(technology, first_year, all_time, mature, tuple(facts), basis, old_concept and not volume_mature,
                    mention.openalex_years or ())


def exclusion_reasons(passport: Passport, curve: CurveAssessment, prediction: Prediction) -> tuple[str, ...]:
    """Почему кандидат не вошёл в ТОП: сначала прямые правила, затем главные вклады модели."""
    reasons: list[str] = []
    if passport.old_concept:
        reasons.append(f"Давно известное понятие (с {passport.first_year} года) без ускорения роста")
    elif passport.mature:
        count = passport.all_time or 0
        reasons.append(f"Зрелая тема: {count} {plural(count, 'работа', 'работы', 'работ')} за всё время"
                       + (f", первое упоминание в {passport.first_year} году" if passport.first_year else ""))
    if curve.trend == "снижается":
        reasons.append("Интерес снижается")
    if curve.confidence is None:
        reasons.append(f"Мало работ для вывода: {curve.materials} за {len(curve.months)} "
                       f"{plural(len(curve.months), 'месяц', 'месяца', 'месяцев')}"
                       + ("" if curve.coverage_complete else " — история неполная: источник отказал"))
    for part in prediction.contributions:
        if part.weight < 0 and len(reasons) < 4:
            reasons.append(f"{part.label}: {part.value}")
    return tuple(dict.fromkeys(reasons))


def decide(model: SignalModel, passport: Passport) -> Prediction:
    return model.predict([passport.technology])[0]


# Выше 60 % — подтверждённый слабый сигнал. ТОП может также показывать
# кандидатов с меньшей вероятностью, сохраняя их отдельный статус.
TOP_PROBABILITY = 0.6


def excluded_by_rule(passport: Passport, curve: CurveAssessment | None = None) -> bool:
    """Зрелость, недостаток данных и спад исключают тему при любом проценте модели.

    Недостаток данных исключает только при полной истории. Если источник
    истории отказал (лимит бесплатных запросов OpenAlex), пустая кривая говорит
    о лимите, а не о теме: кандидат остаётся ниже подтверждённых с этой
    причиной. Замер 28.09.2026: при исчерпанном лимите все 60 кандидатов, среди
    них «3D convolutional neural networks» с 99 %, уходили в журнал исключений.
    """
    if passport.mature or curve is None:
        return passport.mature
    return curve.trend == "снижается" or (curve.confidence is None and curve.coverage_complete)


def is_weak_signal(passport: Passport, prediction: Prediction, curve: CurveAssessment | None = None) -> bool:
    """ТЗ прямо исключает массово внедрённые технологии, даже если интерес к ним растёт.

    Модель этапа 1 не видела зрелых тем с быстрым ростом, поэтому зрелость —
    жёсткое правило поверх её решения. Фраза с единичными работами за период —
    не сигнал, а случайное совпадение слов: вывод по ней делать не из чего,
    в том числе когда история не собрана.
    """
    return (prediction.probability > TOP_PROBABILITY and not excluded_by_rule(passport, curve)
            and (curve is None or curve.confidence is not None))
