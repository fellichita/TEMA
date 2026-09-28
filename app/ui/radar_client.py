"""Разбор ТОПа технологий из ответа API: проверка формата до показа."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, overload

from app.input_safety import is_safe_http_url

_MONTH = re.compile(r"[0-9]{4}-(?:0[1-9]|1[0-2])\Z")


class RadarFormatError(ValueError):
    pass


@dataclass(frozen=True)
class RadarPoint:
    month: str
    materials: int
    weighted: float
    smoothed: float
    fitted: float | None


@dataclass(frozen=True)
class RadarCheck:
    passed: bool
    label: str


@dataclass(frozen=True)
class RadarSource:
    title: str
    url: str
    published: str
    source: str
    source_type: str
    language: str
    trust: str


@dataclass(frozen=True)
class RadarPredictor:
    label: str
    value: str
    weight: float


@dataclass(frozen=True)
class RadarTechnology:
    title: str
    title_ru: str | None
    probability: float
    score: float
    is_signal: bool
    # Исключена правилом ТЗ (зрелая тема, мало работ): процент модели тогда не решает.
    rule_excluded: bool
    curve_confidence: int | None
    trend: str
    materials: int
    points: tuple[RadarPoint, ...]
    checks: tuple[RadarCheck, ...]
    stage: str
    rationale: str
    first_year: int | None
    all_time: int | None
    predictors: tuple[RadarPredictor, ...]
    sources: tuple[RadarSource, ...]
    # Выжимки из найденных текстов: оригинал и машинный перевод.
    description: str | None
    description_ru: str | None
    advantage: str | None
    advantage_ru: str | None
    case: str | None
    case_ru: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RadarResult:
    query: str
    as_of: str
    months: int
    candidates_total: int
    evaluated: int
    high_confidence: int
    sources_processed: int
    technologies: tuple[RadarTechnology, ...]
    excluded: tuple[RadarTechnology, ...]
    # Версия правил отбора, сохранённая вместе с анализом; у старых архивов может отсутствовать.
    policy_version: str | None = None


@dataclass(frozen=True)
class RadarStatus:
    state: str
    completed: int = 0
    total: int = 0
    result: RadarResult | None = None
    message: str | None = None


@overload
def _text(value: object, limit: int, *, optional: Literal[False] = False) -> str: ...


@overload
def _text(value: object, limit: int, *, optional: Literal[True]) -> str | None: ...


def _text(value: object, limit: int, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise RadarFormatError
    return value


@overload
def _count(value: object, *, optional: Literal[False] = False, limit: int = 10_000_000) -> int: ...


@overload
def _count(value: object, *, optional: Literal[True], limit: int = 10_000_000) -> int | None: ...


def _count(value: object, *, optional: bool = False, limit: int = 10_000_000) -> int | None:
    if value is None and optional:
        return None
    if type(value) is not int or not 0 <= value <= limit:
        raise RadarFormatError
    return value


@overload
def _number(value: object, *, low: float = 0.0, high: float = 1_000_000.0,
            optional: Literal[False] = False) -> float: ...


@overload
def _number(value: object, *, low: float = 0.0, high: float = 1_000_000.0,
            optional: Literal[True]) -> float | None: ...


def _number(value: object, *, low: float = 0.0, high: float = 1_000_000.0, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not low <= value <= high:
        raise RadarFormatError
    return float(value)


def _technology(item: object, translation: dict[str, str]) -> RadarTechnology:
    if not isinstance(item, dict) or not isinstance(item.get("curve"), dict) or not isinstance(item.get("passport"), dict):
        raise RadarFormatError
    curve, passport = item["curve"], item["passport"]
    points = []
    for point in curve.get("months") or []:
        if not isinstance(point, dict) or not isinstance(point.get("month"), str) \
                or _MONTH.fullmatch(point["month"]) is None:
            raise RadarFormatError
        points.append(RadarPoint(point["month"], _count(point.get("materials")), _number(point.get("weighted")),
                                 _number(point.get("smoothed")), _number(point.get("fitted"), optional=True)))
    if len(points) > 120 or [point.month for point in points] != sorted({point.month for point in points}):
        raise RadarFormatError
    checks = tuple(RadarCheck(bool(check.get("passed")), _text(check.get("label"), 300))
                   for check in curve.get("checks") or [] if isinstance(check, dict))
    sources = []
    for source in item.get("sources") or []:
        if not isinstance(source, dict) or not is_safe_http_url(source.get("url")):
            continue
        sources.append(RadarSource(*(_text(source.get(key), 1_000) or "" for key in
                                     ("title", "url", "published", "source", "source_type", "language", "trust"))))
    predictors = tuple(RadarPredictor(_text(part.get("label"), 100), _text(part.get("value"), 100),
                                      _number(part.get("weight"), low=-1_000, high=1_000))
                       for part in item.get("predictors") or [] if isinstance(part, dict))
    title = _text(item.get("title"), 300)
    texts = {key: _text(item.get(key), 2_000, optional=True) for key in ("description", "advantage", "case")}
    confidence = curve.get("confidence")
    return RadarTechnology(
        title=title, title_ru=translation.get(title), probability=_number(item.get("probability"), high=1.0),
        score=_number(item.get("score"), high=1.0), is_signal=item.get("is_signal") is True,
        rule_excluded=item.get("rule_excluded") is True,
        curve_confidence=_count(confidence, optional=True, limit=100), trend=_text(curve.get("trend"), 40),
        materials=_count(curve.get("materials")), points=tuple(points), checks=checks,
        stage=_text(passport.get("stage"), 100), rationale=_text(passport.get("rationale"), 3_000),
        first_year=_count(passport.get("first_year"), optional=True, limit=3000),
        all_time=_count(passport.get("all_time"), optional=True), predictors=predictors, sources=tuple(sources[:30]),
        description=texts["description"], description_ru=translation.get(texts["description"] or ""),
        advantage=texts["advantage"], advantage_ru=translation.get(texts["advantage"] or ""),
        case=texts["case"], case_ru=translation.get(texts["case"] or ""),
        reasons=tuple(_text(reason, 300) for reason in item.get("reasons") or []))


def parse_radar(payload: object) -> RadarStatus | None:
    """None — радара в ответе нет (старый сервис); ошибка формата — RadarFormatError."""
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("state") not in {"running", "ready", "unavailable"}:
        raise RadarFormatError
    completed, total = _count(payload.get("completed", 0), limit=1000), _count(payload.get("total", 0), limit=1000)
    if completed > total:
        raise RadarFormatError
    message = _text(payload.get("message"), 500, optional=True)
    raw = payload.get("result")
    if raw is None:
        return RadarStatus(payload["state"], completed, total, None, message)
    if not isinstance(raw, dict):
        raise RadarFormatError
    translation = raw.get("translation") or {}
    if not isinstance(translation, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                                for key, value in translation.items()):
        raise RadarFormatError
    technologies = tuple(_technology(item, translation) for item in raw.get("technologies") or [])
    excluded = tuple(_technology(item, translation) for item in raw.get("excluded") or [])
    if len(technologies) > 15 or len(excluded) > 200:
        raise RadarFormatError
    result = RadarResult(_text(raw.get("query"), 2_000), _text(raw.get("as_of"), 10), _count(raw.get("months"), limit=120),
                         _count(raw.get("candidates_total"), limit=1000), _count(raw.get("evaluated"), limit=1000),
                         _count(raw.get("high_confidence"), limit=15), _count(raw.get("sources_processed")),
                         technologies, excluded, _text(raw.get("policy_version"), 40, optional=True))
    return RadarStatus(payload["state"], completed, total, result, message)
