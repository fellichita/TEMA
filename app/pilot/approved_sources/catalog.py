"""Каталог источников: страна, язык, вид материала — и правила отбора владельца.

Владелец в панели управления выбирает, из источников каких стран брать
сведения, и может выключить отдельные источники. Правило записывается в сам
запуск анализа (`source_policy`), поэтому сохранённый результат всегда
показывает, что именно было опрошено. Научный корпус анализа (Crossref,
OpenAlex) международный и этим правилом не фильтруется: без него анализ не
найдёт ни тем, ни их истории.

Страна источника — страна издателя или организации, которая ведёт выдачу:
ISO 3166-1 alpha-2, «EU» — учреждения Евросоюза, «INT» — международные
каталоги без одной страны. Многострановые источники (Google News, GDELT, DOAJ)
получают выбранные страны сами и ищут только в их изданиях.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any

from app.pilot.approved_sources.contracts import SOURCE_IDS

POLICY_VERSION = "source-policy/1"
POLICY_FILE = "web-source-policy.json"


@dataclass(frozen=True)
class SourceInfo:
    label: str
    country: str
    language: str
    group: str
    description: str
    multi_country: bool = False


SOURCE_INFO: dict[str, SourceInfo] = {
    "arxiv": SourceInfo("arXiv", "INT", "en", "science", "Препринты физики, информатики, математики"),
    "biorxiv": SourceInfo("bioRxiv", "US", "en", "science", "Препринты биологии (свежая выдача)"),
    "openreview": SourceInfo("OpenReview", "INT", "en", "science", "Статьи конференций по ИИ"),
    "zenodo": SourceInfo("Zenodo", "EU", "en", "science", "Исследовательские данные и отчёты CERN/OpenAIRE"),
    "nist_news": SourceInfo("NIST News", "US", "en", "news", "Новости Национального института стандартов США"),
    "mit_research_news": SourceInfo("MIT Research News", "US", "en", "news", "Новости исследований MIT"),
    "horizon_magazine": SourceInfo("Horizon Magazine", "EU", "en", "news", "Журнал программы Horizon Europe"),
    "github": SourceInfo("GitHub", "INT", "en", "community", "Репозитории кода"),
    "hacker_news": SourceInfo("Hacker News", "INT", "en", "community", "Обсуждения технологического сообщества"),
    "gdelt": SourceInfo("GDELT", "INT", "en", "news", "Мировые новости; страны задаются фильтром", True),
    "habr": SourceInfo("Хабр", "RU", "ru", "community", "Русскоязычные технические статьи"),
    "europe_pmc": SourceInfo("Europe PMC", "EU", "en", "science", "Биомедицинские публикации (EMBL-EBI)"),
    "google_news": SourceInfo("Google News", "INT", "en", "news",
                              "Новости по странам: у каждой выбранной страны своё издание", True),
    "semantic_scholar": SourceInfo("Semantic Scholar", "INT", "en", "science", "Научные статьи всех областей"),
    "doaj": SourceInfo("DOAJ", "INT", "en", "science",
                       "Журналы открытого доступа; страна — страна журнала", True),
    "cyberleninka": SourceInfo("КиберЛенинка", "RU", "ru", "science", "Российские научные журналы"),
    "hal": SourceInfo("HAL", "FR", "en", "science", "Открытый архив французских исследований"),
    "osti": SourceInfo("OSTI", "US", "en", "science", "Отчёты и статьи Минэнерго США"),
    "nasa_ntrs": SourceInfo("NASA NTRS", "US", "en", "science", "Технические отчёты NASA"),
    "dblp": SourceInfo("dblp", "INT", "en", "science", "Библиография информатики (только год)"),
    "stack_exchange": SourceInfo("Stack Overflow", "INT", "en", "community", "Вопросы разработчиков"),
    "huggingface": SourceInfo("Hugging Face", "INT", "en", "community", "Модели машинного обучения"),
    "chemrxiv": SourceInfo("ChemRxiv", "INT", "en", "science", "Препринты химии"),
    "openaire": SourceInfo("OpenAIRE", "EU", "en", "science",
                           "Граф исследований Евросоюза: статьи тысяч журналов и репозиториев"),
    "jstage": SourceInfo("J-STAGE", "JP", "en", "science", "Японские научные журналы"),
    "npm": SourceInfo("npm", "INT", "en", "community", "Пакеты JavaScript: что публикуют и обновляют разработчики"),
}
if set(SOURCE_INFO) != set(SOURCE_IDS):
    raise RuntimeError("Каталог источников не совпадает со списком источников")

# Страны, которые предлагает панель. Порядок — порядок показа.
COUNTRIES: dict[str, str] = {
    "INT": "Международные", "RU": "Россия", "US": "США", "EU": "Евросоюз", "GB": "Великобритания",
    "DE": "Германия", "FR": "Франция", "CN": "Китай", "JP": "Япония", "KR": "Южная Корея",
    "IN": "Индия", "CA": "Канада", "AU": "Австралия", "BR": "Бразилия", "IL": "Израиль",
    "IT": "Италия", "ES": "Испания", "NL": "Нидерланды", "CH": "Швейцария", "SG": "Сингапур",
}
EU_MEMBERS = frozenset("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())
# Без выбора стран многострановые источники спрашивают эти издания.
DEFAULT_EDITIONS = ("US", "RU")
MAX_EDITIONS = 4
MIN_WEIGHT, MAX_WEIGHT = 0.4, 1.6


def _country(value: object) -> str:
    if not isinstance(value, str) or value not in COUNTRIES:
        raise ValueError("country")
    return value


@dataclass(frozen=True)
class SourcePolicy:
    """Какие источники и страны опрашивать; пустой список стран — все страны."""

    countries: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()
    # Доли сбора по источникам, которым программа научилась на своих анализах.
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for country in self.countries:
            _country(country)
        if len(set(self.countries)) != len(self.countries):
            raise ValueError("countries")
        if any(source not in SOURCE_INFO for source in self.disabled) or len(set(self.disabled)) != len(self.disabled):
            raise ValueError("disabled")
        for source, weight in self.weights.items():
            if (source not in SOURCE_INFO or isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not math.isfinite(weight) or not MIN_WEIGHT <= weight <= MAX_WEIGHT):
                raise ValueError("weights")

    @property
    def everywhere(self) -> bool:
        return not self.countries

    def _selected(self, country: str) -> bool:
        if self.everywhere:
            return True
        if country in self.countries:
            return True
        return country in EU_MEMBERS and "EU" in self.countries

    def allows(self, source: str) -> bool:
        if source in self.disabled:
            return False
        info = SOURCE_INFO[source]
        if info.multi_country:
            return self.everywhere or any(country != "INT" for country in self.countries) or "INT" in self.countries
        return self._selected(info.country)

    def skip_reason(self, source: str) -> str | None:
        if source in self.disabled:
            return "disabled_by_owner"
        if not self.allows(source):
            return "country_filter"
        return None

    def editions(self, available: Iterable[str]) -> tuple[str, ...]:
        """Страны, в изданиях которых ищет многострановой источник."""
        known = tuple(available)
        if self.everywhere or self.countries == ("INT",):
            return tuple(country for country in DEFAULT_EDITIONS if country in known)[:MAX_EDITIONS]
        chosen = [country for country in self.countries if country in known]
        if "EU" in self.countries:
            chosen.extend(country for country in ("DE", "FR", "IT", "ES", "NL") if country in known)
        return tuple(dict.fromkeys(chosen))[:MAX_EDITIONS]

    def country_filter(self) -> tuple[str, ...]:
        """Страны для фильтра внутри многострановой выдачи; пусто — без фильтра."""
        if self.everywhere or "INT" in self.countries:
            return ()
        countries = [country for country in self.countries if country != "EU"]
        if "EU" in self.countries:
            countries.extend(sorted(EU_MEMBERS))
        return tuple(dict.fromkeys(countries))

    def weight(self, source: str) -> float:
        return float(self.weights.get(source, 1.0))

    def to_json(self) -> dict[str, Any]:
        return {"version": POLICY_VERSION, "countries": list(self.countries), "disabled": list(self.disabled),
                "weights": {source: round(float(value), 4) for source, value in sorted(self.weights.items())}}

    def recorded(self) -> dict[str, Any]:
        """Что записывается в запуск: правило владельца без обученных долей."""
        return {"version": POLICY_VERSION, "countries": list(self.countries), "disabled": list(self.disabled)}

    @classmethod
    def from_json(cls, value: object) -> SourcePolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict) or not set(value) <= {"version", "countries", "disabled", "weights"}:
            raise ValueError("policy")
        countries, disabled = value.get("countries", []), value.get("disabled", [])
        weights = value.get("weights", {})
        if (not isinstance(countries, list) or not isinstance(disabled, list) or not isinstance(weights, dict)
                or len(countries) > len(COUNTRIES) or len(disabled) > len(SOURCE_INFO)
                or not all(isinstance(item, str) for item in disabled)):
            raise ValueError("policy")
        return cls(tuple(_country(item) for item in countries), tuple(disabled),
                   {str(key): item for key, item in weights.items()})


def load_policy(data_dir: Path | None) -> SourcePolicy:
    """Правило владельца из профиля; повреждённый файл — все источники и страны."""
    if data_dir is None:
        return SourcePolicy()
    try:
        return SourcePolicy.from_json(json.loads((Path(data_dir) / POLICY_FILE).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return SourcePolicy()


def save_policy(data_dir: Path, policy: SourcePolicy) -> None:
    path = Path(data_dir) / POLICY_FILE
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(policy.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def catalogue(policy: SourcePolicy) -> list[dict[str, Any]]:
    """Источники для панели: страна, язык, включён ли он при текущем правиле."""
    return [{"source_id": source, "label": info.label, "country": info.country,
             "country_label": COUNTRIES.get(info.country, info.country), "language": info.language,
             "group": info.group, "description": info.description, "multi_country": info.multi_country,
             "enabled": source not in policy.disabled, "active": policy.allows(source),
             "skip_reason": policy.skip_reason(source), "weight": round(policy.weight(source), 3)}
            for source, info in SOURCE_INFO.items()]

