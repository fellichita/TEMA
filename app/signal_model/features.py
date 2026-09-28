"""Интерпретируемые признаки слабого сигнала из текстовых полей описания технологии.

Каждый признак следует из критериев ТЗ: ранняя стадия, растущая динамика,
отсутствие сформированного рынка и лидеров, отсутствие маркетингового хайпа,
новизна. Словари признаков заданы заранее по этим критериям, а не подобраны
под данные. Ссылки на источники в признаки не входят: их наличие отличает
выданный датасет от примеров команды, а не слабый сигнал от зрелой технологии.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from app.signal_model.dataset import Technology

REFERENCE_YEAR = 2026


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    description: str


FEATURES: tuple[Feature, ...] = (
    Feature("stage_level", "Стадия развития",
            "1 — концепция/исследование, 2 — прототип, 3 — пилот, 4 — раннее внедрение, "
            "6 — массовое внедрение, зрелая технология или стандарт; переходы «А → Б» усредняются"),
    Feature("not_technology", "Не технология",
            "Стадия не распознана или описана как зонтичный термин, прогноз, тренд рынка труда"),
    Feature("trend_level", "Динамика упоминаний",
            "−1 — снижается, 0 — стабильно, 1 — растёт, 2 — растёт быстро"),
    Feature("early_market", "Признаки раннего рынка",
            "Выход из stealth, посевные раунды и серии A/B, первые клиенты и пилоты, нишевые обсуждения"),
    Feature("formed_market", "Признаки сформированного рынка",
            "Массовость, выраженные лидеры, поделённый или консолидированный рынок, десятилетия применения"),
    Feature("research", "Научная активность",
            "Препринты, статьи, обзоры, лаборатории и университеты в описании"),
    Feature("hype", "Хайп без подтверждений",
            "Маркетинговые заявления, пресс-релизы и обещания без внедрений и независимых подтверждений"),
    Feature("decline", "Спад интереса",
            "Снижение упоминаний, закрытие или приостановка проектов"),
    Feature("standards", "Стандарты и регулирование",
            "Отраслевые стандарты (ISO, IEC, IEEE, RFC, ГОСТ, PCI DSS) и обязательные требования"),
    Feature("first_mention_age", "Давность первого упоминания",
            f"Сколько лет прошло от самого раннего года в описании до {REFERENCE_YEAR}"),
    Feature("recent_share", "Доля свежих упоминаний",
            f"Доля упомянутых лет не раньше {REFERENCE_YEAR - 1}"),
    Feature("incumbent_share", "Доля крупных корпораций",
            "Доля известных крупных компаний среди названных игроков"),
)
FEATURE_NAMES = tuple(feature.name for feature in FEATURES)

_STAGES = (
    (6, r"массов|зрел|отраслев\w* стандарт|стандарт де-факто"),
    (4, r"раннее внедрение|ранние внедрения|раннее внедрен|первые (?:внедрения|поставки|полисы|интеграции)"
        r"|ранн\w+ (?:серия|поставки|пилоты)|мелкосерийн|early adoption|ограниченн\w+ .*доступ"),
    (3, r"пилот|pilot"),
    (2, r"прототип|poc|proof of concept|testbed|лабораторн\w+ испытан"),
    (1, r"концепц|исследован|research|concept"),
)
_FAMILIES = {
    "early_market": r"stealth|pre-?seed|\bseed\b|series [a-c]\b|серии? [abc]\b|посевн|раунд|стартап|вышл\w* из"
                    r"|первы[ехм] (?:клиент|пилот|внедрен|поставк|полис|интеграц)|нишев|почти не обсужда"
                    r"|единичн|ещё нет|пока нет|только начина",
    "formed_market": r"массов|повсеместн|выраженн\w+ лидер|лидер\w* рынк|доминир|поделён|консолидир"
                     r"|сформированн\w+ рын|де-факто|десятилети|у всех|каждой? (?:организац|отрасл|эмитент)"
                     r"|сотни тысяч|миллион\w* пользоват|стандартн\w+ (?:функци|част|инструмент|способ|процесс)",
    "research": r"препринт|arxiv|стать[яиейю]|обзор|nature|science|лаборатор|университет|исследовател",
    "hype": r"маркетинг|пресс-релиз|обещан|заявлени|громк|хайп|без (?:подтвержд|внедрен|измерим|техническ)"
            r"|не подтвержд|в медиа|демо-ролик|презентаци",
    "decline": r"снижа|спад|упал|закрыт|приостановл|свёрнут|не получил\w* развит|растворил|сошла|не привел",
    "standards": r"\biso\b|\biec\b|\bieee\b|\brfc\b|гост|pci dss|стандарт|директив|регулир|обязател",
}
_INCUMBENTS = (
    "microsoft", "google", "amazon", "aws", "apple", "ibm", "oracle", "sap", "siemens", "abb", "fanuc", "kuka",
    "yaskawa", "schneider", "honeywell", "emerson", "rockwell", "nvidia", "intel", "qualcomm", "samsung",
    "visa", "mastercard", "cisco", "huawei", "сбер", "яндекс", "т-банк", "втб", "альфа-банк", "касперск",
    "palo alto", "fortinet", "check point", "crowdstrike", "splunk", "dji", "hikvision", "dahua",
    "intuitive surgical", "medtronic", "bosch", "continental", "mobileye", "meta", "dassault", "ptc",
    "johnson controls", "broadcom", "akamai", "cloudflare", "equinix", "databricks", "snowflake",
    "binance", "coinbase", "klarna", "fico", "equifax", "openai", "anthropic", "tesla", "mitsubishi",
    "hp inc", "dell", "jpmorgan", "goldman", "swift",
)
_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)")


def _text(technology: Technology) -> str:
    return " ".join((technology.title, technology.rationale, technology.stage, technology.trend)).casefold()


def stage_level(stage: str) -> float | None:
    """Средний уровень упомянутых стадий; текст в скобках — пояснение, а не стадия."""
    text = re.sub(r"\([^)]*\)", " ", stage.casefold())
    if "не технология" in text:
        return None
    levels = []
    for segment in re.split(r"→|->|/|\+", text):
        for level, pattern in _STAGES:
            if re.search(pattern, segment):
                levels.append(level)
                break
    return sum(levels) / len(levels) if levels else None


def trend_level(trend: str) -> float | None:
    text = trend.casefold()
    head = re.split(r"—|:|;", text, maxsplit=1)[0]
    for part in (head, text):
        if re.search(r"снижа|спад|падает", part):
            return -1.0
        growing, stable = "раст" in part, "стабильн" in part
        if stable and (growing or "ускорен" in part):
            return 0.5
        if growing and "быстро" in part:
            return 2.0
        if growing:
            return 1.0
        if stable:
            return 0.0
    return None


def _companies(value: str) -> list[str]:
    plain = re.sub(r"\([^)]*\)", " ", value)
    return [item.strip() for item in re.split(r"[,;]", plain) if len(item.strip()) > 1]


def extract(technology: Technology) -> dict[str, float | None]:
    """Значения признаков; None — признак не измерен (заполняется медианой обучения)."""
    text = _text(technology)
    values: dict[str, float | None] = {
        "stage_level": stage_level(technology.stage),
        "trend_level": trend_level(technology.trend),
    }
    values["not_technology"] = 1.0 if values["stage_level"] is None else 0.0
    for name, pattern in _FAMILIES.items():
        values[name] = math.log1p(len(re.findall(pattern, text)))
    years = [int(year) for year in _YEAR.findall(" ".join((technology.rationale, technology.trend, technology.stage)))]
    values["first_mention_age"] = float(REFERENCE_YEAR - min(years)) if years else None
    values["recent_share"] = sum(year >= REFERENCE_YEAR - 1 for year in years) / len(years) if years else None
    companies = _companies(technology.companies)
    incumbents = sum(any(name in company.casefold() for name in _INCUMBENTS) for company in companies)
    values["incumbent_share"] = incumbents / len(companies) if companies else None
    return values


def describe(name: str, value: float | None) -> str:
    """Понятное значение признака для объяснения конкретного решения."""
    if value is None:
        return "нет данных"
    if name == "stage_level":
        return {1: "концепция", 2: "прототип", 3: "пилот", 4: "раннее внедрение", 6: "массовое/зрелое"}.get(
            round(value), f"между стадиями ({value:.1f})".replace(".", ","))
    if name == "trend_level":
        return {-1: "снижается", 0: "стабильно", 1: "растёт", 2: "растёт быстро"}.get(
            round(value), "стабильно с ростом")
    if name == "not_technology":
        return "да" if value else "нет"
    if name in _FAMILIES:
        count = round(math.expm1(value))
        return f"маркеров: {count}"
    if name == "first_mention_age":
        return f"{value:.0f} лет"
    return f"{value:.0%}"
