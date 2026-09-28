"""Сведения о кандидате: первое упоминание, общий объём и материалы из собранной выдачи."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
import re
from threading import Event, Lock
from time import monotonic
from typing import Any

from defusedxml import ElementTree
import httpx

from app.radar.phrases import NOT_PUBLICATIONS
from app.trend_confidence import SOURCE_RULES, trust_level
from app.trend_history import _ATOM, _OPENSEARCH, HistoryFetchError, _get, arxiv_get, phrase_pattern

SOURCE_TYPES = {
    "scientific_index": "научная публикация", "patent": "патент", "preprint": "препринт",
    "research_artifact": "исследовательский материал", "analytical_report": "аналитический отчёт",
    "institutional_news": "новость научной организации", "repository": "репозиторий кода",
    "news_aggregate": "агрегатор новостей", "community": "сообщество",
}
# Слова стадий в найденных текстах: от поздней к ранней, берётся первая найденная.
STAGE_MARKERS = (
    ("Раннее внедрение", r"commerciali[sz]|deploy(?:ed|ment)|in production|on the market|product launch"
                         r"|customers?|mass production|внедрен|серийн"),
    ("Пилот", r"\bpilot\b|field (?:trial|test)|trial run|пилот"),
    ("Прототип/PoC", r"prototype|proof[- ]of[- ]concept|demonstrat|testbed|прототип"),
)
FUNDING = r"start-?up|raises?|raised|funding|series [a-c]\b|seed round|venture|стартап|раунд|инвест"
STANDARDS = r"\biso\b|\biec\b|\bieee\b|standard|стандарт"
ADVANTAGE = r"improv|enabl|reduc|increas|higher|lower|faster|outperform|achiev|efficien|lower cost|cheaper"
CASE = r"demonstrat|deploy|pilot|prototype|applied to|case study|we show|in practice|real-world|field"
INCUMBENTS = ("Samsung", "Toyota", "Panasonic", "LG", "CATL", "BYD", "Microsoft", "Google", "Amazon", "Apple",
              "IBM", "NVIDIA", "Intel", "Qualcomm", "Siemens", "ABB", "Bosch", "Huawei", "Meta", "OpenAI",
              "Tesla", "Sony", "Hitachi", "Toshiba", "Cisco", "Oracle", "SAP", "Visa", "Mastercard")


@dataclass(frozen=True)
class SourceView:
    """Карточка источника по требованиям ТЗ."""
    title: str
    url: str
    published: str
    source: str
    source_type: str
    language: str
    trust: str


@dataclass(frozen=True)
class PoolEvidence:
    items: tuple[SourceView, ...]
    stage: str | None
    funding_mentions: int
    standard_mentions: int
    incumbents: tuple[str, ...]
    description: str | None
    advantage: str | None
    case: str | None


def language(text: str) -> str:
    return "русский" if re.search(r"[а-яё]", text, re.IGNORECASE) else "английский"


def source_view(source_id: str, title: str, url: str, published: str) -> SourceView:
    rule = SOURCE_RULES.get(source_id)
    return SourceView(title, url, published, rule.label if rule else source_id,
                      SOURCE_TYPES.get(rule.source_class, "источник") if rule else "источник",
                      language(title), trust_level(source_id))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if 40 <= len(part.strip()) <= 400]


def pool_evidence(phrase: str, pool: Sequence[Mapping[str, object]], *, source_limit: int = 8) -> PoolEvidence:
    """Выжимки из подходящих текстов; ссылки — с фразой в заголовке работы."""
    pattern = phrase_pattern([phrase])
    matched = [item for item in pool if item.get("kind") not in NOT_PUBLICATIONS
               and pattern.search(f"{item.get('title') or ''} {item.get('summary') or ''}")]
    matched.sort(key=lambda item: (SOURCE_RULES.get(str(item.get("source_id")), SOURCE_RULES["hacker_news"])
                                   .coefficient, str(item.get("published_at") or "")), reverse=True)
    texts = [f"{item.get('title') or ''}. {item.get('summary') or ''}" for item in matched]
    joined = " ".join(texts)
    stage = next((name for name, marker in STAGE_MARKERS if re.search(marker, joined, re.IGNORECASE)), None)
    sentences = [sentence for text in texts for sentence in _sentences(text) if pattern.search(sentence)] or \
        [sentence for text in texts for sentence in _sentences(text)]

    def first(marker: str) -> str | None:
        return next((sentence for sentence in sentences if re.search(marker, sentence, re.IGNORECASE)), None)

    description = sentences[0] if sentences else None
    # An abstract may compare several sibling mechanisms. Its publication is
    # shown under the technology named in its own title, not under every
    # mechanism it mentions in passing.
    titled = [item for item in matched if pattern.search(str(item.get("title") or ""))]
    views = tuple(source_view(str(item.get("source_id")), str(item.get("title") or ""), str(item.get("url") or ""),
                              str(item.get("published_at") or item.get("publication_year") or ""))
                  for item in titled[:source_limit])
    return PoolEvidence(
        views, stage, len(re.findall(FUNDING, joined, re.IGNORECASE)),
        len(re.findall(STANDARDS, joined, re.IGNORECASE)),
        tuple(name for name in INCUMBENTS if re.search(rf"\b{re.escape(name)}\b", joined)),
        description, first(ADVANTAGE) if description else None, first(CASE) if description else None)


# OpenAlex отказывает при частых запросах; отказ превращал зрелую тему в «нишевую»
# по неполному запасному счёту. Поэтому запросы идут не чаще четырёх в секунду,
# а на отказ 429 — короткая пауза и повтор.
OPENALEX_INTERVAL_SECONDS = 0.25
OPENALEX_RETRIES = 2
OPENALEX_MAX_WAIT_SECONDS = 5.0
_OPENALEX_LOCK = Lock()
_OPENALEX_LAST = [0.0]
# Когда OpenAlex велит прийти через долгое время (исчерпан дневной бюджет без
# ключа: Retry-After — до полуночи UTC), его не спрашивают до этого срока. Иначе
# каждая из 60 фраз ТОПа ждала 5 секунд и повторяла отказ: замер 28.09.2026 —
# 128 с из 202 на пустые ожидания.
_OPENALEX_CLOSED_UNTIL = [0.0]
OPENALEX_LONGEST_PAUSE_SECONDS = 3600.0


def openalex_closed_seconds() -> float:
    """Сколько ещё OpenAlex закрыт для этого процесса; 0 — открыт."""
    return max(0.0, _OPENALEX_CLOSED_UNTIL[0] - monotonic())
# Год считается годом появления фразы, когда в нём не меньше трёх работ:
# одиночная статья десятилетней давности не делает тему старой.
FIRST_YEAR_MINIMUM = 3


@dataclass(frozen=True)
class FirstMention:
    # Самая ранняя дата, когда фраза встречается в названиях arXiv; None — не найдено.
    earliest: date | None
    arxiv_total: int | None
    europe_pmc_total: int | None
    # Работы с фразой в названии по годам во всей науке (OpenAlex); None — источник не ответил.
    openalex_years: tuple[tuple[int, int], ...] | None = None

    @property
    def openalex_total(self) -> int | None:
        return None if self.openalex_years is None else sum(count for _, count in self.openalex_years)

    @property
    def openalex_first_year(self) -> int | None:
        return next((year for year, count in self.openalex_years or () if count >= FIRST_YEAR_MINIMUM), None)


def openalex_years(client: httpx.Client, phrase: str, cancel: Event,
                   api_key: str | None = None) -> tuple[tuple[int, int], ...] | None:
    """Один запрос с группировкой по годам: объём и год появления фразы во всей науке."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"filter": f'title.search:"{phrase.replace(chr(34), "")}"', "group_by": "publication_year",
              "per_page": "200"}
    if openalex_closed_seconds() > 0:
        return None
    try:
        for attempt in range(OPENALEX_RETRIES + 1):
            with _OPENALEX_LOCK:
                wait = _OPENALEX_LAST[0] + OPENALEX_INTERVAL_SECONDS - monotonic()
                if wait > 0 and cancel.wait(wait):
                    return None
                _OPENALEX_LAST[0] = monotonic()
            if cancel.is_set():
                return None
            response = client.get("https://api.openalex.org/works", timeout=30.0, headers=headers, params=params)
            if response.status_code != 429 or attempt == OPENALEX_RETRIES:
                break
            try:
                pause = float(response.headers.get("Retry-After", "1"))
            except ValueError:
                pause = 1.0
            if pause > OPENALEX_MAX_WAIT_SECONDS:
                # Долгий отказ повтором не лечится: закрыть OpenAlex до срока для всех фраз.
                _OPENALEX_CLOSED_UNTIL[0] = max(_OPENALEX_CLOSED_UNTIL[0],
                                                monotonic() + min(pause, OPENALEX_LONGEST_PAUSE_SECONDS))
                return None
            if cancel.wait(min(max(pause, 0.5), OPENALEX_MAX_WAIT_SECONDS)):
                return None
        if response.status_code != 200 or len(response.content) > 2_000_000:
            return None
        groups = response.json().get("group_by") or []
        years = sorted((int(item["key"]), int(item["count"])) for item in groups
                       if str(item.get("key", "")).isdigit() and type(item.get("count")) is int)
        return tuple(years)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None


_UNSET = object()


def first_mention(client: httpx.Client, forms: Sequence[str], cancel: Event,
                  openalex_key: str | None = None, years: object = _UNSET, *,
                  arxiv: Any = None) -> FirstMention:
    """Годы OpenAlex; если OpenAlex не ответил — ранний препринт arXiv и объём Europe PMC.

    `years` — уже полученный ответ OpenAlex, чтобы не спрашивать дважды; `arxiv` —
    пакетный сборщик, отвечающий по нескольким фразам одним запросом.
    """
    if years is _UNSET:
        years = openalex_years(client, forms[0], cancel, openalex_key)
    if years is not None:
        return FirstMention(None, None, None, years)  # type: ignore[arg-type]
    earliest = arxiv_total = europe_total = None
    clean = [form.replace('"', "") for form in forms]
    arxiv_query = " OR ".join(f'ti:"{form}"' for form in clean)
    europe_query = " OR ".join(f'TITLE:"{form}"' for form in clean)
    try:
        batched = arxiv.first(forms) if arxiv is not None else None
        if batched is not None:
            earliest, arxiv_total = batched
        else:
            root = ElementTree.fromstring(arxiv_get(client, {
                "search_query": arxiv_query, "start": "0", "max_results": "1",
                "sortBy": "submittedDate", "sortOrder": "ascending"}, cancel))
            arxiv_total = int(root.findtext(_OPENSEARCH + "totalResults") or 0)
            stamp = root.findtext(f"{_ATOM}entry/{_ATOM}published") or ""
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", stamp):
                earliest = date.fromisoformat(stamp[:10])
    except (HistoryFetchError, ValueError, ElementTree.ParseError):
        pass
    try:
        payload = json.loads(_get(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search", {
            "query": europe_query, "resultType": "idlist", "pageSize": "1", "format": "json"}, cancel))
        europe_total = int(payload.get("hitCount") or 0)
    except (HistoryFetchError, ValueError):
        pass
    return FirstMention(earliest, arxiv_total, europe_total)
