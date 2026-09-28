"""Точное попадание в тему запроса: лексический профиль темы и итоговая релевантность.

Раньше лента материала принималась, если в ней встретилось любое одно слово
запроса («state» из «solid-state batteries» совпадает почти с чем угодно), а
ТОП публикаций веба брал 15 самых свежих записей и только потом оценивал их
смысл. Здесь собраны правила, по которым материал относится к теме:

* профиль темы строится из самого плана анализа: русская и английская
  формулировки, синонимы и поднаправления — каждая даёт свой набор терминов;
* термин узнаётся по основе слова (грубый стеммер для русского и английского),
  общие слова («system», «technology», «метод») весят меньше предметных;
* покрытие считается отдельно для каждой формулировки и берётся лучшее;
  точная фраза формулировки в заголовке или аннотации — сильнейший признак;
* исключения плана понижают оценку материала, который не покрывает тему целиком;
* итог объединяет лексику со смысловой близостью локальной модели E5 (та же
  шкала, что у этапа обнаружения: косинус 0,80 — порог включения) и, если
  программа уже обучилась на своих анализах, с оценкой обученной модели.

Оценка — упорядочивающая эвристика для отбора, а не вероятность истинности.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, TypeVar

RELEVANCE_VERSION = "topic-relevance/1.0.0"

# Шкала смысловой близости та же, что у оценки публикаций и обнаружения.
COSINE_ZERO = 0.75
COSINE_FULL = 0.90
# Порог включения документа на этапе обнаружения; здесь — «явно по теме».
SEMANTIC_RELEVANT = 0.82
# Ниже этого косинуса и без лексического совпадения материал не по теме.
SEMANTIC_OFF_TOPIC = 0.77
RELEVANT_SCORE = 0.40
WEAK_SCORE = 0.25
# Лента (RSS, свежие выдачи) принимает материал только при таком покрытии темы.
FEED_MATCH = 0.67
# Радар технологий обходится без смысловой модели: ему нужен запас, иначе
# пропадут материалы с синонимами, которых нет в профиле.
RADAR_MATCH = 0.34
RADAR_MIN_POOL = 25

Decision = Literal["relevant", "weak", "off_topic"]

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC = re.compile(r"[а-яё]")

# Служебные слова не участвуют в покрытии.
_STOP = frozenset("""
a an the of for and or in on to with by from at as via using based toward towards into over under
its their our this these that those is are was were be been being we how what why which who
yet but than more most less very such also can may new novel recent current future emerging all
и в во на с со к ко по о об от до из за для при без над под про через или либо а но не ни
как что это эти этот эта его её их также так же ли бы у
""".split())
# Общие слова: встречаются во множестве чужих материалов, поэтому весят меньше.
_GENERIC = frozenset("""
system systems technology technologies technique techniques method methods approach approaches
application applications state model models data network networks device devices process processes
material materials development research study analysis design performance based high low power
energy control management platform solution solutions tool tools service services use
система системы технология технологии метод методы подход применение модель модели данные сеть
сети устройство устройства процесс материал материалы разработка исследование анализ решение
решения управление платформа сервис использование новые новый
""".split())
_GENERIC_WEIGHT = 0.3
# Русские окончания для грубого стемминга, от длинных к коротким.
_RU_ENDINGS = tuple(sorted("""
иями ями ами ого его ому ему ыми ими ими иях ах ях ов ев ей ой ый ий ая яя ое ее ые ие ую юю ом ем
ам ям ия ье ья ью ы и а я о е у ю ь й
""".split(), key=len, reverse=True))


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(str(text or "").casefold().replace("ё", "е")))


def stem(word: str) -> str:
    """Основа слова: хватает, чтобы «батареи» и «батарея», «sensors» и «sensor» совпали."""
    word = word.casefold().replace("ё", "е")
    if _CYRILLIC.search(word):
        for ending in _RU_ENDINGS:
            if len(word) - len(ending) >= 4 and word.endswith(ending):
                word = word[:-len(ending)]
                break
        return word[:7]
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("es") and len(word) > 4 and word[-3] in "sxz":
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 5:
            word = word[:-len(suffix)]
            break
    return word[:8]


def _stems(text: str) -> list[str]:
    return [stem(word) for word in normalize(text).split()]


@dataclass(frozen=True)
class Formulation:
    """Одна формулировка темы: её термины с весами и точная фраза."""

    text: str
    terms: tuple[tuple[str, float], ...]
    phrase: tuple[str, ...]

    @property
    def weight(self) -> float:
        return sum(weight for _, weight in self.terms)


def _formulation(text: str) -> Formulation | None:
    words = [word for word in normalize(text).split() if word not in _STOP and len(word) >= 2]
    if not words:
        return None
    terms: dict[str, float] = {}
    for word in words:
        key = stem(word)
        terms[key] = max(terms.get(key, 0.0), _GENERIC_WEIGHT if word in _GENERIC else 1.0)
    phrase = tuple(stem(word) for word in words)
    return Formulation(text=text, terms=tuple(terms.items()), phrase=phrase if len(phrase) >= 2 else ())


@dataclass(frozen=True)
class TopicProfile:
    formulations: tuple[Formulation, ...]
    exclusions: tuple[tuple[str, ...], ...] = field(default=())

    @property
    def empty(self) -> bool:
        return not self.formulations

    @property
    def broad(self) -> bool:
        """Тема из одних общих слов («technology»): по словам её не отличить от чужой."""
        return all(weight < 1.0 for formulation in self.formulations for _, weight in formulation.terms)

    @classmethod
    def build(cls, query: str | None, english_query: str | None = None, *,
              synonyms: Iterable[str] = (), subdirections: Iterable[str] = (),
              exclusions: Iterable[str] = ()) -> TopicProfile:
        texts = [text for text in (query, english_query, *synonyms, *subdirections)
                 if isinstance(text, str) and text.strip()]
        formulations: list[Formulation] = []
        seen: set[tuple[str, ...]] = set()
        for text in texts:
            formulation = _formulation(text[:1000])
            if formulation is None:
                continue
            key = tuple(sorted(term for term, _ in formulation.terms))
            if key in seen:
                continue
            seen.add(key)
            formulations.append(formulation)
            if len(formulations) == 24:
                break
        blocked = []
        for text in exclusions:
            if isinstance(text, str) and text.strip():
                words = tuple(stem(word) for word in normalize(text).split() if word not in _STOP)
                if words:
                    blocked.append(words)
        return cls(tuple(formulations), tuple(blocked[:40]))

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any] | Any) -> TopicProfile:
        """Профиль по плану анализа — словарю из результата или объекту QueryPlan."""
        def value(name: str) -> Any:
            return plan.get(name) if isinstance(plan, Mapping) else getattr(plan, name, None)

        def texts(name: str) -> list[str]:
            raw = value(name)
            return [item for item in raw if isinstance(item, str)] if isinstance(raw, (list, tuple)) else []

        return cls.build(value("original_query"), value("english_query"), synonyms=texts("synonyms"),
                         subdirections=texts("subdirections"), exclusions=texts("exclusions"))


@dataclass(frozen=True)
class LexicalMatch:
    score: float
    coverage_title: float
    coverage_text: float
    phrase_title: bool
    phrase_text: bool
    excluded: bool


def _contains(sequence: Sequence[str], phrase: Sequence[str]) -> bool:
    size = len(phrase)
    if not size or size > len(sequence):
        return False
    first = phrase[0]
    return any(sequence[index] == first and tuple(sequence[index:index + size]) == tuple(phrase)
               for index in range(len(sequence) - size + 1))


def _coverage(formulation: Formulation, words: set[str]) -> float:
    total = formulation.weight
    if total <= 0:
        return 0.0
    return sum(weight for term, weight in formulation.terms if term in words) / total


def lexical_match(profile: TopicProfile, title: str | None, summary: str | None = None) -> LexicalMatch:
    """Насколько заголовок и аннотация покрывают тему по её лучшей формулировке."""
    title_stems = _stems(title or "")
    text_stems = title_stems + ["."] + _stems(summary or "")
    title_set, text_set = set(title_stems), set(text_stems)
    phrase_title = phrase_text = False
    coverage_title = coverage_text = 0.0
    for formulation in profile.formulations:
        coverage_title = max(coverage_title, _coverage(formulation, title_set))
        coverage_text = max(coverage_text, _coverage(formulation, text_set))
        if formulation.phrase:
            if _contains(title_stems, formulation.phrase):
                phrase_title = True
            if _contains(text_stems, formulation.phrase):
                phrase_text = True
    if phrase_title:
        score = 1.0
    elif phrase_text:
        score = max(0.85, 0.6 * coverage_title + 0.4 * coverage_text)
    else:
        score = 0.55 * coverage_title + 0.45 * coverage_text
        # Полное покрытие в тексте без фразы — всё равно уверенное совпадение.
        if coverage_text >= 0.999:
            score = max(score, 0.7)
    excluded = any(_contains(text_stems, words) for words in profile.exclusions)
    if excluded and coverage_text < 0.999 and not phrase_text:
        score *= 0.5
    return LexicalMatch(round(min(1.0, score), 4), round(coverage_title, 4), round(coverage_text, 4),
                        phrase_title, phrase_text, excluded)


def topic_match(profile: TopicProfile, title: str | None, summary: str | None = None, *,
                minimum: float = FEED_MATCH) -> bool:
    """Строгий фильтр ленты: фраза темы или покрытие не ниже порога (пустой профиль пропускает всё)."""
    if profile.empty or profile.broad:
        return True
    match = lexical_match(profile, title, summary)
    if match.phrase_text or match.score >= minimum:
        return True
    # Покрытие в аннотации засчитывается, если материал не про исключённую область.
    return not match.excluded and match.coverage_text >= minimum


def semantic_norm(cosine: float | None) -> float | None:
    if cosine is None or isinstance(cosine, bool) or not isinstance(cosine, (int, float)):
        return None
    if not math.isfinite(cosine):
        return None
    return max(0.0, min(1.0, (float(cosine) - COSINE_ZERO) / (COSINE_FULL - COSINE_ZERO)))


def combine(lexical: float, cosine: float | None, learned: float | None = None,
            learned_weight: float = 0.0) -> tuple[float, Decision]:
    """Итоговая релевантность и решение: по теме, пограничный материал или не по теме."""
    semantic = semantic_norm(cosine)
    if semantic is None:
        heuristic = lexical
    else:
        heuristic = 0.65 * semantic + 0.35 * lexical
    score = heuristic
    if learned is not None and 0 < learned_weight <= 1 and math.isfinite(learned):
        score = (1 - learned_weight) * heuristic + learned_weight * max(0.0, min(1.0, learned))
    score = round(max(0.0, min(1.0, score)), 4)
    if cosine is not None and semantic is not None:
        if cosine >= SEMANTIC_RELEVANT or score >= RELEVANT_SCORE:
            return score, "relevant"
        if cosine < SEMANTIC_OFF_TOPIC and lexical < 0.5:
            return score, "off_topic"
        return score, "weak" if score >= WEAK_SCORE else "off_topic"
    if score >= 0.5:
        return score, "relevant"
    return score, "weak" if score >= WEAK_SCORE else "off_topic"


PoolItem = TypeVar("PoolItem", bound=Mapping[str, Any])


def filter_pool(pool: Sequence[PoolItem], profile: TopicProfile, *,
                scorer: Any = None, minimum: float = RADAR_MATCH,
                minimum_pool: int = RADAR_MIN_POOL) -> list[PoolItem]:
    """Материалы по теме для радара технологий.

    Если по теме набралось меньше `minimum_pool`, добавляются ближайшие по
    оценке материалы с хоть каким-то совпадением; материалы без единого общего
    термина не добавляются никогда. Если совпадений нет вовсе (профиль не понял
    язык выдачи), возвращается вся выдача — радар не должен остаться пустым из-за
    фильтра. `scorer(item, lexical)` — необязательная обученная оценка 0…1 (без
    смысловой модели: радар стартует раньше неё); она смешивается с лексикой пополам.
    """
    if profile.empty or not pool:
        return list(pool)
    scored: list[tuple[float, int, bool]] = []
    for index, item in enumerate(pool):
        match = lexical_match(profile, str(item.get("title") or ""), str(item.get("summary") or ""))
        score = match.score
        if scorer is not None and score > 0:
            try:
                learned = scorer(item, match)
            except Exception:
                learned = None
            if isinstance(learned, (int, float)) and math.isfinite(learned):
                score = 0.5 * score + 0.5 * float(learned)
        scored.append((score, index, match.phrase_text or score >= minimum))
    if not any(score > 0 for score, _, _ in scored):
        return list(pool)
    chosen = {index for _, index, passed in scored if passed}
    if len(chosen) < minimum_pool:
        for score, index, _ in sorted(scored, key=lambda entry: (-entry[0], entry[1])):
            if len(chosen) >= minimum_pool or score <= 0:
                break
            chosen.add(index)
    return [item for index, item in enumerate(pool) if index in chosen]
