"""Кандидаты в технологии: устойчивые фразы из названий найденных материалов.

Фраза из 2–4 слов становится кандидатом, если встречается в названии хотя бы
одного материала и в названиях или аннотациях нескольких независимых
материалов из разных источников. Фраза сразу служит названием технологии и
поисковым запросом для её истории, поэтому языковая модель здесь не нужна.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
from typing import cast

# Two independent sources can identify a narrow mechanism. Its history and
# model assessment still decide whether it belongs in the TOP.
MIN_DOCUMENTS = 2
MIN_SOURCES = 2
# Widely repeated names need visible scope support. This catches homonyms such
# as an atmospheric "lightning network" mixed with the Bitcoin network, while
# leaving small, specific mechanisms open to the later history/model review.
SCOPE_REVIEW_MIN_TITLES = 8
SCOPE_REVIEW_MIN_SHARE = 0.1
# Фразу из одного источника принимаем, если она очень устойчива.
SINGLE_SOURCE_DOCUMENTS = 5
# Более длинная фраза заменяет короткую, если сохраняет большую часть её упоминаний.
SUBSUMPTION_SHARE = 0.6
# A parent phrase can be covered by several distinct, smaller technologies;
# no single child then reaches the ordinary subsumption threshold.
COLLECTIVE_SUBSUMPTION_SHARE = 0.8
NOT_PUBLICATIONS = frozenset({"peer-review"})
EXTENSION_SHARE = 0.6

_STOP = frozenset("""
a an the of for and or in on to with by from at as via using based toward towards into over under
its their our this these that those is are was were be been being we here how what why which who
yet but however than more most less very such also can may new novel study studies analysis approach
approaches method methods review survey overview perspective perspectives case cases toward
high low enhanced improved efficient effective efficiency application applications recent advances
advance progress challenge challenges opportunity opportunities issue issues role impact effect
effects promising potential future next-generation state-of-the-art first report reports all
between among across within versus vs without while where when
""".split())
_INNER = frozenset({"of", "for", "in", "on", "with"})
# Последнее слово-свойство: такая фраза описывает характеристику, а не технологию.
_PROPERTY_HEADS = frozenset("""
conductivity density stability property properties performance efficiency transport capacity mechanism
mechanisms behavior behaviour strategy strategies design designs development temperature condition
conditions characteristic characteristics evaluation assessment measurement investigation insight
insights understanding problem problems limitation limitations benefit benefits cost costs
requirement requirements trend trends result results outcome outcomes rate rates level levels value
values factor factors parameter parameters quality safety reliability durability risk risks
implementation window ionic electrochemical thermal mechanical chemical electronic optical practical
""".split())
_GENERIC = frozenset({
    "machine learning", "deep learning", "neural network", "artificial intelligence", "large language model",
    "language model", "data analysis", "case study", "literature review", "systematic review",
    "proposed method", "experimental result", "real world", "real-world data", "open source",
    "energy storage", "electric vehicle", "climate change", "supply chain", "decision making",
})
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SEGMENT = re.compile(r"[.,;:!?()\[\]{}\"«»|/]+|\s[-–—]\s")


def singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
        return token[:-1]
    return token


def variants(surface: str) -> tuple[str, ...]:
    """Единственное и множественное число последнего слова: точный поиск источников их различает."""
    words = surface.split()
    last = words[-1]
    single = singular(last)
    if single.endswith("y") and len(single) > 3:
        plural = single[:-1] + "ies"
    elif single.endswith(("s", "x", "ch", "sh")):
        plural = single + "es"
    else:
        plural = single + "s"
    return tuple(dict.fromkeys(" ".join(words[:-1] + [form]) for form in (last, single, plural)))


def _segments(text: str) -> list[list[str]]:
    return [_TOKEN.findall(part) for part in _SEGMENT.split(text.casefold())]


def _ngrams(tokens: Sequence[str]) -> Iterable[tuple[str, tuple[str, ...]]]:
    """Нормализованная фраза и её исходная форма."""
    for size in (2, 3, 4):
        for start in range(len(tokens) - size + 1):
            words = tokens[start:start + size]
            if words[0] in _STOP or words[-1] in _STOP or words[0] in _INNER or words[-1] in _INNER:
                continue
            if any(word in _STOP and word not in _INNER for word in words[1:-1]):
                continue
            if sum(word in _INNER for word in words) > 1 or singular(words[-1]) in _PROPERTY_HEADS:
                continue
            # «model-based», «ai-driven» — определение без главного слова, а не технология.
            if words[-1].endswith(("-based", "-driven", "-enabled", "-aware", "-powered", "-like")):
                continue
            if all(len(word) <= 2 for word in words):
                continue
            yield " ".join(singular(word) for word in words), tuple(words)


def _normalized_grams(text: str) -> set[str]:
    return {normal for segment in _segments(text) for normal, _ in _ngrams(segment)}


@dataclass(frozen=True)
class Candidate:
    phrase: str
    # Как фраза чаще всего написана в названиях: её показываем и ищем.
    surface: str
    documents: tuple[str, ...]
    sources: tuple[str, ...]


def mine_candidates(pool: Sequence[Mapping[str, object]], *, query_terms: Iterable[str] = (),
                    limit: int = 30) -> list[Candidate]:
    """Фразы-кандидаты по убыванию устойчивости; фраза самого запроса не кандидат."""
    # Each query wording is a direction of its own. A union of every synonym's
    # words can accidentally erase a narrower combination that the user never
    # requested as a whole.
    query_phrases = [{singular(part) for token in _TOKEN.findall(str(term).casefold())
                      for part in token.split("-")} for term in query_terms]
    primary_scope = (query_phrases[0] - _STOP - _INNER) if query_phrases else set()
    review_scope = len(primary_scope) >= 3
    in_titles: dict[str, Counter[str]] = defaultdict(Counter)
    following: dict[str, Counter[str]] = defaultdict(Counter)
    documents: dict[str, set[str]] = defaultdict(set)
    title_documents: dict[str, set[str]] = defaultdict(set)
    scoped_titles: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    for item in pool:
        # Рецензии и решения редакции Crossref повторяют название статьи, но публикациями не являются.
        if item.get("kind") in NOT_PUBLICATIONS:
            continue
        key = re.sub(r"[^0-9a-z]+", "", str(item.get("title") or "").casefold())[:160]
        if key in seen:
            continue
        seen.add(key)
        identifier = str(item.get("publication_id") or item.get("url") or item.get("title"))
        title = str(item.get("title") or "")
        title_phrases: set[str] = set()
        for segment in _segments(title):
            for normal, words in _ngrams(segment):
                in_titles[normal][" ".join(words)] += 1
                title_phrases.add(normal)
            for size in (2, 3):
                for start in range(len(segment) - size):
                    normal = " ".join(singular(word) for word in segment[start:start + size])
                    following[normal][singular(segment[start + size])] += 1
        scoped = (review_scope and primary_scope <= {
            singular(part) for token in _TOKEN.findall(
                (title + " " + str(item.get("summary") or "")).casefold()) for part in token.split("-")})
        for normal in title_phrases:
            title_documents[normal].add(identifier)
            if scoped:
                scoped_titles[normal].add(identifier)
        for normal in _normalized_grams(title + " . " + str(item.get("summary") or "")):
            documents[normal].add(identifier)
            source_ids = cast(Iterable[object], item.get("source_ids") or (item.get("source_id"),))
            sources[normal].update(str(source) for source in source_ids if source)
    def extended(phrase: str) -> str:
        """Обрывок вроде «sodium metal» достраивается словом, которое почти всегда идёт следом."""
        while len(phrase.split()) < 4:
            options = following.get(phrase)
            if not options:
                break
            word, count = options.most_common(1)[0]
            longer = f"{phrase} {word}"
            if count < MIN_DOCUMENTS or count < EXTENSION_SHARE * sum(options.values()) or longer not in in_titles:
                break
            phrase = longer
        return phrase

    scored: dict[str, int] = {}
    for phrase in list(in_titles):
        phrase = extended(phrase)
        count, source_count = len(documents[phrase]), len(sources[phrase])
        tokens = {singular(part) for token in phrase.split() for part in token.split("-")}
        if count < MIN_DOCUMENTS or (source_count < MIN_SOURCES and count < SINGLE_SOURCE_DOCUMENTS):
            continue
        if phrase in _GENERIC or any(tokens <= query | _INNER | _STOP for query in query_phrases if query):
            continue
        # A common phrase can describe unrelated fields. Require a measurable
        # link to the whole main query in its own title-bearing papers before
        # asking global phrase history, which cannot separate homonyms.
        # A new phrase with no literal overlap may still be a valid mechanism
        # (for example lithium iron phosphate in an EV battery search). Review
        # this lexical relation only when the name reuses one query concept but
        # omits at least two other concepts: the shape of a common homonym.
        if (review_scope and tokens & primary_scope and len(primary_scope - tokens) >= 2
                and len(title_documents[phrase]) >= SCOPE_REVIEW_MIN_TITLES
                and len(scoped_titles[phrase]) < SCOPE_REVIEW_MIN_SHARE * len(title_documents[phrase])):
            continue
        scored[phrase] = count
    kept: dict[str, int] = {}
    for phrase, count in sorted(scored.items(), key=lambda pair: (-len(pair[0].split()), -pair[1])):
        longer = [other for other in kept if f" {phrase} " in f" {other} "]
        if any(kept[other] >= SUBSUMPTION_SHARE * count for other in longer):
            continue
        if len(longer) >= 2 and len(set().union(*(documents[other] for other in longer))) >= \
                COLLECTIVE_SUBSUMPTION_SHARE * count:
            continue
        kept[phrase] = count
    ranked = sorted(kept, key=lambda phrase: (-kept[phrase] * (1 + 0.3 * (len(phrase.split()) - 2)), phrase))
    # Shared papers are not proof of identical mechanisms. Reviews and
    # abstracts often mention sibling technologies side by side; merging on
    # document overlap collapsed them into one broad direction. Contained
    # wording variants were already removed by the subsumption rule above.
    return [Candidate(phrase, in_titles[phrase].most_common(1)[0][0], tuple(sorted(documents[phrase])),
                      tuple(sorted(sources[phrase]))) for phrase in ranked[:limit]]
