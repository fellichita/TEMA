"""Frozen candidate definitions and evidence passports from actual archived text.

An AI may propose names, select quotations and write an explicitly unverified
interpretation. Only exact source quotations receive supported status. No claim
of technological novelty or independently replicated performance is inferred
from an author's use of the word 'novel'.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from functools import lru_cache
from threading import Lock
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.backend.contracts import DocumentRecord
from app.pilot.archive import DocumentArchive, RevisionSource
from app.pilot.contracts import LEGACY_SCOPE_RULE_VERSION, SCOPE_RULE_VERSION, Candidate, Claim, CorpusSnapshot, DocumentRevisionRef, Evidence, MethodologyVersion, QueryPlan, ScopeRuleVersion, TrendCard, content_hash, require_unique, verify_evidence_text
from app.pilot.llm import LlmCancelled, LlmClient, LlmError
from app.runtime.budget import BudgetError, BudgetExceeded, BudgetStateError
from app.runtime.jobs import RunContext, TaskCancelled, TaskFailure

TITLE_ADMISSION_VERSION = "title-phrase-admission/2.0.0"
LABEL_PROMPT_VERSION = "candidate-labels/3.3.0"
PASSPORT_PROMPT_VERSION = "extractive-passport/2.0.0"
PRIMARY_PASSPORT_PROMPT_VERSION = "extractive-primary-passport/3.0.0"
PRIMARY_GROUNDING_METHOD = "exact-contextual-quotation/3.0.0"
RAW_GROUNDING_METHOD = "exact-contextual-quotation/4.0.0"
RAW_PASSPORT_PROMPT_VERSION = "extractive-primary-passport/4.0.0"
RESEARCH_APPLICATION_METHOD = "verified-application/research/2.0.0"
LABEL_BATCH_SIZE = 4
LABEL_BATCH_REPRESENTATIVES = 8  # Bound actual evidence/output work, not only candidate count.
# A small local model continues the nearest example it can see, and the nearest
# one is the material itself: without this it answered by repeating the supplied
# cluster instead of naming it. Ignored by a provider that needs no example.
# The example carries the answer's shape for a model that reads it more closely
# than any instruction, so it has to show the shape actually being asked for:
# a chosen phrase_number and no written phrases. When it showed a "phrases" list
# against a schema that forbids one, schema failures went from 10 to 42 in a
# measured run — the model copied the example and was refused every time.
LOCAL_LABEL_EXAMPLE = (
    "Formatting example for a different cluster. Answer about the supplied clusters, never about this one. "
    "Here phrase_number 2 was chosen from a list whose second entry was «polymer membrane gas separation»:"
    + chr(10) +
    '{"candidates":[{"candidate_id":"the id supplied with that cluster",'
    '"label":"Мембранное разделение газов","definition":"Полимерная мембрана разделяет газовую смесь без охлаждения.",'
    '"phrase_number":2,"specificity":"specific_technology",'
    '"in_scope":true,"scope_support":[{"revision_id":"the 64-character id supplied with that document",'
    '"field":"title","quote":"Polymer membrane gas separation at high pressure"}],'
    '"scope_reason":"Каждая работа описывает мембранное разделение."}]}' + chr(10) +
    "Answer with phrase_number only; never write a phrases list. "
    "Never copy the input fields discovery_label, total_studies or documents into the answer: "
    "they are what you are naming, not what you return.")
# Measured on a real run of this application: with two clusters per call the
# 1.5B model failed every one of seven batches — it renamed the supplied ids,
# left the object unclosed or broke the shape. One cluster per call is the
# configuration that answers correctly.
LOCAL_LABEL_BATCH_SIZE = 1
# The application requires a quotation for every representative it shows the
# model. Measured on a real run, the 1.5B model names the cluster correctly and
# quotes one representative out of three, and the proposal is then rejected for
# the two it left out — eight candidates in, none out. Listing the required
# identifiers in the prompt did not change that and doubled the time. One
# representative is what this model can support, and the result says so.
LOCAL_LABEL_BATCH_REPRESENTATIVES = 1
# How many representatives of one cluster the local model is shown and has to
# quote. Every one of them needs its own quotation, and this model supports one.
LOCAL_LABEL_REPRESENTATIVES = 1
# How much of a document a small model is shown. The quotations it selects are
# still verified against the whole archived document, so a shorter excerpt costs
# nothing but the material it could have quoted from.
LOCAL_TITLE_CHARACTERS = 300
LOCAL_ABSTRACT_CHARACTERS = 700
_PROBLEM = re.compile(r"\b(challenges?|limitations?|bottlenecks?|limited|suffer|difficult|expensive|costly)\b|проблем|ограничен|сложност|недостат", re.IGNORECASE)
_ADVANTAGE = re.compile(r"\b(improv\w*|reduc\w*|enabl\w*|outperform\w*|efficient\w*|faster|superior|enhanc\w*)\b|улучш|сниж|повыш|превосход|позволяет", re.IGNORECASE)
_NEGATED_ADVANTAGE = re.compile(r"\b(not|never|fails?\s+to|unable\s+to|aims?|hopes?|intends?|could|might|may|potential\w*|hypothes\w*|suggest\w*|propos\w*|speculat\w*|negligible|insignificant|unproven|theoretic\w*|simulat\w*|predic\w*|first.principles)\b|\bне\s+(?:улучш|сниж|повыш|превосход|позвол)|гипотез|предполага|моделирован|пренебрежимо", re.IGNORECASE)
_BASELINE_ADVANTAGE = re.compile(r"\b(previous\w*|prior|earlier|conventional|baseline|existing|established|review|survey|roadmap)\b|предыдущ|ранее|обзор", re.IGNORECASE)
_EXPERIMENT = re.compile(r"\b(experiment\w*|measur\w*|laborator\w*|in vivo|in vitro)\b|эксперимент|измерен|лаборатор", re.IGNORECASE)
_ADVERSE_EFFECT = re.compile(r"\b(?:reduc\w*|lower\w*|decreas\w*)\s+(?:(?:the|its|our|sensor|device)\s+)*(?:sensitivity|accuracy|efficiency|selectivity|stability|reliability|performance)\b|\b(?:increas\w*|enhanc\w*)\s+(?:(?:the|its|our)\s+)*(?:error|noise|cost|latency|loss)\b", re.IGNORECASE)
_PRIMARY_RESULT = re.compile(
    r"\b(?:we|our|this\s+(?:work|study|paper))\b.{0,100}\b(?:measur\w*|test\w*|fabricat\w*|observ\w*|"
    r"demonstrat\w*|develop\w*|design\w*|implement\w*|introduc\w*|calculat\w*|deriv\w*|simulat\w*|"
    r"prove\w*|show\w*|found|find|report\w*|present\w*|evaluat\w*)\b|"
    r"\b(?:мы|нами)\b.{0,100}(?:измер|испыт|разработ|показ|получ|вычисл|изготов|наблюд|реализ)", re.I)
_SPECULATIVE_RESULT = re.compile(
    r"\b(?:could|might|may|would|will|hope\w*|aim\w*|intend\w*|propos\w*|future)\b"
    r"|\b(?:not|never|unable|failed)\b|\b(?:планируем|предполагаем|может|не)\b", re.I)
_RESULT_CONTRADICTION = re.compile(
    r"\b(?:our|these|the)\s+(?:results|experiments|measurements|findings)\s+(?:do|did)\s+not\s+"
    r"(?:support|confirm|validate|demonstrate|show)\b|\b(?:we|our\s+(?:experiments|measurements))\s+"
    r"(?:failed|were\s+unable)\s+to\s+(?:reproduce|confirm|validate|demonstrate)\b"
    r"|(?:результаты|эксперименты)\s+не\s+подтверд", re.I)
_PAST_OWN_RESULT = re.compile(r"\b(?:we|our\s+(?:group|team))\s+(?:(?:have|had)\s+)?(?:previously|recently|earlier)\b", re.I)
_LIVING_SYSTEM_RESULT = re.compile(r"\bin\s+(?:living|live)\s+(?:\w+\s+){0,3}(?:cells?|bacteria|animals?|tissue)\b|в\s+живых\s+клетках", re.I)
# These screens belong only to grounding 4. Earlier regexes above and the
# legacy sentence splitter below are immutable replay rules.
_UNREALIZED_V4 = re.compile(
    r"\b(?:could|might|may|would|will|hope\w*|aim\w*|intend\w*|propos\w*|future|"
    r"hypothes\w*|speculat\w*|potential\w*|predict\w*|not|never|unable|cannot|failed)\b"
    r"|\b(?:need\w*|requir\w*)\b.{0,100}\b(?:improv\w*|enhanc\w*|increas\w*|reduc\w*|develop\w*|test\w*|measur\w*)\b"
    r"|\b(?:improv\w*|enhanc\w*|increas\w*|reduc\w*|develop\w*|test\w*)\b.{0,100}\b(?:needed|required|necessary)\b"
    r"|\b(?:no|negligible|insignificant)\s+(?:(?:statistically|significant|measurable|clear)\s+)*(?:improvement|advantage|benefit|increase|reduction)\b"
    r"|\b(?:планируем|предполагаем|может|не|нужно|необходимо|требуется|требуются)\b", re.I)
_PRIOR_ASSERTION_V4 = re.compile(
    r"\b(?:previous|prior|earlier|other)\s+(?:\w+\s+){0,3}(?:work|studies|study|research\w*|reports?|authors?|results)\b"
    r"|\b(?:conventional|baseline|existing|established)\s+(?:\w+\s+){0,4}"
    r"(?:improv\w*|reduc\w*|enabl\w*|outperform\w*|enhanc\w*|achiev\w*|demonstrat\w*)\b"
    r"|\bet\s+al\.\s+(?:show\w*|report\w*|demonstrat\w*|found)\b"
    r"|\b(?:review|survey|roadmap)\b|предыдущ|ранее|обзор", re.I)
_OWN_RESULT_V4 = re.compile(
    r"\b(?:we|our|herein|this\s+(?:work|study|paper))\b.{0,140}\b"
    r"(?:measur\w*|test\w*|fabricat\w*|observ\w*|demonstrat\w*|develop\w*|design\w*|"
    r"implement\w*|introduc\w*|calculat\w*|deriv\w*|simulat\w*|prove\w*|show\w*|found|find|"
    r"report\w*|present\w*|evaluat\w*|investigat\w*|achiev\w*|reduc\w*|improv\w*|enhanc\w*|outperform\w*)\b"
    r"|\b(?:developed|fabricated|demonstrated|reported|measured|implemented)\b.{0,100}\b(?:in|by)\s+(?:this|our)\s+(?:work|study|paper)\b"
    r"|\b(?:our|the|these|experimental)\s+(?:results|measurements|experiments|findings)\s+(?:show\w*|indicat\w*|demonstrat\w*|confirm\w*)\b"
    r"|\b(?:мы|нами|наши|наша|наш)\b.{0,140}(?:измер|испыт|разработ|показ|получ|вычисл|изготов|наблюд|реализ|сниз|повыс)", re.I)
_ADVERSE_V4 = re.compile(
    r"\b(?:reduc\w*|lower\w*|decreas\w*|loss\s+of)\s+(?:(?:the|its|our|sensor|device|measurement|significant|significantly)\s+)*"
    r"(?:sensitivity|accuracy|efficiency|selectivity|stability|reliability|performance)\b"
    r"|\b(?:increas\w*|enhanc\w*|higher)\s+(?:(?:the|its|our|measurement|significant|significantly)\s+)*"
    r"(?:errors?|noise|costs?|latency|losses?|uncertainty)\b"
    r"|(?:снижен|сниз|уменьшен|уменьш).{0,30}(?:точност|чувствительност|эффективност)"
    r"|(?:увеличен|повышен|возрос).{0,30}(?:ошиб|шум|стоимост)", re.I)
_COMPUTATIONAL_V4 = re.compile(r"\b(?:simulat\w*|numerical\w*|computational\w*|in\s+silico)\b|моделирован|численн", re.I)
_THEORETICAL_V4 = re.compile(r"\b(?:theoretic\w*|deriv\w*|prove\w*|analytic\w*|first.principles)\b|теоретич|аналитич|доказ", re.I)
_EMPIRICAL_V4 = re.compile(
    r"\b(?:measured|tested|fabricated|observed|experimentally|laboratory\s+(?:experiments|measurements|tests)|"
    r"experimental\s+(?:results|measurements|validation)|in\s+vivo|in\s+vitro)\b"
    r"|\bexperiments\s+(?:show\w*|demonstrat\w*|confirm\w*|indicat\w*)\b"
    r"|\b(?:измерили|изготовили|испытали|экспериментально)\b|лабораторн.{0,10}(?:эксперимент|измерен)", re.I)
_PHYSICAL_V4 = re.compile(r"\b(?:laborator\w*|fabricat\w*|experimentally|in\s+vivo|in\s+vitro|physical\s+experiment\w*)\b|лаборатор|изготов", re.I)
_CONTRARY_RESULT_V4 = re.compile(
    r"\b(?:we|our\s+(?:results|experiments|measurements|findings))\b.{0,80}"
    r"(?:\b(?:found|show\w*|report\w*)\s+no\s+(?:(?:statistically|significant|measurable|clear)\s+)*(?:improvement|benefit|advantage)\b"
    r"|\b(?:fail\w*|unable)\s+to\s+(?:reproduce|confirm|validate|demonstrate)\b)", re.I)
_GENERIC_TOKENS = frozenset("a an the and or for of in on to with by from using based study studies research review technology technologies approach approaches method methods model models system systems application applications development results new novel efficient advanced high low performance evidence material materials sensing detection".split())
_SCOPE_FUNCTION_WORDS = frozenset("a an the of for in on at to by with from into onto upon through across within over under between among via per".split())
# Words that make a title fragment a clause rather than a name: pronouns,
# auxiliaries, modals and connectives. Domain-neutral English, not a topic list.
_CLAUSE_WORDS = frozenset(
    "and or but if as than then so such not no this that these those there here it its they them their we our "
    "you your he she his her is are was were be been being do does did can could may might will would should "
    "must shall really very how what why which who whom whose when where".split())
_SCOPE_WINDOW_WORDS = 80
_PUBLICATION_TITLE_CLAUSE = re.compile(
    r"\b(?:for|towards?|based\s+on|with\s+applications?\s+(?:to|in)|"
    r"(?:search|study|investigation|analysis|evaluation|assessment|exploration)\s+(?:of|for|into|on))\b",
    re.IGNORECASE,
)


class EvidenceError(TaskFailure):
    pass


class MechanismAliasRequired(EvidenceError):
    def __init__(self) -> None:
        super().__init__("Для проверки истории требуется короткое название конкретного механизма, а не полное название единственной статьи с описанием цели исследования.")


class EvidenceContext(Protocol):
    def check_cancelled(self) -> None: ...


def normalize_title(text: str) -> str:
    """Versioned Unicode tokenization, identical for every publication year."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def title_matches(title: str, phrases: tuple[str, ...], exclusions: tuple[str, ...]) -> bool:
    haystack = " " + normalize_title(title) + " "
    return (any(" " + normalize_title(phrase) + " " in haystack for phrase in phrases)
            and not any(" " + normalize_title(phrase) + " " in haystack for phrase in exclusions))


def _concept_tokens(text: str) -> frozenset[str]:
    # A conservative plural normalization, not a claim of semantic equivalence.
    return frozenset(token[:-1] if len(token) > 4 and token.endswith("s") else token
                     for token in normalize_title(text).split() if token not in _GENERIC_TOKENS)


@lru_cache(maxsize=512)
def coherent_aliases(phrases: tuple[str, ...]) -> bool:
    """Accept complete anchored names, never an OR of unrelated keyword fragments.

    Unrelated lexical synonyms need explicit reviewed admission in a later rule
    version; they are not automatically equivalent merely because an LLM says so.
    """
    expanded = [phrase for phrase in phrases if len(normalize_title(phrase).split()) >= 2]
    if not expanded or any(not _concept_tokens(phrase) for phrase in expanded):
        return False
    # A shared parent such as "ion batteries" cannot make lithium and sodium
    # interchangeable. Unreviewed aliases must retain all identifying modifiers.
    if len({_concept_tokens(phrase) for phrase in expanded}) != 1:
        return False
    acronyms = {"".join(word[0] for word in normalize_title(phrase).split()
                         if word not in {"a", "an", "the", "of", "for", "and"}) for phrase in expanded}
    return all(phrase in expanded or (phrase.isupper() and normalize_title(phrase) in acronyms)
               for phrase in phrases)


def candidate_title_matches(title: str, candidate: Candidate) -> bool:
    """Replay old frozen rules exactly; new rules require coherent full names."""
    if candidate.admission_rule_version == "title-phrase-admission/1.0.0":
        return title_matches(title, candidate.synonyms, candidate.exclusions)
    if candidate.admission_rule_version != TITLE_ADMISSION_VERSION or not coherent_aliases(candidate.synonyms):
        return False
    return _anchored_title_matches(title, candidate.synonyms, candidate.exclusions)


def _anchored_title_matches(title: str, phrases: tuple[str, ...], exclusions: tuple[str, ...]) -> bool:
    def normalize(text: str) -> str:
        return " ".join(token[:-1] if len(token) > 4 and token.endswith("s") else token
                        for token in normalize_title(text).split())
    haystack = " " + normalize(title) + " "
    return (any(" " + normalize(phrase) + " " in haystack for phrase in phrases)
            and not any(" " + normalize(phrase) + " " in haystack for phrase in exclusions))


def admission_hash(candidate: Candidate, *, definition: str | None = None,
                   phrases: tuple[str, ...] | None = None, exclusions: tuple[str, ...] | None = None) -> str:
    # Explicit proposed values freeze a new rule; archived v1 cards keep their hash.
    version = (candidate.admission_rule_version if definition is None and phrases is None and exclusions is None
               and candidate.admission_rule_version == "title-phrase-admission/1.0.0" else TITLE_ADMISSION_VERSION)
    rule = {"version": version, "candidate_id": candidate.candidate_id,
        "definition": definition if definition is not None else candidate.definition,
        "phrases": list(phrases if phrases is not None else candidate.synonyms),
        "exclusions": list(exclusions if exclusions is not None else candidate.exclusions),
        "text_field": "title", "matching": ("NFKC-casefold-contiguous-whole-tokens" if version.endswith("/1.0.0")
            else "coherent-full-name-NFKC-casefold-plural-contiguous-whole-tokens")}
    if candidate.scope_rule_version != LEGACY_SCOPE_RULE_VERSION:
        rule["scope_rule_version"] = candidate.scope_rule_version
    return content_hash(rule)


def member_documents(candidate: Candidate, snapshot: CorpusSnapshot, archive: RevisionSource,
                     context: EvidenceContext) -> tuple[tuple[DocumentRevisionRef, DocumentRecord], ...]:
    if snapshot.snapshot_id != candidate.discovery_snapshot_id or snapshot.plan_hash != candidate.plan_hash:
        raise EvidenceError("Кандидат относится к другому снимку обнаружения.")
    selected = set(candidate.discovery_study_ids)
    by_study: dict[str, tuple[DocumentRevisionRef, DocumentRecord]] = {}
    for reference in snapshot.documents:
        context.check_cancelled()
        if reference.study_id not in selected:
            continue
        document = archive.get(reference.revision_id)
        if document.source != reference.source or document.source_id != reference.source_id:
            raise EvidenceError("Архивная ревизия не соответствует ссылке снимка.")
        old = by_study.get(reference.study_id)
        key = (bool(document.abstract), reference.observed_at, reference.revision_id)
        if old is None or key > (bool(old[1].abstract), old[0].observed_at, old[0].revision_id):
            by_study[reference.study_id] = reference, document
    if set(by_study) != selected:
        raise EvidenceError("Часть исследований кандидата отсутствует в архивном снимке.")
    return tuple(by_study[key] for key in sorted(by_study))


def representative_documents(documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...],
                             limit: int = 5) -> tuple[tuple[DocumentRevisionRef, DocumentRecord], ...]:
    """Deterministic lexical medoid, temporal extremes and diverse boundary records.

    Selection is independent of source IDs; it is a bounded inspection sample,
    while admission coverage below is checked for every cluster member.
    """
    if len(documents) <= limit:
        return documents
    ordered = sorted(documents, key=lambda item: (normalize_title(item[1].title), item[0].revision_id))
    bags = [Counter(_concept_tokens(doc.title + " " + (doc.abstract or "")[:2000])) for _, doc in ordered]
    frequencies = Counter(token for bag in bags for token in bag)
    vectors = []
    for bag in bags:
        vector = {token: count * (1 + math.log((len(bags) + 1) / (frequencies[token] + 1)))
                  for token, count in bag.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        vectors.append({token: value / norm for token, value in vector.items()})
    centroid: dict[str, float] = {}
    for vector in vectors:
        for token, value in vector.items():
            centroid[token] = centroid.get(token, 0.0) + value
    selected = [max(range(len(ordered)), key=lambda index: sum(value * centroid[token]
                 for token, value in vectors[index].items()))]
    dated = {index: year for index, (_, doc) in enumerate(ordered) if (year := doc.publication_year) is not None}
    if dated:
        for index in (min(dated, key=dated.__getitem__), max(dated, key=dated.__getitem__)):
            if index not in selected and len(selected) < limit:
                selected.append(index)
    while len(selected) < min(limit, len(ordered)):
        remaining = [index for index in range(len(ordered)) if index not in selected]
        index = min(remaining, key=lambda i: max(sum(value * vectors[j].get(token, 0)
                    for token, value in vectors[i].items()) for j in selected))
        selected.append(index)
    return tuple(ordered[index] for index in selected)


def scope_is_anchored(documents: tuple[tuple[DocumentRevisionRef | None, DocumentRecord], ...], plan: QueryPlan,
                      *, rule_version: ScopeRuleVersion = LEGACY_SCOPE_RULE_VERSION) -> bool:
    """Require an archived scope phrase, not a material's coincidentally shared word.

    This is a conservative *necessary* grounding check, not semantic proof. Missing
    aliases/context route a proposal to review; they do not prove it off-topic.
    """
    phrases = tuple(phrase for phrase in (plan.english_query, *plan.synonyms, *plan.subdirections)
                    if len(normalize_title(phrase).split()) >= 2 and _concept_tokens(phrase))
    if rule_version == LEGACY_SCOPE_RULE_VERSION:
        return bool(phrases) and all(title_matches(doc.title + " " + (doc.abstract or ""), phrases, ())
                                     for _, doc in documents)
    if rule_version != SCOPE_RULE_VERSION:
        raise EvidenceError("Версия проверки границ области не поддерживается.")

    def tokens(text: str) -> list[str]:
        # Keep technical concepts such as sensing, detection and system. Only
        # light plural normalization; no learned/handwritten technology aliases.
        return [word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word
                for word in normalize_title(text).split()]

    required = set(tokens(plan.english_query)) - _SCOPE_FUNCTION_WORDS

    def local_match(text: str) -> bool:
        words = tokens(text)
        if len(required) < 2:
            return False
        last_seen: dict[str, int] = {}
        for index, word in enumerate(words):
            if word in required:
                last_seen[word] = index
                if len(last_seen) == len(required) and index - min(last_seen.values()) < _SCOPE_WINDOW_WORDS:
                    return True
        return False

    # Each document and paragraph must ground the complete scope on its own.
    # The subsequent LLM scope quotations, in_scope and full membership checks
    # still decide admission: nearby words alone never establish relevance.
    return bool(documents) and all(any(
        title_matches(paragraph, phrases, ()) or local_match(paragraph)
        for field in (doc.title, doc.abstract or "") for paragraph in re.split(r"[\r\n]+", field)
        if paragraph.strip()) for _, doc in documents)


def archived_field(document: DocumentRecord, text_field: str) -> str:
    if text_field == "title":
        return document.title
    if text_field == "abstract":
        return document.abstract or ""
    if text_field == "metadata":
        return json.dumps(document.raw_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if text_field == "full_text":
        from app.pilot.reports import ReportRecord

        if isinstance(document, ReportRecord):
            return document.full_text
    raise EvidenceError("Этот архив не содержит запрошенного поля полного текста.")


def quote_evidence(reference: DocumentRevisionRef, document: DocumentRecord, *,
                   text_field: Literal["title", "abstract", "metadata", "full_text"], quote: str,
                   start: int | None = None) -> Evidence:
    text = archived_field(document, text_field)
    start = text.find(quote) if start is None else start
    if (type(start) is not int or not quote.strip() or start < 0 or len(quote) > 12000
            or text[start:start + len(quote)] != quote):
        raise EvidenceError("Предложенная цитата отсутствует в указанной архивной ревизии.")
    evidence = Evidence(evidence_id="evidence-" + content_hash({"revision": reference.revision_id,
        "field": text_field, "start": start, "quote": quote}), revision_id=reference.revision_id,
        study_id=reference.study_id, source=reference.source, source_url=document.url,
        retrieved_at=document.fetched_at, text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text_field=text_field, start=start, end=start + len(quote), quote=quote,
        external_ai_allowed=(document.source in {"openalex", "crossref", "arxiv"}
                             or getattr(document, "external_ai_allowed", False) is True))
    verify_evidence_text(evidence, text)
    return evidence


def raw_quote_evidence(reference: DocumentRevisionRef, document: DocumentRecord, *, quote: str) -> Evidence:
    """Quote a whole raw abstract sentence, even when an earlier substring repeats it."""
    from app.pilot.sentences import sentence_spans

    text = document.abstract or ""
    span = next((span for span in sentence_spans(text) if text[span.start:span.end] == quote), None)
    if span is None:
        raise EvidenceError("Цитата не совпадает с целым предложением исходного текста.")
    return quote_evidence(reference, document, text_field="abstract", quote=quote, start=span.start)


class ScopeSupport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    revision_id: str = Field(min_length=64, max_length=64)
    field: Literal["title", "abstract"]
    quote: str = Field(min_length=5, max_length=600)


class CandidateName(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str = Field(min_length=1, max_length=2048)
    label: str = Field(min_length=1, max_length=200)
    definition: str = Field(min_length=1, max_length=1500)
    phrases: tuple[str, ...] = Field(min_length=1, max_length=8)
    exclusions: tuple[str, ...] = Field(default=(), max_length=8)
    specificity: Literal["specific_technology", "broad_topic", "uncertain"]
    in_scope: bool = Field(strict=True)
    scope_support: tuple[ScopeSupport, ...] = Field(default=(), max_length=5)
    scope_reason: str = Field(default="", max_length=600)

    @field_validator("phrases", "exclusions")
    @classmethod
    def plain_search_phrases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if (not value.strip() or len(value) > 180 or len(normalize_title(value)) < 3
                    or any(unicodedata.category(character) in ("Cc", "Cf", "Cs") for character in value)
                    or "http" in value.casefold()):
                raise ValueError("Требуются короткие текстовые названия технологии.")
        require_unique(tuple(normalize_title(value) for value in values), "candidate phrases")
        return values


def covering_title_phrases(documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...],
                           *, excluded: tuple[str, ...] = (), limit: int = 6) -> tuple[str, ...]:
    """Legal admission phrases for this group, the ones that admit all of it first.

    `_freeze` accepts a specific technology only when its phrase anchors every
    member title, so a phrase that anchors all of them is not merely the most
    frequent option — it is the only kind that can pass. Measured on a real
    corpus, 11 of 28 multi-document clusters had such a phrase while the model
    named one of them, because it was free to type a phrase instead of choosing.
    Ordering is by the same predicate the admission rule uses, then by the
    frequency ranking of `title_phrase_options`, which stays the source of
    candidates: this computes the rule, it does not relax it.

    The model takes the first option almost every time, so the options are
    shaped like names. Measured on «solid-state batteries», single papers
    entered the TOP as «batteries can they really change the game» and «of
    solid state electrolytes for li type»: seven-word title fragments ranked
    first only because they were longest. Options are at most four words with
    no function or rhetoric word at either edge, and among equals a phrase
    with more words beyond the direction's own name comes first — «li metal»
    before «solid state li».
    """
    from app.pilot.hierarchy import _RHETORIC

    edges = _SCOPE_FUNCTION_WORDS | _GENERIC_TOKENS | _RHETORIC
    scope_words = {word for phrase in excluded for word in _concept_tokens(phrase)}
    ranked = [phrase for phrase, _ in title_phrase_options(documents, limit=60, max_words=4, excluded=excluded)
              if phrase.split()[0] not in edges and phrase.split()[-1] not in edges
              and not _CLAUSE_WORDS.intersection(phrase.split())
              and _concept_tokens(phrase) - scope_words]

    def name_like(phrase: str) -> tuple[bool, bool, int]:
        words = phrase.split()
        return (any(word in _SCOPE_FUNCTION_WORDS for word in words), bool(_concept_tokens(words[0]) & scope_words),
                -len(_concept_tokens(phrase) - scope_words))

    covering = sorted((phrase for phrase in ranked
                       if all(_anchored_title_matches(document.title, (phrase,), ()) for _, document in documents)),
                      key=name_like)
    rest = sorted((phrase for phrase in ranked if phrase not in covering), key=name_like)
    return tuple((covering + rest)[:limit])


_REVISION_ID = re.compile(r"[0-9a-f]{64}\Z")
_EXAMPLE_LABEL = "Мембранное разделение газов"
_EXAMPLE_DEFINITION = "Полимерная мембрана разделяет газовую смесь без охлаждения."


def _first_sentence(text: str) -> str:
    from app.pilot.sentences import sentence_spans

    spans = sentence_spans(text)
    return text[spans[0].start:spans[0].end] if spans else text[:600]


class LocalCandidateName(BaseModel):
    """The same proposal, with the admission phrase chosen rather than written.

    Only the local model answers in this shape. Everything a phrase is checked
    for afterwards is unchanged; the choice simply cannot be a phrase that no
    supplied title contains.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str = Field(min_length=1, max_length=2048)
    label: str = Field(min_length=1, max_length=200)
    definition: str = Field(min_length=1, max_length=1500)
    phrase_number: int = Field(ge=1, le=6, strict=True,
                               description="number of the chosen phrase from the offered list")
    specificity: Literal["specific_technology", "broad_topic", "uncertain"]
    in_scope: bool = Field(strict=True)
    scope_support: tuple[ScopeSupport, ...] = Field(default=(), max_length=5)
    scope_reason: str = Field(default="", max_length=600)

    @model_validator(mode="before")
    @classmethod
    def repair_small_model_habits(cls, data: object) -> object:
        """Undo the 1.5B model's formatting slips without accepting anything unverified.

        Measured on «solid-state batteries»: 9 of 16 candidates failed this
        schema after the retry, although the model had chosen the candidate and
        its phrase correctly. It quoted a whole abstract instead of one sentence,
        dropped the definition, nested scope_reason inside a quotation or added a
        quotation copied from the formatting example. Each repair keeps the
        later checks meaningful: a shortened quotation is still looked up
        verbatim in the archived field, and an entry whose id is a placeholder
        could never be verified, so dropping it loses nothing. A copy of the
        example itself is refused, so it is retried instead of being read as a
        rejection of the supplied cluster.
        """
        if not isinstance(data, dict):
            return data
        data = {key.strip() if isinstance(key, str) else key: value for key, value in data.items()}
        if data.get("label") == _EXAMPLE_LABEL or data.get("definition") == _EXAMPLE_DEFINITION:
            raise ValueError("The answer repeats the formatting example instead of naming the cluster.")
        supports = data.get("scope_support")
        if isinstance(supports, list):
            repaired = []
            for item in supports:
                if not isinstance(item, dict):
                    repaired.append(item)
                    continue
                item = {key.strip() if isinstance(key, str) else key: value for key, value in item.items()}
                reason = item.pop("scope_reason", None)
                if isinstance(reason, str) and not data.get("scope_reason"):
                    data["scope_reason"] = reason
                revision = item.get("revision_id")
                if isinstance(revision, str) and _REVISION_ID.fullmatch(revision) is None:
                    continue
                quote = item.get("quote")
                if isinstance(quote, str) and len(quote) > 600:
                    item["quote"] = _first_sentence(quote)
                repaired.append(item)
            data["scope_support"] = repaired
        definition = data.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            quotes = [item.get("quote") for item in data.get("scope_support") or () if isinstance(item, dict)]
            if quotes and isinstance(quotes[0], str) and quotes[0].strip():
                data["definition"] = quotes[0]
        for key, limit in (("label", 200), ("definition", 1500), ("scope_reason", 600)):
            value = data.get(key)
            if isinstance(value, str) and len(value) > limit:
                data[key] = value[:limit].rsplit(" ", 1)[0]
        return data


class LocalLabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: tuple[LocalCandidateName, ...] = Field(min_length=1, max_length=15)

    @model_validator(mode="after")
    def unique_candidates(self) -> Self:
        require_unique(tuple(item.candidate_id for item in self.candidates), "named candidates")
        return self


def _chosen_name(proposal: LocalCandidateName, phrases: Sequence[str]) -> CandidateName:
    if not phrases or proposal.phrase_number > len(phrases):
        raise EvidenceError("AI выбрал название, которого нет в списке допустимых фраз.")
    return CandidateName(candidate_id=proposal.candidate_id, label=proposal.label,
        definition=proposal.definition, phrases=(phrases[proposal.phrase_number - 1],),
        exclusions=(), specificity=proposal.specificity, in_scope=proposal.in_scope,
        scope_support=proposal.scope_support, scope_reason=proposal.scope_reason)


class LabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: tuple[CandidateName, ...] = Field(min_length=1, max_length=15)

    @model_validator(mode="after")
    def unique_candidates(self) -> Self:
        require_unique(tuple(item.candidate_id for item in self.candidates), "named candidates")
        return self


def _freeze(candidate: Candidate, proposal: CandidateName,
            documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...],
            *, current_proposal: bool = False) -> Candidate:
    if proposal.candidate_id != candidate.candidate_id:
        raise EvidenceError("AI предложил кандидата, которого нет в обнаруженных группах.")
    supported_phrases = tuple(phrase for phrase in proposal.phrases
        if any(title_matches(document.title, (phrase,), ()) for _, document in documents))
    if supported_phrases != proposal.phrases:
        raise EvidenceError("Поисковое название технологии не встречается в названиях её исследований.")
    matched = sum(_anchored_title_matches(document.title, proposal.phrases, proposal.exclusions) for _, document in documents)
    if proposal.specificity == "specific_technology" and (
            matched != len(documents) or not coherent_aliases(proposal.phrases)):
        raise EvidenceError("Новое название не задаёт одну технологию для всех исследований группы.")
    if current_proposal and proposal.specificity == "specific_technology" and len(documents) == 1:
        title = normalize_title(documents[0][1].title)
        # A copied paper title with its purpose/rationale is not a reusable
        # history definition. This is only a gate for new naming proposals:
        # frozen historical rules and short legitimate technology names replay
        # unchanged. Long mechanism names without a narrative clause are allowed.
        if (len(title.split()) >= 8 and _PUBLICATION_TITLE_CLAUSE.search(title)
                and not any(len(_concept_tokens(phrase)) >= 2 and normalize_title(phrase) != title
                            for phrase in proposal.phrases)):
            raise MechanismAliasRequired()
    # Specificity describes the technological object, not its publication count.
    # One study can define a concrete mechanism; temporal/evidence gates decide
    # separately whether it supports any positive signal decision.
    specificity = proposal.specificity
    if (specificity == "specific_technology" and (not matched or not any(
            len(_concept_tokens(phrase)) >= 2 for phrase in proposal.phrases))):
        specificity = "uncertain"
    values = candidate.model_dump(mode="python") | dict(label=proposal.label, definition=proposal.definition,
        synonyms=proposal.phrases, exclusions=proposal.exclusions, specificity=specificity,
        admission_rule_version=TITLE_ADMISSION_VERSION,
        admission_rule_hash=admission_hash(candidate, definition=proposal.definition,
                                           phrases=proposal.phrases, exclusions=proposal.exclusions))
    return Candidate.model_validate(values)


def title_phrase_options(documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...],
                         limit: int = 12, min_words: int = 2, max_words: int = 7,
                         excluded: tuple[str, ...] = ()) -> tuple[tuple[str, int], ...]:
    """Phrases that literally occur in these titles, with how many titles carry each.

    The admission rule accepts only a phrase that appears word for word in the
    titles of the group. A small model invents phrases that do not, and its
    proposal is then refused. Offering the legal options is not a relaxation of
    that rule: it is the same rule, computed, so the model chooses instead of
    guessing. Which of them names the mechanism is still the model's decision,
    and every check afterwards runs unchanged.
    """
    counts: Counter[str] = Counter()
    for _, document in documents:
        words = normalize_title(document.title).split()
        seen = set()
        for size in range(min_words, max_words + 1):
            for index in range(len(words) - size + 1):
                phrase = " ".join(words[index:index + size])
                if phrase not in seen and len(_concept_tokens(phrase)) >= 2:
                    seen.add(phrase)
        counts.update(seen)
    # The name of the search direction itself is never a specific technology
    # here, so offering it only produces a proposal the rules downgrade.
    forbidden = {normalize_title(phrase) for phrase in excluded}
    ordered = sorted(((phrase, count) for phrase, count in counts.items() if phrase not in forbidden),
                     key=lambda item: (-item[1], -len(item[0].split()), item[0]))
    return tuple(ordered[:limit])


def exact_quote(document: DocumentRecord, field: str, quote: str) -> str | None:
    """The document's own wording for a quotation the model retyped.

    A model reproduces a sentence with different quotation marks, spacing or
    case, and the archive then refuses it. The sentence it meant is still in the
    document: this returns that sentence verbatim, and returns nothing at all
    when no single sentence matches, because rewording is not a quotation.
    """
    from app.pilot.sentences import sentence_spans

    text = archived_field(document, field)
    if quote in text:
        return quote
    wanted = normalize_title(quote)
    if len(wanted.split()) < 3:
        return None
    matches = [text[span.start:span.end] for span in sentence_spans(text)
               if normalize_title(text[span.start:span.end]) == wanted]
    return matches[0] if len(matches) == 1 else None


def _local_proposal(candidate: Candidate, documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...]) -> CandidateName:
    phrases = tuple(value.strip() for value in candidate.label.split("/") if len(value.strip()) <= 180
                    and len(normalize_title(value)) >= 3
                    and any(title_matches(document.title, (value.strip(),), ()) for _, document in documents))
    if not phrases:
        title = documents[0][1].title
        phrase = title[:180]
        if len(title) > 180:
            phrase = phrase.rsplit(" ", 1)[0]
        phrases = (phrase,)
    return CandidateName(candidate_id=candidate.candidate_id, label=candidate.label,
        definition="Предварительная группа по названиям исследований. Границы технологии ещё не проверены.",
        phrases=phrases[:8], exclusions=(), specificity="uncertain", in_scope=True,
        scope_reason="Принадлежность области не проверена; только локальная группа для просмотра.")


def frozen_candidate_is_grounded(candidate: Candidate,
                                 documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...],
                                 plan: QueryPlan | None) -> bool:
    """Retain an already frozen definition only after rechecking its whole corpus.

    This does not upgrade discovery labels or legacy rules, rename the object,
    infer novelty or use an LLM's unavailable response as new evidence.
    """
    if (plan is None or candidate.plan_hash != plan.plan_hash
            or candidate.specificity != "specific_technology"
            or candidate.admission_rule_version != TITLE_ADMISSION_VERSION
            or candidate.admission_rule_hash != admission_hash(candidate)
            or not coherent_aliases(candidate.synonyms)
            or not any(len(_concept_tokens(phrase)) >= 2 for phrase in candidate.synonyms)):
        return False
    broad_names = {normalize_title(plan.original_query), normalize_title(plan.english_query)}
    return (bool(documents) and normalize_title(candidate.label) not in broad_names
            and not any(normalize_title(phrase) == normalize_title(plan.english_query) for phrase in candidate.synonyms)
            and all(candidate_title_matches(document.title, candidate) for _, document in documents)
            and scope_is_anchored(documents, plan, rule_version=candidate.scope_rule_version))


def _failure_code(error: LlmError | BudgetError | EvidenceError) -> str:
    if isinstance(error, LlmError):
        return error.code
    if isinstance(error, BudgetExceeded):
        return "budget_exceeded"
    if isinstance(error, MechanismAliasRequired):
        return "mechanism_alias_required"
    return "budget_state_error" if isinstance(error, BudgetError) else "evidence_rejected"


def label_system_prompt(local: bool) -> str:
    """The exact candidate-label instruction shared by production and benchmarks."""
    return ("Name each supplied document cluster with a specific technological mechanism. "
        "Keep exactly the supplied candidate IDs; never invent candidates or change their membership. "
        "Label and definition in Russian; phrases and exclusions in English. "
        + ("The admission phrase is not written: answer with phrase_number, the number of the "
           "phrase you choose from the list offered for that cluster. Prefer the number whose "
           "phrase names the identifying mechanism and material/platform of the whole cluster. "
           if local else
           "Every admission phrase must appear verbatim (ignoring punctuation/case) in a supplied "
           "document TITLE, not just its abstract. "
           "Every phrase must name the SAME concrete mechanism, not an independent parent keyword. "
           "Choose a concise canonical technical phrase, normally 2-7 words, retaining the identifying "
           "mechanism and material/platform. Remove the paper-specific purpose or narrative tail; "
           "do not copy an entire long publication title as its only history admission phrase. "
           "Do not shorten a name to a generic parent technology just to satisfy that guidance. ")
        +
        "If only a theoretical model, simulation or mathematical relation is reported, say so in "
        "the label and definition; never describe it as a demonstrated sensor or experimental device. "
        "Do not call a broad application area or the original search direction a specific technology. "
        "A single study may define a specific mechanism; study count does not establish or refute specificity. "
        "Do not infer a trend or novelty from that specificity. "
        "Check the actual mechanism/function/material of every representative, not shared vocabulary; "
        "a material name does not prove the requested measurement principle or application. "
        "Mark uncertain groups honestly. "
        "First check whether EVERY supplied representative belongs to the requested technological scope. "
        "Set in_scope=false for off-topic or mixed-topic groups, without renaming them to make them fit. "
        "For in_scope=true, cite an exact title/abstract quote for EVERY supplied representative in scope_support; "
        "these quotations must explain its actual mechanism-level connection to the requested scope. "
        "Return uncertain when connection relies on speculation or missing context. Include scope_reason. "
        "Keep definitions under 50 words and each scope quotation under 35 words. "
        "These names are proposals, not scientific evidence or declarations of novelty.")


def label_candidates(candidates: tuple[Candidate, ...], snapshot: CorpusSnapshot, archive: DocumentArchive,
                     context: RunContext, *, client: LlmClient | None = None,
                     scope_ids: Sequence[str] = (), query_plan: QueryPlan | None = None, stage_offset: int = 0,
                     budget_unavailable: Callable[[], bool] | None = None) -> tuple[Candidate, ...]:
    if len(candidates) > 30:
        raise EvidenceError("За один этап можно назвать не более 30 обнаруженных кандидатов.")
    if type(stage_offset) is not int or not 0 <= stage_offset < 60 or stage_offset + len(candidates) > 60:
        raise EvidenceError("Некорректное смещение этапа проверки кандидатов.")
    require_unique(tuple(candidate.candidate_id for candidate in candidates), "candidate labels")
    if query_plan is not None and query_plan.plan_hash != snapshot.plan_hash:
        raise EvidenceError("Область запроса не соответствует снимку обнаружения.")
    documents = {item.candidate_id: member_documents(item, snapshot, archive, context) for item in candidates}
    local = getattr(getattr(client, "config", None), "provider", "") == "local"
    representatives = {key: representative_documents(members)[:LOCAL_LABEL_REPRESENTATIVES] if local
                       else representative_documents(members)
                       for key, members in documents.items()}
    # Admission is candidate-specific. A private or unanchored neighbour must
    # neither block an eligible candidate nor ride along in its paid request.
    eligibility = {item.candidate_id: (
        all(doc.source in {"openalex", "crossref", "arxiv"} for _, doc in documents[item.candidate_id]),
        query_plan is None or scope_is_anchored(documents[item.candidate_id], query_plan,
                                                rule_version=item.scope_rule_version))
        for item in candidates}
    # A 1.5B model answers about one or two clusters correctly and starts
    # retelling the material when asked about four. Measured on a real batch:
    # four clusters took 23 seconds and produced one usable candidate, two take
    # about eight and produce both.
    batch_limit = LOCAL_LABEL_BATCH_SIZE if local else LABEL_BATCH_SIZE
    representative_limit = LOCAL_LABEL_BATCH_REPRESENTATIVES if local else LABEL_BATCH_REPRESENTATIVES
    batches: list[list[Candidate]] = []
    for candidate in candidates:
        size = len(representatives[candidate.candidate_id])
        if (not batches or len(batches[-1]) >= batch_limit
                or sum(len(representatives[item.candidate_id]) for item in batches[-1]) + size
                    > representative_limit):
            batches.append([])
        batches[-1].append(candidate)
    def request_for(eligible: tuple[Candidate, ...]):
        """Запрос именования этих кандидатов: алиасы, варианты фраз и аргументы вызова."""
        assert query_plan is not None
        # A small model has to copy every identifier it is given, and a
        # 64-character hash is where it slips: in a measured run it returned
        # ids that belonged to no candidate. The alias below is short enough
        # to copy and is translated back before anything is checked.
        aliases = {candidate.candidate_id: f"c{number}" for number, candidate in enumerate(eligible)} if local else {}
        title_limit = LOCAL_TITLE_CHARACTERS if local else 1000
        abstract_limit = LOCAL_ABSTRACT_CHARACTERS if local else 1200
        data = [{"candidate_id": aliases.get(candidate.candidate_id, candidate.candidate_id),
                 "keyword_label": candidate.label,
                 "documents": [{"revision_id": ref.revision_id, "title": doc.title[:title_limit],
                                "abstract": (doc.abstract or "")[:abstract_limit]}
                               for ref, doc in representatives[candidate.candidate_id]],
                 "total_studies": len(documents[candidate.candidate_id])} for candidate in eligible]
        hint = ""
        options: dict[str, tuple[str, ...]] = {}
        if local:
            # The options go with the rules, after the material: a list
            # inside the material is one more thing this model copies. They
            # are numbered because the answer names a number: a phrase this
            # model retypes is a phrase it can retype wrongly.
            scope_names = (query_plan.english_query, query_plan.original_query, *query_plan.subdirections)
            options = {}
            for candidate in eligible:
                members = documents[candidate.candidate_id]
                phrases = covering_title_phrases(members, excluded=scope_names, limit=6)
                # A group of several studies becomes a specific technology
                # only when one phrase anchors every member title (_freeze),
                # and covering phrases come first. Without one, every answer
                # is refused afterwards, so the few seconds are not spent.
                if len(members) > 1 and not any(
                        all(_anchored_title_matches(document.title, (phrase,), ()) for _, document in members)
                        for phrase in phrases[:1]):
                    phrases = ()
                options[aliases[candidate.candidate_id]] = phrases
            hint = LOCAL_LABEL_EXAMPLE + chr(10) + chr(10).join(
                f"For cluster {name}, phrase_number chooses one of: "
                + "; ".join(f"{number}) {phrase}" for number, phrase in enumerate(phrases, 1))
                for name, phrases in options.items() if phrases)
        request = {"system_prompt": label_system_prompt(local),
                   "user_content": json.dumps({"requested_scope": query_plan.definition,
                       "english_scope": query_plan.english_query, "excluded_scope": query_plan.exclusions,
                       "clusters": data}, ensure_ascii=False),
                   "local_hint": hint,
                   "max_output_tokens": 256 + 650 * len(eligible)
                       + 250 * sum(len(representatives[item.candidate_id]) for item in eligible)}
        return aliases, options, request

    def prefetch(start: int) -> int:
        """Посчитать ответы модели на следующие запросы одним батчем.

        Локальная модель отвечает на одну группу за запрос, и её ответы не
        зависят друг от друга: восемь ответов батчем декодируются в 2,4 раза
        быстрее, чем по одному (замер 28.09.2026), и совпадают с одиночными.
        Возвращает первый индекс после окна.
        """
        from app.pilot.local_llm import generation_batch

        window: list[dict[str, object]] = []
        index = start
        while index < len(batches) and len(window) < generation_batch():
            pending = batches[index]
            eligible = tuple(item for item in pending if eligibility[item.candidate_id] == (True, True))
            if eligible and context.load_checkpoint(f"labels_{stage_offset + index}") is None:
                _, options, request = request_for(eligible)
                if all(options.values()):
                    window.append(request)
            index += 1
        prefetch_json = getattr(client, "prefetch_json", None)
        if len(window) > 1 and callable(prefetch_json):
            prefetch_json(LocalLabelBatch, window, cancel=context.cancel_event)
        return index

    result: list[Candidate] = []
    processed = 0
    prefetched = 0
    snapshot_digest: str | None = None
    # A batch too large for the remaining tokens may fail while a smaller later
    # batch still fits. Only a proven exhausted scope stops subsequent requests.
    budget_failure = next((prior.get("failure_code") for index in range(stage_offset)
        if (prior := context.load_checkpoint(f"labels_{index}")) is not None
        and prior.get("budget_stopped") is True), None)
    for batch_index, batch in enumerate(batches):
        context.check_cancelled()
        if snapshot_digest is None:
            # Keep this local to one invocation: model_copy(update=...) creates
            # a different snapshot whose content hash must be recomputed.
            snapshot_digest = snapshot.snapshot_hash
        first, last = processed + 1, processed + len(batch)
        position = str(first) if first == last else f"{first}–{last}"
        context.progress("labels", f"Проверяем названия и границы кандидатов: {position} из {len(candidates)}",
                         processed, len(candidates))
        processed += len(batch)
        stage = f"labels_{stage_offset + batch_index}"
        input_hash = content_hash({"candidates": [item.model_dump(mode="json") for item in batch],
                                   "snapshot": snapshot_digest, "prompt": LABEL_PROMPT_VERSION,
                                   "scope": query_plan.plan_hash if query_plan else None})
        checkpoint = context.load_checkpoint(stage)
        if checkpoint is not None:
            if checkpoint.get("input_hash") != input_hash:
                raise EvidenceError("Сохранённые названия относятся к другой выборке.")
            saved_candidates = tuple(Candidate.model_validate(item) for item in checkpoint["candidates"])
            if checkpoint.get("input_candidate_ids") != [item.candidate_id for item in batch]:
                raise EvidenceError("Набор кандидатов изменён в сохранённом этапе.")
            originals = {item.candidate_id: item for item in batch}
            require_unique(tuple(item.candidate_id for item in saved_candidates), "cached labels")
            for saved in saved_candidates:
                original = originals.get(saved.candidate_id)
                if original is None:
                    raise EvidenceError("Сохранённый кандидат отсутствует в исходной выборке.")
                if (saved.discovery_study_ids != original.discovery_study_ids
                        or saved.admission_rule_hash != admission_hash(saved)):
                    raise EvidenceError("Сохранённые границы кандидата изменены.")
                if (saved.specificity == "specific_technology" and query_plan is not None
                        and not scope_is_anchored(documents[saved.candidate_id], query_plan,
                                                  rule_version=saved.scope_rule_version)):
                    raise EvidenceError("Сохранённая технология не связана с областью во всех исследованиях.")
            if checkpoint.get("budget_stopped") is True:
                budget_failure = checkpoint["failure_code"]
            result.extend(saved_candidates)
            continue
        proposals = None
        failure = None
        failure_code = None
        receipt = None
        rejected_candidates = []
        fallback_candidate_ids = []
        eligible = tuple(item for item in batch if eligibility[item.candidate_id] == (True, True))
        if any(not eligibility[item.candidate_id][0] for item in batch):
            failure = "Документы без разрешения на передачу AI названы только локально."
            failure_code = "external_ai_not_allowed"
        # A proposal about a group whose own text does not anchor the requested
        # scope is replaced below whatever the model answers. Asking anyway costs
        # a full generation and changes nothing, so the check comes first and the
        # outcome is recorded exactly as it would have been after the call.
        elif any(not eligibility[item.candidate_id][1] for item in batch):
            failure = ("Прямая связь механизма с областью не подтверждена архивным текстом; "
                       "требуется проверка границ и aliases.")
            failure_code = "evidence_rejected"
        if budget_failure is not None and eligible and failure_code is None:
            failure = "Лимит AI-проверки исчерпан; последующие названия сохранены как непроверенные."
            failure_code = budget_failure
        if client is not None and query_plan is not None and eligible and budget_failure is None:
            if local and batch_index >= prefetched and callable(getattr(client, "prefetch_json", None)):
                prefetched = prefetch(batch_index)
            aliases, options, request = request_for(eligible)
            if local:
                if not all(options.values()):
                    # Nothing in these titles can be admitted, so a generation
                    # would be refused whatever it answered.
                    failure = "В названиях работ группы нет фразы, пригодной для допуска технологии."
                    failure_code = "evidence_rejected"
            try:
                if local and not all(options.values()):
                    raise EvidenceError(failure)
                completion = client.generate_json(LocalLabelBatch if local else LabelBatch,
                    prompt_version=LABEL_PROMPT_VERSION,
                    request_id=f"{context.run_id}:labels:{stage_offset + batch_index}:attempt:{context.attempt}",
                    scope_ids=scope_ids, cancel=context.cancel_event, **request)
                restored = {alias: real for real, alias in aliases.items()}
                # The chosen number becomes its phrase while the alias is still
                # known, because the offered list is keyed by that alias.
                if isinstance(completion.value, LocalLabelBatch):
                    answered = [_chosen_name(item, options.get(item.candidate_id, ()))
                                for item in completion.value.candidates]
                elif isinstance(completion.value, LabelBatch):
                    answered = list(completion.value.candidates)
                else:
                    raise EvidenceError("AI вернул неподходящий формат именования.")
                named = [item.model_copy(update={"candidate_id": restored[item.candidate_id]})
                         if item.candidate_id in restored else item
                         for item in answered]
                proposals = {item.candidate_id: item for item in named}
                if set(proposals) != {item.candidate_id for item in eligible}:
                    raise EvidenceError("AI изменил набор обнаруженных кандидатов.")
            except LlmCancelled:
                raise TaskCancelled() from None
            except (LlmError, BudgetError, EvidenceError) as error:
                failure = str(error)
                failure_code = _failure_code(error)
                if isinstance(error, BudgetStateError) or (isinstance(error, BudgetExceeded)
                        and budget_unavailable is not None and budget_unavailable()):
                    budget_failure = failure_code
                proposals = None
            finally:
                receipt = client.last_receipt.to_dict() if client.last_receipt else None
        frozen_list: list[Candidate] = []
        preserved_frozen = 0
        model_accepted = 0
        for candidate in batch:
            members = documents[candidate.candidate_id]
            proposed = proposals.get(candidate.candidate_id) if proposals else None
            if proposed is None and frozen_candidate_is_grounded(candidate, members, query_plan):
                frozen_list.append(candidate)
                preserved_frozen += 1
                continue
            proposal = proposed if proposed is not None else _local_proposal(candidate, members)
            used_model = proposed is not None
            if proposed is None:
                fallback_candidate_ids.append(candidate.candidate_id)
            if proposed is not None:
                if not proposal.in_scope:
                    rejected_candidates.append({"candidate_id": candidate.candidate_id,
                        "reason_code": "off_scope_or_mixed",
                        "reason": proposal.scope_reason or "Группа не относится к запрошенной области."})
                    continue
                scope_documents = {ref.revision_id: (ref, doc) for ref, doc in representatives[candidate.candidate_id]}
                try:
                    cited = {support.revision_id for support in proposal.scope_support}
                    if cited != set(scope_documents) or len(cited) != len(proposal.scope_support):
                        raise EvidenceError("Нет прямой связи с областью для всех представителей группы.")
                    for support in proposal.scope_support:
                        ref, doc = scope_documents[support.revision_id]
                        field = support.field
                        quote = exact_quote(doc, field, support.quote) if local else support.quote
                        if quote is None and local:
                            # The small model quotes a title and calls it an
                            # abstract: the words are verbatim, only the field
                            # it named is wrong.
                            field = "title" if field == "abstract" else "abstract"
                            quote = exact_quote(doc, field, support.quote)
                        if quote is None:
                            raise EvidenceError("Предложенная цитата отсутствует в указанной архивной ревизии.")
                        quote_evidence(ref, doc, text_field=field, quote=quote)
                    if query_plan is not None and not scope_is_anchored(members, query_plan,
                                                                       rule_version=candidate.scope_rule_version):
                        failure = "Прямая связь механизма с областью не подтверждена архивным текстом; требуется проверка границ и aliases."
                        failure_code = "evidence_rejected"
                        fallback_candidate_ids.append(candidate.candidate_id)
                        proposal = _local_proposal(candidate, members)
                        used_model = False
                    if query_plan is not None and (normalize_title(proposal.label) in {
                            normalize_title(query_plan.original_query), normalize_title(query_plan.english_query)}
                            or any(normalize_title(phrase) == normalize_title(query_plan.english_query)
                                   for phrase in proposal.phrases)):
                        proposal = proposal.model_copy(update={"specificity": "broad_topic"})
                except EvidenceError as error:
                    rejected_candidates.append({"candidate_id": candidate.candidate_id,
                        "reason_code": "scope_evidence_missing", "reason": str(error)})
                    continue
            try:
                frozen = _freeze(candidate, proposal, members, current_proposal=proposed is not None)
            except EvidenceError as error:
                failure = str(error)
                failure_code = _failure_code(error)
                if candidate.candidate_id not in fallback_candidate_ids:
                    fallback_candidate_ids.append(candidate.candidate_id)
                frozen = _freeze(candidate, _local_proposal(candidate, members), members)
                used_model = False
            frozen_list.append(frozen)
            model_accepted += used_model
        context.checkpoint(stage, {"input_hash": input_hash, "candidates": [item.model_dump(mode="json") for item in frozen_list],
                                   "input_candidate_ids": [item.candidate_id for item in batch],
                                   "rejected": rejected_candidates,
                                   "receipt": receipt, "limitation": failure,
                                   "failure_code": failure_code, "budget_stopped": budget_failure is not None,
                                   "fallback_candidate_ids": fallback_candidate_ids,
                                   "label_status": ("model_proposal" if model_accepted == len(frozen_list) and model_accepted else
                                        "mixed_model_and_lexical" if model_accepted else
                                        "retained_frozen_definition" if preserved_frozen == len(batch) else
                                        "mixed_frozen_and_lexical" if preserved_frozen else
                                        "definition_rejected" if not frozen_list else "unverified_lexical_label")})
        result.extend(frozen_list)
    return tuple(result)


class QuoteSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["problem", "advantage", "case"]
    revision_id: str = Field(min_length=1, max_length=64)
    field: Literal["title", "abstract"]
    quote: str = Field(min_length=1, max_length=1500)


class PassportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    selections: tuple[QuoteSelection, ...] = Field(default=(), max_length=6)
    russian_interpretation: str | None = Field(default=None, min_length=1, max_length=1000)


def _role_supported(role: str, quote: str, *, context: str | None = None,
                    method_version: str = "exact-contextual-quotation/2.0.0") -> bool:
    """Conservative contextual screening of an author's direct assertion.

    Exact text alone only proves provenance. A full, unqualified statement of
    benefit is required here; ambiguous claims need a human semantic review.
    This screen is deliberately not a general-purpose entailment classifier.
    """
    if method_version == RAW_GROUNDING_METHOD:
        if context is not None and quote not in grounding_sentences(context, method_version=method_version):
            return False
        return _raw_role_supported(role, quote, context_contradicts=context is not None and _contrary_result_v4(context))
    if role == "problem":
        return bool(_PROBLEM.search(quote)) and (context is None or quote in _sentences(context))
    if role == "advantage":
        return (bool(_ADVANTAGE.search(quote)) and not bool(_NEGATED_ADVANTAGE.search(quote))
                and not bool(_BASELINE_ADVANTAGE.search(quote))
                and not bool(_ADVERSE_EFFECT.search(quote))
                and (context is None or (quote in _sentences(context)
                     and not (_RESULT_CONTRADICTION.search(context) if method_version == PRIMARY_GROUNDING_METHOD
                              else _NEGATED_ADVANTAGE.search(context)))))
    return True


def _sentences(text: str) -> tuple[str, ...]:
    # Preserve original spelling/spacing, without gluing distant fragments.
    return tuple(match.group().strip() for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text)
                 if 20 <= len(match.group().strip()) <= 1500)


def grounding_sentences(text: str, *, method_version: str = "exact-contextual-quotation/2.0.0") -> tuple[str, ...]:
    """Versioned raw quotations; legacy segmentation must not change on replay."""
    if method_version == RAW_GROUNDING_METHOD:
        from app.pilot.sentences import sentences

        return tuple(sentence for sentence in sentences(text) if 20 <= len(sentence) <= 1500)
    if method_version in {"exact-contextual-quotation/2.0.0", PRIMARY_GROUNDING_METHOD}:
        return _sentences(text)
    raise EvidenceError("Неизвестная версия проверки цитат.")


def _screening_text(text: str) -> str:
    """Plain text solely for matching; evidence always retains the raw slice.

    Malformed markup is ineligible rather than allowing hidden qualifications
    to disappear. Tags, entity spelling and whitespace never alter quote bytes.
    """
    from app.pilot.sentences import _markup_spans

    pieces: list[str] = []
    previous = 0
    for start, end, _ in _markup_spans(text):
        if not text[start:end].endswith(">"):
            return ""
        tag = text[start:end]
        block = re.match(r"</?(?:[\w-]+:)?(?:p|div|sec|title|br|li|tr|table|abstract)(?:\s|/?>)", tag, re.I)
        # Inline markup may split a word, including a negation (n<italic>ot).
        # Treating every tag as whitespace would erase that qualification.
        pieces.extend((text[previous:start], " " if block else ""))
        previous = end
    pieces.append(text[previous:])
    return " ".join(html.unescape("".join(pieces)).split())


def _own_result_v4(text: str) -> bool:
    return bool(_OWN_RESULT_V4.search(text) or (_EMPIRICAL_V4.search(text) and _ADVANTAGE.search(text)))


def _raw_role_supported(role: str, quote: str, *, context_contradicts: bool) -> bool:
    screened = _screening_text(quote)
    if role == "problem":
        return bool(_PROBLEM.search(screened))
    if role == "advantage":
        return bool(_ADVANTAGE.search(screened) and _own_result_v4(screened)
            and not _UNREALIZED_V4.search(screened) and not _PRIOR_ASSERTION_V4.search(screened)
            and not _PAST_OWN_RESULT.search(screened) and not _ADVERSE_V4.search(screened)
            and not _COMPUTATIONAL_V4.search(screened) and not _THEORETICAL_V4.search(screened)
            and not context_contradicts)
    return True


def _contrary_result_v4(text: str) -> bool:
    screened = _screening_text(text)
    return bool(_RESULT_CONTRADICTION.search(screened) or _CONTRARY_RESULT_V4.search(screened))


def _primary_sentence_supported_v4(quote: str, abstract: str, *, context_contradicts: bool | None = None) -> bool:
    screened = _screening_text(quote)
    return bool(screened and _own_result_v4(screened)
        and not _UNREALIZED_V4.search(screened) and not _PRIOR_ASSERTION_V4.search(screened)
        and not _PAST_OWN_RESULT.search(screened)
        and not (_contrary_result_v4(abstract) if context_contradicts is None else context_contradicts))


def candidate_claims_grounded(candidate: Candidate,
                               documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...]) -> bool:
    """A quote from one member cannot establish a benefit for an uncertain group."""
    return bool(documents and candidate.specificity == "specific_technology"
        and candidate.admission_rule_version == TITLE_ADMISSION_VERSION
        and candidate.admission_rule_hash == admission_hash(candidate)
        and coherent_aliases(candidate.synonyms)
        and all(candidate_title_matches(document.title, candidate) for _, document in documents))


_SNAPSHOT_STATUS_CACHE: dict[tuple[str, str], tuple[frozenset[str], frozenset[str]]] = {}
_SNAPSHOT_STATUS_CACHE_LIMIT = 8
_SNAPSHOT_STATUS_LOCK = Lock()


def _snapshot_study_status(snapshot: CorpusSnapshot, archive: RevisionSource,
                            context: EvidenceContext, *,
                            snapshot_digest: str | None = None) -> tuple[frozenset[str], frozenset[str]]:
    """Status evidence uses every revision, including poorer alternate records.

    Every candidate of one run asks the same question about the same discovery
    snapshot, so a passport batch used to reread and rescan the whole corpus once
    per candidate. The answer depends only on the archived bytes of that snapshot:
    ``snapshot_hash`` covers every revision reference, and a revision id is the
    SHA-256 of its own content, so the same key cannot describe other documents.
    Both returned sets are immutable and safe to share between callers.
    """
    from app.pilot.retractions import retracted_family_keys
    from app.pilot.sources import supporting_asset_keys

    key = (str(archive.directory), snapshot_digest if snapshot_digest is not None else snapshot.snapshot_hash)
    cached = _SNAPSHOT_STATUS_CACHE.get(key)
    if cached is not None:
        context.check_cancelled()
        return cached
    documents = []
    for reference in snapshot.documents:
        context.check_cancelled()
        documents.append(archive.get(reference.revision_id))
    status = (frozenset(retracted_family_keys(documents, rules_version="publication-status/2.0.0")),
              frozenset(supporting_asset_keys(documents, rules_version="publication-status/2.0.0")))
    # Bounded: a run keeps one discovery snapshot, and history/enrichment
    # snapshots of one analysis stay far below this limit. An analysis and a
    # saved-result view can reach this from different threads, so eviction never
    # assumes the key it picked is still present.
    with _SNAPSHOT_STATUS_LOCK:
        while len(_SNAPSHOT_STATUS_CACHE) >= _SNAPSHOT_STATUS_CACHE_LIMIT:
            oldest = next(iter(_SNAPSHOT_STATUS_CACHE), None)
            if oldest is None:
                break
            _SNAPSHOT_STATUS_CACHE.pop(oldest, None)
        _SNAPSHOT_STATUS_CACHE[key] = status
    return status


def snapshot_retracted_studies(snapshot: CorpusSnapshot, archive: RevisionSource,
                                context: EvidenceContext) -> frozenset[str]:
    return _snapshot_study_status(snapshot, archive, context)[0]


def snapshot_primary_exclusions(snapshot: CorpusSnapshot, archive: RevisionSource,
                                 context: EvidenceContext) -> frozenset[str]:
    retracted, supporting = _snapshot_study_status(snapshot, archive, context)
    return retracted | supporting


def _research_context_v4(document: DocumentRecord) -> tuple[bool, Literal["computational", "theoretical", "research"]]:
    physical = False
    kind: Literal["computational", "theoretical", "research"] = "research"
    for sentence in grounding_sentences(document.abstract or "", method_version=RAW_GROUNDING_METHOD):
        text = _screening_text(sentence)
        if _OWN_RESULT_V4.search(text) and not _UNREALIZED_V4.search(text):
            if _PHYSICAL_V4.search(text) and not _COMPUTATIONAL_V4.search(text) and not _THEORETICAL_V4.search(text):
                physical = True
            if _COMPUTATIONAL_V4.search(text):
                kind = "computational"
            elif _THEORETICAL_V4.search(text) and kind == "research":
                kind = "theoretical"
    return physical, kind


def primary_result_kind(quote: str, document: DocumentRecord, *,
                        study_context: tuple[bool, Literal["computational", "theoretical", "research"]] | None = None,
                        ) -> Literal["experimental", "computational", "theoretical", "research"]:
    """Grounding-4 research type; a simulated measurement is not an experiment."""
    text = _screening_text(quote)
    if _COMPUTATIONAL_V4.search(text):
        return "computational"
    if _THEORETICAL_V4.search(text):
        return "theoretical"
    # A second sentence's 'measured' cannot erase an explicitly computational
    # study. An explicit physical experiment can coexist with a theoretical model.
    if not _PHYSICAL_V4.search(text):
        physical, context_kind = study_context if study_context is not None else _research_context_v4(document)
        if not physical and context_kind != "research":
            return context_kind
    if _EMPIRICAL_V4.search(text) or _LIVING_SYSTEM_RESULT.search(text):
        return "experimental"
    return "research"


def automatic_research_application(quote: str, document: DocumentRecord, *,
                                   method_version: str = RAW_GROUNDING_METHOD) -> bool:
    """New automatic research application requires an actual empirical benefit."""
    if method_version != RAW_GROUNDING_METHOD:
        raise EvidenceError("Неизвестная версия автоматического применения.")
    selection = QuoteSelection(role="advantage", revision_id="screen", field="abstract", quote=quote)
    return bool(_selection_supported(selection, document, method_version=method_version)
        and _primary_sentence_supported_v4(quote, document.abstract or "")
        and primary_result_kind(quote, document) == "experimental")


def primary_result_sentence(document: DocumentRecord, *, method_version: str = PRIMARY_GROUNDING_METHOD) -> str | None:
    """A dated primary paper's own result, with theory left explicitly as theory.

    A contextual limitation elsewhere does not erase an actual reported result;
    an explicit contrary conclusion about those results still vetoes it.
    """
    from app.pilot.sources import primary_research_exclusion
    from app.pilot.retractions import is_explicitly_retracted

    abstract = document.abstract or ""
    if method_version not in {PRIMARY_GROUNDING_METHOD, RAW_GROUNDING_METHOD}:
        raise EvidenceError("Неизвестная версия первичного результата.")
    modern = method_version == RAW_GROUNDING_METHOD
    rules_version = "publication-status/2.0.0" if modern else "publication-status/1.0.0"
    if (primary_research_exclusion(document, rules_version=rules_version)
            or is_explicitly_retracted(document, rules_version=rules_version)
            or (_contrary_result_v4(abstract) if modern else _RESULT_CONTRADICTION.search(abstract))):
        return None
    if modern:
        sentences = [sentence for sentence in grounding_sentences(abstract, method_version=method_version)
                     if _primary_sentence_supported_v4(sentence, abstract, context_contradicts=False)]
        def nonempirical(sentence: str) -> bool:
            screened = _screening_text(sentence)
            return bool(_COMPUTATIONAL_V4.search(screened) or _THEORETICAL_V4.search(screened)
                        or not (_EMPIRICAL_V4.search(screened) or _LIVING_SYSTEM_RESULT.search(screened)))
        return min(sentences, key=nonempirical, default=None)
    sentences = [sentence for sentence in _sentences(abstract)
        if not _SPECULATIVE_RESULT.search(sentence) and not _BASELINE_ADVANTAGE.search(sentence)
        and not _PAST_OWN_RESULT.search(sentence)
        and (_PRIMARY_RESULT.search(sentence) or (_EXPERIMENT.search(sentence) and _ADVANTAGE.search(sentence)))]
    return min(sentences, key=lambda sentence: not (_EXPERIMENT.search(sentence) or _LIVING_SYSTEM_RESULT.search(sentence)),
               default=None)


def _extractive_selections(documents: tuple[tuple[DocumentRevisionRef, DocumentRecord], ...], *,
                          method_version: str = "exact-contextual-quotation/2.0.0") -> tuple[QuoteSelection, ...]:
    result: list[QuoteSelection] = []
    if method_version in {PRIMARY_GROUNDING_METHOD, RAW_GROUNDING_METHOD}:
        for reference, document in documents:
            sentence = primary_result_sentence(document, method_version=method_version)
            if sentence:
                result.append(QuoteSelection(role="case", revision_id=reference.revision_id, field="abstract", quote=sentence))
                break
    elif documents:
        reference, document = documents[0]
        result.append(QuoteSelection(role="case", revision_id=reference.revision_id, field="title", quote=document.title[:1500]))
    for role in ("problem", "advantage"):
        for reference, document in documents:
            if method_version == RAW_GROUNDING_METHOD:
                contradicts = _contrary_result_v4(document.abstract or "")
                sentence = next((item for item in grounding_sentences(document.abstract or "", method_version=method_version)
                                 if _raw_role_supported(role, item, context_contradicts=contradicts)), None)
            else:
                sentence = next((item for item in grounding_sentences(document.abstract or "", method_version=method_version)
                                 if _role_supported(role, item, context=document.abstract or "", method_version=method_version)), None)
            if sentence:
                result.append(QuoteSelection.model_validate(dict(role=role, revision_id=reference.revision_id,
                                                                 field="abstract", quote=sentence)))
                break
    return tuple(result)


def _selection_supported(selection: QuoteSelection, document: DocumentRecord, *,
                         method_version: str = "exact-contextual-quotation/2.0.0") -> bool:
    text = archived_field(document, selection.field)
    if method_version == RAW_GROUNDING_METHOD:
        from app.pilot.sources import primary_research_exclusion
        from app.pilot.retractions import is_explicitly_retracted

        if (selection.field != "abstract" or selection.quote not in grounding_sentences(text, method_version=method_version)
                or is_explicitly_retracted(document, rules_version="publication-status/2.0.0")):
            return False
        if selection.role in {"case", "advantage"} and primary_research_exclusion(document, rules_version="publication-status/2.0.0"):
            return False
        if selection.role == "case":
            return _primary_sentence_supported_v4(selection.quote, text)
        if selection.role == "advantage" and primary_result_kind(selection.quote, document) in {"computational", "theoretical"}:
            return False
        return _role_supported(selection.role, selection.quote, context=text, method_version=method_version)
    if selection.role == "case":
        if method_version == PRIMARY_GROUNDING_METHOD:
            from app.pilot.sources import primary_research_exclusion
            from app.pilot.retractions import is_explicitly_retracted
            return bool(selection.field == "abstract" and selection.quote in _sentences(text)
                and not primary_research_exclusion(document, rules_version="publication-status/1.0.0")
                and not is_explicitly_retracted(document, rules_version="publication-status/1.0.0")
                and not _SPECULATIVE_RESULT.search(selection.quote) and not _BASELINE_ADVANTAGE.search(selection.quote)
                and not _PAST_OWN_RESULT.search(selection.quote)
                and not _RESULT_CONTRADICTION.search(text)
                and (_PRIMARY_RESULT.search(selection.quote)
                     or (_EXPERIMENT.search(selection.quote) and _ADVANTAGE.search(selection.quote))))
        # A literature survey is useful context, not a concrete development case.
        return not bool(re.search(r"\b(review|survey|roadmap)\b|обзор", document.title, re.IGNORECASE))
    if selection.field != "abstract":
        return False
    return _role_supported(selection.role, selection.quote, context=text, method_version=method_version)


def build_passport(candidate: Candidate, discovery_snapshot: CorpusSnapshot, archive: DocumentArchive,
                   context: RunContext, *, client: LlmClient | None = None,
                   scope_ids: Sequence[str] = (), methodology_version: MethodologyVersion = "3.1.0") -> TrendCard:
    documents = member_documents(candidate, discovery_snapshot, archive, context)
    modern = methodology_version == "3.4.0"
    primary = methodology_version in {"3.3.0", "3.4.0"}
    prompt_version = (RAW_PASSPORT_PROMPT_VERSION if modern else
                      PRIMARY_PASSPORT_PROMPT_VERSION if primary else PASSPORT_PROMPT_VERSION)
    grounding_method = (RAW_GROUNDING_METHOD if modern else
                        PRIMARY_GROUNDING_METHOD if primary else "exact-contextual-quotation/2.0.0")
    snapshot_digest = discovery_snapshot.snapshot_hash
    blocked_studies, supporting_studies = (_snapshot_study_status(discovery_snapshot, archive, context,
                                                                  snapshot_digest=snapshot_digest)
                                           if modern else (frozenset(), frozenset()))
    input_hash = content_hash({"candidate": candidate.model_dump(mode="json"), "snapshot": snapshot_digest,
                               "version": prompt_version})
    stage = "passport_" + content_hash({"id": candidate.candidate_id})[:20]
    checkpoint = context.load_checkpoint(stage)
    if checkpoint is not None:
        if checkpoint.get("input_hash") != input_hash:
            raise EvidenceError("Паспорт относится к другой версии кандидата.")
        card = TrendCard.model_validate(checkpoint["card"])
        if card.candidate != candidate:
            raise EvidenceError("Сохранённый паспорт содержит другого кандидата.")
        saved_documents = {reference.revision_id: document for reference, document in documents}
        for evidence in card.evidence:
            document = saved_documents.get(evidence.revision_id)
            if document is None:
                raise EvidenceError("В паспорте обнаружена ссылка за пределами кандидата.")
            verify_evidence_text(evidence, archived_field(document, evidence.text_field))
        return card
    representatives = representative_documents(documents, limit=5)
    selections = _extractive_selections(representatives, method_version=grounding_method)
    interpretation = None
    limitations = ["Поля содержат исходные утверждения авторов с консервативной проверкой контекста; независимая репликация и полная семантическая проверка не подразумеваются.",
                   "Кандидат ещё не проверен по полному историческому корпусу."]
    if candidate.specificity != "specific_technology":
        limitations.append("Связность и конкретные технологические границы группы ещё требуют проверки.")
    receipt = None
    failure_code = None
    public_members = all(doc.source in {"openalex", "crossref", "arxiv"} for _, doc in representatives)
    if not public_members:
        limitations.append("Документы без разрешения на передачу внешнему AI обработаны только локально.")
    if client is not None and public_members:
        try:
            completion = client.generate_json(PassportDraft,
                system_prompt="Select exact contiguous source quotations supporting a problem, advantage and research case "
                    "for this technological candidate. Never change spelling or join separate passages. Return no selection "
                    "for an unsupported field. Use only supplied revision IDs and TITLE/ABSTRACT. An asserted advantage "
                    "must be a complete sentence about this mechanism's actual claimed result, not a hypothesis, prediction, "
                    "simulation, comparator/baseline, earlier work, aspiration or negated result. Check the entire abstract "
                    "for qualifications or a contrary conclusion; omit ambiguous benefits. A Russian interpretation "
                    "may explain the quotes but is explicitly UNVERIFIED and cannot assert novelty or invent companies. "
                    + ("A CASE must quote a full ABSTRACT sentence reporting this primary paper's own concrete result; "
                       "a title, review, dataset or proposed future experiment is not a development case. "
                       "Computational and theoretical results may be research cases, never experimental deployment." if primary else ""),
                user_content=json.dumps({"candidate": candidate.label, "definition": candidate.definition,
                    "documents": [{"revision_id": ref.revision_id, "title": doc.title[:1500],
                                   "abstract": (doc.abstract or "")[:3500]} for ref, doc in representatives]}, ensure_ascii=False),
                prompt_version=prompt_version,
                request_id=f"{context.run_id}:passport:{content_hash({'id': candidate.candidate_id})[:20]}:attempt:{context.attempt}",
                scope_ids=scope_ids, cancel=context.cancel_event, max_output_tokens=3000)
            proposed = completion.value.selections
            by_revision = {ref.revision_id: (ref, doc) for ref, doc in representatives}
            for selection in proposed:
                if selection.revision_id not in by_revision:
                    raise EvidenceError("AI сослался на исследование вне предоставленного кандидата.")
                ref, doc = by_revision[selection.revision_id]
                if modern and selection.field == "abstract":
                    raw_quote_evidence(ref, doc, quote=selection.quote)
                else:
                    quote_evidence(ref, doc, text_field=selection.field, quote=selection.quote)
                if not _selection_supported(selection, doc, method_version=grounding_method):
                    raise EvidenceError("Предложенная цитата не подтверждает роль поля паспорта.")
            selections = proposed or selections
            interpretation = completion.value.russian_interpretation
        except LlmCancelled:
            raise TaskCancelled() from None
        except (LlmError, BudgetError, EvidenceError) as error:
            limitations.append("AI-описание не принято: " + str(error))
            failure_code = _failure_code(error)
        finally:
            receipt = client.last_receipt.to_dict() if client.last_receipt else None
    by_revision = {ref.revision_id: (ref, doc) for ref, doc in documents}
    grounded_candidate = (candidate_claims_grounded(candidate, documents)
                          and not (blocked_studies | supporting_studies).intersection(candidate.discovery_study_ids)) if modern else True
    evidence_items: dict[str, Evidence] = {}
    claims: list[Claim] = []
    for selection in selections:
        context.check_cancelled()
        reference, document = by_revision[selection.revision_id]
        item = (raw_quote_evidence(reference, document, quote=selection.quote)
                if modern and selection.field == "abstract" else
                quote_evidence(reference, document, text_field=selection.field, quote=selection.quote))
        evidence_items[item.evidence_id] = item
        supported = (_selection_supported(selection, document, method_version=grounding_method)
                     and (selection.role != "advantage" or grounded_candidate)
                     and reference.study_id not in blocked_studies and document.document_key not in blocked_studies
                     and (selection.role not in {"advantage", "case"}
                          or (reference.study_id not in supporting_studies and document.document_key not in supporting_studies)))
        claims.append(Claim(claim_id="claim-" + content_hash({"role": selection.role, "evidence": item.evidence_id}),
            role=selection.role, text=item.quote, support="supported" if supported else "unverified",
            evidence_ids=(item.evidence_id,), grounding_method=grounding_method))
        if (selection.role == "advantage" and supported
                and (automatic_research_application(selection.quote, document) if modern else _EXPERIMENT.search(selection.quote))
                and not any(claim.role == "application" for claim in claims)):
            claims.append(Claim(claim_id="application-" + item.evidence_id, role="application", text=item.quote,
                support="supported", evidence_ids=(item.evidence_id,),
                grounding_method=RESEARCH_APPLICATION_METHOD if modern else "verified-application/research"))
    # Duplicate selections cannot create duplicate apparent support.
    claims = list({claim.claim_id: claim for claim in claims}.values())
    if interpretation:
        claims.append(Claim(claim_id="interpretation-" + content_hash({"text": interpretation}), role="summary",
            text=interpretation, support="unverified", evidence_ids=tuple(evidence_items)))
        limitations.append("Русское пояснение сформировано AI и не прошло независимую проверку смысла.")
    roles = {claim.role for claim in claims if claim.support == "supported"}
    for role, label in (("problem", "проблема"), ("advantage", "преимущество"), ("case", "кейс")):
        if role not in roles:
            limitations.append("Не найдено подходящего прямого свидетельства для поля: " + label + ".")
    card = TrendCard(candidate=candidate, methodology_version=methodology_version,
                     category=("insufficient_evidence" if candidate.specificity == "specific_technology"
                                                  else "unassessed_cluster"), quality="partial", claims=tuple(claims),
                     evidence=tuple(evidence_items.values()), limitations=tuple(limitations))
    context.checkpoint(stage, {"input_hash": input_hash, "card": card.model_dump(mode="json"), "receipt": receipt,
                               "failure_code": failure_code})
    return card
