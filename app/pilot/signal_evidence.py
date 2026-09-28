"""Reproducible author assertions for automatic hypotheses, never novelty review.

An archived experimental author's explicit claim to a new concrete mechanism
can justify investigation. It does not establish novelty against all prior art,
independent replication, a trend, or a calibrated likelihood of future growth.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.backend.contracts import DocumentRecord
from app.pilot.archive import RevisionSource, document_text
from app.pilot.contracts import Candidate, CorpusSnapshot, DocumentRevisionRef, Evidence, QueryPlan, require_unique
from app.pilot.evidence import (
    EvidenceContext, EvidenceError, TITLE_ADMISSION_VERSION, _sentences, admission_hash,
    candidate_title_matches, member_documents, quote_evidence, scope_is_anchored, primary_result_sentence,
    _RESULT_CONTRADICTION, _LIVING_SYSTEM_RESULT,
    RAW_GROUNDING_METHOD, PRIMARY_GROUNDING_METHOD, grounding_sentences, primary_result_kind,
    _screening_text, _contrary_result_v4, _UNREALIZED_V4, _PRIOR_ASSERTION_V4,
    snapshot_primary_exclusions, raw_quote_evidence, _research_context_v4,
)
from app.pilot.retractions import is_explicitly_retracted

SIGNAL_EVIDENCE_VERSION: Final = "archived-author-novelty/1.0.0"
CONTEXTUAL_SIGNAL_EVIDENCE_VERSION: Final = "archived-author-novelty/2.0.0"
RAW_SIGNAL_EVIDENCE_VERSION: Final = "archived-author-novelty/3.0.0"
PRIMARY_EVIDENCE_VERSION: Final = "archived-primary-result/1.0.0"
RAW_PRIMARY_EVIDENCE_VERSION: Final = "archived-primary-result/2.0.0"
SignalEvidenceVersion = Literal["archived-author-novelty/1.0.0", "archived-author-novelty/2.0.0", "archived-author-novelty/3.0.0"]
PrimaryEvidenceVersion = Literal["archived-primary-result/1.0.0", "archived-primary-result/2.0.0"]
AUTHOR_NOVELTY_LIMITATION: Final = (
    "Источник содержит заявление авторов о новом механизме и описание собственного эксперимента. "
    "Это основание для автоматической гипотезы, а не независимое подтверждение мировой новизны, "
    "репликации или зарождающегося тренда."
)
_AUTHOR = re.compile(r"\b(?:we|our|herein|this\s+(?:work|study|paper))\b|\bмы\b|\bнаш", re.I)
_NEW = re.compile(r"\b(?:new|novel|first)\b|\bнов(?:ый|ая|ое|ые|ого|ую|ым)\b|\bвпервые\b", re.I)
_INTRODUCE = re.compile(r"\b(?:introduc\w*|develop\w*|demonstrat\w*|report\w*|present\w*|engineer\w*|fabricat\w*|creat\w*|design\w*)\b|разработ|демонстр|созда|представ", re.I)
_EXPERIMENT = re.compile(r"\b(?:measured|tested|fabricated|observed|experimentally\s+(?:demonstrated|validated|confirmed)|laboratory\s+(?:experiments|measurements)|in\s+vivo|in\s+vitro)\b|\bизмерили\b|\bизготовили\b|\bиспытали\b|\bэкспериментально\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:could|might|may|would|hope\w*|aim\w*|intend\w*|potential\w*|hypothes\w*|speculat\w*|propos\w*|simulat\w*|theoretic\w*|predict\w*|first.principles|future|not|never|unable|fail\w*|negligible|insignificant)\b|гипотез|предполага|моделирован|пренебрежимо|\bне\b|\bможет\b", re.I)
_REVIEW = re.compile(r"\b(?:review|survey|roadmap|perspective|meta.analysis)\b|обзор", re.I)
_PRIOR_CLAIM = re.compile(r"\b(?:previous\w*|prior|earlier|conventional|existing|established|baseline|others|colleagues)\b|предыдущ|ранее|существующ", re.I)
_NON_TECH_NOVELTY = re.compile(r"\b(?:new|novel|first)\s+(?:(?:a|an|the|large|comprehensive|systematic)\s+)*(?:dataset|benchmark|review|survey|database|bibliometric|application|perspective|roadmap|analysis)\b", re.I)


class SourceNoveltyEvidence(BaseModel):
    """Two same-study, full-sentence quotations with their complete-field hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    method_version: SignalEvidenceVersion = SIGNAL_EVIDENCE_VERSION
    candidate_id: str = Field(min_length=1, max_length=2048)
    admission_rule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    discovery_snapshot_id: str = Field(min_length=1, max_length=2048)
    novelty: Evidence
    experiment: Evidence
    publication_year: int = Field(ge=1000, le=9999, strict=True)
    limitation: Literal[
        "Источник содержит заявление авторов о новом механизме и описание собственного эксперимента. "
        "Это основание для автоматической гипотезы, а не независимое подтверждение мировой новизны, "
        "репликации или зарождающегося тренда."
    ] = AUTHOR_NOVELTY_LIMITATION

    @model_validator(mode="after")
    def same_source(self) -> Self:
        if (self.novelty.revision_id != self.experiment.revision_id
                or self.novelty.study_id != self.experiment.study_id
                or self.novelty.text_field != "abstract" or self.experiment.text_field != "abstract"):
            raise ValueError("Novelty and experiment must quote the same archived study abstract")
        return self


def _from_source(candidate: Candidate, reference: DocumentRevisionRef, document: DocumentRecord,
                 *, as_of_year: int, method_version: SignalEvidenceVersion = SIGNAL_EVIDENCE_VERSION,
                 as_of_date: date | None = None) -> SourceNoveltyEvidence | None:
    abstract = document.abstract or ""
    modern = method_version == RAW_SIGNAL_EVIDENCE_VERSION
    contextual = method_version in {CONTEXTUAL_SIGNAL_EVIDENCE_VERSION, RAW_SIGNAL_EVIDENCE_VERSION}
    rules_version = "publication-status/2.0.0" if modern else "publication-status/1.0.0"
    from app.pilot.sources import primary_research_exclusion
    if (document.publication_year is None or document.publication_year > as_of_year
            or document.document_type.casefold() in {"review", "patent", "report"}
            or is_explicitly_retracted(document, rules_version=rules_version)
            or (primary_research_exclusion(document, rules_version=rules_version)
                or (_contrary_result_v4(abstract) if modern else _RESULT_CONTRADICTION.search(abstract)) if contextual
                else _REVIEW.search(document.title + " " + abstract) or _UNCERTAIN.search(abstract))):
        return None
    if modern and as_of_date is not None and (
            (document.publication_date is not None and document.publication_date > as_of_date)
            or (document.publication_year == as_of_date.year and document.publication_month is not None
                and document.publication_month > as_of_date.month)):
        return None
    sentences = grounding_sentences(abstract, method_version=RAW_GROUNDING_METHOD) if modern else _sentences(abstract)
    if modern:
        study_context = _research_context_v4(document)
        eligible = [(sentence, _screening_text(sentence)) for sentence in sentences]
        eligible = [(raw, text) for raw, text in eligible
                    if not _UNREALIZED_V4.search(text) and not _PRIOR_ASSERTION_V4.search(text)]
        novelty = next((raw for raw, text in eligible if _AUTHOR.search(text) and _NEW.search(text)
            and _INTRODUCE.search(text) and not _UNCERTAIN.search(text) and not _NON_TECH_NOVELTY.search(text)
            and candidate_title_matches(text, candidate)), None)
        experiment = next((raw for raw, text in eligible if _AUTHOR.search(text) and _EXPERIMENT.search(text)
            and not _UNCERTAIN.search(text) and primary_result_kind(raw, document, study_context=study_context) == "experimental"), None)
    else:
        novelty = next((sentence for sentence in sentences if _AUTHOR.search(sentence)
            and _NEW.search(sentence) and _INTRODUCE.search(sentence)
            and not _PRIOR_CLAIM.search(sentence) and not _NON_TECH_NOVELTY.search(sentence)
            and (not contextual or not _UNCERTAIN.search(sentence))
            and candidate_title_matches(sentence, candidate)), None)
        experiment = next((sentence for sentence in sentences if _AUTHOR.search(sentence)
            and _EXPERIMENT.search(sentence) and not _PRIOR_CLAIM.search(sentence)
            and (not contextual or not _UNCERTAIN.search(sentence))), None)
    if novelty is None or experiment is None:
        return None
    return SourceNoveltyEvidence(method_version=method_version, candidate_id=candidate.candidate_id,
        admission_rule_hash=candidate.admission_rule_hash,
        discovery_snapshot_id=candidate.discovery_snapshot_id,
        novelty=(raw_quote_evidence(reference, document, quote=novelty) if modern else
                 quote_evidence(reference, document, text_field="abstract", quote=novelty)),
        experiment=(raw_quote_evidence(reference, document, quote=experiment) if modern else
                    quote_evidence(reference, document, text_field="abstract", quote=experiment)),
        publication_year=document.publication_year)


def extract_signal_evidence(candidate: Candidate, snapshot: CorpusSnapshot, archive: RevisionSource,
                            context: EvidenceContext, *, query_plan: QueryPlan,
                            method_version: SignalEvidenceVersion = SIGNAL_EVIDENCE_VERSION) -> tuple[SourceNoveltyEvidence, ...]:
    """Strict deterministic author evidence; absence is not a negative novelty verdict.

    Scope, complete candidate names and concrete-object gates precede quotation
    extraction. No generated text, metadata keywords or claimed evidence prefixes
    qualify. The complete abstract is screened, not a cherry-picked fragment.
    """
    if method_version not in {SIGNAL_EVIDENCE_VERSION, CONTEXTUAL_SIGNAL_EVIDENCE_VERSION, RAW_SIGNAL_EVIDENCE_VERSION}:
        raise EvidenceError("Неизвестная версия авторских свидетельств.")
    if query_plan.plan_hash != snapshot.plan_hash or candidate.plan_hash != query_plan.plan_hash:
        raise EvidenceError("Область автоматического свидетельства не соответствует снимку.")
    if (candidate.specificity != "specific_technology"
            or candidate.admission_rule_version != TITLE_ADMISSION_VERSION
            or candidate.admission_rule_hash != admission_hash(candidate)):
        return ()
    documents = member_documents(candidate, snapshot, archive, context)
    if not scope_is_anchored(documents, query_plan, rule_version=candidate.scope_rule_version) or not all(
            candidate_title_matches(document.title, candidate) for _, document in documents):
        return ()
    result = []
    blocked = snapshot_primary_exclusions(snapshot, archive, context) if method_version == RAW_SIGNAL_EVIDENCE_VERSION else frozenset()
    for reference, document in documents:
        context.check_cancelled()
        if reference.study_id in blocked or document.document_key in blocked:
            continue
        item = _from_source(candidate, reference, document, as_of_year=snapshot.as_of.year,
                            method_version=method_version, as_of_date=query_plan.as_of)
        if item is not None:
            result.append(item)
    return tuple(sorted(result, key=lambda item: (item.publication_year, item.novelty.study_id))[:10])


def verify_signal_evidence(evidence: tuple[SourceNoveltyEvidence, ...], candidate: Candidate,
                           snapshot: CorpusSnapshot, archive: RevisionSource, context: EvidenceContext,
                           *, query_plan: QueryPlan,
                           method_version: SignalEvidenceVersion | None = None,
                           excluded_study_ids: frozenset[str] = frozenset()) -> None:
    """Require exact replay, including archive membership and full source context."""
    require_unique(tuple(item.novelty.study_id for item in evidence), "automatic novelty studies")
    if method_version is not None and method_version not in {SIGNAL_EVIDENCE_VERSION, CONTEXTUAL_SIGNAL_EVIDENCE_VERSION, RAW_SIGNAL_EVIDENCE_VERSION}:
        raise EvidenceError("Неизвестная версия авторских свидетельств.")
    versions = {item.method_version for item in evidence}
    if len(versions) > 1:
        raise EvidenceError("Нельзя смешивать версии извлечения авторских свидетельств.")
    effective: SignalEvidenceVersion = method_version if method_version is not None else next(iter(versions), SIGNAL_EVIDENCE_VERSION)
    if versions and versions != {effective}:
        raise EvidenceError("Версия авторского свидетельства не соответствует запуску.")
    expected = extract_signal_evidence(candidate, snapshot, archive, context, query_plan=query_plan, method_version=effective)
    if excluded_study_ids and effective != RAW_SIGNAL_EVIDENCE_VERSION:
        raise EvidenceError("Общий контекст статусов не изменяет прежние версии авторских свидетельств.")
    expected = tuple(item for item in expected if item.novelty.study_id not in excluded_study_ids)
    if evidence != expected:
        raise EvidenceError("Автоматическое свидетельство не воспроизводится из архивного контекста.")


def verify_signal_sources(evidence: tuple[SourceNoveltyEvidence, ...], candidate: Candidate,
                          archive: RevisionSource, context: EvidenceContext, *, query_plan: QueryPlan,
                          method_version: SignalEvidenceVersion | None = None) -> None:
    """Replay each source in a standalone review, without claiming corpus completeness.

    A full result separately runs verify_signal_evidence against its discovery
    snapshot. Review journals have no complete discovery corpus, so this check
    proves their retained assertions only; it cannot detect omitted candidates.
    """
    require_unique(tuple(item.novelty.study_id for item in evidence), "automatic novelty studies")
    if method_version is not None and method_version not in {SIGNAL_EVIDENCE_VERSION, CONTEXTUAL_SIGNAL_EVIDENCE_VERSION, RAW_SIGNAL_EVIDENCE_VERSION}:
        raise EvidenceError("Неизвестная версия авторских свидетельств.")
    versions = {item.method_version for item in evidence}
    if len(versions) > 1 or (method_version is not None and versions and versions != {method_version}):
        raise EvidenceError("Версия авторского свидетельства не соответствует запуску.")
    if evidence and (candidate.specificity != "specific_technology"
            or candidate.admission_rule_version != TITLE_ADMISSION_VERSION
            or candidate.plan_hash != query_plan.plan_hash or candidate.admission_rule_hash != admission_hash(candidate)):
        raise EvidenceError("Границы автоматического свидетельства изменены.")
    for item in evidence:
        context.check_cancelled()
        document = archive.get(item.novelty.revision_id)
        reference = DocumentRevisionRef.model_validate(dict(revision_id=item.novelty.revision_id, study_id=document.document_key,
            source=document.source, source_id=document.source_id,
            text_hash=hashlib.sha256(document_text(document).encode("utf-8")).hexdigest(),
            observed_at=document.fetched_at, publication_year=document.publication_year,
            publicly_available_at=document.publication_date))
        if (document.document_key not in candidate.discovery_study_ids
                or not candidate_title_matches(document.title, candidate)
                or not scope_is_anchored(((reference, document),), query_plan, rule_version=candidate.scope_rule_version)
                or (document.publication_date is not None and document.publication_date > query_plan.as_of)
                or _from_source(candidate, reference, document, as_of_year=query_plan.as_of.year,
                                method_version=item.method_version, as_of_date=query_plan.as_of) != item):
            raise EvidenceError("Автоматическое свидетельство не воспроизводится из архивного контекста.")


class PrimaryStudyEvidence(BaseModel):
    """One exact primary result, separate from completed-year publication counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    method_version: PrimaryEvidenceVersion = PRIMARY_EVIDENCE_VERSION
    candidate_id: str = Field(min_length=1, max_length=2048)
    admission_rule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    discovery_snapshot_id: str = Field(min_length=1, max_length=2048)
    result: Evidence
    publication_year: int = Field(ge=1000, le=9999, strict=True)
    result_kind: Literal["experimental", "computational", "theoretical", "research"]

    @model_validator(mode="after")
    def primary_quotation(self) -> Self:
        if self.result.text_field != "abstract":
            raise ValueError("A primary result requires an exact abstract sentence, not a publication title")
        return self


def _primary_from_source(candidate: Candidate, reference: DocumentRevisionRef, document: DocumentRecord,
                         *, query_plan: QueryPlan,
                         method_version: PrimaryEvidenceVersion = PRIMARY_EVIDENCE_VERSION) -> PrimaryStudyEvidence | None:
    if (document.publication_year is None or document.publication_year > query_plan.as_of.year
            or (document.publication_date is not None and document.publication_date > query_plan.as_of)
            or (document.publication_year == query_plan.as_of.year and document.publication_month is not None
                and document.publication_month > query_plan.as_of.month)):
        return None
    sentence = primary_result_sentence(document, method_version=RAW_GROUNDING_METHOD
                                       if method_version == RAW_PRIMARY_EVIDENCE_VERSION else PRIMARY_GROUNDING_METHOD)
    if sentence is None:
        return None
    kind: Literal["experimental", "computational", "theoretical", "research"] = "research"
    # A simulated experiment remains computational, even if its sentence says measured.
    if re.search(r"\b(?:simulat\w*|numerical\w*|computational\w*)\b|моделирован|численн", sentence, re.I):
        kind = "computational"
    elif re.search(r"\b(?:theoretic\w*|deriv\w*|prove\w*|analytic\w*)\b|теоретич|аналитич|доказ", sentence, re.I):
        kind = "theoretical"
    elif _EXPERIMENT.search(sentence) or _LIVING_SYSTEM_RESULT.search(sentence):
        kind = "experimental"
    if method_version == RAW_PRIMARY_EVIDENCE_VERSION:
        kind = primary_result_kind(sentence, document)
    return PrimaryStudyEvidence(method_version=method_version, candidate_id=candidate.candidate_id, admission_rule_hash=candidate.admission_rule_hash,
        discovery_snapshot_id=candidate.discovery_snapshot_id,
        result=(raw_quote_evidence(reference, document, quote=sentence) if method_version == RAW_PRIMARY_EVIDENCE_VERSION else
                quote_evidence(reference, document, text_field="abstract", quote=sentence)),
        publication_year=document.publication_year, result_kind=kind)


def extract_primary_observations(candidate: Candidate, snapshot: CorpusSnapshot, archive: RevisionSource,
                                 context: EvidenceContext, *, query_plan: QueryPlan,
                                 method_version: PrimaryEvidenceVersion = PRIMARY_EVIDENCE_VERSION) -> tuple[PrimaryStudyEvidence, ...]:
    """Retain specific primary findings, including this year's work, without claiming growth."""
    if method_version not in {PRIMARY_EVIDENCE_VERSION, RAW_PRIMARY_EVIDENCE_VERSION}:
        raise EvidenceError("Неизвестная версия первичных наблюдений.")
    if query_plan.plan_hash != snapshot.plan_hash or candidate.plan_hash != query_plan.plan_hash:
        raise EvidenceError("Область первичного наблюдения не соответствует снимку.")
    if (candidate.specificity != "specific_technology" or candidate.admission_rule_version != TITLE_ADMISSION_VERSION
            or candidate.admission_rule_hash != admission_hash(candidate)):
        return ()
    documents = member_documents(candidate, snapshot, archive, context)
    if not scope_is_anchored(documents, query_plan, rule_version=candidate.scope_rule_version) or not all(
            candidate_title_matches(document.title, candidate) for _, document in documents):
        return ()
    result = []
    blocked = snapshot_primary_exclusions(snapshot, archive, context) if method_version == RAW_PRIMARY_EVIDENCE_VERSION else frozenset()
    for reference, document in documents:
        context.check_cancelled()
        if reference.study_id in blocked or document.document_key in blocked:
            continue
        item = _primary_from_source(candidate, reference, document, query_plan=query_plan, method_version=method_version)
        if item is not None:
            result.append(item)
    # Selection is for a bounded evidence packet, never an estimate of total studies or teams.
    return tuple(sorted(result, key=lambda item: (-item.publication_year, item.result.study_id))[:10])


def verify_primary_observations(evidence: tuple[PrimaryStudyEvidence, ...], candidate: Candidate,
                                snapshot: CorpusSnapshot, archive: RevisionSource, context: EvidenceContext,
                                *, query_plan: QueryPlan,
                                method_version: PrimaryEvidenceVersion | None = None,
                                excluded_study_ids: frozenset[str] = frozenset()) -> None:
    require_unique(tuple(item.result.study_id for item in evidence), "primary observation studies")
    versions = {item.method_version for item in evidence}
    if len(versions) > 1:
        raise EvidenceError("Нельзя смешивать версии первичных наблюдений.")
    effective: PrimaryEvidenceVersion = method_version if method_version is not None else next(iter(versions), PRIMARY_EVIDENCE_VERSION)
    if versions and versions != {effective}:
        raise EvidenceError("Версия первичного наблюдения не соответствует запуску.")
    if excluded_study_ids and effective != RAW_PRIMARY_EVIDENCE_VERSION:
        raise EvidenceError("Общий контекст статусов не изменяет прежние версии первичных наблюдений.")
    expected = extract_primary_observations(candidate, snapshot, archive, context, query_plan=query_plan, method_version=effective)
    expected = tuple(item for item in expected if item.result.study_id not in excluded_study_ids)
    if evidence != expected:
        raise EvidenceError("Первичные наблюдения не воспроизводятся из архивного контекста.")


def verify_primary_sources(evidence: tuple[PrimaryStudyEvidence, ...], candidate: Candidate,
                           archive: RevisionSource, context: EvidenceContext, *, query_plan: QueryPlan,
                           method_version: PrimaryEvidenceVersion | None = None) -> None:
    require_unique(tuple(item.result.study_id for item in evidence), "primary observation studies")
    if method_version is not None and method_version not in {PRIMARY_EVIDENCE_VERSION, RAW_PRIMARY_EVIDENCE_VERSION}:
        raise EvidenceError("Неизвестная версия первичных наблюдений.")
    versions = {item.method_version for item in evidence}
    if len(versions) > 1 or (method_version is not None and versions and versions != {method_version}):
        raise EvidenceError("Версия первичного наблюдения не соответствует запуску.")
    if evidence and (candidate.specificity != "specific_technology" or candidate.admission_rule_version != TITLE_ADMISSION_VERSION
            or candidate.admission_rule_hash != admission_hash(candidate) or candidate.plan_hash != query_plan.plan_hash):
        raise EvidenceError("Границы первичного наблюдения изменены.")
    for item in evidence:
        context.check_cancelled()
        document = archive.get(item.result.revision_id)
        reference = DocumentRevisionRef.model_validate(dict(revision_id=item.result.revision_id, study_id=document.document_key,
            source=document.source, source_id=document.source_id,
            text_hash=hashlib.sha256(document_text(document).encode("utf-8")).hexdigest(),
            observed_at=document.fetched_at, publication_year=document.publication_year,
            publicly_available_at=document.publication_date))
        if (document.document_key not in candidate.discovery_study_ids
                or not candidate_title_matches(document.title, candidate)
                or not scope_is_anchored(((reference, document),), query_plan, rule_version=candidate.scope_rule_version)
                or _primary_from_source(candidate, reference, document, query_plan=query_plan, method_version=item.method_version) != item):
            raise EvidenceError("Первичное наблюдение не воспроизводится из архивного контекста.")
