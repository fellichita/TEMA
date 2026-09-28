"""A resumable, evidence-verified profile job on the existing coordinator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult, content_hash
from app.pilot.export import verify_result
from app.pilot.methodology import AssessmentArtifact
from app.pilot.review import ReviewRecord
from app.runtime.backup import ArchiveError
from app.pilot.multisource.attention import SignalCandidate, build_signal_profile
from app.pilot.multisource.capital import propose_capital_association
from app.pilot.multisource.contracts import (ArxivImportReceipt, ArxivVersion, CapitalDescription,
    CapitalEvent, CapitalImportReceipt, FundingMetric, QueryProfile, QueryTerm, SearchMetric, SearchObservation, SignalProfile, SourceKind,
    SourceSnapshot, TechnologyAssociation, TechnologyConcept, WordstatImportReceipt,
    load_policy)
from app.pilot.multisource.metrics import compute_funding_metric, compute_search_metric
from app.pilot.multisource.store import SignalStore, object_digest
from app.runtime.jobs import RunContext, TaskFailure


class ScientificLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: UUID
    candidate_id: str = Field(min_length=1, max_length=2048)


class SignalRunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["signals"] = "signals"
    query_profile_hash: str
    concept_hashes: tuple[str, ...] = Field(min_length=1, max_length=100)
    wordstat_receipt_hash: str | None = None
    arxiv_receipt_hash: str | None = None
    capital_receipt_hashes: tuple[str, ...] = Field(default=(), max_length=2)
    association_hashes: tuple[str, ...] = Field(default=(), max_length=500)
    base_result_run_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    scientific_links: tuple[ScientificLink, ...] = Field(default=(), max_length=100)
    policy_hash: str
    decision_at: datetime

    @model_validator(mode="after")
    def references(self) -> SignalRunInput:
        hashes = (self.query_profile_hash, self.policy_hash, *self.concept_hashes,
                  *self.capital_receipt_hashes, *self.association_hashes)
        hashes += tuple(value for value in (self.wordstat_receipt_hash, self.arxiv_receipt_hash) if value is not None)
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes):
            raise ValueError("Invalid signal input reference")
        for group in (self.concept_hashes, self.capital_receipt_hashes, self.association_hashes):
            if len(set(group)) != len(group):
                raise ValueError("Duplicate signal input reference")
        if self.decision_at.tzinfo is None or self.decision_at.utcoffset() is None:
            raise ValueError("Signal decision needs a timezone")
        if (self.base_result_run_id is None) != (not self.scientific_links):
            raise ValueError("Scientific links require one base result")
        if not any((self.wordstat_receipt_hash, self.arxiv_receipt_hash,
                    self.capital_receipt_hashes, self.scientific_links)):
            raise ValueError("At least one imported or reviewed source is required")
        if len({item.concept_id for item in self.scientific_links}) != len(self.scientific_links) or len({
                item.candidate_id for item in self.scientific_links}) != len(self.scientific_links):
            raise ValueError("Duplicate scientific mapping")
        return self


def _base_result(read_base: Callable[[str], dict], run_id: str, archive: DocumentArchive,
                 cutoff: datetime) -> tuple[AnalysisResult, str]:
    value = read_base(run_id)
    try:
        result = AnalysisResult.model_validate(value["result"])
        artifacts = tuple(AssessmentArtifact.model_validate(item) for item in value.get("assessments", ()))
        reviews = tuple(ReviewRecord.model_validate(item) for item in value.get("reviews", ()))
        verify_result(result, archive, artifacts, reviews=reviews)
    except (ArchiveError, KeyError, TypeError, ValueError):
        raise TaskFailure("Связанный научный результат не прошёл проверку.") from None
    if result.created_at > cutoff or result.query_plan.as_of > cutoff.date():
        raise TaskFailure("Научный результат появился после среза профиля.")
    return result, content_hash(result)


def _stage(context: RunContext, name: str, value: dict, message: str, number: int) -> None:
    """Replay a stage only when its immutable output still matches the plan."""
    context.progress(name, message, number, 5)
    previous = context.load_checkpoint(name)
    if previous is not None and previous != value:
        raise TaskFailure("Сохранённый этап сигналов не соответствует входным данным.")
    if previous is None:
        context.checkpoint(name, value)


def _snapshot(store: SignalStore, digest: str, source: str, profile_hash: str,
              raw_hash: str, cutoff: datetime, suffix: str, context: RunContext) -> SourceSnapshot:
    value = store.get_object(digest, SourceSnapshot)
    if (value.source != source or value.query_profile_hash != profile_hash or value.raw_hash != raw_hash
            or value.available_at > cutoff or value.observed_at > cutoff or value.retention != "local_allowed"):
        raise TaskFailure("Снимок источника не соответствует области или временному срезу.")
    store.verify_raw(raw_hash, suffix, cancel=context.cancel_event)
    return value


def run_signal_profile(context: RunContext, raw_payload: dict, data_dir: Path,
                       archive: DocumentArchive, read_base: Callable[[str], dict]) -> dict:
    """Only a succeeded coordinator result publishes a signal profile."""
    try:
        plan = SignalRunInput.model_validate(raw_payload)
    except ValueError:
        raise TaskFailure("Некорректный сохранённый план сигналов.") from None
    store = SignalStore(data_dir)
    policy, policy_hash = load_policy()
    if plan.policy_hash != policy_hash:
        raise TaskFailure("Правила сигналов изменились; создайте новый запуск.")
    cutoff = plan.decision_at
    query = store.get_object(plan.query_profile_hash, QueryProfile)
    if query.confirmed_at is None or query.confirmed_at > cutoff:
        raise TaskFailure("Область ещё не была подтверждена на дату анализа.")
    base = None
    base_hash = None
    if plan.base_result_run_id is not None:
        base, base_hash = _base_result(read_base, plan.base_result_run_id, archive, cutoff)
    _stage(context, "signals_plan", {"input_hash": content_hash(plan.model_dump(mode="json"))},
           "Проверка неизменного плана", 0)

    snapshots: dict[str, SourceSnapshot] = {}
    receipt_hashes: list[str] = []
    observations: tuple[SearchObservation, ...] = ()
    events: dict[str, tuple[CapitalEvent, ...]] = {}
    capital_receipts: dict[str, CapitalImportReceipt] = {}
    descriptions: dict[str, CapitalDescription] = {}
    arxiv_revisions: set[str] = set()
    provenance: set[str] = {plan.query_profile_hash}
    if base_hash is not None:
        provenance.add(base_hash)
    if plan.wordstat_receipt_hash is not None:
        receipt = store.get_object(plan.wordstat_receipt_hash, WordstatImportReceipt)
        if receipt.kind != "dynamics" or receipt.query_profile_hash != plan.query_profile_hash or receipt.completed_at > cutoff:
            raise TaskFailure("Неверный или слишком поздний временной ряд Wordstat.")
        source_snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
        raw_suffix = "json" if source_snapshot.adapter_version == "wordstat-api-v2/1" else "csv"
        snapshots["wordstat"] = _snapshot(store, receipt.snapshot_hash, "wordstat", plan.query_profile_hash,
                                           receipt.raw_hash, cutoff, raw_suffix, context)
        observations = tuple(store.get_object(digest, SearchObservation) for digest in receipt.observation_hashes)
        if any(item.snapshot_hash != receipt.snapshot_hash or item.query_profile_hash != plan.query_profile_hash
               for item in observations):
            raise TaskFailure("Поисковое наблюдение принадлежит другому импорту.")
        provenance.update(receipt.observation_hashes)
        receipt_hashes.append(plan.wordstat_receipt_hash)
    if plan.arxiv_receipt_hash is not None:
        receipt_a = store.get_object(plan.arxiv_receipt_hash, ArxivImportReceipt)
        if receipt_a.query_profile_hash != plan.query_profile_hash or receipt_a.completed_at > cutoff:
            raise TaskFailure("arXiv-выгрузка относится к другой области или дате.")
        snapshots["arxiv"] = _snapshot(store, receipt_a.snapshot_hash, "arxiv", plan.query_profile_hash,
                                        receipt_a.raw_hash, cutoff, "xml", context)
        versions = tuple(store.get_object(digest, ArxivVersion) for digest in receipt_a.version_hashes)
        if any(item.raw_hash != receipt_a.raw_hash or item.updated_at > cutoff for item in versions):
            raise TaskFailure("Версия arXiv появилась после временного среза.")
        arxiv_revisions = set(receipt_a.selected_revision_ids)
        if not arxiv_revisions.issubset(item.revision_id for item in versions):
            raise TaskFailure("Выбранный препринт отсутствует в выгрузке.")
        for revision_id in arxiv_revisions:
            context.check_cancelled()
            archive.get(revision_id)
        provenance.update(receipt_a.version_hashes)
        provenance.update(arxiv_revisions)
        receipt_hashes.append(plan.arxiv_receipt_hash)
    for digest in plan.capital_receipt_hashes:
        context.check_cancelled()
        receipt_c = store.get_object(digest, CapitalImportReceipt)
        source = receipt_c.source
        if source in snapshots or receipt_c.query_profile_hash != plan.query_profile_hash or receipt_c.completed_at > cutoff:
            raise TaskFailure("Финансовый импорт повторяется или не соответствует области.")
        snapshots[source] = _snapshot(store, receipt_c.snapshot_hash, source, plan.query_profile_hash,
                                       receipt_c.raw_hash, cutoff, "csv", context)
        if receipt_c.participant_raw_hash is not None:
            store.verify_raw(receipt_c.participant_raw_hash, "csv", cancel=context.cancel_event)
        source_events = tuple(store.get_object(item, CapitalEvent) for item in receipt_c.event_hashes)
        source_descriptions = tuple(store.get_object(item, CapitalDescription)
                                    for item in receipt_c.description_hashes)
        if any(event.source != source or event.source_hash != receipt_c.raw_hash or
               event.available_at > cutoff or event.observed_at > cutoff for event in source_events):
            raise TaskFailure("Финансовое событие не соответствует исходной выгрузке.")
        for event, description, description_hash in zip(source_events, source_descriptions,
                                                        receipt_c.description_hashes, strict=True):
            if (event.event_id != description.event_id or description.source_hash != receipt_c.raw_hash
                    or description.source != source):
                raise TaskFailure("Описание и финансовое событие не совпадают.")
            descriptions[description_hash] = description
        events[source] = source_events
        capital_receipts[source] = receipt_c
        provenance.update(receipt_c.event_hashes)
        provenance.update(receipt_c.description_hashes)
        receipt_hashes.append(digest)
    _stage(context, "signals_imports", {"receipts": receipt_hashes,
            "snapshots": {key: object_digest(value) for key, value in sorted(snapshots.items())}},
           "Проверка импортов и исходных файлов", 1)

    concepts = tuple(store.get_object(digest, TechnologyConcept) for digest in plan.concept_hashes)
    if len({item.concept_id for item in concepts}) != len(concepts):
        raise TaskFailure("Технология включена в анализ более одного раза.")
    science_by_concept = {item.concept_id: item.candidate_id for item in plan.scientific_links}
    if not set(science_by_concept).issubset(item.concept_id for item in concepts):
        raise TaskFailure("Научная связь ссылается на отсутствующую технологию.")
    cards = {item.candidate.candidate_id: (index, item) for index, item in enumerate(base.cards)} if base else {}
    if not set(science_by_concept.values()).issubset(cards):
        raise TaskFailure("Научная связь ссылается на отсутствующую карточку.")
    if any(not set(item.provenance_hashes).intersection(provenance) for item in concepts):
        raise TaskFailure("Происхождение технологии не подтверждено выбранными данными.")
    associations = list(store.get_object(digest, TechnologyAssociation) for digest in plan.association_hashes)
    if any(item.concept_id not in {concept.concept_id for concept in concepts}
           or not set(item.evidence_hashes).issubset(descriptions)
           or item.reviewed_at is not None and item.reviewed_at > cutoff for item in associations):
        raise TaskFailure("Связь технологии с событием не подтверждена выбранным импортом.")
    for concept in concepts:
        for source in ("cordis", "investment_csv"):
            source_receipt = capital_receipts.get(source)
            if source_receipt is None:
                continue
            for event_hash, description_hash in zip(source_receipt.event_hashes,
                                                    source_receipt.description_hashes, strict=True):
                context.check_cancelled()
                event = store.get_object(event_hash, CapitalEvent)
                proposal = propose_capital_association(concept, event, descriptions[description_hash],
                                                       description_hash)
                if proposal is not None and not any(item.concept_id == concept.concept_id
                        and item.subject_id == proposal.subject_id for item in associations):
                    associations.append(proposal)
    for item in associations:
        store.put_object(item, cancel=context.cancel_event)
    _stage(context, "signals_identities", {"concepts": list(plan.concept_hashes),
            "associations": sorted(object_digest(item) for item in associations)},
           "Проверка определений и связей", 2)

    candidates = []
    metrics: list[str] = []
    for concept in concepts:
        context.check_cancelled()
        related = tuple(item for item in associations if item.concept_id == concept.concept_id)
        names = {concept.label.casefold(), *(item.text.casefold() for item in concept.aliases
                                               if item.status == "confirmed" and item.confirmed_at is not None
                                               and item.confirmed_at <= cutoff)}
        search = None
        if "wordstat" in snapshots and query.primary_phrase and query.primary_phrase.casefold() in names:
            search = compute_search_metric(query, snapshots["wordstat"], observations, policy, policy_hash,
                                           decision_at=cutoff, knowledge_cutoff=cutoff,
                                           identity_confirmed=concept.identity_status == "confirmed")
            metrics.append(store.put_object(search, cancel=context.cancel_event))
        equity = grants = None
        if "investment_csv" in snapshots:
            equity = compute_funding_metric("investment_csv", "equity_round", snapshots["investment_csv"],
                                            tuple(item for item in events["investment_csv"] if item.kind == "equity_round"),
                                            related, concept.concept_id, policy_hash,
                                            decision_at=cutoff, knowledge_cutoff=cutoff)
            metrics.append(store.put_object(equity, cancel=context.cancel_event))
        if "cordis" in snapshots:
            grants = compute_funding_metric("cordis", "grant_project", snapshots["cordis"],
                                            events["cordis"], related, concept.concept_id, policy_hash,
                                            decision_at=cutoff, knowledge_cutoff=cutoff)
            metrics.append(store.put_object(grants, cancel=context.cancel_event))
        arxiv_match = bool(arxiv_revisions.intersection(concept.provenance_hashes))
        origins: list[SourceKind] = ["user"]
        science_link = science_by_concept.get(concept.concept_id)
        if science_link is not None:
            origins.append("scientific")
        if arxiv_match:
            origins.append("arxiv")
        if search is not None:
            origins.append("wordstat")
        if any(item.subject_kind == "project" for item in related):
            origins.append("cordis")
        if any(item.subject_kind == "organisation" for item in related):
            origins.append("investment_csv")
        evidence_refs = set(arxiv_revisions.intersection(concept.provenance_hashes))
        evidence_refs.update(digest for item in related for digest in item.evidence_hashes)
        science_position, science_card = cards[science_link] if science_link is not None else (None, None)
        candidates.append(SignalCandidate(concept, tuple(origins), scientific_candidate_id=science_link,
                                          scientific_category=science_card.category if science_card is not None else None,
                                          scientific_order=science_position, search=search, equity=equity, grants=grants,
                                          associations=related, arxiv_unassessed=arxiv_match,
                                          raw_observation_hashes=tuple(sorted(evidence_refs))))
    _stage(context, "signals_metrics", {"metrics": sorted(set(metrics))},
           "Расчёт независимых признаков", 3)
    profile = build_signal_profile(plan.query_profile_hash, policy, policy_hash, tuple(candidates),
                                   profile_id=uuid5(NAMESPACE_URL, "signals:" + context.run_id),
                                   decision_at=cutoff, knowledge_cutoff=cutoff,
                                   base_result_hash=base_hash, base_result_run_id=plan.base_result_run_id,
                                   import_receipt_hashes=tuple(receipt_hashes),
                                   source_snapshot_hashes=tuple(object_digest(item) for item in snapshots.values()))
    profile_hash = store.put_object(profile, cancel=context.cancel_event)
    _stage(context, "signals_profile", {"profile_hash": profile_hash}, "Проверка готового профиля", 4)
    verified = store.get_object(profile_hash, SignalProfile)
    if verified != profile:
        raise TaskFailure("Сохранённый профиль не соответствует расчёту.")
    context.progress("signals_complete", "Профиль сигналов сохранён", 5, 5)
    return {"kind": "signals", "profile_hash": profile_hash, "profile": profile.model_dump(mode="json")}


def verify_signal_profile(store: SignalStore, archive: DocumentArchive, profile_hash: str,
                          read_base: Callable[[str], dict]) -> SignalProfile:
    """Verify the published root and every local source object before display."""
    profile = store.get_object(profile_hash, SignalProfile)
    _, policy_hash = load_policy()
    if profile.policy_hash != policy_hash:
        raise TaskFailure("Для профиля недоступны его исходные правила или научный результат.")
    if profile.base_result_run_id is not None:
        base, base_hash = _base_result(read_base, profile.base_result_run_id, archive, profile.knowledge_cutoff)
        if base_hash != profile.base_result_hash:
            raise TaskFailure("Связанный научный результат изменился.")
        cards = {item.candidate.candidate_id: item for item in base.cards}
        if any(item.scientific_candidate_id not in cards or
               item.scientific_category != cards[item.scientific_candidate_id].category
               for item in profile.findings if item.scientific_candidate_id is not None):
            raise TaskFailure("Научная оценка карточки не соответствует исходному результату.")
    store.get_object(profile.query_profile_hash, QueryProfile)
    snapshots = set()
    facts: set[str] = set()
    event_hashes: set[str] = set()
    observation_hashes: set[str] = set()
    for receipt_hash in profile.import_receipt_hashes:
        receipt: WordstatImportReceipt | ArxivImportReceipt | CapitalImportReceipt
        try:
            receipt = store.get_object(receipt_hash, WordstatImportReceipt)
        except TaskFailure:
            try:
                receipt = store.get_object(receipt_hash, ArxivImportReceipt)
            except TaskFailure:
                receipt = store.get_object(receipt_hash, CapitalImportReceipt)
        if receipt.query_profile_hash != profile.query_profile_hash:
            raise TaskFailure("Импорт профиля относится к другой области.")
        snapshot = store.get_object(receipt.snapshot_hash, SourceSnapshot)
        if (snapshot.query_profile_hash != profile.query_profile_hash or snapshot.raw_hash != receipt.raw_hash
                or snapshot.observed_at > profile.knowledge_cutoff):
            raise TaskFailure("Снимок профиля относится к другому источнику или времени.")
        suffix = ("xml" if isinstance(receipt, ArxivImportReceipt) else
                  "json" if isinstance(receipt, WordstatImportReceipt)
                  and snapshot.adapter_version == "wordstat-api-v2/1" else "csv")
        store.verify_raw(receipt.raw_hash, suffix)
        snapshots.add(receipt.snapshot_hash)
        if isinstance(receipt, WordstatImportReceipt):
            if snapshot.source != "wordstat":
                raise TaskFailure("Тип поискового снимка повреждён.")
            for digest in receipt.observation_hashes:
                item = store.get_object(digest, SearchObservation)
                if item.snapshot_hash != receipt.snapshot_hash:
                    raise TaskFailure("Поисковое наблюдение относится к другому снимку.")
                observation_hashes.add(digest)
            for digest in receipt.term_hashes:
                store.get_object(digest, QueryTerm)
                facts.add(digest)
        elif isinstance(receipt, ArxivImportReceipt):
            if snapshot.source != "arxiv":
                raise TaskFailure("Тип arXiv-снимка повреждён.")
            revisions = set()
            for digest in receipt.version_hashes:
                arxiv_version = store.get_object(digest, ArxivVersion)
                if arxiv_version.raw_hash != receipt.raw_hash:
                    raise TaskFailure("Версия arXiv относится к другой выгрузке.")
                revisions.add(arxiv_version.revision_id)
            if not set(receipt.selected_revision_ids).issubset(revisions):
                raise TaskFailure("Ссылка на препринт отсутствует в выгрузке.")
            for digest in receipt.selected_revision_ids:
                archive.get(digest)
                facts.add(digest)
        else:
            if snapshot.source != receipt.source:
                raise TaskFailure("Тип финансового снимка повреждён.")
            if receipt.participant_raw_hash is not None:
                store.verify_raw(receipt.participant_raw_hash, "csv")
            for event_hash, description_hash in zip(receipt.event_hashes, receipt.description_hashes, strict=True):
                event = store.get_object(event_hash, CapitalEvent)
                description = store.get_object(description_hash, CapitalDescription)
                if (event.source_hash != receipt.raw_hash or event.event_id != description.event_id
                        or description.source_hash != receipt.raw_hash):
                    raise TaskFailure("Финансовое событие не соответствует выгрузке.")
                event_hashes.add(event_hash)
                facts.add(description_hash)
    if snapshots != set(profile.source_snapshot_hashes):
        raise TaskFailure("Профиль содержит неполный перечень исходных снимков.")
    concepts = tuple(store.get_object(digest, TechnologyConcept) for digest in profile.concept_artifact_hashes)
    if {item.concept_id for item in concepts} != {item.concept_id for item in profile.findings}:
        raise TaskFailure("Технологии и карточки профиля не совпадают.")
    for digest in profile.association_artifact_hashes:
        association = store.get_object(digest, TechnologyAssociation)
        if association.concept_id not in {concept.concept_id for concept in concepts} or not set(association.evidence_hashes).issubset(facts):
            raise TaskFailure("Связь технологии не подтверждена сохранённым источником.")
    for digest in profile.metric_artifact_hashes:
        try:
            metric: SearchMetric | FundingMetric = store.get_object(digest, SearchMetric)
        except TaskFailure:
            metric = store.get_object(digest, FundingMetric)
        if (metric.policy_hash != profile.policy_hash or metric.snapshot_hash not in snapshots
                or metric.decision_at != profile.decision_at or metric.knowledge_cutoff != profile.knowledge_cutoff):
            raise TaskFailure("Метрика относится к другому правилу или снимку.")
        if isinstance(metric, SearchMetric):
            if not set(metric.used_observation_hashes).issubset(observation_hashes):
                raise TaskFailure("Метрика ссылается на отсутствующие поисковые данные.")
        elif not set(metric.event_hashes).issubset(event_hashes):
            raise TaskFailure("Метрика ссылается на отсутствующие финансовые события.")
    admissible = facts | event_hashes | observation_hashes
    if any(not set(item.observation_hashes).issubset(admissible)
           or not set(item.metric_hashes).issubset(profile.metric_artifact_hashes)
           for item in profile.findings):
        raise TaskFailure("Карточка ссылается на отсутствующее наблюдение.")
    return profile
