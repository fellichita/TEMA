"""Explainable, three-lane review queue independent of the scientific TOP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from app.pilot.contracts import Category
from app.pilot.multisource.contracts import (FundingMetric, SearchMetric, SignalFinding, SignalPolicy,
                                             SignalProfile, SourceKind, TechnologyAssociation, TechnologyConcept)
from app.pilot.multisource.store import object_digest
from app.runtime.jobs import TaskFailure

_SCIENCE = {"early_signal", "weak_signal_candidate", "emerging_candidate", "confirmed_trend"}
_MATURE = {"established_topic"}


@dataclass(frozen=True)
class SignalCandidate:
    concept: TechnologyConcept
    origins: tuple[SourceKind, ...]
    scientific_candidate_id: str | None = None
    scientific_category: Category | None = None
    scientific_order: int | None = None
    search: SearchMetric | None = None
    equity: FundingMetric | None = None
    grants: FundingMetric | None = None
    associations: tuple[TechnologyAssociation, ...] = ()
    raw_observation_hashes: tuple[str, ...] = ()
    source_snapshot_hashes: tuple[str, ...] = ()
    arxiv_unassessed: bool = False
    explicitly_rejected: bool = False
    mature_reviewed: bool = False


def _funding_rank(metric: FundingMetric | None) -> tuple[int, int, int, int]:
    if metric is None:
        return 9, 0, 0, 0
    priority = 0 if metric.state == "multiple_units" else 1
    date_rank = int((metric.latest_event_period or "0000-00").replace("-", "").ljust(8, "0"))
    return priority, -metric.unit_count, -len(metric.event_hashes), -date_rank


def _round_robin(lanes: tuple[list[UUID], ...], limit: int) -> tuple[UUID, ...]:
    selected: list[UUID] = []
    seen = set()
    positions = [0] * len(lanes)
    while len(selected) < limit:
        progressed = False
        for index, lane in enumerate(lanes):
            while positions[index] < len(lane) and lane[positions[index]] in seen:
                positions[index] += 1
            if positions[index] >= len(lane):
                continue
            identifier = lane[positions[index]]
            positions[index] += 1
            selected.append(identifier)
            seen.add(identifier)
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return tuple(selected)


def _search_eligible(metric: SearchMetric | None) -> bool:
    return bool(metric is not None and (metric.state == "sustained_growth" or metric.state == "new_in_comparison"
                and not metric.low_volume and (metric.positive_recent_months or 0) >= 2))


def _funding_eligible(metric: FundingMetric | None) -> bool:
    return metric is not None and metric.state in {"multiple_units", "single_unit"}


def _explanation(item: SignalCandidate, queue: str, identity_known: bool) -> tuple[str, str, str, tuple[str, ...]]:
    search, funding = item.search, item.equity if _funding_eligible(item.equity) else item.grants
    if queue == "rejected":
        return "explicit_rejection", "Технологическое соответствие отклонено.", "Проверить определение области.", ()
    if queue == "market_only":
        return "verified_mature", "Тема уже отнесена к зрелым в проверенной научной оценке.", (
            "Смотреть изменения рынка отдельно от очереди ранних идей."), ()
    if queue == "deferred":
        return "attention_capacity", "Есть основание для проверки, но карточка не вошла в первые 15 мест.", (
            "Открыть полный список кандидатов и проверить основания."), ()
    if not identity_known:
        return "identity_unconfirmed", "Смысл технологии ещё не подтверждён.", (
            "Подтвердить название и исключить омонимы."), ()
    if queue == "attention" and item.scientific_category in _SCIENCE and search is not None and search.low_volume:
        return "science_with_low_search", (
            f"Есть научная оценка; по выбранной фразе в Яндексе только {search.recent_count} запросов за три месяца."), (
            "Проверить другое осмысленное название и независимую реализацию."), (
            "Число запросов зависит от фразы, региона и устройств.",)
    if queue == "attention" and item.scientific_category in _SCIENCE:
        return "science_assessed", "Технология имеет отдельную научную оценку.", (
            "Проверить независимые результаты и ограничения научного доказательства."), ()
    if queue == "attention" and _search_eligible(search) and search is not None:
        state_name = "устойчивый нормированный рост" if search.state == "sustained_growth" else (
            "новый интерес по сравнению с выбранным прошлогодним окном")
        return "normalized_search_growth", (
            f"По выбранной фразе в Яндексе наблюдается {state_name}; за три месяца {search.recent_count} запросов. "
            "Научная оценка пока отсутствует."), (
            "Проверить технологический смысл фразы и найти исследования."), (
            "Доля относится к выбранной географии и устройствам, а не ко всему рынку.",)
    if queue == "attention" and funding is not None:
        unit = "проектам" if funding.unit_kind == "projects" else "компаниям"
        return "confirmed_funding_units", (
            f"Найдены события, связанные с {funding.unit_count} {unit}; сумма не приписывается автоматически технологии."), (
            "Проверить результаты проектов и независимость получателей."), (
            "Выборка финансового источника может быть неполной.",)
    if item.arxiv_unassessed:
        return "arxiv_unassessed", "Есть препринт arXiv без завершённой научной оценки.", (
            "Проверить конкретный препринт и независимые подтверждения."), ()
    if search is not None and search.state == "new_in_comparison" and (search.positive_recent_months or 0) == 1:
        return "new_single_period_observation", "Запросы отмечены только в одном недавнем месяце.", (
            "Подождать следующих полных месяцев и проверить смысл фразы."), ()
    if search is not None and search.state in {"low_volume_change", "one_period_spike", "new_in_comparison"}:
        return "search_watch_" + search.state, "Поисковые данные требуют дополнительной проверки.", (
            "Сравнить полные месяцы и подтвердить устойчивость спроса."), ()
    if item.equity is not None and item.equity.proposed_count or item.grants is not None and item.grants.proposed_count:
        return "funding_relation_proposed", "Связь финансового события с технологией ещё не проверена.", (
            "Проверить первичный текст и тип связи на дату события."), ()
    if search is not None and search.state not in {"unavailable"}:
        return "search_observed_watch", "Есть поисковые наблюдения без подтверждённого роста.", (
            "Проверить покрытие, единицу доли и полноту месяцев."), ()
    return "insufficient_channels", "Недостаточно проверенных наблюдений для вывода.", (
        "Подключить источник и проверить технологический смысл."), ()


def build_signal_profile(query_profile_hash: str, policy: SignalPolicy, policy_hash: str,
                         candidates: tuple[SignalCandidate, ...], *, profile_id: UUID,
                         decision_at: datetime, knowledge_cutoff: datetime,
                         base_result_hash: str | None = None,
                         base_result_run_id: str | None = None,
                         import_receipt_hashes: tuple[str, ...] = (),
                         source_snapshot_hashes: tuple[str, ...] = ()) -> SignalProfile:
    """Merges only already-confirmed concept IDs, preserving independent lanes."""
    if len(candidates) > policy.concept_limit or decision_at > knowledge_cutoff:
        raise TaskFailure("Превышен лимит технологий или нарушен временной срез профиля.")
    by_id = {item.concept.concept_id: item for item in candidates}
    if len(by_id) != len(candidates):
        raise TaskFailure("Одна технология повторяется в профиле.")
    if any(item.scientific_category is not None and (not item.scientific_candidate_id or base_result_hash is None)
           or item.search is not None and item.search.policy_hash != policy_hash
           or item.equity is not None and item.equity.policy_hash != policy_hash
           or item.grants is not None and item.grants.policy_hash != policy_hash for item in candidates):
        raise TaskFailure("Метрики и научные ссылки относятся к разным правилам профиля.")
    def allowed(item: SignalCandidate) -> bool:
        return (item.concept.identity_status == "confirmed" and item.concept.confirmed_at is not None
                and item.concept.confirmed_at <= knowledge_cutoff and not item.explicitly_rejected
                and item.scientific_category != "off_scope" and not item.mature_reviewed
                and item.scientific_category not in _MATURE)
    science = sorted((item for item in candidates if allowed(item) and item.scientific_category in _SCIENCE),
                     key=lambda item: (item.scientific_order if item.scientific_order is not None else 10_000,
                                       str(item.concept.concept_id)))
    search = sorted((item for item in candidates if allowed(item) and _search_eligible(item.search)),
                    key=lambda item: (0 if item.search and item.search.state == "sustained_growth" else 1,
                                      -(item.search.persistence_numerator or 0) if item.search else 0,
                                      -Decimal(item.search.yoy_share_change or "0") if item.search else Decimal(0),
                                      -(item.search.positive_recent_months or 0) if item.search else 0,
                                      -(item.search.recent_count or 0) if item.search else 0,
                                      str(item.concept.concept_id)))
    equity = sorted((item for item in candidates if allowed(item) and _funding_eligible(item.equity)),
                    key=lambda item: (*_funding_rank(item.equity), str(item.concept.concept_id)))
    grants = sorted((item for item in candidates if allowed(item) and _funding_eligible(item.grants)),
                    key=lambda item: (*_funding_rank(item.grants), str(item.concept.concept_id)))
    funding = _round_robin((([item.concept.concept_id for item in equity]),
                            ([item.concept.concept_id for item in grants])), len(candidates))
    full_order = _round_robin((([item.concept.concept_id for item in science]),
                               ([item.concept.concept_id for item in search]), list(funding)), len(candidates))
    attention = full_order[:policy.attention_limit]
    attention_set = set(attention)
    eligible_set = set(full_order)
    findings = []
    watches = []
    rejected = []
    source_hashes: set[str] = set(source_snapshot_hashes)
    metric_hashes: set[str] = set()
    concept_hashes: set[str] = set()
    association_hashes: set[str] = set()
    for item in sorted(candidates, key=lambda value: str(value.concept.concept_id)):
        concept_hashes.add(object_digest(item.concept))
        association_hashes.update(object_digest(association) for association in item.associations)
        metrics = tuple(metric for metric in (item.search, item.equity, item.grants) if metric is not None)
        metric_refs = tuple(object_digest(metric) for metric in metrics)
        metric_hashes.update(metric_refs)
        source_hashes.update(metric.snapshot_hash for metric in metrics if metric.snapshot_hash is not None)
        source_hashes.update(item.source_snapshot_hashes)
        finding_id = uuid5(NAMESPACE_URL, f"signal:{profile_id}:{item.concept.concept_id}:{decision_at.isoformat()}")
        identity_known = (item.concept.identity_status == "confirmed" and item.concept.confirmed_at is not None
                          and item.concept.confirmed_at <= knowledge_cutoff)
        queue: Literal["attention", "deferred", "watch", "market_only", "insufficient_data", "rejected"]
        if item.explicitly_rejected or item.scientific_category == "off_scope":
            queue = "rejected"
        elif item.mature_reviewed or item.scientific_category in _MATURE:
            queue = "market_only"
        elif item.concept.concept_id in attention_set:
            queue = "attention"
        elif item.concept.concept_id in eligible_set:
            queue = "deferred"
        elif (item.arxiv_unassessed or item.search is not None and item.search.state != "unavailable"
              or item.equity is not None and item.equity.proposed_count
              or item.grants is not None and item.grants.proposed_count
              or not identity_known):
            queue = "watch"
        else:
            queue = "insufficient_data"
        rule_id, explanation, next_check, limitations = _explanation(item, queue, identity_known)
        observations = tuple(dict.fromkeys((*item.raw_observation_hashes,
                                            *(digest for metric in metrics for digest in
                                              (metric.used_observation_hashes if isinstance(metric, SearchMetric)
                                               else metric.event_hashes)))))
        origins = tuple(dict.fromkeys(item.origins))
        funding_metric = item.equity or item.grants
        finding = SignalFinding(finding_id=finding_id, concept_id=item.concept.concept_id,
                                origins=origins, scientific_candidate_id=item.scientific_candidate_id,
                                scientific_category=item.scientific_category,
                                search_state=item.search.state if item.search else None,
                                funding_state=funding_metric.state if funding_metric is not None else None,
                                observation_hashes=observations, metric_hashes=metric_refs,
                                queue=queue, rule_id=rule_id, explanation=explanation,
                                next_check=next_check, limitations=limitations)
        findings.append(finding)
        if queue == "watch":
            watches.append(finding_id)
        elif queue == "rejected":
            rejected.append(finding_id)
    by_concept = {item.concept_id: item.finding_id for item in findings}
    return SignalProfile(profile_id=profile_id, policy_version=policy.version, policy_hash=policy_hash,
                         query_profile_hash=query_profile_hash, import_receipt_hashes=import_receipt_hashes,
                         decision_at=decision_at,
                         collection_finished_at=knowledge_cutoff, knowledge_cutoff=knowledge_cutoff,
                         source_snapshot_hashes=tuple(sorted(source_hashes)),
                         concept_artifact_hashes=tuple(sorted(concept_hashes)),
                         association_artifact_hashes=tuple(sorted(association_hashes)),
                         metric_artifact_hashes=tuple(sorted(metric_hashes)), findings=tuple(findings),
                         attention_ids=tuple(by_concept[identifier] for identifier in attention),
                         deferred_ids=tuple(by_concept[identifier] for identifier in full_order[policy.attention_limit:]),
                         watch_ids=tuple(watches), rejected_ids=tuple(rejected),
                         base_result_hash=base_result_hash, base_result_run_id=base_result_run_id)
