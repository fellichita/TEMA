"""Offline, immutable what-if views over a verified signal profile."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal
from uuid import UUID

from app.pilot.contracts import content_hash
from app.pilot.multisource.attention import SignalCandidate, build_signal_profile
from app.pilot.multisource.contracts import (ArxivImportReceipt, CapitalEvent, CapitalImportReceipt,
                                             FundingMetric, QueryProfile, SearchMetric, SignalProfile, SourceSnapshot,
                                             TechnologyAssociation, TechnologyConcept,
                                             WordstatImportReceipt, load_policy)
from app.pilot.multisource.store import SignalStore, object_digest
from app.pilot.multisource.metrics import compute_funding_metric
from app.runtime.jobs import TaskFailure

Channel = Literal["exclude_wordstat", "exclude_funding", "exclude_science", "exclude_largest_disclosed_event"]


def largest_disclosed_options(store: SignalStore, baseline: SignalProfile, concept_id: UUID) -> tuple[dict, ...]:
    """Offer only the maximum within each comparable kind/currency pair."""
    finding = next((item for item in baseline.findings if item.concept_id == concept_id), None)
    if finding is None:
        raise TaskFailure("Технология отсутствует в сохранённом профиле.")
    groups: dict[tuple[str, str], list[tuple[str, CapitalEvent]]] = {}
    for metric_hash in finding.metric_hashes:
        try:
            metric = store.get_object(metric_hash, FundingMetric)
        except TaskFailure:
            continue
        for digest in metric.event_hashes:
            event = store.get_object(digest, CapitalEvent)
            if event.kind == metric.event_kind and event.currency and event.amount_status == "disclosed":
                groups.setdefault((event.kind, event.currency), []).append((digest, event))
    options = []
    for (kind, currency), members in sorted(groups.items()):
        maximum = max(Decimal(item.amount or "0") for _, item in members)
        for digest, item in sorted(members, key=lambda pair: pair[0]):
            if Decimal(item.amount or "0") == maximum:
                options.append({"event_hash": digest, "event_id": str(item.event_id),
                                "source_event_id": item.source_event_id, "event_kind": kind,
                                "amount": item.amount, "currency": currency})
    return tuple(options)


def _largest_disclosed_event(store: SignalStore, baseline: SignalProfile, concept_id: UUID | None,
                             event_kind: str | None, currency: str | None,
                             event_hash: str | None) -> tuple[str, CapitalEvent, tuple[dict, ...]]:
    if concept_id is None or event_kind not in {"equity_round", "grant_project"} or (
            currency is None or len(currency) != 3 or not currency.isalpha() or currency != currency.upper()):
        raise TaskFailure("Выберите карточку, вид события и валюту для сценария.")
    options = tuple(item for item in largest_disclosed_options(store, baseline, concept_id)
                    if item["event_kind"] == event_kind and item["currency"] == currency)
    if not options:
        raise TaskFailure("Для выбранного вида и валюты нет раскрытого события в карточке.")
    if event_hash is not None and event_hash not in {item["event_hash"] for item in options}:
        raise TaskFailure("Выбранное событие не является одним из крупнейших в этой группе.")
    chosen = next(item for item in options if event_hash is None or item["event_hash"] == event_hash)
    return chosen["event_hash"], store.get_object(chosen["event_hash"], CapitalEvent), options


def exclude_channel(store: SignalStore, baseline: SignalProfile, channel: Channel,
                    scientific_order: dict[str, int] | None = None, *,
                    concept_id: UUID | None = None, event_kind: str | None = None,
                    currency: str | None = None, event_hash: str | None = None) -> dict:
    """Re-rank frozen evidence; a missing channel is excluded, never interpreted as zero."""
    if channel not in {"exclude_wordstat", "exclude_funding", "exclude_science",
                       "exclude_largest_disclosed_event"}:
        raise TaskFailure("Неизвестный сценарий устойчивости.")
    policy, policy_hash = load_policy()
    if baseline.policy_hash != policy_hash:
        raise TaskFailure("Правила профиля изменились; сценарий недоступен.")
    concepts = {item.concept_id: item for item in (
        store.get_object(digest, TechnologyConcept) for digest in baseline.concept_artifact_hashes)}
    associations = tuple(store.get_object(digest, TechnologyAssociation)
                         for digest in baseline.association_artifact_hashes)
    snapshots = {digest: store.get_object(digest, SourceSnapshot)
                 for digest in baseline.source_snapshot_hashes}
    removed_hash = None
    removed_event = None
    equal_maxima: tuple[dict, ...] = ()
    if channel == "exclude_largest_disclosed_event":
        removed_hash, removed_event, equal_maxima = _largest_disclosed_event(
            store, baseline, concept_id, event_kind, currency, event_hash)
    excluded_source = {"exclude_wordstat": {"wordstat"},
                       "exclude_funding": {"cordis", "investment_csv"},
                       "exclude_science": {"scientific"},
                       "exclude_largest_disclosed_event": set()}[channel]
    snapshot_hashes = tuple(digest for digest, item in snapshots.items() if item.source not in excluded_source)
    receipt_hashes = []
    events_by_source: dict[str, tuple[CapitalEvent, ...]] = {}
    for digest in baseline.import_receipt_hashes:
        try:
            store.get_object(digest, WordstatImportReceipt)
            source = "wordstat"
        except TaskFailure:
            try:
                store.get_object(digest, ArxivImportReceipt)
                source = "arxiv"
            except TaskFailure:
                capital_receipt = store.get_object(digest, CapitalImportReceipt)
                source = capital_receipt.source
                if channel == "exclude_largest_disclosed_event":
                    if source in events_by_source:
                        raise TaskFailure("Финансовый источник повторяется в профиле.") from None
                    events_by_source[source] = tuple(store.get_object(ref, CapitalEvent)
                                                     for ref in capital_receipt.event_hashes)
        if source not in excluded_source:
            receipt_hashes.append(digest)
    candidates = []
    affected_units: dict[str, tuple[int, int]] = {}
    for finding in baseline.findings:
        search = None
        equity = grants = None
        for digest in finding.metric_hashes:
            try:
                metric: SearchMetric | FundingMetric = store.get_object(digest, SearchMetric)
            except TaskFailure:
                metric = store.get_object(digest, FundingMetric)
            if isinstance(metric, SearchMetric):
                search = metric
            elif metric.source == "investment_csv":
                equity = metric
            else:
                grants = metric
        selected_associations = (tuple(item for item in associations if item.concept_id == finding.concept_id)
                                 if channel != "exclude_funding" else ())
        if removed_hash is not None and removed_event is not None:
            affected = equity if removed_event.source == "investment_csv" else grants
            if affected is not None and removed_hash in affected.event_hashes:
                available_events = events_by_source.get(affected.source)
                if available_events is None or affected.snapshot_hash not in snapshots:
                    raise TaskFailure("Нельзя пересчитать финансовую метрику без сохранённого импорта.")
                recalculated = compute_funding_metric(
                    affected.source, affected.event_kind, snapshots[affected.snapshot_hash],
                    tuple(item for item in available_events if item.kind == affected.event_kind
                          and (item.source_event_id or str(item.event_id)) !=
                          (removed_event.source_event_id or str(removed_event.event_id))),
                    selected_associations, finding.concept_id,
                    affected.policy_hash, decision_at=affected.decision_at,
                    knowledge_cutoff=affected.knowledge_cutoff, window_days=affected.window_days)
                affected_units[str(finding.concept_id)] = (affected.unit_count, recalculated.unit_count)
                if affected.source == "investment_csv":
                    equity = recalculated
                else:
                    grants = recalculated
        search_refs = set(search.used_observation_hashes) if search is not None else set()
        funding_refs = {digest for metric in (equity, grants) if metric is not None
                        for digest in metric.event_hashes}
        funding_refs.update(digest for item in associations if item.concept_id == finding.concept_id
                            for digest in item.evidence_hashes)
        omitted_refs = (search_refs if channel == "exclude_wordstat" else
                        funding_refs if channel == "exclude_funding" else
                        {removed_hash} if removed_hash is not None else set())
        if channel == "exclude_wordstat":
            search = None
        elif channel == "exclude_funding":
            equity = grants = None
        raw_refs = tuple(digest for digest in finding.observation_hashes if digest not in omitted_refs)
        origins = tuple(source for source in finding.origins if source not in excluded_source)
        candidates.append(SignalCandidate(
            concept=concepts[finding.concept_id], origins=origins,
            scientific_candidate_id=(finding.scientific_candidate_id if channel != "exclude_science" else None),
            scientific_category=(finding.scientific_category if channel != "exclude_science" else None),
            scientific_order=(scientific_order or {}).get(finding.scientific_candidate_id or ""),
            search=search, equity=equity, grants=grants, associations=selected_associations,
            raw_observation_hashes=raw_refs, arxiv_unassessed="arxiv" in origins,
            explicitly_rejected=finding.queue == "rejected",
            mature_reviewed=finding.queue == "market_only"))
    scenario = build_signal_profile(
        baseline.query_profile_hash, policy, policy_hash, tuple(candidates), profile_id=baseline.profile_id,
        decision_at=baseline.decision_at, knowledge_cutoff=baseline.knowledge_cutoff,
        base_result_hash=baseline.base_result_hash if channel != "exclude_science" else None,
        base_result_run_id=baseline.base_result_run_id if channel != "exclude_science" else None,
        import_receipt_hashes=tuple(receipt_hashes),
        source_snapshot_hashes=snapshot_hashes)
    before = {item.concept_id: item for item in baseline.findings}
    after = {item.concept_id: item for item in scenario.findings}
    result = {"baseline_profile_hash": object_digest(baseline), "scenario_profile_hash": object_digest(scenario),
            "scenario": channel, "excluded_by_scenario": tuple(sorted(excluded_source)),
            "changes": [{"concept_id": str(identifier), "before_queue": before[identifier].queue,
                         "after_queue": after[identifier].queue,
                         "before_rule": before[identifier].rule_id, "after_rule": after[identifier].rule_id,
                         "before_funding": before[identifier].funding_state,
                         "after_funding": after[identifier].funding_state,
                         "funding_units": affected_units.get(str(identifier))}
                        for identifier in sorted(before, key=str)],
            "attention_before": len(baseline.attention_ids), "attention_after": len(scenario.attention_ids)}
    if removed_hash is not None and removed_event is not None:
        result["excluded_by_scenario"] = ("event:" + removed_hash,)
        result["excluded_event"] = {"event_hash": removed_hash, "event_id": str(removed_event.event_id),
                                    "event_kind": removed_event.kind, "currency": removed_event.currency,
                                    "amount": removed_event.amount}
        result["equal_maxima"] = equal_maxima
    return result


def _query_semantics(query: QueryProfile) -> str:
    return content_hash({"scope": query.original_query, "definition": query.definition,
                         "exclusions": query.exclusions, "primary_phrase": query.primary_phrase,
                         "terms": tuple((term.text, term.language, term.role, term.status) for term in query.terms),
                         "regions": query.region_ids, "devices": query.devices})


def _events(store: SignalStore, profile: SignalProfile) -> dict[tuple[str, str], CapitalEvent]:
    indexed: dict[tuple[str, str], CapitalEvent] = {}
    for digest in profile.import_receipt_hashes:
        try:
            receipt = store.get_object(digest, CapitalImportReceipt)
        except TaskFailure:
            continue
        for event_hash in receipt.event_hashes:
            item = store.get_object(event_hash, CapitalEvent)
            identifier = item.source_event_id or str(item.event_id)
            key = item.source, identifier
            if key in indexed:
                raise TaskFailure("Событие повторяется между финансовыми импортами профиля.")
            indexed[key] = item
    return indexed


def _event_period(item: CapitalEvent) -> str | None:
    exact = item.agreement_at or item.closed_at or item.announced_at
    return (exact.isoformat() if exact else item.event_month if item.event_month is not None else
            str(item.event_year) if item.event_year is not None else None)


def _associations(store: SignalStore, profile: SignalProfile) -> dict[tuple[str, str, str], TechnologyAssociation]:
    indexed: dict[tuple[str, str, str], TechnologyAssociation] = {}
    for digest in profile.association_artifact_hashes:
        item = store.get_object(digest, TechnologyAssociation)
        key = str(item.concept_id), item.subject_kind, item.subject_id
        if key in indexed:
            raise TaskFailure("Связь с одним событием повторяется в профиле.")
        indexed[key] = item
    return indexed


def compare_profiles(store: SignalStore, before: SignalProfile, after: SignalProfile) -> dict:
    """Compare saved cuts without converting a source change into a priority percentage."""
    if after.decision_at <= before.decision_at:
        raise TaskFailure("Второй профиль должен быть создан позже первого.")
    old_query = store.get_object(before.query_profile_hash, QueryProfile)
    new_query = store.get_object(after.query_profile_hash, QueryProfile)
    reasons = []
    if before.policy_hash != after.policy_hash:
        reasons.append("different_policy")
    if _query_semantics(old_query) != _query_semantics(new_query):
        reasons.append("different_query_semantics")
    old_snapshots = {store.get_object(digest, SourceSnapshot).source for digest in before.source_snapshot_hashes}
    new_snapshots = {store.get_object(digest, SourceSnapshot).source for digest in after.source_snapshot_hashes}
    old_findings = {item.concept_id: item for item in before.findings}
    new_findings = {item.concept_id: item for item in after.findings}
    shared = set(old_findings) & set(new_findings)
    if not shared:
        reasons.append("no_shared_confirmed_concept_id")
    changes = []
    if not reasons:
        for identifier in sorted(shared, key=str):
            left, right = old_findings[identifier], new_findings[identifier]
            if (left.queue, left.search_state, left.funding_state, left.scientific_category) != (
                    right.queue, right.search_state, right.funding_state, right.scientific_category):
                changes.append({"concept_id": str(identifier), "before_queue": left.queue,
                                "after_queue": right.queue, "before_search": left.search_state,
                                "after_search": right.search_state, "before_funding": left.funding_state,
                                "after_funding": right.funding_state,
                                "before_science": left.scientific_category,
                                "after_science": right.scientific_category})
    previous_events, current_events = _events(store, before), _events(store, after)
    event_changes = []
    for key in sorted(current_events):
        item = current_events[key]
        previous = previous_events.get(key)
        if previous is None:
            period = _event_period(item)
            older = period is not None and period[:10] <= before.decision_at.date().isoformat()[:len(period)]
            category = "older_event_first_imported" if older else "new_event_or_date_unknown"
        elif (previous.status, previous.amount, previous.currency, previous.amount_kind,
              _event_period(previous)) != (item.status, item.amount, item.currency, item.amount_kind,
                                           _event_period(item)):
            category = "corrected_event"
        else:
            continue
        event_changes.append({"source": key[0], "source_event_id": key[1], "kind": category,
                              "event_period": _event_period(item)})
    previous_links, current_links = _associations(store, before), _associations(store, after)
    association_changes = []
    for relation_key in sorted(set(previous_links) | set(current_links)):
        old_link, new_link = previous_links.get(relation_key), current_links.get(relation_key)
        if old_link == new_link:
            continue
        if new_link is None:
            category = "removed_relation"
        elif old_link is None:
            category = "new_confirmed_relation" if new_link.status == "confirmed" else "new_unconfirmed_relation"
        elif new_link.status == "confirmed" and old_link.status != "confirmed":
            category = "new_confirmed_relation"
        elif old_link.status == "confirmed" and new_link.status != "confirmed":
            category = "revoked_relation"
        else:
            category = "revised_relation"
        association_changes.append({"concept_id": relation_key[0], "subject_kind": relation_key[1],
                                    "subject_id": relation_key[2], "kind": category,
                                    "before_status": old_link.status if old_link else None,
                                    "after_status": new_link.status if new_link else None})
    core = {"before_profile_hash": object_digest(before), "after_profile_hash": object_digest(after),
            "comparable": not reasons, "reasons": reasons,
            "new_sources": sorted(new_snapshots - old_snapshots),
            "removed_sources": sorted(old_snapshots - new_snapshots),
            "new_concept_ids": sorted(str(item) for item in set(new_findings) - set(old_findings)),
            "removed_concept_ids": sorted(str(item) for item in set(old_findings) - set(new_findings)),
            "changed_findings": changes, "event_change_count": len(event_changes),
            "event_changes": event_changes[:100], "event_changes_truncated": len(event_changes) > 100,
            "association_change_count": len(association_changes),
            "association_changes": association_changes[:100],
            "association_changes_truncated": len(association_changes) > 100}
    return core | {"comparison_hash": content_hash(core)}
