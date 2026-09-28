"""Pure, replayable search and funding states from frozen source observations."""

from __future__ import annotations

import calendar
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from statistics import median
from typing import Literal
from uuid import UUID

from app.pilot.multisource.contracts import (CapitalEvent, FundingAmount, FundingMetric, QueryProfile,
                                             SearchMetric, SearchObservation, SearchState, SignalPolicy,
                                             SourceSnapshot, SourceStatus, TechnologyAssociation)
from app.pilot.multisource.store import object_digest
from app.runtime.jobs import TaskFailure

_QUANTUM = Decimal("0.000000000001")
_SOURCE_UNAVAILABLE = {"not_configured", "not_requested", "stale", "auth_error", "rate_limited",
                       "unavailable", "invalid_data"}


@dataclass(frozen=True)
class _SearchValues:
    recent_count: int | None = None
    base_count: int | None = None
    recent_share: Decimal | None = None
    base_share: Decimal | None = None
    yoy_share: Decimal | None = None
    yoy_count: Decimal | None = None
    persistence: int | None = None
    positive_recent_months: int | None = None
    used: tuple[SearchObservation, ...] = ()


def _formatted(value: Decimal | None) -> str | None:
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 80
        return format(value.quantize(_QUANTUM, rounding=ROUND_HALF_EVEN).normalize(), "f")


def _month_shift(month: date, change: int) -> date:
    index = month.year * 12 + month.month - 1 + change
    return date(index // 12, index % 12 + 1, 1)


def _usable(item: SearchObservation | None) -> bool:
    return bool(item is not None and item.is_complete_period and item.value_status == "observed"
                and item.count is not None and item.normalization_status == "usable"
                and item.share_fraction is not None)


def _selected_observations(items: tuple[SearchObservation, ...], cutoff: datetime) -> tuple[dict[date, SearchObservation], bool]:
    choices: dict[date, list[SearchObservation]] = defaultdict(list)
    for item in items:
        if item.available_at <= cutoff and item.observed_at <= cutoff:
            choices[item.period_start].append(item)
    selected = {}
    conflict = False
    for month, versions in choices.items():
        latest_time = max(item.available_at for item in versions)
        latest = [item for item in versions if item.available_at == latest_time]
        if len(latest) == 1:
            selected[month] = latest[0]
            continue
        signatures = {(item.count, item.value_status, item.share_raw, item.share_unit,
                       item.share_fraction, item.normalization_status, item.is_complete_period,
                       item.period_end) for item in latest}
        if len(signatures) != 1:
            conflict = True
        selected[month] = min(latest, key=object_digest)
    return selected, conflict


def _spike(selected: dict[date, SearchObservation], recent: tuple[date, date, date],
           baseline: tuple[date, date, date], multiplier: Decimal) -> bool:
    shares = [Decimal(selected[month].share_fraction or "0") for month in recent]
    peak_share = max(shares)
    if shares.count(peak_share) != 1:
        return False
    peak = recent[shares.index(peak_share)]
    prior = tuple(_month_shift(peak, -offset) for offset in range(12, 0, -1))
    if not all(_usable(selected.get(month)) for month in prior):
        return False
    centre = median(Decimal(selected[month].share_fraction or "0") for month in prior)
    gains = sum(Decimal(selected[month].share_fraction or "0") >
                Decimal(selected[before].share_fraction or "0") for month, before in zip(recent, baseline, strict=True))
    return centre > 0 and peak_share >= multiplier * centre and gains == 1


def _seasonal(selected: dict[date, SearchObservation], last_month: date,
              yoy: Decimal | None, lower: Decimal, upper: Decimal) -> bool:
    if yoy is None or not lower <= yoy <= upper:
        return False
    span = tuple(_month_shift(last_month, -offset) for offset in range(35, -1, -1))
    if not all(_usable(selected.get(month)) for month in span):
        return False
    def top_three(year: int) -> set[int]:
        months = [month for month in span if month.year == year]
        ranked = sorted(months, key=lambda month: (-Decimal(selected[month].share_fraction or "0"), month.month))
        return {month.month for month in ranked[:3]}
    return all(last_month.month in top_three(year) for year in (last_month.year,
                                                               last_month.year - 1, last_month.year - 2))


def compute_search_metric(profile: QueryProfile, snapshot: SourceSnapshot | None,
                          observations: tuple[SearchObservation, ...], policy: SignalPolicy, policy_hash: str,
                          *, decision_at: datetime, knowledge_cutoff: datetime,
                          identity_confirmed: bool = True, source_status: SourceStatus | None = None) -> SearchMetric:
    """Use one preselected phrase; never maximise growth over its aliases."""
    if decision_at.tzinfo is None or knowledge_cutoff.tzinfo is None or decision_at > knowledge_cutoff:
        raise TaskFailure("Некорректный временной срез поискового анализа.")
    snapshot_hash = object_digest(snapshot) if snapshot is not None else None
    profile_hash = object_digest(profile)
    source_unavailable = (snapshot is None or source_status is not None and source_status.state in _SOURCE_UNAVAILABLE)
    if not profile.primary_phrase:
        raise TaskFailure("Основная поисковая фраза не подтверждена.")
    series_id = observations[0].series_id if observations else None
    def result(state: SearchState, reason: str, values: _SearchValues | None = None) -> SearchMetric:
        values = values or _SearchValues()
        return SearchMetric(series_id=series_id, snapshot_hash=snapshot_hash, policy_hash=policy_hash,
                            decision_at=decision_at, knowledge_cutoff=knowledge_cutoff, state=state,
                            recent_count=values.recent_count, base_count=values.base_count,
                            recent_share=_formatted(values.recent_share), base_share=_formatted(values.base_share),
                            yoy_share_change=_formatted(values.yoy_share), yoy_count_change=_formatted(values.yoy_count),
                            persistence_numerator=values.persistence,
                            positive_recent_months=values.positive_recent_months,
                            low_volume=values.recent_count is not None and values.recent_count < policy.min_recent_count,
                            used_observation_hashes=tuple(object_digest(item) for item in values.used),
                            reason_codes=(reason,))
    if source_unavailable or snapshot is not None and snapshot.observed_at > knowledge_cutoff:
        reason = ("snapshot_after_cutoff" if snapshot is not None and snapshot.observed_at > knowledge_cutoff
                  else source_status.reason_code if source_status else "source_not_configured")
        return result("unavailable", reason)
    assert snapshot is not None
    if snapshot.source != "wordstat" or snapshot.query_profile_hash != profile_hash or any(
            item.snapshot_hash != snapshot_hash or item.query_profile_hash != profile_hash or
            item.phrase.casefold() != profile.primary_phrase.casefold() for item in observations):
        return result("needs_review", "incompatible_search_provenance")
    if profile.confirmed_at is None or profile.confirmed_at > knowledge_cutoff:
        return result("needs_review", "query_profile_confirmed_after_cutoff")
    if len({(item.series_id, item.region_ids, item.devices, item.matching_mode) for item in observations}) > 1:
        return result("needs_review", "incompatible_search_series")
    selected, conflict = _selected_observations(observations, knowledge_cutoff)
    last = _month_shift(decision_at.date().replace(day=1), -1)
    recent = (_month_shift(last, -2), _month_shift(last, -1), last)
    baseline = (_month_shift(recent[0], -12), _month_shift(recent[1], -12), _month_shift(recent[2], -12))
    six = (*baseline, *recent)
    earliest = _month_shift(last, -35)
    used = tuple(selected[month] for month in sorted(selected) if earliest <= month <= last)
    recent_counts = [selected[month].count for month in recent if month in selected]
    base_counts = [selected[month].count for month in baseline if month in selected]
    recent_count = (sum(value for value in recent_counts if value is not None)
                    if len(recent_counts) == 3 and all(value is not None for value in recent_counts) else None)
    base_count = (sum(value for value in base_counts if value is not None)
                  if len(base_counts) == 3 and all(value is not None for value in base_counts) else None)
    count_yoy = (Decimal(recent_count) / Decimal(base_count) - 1
                 if recent_count is not None and base_count else None)
    positive_recent_months = (sum(value > 0 for value in recent_counts if value is not None)
                              if len(recent_counts) == 3 and all(value is not None for value in recent_counts) else None)
    common = _SearchValues(recent_count=recent_count, base_count=base_count, yoy_count=count_yoy,
                           positive_recent_months=positive_recent_months, used=used)
    if not identity_confirmed or conflict or not snapshot.comparable:
        return result("needs_review", "identity_or_coverage_conflict", common)
    if not all(_usable(selected.get(month)) for month in six):
        return result("insufficient_comparison", "missing_normalized_months", common)
    recent_share = sum((Decimal(selected[month].share_fraction or "0") for month in recent), Decimal(0)) / 3
    base_share = sum((Decimal(selected[month].share_fraction or "0") for month in baseline), Decimal(0)) / 3
    persistence = sum(Decimal(selected[month].share_fraction or "0") >
                      Decimal(selected[before].share_fraction or "0") and (selected[month].count or 0) > 0
                      for month, before in zip(recent, baseline, strict=True))
    share_yoy = recent_share / base_share - 1 if base_share > 0 else None
    common = replace(common, recent_share=recent_share, base_share=base_share, yoy_share=share_yoy,
                     persistence=persistence)
    if recent_count == base_count == 0:
        return result("observed_zero", "observed_zero_for_phrase", common)
    if base_share == 0 and base_count == 0 and recent_count is not None and recent_count > 0:
        return result("new_in_comparison", "new_vs_selected_baseline", common)
    if recent_count is not None and recent_count < policy.min_recent_count and recent_share != base_share:
        return result("low_volume_change", "low_recent_volume", common)
    if (share_yoy is not None and share_yoy >= Decimal(policy.growth_threshold)
            and persistence >= policy.min_persistence_numerator and recent_count is not None
            and recent_count >= policy.min_recent_count):
        return result("sustained_growth", "sustained_normalized_growth", common)
    if (share_yoy is not None and share_yoy <= Decimal(policy.seasonal_lower) and
            sum(Decimal(selected[month].share_fraction or "0") <
                Decimal(selected[before].share_fraction or "0") for month, before in
                zip(recent, baseline, strict=True)) >= 2):
        return result("declining", "sustained_normalized_decline", common)
    if _spike(selected, recent, baseline, Decimal(policy.spike_multiplier)):
        return result("one_period_spike", "single_peak_with_twelve_month_baseline", common)
    if _seasonal(selected, last, share_yoy, Decimal(policy.seasonal_lower), Decimal(policy.seasonal_upper)):
        return result("possible_seasonality", "repeated_calendar_peak", common)
    return result("flat_or_mixed", "no_positive_search_rule", common)


def _event_window(event: CapitalEvent, *, event_kind: Literal["equity_round", "grant_project"],
                  lower: date, upper: date) -> Literal["inside", "outside", "uncertain"]:
    day = (event.agreement_at if event_kind == "grant_project" else event.announced_at or event.closed_at)
    if day is not None:
        return "inside" if lower <= day < upper else "outside"
    if event.event_month is not None:
        year, month = (int(part) for part in event.event_month.split("-"))
        first = date(year, month, 1)
        last = date(year, month, calendar.monthrange(year, month)[1])
    elif event.event_year is not None:
        first, last = date(event.event_year, 1, 1), date(event.event_year, 12, 31)
    else:
        return "uncertain"
    if lower <= first and last < upper:
        return "inside"
    if last < lower or first >= upper:
        return "outside"
    return "uncertain"


def _confirmed_relation(event: CapitalEvent, links: Iterable[TechnologyAssociation],
                        cutoff: datetime) -> bool:
    subject = event.project_id if event.kind == "grant_project" else event.recipient_id
    kind = "project" if event.kind == "grant_project" else "organisation"
    for link in links:
        if (link.subject_id != subject or link.subject_kind != kind or link.status != "confirmed"
                or link.reviewed_at is None or link.reviewed_at > cutoff):
            continue
        if link.relation in {"develops", "researches"} and link.relation_at_event == "supported":
            return True
        if link.relation == "finances_specific_project" and event.kind == "grant_project":
            return True
    return False


def _proposed_relation(event: CapitalEvent, links: Iterable[TechnologyAssociation], cutoff: datetime) -> bool:
    subject = event.project_id if event.kind == "grant_project" else event.recipient_id
    kind = "project" if event.kind == "grant_project" else "organisation"
    return any(link.subject_id == subject and link.subject_kind == kind and link.status in {"proposed", "uncertain"}
               and (link.reviewed_at is None or link.reviewed_at <= cutoff) for link in links)


def _latest_events(events: tuple[CapitalEvent, ...], cutoff: datetime) -> tuple[tuple[CapitalEvent, ...], bool]:
    grouped: dict[UUID, list[CapitalEvent]] = defaultdict(list)
    for event in events:
        if event.available_at <= cutoff and event.observed_at <= cutoff:
            grouped[event.event_id].append(event)
    result = []
    conflict = False
    for versions in grouped.values():
        latest_time = max(item.available_at for item in versions)
        latest = [item for item in versions if item.available_at == latest_time]
        if len(latest) == 1:
            result.append(latest[0])
            continue
        signatures = {(item.source_event_id, item.kind, item.status, item.amount, item.currency,
                       item.announced_at, item.agreement_at, item.closed_at, item.event_month,
                       item.event_year) for item in latest}
        if len(signatures) != 1:
            conflict = True
        result.append(min(latest, key=object_digest))
    return tuple(sorted(result, key=lambda item: str(item.event_id))), conflict


def compute_funding_metric(source: Literal["cordis", "investment_csv"],
                           event_kind: Literal["equity_round", "grant_project"],
                           snapshot: SourceSnapshot | None, events: tuple[CapitalEvent, ...],
                           associations: tuple[TechnologyAssociation, ...], concept_id: UUID,
                           policy_hash: str, *, decision_at: datetime, knowledge_cutoff: datetime,
                           window_days: Literal[90, 365] = 90,
                           source_status: SourceStatus | None = None) -> FundingMetric:
    """Count proven projects or companies; disclose money only by kind and currency."""
    if decision_at.tzinfo is None or knowledge_cutoff.tzinfo is None or decision_at > knowledge_cutoff:
        raise TaskFailure("Некорректный временной срез финансового анализа.")
    if source == "cordis" and event_kind != "grant_project" or source == "investment_csv" and event_kind != "equity_round":
        raise TaskFailure("Вид события не соответствует финансовому источнику.")
    snapshot_hash = object_digest(snapshot) if snapshot is not None else None
    unit_kind: Literal["companies", "projects"] = "projects" if event_kind == "grant_project" else "companies"
    def result(state, reason, *, selected: tuple[CapitalEvent, ...] = (), units: int = 0,
               proposed: int = 0, uncertain: int = 0) -> FundingMetric:
        amount_groups: dict[str, list[Decimal]] = defaultdict(list)
        undisclosed = 0
        for event in selected:
            if event.amount_status == "disclosed" and event.amount is not None and event.currency is not None:
                amount_groups[event.currency].append(Decimal(event.amount))
            else:
                undisclosed += 1
        amounts = []
        for currency, values in sorted(amount_groups.items()):
            total = sum(values, Decimal(0))
            if total > 0:
                largest = max(values)
                amounts.append(FundingAmount(event_kind=event_kind, currency=currency,
                                             disclosed_total=format(total, "f"),
                                             largest_event_amount=format(largest, "f"),
                                             largest_event_share=_formatted(largest / total) or "0"))
        periods = [date_value.isoformat() for event in selected for date_value in
                   (event.agreement_at if event_kind == "grant_project" else event.announced_at or event.closed_at,)
                   if date_value is not None]
        periods.extend(event.event_month for event in selected if event.event_month is not None)
        return FundingMetric(source=source, event_kind=event_kind, snapshot_hash=snapshot_hash,
                             policy_hash=policy_hash, decision_at=decision_at, knowledge_cutoff=knowledge_cutoff,
                             window_days=window_days, state=state, unit_kind=unit_kind, unit_count=units,
                             event_hashes=tuple(object_digest(item) for item in selected), proposed_count=proposed,
                             uncertain_date_count=uncertain, undisclosed_count=undisclosed,
                             latest_event_period=max(periods) if periods else None,
                             amounts=tuple(amounts), reason_codes=(reason,))
    if (snapshot is None or snapshot.observed_at > knowledge_cutoff or
            source_status is not None and source_status.state in _SOURCE_UNAVAILABLE):
        reason = ("snapshot_after_cutoff" if snapshot is not None and snapshot.observed_at > knowledge_cutoff
                  else source_status.reason_code if source_status else "source_not_configured")
        return result("unavailable", reason)
    if snapshot.source != source:
        return result("needs_review", "incompatible_funding_snapshot")
    if any(event.source != source or event.kind != event_kind or event.source_hash != snapshot.raw_hash
           for event in events):
        return result("needs_review", "incompatible_funding_events")
    selected_events, conflicting = _latest_events(events, knowledge_cutoff)
    if conflicting:
        return result("needs_review", "conflicting_event_revisions")
    links_by_subject: dict[tuple[str, str | None], list[TechnologyAssociation]] = defaultdict(list)
    for link in associations:
        if link.concept_id == concept_id:
            links_by_subject[link.subject_kind, link.subject_id].append(link)
    lower = decision_at.date() - timedelta(days=window_days)
    upper = decision_at.date()
    selected = []
    units = set()
    proposed = uncertain = 0
    for event in selected_events:
        if event_kind == "grant_project" and (event.status != "confirmed" or event.agreement_at is None):
            continue
        if event_kind == "equity_round" and event.status not in {"announced", "confirmed"}:
            continue
        membership = _event_window(event, event_kind=event_kind, lower=lower, upper=upper)
        if membership == "outside":
            continue
        subject_kind = "project" if event_kind == "grant_project" else "organisation"
        subject_id = event.project_id if event_kind == "grant_project" else event.recipient_id
        links = links_by_subject.get((subject_kind, subject_id), ())
        confirmed = _confirmed_relation(event, links, knowledge_cutoff)
        if membership == "uncertain":
            if confirmed:
                uncertain += 1
            continue
        if confirmed:
            selected.append(event)
            units.add(event.project_id if event_kind == "grant_project" else event.recipient_id)
        elif _proposed_relation(event, links, knowledge_cutoff):
            proposed += 1
    chosen = tuple(selected)
    if len(units) >= 2:
        return result("multiple_units", "multiple_confirmed_units", selected=chosen, units=len(units),
                      proposed=proposed, uncertain=uncertain)
    if len(units) == 1:
        return result("single_unit", "single_confirmed_unit", selected=chosen, units=1,
                      proposed=proposed, uncertain=uncertain)
    if proposed:
        return result("needs_review", "only_proposed_relations", proposed=proposed, uncertain=uncertain)
    if snapshot.coverage == "complete" and not uncertain:
        return result("none_in_observed_scope", "no_confirmed_events_in_complete_scope")
    return result("insufficient_coverage", "sample_or_dates_incomplete", uncertain=uncertain)
