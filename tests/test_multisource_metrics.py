"""Counterexamples for normalized demand and project/company funding states."""

import calendar
from datetime import date, datetime, timezone
from uuid import uuid4

from app.pilot.multisource.contracts import (CapitalEvent, SearchObservation, SourceSnapshot,
                                             TechnologyAssociation, load_policy)
from app.pilot.multisource.metrics import compute_funding_metric, compute_search_metric
from app.pilot.multisource.queries import build_manual_profile
from app.pilot.multisource.store import object_digest


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
HASH = "a" * 64
POLICY, POLICY_HASH = load_policy()


def _profile():
    return build_manual_profile("ДНК память", "Хранение данных в молекулярных носителях",
                                seed_terms=("ДНК память",), primary_phrase="ДНК память", confirmed_at=NOW)


def _snapshot(profile=None, *, source="wordstat", coverage="complete", comparable=True):
    profile = profile or _profile()
    return SourceSnapshot(source=source, adapter_version="test/1", request_hash=HASH,
                          query_profile_hash=object_digest(profile), observed_at=NOW, available_at=NOW,
                          coverage=coverage, comparable=comparable, raw_hash=HASH,
                          retention="local_allowed", export_right="local_only")


def _observation(profile, snapshot, year, month, count, share, *, complete=True):
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    is_quantized = count > 0 and share == "0"
    return SearchObservation(series_id=HASH, query_profile_hash=object_digest(profile),
                             snapshot_hash=object_digest(snapshot), phrase="ДНК память", phrase_role="technology",
                             matching_mode="wordstat-csv-monthly-v1", period_start=start,
                             period_end=end if complete else date(year, month, min(19, end.day)),
                             is_complete_period=complete, count=count, value_status="observed",
                             share_raw=share, share_unit="fraction",
                             share_fraction=None if is_quantized else share,
                             normalization_status="quantized_zero" if is_quantized else "usable",
                             observed_at=NOW, available_at=NOW, row_locator=f"line:{year}-{month}")


def _six(profile, snapshot, base_counts, recent_counts, base_shares, recent_shares):
    return tuple(_observation(profile, snapshot, 2025, month, count, share)
                 for month, count, share in zip((6, 7, 8), base_counts, base_shares, strict=True)) + tuple(
        _observation(profile, snapshot, 2026, month, count, share)
        for month, count, share in zip((6, 7, 8), recent_counts, recent_shares, strict=True))


def _search(items, profile=None, snapshot=None, **changes):
    profile = profile or _profile()
    snapshot = snapshot or _snapshot(profile)
    return compute_search_metric(profile, snapshot, items, POLICY, POLICY_HASH,
                                 decision_at=NOW, knowledge_cutoff=NOW, **changes)


def test_t01_normalized_growth_is_sustained_with_two_or_more_positive_months() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    observations = _six(profile, snapshot, (10, 10, 10), (20, 20, 20),
                        ("0.001",) * 3, ("0.002",) * 3)
    metric = _search(observations, profile, snapshot)
    assert metric.state == "sustained_growth" and metric.yoy_share_change == "1"
    assert metric.persistence_numerator == 3 and metric.recent_count == 60
    assert metric.base_count == 30 and metric.low_volume is False


def test_t02_counts_double_but_share_does_not() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    metric = _search(_six(profile, snapshot, (10,) * 3, (20,) * 3,
                          ("0.001",) * 3, ("0.001",) * 3), profile, snapshot)
    assert metric.yoy_count_change == "1" and metric.yoy_share_change == "0"
    assert metric.state == "flat_or_mixed"


def test_t03_t04_zero_base_is_new_in_comparison_without_percent_growth() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    for counts, shares, low in (((0, 0, 1000), ("0", "0", "0.01"), False),
                                ((20, 20, 20), ("0.001",) * 3, False),
                                ((2, 2, 2), ("0.001",) * 3, True)):
        metric = _search(_six(profile, snapshot, (0,) * 3, counts, ("0",) * 3, shares), profile, snapshot)
        assert metric.state == "new_in_comparison" and metric.yoy_share_change is None
        assert metric.low_volume is low


def test_t05_t06_t10_missing_quantized_or_current_partial_never_becomes_zero() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    observations = list(_six(profile, snapshot, (10,) * 3, (20,) * 3,
                             ("0.001",) * 3, ("0.002",) * 3))
    quantized = _observation(profile, snapshot, 2026, 8, 20, "0")
    assert _search(tuple(observations[:-1] + [quantized]), profile, snapshot).state == "insufficient_comparison"
    assert _search(tuple(observations[:-1]), profile, snapshot).state == "insufficient_comparison"
    partial_current = _observation(profile, snapshot, 2026, 9, 100000, "0.9", complete=False)
    assert _search(tuple(observations + [partial_current]), profile, snapshot).state == "sustained_growth"


def test_t07_missing_only_outside_six_does_not_break_yoy() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    observations = _six(profile, snapshot, (10,) * 3, (20,) * 3, ("0.001",) * 3, ("0.002",) * 3)
    assert _search(observations, profile, snapshot).state == "sustained_growth"


def test_t08_observed_zero_and_t09_single_spike() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    zero = _six(profile, snapshot, (0,) * 3, (0,) * 3, ("0",) * 3, ("0",) * 3)
    assert _search(zero, profile, snapshot).state == "observed_zero"
    prior = tuple(_observation(profile, snapshot, 2025 if month >= 8 else 2026, month, 10, "0.001")
                  for month in (8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7))
    baseline = tuple(_observation(profile, snapshot, 2025, month, 10, "0.001") for month in (6, 7))
    august = _observation(profile, snapshot, 2026, 8, 40, "0.004")
    metric = _search(tuple((*baseline, *prior, august)), profile, snapshot)
    assert metric.state == "one_period_spike" and metric.persistence_numerator == 1


def test_repeated_calendar_peak_is_only_possible_seasonality() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    observations = []
    for year, months in ((2023, range(9, 13)), (2024, range(1, 13)),
                         (2025, range(1, 13)), (2026, range(1, 9))):
        observations.extend(_observation(profile, snapshot, year, month, 10,
                                         "0.003" if month == 8 else "0.001") for month in months)
    metric = _search(tuple(observations), profile, snapshot)
    assert metric.state == "possible_seasonality" and metric.yoy_share_change == "0"


def test_t11_alias_series_cannot_be_summed_or_selected_after_results() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    observations = list(_six(profile, snapshot, (10,) * 3, (20,) * 3,
                             ("0.001",) * 3, ("0.002",) * 3))
    observations[-1] = observations[-1].model_copy(update={"series_id": "b" * 64})
    assert _search(tuple(observations), profile, snapshot).state == "needs_review"


def test_equal_time_conflicting_search_observations_still_need_review() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    observations = _six(profile, snapshot, (10,) * 3, (20,) * 3,
                        ("0.001",) * 3, ("0.002",) * 3)
    conflicting = observations[-1].model_copy(update={"count": 30})
    metric = _search((*observations, conflicting), profile, snapshot)
    assert metric.state == "needs_review"
    assert metric.reason_codes == ("identity_or_coverage_conflict",)


def test_search_snapshot_collected_after_historical_cutoff_cannot_leak_backwards() -> None:
    profile = _profile()
    snapshot = _snapshot(profile)
    metric = compute_search_metric(profile, snapshot, (), POLICY, POLICY_HASH,
                                   decision_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                                   knowledge_cutoff=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert metric.state == "unavailable" and metric.reason_codes == ("snapshot_after_cutoff",)


def _grant(project: str, *, amount="1000000", signed=date(2026, 8, 1), beneficiaries=()):
    return CapitalEvent(event_id=uuid4(), source="cordis", source_hash=HASH,
                        kind="grant_project", status="confirmed", project_id=project,
                        beneficiary_ids=beneficiaries, agreement_at=signed, event_date_precision="day",
                        observed_at=NOW, available_at=NOW, amount=amount, currency="EUR",
                        amount_status="disclosed", amount_kind="eu_contribution")


def _link(concept_id, subject, *, kind="project", relation="researches", status="confirmed",
          relation_at_event="supported"):
    return TechnologyAssociation(concept_id=concept_id, subject_id=subject, subject_kind=kind,
                                 relation=relation, status=status, relation_at_event=relation_at_event,
                                 evidence_hashes=(HASH,), reviewer="Analyst" if status == "confirmed" else None,
                                 reviewed_at=NOW if status == "confirmed" else None)


def test_t13_one_grant_ten_participants_is_one_project_unit() -> None:
    profile = _profile()
    concept_id = uuid4()
    event = _grant("project-1", beneficiaries=tuple(f"ORG{i}" for i in range(10)))
    metric = compute_funding_metric("cordis", "grant_project", _snapshot(profile, source="cordis",
                                    coverage="partial", comparable=False), (event,),
                                    (_link(concept_id, "project-1"),), concept_id, POLICY_HASH,
                                    decision_at=NOW, knowledge_cutoff=NOW)
    assert metric.state == "single_unit" and metric.unit_count == 1 and metric.unit_kind == "projects"
    assert len(metric.event_hashes) == 1 and metric.amounts[0].disclosed_total == "1000000"
    assert metric.amounts[0].amount_scope == "linked_events_not_attributed"


def test_t17_company_mention_and_proposed_link_are_not_positive_funding() -> None:
    profile = _profile()
    concept_id = uuid4()
    event = CapitalEvent(event_id=uuid4(), source="investment_csv", source_hash=HASH,
                         kind="equity_round", status="confirmed", recipient_id="firm-1",
                         recipient_name="Example", announced_at=date(2026, 8, 1), event_date_precision="day",
                         observed_at=NOW, available_at=NOW, amount="2000000", currency="USD",
                         amount_status="disclosed", amount_kind="round_amount")
    snapshot = _snapshot(profile, source="investment_csv", coverage="partial", comparable=False)
    proposed = _link(concept_id, "firm-1", kind="organisation", status="proposed")
    assert compute_funding_metric("investment_csv", "equity_round", snapshot, (event,), (proposed,), concept_id,
                                  POLICY_HASH, decision_at=NOW, knowledge_cutoff=NOW).state == "needs_review"
    mentioned = _link(concept_id, "firm-1", kind="organisation", relation="mentioned")
    metric = compute_funding_metric("investment_csv", "equity_round", snapshot, (event,), (mentioned,), concept_id,
                                    POLICY_HASH, decision_at=NOW, knowledge_cutoff=NOW)
    assert metric.state == "insufficient_coverage" and metric.unit_count == 0 and metric.amounts == ()


def test_t26_partial_month_overlap_is_uncertain_not_first_day() -> None:
    profile = _profile()
    concept_id = uuid4()
    event = CapitalEvent(event_id=uuid4(), source="investment_csv", source_hash=HASH,
                         kind="equity_round", status="confirmed", recipient_id="firm-1",
                         recipient_name="Example", event_month="2026-06", event_date_precision="month",
                         observed_at=NOW, available_at=NOW, amount=None, currency="USD",
                         amount_status="undisclosed", amount_kind="round_amount")
    snapshot = _snapshot(profile, source="investment_csv", coverage="partial", comparable=False)
    metric = compute_funding_metric("investment_csv", "equity_round", snapshot, (event,),
                                    (_link(concept_id, "firm-1", kind="organisation", relation="develops"),),
                                    concept_id, POLICY_HASH, decision_at=NOW, knowledge_cutoff=NOW)
    assert metric.state == "insufficient_coverage" and metric.uncertain_date_count == 1


def test_two_confirmed_grants_are_multiple_even_with_different_beneficiaries() -> None:
    profile = _profile()
    concept_id = uuid4()
    events = (_grant("project-1"), _grant("project-2", amount="2000000"))
    links = (_link(concept_id, "project-1"), _link(concept_id, "project-2"))
    metric = compute_funding_metric("cordis", "grant_project", _snapshot(profile, source="cordis",
                                    coverage="partial", comparable=False), events, links, concept_id, POLICY_HASH,
                                    decision_at=NOW, knowledge_cutoff=NOW)
    assert metric.state == "multiple_units" and metric.unit_count == 2
    assert metric.amounts[0].disclosed_total == "3000000" and metric.amounts[0].largest_event_share == "0.666666666667"


def test_t14_amended_grant_revision_is_one_event_and_latest_amount() -> None:
    profile = _profile()
    concept_id = uuid4()
    original = _grant("project-1", amount="1000000")
    latest = original.model_copy(update={"amount": "1500000", "available_at": NOW,
                                      "observed_at": NOW, "source_hash": HASH})
    earlier = original.model_copy(update={"available_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
                                       "observed_at": datetime(2026, 9, 1, tzinfo=timezone.utc)})
    metric = compute_funding_metric("cordis", "grant_project", _snapshot(profile, source="cordis",
                                    coverage="partial", comparable=False), (earlier, latest),
                                    (_link(concept_id, "project-1"),), concept_id, POLICY_HASH,
                                    decision_at=NOW, knowledge_cutoff=NOW)
    assert metric.state == "single_unit" and len(metric.event_hashes) == 1
    assert metric.amounts[0].disclosed_total == "1500000"


def test_funding_links_keep_concept_subject_and_review_cutoffs() -> None:
    concept_id = uuid4()
    other_concept = uuid4()
    first, second = _grant("project-1"), _grant("project-2")
    future = _link(concept_id, "project-2").model_copy(update={
        "reviewed_at": datetime(2026, 10, 1, tzinfo=timezone.utc)})
    links = (_link(other_concept, "project-2"), future,
             _link(concept_id, "project-2", status="proposed"),
             _link(concept_id, "project-1"))
    snapshot = _snapshot(_profile(), source="cordis", coverage="partial", comparable=False)
    def evaluate(associations):
        return compute_funding_metric("cordis", "grant_project", snapshot, (first, second),
                                      associations, concept_id, POLICY_HASH,
                                      decision_at=NOW, knowledge_cutoff=NOW)

    metric = evaluate(links)
    assert metric == evaluate(tuple(reversed(links)))
    assert metric.state == "single_unit" and metric.unit_count == 1
    assert metric.proposed_count == 1
    assert metric.event_hashes == (object_digest(first),)


def test_equal_time_conflicting_grant_revisions_still_need_review() -> None:
    concept_id = uuid4()
    first = _grant("project-1", amount="1000000")
    conflicting = first.model_copy(update={"amount": "2000000"})
    metric = compute_funding_metric("cordis", "grant_project",
        _snapshot(_profile(), source="cordis", coverage="partial", comparable=False),
        (first, conflicting), (_link(concept_id, "project-1"),), concept_id, POLICY_HASH,
        decision_at=NOW, knowledge_cutoff=NOW)
    assert metric.state == "needs_review"
    assert metric.reason_codes == ("conflicting_event_revisions",)
