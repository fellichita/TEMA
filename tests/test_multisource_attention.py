"""Independent S/Q/F lanes must preserve uncertainty and the old scientific TOP."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.pilot.multisource.attention import SignalCandidate, build_signal_profile
from app.pilot.multisource.contracts import (FundingMetric, SearchMetric, TechnologyConcept, load_policy)
from app.pilot.multisource.store import object_digest


NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
POLICY, POLICY_HASH = load_policy()
HASH = "a" * 64
OTHER = "b" * 64


def _concept(name: str, *, confirmed=True):
    return TechnologyConcept(concept_id=uuid4(), label=name, definition="Реальная технологическая область",
                             identity_status="confirmed" if confirmed else "proposed",
                             confirmed_at=NOW if confirmed else None, provenance_hashes=(HASH,))


def _search(state="sustained_growth", *, positive=3, low=False, count=60, series_id=HASH):
    return SearchMetric(series_id=series_id, snapshot_hash=HASH, policy_hash=POLICY_HASH,
                        decision_at=NOW, knowledge_cutoff=NOW, state=state, recent_count=count, base_count=30,
                        yoy_share_change="1" if state == "sustained_growth" else None,
                        persistence_numerator=3 if state == "sustained_growth" else None,
                        positive_recent_months=positive, low_volume=low,
                        used_observation_hashes=(OTHER,), reason_codes=("fixture",))


def _funding(kind="grant_project", units=2, event_hash=HASH):
    return FundingMetric(source="cordis" if kind == "grant_project" else "investment_csv",
                         event_kind=kind, snapshot_hash=HASH, policy_hash=POLICY_HASH,
                         decision_at=NOW, knowledge_cutoff=NOW, window_days=90,
                         state="multiple_units" if units >= 2 else "single_unit",
                         unit_kind="projects" if kind == "grant_project" else "companies",
                         unit_count=units, event_hashes=(event_hash, OTHER) if units >= 2 else (event_hash,),
                         latest_event_period="2026-08-15", reason_codes=("fixture",))


def _build(*items: SignalCandidate):
    return build_signal_profile(HASH, POLICY, POLICY_HASH, tuple(items), profile_id=uuid4(),
                                decision_at=NOW, knowledge_cutoff=NOW,
                                base_result_hash=HASH if any(item.scientific_category for item in items) else None,
                                base_result_run_id="scientific-run" if any(item.scientific_category for item in items) else None)


def _attention_concepts(profile) -> tuple[UUID, ...]:
    by_id = {item.finding_id: item.concept_id for item in profile.findings}
    return tuple(by_id[item] for item in profile.attention_ids)


def test_round_robin_preserves_science_search_and_funding_order_without_duplication() -> None:
    science = [SignalCandidate(_concept(f"science {index}"), ("scientific",),
                               scientific_candidate_id=f"study-{index}", scientific_category="early_signal",
                               scientific_order=index) for index in range(3)]
    search = [SignalCandidate(_concept(f"search {index}"), ("wordstat",),
                              search=_search(series_id=chr(99 + index) * 64)) for index in range(3)]
    funding = [SignalCandidate(_concept(f"funding {index}"), ("cordis",),
                               grants=_funding(event_hash=str(index) * 64)) for index in range(3)]
    profile = _build(*science, *search, *funding)
    ordered = _attention_concepts(profile)
    assert len(ordered) == 9 and len(set(ordered)) == 9
    assert [ordered[index] for index in (0, 3, 6)] == [item.concept.concept_id for item in science]
    assert set(ordered[1::3]) == {item.concept.concept_id for item in search}
    assert set(ordered[2::3]) == {item.concept.concept_id for item in funding}
    assert profile.base_result_hash == HASH
    assert len(profile.metric_artifact_hashes) == 6


def test_one_concept_with_three_origins_uses_one_attention_slot() -> None:
    concept = _concept("молекулярная память")
    item = SignalCandidate(concept, ("scientific", "wordstat", "cordis"),
                           scientific_candidate_id="study-1", scientific_category="early_signal",
                           scientific_order=0, search=_search(), grants=_funding())
    profile = _build(item)
    assert len(profile.attention_ids) == 1
    assert profile.findings[0].origins == ("scientific", "wordstat", "cordis")
    assert profile.findings[0].scientific_category == "early_signal"
    assert set(profile.findings[0].metric_hashes) == {object_digest(item.search), object_digest(item.grants)}


def test_new_single_month_low_volume_and_unconfirmed_identity_remain_watch() -> None:
    one_month = SignalCandidate(_concept("one month"), ("wordstat",),
                                search=_search("new_in_comparison", positive=1, count=1000))
    low = SignalCandidate(_concept("low count"), ("wordstat",), search=_search("low_volume_change", low=True,
                                                                               count=6))
    ambiguous = SignalCandidate(_concept("RAG", confirmed=False), ("wordstat",), search=_search())
    profile = _build(one_month, low, ambiguous)
    assert profile.attention_ids == () and len(profile.watch_ids) == 3
    rules = {item.rule_id for item in profile.findings}
    assert "new_single_period_observation" in rules and "identity_unconfirmed" in rules


def test_mature_and_off_scope_never_enter_early_attention() -> None:
    mature = SignalCandidate(_concept("mature"), ("scientific",), scientific_candidate_id="mature-1",
                             scientific_category="established_topic")
    rejected = SignalCandidate(_concept("off scope"), ("scientific",), scientific_candidate_id="off-1",
                               scientific_category="off_scope")
    profile = _build(mature, rejected)
    assert profile.attention_ids == () and profile.watch_ids == ()
    assert {item.queue for item in profile.findings} == {"market_only", "rejected"}


def test_single_lane_fills_fifteen_then_deferred_without_watch_padding() -> None:
    items = [SignalCandidate(_concept(f"science {index}"), ("scientific",),
                             scientific_candidate_id=f"study-{index}", scientific_category="early_signal",
                             scientific_order=index) for index in range(17)]
    profile = _build(*items)
    assert len(profile.attention_ids) == 15 and len(profile.deferred_ids) == 2
    assert profile.watch_ids == ()
    assert all(item.queue == "deferred" for item in profile.findings if item.finding_id in profile.deferred_ids)


def test_missing_channels_do_not_receive_zero_scores_or_empty_attention_slots() -> None:
    scientific = SignalCandidate(_concept("science"), ("scientific",),
                                 scientific_candidate_id="study-1", scientific_category="weak_signal_candidate")
    missing = SignalCandidate(_concept("no observed sources"), ("wordstat",))
    profile = _build(scientific, missing)
    assert len(profile.attention_ids) == 1
    assert next(item for item in profile.findings if item.concept_id == missing.concept.concept_id).queue == "insufficient_data"
