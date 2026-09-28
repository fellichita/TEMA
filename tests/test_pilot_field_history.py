"""Frozen source-count provenance and field-growth counterexamples, without network."""

from datetime import UTC, date, datetime
from threading import Event, Lock

import pytest

from app.backend.contracts import SourcePage
from app.backend.errors import BackendError, CancelledError
from app.pilot.field_history import FieldCountPage, FieldCountRequest, FieldExposure, FieldYearExposure, collect_field_exposure, normalize_growth
from app.runtime.credentials import CredentialStore
from app.runtime.jobs import TaskCancelled
from tests.test_pilot_evidence import Context, document, query_plan

NOW = datetime(2026, 9, 14, tzinfo=UTC)


def field_exposure(values=(1000,) * 6, *, plan=None):
    plan = plan or query_plan()
    observations = tuple(FieldYearExposure(year=year,
        request=FieldCountRequest(query=plan.english_query, from_date=date(year, 1, 1), until_date=date(year, 12, 31)),
        observed_at=NOW, status="observed" if count is not None else "unavailable", count=count,
        page=FieldCountPage(total_available=count, scanned=0 if not count else 1, skipped=0,
                           exhausted=not count, source_ids=(f"W{year}",) if count else ()) if count is not None else None,
        reason="source_total_missing" if count is None else None)
        for year, count in zip(plan.completed_years, values, strict=True))
    return FieldExposure.create(plan_hash=plan.plan_hash, fixed_query=plan.english_query, observed_at=NOW, years=observations)


def relative(counts, field_counts=(1000,) * 6):
    plan = query_plan()
    return normalize_growth(counts, field_exposure(field_counts), plan_hash=plan.plan_hash, years=plan.completed_years)


class Provider:
    def __init__(self, pages, calls, *, context=None):
        self.pages = pages
        self.calls = calls
        self.closed = False
        self.context = context

    def iter_pages(self, request, cancel):
        self.calls.append(request)
        assert request.max_results == 1
        if isinstance(self.pages, Exception):
            raise self.pages
        for page in self.pages:
            if self.context:
                self.context.cancel_event.set()
            yield page
        raise AssertionError("The collector requested a second page")

    def close(self):
        self.closed = True


def test_source_reported_total_is_used_without_consuming_capped_discovery_or_second_page():
    calls, providers = [], []

    def factory(source):
        assert source == "openalex"
        provider = Provider([SourcePage(documents=(document(len(calls)),), scanned=1,
                                       total_available=50000, exhausted=False)], calls)
        providers.append(provider)
        return provider

    context = Context()
    result = collect_field_exposure(query_plan(), context, CredentialStore(), provider_factory=factory)
    assert len(calls) == 6 and all(provider.closed for provider in providers)
    # Every year moves a visible counter, so a long field check is never silent.
    assert [(completed, total) for _, _, completed, total in context.progress_events] == [
        (position // 2, 6) if position % 2 == 0 else (position // 2 + 1, 6) for position in range(12)]
    assert all(stage == "history" for stage, *_ in context.progress_events)
    assert [item.count for item in result.years] == [50000] * 6
    assert all(item.status == "observed" and not item.page.exhausted for item in result.years)
    assert {request.topic for request in calls} == {query_plan().english_query}
    assert sorted((request.from_date.year, request.until_date.year) for request in calls) == [
        (year, year) for year in range(2020, 2026)]
    assert FieldExposure.model_validate_json(result.model_dump_json()) == result


def test_ten_completed_years_make_at_most_ten_first_page_calls():
    plan = type(query_plan()).model_validate(query_plan().model_dump(mode="python") | {"completed_years": tuple(range(2016, 2026))})
    calls = []
    result = collect_field_exposure(plan, Context(), CredentialStore(),
        provider_factory=lambda _: Provider([SourcePage(scanned=0, total_available=0, exhausted=True)], calls))
    assert len(calls) == len(result.years) == 10
    assert all(item.count == 0 for item in result.years)
    assert normalize_growth((0,) * 10, result, plan_hash=plan.plan_hash, years=plan.completed_years).reason == "zero_field_exposure"


@pytest.mark.parametrize("page,reason", [
    (SourcePage(scanned=0, exhausted=True), "source_total_missing"),
    (SourcePage(scanned=0, total_available=30, exhausted=True), "empty_page_with_positive_source_total"),
    (SourcePage(scanned=1, skipped=1, total_available=0, exhausted=True), "invalid_source_count_response"),
    (BackendError("network_error", "unavailable"), "source_network_error"),
])
def test_missing_inconsistent_and_failed_totals_are_unknown_never_zero(page, reason):
    calls = []
    result = collect_field_exposure(query_plan(), Context(), CredentialStore(),
        provider_factory=lambda _: Provider(page if isinstance(page, Exception) else [page], calls))
    assert all(item.count is None and item.status == "unavailable" and item.reason == reason for item in result.years)
    assert relative((1, 1, 1, 2, 4, 8), (1000, None, 1000, 1000, 1000, 1000)).reason == "field_exposure_missing"


@pytest.mark.parametrize("backend_cancel", [False, True])
def test_cancellation_propagates_and_closes_provider_without_returning_partial_success(backend_cancel):
    calls, context = [], Context()
    provider = Provider(CancelledError() if backend_cancel else [SourcePage(scanned=0, total_available=0, exhausted=True)],
                        calls, context=None if backend_cancel else context)
    with pytest.raises(TaskCancelled):
        collect_field_exposure(query_plan(), context, CredentialStore(), provider_factory=lambda _: provider)
    assert 1 <= len(calls) <= 3 and provider.closed


def test_year_counts_overlap_in_bounded_batches_and_return_in_calendar_order():
    entered: set[int] = set()
    lock = Lock()
    first_batch_ready = Event()

    class ConcurrentProvider:
        def __init__(self):
            self.closed = False

        def iter_pages(self, request, cancel):
            year = request.from_date.year
            with lock:
                entered.add(year)
                if len(entered) == 3:
                    first_batch_ready.set()
            assert first_batch_ready.wait(2), "year requests did not overlap"
            yield SourcePage(documents=(document(year),), scanned=1,
                             total_available=year, exhausted=False)

        def close(self):
            self.closed = True

    result = collect_field_exposure(query_plan(), Context(), CredentialStore(),
                                    provider_factory=lambda _: ConcurrentProvider())
    assert tuple(item.year for item in result.years) == query_plan().completed_years
    assert tuple(item.count for item in result.years) == query_plan().completed_years


def test_exposure_hash_catches_changed_counts_query_and_request_dates():
    original = field_exposure()
    changed = original.model_dump(mode="json")
    changed["years"][0]["count"] = changed["years"][0]["page"]["total_available"] = 2000
    with pytest.raises(ValueError, match="provenance"):
        FieldExposure.model_validate(changed)
    changed = original.model_dump(mode="json")
    changed["years"][0]["request"]["query"] = "another field"
    with pytest.raises(ValueError, match="identical"):
        FieldExposure.model_validate(changed)
    changed = original.model_dump(mode="json")
    changed["years"][0]["request"]["until_date"] = "2020-06-30"
    with pytest.raises(ValueError, match="exactly one"):
        FieldExposure.model_validate(changed)


def test_exposure_rejects_other_plan_and_unfinished_or_misaligned_years():
    plan, exposure = query_plan(), field_exposure()
    with pytest.raises(ValueError, match="match"):
        normalize_growth((1, 1, 1, 2, 4, 8), exposure, plan_hash="f" * 64, years=plan.completed_years)
    with pytest.raises(ValueError, match="aligned"):
        normalize_growth((1, 1, 2), exposure, plan_hash=plan.plan_hash, years=plan.completed_years)
    changed = exposure.model_dump(mode="json")
    changed["years"][-1]["observed_at"] = "2025-09-14T00:00:00Z"
    with pytest.raises(ValueError, match="unfinished"):
        FieldExposure.model_validate(changed)


def test_identical_tenfold_field_and_candidate_growth_is_not_excess_growth():
    result = relative((1, 1, 1, 10, 10, 10), (100, 100, 100, 1000, 1000, 1000))
    assert result.status == "available" and result.raw_ratio == 1
    assert result.ratio_lower_95 < 1 < result.ratio_upper_95
    assert not result.observed_growth and not result.excess_growth_supported


def test_real_rate_growth_survives_source_scale_and_count_scale_without_volume_bonus_to_raw_ratio():
    values = (10, 10, 10, 30, 60, 90)
    original = relative(values)
    scaled = relative(tuple(value * 10 for value in values), (10000,) * 6)
    source_scaled = relative(values, (10000,) * 6)
    assert original.raw_ratio == scaled.raw_ratio == source_scaled.raw_ratio == 6
    assert original.excess_growth_supported and scaled.excess_growth_supported and source_scaled.excess_growth_supported
    assert original.recent_rates == scaled.recent_rates
    assert original.ratio_lower_95 == pytest.approx(source_scaled.ratio_lower_95)
    assert scaled.ratio_lower_95 > original.ratio_lower_95  # more observations legitimately reduce Poisson uncertainty


def test_burst_collapse_and_absolute_decline_never_pass_even_with_large_aggregate_growth():
    result = relative((1, 1, 1, 2, 100, 3))
    assert result.raw_ratio == 35 and result.ratio_lower_95 > 1
    assert not result.recent_share_non_decreasing and not result.excess_growth_supported
    declining = relative((1, 1, 1, 40, 30, 20), (1000, 1000, 1000, 4000, 2000, 1000))
    assert declining.recent_share_non_decreasing and not declining.observed_growth


def test_sparse_signal_can_be_a_positive_observation_without_statistical_confirmation():
    result = relative((0, 0, 0, 0, 1, 2))
    assert result.raw_ratio is None and result.smoothed_ratio == 7
    assert result.observed_growth and not result.excess_growth_supported
    assert result.ratio_lower_95 < 1 and result.ratio_upper_95 is None
    assert "interval_assumes_poisson_counts_and_does_not_cover_source_or_membership_bias" in result.limitations


@pytest.mark.parametrize("counts,field,reason", [
    ((0, 0, 0, 0, 0, 0), (1000,) * 6, "no_candidate_observations"),
    ((1, 1, 1, 2, 4, 8), (1000, 1000, 1000, 0, 1000, 1000), "zero_field_exposure"),
    ((1, 1, 1, 2, 4, 8), (1000, 1000, 1000, 1000, 1000, 4), "candidate_exceeds_reference_field_exposure"),
])
def test_bad_denominators_and_absent_observations_do_not_create_growth(counts, field, reason):
    result = relative(counts, field)
    assert result.status == "unavailable" and result.reason == reason
    assert not result.excess_growth_supported and not result.observed_growth
    assert result.smoothed_ratio is None


def test_interval_matches_known_conditional_binomial_boundary():
    result = relative((0, 0, 0, 2, 4, 8))
    expected_p = 0.025 ** (1 / 14)
    assert result.ratio_lower_95 == pytest.approx(expected_p / (1 - expected_p))
    assert result.ratio_upper_95 is None and result.excess_growth_supported


def test_ten_year_history_uses_adjacent_final_windows_and_validates_exact_plan_query():
    from app.pilot.field_history import verify_field_exposure

    plan = type(query_plan()).model_validate(query_plan().model_dump(mode="python") | {"completed_years": tuple(range(2016, 2026))})
    exposure = field_exposure((90000,) * 4 + (1000,) * 6, plan=plan)
    result = normalize_growth((100, 100, 100, 100, 1, 1, 1, 2, 4, 8), exposure,
                              plan_hash=plan.plan_hash, years=plan.completed_years)
    assert result.baseline_studies == 3 and result.baseline_exposure == 3000
    assert result.raw_ratio == pytest.approx(14 / 3)
    assert verify_field_exposure(exposure, plan) == exposure
    changed_plan = plan.model_copy(update={"english_query": "another fixed query"})
    with pytest.raises(ValueError, match="query plan"):
        verify_field_exposure(exposure, changed_plan)


def test_exposure_create_normalizes_aware_timestamp_before_hashing():
    from datetime import timedelta, timezone

    original = field_exposure()
    same_instant = NOW.astimezone(timezone(timedelta(hours=3)))
    normalized = FieldExposure.create(plan_hash=original.plan_hash, fixed_query=original.fixed_query,
                                      years=original.years, observed_at=same_instant)
    assert normalized == original


def test_provider_cannot_silently_normalize_the_recorded_field_query():
    plan = type(query_plan()).model_validate(query_plan().model_dump(mode="python") | {"english_query": " direct lithium extraction "})
    calls = []
    result = collect_field_exposure(plan, Context(), CredentialStore(),
        provider_factory=lambda _: Provider([SourcePage(scanned=0, total_available=0, exhausted=True)], calls))
    assert calls == []
    assert all(item.count is None and item.reason == "query_changes_under_provider_normalization" for item in result.years)


@pytest.mark.parametrize("counts,field", [((1, 1, 1, 2, 4, 8), (1000,) * 6),
                                         ((0, 0, 1, 500, 1000, 1499), (10000,) * 6)])
def test_portable_interval_replay_ignores_platform_last_bit_in_inverse_beta(monkeypatch, counts, field):
    import math
    from types import SimpleNamespace
    from app.pilot import field_history

    expected = relative(counts, field)
    beta = field_history.import_module("scipy.stats").beta
    changed_beta = SimpleNamespace(ppf=lambda q, a, b: math.nextafter(float(beta.ppf(q, a, b)), math.inf))
    monkeypatch.setattr(field_history, "import_module", lambda _: SimpleNamespace(beta=changed_beta))
    assert relative(counts, field) == expected
