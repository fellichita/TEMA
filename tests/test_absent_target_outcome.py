"""Regressions for interpretable holdout growth, separate from ranking policy."""

import json

import pytest

from app.ml.validation import PROTOCOL, target_outcome


YEARS = PROTOCOL["target_early_years"] + PROTOCOL["target_recent_years"]


@pytest.mark.parametrize("totals", [
    {2020: 100, 2021: 100, 2022: 100, 2023: 100},
    {2020: 1000, 2021: 1000, 2022: 10, 2023: 10},
    {2020: 10, 2021: 10, 2022: 1000, 2023: 1000},
])
@pytest.mark.parametrize("counts", [{}, dict.fromkeys(YEARS, 0)])
def test_absent_topic_has_no_growth_even_when_direction_shrinks(counts, totals):
    outcome = target_outcome(counts, totals)
    assert outcome == {"growing": False, "growth_ratio": None,
                       "early_documents": 0, "recent_documents": 0,
                       "reason": "absent_topic"}


@pytest.mark.parametrize("year", YEARS)
@pytest.mark.parametrize("denominator", [None, 0, -1, float("nan"), float("inf"),
                                        float("-inf"), "100", False])
def test_unusable_denominator_is_unknown_even_for_absent_topic(year, denominator):
    totals = dict.fromkeys(YEARS, 100)
    totals[year] = denominator
    outcome = target_outcome({}, totals)
    assert outcome["growing"] is None
    assert outcome["growth_ratio"] is None
    assert outcome["reason"] == "unknown_direction_denominator"
    json.dumps(outcome, allow_nan=False)


@pytest.mark.parametrize("year", YEARS)
def test_missing_denominator_is_unknown(year):
    totals = dict.fromkeys(YEARS, 100)
    totals.pop(year)
    outcome = target_outcome({2023: 10}, totals)
    assert outcome["growing"] is None
    assert outcome["growth_ratio"] is None
    assert outcome["reason"] == "unknown_direction_denominator"


@pytest.mark.parametrize("count", [None, -1, float("nan"), float("inf"),
                                  float("-inf"), "1", True])
def test_invalid_topic_count_is_unknown_and_json_safe(count):
    outcome = target_outcome({2020: count}, dict.fromkeys(YEARS, 100))
    assert outcome["growing"] is None
    assert outcome["growth_ratio"] is None
    assert outcome["reason"] == "invalid_topic_counts"
    json.dumps(outcome, allow_nan=False)


@pytest.mark.parametrize("counts,expected", [
    ({2020: 5, 2021: 5}, False),
    ({2020: 5, 2021: 5, 2022: 3, 2023: 3}, False),
    ({2020: 5, 2021: 5, 2022: 5, 2023: 5}, False),
    ({2020: 1, 2021: 1, 2022: 6, 2023: 6}, True),
    ({2022: 5, 2023: 5}, True),
    ({2022: 4, 2023: 5}, False),
])
def test_observed_topics_keep_frozen_smoothing_and_thresholds(counts, expected):
    totals = dict.fromkeys(YEARS, 100)
    outcome = target_outcome(counts, totals)
    early = sum(counts.get(y, 0) for y in PROTOCOL["target_early_years"])
    recent = sum(counts.get(y, 0) for y in PROTOCOL["target_recent_years"])
    alpha = PROTOCOL["smoothing_alpha"]
    expected_ratio = (recent + alpha) / (early + alpha) if recent else 0
    assert outcome["growing"] is expected
    assert outcome["growth_ratio"] == pytest.approx(expected_ratio)
    assert outcome["reason"] == "growth_in_saved_sample"


def test_invalid_values_outside_the_frozen_target_window_are_irrelevant():
    outcome = target_outcome({2019: None, 2023: 10},
                             dict.fromkeys(YEARS, 100) | {2024: None})
    assert outcome["growing"] is True


def test_finite_input_counts_cannot_overflow_the_reported_growth_to_infinity():
    outcome = target_outcome({2022: 10}, {2020: 15 * 10**307, 2021: 1, 2022: 10, 2023: 1})
    assert outcome["growth_ratio"] is None
    assert outcome["growing"] is None
    assert outcome["reason"] == "growth_ratio_out_of_range"
    json.dumps(outcome, allow_nan=False)
