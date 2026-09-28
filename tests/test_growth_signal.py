"""Known growth trajectories and a planted signal through the unchanged NMF model."""

from collections import Counter
from copy import deepcopy
import json
import math

import pytest

from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from app.ml.engine import analyze, metrics
from tests.mvp_fixture import research_groups, GROWTH_PROFILES, growth_snapshot


YEARS = list(range(2020, 2026))
OPTIONS = AnalysisOptions(topic="photonic neuromorphic computing")


def trajectory(counts, totals=None):
    studies = [{"year": year} for year, count in zip(YEARS, counts, strict=True) for _ in range(count)]
    return metrics(studies, totals if totals is not None else {year: 100 for year in YEARS}, YEARS, .8)


def test_planted_signal_has_the_declared_counts_before_model_fit():
    data = growth_snapshot()
    for kind, expected in GROWTH_PROFILES.items():
        actual = Counter(entry["document"]["publication_year"]
                         for batch in data["batches"] for entry in batch["documents"]
                         if entry["document"]["source_id"].startswith(f"growth-{kind}-"))
        assert tuple(actual[year] for year in YEARS) == expected
    assert all(batch["total"] == len(batch["documents"]) for batch in data["batches"])


@pytest.mark.parametrize("counts,expected", [
    ((0, 0, 0, 0, 4, 16), True),    # Newly observed in two completed years.
    ((0, 0, 0, 2, 4, 8), True),     # Newly observed in three completed years.
    ((1, 1, 1, 2, 4, 8), True),     # Existing growing topic.
    ((4, 4, 4, 4, 4, 4), False),    # Stable share.
    ((8, 8, 8, 8, 4, 2), False),    # Decline.
    ((0, 0, 0, 0, 0, 20), False),   # One-year spike is not sustained growth.
    ((0, 0, 0, 0, 2, 3), False),    # Insufficient evidence.
    ((0, 0, 0, 0, 1, 20), False),   # Only one supported year.
    ((0, 0, 0, 4, 0, 16), False),   # Gap after first observation.
    ((0, 0, 0, 4, 16, 4), False),   # Recent decline despite a zero baseline.
    ((1, 1, 1, 0, 4, 16), False),   # Established topic retains the three-year rule.
])
def test_growth_classification(counts, expected):
    result = trajectory(counts)
    assert result["growth_pattern"] is expected
    assert math.isfinite(result["growth_ratio"])
    assert 0 <= result["score"] <= 100
    json.dumps(result, allow_nan=False)


def test_zero_baseline_has_finite_smoothed_growth_and_keeps_raw_ratio_undefined():
    result = trajectory((0, 0, 0, 0, 4, 16))
    assert result["baseline_share"] == 0
    assert result["recent_share"] == pytest.approx(20 / 300)
    assert result["growth_ratio"] == pytest.approx(41)
    assert result["raw_growth_ratio"] is None
    assert result["growth_smoothing"] == .5
    assert result["zero_baseline"]
    assert result["new_topic_in_window"]
    assert result["score_components"]["growth"] > 0


def test_growth_uses_direction_volume_not_just_publication_counts():
    totals = {year: n for year, n in zip(YEARS, (100, 100, 100, 200, 400, 800), strict=True)}
    result = trajectory((2, 2, 2, 4, 8, 16), totals)
    assert result["raw_growth_ratio"] == pytest.approx(1)
    assert result["growth_ratio"] == pytest.approx((28.5 / 1401) / (6.5 / 301))
    assert not result["growth_pattern"]


@pytest.mark.parametrize("year", YEARS)
def test_missing_direction_volume_is_not_a_zero_topic_baseline(year):
    totals = {y: 100 for y in YEARS}
    totals[year] = 0
    result = trajectory((0, 0, 0, 0, 4, 16), totals)
    assert result["growth_ratio"] is None
    assert result["raw_growth_ratio"] is None
    assert not result["growth_pattern"]


def test_missing_direction_year_key_is_handled_without_inventing_data():
    totals = {y: 100 for y in YEARS if y != 2021}
    assert trajectory((0, 0, 0, 0, 4, 16), totals)["growth_ratio"] is None


def test_smoothing_does_not_create_growth_when_the_topic_has_no_recent_observations():
    studies = [{"year": 2019}]
    result = metrics(studies, {y: 100 if y < 2023 else 20 for y in YEARS}, YEARS, .8)
    assert result["growth_ratio"] is None
    assert not result["growth_pattern"]
    assert result["score_components"]["growth"] == 0


def test_older_known_work_prevents_the_new_topic_exception():
    studies = [{"year": 2019}] + [{"year": 2024}] * 4 + [{"year": 2025}] * 16
    result = metrics(studies, {y: 100 for y in YEARS}, YEARS, .8)
    assert not result["new_topic_in_window"]
    assert not result["growth_pattern"]


def test_planted_signal_is_recovered_ranked_and_explained_end_to_end():
    data = growth_snapshot()
    result = analyze(unpack_snapshot(data), OPTIONS)
    assert result["growth_data_comparable"]
    assert sorted(tuple(row["documents"] for row in candidate["metrics"]["years"])
                  for candidate in research_groups(result)) == sorted(GROWTH_PROFILES.values())
    json.dumps(result, allow_nan=False)
    expected_ids = {entry["document_key"] for batch in data["batches"] for entry in batch["documents"]
                    if entry["document"]["source_id"].startswith("growth-new_signal-")}
    matching = [candidate for candidate in research_groups(result)
                if expected_ids.intersection(candidate["study_ids"])]
    assert len(matching) == 1
    signal = matching[0]
    assert set(signal["study_ids"]) == expected_ids
    assert research_groups(result)[0]["id"] == signal["id"]
    assert tuple(row["documents"] for row in signal["metrics"]["years"]) == GROWTH_PROFILES["new_signal"]
    assert signal["metrics"]["first_observed_year"] == 2024
    assert signal["metrics"]["growth_pattern"]
    assert signal["status"] == "established"  # The planted signal occupies >8% of this small fixture.
    assert signal["stage"] == "requires_review"
    assert all(signal["card"].values())
    assert any("сглаж" in message for message in signal["limitations"])
    assert all(candidate["status"] in {"exploratory_candidate", "established"} for candidate in research_groups(result)
               if candidate["id"] != signal["id"])


def test_incomplete_source_coverage_cannot_confirm_the_planted_growth():
    data = deepcopy(growth_snapshot())
    data["history"]["periods"][0]["job"]["source_exhausted"] = False
    result = analyze(unpack_snapshot(data), OPTIONS)
    assert research_groups(result)
    assert not result["growth_data_comparable"]
    assert not result["candidates"]
    assert all(candidate["status"] in {"exploratory_candidate", "established"} for candidate in research_groups(result))


@pytest.mark.parametrize("volumes", [
    (1000, 1000, 1000, 10, 10, 10),  # Regression: smoothing used to invent G > 7.
    (100, 100, 100, 100, 100, 100),
    (10, 10, 10, 1000, 1000, 1000),
    (10**9, 10**9, 10**9, 1, 1, 1),
    (1000, 100, 10, 1, 5, 20),
])
def test_disappeared_topic_gets_zero_growth_regardless_of_direction_volume(volumes):
    result = trajectory((2, 2, 2, 0, 0, 0), dict(zip(YEARS, volumes, strict=True)))
    assert result["growth_ratio"] == 0
    assert result["raw_growth_ratio"] == 0
    assert result["score_components"]["growth"] == 0
    assert result["recent_documents"] == 0
    assert result["recent_share"] == 0
    assert result["baseline_share"] == pytest.approx(6 / sum(volumes[:3]))
    assert result["share_change_pp"] == pytest.approx(-600 / sum(volumes[:3]))
    assert not result["growth_pattern"]
    assert not result["zero_baseline"]
    assert not result["new_topic_in_window"]
    # The other four components and all five weights retain their prior meaning.
    assert result["score_components"]["recency"] == pytest.approx(1 / 6)
    assert result["score_components"]["coherence"] == .8
    assert result["score_components"]["persistence"] == 0
    assert result["score_components"]["support"] == 0
    assert result["score_weights"] == {
        "growth": 35, "recency": 20, "persistence": 15, "coherence": 20, "support": 10}
    assert result["score"] == 19.33
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("missing_year", YEARS)
@pytest.mark.parametrize("missing_key", [False, True])
def test_unknown_direction_volume_does_not_become_a_measured_disappearance(missing_year, missing_key):
    volumes = {year: 1000 if year < 2023 else 10 for year in YEARS}
    if missing_key:
        del volumes[missing_year]
    else:
        volumes[missing_year] = 0
    counts = [2 if year < 2023 and year != missing_year else 0 for year in YEARS]
    result = trajectory(counts, volumes)
    assert result["growth_ratio"] is None
    assert result["raw_growth_ratio"] is None
    assert result["score_components"]["growth"] == 0
    assert not result["growth_pattern"]
    assert result["years"][YEARS.index(missing_year)]["share"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("window_length", [4, 5, 6, 10, 30])
def test_disappearance_works_for_different_baseline_lengths(window_length):
    years = list(range(2026 - window_length, 2026))
    # One observed year in the baseline is sufficient; every direction year is known.
    result = metrics([{"year": years[0]}] * 6,
                     {year: 1000 if year < 2023 else 1 for year in years}, years, .8)
    assert result["growth_ratio"] == 0
    assert result["score_components"]["growth"] == 0
    assert not result["growth_pattern"]


@pytest.mark.parametrize("counts,expected_g,growing", [
    ((8, 8, 8, 8, 4, 2), 29 / 49, False),  # Decline with recent observations.
    ((4, 4, 4, 4, 4, 4), 1, False),        # Stable share.
    ((1, 1, 1, 2, 4, 8), 29 / 7, True),    # Established sustained growth.
    ((0, 0, 0, 0, 4, 16), 41, True),       # Planted new topic: smoothing stays active.
    ((0, 0, 0, 0, 0, 20), 41, False),      # One-year spike still not sustained.
    ((2, 2, 2, 4, 2, 0), 1, False),        # Last year empty, recent window not empty.
])
def test_non_disappearing_trajectories_keep_their_growth(counts, expected_g, growing):
    result = trajectory(counts)
    assert result["growth_ratio"] == pytest.approx(expected_g)
    assert result["growth_pattern"] is growing
    assert result["growth_smoothing"] == .5
    assert result["score_components"]["growth"] == pytest.approx(
        min(1, max(0, math.log2(expected_g) / 3)))


def test_disappeared_topic_and_planted_signal_through_the_real_model():
    data = growth_snapshot()
    vanished_ids = set()
    signal_ids = set()
    for batch, period in zip(data["batches"], data["history"]["periods"], strict=True):
        kept = []
        for entry in batch["documents"]:
            document = entry["document"]
            if document["source_id"].startswith("growth-stable_background-"):
                if document["publication_year"] >= 2023:
                    continue
                vanished_ids.add(entry["document_key"])
            if document["source_id"].startswith("growth-new_signal-"):
                signal_ids.add(entry["document_key"])
            kept.append(entry)
        batch.update(documents=kept, total=len(kept))
        period["job"].update(stored=len(kept), scanned=len(kept), total_available=len(kept))
    result = analyze(unpack_snapshot(data), OPTIONS)
    vanished = [c for c in research_groups(result) if vanished_ids.intersection(c["study_ids"])]
    assert len(vanished) == 1
    candidate = vanished[0]
    assert set(candidate["study_ids"]) == vanished_ids
    assert tuple(row["documents"] for row in candidate["metrics"]["years"]) == (12, 12, 12, 0, 0, 0)
    assert candidate["metrics"]["growth_ratio"] == 0
    assert candidate["metrics"]["score_components"]["growth"] == 0
    assert candidate["status"] in {"exploratory_candidate", "established"}
    assert all(candidate["card"].values())
    signal = research_groups(result)[0]
    assert set(signal["study_ids"]) == signal_ids
    assert signal["status"] == "established"  # The planted signal occupies >8% of this small fixture.
    assert signal["metrics"]["zero_baseline"]
    assert math.isfinite(signal["metrics"]["growth_ratio"])
    assert signal["metrics"]["growth_ratio"] > 1
    json.dumps(result, allow_nan=False)
