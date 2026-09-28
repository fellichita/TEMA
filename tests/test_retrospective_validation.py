"""Temporal leakage and honest denominators for the offline ranking-only pilot."""

from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest

from app.ml.model import fit_topics
from app.ml.validation import (
    PROTOCOL, compare_rankings, exclude_known_versions, ranking_quality, run_retrospective, target_outcome,
)
from tests.mvp_fixture import records


def corpus():
    entries = []
    for year in range(2014, 2024):
        for document in records(year):
            data = document.model_dump(mode="json")
            data["fetched_at"] = "2026-09-06T00:00:00Z"
            # Alphabetic author identities prevent the production name normalizer
            # from collapsing distinct synthetic years after removing digits.
            data["authors"] = ["Researcher " + "".join(
                chr(97 + int(c)) if c.isdigit() else c for c in document.source_id if c.isalnum())]
            entries.append({"document_key": document.document_key, "revision_id": document.source_id,
                            "document": data})
    return {"topic": "photonic neuromorphic computing", "source": "openalex", "entries": entries,
            "periods": [{"from_date": f"{y}-01-01", "until_date": f"{y}-12-31", "issues": []}
                        for y in range(2014, 2024)], "provenance": {"history_id": "offline-validation-test"}}


def test_future_changes_never_change_training_fit_ranking_or_fingerprint():
    data = corpus()
    original = run_retrospective(data)
    changed = deepcopy(data)
    for entry in changed["entries"]:
        if entry["document"]["publication_year"] >= 2020:
            entry["document"]["abstract"] += " futureonlyvocabulary futureonlyvocabulary."
    later_copy = deepcopy(changed["entries"][0])
    later_copy["document"].update(publication_year=2023, publication_date="2023-06-01")
    later_copy["revision_id"] = "a-future-revision-that-must-not-invalidate-training"
    changed["entries"].append(later_copy)
    result = run_retrospective(changed)
    assert result["train_fingerprint"] == original["train_fingerprint"]
    assert result["training_ranking"] == original["training_ranking"]
    assert result["baseline_ranking"] == original["baseline_ranking"]
    assert result["groups"] == original["groups"]
    assert result["model"]["vocabulary"] == original["model"]["vocabulary"]
    assert "futureonlyvocabulary" not in result["model"]["vocabulary"]
    assert later_copy["document_key"] in {s["id"] for s in result["holdout_train_versions_excluded"]}


def test_fit_only_receives_training_work_and_both_rankings_use_the_same_pool():
    seen = []
    def fit(studies):
        seen.extend(studies)
        assert all(s["year"] <= 2019 for s in studies)
        return fit_topics(studies)
    result = run_retrospective(corpus(), fit_topics_fn=fit)
    assert seen and result["groups"]
    assert set(result["training_ranking"]) == set(result["baseline_ranking"])
    assert set(result["outcomes"]) == set(result["training_ranking"])
    assert all(max(row["year"] for row in g["metrics"]["years"]) == 2019 for g in result["groups"])
    assert all(g["metrics"]["score_weights"] ==
               {"growth": 35, "recency": 20, "persistence": 15, "coherence": 20, "support": 10}
               for g in result["groups"])


def test_future_zero_feature_vectors_are_not_assigned_to_the_first_topic():
    from scipy.sparse import csr_matrix
    def fit(studies):
        fitted = fit_topics(studies)
        vectorizer = fitted["vectorizer"]
        fitted["vectorizer"] = SimpleNamespace(
            get_feature_names_out=vectorizer.get_feature_names_out,
            transform=lambda texts: csr_matrix((len(texts), fitted["matrix"].shape[1])),
        )
        return fitted
    result = run_retrospective(corpus(), fit_topics_fn=fit)
    assert result["model"]["holdout_zero_features"] == sum(result["holdout_counts"].values())
    assert result["groups"]
    assert all(sum(outcome["counts"].values()) == 0 for outcome in result["outcomes"].values())


def test_same_snapshot_repeats_exactly_offline_and_input_order_does_not_change_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must never access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    data = corpus()
    first = run_retrospective(data)
    assert first == run_retrospective(data)
    reordered = deepcopy(data)
    reordered["entries"].reverse()
    assert first == run_retrospective(reordered)
    json.dumps(first, allow_nan=False)
    assert first["status"] == "exploratory_retrospective"


@pytest.mark.parametrize("years", [[], [2020], [2020, 2021, 2022]])
def test_missing_future_direction_year_is_unknown_not_zero_growth(years):
    outcome = target_outcome({2023: 20}, {y: 100 for y in years})
    assert outcome["growing"] is None
    assert outcome["growth_ratio"] is None


@pytest.mark.parametrize("counts,expected", [
    ({2020: 5, 2021: 5}, False),
    ({2020: 5, 2021: 5, 2022: 5, 2023: 5}, False),
    ({2020: 1, 2021: 1, 2022: 6, 2023: 6}, True),
    ({2022: 5, 2023: 5}, True),
    ({2022: 4, 2023: 5}, False),
    ({}, False),
])
def test_fixed_future_target_cases(counts, expected):
    value = target_outcome(counts, {y: 100 for y in range(2020, 2024)})
    assert value["growing"] is expected
    if counts == {2020: 5, 2021: 5}:
        assert value["growth_ratio"] == 0


def test_absolute_publication_growth_does_not_imply_relative_growth():
    assert not target_outcome({2020: 5, 2021: 5, 2022: 10, 2023: 10},
                              {2020: 100, 2021: 100, 2022: 300, 2023: 300})["growing"]


def test_partial_future_coverage_is_visible_and_never_reported_as_validated_prediction():
    data = corpus()
    data["periods"][-1]["issues"] = ["skipped source records"]
    result = run_retrospective(data)
    assert not result["data_comparable"]
    assert result["holdout_coverage"]["2023"] == ["skipped source records"]
    assert result["status"] == "exploratory_retrospective"
    assert "not evaluation" in result["scope"]


@pytest.mark.parametrize("missing_year,reason", [
    (2015, "train:unknown_direction_denominator"),
    (2021, "holdout:unknown_direction_denominator"),
])
def test_complete_calendar_with_zero_direction_records_is_not_comparable(missing_year, reason):
    data = corpus()
    data["entries"] = [e for e in data["entries"] if e["document"]["publication_year"] != missing_year]
    result = run_retrospective(data)
    assert not result["data_comparable"]
    assert reason in result["metadata_limitations"]


def test_no_training_work_returns_a_valid_empty_report_without_fitting():
    data = corpus()
    data["entries"] = [e for e in data["entries"] if e["document"]["publication_year"] > 2019]
    result = run_retrospective(data, fit_topics_fn=lambda _: pytest.fail("Empty training fit"))
    assert result["diagnostic"] == "insufficient_training_studies"
    assert result["evaluation"]["proposed"]["k"] == 0
    assert result["evaluation"]["proposed"]["precision_at_k"] is None
    assert result["evaluation"]["lift"] is None


def test_k_below_fifteen_never_fabricates_precision_at_fifteen():
    quality = ranking_quality(["a", "b"], {"a": {"growing": True}, "b": {"growing": False}})
    assert quality["precision_at_k"] == .5
    assert quality["precision_at_15"] is None
    assert quality["coverage_at_15"] == pytest.approx(2 / 15)


def test_unknown_outcome_does_not_shrink_denominator_to_make_precision_better():
    quality = ranking_quality(["a", "b"], {"a": {"growing": True}, "b": {"growing": None}})
    assert quality["precision_at_k"] is None
    assert quality["outcomes_known"] == 1
    assert quality["k"] == 2


def test_zero_baseline_precision_has_no_infinite_lift():
    keys = [str(i) for i in range(16)]
    values = {key: {"growing": key == "15"} for key in keys}
    result = compare_rankings(list(reversed(keys)), keys, values)
    assert result["lift"] is None
    assert result["baseline"]["precision_at_k"] == 0
    assert result["proposed"]["precision_at_15"] == pytest.approx(1 / 15)
    assert result["ranking_can_discriminate_at_k"]


def test_same_complete_small_pool_cannot_claim_ranking_discrimination():
    outcome = {"a": {"growing": True}, "b": {"growing": False}}
    result = compare_rankings(["a", "b"], ["b", "a"], outcome)
    assert result["selected_overlap"] == 2
    assert not result["ranking_can_discriminate_at_k"]
    assert result["lift"] == 1


def test_unknown_future_outcome_cannot_claim_zero_precision_or_lift():
    data = corpus()
    data["entries"] = [e for e in data["entries"] if e["document"]["publication_year"] <= 2019]
    result = run_retrospective(data)
    assert result["evaluation"]["lift"] is None
    assert result["evaluation"]["baseline"]["precision_at_k"] is None


def test_known_identity_and_long_title_versions_excluded_but_short_titles_not_merged():
    train = [{"id": "doi:a", "title": "An experimental optical neural network chip implementation",
              "authors": ["Alice Smith"], "versions": [{"document_key": "doi:alias"}]},
             {"id": "doi:b", "title": "Neural chips", "authors": ["Alice Smith"]}]
    future = [dict(train[0], id="doi:alias", versions=[]),
              dict(train[0], id="doi:other", versions=[]),
              dict(train[0], id="doi:new-author", authors=["Bob Jones"], versions=[]),
              dict(train[1], id="doi:new-short")]
    retained, rejected = exclude_known_versions(train, future)
    assert [s["id"] for s in retained] == ["doi:new-author", "doi:new-short"]
    assert [s["reason"] for s in rejected] == ["train_identity", "train_long_title_first_author"]


@pytest.mark.parametrize("new_identity,expected", [(False, 0), (True, 1)])
def test_early_work_without_usable_text_is_still_known_before_cutoff(new_identity, expected):
    data = corpus()
    early = deepcopy(data["entries"][0])
    early["document"]["abstract"] = "Too short."
    late = deepcopy(data["entries"][0])
    late["document"].update(publication_year=2022, publication_date="2022-06-01")
    late["revision_id"] = "full-later-version"
    if new_identity:
        late["document_key"] = "doi:10.9999/new-independent-study"
        late["document"]["doi"] = "10.9999/new-independent-study"
        late["document"]["title"] += " with an independent architecture"
    data["entries"] = [early, late]
    result = run_retrospective(data)
    assert result["train_preparation"]["rejected"] == {"missing_or_short_abstract": 1}
    assert result["holdout_counts"]["2022"] == expected
    assert len(result["holdout_train_versions_excluded"]) == 1 - expected


def test_returned_protocol_does_not_mutate_predeclared_thresholds():
    result = run_retrospective({**corpus(), "entries": []})
    result["protocol"]["train_scoring_years"].append(2020)
    assert PROTOCOL["train_scoring_years"] == list(range(2014, 2020))
