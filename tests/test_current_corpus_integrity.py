from copy import deepcopy

from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from app.ml.engine import analysis_entries, analyze, metrics
from tests.mvp_fixture import snapshot, research_groups


def test_future_raw_version_cannot_change_earlier_clustering_or_counts():
    corpus = unpack_snapshot(snapshot())
    options = AnalysisOptions(topic=corpus["topic"])
    before = analyze(corpus, options)
    future = deepcopy(corpus["entries"][0])
    future["document"]["publication_year"] = 2026
    future["document"]["abstract"] = "Future previously unseen vocabulary would change merging and the fitted features."
    corpus["entries"].append(future)
    after = analyze(corpus, options)
    assert research_groups(before) == research_groups(after)
    assert before["model"] == after["model"]
    assert before["direction_counts"] == after["direction_counts"]
    assert after["temporal_selection"]["excluded_after_end_year"] == 1


def test_retracted_articles_are_explicitly_excluded_without_mutating_input():
    entries = deepcopy(unpack_snapshot(snapshot())["entries"][:3])
    entries[0]["document"]["title"] = "RETRACTED: " + entries[0]["document"]["title"]
    entries[1]["document"]["raw_metadata"] = {"is_retracted": True}
    entries[2]["document"]["title"] = "Study of retracted publications using photonic neural computing"
    original = deepcopy(entries)
    retained, report = analysis_entries(entries, 2025)
    assert retained == [entries[2]]
    assert report["excluded_retracted_occurrences"] == 2
    assert entries == original


def test_first_years_are_distinct_and_zero_window_observations_stay_missing():
    years = list(range(2020, 2026))
    values = metrics([{"year": 2017}, {"year": 2022}], {year: 10 for year in years}, years, .5)
    assert values["first_observed_year_in_corpus"] == 2017
    assert values["first_observed_year_in_window"] == 2022
    values = metrics([{"year": 2017}], {year: 10 for year in years}, years, .5)
    assert values["first_observed_year_in_window"] is None


def test_verified_date_dispute_is_quarantined_without_inventing_a_replacement_year():
    entries = deepcopy(unpack_snapshot(snapshot())["entries"][:2])
    entries[0]["document"].update(doi="10.1088/2515-7647/ae2e67", publication_year=2025)
    original = deepcopy(entries)
    retained, report = analysis_entries(entries, 2025)
    assert retained == [entries[1]]
    assert report["excluded_disputed_date_occurrences"] == 1
    assert len(report["date_reviews"][0]["sources"]) == 2
    assert entries == original
    entries[0]["document"]["publication_year"] = 2026
    retained, report = analysis_entries(entries, 2026)
    assert retained == entries
    assert not report["date_reviews"]
