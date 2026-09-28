"""Computability, collection completeness and sample quality are separate facts."""

from copy import deepcopy
import json

import pytest

from app.ml import engine
from app.ml.contracts import AnalysisOptions
from app.ml.corpus import unpack_snapshot
from tests.mvp_fixture import snapshot, research_groups


YEARS = list(range(2020, 2026))


def clean_corpus():
    data = snapshot()
    for batch in data["batches"]:
        for index, entry in enumerate(batch["documents"]):
            entry["document"]["authors"] = [f"Researcher Unique{batch['job_id']}_{index}"]
    return unpack_snapshot(data)


def assess(corpus, totals=None, preparation=None, selection=None):
    return engine.growth_assessment(corpus["periods"], YEARS,
        totals if totals is not None else dict.fromkeys(YEARS, 10),
        preparation=preparation, selection=selection)


@pytest.mark.parametrize("empty_year", YEARS)
def test_complete_calendar_with_no_retained_direction_work_cannot_confirm_growth(empty_year):
    corpus = clean_corpus()
    corpus["entries"] = [e for e in corpus["entries"] if e["document"]["publication_year"] != empty_year]
    result = engine.analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert result["coverage"][str(empty_year)] == []
    assert not result["growth_data_comparable"]
    assert not result["candidates"]
    details = result["growth_comparability"]
    assert details["calendar_complete"]
    assert details["denominators"][YEARS.index(empty_year)] == {
        "year": empty_year, "documents": 0, "state": "zero"}
    assert details["blocking_issues"][str(empty_year)]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kind", ["zero", "missing", "none", "negative", "boolean", "string", "nan"])
@pytest.mark.parametrize("year", [2020, 2023, 2025])
def test_denominator_diagnostic_distinguishes_measured_zero_from_unknown_and_invalid(kind, year):
    totals = dict.fromkeys(YEARS, 10)
    if kind == "missing":
        del totals[year]
    else:
        totals[year] = {"zero": 0, "none": None, "negative": -1,
                        "boolean": True, "string": "10", "nan": float("nan")}[kind]
    result = assess(clean_corpus(), totals)
    assert not result["growth_data_comparable"]
    state = "zero" if kind == "zero" else "unknown" if kind in {"missing", "none"} else "invalid"
    assert result["growth_comparability"]["denominators"][YEARS.index(year)] == {
        "year": year, "documents": 0 if state == "zero" else None, "state": state}
    assert result["growth_comparability"]["blocking_issues"][str(year)]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("reason", ["unknown source failure", "skipped source records",
                                   "Выдача не исчерпана или есть пропуски"])
def test_legacy_issue_without_structural_proof_remains_blocking(reason):
    corpus = clean_corpus()
    corpus["periods"][0] = {"from_date": "2020-01-01", "until_date": "2020-12-31", "issues": [reason]}
    result = assess(corpus)
    assert not result["growth_data_comparable"]
    assert result["growth_comparability"]["blocking_issues"]["2020"] == [reason]


def test_calendar_gap_stays_hard_even_when_a_nearby_period_only_has_quality_notes():
    corpus = clean_corpus()
    corpus["periods"][0]["from_date"] = "2020-01-02"
    result = assess(corpus)
    assert not result["growth_data_comparable"]
    assert not result["growth_comparability"]["calendar_complete"]
    assert "Пробел в календарном покрытии" in result["growth_comparability"]["blocking_issues"]["2020"]


@pytest.mark.parametrize("part, key", [
    ("selection", "excluded_retracted_occurrences"),
    ("selection", "excluded_disputed_date_occurrences"),
    ("preparation", "possible_versions_merged"),
    ("rejected", "conflicting_year"),
    ("rejected", "missing_or_short_abstract"),
])
def test_documented_sample_selection_is_visible_but_does_not_cancel_known_denominators(part, key):
    preparation, selection = {"rejected": {}}, {}
    target = selection if part == "selection" else preparation["rejected"] if part == "rejected" else preparation
    target[key] = 7
    original = deepcopy((preparation, selection))
    result = assess(clean_corpus(), preparation=preparation, selection=selection)
    assert result["growth_data_comparable"]
    assert not any(result["growth_comparability"]["blocking_issues"].values())
    assert result["data_quality_notes"] and "7" in " ".join(result["data_quality_notes"])
    assert (preparation, selection) == original


def test_retracted_extra_record_does_not_block_a_complete_surviving_direction_sample():
    corpus = clean_corpus()
    options = AnalysisOptions(topic=corpus["topic"])
    before = engine.analyze(corpus, options)
    assert before["growth_data_comparable"]
    withdrawn = deepcopy(corpus["entries"][0])
    withdrawn["document_key"] = "doi:10.9999/withdrawn-extra"
    withdrawn["revision_id"] = "withdrawn-extra"
    withdrawn["document"]["doi"] = "10.9999/withdrawn-extra"
    withdrawn["document"]["raw_metadata"] = {"is_retracted": True}
    corpus["entries"].append(withdrawn)
    after = engine.analyze(corpus, options)
    assert after["growth_data_comparable"]
    assert after["data_quality_notes"]
    assert after["temporal_selection"]["excluded_retracted_occurrences"] == 1
    assert before["direction_counts"] == after["direction_counts"]
    assert before["model"] == after["model"]
    assert {c["id"]: c["metrics"] for c in research_groups(before)} == {
        c["id"]: c["metrics"] for c in research_groups(after)}


def test_scope_is_explicit_and_healthy_saved_sample_is_comparable():
    result = assess(clean_corpus())
    assert result["growth_data_comparable"]
    assert result["growth_comparability"]["scope"] == "retained_studies_in_saved_corpus"
    assert result["data_quality_notes"] == []


def test_undated_record_in_legacy_memory_corpus_does_not_silently_confirm_comparability():
    corpus = clean_corpus()
    # No raw batch association remains after a caller introduces an undated record.
    corpus["entries"][0]["document"].update(publication_year=None, publication_date=None,
                                             publication_month=None, date_precision="unknown")
    result = engine.analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert not result["growth_data_comparable"]
    assert result["preparation"]["rejected"]["unknown_or_future_year"] == 1
    assert all(result["growth_comparability"]["blocking_issues"].values())


def test_known_outside_window_undated_period_does_not_block_an_independent_window():
    corpus = clean_corpus()
    corpus["periods"].append({"from_date": "2010-01-01", "until_date": "2010-12-31",
        "issues": ["Год публикации неизвестен; принадлежность периоду не подтверждена."], "undated_records": 1})
    document = deepcopy(corpus["entries"][0])
    document["revision_id"] = document["document_key"] = "undated-2010-batch-record"
    document["document"].update(doi="10.9999/undated", publication_year=None, publication_date=None)
    corpus["entries"].append(document)
    result = engine.analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert result["growth_data_comparable"]
