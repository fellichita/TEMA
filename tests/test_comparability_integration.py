"""The same distinction must reach replay audits and publication-date validation."""

from copy import deepcopy

import pytest

from app.ml.contracts import AnalysisOptions
from app.ml.engine import analyze
from app.ml.validation import run_retrospective
from tests.test_growth_comparability import clean_corpus
from tests.test_retrospective_validation import corpus as retrospective_corpus
from tools.audit_ml_result import audit_result


def quality_period(period):
    issues = ["Сбор периода неполный", "Выдача не исчерпана или есть пропуски"]
    period.update(issues=issues, nonblocking_issues=issues[:],
                  data_quality_notes=["Исчерпанная выдача содержит одну пропущенную невалидную запись."])


def test_audit_replays_soft_collection_and_detects_hidden_or_changed_diagnostics():
    corpus = clean_corpus()
    quality_period(corpus["periods"][0])
    result = analyze(corpus, AnalysisOptions(topic=corpus["topic"]))
    assert result["growth_data_comparable"]
    assert result["coverage"]["2020"]
    assert result["data_quality_notes"]
    assert audit_result(corpus, result)["ok"]
    for key, code in (("data_quality_notes", "data_quality_notes_replay"),
                      ("growth_comparability", "growth_comparability_replay")):
        changed = deepcopy(result)
        del changed[key]
        assert code in {e["code"] for e in audit_result(corpus, changed)["errors"]}
    changed = deepcopy(result)
    changed["warnings"] = [w for w in changed["warnings"] if w not in changed["data_quality_notes"]]
    assert "quality_warnings_visible" in {e["code"] for e in audit_result(corpus, changed)["errors"]}


@pytest.mark.parametrize("window, index", [("train", 0), ("holdout", -1)])
def test_retrospective_retains_quality_without_changing_fit_or_rankings(window, index):
    data = retrospective_corpus()
    before = run_retrospective(data)
    quality_period(data["periods"][index])
    result = run_retrospective(data)
    assert result["data_comparable"]
    assert result["data_quality_notes"]
    assert result["growth_comparability"][window]["calendar_complete"]
    assert not any(result["growth_comparability"][window]["blocking_issues"].values())
    assert result["training_ranking"] == before["training_ranking"]
    assert result["groups"] == before["groups"]
    assert result["model"] == before["model"]
    assert result["evaluation"] == before["evaluation"]
    assert result[f"{window}_coverage"][data["periods"][index]["from_date"][:4]]


def test_soft_period_does_not_hide_an_unrelated_new_hard_issue():
    data = retrospective_corpus()
    quality_period(data["periods"][-1])
    data["periods"][-1]["issues"].append("Неполный экспорт периода")
    result = run_retrospective(data)
    assert not result["data_comparable"]
    assert result["data_quality_notes"]
    assert result["growth_comparability"]["holdout"]["blocking_issues"]["2023"] == ["Неполный экспорт периода"]


def test_retrospective_zero_denominator_is_explicit_despite_complete_calendar():
    data = retrospective_corpus()
    data["entries"] = [e for e in data["entries"] if e["document"]["publication_year"] != 2021]
    result = run_retrospective(data)
    assert not result["data_comparable"]
    details = result["growth_comparability"]["holdout"]
    assert details["calendar_complete"]
    assert details["denominators"][1] == {"year": 2021, "documents": 0, "state": "zero"}
    assert details["blocking_issues"]["2021"]


def test_retrospective_cannot_hide_an_undated_record_without_known_period():
    data = retrospective_corpus()
    data["entries"][0]["document"].update(publication_year=None, publication_date=None)
    result = run_retrospective(data)
    assert not result["data_comparable"]
    assert all(result["growth_comparability"]["train"]["blocking_issues"].values())
    assert all(result["growth_comparability"]["holdout"]["blocking_issues"].values())
