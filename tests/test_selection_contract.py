from copy import deepcopy

import pytest

from app.ml.selection import partition_candidates, ranking_key, selection_for


def candidate(identifier="a", count=8, score=50, level="direct"):
    return {"id": identifier, "title": identifier, "study_count": 999,
            "metrics": {"years": [{"year": 2025, "documents": count, "direction_documents": 100}],
                        "growth_pattern": True, "score": score, "recent_documents": count},
            "card": {key: {"text": "evidence"} for key in ("problem", "advantage", "example")},
            "execution": {"evidence_level": level}}


@pytest.mark.parametrize("count,bucket", [(0, "candidates"), (8, "candidates"), (9, "established")])
def test_volume_gate_uses_only_window_counts_and_strict_boundary(count, bucket):
    group = candidate(count=count)
    result = selection_for(group, True)
    assert result["bucket"] == bucket
    assert result["window_document_share"] == count / 100
    assert group["study_count"] == 999


def test_missing_direction_totals_do_not_invent_a_share():
    group = candidate()
    group["metrics"]["years"][0]["direction_documents"] = 0
    assert selection_for(group, False)["window_document_share"] is None
    assert selection_for(group, False)["bucket"] == "preliminary_signals"


@pytest.mark.parametrize("change,reason", [
    ("coverage", "incomparable_growth_coverage"), ("growth", "sustained_growth_not_established"),
    ("nominal", "nominal_execution_only"), ("card", "incomplete_supported_card")])
def test_preliminary_is_not_silently_used_to_fill_top(change, reason):
    group = candidate()
    if change == "growth": group["metrics"]["growth_pattern"] = False
    if change == "nominal": group["execution"]["evidence_level"] = "nominal"
    if change == "card": group["card"]["advantage"] = None
    buckets, summary = partition_candidates([group], change != "coverage", 15)
    assert buckets["candidates"] == []
    assert reason in buckets["preliminary_signals"][0]["selection"]["reasons"]
    assert summary["shortfall"] == 15


def test_off_direction_precedes_volume_and_all_buckets_are_disjoint():
    groups = [candidate("main"), candidate("large", count=20), candidate("nominal", level="nominal"),
              candidate("off", count=30)]
    groups[-1]["direction_guard"] = {"axis_check": "off_direction"}
    original_scores = [c["metrics"]["score"] for c in groups]
    buckets, _ = partition_candidates(groups, True, 15)
    assert [len(v) for v in buckets.values()] == [1, 1, 1, 1]
    assert len({c["id"] for values in buckets.values() for c in values}) == 4
    assert [c["metrics"]["score"] for c in groups] == original_scores


def test_direct_precedes_nominal_without_changing_scores():
    groups = [candidate("nominal", score=99, level="nominal"), candidate("direct", score=12)]
    assert [g["id"] for g in sorted(groups, key=ranking_key)] == ["direct", "nominal"]
    assert groups[0]["metrics"]["score"] == 99


def test_top_limit_is_applied_after_gates_and_reports_nonselected_ids():
    groups = [candidate(str(i), score=i) for i in range(20)] + [candidate("established", count=50, score=100)]
    buckets, summary = partition_candidates(deepcopy(groups), True, 15)
    assert len(buckets["candidates"]) == 15
    assert len(buckets["established"]) == 1
    assert summary["eligible_before_limit"] == 20
    assert len(summary["below_top_ids"]) == 5


@pytest.mark.parametrize("modality,bucket", [("research_reference", "preliminary_signals"),
                                            ("reported_result", "candidates")])
def test_bibliographic_reference_alone_does_not_complete_a_main_top_example(modality, bucket):
    group = candidate()
    group["evidence_annotations"] = {"example": {"modality": modality}}
    result = selection_for(group, True)
    assert result["bucket"] == bucket
    assert ("example_is_bibliographic_reference" in result["reasons"]) == (modality == "research_reference")
