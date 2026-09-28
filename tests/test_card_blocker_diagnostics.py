"""Checks that diagnostics do not confuse rejected alternatives with empty fields."""

from copy import deepcopy

import pytest

from tools.diagnose_card_blockers import diagnose_result, summarize, BUCKETS


def result_with_group(bucket="preliminary_signals"):
    quote = {"text": "A chosen passage.", "study_id": "s", "mode": "source_excerpt"}
    group = {"id": "g", "title": "A group", "study_count": 6,
             "card": {field: deepcopy(quote) for field in ("problem", "advantage", "example")},
             "sources": [{"id": "s", "title": "Source", "url": "https://example.org/source"}],
             "rejected_quotes": {field: [] for field in ("problem", "advantage", "example")},
             "evidence_annotations": {field: {"modality": "source_statement"} for field in ("problem", "advantage", "example")},
             "selection": {"reasons": ["off_direction"] if bucket == "excluded_off_direction" else []}}
    result = {key: [] for key in BUCKETS}
    result.update(growth_data_comparable=False)
    result[bucket] = [group]
    return result, group


def rejection(reason, text="Rejected passage."):
    return {"study_id": "s", "text": text, "reason": reason, "accepted": False}


def test_rejected_alternative_does_not_make_filled_field_a_blocker():
    result, group = result_with_group()
    group["rejected_quotes"]["problem"] = [rejection("not_a_problem_statement")]
    original = deepcopy(result)
    rows = diagnose_result("test", result)
    summary = summarize(rows)
    assert summary["empty_field_count"] == 0
    assert summary["rule_table"] == []
    assert rows[0]["fields"]["problem"]["saved_rejected_count"] == 1
    assert result == original


def test_no_candidate_differs_from_guard_rejected_and_missing_source():
    result, group = result_with_group()
    group["card"]["problem"] = None
    group["card"]["advantage"] = None
    group["rejected_quotes"]["advantage"] = [rejection("desired_benefit_not_reported")]
    rows = diagnose_result("test", result)
    assert rows[0]["fields"]["problem"]["state"] == "no_candidate"
    assert rows[0]["fields"]["advantage"]["state"] == "guard_rejected"
    assert rows[0]["fields"]["sources"]["state"] == "filled"
    assert rows[0]["fields"]["advantage"]["guard_rejections"][0]["url"] == "https://example.org/source"
    group["rejected_quotes"]["advantage"] = []
    group["sources"] = []
    assert diagnose_result("test", result)[0]["fields"]["sources"]["state"] == "no_source"


def test_multiple_rejections_rules_and_fields_count_groups_only_once():
    result, group = result_with_group()
    for field in ("problem", "advantage"):
        group["card"][field] = None
    group["rejected_quotes"]["problem"] = [rejection("rule_a"), rejection("rule_a", "Second.")]
    group["rejected_quotes"]["advantage"] = [rejection("rule_a"), rejection("rule_b")]
    summary = summarize(diagnose_result("test", result))
    rules = {item["rule"]: item for item in summary["rule_table"]}
    assert (summary["groups_with_empty_fields"], summary["empty_field_count"]) == (1, 2)
    assert (rules["rule_a"]["rejected_fragments"], rules["rule_a"]["affected_fields"], rules["rule_a"]["affected_groups"]) == (3, 2, 1)
    assert rules["rule_b"]["affected_groups"] == 1


def test_short_circuit_does_not_hide_empty_field_or_make_reference_empty():
    result, group = result_with_group("excluded_off_direction")
    group["card"]["problem"] = None
    group["evidence_annotations"]["example"]["modality"] = "research_reference"
    summary = summarize(diagnose_result("test", result))
    assert summary["empty_field_count"] == 1
    assert summary["example_research_reference_count"] == 1
    assert summary["selection_reason_counts"] == {"off_direction": 1}


def test_main_top_is_excluded_but_other_buckets_are_included():
    result, group = result_with_group("candidates")
    assert diagnose_result("test", result) == []
    for bucket in BUCKETS[1:]:
        result[bucket] = [{**deepcopy(group), "id": bucket}]
    assert len(diagnose_result("test", result)) == 3


def test_duplicate_groups_are_rejected_and_directions_disambiguate_ids():
    result, group = result_with_group()
    rows = diagnose_result("one", result)
    assert summarize(rows + diagnose_result("two", result))["groups_outside_top"] == 2
    with pytest.raises(ValueError, match="повторена"):
        summarize(rows + deepcopy(rows))
    result["established"] = [deepcopy(group)]
    with pytest.raises(ValueError, match="Повтор группы"):
        diagnose_result("one", result)
