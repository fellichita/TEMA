"""JSON integration contract, independently of ML fitting and business-rule audits."""

from copy import deepcopy
import json
import math

import pytest
from pydantic import ValidationError

from tools.result_schema import ResultContract


def valid_result():
    quote = {"text": "A verbatim source statement.", "study_id": "doi:example",
             "url": "https://example.org/paper", "title": "Example paper", "mode": "source_excerpt"}
    card = {
        "id": "candidate-a", "title": "photonic neural", "keywords": ["photonic neural"],
        "study_count": 1, "study_ids": ["doi:example"],
        "metrics": {"years": [{"year": 2025, "documents": 1, "direction_documents": 10, "share": .1}],
                    "first_observed_year_in_corpus": 2019, "first_observed_year_in_window": 2025,
                    "growth_ratio": 1.5, "score": 25.0, "score_components": {"growth": .1},
                    "score_weights": {"growth": 35}, "growth_pattern": False},
        "card": {"problem": quote, "advantage": None, "example": None},
        "sources": [{"id": "doi:example", "url": "https://example.org/paper"}],
        "status": "exploratory_candidate", "stage": "requires_review",
        "direction_guard": {"axis_check": "partial", "reason": "document_support"},
        "selection": {"bucket": "preliminary_signals", "reasons": ["incomplete_supported_card"],
                      "window_document_share": .1, "established_threshold": .08},
        "execution": {"evidence_level": "nominal"}, "document_evidence": [],
        "evidence_annotations": {}, "explanations": {"problem": "Пояснение"}, "limitations": [],
    }
    return {
        "schema_version": 2, "pipeline_version": "local-mvp-nmf-1.4.0",
        "fingerprint": "a" * 64, "implementation_fingerprint": "b" * 64,
        "options": {"topic": "photonic neuromorphic computing", "start_year": 2020, "end_year": 2025},
        "source": "openalex", "provenance": {"snapshot_sha256": "c" * 64},
        "preparation": {}, "temporal_selection": {}, "coverage": {"2025": []},
        "growth_data_comparable": False, "warnings": ["Пилотная выдача"],
        "candidates": [], "preliminary_signals": [card], "established": [], "excluded_off_direction": [],
        "selection_summary": {"shortfall": 15}, "status": "ranked",
    }


def card(data):
    return data["preliminary_signals"][0]


def test_valid_json_with_original_quotes_and_empty_optional_fields_roundtrips():
    data = valid_result()
    parsed = ResultContract.model_validate(json.loads(json.dumps(data, ensure_ascii=False)))
    assert parsed.model_dump() == data


def test_new_sections_and_axis_decisions_are_represented_in_the_published_schema():
    schema = ResultContract.model_json_schema()
    assert set(("candidates", "preliminary_signals", "established", "excluded_off_direction")) <= set(schema["required"])
    assert schema["$defs"]["Guard"]["properties"]["axis_check"]["enum"] == ["supported", "partial", "off_direction"]
    assert schema["properties"]["candidates"]["maxItems"] == 15
    assert schema["$defs"]["Selection"]["properties"]["established_threshold"] == {
        "const": 0.08, "title": "Established Threshold", "type": "number"}


@pytest.mark.parametrize("value", ["0.08", None, False, True, 0, 1, 0.0, 0.0801, 0.0799,
                                  math.nextafter(0.08, 0.0), math.nextafter(0.08, 1.0)])
def test_legacy_established_threshold_is_an_exact_json_constant_without_coercion(value):
    data = valid_result()
    card(data)["selection"]["established_threshold"] = value
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("field", ["schema_version", "fingerprint", "temporal_selection", "candidates",
                                  "preliminary_signals", "established", "excluded_off_direction"])
def test_missing_required_result_fields_are_rejected(field):
    data = valid_result()
    del data[field]
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("field", ["metrics", "card", "selection", "execution", "document_evidence"])
def test_missing_required_card_fields_are_rejected(field):
    data = valid_result()
    del card(data)[field]
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version="2"),
    lambda d: d.update(growth_data_comparable="false"),
    lambda d: d.update(preliminary_signals={}),
    lambda d: card(d).update(study_count="1"),
    lambda d: card(d).update(study_count=True),
    lambda d: card(d)["metrics"].update(score="25.0"),
    lambda d: card(d)["metrics"]["years"][0].update(documents=-1),
    lambda d: card(d)["metrics"]["years"][0].update(share=1.1),
    lambda d: card(d)["metrics"]["years"][0].update(year="2025"),
    lambda d: card(d)["metrics"]["score_weights"].update(growth=35.5),
    lambda d: card(d)["card"]["problem"].update(url="javascript:alert(1)"),
])
def test_wrong_json_types_and_declared_numeric_bounds_are_rejected(mutate):
    data = valid_result()
    mutate(data)
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("decision", ["supported", "partial", "off_direction"])
def test_all_current_axis_values_are_accepted(decision):
    data = valid_result()
    card(data)["direction_guard"]["axis_check"] = decision
    assert ResultContract.model_validate(data).preliminary_signals[0].direction_guard.axis_check == decision


@pytest.mark.parametrize("decision", ["requires_review", "direct", "", None, 1])
def test_obsolete_or_invalid_axis_values_are_rejected(decision):
    data = valid_result()
    card(data)["direction_guard"]["axis_check"] = decision
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("bucket", ["candidates", "preliminary_signals", "established", "excluded_off_direction"])
def test_each_new_section_validates_its_card_structure(bucket):
    data = valid_result()
    broken = deepcopy(card(data))
    broken["metrics"]["score"] = "invalid-score"
    data[bucket] = [broken]
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


def test_top_limit_does_not_limit_transparent_additional_sections():
    data = valid_result()
    data["preliminary_signals"] = [deepcopy(card(data)) for _ in range(16)]
    ResultContract.model_validate(data)
    data["candidates"] = data["preliminary_signals"]
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", ["score", "provenance", "execution", "guard_extra", "root_extra"])
def test_nonfinite_numbers_are_rejected_in_typed_json_and_extension_fields(number, location):
    data = valid_result()
    if location == "score":
        card(data)["metrics"]["score"] = number
    elif location == "provenance":
        data["provenance"]["nested"] = [True, {"value": number}]
    elif location == "execution":
        card(data)["execution"]["diagnostic"] = {"score": number}
    elif location == "guard_extra":
        card(data)["direction_guard"]["future_extension"] = {"values": [number]}
    else:
        data["future_extension"] = {"values": [number]}
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


def test_finite_json_extensions_remain_compatible_without_mutating_the_input():
    data = valid_result()
    data["future_extension"] = {"nested": [None, True, False, -1, 2.5, "описание", {"x": 0}]}
    card(data)["direction_guard"]["future_extension"] = {"value": [1, 2.5]}
    before = deepcopy(data)
    assert ResultContract.model_validate(data).model_dump() == data
    assert data == before
