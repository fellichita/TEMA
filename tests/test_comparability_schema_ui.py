"""Typed optional growth diagnostics and their presentation, including legacy v2."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ui.window import Application
from tests.test_result_schema import valid_result
from tests.ui.test_desktop import TkCase
from tests.ui.test_trend_result_sections import result_sections
from tools.result_schema import ResultContract


def diagnostic():
    return {"scope": "retained_studies_in_saved_corpus", "calendar_complete": True,
            "blocking_issues": {"2020": [], "2021": ["Нулевой знаменатель направления"]},
            "denominators": [{"year": 2020, "documents": 12, "state": "positive"},
                             {"year": 2021, "documents": 0, "state": "zero"}]}


def result_with_diagnostic():
    return {**valid_result(), "growth_comparability": diagnostic(),
            "data_quality_notes": ["Пропущены непригодные записи источника: 2."]}


def test_optional_diagnostics_preserve_legacy_v2_roundtrip():
    data = valid_result()
    assert ResultContract.model_validate(data).model_dump() == data


def test_typed_diagnostics_roundtrip_without_mutating_input():
    data = result_with_diagnostic()
    before = deepcopy(data)
    assert ResultContract.model_validate(json.loads(json.dumps(data))).model_dump() == data
    assert data == before


@pytest.mark.parametrize("path,value", [
    (("growth_comparability",), []),
    (("growth_comparability", "scope"), "whole_scientific_literature"),
    (("growth_comparability", "calendar_complete"), "true"),
    (("growth_comparability", "calendar_complete"), 1),
    (("growth_comparability", "blocking_issues"), []),
    (("growth_comparability", "blocking_issues", "2020"), "a single issue"),
    (("growth_comparability", "blocking_issues", "2020"), [1]),
    (("growth_comparability", "denominators"), {}),
    (("growth_comparability", "denominators", 0, "year"), "2020"),
    (("growth_comparability", "denominators", 0, "year"), True),
    (("growth_comparability", "denominators", 0, "documents"), "12"),
    (("growth_comparability", "denominators", 0, "documents"), 12.0),
    (("growth_comparability", "denominators", 0, "documents"), True),
    (("growth_comparability", "denominators", 0, "state"), "complete"),
    (("growth_comparability", "denominators", 0, "state"), None),
    (("data_quality_notes",), "quality warning"),
    (("data_quality_notes",), [1]),
    (("data_quality_notes",), [None]),
])
def test_diagnostics_reject_wrong_types_and_unknown_states(path, value):
    data = result_with_diagnostic()
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("field", ["scope", "calendar_complete", "blocking_issues", "denominators"])
def test_present_diagnostic_requires_its_declared_fields(field):
    data = result_with_diagnostic()
    del data["growth_comparability"][field]
    with pytest.raises(ValidationError):
        ResultContract.model_validate(data)


@pytest.mark.parametrize("state,documents", [
    ("positive", 12), ("zero", 0), ("unknown", None), ("invalid", -1),
])
def test_all_denominator_diagnostic_states_remain_representable(state, documents):
    data = result_with_diagnostic()
    data["growth_comparability"]["denominators"] = [{"year": 2020, "documents": documents, "state": state}]
    assert ResultContract.model_validate(data).model_dump() == data


def test_published_schema_includes_optional_typed_diagnostics():
    generated = ResultContract.model_json_schema()
    generated.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                      "$id": "urn:trendanalizer:result:2"})
    saved = json.loads((Path(__file__).resolve().parents[1] / "docs/schemas/result-v2.schema.json").read_text(encoding="utf-8"))
    assert saved == generated
    assert {"growth_comparability", "data_quality_notes"} <= set(generated["properties"])
    assert not {"growth_comparability", "data_quality_notes"} & set(generated["required"])
    assert generated["$defs"]["DirectionDenominator"]["properties"]["state"]["enum"] == [
        "positive", "zero", "unknown", "invalid"]


class ComparabilityDiagnosticUITests(TkCase):
    def open_panel(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready)
        return self.app.trends_panel

    def test_hard_reasons_denominators_and_saved_sample_scope_are_visible(self):
        panel = self.open_panel()
        result = result_sections()
        result.update(growth_data_comparable=False, growth_comparability=diagnostic(),
                      data_quality_notes=["Пропущены непригодные записи источника: 2."])
        result["growth_comparability"]["denominators"].extend([
            {"year": 2022, "documents": None, "state": "unknown"},
            {"year": 2023, "documents": -1, "state": "invalid"}])
        before = deepcopy(result)
        panel.render_result(result)
        displayed = panel.diagnostics.get("1.0", "end")
        for phrase in ("Сопоставимость динамики", "сохранённым исследованиям после подготовки",
                       "Календарное покрытие: полное", "сопоставимость не подтверждена",
                       "Нулевой знаменатель направления", "2020: 12 — положительный",
                       "2021: 0 — нулевой", "2022: неизвестен — неизвестный",
                       "2023: -1 — некорректный", result["data_quality_notes"][0]):
            self.assertIn(phrase, displayed)
        self.assertEqual(result, before)
        self.assertTrue(panel.export_button.instate(["!disabled"]))
        self.assertTrue(panel.link_button.instate(["!disabled"]))

    def test_soft_warnings_do_not_hide_comparability_or_duplicate_notes(self):
        panel = self.open_panel()
        result = result_sections()
        note = "Пропущены непригодные записи источника: 2."
        result["warnings"].append(note)
        result.update(growth_data_comparable=True, data_quality_notes=[note],
                      growth_comparability={"scope": "retained_studies_in_saved_corpus",
                                            "calendar_complete": True, "blocking_issues": {"2020": []},
                                            "denominators": [{"year": 2020, "documents": 12, "state": "positive"}]})
        panel.render_result(result)
        displayed = panel.diagnostics.get("1.0", "end")
        self.assertEqual(displayed.count(note), 1)
        self.assertIn("сопоставим в сохранённой выборке", displayed)
        self.assertIn("Блокирующие причины: не обнаружены", displayed)

    def test_legacy_result_shows_existing_coverage_without_inventing_a_diagnostic(self):
        panel = self.open_panel()
        result = result_sections()
        result["coverage"] = {"2020": ["Пробел в календарном покрытии"]}
        before = deepcopy(result)
        panel.render_result(result)
        displayed = panel.diagnostics.get("1.0", "end")
        self.assertIn("2020: Пробел в календарном покрытии", displayed)
        self.assertIn("Подробная оценка сопоставимости отсутствует в сохранённом результате", displayed)
        self.assertNotIn("сопоставим в сохранённой выборке", displayed)
        self.assertEqual(result, before)
