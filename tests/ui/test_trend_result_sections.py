"""The result contract stays useful when the main TOP is empty."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from app.ui.trends_panel import RESULT_BLOCKS
from app.ui.window import Application
from tests.ui.test_desktop import TkCase


def result_sections():
    result = {key: [] for key, _ in RESULT_BLOCKS}
    result.update(schema_version=2, pipeline_version="test", fingerprint="a" * 64,
                  warnings=["Профиль направления не настроен, проверка релевантности ограничена."],
                  coverage={"2020": []}, preparation={"input_occurrences": 100, "retained_studies": 90,
                                                     "possible_versions_merged": 0, "rejected": {}})
    for number, (key, _) in enumerate(RESULT_BLOCKS):
        source = {"id": key, "title": f"Original study {number}", "url": f"https://example.org/{key}",
                  "year": 2017, "evidence_level": "nominal"}
        passage = {"text": "The original evidence stays unchanged.", "url": source["url"], "study_id": key}
        candidate = {"id": key, "title": f"Technology {number}", "study_count": 10, "sources": [source],
                     "limitations": ["Доступна только сохранённая выборка."],
                     "metrics": {"score": 42., "first_observed_year": 2017,
                                 "first_observed_year_in_corpus": 2017, "first_observed_year_in_window": 2020,
                                 "growth_ratio": 2., "years": [{"year": 2020, "documents": 10, "direction_documents": 90}]},
                     "card": {"problem": deepcopy(passage), "advantage": None, "example": deepcopy(passage)},
                     "explanations": {"advantage": "Измеренное преимущество не подтверждено."},
                     "evidence_annotations": {"advantage": {"modality": "future", "reason": "Заявлено перспективное применение."}},
                     "execution": {"evidence_level": "nominal", "direct_documents": 0, "nominal_documents": 10},
                     "rejected_quotes": {"problem": [], "advantage": [{"text": "A future application."}], "example": []},
                     "selection": {"bucket": key, "window_document_share": .1, "established_threshold": .08,
                                   "reasons": ["off_direction" if key == "excluded_off_direction" else "nominal_only"]}}
        result[key].append(candidate)
    return result


class TrendResultSectionTests(TkCase):
    def open_panel(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready)
        return self.app.trends_panel

    def choose(self, panel, key):
        panel.result_block.set(next(label for label, block in panel.block_labels.items() if block == key))
        panel.select_block()

    def test_empty_top_selects_preliminary_and_keeps_source_and_export(self):
        panel = self.open_panel()
        result = result_sections()
        result["candidates"] = []
        panel.render_result(result)
        self.assertEqual(panel.tree.get_children(), ("preliminary_signals",))
        self.assertEqual(panel.current_sources[0]["id"], "preliminary_signals")
        self.assertTrue(panel.link_button.instate(["!disabled"]))
        self.assertTrue(panel.export_button.instate(["!disabled"]))
        self.assertIn("TOP: 0", panel.status.get())
        self.assertIn("добивки до 15 нет", panel.status.get())

    def test_all_sections_show_correct_sources_reasons_and_untranslated_quote(self):
        panel = self.open_panel()
        result = result_sections()
        before = deepcopy(result)
        panel.render_result(result)
        for key, _ in RESULT_BLOCKS:
            self.choose(panel, key)
            self.assertEqual(panel.tree.get_children(), (key,))
            self.assertEqual(panel.current_sources[0]["id"], key)
            detail = panel.detail.get("1.0", "end")
            self.assertIn("The original evidence stays unchanged.", detail)
            self.assertIn("доступном корпусе: 2017", detail)
            self.assertIn("выбранном окне: 2020", detail)
            self.assertIn("Измеренное преимущество не подтверждено.", detail)
            self.assertIn("перспективное применение", detail)
            self.assertIn("тематическое упоминание", detail)
            self.assertIn("Отклонено неподходящих фрагментов: 1", detail)
        self.choose(panel, "excluded_off_direction")
        self.assertIn("Связь с направлением не подтверждена.", panel.detail.get("1.0", "end"))
        self.assertEqual(result, before)

    def test_growth_candidates_are_not_presented_as_verified_emergence(self):
        panel = self.open_panel()
        result = result_sections()
        before = deepcopy(result)
        panel.render_result(result)
        self.choose(panel, "candidates")
        self.assertIn("Кандидаты с ростом", panel.result_block.get())
        self.assertIn("сохранённом корпусе", panel.block_description.get())
        self.assertIn("Новизна и стадия зарождения требуют отдельной проверки", panel.block_description.get())
        self.assertIn("новизна требует отдельной проверки", panel.status.get())
        self.assertIn("не вероятность новизны", panel.diagnostics.get("1.0", "end"))
        self.assertEqual(result, before)

    def test_export_includes_other_sections_and_empty_section_clears_stale_source(self):
        panel = self.open_panel()
        result = result_sections()
        panel.render_result(result)
        self.choose(panel, "established")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "all-results.json"
            with patch("app.ui.trends_panel.filedialog.asksaveasfilename", return_value=str(output)):
                panel.export()
                self.pump(lambda: output.exists() and not panel.busy)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
        result["preliminary_signals"] = []
        self.choose(panel, "preliminary_signals")
        self.assertFalse(panel.current_sources)
        self.assertTrue(panel.link_button.instate(["disabled"]))
        self.assertTrue(panel.export_button.instate(["!disabled"]))

    def test_only_excluded_cards_remain_inspectable(self):
        panel = self.open_panel()
        result = result_sections()
        for key in ("candidates", "preliminary_signals", "established"):
            result[key] = []
        panel.render_result(result)
        self.assertEqual(panel.tree.get_children(), ("excluded_off_direction",))
        self.assertTrue(panel.link_button.instate(["!disabled"]))

    def test_unknown_russian_preparation_uses_original_query_without_guessing_translation(self):
        panel = self.open_panel()
        panel.topic.set("Неизвестная русская технология")
        panel.snapshot_path = "existing-photonic-corpus.json"
        with patch.object(self.controller, "call") as call:
            panel.collect()
        call.assert_called_once()
        self.assertEqual(call.call_args.args[4]["topic"], "Неизвестная русская технология")
        self.assertEqual(panel.snapshot_path, "existing-photonic-corpus.json")
        self.assertEqual(panel.topic.get(), "Неизвестная русская технология")
        self.assertIn("источник получит исходную строку", panel.status.get())
        self.assertTrue(panel.busy)
