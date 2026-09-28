"""Actual Tk controls exercise optional semantic mode without loading model weights."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from app.ml.contracts import AnalysisInputError
from app.ui.window import Application
from tests.ui.test_desktop import TkCase
from tests.ui.test_trend_result_sections import result_sections


class SemanticModeTests(TkCase):
    def open_panel(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready)
        return self.app.trends_panel

    def test_defaults_are_lexical_and_chooser_is_disabled(self):
        panel = self.open_panel()
        self.assertFalse(panel.semantic_mode.get())
        self.assertEqual(panel.options()["relevance_mode"], "lexical")
        self.assertTrue(panel.model_button.instate(["disabled"]))

    def test_semantic_analysis_forwards_directory_and_restores_controls_on_failure(self):
        panel = self.open_panel()
        panel.snapshot_path = Path("local-corpus.json")
        panel.semantic_check.invoke()
        self.assertTrue(panel.model_button.instate(["!disabled"]))
        with patch("app.ui.trends_panel.filedialog.askdirectory", return_value="/tmp/local model"):
            panel.model_button.invoke()
        with patch("app.ml.service.run_analysis", side_effect=AnalysisInputError(
                "Локальная модель не установлена. python -m scripts.install_ml_model")) as analyze:
            panel.analyze()
            self.assertTrue(panel.semantic_check.instate(["disabled"]))
            self.assertTrue(panel.model_button.instate(["disabled"]))
            self.pump(lambda: not panel.busy)
        self.assertEqual(analyze.call_args.args[1]["relevance_mode"], "semantic")
        self.assertEqual(analyze.call_args.kwargs["model_dir"], Path("/tmp/local model"))
        self.assertIn("scripts.install_ml_model", panel.status.get())
        self.assertTrue(panel.run_button.instate(["!disabled"]))
        self.assertTrue(panel.model_button.instate(["!disabled"]))
        self.assertIsNone(panel.result)

    def test_turning_semantic_off_does_not_forward_stale_model_directory(self):
        panel = self.open_panel()
        panel.snapshot_path = Path("local-corpus.json")
        panel.model_dir.set("/tmp/stale weights")
        with patch("app.ml.service.run_analysis", return_value=result_sections()) as analyze:
            panel.analyze()
            self.pump(lambda: not panel.busy)
        self.assertEqual(analyze.call_args.args[1]["relevance_mode"], "lexical")
        self.assertNotIn("model_dir", analyze.call_args.kwargs)

    def test_semantic_diagnostics_show_uncalibrated_similarity_without_mutating_result(self):
        panel = self.open_panel()
        result = result_sections()
        result["semantic_relevance"] = {
            "mode": "semantic_assisted", "model": {"model_id": "intfloat/e5-small-v2", "revision": "fixed-revision"},
            "threshold": .82, "calibrated": False, "scored_texts": 100,
            "retained_studies": 90, "lexical_studies": 70, "semantic_only_studies": 20,
            "unscored_studies": 5,
        }
        before = deepcopy(result)
        panel.render_result(result)
        text = panel.diagnostics.get("1.0", "end")
        for expected in ("intfloat/e5-small-v2", "fixed-revision", "0.82", "не вероятность",
                         "не откалиброван", "20", "предварительными"):
            self.assertIn(expected, text)
        self.assertEqual(result, before)

    def test_cancelled_directory_choice_preserves_current_location(self):
        panel = self.open_panel()
        panel.semantic_check.invoke()
        before = panel.model_dir.get()
        with patch("app.ui.trends_panel.filedialog.askdirectory", return_value=""):
            panel.model_button.invoke()
        self.assertEqual(panel.model_dir.get(), before)

    def test_missing_dependencies_explain_semantic_lock_and_restore_controls(self):
        panel = self.open_panel()
        panel.semantic_check.invoke()
        panel.failed(ImportError("onnxruntime"))
        self.assertIn("requirements/semantic.lock", panel.status.get())
        self.assertTrue(panel.model_button.instate(["!disabled"]))

    def test_guarded_profile_explains_why_semantic_scores_do_not_expand_admission(self):
        panel = self.open_panel()
        result = result_sections()
        result["semantic_relevance"] = {"mode": "semantic_assisted", "guarded_direction": True,
                                         "model": {"model_id": "intfloat/e5-small-v2", "revision": "fixed"},
                                         "threshold": .82, "calibrated": False}
        panel.render_result(result)
        self.assertIn("допуск сохраняет существующие предметные правила", panel.diagnostics.get("1.0", "end"))

    def test_semantic_preliminary_reason_is_readable_on_card(self):
        panel = self.open_panel()
        result = result_sections()
        result["candidates"] = []
        result["preliminary_signals"][0]["selection"]["reasons"] = ["semantic_scope_requires_review"]
        panel.render_result(result)
        self.assertIn("Смысловая близость требует проверки", panel.detail.get("1.0", "end"))
