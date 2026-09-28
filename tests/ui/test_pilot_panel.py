"""Independent real-Tk pilot UI QA with a bounded test-only controller fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from threading import Event
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

from app.pilot.archive import DocumentArchive
from app.pilot.contracts import AnalysisResult
from app.pilot.evidence import build_passport
from app.pilot.history import assess_snapshot
from app.pilot.query import QueryError
from app.ui.pilot_panel import PilotPanel
from app.ui.theme import apply_theme
from app.ui.viewport import ScrollViewport
from app.ui.window import Application
from app.runtime.jobs import TaskFailure
from tests.test_pilot_evidence import Context, NOW, candidate, document, query_plan, snapshot
from tests.test_pilot_history import frozen
from tests.ui.test_desktop import TkCase


class DeferredController:
    """Keep real callback ordering and duplicate suppression; never used by the app."""
    def __init__(self):
        self.pending = {}
        self.calls = []

    def call(self, key, method, success, failure, *args, **kwargs):
        if key in self.pending:
            return False
        self.pending[key] = (method, success, failure, args, kwargs)
        self.calls.append((method, args, kwargs))
        return True

    def complete(self, method, result=None, error=None):
        key = next(key for key, entry in self.pending.items() if entry[0] == method)
        _, success, failure, _, _ = self.pending.pop(key)
        (failure if error else success)(error if error else result)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


class PilotPanelTests(TkCase):
    def setUp(self):
        super().setUp()
        apply_theme(self.root)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)
        overview = ttk.Frame(self.tabs)
        self.tabs.add(overview, text="Overview")
        viewport = ScrollViewport(self.tabs, padding=16)
        self.tabs.add(viewport, text="Новый анализ")
        self.fixture_controller = DeferredController()
        self.app = SimpleNamespace(root=self.root, tabs=self.tabs, pilot_tab=viewport,
            ready=True, closing=False, child_windows=[], controller=self.fixture_controller)
        self.panel = PilotPanel(self.app, viewport.content)
        self.panel._status({"model_installed": True})
        self.directory = TemporaryDirectory()
        archive = DocumentArchive(Path(self.directory.name) / "revisions")
        docs = tuple(document(1000 + year * 10 + index, year=year) for year, count in zip(
            range(2020, 2026), (1, 1, 1, 2, 4, 8), strict=True) for index in range(count))
        discovery = snapshot(docs[-4:], archive)
        historical = snapshot(docs, archive, purpose="history")
        item = frozen(candidate(discovery))
        context = Context()
        passport = build_passport(item, discovery, archive, context)
        artifact, card, _ = assess_snapshot(item, query_plan(), historical, archive, context, passport=passport)
        result = AnalysisResult(result_id="ui-result", run_id="run-one", query_plan=query_plan(), created_at=NOW,
            quality=card.quality, snapshots=(discovery, historical), cards=(card,))
        self.payload = {"result": result.model_dump(mode="json"), "assessments": [artifact.model_dump(mode="json")]}

    def tearDown(self):
        self.app.closing = True
        super().tearDown()
        self.directory.cleanup()

    def show(self, width=900, height=650):
        self.root.geometry(f"{width}x{height}")
        self.root.deiconify()
        self.tabs.select(self.app.pilot_tab)
        self.pump(lambda: self.root.winfo_width() == width and self.app.pilot_tab.canvas.winfo_width() > 100)

    def settle_geometry(self):
        finished = []
        self.root.after(80, lambda: finished.append(True))
        self.pump(lambda: bool(finished))

    def load_result(self):
        self.panel.run_id = "run-one"
        self.panel._result(self.payload)

    def submissions(self):
        """Method and positional arguments of every queued call; kwargs vary by run."""
        return [(method, args) for method, args, _ in self.fixture_controller.calls]

    def russian_without_ai(self):
        self.panel._status({"model_installed": True, "settings": {"provider": "deepseek"},
                            "keys": {"deepseek_api_key": False, "yandex_api_key": False}})
        self.panel.query_text.set("Извлечение лития из рассолов")

    def test_russian_direction_starts_on_the_local_translation_without_confirmation(self):
        """No dead end and no second press: the run begins with the translation."""
        self.show()
        self.russian_without_ai()
        self.panel.start()
        self.assertEqual(self.submissions(), [("pilot_translate_query", ("Извлечение лития из рассолов",))])
        self.assertNotIn("неточно", self.panel.message.get(), "предупреждения о переводе больше нет")
        self.fixture_controller.calls.clear()
        self.fixture_controller.complete("pilot_translate_query", {
            "english_query": "Extraction of lithium from brines",
            "model_id": "Xenova/opus-mt-ru-en", "revision": "a" * 40, "reviewed": False})
        self.assertEqual(self.submissions(), [("pilot_start", (
            "Извлечение лития из рассолов", "Extraction of lithium from brines"))])
        self.assertEqual(self.fixture_controller.calls[0][2].get("english_source"), "local_translation",
                         "результат помечает формулировку машинной")
        self.assertFalse(self.panel.manual.get(), "ручное поле не открывается без необходимости")

    def test_the_local_provider_reads_a_russian_direction_without_translating_it(self):
        self.show()
        self.panel._status({"model_installed": True, "settings": {"provider": "local"},
                            "keys": {"deepseek_api_key": False, "yandex_api_key": False}})
        self.panel.query_text.set("Извлечение лития из рассолов")
        self.panel.start()
        self.assertEqual(self.submissions(), [("pilot_start", ("Извлечение лития из рассолов", None))])
        self.assertFalse(self.panel.manual.get())

    def test_edited_direction_is_never_started_with_the_previous_translation(self):
        self.show()
        self.russian_without_ai()
        # A translation of the wording the user has since replaced.
        self.panel._translation = ("Извлечение лития", "Extraction of lithium")
        self.panel.start()
        self.assertEqual(self.submissions(), [("pilot_translate_query", ("Извлечение лития из рассолов",))])

    def test_manual_english_formulation_still_wins_and_stays_the_users_own(self):
        self.show()
        self.russian_without_ai()
        self.panel.manual.set(True)
        self.panel._manual_changed()
        self.panel.english.insert(0, "lithium selective membranes")
        self.panel.start()
        self.assertEqual(self.submissions(),
                         [("pilot_start", ("Извлечение лития из рассолов", "lithium selective membranes"))])
        self.assertEqual(self.fixture_controller.calls[0][2].get("english_source"), "user")

    def test_absent_translation_model_leaves_a_usable_editable_field(self):
        self.show()
        self.russian_without_ai()
        self.panel.start()
        self.fixture_controller.complete("pilot_translate_query", error=TaskFailure(
            "Модель перевода отсутствует или повреждена. "
            "Установите её командой python -m scripts.install_translation_model."))
        self.assertIn("install_translation_model", self.panel.message.get())
        self.assertTrue(self.panel.manual.get())
        self.assertTrue(self.panel.english.instate(["!disabled"]))
        self.assertEqual(self.panel.english.get(), "")
        self.assertTrue(self.panel.start_button.instate(["!disabled"]), "кнопка не остаётся заблокированной")

    def test_english_direction_without_ai_is_submitted_unchanged(self):
        self.show()
        self.panel._status({"model_installed": True, "settings": {"provider": "deepseek"},
                            "keys": {"deepseek_api_key": False}})
        self.panel.query_text.set("lithium selective membranes")
        self.panel.start()
        self.assertFalse(self.panel.manual.get())
        self.assertEqual(self.submissions(), [("pilot_start", ("lithium selective membranes", None))])

    def test_configured_ai_keeps_translating_the_russian_direction(self):
        self.show()
        self.panel._status({"model_installed": True, "settings": {"provider": "deepseek"},
                            "keys": {"deepseek_api_key": True}})
        self.panel.query_text.set("Извлечение лития из рассолов")
        self.panel.start()
        self.assertFalse(self.panel.manual.get(), "с ключом перевод делает AI")
        self.assertEqual(self.submissions(), [("pilot_start", ("Извлечение лития из рассолов", None))])

    def test_unreadable_key_store_leaves_the_decision_to_the_service(self):
        self.show()
        self.panel._status({"model_installed": True, "settings": {"provider": "deepseek"},
                            "keys": {"deepseek_api_key": None}})
        self.panel.query_text.set("Извлечение лития из рассолов")
        self.panel.start()
        self.assertEqual(self.submissions(), [("pilot_start", ("Извлечение лития из рассолов", None))])

    def test_bundled_model_failure_shows_repair_message_without_download_action(self):
        error = "Встроенная модель повреждена. Переустановите приложение."
        self.panel._status({"model_installed": False, "model_origin": "bundled",
                            "model_state": "corrupt", "model_error": error})
        assert self.panel.message.get() == error
        assert not self.panel.setup_action.winfo_manager()
        self.panel.query.insert(0, "Квантовые сенсоры")
        self.panel.start()
        assert self.panel.message.get() == error
        assert not any(method == "pilot_start" for method, _, _ in self.fixture_controller.calls)

    def test_rare_queue_pages_stay_bounded_and_open_the_saved_source(self):
        source = self.payload["result"]["cards"][0]["candidate"]
        self.payload["result"]["candidate_queue"] = [dict(source, candidate_id=f"queued-{index}") for index in range(101)]
        self.load_result()
        self.assertTrue(self.panel.queue_button.instate(["!disabled"]))
        self.panel.queue_button.invoke()
        window = self.app.child_windows[-1]
        tree = next(item for item in descendants(window) if isinstance(item, ttk.Treeview))
        self.assertEqual(len(tree.get_children()), 50)
        self.button(window, "Следующие 50").invoke()
        self.assertEqual(tree.get_children()[0], "queued-50")
        self.button(window, "Следующие 50").invoke()
        self.assertEqual(tree.get_children(), ("queued-100",))
        self.assertTrue(self.button(window, "Следующие 50").instate(["disabled"]))
        tree.selection_set("queued-100")
        self.button(window, "Открыть первоисточник выбранной находки").invoke()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "open_url")
        self.assertEqual(args, ("https://doi.org/" + source["discovery_study_ids"][0][4:],))
        self.assertEqual(self.callback_errors, [])

    def test_top_is_a_separate_ordered_table_and_unknown_cards_remain_reviewable(self):
        source = self.payload["result"]["cards"][0]
        def card(identifier, category):
            return dict(source, candidate=dict(source["candidate"], candidate_id=identifier, label=identifier),
                        category=category, methodology_version="3.2.0")
        self.payload["result"]["cards"] = [card("unknown", "insufficient_evidence"),
            card("emerging", "emerging_candidate"), card("early", "weak_signal_candidate"),
            card("reviewed", "early_signal"), card("established", "established_topic")]
        self.payload["result"]["top_trend_ids"] = ["reviewed", "early", "emerging"]
        self.payload["assessments"] = [dict(assessment={"candidate_id": "early", "recent_studies": 0})]
        self.load_result()
        self.assertEqual(self.panel.tree.get_children(), ("reviewed", "early", "emerging"))
        self.assertEqual(self.panel.other_tree.get_children(), ("unknown", "established"))
        self.assertIn("TOP 2", self.panel.tree.item("early", "values")[1])
        self.assertIn("автоматическая оценка", self.panel.tree.item("early", "values")[1])
        self.assertEqual(str(self.panel.tree.item("early", "values")[2]), "0")
        self.assertEqual(self.panel.tree.item("emerging", "values")[2], "Не проверено")
        self.assertIn("3 из 15", self.panel.summary.get())
        self.panel.result_tabs.select(1)
        self.panel.other_tree.selection_set("unknown")
        with patch("app.ui.pilot_passport.show_passport") as passport:
            self.panel.passport()
            self.assertEqual(passport.call_args.args[1]["candidate"]["candidate_id"], "unknown")

    def test_queue_refinement_starts_one_job_and_opens_its_result(self):
        source = self.payload["result"]["cards"][0]["candidate"]
        self.payload["result"]["candidate_queue"] = [dict(source, candidate_id="rare-candidate")]
        self.load_result()
        self.panel.show_candidate_queue()
        window = self.app.child_windows[-1]
        tree = next(item for item in descendants(window) if isinstance(item, ttk.Treeview))
        refine = self.button(window, "Развить гипотезу")
        self.assertTrue(refine.instate(["disabled"]))
        tree.selection_set("rare-candidate")
        tree.event_generate("<<TreeviewSelect>>")
        refine.invoke()
        self.assertFalse(window.winfo_exists())
        self.assertTrue(self.panel.active)
        self.assertEqual(self.fixture_controller.calls[-1][:2],
                         ("pilot_refine_candidate", ("run-one", "rare-candidate")))
        self.fixture_controller.complete("pilot_refine_candidate", "refined-job")
        self.assertFalse(self.panel.other_tree.get_children())
        self.fixture_controller.complete("pilot_get", {"id": "refined-job", "state": "succeeded", "message": "Готово",
            "completed": 1, "total": 1, "error": None})
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_result", ("refined-job",)))
        revised = {"result": dict(self.payload["result"], run_id="refined-job"), "assessments": self.payload["assessments"]}
        self.fixture_controller.complete("pilot_result", revised)
        self.assertEqual(self.panel.displayed_id, "refined-job")
        self.assertFalse(self.panel.active)
        self.assertEqual(self.callback_errors, [])

    def test_explicit_empty_top_is_not_repopulated_from_other_cards(self):
        source = self.payload["result"]["cards"][0]
        self.payload["result"]["cards"] = [dict(source, category="confirmed_trend")]
        self.payload["result"]["top_trend_ids"] = []
        self.load_result()
        self.assertEqual(self.panel.tree.get_children(), ())
        self.assertEqual(len(self.panel.other_tree.get_children()), 1)
        self.assertIn("Нет кандидатов, допущенных", self.panel.summary.get())
        self.assertTrue(self.panel.passport_button.instate(["disabled"]))
        self.panel.result_tabs.select(1)
        self.panel._selection()
        self.assertTrue(self.panel.passport_button.instate(["!disabled"]))

    def test_unassessed_live_shape_explains_missing_history_in_actual_ui(self):
        source = self.payload["result"]["cards"][0]
        self.payload["result"]["cards"] = [dict(source, category="unassessed_cluster",
            methodology_version="3.2.0", assessment_hash=None, historical_snapshot_id=None,
            candidate=dict(source["candidate"], candidate_id=f"pending-{index}", specificity="uncertain"))
            for index in range(30)]
        self.payload["result"]["top_trend_ids"] = []
        self.payload["assessments"] = []
        self.load_result()
        self.assertFalse(self.panel.tree.get_children())
        self.assertEqual(len(self.panel.other_tree.get_children()), 30)
        self.assertIn("Научная оценка кандидатов не завершена", self.panel.message.get())
        self.assertIn("Исторические оценки: 0 из 30", self.panel.message.get())
        self.assertTrue(all(self.panel.other_tree.item(identifier, "values")[1] == "Непроверенная группа"
                            for identifier in self.panel.other_tree.get_children()))
        self.assertEqual(self.callback_errors, [])

    def test_completed_negative_history_is_visible_without_error_or_padding(self):
        source = self.payload["result"]["cards"][0]
        self.payload["result"]["cards"] = [dict(source, category="established_topic", methodology_version="3.2.0")]
        self.payload["result"]["top_trend_ids"] = []
        self.load_result()
        self.assertFalse(self.panel.tree.get_children())
        self.assertIn("Автоматическая проверка выполнена;", self.panel.message.get())
        self.assertNotIn("не завершена", self.panel.message.get())
        self.assertIn("Известная технология", self.panel.other_tree.item(self.panel.other_tree.get_children()[0], "values")[1])

    def test_empty_cards_with_queue_do_not_claim_there_were_no_relevant_documents(self):
        source = self.payload["result"]["cards"][0]["candidate"]
        self.payload["result"].update(cards=[], top_trend_ids=[], candidate_queue=[source])
        self.payload["assessments"] = []
        self.load_result()
        self.assertIn("Непроверенные находки доступны", self.panel.summary.get())
        self.assertNotIn("недостаточно релевантных документов", self.panel.summary.get())
        self.assertTrue(self.panel.queue_button.instate(["!disabled"]))
        self.assertFalse(self.panel.tree.get_children())

    def test_legacy_unassessed_card_is_not_promoted_to_top(self):
        source = self.payload["result"]["cards"][0]
        self.payload["result"]["cards"] = [dict(source, category="early_signal", methodology_version=None)]
        self.payload["result"].pop("top_trend_ids", None)
        self.load_result()
        self.assertEqual(self.panel.tree.get_children(), ())
        values = self.panel.other_tree.item(self.panel.other_tree.get_children()[0], "values")
        self.assertEqual(values[1], "Предварительный кандидат (архив 3.0)")

    def button(self, window, text):
        return next(item for item in descendants(window) if isinstance(item, ttk.Button) and item.cget("text") == text)

    def test_manual_mode_and_invalid_query_restore_enabled_controls(self):
        self.assertTrue(self.panel.english.instate(["disabled"]))
        self.panel.manual.set(True)
        self.panel._manual_changed()
        self.panel.query.insert(0, "извлечение лития")
        self.panel.english.insert(0, "lithium selective membranes")
        self.panel.start()
        self.assertTrue(self.panel.active)
        self.assertTrue(self.panel.start_button.instate(["disabled"]))
        self.assertTrue(self.panel.query.instate(["disabled"]))
        self.assertEqual(self.fixture_controller.calls[-1][:2],
                         ("pilot_start", ("извлечение лития", "lithium selective membranes")))
        self.fixture_controller.complete("pilot_start", error=QueryError("Уточните технологическое направление."))
        self.assertFalse(self.panel.active)
        self.assertTrue(self.panel.start_button.instate(["!disabled"]))
        self.assertTrue(self.panel.english.instate(["!disabled"]))
        self.assertIn("Уточните", self.panel.message.get())

    def test_duplicate_start_is_bounded_and_cancel_waits_for_terminal_state(self):
        self.panel.query.insert(0, "технологии")
        self.panel.start()
        self.panel.start()
        self.assertEqual(sum(method == "pilot_start" for method, _, _ in self.fixture_controller.calls), 1)
        self.fixture_controller.complete("pilot_start", "run-two")
        self.panel.cancel()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_cancel", ("run-two",)))
        self.fixture_controller.complete("pilot_cancel", True)
        self.assertTrue(self.panel.active)
        self.fixture_controller.complete("pilot_get", {"id": "run-two", "state": "cancelled", "message": "",
            "completed": 0, "total": 0, "error": None})
        self.assertFalse(self.panel.active)
        self.assertIn("Отменён", self.panel.message.get())

    def test_cancel_while_new_run_is_starting_never_targets_previous_result(self):
        self.load_result()
        self.panel.query.insert(0, "Квантовые сенсоры")
        self.panel.start()
        self.panel.cancel()
        self.assertFalse(any(method == "pilot_cancel" and args == ("run-one",)
                             for method, args, _ in self.fixture_controller.calls))
        self.fixture_controller.complete("pilot_start", "run-two")
        self.assertTrue(any(method == "pilot_cancel" and args == ("run-two",)
                            for method, args, _ in self.fixture_controller.calls))

    def test_empty_success_is_explained_and_keeps_export_and_documents_available(self):
        self.load_result()
        empty = {"result": self.payload["result"] | {"cards": [], "quality": "insufficient_data", "limitations": ["Мало данных."]},
                 "assessments": []}
        self.panel._result(empty)
        self.assertFalse(self.panel.tree.get_children())
        self.assertIn("недостаточно", self.panel.summary.get())
        self.assertTrue(self.panel.export_button.instate(["!disabled"]))
        self.assertTrue(self.panel.documents_button.instate(["!disabled"]))
        self.panel._selection()
        self.assertTrue(self.panel.passport_button.instate(["disabled"]))

    def test_unexpected_error_does_not_expose_secrets_and_terminal_failure_unlocks_ui(self):
        self.panel.error(RuntimeError("sk-secret token private-path"), stop=True)
        self.assertNotIn("secret", self.panel.message.get())
        self.assertNotIn("private-path", self.panel.message.get())
        self.panel.run_id = "run-one"
        self.panel._busy(True)
        self.panel._progress({"id": "run-one", "state": "failed", "message": "", "completed": 0,
            "total": 0, "error": "Источник недоступен."})
        self.assertFalse(self.panel.active)
        self.assertEqual(self.panel.message.get(), "Источник недоступен.")

    def test_result_race_rejects_a_response_for_a_different_run(self):
        self.panel.run_id = "run-new"
        self.panel._result(self.payload)
        self.assertIsNone(self.panel.payload)
        self.assertFalse(self.panel.tree.get_children())

    def test_document_window_retains_requested_run_when_response_arrives_after_switch(self):
        self.panel.run_id = "run-one"
        self.panel.documents()
        self.panel.run_id = "run-two"
        self.fixture_controller.complete("pilot_documents", {"total": 120, "items": []})
        window = self.app.child_windows[-1]
        self.button(window, "Далее").invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_documents", ("run-one", 50)))

    def test_failed_document_page_does_not_skip_a_page_on_retry(self):
        self.panel._documents({"total": 120, "items": []}, "run-one")
        window = self.app.child_windows[-1]
        following = self.button(window, "Далее")
        following.invoke()
        self.fixture_controller.complete("pilot_documents", error=OSError("offline"))
        following.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_documents", ("run-one", 50)))

    def test_history_opens_saved_result_and_can_resume_interrupted_work(self):
        rows = [{"id": "run-one", "state": "succeeded", "created_at": NOW.isoformat(),
                 "input_json": '{"payload":{"query":"Литий"}}'},
                {"id": "interrupted", "state": "interrupted", "created_at": NOW.isoformat(),
                 "input_json": '{"payload":{"query":"Квантовые системы"}}'}]
        for identifier, method in (("run-one", "pilot_result"), ("interrupted", "pilot_resume")):
            self.panel._history(rows)
            window = self.app.child_windows[-1]
            tree = next(item for item in descendants(window) if isinstance(item, ttk.Treeview))
            tree.selection_set(identifier)
            self.button(window, "Открыть / продолжить").invoke()
            self.assertEqual(self.fixture_controller.calls[-1][:2], (method, (identifier,)))
            if method == "pilot_result":
                self.fixture_controller.complete(method, self.payload)
                self.assertIsNotNone(self.panel.payload)
            else:
                self.fixture_controller.complete(method, "interrupted")
                self.assertTrue(self.panel.active)

    @staticmethod
    def history_rows(count=3, prefix="keyboard"):
        return [{"id": f"{prefix}-{number}", "state": "succeeded", "created_at": NOW.isoformat(),
                 "input_json": '{"query":"Keyboard-only history fixture"}'} for number in range(count)]

    def press_key(self, widget, key):
        widget.event_generate(f"<KeyPress-{key}>", state=0)
        target = widget if widget.winfo_exists() else self.root
        target.event_generate(f"<KeyRelease-{key}>", state=0)
        self.settle_geometry()

    def show_history(self, rows):
        self.show()
        self.panel._history(rows)
        window = self.app.child_windows[-1]
        tree = next(item for item in descendants(window) if isinstance(item, ttk.Treeview))
        # Tk's focus traversal skips unmapped descendants. A toplevel or a
        # fixed-delay tick alone does not establish keyboard-test readiness.
        self.wait_mapped([window, tree])
        if rows:
            self.pump(lambda: bool(tree.bbox(tree.get_children()[0])))
        window.focus_force()
        self.pump(lambda: self.root.focus_get() is window)
        self.assertIs(window.tk_focusNext(), tree,
                      f"First Tab target differs: window={window.winfo_geometry()}, "
                      f"tree={tree.winfo_geometry()}, mapped={tree.winfo_viewable()}")
        self.press_key(window, "Tab")
        self.pump(lambda: self.root.focus_get() is tree)
        return window, tree

    def test_history_keyboard_selects_rows_and_return_opens_without_mouse(self):
        window, tree = self.show_history(self.history_rows())
        self.assertEqual(tree.selection(), ("keyboard-0",))
        self.assertEqual(tree.focus(), "keyboard-0")
        self.press_key(tree, "Down")
        self.assertEqual(tree.selection(), ("keyboard-1",))
        self.assertEqual(tree.focus(), "keyboard-1")
        self.press_key(tree, "Up")
        self.assertEqual(tree.selection(), ("keyboard-0",))
        self.press_key(tree, "Return")
        self.assertFalse(window.winfo_exists())
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_result", ("keyboard-0",)))

    def test_empty_local_history_switches_to_imported_with_keyboard_selection_and_open(self):
        window, tree = self.show_history([])
        self.assertEqual(tree.selection(), ())
        self.press_key(tree, "Return")
        self.assertTrue(window.winfo_exists())
        self.assertFalse(self.fixture_controller.calls)
        choice = next(item for item in descendants(window) if isinstance(item, ttk.Combobox))
        choice.current(1)
        choice.event_generate("<<ComboboxSelected>>")
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_list_runs", (0, 50, "imported")))
        self.fixture_controller.complete("pilot_list_runs", [row | {"imported": True}
            for row in self.history_rows(prefix="imported")])
        self.assertEqual(tree.focus(), "imported-0")
        self.press_key(tree, "Down")
        self.assertEqual(tree.selection(), ("imported-1",))
        self.press_key(tree, "Return")
        self.assertFalse(window.winfo_exists())
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_result", ("imported-1",)))

    def test_history_failed_page_preserves_keyboard_selection_and_new_page_selects_first(self):
        window, tree = self.show_history(self.history_rows(50))
        for _ in range(2):
            self.press_key(tree, "Down")
        self.assertEqual(tree.selection(), ("keyboard-2",))
        following = self.button(window, "Далее")
        following.invoke()
        self.fixture_controller.complete("pilot_list_runs", error=OSError("Test-only page unavailable"))
        self.assertEqual(tree.selection(), ("keyboard-2",))
        self.assertEqual(tree.focus(), "keyboard-2")
        self.press_key(tree, "Down")
        self.assertEqual(tree.selection(), ("keyboard-3",))
        following.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_list_runs", (50, 50, "local")))
        self.fixture_controller.complete("pilot_list_runs", self.history_rows(prefix="next-page"))
        self.assertEqual(tree.selection(), ("next-page-0",))
        self.assertEqual(tree.focus(), "next-page-0")
        self.press_key(tree, "Down")
        self.assertEqual(tree.selection(), ("next-page-1",))

    def test_passport_uses_actual_counts_original_quotes_and_safe_link_controller(self):
        self.show()
        self.load_result()
        self.panel.result_tabs.select(1)
        self.panel.passport()
        window = self.app.child_windows[-1]
        self.settle_geometry()
        labels = [item for item in descendants(window) if isinstance(item, ttk.Label)]
        text = "\n".join(str(item.cget("text")) for item in labels)
        self.assertIn("14 исследований", text)
        self.assertIn("Existing extraction methods suffer from limited selectivity.", text)
        self.assertIn("Дата первого наблюдения не означает дату изобретения", text)
        link = next(item for item in labels if str(item.cget("text")).startswith("https://"))
        link.event_generate("<Button-1>", x=3, y=3)
        self.pump(lambda: any(method == "open_url" for method, _, _ in self.fixture_controller.calls))
        self.assertEqual(self.fixture_controller.calls[-1][0], "open_url")

    def test_passport_displays_relative_growth_uncertainty_and_zero_priority(self):
        self.payload["assessments"][0]["assessment"].update(methodology_version="3.2.0", signal_priority=0,
            relative_growth={"status": "available", "raw_ratio": None, "smoothed_ratio": 7,
                             "ratio_lower_95": 0, "ratio_upper_95": None, "excess_growth_supported": False})
        self.load_result()
        self.panel.result_tabs.select(1)
        self.panel.passport()
        window = self.app.child_windows[-1]
        text = "\n".join(str(item.cget("text")) for item in descendants(window) if isinstance(item, ttk.Label))
        self.assertIn("Приоритет проверки гипотезы: 0.00/100", text)
        self.assertIn("нижняя граница 0.00", text)
        self.assertIn("без конечной границы", text)
        self.assertIn("не вероятность истинного слабого сигнала", text)
        self.assertIn("Кандидат вне TOP-15", text)
        self.assertNotIn("Приоритет проверки гипотезы не определён", text)

    def test_verified_automatic_result_reaches_top_and_passport_with_archived_counts(self):
        from app.pilot.export import verify_result
        from tests.test_pilot_result_signals import automatic_result

        result, archive, artifact = automatic_result(Path(self.directory.name) / "automatic")
        result = AnalysisResult.model_validate(result.model_dump(mode="python") | {"top_limit": 3})
        verify_result(result, archive, (artifact,))
        self.panel.run_id = result.run_id
        self.panel._result({"result": result.model_dump(mode="json"), "assessments": [artifact.model_dump(mode="json")]})
        identifier = result.cards[0].candidate.candidate_id
        self.assertEqual(self.panel.tree.get_children(), (identifier,))
        self.assertFalse(self.panel.other_tree.get_children())
        row = self.panel.tree.item(identifier, "values")
        self.assertIn("Кандидат в слабые сигналы · автоматическая оценка", row[1])
        self.assertEqual(int(row[2]), artifact.assessment.recent_studies)
        self.assertIn("В TOP-15: 1 из 3", self.panel.summary.get())
        self.assertEqual(self.panel.result_tabs.tab(0, "text"), "TOP-15 · 1 из 3")
        self.panel.passport()
        window = self.app.child_windows[-1]
        text = "\n".join(str(item.cget("text")) for item in descendants(window) if isinstance(item, ttk.Label))
        self.assertIn("Позиция в сохранённом TOP-15: 1", text)
        self.assertIn("Последние 3 полных года: 7 исследований; предыдущие 3 года: 0", text)
        self.assertIn("Первое наблюдение в проверенной выборке: 2023", text)
        self.assertIn(f"Приоритет проверки гипотезы: {artifact.assessment.signal_priority:.2f}/100", text)
        self.assertIn("[Авторская гипотеза · новизна независимо не подтверждена]", text)
        self.assertIn(artifact.inputs.source_novelty[0].novelty.quote, text)
        self.assertNotIn("Приоритет не полностью определён:", text)
        self.assertNotIn("[Экспертная оценка новизны]", text)
        self.assertEqual(self.callback_errors, [])

    def test_controls_remain_reachable_at_900_by_650_using_the_scroll_viewport(self):
        self.show()
        self.load_result()
        self.settle_geometry()
        viewport = self.app.pilot_tab
        for widget in (self.panel.query, self.panel.start_button, self.panel.card_list.frame, self.panel.export_button):
            viewport.reveal(widget)
            self.settle_geometry()
            left = widget.winfo_rootx() - viewport.canvas.winfo_rootx()
            top = widget.winfo_rooty() - viewport.canvas.winfo_rooty()
            self.assertLess(left, viewport.canvas.winfo_width())
            self.assertLess(top, viewport.canvas.winfo_height())
            self.assertGreater(left + widget.winfo_width(), 0)
            self.assertGreater(top + widget.winfo_height(), 0)
        self.assertEqual(self.callback_errors, [])


class PilotDialogShutdownTests(TkCase):
    def test_application_closes_raw_pilot_toplevels_and_original_dialog_lifecycle(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready)
        self.app.pilot_panel._history([])
        raw_window = self.app.child_windows[-1]
        self.assertTrue(raw_window.winfo_exists())
        self.app.close()
        self.pump(lambda: self.controller.stopped)
        self.assertEqual(self.callback_errors, [])


class PilotDialogGeometryTests(TkCase):
    """A callable button is insufficient: every action must fit the visible window."""

    def show_application(self, scale):
        self.app = Application(self.root, lambda: self.backend, ui_scale=scale)
        self.controller = self.app.controller
        # Finish the fixture's initial data load while its root is withdrawn.
        # Native first paint must not compete with an unrelated document-load
        # deadline in tests that measure dialog geometry.
        self.pump(lambda: self.app.ready and not self.app.loading)
        self.root.deiconify()
        self.wait_mapped([self.root])
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root)
        return self.app.pilot_panel

    def settle(self):
        # A mapped toplevel and a single idle callback do not guarantee that
        # native Configure/Map events for its controls have arrived on X11.
        windows = [window for window in self.app.child_windows if window.winfo_exists()]
        self.wait_mapped([widget for window in windows for widget in (window, *descendants(window))])

    def assert_unclipped(self, widget, window):
        context = str({"widget": str(widget), "widget_geometry": widget.winfo_geometry(),
                       "parent_geometry": widget.master.winfo_geometry(),
                       "window_geometry": window.winfo_geometry(), "window_state": window.state(),
                       "window_viewable": window.winfo_viewable()})
        self.assertTrue(widget.winfo_viewable(), context)
        self.assertGreater(widget.winfo_height(), 15, context)
        self.assertGreaterEqual(widget.winfo_width(), widget.winfo_reqwidth(), str(widget))
        current = widget
        while current is not window:
            parent = current.master
            left = widget.winfo_rootx() - parent.winfo_rootx()
            top = widget.winfo_rooty() - parent.winfo_rooty()
            self.assertGreaterEqual(left, 0, str(widget))
            self.assertGreaterEqual(top, 0, str(widget))
            self.assertLessEqual(left + widget.winfo_width(), parent.winfo_width(), str(widget))
            self.assertLessEqual(top + widget.winfo_height(), parent.winfo_height(), str(widget))
            current = parent

    def check_history(self, scale):
        panel = self.show_application(scale)
        rows = [{"id": f"row-{number}", "state": "succeeded", "created_at": NOW.isoformat(),
                 "input_json": '{"query":"Сохранённое направление для проверки размещения"}'} for number in range(50)]
        panel._history(rows)
        window = self.app.child_windows[-1]
        window.geometry("900x440")
        self.settle()
        self.assertEqual((window.winfo_width(), window.winfo_height()), (900, 440))
        self.assert_history_rows_and_controls(window)

    def assert_history_rows_and_controls(self, window):
        for widget in descendants(window):
            if isinstance(widget, (ttk.Button, ttk.Combobox, ttk.Entry)):
                self.assert_unclipped(widget, window)
        tree = next(widget for widget in descendants(window) if isinstance(widget, ttk.Treeview))
        self.assertGreater(tree.winfo_height(), 50)
        first_row = tree.bbox(tree.get_children()[0])
        self.assertTrue(first_row, "The history must show a data row below its heading")
        self.assertLessEqual(first_row[1] + first_row[3], tree.winfo_height())
        scrollbars = [widget for widget in descendants(window) if isinstance(widget, ttk.Scrollbar)]
        self.assertEqual(len(scrollbars), 2)
        self.assertTrue(all(widget.winfo_viewable() for widget in scrollbars))
        self.assertEqual(self.callback_errors, [])

    def test_history_pending_and_errors_keep_a_complete_row_at_200_percent(self):
        from app.runtime.jobs import TaskFailure

        self.check_history(2)
        window = self.app.child_windows[-1]
        deferred = DeferredController()
        following = next(widget for widget in descendants(window)
                         if isinstance(widget, ttk.Button) and widget.cget("text") == "Далее")
        with patch.object(self.app.controller, "call", side_effect=deferred.call):
            following.invoke()
            self.settle()
            self.assert_history_rows_and_controls(window)
            error = "Страница временно недоступна. " * 30
            deferred.complete("pilot_list_runs", error=TaskFailure(error))
            self.settle()
            self.assert_history_rows_and_controls(window)
            status = next(widget for widget in descendants(window)
                          if type(widget) is ttk.Entry and widget.instate(["readonly"]))
            self.assertIn(error, status.get())

    def check_documents(self, scale):
        panel = self.show_application(scale)
        panel._documents({"total": 120, "items": [{"title": "A real-shaped document title", "publication_year": 2025,
            "source": "openalex", "url": "https://openalex.org/W123"}]}, "layout-only-run")
        window = self.app.child_windows[-1]
        window.geometry("900x440")
        self.settle()
        self.assertEqual((window.winfo_width(), window.winfo_height()), (900, 440))
        buttons = [widget for widget in descendants(window) if isinstance(widget, ttk.Button)]
        self.assertEqual(len(buttons), 2)
        for widget in buttons:
            self.assert_unclipped(widget, window)
        box = next(widget for widget in descendants(window) if isinstance(widget, tk.Text))
        self.assertGreater(box.winfo_height(), 50)
        self.assertEqual(self.callback_errors, [])

    def test_history_controls_visible_at_100_percent(self):
        self.check_history(1)

    def test_history_controls_visible_at_150_percent(self):
        self.check_history(1.5)

    def test_history_controls_visible_at_200_percent(self):
        self.check_history(2)

    def check_minimum_history(self, scale):
        self.check_history(scale)
        window = self.app.child_windows[-1]
        width, height = window.minsize()
        available_width, available_height = window.maxsize()
        self.assertLessEqual(width, min(available_width, window.winfo_screenwidth()))
        self.assertLessEqual(height, min(available_height, window.winfo_screenheight()))
        window.geometry(f"{width}x{height}")
        self.pump(lambda: (window.winfo_width(), window.winfo_height()) == (width, height))
        self.settle()
        def ready():
            try:
                self.assert_history_rows_and_controls(window)
            except AssertionError:
                return False
            return True
        try:
            self.pump(ready)
        except AssertionError:
            # Keep the exact clipped-widget/row assertion in failure reports.
            self.assert_history_rows_and_controls(window)
            raise
        self.assert_history_rows_and_controls(window)

    def test_history_minimum_size_at_100_percent(self):
        self.check_minimum_history(1)

    def test_history_minimum_size_at_150_percent(self):
        self.check_minimum_history(1.5)

    def test_history_minimum_size_at_200_percent(self):
        self.check_minimum_history(2)

    def test_document_controls_visible_at_100_percent(self):
        self.check_documents(1)

    def test_document_controls_visible_at_150_percent(self):
        self.check_documents(1.5)

    def test_document_controls_visible_at_200_percent(self):
        self.check_documents(2)

    def check_sensitivity_and_quote(self, scale):
        from app.ui.pilot_materials import sensitivity_dialog
        from app.ui.pilot_review import evidence_choices

        panel = self.show_application(scale)
        sensitivity_dialog(panel, "layout-only-run", "layout-only-candidate")
        sensitivity = self.app.child_windows[-1]
        sensitivity.geometry("850x620")
        self.settle()
        self.assertEqual((sensitivity.winfo_width(), sensitivity.winfo_height()), (850, 620))
        for widget in descendants(sensitivity):
            if isinstance(widget, (ttk.Button, ttk.Combobox)):
                self.assert_unclipped(widget, sensitivity)
        sensitivity.destroy()
        selector = tk.Toplevel(self.root)
        self.app.child_windows.append(selector)
        selector.geometry("900x500")
        frame = ttk.Frame(selector)
        frame.pack(fill="both", expand=True)
        tree = evidence_choices(panel, frame, "Test-only source selection", [{"evidence_id": "layout-only-quote",
            "quote": "Exact source text for layout testing. " * 80, "source": "openalex", "source_url": "https://openalex.org/W123"}])
        self.settle()
        # Mapping only guarantees a native window; Treeview can still have a
        # pending item layout. Click the real, fully visible row once ready.
        tree.see("layout-only-quote")
        self.pump(lambda: bool(tree.bbox("layout-only-quote")))
        x, y, _, height = tree.bbox("layout-only-quote")
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y + height, tree.winfo_height())
        x, y = x + 20, y + height // 2
        count = len(self.app.child_windows)
        for _ in range(2):
            tree.event_generate("<ButtonPress-1>", x=x, y=y)
            tree.event_generate("<ButtonRelease-1>", x=x, y=y)
            self.settle()
        self.pump(lambda: len(self.app.child_windows) > count)
        quote = self.app.child_windows[-1]
        quote.geometry("820x470")
        self.settle()
        self.assertEqual((quote.winfo_width(), quote.winfo_height()), (820, 470))
        action = next(widget for widget in descendants(quote) if isinstance(widget, ttk.Button))
        self.assert_unclipped(action, quote)
        self.assertEqual(action.cget("text"), "Открыть источник")
        self.assertEqual(self.callback_errors, [])

    def test_sensitivity_and_quote_controls_visible_at_100_percent(self):
        self.check_sensitivity_and_quote(1)

    def test_sensitivity_and_quote_controls_visible_at_150_percent(self):
        self.check_sensitivity_and_quote(1.5)

    def test_sensitivity_and_quote_controls_visible_at_200_percent(self):
        self.check_sensitivity_and_quote(2)


class PilotNativeLifecycleTests(TkCase):
    def setUp(self):
        super().setUp()
        from app.ui.controller import create_backend

        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)
        credentials = patch("app.runtime.credentials.CredentialStore.get", lambda *_: None)
        credentials.start()
        self.addCleanup(credentials.stop)
        self.original_quit_command = self.root.tk.call("info", "commands", "tk::mac::Quit")
        self.app = Application(self.root, factory=lambda: create_backend(self.data_dir), ui_scale=1)
        self.controller = self.app.controller
        self.root.deiconify()
        self.pump(lambda: self.app.ready)
        self.app.tabs.select(self.app.pilot_tab)
        self.pump(lambda: self.app.pilot_panel.loaded)

    def test_real_settings_page_carries_the_keys_form_and_opens_no_window(self):
        """Settings and keys belong to the settings page in the running application."""
        panel = self.app.pilot_panel
        self.pump(lambda: panel.settings_form is not None)
        self.assertTrue(panel.settings_form.embedded)
        form = self.app.pilot_settings_form
        self.assertIs(panel.settings_form.save_button.winfo_toplevel(), self.root)
        self.assertIn(panel.settings_form.save_button, list(descendants(form)))
        self.assertIn(panel.settings_form.model_button, list(descendants(form)))
        self.assertTrue(any(isinstance(item, ttk.Entry) and item.cget("show") == "•"
                            for item in descendants(form)), "поля ключей скрыты и лежат на странице")
        before = list(self.app.child_windows)
        panel.settings()
        self.pump(lambda: str(self.app.tabs.select()) == str(self.app.sources_tab))
        self.assertEqual(self.app.child_windows, before, "отдельное окно настроек не открывается")
        self.assertFalse(hasattr(panel, "settings_button"))
        self.assertEqual(self.callback_errors, [])

    def test_native_quit_cancels_coordinator_and_closes_backend_before_destroy(self):
        if self.app.display.system != "aqua":
            self.assertEqual(self.root.tk.call("info", "commands", "tk::mac::Quit"), self.original_quit_command)
            return
        from app.sqlite_runtime import sqlite3

        entered = Event()
        pilot, backend = self.controller.pilot, self.controller.backend
        def waiting_test_processor(context, _payload):
            context.checkpoint("quit_fixture", {"retained": True})
            entered.set()
            if not context.cancel_event.wait(5):
                raise TimeoutError("Test-only processor was not cancelled")
            context.check_cancelled()
        pilot.coordinator._processor = waiting_test_processor
        run_id = pilot.coordinator.submit({"test_only": "native quit lifecycle"})
        self.pump(entered.is_set)
        self.root.tk.call("tk::mac::Quit")
        self.assertTrue(self.app.closing)
        self.pump(lambda: self.controller.stopped, timeout=8)
        self.assertTrue(pilot.coordinator._closed)
        self.assertTrue(backend._closed)
        self.assertTrue(pilot.model_cancel.is_set())
        self.assertTrue(pilot.view_cancel.is_set())
        with sqlite3.connect(self.data_dir / "pilot.sqlite3") as database:
            self.assertEqual(database.execute("SELECT state FROM analysis_runs WHERE id=?", (run_id,)).fetchone(), ("cancelled",))
            self.assertEqual(database.execute("SELECT count(*) FROM analysis_checkpoints WHERE run_id=?", (run_id,)).fetchone(), (1,))
        self.assertEqual(self.callback_errors, [])

    def test_visible_start_button_mouse_press_release_runs_real_empty_query_validation(self):
        panel = self.app.pilot_panel
        viewport = self.app.pilot_tab
        button = panel.start_button
        viewport.reveal(button)
        # A hit test is only meaningful while this window is the top one:
        # winfo_containing reports whichever window the desktop has above it.
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.addCleanup(self.drop_topmost)
        settled = []
        self.root.after(100, lambda: settled.append(True))
        self.pump(lambda: bool(settled))
        x, y = button.winfo_width() // 2, button.winfo_height() // 2
        self.assertIs(self.root.winfo_containing(button.winfo_rootx() + x, button.winfo_rooty() + y), button)
        generation = panel.generation
        with patch("httpx.Client.send", side_effect=AssertionError("Empty query must not reach the network")):
            button.event_generate("<ButtonPress-1>", x=x, y=y)
            button.event_generate("<ButtonRelease-1>", x=x, y=y)
            self.pump(lambda: "Введите направление" in panel.message.get() and not panel.active)
        self.assertEqual(panel.generation, generation)
        self.assertEqual(panel.query.get(), "")
        self.assertIsNone(panel.run_id)
        self.assertTrue(panel.start_button.instate(["!disabled"]))
        self.assertEqual(self.controller.pilot.coordinator.list_runs(), [])
        self.assertIn("направлен", panel.message.get().casefold())
        self.assertEqual(self.callback_errors, [])

    def test_credential_presence_refreshes_existing_source_controls_without_restart(self):
        panel = self.app.pilot_panel
        status = dict(panel.settings_status)
        keys = dict(status["keys"])
        def present(openalex, epo_key, epo_secret):
            panel._status(status | {"keys": keys | {"openalex_api_key": openalex,
                "epo_ops_key": epo_key, "epo_ops_secret": epo_secret}})
        present(False, False, False)
        self.assertTrue(self.app.source_checks["epo"].instate(["disabled"]))
        self.assertFalse(self.app.source_vars["epo"].get())
        present(True, True, True)
        self.assertTrue(self.app.source_checks["epo"].instate(["!disabled"]))
        self.assertFalse(self.app.source_vars["epo"].get())  # Presence does not opt the user into a source.
        self.assertIn("API-ключ настроен", self.app.source_descriptions["openalex"].get())
        self.assertIn("Ключи настроены", self.app.source_descriptions["epo"].get())
        self.app.source_vars["epo"].set(True)
        present(False, True, False)
        self.assertTrue(self.app.source_checks["epo"].instate(["disabled"]))
        self.assertFalse(self.app.source_vars["epo"].get())
        self.assertIn("не настроен", self.app.source_descriptions["openalex"].get())
        present(None, None, True)
        self.assertTrue(self.app.source_checks["epo"].instate(["disabled"]))
        self.assertIn("недоступно", self.app.source_descriptions["epo"].get())
        self.assertIn("недоступно", self.app.source_descriptions["openalex"].get())
        # Reopening an enabled library uses the same presence transition.
        self.app._opened((str(self.data_dir), ({"id": "crossref"},
            {"id": "openalex", "key_configured": True}, {"id": "epo", "credentials_configured": True})))
        self.assertTrue(self.app.source_checks["epo"].instate(["!disabled"]))
        self.assertEqual(self.callback_errors, [])
