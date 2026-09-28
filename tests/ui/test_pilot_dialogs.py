"""Native dialog QA; review decisions and provider responses are explicit test fixtures."""

from datetime import datetime, UTC
from decimal import InvalidOperation
import json
from threading import Event
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

import pytest

from app.pilot.settings import PilotSettings
from app.pilot.sensitivity import SensitivityScenario, evaluate_sensitivity
from app.runtime.jobs import TaskFailure
from app.ui.pilot_budget import budget_dialog, money_micro
from app.ui.pilot_materials import background, import_report_dialog, restore_dialog, sensitivity_dialog, show_library, show_matches
from app.ui.pilot_panel import PilotPanel
from app.ui.pilot_review import start_review, field_review_controls, read_field_reviews
from app.ui.pilot_settings import SettingsDialog
from app.ui.theme import apply_theme
from app.ui.viewport import ScrollViewport
from app.ui.window import Application
from app.ui.controller import create_backend
from tests.test_pilot_antecedents import bundle_for
from tests.test_pilot_evidence import Context, document
from tests.test_pilot_export import make_result
from tests.ui.test_desktop import TkCase
from tests.ui.test_pilot_panel import DeferredController, descendants
from tests.platform_support import require_symlinks


@pytest.mark.parametrize("text,expected", [("0", 0), ("0.000001", 1), ("1,234567", 1234567),
    ("1000000", 10**12), ("0.0000000", 0)])
def test_money_parser_preserves_exact_micro_units(text, expected):
    assert money_micro(text) == expected


@pytest.mark.parametrize("text", ["", "NaN", "Infinity", "-1", "0.0000001", "1e1000000",
    "1e-1000000000", "1000001", "9" * 1000])
def test_money_parser_rejects_extremes_before_overflow_underflow_or_large_integer_work(text):
    with pytest.raises((ValueError, InvalidOperation)):
        money_micro(text)


class PilotDialogTests(TkCase):
    def setUp(self):
        super().setUp()
        apply_theme(self.root)
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True)
        overview = ttk.Frame(tabs)
        tabs.add(overview, text="Overview")
        viewport = ScrollViewport(tabs, padding=16)
        tabs.add(viewport, text="Новый анализ")
        self.fixture_controller = DeferredController()
        self.app = SimpleNamespace(root=self.root, tabs=tabs, pilot_tab=viewport, ready=True,
            closing=False, child_windows=[], controller=self.fixture_controller)
        self.panel = PilotPanel(self.app, viewport.content)
        self.status = {"settings": PilotSettings().model_dump(mode="json"), "model_installed": True,
            "keys": {name: name == "deepseek_api_key" for name in ("deepseek_api_key", "yandex_api_key", "openalex_api_key", "epo_ops_key", "epo_ops_secret")},
            "persistent_keys_available": False, "key_errors": []}
        self.panel._status(self.status)
        self.directory = TemporaryDirectory()
        self.result, self.archive, self.artifacts = make_result(Path(self.directory.name), historical=True)

    def tearDown(self):
        self.app.closing = True
        super().tearDown()
        self.directory.cleanup()

    def button(self, window, text):
        return next(item for item in descendants(window) if isinstance(item, ttk.Button) and item.cget("text") == text)

    def text(self, window):
        text = []
        for item in descendants(window):
            if isinstance(item, ttk.Label):
                variable = str(item.cget("textvariable"))
                text.append(str(item.getvar(variable)) if variable else str(item.cget("text")))
            elif isinstance(item, tk.Text):
                text.append(item.get("1.0", "end-1c"))
            elif isinstance(item, ttk.Entry) and item.instate(["readonly"]):
                text.append(item.get())
        return "\n".join(text)

    def tick(self):
        completed = []
        self.root.after(80, completed.append, True)
        self.pump(lambda: bool(completed))

    def review(self, *, finish=True):
        start_review(self.panel, "source-run", self.result.cards[0].candidate.candidate_id)
        window = self.app.child_windows[-1]
        if not finish:
            return window
        self.fixture_controller.complete("pilot_begin_review", "review-job")
        card = self.result.cards[0]
        bundle, _ = bundle_for((self.archive, None, None, None, card.candidate, None, None),
                                (document(81, year=2010),))
        self.prepared = {"bundle": bundle.model_dump(mode="json"), "card": card.model_dump(mode="json")}
        self.fixture_controller.complete("pilot_review_progress", {"state": "succeeded", "message": "",
            "error": None, "review_data": self.prepared})
        return window

    @staticmethod
    def budget_fixture():
        return {"scope_total": 1, "unknown_total": 1, "reconciliation_required": 1,
            "notice": "Неизвестный ответ удержан до явной сверки владельцем.",
            "scopes": [{"scope_id": "run:one", "currency": "USD", "used": {"cost_micro": 250000},
                        "remaining": {"cost_micro": 750000}, "requires_reconciliation": True}],
            "unknown_requests": [{"request_id": "request-unknown", "currency": "USD", "charged": {"cost_micro": 250000}}],
            "next_scope_after": None, "next_request_after": None}

    def budget(self):
        budget_dialog(self.panel)
        window = self.app.child_windows[-1]
        self.fixture_controller.complete("pilot_budget_status", self.budget_fixture())
        return window

    def inline_panel(self):
        """A panel whose settings and keys live on the settings page, as in the app."""
        page = ScrollViewport(self.app.tabs, padding=16)
        self.app.tabs.add(page, text="Настройки")
        form = ttk.Frame(page.content)
        form.pack(fill="x")
        visited = []
        self.app.navigation = SimpleNamespace(select=visited.append)
        panel = PilotPanel(self.app, self.app.pilot_tab.content, settings_form_parent=form)
        panel._status(self.status)
        return panel, form, visited

    def test_settings_and_keys_are_built_on_the_page_without_opening_a_window(self):
        before = len(self.app.child_windows)
        panel, form, _ = self.inline_panel()
        self.assertIsNotNone(panel.settings_form)
        self.assertTrue(panel.settings_form.embedded)
        self.assertEqual(len(self.app.child_windows), before, "окно не открывается")
        self.assertFalse(hasattr(panel, "settings_button"), "кнопка внутри самих настроек не нужна")
        entries = [widget for widget in descendants(form) if isinstance(widget, ttk.Entry)]
        self.assertTrue(entries, "поля формы действительно лежат на странице настроек")
        self.assertIs(panel.settings_form.save_button.winfo_toplevel(), self.root)

    def test_the_settings_action_goes_to_the_page_instead_of_a_window(self):
        panel, _, visited = self.inline_panel()
        before = len(self.app.child_windows)
        panel.settings()
        self.assertEqual(visited, ["settings"])
        self.assertEqual(len(self.app.child_windows), before)

    def test_a_status_refresh_updates_the_marks_and_keeps_unsaved_input(self):
        panel, _, _ = self.inline_panel()
        form = panel.settings_form
        form.keys["yandex_api_key"].insert(0, "не сохранено")
        form.folder.set("каталог-черновик")
        label = form.key_labels["yandex_api_key"]
        self.assertNotIn("настроен", label.cget("text"))
        panel._status(self.status | {"keys": self.status["keys"] | {"yandex_api_key": True}})
        self.assertIs(panel.settings_form, form, "форма не пересобирается")
        self.assertIn("настроен", label.cget("text"))
        self.assertEqual(form.keys["yandex_api_key"].get(), "не сохранено")
        self.assertEqual(form.folder.get(), "каталог-черновик")

    def test_without_a_page_the_window_still_opens(self):
        """Contexts that have no settings page keep the previous behaviour."""
        before = len(self.app.child_windows)
        self.panel.settings()
        self.assertEqual(len(self.app.child_windows), before + 1)

    def test_settings_masks_keys_uses_explicit_persistence_and_saves_no_placeholder(self):
        dialog = SettingsDialog(self.panel)
        self.assertFalse(dialog.persistent.get())
        self.assertTrue(all(entry.get() == "" and entry.cget("show") == "•" for entry in dialog.keys.values()))
        dialog.save()
        method, args, kwargs = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_configure")
        self.assertEqual(args[1], {})
        self.assertIs(kwargs["persistent"], False)
        self.fixture_controller.complete(method, self.status)
        self.assertTrue(dialog.save_button.instate(["!disabled"]))
        self.assertIn("сохранены", dialog.message.get())

    def test_bundled_settings_offer_local_integrity_check_and_show_native_error(self):
        status = self.status | {"model_origin": "bundled", "model_installed": False,
                                "model_state": "unavailable",
                                "model_error": "Встроенная модель не загружается. Переустановите приложение."}
        self.panel._status(status)
        dialog = SettingsDialog(self.panel)
        self.assertEqual(dialog.model_button.cget("text"), "Проверить встроенную модель")
        self.assertEqual(dialog.model_text.get(), status["model_error"])
        dialog.install()
        self.assertEqual(self.fixture_controller.calls[-1][0], "pilot_install_model")
        self.assertIn("Проверяем встроенную модель", dialog.model_text.get())

    def test_local_model_button_downloads_once_and_draws_the_bytes_as_they_arrive(self):
        dialog = SettingsDialog(self.panel)
        self.assertIn("не установлена", dialog.local_llm_text.get())
        self.assertFalse(dialog.local_llm_bar.winfo_ismapped())
        self.button(dialog.window, dialog.local_llm_button.cget("text")).invoke()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_install_local_llm")
        self.assertTrue(self.panel.loading)
        self.assertTrue(dialog.local_llm_button.instate(["disabled"]))
        progress = args[0]
        progress(450 * 10 ** 6, 1800 * 10 ** 6)
        self.pump(lambda: "МБ" in dialog.local_llm_text.get())
        self.assertEqual(round(float(dialog.local_llm_bar.cget("value"))), 25)
        self.assertIn("450 МБ из 1800 МБ", dialog.local_llm_text.get())
        progress(1800 * 10 ** 6, 1800 * 10 ** 6)
        self.pump(lambda: "контрольные суммы" in dialog.local_llm_text.get())
        self.fixture_controller.complete("pilot_install_local_llm", self.status | {"local_llm_installed": True})
        self.assertIn("установлена", dialog.local_llm_text.get())
        self.assertNotIn("Загрузка", dialog.local_llm_text.get())
        self.assertIsNone(dialog.local_llm_tick)
        self.assertFalse(dialog.local_llm_bar.winfo_ismapped())
        self.assertTrue(dialog.local_llm_button.instate(["!disabled"]))
        self.assertFalse(self.panel.loading)
        self.assertEqual(self.callback_errors, [])

    def test_failed_local_model_download_reports_the_error_and_stops_the_bar(self):
        dialog = SettingsDialog(self.panel)
        dialog.install_local_llm()
        self.fixture_controller.complete("pilot_install_local_llm",
                                         error=TaskFailure("Не удалось загрузить локальную AI-модель."))
        self.assertIn("Не удалось", dialog.local_llm_text.get())
        self.assertIsNone(dialog.local_llm_tick)
        self.assertTrue(dialog.local_llm_button.instate(["!disabled"]))
        self.assertFalse(self.panel.loading)
        self.tick()
        self.assertEqual(self.callback_errors, [])

    def test_closed_settings_window_does_not_break_local_model_completion(self):
        dialog = SettingsDialog(self.panel)
        dialog.install_local_llm()
        dialog.window.destroy()
        self.fixture_controller.complete("pilot_install_local_llm", self.status | {"local_llm_installed": True})
        self.tick()
        self.assertFalse(self.panel.loading)
        self.assertEqual(self.callback_errors, [])

    def test_local_model_line_follows_a_later_status_but_not_during_a_download(self):
        dialog = SettingsDialog(self.panel)
        self.panel._status(self.status | {"local_llm_installed": True})
        self.assertIn("установлена", dialog.local_llm_text.get())
        dialog.install_local_llm()
        self.panel._status(self.status | {"local_llm_installed": False})
        self.assertIn("Загрузка", dialog.local_llm_text.get())
        self.fixture_controller.complete("pilot_install_local_llm", self.status | {"local_llm_installed": True})
        self.assertIn("установлена", dialog.local_llm_text.get())

    def test_settings_invalid_money_never_escapes_into_a_tk_callback_or_submits(self):
        dialog = SettingsDialog(self.panel)
        for amount in ("NaN", "1e1000000", "1e-1000000000", "-1", "0.0000001", "10000000000000000"):
            with self.subTest(amount=amount):
                dialog.run_limit.set(amount)
                dialog.save_button.invoke()
                self.assertEqual(self.callback_errors, [])
                self.assertFalse(any(method == "pilot_configure" for method, _, _ in self.fixture_controller.calls))
                self.assertTrue(dialog.save_button.instate(["!disabled"]))
                self.assertIn("Проверьте", dialog.message.get())

    def test_second_settings_window_does_not_stay_disabled_when_first_save_is_pending(self):
        first, second = SettingsDialog(self.panel), SettingsDialog(self.panel)
        first.save()
        second.save()
        self.assertTrue(first.save_button.instate(["disabled"]))
        self.assertTrue(second.save_button.instate(["!disabled"]))
        self.fixture_controller.complete("pilot_configure", self.status)
        second.save()
        self.assertEqual(sum(method == "pilot_configure" for method, _, _ in self.fixture_controller.calls), 2)

    def test_closed_settings_window_does_not_break_model_completion(self):
        dialog = SettingsDialog(self.panel)
        dialog.install()
        self.assertTrue(self.panel.loading)
        dialog.window.destroy()
        self.fixture_controller.complete("pilot_install_model", self.status)
        self.assertFalse(self.panel.loading)
        self.assertTrue(self.panel.start_button.instate(["!disabled"]))
        self.assertEqual(self.callback_errors, [])

    def test_settings_error_is_visible_without_losing_unsaved_values(self):
        dialog = SettingsDialog(self.panel)
        dialog.run_limit.set("0,125")
        dialog.save()
        self.fixture_controller.complete("pilot_configure", error=TaskFailure("Не удалось сохранить настройки."))
        self.assertEqual(dialog.run_limit.get(), "0,125")
        self.assertIn("Не удалось", dialog.message.get())
        self.assertTrue(dialog.save_button.instate(["!disabled"]))

    def test_key_deletion_is_explicit_and_reports_the_submitted_storage_choice(self):
        dialog = SettingsDialog(self.panel)
        cleanup = next(item for item in descendants(dialog.window)
                       if isinstance(item, ttk.LabelFrame) and item.cget("text") == "Удаление ключа доступа")
        choice = next(item for item in descendants(cleanup) if isinstance(item, ttk.Combobox))
        persistence = next(item for item in descendants(cleanup) if isinstance(item, ttk.Checkbutton))
        delete = self.button(cleanup, "Удалить выбранный ключ")
        self.assertEqual(choice.get(), "")
        self.assertFalse(persistence.instate(["selected"]))
        delete.invoke()
        self.assertFalse(any(method == "pilot_delete_credential" for method, _, _ in self.fixture_controller.calls))
        choice.set("DeepSeek")
        delete.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_delete_credential", ("deepseek_api_key", False)))
        persistence.invoke()  # A later edit must not rewrite which request actually succeeded.
        self.fixture_controller.complete("pilot_delete_credential", self.status)
        self.assertIn("может оставаться", dialog.message.get())
        delete.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_delete_credential", ("deepseek_api_key", True)))
        dialog.window.destroy()
        self.fixture_controller.complete("pilot_delete_credential", self.status)
        self.assertEqual(self.callback_errors, [])

    @staticmethod
    def history_rows():
        return [{"id": f"local-{number}", "state": "succeeded", "input_json": json.dumps({"query": f"Test-only run {number}"}),
                 "created_at": "2026-09-10T12:00:00+00:00", "error": None} for number in range(50)]

    def test_history_page_failure_keeps_offset_and_retry_requests_the_same_page(self):
        self.panel._history(self.history_rows())
        window = self.app.child_windows[-1]
        following = self.button(window, "Далее")
        following.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_list_runs", (50, 50, "local")))
        self.fixture_controller.complete("pilot_list_runs", error=TaskFailure("Test-only page unavailable"))
        following.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_list_runs", (50, 50, "local")))
        window.destroy()
        self.fixture_controller.complete("pilot_list_runs", [])
        self.assertEqual(self.callback_errors, [])

    def test_two_history_windows_own_page_requests_and_display_local_errors(self):
        self.panel._history(self.history_rows())
        first = self.app.child_windows[-1]
        self.panel._history(self.history_rows())
        second = self.app.child_windows[-1]
        self.button(first, "Далее").invoke()
        self.button(second, "Далее").invoke()
        self.assertEqual(sum(method == "pilot_list_runs" for method, _, _ in self.fixture_controller.calls), 2)
        self.fixture_controller.complete("pilot_list_runs", error=TaskFailure("Страница временно недоступна"))
        self.assertIn("Страница временно недоступна", self.text(first))
        self.assertNotIn("Страница временно недоступна", self.text(second))
        self.fixture_controller.complete("pilot_list_runs", [])
        self.assertFalse(self.button(first, "Далее").instate(["disabled"]))
        self.assertTrue(self.button(second, "Далее").instate(["disabled"]))

    def test_history_failed_source_switch_keeps_the_previous_rows_with_their_actual_source(self):
        self.panel._history(self.history_rows())
        window = self.app.child_windows[-1]
        choice = next(item for item in descendants(window) if isinstance(item, ttk.Combobox))
        choice.current(1)
        choice.event_generate("<<ComboboxSelected>>")
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_list_runs", (0, 50, "imported")))
        self.fixture_controller.complete("pilot_list_runs", error=TaskFailure("Test-only imported library unavailable"))
        self.assertEqual(choice.get(), "Локальные запуски")
        tree = next(item for item in descendants(window) if isinstance(item, ttk.Treeview))
        self.assertEqual(tuple(tree.get_children()), tuple(f"local-{number}" for number in range(50)))

    def test_closing_review_before_job_id_arrives_cancels_that_job_without_a_form(self):
        window = self.review(finish=False)
        self.assertTrue(self.panel.loading)
        self.button(window, "Отменить обзор").invoke()
        self.assertFalse(window.winfo_exists())
        self.assertFalse(self.panel.loading)
        self.fixture_controller.complete("pilot_begin_review", "review-job")
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_cancel", ("review-job",)))
        self.assertFalse(any(method == "pilot_apply_review" for method, _, _ in self.fixture_controller.calls))

    def test_late_failure_from_closed_review_does_not_release_new_material_operation(self):
        window = self.review(finish=False)
        self.fixture_controller.complete("pilot_begin_review", "review-old")
        self.button(window, "Отменить обзор").invoke()
        self.assertTrue(background(self.panel, "new-import", "import_arxiv", lambda _: None, "/tmp/test-only.atom"))
        self.assertTrue(self.panel.loading)
        self.fixture_controller.complete("pilot_review_progress", error=TaskFailure("Прежний обзор отменён."))
        self.assertTrue(self.panel.loading)
        self.assertTrue(self.panel.start_button.instate(["disabled"]))
        self.fixture_controller.complete("pilot_import_arxiv", {"documents": 1})
        self.assertFalse(self.panel.loading)

    def test_review_does_not_preselect_identity_conclusion_evidence_or_confirmations(self):
        window = self.review()
        self.assertFalse(self.panel.loading)
        kind = next(item for item in descendants(window) if isinstance(item, ttk.Combobox))
        reviewer = next(item for item in descendants(window) if isinstance(item, ttk.Entry) and not isinstance(item, ttk.Combobox))
        trees = [item for item in descendants(window) if isinstance(item, ttk.Treeview)]
        checkboxes = [item for item in descendants(window) if isinstance(item, ttk.Checkbutton)]
        self.assertEqual(kind.get(), "")
        self.assertEqual(reviewer.get(), "")
        self.assertTrue(all(not tree.selection() for tree in trees))
        self.assertTrue(all(not check.instate(["selected"]) for check in checkboxes))
        self.button(window, "Записать мою оценку и пересчитать статус").invoke()
        self.assertFalse(any(method == "pilot_apply_review" for method, _, _ in self.fixture_controller.calls))
        self.assertIn("Выберите свой вывод", self.text(window))
        self.assertIn("не доказывает мировую новизну", self.text(window))

    def test_review_field_controls_show_archived_context_and_collect_only_explicit_verdict(self):
        window = tk.Toplevel(self.root)
        self.app.child_windows.append(window)
        source = next(item for item in self.result.cards[0].evidence if item.text_field == "abstract")
        doc = self.archive.get(source.revision_id)
        controls = field_review_controls(window, [source.model_dump(mode="json")],
            {source.revision_id: {"title": doc.title, "abstract": doc.abstract}})
        self.assertEqual(read_field_reviews(controls), [])
        item = controls["advantage"]
        item["source"].set(next(iter(item["sources"])))
        self.button(item["source"].master, "Посмотреть архивный контекст").invoke()
        context_window = next(child for child in window.winfo_children() if isinstance(child, tk.Toplevel))
        self.assertIn(doc.abstract, self.text(context_window))
        item["verdict"].set("Подтверждается источником")
        item["text_field"].set("Аннотация")
        item["quote"].insert("1.0", source.quote)
        item["rationale"].insert("1.0", "The reviewer explicitly checked the archived source and candidate mechanism for this field.")
        with self.assertRaises(ValueError):
            read_field_reviews(controls)
        item["context_checked"].set(True)
        reviewed, = read_field_reviews(controls)
        self.assertEqual(reviewed["role"], "advantage")
        self.assertEqual(reviewed["source_evidence_id"], source.evidence_id)
        self.assertEqual(reviewed["quote"], source.quote)
        self.assertEqual(self.callback_errors, [])

    def test_explicit_test_reviewer_submission_contains_only_chosen_evidence_and_assertions(self):
        window = self.review()
        kind = next(item for item in descendants(window) if isinstance(item, ttk.Combobox))
        kind.set("Новая комбинация технологий")
        reviewer = next(item for item in descendants(window) if isinstance(item, ttk.Entry) and not isinstance(item, ttk.Combobox))
        reviewer.insert(0, "Explicit GUI test reviewer")
        texts = [item for item in descendants(window) if isinstance(item, tk.Text)]
        explanations = ("This is an explicit fixture explanation of the selected scientific interpretation.",
                        "The test reviewer compared selected source quotations without a model decision.",
                        "The test reviewer explicitly checked terminology in the supplied fixture sources.")
        for widget, value in zip(texts[:3], explanations, strict=True):
            widget.insert("1.0", value)
        trees = [item for item in descendants(window) if isinstance(item, ttk.Treeview)]
        selected = []
        for tree, items in zip(trees, (self.prepared["card"]["evidence"], self.prepared["bundle"]["evidence"]), strict=True):
            identifier = next(item["evidence_id"] for item in items if item["text_field"] == "abstract")
            tree.selection_set(identifier)
            selected.append(identifier)
        for checkbox in (item for item in descendants(window) if isinstance(item, ttk.Checkbutton)):
            checkbox.invoke()
        self.assertFalse(any(method == "pilot_apply_review" for method, _, _ in self.fixture_controller.calls))
        self.button(window, "Записать мою оценку и пересчитать статус").invoke()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_apply_review")
        self.assertEqual(args[0], "review-job")
        self.assertEqual(args[1]["reviewer_name"], "Explicit GUI test reviewer")
        self.assertEqual(args[1]["kind"], "new_combination")
        self.assertEqual(args[1]["current_evidence_ids"], [selected[0]])
        self.assertEqual(args[1]["earlier_evidence_ids"], [selected[1]])
        self.assertIs(args[1]["earlier_analogues_checked"], True)
        self.assertIs(args[1]["terminology_changes_checked"], True)
        self.assertEqual(args[1]["field_reviews"], [])
        self.assertTrue(self.panel.loading)
        # Execute the actual local review/assessment/journal code on the explicit
        # fixture decision, then return its real library response to the UI.
        from app.pilot.antecedents import AntecedentBundle
        from app.pilot.contracts import AnalysisResult
        from app.pilot.history import assess_snapshot
        from app.pilot.library import ResultLibrary
        from app.pilot.review import ReviewDecision, apply_novelty_review, record_review

        source_card = self.result.cards[0]
        bundle = AntecedentBundle.model_validate(self.prepared["bundle"])
        decision = ReviewDecision(**args[1], reviewed_at=datetime.now(UTC),
            candidate_id=source_card.candidate.candidate_id, admission_rule_hash=source_card.candidate.admission_rule_hash,
            bundle_hash=bundle.bundle_hash)
        expanded, novelty = apply_novelty_review(decision, bundle, source_card, self.archive)
        history = next(item for item in self.result.snapshots if item.purpose == "history")
        artifact, card, _ = assess_snapshot(source_card.candidate, self.result.query_plan, history, self.archive,
            Context(), passport=expanded, verified_novelty=novelty, antecedents=bundle)
        record = record_review(Path(self.directory.name) / "reviews", decision, bundle, source_card=source_card,
            reviewed_card=card, artifact=artifact, historical_snapshot=history, archive=self.archive)
        revised = AnalysisResult.model_validate(self.result.model_dump(mode="python") | {
            "result_id": "explicit-ui-review-fixture", "cards": (card,), "snapshots": (*self.result.snapshots, bundle.snapshot)})
        saved = ResultLibrary(Path(self.directory.name), self.archive).save_result(revised, (artifact,), (record,))
        self.fixture_controller.complete("pilot_apply_review", saved)
        self.assertFalse(window.winfo_exists())
        self.assertFalse(self.panel.loading)
        self.assertEqual(self.panel.run_id, saved["id"])
        self.assertEqual(self.panel.cards[card.candidate.candidate_id]["category"], "confirmed_trend")
        self.assertEqual(self.result.cards[0].category, "insufficient_evidence")

    def test_unknown_review_payload_is_reported_without_callback_crash_or_automatic_decision(self):
        window = self.review(finish=False)
        self.fixture_controller.complete("pilot_begin_review", "review-job")
        with self.assertNoLogs(level="CRITICAL"):
            self.root.after(0, lambda: self.fixture_controller.complete("pilot_review_progress", {
                "state": "succeeded", "message": "", "error": None, "review_data": {"unexpected": True}}))
            self.tick()
        self.assertTrue(window.winfo_exists())
        self.assertFalse(self.panel.loading)
        self.assertFalse(any(method == "pilot_apply_review" for method, _, _ in self.fixture_controller.calls))

    def test_budget_unknown_charge_is_held_and_reconciliation_requires_explicit_values(self):
        window = self.budget()
        trees = [item for item in descendants(window) if isinstance(item, ttk.Treeview)]
        unknown = trees[1]
        self.assertEqual(unknown.item("request-unknown", "values")[-1], "0.25")
        self.assertIn("удержан", self.text(window))
        self.assertFalse(any(method == "pilot_budget_reconcile" for method, _, _ in self.fixture_controller.calls))
        unknown.selection_set("request-unknown")
        self.button(window, "Сверить выбранный запрос").invoke()
        form = self.app.child_windows[-1]
        entries = [item for item in descendants(form) if isinstance(item, ttk.Entry)]
        submit = self.button(form, "Подтвердить сверенные значения")
        submit.invoke()
        for entry, value in zip(entries, ("12", "34", "0,125"), strict=True):
            entry.insert(0, value)
        submit.invoke()
        self.assertFalse(any(method == "pilot_budget_reconcile" for method, _, _ in self.fixture_controller.calls))
        next(item for item in descendants(form) if isinstance(item, ttk.Checkbutton)).invoke()
        submit.invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_budget_reconcile", ("request-unknown", {
            "input_tokens": 12, "output_tokens": 34, "cost_micro": 125000, "confirmed": True})))
        self.fixture_controller.complete("pilot_budget_reconcile", error=TaskFailure("Сверка не завершена."))
        self.assertTrue(form.winfo_exists())
        self.assertEqual(unknown.item("request-unknown", "values")[-1], "0.25")

    def test_budget_invalid_money_does_not_crash_or_submit(self):
        window = self.budget()
        tree = next(item for item in descendants(window) if isinstance(item, ttk.Treeview))
        tree.selection_set("run:one")
        self.button(window, "Сверить выбранный период").invoke()
        form = self.app.child_windows[-1]
        entry = next(item for item in descendants(form) if isinstance(item, ttk.Entry))
        next(item for item in descendants(form) if isinstance(item, ttk.Checkbutton)).invoke()
        for amount in ("1e1000000", "1e-1000000000", "-1", "NaN"):
            entry.delete(0, "end")
            entry.insert(0, amount)
            self.button(form, "Подтвердить сверенные значения").invoke()
            self.assertEqual(self.callback_errors, [])
            self.assertFalse(any(method == "pilot_budget_acknowledge" for method, _, _ in self.fixture_controller.calls))

    def test_finish_restore_requires_separate_confirmation(self):
        window = self.budget()
        self.button(window, "Завершить сверку восстановления").invoke()
        self.assertFalse(any(method == "pilot_budget_acknowledge" for method, _, _ in self.fixture_controller.calls))
        check = next(item for item in descendants(window) if isinstance(item, ttk.Checkbutton))
        check.invoke()
        self.button(window, "Завершить сверку восстановления").invoke()
        self.assertEqual(self.fixture_controller.calls[-1][:2], ("pilot_budget_acknowledge", (None, 0, True)))

    def test_closed_budget_and_material_library_ignore_late_read_results(self):
        budget_dialog(self.panel)
        self.app.child_windows[-1].destroy()
        self.fixture_controller.complete("pilot_budget_status", self.budget_fixture())
        show_library(self.panel)
        self.app.child_windows[-1].destroy()
        self.fixture_controller.complete("pilot_supplemental_list", {"items": [], "total": 0, "offset": 0, "limit": 50})
        self.assertEqual(self.callback_errors, [])

    def test_pdf_import_requires_rights_and_does_not_enable_external_ai(self):
        with patch("app.ui.pilot_materials.filedialog.askopenfilename", return_value="/tmp/public-fixture.pdf"):
            import_report_dialog(self.panel, lambda _: None)
        window = self.app.child_windows[-1]
        entries = [item for item in descendants(window) if isinstance(item, ttk.Entry)]
        for entry, value in zip(entries, ("Public test report", "https://example.org/report.pdf", "2025", "CC BY 4.0"), strict=True):
            entry.delete(0, "end")
            entry.insert(0, value)
        self.button(window, "Импортировать и проверить").invoke()
        self.assertFalse(any(method == "pilot_import_report" for method, _, _ in self.fixture_controller.calls))
        next(item for item in descendants(window) if isinstance(item, ttk.Checkbutton)).invoke()
        self.button(window, "Импортировать и проверить").invoke()
        method, args, _ = self.fixture_controller.calls[-1]
        self.assertEqual(method, "pilot_import_report")
        self.assertIs(args[1]["public_license_allowed"], True)
        self.assertIs(args[1]["external_ai_allowed"], False)
        self.assertTrue(window.winfo_exists())
        self.assertTrue(self.button(window, "Импортировать и проверить").instate(["disabled"]))
        self.fixture_controller.complete("pilot_import_report", {"documents": 1})
        self.assertFalse(window.winfo_exists())

    def test_empty_material_matches_explain_absence_without_changing_candidate_counts(self):
        show_matches(self.panel, "source-run", self.result.cards[0].candidate.candidate_id)
        self.fixture_controller.complete("pilot_supplemental_matches", {"items": [], "limited": False})
        window = self.app.child_windows[-1]
        self.assertIn("Совпадений не найдено", self.text(window))
        self.assertFalse(self.panel.loading)
        self.assertFalse(any(method == "pilot_start" for method, _, _ in self.fixture_controller.calls))

    def test_sensitivity_without_reference_source_displays_unknown_not_zero(self):
        card = self.result.cards[0]
        historical = next(item for item in self.result.snapshots if item.purpose == "history")
        report = evaluate_sensitivity(card.candidate, self.result.query_plan, historical, self.archive, Context(),
            passport=card, scenario=SensitivityScenario(excluded_sources=("openalex",)))
        sensitivity_dialog(self.panel, "source-run", card.candidate.candidate_id)
        window = self.app.child_windows[-1]
        choice = next(item for item in descendants(window) if isinstance(item, ttk.Combobox))
        choice.current(1)
        self.button(window, "Пересчитать по сохранённым данным").invoke()
        self.assertEqual(self.fixture_controller.calls[-1][1][-1], {"excluded_sources": ["openalex"]})
        self.fixture_controller.complete("pilot_sensitivity", report.model_dump(mode="json"))
        text = self.text(window)
        self.assertIn("Это не нулевой рост", text)
        self.assertNotIn("→ 0.00", text)
        self.assertFalse(self.panel.loading)

    def test_restore_cancel_at_either_file_dialog_has_no_side_effect(self):
        before = list(self.fixture_controller.calls)
        with patch("app.ui.pilot_materials.filedialog.askopenfilename", return_value=""), \
                patch("app.ui.pilot_materials.filedialog.askdirectory") as folder:
            restore_dialog(self.panel)
            folder.assert_not_called()
        with patch("app.ui.pilot_materials.filedialog.askopenfilename", return_value="/tmp/test-only.trendbackup"), \
                patch("app.ui.pilot_materials.filedialog.askdirectory", return_value=""):
            restore_dialog(self.panel)
        self.assertEqual(self.fixture_controller.calls, before)
        self.assertFalse(self.panel.loading)

    def begin_restore(self):
        with patch("app.ui.pilot_materials.filedialog.askopenfilename", return_value="/tmp/test-only.trendbackup"), \
                patch("app.ui.pilot_materials.filedialog.askdirectory", return_value="/tmp/test-only-restore"):
            restore_dialog(self.panel)

    def test_pending_restore_prevents_second_restore_and_analysis_failure_preserves_current_result(self):
        payload = {"result": self.result.model_dump(mode="json"), "assessments": [
            item.model_dump(mode="json") for item in self.artifacts]}
        self.panel.run_id = self.result.run_id
        self.panel._result(payload)
        self.begin_restore()
        self.assertTrue(self.panel.loading)
        with patch("app.ui.pilot_materials.filedialog.askopenfilename") as select:
            restore_dialog(self.panel)
            select.assert_not_called()
        self.panel.start()
        self.assertFalse(any(method == "pilot_start" for method, _, _ in self.fixture_controller.calls))
        self.fixture_controller.complete("pilot_restore", error=TaskFailure("Копия повреждена. Прежняя библиотека открыта."))
        self.assertFalse(self.panel.loading)
        self.assertEqual(self.panel.run_id, self.result.run_id)
        self.assertEqual(self.panel.payload, payload)
        self.assertEqual(len(self.panel.other_tree.get_children()), 1)
        self.assertFalse(self.panel.tree.get_children())
        self.assertTrue(self.panel.export_button.instate(["!disabled"]))
        self.assertIn("Прежняя библиотека", self.panel.message.get())

    def test_settings_review_and_budget_primary_controls_are_reachable_at_900_by_650(self):
        self.root.deiconify()
        settings = SettingsDialog(self.panel)
        windows = [(settings.window, (settings.keys["deepseek_api_key"], settings.model_button, settings.save_button))]
        review = self.review()
        windows.append((review, (self.button(review, "Записать мою оценку и пересчитать статус"),)))
        budget = self.budget()
        windows.append((budget, (self.button(budget, "Сверить выбранный период"),
                                 self.button(budget, "Завершить сверку восстановления"))))
        for window, controls in windows:
            with self.subTest(dialog=window.title()):
                window.geometry("900x650")
                window.deiconify()
                self.pump(lambda window=window: (window.winfo_width(), window.winfo_height()) == (900, 650))
                self.assertEqual((window.winfo_width(), window.winfo_height()), (900, 650))
                viewport = next(item for item in descendants(window) if isinstance(item, ScrollViewport))
                for control in controls:
                    self.wait_mapped([viewport.canvas, control])
                    viewport.reveal(control)
                    def reached(control=control, viewport=viewport):
                        left = control.winfo_rootx() - viewport.canvas.winfo_rootx()
                        top = control.winfo_rooty() - viewport.canvas.winfo_rooty()
                        return (viewport.timer is None and left >= 0 and top >= 0
                                and left + control.winfo_width() <= viewport.canvas.winfo_width()
                                and top + control.winfo_height() <= viewport.canvas.winfo_height())
                    self.pump(reached)
                    left = control.winfo_rootx() - viewport.canvas.winfo_rootx()
                    top = control.winfo_rooty() - viewport.canvas.winfo_rooty()
                    self.assertGreaterEqual(left, 0)
                    self.assertGreaterEqual(top, 0)
                    self.assertLessEqual(left + control.winfo_width(), viewport.canvas.winfo_width())
                    self.assertLessEqual(top + control.winfo_height(), viewport.canvas.winfo_height())
        self.assertEqual(self.callback_errors, [])


class PilotProfileStartupTests(TkCase):
    """Real backend startup stays closed on invalid saved library selection."""

    def setUp(self):
        super().setUp()
        self.directory = TemporaryDirectory()
        self.anchor = Path(self.directory.name) / "main2-anchor"
        self.anchor.mkdir()
        self.pointer = self.anchor / "active-library.json"
        # This test must never inspect the user's operating-system credentials.
        self.keys = patch("app.runtime.credentials.CredentialStore.get", lambda *_: None)
        self.keys.start()

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            self.keys.stop()
            self.directory.cleanup()

    def assert_safe_startup_failure(self):
        self.app = Application(self.root, factory=lambda: create_backend(self.anchor))
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready or self.app.retry.winfo_manager() == "pack", timeout=5)
        self.assertFalse(self.app.ready)
        self.assertIsNone(self.controller.backend)
        self.assertEqual(self.app.page_label.get(), "Хранилище не открыто")
        self.assertFalse(self.anchor.joinpath("documents.sqlite3").exists())
        self.assertFalse(self.anchor.joinpath("pilot.sqlite3").exists())
        self.assertEqual(self.callback_errors, [])

    def test_corrupt_pointer_has_safe_error_and_retry_recovers_after_explicit_local_repair(self):
        original = b'{"version": 1, "path": "private-path-secret"'
        self.pointer.write_bytes(original)
        self.assert_safe_startup_failure()
        self.assertEqual(self.pointer.read_bytes(), original)
        self.assertNotIn("private-path-secret", self.app.status.get())
        self.pointer.unlink()  # Explicit test-only equivalent of repairing the selected profile.
        self.app.retry.invoke()
        self.pump(lambda: self.app.ready, timeout=5)
        self.assertTrue(self.anchor.joinpath("documents.sqlite3").is_file())
        self.assertEqual(self.app.retry.winfo_manager(), "")

    def test_missing_selected_library_does_not_become_a_new_empty_library(self):
        target = Path(self.directory.name) / "missing-library"
        self.pointer.write_text(json.dumps({"version": 1, "path": str(target)}))
        self.assert_safe_startup_failure()
        self.assertFalse(target.exists())

    def test_dangling_pointer_is_an_error_instead_of_silent_default_library(self):
        require_symlinks()
        self.pointer.symlink_to(Path(self.directory.name) / "missing-pointer")
        self.assert_safe_startup_failure()
        self.assertTrue(self.pointer.is_symlink())

    def test_successful_real_restore_resets_all_library_views_and_closes_previous_dialogs(self):
        self.app = Application(self.root, factory=lambda: create_backend(self.anchor))
        self.controller = self.app.controller
        self.controller.profile_anchor = self.anchor
        self.pump(lambda: self.app.ready and not self.controller.pending, timeout=5)
        saved, errors = [], []
        self.controller.call("test-backup", "pilot_backup", saved.append, errors.append,
                             str(Path(self.directory.name) / "backups"))
        self.pump(lambda: bool(saved or errors), timeout=5)
        self.assertEqual(errors, [])
        panel = self.app.pilot_panel
        panel.load()
        self.pump(lambda: panel.loaded, timeout=5)
        result, _, artifacts = make_result(Path(self.directory.name) / "fixture", historical=True)
        panel.run_id = result.run_id
        panel._result({"result": result.model_dump(mode="json"), "assessments": [item.model_dump(mode="json") for item in artifacts]})
        old_scope, old_summary, old_generation = panel.scope.get(), panel.summary.get(), panel.generation
        window = tk.Toplevel(self.root)
        self.app.child_windows.append(window)
        self.app.offset, self.app.total = 100, 500
        self.app.query = "old filter"
        self.app.search.insert(0, self.app.query)
        self.app.job_filter, self.app.history_filter = "old-job", "old-history"
        self.app.document_scope = "Old selected collection"
        self.app.sort_by, self.app.descending = "citations", True
        self.app.sort_choice.set("Больше цитирований")
        self.app.selected_document = self.backend.records[0]
        self.app.open_link.state(["!disabled"])
        self.app.history.selected_id = "old-history"
        self.app.history.lookup.insert(0, "old-history")
        self.app.trends_panel.history_id = "old-history"
        self.app.trends_panel.snapshot_path = "/tmp/old-test-only-corpus.json"
        self.app.trends_panel.input_label.set("Old corpus")
        self.app.trends_panel.history_box.set("Old collection")
        with patch("app.ui.pilot_materials.filedialog.askopenfilename", return_value=saved[0]["path"]), \
                patch("app.ui.pilot_materials.filedialog.askdirectory", return_value=str(Path(self.directory.name) / "restored")):
            restore_dialog(panel)
        self.pump(lambda: not panel.loading, timeout=8)
        self.assertIn("Прежние данные сохранены", panel.message.get())
        self.pump(lambda: not self.controller.pending, timeout=5)
        self.assertIsNone(panel.run_id)
        self.assertIsNone(panel.payload)
        self.assertEqual(panel.cards, {})
        self.assertFalse(panel.tree.get_children())
        self.assertFalse(panel.other_tree.get_children())
        for button in (panel.export_button, panel.passport_button, panel.documents_button):
            self.assertTrue(button.instate(["disabled"]))
        self.assertNotEqual(panel.summary.get(), old_summary)
        self.assertNotEqual(panel.scope.get(), old_scope)
        self.assertGreater(panel.generation, old_generation)
        self.assertFalse(window.winfo_exists())
        self.assertEqual(self.app.child_windows, [])
        self.assertEqual((self.app.offset, self.app.total, self.app.query, self.app.search.get()), (0, 0, "", ""))
        self.assertEqual(self.app._displayed_document_view, self.app._document_view())
        self.assertEqual((self.app.job_filter, self.app.history_filter, self.app.selected_document), (None, None, None))
        self.assertEqual((self.app.sort_by, self.app.descending), ("default", False))
        self.assertEqual(self.app.documents, {})
        self.assertTrue(self.app.open_link.instate(["disabled"]))
        self.assertIsNone(self.app.history.selected_id)
        self.assertEqual(self.app.history.lookup.get(), "")
        self.assertIsNone(self.app.trends_panel.history_id)
        self.assertIsNone(self.app.trends_panel.snapshot_path)
        self.assertIsNone(self.app.trends_panel.result)
        self.assertEqual(self.app.trends_panel.history_box.get(), "")
        self.assertEqual(self.callback_errors, [])

    def test_close_during_restore_finishes_real_copy_and_shuts_down_without_late_tk_calls(self):
        from app.profiles import resolve_profile
        from app.runtime.backup import restore_backup as real_restore

        self.app = Application(self.root, factory=lambda: create_backend(self.anchor))
        self.controller = self.app.controller
        self.controller.profile_anchor = self.anchor
        self.pump(lambda: self.app.ready and not self.controller.pending, timeout=5)
        saved, errors = [], []
        self.controller.call("test-backup", "pilot_backup", saved.append, errors.append,
                             str(Path(self.directory.name) / "backups"))
        self.pump(lambda: bool(saved or errors), timeout=5)
        self.assertEqual(errors, [])
        entered, release = Event(), Event()
        def delayed_real_restore(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test failed to release restoration worker")
            return real_restore(*args, **kwargs)
        results = []
        with patch("app.runtime.backup.restore_backup", side_effect=delayed_real_restore):
            self.controller.call("test-restore", "pilot_restore", results.append, errors.append,
                saved[0]["path"], str(Path(self.directory.name) / "restored"))
            try:
                self.pump(entered.is_set, timeout=5)
                self.app.close()
            finally:
                release.set()
            self.pump(lambda: self.controller.stopped, timeout=8)
        self.assertEqual(results, [])  # A closed window receives no completion callback.
        self.assertEqual(errors, [])
        self.assertEqual(self.callback_errors, [])
        selected = resolve_profile(self.anchor)
        self.assertNotEqual(selected, self.anchor)
        self.assertTrue((selected / "documents.sqlite3").is_file())
        self.assertTrue((selected / "pilot.sqlite3").is_file())
        self.assertTrue((self.anchor / "documents.sqlite3").is_file())
