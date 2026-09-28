"""Deliver dialog actions and delayed replies without Tk, workers, or real keys."""

from types import SimpleNamespace

import pytest

from app.pilot.settings import PilotSettings
from app.runtime.jobs import TaskFailure
from app.ui import pilot_budget, pilot_materials, pilot_review, pilot_settings
from app.ui.pilot_panel import PilotPanel
from tests.test_pilot_operation_ownership import Control, DeferredController, Value


@pytest.fixture
def ui(monkeypatch):
    widgets = []

    class Widget:
        kind = "widget"

        def __init__(self, master=None, **options):
            self.master, self.options = master, options
            self.exists, self.states = True, set()
            self.value = ""
            self.rows, self.selected = {}, ()
            self.bindings, self.protocols, self.timers = {}, {}, {}
            self.next_timer = 0
            widgets.append(self)

        def pack(self, **_):
            pass

        def pack_forget(self):
            pass

        grid = grid_remove = columnconfigure = rowconfigure = lambda self, *args, **kwargs: None

        def geometry(self, _):
            pass

        title = geometry

        def winfo_exists(self):
            return self.exists

        def winfo_toplevel(self):
            return self if self.kind == "Toplevel" or self.master is None else self.master.winfo_toplevel()

        def destroy(self):
            for child in tuple(widgets):
                if child.master is self and child.exists:
                    child.destroy()
            self.exists = False
            if "<Destroy>" in self.bindings:
                self.bindings["<Destroy>"](SimpleNamespace(widget=self))

        def configure(self, **options):
            self.options.update(options)

        def state(self, values):
            for value in values:
                if value.startswith("!"):
                    self.states.discard(value[1:])
                else:
                    self.states.add(value)

        def get(self, *_):
            variable = self.options.get("textvariable")
            return variable.get() if variable is not None else self.value

        def set(self, value):
            variable = self.options.get("textvariable")
            if variable is not None:
                variable.set(value)
            else:
                self.value = value

        def insert(self, index, value, *, iid=None, values=None):
            if iid is not None:
                self.rows[iid] = values
            else:
                self.set(self.get() + value)

        def delete(self, *values):
            if self.kind == "Treeview":
                for identifier in values:
                    self.rows.pop(identifier, None)
                self.selected = ()
            else:
                self.set("")

        def current(self, index):
            self.set(self.options["values"][index])

        def invoke(self):
            if "disabled" not in self.states:
                return self.options["command"]()

        def bind(self, event, callback, **_):
            self.bindings[event] = callback

        def protocol(self, name, callback):
            self.protocols[name] = callback

        def after(self, delay, callback):
            self.next_timer += 1
            identifier = f"after-{self.next_timer}"
            self.timers[identifier] = (delay, callback)
            return identifier

        def after_cancel(self, identifier):
            del self.timers[identifier]

        def get_children(self):
            return tuple(self.rows)

        def selection(self):
            return self.selected

        def selection_set(self, identifier):
            self.selected = (identifier,)

        heading = column = add = lambda self, *args, **kwargs: None

    class Viewport(Widget):
        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            self.content = Widget(self)

    for name in ("Frame", "Label", "LabelFrame", "Entry", "Button", "Checkbutton", "Combobox", "Treeview",
                 "Notebook", "Progressbar"):
        monkeypatch.setattr(pilot_settings.ttk, name, type(name, (Widget,), {"kind": name}))
    monkeypatch.setattr(pilot_settings.tk, "Toplevel", type("Toplevel", (Widget,), {"kind": "Toplevel"}))
    monkeypatch.setattr(pilot_settings.tk, "StringVar", lambda value="": Value(value))
    monkeypatch.setattr(pilot_settings.tk, "BooleanVar", lambda value=False: Value(value))
    for module in (pilot_settings, pilot_materials, pilot_budget, pilot_review):
        monkeypatch.setattr(module, "ScrollViewport", Viewport)
    for module in (pilot_materials, pilot_review):
        monkeypatch.setattr(module, "ScrolledText", type("Text", (Widget,), {"kind": "Text"}))

    panel = object.__new__(PilotPanel)
    # This panel stands for one without a settings page, so its form stays a window.
    panel.settings_form_parent = panel.settings_form = None
    panel.active = panel.loading = panel.pending_cancel = False
    panel.loaded, panel.generation = True, 0
    panel._loading_token = panel.payload = panel.run_id = None
    panel.start_button, panel.message = Control(), Value()
    panel.app = SimpleNamespace(root=Widget(), closing=False, child_windows=[], controller=DeferredController())
    # The main2 panel owns its setup shortcut even when no modal is open.
    panel.setup_action = Widget(panel.app.root)
    panel.message_label = Widget(panel.app.root)
    status = {"settings": PilotSettings().model_dump(mode="json"), "model_installed": True,
              "keys": dict.fromkeys(("deepseek_api_key", "yandex_api_key", "openalex_api_key", "epo_ops_key", "epo_ops_secret"), False),
              "persistent_keys_available": False, "key_errors": []}
    panel._status(status)

    def children(window, kind=None):
        def belongs(widget):
            current = widget.master
            while current is not None:
                if current is window:
                    return True
                current = current.master
            return False
        return [widget for widget in widgets if belongs(widget) and (kind is None or widget.kind == kind)]

    def button(window, text):
        return next(widget for widget in children(window, "Button") if widget.options.get("text") == text)

    def text(window):
        parts = []
        for widget in children(window):
            if widget.kind == "Label":
                variable = widget.options.get("textvariable")
                parts.append(str(variable.get()) if variable is not None else str(widget.options.get("text", "")))
            elif widget.kind == "Text":
                parts.append(widget.get())
        return "\n".join(parts)

    return SimpleNamespace(panel=panel, controller=panel.app.controller, status=status, widgets=widgets,
                           children=children, button=button, text=text)


def test_settings_delayed_save_preserves_new_key_and_reports_unsaved_changes(ui):
    dialog = pilot_settings.SettingsDialog(ui.panel)
    dialog.keys["deepseek_api_key"].set("synthetic-key-a")
    dialog.save()
    dialog.keys["deepseek_api_key"].set("synthetic-key-b")
    dialog.keys["openalex_api_key"].set("synthetic-new-field")
    dialog.run_limit.set("0.125")
    ui.controller.complete("pilot_configure", ui.status)
    assert dialog.keys["deepseek_api_key"].get() == "synthetic-key-b"
    assert dialog.keys["openalex_api_key"].get() == "synthetic-new-field"
    assert dialog.run_limit.get() == "0.125"
    assert "несохранён" in dialog.message.get().lower()
    assert ui.controller.arguments("pilot_configure")[0][1] == {"deepseek_api_key": "synthetic-key-a"}


def test_settings_success_clears_only_the_submitted_unchanged_key(ui):
    dialog = pilot_settings.SettingsDialog(ui.panel)
    dialog.keys["deepseek_api_key"].set("synthetic-key-a")
    dialog.save()
    dialog.save()
    assert len(ui.controller.arguments("pilot_configure")) == 1
    ui.controller.complete("pilot_configure", ui.status)
    assert dialog.keys["deepseek_api_key"].get() == ""
    assert "несохранён" not in dialog.message.get().lower()


def test_settings_delete_keeps_later_input_and_reports_errors_locally(ui):
    dialog = pilot_settings.SettingsDialog(ui.panel)
    choice = ui.children(dialog.window, "Combobox")[-1]
    choice.set("DeepSeek")
    action = ui.button(dialog.window, "Удалить выбранный ключ")
    action.invoke()
    dialog.keys["deepseek_api_key"].set("synthetic-unsaved-value")
    ui.controller.complete("pilot_delete_credential", ui.status)
    assert dialog.keys["deepseek_api_key"].get() == "synthetic-unsaved-value"
    action.invoke()
    ui.controller.complete("pilot_delete_credential", error=TaskFailure("Удаление не завершено."))
    assert "Удаление не завершено" in dialog.message.get()
    assert "disabled" not in action.states


def test_settings_open_form_cannot_save_or_delete_during_analysis(ui):
    dialog = pilot_settings.SettingsDialog(ui.panel)
    ui.children(dialog.window, "Combobox")[-1].set("DeepSeek")
    ui.panel.active = True
    dialog.save()
    ui.button(dialog.window, "Удалить выбранный ключ").invoke()
    assert ui.controller.arguments("pilot_configure") == []
    assert ui.controller.arguments("pilot_delete_credential") == []
    assert "анализ" in dialog.message.get().lower()


def open_pdf(ui, monkeypatch):
    monkeypatch.setattr(pilot_materials.filedialog, "askopenfilename", lambda **_: "/synthetic/public.pdf")
    results = []
    pilot_materials.import_report_dialog(ui.panel, results.append)
    window = ui.panel.app.child_windows[-1]
    values = ("Public test report", "https://example.org/report.pdf", "2025", "CC BY 4.0")
    for entry, value in zip(ui.children(window, "Entry"), values, strict=True):
        entry.set(value)
    ui.children(window, "Checkbutton")[0].options["variable"].set(True)
    return window, results


def test_pdf_import_preserves_form_until_success_and_retries_with_its_draft(ui, monkeypatch):
    window, results = open_pdf(ui, monkeypatch)
    action = ui.button(window, "Импортировать и проверить")
    action.invoke()
    assert window.winfo_exists()
    assert "disabled" in action.states
    ui.controller.complete("pilot_import_report", error=TaskFailure("PDF не удалось прочитать."))
    assert "PDF не удалось прочитать" in ui.text(window)
    assert ui.children(window, "Entry")[0].get() == "Public test report"
    assert "disabled" not in action.states
    action.invoke()
    ui.controller.complete("pilot_import_report", {"documents": 1})
    assert not window.winfo_exists()
    assert results == [{"documents": 1}]


def test_pdf_submit_rechecks_analysis_started_after_metadata_form_opened(ui, monkeypatch):
    window, _ = open_pdf(ui, monkeypatch)
    ui.panel.active = True
    ui.button(window, "Импортировать и проверить").invoke()
    assert ui.controller.arguments("pilot_import_report") == []
    assert "анализ" in ui.text(window).lower()


def test_sensitivity_labels_the_applied_scenario_after_selection_changes(ui):
    pilot_materials.sensitivity_dialog(ui.panel, "run", "candidate")
    window = ui.panel.app.child_windows[-1]
    choice = ui.children(window, "Combobox")[0]
    choice.current(1)
    requested = choice.get()
    ui.button(window, "Пересчитать по сохранённым данным").invoke()
    choice.current(2)
    ui.controller.complete("pilot_sensitivity", {"baseline": {"assessment": {}}, "status": "unavailable",
                           "excluded_study_ids": [], "limitations": []})
    assert requested in ui.text(window)
    assert ui.controller.arguments("pilot_sensitivity")[0][-1] == {"excluded_sources": ["openalex"]}


def test_sensitivity_error_is_visible_in_its_window_and_retry_is_enabled(ui):
    pilot_materials.sensitivity_dialog(ui.panel, "run", "candidate")
    window = ui.panel.app.child_windows[-1]
    action = ui.button(window, "Пересчитать по сохранённым данным")
    action.invoke()
    ui.controller.complete("pilot_sensitivity", error=TaskFailure("Расчёт недоступен."))
    assert "Расчёт недоступен" in ui.text(window)
    assert "disabled" not in action.states


def test_material_library_windows_load_independently(ui):
    pilot_materials.show_library(ui.panel)
    first = ui.panel.app.child_windows[-1]
    pilot_materials.show_library(ui.panel)
    second = ui.panel.app.child_windows[-1]
    assert len(ui.controller.arguments("pilot_supplemental_list")) == 2
    ui.controller.complete("pilot_supplemental_list", {"items": [], "total": 0, "offset": 0, "limit": 50})
    ui.controller.complete("pilot_supplemental_list", {"items": [], "total": 0, "offset": 0, "limit": 50})
    assert "Сохранено импортов: 0" in ui.text(first)
    assert "Сохранено импортов: 0" in ui.text(second)


def test_material_library_commits_page_after_success_and_keeps_retry_offset(ui):
    pilot_materials.show_library(ui.panel)
    window = ui.panel.app.child_windows[-1]
    page = {"items": [{"id": "one", "title": "Public report", "kind": "report", "documents": 1}],
            "total": 120, "offset": 0, "limit": 50}
    assert ui.controller.arguments("pilot_supplemental_list") == [(0, 50)]
    ui.controller.complete("pilot_supplemental_list", page)
    following = ui.button(window, "Далее")
    following.invoke()
    assert "disabled" in following.states
    ui.controller.complete("pilot_supplemental_list", error=TaskFailure("Следующая страница недоступна."))
    assert tuple(ui.children(window, "Treeview")[0].rows) == ("one",)
    assert "Следующая страница недоступна" in ui.text(window)
    following.invoke()
    assert ui.controller.arguments("pilot_supplemental_list")[-1] == (50, 50)
    ui.controller.complete("pilot_supplemental_list", page | {"offset": 50})
    ui.button(window, "Назад").invoke()
    assert ui.controller.arguments("pilot_supplemental_list")[-1] == (0, 50)


def test_backup_rotation_warning_reaches_the_user(ui, monkeypatch):
    monkeypatch.setattr(pilot_materials.filedialog, "askdirectory", lambda **_: "/synthetic/backups")
    pilot_materials.backup_dialog(ui.panel)
    ui.controller.complete("pilot_backup", {"path": "/synthetic/backups/new.zip", "rotation_warning": True})
    assert "new.zip" in ui.panel.message.get()
    assert "стар" in ui.panel.message.get().lower()


def budget_page(prefix="first", *, following="scope-100"):
    return {"scope_total": 120, "unknown_total": 1, "reconciliation_required": 1, "notice": "Local journal",
            "scopes": [{"scope_id": prefix, "currency": "USD", "used": {"cost_micro": 250000},
                        "remaining": {"cost_micro": 750000}, "requires_reconciliation": True}],
            "unknown_requests": [{"request_id": "unknown", "currency": "USD", "charged": {"cost_micro": 250000}}],
            "next_scope_after": following, "next_request_after": None}


def open_budget(ui):
    pilot_budget.budget_dialog(ui.panel)
    window = ui.panel.app.child_windows[-1]
    ui.controller.complete("pilot_budget_status", budget_page())
    return window


def test_budget_window_read_is_independent_of_another_open_window(ui):
    pilot_budget.budget_dialog(ui.panel)
    first = ui.panel.app.child_windows[-1]
    pilot_budget.budget_dialog(ui.panel)
    second = ui.panel.app.child_windows[-1]
    assert len(ui.controller.arguments("pilot_budget_status")) == 2
    ui.controller.complete("pilot_budget_status", budget_page("one"))
    ui.controller.complete("pilot_budget_status", budget_page("two"))
    assert tuple(ui.children(first, "Treeview")[0].rows) == ("one",)
    assert tuple(ui.children(second, "Treeview")[0].rows) == ("two",)


def test_budget_navigation_has_pending_state_and_commits_only_successful_pages(ui):
    window = open_budget(ui)
    following = ui.button(window, "Следующие периоды")
    reset = ui.button(window, "Обновить с начала")
    following.invoke()
    assert "disabled" in reset.states
    assert "disabled" in following.states
    reset.invoke()
    assert ui.controller.arguments("pilot_budget_status") == [("", ""), ("scope-100", "")]
    ui.controller.complete("pilot_budget_status", error=TaskFailure("Страница недоступна."))
    assert "Страница недоступна" in ui.text(window)
    assert tuple(ui.children(window, "Treeview")[0].rows) == ("first",)
    following.invoke()
    assert ui.controller.arguments("pilot_budget_status")[-1] == ("scope-100", "")
    ui.controller.complete("pilot_budget_status", budget_page("second", following=None))
    reset.invoke()
    assert ui.controller.arguments("pilot_budget_status")[-1] == ("", "")


def test_budget_reconciliation_failure_stays_local_and_does_not_submit_twice(ui):
    window = open_budget(ui)
    ui.children(window, "Treeview")[1].selection_set("unknown")
    ui.button(window, "Сверить выбранный запрос").invoke()
    form = ui.panel.app.child_windows[-1]
    for entry, value in zip(ui.children(form, "Entry"), ("12", "34", "0.125"), strict=True):
        entry.set(value)
    ui.children(form, "Checkbutton")[0].options["variable"].set(True)
    submit = ui.button(form, "Подтвердить сверенные значения")
    submit.invoke()
    assert "disabled" in submit.states
    submit.invoke()
    assert len(ui.controller.arguments("pilot_budget_reconcile")) == 1
    ui.controller.complete("pilot_budget_reconcile", error=TaskFailure("Сверка ещё не завершена."))
    assert "Сверка ещё не завершена" in ui.text(form)
    assert form.winfo_exists()
    assert "disabled" not in submit.states


def test_budget_mutation_refresh_is_not_lost_behind_a_pending_page_read(ui):
    window = open_budget(ui)
    ui.children(window, "Treeview")[1].selection_set("unknown")
    ui.button(window, "Сверить выбранный запрос").invoke()
    form = ui.panel.app.child_windows[-1]
    for entry, value in zip(ui.children(form, "Entry"), ("12", "34", "0.125"), strict=True):
        entry.set(value)
    ui.children(form, "Checkbutton")[0].options["variable"].set(True)
    ui.button(window, "Следующие периоды").invoke()
    ui.button(form, "Подтвердить сверенные значения").invoke()
    ui.controller.complete("pilot_budget_reconcile", {})
    assert len(ui.controller.arguments("pilot_budget_status")) == 2
    ui.controller.complete("pilot_budget_status", budget_page("second", following=None))
    assert ui.controller.arguments("pilot_budget_status") == [("", ""), ("scope-100", ""), ("scope-100", "")]


def test_budget_finish_failure_is_visible_and_reenables_action(ui):
    window = open_budget(ui)
    ui.children(window, "Checkbutton")[0].options["variable"].set(True)
    button = ui.button(window, "Завершить сверку восстановления")
    button.invoke()
    assert "disabled" in button.states
    ui.controller.complete("pilot_budget_acknowledge", error=TaskFailure("Сначала сверьте каждый период."))
    assert "Сначала сверьте каждый период" in ui.text(window)
    assert "disabled" not in button.states


def test_review_save_error_reaches_form_feedback_and_can_be_retried(ui):
    window = pilot_review.tk.Toplevel(ui.panel.app.root)
    feedback = Value()
    assert pilot_review.submit_review(ui.panel, window, feedback, "review", {}, owner=0, source_id="source")
    ui.controller.complete("pilot_apply_review", error=TaskFailure("Укажите проверенные доказательства."))
    assert "Укажите проверенные доказательства" in feedback.get()
    assert not ui.panel.loading
    assert window.winfo_exists()
    assert pilot_review.submit_review(ui.panel, window, feedback, "review", {}, owner=0, source_id="source")


@pytest.mark.parametrize("native_destroy", [False, True])
def test_review_close_cancels_its_scheduled_poll(ui, native_destroy):
    pilot_review.start_review(ui.panel, "source", "candidate")
    window = ui.panel.app.child_windows[-1]
    ui.controller.complete("pilot_begin_review", "review")
    ui.controller.complete("pilot_review_progress", {"state": "running", "message": "Working"})
    assert len(window.timers) == 1
    if native_destroy:
        window.destroy()
    else:
        window.protocols["WM_DELETE_WINDOW"]()
    assert window.timers == {}
    assert not ui.panel.loading


def test_finished_review_destroy_cannot_cancel_finished_job_or_release_new_operation(ui):
    pilot_review.start_review(ui.panel, "source", "candidate")
    window = ui.panel.app.child_windows[-1]
    ui.controller.complete("pilot_begin_review", "review")
    ui.controller.complete("pilot_review_progress", {"state": "failed", "message": "", "error": "Stopped"})
    new_token = ui.panel.begin_loading()
    window.destroy()
    assert ui.panel.loading
    assert ui.panel._loading_token is new_token
    assert ui.controller.arguments("pilot_cancel") == []


def test_review_manual_form_disables_pending_edits_and_restores_after_failure(ui):
    window = pilot_review.tk.Toplevel(ui.panel.app.root)
    frame = pilot_review.ttk.Frame(window)
    message = Value()
    prepared = {"bundle": {"operational_status": "none_found_within_queries", "earliest_observed_year": None,
                           "evidence": []}, "card": {"evidence": []}}
    pilot_review.render_review_form(ui.panel, window, frame, message, prepared, "review", owner=0, source_id="source")
    ui.children(window, "Combobox")[0].set("Новая комбинация технологий")
    action = ui.button(window, "Записать мою оценку и пересчитать статус")
    action.invoke()
    assert "disabled" in action.states
    assert all("disabled" in widget.states for widget in ui.children(window, "Entry"))
    assert all("disabled" in widget.states for widget in ui.children(window, "Combobox"))
    assert all(widget.options["state"] == "disabled" for widget in ui.children(window, "Text"))
    assert all(widget.options["selectmode"] == "none" for widget in ui.children(window, "Treeview"))
    ui.controller.complete("pilot_apply_review", error=TaskFailure("Укажите имя и доказательства."))
    assert "Укажите имя и доказательства" in ui.text(window)
    assert "disabled" not in action.states
    assert all(widget.options["state"] == "normal" for widget in ui.children(window, "Text"))
    assert all(widget.options["selectmode"] == "extended" for widget in ui.children(window, "Treeview"))


def test_manual_field_controls_start_blank_and_require_explicit_context(ui):
    frame = pilot_review.ttk.Frame(ui.panel.app.root)
    evidence = [{"evidence_id": "source-one", "revision_id": "a" * 64,
                 "quote": "An archived research source excerpt.", "source_url": "https://example.org/study"}]
    controls = pilot_review.field_review_controls(frame, evidence, {})
    assert pilot_review.read_field_reviews(controls) == []
    assert all(not items["context_checked"].get() for items in controls.values())
    item = controls["advantage"]
    item["verdict"].set("Подтверждается источником")
    item["source"].set(next(iter(item["sources"])))
    item["text_field"].set("Аннотация")
    item["quote"].set("The measured membrane reduces energy consumption in laboratory experiments.")
    item["rationale"].set("The complete archived source explicitly attributes this measured result to the candidate mechanism.")
    with pytest.raises(ValueError, match="контекста"):
        pilot_review.read_field_reviews(controls)
    item["context_checked"].set(True)
    reviewed, = pilot_review.read_field_reviews(controls)
    assert reviewed["role"] == "advantage" and reviewed["verdict"] == "supported"
    assert reviewed["source_evidence_id"] == "source-one"
    assert reviewed["quote"] == item["quote"].get()


def test_review_evidence_return_opens_exact_selected_quote(ui):
    frame = pilot_review.ttk.Frame(ui.panel.app.root)
    tree = pilot_review.evidence_choices(ui.panel, frame, "Sources", [{"evidence_id": "chosen",
        "quote": "Exact original quote", "source": "openalex", "source_url": "https://openalex.org/W123"}])
    tree.selection_set("chosen")
    assert "<Return>" in tree.bindings
    tree.bindings["<Return>"]()
    assert "Exact original quote" in ui.text(ui.panel.app.child_windows[-1])
