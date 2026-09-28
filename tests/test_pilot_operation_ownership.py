"""Deliver real panel callbacks in chosen orders without Tk or background work."""

from types import SimpleNamespace

import pytest

from app.ui.pilot_panel import PilotPanel
from app.ui import pilot_passport, pilot_review


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Control(Value):
    def __init__(self, value=""):
        super().__init__(value)
        self.options = {}
        self.states = set()
        self.running = False

    def configure(self, **options):
        self.options.update(options)

    def pack(self, **options):
        self.options.update(options)
        self.mapped = True

    pack_configure = pack

    def pack_forget(self):
        self.mapped = False

    def state(self, states):
        for state in states:
            if state.startswith("!"):
                self.states.discard(state[1:])
            else:
                self.states.add(state)

    def start(self, _interval):
        self.running = True

    def stop(self):
        self.running = False


class Tree:
    def __init__(self):
        self.rows = {}
        self.selected = ()

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *identifiers):
        for identifier in identifiers:
            del self.rows[identifier]
        self.selected = ()

    def insert(self, _parent, _position, *, iid, values):
        self.rows[iid] = values

    def selection_set(self, identifier):
        self.selected = (identifier,)

    def selection(self):
        return self.selected

    def focus(self, identifier):
        self.focused = identifier


class ResultTabs:
    def __init__(self):
        self.current = 0
        self.titles = {}

    def tab(self, index, **options):
        self.titles[index] = options

    def index(self, _):
        return self.current

    def select(self, index):
        self.current = index


class DeferredController:
    """Use the controller's keyed deduplication and explicit completion contract."""

    def __init__(self):
        self.pending = {}
        self.calls = []

    def call(self, key, method, success, failure, *args, **kwargs):
        if key in self.pending:
            return False
        self.calls.append((method, args, kwargs))
        self.pending[key] = (method, success, failure, args)
        return True

    def arguments(self, method):
        return [args for called, args, _ in self.calls if called == method]

    def complete(self, method, value=None, *, error=None):
        key = next(key for key, call in self.pending.items() if call[0] == method)
        _, success, failure, _ = self.pending.pop(key)
        if error is None:
            success(value)
        else:
            failure(error)


class Window:
    def __init__(self):
        self.exists = True

    def winfo_exists(self):
        return self.exists

    def destroy(self):
        self.exists = False


def payload(identifier, *, view_id=None):
    result = {"result": {"run_id": identifier, "query_plan": {"definition": identifier},
                         "cards": [{"candidate": {"candidate_id": "candidate-" + identifier,
                                                  "label": identifier, "definition": identifier},
                                    "category": "early_signal", "claims": [], "evidence": [],
                                    "limitations": []}], "limitations": []}}
    if view_id:
        result["view_id"] = view_id
    return result


def saved(identifier):
    return {"id": identifier, "payload": payload("original-run", view_id=identifier)}


@pytest.fixture
def panel(monkeypatch):
    panel = object.__new__(PilotPanel)
    panel.settings_form_parent = panel.settings_form = None
    panel.run_id = panel.payload = panel.settings_status = None
    panel.active = panel.loading = panel.pending_cancel = False
    panel._loading_token = None
    panel.loaded = True
    panel.generation = 0
    panel.cards = {}
    panel.tree = Tree()
    panel.other_tree = Tree()
    panel.result_tabs = ResultTabs()
    for name in ("start_button", "cancel_button", "export_button", "passport_button", "documents_button", "queue_button",
                 "query", "english", "progress", "heading", "examples", "execution", "message_label",
                 "manual_note", "result_actions", "setup_action"):
        setattr(panel, name, Control())
    panel.query.set("new question")
    panel.manual = Value(False)
    for name in ("message", "scope", "summary", "direction", "progress_text"):
        setattr(panel, name, Value())
    panel.card_list = SimpleNamespace(clear=lambda: None, render=lambda cards, assessments: None)
    scheduled = []
    panel.app = SimpleNamespace(closing=False, ready=True, child_windows=[], controller=DeferredController(),
                                root=SimpleNamespace(after=lambda delay, callback: scheduled.append((delay, callback))))
    panel.scheduled = scheduled
    monkeypatch.setattr("tkinter.filedialog.askopenfilename", lambda **_: "/synthetic/result.trendresult")
    monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: "/synthetic/export.trendresult")
    return panel


def show(panel, identifier):
    value = payload(identifier)
    assert panel._result(value, expected_id=identifier)
    return value


def start_acknowledged(panel, identifier="active-run"):
    panel.start()
    panel.app.controller.complete("pilot_start", identifier)
    assert panel.active_run_id == identifier


def test_refine_uses_saved_source_and_cancellation_cannot_target_previous_run(panel):
    show(panel, "displayed-source")
    assert panel.refine_candidate("captured-source", "rare-id")
    assert panel.active
    assert panel.app.controller.arguments("pilot_refine_candidate") == [("captured-source", "rare-id")]
    assert not panel.refine_candidate("another-source", "rare-two")
    panel.cancel()
    assert panel.start_cancel.is_set()
    assert not panel.app.controller.arguments("pilot_cancel")
    panel.app.controller.complete("pilot_refine_candidate", "refinement-job")
    assert panel.app.controller.arguments("pilot_cancel") == [("refinement-job",)]
    assert panel.payload is None
    assert not panel.tree.rows and not panel.other_tree.rows


def test_refine_start_failure_keeps_saved_result_reviewable(panel):
    original = show(panel, "saved-source")
    assert panel.refine_candidate("saved-source", "rare-id")
    panel.app.controller.complete("pilot_refine_candidate", error=RuntimeError("failure"))
    assert not panel.active
    assert panel.payload is original
    assert "candidate-saved-source" in panel.other_tree.rows
    assert panel.displayed_id == "saved-source"


def test_signal_metric_text_preserves_unknown_zero_and_conditional_intervals():
    assert pilot_passport.signal_metric_text({"methodology_version": "3.1.0"}) == ()
    missing = " ".join(pilot_passport.signal_metric_text({"methodology_version": "3.2.0", "signal_priority": None}))
    assert "Приоритет проверки гипотезы не определён" in missing
    assert "Рост относительно направления не оценён" in missing
    unavailable = " ".join(pilot_passport.signal_metric_text({"methodology_version": "3.2.0",
        "relative_growth": {"status": "unavailable", "reason": "zero_field_exposure"}}))
    assert "нулевым знаменателем" in unavailable
    observed = " ".join(pilot_passport.signal_metric_text({"methodology_version": "3.2.0", "signal_priority": 0,
        "relative_growth": {"status": "available", "raw_ratio": 0, "smoothed_ratio": 0.3,
                            "ratio_lower_95": 0, "ratio_upper_95": None, "excess_growth_supported": False}}))
    assert "Отношение частот с учётом объёма направления: 0.00" in observed
    assert "Приоритет проверки гипотезы: 0.00/100" in observed
    assert "нижняя граница 0.00" in observed and "без конечной границы" in observed
    assert "не вероятность истинного слабого сигнала" in observed


def test_novelty_text_does_not_present_author_assertions_as_independent_review():
    assert "Авторская гипотеза" in pilot_passport.novelty_prefix({
        "support": "unverified", "grounding_method": "archived-author-novelty/1.0.0"})
    assert "Авторская гипотеза" in pilot_passport.novelty_prefix({
        "support": "supported", "grounding_method": "archived-author-novelty/1.0.0"})
    assert "Экспертная оценка" in pilot_passport.novelty_prefix({
        "support": "supported", "grounding_method": "reviewed-novelty/manual-v1"})
    assert "требует проверки" in pilot_passport.novelty_prefix({
        "support": "unverified", "grounding_method": "reviewed-novelty/manual-v1"})
    assert "опровергнуто" in pilot_passport.novelty_prefix({
        "support": "contradicted", "grounding_method": "archived-author-novelty/1.0.0"})


@pytest.mark.parametrize("failure", [False, True])
def test_late_import_cannot_replace_active_run_or_its_progress_and_cancellation(panel, failure):
    show(panel, "shown-before-start")
    panel.import_result()
    start_acknowledged(panel)
    before = panel.message.get()
    panel.app.controller.complete("pilot_import_result", saved("imported-view"),
                                  error=RuntimeError("old import failed") if failure else None)
    assert panel.run_id == panel.active_run_id == "active-run"
    assert panel.payload is None
    assert panel.message.get() == before
    assert panel.active
    panel.app.controller.complete("pilot_get", {"id": "active-run", "state": "running", "message": "active progress",
                                               "completed": 3, "total": 4})
    assert panel.message.get() == "active progress"
    assert panel.progress.options["value"] == 75
    panel.cancel()
    assert panel.app.controller.arguments("pilot_cancel") == [("active-run",)]


def test_import_before_start_ack_cannot_steal_pending_cancellation(panel):
    panel.import_result()
    panel.start()
    panel.cancel()
    panel.app.controller.complete("pilot_import_result", saved("imported-view"))
    assert panel.active and panel.pending_cancel
    assert panel.run_id is None
    assert panel.app.controller.arguments("pilot_cancel") == []
    panel.app.controller.complete("pilot_start", "actual-run")
    assert panel.app.controller.arguments("pilot_cancel") == [("actual-run",)]
    assert panel.app.controller.arguments("pilot_get") == [("actual-run",)]


def test_duplicate_import_does_not_invalidate_the_accepted_request(panel):
    panel.import_result()
    accepted_generation = panel.generation
    panel.import_result()
    assert panel.generation == accepted_generation
    assert panel.app.controller.arguments("pilot_import_result") == [("/synthetic/result.trendresult",)]
    imported = saved("imported-view")
    panel.app.controller.complete("pilot_import_result", imported)
    assert panel.displayed_id == panel.run_id == "imported-view"
    assert panel.payload is imported["payload"]
    assert panel.tree.rows == {}
    assert panel.other_tree.rows == {"candidate-original-run": ("original-run", "Предварительный кандидат (архив 3.0)", "Не проверено")}


def test_history_load_keeps_export_bound_to_displayed_payload_on_pending_and_failure(panel):
    previous = show(panel, "shown")
    assert panel.open_result("requested")
    assert panel.run_id == panel.displayed_id == "shown"
    assert panel.payload is previous
    panel.export()
    assert panel.app.controller.arguments("pilot_export_result") == [("shown", "/synthetic/export.trendresult")]
    panel.app.controller.complete("pilot_export_result")
    panel.app.controller.complete("pilot_result", error=RuntimeError("load failed"))
    assert panel.run_id == panel.displayed_id == "shown"
    assert panel.payload is previous
    panel.export()
    assert panel.app.controller.arguments("pilot_export_result") == [("shown", "/synthetic/export.trendresult")] * 2


def test_history_publication_updates_identity_and_payload_together(panel):
    previous = show(panel, "shown")
    assert panel.open_result("requested")
    assert not panel.open_result("deduplicated")
    assert panel.payload is previous
    incoming = payload("origin", view_id="requested")
    panel.app.controller.complete("pilot_result", incoming)
    assert panel.payload is incoming
    assert panel.run_id == panel.displayed_id == "requested"
    panel.documents()
    assert panel.app.controller.arguments("pilot_documents") == [("requested",)]


def test_mismatched_history_payload_cannot_change_displayed_identity(panel):
    previous = show(panel, "shown")
    assert panel.open_result("requested")
    panel.app.controller.complete("pilot_result", payload("unexpected"))
    assert panel.payload is previous
    assert panel.run_id == panel.displayed_id == "shown"


def test_late_history_load_cannot_overwrite_a_new_analysis(panel):
    show(panel, "shown")
    assert panel.open_result("requested")
    start_acknowledged(panel)
    panel.app.controller.complete("pilot_result", payload("requested"))
    assert panel.active_run_id == "active-run"
    assert panel.payload is None
    panel.cancel()
    assert panel.app.controller.arguments("pilot_cancel") == [("active-run",)]


def test_export_captures_payload_identity_before_native_dialog_processes_other_events(panel, monkeypatch):
    shown = payload("origin", view_id="shown-view")
    assert panel._result(shown, expected_id="shown-view")

    def save_dialog(**_):
        assert panel.open_result("new-view")
        panel.app.controller.complete("pilot_result", payload("new-view"))
        return "/synthetic/export.trendresult"

    monkeypatch.setattr(pilot_passport.filedialog, "asksaveasfilename", save_dialog)
    pilot_passport.export_payload(panel, shown)
    before = panel.message.get()
    assert panel.displayed_id == "new-view"
    assert panel.app.controller.arguments("pilot_export_result") == [("shown-view", "/synthetic/export.trendresult")]
    panel.app.controller.complete("pilot_export_result")
    assert panel.message.get() == before


def test_import_rechecks_active_analysis_after_native_file_dialog(panel, monkeypatch):
    def open_dialog(**_):
        start_acknowledged(panel)
        return "/synthetic/result.trendresult"

    monkeypatch.setattr("tkinter.filedialog.askopenfilename", open_dialog)
    panel.import_result()
    assert panel.active_run_id == "active-run"
    assert panel.app.controller.arguments("pilot_import_result") == []


def test_old_review_form_cannot_submit_while_new_analysis_is_active(panel):
    show(panel, "source")
    owner = panel.generation
    start_acknowledged(panel)
    feedback, window = Value(), Window()
    assert not pilot_review.submit_review(panel, window, feedback, "old-review", {"reviewer_name": "Expert"},
                                         owner=owner, source_id="source")
    assert panel.app.controller.arguments("pilot_apply_review") == []
    assert "Дождитесь" in feedback.get()
    assert not panel.loading
    assert window.exists
    panel.cancel()
    assert panel.app.controller.arguments("pilot_cancel") == [("active-run",)]


def test_late_review_save_does_not_replace_newer_history_selection(panel):
    show(panel, "source")
    owner = panel.generation
    feedback, window = Value(), Window()
    assert pilot_review.submit_review(panel, window, feedback, "review", {"reviewer_name": "Expert"},
                                      owner=owner, source_id="source")
    assert panel.loading
    panel.start()
    assert panel.app.controller.arguments("pilot_start") == []
    assert panel.open_result("new-history")
    incoming = payload("new-history")
    panel.app.controller.complete("pilot_result", incoming)
    panel.app.controller.complete("pilot_apply_review", saved("review-version"))
    assert panel.payload is incoming
    assert panel.displayed_id == panel.run_id == "new-history"
    assert not panel.loading
    assert not window.exists


def test_review_saved_from_still_current_view_publishes_new_version(panel):
    show(panel, "source")
    feedback, window = Value(), Window()
    assert pilot_review.submit_review(panel, window, feedback, "review", {"reviewer_name": "Expert"},
                                      owner=panel.generation, source_id="source")
    reviewed = saved("review-version")
    panel.app.controller.complete("pilot_apply_review", reviewed)
    assert panel.payload is reviewed["payload"]
    assert panel.displayed_id == panel.run_id == "review-version"
    assert panel.active_run_id is None
    assert not panel.loading
    assert not window.exists


def test_old_review_can_be_saved_after_new_analysis_without_replacing_new_result(panel):
    show(panel, "source")
    owner = panel.generation
    start_acknowledged(panel)
    panel.app.controller.complete("pilot_get", {"id": "active-run", "state": "succeeded", "message": "done",
                                               "completed": 4, "total": 4})
    incoming = payload("active-run")
    panel.app.controller.complete("pilot_result", incoming)
    assert not panel.active
    feedback, window = Value(), Window()
    assert pilot_review.submit_review(panel, window, feedback, "old-review", {"reviewer_name": "Expert"},
                                      owner=owner, source_id="source")
    panel.app.controller.complete("pilot_apply_review", saved("review-version"))
    assert panel.payload is incoming
    assert panel.displayed_id == panel.run_id == "active-run"
    assert not panel.loading
    assert not window.exists


def test_status_refresh_does_not_enable_start_during_review_save(panel):
    token = panel.begin_loading()
    panel._status({"model_installed": True})
    assert "disabled" in panel.start_button.states
    panel.finish_loading(token)
    assert "disabled" not in panel.start_button.states


def test_closing_application_discards_late_import_without_rendering(panel):
    previous = show(panel, "shown")
    panel.import_result()
    before = panel.message.get()
    panel.app.closing = True
    panel.app.controller.complete("pilot_import_result", saved("imported-view"))
    assert panel.payload is previous
    assert panel.displayed_id == "shown"
    assert panel.message.get() == before


def test_passport_actions_keep_the_identity_of_the_passport_payload(panel, monkeypatch):
    from app.ui import pilot_materials

    class Widget(Window):
        def __init__(self, *_, **options):
            super().__init__()
            self.options = options
            if "command" in options:
                buttons[options["text"]] = options["command"]

        def pack(self, **_):
            pass

        def title(self, _):
            pass

        def geometry(self, _):
            pass

    class Viewport(Widget):
        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            self.content = Widget()

    buttons, requests = {}, []
    monkeypatch.setattr(pilot_passport.tk, "Toplevel", Widget)
    monkeypatch.setattr(pilot_passport, "ScrollViewport", Viewport)
    for name in ("Frame", "Label", "Button"):
        monkeypatch.setattr(pilot_passport.ttk, name, Widget)
    monkeypatch.setattr(pilot_materials, "show_matches", lambda _panel, run_id, candidate_id:
                        requests.append((run_id, candidate_id)))
    show(panel, "different-current-result")
    original = payload("origin", view_id="passport-version")
    pilot_passport.show_passport(panel, original["result"]["cards"][0], original)
    buttons["Отчёты и препринты по технологии"]()
    assert requests == [("passport-version", "candidate-origin")]
