"""The main2 sidebar and cards retain prerequisite operation ownership."""

from types import SimpleNamespace

import pytest

from app.runtime.jobs import TaskCancelled
from app.ui.analysis_history import AnalysisHistory, history_view
from app.ui.window import Application
from tests import test_pilot_operation_ownership as ownership
from tests.test_controller_lanes import test_control_completes_while_bulk_operation_is_blocked as bulk_lane_check


class Control(ownership.Control):
    def pack(self, **options):
        self.options.update(options)
        self.mapped = True

    pack_configure = pack

    def pack_forget(self):
        self.mapped = False


class CardList:
    def __init__(self):
        self.rendered = []

    def clear(self):
        self.rendered.clear()

    def render(self, cards, assessments):
        self.rendered.append((cards, assessments))


@pytest.fixture
def panel(monkeypatch):
    panel = ownership.panel.__wrapped__(monkeypatch)
    for name in ("heading", "examples", "execution", "message_label", "manual_note", "result_actions",
                 "progress", "setup_action"):
        setattr(panel, name, Control())
    panel.direction = ownership.Value()
    panel.progress_text = ownership.Value()
    panel.card_list = CardList()
    panel.navigation_events = []
    panel.app.navigation = SimpleNamespace(select=panel.navigation_events.append)
    return panel


@pytest.fixture
def history(panel):
    view = object.__new__(AnalysisHistory)
    view.panel, view.app = panel, panel.app
    view.loaded = True
    view.pending = view.dirty = False
    view.generation = view.offset = 0
    view.rows = {"saved": {"id": "saved", "state": "succeeded", "input_json": '{"query":"saved question"}'}}
    view.tree = ownership.Tree()
    view.tree.rows = {"saved": ()}
    view.tree.selection_set("saved")
    view.message = ownership.Value()
    view.source = ownership.Value("На этом компьютере")
    view.loaded_source = view.source.get()
    for name in ("selector", "open_button", "resume_button", "previous", "following"):
        setattr(view, name, Control())
    panel.app.analysis_history = view
    return view


def test_inline_history_opens_matching_payload_and_then_navigates(history, panel):
    previous = ownership.show(panel, "previous")
    history.open_selected()
    assert panel.payload is previous and panel.displayed_id == panel.run_id == "previous"
    assert panel.navigation_events == []
    incoming = ownership.payload("saved")
    panel.app.controller.complete("pilot_result", incoming)
    assert panel.payload is incoming
    assert panel.displayed_id == panel.run_id == "saved"
    assert panel.navigation_events == ["analysis"]
    assert panel.card_list.rendered[-1][0] is panel.cards


def test_inline_history_late_result_cannot_replace_new_analysis_or_navigate(history, panel):
    history.open_selected()
    ownership.start_acknowledged(panel)
    panel.app.controller.complete("pilot_result", ownership.payload("saved"))
    assert panel.active_run_id == "active-run"
    assert panel.payload is None
    assert panel.navigation_events == []
    assert panel.card_list.rendered == []


def test_inline_resume_has_new_owner_and_cancellable_admission(history, panel):
    panel.import_result()
    generation = panel.generation
    history.rows["saved"]["state"] = "interrupted"
    history.resume()
    assert panel.generation > generation
    assert panel.active and panel.run_id is None
    assert panel.direction.get() == "saved question"
    assert panel.navigation_events == ["analysis"]
    panel.cancel()
    assert panel.start_cancel.is_set()
    panel.app.controller.complete("pilot_import_result", ownership.saved("imported"))
    assert panel.run_id is None and panel.active
    assert panel.navigation_events == ["analysis"]
    panel.app.controller.complete("pilot_resume", "resumed")
    assert panel.app.controller.arguments("pilot_cancel") == [("resumed",)]


def test_inline_resume_rejected_by_controller_leaves_panel_idle(history, panel):
    history.rows["saved"]["state"] = "interrupted"
    panel.app.controller.call = lambda *_args, **_kwargs: False
    history.resume()
    assert not panel.active
    assert panel.navigation_events == []


def test_inline_history_profile_reset_discards_pending_rows_and_callbacks(history, panel):
    history.request(50)
    assert history.pending
    history.reset()
    assert not history.pending and not history.loaded and history.rows == {}
    panel.app.controller.complete("pilot_history_view", [{"id": "old-profile"}])
    assert not history.loaded and history.rows == {}
    assert "disabled" in history.open_button.states


def test_inline_history_cancelled_count_read_cannot_publish_a_partial_page():
    def cancelled(_identifier):
        raise TaskCancelled()
    service = SimpleNamespace(list_runs=lambda *_: [{"id": "saved", "state": "succeeded"}], result=cancelled)
    with pytest.raises(TaskCancelled):
        history_view(service)


def test_inline_history_read_does_not_block_cancellation_lane():
    bulk_lane_check("pilot_history_view")


def test_compact_sidebar_layout_does_not_require_legacy_global_header_or_repack():
    root = object()
    app = object.__new__(Application)
    app.root, app.display = root, SimpleNamespace(px=lambda value: value)
    app._compact_chrome = False
    Application._window_layout(app, SimpleNamespace(widget=root, height=500))
    assert app._compact_chrome
    Application._window_layout(app, SimpleNamespace(widget=root, height=900))
    assert not app._compact_chrome


def test_profile_transition_disables_sidebar_navigation_and_restores_prior_states():
    class StatefulControl(Control):
        def state(self, states=None):
            if states is None:
                return tuple(self.states)
            return super().state(states)

    class Tabs:
        states = {"analysis": "normal", "settings": "disabled"}

        def tabs(self):
            return tuple(self.states)

        def tab(self, key, option=None, **changes):
            if option == "state":
                return self.states[key]
            self.states[key] = changes["state"]

    app = object.__new__(Application)
    app.tabs = Tabs()
    app.ready, app.closing = True, False
    app._message = lambda _text: None
    buttons = {key: StatefulControl() for key in app.tabs.tabs()}
    buttons["settings"].state(["disabled"])
    app.navigation = SimpleNamespace(buttons=buttons)
    app._profile_transition(True)
    assert not app.ready
    assert all("disabled" in button.states for button in buttons.values())
    app._profile_transition(False)
    assert app.ready
    assert buttons["analysis"].states == set()
    assert buttons["settings"].states == {"disabled"}
    assert app.tabs.states == {"analysis": "normal", "settings": "disabled"}


@pytest.mark.parametrize("directory, expected_reset", [("same", False), ("restored", True)])
def test_opening_replacement_profile_clears_main2_history_and_visible_cards(directory, expected_reset):
    app = object.__new__(Application)
    app._opened_directory = "same"
    resets = []
    app.analysis_history = SimpleNamespace(reset=lambda: resets.append("history"))
    app.pilot_panel = SimpleNamespace(card_list=CardList(), result_actions=Control(), load=lambda: None)
    app.pilot_panel.card_list.render({"old": {}}, {})
    app.pilot_panel.result_actions.pack()
    for name in ("sort_control", "retry", "search_button", "clear_button", "refresh", "start_button", "all_documents"):
        setattr(app, name, Control())
    app.storage = ownership.Value()
    app.refresh_source_credentials = lambda _keys: None
    app._message = lambda _text: None
    app.refresh_documents = app._request_jobs = lambda: None
    app.history = SimpleNamespace(opened=lambda: None)
    app.trends_panel = SimpleNamespace(ready=lambda: None)
    app._opened((directory, [{"id": "epo"}, {"id": "openalex"}]))
    assert bool(resets) == expected_reset
    assert bool(app.pilot_panel.card_list.rendered) != expected_reset
    assert app.pilot_panel.result_actions.mapped != expected_reset
    assert app._opened_directory == directory
