"""Native interaction coverage for the persistent sidebar and analysis views."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.ui.window import Application
from app.ui.analysis_history import history_view
from tests.ui.test_desktop import TkCase
from tests.ui.test_pilot_panel import DeferredController, descendants
from tkinter import ttk


def saved_payload():
    return {"result": {"run_id": "saved", "query_plan": {"original_query": "Сохранённый запрос",
        "definition": "Конкретный механизм"}, "limitations": ["Неполная история публикаций"],
        "cards": [{"candidate": {"candidate_id": "one", "label": "Пример технологии", "definition": "Описание подтверждено публикацией"},
            "category": "early_signal", "claims": [{"role": "case", "support": "supported", "text": "Получен экспериментальный результат"}],
            "evidence": [{"source_url": "https://example.org/paper"}], "limitations": ["Новизна требует проверки"]}]},
        "assessments": [{"assessment": {"candidate_id": "one", "recent_studies": 2, "gate_failures": []}}]}


class SidebarAnalysisTests(TkCase):
    def build(self, scale=1):
        self.fixture = DeferredController()
        with patch("app.ui.window.Controller", return_value=self.fixture):
            self.app = Application(self.root, ui_scale=scale)
        self.app.ready = True
        self.panel = self.app.pilot_panel
        self.panel._status({"model_installed": True})
        self.root.deiconify()
        self.root.minsize(640, 480)
        self.root.geometry("900x650")
        # Mapping the window is the window manager's work; wait for it instead
        # of assuming a fixed pause is long enough on a loaded machine.
        self.pump(lambda: self.panel.query.winfo_viewable()
                  and self.panel.start_button.winfo_viewable())
        self.settle()

    def settle(self):
        until = time.monotonic() + .08
        self.pump(lambda: time.monotonic() >= until)

    def test_hidden_pages_stay_reachable_and_mark_their_location(self):
        """A page kept out of the rail must still be openable and still show where you are."""
        self.build()
        rail = self.app.navigation
        self.assertEqual(set(rail.hidden), {"collection", "periods", "ml"})
        for key in ("analysis", "analyses", "documents", "jobs", "settings"):
            self.assertNotIn(key, rail.hidden)
            self.assertTrue(rail.rows[key].winfo_manager(), key)
        # Hidden items occupy no space until their page becomes the current one.
        for key in rail.hidden:
            self.assertFalse(rail.rows[key].winfo_manager(), key)
        entry_points = [widget for widget in descendants(self.app.sources_tab)
                        if isinstance(widget, ttk.Button)
                        and str(widget.cget("text")) in {"Разовый сбор", "Сбор по периодам", "Технический ML"}]
        self.assertEqual(len(entry_points), 3)
        for button in entry_points:
            button.invoke()
            self.settle()
            current = self.app.tabs.select()
            key = next(name for name, page in rail.pages.items() if str(page) == current)
            self.assertIn(key, rail.hidden)
            self.assertTrue(rail.rows[key].winfo_manager(), key)
            # The rail is collapsed at this window size, so either selected style counts.
            self.assertIn(rail.buttons[key].cget("style"),
                          ("Selected.Nav.TButton", "Selected.Compact.Nav.TButton"))
        # Leaving the page puts the rail back to its five permanent items.
        rail.select("analysis")
        self.settle()
        for key in rail.hidden:
            self.assertFalse(rail.rows[key].winfo_manager(), key)

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.closing = True
        super().tearDown()

    def test_default_screen_has_one_input_and_native_pages_keep_state(self):
        self.build()
        self.assertEqual(self.app.tabs.select(), str(self.app.pilot_tab))
        self.assertEqual(self.app.style.layout("Pages.TNotebook.Tab"), [("null", {"sticky": "nswe"})])
        inputs = [w for w in descendants(self.app.pilot_tab) if isinstance(w, ttk.Entry) and w.winfo_ismapped()]
        self.assertEqual(inputs, [self.panel.query])
        self.panel.query.insert(0, "Черновик направления")
        self.app.fields["topic"].insert(0, "Сохранённый сбор")
        children = tuple(self.app.tabs.winfo_children())
        # Isolate navigation from the independent periodic job/history poll,
        # whose timer may otherwise fire while the Periods page is selected.
        self.root.after_cancel(self.app.job_timer)
        calls = len(self.fixture.calls)
        for _ in range(4):
            for key in ("documents", "collection", "jobs", "periods", "ml", "settings", "analysis"):
                self.app.navigation.select(key)
                self.settle()
        self.assertEqual(tuple(self.app.tabs.winfo_children()), children)
        self.assertEqual(self.panel.query.get(), "Черновик направления")
        self.assertEqual(self.app.fields["topic"].get(), "Сохранённый сбор")
        self.assertFalse(any(method in {"pilot_start", "pilot_history_view", "list_history", "list_documents"}
                             for method, _, _ in self.fixture.calls[calls:]))
        self.assertEqual(self.callback_errors, [])

    def test_enter_duplicate_cancel_and_progress_without_estimates(self):
        self.build()
        self.panel.query.insert(0, "Квантовые сенсоры")
        self.panel.query.focus_force()
        self.settle()
        self.panel.query.event_generate("<Return>")
        self.panel.query.event_generate("<Return>")
        self.settle()
        self.assertEqual(sum(method == "pilot_start" for method, _, _ in self.fixture.calls), 1)
        self.assertFalse(self.panel.progress.winfo_ismapped())
        self.fixture.complete("pilot_start", "run")
        self.fixture.complete("pilot_get", {"id": "run", "state": "running", "message": "Проверяем источники", "completed": 2, "total": 5})
        self.assertEqual(self.panel.progress_text.get(), "В текущем этапе: 2 из 5")
        self.assertEqual(float(self.panel.progress.cget("value")), 40)
        self.panel._progress({"id": "run", "state": "running", "message": "Группировка", "completed": 0, "total": 0})
        self.assertEqual(self.panel.progress_text.get(), "")
        self.assertFalse(self.panel.progress.winfo_ismapped())
        self.panel.cancel_button.invoke()
        self.assertTrue(self.panel.active)
        self.fixture.complete("pilot_cancel", True)
        self.panel._progress({"id": "run", "state": "cancelled", "message": "", "completed": 0, "total": 0, "error": None})
        self.assertFalse(self.panel.active)
        self.assertIn("Отменён", self.panel.message.get())

    def test_native_click_and_keyboard_navigation_stays_responsive(self):
        from statistics import median
        self.build()
        # A hit test is only meaningful while this window is the top one:
        # winfo_containing reports whichever window the desktop has above it.
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.addCleanup(self.drop_topmost)
        self.root.focus_force()
        self.settle()
        timings = []
        for _ in range(3):
            for key in ("documents", "analyses", "analysis"):
                button = self.app.navigation.buttons[key]
                self.app.navigation.body.reveal(button)
                self.settle()
                x, y = button.winfo_width() // 2, button.winfo_height() // 2
                self.assertIs(self.root.winfo_containing(button.winfo_rootx() + x, button.winfo_rooty() + y), button)
                started = time.monotonic()
                button.event_generate("<ButtonPress-1>", x=x, y=y)
                button.event_generate("<ButtonRelease-1>", x=x, y=y)
                tick = []
                self.root.after(20, lambda tick=tick: tick.append(True))
                self.pump(lambda tick=tick, key=key: bool(tick) and self.app.tabs.select() == str(self.app.navigation.pages[key]))
                timings.append(1000 * (time.monotonic() - started))
        button = self.app.navigation.buttons["settings"]
        button.focus_force()
        self.settle()
        button.event_generate("<KeyPress-space>")
        button.event_generate("<KeyRelease-space>")
        self.pump(lambda: self.app.tabs.select() == str(self.app.sources_tab))
        print(f"Sidebar navigation, including 20 ms event-loop tick: median={median(timings):.1f} ms, max={max(timings):.1f} ms")
        self.assertLess(max(timings), 1000)
        self.assertEqual(self.callback_errors, [])

    def test_missing_model_errors_and_clarification_remain_actionable(self):
        self.build()
        self.panel._status({"model_installed": False})
        self.panel.query_text.set("Литий")
        self.panel.start()
        self.settle()
        self.assertFalse(self.panel.active)
        self.assertTrue(self.panel.setup_action.winfo_viewable())
        self.assertFalse(any(method == "pilot_start" for method, _, _ in self.fixture.calls))
        self.panel._status({"model_installed": True})
        self.panel.start()
        self.fixture.complete("pilot_start", error=RuntimeError("private-key"))
        self.assertNotIn("private-key", self.panel.message.get())
        self.assertFalse(self.panel.active)
        self.panel._clarification(["Литий-селективные мембраны"])
        window = self.app.child_windows[-1]
        next(w for w in descendants(window) if isinstance(w, ttk.Button)).invoke()
        self.assertEqual(self.panel.query.get(), "Литий-селективные мембраны")

    def test_history_opens_without_restart_and_retains_cards_scroll_and_selection(self):
        self.build()
        history = self.app.analysis_history
        self.panel.query_text.set("Мой несохранённый ввод")
        self.app.navigation.select("analyses")
        self.settle()
        rows = [{"id": "saved", "state": "succeeded", "created_at": "2026-09-14T12:00:00", "result_count": 1,
            "input_json": json.dumps({"payload": {"query": "Сохранённый запрос"}})}]
        self.fixture.complete("pilot_history_view", rows)
        history.tree.focus_force()
        self.settle()
        history.tree.event_generate("<Return>")
        self.settle()
        self.assertTrue(any(entry[0] == "pilot_result" for entry in self.fixture.pending.values()))
        self.fixture.complete("pilot_result", saved_payload())
        self.settle()
        self.assertEqual(self.app.tabs.select(), str(self.app.pilot_tab))
        self.assertEqual(self.panel.query.get(), "Мой несохранённый ввод")
        button = next(w for w in descendants(self.panel.card_list.frame) if isinstance(w, ttk.Button) and "Показатели" in str(w.cget("text")))
        button.invoke()
        self.settle()
        self.app.pilot_tab.canvas.yview_moveto(.25)
        # Read the position only after the page's own layout pass: a pending
        # scroll region changes what the very same offset means as a fraction.
        self.pump(lambda: self.app.pilot_tab.timer is None)
        scroll = self.app.pilot_tab.canvas.yview()
        item_widgets = tuple(self.panel.card_list.items.winfo_children())
        for _ in range(3):
            self.app.navigation.select("analyses")
            self.settle()
            self.app.navigation.select("analysis")
            self.settle()
        self.assertEqual(tuple(self.panel.card_list.items.winfo_children()), item_widgets)
        self.assertEqual(self.panel.selected_result_tree.selection(), ("one",))
        self.assertIn("one", self.panel.card_list.expanded)
        self.pump(lambda: self.app.pilot_tab.timer is None)
        self.assertAlmostEqual(self.app.pilot_tab.canvas.yview()[0], scroll[0], places=2)
        self.assertEqual(sum(method == "pilot_history_view" for method, _, _ in self.fixture.calls), 1)
        self.assertFalse(any(method in {"pilot_start", "pilot_resume"} for method, _, _ in self.fixture.calls))

    def test_new_analysis_rejects_late_payload_from_pending_history_open(self):
        self.build()
        self.panel.query_text.set("Новое направление")
        self.app.navigation.select("analyses")
        self.settle()
        self.fixture.complete("pilot_history_view", [{"id": "saved", "state": "succeeded",
            "created_at": "2026-09-14T12:00:00", "result_count": 1,
            "input_json": '{"query":"Старое направление"}'}])
        self.app.analysis_history.tree.focus_force()
        self.settle()
        self.app.analysis_history.tree.event_generate("<Return>")
        self.pump(lambda: any(entry[0] == "pilot_result" for entry in self.fixture.pending.values()))
        reading_owner = self.panel.generation
        self.app.navigation.select("analysis")
        self.settle()
        self.panel.start_button.invoke()
        self.assertEqual(sum(method == "pilot_start" for method, _, _ in self.fixture.calls), 1)
        self.assertGreater(self.panel.generation, reading_owner)
        self.fixture.complete("pilot_start", "new-run")
        self.fixture.complete("pilot_get", {"id": "new-run", "state": "running",
            "message": "Собираем новые публикации", "completed": 2, "total": 5})
        self.app.navigation.select("documents")
        self.settle()
        before = (self.panel.message.get(), self.panel.summary.get(), self.panel.progress_text.get())
        self.fixture.complete("pilot_result", saved_payload())
        self.settle()
        self.assertEqual(self.app.tabs.select(), str(self.app.document_tab))
        self.assertEqual(self.panel.run_id, "new-run")
        self.assertTrue(self.panel.active)
        self.assertIsNone(self.panel.payload)
        self.assertEqual(self.panel.cards, {})
        self.assertEqual(self.panel.tree.get_children(), ())
        self.assertEqual(self.panel.other_tree.get_children(), ())
        self.assertEqual((self.panel.message.get(), self.panel.summary.get(), self.panel.progress_text.get()), before)
        self.assertEqual(self.panel.query.get(), "Новое направление")
        self.assertTrue(self.panel.cancel_button.instate(["!disabled"]))
        self.assertEqual(self.callback_errors, [])

    def test_history_failed_row_does_not_resume_until_explicit_action(self):
        self.build()
        history = self.app.analysis_history
        history.render([{"id": "failed", "state": "failed", "error": "Проверьте подключение", "created_at": "2026-09-14",
                         "input_json": '{"query":"Литий"}'}])
        history.open_selected()
        self.assertIn("Проверьте подключение", history.message.get())
        self.assertFalse(any(method == "pilot_resume" for method, _, _ in self.fixture.calls))
        history.resume_button.invoke()
        self.assertTrue(any(method == "pilot_resume" for method, _, _ in self.fixture.calls))

    def check_scale(self, scale):
        self.build(scale)
        self.assertTrue(self.app.navigation.collapsed)
        for widget in (self.panel.query, self.panel.start_button):
            self.assertTrue(widget.winfo_viewable())
            self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), self.root.winfo_rootx() + self.root.winfo_width())
        self.app.navigation.toggle_width()
        self.settle()
        self.assertFalse(self.app.navigation.collapsed)
        for key in ("documents", "collection", "jobs", "periods", "ml", "settings"):
            if key in self.app.navigation.hidden:
                # Hidden destinations enter through their owning flow and
                # appear in the rail only while that page is active.
                self.app.navigation.select(key)
                self.settle()
            button = self.app.navigation.buttons[key]
            if key != "settings":
                self.app.navigation.body.reveal(button)
                self.settle()
            self.assertTrue(button.winfo_viewable(), str({"key": key, "button": button.winfo_geometry(),
                "body": self.app.navigation.body.winfo_geometry(), "canvas": self.app.navigation.body.canvas.winfo_geometry(),
                "content": self.app.navigation.body.content.winfo_geometry(), "tools": self.app.navigation.tools.winfo_geometry(),
                "scroll": self.app.navigation.body.canvas.yview(), "root": self.root.winfo_geometry()}))
            if key not in self.app.navigation.hidden:
                button.invoke()
            self.settle()
            self.assertEqual(self.app.tabs.select(), str(self.app.navigation.pages[key]))
        self.app.navigation.select("analysis")
        self.app.navigation.toggle_width()
        self.settle()
        self.assertTrue(self.app.navigation.collapsed)
        self.assertEqual(self.callback_errors, [])

    def test_small_window_100_percent(self):
        self.check_scale(1)

    def test_small_window_150_percent(self):
        self.check_scale(1.5)

    def test_small_window_200_percent(self):
        self.check_scale(2)

    def test_hidden_destination_reveals_new_row_at_200_percent(self):
        self.build(2)
        rail = self.app.navigation
        rail.toggle_width()
        self.settle()
        for key in ("collection", "periods", "ml"):
            rail.body.canvas.yview_moveto(0)
            rail.select(key)
            button = rail.buttons[key]
            # On Windows the row maps only after the rail scrolls to it, and its
            # button gets geometry one mapping later: wait for that state, not a
            # fixed pause that a loaded machine can outrun.
            self.pump(lambda button=button: button.winfo_viewable() and button.winfo_height() > 1)
            self.settle()
            canvas = rail.body.canvas
            top = canvas.winfo_rooty()
            bottom = top + canvas.winfo_height()
            self.assertTrue(button.winfo_viewable(), str({"key": key, "button": button.winfo_geometry(),
                                                          "tools": rail.tools.winfo_geometry(),
                                                          "scroll": canvas.yview()}))
            self.assertGreaterEqual(button.winfo_rooty(), top, key)
            self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), bottom, key)


def test_history_count_distinguishes_empty_unavailable_and_pending():
    rows = [{"id": key, "state": state} for key, state in (("empty", "succeeded"), ("broken", "succeeded"), ("pending", "running"))]
    def result(key):
        if key == "broken":
            raise ValueError("damaged")
        return {"result": {"cards": []}}
    service = SimpleNamespace(list_runs=lambda *args: rows, result=result)
    values = history_view(service)
    assert values[0]["result_count"] == 0
    assert values[1]["result_count"] is None
    assert values[2]["result_count"] is None


def test_history_adapter_is_dispatched_outside_the_ui_thread():
    from threading import get_ident
    from app.ui.controller import Controller
    owner = get_ident()
    threads = []
    def list_runs(*args):
        threads.append(get_ident())
        return []
    scheduler = SimpleNamespace(after=lambda *args: None)
    controller = Controller(scheduler)
    controller.backend = object()
    controller.pilot = SimpleNamespace(list_runs=list_runs)
    try:
        assert controller.call("history", "pilot_history_view", lambda _: None, lambda _: None)
        controller.pending["history"][0].result(timeout=3)
        assert threads and all(thread != owner for thread in threads)
    finally:
        controller.executor.shutdown(wait=True)
        controller.ml_executor.shutdown(wait=True)
