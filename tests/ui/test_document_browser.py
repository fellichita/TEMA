"""Real text-widget interaction, failed pagination, and destroyed-window replies."""

import gc
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import weakref

from app.runtime.jobs import TaskFailure
from app.ui.pilot_panel import PilotPanel
from app.ui.theme import apply_theme
from tests.test_pilot_operation_ownership import DeferredController
from tests.ui.test_desktop import TkCase
from tests.ui.test_document_links import page
from tests.ui.test_pilot_panel import descendants


class DocumentBrowserTests(TkCase):
    def setUp(self):
        super().setUp()
        apply_theme(self.root)
        self.calls = DeferredController()
        self.panel = object.__new__(PilotPanel)
        self.panel.message = tk.StringVar()
        self.panel.error = lambda error: self.panel.message.set(str(error))
        self.panel.generation = 0
        self.panel.app = SimpleNamespace(root=self.root, closing=False, child_windows=[], controller=self.calls)

    def browser(self):
        self.panel._documents(page(), "saved-run")
        window = self.panel.app.child_windows[-1]
        box = next(widget for widget in descendants(window) if isinstance(widget, tk.Text))
        return window, box

    def button(self, window, title):
        return next(widget for widget in descendants(window) if isinstance(widget, ttk.Button) and widget.cget("text") == title)

    def focus(self, window, box):
        self.root.deiconify()
        self.wait_mapped([window, box])
        box.focus_force()
        self.pump(lambda: self.root.focus_get() is box)

    def test_page_failure_preserves_text_and_selection_with_local_feedback(self):
        window, box = self.browser()
        self.focus(window, box)
        box.event_generate("<Down>")
        before = box.get("1.0", "end-1c")
        following = self.button(window, "Далее")
        following.invoke()
        self.assertTrue(following.instate(["disabled"]))
        self.calls.complete("pilot_documents", error=TaskFailure("Страница недоступна."))
        self.assertEqual(box.get("1.0", "end-1c"), before)
        labels = [widget for widget in descendants(window) if isinstance(widget, ttk.Label)]
        self.assertTrue(any("Страница недоступна" in str(widget.getvar(str(widget.cget("textvariable"))))
                            for widget in labels if str(widget.cget("textvariable"))))
        box.event_generate("<space>")
        self.assertEqual(self.calls.arguments("open_url"), [("https://example.org/publication/1",)])
        following.invoke()
        self.assertEqual(self.calls.arguments("pilot_documents"), [("saved-run", 50), ("saved-run", 50)])

    def test_keyboard_highlights_selected_link_and_tab_leaves_text(self):
        window, box = self.browser()
        self.focus(window, box)
        before = box.get("1.0", "end-1c")
        box.event_generate("<Down>")
        ranges = box.tag_ranges("active-link")
        self.assertEqual(len(ranges), 2)
        self.assertEqual(box.get(*ranges), "https://example.org/publication/1")
        box.event_generate("<Up>")
        box.event_generate("<space>")
        self.assertEqual(self.calls.arguments("open_url"), [("https://example.org/publication/0",)])
        box.event_generate("<Tab>")
        self.pump(lambda: self.root.focus_get() is self.button(window, "Далее"))
        self.assertEqual(box.get("1.0", "end-1c"), before)

    def test_mouse_url_click_opens_exact_original_source(self):
        window, box = self.browser()
        self.focus(window, box)
        self.pump(lambda: box.bbox("3.0") is not None)
        x, y, width, height = box.bbox("3.0")
        box.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
        self.assertEqual(self.calls.arguments("open_url"), [("https://example.org/publication/0",)])

    def test_destroyed_window_ignores_pending_page_and_releases_callbacks(self):
        window, box = self.browser()
        self.button(window, "Далее").invoke()
        reference = weakref.ref(box)
        window.destroy()
        self.panel.app.child_windows.clear()  # This fixture deliberately has a plain registry list.
        self.calls.complete("pilot_documents", page(50))
        del window, box
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(self.callback_errors, [])
