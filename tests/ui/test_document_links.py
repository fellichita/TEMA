"""Native Tcl callback lifetime and independent document browser navigation."""

import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

from app.ui.pilot_panel import PilotPanel
from app.ui.theme import apply_theme
from tests.test_pilot_operation_ownership import DeferredController
from tests.ui.test_desktop import TkCase
from tests.ui.test_pilot_panel import descendants


def page(offset=0):
    return {"total": 100, "items": [{"title": f"Publication {i}", "source": "crossref",
            "publication_year": 2025, "url": f"https://example.org/publication/{i}"}
            for i in range(offset, offset + 50)]}


class DocumentLinkTests(TkCase):
    def setUp(self):
        super().setUp()
        apply_theme(self.root)
        self.calls = DeferredController()
        self.panel = object.__new__(PilotPanel)
        self.panel.generation = 0
        self.panel.error = lambda error: self.fail(str(error))
        self.panel.app = SimpleNamespace(root=self.root, closing=False, child_windows=[], controller=self.calls)

    def window(self, identifier):
        self.panel._documents(page(), identifier)
        return self.panel.app.child_windows[-1]

    def button(self, window, text):
        return next(w for w in descendants(window) if isinstance(w, ttk.Button) and w.cget("text") == text)

    def test_repeated_pages_do_not_accumulate_tcl_commands(self):
        window = self.window("first")
        box = next(w for w in descendants(window) if isinstance(w, tk.Text))
        initial = len(box._tclCommands)
        for i in range(12):
            offset = 50 if i % 2 == 0 else 0
            self.button(window, "Далее" if offset else "Назад").invoke()
            self.calls.complete("pilot_documents", page(offset))
            self.assertEqual(len(box._tclCommands), initial)

    def test_two_document_windows_can_page_independently(self):
        first, second = self.window("first"), self.window("second")
        self.button(first, "Далее").invoke()
        self.button(second, "Далее").invoke()
        self.assertEqual(self.calls.arguments("pilot_documents"), [("first", 50), ("second", 50)])

    def test_keyboard_opens_focused_publication(self):
        window = self.window("first")
        self.root.deiconify()
        box = next(w for w in descendants(window) if isinstance(w, tk.Text))
        self.wait_mapped([window, box])
        box.focus_force()
        self.pump(lambda: self.root.focus_get() is box)
        box.event_generate("<Down>")
        box.event_generate("<Return>")
        self.assertEqual(self.calls.arguments("open_url"), [("https://example.org/publication/1",)])
