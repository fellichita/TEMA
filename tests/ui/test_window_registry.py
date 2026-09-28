"""Real Tk lifecycle checks for the registry shared by all dialog factories."""

import gc
import tkinter as tk
import weakref

from app.ui.history import ParametersWindow
from app.ui.pilot_materials import _reset_library_views
from app.ui.window import Application
from tests.ui.test_desktop import TkCase


class WindowRegistryTests(TkCase):
    def open_application(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)

    def test_destroyed_raw_and_history_dialogs_release_owners(self):
        self.open_application()
        raw = tk.Toplevel(self.root)
        self.app.child_windows.append(raw)
        child = tk.Frame(raw)
        child.destroy()
        self.assertIn(raw, self.app.child_windows)
        raw_ref = weakref.ref(raw)
        raw.destroy()
        del child, raw
        gc.collect()
        self.assertIsNone(raw_ref())
        dialog = ParametersWindow(self.app, "Lifecycle check")
        self.app.child_windows.append(dialog)
        dialog_ref = weakref.ref(dialog)
        dialog.close()
        del dialog
        gc.collect()
        self.assertIsNone(dialog_ref())
        self.assertEqual(self.app.child_windows, [])

    def test_restore_closes_all_children_while_registry_shrinks(self):
        self.open_application()
        for _ in range(3):
            self.app.child_windows.append(tk.Toplevel(self.root))
        self.app.history.new()
        for _ in range(2):
            self.app.child_windows.append(ParametersWindow(self.app, "Old library"))
        references = [weakref.ref(owner) for owner in self.app.child_windows]
        windows = [getattr(owner, "window", owner) for owner in self.app.child_windows]
        self.assertEqual(len(windows), 6)
        _reset_library_views(self.app.pilot_panel)
        self.assertTrue(all(not window.winfo_exists() for window in windows))
        self.assertEqual(self.app.child_windows, [])
        self.assertIsNone(self.app.history.form)
        del windows
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(self.callback_errors, [])
