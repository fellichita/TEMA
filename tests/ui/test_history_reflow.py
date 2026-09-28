"""The history lookup must settle instead of feeding back into Tk Configure."""

from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from app.ui.history import HistoryPanel


@pytest.mark.gui
def test_lookup_action_layout_stabilizes_at_the_fit_boundary():
    root = tk.Tk()
    root.withdraw()
    try:
        container = ttk.Frame(root)
        row = ttk.Frame(container)
        row.pack(fill="x")
        ttk.Entry(row, width=20).pack(side="left")
        button = ttk.Button(container, text="Показать периоды выбранного сбора")
        panel = HistoryPanel.__new__(HistoryPanel)
        panel.lookup_row = row
        panel.view_button = button
        panel._lookup_wrapped = None
        root.update_idletasks()
        width = row.winfo_reqwidth() + button.winfo_reqwidth() + 12
        for _ in range(6):
            HistoryPanel._reflow_lookup(panel, SimpleNamespace(width=width))
            root.update_idletasks()
            assert panel._lookup_wrapped is False
            assert button.winfo_manager() == "pack"
            assert button.master is container
    finally:
        root.destroy()
