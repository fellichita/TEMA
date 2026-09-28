"""A user can open the bundled desktop mode from the one-app start window."""

import tkinter as tk
from tkinter import ttk

import pytest

from app import launcher
from app.runtime import session
from app.ui import window

pytestmark = pytest.mark.gui


def test_launcher_opens_desktop_without_command_line(monkeypatch):
    original_tk = tk.Tk
    opened = []

    def tk_with_desktop_click():
        root = original_tk()

        def click():
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)

            button = next(item for item in descendants(root)
                          if isinstance(item, ttk.Button) and item.cget("text") == "Открыть настольную версию")
            button.invoke()

        root.after(30, click)
        return root

    monkeypatch.setattr(tk, "Tk", tk_with_desktop_click)
    monkeypatch.setattr(session, "credentials", lambda: opened.append("credentials"))
    monkeypatch.setattr(window, "run_app", lambda: opened.append("desktop"))

    assert launcher.run_launcher() == 0
    assert opened == ["credentials", "desktop"]
