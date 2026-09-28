"""Structural regressions for expensive redraws; no hardware-specific deadlines."""

import tkinter as tk
import pytest
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import patch

from app.ui.display import Display
from app.ui.history import HistoryPanel
from app.ui.theme import COLORS, apply_theme
from app.ui.viewport import ScrollViewport

pytestmark = pytest.mark.gui


def test_theme_uses_native_layouts_and_keeps_dark_palette():
    root = tk.Tk()
    root.withdraw()
    original = root.tk.call('tk', 'scaling')
    try:
        Display(root, 1.5)
        style = apply_theme(root)
        assert not any(name.startswith('Rounded.') for name in style.element_names())
        assert 'Documents.separator' not in style.element_names()
        for part in ('Cell', 'Heading'):
            assert style.layout('Documents.Treeview.' + part) == style.layout('Treeview.' + part)
        assert style.lookup('Documents.Treeview', 'background') == COLORS['field']
        assert style.lookup('TNotebook.Tab', 'background', ('selected',)) == COLORS['blue']
    finally:
        root.tk.call('tk', 'scaling', original)
        root.destroy()


def test_remapping_known_page_does_not_walk_subtree_but_new_children_scale():
    root = tk.Tk()
    root.withdraw()
    original = root.tk.call('tk', 'scaling')
    try:
        display = Display(root, 2)
        page = ttk.Frame(root, padding=5)
        page.pack()
        display.widgets(root)
        with patch.object(display, 'widgets', wraps=display.widgets) as visit:
            display._mapped(SimpleNamespace(widget=page))
            visit.assert_not_called()
        child = ttk.Frame(page, padding=7)
        child.pack()
        display._mapped(SimpleNamespace(widget=child))
        assert int(str(child.cget('padding')[0])) == 14
        display._mapped(SimpleNamespace(widget=child))
        assert int(str(child.cget('padding')[0])) == 14
    finally:
        root.tk.call('tk', 'scaling', original)
        root.destroy()


def test_viewport_only_reconfigures_when_required_size_changes():
    root = tk.Tk()
    root.withdraw()
    try:
        viewport = ScrollViewport(root)
        with patch.object(viewport.canvas, 'winfo_width', return_value=600), \
             patch.object(viewport.canvas, 'winfo_height', return_value=400), \
             patch.object(viewport.content, 'winfo_reqwidth', return_value=300), \
             patch.object(viewport.content, 'winfo_reqheight', return_value=200), \
             patch.object(viewport.canvas, 'configure', wraps=viewport.canvas.configure) as resize, \
             patch.object(viewport.canvas, 'itemconfigure', wraps=viewport.canvas.itemconfigure) as fixed_size:
            # Intrinsic canvas-window sizing lets asynchronous content grow or
            # shrink. Coalesce the scrollregion updates without freezing that
            # geometry through explicit canvas item width/height.
            viewport._layout()
            viewport._layout()
            resize.assert_called_once_with(scrollregion=(0, 0, 600, 400))
            with patch.object(viewport.content, 'winfo_reqheight', return_value=800):
                viewport._layout()
                viewport._layout()
            assert resize.call_count == 2
            resize.assert_called_with(scrollregion=(0, 0, 600, 800))
            fixed_size.assert_not_called()
            assert viewport._layout_size == (600, 800)
    finally:
        root.destroy()


def test_history_status_update_touches_one_row_and_preserves_selection():
    root = tk.Tk()
    root.withdraw()
    try:
        panel = SimpleNamespace(enabled=lambda: True, rows=None, _visible_rows={}, selected_id="1",
                                tree=ttk.Treeview(root), message=tk.StringVar())
        rows = [dict(id=str(index), state="queued", request=dict(topic=str(index),
                from_date="2020-01-01", until_date="2025-01-01", sources=("crossref",)))
                for index in range(100)]
        HistoryPanel.render_list(panel, rows)
        panel.tree.selection_set("1")
        changed = [row.copy() for row in rows]
        changed[-1]["state"] = "running"
        with patch.object(panel.tree, "delete", wraps=panel.tree.delete) as deleted, \
                patch.object(panel.tree, "insert", wraps=panel.tree.insert) as inserted, \
                patch.object(panel.tree, "item", wraps=panel.tree.item) as updated:
            HistoryPanel.render_list(panel, changed)
            deleted.assert_not_called()
            inserted.assert_not_called()
            assert updated.call_count == 1
            assert updated.call_args.args[0] == "99"
        assert panel.tree.selection() == ("1",)
        HistoryPanel.render_list(panel, list(reversed(changed)))
        assert panel.tree.get_children() == tuple(str(index) for index in reversed(range(100)))
        assert panel.tree.selection() == ("1",)
    finally:
        root.destroy()
