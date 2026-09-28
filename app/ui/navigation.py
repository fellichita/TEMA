"""Persistent native pages with a compact, keyboard-accessible sidebar."""

import tkinter as tk
from tkinter import ttk

from app.ui.display import px
from app.ui.icons import attach
from app.ui.theme import COLORS
from app.ui.viewport import ScrollViewport


# ttk offers no spacing between a button's image and its label, so the gap is
# part of the text. The clean label is kept separately for the tooltip.
NAV_GAP = "  "


class SectionNotebook(ttk.Notebook):
    """Keep tool subpages accessible through a small selector, without tab bars."""
    def __init__(self, parent):
        self.selector = ttk.Combobox(parent, state="readonly", width=28)
        self.selector.pack(anchor="w", pady=(4, 8))
        super().__init__(parent, style="Pages.TNotebook")
        self.selector.bind("<<ComboboxSelected>>", lambda _: self.select(self.selector.current()))
        self.bind("<<NotebookTabChanged>>", self._sync)

    def add(self, child, **kwargs):
        super().add(child, **kwargs)
        self.selector.configure(values=tuple(self.tab(tab, "text") for tab in self.tabs()))
        self._sync()

    def _sync(self, _=None):
        if self.select():
            self.selector.current(self.index(self.select()))


class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text = widget, text
        self.timer = self.window = None
        for event in ("<Enter>", "<FocusIn>"):
            widget.bind(event, self.schedule, add="+")
        for event in ("<Leave>", "<FocusOut>", "<ButtonPress>", "<Unmap>", "<Destroy>"):
            widget.bind(event, self.hide, add="+")

    def schedule(self, _=None):
        self.hide()
        self.timer = self.widget.after(450, self.show)

    def show(self):
        self.timer = None
        if not self.widget.winfo_ismapped():
            return
        self.window = tk.Toplevel(self.widget)
        self.window.overrideredirect(True)
        ttk.Label(self.window, text=self.text, padding=6).pack()
        self.window.geometry(f"+{self.widget.winfo_rootx() + self.widget.winfo_width()}+{self.widget.winfo_rooty()}")

    def hide(self, _=None):
        if self.timer is not None:
            self.widget.after_cancel(self.timer)
            self.timer = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class Sidebar(ttk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent, style="Sidebar.TFrame", width=px(app.root, 220))
        self.app = app
        self.collapsed = False
        self.manual_width = False
        self.expanded_width = px(app.root, 220)
        self.buttons, self.labels, self.pages, self.markers = {}, {}, {}, {}
        self.rows, self.hidden = {}, set()
        self._selected_page = None
        self.pack_propagate(False)
        # An explicit rule, not a fill difference alone: in the dark palette the
        # sidebar and the content field are close in luminance by design.
        self.edge = ttk.Frame(self, style="Divider.TFrame", width=px(app.root, 1))
        self.edge.pack(side="right", fill="y")
        header = ttk.Frame(self, style="Sidebar.TFrame", padding=6)
        header.pack(fill="x")
        self.brand = ttk.Label(header, text="Trendanalyser", style="Brand.TLabel")
        self.brand.pack(side="left", padx=4)
        self.toggle = ttk.Button(header, width=2, style="Compact.Nav.TButton", command=self.toggle_width)
        attach(self.toggle, "menu", size="sm", role="muted")
        self.toggle.pack(side="right")
        Tooltip(self.toggle, "Свернуть / развернуть навигацию")
        self.body = ScrollViewport(self, padding=6, auto_hide=True)
        self.body.pack(fill="both", expand=True)
        self.body.canvas.configure(background=COLORS["sidebar"])
        self.body.content.configure(style="Sidebar.TFrame")
        self.primary = ttk.Frame(self.body.content, style="Sidebar.TFrame")
        self.primary.pack(fill="x")
        self.tools_caption = ttk.Label(self.body.content, text="ИНСТРУМЕНТЫ", style="NavGroup.TLabel")
        self.tools_caption.pack(anchor="w", padx=px(app.root, 8), pady=(px(app.root, 24), px(app.root, 8)))
        self.tools = ttk.Frame(self.body.content, style="Sidebar.TFrame")
        self.tools.pack(fill="x")
        self.bottom = ttk.Frame(self, padding=6, style="Sidebar.TFrame")
        self.bottom.pack(fill="x", side="bottom", pady=(4, 8), before=self.body)
        app.root.bind("<Configure>", self._resize, add="+")
        app.tabs.bind("<<NotebookTabChanged>>", self.sync, add="+")

    def add(self, key, label, glyph, page, *, group="primary", hidden=False):
        """`glyph` names an icon in app.ui.icons, drawn beside the label.

        A hidden page keeps working and stays reachable from the flows that open
        it; its item only appears while that page is the current one, so the rail
        stays short without ever leaving the user somewhere unmarked.
        """
        parent = {"primary": self.primary, "tools": self.tools, "bottom": self.bottom}[group]
        row = ttk.Frame(parent, style="Sidebar.TFrame")
        if hidden:
            self.hidden.add(key)
        else:
            row.pack(fill="x", pady=px(self.app.root, 2))
        # The active page is marked by this rule as well as by the fill, so the
        # current location is not carried by colour alone.
        marker = ttk.Frame(row, style="NavMarkerIdle.TFrame", width=px(self.app.root, 3))
        marker.pack(side="left", fill="y")
        button = ttk.Button(row, text=NAV_GAP + label, style="Nav.TButton", command=lambda: self.select(key))
        button.pack(side="left", fill="x", expand=True)
        attach(button, glyph, size="md", role="muted")
        Tooltip(button, label)
        self.buttons[key] = button
        self.markers[key] = marker
        self.rows[key] = row
        self.labels[key] = (label, glyph)
        self.pages[key] = page

    def fit(self):
        """Size the rail to its widest label so no item needs horizontal scrolling."""
        needed = max((button.winfo_reqwidth() for button in self.buttons.values()), default=0)
        # Marker rule, both viewport paddings and room for the vertical scrollbar.
        chrome = px(self.app.root, 3 + 6 * 2 + 16)
        self.expanded_width = min(px(self.app.root, 320),
                                  max(px(self.app.root, 220), needed + chrome))
        self._layout()

    def select(self, key):
        if self.app.closing:
            return
        self.app.tabs.select(self.pages[key])
        self.sync()

    def sync(self, _=None):
        current = self.app.tabs.select()
        style = "Compact.Nav.TButton" if self.collapsed else "Nav.TButton"
        selected_hidden = False
        for key, button in self.buttons.items():
            active = str(self.pages[key]) == current
            if key in self.hidden:
                if active:
                    self.rows[key].pack(fill="x", pady=px(self.app.root, 2))
                    selected_hidden = True
                else:
                    self.rows[key].pack_forget()
            button.configure(style="Selected." + style if active else style)
            self.markers[key].configure(style="NavMarker.TFrame" if active else "NavMarkerIdle.TFrame")
            attach(button, self.labels[key][1], size="md", role="accent" if active else "muted")
        if current != self._selected_page:
            self._selected_page = current
            if selected_hidden:
                # On Windows the newly packed row may start outside the canvas
                # and be unmapped (1x1). Reveal its mapped group first; the
                # viewport then follows the group's resized bounds.
                self.body.reveal(self.tools)
            else:
                self.body._clear_reveal()

    def toggle_width(self):
        self.manual_width = True
        self.collapsed = not self.collapsed
        self._layout()

    def _resize(self, event):
        if event.widget is self.app.root and not self.manual_width:
            collapsed = event.width < px(self.app.root, 960)
            if collapsed != self.collapsed:
                self.collapsed = collapsed
                self._layout()

    def _layout(self):
        self.configure(width=px(self.app.root, 58) if self.collapsed else self.expanded_width)
        if self.collapsed:
            self.brand.pack_forget()
            self.toggle.pack_configure(side="top", fill="x")
        else:
            self.brand.pack(side="left", padx=4, before=self.toggle)
            self.toggle.pack_configure(side="right", fill="none")
        if self.collapsed:
            self.tools_caption.pack_forget()
        else:
            self.tools_caption.pack(anchor="w", padx=px(self.app.root, 8),
                                    pady=(px(self.app.root, 24), px(self.app.root, 8)),
                                    before=self.tools)
        for key, button in self.buttons.items():
            label, _ = self.labels[key]
            button.configure(text="" if self.collapsed else NAV_GAP + label,
                             width=2 if self.collapsed else 1)
        self.sync()
