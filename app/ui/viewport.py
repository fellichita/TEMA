"""Notebook pages that retain usable content when the window cannot fit it."""

import tkinter as tk
from tkinter import ttk

from app.ui.display import PixelWheel, px


class ScrollViewport(ttk.Frame):
    def __init__(self, parent, *, padding=12, auto_hide=False):
        super().__init__(parent)
        self.auto_hide = auto_hide
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                                takefocus=False, xscrollincrement=1)
        self.vertical = ttk.Scrollbar(self, command=lambda *args: self._scroll_to("y", *args))
        self.horizontal = ttk.Scrollbar(self, orient="horizontal", command=lambda *args: self._scroll_to("x", *args))
        self.canvas.configure(yscrollcommand=self.vertical.set, xscrollcommand=self.horizontal.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vertical.grid(row=0, column=1, sticky="ns")
        self.horizontal.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Let Tk propagate changing child requests into the canvas window.
        # Explicit item width/height would freeze this size: longer result text
        # can then request more room without a Configure event on the content.
        self._window_frame = ttk.Frame(self.canvas)
        self._window_frame.rowconfigure(0, weight=1)
        self._window_frame.columnconfigure(0, weight=1)
        self.content = ttk.Frame(self._window_frame, padding=padding)
        self.content.grid(row=0, column=0, sticky="nsew")
        self.window = self.canvas.create_window(0, 0, window=self._window_frame, anchor="nw")
        self.timer = None
        self._layout_size = None
        self.focus_target = None
        self.focus_bindings = []
        self.reveal_target = None
        self.reveal_bindings = []
        self.wheel = PixelWheel(self.canvas)
        self.canvas.bind("<Configure>", self._schedule)
        # Returning to a same-size Notebook page need not emit Configure.
        # The embedded window may still be unmapped, so its own Map event
        # cannot be the event that restarts canvas layout.
        self.canvas.bind("<Map>", self._mapped)
        self.canvas.bind("<Unmap>", self._hidden)
        self.content.bind("<Configure>", self._schedule)
        self.content.bind("<Map>", self._schedule)
        # On X11, shrinking below the current scroll offset unmaps the canvas
        # window before its children receive Configure. Recompute the region
        # from their requests so Tk can clamp the offset and map it again.
        self._window_frame.bind("<Unmap>", self._schedule)
        self.bind("<Destroy>", self._destroyed)
        root = self.winfo_toplevel()
        self.bindings = [(sequence, root.bind(sequence, callback, add="+")) for sequence, callback in (
            ("<FocusIn>", self._focus), ("<MouseWheel>", self._scroll),
            ("<Button-4>", self._scroll), ("<Button-5>", self._scroll),
            ("<Map>", self._request_changed), ("<Unmap>", self._request_changed),
            ("<Configure>", self._request_changed))]

    def _request_changed(self, event):
        # A child can request more room after DPI scaling without changing the
        # canvas window's current size. Coalesce those requests, never walk it.
        if self._contains(event.widget) and self.winfo_ismapped():
            self._schedule()

    def _contains(self, widget):
        return widget in (self.canvas, self.content) or str(widget).startswith(str(self.content) + ".")

    def _mapped(self, _event):
        # Tk may leave the embedded window unmapped after a same-size tab
        # return. Reapply the region once for the new native mapping, while
        # ordinary repeated layout requests still share the size cache.
        self._layout_size = None
        self._schedule()

    def _schedule(self, event=None):
        if self.timer is None:
            self.timer = self.after_idle(self._layout)

    def _layout(self):
        self.timer = None
        # Grid minimums fill a larger viewport while preserving intrinsic
        # growth/shrink propagation when asynchronously rendered text changes.
        self._window_frame.columnconfigure(0, minsize=self.canvas.winfo_width())
        self._window_frame.rowconfigure(0, minsize=self.canvas.winfo_height())
        width = max(self.canvas.winfo_width(), self.content.winfo_reqwidth())
        height = max(self.canvas.winfo_height(), self.content.winfo_reqheight())
        if self._layout_size != (width, height):
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self._layout_size = (width, height)
        if self.auto_hide:
            for bar, needed in ((self.vertical, height > self.canvas.winfo_height()),
                                (self.horizontal, width > self.canvas.winfo_width())):
                if needed and not bar.winfo_manager():
                    bar.grid()
                elif not needed and bar.winfo_manager():
                    bar.grid_remove()
        if self.focus_target is not None:
            widget = self.focus_target
            if not widget.winfo_exists():
                self._clear_focus()
            elif (widget.winfo_viewable() and widget.winfo_width() > 1 and widget.winfo_height() > 1
                    and self.content.winfo_width() == width and self.content.winfo_height() == height):
                self._clear_focus()
                widget.focus_set()
                self.reveal(widget)
        if self.reveal_target is not None:
            if self.reveal_target.winfo_exists():
                self._reveal(self.reveal_target)
            else:
                self._clear_reveal()

    def _clear_reveal(self):
        for widget, sequence, binding in self.reveal_bindings:
            if widget.winfo_exists():
                widget.unbind(sequence, binding)
        self.reveal_bindings.clear()
        self.reveal_target = None

    def _target_destroyed(self, event):
        if event.widget is self.reveal_target:
            self._clear_reveal()

    def _hidden(self, _event):
        self._clear_focus()
        self._clear_reveal()

    def _clear_focus(self):
        for widget, sequence, binding in self.focus_bindings:
            if widget.winfo_exists():
                widget.unbind(sequence, binding)
        self.focus_bindings.clear()
        self.focus_target = None

    def focus_when_visible(self, widget):
        """Focus after Notebook mapping, without entering a nested Tk event loop."""
        self._clear_focus()
        self.focus_target = widget
        # A mapped page does not mean that its descendants are mapped yet on
        # X11. Retain the request until the actual target has usable geometry.
        self.focus_bindings = [(widget, sequence, widget.bind(sequence, self._schedule, add="+"))
                               for sequence in ("<Map>", "<Configure>")]
        self._schedule()

    def _focus(self, event):
        if self._contains(event.widget) and self.content.winfo_ismapped():
            self.reveal(event.widget)

    def reveal(self, widget):
        # Native window resizing and nested pane layout can arrive after the
        # initial request. Keep its target through those Configure events;
        # explicit scrolling or a new request replaces this navigation intent.
        if widget is not self.reveal_target:
            self._clear_reveal()
            self.reveal_target = widget
            ancestor = widget
            while ancestor is not None and self._contains(ancestor):
                self.reveal_bindings.extend((ancestor, sequence, ancestor.bind(sequence, self._schedule, add="+"))
                                           for sequence in ("<Map>", "<Configure>"))
                ancestor = ancestor.master
            self.reveal_bindings.append((widget, "<Destroy>", widget.bind("<Destroy>", self._target_destroyed, add="+")))
        self._reveal(widget)
        self._schedule()

    def _reveal(self, widget):
        if not widget.winfo_viewable() or not self.content.winfo_ismapped():
            return
        margin = px(self.nametowidget("."), 8)
        for axis in ("x", "y"):
            position = getattr(widget, "winfo_root" + axis)() - getattr(self.content, "winfo_root" + axis)()
            dimension = "width" if axis == "x" else "height"
            extent = getattr(widget, "winfo_" + dimension)()
            visible = getattr(self.canvas, "winfo_" + dimension)()
            total = max(1, getattr(self.content, "winfo_" + dimension)())
            start = getattr(self.canvas, "canvas" + axis)(0)
            if extent > visible:
                # Fill the viewport at the nearest edge instead of accepting a
                # tiny overlap. This retains every previously visible row/pixel
                # and leaves a viewport already inside the target unchanged.
                nearest = max(position, min(start, position + extent - visible))
                if nearest != start:
                    getattr(self.canvas, axis + "view_moveto")(nearest / total)
                continue
            # A target filling the viewport cannot also fit two margins.
            # Reapplying opposite edges must converge, not alternate forever.
            available_margin = min(margin, visible - extent)
            if position < start:
                getattr(self.canvas, axis + "view_moveto")(max(0, position - available_margin) / total)
            elif position + extent > start + visible:
                getattr(self.canvas, axis + "view_moveto")((position + extent - visible + available_margin) / total)

    def _scroll_to(self, axis, *args):
        self._clear_focus()
        self._clear_reveal()
        getattr(self.canvas, axis + "view")(*args)

    def _scroll(self, event):
        widget = event.widget
        if not self.content.winfo_ismapped() or not self._contains(widget):
            return
        # Text, lists and trees own their scrolling; the surrounding page owns
        # wheel events on labels, forms and other non-scrolling controls.
        self._clear_focus()
        self._clear_reveal()
        if isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview, ttk.Combobox)):
            return
        self.wheel.scroll(event)

    def _destroyed(self, event):
        if event.widget is not self:
            return
        if self.timer is not None:
            self.after_cancel(self.timer)
        self._clear_focus()
        self._clear_reveal()
        root = self.winfo_toplevel()
        for sequence, binding in self.bindings:
            root.unbind(sequence, binding)
