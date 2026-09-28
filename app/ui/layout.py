"""Layout primitives that survive a narrow window and a scaled display.

ttk packs a row of buttons until it runs out of room and then quietly clips the
last one: the control is still mapped and still callable, so nothing reports a
problem, but part of it is off-screen. `ActionRow` wraps onto another line
instead — labels keep their full text and every action stays reachable.
"""

from typing import Literal

from tkinter import ttk

from app.ui.display import px
from app.ui.tokens import SPACE


class ActionRow(ttk.Frame):
    """A row of actions that reflows onto further lines when space runs out."""

    def __init__(self, parent, *, gap=SPACE["sm"], **options):
        super().__init__(parent, **options)
        self.gap = gap
        self.items: list[tuple[ttk.Widget, Literal["left", "right"]]] = []
        self._wrapped: bool | None = None
        self._width = 0
        self.bind("<Configure>", self._reflow)

    def add(self, widget, side: Literal["left", "right"] = "left"):
        """Register a control; `side` is honoured only while the row fits."""
        self.items.append((widget, side))
        self._wrapped = None
        self._reflow()
        return widget

    def _reflow(self, event=None):
        width = event.width if event is not None else self.winfo_width()
        gap = px(self, self.gap)
        requested = [widget.winfo_reqwidth() for widget, _ in self.items]
        needed = sum(requested) + gap * max(0, len(self.items) - 1)
        # A width of 1 means Tk has not allocated the row yet. Start wrapped:
        # growing into one line later is cheaper than being clipped meanwhile.
        wrapped = width <= 1 or needed > width
        if wrapped == self._wrapped and width == self._width:
            return
        self._wrapped, self._width = wrapped, width
        for widget, _ in self.items:
            widget.pack_forget()
            widget.grid_forget()
        if not wrapped:
            for widget, side in self.items:
                padding = (0, gap) if side == "left" else (gap, 0)
                widget.pack(side=side, padx=padding)
            return
        row = column = column_width = 0
        for widget, size in zip((widget for widget, _ in self.items), requested, strict=True):
            if column and column_width + gap + size > width:
                row, column, column_width = row + 1, 0, 0
            widget.grid(row=row, column=column, sticky="w", padx=(0, gap), pady=(0, gap))
            column_width += (gap if column else 0) + size
            column += 1
