"""Loading, error and empty blocks shared by every panel.

Three rules the panels used to solve individually, each in its own way:

* loading reserves the space the content will occupy, so nothing jumps when
  the result arrives;
* an error states the cause *and* the way out, next to the thing that failed,
  and never replaces the retry path with a dead end;
* an empty view offers the one action that would fill it.

Tk exposes no accessibility tree, so "announced" here means focus moves to the
recovery control and the text is readable — not that a screen reader is told.
"""

from tkinter import ttk

from app.ui.display import px
from app.ui.icons import attach
from app.ui.tokens import SPACE


class StateBlock(ttk.Frame):
    """Icon, title, explanation and at most one action."""

    def __init__(self, parent, *, icon="info", title="", description="",
                 action=None, command=None, tone="text", style="TButton"):
        super().__init__(parent, style="Card.TFrame", padding=px(parent, SPACE["lg"]))
        self.icon_label = ttk.Label(self, style="Muted.TLabel")
        attach(self.icon_label, icon, size="lg", role=tone, compound="image")
        self.icon_label.pack(anchor="w", pady=(0, px(parent, SPACE["sm"])))
        self.title_label = ttk.Label(self, text=title, style="CardTitle.TLabel",
                                     wraplength=px(parent, 520), justify="left")
        self.title_label.pack(anchor="w", fill="x")
        self.description_label = ttk.Label(self, text=description, style="Muted.TLabel",
                                           wraplength=px(parent, 520), justify="left")
        self.action_button = ttk.Button(self, style=style)
        self._tone = tone
        self.update_content(title=title, description=description, action=action, command=command)

    def update_content(self, *, title=None, description=None, action=None, command=None):
        if title is not None:
            self.title_label.configure(text=title)
        if description is not None:
            self.description_label.configure(text=description)
            if description:
                self.description_label.pack(anchor="w", fill="x",
                                            pady=(px(self, SPACE["xs"]), 0))
            else:
                self.description_label.pack_forget()
        if action is None and command is None:
            return
        if action:
            self.action_button.configure(text=action, command=command or (lambda: None))
            self.action_button.pack(anchor="w", pady=(px(self, SPACE["md"]), 0))
        else:
            self.action_button.pack_forget()

    def show(self, **options):
        options.setdefault("fill", "x")
        self.pack(**options)
        return self

    def hide(self):
        self.pack_forget()
        return self


class EmptyBlock(StateBlock):
    """Nothing here yet — and the one thing that would change that."""

    def __init__(self, parent, *, title, description="", action=None, command=None, icon="inbox"):
        super().__init__(parent, icon=icon, title=title, description=description,
                         action=action, command=command, tone="muted")


class ErrorBlock(StateBlock):
    """Something failed: what happened, and how to try again."""

    def __init__(self, parent, *, title="Не удалось выполнить", description="",
                 action="Повторить", command=None):
        super().__init__(parent, icon="alert", title=title, description=description,
                         action=action, command=command, tone="error")
        self.title_label.configure(style="Error.TLabel")

    def report(self, description, command=None, *, title=None, action="Повторить"):
        """Show one failure and put keyboard focus on its recovery control."""
        self.update_content(title=title or "Не удалось выполнить", description=description,
                            action=action if command else "", command=command)
        self.show()
        if command:
            self.action_button.focus_set()
        return self


class LoadingBlock(ttk.Frame):
    """Skeleton rows that reserve the incoming layout, plus a progress line."""

    WIDTHS = (0.55, 0.9, 0.75)

    def __init__(self, parent, *, rows=3, text="Загружаем…"):
        super().__init__(parent, style="Card.TFrame", padding=px(parent, SPACE["lg"]))
        self.status = ttk.Label(self, text=text, style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(0, px(parent, SPACE["md"])))
        self.bars = []
        for _ in range(rows):
            bar = ttk.Frame(self, style="Skeleton.TFrame", height=px(parent, SPACE["md"]))
            bar.pack(anchor="w", pady=(0, px(parent, SPACE["sm"])))
            self.bars.append(bar)
        self.bind("<Configure>", self._resize)

    def _resize(self, event):
        # Reserve a realistic line length instead of a full-width grey slab.
        usable = max(px(self, 160), event.width - px(self, SPACE["lg"] * 2))
        for bar, fraction in zip(self.bars, self.WIDTHS * len(self.bars), strict=False):
            bar.configure(width=round(usable * fraction))

    def start(self, text=None, **options):
        if text:
            self.status.configure(text=text)
        options.setdefault("fill", "x")
        self.pack(**options)
        return self

    def stop(self):
        self.pack_forget()
        return self
