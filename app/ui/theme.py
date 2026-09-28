"""Shared light/dark palettes for existing widgets and future dialogs.

Every colour, size and spacing value here comes from app.ui.tokens; nothing in
this module invents a literal. Widgets ask for a named style ("Section.TLabel",
"Accent.TButton") and never for a colour, so retheming stays a token change.
"""

import json
import tkinter as tk
from tkinter import font, ttk
from uuid import uuid4

from app.identity import default_data_dir
from app.ui.display import px
from app.ui.fonts import load_fonts
from app.ui.tokens import (CONTROL, DARK, FONT_FAMILY, FONT_FAMILY_STRONG, LIGHT, PALETTES,
                            RADIUS, SPACE, TEXT)

__all__ = ["COLORS", "DARK", "LIGHT", "PALETTES", "apply_theme", "load_theme", "save_theme"]

# Keep the dictionary identity stable for modules importing COLORS.
COLORS = LIGHT.copy()


def load_theme(path=None):
    path = path if path is not None else default_data_dir() / "appearance.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("theme") if isinstance(data, dict) else None
        return name if isinstance(name, str) and name in PALETTES else "light"
    except (OSError, ValueError):
        return "light"


def save_theme(name, path=None):
    if name not in PALETTES:
        raise ValueError("Unknown theme")
    path = path if path is not None else default_data_dir() / "appearance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps({"theme": name}), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _options(root, c):
    """Defaults for classic Tk widgets, including dialogs built later."""
    root.configure(background=c["background"])
    for option, value in {
        "*background": c["surface"], "*foreground": c["text"],
        "*selectBackground": c["selected"], "*selectForeground": c["text"],
        "*insertBackground": c["accent"], "*highlightBackground": c["border"],
        "*highlightColor": c["accent"], "*Text.background": c["field"],
        "*Text.font": "TkTextFont", "*Text.relief": "flat",
        "*Text.borderWidth": 0, "*Text.highlightThickness": 1,
        "*Text.padX": SPACE["md"], "*Text.padY": SPACE["md"], "*Text.spacing1": 3,
        "*Text.spacing3": 5, "*Listbox.background": c["field"],
        "*Listbox.font": "TkDefaultFont", "*Scrollbar.troughColor": c["field"],
        "*Scrollbar.activeBackground": c["raised"],
    }.items():
        root.option_add(option, value)


def _typography(root, c, style):
    """One scale, three weights. Hierarchy is carried by size and weight."""
    load_fonts()
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        font.nametofont(name, root=root).configure(family=FONT_FAMILY, size=TEXT["body"],
                                                   weight="normal")
    for name, size, family, colour in (
        ("Display.TLabel", TEXT["display"], FONT_FAMILY_STRONG, "text"),
        ("Hero.TLabel", TEXT["hero"], FONT_FAMILY_STRONG, "text"),
        ("Section.TLabel", TEXT["section"], FONT_FAMILY_STRONG, "text"),
        ("CardTitle.TLabel", TEXT["lead"], FONT_FAMILY_STRONG, "text"),
        ("Lead.TLabel", TEXT["lead"], FONT_FAMILY, "text"),
        ("Muted.TLabel", TEXT["body"], FONT_FAMILY, "muted"),
        ("Caption.TLabel", TEXT["caption"], FONT_FAMILY, "muted"),
        ("Error.TLabel", TEXT["body"], FONT_FAMILY, "error"),
        ("Success.TLabel", TEXT["body"], FONT_FAMILY, "success"),
        ("Warning.TLabel", TEXT["body"], FONT_FAMILY, "warning"),
        ("Link.TLabel", TEXT["body"], FONT_FAMILY, "accent"),
    ):
        style.configure(name, font=(family, size), foreground=c[colour])
    style.configure("Brand.TLabel", background=c["sidebar"], foreground=c["text"],
                    font=(FONT_FAMILY_STRONG, TEXT["label"]))
    style.configure("NavGroup.TLabel", background=c["sidebar"], foreground=c["muted"],
                    font=(FONT_FAMILY, TEXT["caption"]))
    style.configure("Placeholder.TLabel", background=c["field"], foreground=c["muted"])
    style.configure("Footer.TLabel", background=c["background"], foreground=c["muted"])
    style.configure("Title.TLabel", background=c["selected"], font=(FONT_FAMILY_STRONG, TEXT["hero"]))
    style.configure("Subtitle.TLabel", background=c["selected"], foreground=c["muted"])
    style.configure("Skeleton.TLabel", background=c["skeleton"], foreground=c["skeleton"])


def _surfaces(root, c, style):
    """Frames, separators and the elevation ladder: page, card, raised."""
    style.configure("Shell.TFrame", background=c["background"])
    style.configure("Sidebar.TFrame", background=c["sidebar"])
    style.configure("Card.TFrame", background=c["surface"], borderwidth=1, relief="solid",
                    bordercolor=c["divider"])
    style.configure("CardHeader.TFrame", background=c["surface"])
    style.configure("Raised.TFrame", background=c["raised"])
    style.configure("Toolbar.TFrame", background=c["background"])
    style.configure("Header.TFrame", background=c["selected"])
    style.configure("Skeleton.TFrame", background=c["skeleton"])
    # A one-pixel rule. Height is applied by the caller so it survives scaling.
    style.configure("Divider.TFrame", background=c["divider"])
    style.configure("TSeparator", background=c["divider"])
    style.configure("TLabelframe", background=c["surface"], bordercolor=c["divider"],
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=c["surface"], foreground=c["muted"],
                    font=(FONT_FAMILY, TEXT["label"]))
    style.configure("TPanedwindow", background=c["background"])


def _buttons(root, c, style):
    """Three weights of action, plus the destructive one, plus navigation."""
    # Secondary — the default. Quiet fill, visible border, real focus ring.
    style.configure("TButton", background=c["raised"], foreground=c["text"],
                    bordercolor=c["border"], borderwidth=1, relief="flat",
                    focuscolor=c["accent"], anchor="center")
    style.map("TButton",
              background=[("disabled", c["surface"]), ("pressed", c["selected"]),
                          ("active", c["selected"])],
              foreground=[("disabled", c["disabled"])],
              bordercolor=[("focus", c["accent"]), ("active", c["border"])])

    # Accent — one per screen. Solid fill carries the primary call to action.
    style.configure("Accent.TButton", background=c["accent"], foreground=c["on_accent"],
                    bordercolor=c["accent"], borderwidth=1, relief="flat",
                    font=(FONT_FAMILY_STRONG, TEXT["body"]))
    style.map("Accent.TButton",
              background=[("disabled", c["raised"]), ("pressed", c["accent_pressed"]),
                          ("active", c["accent_hover"])],
              foreground=[("disabled", c["disabled"])],
              bordercolor=[("disabled", c["border"]), ("focus", c["text"]),
                           ("pressed", c["accent_pressed"]), ("active", c["accent_hover"])])

    # Primary — the tonal fill kept for dialogs that already rely on it.
    style.configure("Primary.TButton", background=c["blue"], foreground=c["text"],
                    bordercolor=c["border"], borderwidth=1,
                    font=(FONT_FAMILY_STRONG, TEXT["body"]))
    style.map("Primary.TButton",
              background=[("disabled", c["raised"]), ("pressed", c["selected"]),
                          ("active", c["hover"])],
              foreground=[("disabled", c["disabled"])],
              bordercolor=[("focus", c["accent"])])

    # Ghost — toolbar and inline actions that must not compete with the CTA.
    style.configure("Ghost.TButton", background=c["surface"], foreground=c["accent"],
                    bordercolor=c["surface"], borderwidth=1, relief="flat")
    style.map("Ghost.TButton",
              background=[("disabled", c["surface"]), ("pressed", c["selected"]),
                          ("active", c["blue"])],
              foreground=[("disabled", c["disabled"])],
              bordercolor=[("focus", c["accent"])])

    # Danger — semantic colour plus an explicit label at every call site.
    style.configure("Danger.TButton", background=c["surface"], foreground=c["error"],
                    bordercolor=c["error"], borderwidth=1, relief="flat")
    style.map("Danger.TButton",
              background=[("disabled", c["surface"]), ("pressed", c["selected"]),
                          ("active", c["raised"])],
              foreground=[("disabled", c["disabled"])],
              bordercolor=[("disabled", c["border"])])

    # Chip — a suggestion, not a command. Reads as a rounded tonal tag.
    style.configure("Chip.TButton", background=c["field"], foreground=c["text"],
                    bordercolor=c["divider"], borderwidth=1, relief="flat")
    style.map("Chip.TButton",
              background=[("disabled", c["surface"]), ("pressed", c["selected"]),
                          ("active", c["blue"])],
              foreground=[("disabled", c["disabled"])],
              bordercolor=[("focus", c["accent"]), ("active", c["border"])])

    for name in ("Nav.TButton", "Selected.Nav.TButton",
                 "Compact.Nav.TButton", "Selected.Compact.Nav.TButton"):
        selected = name.startswith("Selected.")
        compact = "Compact" in name
        style.configure(name, background=c["selected"] if selected else c["sidebar"],
                        foreground=c["text"] if selected else c["muted"],
                        bordercolor=c["sidebar"], borderwidth=1, relief="flat",
                        anchor="center" if compact else "w",
                        font=(FONT_FAMILY_STRONG if selected else FONT_FAMILY, TEXT["body"]))
        style.map(name, background=[("active", c["raised"]), ("pressed", c["selected"])],
                  foreground=[("active", c["text"]), ("disabled", c["disabled"])],
                  bordercolor=[("focus", c["accent"])])
    # The active page is marked by this rule beside the button, not by colour alone.
    style.configure("NavMarker.TFrame", background=c["accent"])
    style.configure("NavMarkerIdle.TFrame", background=c["sidebar"])


def _inputs(root, c, style):
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(name, fieldbackground=c["field"], background=c["raised"],
                        foreground=c["text"], arrowcolor=c["accent"],
                        bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                        insertcolor=c["text"], borderwidth=1)
        style.map(name,
                  fieldbackground=[("disabled", c["surface"]), ("readonly", c["field"])],
                  foreground=[("disabled", c["disabled"]), ("readonly", c["text"])],
                  bordercolor=[("invalid", c["error"]), ("focus", c["accent"])],
                  lightcolor=[("focus", c["accent"])], darkcolor=[("focus", c["accent"])],
                  arrowcolor=[("disabled", c["disabled"])])
    style.configure("Invalid.TEntry", bordercolor=c["error"], lightcolor=c["error"],
                    darkcolor=c["error"])
    for name in ("TCheckbutton", "TRadiobutton"):
        style.configure(name, background=c["surface"], foreground=c["text"],
                        indicatorbackground=c["field"], indicatorforeground=c["accent"],
                        bordercolor=c["border"], focuscolor=c["accent"])
        style.map(name, background=[("active", c["surface"])],
                  foreground=[("disabled", c["disabled"])],
                  indicatorbackground=[("disabled", c["surface"]), ("selected", c["accent"]),
                                       ("active", c["raised"])],
                  indicatorforeground=[("selected", c["on_accent"])])


def _collections(root, c, style):
    """Notebooks, trees, scrollbars and progress."""
    style.configure("TNotebook", background=c["background"], borderwidth=0, tabmargins=(0, 4, 0, 0))
    # Retain native page ownership and state, render navigation only in the sidebar.
    style.layout("Pages.TNotebook.Tab", [])
    style.configure("Pages.TNotebook", borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab", background=c["background"], foreground=c["muted"],
                    bordercolor=c["divider"])
    style.map("TNotebook.Tab", background=[("selected", c["blue"]), ("active", c["raised"])],
              foreground=[("selected", c["text"]), ("active", c["text"])])
    style.configure("Treeview", background=c["field"], fieldbackground=c["field"],
                    foreground=c["text"], bordercolor=c["divider"], borderwidth=0)
    style.map("Treeview", background=[("selected", c["selected"])],
              foreground=[("selected", c["text"])])
    style.configure("Treeview.Heading", background=c["surface"], foreground=c["muted"],
                    font=(FONT_FAMILY_STRONG, TEXT["label"]), relief="flat",
                    bordercolor=c["divider"])
    style.map("Treeview.Heading", background=[("active", c["raised"])],
              foreground=[("active", c["text"])])
    style.configure("Horizontal.TProgressbar", background=c["accent"], troughcolor=c["raised"],
                    bordercolor=c["raised"], lightcolor=c["accent"], darkcolor=c["accent"],
                    borderwidth=0)
    for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(name, background=c["raised"], troughcolor=c["background"],
                        arrowcolor=c["muted"], bordercolor=c["background"], borderwidth=0)
        style.map(name, background=[("active", c["border"]), ("pressed", c["accent"])],
                  arrowcolor=[("active", c["text"])])


def apply_theme(root, name="light"):
    """Install before constructing widgets, including future child windows."""
    previous = COLORS.copy()
    COLORS.update(PALETTES[name])
    c = COLORS
    style = ttk.Style(root)
    # Native Aqua/Windows themes ignore several essential background colors.
    style.theme_use("clam")
    _options(root, c)
    style.configure(".", background=c["surface"], foreground=c["text"],
                    font="TkDefaultFont", bordercolor=c["border"],
                    lightcolor=c["border"], darkcolor=c["border"],
                    troughcolor=c["field"], selectbackground=c["selected"],
                    selectforeground=c["text"], focuscolor=c["accent"])
    _typography(root, c, style)
    _surfaces(root, c, style)
    _buttons(root, c, style)
    _inputs(root, c, style)
    _collections(root, c, style)
    # Native clam layouts retain our palette without expensive tiled-image
    # repainting on every tab switch (especially visible on Windows/high DPI).
    _document_columns(root, style)
    # One 8pt rhythm for every control's internal padding, scaled for the display.
    dimensions: dict[str, dict[str, int | tuple[int, ...]]] = {
        "TButton": {"padding": (CONTROL["pad_x"], CONTROL["pad_y"])},
        "Accent.TButton": {"padding": (CONTROL["pad_x"], CONTROL["pad_y"])},
        "Primary.TButton": {"padding": (CONTROL["pad_x"], CONTROL["pad_y"])},
        "Ghost.TButton": {"padding": (SPACE["sm"], SPACE["sm"])},
        "Danger.TButton": {"padding": (CONTROL["pad_x"], CONTROL["pad_y"])},
        "Chip.TButton": {"padding": (CONTROL["pad_x"], CONTROL["pad_y"])},
        "Nav.TButton": {"padding": (SPACE["sm"], SPACE["sm"])},
        "Selected.Nav.TButton": {"padding": (SPACE["sm"], SPACE["sm"])},
        "Compact.Nav.TButton": {"padding": (SPACE["xs"], SPACE["sm"])},
        "Selected.Compact.Nav.TButton": {"padding": (SPACE["xs"], SPACE["sm"])},
        "TEntry": {"padding": SPACE["sm"]}, "TCombobox": {"padding": SPACE["sm"]},
        "TSpinbox": {"padding": (SPACE["sm"], 1)},
        "TCheckbutton": {"padding": (0, SPACE["xs"] + 1)},
        "TRadiobutton": {"padding": (0, SPACE["xs"] + 1)},
        "TNotebook.Tab": {"padding": (SPACE["md"], SPACE["sm"] + 2)},
        "Treeview": {"rowheight": 40},
        "Treeview.Heading": {"padding": (SPACE["sm"] + 2, SPACE["sm"] + 2)},
        "TPanedwindow": {"sashwidth": SPACE["sm"]},
        "Horizontal.TProgressbar": {"thickness": SPACE["sm"]},
        "Vertical.TScrollbar": {"arrowsize": RADIUS["lg"]},
        "Horizontal.TScrollbar": {"arrowsize": RADIUS["lg"]},
    }
    for target, options in dimensions.items():
        style.configure(target, **{key: px(root, value) for key, value in options.items()})
    from app.ui.icons import clear_icons
    from app.ui.rounded import apply_composer_shape
    clear_icons(root)
    apply_composer_shape(root, style, c)
    _recolor(root, previous, c)
    return style


def _recolor(widget, previous, current):
    """ttk styles update automatically; explicit Tk colors need refreshing."""
    from app.ui.icons import retint

    retint(widget)
    replacements = {value.lower(): current[key] for key, value in previous.items()}
    for option in ("background", "foreground", "selectbackground", "selectforeground",
                   "insertbackground", "highlightbackground", "highlightcolor",
                   "activebackground", "activeforeground", "troughcolor"):
        if option in widget.keys():
            value = str(widget.cget(option)).lower()
            if value in replacements:
                widget.configure(**{option: replacements[value]})
    if isinstance(widget, tk.Canvas):
        for item in widget.find_all():
            options = widget.itemconfigure(item) or {}
            for option in ("fill", "outline"):
                if option in options:
                    value = widget.itemcget(item, option).lower()
                    if value in replacements:
                        widget.itemconfigure(item, **{option: replacements[value]})
    if isinstance(widget, tk.Text):
        for tag in widget.tag_names():
            for option in ("foreground", "background"):
                value = widget.tag_cget(tag, option).lower()
                if value in replacements:
                    widget.tag_configure(tag, **{option: replacements[value]})
    if isinstance(widget, ttk.Combobox):
        popup = f"{widget}.popdown.f.l"
        if widget.tk.call("winfo", "exists", popup):
            widget.tk.call(popup, "configure", "-background", current["field"],
                           "-foreground", current["text"], "-selectbackground", current["selected"],
                           "-selectforeground", current["text"])
    for child in widget.winfo_children():
        _recolor(child, previous, current)


def _document_columns(root, style):
    """Native cells and headings inherit the palette without tiled images."""
    style.layout("Documents.Treeview.Cell", style.layout("Treeview.Cell"))
    style.layout("Documents.Treeview.Heading", style.layout("Treeview.Heading"))
