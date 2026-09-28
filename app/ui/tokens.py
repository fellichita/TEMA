"""Design tokens for the main2 desktop shell.

One neutral surface family plus a single accent. Light and dark are authored
separately: dark is not an inversion of light, because inverted tonal fills
lose their contrast against body text. Components read these names, never a
literal hex value, so a palette change stays a one-file change.

Contrast budget, verified by tests/ui/test_theme.py:
  * body and muted text     >= 4.5:1 on every surface they are placed on
  * control borders, focus  >= 3.0:1 against the surface behind them
  * accent fills            >= 4.5:1 against ``on_accent``
``divider`` is decorative only and is deliberately below the 3:1 floor; it
never carries meaning that is not also carried by spacing or a label.
"""

LIGHT = {
    # Surfaces, from furthest back to nearest front.
    "background": "#FFFFFF",
    "surface": "#FFFFFF",
    "sidebar": "#F5F7FA",
    "field": "#F7F9FC",
    "raised": "#EDF1F6",
    "skeleton": "#E7ECF3",
    # Lines.
    "border": "#7F8FA3",
    "divider": "#E4E9EF",
    # Text.
    "text": "#1A2433",
    "muted": "#566477",
    "disabled": "#8A94A3",
    # Accent, and the tonal fills derived from it.
    "cyan": "#1F5FBF",
    "accent": "#1F5FBF",
    "accent_hover": "#1A4FA0",
    "accent_pressed": "#163F80",
    "on_accent": "#FFFFFF",
    "blue": "#DCE8F7",
    "hover": "#C9DCF2",
    "selected": "#E3ECF8",
    # Status.
    "error": "#B3261E",
    "success": "#1B6B3A",
    "warning": "#8A5A00",
}

DARK = {
    # The sidebar stays the darkest plane and the content field is lifted above
    # it, so the "navigation is one block on the content field" metaphor reads
    # the same in both themes. Mirroring the roles instead was rejected.
    "background": "#22262E",
    "surface": "#262B34",
    "sidebar": "#12151A",
    "field": "#1B1F26",
    "raised": "#313845",
    "skeleton": "#313845",
    "border": "#7D8896",
    "divider": "#38404D",
    "text": "#EDF0F5",
    "muted": "#AEB8C6",
    "disabled": "#78828F",
    "cyan": "#8FBCF5",
    "accent": "#5A97E8",
    "accent_hover": "#6FA6EE",
    "accent_pressed": "#4886D8",
    "on_accent": "#0E1620",
    "blue": "#2C4359",
    "hover": "#37536E",
    "selected": "#2E3A48",
    "error": "#FF9AA2",
    "success": "#7BD2A0",
    "warning": "#E6B860",
}

PALETTES = {"light": LIGHT, "dark": DARK}

# Bundled first, so the application renders identically on every machine and
# needs no network or system install; the system stack only covers the case of
# the private font failing to register.
FONT_FAMILY = "Open Sans Medium"
FONT_FAMILY_STRONG = "Open Sans SemiBold"
FONT_FALLBACK = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "TkDefaultFont")

# Tk point sizes. The base is 13 because Display.px scales the layout around it.
TEXT = {
    "caption": 11,
    "label": 12,
    "body": 13,
    "lead": 15,
    "section": 19,
    "hero": 24,
    "display": 30,
}

# 8pt rhythm. ``xs`` is the one permitted half-step, for gaps inside a single
# control (icon to its label), never between components.
SPACE = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32, "xxl": 48}

# Padding *inside* a control is a control metric, not page rhythm. 12 is the
# widest horizontal value that still lets the history and period dialogs fit
# every action at 150% and 200% scaling; the 8pt scale above governs the space
# between components, which is what the grid is actually for.
CONTROL = {"pad_x": 12, "pad_y": SPACE["sm"]}

RADIUS = {"sm": 4, "md": 8, "lg": 12}

# Motion durations in milliseconds. Exit is deliberately shorter than enter.
MOTION = {"fast": 120, "base": 180, "slow": 240, "exit": 120}

# Stroke width and box size for app/ui/icons.py, in logical pixels.
ICON = {"stroke": 1.5, "sm": 14, "md": 18, "lg": 24}


def space(*names):
    """Resolve one or more spacing names to logical pixels."""
    return tuple(SPACE[name] for name in names) if len(names) > 1 else SPACE[names[0]]
