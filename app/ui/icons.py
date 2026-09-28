"""Monochrome line icons rasterised at runtime; no image assets, no network.

Every glyph is described on the same 24x24 grid with one stroke width, so the
set stays visually consistent the way an SVG icon family would. Text glyphs
would depend on whichever font happens to resolve, which is exactly what the
bundled-font policy avoids elsewhere.

Icons here are decorative: they always sit next to a visible label, or on a
control that carries its own tooltip and accessible text. Nothing in the UI
relies on an icon alone to convey state.
"""

import math
import struct
import zlib
from functools import lru_cache

import tkinter as tk

from app.ui.display import px
from app.ui.tokens import ICON

GRID = 24

# Primitives: ("line", x1, y1, x2, y2), ("circle", cx, cy, r),
# ("arc", cx, cy, r, start_degrees, end_degrees).
PATHS: dict[str, tuple[tuple, ...]] = {
    "menu": (("line", 4, 7, 20, 7), ("line", 4, 12, 20, 12), ("line", 4, 17, 20, 17)),
    "chevron-left": (("line", 15, 5, 9, 12), ("line", 9, 12, 15, 19)),
    "chevron-right": (("line", 9, 5, 15, 12), ("line", 15, 12, 9, 19)),
    "chevron-down": (("line", 5, 9, 12, 15), ("line", 12, 15, 19, 9)),
    "chevron-up": (("line", 5, 15, 12, 9), ("line", 12, 9, 19, 15)),
    "arrow-right": (("line", 4, 12, 19, 12), ("line", 13, 6, 19, 12), ("line", 19, 12, 13, 18)),
    "search": (("circle", 11, 11, 6), ("line", 15.5, 15.5, 20, 20)),
    "sliders": (("line", 4, 8, 14, 8), ("line", 18, 8, 20, 8), ("circle", 16, 8, 2),
                ("line", 4, 16, 6, 16), ("line", 10, 16, 20, 16), ("circle", 8, 16, 2)),
    "clock": (("circle", 12, 12, 8), ("line", 12, 7, 12, 12), ("line", 12, 12, 16, 14)),
    "file": (("line", 6, 3, 14, 3), ("line", 14, 3, 18, 7), ("line", 18, 7, 18, 21),
             ("line", 18, 21, 6, 21), ("line", 6, 21, 6, 3), ("line", 14, 3, 14, 7),
             ("line", 14, 7, 18, 7)),
    "layers": (("line", 12, 3, 21, 8), ("line", 21, 8, 12, 13), ("line", 12, 13, 3, 8),
               ("line", 3, 8, 12, 3), ("line", 3, 13, 12, 18), ("line", 12, 18, 21, 13)),
    "activity": (("line", 3, 12, 8, 12), ("line", 8, 12, 11, 5), ("line", 11, 5, 14, 19),
                 ("line", 14, 19, 17, 12), ("line", 17, 12, 21, 12)),
    "calendar": (("line", 4, 6, 20, 6), ("line", 20, 6, 20, 20), ("line", 20, 20, 4, 20),
                 ("line", 4, 20, 4, 6), ("line", 4, 11, 20, 11), ("line", 8, 3, 8, 7),
                 ("line", 16, 3, 16, 7)),
    "chart": (("line", 4, 20, 20, 20), ("line", 4, 20, 4, 4), ("line", 8, 20, 8, 14),
              ("line", 13, 20, 13, 9), ("line", 18, 20, 18, 12)),
    "download": (("line", 12, 4, 12, 15), ("line", 7, 10, 12, 15), ("line", 12, 15, 17, 10),
                 ("line", 4, 20, 20, 20)),
    "upload": (("line", 12, 20, 12, 9), ("line", 7, 14, 12, 9), ("line", 12, 9, 17, 14),
               ("line", 4, 4, 20, 4)),
    "check": (("line", 5, 13, 10, 18), ("line", 10, 18, 19, 6)),
    "alert": (("line", 12, 4, 21, 20), ("line", 21, 20, 3, 20), ("line", 3, 20, 12, 4),
              ("line", 12, 10, 12, 15), ("line", 12, 17.5, 12, 17.6)),
    "info": (("circle", 12, 12, 8), ("line", 12, 11, 12, 16), ("line", 12, 8.4, 12, 8.5)),
    "close": (("line", 6, 6, 18, 18), ("line", 18, 6, 6, 18)),
    "plus": (("line", 12, 5, 12, 19), ("line", 5, 12, 19, 12)),
    "refresh": (("arc", 12, 12, 7, 130, 400), ("line", 17, 3, 17.5, 8), ("line", 17.5, 8, 12.5, 8)),
    "inbox": (("line", 3, 12, 8, 12), ("line", 8, 12, 10, 15), ("line", 10, 15, 14, 15),
              ("line", 14, 15, 16, 12), ("line", 16, 12, 21, 12), ("line", 3, 12, 6, 5),
              ("line", 6, 5, 18, 5), ("line", 18, 5, 21, 12), ("line", 3, 12, 3, 19),
              ("line", 3, 19, 21, 19), ("line", 21, 19, 21, 12)),
}


def _distance(shape, x, y):
    kind = shape[0]
    if kind == "line":
        _, x1, y1, x2, y2 = shape
        dx, dy = x2 - x1, y2 - y1
        length = dx * dx + dy * dy
        t = 0.0 if not length else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
    if kind == "circle":
        _, cx, cy, radius = shape
        return abs(math.hypot(x - cx, y - cy) - radius)
    _, cx, cy, radius, start, end = shape
    angle = math.degrees(math.atan2(y - cy, x - cx)) % 360
    span = (angle - start) % 360
    if span <= (end - start) % 360 or (end - start) % 360 == 0:
        return abs(math.hypot(x - cx, y - cy) - radius)
    return min(math.hypot(x - (cx + radius * math.cos(math.radians(edge))),
                          y - (cy + radius * math.sin(math.radians(edge))))
               for edge in (start, end))


def _chunk(kind, data):
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))


@lru_cache(maxsize=512)
def _alpha_mask(name, size, stroke):
    """Compute geometry once so changing themes only recolours cached pixels."""
    shapes = PATHS[name]
    scale = GRID / size
    half = stroke / 2
    alpha_mask = bytearray()
    for row in range(size):
        for column in range(size):
            coverage = 0.0
            for sub_y in range(3):
                for sub_x in range(3):
                    x = (column + (sub_x + .5) / 3) * scale
                    y = (row + (sub_y + .5) / 3) * scale
                    distance = min(_distance(shape, x, y) for shape in shapes)
                    coverage += max(0.0, min(1.0, half + .5 - distance))
            alpha_mask.append(round(255 * coverage / 9))
    return bytes(alpha_mask)


@lru_cache(maxsize=512)
def icon_png(name, size, color, stroke):
    """Rasterise one glyph as an RGBA PNG, antialiased by 3x3 supersampling."""
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    mask = _alpha_mask(name, size, stroke)
    pixels = bytearray()
    for start in range(0, len(mask), size):
        pixels.append(0)  # PNG filter: None
        for alpha in mask[start:start + size]:
            pixels.extend((red, green, blue, alpha) if alpha else (0, 0, 0, 0))
    header = struct.pack("!2I5B", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(bytes(pixels)))
            + _chunk(b"IEND", b""))


def icon(root, name, *, size="md", color):
    """Return a cached PhotoImage; Tcl owns the image, Python must retain it."""
    pixels = max(8, px(root, ICON[size] if isinstance(size, str) else size))
    store = getattr(root, "_icon_images", None)
    if store is None:
        store = root._icon_images = {}
    key = (name, pixels, color)
    if key not in store:
        # The stroke is kept in grid units, so every size renders the same weight.
        stroke = round(ICON["stroke"] * GRID / ICON["md"], 2)
        store[key] = tk.PhotoImage(master=root, data=icon_png(name, pixels, color, stroke))
    return store[key]


def attach(widget, name, *, size="md", role="text", compound="left"):
    """Place a tinted glyph on a label or button and remember how to redraw it.

    The palette role is stored rather than the colour, so a later theme change
    can re-tint the glyph instead of leaving the previous palette on screen.
    """
    from app.ui.theme import COLORS

    widget._icon_spec = (name, size, role)
    widget.configure(image=icon(widget.winfo_toplevel(), name, size=size, color=COLORS[role]),
                     compound=compound if widget.cget("text") else "image")
    return widget


def retint(widget):
    """Redraw a glyph attached earlier; called after the palette changes."""
    spec = getattr(widget, "_icon_spec", None)
    if spec is None:
        return
    from app.ui.theme import COLORS

    name, size, role = spec
    widget.configure(image=icon(widget.winfo_toplevel(), name, size=size, color=COLORS[role]))


def clear_icons(root):
    """Drop cached images so a theme change can re-tint the whole set."""
    store = getattr(root, "_icon_images", None)
    if store is not None:
        store.clear()
