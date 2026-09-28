"""Antialiased, stretchable ttk control shapes; no additional GUI dependencies."""

import math
from functools import lru_cache
from app.ui.display import px
import struct
import zlib
import tkinter as tk


BACKGROUND_TILE_SIZE = 64


@lru_cache(maxsize=256)
def rounded_png(fill, outline, radius=7, size=20, stroke=1):
    """Render a small RGBA nine-slice tile using 4x supersampling."""
    def rgb(color):
        return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

    inside, edge = rgb(fill), rgb(outline)
    # The extrema of the actual 4x4 sample grid classify uniform pixels exactly.
    # Only corner/edge transitions need sixteen color/coverage samples. Keeping
    # actual sample extrema also handles odd sizes and stroke > radius.
    distances = []
    for coordinate in range(size):
        axis = [max(radius - (coordinate + (sample + .5) / 4),
                    coordinate + (sample + .5) / 4 - (size - radius), 0) for sample in range(4)]
        distances.append((axis, min(axis), max(axis)))
    pixels = bytearray()
    for y in range(size):
        pixels.append(0)  # PNG filter: None
        for x in range(size):
            x_axis, x_min, x_max = distances[x]
            y_axis, y_min, y_max = distances[y]
            nearest = math.hypot(x_min, y_min) - radius
            farthest = math.hypot(x_max, y_max) - radius
            if nearest > 0:
                pixels.extend((0, 0, 0, 0))
                continue
            if farthest <= 0:
                if farthest <= -stroke:
                    pixels.extend((*inside, 255))
                    continue
                if nearest > -stroke:
                    pixels.extend((*edge, 255))
                    continue
            channels = [0, 0, 0]
            coverage = 0
            for dy in y_axis:
                for dx in x_axis:
                    distance = math.hypot(dx, dy) - radius
                    if distance <= 0:
                        coverage += 1
                        color = edge if distance > -stroke else inside
                        for i in range(3):
                            channels[i] += color[i]
            pixels.extend([round(v / coverage) if coverage else 0 for v in channels])
            pixels.append(round(255 * coverage / 16))

    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))

    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', size, size, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def apply_composer_shape(root, style, colors):
    """One rounded outline around the composer; native input stays borderless."""
    radius = max(1, px(root, 14))
    size = 2 * radius + max(4, px(root, 4))
    images = getattr(root, "_composer_images", None)
    data = [rounded_png(colors["field"], colors[border], radius, size, max(1, px(root, 1)))
            for border in ("border", "cyan")]
    if images is None:
        images = root._composer_images = [tk.PhotoImage(master=root, data=value) for value in data]
        style.element_create("Composer.surface", "image", images[0], ("focus", images[1]),
                             border=radius, sticky="nsew")
    else:
        for image, value in zip(images, data, strict=True):
            image.configure(data=value)
    style.layout("Composer.TFrame", [("Composer.surface", {"sticky": "nsew"})])
    style.configure("Composer.TFrame", background=colors["surface"], borderwidth=0)
    style.configure("ComposerContent.TFrame", background=colors["field"])
    style.layout("Composer.TEntry", [("Entry.padding", {"sticky": "nsew", "children": [
        ("Entry.textarea", {"sticky": "nsew"})]})])
    style.configure("Composer.TEntry", padding=px(root, (0, 3)), borderwidth=0,
                    relief="flat", background=colors["field"],
                    insertcolor=colors["text"], insertwidth=max(2, px(root, 2)))
    style.map("Composer.TEntry", fieldbackground=[("disabled", colors["field"]),
                                                  ("readonly", colors["field"])])
    style.layout("Send.TButton", [("Button.padding", {"sticky": "nsew", "children": [
        ("Button.label", {"sticky": "nsew"})]})])
    style.configure("Send.TButton", padding=px(root, (6, 3)), borderwidth=0,
                    background=colors["field"], foreground=colors["text"], anchor="center")
    style.map("Send.TButton", background=[("disabled", colors["field"]),
              ("pressed", colors["field"]), ("active", colors["field"])],
              foreground=[("disabled", colors["disabled"]), ("pressed", colors["cyan"]),
                          ("active", colors["cyan"])])


def apply_rounded_controls(root, style, colors):
    c = colors
    # Tcl owns the elements; Python must retain their PhotoImage objects as long as the root lives.
    images = root._rounded_images = []

    def tile(fill, outline, radius=7, size=20):
        image = tk.PhotoImage(master=root, data=rounded_png(fill, outline, max(1, px(root, radius)),
                                                              max(4, px(root, size)), max(1, px(root, 1))))
        images.append(image)
        return image

    def element(name, fill, states, radius=7):
        base = tile(fill, c['border'], radius, BACKGROUND_TILE_SIZE)
        specs = [(*state, tile(color, border, radius, BACKGROUND_TILE_SIZE)) for state, color, border in states]
        # A larger flat center reduces native tiling; explicit natural dimensions
        # preserve control layout. Indicators and arrows keep their original sizes.
        style.element_create(name, 'image', base, *specs, border=px(root, radius + 1), padding=0,
                             width=px(root, 20), height=px(root, 20), sticky='nsew')
        return name

    def button_layout(name):
        return [(name, {'sticky': 'nsew', 'children': [
            ('Button.padding', {'sticky': 'nsew', 'children': [('Button.label', {'sticky': 'nsew'})]})]})]

    for name, fill, hover in [('TButton', c['raised'], c['selected']),
                              ('Primary.TButton', c['blue'], c['hover'])]:
        shape = element('Rounded.' + name + '.border', fill, [
            (('disabled',), c['surface'] if name == 'TButton' else c['raised'], c['border']),
            (('pressed', 'focus'), c['selected'], c['cyan']),
            (('pressed',), c['selected'], c['border']),
            (('active', 'focus'), hover, c['cyan']),
            (('focus',), fill, c['cyan']), (('active',), hover, c['border'])])
        style.layout(name, button_layout(shape))

    field = element('Rounded.field', c['field'], [
        (('disabled',), c['surface'], c['border']),
        (('focus',), c['field'], c['cyan']), (('active',), c['field'], c['muted'])])
    style.layout('TEntry', [(field, {'sticky': 'nsew', 'children': [
        ('Entry.padding', {'sticky': 'nsew', 'children': [('Entry.textarea', {'sticky': 'nsew'})]})]})])

    def arrow(name, upwards=False):
        image = tk.PhotoImage(master=root, width=px(root, 14), height=px(root, 14))
        for row in range(4):
            y = 5 + row if not upwards else 8 - row
            for x in range(3 + row, 10 - row):
                image.put(c['cyan'], (px(root, x), px(root, y), px(root, x + 1), px(root, y + 1)))
        images.append(image)
        disabled = tk.PhotoImage(master=root, width=px(root, 14), height=px(root, 14))
        for row in range(4):
            y = 5 + row if not upwards else 8 - row
            for x in range(3 + row, 10 - row):
                disabled.put(c['disabled'], (px(root, x), px(root, y), px(root, x + 1), px(root, y + 1)))
        images.append(disabled)
        style.element_create(name, 'image', image, ('disabled', disabled), sticky='')
        return name

    down = arrow('Rounded.Combobox.downarrow')
    # Keep the arrow inside the rounded field and retain the downarrow suffix used by ttk bindings.
    style.layout('TCombobox', [(field, {'sticky': 'nsew', 'children': [
        ('Combobox.padding', {'sticky': 'nsew', 'children': [
            (down, {'side': 'right', 'sticky': 'ns'}),
            ('Combobox.textarea', {'sticky': 'nsew'})]})]})])
    up = arrow('Rounded.Spinbox.uparrow', True)
    down = arrow('Rounded.Spinbox.downarrow')
    style.layout('TSpinbox', [(field, {'sticky': 'nsew', 'children': [
        ('Spinbox.padding', {'sticky': 'nsew', 'children': [
            ('null', {'side': 'right', 'children': [(up, {'side': 'top'}), (down, {'side': 'bottom'})]}),
            ('Spinbox.textarea', {'sticky': 'nsew'})]})]})])
    # A stacked arrow pair already supplies vertical space; avoid inflating the options row.
    style.configure('TSpinbox', padding=(7, 1))
    tab = element('Rounded.Notebook.tab', c['background'], [
        (('selected', 'focus'), c['blue'], c['cyan']),
        (('selected',), c['blue'], c['blue']),
        (('focus',), c['raised'], c['cyan']), (('active',), c['raised'], c['border'])])
    style.layout('TNotebook.Tab', [(tab, {'sticky': 'nsew', 'children': [
        ('Notebook.padding', {'sticky': 'nsew', 'children': [('Notebook.label', {'sticky': 'nsew'})]})]})])
    def checkbox(selected=False, disabled=False):
        fill = c['raised'] if disabled else c['blue'] if selected else c['field']
        image = tile(fill, c['border'], radius=4)
        if selected:
            # Small check mark, kept separate from the stretchable control backgrounds.
            ink = c['disabled'] if disabled else c['text']
            for x, y in ((5, 10), (6, 11), (7, 12), (8, 13), (9, 12),
                         (10, 11), (11, 10), (12, 9), (13, 8), (14, 7)):
                image.put(ink, px(root, (x, y, x + 2, y + 2)))
        return image

    style.element_create('Rounded.Checkbutton.indicator', 'image', checkbox(),
                         ('disabled', 'selected', checkbox(True, True)),
                         ('disabled', checkbox(False, True)), ('selected', checkbox(True)),
                         sticky='')
    style.layout('TCheckbutton', [('Checkbutton.padding', {'sticky': 'nsew', 'children': [
        ('Rounded.Checkbutton.indicator', {'side': 'left', 'sticky': ''}),
        ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [
            ('Checkbutton.label', {'sticky': 'nsew'})]})]})])
    for orientation in ('Horizontal', 'Vertical'):
        name = orientation + '.TScrollbar'
        thumb = element('Rounded.' + orientation + '.Scrollbar.thumb', c['raised'], [
            (('pressed',), c['blue'], c['blue']), (('active',), c['selected'], c['selected'])], radius=6)
        def replace(layout):
            return [(thumb if key.endswith('.thumb') else key,  # noqa: B023 — called synchronously in this iteration
                     {k: replace(v) if k == 'children' else v for k, v in options.items()})
                    for key, options in layout]
        style.layout(name, replace(style.layout(name)))
