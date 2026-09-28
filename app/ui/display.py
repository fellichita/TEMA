"""Logical UI dimensions and native high-DPI startup configuration."""

import ctypes
import sys
import tkinter as tk
from tkinter import ttk


def enable_high_dpi():
    """Call before Tk on Windows; leave existing host DPI policy intact."""
    if sys.platform != 'win32':
        return
    try:
        # System-aware is supported by Tk 8.6; do not opt old Tk into unsupported
        # per-monitor transitions. Modern host manifests can supply newer policy.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def work_area():
    """Usable desktop size in physical pixels, or None when it is unavailable."""
    if sys.platform != 'win32':
        return None

    class Rect(ctypes.Structure):
        _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                    ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

    try:
        rect = Rect()
        # SPI_GETWORKAREA: the desktop minus the taskbar and other appbars.
        if not ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return None
    except (AttributeError, OSError):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    return (width, height) if width > 0 and height > 0 else None


def screen_size():
    """Physical screen size as the OS reports it, or None off Windows."""
    if sys.platform != 'win32':
        return None
    try:
        metric = ctypes.windll.user32.GetSystemMetrics
        return metric(0), metric(1)  # SM_CXSCREEN, SM_CYSCREEN
    except (AttributeError, OSError):
        return None


def reserved_desktop(window):
    """Pixels the taskbar takes from the screen Tk reports, as (width, height).

    Tests simulate other displays by patching winfo_screenwidth/height. The
    taskbar measurement belongs to the real screen, so it is applied only when
    Tk is reporting that same screen.
    """
    area, real = work_area(), screen_size()
    if not area or not real:
        return 0, 0
    if (window.winfo_screenwidth(), window.winfo_screenheight()) != real:
        return 0, 0
    return max(0, real[0] - area[0]), max(0, real[1] - area[1])


def scale_value(value):
    number = float(value)
    if not .75 <= number <= 3:
        raise ValueError('Масштаб должен быть от 0.75 до 3.0')
    return number


class Display:
    def __init__(self, root, scale=None):
        self.root = root
        native = float(root.tk.call('tk', 'scaling'))
        # Aqua exposes logical points; doubling again would over-scale Retina.
        self.system = root.tk.call('tk', 'windowingsystem')
        baseline = native if self.system == 'aqua' else 96 / 72
        automatic = 1.0 if self.system == 'aqua' else native / baseline
        self.scale = scale_value(scale) if scale is not None else min(3.0, max(.75, automatic))
        if scale is not None:
            root.tk.call('tk', 'scaling', baseline * self.scale)
        root._display = self
        self._scaled = set()
        root.bind_all('<Map>', self._mapped, add='+')

    def px(self, value):
        if isinstance(value, (tuple, list)):
            return tuple(self.px(item) for item in value)
        # Some Tk builds return typed Tcl pixel objects from padding options.
        # They expose their numeric value as text, but do not support float().
        number = value if isinstance(value, (int, float)) else str(value)
        return round(float(number) * self.scale)

    def window(self, window, width, height, minimum=None):
        # Leave room for OS title bars, Dock/taskbar and window borders.
        # winfo_screenheight covers the whole screen, so a window sized from it
        # alone runs under the taskbar and hides whatever sits at its bottom.
        reserved = reserved_desktop(window)
        usable_width = window.winfo_screenwidth() - reserved[0]
        usable_height = window.winfo_screenheight() - reserved[1]
        max_width = max(320, usable_width - self.px(48))
        max_height = max(240, usable_height - self.px(80))
        size = min(self.px(width), max_width), min(self.px(height), max_height)
        # Place it explicitly: a cascaded window near the maximum height ends up
        # partly below the desktop however carefully it was sized.
        left, top = max(0, (usable_width - size[0]) // 2), max(0, (usable_height - size[1]) // 2)
        window.geometry(f'{size[0]}x{size[1]}+{left}+{top}')
        if minimum:
            window.minsize(min(self.px(minimum[0]), max_width), min(self.px(minimum[1]), max_height))

    def _mapped(self, event):
        widget = event.widget
        # New descendants receive their own Map event. Reopening a known page
        # must not recursively traverse its entire widget tree again.
        if isinstance(widget, tk.Misc) and widget not in self._scaled:
            self.widgets(widget)

    def widgets(self, widget):
        """Scale only authored pixel options once; character/line counts stay intact."""
        if widget not in self._scaled:
            self._scaled.add(widget)
            widget.bind('<Destroy>', lambda event, w=widget: self._scaled.discard(w)
                        if event.widget is w else None, add='+')
            if abs(self.scale - 1) > .01:
                for option in ('padding', 'padx', 'pady', 'wraplength', 'length',
                               'spacing1', 'spacing2', 'spacing3'):
                    if option in widget.keys():
                        value = widget.cget(option)
                        values = (value,) if isinstance(value, (int, float)) else widget.tk.splitlist(value)
                        if values:
                            try:
                                scaled = self.px(values)
                                widget.configure(**{option: scaled if len(scaled) > 1 else scaled[0]})
                            except (ValueError, tk.TclError):
                                pass
                manager = widget.winfo_manager()
                if manager in ('pack', 'grid'):
                    # ScrolledText forwards geometry methods to its outer frame;
                    # scale the actual widget identified by winfo_manager/_w.
                    geometry = tk.Pack if manager == 'pack' else tk.Grid
                    info = getattr(geometry, manager + '_info')(widget)
                    options = {}
                    for key in ('padx', 'pady', 'ipadx', 'ipady'):
                        if key in info:
                            value = info[key]
                            values = (value,) if isinstance(value, (int, float)) else widget.tk.splitlist(value)
                            scaled = self.px(values)
                            options[key] = scaled if len(scaled) > 1 else scaled[0]
                    getattr(geometry, manager + '_configure')(widget, **options)
                if isinstance(widget, ttk.Treeview):
                    for column in widget['columns']:
                        widget.column(column, width=self.px(widget.column(column, 'width')),
                                      minwidth=self.px(widget.column(column, 'minwidth')))
        for child in widget.winfo_children():
            self.widgets(child)


def px(root, value):
    display = getattr(root, '_display', None)
    if display is None and hasattr(root, 'winfo_toplevel'):
        # Panels hold a child frame, not the root. Resolve the window they
        # belong to rather than silently returning an unscaled value.
        display = getattr(root.winfo_toplevel(), '_display', None)
    return display.px(value) if display else value


def size_window(window, width, height, minimum=None):
    display = getattr(window._root(), '_display', None)
    if display:
        display.window(window, width, height, minimum)
    else:
        window.geometry(f'{width}x{height}')
        if minimum:
            window.minsize(*minimum)


class PixelWheel:
    """Preserve precision and batch wheel bursts at idle, without a frame-rate cap."""
    def __init__(self, canvas):
        self.canvas = canvas
        self.pending = 0.0
        self.timer = None
        self.system = canvas.tk.call('tk', 'windowingsystem')
        canvas.configure(yscrollincrement=1)
        canvas.bind('<Destroy>', self.close, add='+')

    def scroll(self, event):
        if getattr(event, 'num', None) in (4, 5):
            amount = -40 if event.num == 4 else 40
        else:
            amount = -event.delta * (1 if self.system == 'aqua' else 40 / 120)
        self.pending += amount * getattr(getattr(self.canvas._root(), "_display", None), "scale", 1)
        if self.timer is None:
            self.timer = self.canvas.after_idle(self.flush)

    def flush(self):
        self.timer = None
        pixels = int(self.pending)
        self.pending -= pixels
        if pixels:
            self.canvas.yview_scroll(pixels, 'units')

    def close(self, event):
        if event.widget is self.canvas and self.timer is not None:
            self.canvas.after_cancel(self.timer)
            self.timer = None
