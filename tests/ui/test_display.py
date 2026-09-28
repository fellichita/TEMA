"""DPI geometry and event-driven precision scrolling regressions."""

import tkinter as tk
from tkinter import font, ttk
from tkinter.scrolledtext import ScrolledText
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest

from app.ui.display import Display, PixelWheel, scale_value
from app.ui.theme import apply_theme


class PixelValueTests(unittest.TestCase):
    def test_tcl_style_pixel_values_scale_without_numeric_python_protocol(self):
        class PixelValue:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return self.value

        display = Display.__new__(Display)
        display.scale = 1.5
        self.assertEqual(display.px((PixelValue('10'), PixelValue('2.5'))), (15, 4))
        self.assertEqual(display.px([0, '8', 3.0]), (0, 12, 4))


@pytest.mark.gui
class DisplayTests(unittest.TestCase):
    def test_scrolled_text_geometry_scales_once_without_repacking_its_pane(self):
        root = tk.Tk()
        root.withdraw()
        original = root.tk.call('tk', 'scaling')
        try:
            display = Display(root, 2)
            panes = ttk.Panedwindow(root, orient='horizontal')
            panes.pack(fill='both', expand=True, padx=5)
            text = ScrolledText(panes, width=30, height=5, padx=3)
            panes.add(text, weight=1)
            tk.Pack.pack_configure(text, padx=4)
            display.widgets(root)
            display.widgets(root)
            self.assertEqual(text.frame.winfo_manager(), 'panedwindow')
            self.assertEqual(tuple(panes.panes()), (str(text.frame),))
            self.assertEqual(tk.Pack.pack_info(text)['padx'], 8)
            self.assertEqual(int(text.cget('padx')), 6)
            self.assertEqual(int(text.cget('width')), 30)
            self.assertEqual(int(text.cget('height')), 5)
            self.assertEqual(panes.pack_info()['padx'], 10)
        finally:
            root.tk.call('tk', 'scaling', original)
            root.destroy()

    def test_invalid_scales_are_rejected(self):
        for value in ('nan', 'inf', 0, 4):
            with self.assertRaises(ValueError):
                scale_value(value)

    def test_native_layout_scales_without_raster_backgrounds(self):
        for scale in (1, 1.5, 2):
            with self.subTest(scale=scale):
                root = tk.Tk()
                root.withdraw()
                original = root.tk.call('tk', 'scaling')
                try:
                    display = Display(root, scale)
                    style = apply_theme(root)
                    frame = ttk.Frame(root, padding=10)
                    frame.pack(padx=8)
                    tree = ttk.Treeview(frame, columns=('title',), show='headings')
                    tree.column('title', width=200, minwidth=70)
                    tree.pack()
                    text = tk.Text(frame, height=3, width=40)
                    text.pack()
                    for _ in range(2):  # repeated Map events must not compound scaling
                        display.widgets(root)
                        root.update_idletasks()
                        self.assertEqual(tree.column('title', 'width'), round(200 * scale))
                    self.assertEqual(tree.column('title', 'width'), round(200 * scale))
                    self.assertEqual(int(style.lookup('Treeview', 'rowheight')), round(40 * scale))
                    self.assertGreater(int(style.lookup('Treeview', 'rowheight')),
                                       font.nametofont('TkDefaultFont').metrics('linespace'))
                    self.assertFalse(hasattr(root, '_rounded_images'))
                    self.assertEqual(int(text.cget('width')), 40)
                    self.assertEqual(int(text.cget('height')), 3)
                    with patch.object(root, 'winfo_screenwidth', return_value=3840), \
                         patch.object(root, 'winfo_screenheight', return_value=2160):
                        display.window(root, 1280, 880, (940, 740))
                        self.assertEqual(root.minsize(), (round(940 * scale), round(740 * scale)))
                finally:
                    root.tk.call('tk', 'scaling', original)
                    root.destroy()

    def test_window_fits_small_work_area(self):
        root = tk.Tk()
        root.withdraw()
        original = root.tk.call('tk', 'scaling')
        try:
            display = Display(root, 2)
            with patch.object(root, 'winfo_screenwidth', return_value=1280), \
                 patch.object(root, 'winfo_screenheight', return_value=800):
                display.window(root, 1280, 880, (940, 740))
                self.assertEqual(root.minsize(), (1184, 640))
        finally:
            root.tk.call('tk', 'scaling', original)
            root.destroy()

    def test_wheel_preserves_fractional_input_and_coalesces_bursts(self):
        root = tk.Tk()
        root.withdraw()
        try:
            canvas = tk.Canvas(root)
            wheel = PixelWheel(canvas)
            wheel.system = 'win32'
            with patch.object(canvas, 'yview_scroll') as scroll:
                for _ in range(144):
                    wheel.scroll(SimpleNamespace(delta=1, num=None))
                pending = wheel.timer
                self.assertIsNotNone(pending)
                root.update_idletasks()
                self.assertIsNone(wheel.timer)
                self.assertEqual(scroll.call_count, 1)
                # Floating point remainder stays available for the next burst.
                self.assertAlmostEqual(scroll.call_args.args[0] + wheel.pending, -48)
                wheel.scroll(SimpleNamespace(delta=0, num=None))
                canvas.destroy()  # pending idle callback is cancelled with the widget
                root.update_idletasks()
        finally:
            root.destroy()


@pytest.mark.gui
class ApplicationDpiTests(unittest.TestCase):
    def test_full_application_and_late_dialog_at_double_scale(self):
        from app.ui.window import Application
        from tests.ui.test_desktop import FakeBackend
        root = tk.Tk()
        root.withdraw()
        original = root.tk.call('tk', 'scaling')
        errors = []
        root.report_callback_exception = lambda *args: errors.append(args)
        app = Application(root, FakeBackend, ui_scale=2)
        try:
            root.after(300, root.quit)
            root.mainloop()
            self.assertTrue(app.ready)
            self.assertEqual(app.document_tree.column('source', 'width'), 250)
            self.assertEqual(int(app.style.lookup('Treeview', 'rowheight')), 80)
            app.history.new()
            dialog = app.history.form
            root.after(100, root.quit)
            root.mainloop()
            self.assertEqual(int(dialog.fields['topic'].cget('width')), 28)
            self.assertEqual(root.winfo_pixels(app.style.lookup('TButton', 'padding')[0]), 24)
            self.assertEqual(errors, [])
        finally:
            root.tk.call('tk', 'scaling', original)
            app.close()
            root.after(500, root.quit)
            root.mainloop()
