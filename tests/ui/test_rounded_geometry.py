"""Native control geometry must fit its text and preserve arrow/action targets."""

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

import pytest

from app.ui.display import Display, px
from app.ui.theme import apply_theme
from tests.ui.test_desktop import TkCase


@pytest.mark.gui
class RoundedGeometryTests(TkCase):
    def verify_geometry(self, scale):
        root = self.root
        Display(root, scale)
        style = apply_theme(root)
        root.geometry('850x650')
        calls = []
        button = ttk.Button(root, text='Действие', command=lambda: calls.append(True))
        button.pack()
        selected = tk.BooleanVar(root)
        checkbox = ttk.Checkbutton(root, text='Выбор', variable=selected)
        checkbox.pack()
        combo = ttk.Combobox(root, values=['Первый', 'Второй'], state='readonly')
        combo.pack()
        spin = ttk.Spinbox(root, from_=1, to=15)
        spin.pack()
        root.deiconify()
        self.wait_mapped([button, checkbox, combo, spin])
        # main2 intentionally has no raster background tiles. Measure the native
        # controls against their real font/padding, not removed image dimensions.
        for control, name, text in ((button, 'TButton', 'Действие'), (checkbox, 'TCheckbutton', 'Выбор'),
                                     (combo, 'TCombobox', 'Первый'), (spin, 'TSpinbox', '15')):
            font = tkfont.Font(root=root, font=(control.cget('font') if 'font' in control.keys()
                                               else style.lookup(name, 'font')))
            padding = tuple(int(value) for value in root.tk.splitlist(style.lookup(name, 'padding')))
            vertical = (2 * padding[0] if len(padding) == 1 else
                        padding[1] + (padding[3] if len(padding) == 4 else padding[1]))
            text_height = font.metrics('linespace')
            self.assertGreaterEqual(control.winfo_reqheight(), text_height + vertical)
            self.assertLessEqual(control.winfo_reqheight(), text_height + vertical + px(root, 8))
            self.assertGreaterEqual(control.winfo_reqwidth(), font.measure(text))
            self.assertEqual(control.winfo_height(), control.winfo_reqheight())
        button.invoke()
        button.state(['disabled'])
        button.invoke()
        self.assertEqual(calls, [True])
        checkbox.invoke()
        self.assertTrue(selected.get())
        for control, arrows in ((combo, ['downarrow']), (spin, ['uparrow', 'downarrow'])):
            self.assert_arrow_hit_regions(control, arrows)

    def test_hit_regions_wait_for_replaced_native_layout(self):
        root = self.root
        Display(root, 1.5)
        apply_theme(root)
        combo = ttk.Combobox(root, values=['Первый', 'Второй'], state='readonly')
        combo.pack()
        root.deiconify()
        self.wait_mapped([combo])
        self.pump(lambda: bool(combo.identify(combo.winfo_width() // 2, combo.winfo_height() // 2)))
        before = (combo.winfo_width(), combo.winfo_height())
        # Tk replaces the native layout synchronously, but places its elements
        # at redisplay idle. Mapping and positive allocation do not prove placement.
        combo.event_generate('<<ThemeChanged>>')
        self.assertTrue(combo.winfo_viewable())
        self.assertEqual((combo.winfo_width(), combo.winfo_height()), before)
        self.assertGreater(before[0], 1)
        self.assertGreater(before[1], 1)
        self.assertEqual(combo.identify(before[0] // 2, before[1] // 2), '')
        self.assert_arrow_hit_regions(combo, ['downarrow'])

    def test_placed_layout_without_arrow_still_fails_hit_assertion(self):
        root = self.root
        style = apply_theme(root)
        style.layout('MissingArrow.TCombobox', [('Combobox.field', {'sticky': 'nsew'})])
        combo = ttk.Combobox(root, style='MissingArrow.TCombobox', state='readonly')
        combo.pack()
        root.deiconify()
        self.wait_mapped([combo])
        with self.assertRaisesRegex(AssertionError, 'downarrow'):
            self.assert_arrow_hit_regions(combo, ['downarrow'])

    def test_natural_geometry_and_hit_targets_at_100_percent(self):
        self.verify_geometry(1)

    def test_natural_geometry_and_hit_targets_at_150_percent(self):
        self.verify_geometry(1.5)

    def test_natural_geometry_and_hit_targets_at_200_percent(self):
        self.verify_geometry(2)
