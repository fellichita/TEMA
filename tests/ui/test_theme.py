"""Theme contrast and coverage for widgets created after startup."""

import tkinter as tk
from tkinter import font, ttk
import unittest

import pytest

from app.ui.theme import COLORS, PALETTES, apply_theme
from tests.ui.test_desktop import TkCase


def luminance(value):
    channels = [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
    return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722), strict=True))


def ratio(foreground, background):
    light, dark = sorted((luminance(foreground), luminance(background)))
    return (dark + .05) / (light + .05)


class ContrastTests(unittest.TestCase):
    TEXT_PAIRS = {
        "text": ("background", "surface", "field", "raised", "blue", "hover", "selected",
                 "sidebar", "skeleton"),
        "muted": ("background", "surface", "sidebar", "field"),
        "error": ("surface",), "cyan": ("surface",),
        "success": ("surface",), "warning": ("surface",),
        "on_accent": ("accent", "accent_hover", "accent_pressed"),
    }
    # Borders and the focus ring are non-text UI: WCAG asks for 3:1, not 4.5:1.
    OUTLINE_PAIRS = {name: ("surface", "background", "field", "sidebar")
                     for name in ("border", "accent")}

    def test_text_has_readable_contrast_on_all_text_surfaces(self):
        for foreground, backgrounds in self.TEXT_PAIRS.items():
            for background in backgrounds:
                with self.subTest(foreground=foreground, background=background):
                    self.assertGreaterEqual(ratio(COLORS[foreground], COLORS[background]), 4.5)

    def test_both_palettes_meet_the_same_budget(self):
        """Dark is authored separately, so it is measured separately."""
        for name, palette in PALETTES.items():
            for foreground, backgrounds in self.TEXT_PAIRS.items():
                for background in backgrounds:
                    with self.subTest(palette=name, foreground=foreground, background=background):
                        self.assertGreaterEqual(ratio(palette[foreground], palette[background]), 4.5)
            for outline, surfaces in self.OUTLINE_PAIRS.items():
                for surface in surfaces:
                    with self.subTest(palette=name, outline=outline, surface=surface):
                        self.assertGreaterEqual(ratio(palette[outline], palette[surface]), 3.0)


@pytest.mark.gui
class ThemeWidgetTests(TkCase):
    def test_future_dialogs_and_readonly_fields_use_dark_palette(self):
        root = self.root
        root.withdraw()
        try:
            style = apply_theme(root)
            self.assertEqual(font.nametofont('TkDefaultFont').actual('family'), 'Open Sans Medium')
            self.assertEqual(font.nametofont('TkDefaultFont').actual('size'), 13)
            self.assertEqual(font.Font(root, font=style.lookup('Primary.TButton', 'font')).actual('family'), 'Open Sans SemiBold')
            dialog = tk.Toplevel(root)
            dialog.withdraw()
            text = tk.Text(dialog)
            canvas = tk.Canvas(dialog)
            popup = tk.Listbox(dialog)
            self.assertEqual(text.cget("background"), COLORS["field"])
            self.assertEqual(text.cget("foreground"), COLORS["text"])
            self.assertEqual(canvas.cget("background"), COLORS["surface"])
            self.assertEqual(popup.cget("background"), COLORS["field"])
            for widget in ("TCombobox", "TEntry", "TSpinbox"):
                self.assertEqual(style.lookup(widget, "fieldbackground", ("readonly",)), COLORS["field"])
            self.assertEqual(style.lookup("Treeview", "foreground", ("selected",)), COLORS["text"])
            self.assertEqual(style.lookup("Primary.TButton", "background", ("active",)), COLORS["hover"])
        finally:
            root.destroy()

    def test_native_controls_preserve_actions_and_arrow_hit_regions(self):
        root = self.root
        try:
            apply_theme(root)
            calls = []
            button = ttk.Button(root, text='Действие', command=lambda: calls.append(True))
            button.pack()
            button.invoke()
            button.state(['disabled'])
            button.invoke()
            self.assertEqual(calls, [True])
            selected = tk.BooleanVar(root)
            check = ttk.Checkbutton(root, variable=selected)
            check.pack()
            check.invoke()
            self.assertTrue(selected.get())
            combo = ttk.Combobox(root, values=['Первый', 'Второй'], state='readonly')
            combo.pack()
            spin = ttk.Spinbox(root, from_=1, to=15)
            spin.pack()
            root.deiconify()
            # Native Windows Map/Configure events can arrive after idle work.
            # Hit testing requires the controls' actual mapped allocation.
            self.wait_mapped([button, check, combo, spin])
            for control, arrows in ((combo, ['downarrow']), (spin, ['uparrow', 'downarrow'])):
                self.assert_arrow_hit_regions(control, arrows)
        finally:
            root.destroy()
