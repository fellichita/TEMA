"""Reveal intent survives queued geometry, but yields to explicit user scrolling."""

import gc
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import weakref

from app.ui.viewport import ScrollViewport
from tests.ui.test_desktop import TkCase


class ViewportRevealTests(TkCase):
    def fixture(self):
        page = ScrollViewport(self.root)
        page.pack(fill="both", expand=True)
        fixed = ttk.Frame(page.content, width=900, height=900)
        fixed.pack()
        ancestor = ttk.Frame(fixed, width=180, height=50)
        ancestor.place(x=500, y=550)
        action = ttk.Button(ancestor, text="Action to reveal")
        action.place(x=0, y=0)
        self.root.geometry("700x700")
        self.root.deiconify()
        self.wait_mapped([self.root, page.canvas, page.content, action])
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root)
        self.pump(lambda: page.canvas.winfo_height() > 600 and page.timer is None)
        return page, ancestor, action

    def settled(self, page):
        return (page.timer is None
                and page.content.winfo_width() == max(page.canvas.winfo_width(), page.content.winfo_reqwidth())
                and page.content.winfo_height() == max(page.canvas.winfo_height(), page.content.winfo_reqheight())
                and page.content.winfo_rootx() == page.canvas.winfo_rootx() - round(page.canvas.canvasx(0))
                and page.content.winfo_rooty() == page.canvas.winfo_rooty() - round(page.canvas.canvasy(0)))

    def assert_visible(self, page, action):
        x = action.winfo_rootx() - page.canvas.winfo_rootx()
        y = action.winfo_rooty() - page.canvas.winfo_rooty()
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + action.winfo_width(), page.canvas.winfo_width())
        self.assertLessEqual(y + action.winfo_height(), page.canvas.winfo_height())

    def test_reveal_before_native_resize_keeps_the_action_visible_after_resize(self):
        page, _, action = self.fixture()
        self.root.geometry("450x350")
        # The native Configure is still pending: this target fits the old canvas.
        self.assertGreater(page.canvas.winfo_height(), 600)
        page.reveal(action)
        self.pump(lambda: page.canvas.winfo_height() < 350 and self.settled(page))
        self.assert_visible(page, action)

    def test_ancestor_move_with_unchanged_content_size_keeps_revealed_action_visible(self):
        page, ancestor, action = self.fixture()
        page.reveal(action)
        self.pump(lambda: self.settled(page))
        before = (page.content.winfo_width(), page.content.winfo_height())
        ancestor.place_configure(x=750, y=800)
        self.pump(lambda: ancestor.winfo_y() == 800 and self.settled(page))
        self.assertEqual((page.content.winfo_width(), page.content.winfo_height()), before)
        self.assert_visible(page, action)

    def test_user_scroll_cancels_reveal_and_destruction_releases_target_bindings(self):
        page, ancestor, action = self.fixture()
        ancestor.place_configure(y=800)
        page.reveal(action)
        self.pump(lambda: ancestor.winfo_y() == 800 and self.settled(page))
        self.assert_visible(page, action)
        self.root.tk.call(page.vertical.cget("command"), "moveto", "0")
        ancestor.place_configure(y=850)
        self.pump(lambda: ancestor.winfo_y() == 850 and self.settled(page))
        self.assertEqual(page.canvas.canvasy(0), 0)
        page.reveal(action)
        self.pump(lambda: self.settled(page))
        page._scroll(SimpleNamespace(widget=ancestor, delta=1000, num=4))
        self.pump(lambda: page.wheel.timer is None and self.settled(page))
        old_offset = page.canvas.canvasy(0)
        ancestor.place_configure(y=880)
        self.pump(lambda: ancestor.winfo_y() == 880 and self.settled(page))
        self.assertEqual(page.canvas.canvasy(0), old_offset)
        page.reveal(action)
        reference = weakref.ref(action)
        action.destroy()
        del action
        self.pump(lambda: self.settled(page))
        gc.collect()
        self.assertIsNone(reference())
        page._schedule()
        timer = page.timer
        page.destroy()
        self.assertNotIn(timer, self.root.tk.call("after", "info"))
        self.assertEqual(self.callback_errors, [])

    def test_target_as_large_as_viewport_settles_without_margin_oscillation(self):
        page, _, _ = self.fixture()
        target = ttk.Frame(page.content)
        target.place(x=100, y=200, width=page.canvas.winfo_width(), height=page.canvas.winfo_height())
        self.wait_mapped([target])
        page.reveal(target)
        self.pump(lambda: self.settled(page))
        self.assert_visible(page, target)

    def oversized_fixture(self):
        page = ScrollViewport(self.root)
        page.pack(fill="both", expand=True)
        ttk.Frame(page.content, width=3000, height=3000).pack()
        target = ttk.Frame(page.content)
        target.place(x=900, y=1100, width=1200, height=1200)
        self.root.geometry("700x700")
        self.root.deiconify()
        self.wait_mapped([self.root, page.canvas, page.content, target])
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root and self.settled(page))
        return page, target

    def test_oversized_target_with_only_fifteen_pixels_visible_preserves_that_region_and_fills_viewport(self):
        page, target = self.oversized_fixture()
        for axis in ("x", "y"):
            for edge in ("leading", "trailing"):
                with self.subTest(axis=axis, edge=edge):
                    for direction in ("x", "y"):
                        dimension = "width" if direction == "x" else "height"
                        position = (getattr(target, "winfo_root" + direction)()
                                    - getattr(page.content, "winfo_root" + direction)())
                        extent = getattr(target, "winfo_" + dimension)()
                        visible = getattr(page.canvas, "winfo_" + dimension)()
                        total = getattr(page.content, "winfo_" + dimension)()
                        # Keep the other axis well inside the large target.
                        start = position + 100
                        if direction == axis:
                            start = position - visible + 15 if edge == "leading" else position + extent - 15
                        page._scroll_to(direction, "moveto", start / total)
                    self.pump(lambda: self.settled(page))
                    dimension = "width" if axis == "x" else "height"
                    position = getattr(target, "winfo_root" + axis)() - getattr(page.content, "winfo_root" + axis)()
                    extent = getattr(target, "winfo_" + dimension)()
                    visible = getattr(page.canvas, "winfo_" + dimension)()
                    before = getattr(page.canvas, "canvas" + axis)(0)
                    seen_before = (max(position, before), min(position + extent, before + visible))
                    self.assertEqual(seen_before[1] - seen_before[0], 15)
                    expected = position if edge == "leading" else position + extent - visible
                    page.reveal(target)
                    self.pump(lambda: self.settled(page))
                    after = getattr(page.canvas, "canvas" + axis)(0)
                    self.assertEqual(after, expected)
                    self.assertLessEqual(after, seen_before[0])
                    self.assertGreaterEqual(after + visible, seen_before[1])
                    self.assertGreaterEqual(after, position)
                    self.assertLessEqual(after + visible, position + extent)
                    page.reveal(target)
                    self.pump(lambda: self.settled(page))
                    self.assertEqual(getattr(page.canvas, "canvas" + axis)(0), after)
        self.assertEqual(self.callback_errors, [])

    def test_oversized_target_already_filling_viewport_keeps_both_scroll_offsets(self):
        page, target = self.oversized_fixture()
        for axis in ("x", "y"):
            dimension = "width" if axis == "x" else "height"
            position = getattr(target, "winfo_root" + axis)() - getattr(page.content, "winfo_root" + axis)()
            total = getattr(page.content, "winfo_" + dimension)()
            page._scroll_to(axis, "moveto", (position + 100) / total)
        self.pump(lambda: self.settled(page))
        before = (page.canvas.canvasx(0), page.canvas.canvasy(0))
        for _ in range(3):
            page.reveal(target)
            self.pump(lambda: self.settled(page))
            self.assertEqual((page.canvas.canvasx(0), page.canvas.canvasy(0)), before)
        self.assertEqual(self.callback_errors, [])

    def test_scrolling_nested_text_releases_the_outer_reveal_intent(self):
        page, ancestor, action = self.fixture()
        text = tk.Text(page.content, height=2, width=20)
        text.place(x=20, y=20)
        self.wait_mapped([text])
        page.reveal(action)
        self.pump(lambda: self.settled(page))
        page._scroll(SimpleNamespace(widget=text, delta=1000, num=4))
        old_offset = page.canvas.canvasy(0)
        ancestor.place_configure(y=850)
        self.pump(lambda: ancestor.winfo_y() == 850 and self.settled(page))
        self.assertEqual(page.canvas.canvasy(0), old_offset)

    def test_hiding_a_page_cancels_its_unfulfilled_focus_request(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True)
        first, second = ScrollViewport(tabs), ScrollViewport(tabs)
        tabs.add(first, text="First")
        tabs.add(second, text="Second")
        entry = ttk.Entry(first.content)
        entry.pack()
        self.root.deiconify()
        self.wait_mapped([first.canvas, entry])
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root)
        first.focus_when_visible(entry)
        tabs.select(second)
        self.wait_mapped([second.canvas])
        self.pump(lambda: not first.canvas.winfo_ismapped())
        self.assertIsNone(first.focus_target)
        tabs.select(first)
        self.wait_mapped([first.canvas])
        self.assertIsNot(self.root.focus_get(), entry)
