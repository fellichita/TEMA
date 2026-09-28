"""The available work area must survive a small display at native large fonts."""

from unittest.mock import patch

from app.ui.window import Application
from tests.ui import test_history_versions, test_visible_pages
from tests.ui.test_desktop import TkCase


class CompactWindowTests(TkCase):
    start = test_visible_pages.VisiblePagesTests.start
    settle = test_visible_pages.VisiblePagesTests.settle
    assert_reachable = test_visible_pages.VisiblePagesTests.assert_reachable

    def test_small_display_at_200_percent_retains_reachable_page_controls(self):
        with patch.object(self.root, "winfo_screenwidth", return_value=1024), \
                patch.object(self.root, "winfo_screenheight", return_value=768):
            test_visible_pages.VisiblePagesTests.check_scale(self, 2)

    def test_small_display_at_200_percent_retains_history_actions_and_table(self):
        with patch.object(self.root, "winfo_screenwidth", return_value=1024), \
                patch.object(self.root, "winfo_screenheight", return_value=768):
            test_history_versions.HistorySelectionTests.check_history_actions_at_minimum_window_size(self, 2)

    def test_resizing_sidebar_keeps_analysis_actions_reachable_and_preserves_drafts(self):
        # Both native sizes fit the hosted Windows display. Crossing the
        # sidebar's logical-width threshold must preserve the existing pages.
        self.app = Application(self.root, lambda: self.backend, ui_scale=.75)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        app = self.app
        app.pilot_panel.query.insert(0, "Черновик направления")
        app.fields["topic"].insert(0, "Сохранённый черновик сбора")
        pages = tuple(app.tabs.winfo_children())
        self.root.minsize(1, 1)
        self.root.deiconify()
        # A hit test is only meaningful while this window is the top one:
        # winfo_containing reports whichever window the desktop has above it.
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.addCleanup(self.drop_topmost)
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root)

        def assert_inside(widget, bounds):
            self.assertTrue(widget.winfo_viewable())
            self.assertGreater(widget.winfo_width(), 20)
            self.assertGreater(widget.winfo_height(), 15)
            x = widget.winfo_rootx() - bounds.winfo_rootx()
            y = widget.winfo_rooty() - bounds.winfo_rooty()
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + widget.winfo_width(), bounds.winfo_width())
            self.assertLessEqual(y + widget.winfo_height(), bounds.winfo_height(), str(widget))
            self.assertIs(self.root.winfo_containing(
                widget.winfo_rootx() + widget.winfo_width() // 2,
                widget.winfo_rooty() + widget.winfo_height() // 2), widget)

        for width, height, collapsed in ((640, 500, True), (900, 650, False), (640, 500, True)):
            with self.subTest(width=width, height=height, collapsed=collapsed):
                self.root.geometry(f"{width}x{height}")
                # The rail lays itself out one step behind the window, so its
                # bottom controls still sit outside the new size for a moment.
                # It fills the window's height: wait until it reports exactly
                # that, or a stale geometry passes for a settled one.
                self.pump(lambda width=width, height=height, collapsed=collapsed:
                          (self.root.winfo_width(), self.root.winfo_height()) == (width, height)
                          and app.navigation.collapsed == collapsed
                          and app.navigation.winfo_height() == height
                          # The rail's stale width would place the buttons over
                          # the page, where a hit test finds the page instead.
                          and app.navigation.winfo_rootx() + app.navigation.winfo_width()
                          <= app.tabs.winfo_rootx())
                self.settle(app.pilot_tab, [app.pilot_panel.query, app.pilot_panel.start_button])
                # The expanded rail shows the brand; mapping it is the window
                # manager's work, so wait for it rather than assume it happened.
                self.wait_mapped([app.navigation, app.navigation.toggle,
                                  app.navigation.buttons["analysis"], app.navigation.buttons["settings"],
                                  *(() if collapsed else (app.navigation.brand,))])
                # The rail is a scrolling viewport of its own and re-lays out
                # after the resize: wait for it, or the buttons are measured
                # while they still hold their previous width and position.
                self.settle(app.navigation.body, [app.navigation.toggle,
                            app.navigation.buttons["analysis"], app.navigation.buttons["settings"]])
                self.assertEqual(bool(app.navigation.brand.winfo_ismapped()), not collapsed)
                self.assertEqual(app.tabs.select(), str(app.pilot_tab))
                self.assertEqual(tuple(app.tabs.winfo_children()), pages)
                self.assertGreater(app.pilot_tab.canvas.winfo_width(), 300)
                self.assertGreater(app.pilot_tab.canvas.winfo_height(), 300)
                rail_controls = (app.navigation.toggle, app.navigation.buttons["analysis"],
                                 app.navigation.buttons["settings"])
                # Controls inside the rail keep the previous width until their
                # own layout pass runs, which would aim the hit test at the page.
                self.pump(lambda rail_controls=rail_controls: all(widget.winfo_rootx() + widget.winfo_width()
                                      <= app.navigation.winfo_rootx() + app.navigation.winfo_width()
                                      for widget in rail_controls))
                for widget in rail_controls:
                    assert_inside(widget, self.root)
                for widget in (app.pilot_panel.query, app.pilot_panel.start_button):
                    app.pilot_tab.reveal(widget)
                    self.settle(app.pilot_tab, [widget])
                    # A field keeps the wider window's size until its own layout
                    # pass runs. Wait for it to land inside the page, then check
                    # where it landed; a field that never fits still fails here.
                    canvas = app.pilot_tab.canvas
                    self.pump(lambda widget=widget, canvas=canvas:
                              widget.winfo_rootx() + widget.winfo_width()
                              <= canvas.winfo_rootx() + canvas.winfo_width())
                    assert_inside(widget, canvas)
                self.assertEqual(app.pilot_panel.query.get(), "Черновик направления")
                self.assertEqual(app.fields["topic"].get(), "Сохранённый черновик сбора")
        self.assertEqual(self.callback_errors, [])
