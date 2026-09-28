"""Real mapped-window regressions: geometry assertions require an event loop."""

from datetime import date
import time
from types import SimpleNamespace as Record

from app.ui.window import Application
from tests.ui.test_desktop import TkCase


class VisiblePagesTests(TkCase):
    def settle(self, page, widgets=()):
        try:
            self.wait_mapped([page, page.canvas, page.content, *widgets])
        except AssertionError as error:
            raise self.failureException(f"{error}; canvas window=" + str({
                "state": page.canvas.itemcget(page.window, "state"),
                "coords": page.canvas.coords(page.window), "bbox": page.canvas.bbox(page.window),
                "scrollregion": page.canvas.cget("scrollregion"),
                "xview": page.canvas.xview(), "yview": page.canvas.yview(), "layout_pending": page.timer,
            })) from error
        # Wait for the canvas window to accept its requested size and scroll
        # position. A time delay does not establish descendant readiness.
        self.pump(lambda: page.timer is None
                  and page.content.winfo_width() == max(page.canvas.winfo_width(), page.content.winfo_reqwidth())
                  and page.content.winfo_height() == max(page.canvas.winfo_height(), page.content.winfo_reqheight())
                  and page.content.winfo_rootx() == page.canvas.winfo_rootx() - round(page.canvas.canvasx(0))
                  and page.content.winfo_rooty() == page.canvas.winfo_rooty() - round(page.canvas.canvasy(0)))

    def start(self, scale):
        self.focus_events = []
        started = time.monotonic()
        def focus_event(event):
            self.focus_events.append((round(time.monotonic() - started, 3), str(event.type), str(event.widget)))
            self.focus_events[:] = self.focus_events[-24:]
        self.root.bind_all('<FocusIn>', focus_event, add='+')
        self.root.bind_all('<FocusOut>', focus_event, add='+')
        self.app = Application(self.root, lambda: self.backend, ui_scale=scale)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        # main2 opens New analysis; this fixture measures the document workspace.
        self.app.navigation.select("documents")
        self.root.deiconify()
        self.wait_mapped([self.root])
        width, height = self.root.minsize()
        initial_geometry = [self.root.winfo_width(), self.root.winfo_height()]
        resize_started, resize_cpu = time.monotonic(), time.process_time()
        self.root.geometry(f'{width}x{height}')
        self.pump(lambda: (self.root.winfo_width(), self.root.winfo_height()) == (width, height))
        self.settle(self.app.document_tab)
        # tools.measure_resize applies a separate performance gate to this
        # interval after the event loop returns; pump's timeout is functional.
        self.resize_observation = {
            'started': resize_started, 'ended': time.monotonic(),
            'cpu_seconds': time.process_time() - resize_cpu,
            'initial_geometry': initial_geometry, 'geometry': [width, height],
            'rows': len(self.app.document_tree.get_children()),
        }
        # A delayed window-manager FocusIn can reveal the search field and
        # undo the first explicit scroll. Establish focus before testing it.
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root)
        self.settle(self.app.document_tab)

    def assert_reachable(self, page, widgets):
        self.app.tabs.select(page)
        self.settle(page, widgets)
        for widget in widgets:
            with self.subTest(widget=str(widget)):
                self.assertTrue(widget.winfo_viewable())
                self.assertGreater(widget.winfo_width(), 20)
                self.assertGreater(widget.winfo_height(), 15)
                page.reveal(widget)
                self.settle(page, [widget])
                # A control must fit into the viewport after scrolling; large
                # result panes must at least intersect it with useful area.
                x = widget.winfo_rootx() - page.canvas.winfo_rootx()
                y = widget.winfo_rooty() - page.canvas.winfo_rooty()
                self.assertLess(x, page.canvas.winfo_width() - 15)
                self.assertLess(y, page.canvas.winfo_height() - 15, str({
                    "widget": widget.winfo_geometry(), "content": page.content.winfo_geometry(),
                    "viewport": page.canvas.winfo_geometry(), "scrollregion": page.canvas.cget("scrollregion"),
                    "view": page.canvas.yview(), "focus": str(self.root.focus_get())}))
                self.assertGreater(x + widget.winfo_width(), 15)
                self.assertGreater(y + widget.winfo_height(), 15)
                if widget.winfo_class() in ('TButton', 'TEntry'):
                    self.assertGreaterEqual(x, -1)
                    self.assertGreaterEqual(y, -1)
                    self.assertLessEqual(x + widget.winfo_width(), page.canvas.winfo_width() + 1)
                    self.assertLessEqual(y + widget.winfo_height(), page.canvas.winfo_height() + 1)

    def check_scale(self, scale):
        self.start(scale)
        app = self.app
        self.assert_reachable(app.document_tab, [app.previous, app.next, app.refresh,
            app.all_documents, app.open_link, app.versions_button, app.document_tree, app.detail])
        panel = app.trends_panel
        self.assert_reachable(app.trends_tab, [panel.run_button, panel.export_button,
                                             panel.tree, panel.detail])
        self.assert_reachable(app.collection_tab, [app.fields['topic'], app.start_button])
        self.assert_reachable(app.jobs_tab, [app.repeat_button, app.job_tree])
        self.assertEqual(self.callback_errors, [])

    def test_minimum_window_at_100_percent(self):
        self.check_scale(1)

    def test_minimum_window_at_150_percent(self):
        self.check_scale(1.5)

    def test_minimum_window_at_200_percent(self):
        self.check_scale(2)

    def test_repeat_by_space_after_tab_visits_maps_form_and_focus(self):
        self.backend.jobs = [Record(id='repeat', request=Record(topic='фотон', source='crossref',
            from_date=date(2024, 1, 1), until_date=date(2025, 12, 31), max_results=2),
            state='succeeded', updated_at=1, scanned=2, stored=2, skipped=0,
            total_available=7, error_message=None, error_code=None,
            coverage_complete=False, source_exhausted=False)]
        self.start(1)
        app = self.app
        self.pump(lambda: 'repeat' in app.jobs)
        app.job_tree.selection_set('repeat')
        for page in (app.collection_tab, app.document_tab, app.jobs_tab):
            app.tabs.select(page)
            self.settle(page)
        app.repeat_button.focus_force()
        self.pump(lambda: self.root.focus_get() is app.repeat_button)
        self.settle(app.jobs_tab, [app.repeat_button])
        # Deliver keys through the event queue, like native input. The default
        # synchronous dispatch would run handlers between stopped mainloops.
        app.repeat_button.event_generate('<KeyPress-space>', when='tail')
        app.repeat_button.event_generate('<KeyRelease-space>', when='tail')
        self.pump(lambda: app.tabs.select() == str(app.collection_tab))
        self.settle(app.collection_tab, [app.form, app.fields['topic']])
        self.pump(lambda: self.root.focus_get() is app.fields['topic'])
        self.assertEqual(app.tabs.select(), str(app.collection_tab))
        self.assertEqual(app.fields['topic'].get(), 'фотон')
        self.assertTrue(app.form.winfo_viewable())
        self.assertTrue(app.fields['topic'].winfo_viewable())
        self.assertEqual(self.root.focus_get(), app.fields['topic'])
        if self.root.tk.call('tk', 'windowingsystem') == 'win32':
            # The physical F key, as native input sends it: with a Russian
            # layout Tk cannot even build a synthetic <Control-f> («??», code 0).
            self.root.event_generate('<Control-KeyPress>', keycode=0x46, when='tail')
        else:
            self.root.event_generate('<Control-f>', when='tail')
        self.pump(lambda: app.tabs.select() == str(app.document_tab))
        self.settle(app.document_tab, [app.search])
        try:
            self.pump(lambda: self.root.focus_get() is app.search)
        except AssertionError as error:
            raise self.failureException(f"{error}; focus_events={self.focus_events}; "
                                        f"requested_focus={app.document_tab.focus_target}; "
                                        f"layout_pending={app.document_tab.timer}") from error
        self.assertEqual(app.tabs.select(), str(app.document_tab))
        self.assertEqual(self.root.focus_get(), app.search)
        self.assertEqual(self.callback_errors, [])
