"""Scroll geometry follows content requests even while the viewport stays fixed."""

from tkinter import ttk

from app.ui.viewport import ScrollViewport
from tests.ui.test_desktop import TkCase


class ViewportRequestTests(TkCase):
    def test_result_growth_and_shrink_update_scrollregion_without_resizing_window(self):
        self.check_growth_and_shrink(scroll_right=False)

    def test_shrink_after_horizontal_and_vertical_reveal_remaps_content(self):
        self.check_growth_and_shrink(scroll_right=True)

    def check_growth_and_shrink(self, *, scroll_right):
        page = ScrollViewport(self.root)
        page.pack(fill='both', expand=True)
        label = ttk.Label(page.content, text='Short result')
        label.pack(fill='x')
        action = ttk.Button(page.content, text='Open evidence')
        action.pack(anchor='e' if scroll_right else 'w')
        self.root.geometry('400x300')
        self.root.deiconify()
        self.wait_mapped([self.root, page, page.canvas, page.content, action])
        self.root.focus_force()
        self.pump(lambda: self.root.focus_get() is self.root)

        def settled():
            return (page.timer is None
                    and page.content.winfo_width() == max(page.canvas.winfo_width(), page.content.winfo_reqwidth())
                    and page.content.winfo_height() == max(page.canvas.winfo_height(), page.content.winfo_reqheight()))

        self.pump(settled)
        initial = (page.content.winfo_width(), page.content.winfo_height())
        label.configure(text=('Saved evidence with a longer description. ' * 4 + '\n') * 30)
        self.pump(lambda: page.content.winfo_reqheight() > initial[1])
        self.pump(settled)
        self.assertGreater(page.content.winfo_width(), initial[0])
        self.assertGreater(page.content.winfo_height(), initial[1])
        page.reveal(action)
        self.wait_mapped([action])
        self.pump(lambda: 0 <= action.winfo_rooty() - page.canvas.winfo_rooty()
                  <= page.canvas.winfo_height() - action.winfo_height())
        self.assertGreater(page.canvas.canvasy(0), 0)
        if scroll_right:
            self.assertGreater(page.canvas.canvasx(0), initial[0])
        label.configure(text='Short result')
        try:
            self.pump(lambda: settled() and (page.content.winfo_width(), page.content.winfo_height()) == initial)
        except AssertionError as error:
            geometry = {str(widget): {"geometry": widget.winfo_geometry(),
                        "requested": (widget.winfo_reqwidth(), widget.winfo_reqheight()),
                        "mapped": widget.winfo_viewable()}
                        for widget in (page.canvas, page._window_frame, page.content, label, action)}
            raise AssertionError(f"{error}; initial={initial}; geometry={geometry}; "
                                 f"scrollregion={page.canvas.cget('scrollregion')}; "
                                 f"offset=({page.canvas.canvasx(0)}, {page.canvas.canvasy(0)})") from error
        self.assertEqual(tuple(map(float, page.canvas.cget('scrollregion').split())), (0, 0, *initial))
        self.assertEqual((page.canvas.canvasx(0), page.canvas.canvasy(0)), (0, 0))
        self.wait_mapped([action])
        page._schedule()
        pending = page.timer
        page.destroy()
        self.assertNotIn(pending, self.root.tk.call('after', 'info'))
        self.pump(lambda: not page.winfo_exists())
        self.assertEqual(self.callback_errors, [])

    def test_returning_to_same_size_page_remaps_its_content(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True)
        first, second = ScrollViewport(tabs), ScrollViewport(tabs)
        tabs.add(first, text="Documents")
        tabs.add(second, text="Other page")
        action = ttk.Button(first.content, text="Open evidence")
        action.pack()
        ttk.Frame(first.content, width=900, height=700).pack()
        ttk.Label(second.content, text="Other content").pack()
        self.root.geometry("500x400")
        self.root.deiconify()
        self.wait_mapped([self.root, first.canvas, first.content, action])
        self.pump(lambda: first.timer is None)
        original_size = (first.canvas.winfo_width(), first.canvas.winfo_height())
        original_content = (first.content.winfo_width(), first.content.winfo_height())
        for _ in range(3):
            tabs.select(second)
            self.wait_mapped([second.canvas, second.content])
            self.pump(lambda: not first.canvas.winfo_ismapped())
            tabs.select(first)
            self.wait_mapped([first.canvas, first.content, action])
            self.pump(lambda: first.timer is None)
            self.assertEqual((first.canvas.winfo_width(), first.canvas.winfo_height()), original_size)
            self.assertEqual((first.content.winfo_width(), first.content.winfo_height()), original_content)
        self.assertEqual(self.callback_errors, [])
