"""Native text, heading actions and resize boundaries survive resizing/scrolling."""

from tkinter import ttk

from app.ui.display import Display, px
from app.ui.theme import apply_theme
from tests.ui.test_desktop import TkCase


class DocumentRenderingTests(TkCase):
    def test_native_separator_text_and_header_hits_at_100_percent(self):
        self.check_columns(1)

    def test_native_separator_text_and_header_hits_at_200_percent(self):
        self.check_columns(2)

    def check_columns(self, scale):
        Display(self.root, scale)
        apply_theme(self.root)
        tree = ttk.Treeview(self.root, columns=('title', 'source', 'date'),
                            show='headings', style='Documents.Treeview', height=4)
        clicked = []
        for column in tree['columns']:
            tree.heading(column, text=column.title(), command=lambda c=column: clicked.append(c))
            tree.column(column, width=px(self.root, 160), minwidth=px(self.root, 80), stretch=False)
        for number in range(5):
            tree.insert('', 'end', iid=str(number), values=(f'Research {number}', 'Crossref', '2024'))
        tree.pack(fill='both', expand=True)
        self.root.geometry(f'{px(self.root, 360)}x{px(self.root, 240)}')
        self.root.deiconify()
        self.wait_mapped([self.root, tree])

        def check_cell(column):
            # Native Map can precede Treeview's first item layout on X11.
            # Hit testing requires the cell's actual allocated rectangle.
            self.pump(lambda: len(tree.bbox('0', column)) == 4)
            x, y, width, height = tree.bbox('0', column)
            self.assertGreater(width, px(self.root, 80))
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + width, tree.winfo_width())
            elements = [tree.identify_element(x + offset, y + height // 2) for offset in range(width)]
            # main2 uses the native cell layout; it has no tiled image separator.
            # The actual header resize target must still stay narrow and usable.
            self.assertIn('text', elements)
            heading_regions = [tree.identify_region(x + offset, y // 2) for offset in range(width)]
            trailing_separator = [offset for offset, region in enumerate(heading_regions)
                                  if region == 'separator' and offset > width // 2]
            self.assertTrue(trailing_separator)
            self.assertEqual(trailing_separator[-1], width - 1)
            self.assertLessEqual(len(trailing_separator), px(self.root, 8))
            self.assertEqual(tree.identify_region(x + width - 1, y // 2), 'separator')
            self.assertEqual(tree.identify_region(x + width // 2, y // 2), 'heading')
            return x, y, width, height

        x, y, width, height = check_cell('title')
        tree.event_generate('<Button-1>', x=x + width // 2, y=y + height // 2)
        tree.event_generate('<ButtonRelease-1>', x=x + width // 2, y=y + height // 2)
        self.pump(lambda: tree.selection() == ('0',))
        tree.event_generate('<Button-1>', x=x + width // 2, y=y // 2)
        tree.event_generate('<ButtonRelease-1>', x=x + width // 2, y=y // 2)
        self.pump(lambda: clicked == ['title'])
        tree.column('title', width=px(self.root, 210))
        self.pump(lambda: len(tree.bbox('0', 'title')) == 4
                  and tree.bbox('0', 'title')[2] == px(self.root, 210))
        check_cell('title')
        tree.xview_moveto(1)
        self.pump(lambda: tree.xview()[1] == 1)
        check_cell('date')
        self.assertEqual(self.callback_errors, [])
