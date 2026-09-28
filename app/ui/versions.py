"""Read-only version browser; closed windows ignore pending backend responses."""

import json
import tkinter as tk
from tkinter import ttk
from app.ui.display import size_window
from tkinter.scrolledtext import ScrolledText

from app.ui.presentation import SOURCES, document_detail, error_message, publication_date


class VersionsWindow:
    PAGE_SIZE = 50

    def __init__(self, app, snapshot):
        self.app, self.snapshot = app, snapshot
        self.closed = self.loading = False
        self.offset = self.total = 0
        self.items = {}
        self.selected = None
        self._raw_rendered = None
        self.window = tk.Toplevel(app.root)
        self.window.title('Версии документа')
        size_window(self.window, 1000, 700, (720, 520))
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text=snapshot.document.title[:150] + ('…' if len(snapshot.document.title) > 150 else ''), wraplength=650).pack(anchor='w')
        ttk.Label(outer, text='Сохранённые версии из всех источников. Просмотр не изменяет документы задания.',
                  wraplength=850).pack(anchor='w', pady=6)
        self.status = tk.StringVar(value='Читаем версии…')
        ttk.Label(outer, textvariable=self.status, wraplength=850).pack(anchor='w')
        box, self.tree = app._tree(outer, [('source', 'Источник', 120), ('date', 'Публикация', 130),
                                            ('received', 'Получено (UTC)', 150), ('title', 'Название', 300)], 5)
        box.pack(fill='x', pady=8)
        self.tree.bind('<<TreeviewSelect>>', self.select)
        buttons = ttk.Frame(outer)
        buttons.pack(fill='x')
        self.previous = ttk.Button(buttons, text='Назад', command=lambda: self.page(-1), state='disabled')
        self.previous.pack(side='left')
        self.next = ttk.Button(buttons, text='Далее', command=lambda: self.page(1), state='disabled')
        self.next.pack(side='left', padx=6)
        self.refresh = ttk.Button(buttons, text='Обновить', command=self.load)
        self.refresh.pack(side='left')
        self.open_link = ttk.Button(buttons, text='Открыть источник версии', command=self.open_url, state='disabled')
        self.open_link.pack(side='right')
        self.notebook = notebook = ttk.Notebook(outer)
        notebook.pack(fill='both', expand=True, pady=(8, 0))
        self.detail = ScrolledText(notebook, wrap='word', state='disabled', width=50, height=8)
        self.raw = ScrolledText(notebook, wrap='word', state='disabled', width=50, height=8)
        notebook.add(self.detail, text='Карточка версии')
        notebook.add(self.raw, text='Исходные метаданные')
        notebook.bind('<<NotebookTabChanged>>', self.show_raw)
        self.load()

    def alive(self):
        return not self.closed and not self.app.closing

    def load(self):
        if not self.alive() or self.loading:
            return
        self.loading = True
        self.previous.state(['disabled'])
        self.next.state(['disabled'])
        self.refresh.state(['disabled'])
        self.status.set('Читаем версии…')
        self.app.controller.call(('versions', self), 'list_document_versions', self.loaded, self.failed,
                                 self.snapshot.document_key, limit=self.PAGE_SIZE, offset=self.offset)

    def loaded(self, page):
        if not self.alive():
            return
        self.loading = False
        old = self.selected.revision_id if self.selected else self.snapshot.revision_id
        self.items = {item.revision_id: item for item in page.items}
        self.total = page.total
        self.tree.delete(*self.tree.get_children())
        for item in page.items:
            doc = item.document
            self.tree.insert('', 'end', iid=item.revision_id, values=(SOURCES.get(doc.source, doc.source),
                             publication_date(doc), doc.fetched_at.strftime('%d.%m.%Y %H:%M'), doc.title))
        if self.items:
            self.tree.selection_set(old if old in self.items else next(iter(self.items)))
        self.select()
        self.status.set(f'{self.offset + 1}–{self.offset + len(page.items)} из {page.total}. '
                        'Новые сохранённые версии — первыми.' if page.total else 'Сохранённых версий нет.')
        self.navigation()

    def navigation(self):
        self.refresh.state(['!disabled'])
        self.previous.state(['!disabled'] if self.offset else ['disabled'])
        self.next.state(['!disabled'] if self.offset + self.PAGE_SIZE < self.total else ['disabled'])

    def failed(self, error):
        if self.alive():
            self.loading = False
            self.status.set(error_message(error) + ' Нажмите «Обновить», чтобы повторить чтение.')
            self.refresh.state(['!disabled'])
            # Keep old content, but do not navigate using an offset that failed to load.

    def page(self, direction):
        if not self.loading and self.alive():
            self.offset = max(0, self.offset + direction * self.PAGE_SIZE)
            self.load()

    @staticmethod
    def set_text(widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def select(self, event=None):
        if not self.alive():
            return
        selection = self.tree.selection()
        self.selected = self.items.get(selection[0]) if selection else None
        if self._raw_rendered is not self.selected:
            self._raw_rendered = None
            self.set_text(self.raw, 'Откройте вкладку, чтобы показать исходные метаданные.' if self.selected else '')
        self.open_link.state(['!disabled'] if self.selected else ['disabled'])
        text = document_detail(self.selected) if self.selected else 'Выберите версию.'
        if self.selected and self.selected.revision_id == self.snapshot.revision_id:
            text = 'Эта версия показана в исходной карточке.\n\n' + text
        self.set_text(self.detail, text)
        self.show_raw()

    def show_raw(self, event=None):
        if (not self.alive() or not self.selected or self._raw_rendered is self.selected
                or self.notebook.select() != str(self.raw)):
            return
        self.set_text(self.raw, json.dumps(getattr(self.selected.document, 'raw_metadata', {}),
                                          ensure_ascii=False, indent=2))
        self._raw_rendered = self.selected

    def open_url(self):
        if self.alive() and self.selected:
            self.app.controller.call(('version-browser', self), 'open_url', lambda result: None,
                                     self.browser_error, self.selected.document.url)

    def browser_error(self, error):
        if self.alive():
            self.status.set('Не удалось открыть браузер. Скопируйте ссылку из карточки версии.')

    def close(self):
        if not self.closed:
            self.closed = True
            self.loading = False
            # Pending callbacks may retain this object until the backend finishes.
            # Release potentially large revision payloads immediately, and stop
            # the application registry retaining every previously opened page.
            self.items.clear()
            self.selected = self.snapshot = None
            self._raw_rendered = None
            self.app.child_windows[:] = [dialog for dialog in self.app.child_windows if dialog is not self]
            self.window.destroy()
