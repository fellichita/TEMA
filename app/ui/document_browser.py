"""Paged publication text with a fixed set of mouse/keyboard Tcl callbacks."""

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import NotRequired, TypedDict

from app.runtime.jobs import TaskFailure
from app.ui.theme import COLORS


class Publication(TypedDict):
    title: str
    source: str
    publication_year: NotRequired[int | None]
    url: str


class PublicationPage(TypedDict):
    total: int
    items: list[Publication]


class DocumentBrowser:
    def __init__(self, panel, page: PublicationPage, run_id: str):
        self.panel, self.run_id = panel, run_id
        self.closed = self.pending = self.link_pending = False
        self.offset = self.total = 0
        self.selected: int | None = None
        self.links: list[tuple[str, str, str]] = []
        self.link_tags: dict[str, int] = {}
        self.document_tags: dict[str, int] = {}
        self.window = window = tk.Toplevel(panel.app.root)
        window.title(f"Публикации выборки — всего {page['total']}")
        window.geometry("900x550")
        window.minsize(480, 280)
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        panel.app.child_windows.append(window)
        self.box = ScrolledText(window, wrap="word", padx=12, pady=12, width=1, height=5, takefocus=True)
        self.box.grid(row=0, column=0, sticky="nsew")
        self.box.tag_configure("link", foreground=COLORS["cyan"], underline=True)
        self.box.tag_configure("active-link", background=COLORS["selected"], foreground=COLORS["text"])
        self.box.bind("<Button-1>", self._click)
        self.box.bind("<Up>", lambda event: self._step(-1))
        self.box.bind("<Down>", lambda event: self._step(1))
        self.box.bind("<Return>", self._open)
        self.box.bind("<space>", self._open)
        self.box.bind("<Tab>", self._leave_text)
        self.box.bind("<Shift-Tab>", lambda event: self._leave_text(event, reverse=True))
        self.box.bind("<ISO_Left_Tab>", lambda event: self._leave_text(event, reverse=True))
        self.notice = tk.StringVar(value="↑ / ↓ — выбрать публикацию; Enter / пробел — открыть источник; Tab — перейти к страницам.")
        ttk.Label(window, textvariable=self.notice, wraplength=840).grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 0))
        controls = ttk.Frame(window)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        controls.columnconfigure(1, weight=1)
        self.position = tk.StringVar()
        self.previous = ttk.Button(controls, text="Назад", command=lambda: self._page(-50))
        self.previous.grid(row=0, column=0)
        ttk.Label(controls, textvariable=self.position).grid(row=0, column=1, padx=15)
        self.following = ttk.Button(controls, text="Далее", command=lambda: self._page(50))
        self.following.grid(row=0, column=2)
        window.bind("<Destroy>", self._destroyed, add="+")
        self._render(page)

    def _alive(self):
        return not self.closed and not self.panel.app.closing

    def _destroyed(self, event):
        if event.widget is self.window:
            self.closed = True
            self.links.clear()
            self.link_tags.clear()
            self.document_tags.clear()
            self.selected = None

    def _pagers(self):
        self.previous.state(["disabled"] if self.pending or self.offset == 0 else ["!disabled"])
        self.following.state(["disabled"] if self.pending or self.offset + 50 >= self.total else ["!disabled"])

    def _render(self, page: PublicationPage, requested_offset: int = 0):
        if not self._alive():
            return
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        for tag in (*self.link_tags, *self.document_tags):
            self.box.tag_delete(tag)
        self.links.clear()
        self.link_tags.clear()
        self.document_tags.clear()
        for index, item in enumerate(page["items"]):
            begin = self.box.index("end-1c")
            self.box.insert("end", f"{item['title']}\n{item.get('publication_year') or 'Год неизвестен'} · {item['source']}\n")
            start = self.box.index("end-1c")
            link_tag, document_tag = f"source-{index}", f"document-{index}"
            self.box.insert("end", item["url"], ("link", link_tag))
            end = self.box.index("end-1c")
            self.box.insert("end", "\n\n")
            self.box.tag_add(document_tag, begin, self.box.index("end-1c"))
            self.links.append((item["url"], start, end))
            self.link_tags[link_tag] = index
            self.document_tags[document_tag] = index
        self.box.configure(state="disabled")
        self.offset, self.total = requested_offset, page["total"]
        self.pending = False
        self.window.title(f"Публикации выборки — всего {self.total}")
        self.position.set(f"{self.offset + 1 if self.links else 0}–{self.offset + len(self.links) if self.links else 0} из {self.total}")
        self._select(0)
        self._pagers()

    def _select(self, index: int):
        self.box.tag_remove("active-link", "1.0", "end")
        if not self.links:
            self.selected = None
            return
        self.selected = min(max(0, index), len(self.links) - 1)
        _, start, end = self.links[self.selected]
        self.box.tag_add("active-link", start, end)
        self.box.tag_raise("active-link")
        self.box.mark_set("insert", start)
        self.box.see(start)

    def _step(self, delta: int):
        if self._alive():
            self._select((self.selected if self.selected is not None else 0) + delta)
        return "break"

    def _click(self, event):
        if not self._alive():
            return "break"
        self.box.focus_set()
        tags = self.box.tag_names(self.box.index(f"@{event.x},{event.y}"))
        for tag in tags:
            if tag in self.link_tags:
                self._select(self.link_tags[tag])
                return self._open()
        for tag in tags:
            if tag in self.document_tags:
                self._select(self.document_tags[tag])
                break
        return None

    def _leave_text(self, _event=None, *, reverse=False):
        choices = (self.following, self.previous) if reverse else (self.previous, self.following)
        target = next((button for button in choices if button.instate(["!disabled"])), self.window)
        target.focus_set()
        return "break"

    def _error(self, error):
        if not self._alive():
            return
        self.panel.error(error)
        message = getattr(self.panel, "message", None)
        self.notice.set(message.get() if message is not None else "Операция не завершена. Повторите после проверки подключения и доступа к данным.")

    def _open(self, _event=None):
        if not self._alive() or self.selected is None:
            return "break"
        if self.link_pending:
            self.notice.set("Открываем выбранный источник…")
            return "break"
        url = self.links[self.selected][0]
        self.link_pending = True
        self.notice.set("Открываем источник в браузере…")
        def opened(_):
            self.link_pending = False
            if self._alive():
                self.notice.set("Источник открыт в браузере.")
        def failed(error):
            self.link_pending = False
            self._error(error)
        if not self.panel.app.controller.call("pilot-document-link-" + str(self.window), "open_url", opened, failed, url):
            failed(TaskFailure("Открытие источника уже выполняется. Повторите после его завершения."))
        return "break"

    def _page(self, delta: int):
        if not self._alive() or self.pending:
            return
        requested = max(0, self.offset + delta)
        self.pending = True
        self._pagers()
        self.notice.set("Читаем страницу публикаций…")
        def loaded(page):
            if self._alive():
                self._render(page, requested)
                self.notice.set("Страница загружена. ↑ / ↓ — выбрать публикацию; Enter / пробел — открыть источник.")
        def failed(error):
            self.pending = False
            if self._alive():
                self._error(error)
                self._pagers()
        if not self.panel.app.controller.call("pilot-documents-page-" + str(self.window), "pilot_documents", loaded, failed,
                                              self.run_id, requested):
            failed(TaskFailure("Чтение страницы уже выполняется. Повторите после его завершения."))
