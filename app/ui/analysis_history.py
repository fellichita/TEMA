"""Inline analysis history backed by existing asynchronous result operations."""

import json
from concurrent.futures import CancelledError
import tkinter as tk
from tkinter import ttk

from app.runtime.jobs import TaskCancelled


def history_view(service, offset=0, limit=50, source="local"):
    """Presentation adapter, run only on the controller worker; never writes data."""
    rows = service.list_runs(offset, limit, source)
    for row in rows:
        row["result_count"] = None
        if row["state"] == "succeeded":
            try:
                row["result_count"] = len(service.result(row["id"])["result"]["cards"])
            except (TaskCancelled, CancelledError):
                raise
            except Exception:
                row["count_message"] = "Результат недоступен; откройте запись для проверки."
    return rows


class AnalysisHistory:
    def __init__(self, panel, parent):
        self.panel, self.app = panel, panel.app
        self.loaded = self.pending = False
        self.dirty = False
        self.offset = 0
        self.generation = 0
        self.rows = {}
        self.source = tk.StringVar(value="На этом компьютере")
        self.loaded_source = self.source.get()
        self.message = tk.StringVar(value="История ещё не загружена.")
        ttk.Label(parent, text="История анализов", style="Section.TLabel").pack(anchor="w", pady=(0, 16))
        self.selector = ttk.Combobox(parent, textvariable=self.source, state="readonly", width=28,
                                    values=("На этом компьютере", "Импорт и экспертные версии"))
        self.selector.pack(anchor="w")
        self.selector.bind("<<ComboboxSelected>>", lambda _: self.request(0))
        ttk.Label(parent, textvariable=self.message, wraplength=520).pack(anchor="w", pady=8)
        listing, self.tree = self.app._tree(parent, [("title", "Направление", 320),
            ("date", "Дата", 155), ("state", "Состояние", 140), ("count", "Результатов", 95)], 8)
        listing.pack(fill="both", expand=True)
        self.tree.bind("<Return>", lambda _: self.open_selected())
        self.tree.bind("<Double-1>", lambda _: self.open_selected())
        self.tree.bind("<<TreeviewSelect>>", self.selected)
        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=12)
        self.open_button = ttk.Button(actions, text="Открыть результат", command=self.open_selected, state="disabled")
        self.open_button.pack(anchor="w", pady=3)
        self.resume_button = ttk.Button(actions, text="Продолжить прерванный анализ", command=self.resume, state="disabled")
        self.resume_button.pack(anchor="w", pady=3)
        ttk.Button(actions, text="Открыть файл результата", command=panel.import_result).pack(anchor="w", pady=3)
        self.refresh_button = ttk.Button(actions, text="Обновить историю", command=lambda: self.request(self.offset))
        self.refresh_button.pack(anchor="w", pady=3)
        pager = ttk.Frame(parent)
        pager.pack(fill="x")
        self.previous = ttk.Button(pager, text="Назад", command=lambda: self.request(max(0, self.offset - 50)), state="disabled")
        self.previous.pack(side="left")
        self.following = ttk.Button(pager, text="Далее", command=lambda: self.request(self.offset + 50), state="disabled")
        self.following.pack(side="left", padx=8)

    def ensure_loaded(self):
        if not self.loaded:
            self.request(0)
        elif self.dirty:
            self.message.set("Есть новые изменения. Нажмите «Обновить историю».")

    def reset(self):
        """Discard rows and pending callbacks from a replaced library."""
        self.generation += 1
        self.loaded = self.pending = self.dirty = False
        self.offset = 0
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        self.selector.configure(state="readonly")
        self.loaded_source = self.source.get()
        for button in (self.open_button, self.resume_button, self.previous, self.following):
            button.state(["disabled"])
        self.message.set("История восстановленной библиотеки ещё не загружена.")

    def request(self, offset):
        if self.pending or not self.app.ready or self.app.closing:
            return
        self.pending = True
        self.generation += 1
        owner = self.generation
        self.selector.configure(state="disabled")
        self.message.set("Открываем сохранённую историю…")
        selected_source = self.source.get()
        source = "local" if selected_source == "На этом компьютере" else "imported"
        def done(rows):
            if self.app.closing or owner != self.generation:
                return
            self.pending = False
            self.loaded, self.dirty = True, False
            self.selector.configure(state="readonly")
            self.loaded_source = selected_source
            self.offset = offset
            self.render(rows)
        def failed(error):
            if owner != self.generation:
                return
            self.pending = False
            if self.app.closing:
                return
            self.selector.configure(state="readonly")
            self.source.set(self.loaded_source)
            self.message.set("Не удалось открыть историю. Сохранённый список оставлен; можно повторить обновление.")
        if not self.app.controller.call("pilot-history-view", "pilot_history_view", done, failed, offset, 50, source):
            failed(None)

    def render(self, rows):
        from app.ui.pilot_panel import STATES
        selected = self.tree.selection()
        scroll = self.tree.yview()
        self.rows = {row["id"]: row for row in rows}
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            saved = json.loads(row["input_json"])
            query = saved.get("payload", saved).get("query", "Анализ")
            self.tree.insert("", "end", iid=row["id"], values=(query, row["created_at"][:16],
                STATES.get(row["state"], row["state"]), row.get("result_count") if row.get("result_count") is not None else "—"))
        if rows:
            identifier = next((key for key in selected if key in self.rows), rows[0]["id"])
            self.tree.selection_set(identifier)
            self.tree.focus(identifier)
            if identifier in selected and scroll:
                self.tree.yview_moveto(scroll[0])
        self.previous.state(["!disabled"] if self.offset else ["disabled"])
        self.following.state(["!disabled"] if len(rows) == 50 else ["disabled"])
        self.message.set(f"{self.offset + 1}–{self.offset + len(rows)} · «—» означает: готовый результат отсутствует или недоступен." if rows else "Сохранённых анализов пока нет.")
        self.selected()

    def selected(self, _=None):
        selected = self.tree.selection()
        row = self.rows.get(selected[0], {}) if selected else {}
        self.open_button.state(["!disabled"] if row else ["disabled"])
        self.resume_button.state(["!disabled"] if row.get("state") in {"failed", "cancelled", "interrupted"}
                                 and not row.get("imported") else ["disabled"])

    def open_selected(self):
        selected = self.tree.selection()
        if not selected or self.app.closing or self.pending or self.panel.active or self.panel.loading:
            if self.panel.active or self.panel.loading:
                self.message.set("Дождитесь текущей операции или отмените анализ.")
            return "break"
        row = self.rows[selected[0]]
        if row["state"] != "succeeded":
            self.message.set(row.get("error") or row.get("message") or "Готового результата нет. Продолжение запускается отдельной кнопкой.")
            return "break"
        identifier = row["id"]
        if not self.panel.open_result(identifier, on_ready=lambda: self.app.navigation.select("analysis")):
            self.message.set("Открытие результата пока недоступно. Повторите после текущей операции.")
        return "break"

    def resume(self):
        selected = self.tree.selection()
        if not selected or self.app.closing or self.pending or self.panel.active or self.panel.loading:
            return
        row = self.rows[selected[0]]
        if row["state"] not in {"failed", "cancelled", "interrupted"} or row.get("imported"):
            return
        saved = json.loads(row["input_json"])
        if self.panel.resume_run(row["id"], query=saved.get("payload", saved).get("query", "Анализ")):
            self.app.navigation.select("analysis")
