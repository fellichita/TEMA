"""Visible local spending and explicit owner reconciliation after failures/restore."""

from decimal import Decimal, InvalidOperation
import tkinter as tk
from tkinter import ttk
from typing import Literal, TypedDict

from app.runtime.jobs import TaskFailure
from app.ui.viewport import ScrollViewport


class _Cost(TypedDict):
    cost_micro: int


class _ScopeRow(TypedDict):
    scope_id: str
    currency: str
    used: _Cost
    remaining: _Cost
    requires_reconciliation: bool


class _UnknownRow(TypedDict):
    request_id: str
    currency: str
    charged: _Cost


class _BudgetPage(TypedDict):
    scope_total: int
    unknown_total: int
    reconciliation_required: int
    notice: str
    scopes: list[_ScopeRow]
    unknown_requests: list[_UnknownRow]
    next_scope_after: str | None
    next_request_after: str | None


class _PageState(TypedDict):
    scope_after: str
    request_after: str
    data: _BudgetPage | None
    pending: bool
    refresh_again: bool


def money_micro(value):
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("Сумма слишком длинная.")
    amount = Decimal(value.replace(",", "."))
    if not amount.is_finite() or amount < 0 or amount > 1000000 or (amount != 0 and amount < Decimal("0.000001")):
        raise ValueError("Сумма вне допустимого диапазона.")
    if amount == 0:
        return 0
    if amount.quantize(Decimal("0.000001")) != amount:
        raise ValueError("Введите неотрицательную сумму, не более 6 знаков после запятой.")
    return int(amount * 1000000)


def budget_dialog(panel):
    if panel.active or panel.loading:
        panel.message.set("Журнал бюджета доступен после завершения текущей операции.")
        return
    window = tk.Toplevel(panel.app.root)
    window.title("Расходы и сверка бюджета")
    window.geometry("940x700")
    panel.app.child_windows.append(window)
    viewport = ScrollViewport(window, padding=16)
    viewport.pack(fill="both", expand=True)
    frame = viewport.content
    notice = tk.StringVar(value="Читаем локальный журнал расходов…")
    ttk.Label(frame, textvariable=notice, wraplength=850).pack(anchor="w", pady=8)
    scopes = ttk.Treeview(frame, columns=("scope", "currency", "used", "remaining", "state"), show="headings", height=6)
    for key, title, width in (("scope", "Период / анализ", 340), ("currency", "Валюта", 80), ("used", "Списано / удержано", 130),
                              ("remaining", "Доступно по лимиту", 130), ("state", "Сверка", 90)):
        scopes.heading(key, text=title)
        scopes.column(key, width=width, minwidth=60)
    scopes.pack(fill="x", pady=8)
    unknown = ttk.Treeview(frame, columns=("id", "currency", "held"), show="headings", height=5)
    for key, title, width in (("id", "Запрос с неизвестным ответом", 580), ("currency", "Валюта", 80), ("held", "Удержано", 140)):
        unknown.heading(key, text=title)
        unknown.column(key, width=width, minwidth=60)
    ttk.Label(frame, text="Неизвестные ответы", style="Section.TLabel").pack(anchor="w", pady=(12, 3))
    unknown.pack(fill="x", pady=8)
    state: _PageState = {"scope_after": "", "request_after": "", "data": None, "pending": False, "refresh_again": False}
    feedback = tk.StringVar()
    ttk.Label(frame, textvariable=feedback, wraplength=850).pack(anchor="w", pady=8)
    def render(data: _BudgetPage):
        if not window.winfo_exists():
            return
        state["data"] = data
        notice.set(f"Периодов: {data['scope_total']}. Неизвестных ответов: {data['unknown_total']}. "
                   f"Требуют сверки: {data['reconciliation_required']}. " + data["notice"])
        scopes.delete(*scopes.get_children())
        unknown.delete(*unknown.get_children())
        for item in data["scopes"]:
            scopes.insert("", "end", iid=item["scope_id"], values=(item["scope_id"], item["currency"],
                str(Decimal(item["used"]["cost_micro"]) / 1000000), str(Decimal(item["remaining"]["cost_micro"]) / 1000000),
                "Нужна" if item["requires_reconciliation"] else "Готово"))
        for request in data["unknown_requests"]:
            unknown.insert("", "end", iid=request["request_id"], values=(request["request_id"], request["currency"],
                str(Decimal(request["charged"]["cost_micro"]) / 1000000)))
    def page_states():
        if not window.winfo_exists():
            return
        data = state["data"]
        next_scopes.state(["disabled"] if state["pending"] or not data or not data["next_scope_after"] else ["!disabled"])
        next_requests.state(["disabled"] if state["pending"] or not data or not data["next_request_after"] else ["!disabled"])
        reset_button.state(["disabled"] if state["pending"] else ["!disabled"])
    def settled():
        state["pending"] = False
        page_states()
        if state["refresh_again"]:
            state["refresh_again"] = False
            refresh()
    def load(scope_after: str, request_after: str):
        if state["pending"] or panel.app.closing or not window.winfo_exists():
            return
        state["pending"] = True
        feedback.set("Читаем страницу журнала…")
        page_states()
        def loaded(data: _BudgetPage):
            if not window.winfo_exists():
                return
            state["scope_after"], state["request_after"] = scope_after, request_after
            render(data)
            feedback.set("")
            settled()
        def failed(error):
            if not window.winfo_exists():
                return
            panel.error(error)
            feedback.set(panel.message.get())
            settled()
        if not panel.app.controller.call("pilot-budget-status-" + str(window), "pilot_budget_status", loaded, failed,
                                          scope_after, request_after):
            failed(TaskFailure("Чтение журнала уже выполняется. Повторите после его завершения."))
    def refresh():
        if state["pending"]:
            state["refresh_again"] = True
        else:
            load(state["scope_after"], state["request_after"])
    def next_page(key: Literal["scope_after", "request_after"], result_key: Literal["next_scope_after", "next_request_after"]):
        data = state["data"]
        cursor = data[result_key] if data else None
        if not state["pending"] and cursor:
            requested = {"scope_after": state["scope_after"], "request_after": state["request_after"], key: cursor}
            load(requested["scope_after"], requested["request_after"])
    page_actions = ttk.Frame(frame)
    page_actions.pack(fill="x", pady=8)
    next_scopes = ttk.Button(page_actions, text="Следующие периоды", command=lambda: next_page("scope_after", "next_scope_after"))
    next_scopes.pack(side="left")
    next_requests = ttk.Button(page_actions, text="Следующие запросы", command=lambda: next_page("request_after", "next_request_after"))
    next_requests.pack(side="left", padx=8)
    def reset():
        load("", "")
    reset_button = ttk.Button(page_actions, text="Обновить с начала", command=reset)
    reset_button.pack(side="left")
    def form(kind):
        selected = unknown.selection() if kind == "request" else scopes.selection()
        if not selected:
            feedback.set("Выберите запрос или период в таблице.")
            return
        identifier = selected[0]
        dialog = tk.Toplevel(window)
        dialog.title("Явная сверка владельцем бюджета")
        panel.app.child_windows.append(dialog)
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=identifier, wraplength=680).pack(anchor="w", pady=6)
        entries = {}
        fields = (("input_tokens", "Фактически оплачено входных токенов"), ("output_tokens", "Фактически оплачено выходных токенов"),
                  ("cost_micro", "Фактическая сумма списания в валюте запроса")) if kind == "request" else (
                  ("cost_micro", "Новый допустимый расход сверх всех сохранённых удержаний (0 — без нового запаса)"),)
        for name, title in fields:
            ttk.Label(body, text=title, wraplength=680).pack(anchor="w", pady=(8, 2))
            entry = ttk.Entry(body)
            entry.pack(fill="x")
            entries[name] = entry
        confirmed = tk.BooleanVar(value=False)
        confirmation = ttk.Checkbutton(body, text="Я сверил расходы у провайдера и подтверждаю эти значения", variable=confirmed)
        confirmation.pack(anchor="w", pady=12)
        message = tk.StringVar()
        ttk.Label(body, textvariable=message, wraplength=680).pack(anchor="w", pady=6)
        pending = [False]
        def set_pending(value):
            pending[0] = value
            if dialog.winfo_exists():
                for widget in (*entries.values(), confirmation, submit_button):
                    widget.state(["disabled"] if value else ["!disabled"])
        def failed(error):
            set_pending(False)
            if dialog.winfo_exists():
                panel.error(error)
                message.set(panel.message.get())
        def submit():
            if pending[0]:
                return
            if panel.active or panel.loading or panel.app.closing:
                message.set("Дождитесь завершения текущего анализа или операции перед сверкой.")
                return
            try:
                values = {name: (money_micro(entry.get()) if name == "cost_micro" else int(entry.get()))
                          for name, entry in entries.items()}
                if not confirmed.get() or any(value < 0 for value in values.values()):
                    raise ValueError("Explicit values required")
            except (ValueError, InvalidOperation, OverflowError):
                message.set("Заполните все поля фактическими неотрицательными значениями и подтвердите сверку.")
                return
            def saved(_):
                if dialog.winfo_exists():
                    dialog.destroy()
                refresh()
            set_pending(True)
            message.set("Сохраняем сверенные значения…")
            if kind == "request":
                accepted = panel.app.controller.call("pilot-budget-reconcile-" + str(dialog), "pilot_budget_reconcile",
                    saved, failed, identifier, values | {"confirmed": True})
            else:
                accepted = panel.app.controller.call("pilot-budget-acknowledge-" + str(dialog), "pilot_budget_acknowledge",
                    saved, failed, identifier, values["cost_micro"], True)
            if not accepted:
                failed(TaskFailure("Сверка уже выполняется. Повторите после её завершения."))
        submit_button = ttk.Button(body, text="Подтвердить сверенные значения", command=submit)
        submit_button.pack(anchor="e", pady=12)
    actions = ttk.Frame(frame)
    actions.pack(fill="x", pady=10)
    ttk.Button(actions, text="Сверить выбранный запрос", command=lambda: form("request")).pack(side="left")
    ttk.Button(actions, text="Сверить выбранный период", command=lambda: form("scope")).pack(side="left", padx=8)
    finish_confirmed = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Все периоды сверены; завершить восстановление разрешаю явно", variable=finish_confirmed).pack(anchor="w", pady=8)
    finishing = [False]
    def finish_settled():
        finishing[0] = False
        if window.winfo_exists():
            finish_button.state(["!disabled"])
    def finish_failed(error):
        finish_settled()
        if window.winfo_exists():
            panel.error(error)
            feedback.set(panel.message.get())
    def finish_saved(_):
        finish_settled()
        refresh()
    def finish():
        if finishing[0]:
            return
        if panel.active or panel.loading or panel.app.closing:
            feedback.set("Дождитесь завершения текущего анализа или операции перед сверкой.")
            return
        if not finish_confirmed.get():
            feedback.set("Сначала подтвердите завершение сверки.")
            return
        finishing[0] = True
        finish_button.state(["disabled"])
        feedback.set("Завершаем сверку восстановления…")
        if not panel.app.controller.call("pilot-budget-finish-" + str(window), "pilot_budget_acknowledge", finish_saved,
                                          finish_failed, None, 0, True):
            finish_failed(TaskFailure("Завершение сверки уже выполняется. Повторите после его окончания."))
    finish_button = ttk.Button(frame, text="Завершить сверку восстановления", command=finish)
    finish_button.pack(anchor="e", pady=8)
    refresh()
