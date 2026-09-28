"""Explicit expert novelty review; no automatic submission or invented reviewer."""

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import TypedDict, Any, Literal

from app.ui.pilot_materials import background
from app.ui.viewport import ScrollViewport

KINDS = {"Новый механизм": "new_mechanism", "Новая комбинация технологий": "new_combination",
         "Новое применение": "new_application", "Переименование известной технологии": "renamed",
         "Известная технология": "established"}


class _ReviewValues(TypedDict):
    rationale: str
    mechanism_comparison: str
    terminology_review: str
    reviewer_name: str
    kind: str
    current_evidence_ids: list[str]
    earlier_evidence_ids: list[str]
    earlier_analogues_checked: bool
    terminology_changes_checked: bool
    field_reviews: list[dict[str, Any]]


def start_review(panel, run_id, candidate_id):
    if panel.active or panel.loading:
        panel.message.set("Завершите текущую операцию перед обзором ранних аналогов.")
        return
    window = tk.Toplevel(panel.app.root)
    window.title("Ранние аналоги и экспертная оценка новизны")
    window.geometry("960x720")
    panel.app.child_windows.append(window)
    viewport = ScrollViewport(window, padding=18)
    viewport.pack(fill="both", expand=True)
    frame = viewport.content
    message = tk.StringVar(value="Ищем ранние аналоги по зафиксированной формулировке технологии…")
    ttk.Label(frame, textvariable=message, wraplength=860).pack(anchor="w", pady=8)
    job_id: list[str | None] = [None]
    token = panel.begin_loading()
    owner = panel.generation
    if token is None:
        window.destroy()
        return
    poll_after: list[str | None] = [None]
    search_finished = [False]
    cancel_sent = [False]
    def release():
        return panel.finish_loading(token)
    def stop_poll():
        identifier, poll_after[0] = poll_after[0], None
        if identifier is not None:
            try:
                window.after_cancel(identifier)
            except tk.TclError:
                pass
    def cancel_search():
        if job_id[0] and not search_finished[0] and not cancel_sent[0]:
            cancel_sent[0] = True
            panel._call("review-cancel-" + job_id[0], "cancel", lambda _: None, job_id[0])
    def cleanup():
        stop_poll()
        cancel_search()
        release()
    def destroyed(event):
        if event.widget is window:
            cleanup()
    window.bind("<Destroy>", destroyed, add="+")
    def close():
        cleanup()
        window.destroy()
    window.protocol("WM_DELETE_WINDOW", close)
    cancel = ttk.Button(frame, text="Отменить обзор", command=close)
    cancel.pack(anchor="e", pady=8)
    def failed(error):
        if not release():
            return
        panel.error(error)
        if window.winfo_exists():
            message.set(panel.message.get())
    def started(identifier):
        job_id[0] = identifier
        if not window.winfo_exists():
            cancel_search()
            return
        poll()
    def poll():
        poll_after[0] = None
        if not window.winfo_exists() or panel.app.closing:
            return
        # Closing an old dialog does not instantly finish its pending read.
        # A new dialog must still be able to start its own polling chain.
        panel.app.controller.call("pilot-review-progress-" + str(window), "pilot_review_progress", progress, failed, job_id[0])
    def progress(row):
        if not window.winfo_exists():
            return
        message.set(row["message"] or "Проверяем обзор ранних источников…")
        if row["state"] in {"queued", "running"}:
            poll_after[0] = window.after(700, poll)
            return
        search_finished[0] = True
        release()
        if row["state"] == "succeeded":
            cancel.destroy()
            # Collection is complete. Closing the manual form merely hides it;
            # an explicitly submitted review is a save, not a cancellable search.
            window.protocol("WM_DELETE_WINDOW", window.destroy)
            try:
                from app.pilot.antecedents import AntecedentBundle
                from app.pilot.contracts import TrendCard

                prepared = row["review_data"]
                bundle = AntecedentBundle.model_validate(prepared["bundle"])
                card = TrendCard.model_validate(prepared["card"])
                if bundle.candidate_id != card.candidate.candidate_id or bundle.admission_rule_hash != card.candidate.admission_rule_hash:
                    raise ValueError("Review data mismatch")
            except (ValueError, KeyError, TypeError):
                message.set("Сохранённый обзор повреждён или имеет неподдерживаемый формат. Экспертная оценка не записана.")
                return
            render_review_form(panel, window, frame, message, prepared, job_id[0], owner=owner, source_id=run_id)
        else:
            message.set(row["error"] or "Обзор остановлен. Повторное открытие продолжит сохранённую работу.")
    if not panel.app.controller.call("pilot-begin-review", "pilot_begin_review", started, failed, run_id, candidate_id):
        release()
        window.destroy()


def submit_review(panel, window, feedback, job_id, values, *, owner, source_id, on_error=None):
    """Save an explicit old review without replacing a newer analysis or view."""
    if panel.active or panel.app.closing:
        feedback.set("Дождитесь завершения текущего анализа перед сохранением экспертной оценки.")
        return False

    def saved(result):
        if panel.displayed_id is None or panel.displayed_id == source_id:
            panel._imported(result, owner=owner)
        if window.winfo_exists():
            window.destroy()

    def failed(error):
        if window.winfo_exists():
            feedback.set(panel.message.get())
            if on_error is not None:
                on_error(error)

    accepted = background(panel, "review-submit", "apply_review", saved, job_id, values, on_error=failed)
    if accepted and window.winfo_exists():
        feedback.set("Сохраняем экспертную оценку…")
    return accepted


def render_review_form(panel, window, frame, message, prepared, job_id, *, owner, source_id):
    bundle, card = prepared["bundle"], prepared["card"]
    statuses = {"earlier_matches_found": "Найдены более ранние совпадения", "none_found_within_queries": "Ранние совпадения в этом поиске не найдены",
                "incomplete_search": "Поиск ранних аналогов неполный"}
    message.set(statuses[bundle["operational_status"]] + ". Первое наблюдение: "
                + str(bundle["earliest_observed_year"] or "не установлено") + ". "
                "Отсутствие совпадений не доказывает мировую новизну. Ниже требуется самостоятельная оценка эксперта.")
    ttk.Label(frame, text="Кто проводит оценку (имя будет записано в отчёт)").pack(anchor="w", pady=(10, 3))
    reviewer = ttk.Entry(frame)
    reviewer.pack(fill="x")
    ttk.Label(frame, text="Ваш вывод о новизне").pack(anchor="w", pady=(10, 3))
    kind = ttk.Combobox(frame, values=tuple(KINDS), state="readonly")
    kind.pack(fill="x")
    # Deliberately no preselected scientific conclusion.
    current = evidence_choices(panel, frame, "Текущие доказательства: выберите цитаты содержания", card["evidence"])
    earlier = evidence_choices(panel, frame, "Ранние аналоги: выберите рассмотренные источники", bundle["evidence"])
    fields = {}
    for name, title in (("rationale", "Обоснование вывода (от 40 символов)"),
                        ("mechanism_comparison", "Чем механизм отличается от проверенных ранних аналогов (от 40 символов)"),
                        ("terminology_review", "Какие старые названия и переименования проверены (от 30 символов)")):
        ttk.Label(frame, text=title, wraplength=850).pack(anchor="w", pady=(10, 3))
        widget = ScrolledText(frame, height=3, wrap="word")
        widget.pack(fill="x")
        fields[name] = widget
    # No scientific verdict is selected in advance. These controls allow a
    # reviewer to replace an uncertain extract without changing the technology.
    field_controls = field_review_controls(frame, card["evidence"], prepared.get("source_context", {}))
    analogue_check, terminology_check = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    analogue_control = ttk.Checkbutton(frame, text="Я рассмотрел более ранние аналоги и ограничения поиска", variable=analogue_check)
    analogue_control.pack(anchor="w", pady=(12, 4))
    terminology_control = ttk.Checkbutton(frame, text="Я проверил изменения терминологии; новое название само по себе не означает новую технологию",
                                        variable=terminology_check)
    terminology_control.pack(anchor="w", pady=4)
    ttk.Label(frame, text="Оценка сохранится отдельной версией с вашим именем и источниками. "
              "После нажатия «Записать» закрытие окна не отменяет сохранение. "
              "Имя заявлено локально: приложение не удостоверяет личность и не заменяет независимую экспертизу.",
              wraplength=850, style="Muted.TLabel").pack(anchor="w", pady=12)
    feedback = tk.StringVar()
    ttk.Label(frame, textvariable=feedback, wraplength=850).pack(anchor="w", pady=6)
    pending = [False]
    def set_pending(value):
        pending[0] = value
        if window.winfo_exists():
            for widget in (reviewer, kind, current, earlier, analogue_control, terminology_control, submit_button):
                widget.state(["disabled"] if value else ["!disabled"])
            for widget in fields.values():
                widget.configure(state="disabled" if value else "normal")
            for controls in field_controls.values():
                for name in ("verdict", "source", "text_field", "application_kind", "context_control"):
                    widget = controls.get(name)
                    if widget is not None:
                        widget.state(["disabled"] if value else ["!disabled"])
                for name in ("quote", "rationale"):
                    controls[name].configure(state="disabled" if value else "normal")
            for widget in (current, earlier):
                widget.configure(selectmode="none" if value else "extended")
    def submit():
        if pending[0]:
            return
        if kind.get() not in KINDS:
            feedback.set("Выберите свой вывод о новизне.")
            return
        try:
            field_reviews = read_field_reviews(field_controls)
        except ValueError as error:
            feedback.set(str(error))
            return
        values: _ReviewValues = {
            "rationale": fields["rationale"].get("1.0", "end-1c").strip(),
            "mechanism_comparison": fields["mechanism_comparison"].get("1.0", "end-1c").strip(),
            "terminology_review": fields["terminology_review"].get("1.0", "end-1c").strip(),
            "reviewer_name": reviewer.get().strip(), "kind": KINDS[kind.get()],
            "current_evidence_ids": list(current.selection()), "earlier_evidence_ids": list(earlier.selection()),
            "earlier_analogues_checked": analogue_check.get(), "terminology_changes_checked": terminology_check.get(),
            "field_reviews": field_reviews}
        if submit_review(panel, window, feedback, job_id, values, owner=owner, source_id=source_id,
                         on_error=lambda _error: set_pending(False)):
            set_pending(True)
    submit_button = ttk.Button(frame, text="Записать мою оценку и пересчитать статус", style="Primary.TButton", command=submit)
    submit_button.pack(anchor="e", pady=12)


FIELD_VERDICTS: dict[str, Literal["supported", "unverified", "contradicted"]] = {"Подтверждается источником": "supported", "Недостаточно данных": "unverified",
                  "Противоречит источнику": "contradicted"}
TEXT_FIELDS: dict[str, Literal["title", "abstract", "full_text"]] = {"Аннотация": "abstract", "Название": "title", "Полный текст": "full_text"}
APPLICATION_KINDS: dict[str, Literal["research", "demonstrator", "deployment"]] = {"Исследование": "research", "Демонстратор": "demonstrator", "Внедрение": "deployment"}


def field_review_controls(frame, evidence, source_context):
    group = ttk.LabelFrame(frame, text="Проверка смысла полей паспорта — необязательно", padding=10)
    group.pack(fill="x", pady=12)
    ttk.Label(group, text="Оценивайте только проверенные поля. Выберите архивный источник, вставьте точную цитату "
        "из указанного поля и объясните её связь с конкретной технологией. Проверка происхождения текста "
        "не заменяет вашего содержательного решения. Новая оценка заменит прежнее утверждение выбранной роли.",
        wraplength=800).pack(anchor="w", pady=5)
    sources = {f"{index + 1}. {entry['quote'][:90]}": entry["evidence_id"] for index, entry in enumerate(evidence)}
    evidence_by_id = {entry["evidence_id"]: entry for entry in evidence}
    tabs = ttk.Notebook(group)
    tabs.pack(fill="x", pady=5)
    controls = {}
    for role, label in (("problem", "Проблема"), ("advantage", "Преимущество"),
                        ("case", "Кейс исследования / разработки"), ("application", "Стадия применения")):
        box = ttk.Frame(tabs, padding=8)
        tabs.add(box, text=label)
        ttk.Label(box, text="Ваш вывод (оставьте пустым, если поле не проверяли)").pack(anchor="w")
        verdict = ttk.Combobox(box, values=tuple(FIELD_VERDICTS), state="readonly")
        verdict.pack(fill="x")
        ttk.Label(box, text="Архивный источник из паспорта").pack(anchor="w", pady=(5, 0))
        source = ttk.Combobox(box, values=tuple(sources), state="readonly")
        source.pack(fill="x")
        def show_context(selector=source):
            item = evidence_by_id.get(sources.get(selector.get()))
            if item is None:
                return
            archived = source_context.get(item["revision_id"], {})
            window = tk.Toplevel(frame.winfo_toplevel())
            window.title("Архивный контекст выбранного источника")
            window.geometry("820x580")
            text = ScrolledText(window, wrap="word", padx=12, pady=12)
            text.pack(fill="both", expand=True)
            text.insert("end", (archived.get("title", "") + "\n\n" + archived.get("abstract", "")
                if archived else "Полная аннотация не включена в этот старый обзор. Сохранённая цитата:\n\n" + item["quote"])
                + "\n\n" + item["source_url"])
            text.configure(state="disabled")
        ttk.Button(box, text="Посмотреть архивный контекст", command=show_context).pack(anchor="w", pady=4)
        text_field = ttk.Combobox(box, values=tuple(TEXT_FIELDS), state="readonly")
        text_field.pack(fill="x", pady=5)
        ttk.Label(box, text="Точная цитата выбранного поля источника").pack(anchor="w")
        quote = ScrolledText(box, height=3, wrap="word")
        quote.pack(fill="x")
        ttk.Label(box, text="Почему источник подтверждает / не подтверждает это поле (от 40 символов)").pack(anchor="w")
        rationale = ScrolledText(box, height=3, wrap="word")
        rationale.pack(fill="x")
        checked = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(box, text="Я проверил контекст, ограничения и соответствие механизму технологии",
                                variable=checked)
        check.pack(anchor="w", pady=5)
        item: dict[str, Any] = {"verdict": verdict, "source": source, "sources": sources, "text_field": text_field,
                "quote": quote, "rationale": rationale, "context_checked": checked, "context_control": check}
        if role == "application":
            ttk.Label(box, text="Стадия, подтверждённая выбранной цитатой").pack(anchor="w")
            item["application_kind"] = ttk.Combobox(box, values=tuple(APPLICATION_KINDS), state="readonly")
            item["application_kind"].pack(fill="x")
        controls[role] = item
    return controls


def read_field_reviews(controls):
    from app.pilot.review import FieldReview

    result = []
    for role, items in controls.items():
        selected = items["verdict"].get()
        if not selected:
            continue
        source = items["sources"].get(items["source"].get())
        text_field = TEXT_FIELDS.get(items["text_field"].get())
        if source is None or text_field is None:
            raise ValueError("Для проверяемого поля выберите архивный источник и поле текста.")
        application = items.get("application_kind")
        try:
            review = FieldReview(role=role, source_evidence_id=source, text_field=text_field,
                quote=items["quote"].get("1.0", "end-1c").strip(), verdict=FIELD_VERDICTS[selected],
                rationale=items["rationale"].get("1.0", "end-1c").strip(),
                context_checked=items["context_checked"].get(),
                application_kind=APPLICATION_KINDS.get(application.get()) if application is not None else None)
        except ValueError:
            raise ValueError("Заполните точную цитату и обоснование (от 40 символов), подтвердите проверку контекста; "
                             "для стадии применения укажите исследование, демонстратор или внедрение.") from None
        result.append(review.model_dump(mode="json"))
    return result


def evidence_choices(panel, frame, title, evidence):
    ttk.Label(frame, text=title, style="Section.TLabel", wraplength=850).pack(anchor="w", pady=(16, 4))
    tree = ttk.Treeview(frame, columns=("quote", "source"), show="headings", height=5, selectmode="extended")
    tree.heading("quote", text="Цитата (Enter / двойной щелчок — полный текст и ссылка)")
    tree.heading("source", text="Источник")
    tree.column("quote", width=680, minwidth=300)
    tree.column("source", width=100, minwidth=80)
    tree.pack(fill="x")
    items = {item["evidence_id"]: item for item in evidence}
    for identifier, item in items.items():
        tree.insert("", "end", iid=identifier, values=(item["quote"], item["source"]))
    def detail(_=None):
        selected = tree.selection()
        if not selected:
            return
        item = items[selected[0]]
        window = tk.Toplevel(panel.app.root)
        panel.app.child_windows.append(window)
        window.title("Точная цитата источника")
        window.geometry("820x470")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        text = ScrolledText(window, wrap="word", padx=15, pady=15)
        text.grid(row=0, column=0, sticky="nsew")
        text.insert("end", item["quote"] + "\n\n" + item["source_url"])
        text.configure(state="disabled")
        ttk.Button(window, text="Открыть источник", command=lambda:
            panel.app.controller.call("pilot-review-source", "open_url", lambda _: None, panel.error, item["source_url"])).grid(row=1, column=0, sticky="e", padx=15, pady=10)
    tree.bind("<Double-1>", detail)
    tree.bind("<Return>", detail)
    return tree
