"""New analysis UI; every network/CPU operation stays outside Tk callbacks."""

import json
import re
from functools import partial
import tkinter as tk
from threading import Event
from tkinter import ttk
from typing import NotRequired, TypedDict

from pydantic import ValidationError

from app.pilot.completion import evaluation_progress
from app.pilot.query import QueryError
from app.ui.icons import attach
from app.runtime.jobs import TaskCancelled, TaskFailure

CATEGORIES = {"confirmed_trend": "Тренд с экспертной проверкой", "early_signal": "Ранняя гипотеза с экспертной проверкой",
              "weak_signal_candidate": "Кандидат в слабые сигналы · автоматическая оценка",
              "emerging_candidate": "Зарождающийся кандидат · автоматическая оценка",
              "renewed_interest": "Новая волна интереса", "established_topic": "Известная технология",
              "unassessed_cluster": "Непроверенная группа", "insufficient_evidence": "Недостаточно доказательств",
              "off_scope": "Не соответствует направлению", "transient_burst": "Всплеск с последующим спадом",
              "declining": "Снижение активности"}
# The same letters the analysis service tests before it refuses a direction it
# cannot translate without an AI key.
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def category_label(card):
    if card.get("methodology_version") is None and card["category"] == "early_signal":
        return "Предварительный кандидат (архив 3.0)"
    return CATEGORIES.get(card["category"], "Неизвестный статус · требуется проверка")


STATES = {"queued": "В очереди", "running": "Выполняется", "succeeded": "Результат сохранён",
          "failed": "Не завершён", "cancelled": "Отменён", "interrupted": "Прерван"}


class _HistoryPageState(TypedDict):
    offset: int
    pending: bool
    source: str


class _RunRow(TypedDict):
    id: str
    state: str
    input_json: str
    created_at: str
    error: str | None
    imported: NotRequired[bool]


class PilotPanel:
    # Comfortable measure for body text, in logical pixels (~75 characters).
    MEASURE = 720

    def __init__(self, app, parent, settings_parent=None, settings_form_parent=None):
        self.app, self.parent = app, parent
        self.run_id: str | None = None
        self.active = self.loading = False
        self.pending_cancel = False
        self.start_cancel: Event | None = None
        self._loading_token = None
        self.loaded = False
        self.payload = None
        self.settings_status = None
        # The settings and keys form lives on the settings page, not in a window.
        # It is built once, when the first status arrives with what it needs.
        self.settings_form_parent = settings_form_parent
        self.settings_form = None
        self.cards = {}
        self.generation = 0
        # The direction this local translation belongs to, so an edited
        # query is never started with the previous wording's formulation.
        self._translation: tuple[str, str] | None = None
        self.heading = ttk.Label(parent, text="Какое направление исследуем?", style="Hero.TLabel", wraplength=620, width=1, anchor="center")
        self.heading.pack(fill="x", pady=(72, 24))
        composer = ttk.Frame(parent, padding=8, style="Card.TFrame")
        composer.pack(fill="x", padx=16)
        self.query_text = tk.StringVar()
        entry_frame = ttk.Frame(composer)
        entry_frame.pack(side="left", fill="x", expand=True)
        self.query = ttk.Entry(entry_frame, textvariable=self.query_text, width=1)
        self.query.pack(fill="x")
        self.placeholder = ttk.Label(entry_frame, text="Введите технологическое направление…", style="Placeholder.TLabel")
        self.placeholder.place(x=10, rely=.5, anchor="w")
        self.placeholder.bind("<Button-1>", lambda _: self.query.focus_set())
        self.query_text.trace_add("write", self._placeholder)
        self.query.bind("<Return>", lambda _: self.start())
        self.start_button = ttk.Button(composer, style="Accent.TButton",
                                       command=self.start, state="disabled")
        attach(self.start_button, "arrow-right", size="sm", role="on_accent")
        self.start_button.pack(side="right", padx=(8, 0))
        from app.ui.navigation import Tooltip
        Tooltip(self.start_button, "Начать анализ · Enter")
        self.examples = ttk.Frame(parent)
        self.examples.pack(fill="x", padx=16, pady=(10, 22))
        self.example_buttons: list[ttk.Button] = []
        for text in ("Квантовые сенсоры", "Хранение энергии", "Биоматериалы"):
            button = ttk.Button(self.examples, text=text, style="Chip.TButton",
                                command=partial(self._example, text))
            button.grid(row=len(self.example_buttons), column=0, sticky="w", padx=(0, 6), pady=2)
            self.example_buttons.append(button)
        controls = settings_parent if settings_parent is not None else ttk.Frame(parent)
        ttk.Label(controls, text="Анализ и данные", style="Section.TLabel").pack(anchor="w", pady=(0, 12))
        self.manual = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Ручной поиск с английской формулировкой",
                        variable=self.manual, command=self._manual_changed).pack(anchor="w", pady=(8, 0))
        self.english = ttk.Entry(controls)
        self.english.pack(fill="x", pady=(4, 6))
        self.english.configure(state="disabled")
        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=8)
        self.execution = ttk.Frame(parent, padding=(16, 8))
        self.direction = tk.StringVar()
        self.direction_label = ttk.Label(self.execution, textvariable=self.direction, style="Section.TLabel", wraplength=600, width=1)
        self.direction_label.pack(fill="x")
        self.cancel_button = ttk.Button(self.execution, text="Отменить", command=self.cancel, state="disabled")
        self.cancel_button.pack(anchor="w", pady=8)
        self.history_button = ttk.Button(actions, text="История анализов", command=self.history)
        self.history_button.pack(side="left")
        from app.ui.pilot_signals import open_signals
        ttk.Button(actions, text="Сигналы из CSV и arXiv",
                   command=lambda: open_signals(self.app)).pack(side="left", padx=6)
        self.message = tk.StringVar(value="Открываем настройки анализа…")
        self.message_label = ttk.Label(parent, textvariable=self.message, wraplength=600, width=1)
        self.message_label.pack(fill="x", padx=16, pady=6)
        self.setup_action = ttk.Button(parent, text="Открыть настройки анализа", command=self.settings)
        self.manual_note = ttk.Label(parent, text="Используется английская формулировка из настроек.", style="Muted.TLabel", wraplength=600)
        self.progress_text = tk.StringVar()
        ttk.Label(parent, textvariable=self.progress_text, style="Muted.TLabel").pack(anchor="w", padx=16)
        self.progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.scope = tk.StringVar()
        self.scope_label = ttk.Label(parent, textvariable=self.scope, wraplength=600, width=1, style="Muted.TLabel")
        self.scope_label.pack(fill="x", padx=16, pady=6)
        self.summary = tk.StringVar()
        self.summary_label = ttk.Label(parent, textvariable=self.summary, wraplength=600, width=1)
        self.summary_label.pack(fill="x", padx=16, pady=(0, 8))
        # The full sentence on one line widened the whole page: at 200% scale it
        # pushed the direction field and the start button out of the window.
        self.queue_button = ttk.Button(parent, text="Непроверенные гипотезы",
                                       command=self.show_candidate_queue, state="disabled")
        Tooltip(self.queue_button, "Непроверенные гипотезы вне основного списка")
        self.queue_button.pack(anchor="w", padx=16, pady=(0, 6))
        # Preserve native selection for passport/legacy actions; main2 displays
        # the bounded cards, so this compatibility model is intentionally hidden.
        self.result_tabs = ttk.Notebook(parent)
        self.tree, self.other_tree = (self._result_table(title) for title in ("TOP-15", "Остальные кандидаты"))
        self.result_tabs.bind("<<NotebookTabChanged>>", self._selection)
        from app.ui.result_cards import ResultCards
        self.card_list = ResultCards(self, parent)
        self.result_actions = result_actions = ttk.Frame(parent)
        self.passport_button = ttk.Button(result_actions, text="Паспорт и доказательства", command=self.passport, state="disabled")
        self.passport_button.pack(anchor="w", pady=3)
        self.documents_button = ttk.Button(result_actions, text="Публикации выборки", command=self.documents, state="disabled")
        self.documents_button.pack(anchor="w", pady=3)
        self.export_button = ttk.Button(result_actions, text="Экспорт результата", command=self.export, state="disabled")
        self.export_button.pack(anchor="w", pady=3)
        ttk.Button(controls, text="Открыть файл результата", command=self.import_result).pack(anchor="w", pady=6)
        library_actions = ttk.Frame(controls)
        library_actions.pack(fill="x", pady=(0, 8))
        from app.ui.pilot_materials import backup_dialog, restore_dialog, show_library
        ttk.Button(library_actions, text="Отчёты и препринты", command=lambda: show_library(self)).pack(anchor="w", pady=3)
        ttk.Button(library_actions, text="Резервная копия", command=lambda: backup_dialog(self)).pack(anchor="w", pady=3)
        ttk.Button(library_actions, text="Восстановить копию", command=lambda: restore_dialog(self)).pack(anchor="w", pady=3)
        from app.ui.pilot_budget import budget_dialog
        ttk.Button(library_actions, text="Расходы и сверка", command=lambda: budget_dialog(self)).pack(anchor="w", pady=3)
        self.app.tabs.bind("<<NotebookTabChanged>>", self._tab, add="+")
        parent.bind("<Configure>", self._resize, add="+")

    def _placeholder(self, *_):
        if self.query_text.get():
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=10, rely=.5, anchor="w")

    def _example(self, text):
        if not self.active and not self.loading:
            self.query_text.set(text)
            self.query.focus_set()

    def _resize(self, event):
        from app.ui.display import px
        width = max(px(self.app.root, 180), event.width - px(self.app.root, 48))
        if getattr(self, "_wrap_width", None) == width:
            return
        self._wrap_width = width
        columns = 3 if sum(button.winfo_reqwidth() for button in self.example_buttons) + px(self.app.root, 18) < width else 1
        for index, button in enumerate(self.example_buttons):
            button.grid_configure(row=index // columns, column=index % columns)
        # Cap the measure rather than letting prose run the full window width:
        # past roughly 75 characters the eye loses the start of the next line.
        text_width = min(width, px(self.app.root, self.MEASURE))
        for label in (self.heading, self.message_label, self.manual_note, self.direction_label, self.scope_label, self.summary_label):
            label.configure(wraplength=text_width)

    def set_compact(self, compact):
        """A short window spends its height on content, not on the hero."""
        from app.ui.display import px
        if getattr(self, "_compact", None) == compact:
            return
        self._compact = compact
        if not self.active:
            top = px(self.app.root, 24 if compact else 72)
            self.heading.pack_configure(pady=(top, px(self.app.root, 24)))

    def _show_execution(self):
        self.heading.pack_configure(pady=(20, 16))
        self.examples.pack_forget()
        self.execution.pack(fill="x", before=self.message_label)
        self.direction.set(self.query.get())

    def _history_changed(self):
        if hasattr(self.app, "analysis_history"):
            self.app.analysis_history.dirty = True

    def _result_table(self, title):
        frame = ttk.Frame(self.result_tabs)
        self.result_tabs.add(frame, text=title)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=("title", "category", "studies"), show="headings", height=8,
                           selectmode="browse")
        for name, heading, width in (("title", "Технология / кандидат", 400), ("category", "Статус", 240),
                                     ("studies", "Исследований за 3 года", 160)):
            tree.heading(name, text=heading)
            tree.column(name, width=width, minwidth=100, stretch=name != "studies")
        tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, command=tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.bind("<Double-1>", lambda _: self.passport())
        tree.bind("<Return>", lambda _: self.passport())
        tree.bind("<<TreeviewSelect>>", self._selection)
        return tree

    @property
    def selected_result_tree(self):
        return self.tree if self.result_tabs.index("current") == 0 else self.other_tree

    def clear_result_tables(self):
        for tree in (self.tree, self.other_tree):
            tree.delete(*tree.get_children())
        self.result_tabs.tab(0, text="TOP-15")
        self.result_tabs.tab(1, text="Остальные кандидаты")
        self.queue_button.state(["disabled"])

    def _tab(self, _=None):
        if (not self.app.closing and self.app.ready and not self.loaded
                and self.app.tabs.select() == str(self.app.pilot_tab)):
            self.load()

    @property
    def displayed_id(self):
        """Identity of the payload actually on screen, independent of pending work."""
        return self.payload.get("view_id", self.payload["result"]["run_id"]) if self.payload else None

    @property
    def active_run_id(self):
        return self.run_id if self.active else None

    def _call(self, key, method, callback, *args, owner=None, **kwargs):
        if self.app.closing:
            return False
        def current():
            return not self.app.closing and (owner is None or owner == self.generation)
        def completed(value):
            if current():
                callback(value)
        def failure(error):
            if not current():
                return
            self.error(error, stop=method in {"start", "resume", "refine_candidate"})
            if method == "get" and self.active and not self.app.closing:
                self.app.root.after(1500, self._poll)
        return self.app.controller.call("pilot-" + key, "pilot_" + method, completed, failure, *args, **kwargs)

    def load(self):
        if not self.app.ready or self.app.closing:
            return
        # Place models this computer already holds before asking what is missing,
        # so a fresh profile opens ready instead of offering an installation that
        # would download what is already here. Only local copying happens here.
        self._call("prepare", "prepare_local_models", self._prepared)

    def _prepared(self, _result):
        self._call("status", "status", self._status)

    def _status(self, result):
        self.loaded = True
        self.settings_status = result
        self._settings_form(result)
        if "keys" in result and hasattr(self.app, "refresh_source_credentials"):
            self.app.refresh_source_credentials(result["keys"])
        self.start_button.state(["disabled"] if self.active or self.loading else ["!disabled"])
        if not self.active:
            bundled = result.get("model_origin") == "bundled"
            missing_model = ((result.get("model_error") or "Встроенная модель недоступна. Переустановите приложение.")
                             if bundled else "Для первого анализа загрузите локальную модель в настройках (около 490 МБ).")
            self.message.set(result.get("settings_error") or
                             ("Готово к поиску. Сохранённые результаты доступны без интернета." if result["model_installed"]
                              else missing_model))
            if (not bundled and not result["model_installed"]) or result.get("settings_error"):
                self.setup_action.pack(anchor="w", padx=16, after=self.message_label)
            else:
                self.setup_action.pack_forget()

    def _needs_local_translation(self):
        """Report whether this run has to translate the direction by itself.

        Without an AI key the service cannot read a Russian direction. An
        unreadable key store leaves the decision to the service, and a manual
        English formulation is always the user's own wording.
        """
        status = self.settings_status or {}
        keys = status.get("keys") or {}
        provider = (status.get("settings") or {}).get("provider")
        if provider == "local":
            return False  # The local model reads the direction in Russian itself.
        name = "yandex_api_key" if provider == "yandex" else "deepseek_api_key"
        if keys.get(name) is not False or not _CYRILLIC.search(self.query.get()):
            return False
        return not (self.manual.get() and self.english.get().strip())

    def _translate_then_start(self):
        """Translate locally and continue the same start, without a confirmation.

        The draft used to wait in the English field for a second press. The run
        now begins as soon as the formulation exists; the result records it as a
        machine translation, and the direction keeps its Russian wording.
        """
        token = self.begin_loading()
        if token is None:
            return
        self.message.set("Готовим поисковую формулировку…")
        generation = self.generation
        # The wording as it was sent: editing the field while the translator
        # works must not relabel this draft as the new direction's formulation.
        requested = self.query.get()

        def translated(result):
            if self.app.closing or generation != self.generation:
                return
            self.finish_loading(token)
            self._translation = (requested, result["english_query"])
            self.start()

        def failed(error):
            if self.app.closing or generation != self.generation:
                return
            self.finish_loading(token)
            self._offer_manual_field()
            self.error(error)

        if not self.app.controller.call("pilot-translate", "pilot_translate_query",
                                        translated, failed, requested):
            self.finish_loading(token)
            self._offer_manual_field()
            self.message.set("Впишите английскую поисковую формулировку ниже — русское "
                             "название направления сохранится в результате.")

    def _offer_manual_field(self):
        """Only a failed translation asks the user for the wording."""
        if not self.manual.get():
            self.manual.set(True)
            self._manual_changed()
        self.english.focus_set()

    def _manual_changed(self):
        self.english.configure(state="normal" if self.manual.get() and not self.active else "disabled")
        if self.manual.get():
            self.manual_note.pack(anchor="w", padx=16, after=self.message_label)
        else:
            self.manual_note.pack_forget()

    def begin_loading(self):
        if self.loading or self.app.closing:
            return None
        token = object()
        self._loading_token = token
        self.loading = True
        self.start_button.state(["disabled"])
        return token

    def finish_loading(self, token):
        if token is None or token is not self._loading_token:
            return False
        self._loading_token = None
        self.loading = False
        self.start_button.state(["disabled"] if self.active or not self.loaded else ["!disabled"])
        return True

    def _busy(self, active):
        self.active = active
        self.start_button.state(["disabled"] if active or self.loading or not self.loaded else ["!disabled"])
        self.cancel_button.state(["!disabled"] if active else ["disabled"])
        self.query.configure(state="disabled" if active else "normal")
        self._manual_changed()
        if active:
            self.progress.stop()
            self.progress.pack_forget()
            self.progress_text.set("")
            self._show_execution()
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self.progress.pack_forget()

    def start(self):
        if self.active or self.loading or self.app.closing or not self.loaded:
            return
        if not self.query.get().strip():
            self.message.set("Введите направление исследования.")
            self.query.focus_set()
            return
        if self.settings_status and self.settings_status.get("model_installed") is False:
            bundled = self.settings_status.get("model_origin") == "bundled"
            self.message.set((self.settings_status.get("model_error") or
                              "Встроенная модель недоступна. Переустановите приложение.") if bundled else
                             "Сначала установите локальную модель в настройках анализа.")
            if not bundled:
                self.setup_action.pack(anchor="w", padx=16, after=self.message_label)
            return
        english = self.english.get().strip() if self.manual.get() else None
        source = "user"
        if english is None and self._needs_local_translation():
            translated = self._translation
            if translated is None or translated[0] != self.query.get():
                self._translate_then_start()
                return
            english, source = translated[1], "local_translation"
        self.generation += 1
        self.run_id = None
        self.pending_cancel = False
        self.start_cancel = Event()
        self._busy(True)
        self.message.set("Проверяем настройки и локальную модель…")
        if not self._call("start", "start", self._started, self.query.get(), english,
                          english_source=source, owner=self.generation, cancel=self.start_cancel):
            self._busy(False)

    def _started(self, run_id):
        self.run_id = run_id
        self.payload = None
        self.clear_result_tables()
        self.cards.clear()
        self.card_list.clear()
        self.result_actions.pack_forget()
        self._history_changed()
        self.export_button.state(["disabled"])
        self.passport_button.state(["disabled"])
        self.documents_button.state(["disabled"])
        self.scope.set("")
        self.summary.set("Получаем реальные документы. Первичная группировка ещё не подтверждает зарождение тренда.")
        if self.pending_cancel:
            self.cancel()
        self._poll()

    def refine_candidate(self, source_id, candidate_id):
        """Explicitly develop a saved finding; source identity belongs to its window."""
        if self.active or self.loading or self.app.closing or not self.loaded:
            self.message.set("Дождитесь завершения текущей операции или отмените её.")
            return False
        self.generation += 1
        self.run_id = None
        self.pending_cancel = False
        self.start_cancel = Event()
        self._busy(True)
        self.message.set("Уточняем механизм, собираем историю и проверяем ранние аналоги выбранной гипотезы…")
        submitted = self._call("refine", "refine_candidate", self._started, source_id, candidate_id,
                               owner=self.generation, cancel=self.start_cancel)
        if not submitted:
            self._busy(False)
        return submitted

    def _poll(self):
        if not self.active or self.app.closing or not self.run_id:
            return
        self._call("progress", "get", self._progress, self.active_run_id, owner=self.generation)

    def _progress(self, row):
        if row["id"] != self.run_id:
            return
        self.message.set(row["message"] or STATES.get(row["state"], row["state"]))
        if row["total"]:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=100 * row["completed"] / row["total"])
            self.progress.pack(fill="x", padx=16, after=self.message_label)
            self.progress_text.set(f"В текущем этапе: {row['completed']} из {row['total']}")
        else:
            self.progress.pack_forget()
            self.progress_text.set("")
        self.documents_button.state(["!disabled"])
        if row["state"] in {"queued", "running"}:
            self.app.root.after(700, self._poll)
        else:
            self._busy(False)
            self._history_changed()
            if row["state"] == "succeeded":
                self.open_result(row["id"])
            else:
                self.message.set(row["error"] or STATES[row["state"]] + ". Сохранённые этапы доступны в истории.")
                if row.get("clarification"):
                    self._clarification(row["clarification"]["options"])

    def open_result(self, identifier, *, on_ready=None):
        """Keep the previous payload/ID paired until this requested view is ready."""
        if self.active or self.app.closing:
            return False
        previous = self.generation
        self.generation += 1
        def ready(payload):
            if self._result(payload, expected_id=identifier) and on_ready is not None:
                on_ready()
        submitted = self._call("result", "result", ready, identifier, owner=self.generation)
        if not submitted:
            self.generation = previous
        return submitted

    def resume_run(self, identifier, *, query=None):
        """Both history views resume through the same cancellable owner."""
        if self.active or self.loading or self.app.closing:
            return False
        self.generation += 1
        self.run_id = None
        self.pending_cancel = False
        self.start_cancel = Event()
        self._busy(True)
        if query is not None:
            self.direction.set(query)
        submitted = self._call("resume", "resume", self._started, identifier,
                               owner=self.generation, cancel=self.start_cancel)
        if not submitted:
            self._busy(False)
        return submitted

    def _result(self, payload, *, expected_id=None):
        result = payload["result"]
        identifier = payload.get("view_id", result["run_id"])
        if self.app.closing or identifier != (self.run_id if expected_id is None else expected_id):
            return False
        self.run_id, self.payload = identifier, payload
        self._show_execution()
        self.direction.set(result["query_plan"].get("original_query", self.query.get()))
        self.clear_result_tables()
        self.cards = {card["candidate"]["candidate_id"]: card for card in result["cards"]}
        assessments = {item["assessment"]["candidate_id"]: item["assessment"] for item in payload.get("assessments", [])}
        top_ids = result.get("top_trend_ids")
        if top_ids is None:
            top_ids = [identifier for identifier, card in self.cards.items() if card["category"] == "confirmed_trend"][:15]
        top_ranks = {identifier: rank for rank, identifier in enumerate(top_ids, 1)}
        other_ids = [identifier for identifier in self.cards if identifier not in top_ranks]
        self.cards = {identifier: self.cards[identifier] for identifier in (*top_ids, *other_ids)}
        for identifier in self.cards:
            card = self.cards[identifier]
            assessment = assessments.get(identifier, {})
            label = category_label(card)
            if identifier in top_ranks:
                label = f"TOP {top_ranks[identifier]} · {label}"
            tree = self.tree if identifier in top_ranks else self.other_tree
            studies = assessment.get("recent_studies")
            tree.insert("", "end", iid=identifier, values=(card["candidate"]["label"],
                label, studies if studies is not None else "Не проверено"))
        self.scope.set("Область: " + result["query_plan"]["definition"])
        confirmed = sum(card["category"] == "confirmed_trend" for card in self.cards.values())
        queued = len(result.get("candidate_queue", ()))
        top_limit = result.get("top_limit") or 15
        self.queue_button.state(["!disabled"] if queued else ["disabled"])
        self.result_tabs.tab(0, text=f"TOP-15 · {len(top_ids)} из {top_limit}")
        self.result_tabs.tab(1, text=f"Остальные кандидаты · {len(other_ids)}")
        reason = ("Карточки не сформированы. Непроверенные находки доступны в очереди гипотез; откройте выбранную находку для уточнения."
                  if not self.cards and queued else
                  "Для выделения групп недостаточно релевантных документов. Откройте публикации и уточните направление."
                  if not self.cards else
                  "Нет кандидатов, допущенных в TOP. Откройте карточки вне TOP и их паспорта: там указаны недостающие доказательства; "
                  "редкие находки доступны в очереди гипотез." if not top_ids else
                  f"Список содержит только допущенные кандидаты и не дополняется непроверенными группами до {top_limit}. "
                  "Автоматическая оценка и экспертная проверка указаны отдельно.")
        self.summary.set(f"В TOP-15: {len(top_ids)} из {top_limit}. Трендов с экспертной проверкой: {confirmed}. "
                         f"Вне TOP: {len(other_ids)}. Гипотез в очереди: {queued}. " + reason)
        completion = evaluation_progress(result)
        self.message.set(" ".join(part for part in (completion.message, completion.details,
                          *result["limitations"]) if part))
        self.export_button.state(["!disabled"])
        self.documents_button.state(["!disabled"])
        for tree in (self.tree, self.other_tree):
            identifiers = tree.get_children()
            if identifiers:
                tree.selection_set(identifiers[0])
                tree.focus(identifiers[0])
        self.result_tabs.select(0)
        self._selection()
        self.card_list.render(self.cards, assessments)
        self.result_actions.pack(fill="x", padx=16, pady=12)
        return True

    def show_candidate_queue(self):
        from app.pilot.completion import REJECTION_REASONS

        if not self.payload or self.app.closing:
            return
        queue = self.payload["result"].get("candidate_queue", ())
        source_id = self.displayed_id
        if not queue:
            return
        window = tk.Toplevel(self.app.root)
        self.app.child_windows.append(window)
        window.title("Сохранённые гипотезы для предметной проверки")
        window.geometry("920x560")
        window.minsize(480, 300)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        heading = ttk.Label(window, text="Эти находки не прошли полную проверку. Размер группы не доказывает новизну. "
                  "Они сохранены вместе с результатом и источниками.", wraplength=840)
        heading.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        tree = ttk.Treeview(window, columns=("title", "count"), show="headings", selectmode="browse")
        tree.heading("title", text="Исследовательская гипотеза")
        tree.heading("count", text="Работ в группе")
        tree.column("title", width=700, minwidth=250)
        tree.column("count", width=100, minwidth=80, stretch=False)
        tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0))
        scroll = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        scroll.grid(row=1, column=1, sticky="ns", padx=(0, 12))
        tree.configure(yscrollcommand=scroll.set)
        by_id = {item["candidate_id"]: item for item in queue}
        review_states = {item["candidate_id"]: item
                         for item in self.payload["result"].get("candidate_review_states") or ()}
        notice = tk.StringVar(value="Выберите находку: откройте источник или запустите уточнение механизма и проверку истории.")
        notice_label = ttk.Label(window, textvariable=notice, wraplength=840)
        notice_label.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        window.bind("<Configure>", lambda event: (
            heading.configure(wraplength=max(200, event.width - 40)),
            notice_label.configure(wraplength=max(200, event.width - 40))) if event.widget == window else None)
        page_controls = ttk.Frame(window)
        page_controls.grid(row=3, column=0, sticky="ew", padx=12, pady=8)
        page_text = tk.StringVar()
        page = [0]

        def render_page(change=0):
            page[0] = min(max(0, page[0] + change), (len(queue) - 1) // 50)
            tree.delete(*tree.get_children())
            start = page[0] * 50
            for item in queue[start:start + 50]:
                tree.insert("", "end", iid=item["candidate_id"],
                            values=(item["label"], len(item["discovery_study_ids"])))
            page_text.set(f"{start + 1}–{min(start + 50, len(queue))} из {len(queue)}")
            previous.state(["disabled"] if page[0] == 0 else ["!disabled"])
            following.state(["disabled"] if start + 50 >= len(queue) else ["!disabled"])

        previous = ttk.Button(page_controls, text="Предыдущие 50", command=lambda: render_page(-1))
        previous.pack(side="left")
        ttk.Label(page_controls, textvariable=page_text).pack(side="left", padx=12)
        following = ttk.Button(page_controls, text="Следующие 50", command=lambda: render_page(1))
        following.pack(side="left")
        render_page()

        def open_source(_event=None):
            selected = tree.selection()
            if not selected or not window.winfo_exists():
                return
            item = by_id[selected[0]]
            doi = next((key[4:] for key in item["discovery_study_ids"] if key.startswith("doi:")), None)
            if doi is None:
                notice.set("DOI отсутствует. Исследования: " + "; ".join(item["discovery_study_ids"][:5]))
                return
            self.app.controller.call("pilot-queue-source-" + str(window), "open_url",
                                     lambda _: None, self.error, "https://doi.org/" + doi)

        tree.bind("<Return>", open_source)
        tree.bind("<Double-1>", open_source)
        def refine():
            selected = tree.selection()
            if selected and window.winfo_exists():
                if self.refine_candidate(source_id, selected[0]):
                    window.destroy()
                else:
                    notice.set(self.message.get())
        footer = ttk.Frame(window)
        footer.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))
        source_button = ttk.Button(footer, text="Открыть первоисточник выбранной находки", command=open_source)
        source_button.pack(anchor="e")
        refine_button = ttk.Button(footer, text="Развить гипотезу", command=refine, state="disabled")
        refine_button.pack(anchor="e", pady=(6, 0))
        def select_hypothesis(_event=None):
            selected = tree.selection()
            refine_button.state(["!disabled"] if selected else ["disabled"])
            state = review_states.get(selected[0], {}) if selected else {}
            if state.get("outcome") == "definition_rejected":
                code = state.get("reason_code")
                notice.set(REJECTION_REASONS.get(code if isinstance(code, str) else "definition_rejected",
                                                REJECTION_REASONS["definition_rejected"]))
            else:
                notice.set("Паспорт этой гипотезы ещё не сформирован. Откройте источник или запустите уточнение.")
        tree.bind("<<TreeviewSelect>>", select_hypothesis)

    def _selection(self, _=None):
        self.passport_button.state(["!disabled"] if self.selected_result_tree.selection() else ["disabled"])

    def cancel(self):
        if self.active:
            token = getattr(self, "start_cancel", None)
            if token is not None:
                token.set()
            self.pending_cancel = True
            self.cancel_button.state(["disabled"])
            self.message.set("Отмена запрошена; ждём безопасной остановки…")
        if self.active_run_id:
            self._call("cancel", "cancel", lambda _: self.message.set("Отмена запрошена; ждём безопасной остановки…"),
                       self.active_run_id, owner=self.generation)

    def error(self, error, *, stop=False):
        from app.pilot.encoder import EncoderError
        from app.pilot.llm import LlmError
        from app.runtime.credentials import CredentialUnavailable
        from app.runtime.backup import ArchiveError
        from app.pilot.enrichment import EnrichmentError
        from app.pilot.reports import ReportImportError
        from app.runtime.budget import BudgetError

        if stop:
            self._busy(False)
        if isinstance(error, TaskCancelled):
            self.message.set("Операция отменена. Сохранённые результаты остаются доступны.")
        elif isinstance(error, (TaskFailure, QueryError, EncoderError, LlmError, CredentialUnavailable, ArchiveError, EnrichmentError, ReportImportError, BudgetError)):
            self.message.set(str(error))
        elif isinstance(error, ValidationError):
            self.message.set("Проверьте поля: " + "; ".join(str(item["msg"]) for item in error.errors(include_input=False, include_url=False)))
        else:
            self.message.set("Операция не завершена. Проверьте подключение и доступ к данным. Ранее сохранённые результаты доступны в истории.")
        if stop:
            self.setup_action.pack(anchor="w", padx=16, after=self.message_label)

    # Everything the settings and keys form reads when it is built.
    _SETTINGS_FIELDS = frozenset({"settings", "keys", "model_installed", "persistent_keys_available"})

    def _settings_form(self, status):
        """Build the settings and keys form in place once, then only refresh it.

        A partial status simply waits: the form is built from a complete one, and
        callers that report only part of the state must not raise here.
        """
        if self.settings_form_parent is None or self.app.closing:
            return
        if not self._SETTINGS_FIELDS <= status.keys():
            return
        from app.ui.pilot_settings import SettingsDialog

        if self.settings_form is None:
            self.settings_form = SettingsDialog(self, parent=self.settings_form_parent)
        else:
            self.settings_form.refresh(status)

    def settings(self):
        """Go to the settings page; a window only remains where there is no page."""
        if self.settings_status is None:
            self.load()
            return
        navigation = getattr(self.app, "navigation", None)
        if self.settings_form is not None and navigation is not None:
            navigation.select("settings")
            return
        from app.ui.pilot_settings import SettingsDialog

        SettingsDialog(self)

    def passport(self):
        from app.ui.pilot_passport import show_passport

        selected = self.selected_result_tree.selection()
        if selected and self.payload:
            show_passport(self, self.cards[selected[0]], self.payload)

    def history(self):
        if hasattr(self.app, "analysis_history"):
            self.app.navigation.select("analyses")
            self.app.analysis_history.ensure_loaded()
            return
        self._call("history", "list_runs", self._history, 0, 50, "local")

    def _history(self, rows):
        from app.ui.display import px

        window = tk.Toplevel(self.app.root)
        window.title("История анализов")
        window.geometry("900x440")
        available_width, available_height = window.maxsize()
        # Retain a complete data row and the two footer rows at larger fonts,
        # while respecting the native window manager's available client area.
        window.minsize(min(available_width, window.winfo_screenwidth(), max(560, px(self.app.root, 450))),
                       min(available_height, window.winfo_screenheight(), max(320, px(self.app.root, 220))))
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        self.app.child_windows.append(window)
        listing = ttk.Frame(window)
        listing.grid(row=0, column=0, sticky="nsew", padx=12, pady=8)
        listing.rowconfigure(0, weight=1)
        listing.columnconfigure(0, weight=1)
        tree = ttk.Treeview(listing, columns=("query", "state", "date"), show="headings", selectmode="browse", height=5)
        for key, title in (("query", "Направление"), ("state", "Состояние"), ("date", "Создан")):
            tree.heading(key, text=title)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(listing, command=tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(listing, orient="horizontal", command=tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        by_id: dict[str, _RunRow] = {}
        controls = ttk.Frame(window)
        controls.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        controls.columnconfigure(0, weight=1)
        source_labels = {"Локальные запуски": "local", "Импорт и версии": "imported"}
        source = tk.StringVar(value="Локальные запуски")
        selector = ttk.Combobox(controls, textvariable=source, state="readonly", width=1,
                               values=tuple(source_labels))
        selector.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        position = tk.StringVar()
        notice = tk.StringVar()
        # A single selectable status line cannot take height away from the
        # result rows when an error wraps at larger native font scales.
        ttk.Entry(controls, textvariable=notice, state="readonly", width=1).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=(0, 6))
        state: _HistoryPageState = {"offset": 0, "pending": False, "source": source.get()}
        def render(page, offset, selected_source=None):
            if not window.winfo_exists():
                return
            selected = tree.selection()
            state["offset"] = offset
            state["pending"] = False
            state["source"] = selected_source or state["source"]
            source.set(state["source"])
            selector.configure(state="readonly")
            by_id.clear()
            by_id.update((row["id"], row) for row in page)
            tree.delete(*tree.get_children())
            for row in page:
                saved = json.loads(row["input_json"])
                query = saved.get("payload", saved).get("query", "Анализ")
                tree.insert("", "end", iid=row["id"], values=(query, STATES[row["state"]], row["created_at"][:16]))
            if page:
                identifier = next((item for item in selected if item in by_id), page[0]["id"])
                tree.selection_set(identifier)
                tree.focus(identifier)
                tree.see(identifier)
            position.set(f"{offset + 1}–{offset + len(page)}" if page else "Записей нет")
            previous.state(["!disabled"] if offset else ["disabled"])
            following.state(["!disabled"] if len(page) == 50 else ["disabled"])
            notice.set(position.get())
        def request(offset):
            if state["pending"]:
                return
            state["pending"] = True
            notice.set(position.get() + " · Читаем страницу истории…")
            selector.configure(state="disabled")
            previous.state(["disabled"])
            following.state(["disabled"])
            requested_source = source.get()
            chosen = source_labels[requested_source]
            def failed(error):
                if window.winfo_exists():
                    render(list(by_id.values()), state["offset"])
                    self.error(error)
                    notice.set(position.get() + " · " + self.message.get())
            if not self.app.controller.call("pilot-history-page-" + str(window), "pilot_list_runs", lambda page: render(page, offset, requested_source),
                                            failed, offset, 50, chosen):
                failed(TaskFailure("Чтение истории пока недоступно. Повторите после текущей операции."))
        previous = ttk.Button(controls, text="Назад", width=0, command=lambda: request(max(0, state["offset"] - 50)))
        previous.grid(row=1, column=1, padx=4)
        following = ttk.Button(controls, text="Далее", width=0, command=lambda: request(state["offset"] + 50))
        following.grid(row=1, column=2, padx=4)
        selector.bind("<<ComboboxSelected>>", lambda _: request(0))
        render(rows, 0)
        def open_selected():
            selected = tree.selection()
            if not selected or self.active or state["pending"]:
                return
            row = by_id[selected[0]]
            if row["state"] == "succeeded":
                if self.open_result(row["id"]):
                    window.destroy()
            elif row.get("imported"):
                self.message.set(row["error"] or "Импортируйте исходный пакет заново.")
            elif row["state"] in {"cancelled", "failed", "interrupted"}:
                if self.loading:
                    self.message.set("Дождитесь завершения текущей операции с локальными материалами.")
                    return
                if self.resume_run(row["id"]):
                    window.destroy()
        def open_from_keyboard(_event):
            open_selected()
            return "break"
        tree.bind("<Return>", open_from_keyboard)
        ttk.Button(controls, text="Открыть / продолжить", width=0, command=open_selected).grid(
            row=1, column=0, sticky="ew", padx=(0, 4))

    def documents(self):
        identifier = self.active_run_id or self.displayed_id or self.run_id
        if identifier:
            self._call("documents", "documents", lambda page: self._documents(page, identifier), identifier)

    def _documents(self, page, run_id):
        from app.ui.document_browser import DocumentBrowser

        DocumentBrowser(self, page, run_id)

    def _clarification(self, options):
        window = tk.Toplevel(self.app.root)
        window.title("Уточнение направления")
        self.app.child_windows.append(window)
        for option in options:
            def choose(text=option):
                self.query.delete(0, "end")
                self.query.insert(0, text)
                window.destroy()
            ttk.Button(window, text=option, command=choose).pack(fill="x", padx=12, pady=8)

    def export(self):
        from app.ui.pilot_passport import export_payload

        if self.payload:
            export_payload(self, self.payload)

    def import_result(self):
        from tkinter import filedialog

        if self.active or self.app.closing:
            self.message.set("Дождитесь завершения текущего анализа или отмените его.")
            return
        path = filedialog.askopenfilename(parent=self.app.root, title="Открыть проверяемый результат",
                                          filetypes=[("Результат анализа", "*.trendresult")])
        if path:
            # The file dialog runs a nested native event loop. Recheck ownership
            # after it returns, then fence a late import behind newer user intent.
            if self.active or self.app.closing:
                return
            previous = self.generation
            self.generation += 1
            owner = self.generation
            if not self._call("import", "import_result", lambda saved: self._imported(saved, owner=owner),
                              path, owner=owner):
                self.generation = previous

    def _imported(self, saved, *, owner=None):
        if self.active or self.app.closing or owner is not None and owner != self.generation:
            return False
        if not self._result(saved["payload"], expected_id=saved["id"]):
            return False
        self._history_changed()
        if hasattr(self.app, "navigation"):
            self.app.navigation.select("analysis")
        return True
