"""Local MVP workflow: saved corpus or historical collection → candidate cards."""

import tkinter as tk
import unicodedata
from concurrent.futures import CancelledError
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from app.ml.directions import direction_profile, resolve_direction

RESULT_BLOCKS = (
    ("candidates", "Кандидаты с ростом"),
    ("preliminary_signals", "Предварительные сигналы"),
    ("established", "Крупные темы"),
    ("excluded_off_direction", "Отклонено фильтром"),
)
SELECTION_REASONS = {
    "incomparable_growth_coverage": "Покрытие данных не позволяет подтвердить сопоставимость роста.",
    "incomplete_coverage": "Покрытие данных неполное; рост требует проверки.",
    "growth_not_confirmed": "Устойчивый рост не подтверждён.",
    "no_growth_pattern": "Признаки устойчивого роста не установлены.",
    "nominal_only": "Есть тематическая связь, но прямое выполнение операции не подтверждено.",
    "missing_direct_execution": "Прямое выполнение операции устройством не подтверждено.",
    "incomplete_evidence_card": "Для всех полей карточки недостаточно подходящих свидетельств.",
    "insufficient_evidence": "Доказательств недостаточно для основного списка.",
    "window_share_above_threshold": "Тема занимает более 8% доступного корпуса в выбранном окне.",
    "established_share": "Тема занимает более 8% доступного корпуса в выбранном окне.",
    "off_direction": "Связь с направлением не подтверждена.",
    "no_document_support_for_missing_cluster_axis": "Недостающие признаки направления не подтверждены документами.",
    "profile_not_configured": "Профиль направления не настроен, проверка релевантности ограничена.",
    "large_share_in_available_window": "Тема занимает более 8% доступного корпуса в выбранном окне.",
    "sustained_growth_not_established": "Устойчивый рост в выбранном окне не подтверждён.",
    "nominal_execution_only": "Есть тематическая связь, но выполнение операции устройством не подтверждено.",
    "insufficient_direction_evidence": "Документов недостаточно для проверки направления.",
    "incomplete_supported_card": "Не для всех полей найдены подходящие подтверждающие фрагменты.",
    "example_is_bibliographic_reference": "Есть ссылка на исследование, но содержательный пример разработки не извлечён.",
    "bibliographic_example_not_an_execution_claim": "Указано исследование; название не доказывает выполнение операции.",
    "source_role_supported": "Фрагмент соответствует назначению поля по выполненной проверке.",
    "not_a_problem_statement": "Фрагмент не описывает решаемую проблему.",
    "benefit_not_reported_as_a_result_or_property": "Фрагмент не подтверждает преимущество как результат или свойство.",
    "not_a_specific_research_example": "Фрагмент не описывает конкретное исследование или разработку.",
    "no_supported_excerpt": "Подходящий подтверждающий фрагмент не найден.",
    "semantic_scope_requires_review": "Смысловая близость требует проверки; группа остаётся предварительной.",
}
EVIDENCE_LEVELS = {"direct": "прямое свидетельство", "nominal": "тематическое упоминание",
                   "not_applicable": "правило оптического исполнителя неприменимо"}
MODALITIES = {"demonstrated": "подтверждённый результат", "observed": "наблюдаемый результат",
              "measured": "измеренный результат", "capability": "заявленная возможность",
              "future": "перспективное применение", "prospective": "перспективное применение",
              "goal": "цель разработки", "requirement": "необходимое условие",
              "background": "общие сведения", "negative": "отрицательный результат",
              "unsupported": "подтверждение не найдено", "unknown": "модальность не установлена",
              "research_reference": "ссылка на исследование", "review": "обзор",
              "outlook": "перспективное применение", "simulation": "расчёт или симуляция",
              "simulation_and_experiment": "симуляция и физический эксперимент",
              "reported_result": "результат, заявленный в источнике", "proposal": "предлагаемый подход",
              "source_statement": "утверждение источника", "reviewed_result": "результат из обзора",
              "not_found": "подходящий фрагмент не найден"}


def _reason_text(value):
    return SELECTION_REASONS.get(value, value if isinstance(value, str) and " " in value
                                 else "Требуется дополнительная содержательная проверка.")


@dataclass(eq=False)
class _Operation:
    """Identity and cancellation belong to one accepted foreground request."""

    kind: str
    cancel: Event = dataclass_field(default_factory=Event)


class TrendsPanel:
    def __init__(self, app, parent):
        self.app, self.parent = app, parent
        self.snapshot_path = self.history_id = self.collecting_id = None
        self._corpus_sources: tuple[str, ...] = ()
        self._corpus_error = ""
        self.demo_path = Path(__file__).resolve().parents[2] / "storage/ml-validation/corpus-portable.json"
        self.demo_available = self.demo_path.is_file()
        self.result = None
        self._operation: _Operation | None = None
        self._result_context = None
        self._cancel_pending = self._collection_cancelled = False
        self.cancel_event = Event()
        self.messages: Queue[tuple[_Operation, float, str]] = Queue()
        self.histories = {}
        self.current_sources = []
        self.result_block = tk.StringVar(value=RESULT_BLOCKS[0][1])
        self.block_labels = {label: key for key, label in RESULT_BLOCKS}
        self.topic = tk.StringVar(value="Фотонные нейроморфные вычисления")
        last = datetime.now(timezone.utc).year - 1
        self.start_year, self.end_year = tk.StringVar(value=str(last - 5)), tk.StringVar(value=str(last))
        self.source = tk.StringVar(value="openalex")
        from app.ml.local_encoder import DEFAULT_MODEL_DIR, load_spec
        from app.runtime.model_resources import frozen, resolve_model
        self.semantic_mode = tk.BooleanVar(value=False)
        self.bundled_model = frozen()
        model = resolve_model("e5-small-v2", load_spec()["revision"], development_default=DEFAULT_MODEL_DIR)
        self.model_dir = tk.StringVar(value=str(model.path))
        form = ttk.Frame(parent)
        form.pack(fill="x")
        ttk.Label(form, text="Направление").grid(row=0, column=0, sticky="w")
        self.topic_entry = ttk.Combobox(form, textvariable=self.topic, values=tuple(
            direction_profile(query)["display_name_ru"] for query in
            ("photonic neuromorphic computing", "quantum reservoir computing", "dna data storage")))
        self.topic_entry.grid(row=0, column=1, columnspan=5, sticky="ew", padx=8)
        ttk.Label(form, text="Период").grid(row=1, column=0, sticky="w", pady=6)
        self.start_entry = ttk.Entry(form, textvariable=self.start_year, width=8)
        self.start_entry.grid(row=1, column=1, sticky="w", padx=8)
        ttk.Label(form, text="—").grid(row=1, column=2)
        self.end_entry = ttk.Entry(form, textvariable=self.end_year, width=8)
        self.end_entry.grid(row=1, column=3, sticky="w", padx=8)
        ttk.Label(form, text="Источник").grid(row=1, column=4, sticky="e")
        self.source_box = ttk.Combobox(form, textvariable=self.source, values=("openalex", "crossref"),
                                      state="readonly", width=13)
        self.source_box.grid(row=1, column=5, sticky="e", padx=8)
        self.semantic_check = ttk.Checkbutton(form, text="Смысловая проверка (локальная модель)",
                                              variable=self.semantic_mode, command=self.controls)
        self.semantic_check.grid(row=2, column=0, columnspan=4, sticky="w", pady=3)
        self.model_button = ttk.Button(form, text="Каталог модели…", command=self.choose_model_dir)
        self.model_button.grid(row=2, column=4, columnspan=2, sticky="e", padx=8)
        if self.bundled_model:
            self.model_button.grid_remove()
        self.model_description = tk.StringVar(value="Встроенная модель готова к локальному расчёту." if self.bundled_model else
                                               str(model.path))
        self.model_location = ttk.Label(form, textvariable=self.model_description, wraplength=880, style="Muted.TLabel")
        self.model_location.grid(row=3, column=0, columnspan=6, sticky="w", pady=(0, 3))
        source_help = ttk.Frame(form)
        source_help.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(0, 3))
        self.source_hint = tk.StringVar()
        ttk.Label(source_help, textvariable=self.source_hint, wraplength=700,
                  style="Muted.TLabel").pack(side="left", fill="x", expand=True)
        self.source_reset = ttk.Button(source_help, text="Другой источник…", command=self.choose_source)
        self.source_reset.pack(side="right", padx=8)
        form.columnconfigure(1, weight=1)
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=5)
        self.collect_button = ttk.Button(bar, text="Сбор через интернет", command=self.collect)
        self.collect_button.pack(side="left")
        self.file_button = ttk.Button(bar, text="Открыть JSON-корпус…", command=self.choose_snapshot)
        self.file_button.pack(side="left", padx=5)
        self.demo_button = ttk.Button(bar, text="Локальное демо", command=self.demo)
        self.demo_button.pack(side="left")
        self.run_button = ttk.Button(bar, text="Анализировать", command=self.analyze)
        self.run_button.pack(side="left", padx=5)
        self.cancel_button = ttk.Button(bar, text="Отмена", command=self.cancel)
        self.cancel_button.pack(side="right")
        stored = ttk.Frame(parent)
        stored.pack(fill="x", pady=3)
        ttk.Label(stored, text="История").pack(side="left")
        self.history_box = ttk.Combobox(stored, state="readonly")
        self.history_box.pack(side="left", fill="x", expand=True, padx=8)
        self.history_box.bind("<<ComboboxSelected>>", self.select_history)
        self.refresh_button = ttk.Button(stored, text="Обновить", command=self.refresh_histories)
        self.refresh_button.pack(side="left")
        self.input_label = tk.StringVar(value="Откройте сохранённый корпус или запустите сбор. Анализ выполняется локально.")
        ttk.Label(parent, textvariable=self.input_label, wraplength=1020, style="Muted.TLabel").pack(anchor="w", pady=4)
        self.notice = tk.StringVar()
        ttk.Label(parent, textvariable=self.notice, wraplength=1020, style="Muted.TLabel").pack(anchor="w")
        self.status = tk.StringVar(value="Результат — до 15 кандидатов. Новизна и стадия требуют проверки.")
        ttk.Label(parent, textvariable=self.status, wraplength=1020).pack(anchor="w")
        self.progress = ttk.Progressbar(parent, maximum=100)
        self.progress.pack(fill="x", pady=4)
        from app.ui.navigation import SectionNotebook
        self.tabs = SectionNotebook(parent)
        self.tabs.pack(fill="both", expand=True)
        listing = ttk.Frame(self.tabs)
        self.tabs.add(listing, text="Результаты")
        section = ttk.Frame(listing)
        section.pack(fill="x", pady=(4, 6))
        ttk.Label(section, text="Раздел").pack(side="left")
        self.block_box = ttk.Combobox(section, textvariable=self.result_block,
                                     values=tuple(self.block_labels), state="readonly", width=42)
        self.block_box.pack(side="left", fill="x", expand=True, padx=8)
        self.block_box.bind("<<ComboboxSelected>>", self.select_block)
        self.block_description = tk.StringVar(value="Главный список не дополняется неподтверждёнными карточками.")
        ttk.Label(listing, textvariable=self.block_description, wraplength=1000,
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        diagnostics = ttk.Frame(self.tabs)
        self.tabs.add(diagnostics, text="Качество данных и методика")
        self.diagnostics = ScrolledText(diagnostics, wrap="word", height=12, state="disabled")
        self.diagnostics.pack(fill="both", expand=True)
        panes = ttk.Panedwindow(listing, orient="horizontal")
        panes.pack(fill="both", expand=True)
        frame, self.tree = app._tree(panes, [("rank", "№", 35), ("title", "Технология / тема", 260),
                                             ("score", "Рейтинг", 65), ("docs", "Работ", 60)], 9)
        panes.add(frame, weight=2)
        self.detail = ScrolledText(panes, wrap="word", width=48, height=12, state="disabled", padx=8, pady=8)
        panes.add(self.detail, weight=3)
        self.tree.bind("<<TreeviewSelect>>", self.select_candidate)
        footer = ttk.Frame(parent)
        footer.pack(side="bottom", fill="x", pady=(5, 0), before=self.tabs)
        self.export_button = ttk.Button(footer, text="Сохранить результат JSON…", command=self.export)
        self.export_button.pack(side="right")
        self.link_button = ttk.Button(footer, text="Открыть основной источник", command=self.open_source)
        self.link_button.pack(side="left")
        self.set_text(self.detail, "После анализа выберите кандидата: здесь появятся описание, динамика и источники.")
        for variable in (self.topic, self.start_year, self.end_year, self.source,
                         self.semantic_mode, self.model_dir):
            variable.trace_add("write", self._input_changed)
        self.controls()

    @property
    def busy(self):
        return self._operation is not None

    def _can_start(self):
        return self.app.ready and not self.app.closing and not self.busy and not self.collecting_id

    def _input_context(self):
        return (str(self.snapshot_path) if self.snapshot_path else None, self.history_id,
                self.topic.get(), self.start_year.get(), self.end_year.get(), self.source.get(),
                self.semantic_mode.get(), self.model_dir.get() if self.semantic_mode.get() else None)

    def _input_changed(self, *_):
        if self.result is not None and self._result_context != self._input_context():
            self.clear_result()
            self.status.set("Параметры изменены. Запустите анализ для нового выбора.")
            self.controls()

    def _submit(self, operation: _Operation, key, method, success, *args, **kwargs):
        """Only this request may release the foreground controls or finish its result."""
        if not self._can_start():
            return False

        def failed(error):
            if self._operation is operation and not self.app.closing:
                self._operation = None
                self.failed(error)

        def completed(value):
            if self._operation is not operation or self.app.closing:
                return
            self._operation = None
            if operation.cancel.is_set() and operation.kind in {"inspection", "analysis"}:
                self.failed(CancelledError())
                return
            success(value)

        accepted = self.app.controller.call(key, method, completed, failed, *args, **kwargs)
        if not accepted:
            self.status.set("Операция не запущена: предыдущий запрос ещё выполняется или приложение закрывается.")
            self.controls()
            return False
        self._operation = operation
        self.cancel_event = operation.cancel
        self.notice.set("")
        self.controls()
        return True

    @staticmethod
    def set_text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def controls(self):
        ready = self.app.ready and not self.app.closing
        idle = ready and not self.busy and not self.collecting_id
        for widget in (self.file_button, self.refresh_button,
                       self.topic_entry, self.start_entry, self.end_entry, self.semantic_check):
            widget.state(["!disabled"] if idle else ["disabled"])
        corpus = bool(self.snapshot_path or self.history_id)
        self.collect_button.state(["!disabled"] if idle and not self._corpus_error else ["disabled"])
        self.demo_button.state(["!disabled"] if idle and self.demo_available else ["disabled"])
        self.source_reset.state(["!disabled"] if idle and corpus else ["disabled"])
        explanation = (self._corpus_error or
                       ("Источник закреплён в корпусе. «Другой источник…» очищает выбор корпуса и результат."
                        if corpus else "Выберите источник для нового сбора."))
        if not self.demo_available:
            explanation += " Демо-корпус не установлен."
        self.source_hint.set(explanation)
        self.model_button.state(["!disabled"] if idle and self.semantic_mode.get() else ["disabled"])
        if self.semantic_mode.get():
            self.model_location.grid()
        else:
            self.model_location.grid_remove()
        self.source_box.configure(state="readonly" if idle and not corpus else "disabled")
        self.history_box.configure(state="readonly" if idle else "disabled")
        self.run_button.state(["!disabled"] if idle and corpus and not self._corpus_error else ["disabled"])
        cancellable = (self._operation is not None and self._operation.kind != "export"
                       and not self._operation.cancel.is_set()) or (
                           self.collecting_id and not self._cancel_pending and not self._collection_cancelled)
        self.cancel_button.state(["!disabled"] if ready and cancellable else ["disabled"])
        self.export_button.state(["!disabled"] if idle and self.result else ["disabled"])
        self.link_button.state(["!disabled"] if ready and self.current_sources else ["disabled"])

    def ready(self):
        self.controls()
        self.refresh_histories()

    def refresh_histories(self):
        if self.app.ready and not self.app.closing:
            accepted = self.app.controller.call("ml_histories", "list_history", self.render_histories,
                lambda error: self.background_failed(error, "Не удалось обновить список историй. Повторите обновление."),
                limit=50)
            if not accepted:
                self.notice.set("Обновление списка историй уже выполняется или приложение закрывается.")

    def render_histories(self, histories):
        self.histories = {f"{h['request']['topic']} · {h['request']['from_date'][:4]}–{h['request']['until_date'][:4]} · {h['id'][:8]}": h
                          for h in histories}
        self.history_box.configure(values=list(self.histories))

    def select_history(self, event=None):
        history = self.histories.get(self.history_box.get())
        if not history or not self._can_start():
            return
        request = history["request"]
        self.history_id, self.snapshot_path = history["id"], None
        self._corpus_sources = tuple(request["sources"])
        self._corpus_error = ("Для MVP нужна история одного научного источника: OpenAlex или Crossref. "
                              "Выберите другую историю или создайте новый сбор."
                              if len(self._corpus_sources) != 1 or self._corpus_sources[0] not in
                              {"openalex", "crossref"} else "")
        self.topic.set(request["topic"])
        self.source.set(", ".join(self._corpus_sources))
        self.start_year.set(request["from_date"][:4])
        self.end_year.set(str(min(int(request["until_date"][:4]), datetime.now(timezone.utc).year - 1)))
        self.input_label.set("Корпус: сохранённая история " + history["id"][:8]
                             + " · Источники: " + ", ".join(self._corpus_sources))
        self.clear_result()
        self.controls()

    def choose_source(self):
        if not self._can_start():
            return
        self.snapshot_path = self.history_id = None
        self._corpus_sources, self._corpus_error = (), ""
        self.clear_result()
        if self.source.get() not in {"openalex", "crossref"}:
            self.source.set("openalex")
        self.history_box.set("")
        self.input_label.set("Корпус не выбран. Выберите источник и запустите новый сбор.")
        self.status.set("Выбор корпуса и результат очищены. Направление и период сохранены.")
        self.controls()
        self.source_box.focus_set()

    def choose_snapshot(self):
        if not self._can_start():
            return
        path = filedialog.askopenfilename(parent=self.parent, title="Исторический корпус",
                                          filetypes=[("JSON", "*.json")])
        if path:
            self.load_snapshot(path)

    def choose_model_dir(self):
        if self.bundled_model or not self._can_start() or not self.semantic_mode.get():
            return
        path = filedialog.askdirectory(parent=self.parent, title="Каталог установленной локальной модели",
                                       initialdir=self.model_dir.get(), mustexist=True)
        if path:
            self.model_dir.set(path)
            self.model_description.set(path)

    def load_snapshot(self, path, auto_run=False):
        if not self._can_start():
            return
        operation = _Operation("inspection")

        def loaded(info):
            self.snapshot_path, self.history_id = str(path), None
            self._corpus_sources, self._corpus_error = (info["source"],), ""
            self.topic.set(info["topic"])
            self.source.set(info["source"])
            self.start_year.set(str(info["start_year"]))
            self.end_year.set(str(info["end_year"]))
            self.input_label.set(f"Корпус: {Path(path).name} · {info['source']} · {info['occurrences']} исходных записей")
            self.status.set("Корпус открыт. Можно запустить локальный анализ.")
            self.controls()
            if auto_run:
                self.analyze()

        if self._submit(operation, "ml_inspect", "ml_inspect", loaded, str(path), cancel=operation.cancel):
            self.clear_result()
            self.status.set("Проверяем JSON-корпус…")

    def demo(self):
        if self.demo_path.is_file():
            self.load_snapshot(self.demo_path, auto_run=True)
        else:
            self.demo_available = False
            self.controls()
            self.status.set("Локальный демо-корпус отсутствует. Откройте JSON-корпус или соберите историю.")

    def options(self):
        from app.ml.contracts import AnalysisOptions
        return AnalysisOptions(topic=self.topic.get(), start_year=int(self.start_year.get()),
                               end_year=int(self.end_year.get()),
                               relevance_mode="semantic" if self.semantic_mode.get() else "lexical").model_dump()

    def analyze(self):
        if not self._can_start():
            return
        if self._corpus_error:
            self.status.set(self._corpus_error)
            self.controls()
            return
        try:
            options = self.options()
        except ValueError as error:
            self.failed(error)
            return
        operation = _Operation("analysis")
        context = self._input_context()

        def analyzed(result):
            if context != self._input_context():
                self.status.set("Параметры изменились во время расчёта. Запустите анализ для нового выбора.")
                self.controls()
                return
            self.render_result(result)

        if self._submit(operation, "ml_analysis", "ml_analyze", analyzed, options,
                        snapshot_path=self.snapshot_path, history_id=self.history_id,
                        cancel=operation.cancel,
                        progress=lambda value, text: self.messages.put((operation, value, text)),
                        **({"model_dir": Path(self.model_dir.get())}
                           if options["relevance_mode"] == "semantic" else {})):
            self.clear_result()
            self.progress["value"] = 0
            self.status.set("Запускаем локальный анализ…")

    def collect(self):
        if not self._can_start():
            return
        if self._corpus_error:
            self.status.set(self._corpus_error)
            return
        try:
            options = self.options()
        except ValueError as error:
            self.failed(error)
            return
        query = resolve_direction(options["topic"])
        untranslated = any("CYRILLIC" in unicodedata.name(letter, "") for letter in query)
        operation = _Operation("collection")

        def started(history_id):
            self.collecting_id = self.history_id = history_id
            self.snapshot_path = None
            self._corpus_sources, self._corpus_error = (values["sources"][0],), ""
            self.topic.set(query)
            self.input_label.set("Корпус: новый исторический сбор " + history_id[:8])
            if self.cancel_event.is_set():
                self.cancel()
            self.controls()

        values = {"topic": query, "from_date": f"{options['start_year']}-01-01",
                  "until_date": f"{options['end_year']}-12-31", "sources": [self.source.get()],
                  "period": "year", "max_results_per_period": 2500, "auto_split": False}
        if self._submit(operation, "ml_collect", "ml_collect", started, values):
            self._cancel_pending = self._collection_cancelled = False
            self.clear_result()
            self.status.set("Запускаем подготовку корпуса через интернет; лимит — 2500 записей на год. " +
                            ("Локального перевода этого запроса нет: источник получит исходную строку. "
                             "Для автономной работы откройте соответствующий сохранённый корпус."
                             if untranslated else "Анализ сохранённого корпуса выполняется без сети."))

    def poll(self):
        while True:
            try:
                operation, value, message = self.messages.get_nowait()
            except Empty:
                break
            if self._operation is operation and not operation.cancel.is_set():
                self.progress["value"] = value
                self.status.set(message)
        if self.collecting_id and not self.app.closing:
            history_id = self.collecting_id
            self.app.controller.call("ml_history_progress", "get_history",
                                     lambda value: self.history_progress(value, history_id),
                                     lambda error: self.history_poll_failed(error, history_id), history_id)

    def history_poll_failed(self, error, history_id=None):
        if history_id is not None and history_id != self.collecting_id:
            return
        message = ("Не удалось проверить сбор. Проверка повторится автоматически; "
                   "сбор может продолжаться, сохранённые документы доступны в истории.")
        # A later background failure must retain the actionable retry after a
        # cancellation failed. Pending/accepted cancellation must not be repeated.
        if self.cancel_event.is_set() and not self._cancel_pending and not self._collection_cancelled:
            message += " Отмена не подтверждена. Повторите отмену."
        self.background_failed(error, message)

    def history_progress(self, history, history_id=None):
        if self.app.closing or not self.collecting_id or (
                history_id is not None and history_id != self.collecting_id):
            return
        self.status.set(f"Сбор: обработано периодов {history.processed_periods}/{history.total_periods}. "
                        "Полнота данных проверяется отдельно.")
        if history.state not in {"running", "queued"}:
            self.collecting_id = None
            self.refresh_histories()
            self.controls()
            if self.cancel_event.is_set() or history.state in {"cancelled", "interrupted", "failed"}:
                self.status.set("Сбор остановлен. Сохранённые документы доступны; анализ можно запустить отдельно.")
            else:
                self.analyze()

    def cancel(self):
        if self.app.closing:
            return
        if self.collecting_id:
            if self._cancel_pending or self._collection_cancelled:
                return
            history_id = self.collecting_id
            self.cancel_event.set()

            def cancelled(accepted):
                if self.collecting_id != history_id:
                    return
                self._cancel_pending = False
                self._collection_cancelled = bool(accepted)
                if not accepted:
                    self.notice.set("Сбор уже завершён или не принял отмену. Проверяем его состояние.")
                self.controls()

            def failed(error):
                if self.collecting_id != history_id:
                    return
                self._cancel_pending = False
                self.background_failed(error, "Не удалось отменить сбор. Повторите отмену; проверка состояния продолжается.")
                self.controls()

            accepted = self.app.controller.call("ml_cancel_history", "cancel_history", cancelled, failed, history_id)
            if not accepted:
                self.notice.set("Запрос отмены не запущен: предыдущий запрос ещё выполняется или приложение закрывается.")
                return
            self._cancel_pending = True
        elif self._operation is None or self._operation.kind == "export":
            return
        else:
            self._operation.cancel.set()
        self.status.set("Отмена запрошена. Ожидаем завершения текущего этапа.")
        self.controls()

    def close(self):
        self.cancel_event.set()

    def clear_result(self):
        self.result, self.current_sources = None, []
        self._result_context = None
        while not self.messages.empty():
            try:
                self.messages.get_nowait()
            except Empty:
                break
        self.tree.delete(*self.tree.get_children())
        self.block_description.set("Выберите соответствующий направлению корпус. Анализ работает без сети.")
        self.set_text(self.detail, "Выберите кандидата после завершения анализа.")
        self.set_text(self.diagnostics, "Показатели появятся после расчёта.")

    def render_result(self, result):
        self.result = result
        self._result_context = self._input_context()
        self.progress["value"] = 100
        counts = {key: len(result.get(key, [])) for key, _ in RESULT_BLOCKS}
        self.status.set(f"TOP: {counts['candidates']}; предварительные: {counts['preliminary_signals']}; "
                        f"крупные темы: {counts['established']}; отклонено: {counts['excluded_off_direction']}. "
                        f"Исследований после подготовки: {result['preparation']['retained_studies']}. "
                        "TOP содержит кандидатов с ростом в сохранённом корпусе; новизна требует отдельной проверки; добивки до 15 нет.")
        self.block_labels = {f"{label} ({counts[key]})": key for key, label in RESULT_BLOCKS}
        self.block_box.configure(values=tuple(self.block_labels))
        first_block = next((label for label, key in self.block_labels.items() if counts[key]),
                           next(iter(self.block_labels)))
        self.result_block.set(first_block)
        preparation = result["preparation"]
        reasons = {"conflicting_year": "Конфликт года публикации",
                   "unknown_or_future_year": "Неизвестный год или год за пределами окна",
                   "unsupported_document_type": "Неподходящий тип документа",
                   "service_or_short_title": "Служебная запись или короткое название",
                   "missing_or_short_abstract": "Нет достаточной аннотации",
                   "oversized_abstract_requires_review": "Слишком длинная аннотация",
                   "invalid_source_url": "Некорректная ссылка",
                   "connection_not_established": "Связь с направлением не установлена"}
        warnings = list(result["warnings"])
        for note in result.get("data_quality_notes") or []:
            if note not in warnings:
                warnings.append(note)
        lines = [*warnings, "", "Разделы выдачи:",
                 *(f"{label}: {counts[key]}" for key, label in RESULT_BLOCKS),
                 "Крупная тема: доля работ в доступном корпусе за выбранное окно выше 8%. "
                 "Это объёмный критерий, а не доказательство зрелости технологии в мире.",
                 "", "Подготовка данных:",
                 f"Исходных записей: {result.get('temporal_selection', {}).get('raw_occurrences', preparation['input_occurrences'])}; осталось исследований: {preparation['retained_studies']}.",
                 f"Объединено вероятных версий: {preparation['possible_versions_merged']}.",
                 "Исключено по причинам:"]
        lines.extend(f"• {reasons.get(reason, 'Недостаточно признаков направления')}: {count}"
                     for reason, count in preparation["rejected"].items())
        input_selection = result.get("temporal_selection", {})
        if input_selection.get("excluded_retracted_occurrences"):
            lines.append(f"• Явно отозванные публикации: {input_selection['excluded_retracted_occurrences']}.")
        for review in input_selection.get("date_reviews", []):
            lines.extend([f"• Спорная дата: {review['doi']}. {review['reason_ru']}",
                          *(source["url"] for source in review["sources"])])
        lines.extend(["", "Покрытие:"])
        lines.extend(f"{year}: " + ("; ".join(issues) if issues else "выдача запроса полная")
                     for year, issues in result["coverage"].items())
        comparability = result.get("growth_comparability")
        if comparability is not None:
            lines.extend(["", "Сопоставимость динамики:",
                          "Сравнение относится к сохранённым исследованиям после подготовки данных, "
                          "а не ко всей научной литературе.",
                          "Календарное покрытие: " + ("полное" if comparability["calendar_complete"] else "неполное") + ".",
                          "Рост: " + ("сопоставим в сохранённой выборке" if result.get("growth_data_comparable")
                                      else "сопоставимость не подтверждена") + "."])
            blockers = [(year, issues) for year, issues in comparability["blocking_issues"].items() if issues]
            if blockers:
                lines.append("Блокирующие причины:")
                lines.extend(f"{year}: " + "; ".join(issues) for year, issues in blockers)
            else:
                lines.append("Блокирующие причины: не обнаружены.")
            labels = {"positive": "положительный", "zero": "нулевой",
                      "unknown": "неизвестный", "invalid": "некорректный"}
            lines.append("Знаменатель направления по годам:")
            for row in comparability["denominators"]:
                value = row["documents"] if row["documents"] is not None else "неизвестен"
                lines.append(f"{row['year']}: {value} — {labels[row['state']]}")
        else:
            lines.append("Подробная оценка сопоставимости отсутствует в сохранённом результате.")
        semantic = result.get("semantic_relevance")
        if semantic:
            model = semantic.get("model") or {}
            model_id = semantic.get("model_id") or model.get("model_id", "не указана")
            revision = semantic.get("model_revision") or model.get("revision", "не указана")
            lines.extend(["", "Смысловая проверка направления:",
                          f"Локальная модель: {model_id}; ревизия: {revision}.",
                          f"Порог сходства: {semantic.get('threshold', 'не указан')}; "
                          + ("порог не откалиброван на независимой разметке."
                             if not semantic.get("calibrated") else "калибровка указана в результате."),
                          "Сходство текстов — не вероятность релевантности или успеха технологии.",
                          f"Оценено текстов: {semantic.get('scored_texts', 0)}; "
                          f"сохранено исследований: {semantic.get('retained_studies', 0)}.",
                          f"Прошли лексические правила: {semantic.get('lexical_studies', 0)}; "
                          f"добавлены только по сходству: {semantic.get('semantic_only_studies', 0)}; "
                          f"без смысловой оценки: {semantic.get('unscored_studies', 0)}.",
                          "Если смысловая проверка расширила корпус, все его группы остаются предварительными: "
                          "добавленные работы влияют на общую тематическую модель и годовые доли."])
            if semantic.get("guarded_direction"):
                lines.append("Для этого направления смысловая оценка показана как диагностика; "
                             "допуск сохраняет существующие предметные правила.")
        lines.extend(["", "Рейтинг 0–100 — предварительная эвристика, не вероятность успеха.",
                      "35% рост доли + 20% недавнее первое наблюдение + 15% устойчивость + 20% связность + 10% объём свидетельств.",
                      "G рассчитан со сглаживанием α=0.5, в том числе при нулевой базе. При отсутствии данных G не определён. Стадия не устанавливается по возрасту автоматически.",
                      "Группировка: локальная тематическая модель TF-IDF + NMF.",
                      "Score задаёт порядок исследовательской проверки; это не вероятность новизны или раннего weak signal.",
                      f"Версия: {result['pipeline_version']}; отпечаток: {result['fingerprint'][:16]}."])
        self.set_text(self.diagnostics, "\n".join(lines))
        self.select_block()
        self.controls()

    def select_block(self, event=None):
        self.tree.delete(*self.tree.get_children())
        self.current_sources = []
        if not self.result:
            self.controls()
            return
        key = self.block_labels.get(self.result_block.get(), "candidates")
        candidates = self.result.get(key, [])
        descriptions = {
            "candidates": "До 15 кандидатов с ростом в сохранённом корпусе. Новизна и стадия зарождения требуют отдельной проверки.",
            "preliminary_signals": "Предварительные сигналы: рост или подтверждающие сведения требуют проверки.",
            "established": "Крупные темы занимают более 8% доступного корпуса в окне. Это не оценка мировой зрелости.",
            "excluded_off_direction": "Отклонено фильтром направления. Причины и исходные свидетельства доступны в карточке.",
        }
        self.block_description.set(descriptions[key])
        for rank, candidate in enumerate(candidates, 1):
            self.tree.insert("", "end", iid=candidate["id"], values=(rank, candidate["title"],
                             f"{candidate['metrics']['score']:.1f}", candidate["study_count"]))
        if candidates:
            self.tree.selection_set(candidates[0]["id"])
            self.select_candidate()
        else:
            self.set_text(self.detail, "В этом разделе нет результатов. Проверьте другие разделы и качество данных.")
        self.controls()

    def select_candidate(self, event=None):
        selected = self.tree.selection()
        if not selected or not self.result:
            return
        key = self.block_labels.get(self.result_block.get(), "candidates")
        candidate = next((c for c in self.result.get(key, []) if c["id"] == selected[0]), None)
        if candidate is None:
            return
        metrics = candidate["metrics"]
        corpus_year = metrics.get("first_observed_year_in_corpus", metrics.get("first_observed_year"))
        window_year = metrics.get("first_observed_year_in_window")
        if "first_observed_year_in_window" not in metrics:
            window_year = next((row["year"] for row in metrics["years"] if row["documents"]), None)
        lines = [candidate["title"], f"Предварительный рейтинг: {metrics['score']:.1f}/100", "",
                 f"Первое подтверждённое наблюдение в доступном корпусе: {corpus_year if corpus_year is not None else 'не найдено'}",
                 f"Первое наблюдение в выбранном окне: {window_year if window_year is not None else 'не найдено'}",
                 "Первый год упоминания в мире не устанавливается.",
                 "Рост доли (сглаженный G): " + (f"×{metrics['growth_ratio']:.2f}" if metrics["growth_ratio"] is not None else "не определён"),
                 "Динамика в сохранённом корпусе (год: работ / направление):"]
        lines.extend(f"{r['year']}: {r['documents']} / {r['direction_documents']}" for r in metrics["years"])
        selection = candidate.get("selection", {})
        if selection:
            lines.extend(["", "Причина размещения:", *(_reason_text(reason) for reason in selection.get("reasons", []))])
            share = selection.get("window_document_share")
            if share is not None:
                lines.append(f"Доля работ темы в доступном корпусе за окно: {share:.1%}.")
        guard = candidate.get("direction_guard")
        if guard:
            labels = {"supported": "обе оси найдены", "partial": "частичная поддержка",
                      "off_direction": "направление не подтверждено"}
            lines.extend(["", "Проверка направления: " + labels.get(guard.get("axis_check"), "требуется проверка")])
            for axis, matches in guard.get("cluster_matches", {}).items():
                lines.append(("Носитель" if axis == "carrier" else "Операция") + ": " + (", ".join(matches) or "не найдено в названии"))
            if guard.get("axis_check") == "off_direction":
                lines.append(_reason_text(guard.get("reason")))
        execution = candidate.get("execution")
        if execution:
            lines.extend(["", "Исполнитель: " + EVIDENCE_LEVELS.get(execution.get("evidence_level"), "требуется проверка")])
            if execution.get("evidence_level") != "not_applicable":
                lines.append(f"Документов с прямым свидетельством: {execution.get('direct_documents', 0)}; "
                             f"с тематическим упоминанием: {execution.get('nominal_documents', 0)}.")
        rejected = candidate.get("rejected_quotes", {})
        rejected_count = sum(len(items) for items in rejected.values()) if isinstance(rejected, dict) else len(rejected)
        if rejected_count:
            lines.append(f"Отклонено неподходящих фрагментов: {rejected_count}. Подробности сохранены в JSON.")
        for field, title in (("problem", "Проблема"), ("advantage", "Преимущество"), ("example", "Пример исследования")):
            item = candidate["card"][field]
            explanation = candidate.get("explanations", {}).get(field)
            if explanation:
                lines.extend(["", title + " — краткое пояснение:", explanation])
            lines.extend(["", title + " — фрагмент источника:",
                          item["text"] if item else "Явного описания в выбранных аннотациях не найдено."])
            if item:
                lines.append(item["url"])
            annotation = candidate.get("evidence_annotations", {}).get(field)
            if annotation and annotation.get("context"):
                lines.extend(["Контекст из той же аннотации:", *annotation["context"]])
            if annotation:
                modality = annotation.get("modality", "unknown")
                lines.append("Характер свидетельства: " + MODALITIES.get(modality, "требуется проверка"))
                reason = annotation.get("reason")
                if reason:
                    lines.append(_reason_text(reason))
        lines.extend(["", "Ограничения:", *candidate["limitations"], "", "Основные источники:"])
        lines.extend(f"{index}. {source['title']} ({source['year']})\n{source['url']}" +
                     ("\nСвидетельство: " + EVIDENCE_LEVELS.get(source["evidence_level"], "требуется проверка")
                      if "evidence_level" in source else "")
                     for index, source in enumerate(candidate["sources"], 1))
        self.current_sources = candidate["sources"]
        self.set_text(self.detail, "\n".join(lines))
        self.controls()

    def open_source(self):
        if self.current_sources and self.app.ready and not self.app.closing:
            accepted = self.app.controller.call("ml_open_source", "open_url", lambda value: None,
                lambda error: self.background_failed(error, "Не удалось открыть источник. Проверьте браузер и ссылку."),
                self.current_sources[0]["url"])
            if not accepted:
                self.notice.set("Открытие источника уже выполняется или приложение закрывается.")

    def export(self):
        if not self.result or not self._can_start():
            return
        if self._result_context != self._input_context():
            self._input_changed()
            return
        path = filedialog.asksaveasfilename(parent=self.parent, title="Результат анализа", defaultextension=".json",
                                            initialfile="trends.json", filetypes=[("JSON", "*.json")])
        if path and self._submit(_Operation("export"), "ml_export", "ml_export", self.exported,
                                 self.result, path, protected_paths=[self.snapshot_path], overwrite=True):
            self.status.set("Сохраняем результат JSON…")

    def exported(self, path):
        self.status.set("Результат сохранён: " + path)
        self.controls()

    def background_failed(self, error, message):
        """An auxiliary request never changes ownership of analysis or export."""
        if not self.app.closing:
            self.notice.set(message)

    def failed(self, error):
        from app.ml.contracts import AnalysisInputError
        if isinstance(error, CancelledError):
            message = "Операция отменена. Исходные документы сохранены."
        elif isinstance(error, AnalysisInputError):
            message = str(error)
        elif isinstance(error, ImportError):
            lock = "requirements/semantic.lock" if self.semantic_mode.get() else "requirements/ml.lock"
            message = f"Установите ML-зависимости: python -m pip install -r {lock}"
        elif isinstance(error, ValueError):
            message = "Проверьте тему и годы: нужны 4–30 завершённых лет."
        else:
            message = "Не удалось выполнить операцию. Проверьте источник или JSON-корпус и повторите."
        self.status.set(message)
        self.controls()
