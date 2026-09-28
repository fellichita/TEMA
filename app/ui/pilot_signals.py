"""Desktop evidence pilot: explicit imports, immutable runs, and readable queues."""

from __future__ import annotations

from datetime import date
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from app.ui.viewport import ScrollViewport

_SOURCES = ("Wordstat — динамика", "CORDIS — проекты", "Инвестиции — CSV", "arXiv — Atom")
_KINDS = dict(zip(_SOURCES, ("wordstat", "cordis", "investment_csv", "arxiv"), strict=True))
_FIELDS = {
    "wordstat": (("date_column", "Колонка месяца", "Месяц"), ("count_column", "Колонка запросов", "Запросов"),
                 ("share_column", "Колонка доли", "Доля"), ("date_format", "Формат месяца", "MM.YYYY"),
                 ("share_unit", "Единица доли", "percent"), ("expected_from", "Первый месяц ГГГГ-ММ", ""),
                 ("expected_to", "Последний полный месяц ГГГГ-ММ", "")),
    "cordis": (("project_id_column", "ID проекта", "id"), ("title_column", "Название", "title"),
               ("objective_column", "Описание цели", "objective"),
               ("ec_contribution_column", "Вклад ЕС", "ecMaxContribution"),
               ("ec_signature_column", "Дата соглашения", "ecSignatureDate"),
               ("money_format", "Формат денег", "decimal_dot"),
               ("signature_format", "Формат даты", "YYYY-MM-DD")),
    "investment_csv": (("event_id_column", "ID раунда", "id"),
                       ("recipient_id_column", "ID компании", "company_id"),
                       ("recipient_name_column", "Компания", "company"),
                       ("kind_column", "Тип сделки", "kind"), ("status_column", "Статус", "status"),
                       ("date_column", "Дата", "date"), ("date_format", "Формат даты", "YYYY-MM-DD"),
                       ("amount_column", "Сумма", "amount"), ("currency_column", "Валюта", "currency"),
                       ("money_format", "Формат денег", "decimal_dot"),
                       ("description_column", "Описание", "description")),
}


class SignalsWindow:
    def __init__(self, app) -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Сигналы из нескольких источников")
        self.window.geometry("920x740")
        self.window.minsize(680, 520)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.query_hash: str | None = None
        self.concept_hash: str | None = None
        self.concept_id: str | None = None
        self.scientific_run_id: str | None = None
        self.scientific_candidate_id: str | None = None
        self.run_id: str | None = None
        self.receipts: dict[str, str] = {}
        self.association_hashes: tuple[str, ...] = ()
        self.profile: dict | None = None
        self.findings: dict[str, dict] = {}
        self.watch_active = False
        self.mapping: dict[str, tk.StringVar] = {}
        self.message = tk.StringVar(value="Сначала подтвердите область и название технологии.")
        ttk.Label(self.window, textvariable=self.message, wraplength=850, style="Muted.TLabel").pack(fill="x", padx=18, pady=10)
        self.sources_note = tk.StringVar(value="Проверяем доступность источников…")
        ttk.Label(self.window, textvariable=self.sources_note, wraplength=850,
                  style="Muted.TLabel").pack(fill="x", padx=18, pady=(0, 8))
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.input_page = ScrollViewport(notebook, padding=14, auto_hide=True)
        notebook.add(self.input_page, text="Подготовка")
        self.result_page = ScrollViewport(notebook, padding=14, auto_hide=True)
        notebook.add(self.result_page, text="Карточки и история")
        self.notebook = notebook
        self._build_input(self.input_page.content)
        self._build_result(self.result_page.content)
        self._call("source-status", "status", self._source_status)
        self._refresh_history()

    def _source_status(self, status: dict) -> None:
        source = status.get("signal_sources", {})
        api = source.get("wordstat_api", "disabled")
        api_text = {"disabled": "выключен", "key_missing": "нет ключа",
                    "key_unavailable": "ключ недоступен", "configured_unverified": "настроен, доступ не проверен"}.get(api, api)
        self.sources_note.set("Источники: научная модель — " +
                              ("готова" if source.get("scientific") == "model_ready" else "недоступна") +
                              f"; arXiv, Wordstat CSV, CORDIS CSV — локальный импорт; Wordstat API — {api_text}.")

    def _call(self, key, method, success, *args, **kwargs) -> bool:
        def done(value):
            if self.window.winfo_exists():
                success(value)

        def failed(error):
            if self.window.winfo_exists():
                self.message.set(str(error))
                messagebox.showerror("Сигналы", str(error), parent=self.window)

        accepted = self.app.controller.call(("signals", str(self.window), key), "pilot_" + method,
                                            done, failed, *args, **kwargs)
        if not accepted:
            self.message.set("Дождитесь завершения текущего действия.")
        return accepted

    def _row(self, parent, title: str, variable: tk.StringVar, *, width=44) -> None:
        line = ttk.Frame(parent)
        line.pack(fill="x", pady=3)
        ttk.Label(line, text=title, width=31, anchor="w").pack(side="left")
        ttk.Entry(line, textvariable=variable, width=width).pack(side="left", fill="x", expand=True)

    def _build_input(self, parent) -> None:
        ttk.Label(parent, text="1. Подтвердите смысл технологии", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.query = tk.StringVar()
        self.phrase = tk.StringVar()
        self._row(parent, "Направление", self.query)
        self._row(parent, "Точное название / запрос", self.phrase)
        ttk.Label(parent, text="Определение: чем эта технология отличается от соседних тем?").pack(anchor="w", pady=(6, 2))
        self.definition = tk.Text(parent, height=3, wrap="word")
        self.definition.pack(fill="x")
        ttk.Button(parent, text="Подтвердить область и название", command=self._confirm_query).pack(anchor="w", pady=(8, 22))

        ttk.Label(parent, text="2. Импортируйте разрешённую выгрузку", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Оригинальный файл хранится локально. CSV: выберите колонки и единицы доли явно; "
                  "для сравнения Wordstat укажите весь диапазон полных месяцев.", wraplength=740,
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 10))
        self.source = tk.StringVar(value=_SOURCES[0])
        source = ttk.Combobox(parent, textvariable=self.source, values=_SOURCES, state="readonly")
        source.pack(fill="x")
        source.bind("<<ComboboxSelected>>", lambda _: self._source_changed())
        self.path = tk.StringVar()
        file_line = ttk.Frame(parent)
        file_line.pack(fill="x", pady=6)
        ttk.Entry(file_line, textvariable=self.path).pack(side="left", fill="x", expand=True)
        ttk.Button(file_line, text="Выбрать файл", command=self._browse).pack(side="left", padx=6)
        self.encoding = tk.StringVar(value="utf-8-sig")
        self.delimiter = tk.StringVar(value=";")
        formats = ttk.Frame(parent)
        formats.pack(anchor="w", pady=4)
        ttk.Label(formats, text="Кодировка").pack(side="left")
        ttk.Combobox(formats, textvariable=self.encoding, values=("utf-8-sig", "cp1251"),
                     state="readonly", width=12).pack(side="left", padx=6)
        ttk.Label(formats, text="Разделитель").pack(side="left")
        ttk.Combobox(formats, textvariable=self.delimiter, values=(";", ",", "\t"),
                     state="readonly", width=5).pack(side="left", padx=6)
        ttk.Button(formats, text="Показать 5 строк", command=self._preview).pack(side="left", padx=10)
        self.preview = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.preview, wraplength=780, justify="left",
                  style="Muted.TLabel").pack(anchor="w", pady=6)
        self.mapping_frame = ttk.Frame(parent)
        self.mapping_frame.pack(fill="x", pady=8)
        self._source_changed()
        self.rights = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Подтверждаю право локально хранить этот файл и использовать его в анализе",
                        variable=self.rights).pack(anchor="w", pady=6)
        self.share_rights = tk.BooleanVar(value=False)
        self.license_ref = tk.StringVar(value="")
        ttk.Checkbutton(parent, text="Подтверждаю право передать этот источник вместе с карточкой другому пользователю",
                        variable=self.share_rights).pack(anchor="w", pady=4)
        self._row(parent, "Ссылка или основание лицензии для передачи", self.license_ref)
        ttk.Button(parent, text="Импортировать выбранный источник", command=self._import).pack(anchor="w", pady=(3, 20))
        api = ttk.LabelFrame(parent, text="Wordstat Cloud API — если настроен", padding=10)
        api.pack(fill="x", pady=(0, 20))
        today = date.today()
        last_year = today.year if today.month > 1 else today.year - 1
        last_month = today.month - 1 if today.month > 1 else 12
        first_index = last_year * 12 + last_month - 35
        self.api_first = tk.StringVar(value=f"{(first_index - 1) // 12:04d}-{(first_index - 1) % 12 + 1:02d}")
        self.api_last = tk.StringVar(value=f"{last_year:04d}-{last_month:02d}")
        ttk.Label(api, text="Платный запрос только по основной фразе; до 3 попыток в час и 20 в сутки по умолчанию. "
                  "Единица доли API пока не подтверждена реальным образцом: рост по ней не оценивается.",
                  wraplength=730, style="Muted.TLabel").pack(anchor="w", pady=(0, 5))
        self._row(api, "Первый полный месяц ГГГГ-ММ", self.api_first)
        self._row(api, "Последний полный месяц ГГГГ-ММ", self.api_last)
        ttk.Button(api, text="Запросить динамику Wordstat (платно)", command=self._fetch_wordstat).pack(anchor="w", pady=5)
        ttk.Label(parent, text="3. Рассчитайте профиль", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Импорты считаются отдельно. Подтверждение финансовой связи доступно после первого просмотра.",
                  wraplength=740, style="Muted.TLabel").pack(anchor="w", pady=5)
        ttk.Button(parent, text="Связать с сохранённой научной карточкой", command=self._choose_science).pack(anchor="w", pady=5)
        self.science_note = tk.StringVar(value="Научная карточка не выбрана.")
        ttk.Label(parent, textvariable=self.science_note, wraplength=740,
                  style="Muted.TLabel").pack(anchor="w", pady=4)
        ttk.Button(parent, text="Рассчитать карточки", command=self._start).pack(anchor="w", pady=5)

    def _source_changed(self) -> None:
        for child in self.mapping_frame.winfo_children():
            child.destroy()
        self.mapping.clear()
        kind = _KINDS[self.source.get()]
        for key, title, default in _FIELDS.get(kind, ()):
            variable = tk.StringVar(value=default)
            self.mapping[key] = variable
            self._row(self.mapping_frame, title, variable)
        if kind == "wordstat":
            ttk.Label(self.mapping_frame, text="Доли: percent = 0,25%; fraction = 0,0025; unknown не даёт рост.",
                      style="Muted.TLabel").pack(anchor="w")

    def _browse(self) -> None:
        extension = "*.xml" if _KINDS[self.source.get()] == "arxiv" else "*.csv"
        chosen = filedialog.askopenfilename(parent=self.window, filetypes=(("Выгрузка", extension), ("Все файлы", "*")))
        if chosen:
            self.path.set(chosen)

    def _confirm_query(self) -> None:
        query, phrase = self.query.get().strip(), self.phrase.get().strip()
        definition = self.definition.get("1.0", "end-1c").strip()
        self._call("query", "create_signal_query", self._query_ready, query, definition, phrase)

    def _query_ready(self, value: dict) -> None:
        self.query_hash, self.concept_hash = value["query_profile_hash"], value["concept_hash"]
        self.concept_id = value["concept_id"]
        self.scientific_run_id = self.scientific_candidate_id = None
        self.science_note.set("Научная карточка не выбрана.")
        self.receipts.clear()
        self.association_hashes = ()
        self.message.set("Область и название сохранены. Теперь выберите выгрузку.")

    def _preview(self) -> None:
        if _KINDS[self.source.get()] == "arxiv":
            self.message.set("Atom-файл проверяется во время импорта.")
            return
        self._call("preview", "preview_signal_csv", self._preview_ready,
                   self.path.get(), self.encoding.get(), self.delimiter.get())

    def _preview_ready(self, value: dict) -> None:
        rows = ["Колонки: " + ", ".join(value["headers"]), f"Всего строк: {value['row_count']}"]
        rows.extend(" | ".join(str(cell)[:50] for cell in row) for row in value["rows"])
        self.preview.set("\n".join(rows)[:2500])

    def _import(self) -> None:
        if self.query_hash is None:
            self.message.set("Сначала подтвердите область и название технологии.")
            return
        if not self.rights.get():
            self.message.set("Подтвердите право локально хранить этот файл.")
            return
        if self.share_rights.get() and not self.license_ref.get().strip():
            self.message.set("Для передачи укажите ссылку или основание лицензии.")
            return
        share_options = {"share_confirmed": self.share_rights.get(),
                         "license_ref": self.license_ref.get().strip() if self.share_rights.get() else None}
        kind = _KINDS[self.source.get()]
        if kind == "arxiv":
            self._call("import", "import_signal_atom", self._imported, self.query_hash,
                       self.path.get(), retention_confirmed=True, **share_options)
            return
        mapping = {key: variable.get().strip() for key, variable in self.mapping.items()}
        if kind == "wordstat":
            mapping["phrase"] = self.phrase.get().strip()
            for field in ("expected_from", "expected_to"):
                if mapping[field]:
                    try:
                        date.fromisoformat(mapping[field] + "-01")
                    except ValueError:
                        self.message.set("Укажите месяцы как ГГГГ-ММ.")
                        return
        self._call("import", "import_signal_csv", self._imported, self.query_hash,
                   self.path.get(), kind, mapping, self.encoding.get(), self.delimiter.get(),
                   retention_confirmed=True, **share_options)

    def _imported(self, value: dict) -> None:
        self.receipts[value["source"]] = value["receipt_hash"]
        self.message.set(f"{value['source']}: принято {value['accepted']}, отклонено {value['rejected']}. "
                         "Можно добавить другой источник или рассчитать профиль.")
        if value["source"] == "arxiv" and value["accepted"]:
            self._call("arxiv-candidates", "signal_arxiv_candidates", self._arxiv_dialog,
                       value["receipt_hash"])

    def _fetch_wordstat(self) -> None:
        if self.query_hash is None:
            self.message.set("Сначала подтвердите область и основную фразу.")
            return
        self._call("wordstat-api", "fetch_signal_wordstat", self._imported,
                   self.query_hash, self.api_first.get().strip(), self.api_last.get().strip())

    def _arxiv_dialog(self, rows: list[dict]) -> None:
        if not rows:
            return
        dialog = tk.Toplevel(self.window)
        dialog.title("Привязать препринт к технологии")
        dialog.geometry("720x450")
        ttk.Label(dialog, text="Выберите препринт только если он действительно описывает подтверждённую технологию.",
                  wraplength=670).pack(anchor="w", padx=12, pady=12)
        selected = tk.StringVar()
        by_label = {f"{index + 1}. {row['title'][:90]}": row for index, row in enumerate(rows)}
        chooser = ttk.Combobox(dialog, textvariable=selected, values=tuple(by_label), state="readonly")
        chooser.pack(fill="x", padx=12)
        abstract = tk.Text(dialog, height=12, wrap="word", state="disabled")
        abstract.pack(fill="both", expand=True, padx=12, pady=10)

        def show(_=None):
            item = by_label.get(selected.get())
            if item:
                abstract.configure(state="normal")
                abstract.delete("1.0", "end")
                abstract.insert("1.0", item["abstract"] or "Аннотация отсутствует; проверьте оригинал отдельно.")
                abstract.configure(state="disabled")

        def link():
            item = by_label.get(selected.get())
            receipt = self.receipts.get("arxiv")
            if item and receipt and self.concept_hash:
                self._call("arxiv-link", "link_signal_arxiv", lambda digest: self._arxiv_linked(digest, dialog),
                           self.concept_hash, receipt, item["revision_id"])

        chooser.bind("<<ComboboxSelected>>", show)
        ttk.Button(dialog, text="Привязать выбранный препринт", command=link).pack(anchor="w", padx=12, pady=8)

    def _arxiv_linked(self, digest: str, dialog) -> None:
        self.concept_hash = digest
        dialog.destroy()
        self.message.set("Препринт привязан как непроверенная научная публикация. Можно рассчитать профиль.")

    def _choose_science(self) -> None:
        if self.concept_hash is None:
            self.message.set("Сначала подтвердите область и название технологии.")
            return
        self._call("science-runs", "signal_scientific_runs", self._science_runs)

    def _science_runs(self, rows: list[dict]) -> None:
        if not rows:
            self.message.set("Сохранённых научных результатов пока нет.")
            return
        dialog = tk.Toplevel(self.window)
        dialog.title("Связать научную карточку")
        dialog.geometry("760x420")
        ttk.Label(dialog, text="Выберите уже проверенный результат, затем карточку именно этой технологии.",
                  wraplength=700).pack(anchor="w", padx=12, pady=10)
        run = tk.StringVar()
        names = {f"{row['created_at'][:16]} · {row['query'][:75]} · {row['run_id'][:10]}": row for row in rows}
        run_picker = ttk.Combobox(dialog, textvariable=run, values=tuple(names), state="readonly")
        run_picker.pack(fill="x", padx=12)
        card = tk.StringVar()
        picker = ttk.Combobox(dialog, textvariable=card, values=(), state="readonly")
        picker.pack(fill="x", padx=12, pady=10)
        detail = tk.Text(dialog, height=8, wrap="word", state="disabled")
        detail.pack(fill="both", expand=True, padx=12)
        cards: dict[str, dict] = {}

        def show_cards(values: list[dict]) -> None:
            cards.clear()
            cards.update({f"{value['label'][:80]} · {value['category']} · {value['candidate_id'][:10]}": value
                          for value in values})
            picker.configure(values=tuple(cards))
            card.set("")

        def select_run(_=None) -> None:
            selected = names.get(run.get())
            if selected:
                self._call("science-cards", "signal_scientific_cards", show_cards, selected["run_id"])

        def show_card(_=None) -> None:
            selected = cards.get(card.get())
            if selected:
                detail.configure(state="normal")
                detail.delete("1.0", "end")
                detail.insert("1.0", selected["definition"])
                detail.configure(state="disabled")

        def confirm() -> None:
            selected_run, selected_card = names.get(run.get()), cards.get(card.get())
            if selected_run and selected_card and self.concept_id:
                self.scientific_run_id = selected_run["run_id"]
                self.scientific_candidate_id = selected_card["candidate_id"]
                self.science_note.set(f"Связано: {selected_card['label']} · {selected_card['category']}")
                dialog.destroy()

        ttk.Button(dialog, text="Подтвердить связь", command=confirm).pack(anchor="e", padx=12, pady=10)
        picker.bind("<<ComboboxSelected>>", show_card)
        run_picker.bind("<<ComboboxSelected>>", select_run)

    def _start(self) -> None:
        if self.query_hash is None or self.concept_hash is None or (not self.receipts and not self.scientific_run_id):
            self.message.set("Подтвердите область и выберите хотя бы один источник или научную карточку.")
            return
        self._call("start", "start_signals", self._started, self.query_hash, (self.concept_hash,),
                   wordstat_receipt_hash=self.receipts.get("wordstat"),
                   arxiv_receipt_hash=self.receipts.get("arxiv"),
                   capital_receipt_hashes=tuple(self.receipts[source] for source in ("cordis", "investment_csv")
                                                if source in self.receipts),
                   association_hashes=self.association_hashes,
                   base_result_run_id=self.scientific_run_id,
                   scientific_links=({"concept_id": self.concept_id,
                                      "candidate_id": self.scientific_candidate_id},)
                   if self.scientific_run_id and self.concept_id and self.scientific_candidate_id else ())

    def _started(self, run_id: str) -> None:
        self.run_id = run_id
        self.message.set("Расчёт запущен. Проверяем сохранённые источники…")
        self._poll()

    def _poll(self) -> None:
        if self.run_id is not None and self.window.winfo_exists():
            self._call("poll", "get", self._status, self.run_id)

    def _status(self, value: dict) -> None:
        state = value["state"]
        self.message.set(value.get("error") or value.get("message") or f"Состояние: {state}")
        if state == "succeeded":
            self._call("result", "signal_result", self._render, self.run_id)
        elif state in {"queued", "running"}:
            self.window.after(800, self._poll)
        self._refresh_history()

    def _build_result(self, parent) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 10))
        controls.columnconfigure((0, 1), weight=1)
        for row, column, label, action in (
                (0, 0, "История сигналов", self._refresh_history),
                (0, 1, "Открыть пакет…", self._import_package),
                (1, 0, "Сохранить пакет…", self._export_package),
                (1, 1, "Сравнить с выбранным", self._compare_selected),
                (2, 0, "Отменить", self._cancel),
                (2, 1, "Продолжить", self._resume)):
            ttk.Button(controls, text=label, command=action).grid(row=row, column=column,
                                                                   sticky="ew", padx=3, pady=2)
        scenario_controls = ttk.Frame(parent)
        scenario_controls.pack(fill="x", pady=(0, 8))
        scenario_names = {"Без Wordstat": "exclude_wordstat", "Без финансирования": "exclude_funding",
                          "Без научной оценки": "exclude_science",
                          "Без крупнейшего события": "exclude_largest_disclosed_event"}
        self.scenario_names = scenario_names
        self.scenario = tk.StringVar(value="Без Wordstat")
        ttk.Combobox(scenario_controls, textvariable=self.scenario, values=tuple(scenario_names),
                     state="readonly", width=22).pack(side="left", padx=(0, 4))
        ttk.Button(scenario_controls, text="Проверить устойчивость", command=self._scenario).pack(side="left")
        self.history = ttk.Treeview(parent, columns=("date", "state", "id"), show="headings", height=5)
        for key, label, width in (("date", "Дата", 180), ("state", "Состояние", 120), ("id", "Запуск", 340)):
            self.history.heading(key, text=label)
            self.history.column(key, width=width)
        self.history.pack(fill="x", pady=(0, 10))
        self.history.bind("<Double-1>", lambda _: self._open_history())
        self.cards = ttk.Treeview(parent, columns=("queue", "name", "sources", "state"), show="headings", height=9)
        for key, label, width in (("queue", "Очередь", 100), ("name", "Технология", 230),
                                  ("sources", "Источники", 180), ("state", "Признак", 180)):
            self.cards.heading(key, text=label)
            self.cards.column(key, width=width)
        self.cards.pack(fill="both", expand=True)
        self.cards.bind("<<TreeviewSelect>>", lambda _: self._selected())
        self.detail = tk.Text(parent, height=8, wrap="word", state="disabled")
        self.detail.pack(fill="x", pady=8)
        ttk.Button(parent, text="Открыть исходные данные карточки", command=self._open_evidence).pack(anchor="w", pady=4)
        ttk.Button(parent, text="Проверить финансовые связи", command=self._show_associations).pack(anchor="w")
        self.watch_button = ttk.Button(parent, text="В наблюдение", command=self._toggle_watch, state="disabled")
        self.watch_button.pack(anchor="w", pady=4)

    def _refresh_history(self) -> None:
        self._call("history", "list_signal_runs", self._history_ready, 0, 50)

    def _export_package(self) -> None:
        if self.run_id is None:
            self.message.set("Сначала откройте сохранённый профиль.")
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".trendsignals",
                                            filetypes=(("Пакет сигналов", "*.trendsignals"),))
        if path:
            self._call("export-package", "export_signal", self._exported_package, self.run_id, path)

    def _exported_package(self, value: dict) -> None:
        self.message.set(f"Пакет сохранён: {value['path']}. Перед передачей проверьте содержимое и право на источники.")

    def _import_package(self) -> None:
        path = filedialog.askopenfilename(parent=self.window,
                                          filetypes=(("Пакет сигналов", "*.trendsignals"),))
        if path:
            self._call("import-package", "import_signal", self._imported_package, path)

    def _imported_package(self, value: dict) -> None:
        self._refresh_history()
        self._call("result", "signal_result", self._render, value["id"])

    def _history_ready(self, rows: list[dict]) -> None:
        self.history.delete(*self.history.get_children())
        for row in rows:
            self.history.insert("", "end", iid=row["id"],
                                values=(row["created_at"][:19], row["state"], row["id"]))

    def _open_history(self) -> None:
        selected = self.history.selection()
        if not selected:
            return
        self.run_id = selected[0]
        self._call("result", "signal_result", self._render, self.run_id)

    def _render(self, value: dict) -> None:
        profile = value["profile"]
        self.profile = profile
        self.run_id = value["view_id"]
        self.query_hash = profile["query_profile_hash"]
        self.concept_hash = profile["concept_artifact_hashes"][0] if len(profile["concept_artifact_hashes"]) == 1 else None
        self.concept_id = profile["findings"][0]["concept_id"] if len(profile["findings"]) == 1 else None
        self.scientific_run_id = value.get("scientific_reuse_run_id", profile.get("base_result_run_id"))
        self.scientific_candidate_id = (profile["findings"][0].get("scientific_candidate_id")
                                        if len(profile["findings"]) == 1 else None)
        self.science_note.set("Научная карточка связана." if self.scientific_candidate_id else
                              "Научная карточка не выбрана.")
        self.receipts = value["imports"]
        self.association_hashes = tuple(value["confirmed_association_hashes"])
        self.findings = {item["finding_id"]: item for item in profile["findings"]}
        self.cards.delete(*self.cards.get_children())
        self.watch_button.configure(text="В наблюдение", state="disabled")
        for item in profile["findings"]:
            self.cards.insert("", "end", iid=item["finding_id"],
                              values=(item["queue"], value["concepts"].get(item["concept_id"], item["concept_id"][:8]),
                                      ", ".join(item["origins"]),
                                      item["search_state"] or item["funding_state"] or item["scientific_category"] or "—"))
        self.message.set(f"Профиль сохранён: {len(profile['attention_ids'])} в очереди внимания, "
                         f"{len(profile['watch_ids'])} под наблюдением.")
        self.notebook.select(self.result_page)

    def _selected(self) -> None:
        selected = self.cards.selection()
        if not selected:
            return
        item = self.findings[selected[0]]
        content = (f"Основание: {item['explanation']}\nСледующая проверка: {item['next_check']}\n"
                   f"Правило: {item['rule_id']}\nИсходные наблюдения: {', '.join(item['observation_hashes']) or 'нет'}")
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", content)
        self.detail.configure(state="disabled")
        if self.run_id is not None:
            self.watch_button.configure(state="disabled")
            self._call("watch-state", "signal_watch_state",
                       lambda value, run=self.run_id: self._watch_ready(value, run),
                       self.run_id, item["concept_id"])

    def _watch_ready(self, value: dict, run_id: str | None) -> None:
        selected = self.cards.selection()
        if (not selected or run_id != self.run_id or
                self.findings[selected[0]]["concept_id"] != value["concept_id"]):
            return
        self.watch_active = bool(value["watched"])
        self.watch_button.configure(text="Убрать из наблюдения" if self.watch_active else "В наблюдение",
                                    state="normal")

    def _toggle_watch(self) -> None:
        selected = self.cards.selection()
        if not selected or self.run_id is None:
            self.message.set("Выберите сохранённую карточку.")
            return
        self.watch_button.configure(state="disabled")
        self._call("watch-change", "set_signal_watch",
                   lambda value, run=self.run_id: self._watch_ready(value, run), self.run_id,
                   self.findings[selected[0]]["concept_id"], not self.watch_active)

    def _open_evidence(self) -> None:
        selected = self.cards.selection()
        if not selected or not self.run_id:
            self.message.set("Выберите карточку в таблице.")
            return
        self._call("evidence", "signal_finding_evidence", self._evidence_dialog, self.run_id, selected[0])

    def _evidence_dialog(self, value: dict) -> None:
        dialog = tk.Toplevel(self.window)
        dialog.title("Исходные данные сигнала")
        dialog.geometry("850x600")
        dialog.minsize(620, 420)
        panel = ScrollViewport(dialog, padding=12, auto_hide=True)
        panel.pack(fill="both", expand=True)
        body = panel.content
        lines = []
        for snapshot in value["snapshots"]:
            lines.append(f"{snapshot['source']}: {snapshot['observed_at'][:19]}, "
                         f"покрытие {snapshot['coverage']}; перенос {snapshot['export_right']}.")
        for item in value["metrics"]:
            if item["metric_kind"] == "SearchMetric":
                lines.append(f"Wordstat: {item['state']}; последние 3 месяца — {item['recent_count']}, "
                             f"год назад — {item['base_count']}; изменение доли — {item['yoy_share_change'] or 'нет данных'}. "
                             f"Причины: {', '.join(item['reason_codes'])}.")
            else:
                lines.append(f"{item['source']}: {item['state']}; связанных {item['unit_kind']} — "
                             f"{item['unit_count']}; суммы событий не приписываются технологии. "
                             f"Причины: {', '.join(item['reason_codes'])}.")
        observations = value["observations"]
        search = sorted((item for item in observations if item["kind"] == "SearchObservation"),
                        key=lambda item: item["period_start"])
        if search:
            ttk.Label(body, text="Wordstat: число запросов по месяцам", style="Section.TLabel").pack(anchor="w", pady=6)
            graph = tk.Canvas(body, width=740, height=155, background="#ffffff", highlightthickness=1)
            graph.pack(fill="x", pady=4)
            counted = [(item["period_start"], item["count"]) for item in search if item["count"] is not None]
            if counted:
                peak = max(1, *(count for _, count in counted))
                def redraw(event) -> None:
                    graph.delete("all")
                    width = max(100, event.width - 30)
                    for index, (_, count) in enumerate(counted):
                        left = 12 + index * width / len(counted)
                        right = 12 + (index + 1) * width / len(counted) - 2
                        top = 133 - 112 * count / peak
                        graph.create_rectangle(left, top, right, 133, fill="#50699a", outline="")
                    graph.create_text(12, 145, text=counted[0][0][:7], anchor="w")
                    graph.create_text(event.width - 12, 145, text=counted[-1][0][:7], anchor="e")
                    graph.create_text(12, 8, text=f"макс. {peak} запросов", anchor="w")
                graph.bind("<Configure>", redraw)
        for item in observations:
            if item["kind"] == "SearchObservation":
                lines.append(f"{item['period_start'][:7]} · {item['count'] if item['count'] is not None else 'нет данных'} "
                             f"запросов · доля {item['share_raw'] or 'неизвестна'} ({item['share_unit']})")
            elif item["kind"] == "CapitalEvent":
                lines.append(f"{item['source']} · {item['kind']} · {item['source_event_id']} · "
                             f"{item['amount'] or 'сумма не раскрыта'} {item['currency'] or ''} "
                             "(сумма события, не инвестиция в технологию)")
            elif item["kind"] == "CapitalDescription":
                lines.append(f"Описание: {item['title']} — {item['description'] or 'нет текста'}")
            else:
                lines.append(f"Препринт arXiv без научной оценки: {item['title']} · {item['source_url']}")
        if value["total_observations"] > len(observations):
            lines.append(f"Показаны первые {len(observations)} из {value['total_observations']} наблюдений.")
        details = tk.Text(body, height=min(30, max(10, len(lines) + 2)), wrap="word", state="normal")
        details.pack(fill="both", expand=True, pady=8)
        details.insert("1.0", "\n".join(lines))
        details.configure(state="disabled")

    def _scenario(self) -> None:
        if self.run_id is None:
            self.message.set("Сначала откройте сохранённый профиль.")
            return
        if self.scenario_names[self.scenario.get()] == "exclude_largest_disclosed_event":
            selected = self.cards.selection()
            if not selected:
                self.message.set("Выберите карточку с раскрытым финансовым событием.")
                return
            self._call("scenario-options", "signal_largest_event_options", self._largest_event_dialog,
                       self.run_id, self.findings[selected[0]]["concept_id"])
            return
        self._call("scenario", "signal_scenario", self._scenario_dialog,
                   self.run_id, self.scenario_names[self.scenario.get()])

    def _largest_event_dialog(self, options: tuple[dict, ...]) -> None:
        if not options:
            self.message.set("В карточке нет раскрытого финансового события для этого сценария.")
            return
        dialog = tk.Toplevel(self.window)
        dialog.title("Исключить крупнейшее событие")
        dialog.geometry("650x180")
        dialog.minsize(560, 160)
        ttk.Label(dialog, text="Выберите вид, валюту и событие. Равные максимумы показаны отдельно.",
                  wraplength=610).pack(anchor="w", padx=12, pady=12)
        labels = tuple(f"{item['event_kind']} · {item['currency']} {item['amount']} · "
                       f"{item['source_event_id'] or item['event_id']}" for item in options)
        choice = tk.StringVar(value=labels[0])
        ttk.Combobox(dialog, textvariable=choice, values=labels, state="readonly", width=76).pack(
            fill="x", padx=12, pady=6)

        def submit() -> None:
            index = labels.index(choice.get())
            event = options[index]
            selected = self.cards.selection()
            if not selected:
                self.message.set("Выберите карточку для проверки устойчивости.")
                return
            concept_id = self.findings[selected[0]]["concept_id"]
            if self._call("scenario", "signal_scenario", self._scenario_dialog,
                          self.run_id, "exclude_largest_disclosed_event", concept_id,
                          event["event_kind"], event["currency"], event["event_hash"]):
                dialog.destroy()

        ttk.Button(dialog, text="Пересчитать", command=submit).pack(anchor="e", padx=12, pady=8)

    def _scenario_dialog(self, value: dict) -> None:
        dialog = tk.Toplevel(self.window)
        dialog.title("Устойчивость вывода")
        dialog.geometry("670x430")
        excluded = value.get("excluded_event")
        if excluded is not None:
            description = (f"Исключено событие {excluded['event_id']} · "
                           f"{excluded['amount']} {excluded['currency']}. ")
        else:
            description = f"Сценарий: исключены {', '.join(value['excluded_by_scenario'])}. "
        ttk.Label(dialog, text=description +
                  "Исходные файлы, научная оценка и сохранённый профиль не меняются.",
                  wraplength=630).pack(anchor="w", padx=12, pady=12)
        lines = [f"В очереди внимания: {value['attention_before']} → {value['attention_after']}"]
        for item in value["changes"]:
            line = (f"{item['concept_id'][:10]}: {item['before_queue']} → {item['after_queue']} "
                    f"({item['before_rule']} → {item['after_rule']}); "
                    f"финансирование {item['before_funding'] or 'нет данных'} → "
                    f"{item['after_funding'] or 'нет данных'}")
            if item["funding_units"] is not None:
                line += f"; независимые единицы {item['funding_units'][0]} → {item['funding_units'][1]}"
            lines.append(line)
        output = tk.Text(dialog, wrap="word", state="normal")
        output.pack(fill="both", expand=True, padx=12, pady=10)
        output.insert("1.0", "\n".join(lines))
        output.configure(state="disabled")

    def _compare_selected(self) -> None:
        selection = self.history.selection()
        if not selection or self.run_id is None or selection[0] == self.run_id:
            self.message.set("Откройте один профиль и выделите в истории другой завершённый запуск.")
            return
        self._call("compare", "signal_compare", self._comparison_dialog, self.run_id, selection[0])

    def _comparison_dialog(self, value: dict) -> None:
        dialog = tk.Toplevel(self.window)
        dialog.title("Что изменилось между запусками")
        dialog.geometry("760x480")
        dialog.minsize(600, 380)
        lines = ["Настройки сопоставимы." if value["comparable"] else
                 "Настройки не сопоставимы: " + ", ".join(value["reasons"]),
                 "Новые источники: " + (", ".join(value["new_sources"]) or "нет"),
                 "Удалённые источники: " + (", ".join(value["removed_sources"]) or "нет"),
                 f"Изменения финансовых событий: {value['event_change_count']}",
                 f"Изменения проверенных связей: {value['association_change_count']}"]
        lines.extend(f"{item['source']} · {item['source_event_id']}: {item['kind']} "
                     f"({item['event_period'] or 'дата неизвестна'})" for item in value["event_changes"])
        if value["event_changes_truncated"]:
            lines.append("Показаны первые 100 событий; полный счёт сохранён выше.")
        lines.extend(f"Связь {item['concept_id'][:10]} · {item['subject_id']}: {item['kind']} "
                     f"({item['before_status'] or 'нет'} → {item['after_status'] or 'нет'})"
                     for item in value["association_changes"])
        if value["association_changes_truncated"]:
            lines.append("Показаны первые 100 изменений связей; полный счёт сохранён выше.")
        lines.extend(f"{item['concept_id'][:10]}: {item['before_queue']} → {item['after_queue']} "
                     f"(поиск {item['before_search']} → {item['after_search']})"
                     for item in value["changed_findings"])
        output = tk.Text(dialog, wrap="word", state="normal")
        output.pack(fill="both", expand=True, padx=12, pady=12)
        output.insert("1.0", "\n".join(lines))
        output.configure(state="disabled")

    def _show_associations(self) -> None:
        if self.run_id:
            self._call("associations", "signal_associations", self._association_dialog, self.run_id)

    def _association_dialog(self, rows: list[dict]) -> None:
        if not rows:
            self.message.set("В выбранном профиле нет предложенных финансовых связей.")
            return
        dialog = tk.Toplevel(self.window)
        dialog.title("Проверка связей")
        dialog.geometry("700x510")
        selected = tk.StringVar()
        options = {f"{item['subject_id']} · {item['status']} · {item['title'][:45]}": item for item in rows}
        chooser = ttk.Combobox(dialog, textvariable=selected, values=tuple(options), state="readonly")
        chooser.pack(fill="x", padx=12, pady=12)
        evidence = tk.Text(dialog, wrap="word", height=14, state="disabled")
        evidence.pack(fill="both", expand=True, padx=12)
        reviewer = tk.StringVar()
        self._row(dialog, "Проверяющий", reviewer)

        def show(_=None):
            item = options.get(selected.get())
            if item is None:
                return
            evidence.configure(state="normal")
            evidence.delete("1.0", "end")
            evidence.insert("1.0", f"{item['title']}\n\n{item['description']}\n\nИсточник: {item['source_url'] or 'локальный CSV'}\n" +
                            ("Подтвердите связь, только если проект действительно исследует выбранную технологию."
                             if item["subject_kind"] == "project" else
                             "Инвестиционная связь здесь доступна только для просмотра; упоминание компании не доказывает вложение в технологию."))
            evidence.configure(state="disabled")
            review_button.state(["!disabled"] if item["subject_kind"] == "project" and
                                item["relation"] == "researches" and item["status"] == "proposed" else ["disabled"])

        def confirm():
            item = options.get(selected.get())
            if item is None or self.run_id is None:
                return
            self._call("review", "confirm_signal_grant", lambda digest: self._reviewed(digest, dialog),
                       self.run_id, item["hash"], reviewer.get())

        chooser.bind("<<ComboboxSelected>>", show)
        review_button = ttk.Button(dialog, text="Подтвердить связь гранта", command=confirm, state="disabled")
        review_button.pack(anchor="w", padx=12, pady=12)

    def _reviewed(self, digest: str, dialog) -> None:
        self.association_hashes = tuple(dict.fromkeys((*self.association_hashes, digest)))
        dialog.destroy()
        self.message.set("Связь подтверждена. Вернитесь к подготовке и пересчитайте новый профиль.")
        self.notebook.select(self.input_page)

    def _cancel(self) -> None:
        if self.run_id and not self.run_id.startswith("signal-import-"):
            self._call("cancel", "cancel", lambda _: self.message.set("Отмена запрошена."), self.run_id)

    def _resume(self) -> None:
        if self.run_id and not self.run_id.startswith("signal-import-"):
            self._call("resume", "resume", self._started, self.run_id)


def open_signals(app) -> None:
    existing = getattr(app, "signals_window", None)
    if existing is not None and existing.window.winfo_exists():
        existing.window.lift()
        existing.window.focus_set()
        return
    app.signals_window = SignalsWindow(app)
