"""Desktop UI: python -m app.main [--data-dir PATH]."""

import argparse
from functools import partial
from dataclasses import dataclass
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import ttk, font
from tkinter.scrolledtext import ScrolledText

from app.ui.theme import COLORS, apply_theme, load_theme, save_theme
from app.ui.display import Display, enable_high_dpi, size_window, scale_value
from app.ui.viewport import ScrollViewport
from app.ui.windows import WindowRegistry
from app.ui.controller import Controller, create_backend
from app.ui.trends_panel import TrendsPanel
from app.ui.presentation import ACTIVE, SOURCES, collection_values, error_message, job_state, publication_date, document_detail, empty_collection, empty_collection_hint

# Код виртуальной клавиши F в Windows: одинаков при любой раскладке.
WINDOWS_KEY_F = 0x46


@dataclass(frozen=True)
class _DocumentView:
    query: str
    offset: int
    job_filter: str | None
    history_filter: str | None
    document_scope: str
    sort_by: str
    descending: bool
    sort_label: str

    @property
    def label(self) -> str:
        order = self.sort_label
        if self.sort_by == "default":
            order = "порядок по идентификатору" if self.job_filter or self.history_filter else "последние полученные — первыми"
        query = f" • Поиск: «{self.query}»" if self.query else ""
        return f"{self.document_scope} • {order}{query}"


class Application:
    PAGE_SIZE = 50

    def __init__(self, root, factory=create_backend, ui_scale=None):
        self.root = root
        self.ready = self.closing = self.loading = False
        self.offset = self.total = self.generation = 0
        self.query = ""
        self._displayed_document_view: _DocumentView | None = None
        self.history_filter = None
        self.document_scope = "Все документы"
        self.child_windows = WindowRegistry()
        self.sort_by, self.descending = "default", False
        self.documents, self.jobs = {}, {}
        self._job_rows = {}
        self.job_filter = self.selected_document = None
        # The initially empty job view already matches an empty first result.
        # Only a changed snapshot should invalidate the startup document read.
        self.fingerprint: tuple[tuple[str, datetime, str, int, int, int], ...] | None = ()
        self._document_jobs_fingerprint: tuple[tuple[str, int, int], ...] = ()
        self.cancel_requested = set()
        root.title("Trendanalyser Pilot main2 — локальный поиск технологических трендов")
        self.display = Display(root, ui_scale)
        size_window(root, 1280, 820, (640, 480))
        self.theme_name = load_theme()
        self.style = apply_theme(root, self.theme_name)
        self._build()
        self._compact_chrome = False
        root.bind("<Configure>", self._window_layout, add="+")
        # Keep an explicitly installed observer (e.g. the native test harness).
        # Replace Tk's default stderr traceback in the actual desktop app.
        if getattr(root.report_callback_exception, "__func__", None) is tk.Tk.report_callback_exception:
            root.report_callback_exception = self._callback_exception
        self.display.widgets(root)
        self.controller = Controller(root, factory)
        self.controller.on_profile_transition = self._profile_transition
        self.controller.call("startup", "open", self._opened, self._startup_error)
        root.protocol("WM_DELETE_WINDOW", self.close)
        if self.display.system == "aqua":
            root.createcommand("tk::mac::Quit", self.close)
        root.bind("<Control-f>", self._focus_search)
        if self.display.system == "aqua":
            root.bind("<Command-f>", self._focus_search)
        if self.display.system == "win32":
            # При русской раскладке клавиша F даёт «а», и <Control-f> молчит.
            # Сочетание узнаём по самой клавише (VK_F), а не по её букве.
            root.bind("<Control-KeyPress>", self._control_key)
        root.bind("<F5>", lambda event: self.refresh_documents())
        self.job_timer = root.after(700, self._poll_jobs)

    def _build(self):
        self.shell = ttk.Frame(self.root, style="Shell.TFrame")
        self.shell.pack(fill="both", expand=True)
        outer = self.outer = ttk.Frame(self.shell, style="Shell.TFrame")
        outer.pack(side="right", fill="both", expand=True)
        self.status = tk.StringVar(value="Открываем локальное хранилище…")
        self.status_label = ttk.Label(outer, textvariable=self.status, wraplength=880, padding=(12, 5))
        self.retry = ttk.Button(outer, text="Повторить открытие", command=self._retry_startup)
        self.tabs = ttk.Notebook(outer, style="Pages.TNotebook")
        self.tabs.pack(fill="both", expand=True)
        self.document_tab = ScrollViewport(self.tabs, padding=12)
        self.collection_tab = ScrollViewport(self.tabs, padding=16)
        self.trends_tab = ScrollViewport(self.tabs, padding=12)
        self.tabs.add(self.document_tab, text="Документы")
        self.tabs.add(self.collection_tab, text="Разовый сбор")
        self.tabs.add(self.trends_tab, text="Технический ML")
        self._build_documents()
        self._build_collection()
        self.trends_panel = TrendsPanel(self, self.trends_tab.content)
        self.jobs_tab = ScrollViewport(self.tabs, padding=12)
        self.tabs.add(self.jobs_tab, text="Процессы")
        self._build_jobs(self.jobs_tab.content)
        from app.ui.history import HistoryPanel
        self.history_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.history_tab, text="Сбор по периодам")
        self.history = HistoryPanel(self, self.history_tab)
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        self.sources_tab = ScrollViewport(self.tabs, padding=16)
        self.tabs.add(self.sources_tab, text="Настройки")
        ttk.Label(self.sources_tab.content, text="Настройки", style="Section.TLabel").pack(anchor="w", pady=(0, 12))
        self._build_appearance(self.sources_tab.content)
        self._build_extra_tools(self.sources_tab.content)
        self.pilot_settings_area = ttk.Frame(self.sources_tab.content)
        self.pilot_settings_area.pack(fill="x", pady=(0, 16))
        # Settings and keys are part of this page; they no longer open a window.
        self.pilot_settings_form = ttk.Frame(self.sources_tab.content)
        self.pilot_settings_form.pack(fill="x", pady=(0, 16))
        self.source_urls = {"crossref": "https://www.crossref.org/", "openalex": "https://openalex.org/",
                            "epo": "https://www.epo.org/"}
        self.source_link_font = font.nametofont("TkDefaultFont").copy()
        self.source_link_font.configure(underline=True)
        self.source_links, self.source_descriptions = {}, {}
        for source, title in SOURCES.items():
            link = ttk.Label(self.sources_tab.content, text=title, foreground=COLORS["cyan"], cursor="hand2",
                             font=self.source_link_font, takefocus=True, borderwidth=1)
            link.pack(anchor="w", pady=(8, 4))
            for event in ("<Button-1>", "<Return>", "<space>"):
                link.bind(event, lambda event, source=source: self._open_source(source))
            link.bind("<FocusIn>", lambda event: event.widget.configure(relief="solid"))
            link.bind("<FocusOut>", lambda event: event.widget.configure(relief="flat"))
            description = tk.StringVar(value="Читаем настройки источника…")
            ttk.Label(self.sources_tab.content, textvariable=description, wraplength=740,
                      justify="left").pack(anchor="w", pady=(0, 12))
            self.source_links[source], self.source_descriptions[source] = link, description
        ttk.Label(self.sources_tab.content, text="Названия источников открывают их сайты в браузере.\n"
                  "Доступ по сети проверяется при сборе. Сохранённые документы доступны без интернета.",
                  wraplength=740, style="Muted.TLabel").pack(anchor="w", pady=(12, 0))
        self.storage = tk.StringVar()
        ttk.Label(self.sources_tab.content, textvariable=self.storage, style="Footer.TLabel", wraplength=600).pack(
            anchor="w", pady=(8, 0))
        from app.ui.pilot_panel import PilotPanel
        self.pilot_tab = ScrollViewport(self.tabs, padding=16, auto_hide=True)
        self.tabs.add(self.pilot_tab, text="Новый анализ")
        self.pilot_panel = PilotPanel(self, self.pilot_tab.content, settings_parent=self.pilot_settings_area,
                                      settings_form_parent=self.pilot_settings_form)
        from app.ui.analysis_history import AnalysisHistory
        self.analysis_history_tab = ScrollViewport(self.tabs, padding=20, auto_hide=True)
        self.tabs.add(self.analysis_history_tab, text="История анализов")
        self.analysis_history = AnalysisHistory(self.pilot_panel, self.analysis_history_tab.content)
        from app.ui.navigation import Sidebar
        self.navigation = Sidebar(self, self.shell)
        self.navigation.pack(side="left", fill="y", before=outer)
        # Five permanent destinations. The rest stay working pages, opened by the
        # flows that own them, and surface in the rail only while they are current.
        for key, title, glyph, page, group, hidden in (
            ("analysis", "Новый анализ", "plus", self.pilot_tab, "primary", False),
            ("analyses", "История анализов", "clock", self.analysis_history_tab, "primary", False),
            ("documents", "Документы", "file", self.document_tab, "tools", False),
            ("jobs", "Процессы", "activity", self.jobs_tab, "tools", False),
            ("collection", "Разовый сбор", "inbox", self.collection_tab, "tools", True),
            ("periods", "Сбор по периодам", "calendar", self.history_tab, "tools", True),
            ("ml", "Технический ML", "chart", self.trends_tab, "tools", True),
            ("settings", "Настройки", "sliders", self.sources_tab, "bottom", False),
        ):
            self.navigation.add(key, title, glyph, page, group=group, hidden=hidden)
        self.navigation.fit()
        self.navigation.select("analysis")

    def _build_appearance(self, parent):
        """Theme choice lives next to the other per-install preferences."""
        section = ttk.Frame(parent)
        section.pack(fill="x", pady=(0, 16))
        ttk.Label(section, text="Оформление", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(section, text="Выбор сохраняется для этой установки.",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 8))
        self.theme_choice = tk.StringVar(value=self.theme_name)
        row = ttk.Frame(section)
        row.pack(anchor="w")
        for value, label in (("light", "Светлая"), ("dark", "Тёмная")):
            ttk.Radiobutton(row, text=label, value=value, variable=self.theme_choice,
                            command=self._theme_changed).pack(side="left", padx=(0, 16))

    def _build_extra_tools(self, parent):
        """The pages kept out of the rail still need one deliberate way in.

        Two of them are opened by nothing else, and a one-off collection could
        otherwise only be started by copying an existing process. Hiding them
        without this would remove the features, not just the clutter.
        """
        from app.ui.layout import ActionRow
        section = ttk.Frame(parent)
        section.pack(fill="x", pady=(0, 16))
        ttk.Label(section, text="Дополнительные инструменты", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(section, text="Открываются по требованию и не занимают место в навигации.",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 8))
        row = ActionRow(section)
        row.pack(fill="x")
        for key, label in (("collection", "Разовый сбор"), ("periods", "Сбор по периодам"),
                           ("ml", "Технический ML")):
            row.add(ttk.Button(row, text=label, style="Ghost.TButton",
                               command=partial(self._open_page, key)))

    def _open_page(self, key):
        self.navigation.select(key)

    def _theme_changed(self):
        name = self.theme_choice.get()
        if name == self.theme_name:
            return
        self.theme_name = name
        # Dialogs are Toplevel children of the root, so one pass repaints them too.
        self.style = apply_theme(self.root, name)
        try:
            save_theme(name)
        except OSError as error:
            self._message(f"Тема применена, но не сохранена: {error}", True)

    def _window_layout(self, event):
        if event.widget is not self.root:
            return
        compact = event.height < self.display.px(740)
        if compact == self._compact_chrome:
            return
        self._compact_chrome = compact
        # main2 keeps branding in the responsive sidebar and the storage path
        # on Settings. It has no global header/footer to repack on resize, so
        # the only chrome worth reclaiming on a short window is the hero.
        panel = getattr(self, "pilot_panel", None)
        if panel is not None:
            panel.set_compact(compact)

    @staticmethod
    def _tree(parent, columns, height):
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=[col[0] for col in columns], show="headings",
                            selectmode="browse", height=height)
        for name, title, width in columns:
            tree.heading(name, text=title)
            tree.column(name, width=width, minwidth=70, stretch=name in {"title", "state"})
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        return frame, tree

    def _build_documents(self):
        ttk.Label(self.document_tab.content, text="Библиотека документов", style="Section.TLabel").pack(anchor="w", pady=(0, 12))
        bar = ttk.Frame(self.document_tab.content)
        bar.pack(fill="x")
        ttk.Label(bar, text="Поиск в сохранённых:").pack(side="left")
        self.search = ttk.Entry(bar)
        self.search.pack(side="left", fill="x", expand=True, padx=8)
        self.search.bind("<Return>", lambda event: self.search_documents())
        self.search_button = ttk.Button(bar, text="Найти", style="Primary.TButton", command=self.search_documents, state="disabled")
        self.search_button.pack(side="left")
        self.clear_button = ttk.Button(bar, text="Сбросить", command=self.clear_search, state="disabled")
        self.clear_button.pack(side="left", padx=6)
        self.scope = tk.StringVar(value="Все документы • последние полученные — первыми")
        ttk.Label(self.document_tab.content, textvariable=self.scope, style="Muted.TLabel").pack(anchor="w", pady=8)
        sorting = ttk.Frame(self.document_tab.content)
        sorting.pack(fill="x", pady=(0, 8))
        ttk.Label(sorting, text="Сортировать:").pack(side="left")
        self.sort_options = {"По умолчанию": ("default", False), "Название: А → Я": ("title", False),
                             "Название: Я → А": ("title", True), "Сначала новые публикации": ("date", True),
                             "Сначала старые публикации": ("date", False),
                             "Больше цитирований": ("citations", True), "Меньше цитирований": ("citations", False)}
        self.sort_choice = tk.StringVar(value="По умолчанию")
        self.sort_control = ttk.Combobox(sorting, textvariable=self.sort_choice,
                                        values=list(self.sort_options), state="disabled", width=32)
        self.sort_control.pack(side="left", padx=8)
        self.sort_control.bind("<<ComboboxSelected>>", self.change_sort)
        panes = ttk.Panedwindow(self.document_tab.content, orient="horizontal")
        panes.pack(fill="both", expand=True)
        listing, self.document_tree = self._tree(panes, [
            ("title", "Название", 310), ("source", "Источник", 125),
            ("date", "Публикация", 145), ("citations", "Цитирования", 140)], 8)
        self.document_tree.configure(style="Documents.Treeview")
        panes.add(listing, weight=3)
        detail = ttk.Frame(panes, padding=(12, 0, 0, 0))
        panes.add(detail, weight=2)
        ttk.Label(detail, text="Карточка документа", style="Section.TLabel").pack(anchor="w")
        self.detail = ScrolledText(detail, wrap="word", state="disabled", width=35,
                                   height=10, font="TkTextFont", padx=8, pady=8)
        self.detail.pack(fill="both", expand=True, pady=6)
        self.open_link = ttk.Button(detail, text="Открыть источник в браузере",
                                    command=self._open_link, state="disabled")
        self.open_link.pack(anchor="w")
        self.versions_button = ttk.Button(detail, text="Сохранённые версии", command=self.show_versions, state="disabled")
        self.versions_button.pack(anchor="w", pady=(6, 0))
        self.document_tree.bind("<<TreeviewSelect>>", self._select_document)
        self._set_detail("Выберите документ в списке.")
        footer = ttk.Frame(self.document_tab.content)
        footer.pack(fill="x", pady=(10, 0))
        self.previous = ttk.Button(footer, text="Назад", command=lambda: self.change_page(-1), state="disabled")
        self.previous.pack(side="left")
        self.next = ttk.Button(footer, text="Далее", command=lambda: self.change_page(1), state="disabled")
        self.next.pack(side="left", padx=6)
        self.page_label = tk.StringVar(value="Загрузка…")
        ttk.Label(footer, textvariable=self.page_label).pack(side="left", padx=8)
        self.refresh = ttk.Button(footer, text="Обновить", command=self.refresh_documents, state="disabled")
        self.refresh.pack(side="right")
        self.all_documents = ttk.Button(footer, text="Все документы", command=self._show_all, state="disabled")
        self.all_documents.pack(side="right", padx=6)

    def _build_collection(self):
        explanation = ttk.LabelFrame(self.collection_tab.content, text="Разовый сбор — быстро получить выборку по теме", padding=10)
        explanation.pack(fill="x", pady=(0, 12))
        ttk.Label(explanation, wraplength=700, justify="left", text=
                  "Используйте, чтобы познакомиться с темой или получить несколько публикаций. "
                  "Один запрос охватывает весь выбранный диапазон дат.\n"
                  "Лимит общий для диапазона у каждого источника: например, до 200 записей за 2020–2025 годы.\n"
                  "Для подготовки данных по месяцам или годам откройте вкладку «Сбор по периодам».").pack(anchor="w")
        self.form = ttk.Frame(self.collection_tab.content)
        self.form.pack(fill="both", expand=True)
        self.fields, self.field_errors = {}, {}
        labels = [("topic", "Технологическое направление", ""),
                  ("start", "Начало (ГГГГ-ММ-ДД; необязательно)", ""),
                  ("end", "Окончание (ГГГГ-ММ-ДД; пусто — сегодня UTC)", ""),
                  ("limit", "Лимит на весь диапазон у каждого источника", "200")]
        for row, (key, label, default) in enumerate(labels):
            ttk.Label(self.form, text=label).grid(row=row * 2, column=0, sticky="w", pady=(4, 0))
            field = ttk.Entry(self.form, width=42)
            field.insert(0, default)
            field.grid(row=row * 2, column=1, sticky="ew", padx=(16, 0), pady=(4, 0))
            self.fields[key] = field
            self.field_errors[key] = message = tk.StringVar()
            ttk.Label(self.form, textvariable=message, style="Error.TLabel", wraplength=390).grid(
                row=row * 2 + 1, column=1, sticky="w", padx=(16, 0))
        ttk.Label(self.form, text="Источники").grid(row=8, column=0, sticky="nw", pady=(8, 0))
        source_frame = ttk.Frame(self.form)
        source_frame.grid(row=8, column=1, sticky="w", padx=(16, 0), pady=(8, 0))
        self.source_vars, self.source_checks = {}, {}
        for source, label in SOURCES.items():
            value = tk.BooleanVar(value=source == "crossref")
            check = ttk.Checkbutton(source_frame, text=label, variable=value)
            check.pack(side="left", padx=(0, 12))
            self.source_vars[source], self.source_checks[source] = value, check
        self.field_errors["sources"] = tk.StringVar()
        ttk.Label(self.form, textvariable=self.field_errors["sources"], style="Error.TLabel").grid(
            row=9, column=1, sticky="w", padx=(16, 0))
        self.source_info = tk.StringVar(value="Проверяем настройки источников…")
        ttk.Label(self.form, textvariable=self.source_info, wraplength=760,
                  style="Muted.TLabel").grid(row=10, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Label(self.form, wraplength=760,
                  text="Crossref и OpenAlex — научные публикации, EPO — патенты. "
                       "Лимит включает пропущенные записи. Для EPO — до 2000 записей за запрос. "
                       "Сохранение выполняется автоматически по мере загрузки.", style="Muted.TLabel").grid(
            row=11, column=0, columnspan=2, sticky="w", pady=6)
        self.start_button = ttk.Button(self.form, text="Начать разовый сбор", style="Primary.TButton", command=self.start_collection,
                                       state="disabled")
        self.start_button.grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.form.columnconfigure(1, weight=1)
        self.collection_tab.canvas.bind("<Configure>", self._collection_width, add="+")

    def _collection_width(self, event):
        # Wide source selectors determine this grid column's natural size.
        # On narrow displays, keep each entry fully reachable instead of
        # stretching it wider than the surrounding horizontal viewport.
        compact = event.width < self.display.px(1000)
        character = max(1, font.nametofont("TkTextFont", root=self.root).measure("0"))
        for field in self.fields.values():
            inset = max(0, field.winfo_reqwidth() - int(field.cget("width")) * character)
            columns = max(8, min(42, (event.width - self.display.px(32) - inset) // character)) if compact else 42
            if int(field.cget("width")) != columns:
                field.configure(width=columns)
            sticky = "w" if compact else "ew"
            if str(field.grid_info()["sticky"]) != sticky:
                field.grid_configure(sticky=sticky)

    def _build_jobs(self, outer):
        frame = ttk.LabelFrame(outer, text="Последние 100 процессов — состояние и результаты", padding=8)
        frame.pack(fill="both", expand=True)
        box, self.job_tree = self._tree(frame, [
            ("title", "Тема", 200), ("source", "Источник", 100), ("state", "Состояние", 245),
            ("scanned", "Просмотрено", 100), ("stored", "Сохранено", 90), ("skipped", "Пропущено", 90)], 3)
        box.pack(fill="both", expand=True)
        self.job_tree.bind("<<TreeviewSelect>>", self._select_job)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(6, 0))
        self.cancel_button = ttk.Button(actions, text="Отменить процесс", command=self.cancel_job, state="disabled")
        self.cancel_button.pack(side="left")
        self.job_documents = ttk.Button(actions, text="Документы процесса", command=self._show_job, state="disabled")
        self.job_documents.pack(side="left", padx=6)
        self.repeat_button = ttk.Button(actions, text="Повторить с этими параметрами",
                                        command=self.repeat_job, state="disabled")
        self.repeat_button.pack(side="left", padx=6)
        self.job_detail = tk.StringVar(value="Выберите процесс, чтобы посмотреть результат или причину ошибки.")
        ttk.Label(frame, textvariable=self.job_detail, wraplength=860).pack(anchor="w", pady=(6, 0))

    def refresh_source_credentials(self, keys):
        """Refresh both collection workflows from credential presence, never values."""
        if self.closing:
            return
        epo_key, epo_secret = keys.get("epo_ops_key", False), keys.get("epo_ops_secret", False)
        configured = epo_key is True and epo_secret is True
        unavailable = epo_key is None or epo_secret is None
        if configured:
            self.source_checks["epo"].state(["!disabled"])
            state = "Ключи настроены"
            self.source_info.set("Ключи EPO настроены. Доступ к источникам по сети проверяется при сборе.")
        else:
            self.source_vars["epo"].set(False)
            self.source_checks["epo"].state(["disabled"])
            state = ("Системное хранилище ключей недоступно; локальные документы можно читать" if unavailable else
                     "Задайте ключ и секрет EPO в настройках нового анализа")
            self.source_info.set("EPO недоступен: " + ("системное хранилище ключей не отвечает. " if unavailable else
                                 "задайте ключ и секрет в настройках нового анализа. ") +
                                 "Доступ к источникам по сети проверяется при сборе.")
        self.source_descriptions["epo"].set(f"Патенты. {state}. Лимит: 2000 записей за запрос.")
        openalex = keys.get("openalex_api_key", False)
        state = ("Системное хранилище ключей недоступно; локальные документы можно читать" if openalex is None else
                 "API-ключ настроен" if openalex is True else "Необязательный API-ключ не настроен")
        self.source_descriptions["openalex"].set(f"Научные публикации. {state}.")
        self.source_descriptions["crossref"].set("Научные публикации. Ключ не требуется.")

    def _profile_transition(self, active):
        if active:
            self._ready_before_restore = self.ready
            self.ready = False
            self._tab_states_before_restore = {tab: self.tabs.tab(tab, "state") for tab in self.tabs.tabs()}
            for tab in self._tab_states_before_restore:
                self.tabs.tab(tab, state="disabled")
            self._navigation_states_before_restore = {key: button.state()
                for key, button in self.navigation.buttons.items()}
            for button in self.navigation.buttons.values():
                button.state(["disabled"])
            self._message("Восстанавливаем библиотеку. Дождитесь завершения перед следующей операцией.")
        elif not self.closing:
            self.ready = self._ready_before_restore
            for tab, state in self._tab_states_before_restore.items():
                self.tabs.tab(tab, state=state)
            for key, state in self._navigation_states_before_restore.items():
                self.navigation.buttons[key].state(("!disabled", *state))

    def _opened(self, result):
        directory, sources = result
        if getattr(self, "_opened_directory", directory) != directory:
            self.analysis_history.reset()
            self.pilot_panel.card_list.clear()
            self.pilot_panel.result_actions.pack_forget()
        self._opened_directory = directory
        self.ready = True
        self.sort_control.configure(state="readonly")
        by_id = {source["id"]: source for source in sources}
        epo, openalex = by_id["epo"], by_id["openalex"]
        epo_present = None if epo.get("credential_state") == "unavailable" else bool(epo.get("credentials_configured"))
        self.refresh_source_credentials({"epo_ops_key": epo_present, "epo_ops_secret": epo_present,
            "openalex_api_key": None if openalex.get("credential_state") == "unavailable" else bool(openalex.get("key_configured"))})
        self.retry.pack_forget()
        self.storage.set(f"Хранилище: {directory}")
        for widget in (self.search_button, self.clear_button, self.refresh, self.start_button, self.all_documents):
            widget.state(["!disabled"])
        self._message("Хранилище открыто. Просмотр и поиск работают без интернета.")
        self.refresh_documents()
        self._request_jobs()
        self.history.opened()
        self.trends_panel.ready()
        self.pilot_panel.load()

    def _startup_error(self, error):
        self._operation_error(error)
        self.page_label.set("Хранилище не открыто")
        self.retry.pack(before=self.tabs, anchor="w", pady=(0, 8))

    def _retry_startup(self):
        if not self.closing:
            self.retry.pack_forget()
            self._message("Открываем локальное хранилище…")
            self.controller.call("startup", "open", self._opened, self._startup_error)

    def _message(self, text, error=False):
        self.status.set(text)
        self.status_label.configure(style="Error.TLabel" if error else "TLabel")
        if error or self.tabs.select() not in {str(self.pilot_tab), str(self.analysis_history_tab)}:
            self.status_label.pack(fill="x", before=self.tabs)
        else:
            self.status_label.pack_forget()

    def _operation_error(self, error):
        self._message(error_message(error), True)

    def _callback_exception(self, exception_type, error, traceback):
        from app.diagnostics import log_internal_error

        log_internal_error("ui.callback_failed", error)
        self._message("Не удалось обновить интерфейс. Повторите действие или обновите список. "
                      "Сохранённые данные остаются в библиотеке.", True)

    def _control_key(self, event):
        if event.keycode == WINDOWS_KEY_F:
            return self._focus_search(event)
        return None

    def _focus_search(self, event=None):
        if not self.closing:
            self.tabs.select(self.document_tab)
            self.document_tab.focus_when_visible(self.search)
        return "break"

    def search_documents(self):
        query = self.search.get().strip()
        if len(query) > 500:
            self._message("Поиск: не больше 500 символов.", True)
            return
        self.query, self.offset = query, 0
        self.refresh_documents()

    def clear_search(self):
        self.search.delete(0, "end")
        self.search_documents()

    def change_sort(self, event=None):
        if self.ready and not self.closing:
            self.sort_by, self.descending = self.sort_options[self.sort_choice.get()]
            self.offset = 0
            self.refresh_documents()

    def change_page(self, direction):
        if not self.loading and self.ready and not self.closing:
            self.offset = max(0, self.offset + direction * self.PAGE_SIZE)
            self.refresh_documents()

    def refresh_documents(self):
        if self.ready and not self.closing:
            self.generation += 1
            self._load_documents()

    def _document_view(self) -> _DocumentView:
        return _DocumentView(self.query, self.offset, self.job_filter, self.history_filter,
                             self.document_scope, self.sort_by, self.descending, self.sort_choice.get())

    def _restore_document_view(self) -> None:
        view = self._displayed_document_view
        if view is None:
            return
        self.query, self.offset = view.query, view.offset
        self.job_filter, self.history_filter = view.job_filter, view.history_filter
        self.document_scope, self.sort_by, self.descending = view.document_scope, view.sort_by, view.descending
        self.sort_choice.set(view.sort_label)
        self.scope.set(view.label)
        self.previous.state(["!disabled"] if view.offset else ["disabled"])
        self.next.state(["!disabled"] if view.offset + len(self.documents) < self.total else ["disabled"])

    def _load_documents(self):
        if self.loading or self.closing:
            return
        self.loading = True
        generation = self.generation
        requested = self._document_view()
        self.previous.state(["disabled"])
        self.next.state(["disabled"])
        self.page_label.set("Читаем сохранённые документы…")

        def success(page):
            self.loading = False
            if generation != self.generation:
                self._load_documents()
                return
            if not page.items and page.total and self.offset >= page.total:
                self.offset = ((page.total - 1) // self.PAGE_SIZE) * self.PAGE_SIZE
                self.refresh_documents()
                return
            selected = self.document_tree.selection()
            old_key = selected[0] if selected else None
            documents = {item.document_key: item for item in page.items}
            order = tuple(item.document_key for item in page.items)
            redraw = documents != self.documents or self.document_tree.get_children() != order
            self.documents = documents
            if redraw:
                self.document_tree.delete(*self.document_tree.get_children())
                for item in page.items:
                    doc = item.document
                    self.document_tree.insert("", "end", iid=item.document_key, values=(
                        doc.title, ", ".join(SOURCES.get(source, source) for source in item.sources),
                        publication_date(doc), doc.citation_count if doc.citation_count is not None else "—"))
            self.total = page.total
            self._displayed_document_view = requested
            self.scope.set(requested.label)
            if old_key in self.documents:
                if redraw:
                    self.document_tree.selection_set(old_key)
                    self._select_document()
            else:
                self.selected_document = None
                self.open_link.state(["disabled"])
                self.versions_button.state(["disabled"])
                self._set_detail("Ничего не найдено. Измените поиск." if self.query and not page.total else
                                 self._empty_documents_message() if not page.total else
                                 "Выберите документ в списке.")
            self.page_label.set(f"{self.offset + 1}–{self.offset + len(page.items)} из {page.total}" if page.total
                                else "Документов: 0")
            self.previous.state(["!disabled"] if self.offset else ["disabled"])
            self.next.state(["!disabled"] if self.offset + len(page.items) < page.total else ["disabled"])

        def failure(error):
            self.loading = False
            if generation != self.generation:
                self._load_documents()
                return
            self.page_label.set("Ошибка обновления; показаны прежние данные")
            self._restore_document_view()
            self._operation_error(error)

        submitted = self.controller.call("documents", "list_documents", success, failure, query=self.query or None,
                             job_id=self.job_filter, history_id=self.history_filter, limit=self.PAGE_SIZE, offset=self.offset,
                             sort_by=self.sort_by, descending=self.descending)
        if not submitted:
            self.loading = False
            self._restore_document_view()
            self.page_label.set("Чтение пока недоступно. Повторите обновление после текущей операции.")

    def _empty_documents_message(self):
        if self.history_filter:
            return ("В выборке исторического сбора пока нет сохранённых документов. "
                    "Откройте «Сбор по периодам» → «Периоды и прогресс», чтобы проверить состояние и причины. "
                    "После дробления документы родительских периодов доступны отдельно в их попытках.")
        if self.job_filter:
            job = self.jobs.get(self.job_filter)
            jobs = [job] if job else []
        else:
            jobs = list(self.jobs.values())
        if any(job.state in ACTIVE for job in jobs):
            return "Сбор ещё выполняется или ожидает очереди. Документы появятся после сохранения первой порции. Состояние — во вкладке «Процессы»."
        if jobs:
            job = jobs[0]
            if empty_collection(job):
                return empty_collection_hint(job)
            if job.state == "failed":
                return "Документы не сохранены. Последний процесс завершился с ошибкой. Откройте «Процессы», чтобы посмотреть причину и повторить сбор."
            return "Документы пока не сохранены. Проверьте состояние и число пропущенных записей во вкладке «Процессы»."
        return "Пока нет документов. Для поиска у источников откройте «Разовый сбор». Поле над таблицей ищет только среди уже сохранённых документов."

    def _set_detail(self, text):
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def _select_document(self, event=None):
        selected = self.document_tree.selection()
        if not selected or selected[0] not in self.documents:
            return
        self.selected_document = snapshot = self.documents[selected[0]]
        self._set_detail(document_detail(snapshot))
        self.open_link.state(["!disabled"])
        self.versions_button.state(["!disabled"])

    def show_versions(self):
        if self.selected_document and self.ready and not self.closing:
            from app.ui.versions import VersionsWindow
            dialog = VersionsWindow(self, self.selected_document)
            self.child_windows.append(dialog)

    def _open_source(self, source):
        if not self.closing:
            url = self.source_urls[source]
            self.controller.call(("source-browser", source), "open_url", lambda result: None,
                                 lambda error: self._message(f"Не удалось открыть браузер. Адрес источника: {url}", True), url)
        return "break"

    def _open_link(self):
        if self.selected_document and not self.closing:
            self.controller.call("browser", "open_url", lambda result: None,
                                 lambda error: self._message("Не удалось открыть браузер. Ссылка есть в карточке.", True),
                                 self.selected_document.document.url)

    def start_collection(self):
        if not self.ready or self.closing or "start" in self.controller.pending:
            return
        sources = tuple(source for source, value in self.source_vars.items() if value.get())
        values, errors = collection_values(self.fields["topic"].get(), self.fields["start"].get(),
                                          self.fields["end"].get(), self.fields["limit"].get(),
                                           sources)
        for key, message in self.field_errors.items():
            message.set(errors.get(key, ""))
        if errors:
            first = next(iter(errors))
            (self.fields.get(first) or self.source_checks["crossref"]).focus_set()
            return
        self.start_button.state(["disabled"])
        self._message("Ставим сбор в очередь…")

        def started(ids):
            self.start_button.state(["!disabled"])
            self._message(f"Создано процессов: {len(ids)}. Результаты сохраняются автоматически.")
            self.clear_search()
            self._show_all()
            self._request_jobs()
            self.tabs.select(self.jobs_tab)

        def failed(error):
            self.start_button.state(["!disabled"])
            self._operation_error(error)

        self.controller.call("start", "start_collection", started, failed, values, sources)

    def _tab_changed(self, event=None):
        if self.ready and not self.closing:
            if self.tabs.select() in {str(self.pilot_tab), str(self.analysis_history_tab)}:
                self.status_label.pack_forget()
            if self.tabs.select() == str(self.analysis_history_tab):
                self.analysis_history.ensure_loaded()

    def _poll_jobs(self):
        if not self.closing:
            self._request_jobs()
            if self.ready and (self.tabs.select() == str(self.history_tab) or self.history_filter):
                self.history.poll()
            self.trends_panel.poll()
            self.job_timer = self.root.after(1000, self._poll_jobs)

    def _request_jobs(self):
        if self.ready and not self.closing:
            self.controller.call("jobs", "list_jobs", self._render_jobs, self._operation_error, limit=100)

    def _render_jobs(self, jobs):
        fingerprint = tuple((job.id, job.updated_at, job.state, job.scanned, job.stored, job.skipped) for job in jobs)
        if fingerprint == self.fingerprint:
            return
        forced = self.fingerprint is None
        self.fingerprint = fingerprint
        document_fingerprint = tuple(sorted((job.id, job.scanned, job.stored) for job in jobs))
        documents_changed = forced or document_fingerprint != self._document_jobs_fingerprint
        self._document_jobs_fingerprint = document_fingerprint
        selected = self.job_tree.selection()
        self.jobs = {job.id: job for job in jobs}
        existing = self.job_tree.get_children()
        existing_set = set(existing)
        row_cache = getattr(self, "_job_rows", None)
        if row_cache is None:
            row_cache = self._job_rows = {}
        removed = [item for item in existing if item not in self.jobs]
        if removed:
            self.job_tree.delete(*removed)
            for item in removed:
                row_cache.pop(item, None)
        current_order = [item for item in existing if item in self.jobs]
        for index, job in enumerate(jobs):
            state = "Отмена запрошена…" if job.id in self.cancel_requested and job.state in ACTIVE else job_state(job)
            values = (job.request.topic, SOURCES[job.request.source], state, job.scanned, job.stored, job.skipped)
            if job.id in existing_set:
                if row_cache.get(job.id) != values:
                    self.job_tree.item(job.id, values=values)
            else:
                self.job_tree.insert("", "end", iid=job.id, values=values)
                existing_set.add(job.id)
                current_order.append(job.id)
            row_cache[job.id] = values
            if current_order[index] != job.id:
                self.job_tree.move(job.id, "", index)
                current_order.remove(job.id)
                current_order.insert(index, job.id)
        if selected and selected[0] in self.jobs:
            self.job_tree.selection_set(selected[0])
        elif jobs:
            self.job_tree.selection_set(jobs[0].id)
        self._select_job()
        if documents_changed:
            self.refresh_documents()
        elif not self.documents and not self.query and not self.loading:
            self._set_detail(self._empty_documents_message())

    def _select_job(self, event=None):
        selected = self.job_tree.selection()
        job = self.jobs.get(selected[0]) if selected else None
        self.cancel_button.state(["!disabled"] if job and job.state in ACTIVE
                                 and job.id not in self.cancel_requested and not self.closing else ["disabled"])
        self.job_documents.state(["!disabled"] if job and not self.closing else ["disabled"])
        self.repeat_button.state(["!disabled"] if job and not self.closing else ["disabled"])
        if job:
            detail = f"{job_state(job)}. Найдено у источника: {job.total_available if job.total_available is not None else 'неизвестно'}."
            if empty_collection(job):
                detail += " " + empty_collection_hint(job)
            if job.error_message:
                detail += f" {job.error_message} [{job.error_code}]"
            if job.state == "succeeded" and not job.coverage_complete:
                detail += " Полнота выдачи не подтверждена."
            self.job_detail.set(detail)

    def repeat_job(self):
        selected = self.job_tree.selection()
        job = self.jobs.get(selected[0]) if selected else None
        if not job or self.closing:
            return
        request = job.request
        values = {"topic": request.topic, "start": str(request.from_date or ""),
                  "end": str(request.until_date), "limit": str(request.max_results)}
        for key, value in values.items():
            self.fields[key].delete(0, "end")
            self.fields[key].insert(0, value)
        for message in self.field_errors.values():
            message.set("")
        for source, value in self.source_vars.items():
            value.set(source == request.source and not self.source_checks[source].instate(["disabled"]))
        self.tabs.select(self.collection_tab)
        self.collection_tab.focus_when_visible(self.fields["topic"])
        self._message("Параметры процесса скопированы. Проверьте их и нажмите «Начать разовый сбор».")

    def cancel_job(self):
        selected = self.job_tree.selection()
        if not selected or self.closing:
            return
        job_id = selected[0]
        self.cancel_button.state(["disabled"])

        def cancelled(accepted):
            if accepted:
                self.cancel_requested.add(job_id)
            self._message("Отмена запрошена. Ожидаем завершения текущего сетевого чтения." if accepted else
                          "Процесс уже завершился; сохранённые документы доступны.")
            self.fingerprint = None
            self._request_jobs()

        def failed(error):
            self._operation_error(error)
            self._select_job()

        if not self.controller.call(("cancel", job_id), "cancel", cancelled, failed, job_id):
            self._select_job()
            self._message("Запрос отмены уже обрабатывается или библиотека переключается.")

    def show_documents(self, *, job_id=None, history_id=None, label="Все документы"):
        if not self.ready or self.closing:
            return
        self.job_filter, self.history_filter, self.offset = job_id, history_id, 0
        self.document_scope = label
        self.search.delete(0, "end")
        self.query = ""
        self.tabs.select(self.document_tab)
        self.refresh_documents()

    def _show_job(self):
        selected = self.job_tree.selection()
        if selected and not self.closing:
            self.show_documents(job_id=selected[0], label=f"Документы процесса {selected[0]}")

    def _show_all(self):
        self.show_documents()

    def close(self):
        if not self.closing:
            self.closing, self.ready = True, False
            token = getattr(self.pilot_panel, "start_cancel", None)
            if token is not None:
                token.set()
            self.root.after_cancel(self.job_timer)
            self.trends_panel.close()
            for dialog in tuple(self.child_windows):
                if isinstance(dialog, tk.Toplevel):
                    if dialog.winfo_exists():
                        dialog.destroy()
                else:
                    dialog.close()
            for tab in self.tabs.tabs():
                self.tabs.tab(tab, state="disabled")
            self.cancel_button.state(["disabled"])
            self.job_documents.state(["disabled"])
            self.retry.pack_forget()
        self._message("Закрываем приложение: ждём расчёта и завершаем фоновые процессы…")
        self.controller.close(self.root.destroy, self._close_error)

    def _close_error(self, error):
        self._message(error_message(error) + " Нажмите закрытие окна ещё раз, чтобы повторить остановку.", True)


def run_app():
    parser = argparse.ArgumentParser(description="Trendanalyser — сбор публикаций и анализ трендов")
    parser.add_argument("--data-dir", type=Path, help="Каталог базы, общий с backend CLI")
    parser.add_argument("--ui-scale", type=scale_value, default=None,
                        help="Масштаб интерфейса: 1, 1.5, 2 и т.д. (по умолчанию — системный)")
    args = parser.parse_args()
    enable_high_dpi()
    root = tk.Tk()
    application = Application(root, factory=lambda: create_backend(args.data_dir), ui_scale=args.ui_scale)
    from app.identity import default_data_dir
    application.controller.profile_anchor = args.data_dir if args.data_dir is not None else default_data_dir()
    root.mainloop()
