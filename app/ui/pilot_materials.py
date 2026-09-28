"""Supplementary materials and transparent offline robustness checks."""

from datetime import datetime
from threading import Event
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import TypedDict

from app.runtime.jobs import TaskFailure
from app.ui.viewport import ScrollViewport


class _LibraryState(TypedDict):
    pending: bool
    again: bool
    offset: int
    total: int


def _window(panel, title, size="900x650"):
    window = tk.Toplevel(panel.app.root)
    window.title(title)
    window.geometry(size)
    panel.app.child_windows.append(window)
    return window


def background(panel, key, method, finished, *args, on_error=None):
    def report(error):
        panel.error(error)
        if on_error is not None:
            on_error(error)
    token = panel.begin_loading()
    if token is None:
        report(TaskFailure("Дождитесь текущей операции с локальными материалами."))
        return False
    def complete(value):
        if panel.finish_loading(token):
            finished(value)
    def failed(error):
        if panel.finish_loading(token):
            report(error)
    submitted = panel.app.controller.call("pilot-" + key, "pilot_" + method, complete, failed, *args)
    if not submitted:
        panel.finish_loading(token)
        report(TaskFailure("Эта операция уже выполняется или приложение закрывается. Повторите после её завершения."))
    return submitted


def show_library(panel):
    window = _window(panel, "Дополнительные материалы")
    viewport = ScrollViewport(window, padding=16)
    viewport.pack(fill="both", expand=True)
    frame = viewport.content
    ttk.Label(frame, text="Отчёты и препринты", style="Section.TLabel").pack(anchor="w")
    ttk.Label(frame, text="Импортированные материалы доступны локально. Они помогают проверить кандидатов, "
              "но не добавляются к ряду публикаций OpenAlex и не повышают баллы автоматически. "
              "Текст PDF не отправляется внешнему AI.", wraplength=820).pack(anchor="w", pady=8)
    tree = ttk.Treeview(frame, columns=("title", "type", "count"), show="headings", height=10)
    for name, title, width in (("title", "Материал", 540), ("type", "Тип", 90), ("count", "Документов", 100)):
        tree.heading(name, text=title)
        tree.column(name, width=width, minwidth=60)
    tree.pack(fill="both", expand=True, pady=8)
    notice = tk.StringVar()
    ttk.Label(frame, textvariable=notice, wraplength=820).pack(anchor="w", pady=8)
    reading: _LibraryState = {"pending": False, "again": False, "offset": 0, "total": 0}
    def page_states():
        refresh_button.state(["disabled"] if reading["pending"] else ["!disabled"])
        previous.state(["disabled"] if reading["pending"] or reading["offset"] == 0 else ["!disabled"])
        following.state(["disabled"] if reading["pending"] or reading["offset"] + 50 >= reading["total"] else ["!disabled"])
    def settled():
        reading["pending"] = False
        if window.winfo_exists():
            page_states()
            if reading["again"]:
                reading["again"] = False
                refresh()
    def render(page):
        if not window.winfo_exists():
            return
        rows = page["items"]
        reading["offset"], reading["total"] = page["offset"], page["total"]
        tree.delete(*tree.get_children())
        for item in rows:
            tree.insert("", "end", iid=item["id"], values=(item["title"], item["kind"], item["documents"]))
        first = page["offset"] + 1 if rows else 0
        notice.set(f"Сохранено импортов: {page['total']}. Показано {first}–{page['offset'] + len(rows) if rows else 0}. "
                   "Для поиска совпадений откройте паспорт технологии.")
        settled()
    def failed(error):
        if not window.winfo_exists():
            return
        panel.error(error)
        notice.set(panel.message.get())
        settled()
    def load(offset):
        if not window.winfo_exists() or panel.app.closing:
            return
        if reading["pending"]:
            return
        reading["pending"] = True
        page_states()
        notice.set("Читаем дополнительные материалы…")
        if not panel.app.controller.call("pilot-supplemental-list-" + str(window), "pilot_supplemental_list", render, failed,
                                          offset, 50):
            failed(TaskFailure("Чтение библиотеки уже выполняется. Повторите после его завершения."))
    def refresh():
        if reading["pending"]:
            reading["again"] = True
        else:
            load(reading["offset"])
    def saved(value):
        panel.message.set(f"Импорт завершён: документов {value['documents']}. " + " ".join(value.get("limitations", [])))
        refresh()
    def arxiv():
        if panel.active:
            panel.message.set("Завершите текущий анализ перед импортом.")
            return
        path = filedialog.askopenfilename(parent=window, title="Импорт выгрузки arXiv Atom",
                                          filetypes=[("arXiv Atom", "*.xml *.atom")])
        if path and not panel.active:
            background(panel, "arxiv-import", "import_arxiv", saved, path)
        elif path:
            notice.set("Завершите текущий анализ перед импортом.")
    actions = ttk.Frame(frame)
    actions.pack(fill="x", pady=8)
    pages = ttk.Frame(frame)
    pages.pack(fill="x", pady=8)
    previous = ttk.Button(pages, text="Назад", command=lambda: load(max(0, reading["offset"] - 50)))
    previous.pack(side="left")
    following = ttk.Button(pages, text="Далее", command=lambda: load(reading["offset"] + 50))
    following.pack(side="left", padx=8)
    ttk.Button(actions, text="Добавить PDF-отчёт", command=lambda: import_report_dialog(panel, saved)).pack(side="left")
    ttk.Button(actions, text="Импорт arXiv Atom", command=arxiv).pack(side="left", padx=8)
    refresh_button = ttk.Button(actions, text="Обновить", command=refresh)
    refresh_button.pack(side="left")
    ttk.Label(frame, text="arXiv загружается из подготовленной Atom-выгрузки. Приложение не запускает "
              "независимый опрос arXiv на каждом компьютере: это позволяет соблюдать общий лимит источника.",
              wraplength=820, style="Muted.TLabel").pack(anchor="w", pady=8)
    refresh()


def import_report_dialog(panel, finished):
    if panel.active:
        panel.message.set("Завершите текущий анализ перед импортом.")
        return
    path = filedialog.askopenfilename(parent=panel.app.root, title="Выберите публичный аналитический отчёт",
                                      filetypes=[("PDF", "*.pdf")])
    if not path:
        return
    window = _window(panel, "Описание и права на отчёт", "800x570")
    viewport = ScrollViewport(window, padding=16)
    viewport.pack(fill="both", expand=True)
    frame = viewport.content
    entries = {}
    for key, title, value in (("title", "Название", ""), ("source_url", "Публичная ссылка на источник", ""),
                               ("publication_year", "Год публикации", str(datetime.now().year)),
                               ("license_note", "Основание использования: лицензия или условия источника", "")):
        ttk.Label(frame, text=title).pack(anchor="w", pady=(8, 2))
        entry = ttk.Entry(frame)
        entry.pack(fill="x")
        entry.insert(0, value)
        entries[key] = entry
    allowed = tk.BooleanVar(value=False)
    rights = ttk.Checkbutton(frame, text="Я проверил право на локальную обработку этого публичного отчёта", variable=allowed)
    rights.pack(anchor="w", pady=12)
    ttk.Label(frame, text="До 25 МБ и 300 страниц. Из сканов без текстового слоя текст не извлекается. "
              "Полный текст хранится локально и включается только в личную резервную копию, "
              "а не в пакет публичных библиографических результатов.", wraplength=720).pack(anchor="w", pady=8)
    message = tk.StringVar()
    ttk.Label(frame, textvariable=message, wraplength=720).pack(anchor="w", pady=8)
    pending = [False]
    def set_pending(value):
        pending[0] = value
        if window.winfo_exists():
            for widget in (*entries.values(), rights, submit_button):
                widget.state(["disabled"] if value else ["!disabled"])
    def failed(_error):
        set_pending(False)
        if window.winfo_exists():
            message.set(panel.message.get())
    def saved(value):
        if window.winfo_exists():
            window.destroy()
        finished(value)
    def submit():
        if pending[0]:
            return
        if panel.active or panel.app.closing:
            message.set("Завершите текущий анализ перед импортом.")
            return
        try:
            from app.pilot.reports import ReportMetadata
            values = {name: entry.get().strip() for name, entry in entries.items()}
            metadata = ReportMetadata.model_validate(values | {"publication_year": int(values["publication_year"]),
                "public_license_allowed": allowed.get(), "external_ai_allowed": False})
            if not allowed.get():
                raise ValueError("Rights required")
        except ValueError:
            message.set("Укажите название, действительную HTTP(S)-ссылку, год и основание использования; подтвердите право обработки.")
            return
        set_pending(True)
        message.set("Проверяем и импортируем PDF…")
        background(panel, "report-import", "import_report", saved, path, metadata.model_dump(mode="json"), on_error=failed)
    submit_button = ttk.Button(frame, text="Импортировать и проверить", command=submit, style="Primary.TButton")
    submit_button.pack(anchor="e", pady=8)


def show_matches(panel, run_id, candidate_id):
    def render(result):
        window = _window(panel, "Дополнительные доказательства")
        viewport = ScrollViewport(window, padding=16)
        viewport.pack(fill="both", expand=True)
        frame = viewport.content
        ttk.Label(frame, text="Совпадения в локальных отчётах и препринтах", style="Section.TLabel", wraplength=820).pack(anchor="w")
        if not result["items"]:
            ttk.Label(frame, text="Совпадений не найдено. Добавьте материалы в библиотеку или проверьте формулировку технологии.",
                      wraplength=820).pack(anchor="w", pady=12)
        for item in result["items"]:
            evidence = item["evidence"]
            pages = " · страницы " + ", ".join(map(str, item["pages"])) if item["pages"] else ""
            ttk.Label(frame, text=item["title"] + pages, style="Section.TLabel", wraplength=820).pack(anchor="w", pady=(16, 4))
            ttk.Label(frame, text=evidence["quote"], wraplength=820, justify="left").pack(anchor="w", pady=4)
            def open_source(url: str = evidence["source_url"]):
                panel.app.controller.call("pilot-material-link", "open_url", lambda _: None, panel.error, url)
            ttk.Button(frame, text="Открыть первоисточник", command=open_source).pack(anchor="w")
            ttk.Label(frame, text=item["notice"], wraplength=820, style="Muted.TLabel").pack(anchor="w", pady=4)
        if result["limited"]:
            ttk.Label(frame, text="Достигнут лимит просмотра; показана часть совпадений.", wraplength=820).pack(anchor="w", pady=12)
    background(panel, "material-matches", "supplemental_matches", render, run_id, candidate_id)


def sensitivity_dialog(panel, run_id, candidate_id):
    window = _window(panel, "Проверить устойчивость", "850x620")
    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=1)
    ttk.Label(frame, text="Как изменится вывод при исключении части доказательств?", style="Section.TLabel", wraplength=790).grid(row=0, column=0, sticky="w")
    options = {"Без крупнейшей проверенной исследовательской группы": {"exclude_largest_verified_group": True},
               "Без OpenAlex: проверка зависимости от основной базы": {"excluded_sources": ["openalex"]},
               "Без Crossref: проверка зависимости от второго источника": {"excluded_sources": ["crossref"]}}
    # The grid supplies the available width. A character-based minimum of 80
    # exceeds this window on X11 and at larger font scales.
    choice = ttk.Combobox(frame, values=tuple(options), state="readonly", width=1)
    choice.grid(row=1, column=0, sticky="ew", pady=12)
    choice.current(0)
    box = ScrolledText(frame, wrap="word", width=1, height=5, padx=10, pady=10)
    box.grid(row=2, column=0, sticky="nsew")
    pending = [False]
    def display(text):
        if not window.winfo_exists():
            return
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", text)
        box.configure(state="disabled")
    def settled():
        pending[0] = False
        if window.winfo_exists():
            calculate.state(["!disabled"])
    def render(report, scenario):
        settled()
        if not window.winfo_exists():
            return
        lines = ["Применённый сценарий: " + scenario,
                 "Исходный результат сохранён. Новый поиск и платные AI-запросы не выполнялись."]
        before = report["baseline"]["assessment"]
        if report["status"] == "evaluated" and report.get("after"):
            after = report["after"]["assessment"]
            lines += [f"Исследований за 3 года: {before['recent_studies']} → {after['recent_studies']}.",
                      f"Сглаженный рост: {before['smoothed_growth']:.2f} → {after['smoothed_growth']:.2f}.",
                      "Проверка роста: " + ("сохранилась" if before["growth_confirmed"] == after["growth_confirmed"] else "изменилась") + "."]
        else:
            lines.append("Сопоставимый пересчёт недоступен: отсутствует основная база или проверенная принадлежность работ группе. Это не нулевой рост.")
        lines += ["Исключено исследований: " + str(len(report["excluded_study_ids"])), *report["limitations"]]
        display("\n\n".join(lines))
    def failed(_error):
        settled()
        display(panel.message.get())
    def submit():
        if pending[0]:
            return
        scenario = choice.get()
        pending[0] = True
        calculate.state(["disabled"])
        display("Выполняется пересчёт: " + scenario)
        background(panel, "sensitivity", "sensitivity", lambda report: render(report, scenario),
                   run_id, candidate_id, options[scenario], on_error=failed)
    calculate = ttk.Button(frame, text="Пересчитать по сохранённым данным", command=submit)
    calculate.grid(row=3, column=0, sticky="e", pady=12)


def backup_dialog(panel):
    if panel.active or panel.loading:
        panel.message.set("Завершите текущую операцию перед резервным копированием.")
        return
    directory = filedialog.askdirectory(parent=panel.app.root, title="Папка для личной резервной копии")
    if directory:
        def saved(result):
            message = "Резервная копия проверена: " + result["path"] + ". Ключи доступа в неё не включены."
            if result.get("rotation_warning"):
                message += " Не удалось завершить очистку старых резервных копий. Проверьте папку и свободное место; новая копия сохранена."
            panel.message.set(message)
        background(panel, "backup", "backup", saved, directory)


def _reset_library_views(panel):
    """Discard identifiers and widgets tied to the previously opened library."""
    app = panel.app
    for window in tuple(app.child_windows):
        if isinstance(window, tk.Toplevel):
            if window.winfo_exists():
                window.destroy()
        else:
            window.close()
    app.child_windows.clear()
    panel.generation += 1
    panel.run_id = panel.payload = None
    panel.pending_cancel = False
    panel.cards.clear()
    panel.clear_result_tables()
    panel.scope.set("")
    panel.summary.set("Результаты восстановленной библиотеки доступны в истории анализов.")
    panel.progress.configure(value=0)
    for button in (panel.export_button, panel.passport_button, panel.documents_button):
        button.state(["disabled"])

    app.generation += 1
    app._displayed_document_view = None
    app.offset = app.total = 0
    app.query = ""
    app.loading = False
    app.history_filter = app.job_filter = app.selected_document = app.fingerprint = None
    app.document_scope = "Все документы"
    app.sort_by, app.descending = "default", False
    app.sort_choice.set("По умолчанию")
    app.search.delete(0, "end")
    app.documents.clear()
    app.jobs.clear()
    app.cancel_requested.clear()
    app.document_tree.delete(*app.document_tree.get_children())
    app.job_tree.delete(*app.job_tree.get_children())
    app.job_detail.set("Выберите процесс восстановленной библиотеки.")
    app._set_detail("Выберите документ восстановленной библиотеки.")
    for button in (app.open_link, app.versions_button, app.previous, app.next,
                   app.cancel_button, app.job_documents, app.repeat_button):
        button.state(["disabled"])

    history = app.history
    history.selected_id = history.progress = history.rows = history.form = None
    history.busy = False
    history.periods.clear()
    history.cancel_requested.clear()
    history.lookup.delete(0, "end")
    history.tree.delete(*history.tree.get_children())
    history.period_tree.delete(*history.period_tree.get_children())
    history.retry_incomplete.set(False)
    history.summary.set("Выберите сбор или создайте новый.")
    history.period_detail.set("Выберите период для подробностей.")
    history.bar.configure(value=0)
    history.update_actions()
    history.select_period()

    trends = app.trends_panel
    trends.cancel_event.set()
    trends.cancel_event = Event()
    trends._operation = None
    trends._cancel_pending = trends._collection_cancelled = False
    trends.snapshot_path = trends.history_id = trends.collecting_id = None
    trends._corpus_sources = ()
    trends._corpus_error = ""
    if trends.source.get() not in {"openalex", "crossref"}:
        trends.source.set("openalex")
    trends.histories.clear()
    trends.history_box.set("")
    trends.history_box.configure(values=())
    trends.clear_result()
    trends.progress.configure(value=0)
    trends.input_label.set("Выберите корпус или историю восстановленной библиотеки.")
    trends.notice.set("")
    trends.status.set("Результат — до 15 кандидатов. Новизна и стадия требуют проверки.")


def restore_dialog(panel):
    if panel.active or panel.loading:
        panel.message.set("Завершите текущую операцию перед восстановлением.")
        return
    package = filedialog.askopenfilename(parent=panel.app.root, title="Восстановить личную резервную копию main2",
                                         filetypes=[("Резервная копия", "*.zip *.trendbackup")])
    if not package:
        return
    directory = filedialog.askdirectory(parent=panel.app.root, title="Родительская папка: внутри будет создана новая библиотека")
    if not directory:
        return
    def restored(value):
        _reset_library_views(panel)
        panel.app._opened((value["path"], value["sources"]))
        panel._status(value["status"])
        model_note = ("Встроенная модель доступна в приложении." if value["status"].get("model_origin") == "bundled"
                      else "Локальная модель при необходимости загружается отдельно.")
        panel.message.set("Открыта восстановленная библиотека. Прежние данные сохранены. "
                          "Перед платными AI-запросами сверьте расходы в разделе «Расходы и сверка». " + model_note)
        if value.get("durability_warning"):
            panel.message.set(panel.message.get() + " " + value["durability_warning"])
    background(panel, "restore", "restore", restored, package, directory)
