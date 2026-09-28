"""First-run model and credential settings, with explicit persistence choice."""

from decimal import Decimal, InvalidOperation
from threading import Lock
import tkinter as tk
from tkinter import ttk

from app.pilot.settings import PilotSettings
from app.ui.viewport import ScrollViewport


def local_llm_size() -> str:
    """The download size taken from the pinned specification, never a stale number."""
    from app.pilot.local_llm import LocalModelError, load_spec

    try:
        total = sum(item["bytes"] for item in load_spec()["files"])
    except (LocalModelError, OSError):
        return ""
    return f" (~{total / 10 ** 9:.1f} ГБ)".replace(".", ",")


class DownloadProgress:
    """Bytes written so far, handed to the worker thread and read by Tk's timer.

    The installation runs on the backend lane, where no widget may be touched.
    Only these two numbers cross back, under a lock, and the interface draws
    them on its own clock.
    """

    def __init__(self):
        self._lock = Lock()
        self._received = self._total = 0

    def __call__(self, received: int, total: int) -> None:
        with self._lock:
            self._received, self._total = received, total

    def read(self) -> tuple[int, int]:
        with self._lock:
            return self._received, self._total

    def reset(self) -> None:
        self(0, 0)


class SettingsDialog:
    """Model and credential settings, either as a window or inside the settings page.

    Passing `parent` builds the same form into that frame instead of opening a
    window, so the settings page shows them in place. Every later update keeps
    working in both shapes: the form is never rebuilt and never destroyed, so an
    unsaved field survives a status refresh.
    """

    def __init__(self, panel, parent=None):
        self.panel = panel
        self.pending = False
        self.embedded = parent is not None
        if self.embedded:
            # The settings page is already a scrolling viewport; another one here
            # would nest two scrollbars for the same content.
            self.window = parent.winfo_toplevel()
            frame = parent
        else:
            self.window = window = tk.Toplevel(panel.app.root)
            panel.app.child_windows.append(window)
            window.title("Настройки анализа main2")
            window.geometry("800x720")
            viewport = ScrollViewport(window, padding=18)
            viewport.pack(fill="both", expand=True)
            frame = viewport.content
        frame.columnconfigure(1, weight=1)
        status = panel.settings_status
        self.bundled_model = status.get("model_origin") == "bundled"
        self.current = PilotSettings.model_validate(status["settings"])
        self.provider = tk.StringVar(value=self.current.provider)
        self.folder = tk.StringVar(value=self.current.yandex_folder)
        self.run_limit = tk.StringVar(value=str(Decimal(self.current.run_cost_micro) / 1000000))
        self.day_limit = tk.StringVar(value=str(Decimal(self.current.day_cost_micro) / 1000000))
        self.count = tk.StringVar(value=str(self.current.discovery_documents))
        self.history = tk.BooleanVar(value=self.current.history_enabled)
        self.patents = tk.BooleanVar(value=self.current.patents_enabled)
        self.ai_text = tk.BooleanVar(value=self.current.external_ai_allowed)
        self.wordstat_enabled = tk.BooleanVar(value=self.current.wordstat_api_enabled)
        self.wordstat_folder = tk.StringVar(value=self.current.wordstat_folder_id)
        self.wordstat_hourly_cap = tk.StringVar(value=str(self.current.wordstat_hourly_cap))
        self.wordstat_daily_cap = tk.StringVar(value=str(self.current.wordstat_daily_cap))
        self.persistent = tk.BooleanVar(value=False)
        self.note = tk.StringVar()
        fields = (("Поставщик AI", self.provider), ("Каталог Yandex (только для Yandex)", self.folder),
                  ("Лимит стоимости одного анализа", self.run_limit), ("Лимит стоимости за сутки UTC", self.day_limit),
                  ("Максимум публикаций для первичного поиска", self.count))
        for row, (title, variable) in enumerate(fields):
            ttk.Label(frame, text=title).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
            widget: ttk.Entry
            if row == 0:
                widget = ttk.Combobox(frame, textvariable=variable,
                                      values=("deepseek", "yandex", "local"), state="readonly")
                widget.bind("<<ComboboxSelected>>", self._provider_changed)
            else:
                widget = ttk.Entry(frame, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Label(frame, textvariable=self.note, wraplength=730, style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=8)
        self.keys = {}
        self.key_titles = {}
        self.key_labels = {}
        for row, (name, title) in enumerate((("deepseek_api_key", "Ключ DeepSeek"), ("yandex_api_key", "Ключ Yandex"),
                                            ("openalex_api_key", "Ключ OpenAlex (необязательно)"),
                                            ("epo_ops_key", "EPO: ключ"), ("epo_ops_secret", "EPO: секрет")), start=6):
            state = " · настроен" if status["keys"][name] else ""
            self.key_titles[name] = title
            self.key_labels[name] = ttk.Label(frame, text=title + state)
            self.key_labels[name].grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(frame, show="•")
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self.keys[name] = entry
        ttk.Label(frame, text="Пустое поле оставляет прежний ключ. Значения ключей не сохраняются в файле настроек.",
                  wraplength=730, style="Muted.TLabel").grid(row=11, column=0, columnspan=2, sticky="w", pady=5)
        persistence = ttk.Checkbutton(frame, text="Сохранить новые ключи в системном защищённом хранилище",
                                      variable=self.persistent)
        persistence.grid(row=12, column=0, columnspan=2, sticky="w")
        if not status["persistent_keys_available"]:
            persistence.state(["disabled"])
        ttk.Checkbutton(frame, text="Проверять отдельную историю публикаций для конкретных технологий", variable=self.history).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(frame, text="Использовать AI для названий и объяснений по текстам публичных публикаций", variable=self.ai_text).grid(
            row=14, column=0, columnspan=2, sticky="w", pady=5)
        model_ready = status["model_installed"]
        self.model_text = tk.StringVar(value=(
            "Встроенная модель проверена и готова" if model_ready and self.bundled_model else
            "Локальная модель установлена" if model_ready else
            (status.get("model_error") or "Встроенная модель недоступна. Переустановите приложение.") if self.bundled_model else
            "Локальная модель ещё не установлена"))
        ttk.Label(frame, textvariable=self.model_text).grid(row=15, column=0, sticky="w", pady=10)
        self.model_button = ttk.Button(frame, text=("Проверить встроенную модель" if self.bundled_model else
                                                    "Загрузить / проверить модель (~490 МБ)"), command=self.install)
        self.model_button.grid(row=15, column=1, sticky="e")
        self.local_llm_progress = DownloadProgress()
        self.local_llm_tick = None
        local = ttk.Frame(frame)
        local.grid(row=16, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        local.columnconfigure(0, weight=1)
        self.local_llm_text = tk.StringVar(value=self._local_llm_state(status))
        ttk.Label(local, textvariable=self.local_llm_text, wraplength=520).grid(row=0, column=0, sticky="w")
        self.local_llm_button = ttk.Button(local, text="Загрузить / проверить AI-модель" + local_llm_size(),
                                           command=self.install_local_llm)
        self.local_llm_button.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.local_llm_bar = ttk.Progressbar(local, maximum=100, mode="determinate")
        self.local_llm_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.local_llm_bar.grid_remove()
        self.message = tk.StringVar(value="Без сохранения в системе новые ключи действуют только до закрытия приложения.")
        ttk.Label(frame, textvariable=self.message, wraplength=730).grid(row=17, column=0, columnspan=2, sticky="w", pady=10)
        self.save_button = ttk.Button(frame, text="Сохранить настройки", style="Primary.TButton", command=self.save)
        self.save_button.grid(row=18, column=0, columnspan=2, sticky="e", pady=10)
        ttk.Checkbutton(frame, text="Проверять патентные семейства EPO (нужны ключ и секрет; увеличивает время анализа)",
                        variable=self.patents).grid(row=19, column=0, columnspan=2, sticky="w", pady=8)
        cleanup = ttk.LabelFrame(frame, text="Удаление ключа доступа", padding=10)
        cleanup.grid(row=20, column=0, columnspan=2, sticky="ew", pady=10)
        key_names = {"DeepSeek": "deepseek_api_key", "Yandex": "yandex_api_key", "OpenAlex": "openalex_api_key",
                     "EPO — ключ": "epo_ops_key", "EPO — секрет": "epo_ops_secret",
                     "Wordstat API": "wordstat_api_key"}
        selected_key = ttk.Combobox(cleanup, values=tuple(key_names), state="readonly")
        selected_key.pack(fill="x", pady=4)
        remove_system = tk.BooleanVar(value=False)
        ttk.Checkbutton(cleanup, text="Удалить также сохранённый ключ из системного хранилища", variable=remove_system).pack(anchor="w", pady=4)
        def delete_selected():
            if not self._can_submit():
                return
            if selected_key.get() not in key_names:
                self.message.set("Выберите ключ для удаления.")
                return
            name = key_names[selected_key.get()]
            requested_persistent = remove_system.get()
            submitted_text = self.keys[name].get()
            self._pending(True)
            self.message.set("Удаляем выбранный ключ…")
            def deleted(current):
                self.panel._status(current)
                self._pending(False)
                if self.window.winfo_exists():
                    if self.keys[name].get() == submitted_text:
                        self.keys[name].delete(0, "end")
                    self.message.set("Ключ удалён из выбранного места хранения." +
                                     (" В системном хранилище может оставаться отдельная сохранённая копия." if not requested_persistent else ""))
            if not self.panel.app.controller.call("pilot-delete-key", "pilot_delete_credential", deleted,
                                                  self._failed, name, requested_persistent):
                self._pending(False)
                self.message.set("Другая операция удаления ещё выполняется. Повторите после её завершения.")
        self.delete_button = ttk.Button(cleanup, text="Удалить выбранный ключ", command=delete_selected)
        self.delete_button.pack(anchor="e", pady=4)
        wordstat = ttk.LabelFrame(frame, text="Wordstat Cloud API — необязательно", padding=10)
        wordstat.grid(row=21, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Checkbutton(wordstat, text="Разрешить платные запросы после нажатия кнопки в окне сигналов",
                        variable=self.wordstat_enabled).pack(anchor="w")
        ttk.Label(wordstat, text="GetDynamics: ориентир 0,02 ₽ за вызов. Требуются роль search-api.webSearch.user "
                  "и ключ с областью yc.search-api.execute. CSV работает и без API.",
                  wraplength=700, style="Muted.TLabel").pack(anchor="w", pady=5)
        for title, variable in (("Каталог Yandex Cloud", self.wordstat_folder),
                                ("Максимум вызовов в час", self.wordstat_hourly_cap),
                                ("Максимум вызовов за сутки UTC", self.wordstat_daily_cap)):
            line = ttk.Frame(wordstat)
            line.pack(fill="x", pady=2)
            ttk.Label(line, text=title, width=32).pack(side="left")
            ttk.Entry(line, textvariable=variable).pack(side="left", fill="x", expand=True)
        self.key_titles["wordstat_api_key"] = "Отдельный ключ Wordstat API"
        self.key_labels["wordstat_api_key"] = ttk.Label(wordstat, text="Отдельный ключ Wordstat API" +
                  (" · настроен" if status["keys"].get("wordstat_api_key") else ""))
        self.key_labels["wordstat_api_key"].pack(anchor="w", pady=(6, 2))
        self.keys["wordstat_api_key"] = ttk.Entry(wordstat, show="•")
        self.keys["wordstat_api_key"].pack(fill="x")
        self._note()

    def refresh(self, status):
        """Reflect a newer status without touching what the user is typing."""
        if not self.window.winfo_exists():
            return
        for name, label in self.key_labels.items():
            if label.winfo_exists():
                label.configure(text=self.key_titles[name] +
                                (" · настроен" if status["keys"].get(name) else ""))
        if not self.local_llm_text.get().endswith("…"):
            self.local_llm_text.set(self._local_llm_state(status))
        if self.model_text.get().endswith("…"):
            return  # An installation is running and owns this line.
        ready = status["model_installed"]
        self.model_text.set("Встроенная модель проверена и готова" if ready and self.bundled_model else
                            "Локальная модель установлена" if ready else
                            (status.get("model_error") or "Встроенная модель недоступна. Переустановите приложение.")
                            if self.bundled_model else "Локальная модель ещё не установлена")

    def _can_submit(self):
        if self.pending:
            return False
        if self.panel.active or self.panel.loading or self.panel.app.closing:
            self.message.set("Дождитесь завершения анализа или текущей операции перед изменением настроек и ключей.")
            return False
        return True

    def _pending(self, pending):
        self.pending = pending
        if self.window.winfo_exists():
            for button in (self.save_button, self.delete_button):
                button.state(["disabled"] if pending else ["!disabled"])

    def _failed(self, error):
        self.panel.error(error)
        self._pending(False)
        if self.window.winfo_exists():
            self.message.set(self.panel.message.get())

    def _draft(self):
        return tuple(variable.get() for variable in (self.provider, self.folder, self.run_limit,
            self.day_limit, self.count, self.history, self.patents, self.ai_text, self.persistent,
            self.wordstat_enabled, self.wordstat_folder, self.wordstat_hourly_cap, self.wordstat_daily_cap)), {
                name: entry.get() for name, entry in self.keys.items()}

    def _note(self):
        if self.current.provider == "local":
            self.note.set("Локальная модель: ключ и интернет не нужны, запросы бесплатны. "
                          "Она медленнее облачной — минуты на анализ вместо секунд, — и требует "
                          "однократной загрузки весов кнопкой ниже" + local_llm_size() + ".")
            return
        self.note.set(f"Валюта лимитов: {self.current.currency}. Тариф: {self.current.pricing_version}. "
                      "Расходы резервируются до запроса; при неизвестном ответе резерв сохраняется. "
                      "Стоимость оценивается по сохранённому тарифу поставщика.")

    def _provider_changed(self, _=None):
        self.current = PilotSettings.for_provider(self.provider.get())
        self.run_limit.set(str(Decimal(self.current.run_cost_micro) / 1000000))
        self.day_limit.set(str(Decimal(self.current.day_cost_micro) / 1000000))
        self._note()

    def save(self):
        if not self._can_submit():
            return
        try:
            from app.ui.pilot_budget import money_micro

            run, day = money_micro(self.run_limit.get()), money_micro(self.day_limit.get())
            settings = PilotSettings.model_validate(self.current.model_dump() | {
                "provider": self.provider.get(), "yandex_folder": self.folder.get().strip(),
                "run_cost_micro": run, "day_cost_micro": day,
                "discovery_documents": int(self.count.get()), "history_enabled": self.history.get(),
                "patents_enabled": self.patents.get(),
                "external_ai_allowed": self.ai_text.get(),
                "wordstat_api_enabled": self.wordstat_enabled.get(),
                "wordstat_folder_id": self.wordstat_folder.get().strip(),
                "wordstat_hourly_cap": int(self.wordstat_hourly_cap.get()),
                "wordstat_daily_cap": int(self.wordstat_daily_cap.get()),
            })
            keys = {name: entry.get().strip() for name, entry in self.keys.items() if entry.get().strip()}
        except (ValueError, InvalidOperation, OverflowError):
            self.message.set("Проверьте лимиты, число публикаций (100–10000) и каталог Yandex. Дневной лимит должен быть не меньше разового.")
            return
        submitted_draft, submitted_keys = self._draft()
        self._pending(True)
        self.message.set("Сохраняем настройки…")
        def saved(status):
            self.panel._status(status)
            self._pending(False)
            if self.window.winfo_exists():
                changed = self._draft() != (submitted_draft, submitted_keys)
                for name, entry in self.keys.items():
                    if entry.get() == submitted_keys[name]:
                        entry.delete(0, "end")
                self.message.set("Отправленные настройки сохранены. В форме есть новые несохранённые изменения." if changed
                                 else "Настройки сохранены. Можно запустить анализ.")
        if not self.panel.app.controller.call("pilot-configure", "pilot_configure", saved, self._failed,
                                              settings.model_dump(mode="json"), keys, persistent=self.persistent.get()):
            self._pending(False)
            self.message.set("Другая операция сохранения ещё выполняется. Повторите после её завершения.")

    def _local_llm_state(self, status):
        if status.get("local_llm_installed"):
            return "Локальная AI-модель установлена: поставщик «local» работает без ключа и без сети."
        return ("Локальная AI-модель не установлена. Без неё поставщик «local» не называет технологии, "
                "и ТОП остаётся пустым.")

    def _local_llm_poll(self):
        """Draw what the worker thread has written since the last tick."""
        self.local_llm_tick = None
        if not self.window.winfo_exists():
            return
        received, total = self.local_llm_progress.read()
        if total:
            self.local_llm_bar.configure(value=received * 100 / total)
            # The last bytes are followed by hashing the whole download, which
            # takes seconds of its own: saying so beats a bar stuck at the end.
            self.local_llm_text.set("Проверяем контрольные суммы локальной AI-модели…" if received >= total else
                                    f"Загрузка локальной AI-модели: {received // 10 ** 6} МБ "
                                    f"из {total // 10 ** 6} МБ…")
        self.local_llm_tick = self.window.after(300, self._local_llm_poll)

    def _local_llm_done(self):
        tick, self.local_llm_tick = self.local_llm_tick, None
        if not self.window.winfo_exists():
            return
        if tick is not None:
            self.window.after_cancel(tick)
        self.local_llm_bar.grid_remove()
        self.local_llm_bar.configure(value=0)
        self.local_llm_button.state(["!disabled"])

    def install_local_llm(self):
        """Download the offline weights from here, with the bytes visible as they arrive."""
        if self.pending or self.panel.loading or self.panel.active:
            self.message.set("Дождитесь текущей операции.")
            return
        token = self.panel.begin_loading()
        if token is None:
            return
        self.local_llm_button.state(["disabled"])
        self.local_llm_progress.reset()
        self.local_llm_bar.configure(value=0)
        self.local_llm_bar.grid()
        self.local_llm_text.set("Загрузка локальной AI-модели: подготовка…")

        def finished(status):
            # The timer is stopped first: a superseded token must not leave this
            # form drawing a download that has already ended.
            self._local_llm_done()
            if not self.panel.finish_loading(token):
                return
            self.panel._status(status)
            if self.window.winfo_exists():
                self.local_llm_text.set(self._local_llm_state(status))

        def failed(error):
            self._local_llm_done()
            if not self.panel.finish_loading(token):
                return
            self.panel.error(error)
            if self.window.winfo_exists():
                self.local_llm_text.set(self.panel.message.get())

        if not self.panel.app.controller.call("pilot-local-llm", "pilot_install_local_llm", finished, failed,
                                              self.local_llm_progress):
            self.panel.finish_loading(token)
            self._local_llm_done()
            self.local_llm_text.set("Загрузка локальной AI-модели уже выполняется.")
            return
        self._local_llm_poll()

    def install(self):
        if self.pending or self.panel.loading or self.panel.active:
            self.message.set("Дождитесь текущей операции.")
            return
        token = self.panel.begin_loading()
        if token is None:
            return
        self.model_button.state(["disabled"])
        self.model_text.set("Проверяем встроенную модель…" if self.bundled_model else
                            "Загрузка модели и проверка контрольных сумм…")
        def finished(status):
            if not self.panel.finish_loading(token):
                return
            self.panel._status(status)
            if self.window.winfo_exists():
                self.model_text.set("Модель проверена и готова")
                self.model_button.state(["!disabled"])
        def failed(error):
            if not self.panel.finish_loading(token):
                return
            self.panel.error(error)
            if self.window.winfo_exists():
                self.model_text.set(self.panel.message.get())
                self.model_button.state(["!disabled"])
        if not self.panel.app.controller.call("pilot-model", "pilot_install_model", finished, failed):
            self.panel.finish_loading(token)
            self.model_button.state(["!disabled"])
            self.model_text.set("Проверка модели уже выполняется." if self.bundled_model else
                                "Операция загрузки уже выполняется.")
