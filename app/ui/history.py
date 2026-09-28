"""Historical collection UI over the existing asynchronous backend API."""

import tkinter as tk
from typing import TYPE_CHECKING
from tkinter import ttk
from app.ui.display import size_window, PixelWheel
from app.ui.layout import ActionRow
from app.ui.navigation import SectionNotebook
from app.ui.viewport import ScrollViewport
from tkinter.scrolledtext import ScrolledText

from app.ui.presentation import ACTIVE, SOURCES, error_message, history_values

if TYPE_CHECKING:
    from app.backend.history_progress import HistoryProgress, PeriodProgress

HISTORY_STATES = {'queued': 'В очереди', 'running': 'Загрузка', 'pending': 'Не начат',
                  'complete': 'Полная выдача', 'succeeded': 'Завершено', 'partial': 'Неполная выдача',
                  'failed': 'Ошибка', 'cancelled': 'Отменён', 'interrupted': 'Прерван', 'split': 'Разделён'}


class HistoryPanel:
    def __init__(self, app, parent):
        self.app = app
        self.selected_id = None
        self.progress: HistoryProgress | None = None
        self.rows = None
        self._visible_rows = {}
        self.periods: dict[str, PeriodProgress] = {}
        self.busy = False
        self.cancel_requested = set()
        self.form = None
        self.tabs = SectionNotebook(parent)
        self.tabs.pack(fill='both', expand=True)
        self.list_tab = ttk.Frame(self.tabs, padding=8)
        self.progress_tab = ttk.Frame(self.tabs, padding=4)
        # A Notebook pane receives its actual size from the Notebook. Letting
        # its scrollable child set the pane's requested height can instead make
        # the whole Notebook extend beyond the available window on X11.
        self.progress_tab.pack_propagate(False)
        self.tabs.add(self.list_tab, text='Список сборов')
        self.tabs.add(self.progress_tab, text='Периоды и прогресс')
        self.message = tk.StringVar(value='Исторические сборы загружаются из локальной базы.')
        parent = self.list_tab
        explanation = ttk.LabelFrame(parent, text='Сбор по периодам — подготовить историю темы', padding=10)
        explanation.pack(fill='x', pady=(0, 8))
        bar = ttk.Frame(explanation)
        bar.pack(fill='x')
        self.create_button = ttk.Button(bar, text='Создать сбор по периодам', style='Primary.TButton', command=self.new, state='disabled')
        self.create_button.pack(side='left')
        self.refresh_button = ttk.Button(bar, text='Обновить', command=self.poll, state='disabled')
        self.refresh_button.pack(side='left', padx=6)
        ttk.Label(explanation, wraplength=700, justify='left', text=
                  'Используйте, когда нужны данные о теме за несколько месяцев или лет для дальнейшего изучения её развития.\n'
                  'Диапазон делится на месяцы или годы. Лимит действует отдельно для каждого периода и источника: '
                  'например, до 200 записей за каждый месяц, а не за все годы сразу.\n'
                  'Вы получаете сохранённый план, полноту по периодам и возможность продолжить после остановки. '
                  'Такой сбор обычно требует больше времени и запросов. Сам анализ трендов он не выполняет.').pack(anchor='w', pady=(8, 0))
        ttk.Label(parent, text='Последние 100 сборов. Более старый можно открыть по ID.').pack(anchor='w')
        ttk.Label(parent, textvariable=self.message, wraplength=800).pack(anchor='w', pady=6)
        box, self.tree = app._tree(parent, [('title', 'Тема', 270), ('dates', 'Период', 190),
                                            ('source', 'Источники', 160), ('state', 'Состояние', 140)], 4)
        box.pack(fill='both', expand=True, pady=8)
        self.tree.bind('<<TreeviewSelect>>', self.select)
        self.tree.bind('<Double-1>', lambda event: self.view_selected())
        lookup = ttk.Frame(parent)
        lookup.pack(side='bottom', fill='x', before=box)
        self.lookup_row = ttk.Frame(lookup)
        self.lookup_row.pack(fill='x')
        ttk.Label(self.lookup_row, text='ID сбора:').pack(side='left')
        self.lookup = ttk.Entry(self.lookup_row, width=40)
        self.lookup.pack(side='left', padx=6)
        self.lookup.bind('<Return>', lambda event: self.open_id())
        ttk.Button(self.lookup_row, text='Открыть', command=self.open_id).pack(side='left')
        # The long action wraps onto its own line instead of being clipped when
        # the dialog is narrow or the display is scaled up.
        self.view_button = ttk.Button(lookup, text='Показать периоды выбранного сбора',
                                      command=self.view_selected)
        self._lookup_wrapped = None
        lookup.bind('<Configure>', self._reflow_lookup)
        self._reflow_lookup()
        # Text and font metrics vary between Tk platforms. Keep the period
        # actions outside the scrollable content so verbose summaries cannot
        # consume their space. The table retains its requested height and is
        # reachable by scrolling even on a small display at 200% scaling.
        self.progress_overview = ScrollViewport(self.progress_tab, padding=0)
        parent = self.progress_overview.content
        ttk.Label(parent, textvariable=self.message, wraplength=800).pack(anchor='w', pady=6)
        actions = ActionRow(parent)
        actions.pack(fill='x')
        self.cancel_button = actions.add(
            ttk.Button(actions, text='Отменить сбор', command=self.cancel, state='disabled'))
        self.resume_button = actions.add(
            ttk.Button(actions, text='Продолжить', command=self.resume, state='disabled'))
        self.retry_incomplete = tk.BooleanVar(value=False)
        self.retry_check = ttk.Checkbutton(parent, text='Повторить также неполные периоды', variable=self.retry_incomplete)
        self.retry_check.pack(anchor='w', pady=(4, 0))
        self.documents_button = actions.add(
            ttk.Button(actions, text='Документы сбора', command=self.documents, state='disabled'), 'right')
        self.parameters_button = actions.add(
            ttk.Button(actions, text='Параметры сбора', command=self.parameters, state='disabled'), 'right')
        self.summary = tk.StringVar(value='Выберите сбор или создайте новый.')
        self.readonly_text(parent, self.summary, 3).pack(fill='x', pady=6)
        self.bar = ttk.Progressbar(parent, maximum=100)
        self.bar.pack(fill='x')
        ttk.Label(parent, text='Процент — выполнение периодов, а не полнота данных. При дроблении он может уменьшаться.',
                  wraplength=800, style='Muted.TLabel').pack(anchor='w', pady=4)
        self.period_detail = tk.StringVar(value='Выберите период для подробностей.')
        ttk.Label(parent, text='Выбранный период · документы из последней попытки',
                  style='Muted.TLabel').pack(anchor='w', pady=(6, 0))
        self.readonly_text(parent, self.period_detail, 3).pack(fill='x', pady=6)
        actions = ActionRow(self.progress_tab)
        # Pack this control row first from the bottom. At large Linux font
        # scales it can wrap to two lines after the first layout pass; the
        # viewport must yield space to it instead of extending below the tab.
        actions.pack(side='bottom', fill='x', pady=(2, 0))
        self.period_documents = actions.add(
            ttk.Button(actions, text='Документы попытки', command=self.open_period, state='disabled'))
        self.attempts_button = actions.add(
            ttk.Button(actions, text='Все попытки', command=self.attempts, state='disabled'))
        box, self.period_tree = app._tree(parent, [('dates', 'Период / вложенность', 240), ('source', 'Источник', 100),
                                                   ('state', 'Состояние', 125), ('scanned', 'Просмотрено', 90),
                                                   ('stored', 'Сохранено', 85), ('skipped', 'Пропущено', 85)], 5)
        box.pack(fill='both', expand=True)
        self.period_tree.bind('<<TreeviewSelect>>', self.select_period)
        self.progress_overview.pack(side='top', fill='both', expand=True, pady=(0, 2))

    @staticmethod
    def readonly_text(parent, variable, height):
        text = ScrolledText(parent, height=height, width=1, wrap='word', font='TkDefaultFont',
                            padx=8, pady=4, spacing1=0, spacing3=0)
        def update(*args):
            text.configure(state='normal')
            text.delete('1.0', 'end')
            text.insert('1.0', variable.get())
            text.configure(state='disabled')
        variable.trace_add('write', update)
        update()
        return text

    def enabled(self):
        return self.app.ready and not self.app.closing

    def opened(self):
        self.create_button.state(['!disabled'])
        self.refresh_button.state(['!disabled'])
        self.poll()

    def poll(self):
        if self.enabled():
            self.app.controller.call('history-list', 'list_history', self.render_list, self.error, limit=100)
            self.load_progress()

    def error(self, error):
        if self.enabled():
            self.message.set(error_message(error) + ' Нажмите «Обновить», чтобы повторить.')

    def render_list(self, rows):
        if not self.enabled() or rows == self.rows:
            return
        self.rows = rows
        visible = getattr(self, '_visible_rows', None)
        if visible is None:
            visible = self._visible_rows = {}
        existing = self.tree.get_children()
        wanted = {row['id'] for row in rows}
        removed = [item for item in existing if item not in wanted]
        if removed:
            self.tree.delete(*removed)
            for item in removed:
                visible.pop(item, None)
        order = [item for item in existing if item in wanted]
        present = set(order)
        for index, row in enumerate(rows):
            request = row['request']
            values = (request['topic'], f"{request['from_date']} — {request['until_date']}",
                      ', '.join(SOURCES.get(s, s) for s in request['sources']),
                      HISTORY_STATES.get(row['state'], row['state']))
            run_id = row['id']
            if run_id in present:
                if visible.get(run_id) != values:
                    self.tree.item(run_id, values=values)
            else:
                self.tree.insert('', 'end', iid=run_id, values=values)
                order.append(run_id)
                present.add(run_id)
            visible[run_id] = values
            if order[index] != run_id:
                self.tree.move(run_id, '', index)
                order.remove(run_id)
                order.insert(index, run_id)
        if self.selected_id and self.selected_id in wanted:
            if self.tree.selection() != (self.selected_id,):
                self.tree.selection_set(self.selected_id)
        elif not self.selected_id and rows:
            self.tree.selection_set(rows[0]['id'])
            self.choose(rows[0]['id'], show=False)
        if not rows and not self.selected_id:
            self.message.set('Пока нет исторических сборов. Нажмите «Создать сбор по периодам».')

    def select(self, event=None):
        selected = self.tree.selection()
        if selected:
            self.choose(selected[0], show=False)

    def view_selected(self):
        if self.enabled() and self.selected_id:
            self.tabs.select(self.progress_tab)
            self.load_progress()

    def open_id(self):
        value = self.lookup.get().strip()
        if len(value) > 128:
            self.message.set('Проверьте ID: не больше 128 символов.')
            return
        if self.enabled() and value:
            self.choose(value)
            self.tabs.select(self.progress_tab)
            self.load_progress()

    def choose(self, run_id, show=True):
        if not self.enabled() or run_id == self.selected_id:
            return
        self.selected_id = run_id
        if show:
            self.tabs.select(self.progress_tab)
        self.progress = None
        self.periods = {}
        self.period_tree.delete(*self.period_tree.get_children())
        self.retry_incomplete.set(False)
        self.summary.set('Читаем состояние выбранного сбора…')
        self.message.set(f'Сбор: {run_id}')
        self.lookup.delete(0, 'end')
        self.lookup.insert(0, run_id)
        self.bar['value'] = 0
        self.update_actions()
        self.select_period()
        self.load_progress()

    def load_progress(self):
        if not self.enabled() or not self.selected_id:
            return
        run_id = self.selected_id

        def loaded(progress):
            if not self.enabled():
                return
            if run_id != self.selected_id:
                self.load_progress()
                return
            self.render_progress(progress)

        def failed(error):
            if not self.enabled():
                return
            if run_id != self.selected_id:
                self.load_progress()
                return
            self.error(error)
            self.summary.set('Не удалось обновить состояние. Показаны последние полученные данные.')
            self.progress = None
            self.update_actions()

        self.app.controller.call('history-progress', 'get_history_progress', loaded, failed, run_id)

    def render_progress(self, progress):
        from app.backend.history_progress import progress_summary
        changed = progress != self.progress
        self.progress = progress
        self.update_actions()
        if not changed:
            return
        topic = progress.topic[:120] + ('…' if len(progress.topic) > 120 else '')
        self.message.set(f'Сбор: {progress.id} — {topic}')
        summary = progress_summary(progress)
        if progress.partial_calendar_years:
            summary += ' Неполные календарные годы: ' + ', '.join(map(str, progress.partial_calendar_years)) + '.'
        if progress.error_code:
            summary += f' Код ошибки: {progress.error_code}.'
        self.summary.set(summary)
        self.bar['value'] = progress.execution_percent
        selected = self.period_tree.selection()
        yview = self.period_tree.yview()[0]
        previous_periods = self.periods
        self.periods = {p.id: p for p in progress.periods}
        rebuild = previous_periods.keys() != self.periods.keys()
        if rebuild:
            self.period_tree.delete(*self.period_tree.get_children())
        children: dict[str | None, list[PeriodProgress]] = {}
        for period in progress.periods:
            children.setdefault(period.parent_id, []).append(period)

        def insert(parent=None, depth=0):
            for period in children.get(parent, ()):
                number = lambda n: n if n is not None else '?'
                dates = '  ' * depth + ('↳ ' if depth else '') + f'{period.from_date} — {period.until_date}'
                if rebuild or previous_periods.get(period.id) != period:
                    values = (dates, SOURCES.get(period.source, period.source),
                              HISTORY_STATES.get(period.state, period.state), number(period.scanned),
                              number(period.stored), number(period.skipped))
                    if rebuild:
                        self.period_tree.insert('', 'end', iid=period.id, values=values)
                    else:
                        self.period_tree.item(period.id, values=values)
                insert(period.id, depth + 1)
        insert()
        if selected and selected[0] in self.periods:
            self.period_tree.selection_set(selected[0])
        self.period_tree.yview_moveto(yview)
        self.select_period()
        if self.app.history_filter == progress.id:
            self.app.refresh_documents()

    def _reflow_lookup(self, event=None):
        container = self.lookup_row.master
        width = event.width if event is not None else container.winfo_width()
        # Once the action is inside lookup_row its width is already included.
        # Counting it twice makes each Configure alternate between layouts.
        needed = self.lookup_row.winfo_reqwidth()
        if self._lookup_wrapped is not False:
            needed += self.view_button.winfo_reqwidth()
        wrapped = width <= 1 or needed > width
        if wrapped == self._lookup_wrapped:
            return
        self._lookup_wrapped = wrapped
        self.view_button.pack_forget()
        if wrapped:
            self.view_button.pack(anchor='w', pady=(6, 0))
        else:
            self.view_button.pack(in_=self.lookup_row, side='right')

    def update_actions(self):
        progress = self.progress
        available = self.enabled() and progress is not None
        active = progress is not None and progress.state in ACTIVE
        idle = self.enabled() and not self.busy
        cancellable = idle and progress is not None and active and progress.id not in self.cancel_requested
        resumable = idle and progress is not None and not active and not progress.coverage_complete
        self.cancel_button.state(['!disabled'] if cancellable else ['disabled'])
        self.resume_button.state(['!disabled'] if resumable else ['disabled'])
        self.retry_check.state(['!disabled'] if resumable else ['disabled'])
        self.documents_button.state(['!disabled'] if available else ['disabled'])
        self.parameters_button.state(['!disabled'] if available else ['disabled'])

    def selected_period(self):
        selected = self.period_tree.selection()
        return self.periods.get(selected[0]) if selected else None

    def select_period(self, event=None):
        from app.backend.history_progress import period_line
        period = self.selected_period()
        self.period_detail.set(period_line(period) if period else 'Выберите период для подробностей.')
        self.period_documents.state(['!disabled'] if period and period.job_id and self.enabled() else ['disabled'])
        self.attempts_button.state(['!disabled'] if period and period.attempt_count and self.enabled() else ['disabled'])

    def documents(self):
        if self.enabled() and self.progress:
            self.app.show_documents(history_id=self.selected_id, label=f'Исторический сбор «{self.progress.topic}»')

    def open_period(self):
        period = self.selected_period()
        if self.enabled() and period and period.job_id:
            self.app.show_documents(job_id=period.job_id, label=f'Попытка {period.job_id} • {period.from_date} — {period.until_date}')

    def parameters(self):
        if not self.enabled() or not self.selected_id:
            return
        run_id = self.selected_id

        def loaded(report):
            if not self.enabled() or run_id != self.selected_id:
                return
            request = report.request
            text = (f'Тема: {request.topic}\n\n'
                    f'Даты: {request.from_date} — {request.until_date}\n'
                    f'Источники: {", ".join(SOURCES[s] for s in request.sources)}\n'
                    f'Начальная разбивка: {"месяцы" if request.period == "month" else "годы"}\n'
                    f'Лимит записей на период и источник: {request.max_results_per_period}\n'
                    f'Автоматическое дробление: {"включено" if request.auto_split else "выключено"}\n'
                    f'Максимум узлов плана: {request.max_periods}\n\n'
                    f'Создан: {report.created_at.strftime("%d.%m.%Y %H:%M UTC")}\n'
                    f'ID: {report.id}\n\n'
                    'При продолжении параметры и конечная дата сохраняются. Для других параметров создайте новый сбор.')
            if report.error_message:
                text += f'\n\nОшибка: {report.error_message} [{report.error_code}]'
            dialog = ParametersWindow(self.app, text)
            self.app.child_windows.append(dialog)
        self.app.controller.call('history-parameters', 'get_history', loaded, self.error, run_id)

    def attempts(self):
        period = self.selected_period()
        if not self.enabled() or not period:
            return
        run_id, period_id = self.selected_id, period.id

        def loaded(report):
            if not self.enabled() or run_id != self.selected_id:
                return
            selected = self.selected_period()
            if not selected or selected.id != period_id:
                return
            item = next(p for p in report.periods if p.id == period_id)
            dialog = AttemptsWindow(self.app, item)
            self.app.child_windows.append(dialog)
        self.app.controller.call('history-attempts', 'get_history', loaded, self.error, run_id)

    def cancel(self):
        if self.progress and self.progress.state in ACTIVE and not self.busy and self.enabled():
            self.action('cancel_history')

    def resume(self):
        if self.progress and self.progress.state not in ACTIVE and not self.progress.coverage_complete and not self.busy and self.enabled():
            self.action('resume_history', retry_incomplete=self.retry_incomplete.get())

    def action(self, method, **kwargs):
        run_id = self.selected_id
        self.busy = True
        self.update_actions()

        def done(result):
            if not self.enabled():
                return
            self.busy = False
            if method == 'cancel_history' and result:
                self.cancel_requested.add(run_id)
            if method == 'resume_history':
                self.cancel_requested.discard(run_id)
            if self.selected_id == run_id:
                self.message.set('Отмена запрошена; ждём завершения текущего чтения.' if method == 'cancel_history' and result
                                 else 'Сбор уже остановлен.' if method == 'cancel_history'
                                 else 'Продолжение поставлено в очередь. Завершённые периоды сохраняются.')
            self.update_actions()
            self.poll()

        def failed(error):
            if self.enabled():
                self.busy = False
                self.update_actions()
                self.error(error)
        self.app.controller.call('history-action', method, done, failed, run_id, **kwargs)

    def new(self):
        if self.enabled():
            if 'history-start' in self.app.controller.pending:
                self.message.set('Создаём предыдущий план; дождитесь завершения операции.')
                return
            if self.form and not self.form.closed:
                self.form.window.lift()
                return
            self.form = HistoryForm(self)
            self.app.child_windows.append(self.form)


class HistoryForm:
    def __init__(self, panel):
        self.panel, self.app = panel, panel.app
        self.closed = self.busy = False
        self.window = tk.Toplevel(self.app.root)
        self.window.title('Сбор по периодам — новый план')
        size_window(self.window, 780, 650, (650, 420))
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill='both', expand=True)
        footer = ttk.Frame(outer)
        footer.pack(side='bottom', fill='x', pady=(8, 0))
        self.start = ttk.Button(footer, text='Запустить сбор по периодам', command=self.submit)
        self.start.pack(anchor='w')
        self.error_text = tk.StringVar()
        ttk.Label(footer, textvariable=self.error_text, wraplength=600, style='Error.TLabel').pack(fill='x', pady=(4, 0))
        ttk.Label(outer, text='Исторический сбор: отдельная выборка за каждый месяц или год.\n'
                  'Лимит применяется к каждому периоду и источнику. План можно продолжить после остановки.\n'
                  'Для одной выборки за весь диапазон используйте вкладку «Разовый сбор».',
                  wraplength=590, justify='left').pack(anchor='w', pady=(0, 10))
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient='vertical', command=self.canvas.yview)
        scroll.pack(side='right', fill='y')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.body = ttk.Frame(self.canvas)
        item = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfigure(item, width=event.width))
        self.wheel = PixelWheel(self.canvas)
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.window.bind(sequence, self.scroll)
        self.window.bind('<FocusIn>', self.reveal)
        self.fields = {}
        for i, (key, label, default) in enumerate([
            ('topic', 'Технологическое направление', ''), ('start', 'Начало: ГГГГ-ММ-ДД (обязательно)', ''),
            ('end', 'Окончание: ГГГГ-ММ-ДД (пусто — сегодня UTC)', ''),
            ('limit', 'Лимит записей на период и источник', '1000'), ('budget', 'Максимум узлов плана, включая дробление', '1200')]):
            ttk.Label(self.body, text=label, wraplength=320).grid(row=i, column=0, sticky='w', pady=8)
            field = ttk.Entry(self.body, width=28)
            field.insert(0, default)
            field.grid(row=i, column=1, sticky='ew', padx=12, pady=8)
            self.fields[key] = field
        self.body.columnconfigure(1, weight=1)
        ttk.Label(self.body, text='Разбивка периода').grid(row=5, column=0, sticky='w', pady=8)
        self.period = tk.StringVar(value='По месяцам')
        self.period_field = ttk.Combobox(self.body, textvariable=self.period, values=['По месяцам', 'По годам'], state='readonly')
        self.period_field.grid(row=5, column=1, sticky='ew', padx=12)
        ttk.Label(self.body, text='Источники').grid(row=6, column=0, sticky='nw', pady=8)
        source_box = ttk.Frame(self.body)
        source_box.grid(row=6, column=1, sticky='w', padx=12)
        self.sources = {}
        for name, title in SOURCES.items():
            enabled = not self.app.source_checks[name].instate(['disabled'])
            value = tk.BooleanVar(value=enabled and name in ('crossref', 'openalex'))
            self.sources[name] = value
            check = ttk.Checkbutton(source_box, text=title, variable=value)
            check.pack(anchor='w')
            if not enabled:
                check.state(['disabled'])
        self.auto_split = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.body, text='Автоматически дробить переполненные периоды', variable=self.auto_split).grid(
            row=7, column=0, columnspan=2, sticky='w', pady=12)
        ttk.Label(self.body, wraplength=640, text='Годы делятся на месяцы, месяцы — на дни. Для EPO действует предел '
                  '2000 записей за запрос; доступ требует ключей. Лимит включает пропущенные записи.\n\n'
                  'Сохранение происходит автоматически. После остановки продолжайте существующий сбор: '
                  'создание нового запускает отдельный план. Конечная дата фиксируется при создании.\n\n'
                  'Обычные и исторические сборы используют общую очередь. Полные периоды при продолжении '
                  'не скачиваются заново; незавершённые начинаются с начала периода.').grid(
            row=8, column=0, columnspan=2, sticky='w', pady=8)

    def scroll(self, event):
        if not self.closed and self.body.winfo_height() > self.canvas.winfo_height():
            self.wheel.scroll(event)

    def reveal(self, event):
        if self.closed or not str(event.widget).startswith(str(self.body) + '.'):
            return
        self.window.update_idletasks()
        y = event.widget.winfo_rooty() - self.body.winfo_rooty()
        top, visible = self.canvas.canvasy(0), self.canvas.winfo_height()
        height = max(1, self.body.winfo_height())
        if y < top:
            self.canvas.yview_moveto(max(0, y - 8) / height)
        elif y + event.widget.winfo_height() > top + visible:
            self.canvas.yview_moveto((y + event.widget.winfo_height() - visible + 8) / height)

    def submit(self):
        if self.closed or self.busy or not self.panel.enabled():
            return
        topic, start, end, limit = (self.fields[key].get() for key in ('topic', 'start', 'end', 'limit'))
        values, errors = history_values(topic, start, end, limit,
                                         tuple(s for s, value in self.sources.items() if value.get()),
                                         {'По месяцам': 'month', 'По годам': 'year'}.get(self.period.get()),
                                         self.auto_split.get(), self.fields['budget'].get())
        if errors:
            self.error_text.set('\n'.join(errors.values()))
            field = self.fields.get(next(iter(errors)), self.period_field)
            field.focus_set()
            return
        self.busy = True
        self.start.state(['disabled'])
        self.error_text.set('Создаём план…')

        def done(run_id):
            self.busy = False
            if not self.panel.enabled():
                return
            if self.closed:
                # The accepted collection continues, but a dismissed form no
                # longer owns navigation or the user's current selection.
                self.panel.poll()
                return
            self.close()
            self.app.tabs.select(self.app.history_tab)
            self.panel.choose(run_id)
            self.panel.poll()

        def failed(error):
            self.busy = False
            if not self.closed and self.panel.enabled():
                self.error_text.set(error_message(error))
                self.start.state(['!disabled'])
            elif self.panel.enabled():
                self.panel.error(error)
        if not self.app.controller.call('history-start', 'submit_history', done, failed, values):
            self.busy = False
            self.start.state(['!disabled'])
            self.error_text.set('Запрос не запущен: предыдущий запрос ещё выполняется или приложение закрывается.')

    def close(self):
        if not self.closed:
            self.closed = True
            self.window.destroy()


class AttemptsWindow:
    def __init__(self, app, period):
        self.app, self.period = app, period
        self.closed = False
        self.window = tk.Toplevel(app.root)
        self.window.title('Попытки исторического периода')
        size_window(self.window, 720, 350)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text=f'{SOURCES[period.source]} • {period.from_date} — {period.until_date}').pack(anchor='w')
        box, self.tree = app._tree(outer, [('number', 'Попытка', 85), ('title', 'ID задания', 360),
                                         ('current', 'Выборка', 150)], 6)
        box.pack(fill='both', expand=True, pady=8)
        for i, job_id in enumerate(period.attempts, 1):
            self.tree.insert('', 'end', iid=job_id, values=(i, job_id,
                             'Последняя' if period.job and period.job.id == job_id else 'Предыдущая'))
        ttk.Label(outer, text='Каждая попытка хранит собственные неизменяемые версии документов.', wraplength=650).pack(anchor='w')
        self.open_button = ttk.Button(outer, text='Открыть документы попытки', command=self.open, state='disabled')
        self.open_button.pack(anchor='w', pady=8)
        self.tree.bind('<<TreeviewSelect>>', lambda event: self.open_button.state(
            ['!disabled'] if self.tree.selection() else ['disabled']))

    def open(self):
        selected = self.tree.selection()
        if selected and not self.closed and not self.app.closing:
            self.app.show_documents(job_id=selected[0], label=f'Попытка {selected[0]}')
            self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self.window.destroy()


class ParametersWindow:
    def __init__(self, app, text):
        from tkinter.scrolledtext import ScrolledText
        self.closed = False
        self.window = tk.Toplevel(app.root)
        self.window.title('Параметры исторического сбора')
        size_window(self.window, 700, 420)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        content = ScrolledText(self.window, wrap='word', padx=12, pady=12, width=60, height=12)
        content.pack(fill='both', expand=True)
        content.insert('1.0', text)
        content.configure(state='disabled')

    def close(self):
        if not self.closed:
            self.closed = True
            self.window.destroy()
