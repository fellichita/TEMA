"""Run Tk scenarios in separate processes on macOS (see docs/architecture/desktop-ui.md)."""

import threading
from datetime import date
from pathlib import Path
from types import SimpleNamespace as Record
import tempfile
import unittest

from app.backend.config import BackendSettings
from app.backend.contracts import DocumentRecord, SourcePage
from app.backend.history import HistoryRequest
from app.backend.service import Backend
from app.ui.presentation import history_values
from app.ui.window import Application
from tests.ui.test_desktop import TkCase, document


class HistoryValidationTests(unittest.TestCase):
    def test_limits_dates_and_initial_plan(self):
        values, errors = history_values('robotics', '', '', '1000', ('crossref',), 'month', True, '1200',
                                        today=date(2026, 9, 7))
        self.assertIn('start', errors)
        self.assertEqual(values['until_date'], date(2026, 9, 7))
        _, errors = history_values('robotics', '1900-01-01', '2024-01-01', '1000',
                                   ('crossref', 'openalex'), 'month', True, '1200')
        self.assertIn('start', errors)
        self.assertIn('budget', errors)
        values, errors = history_values('robotics', '2024-01-01', '2024-02-29', '1000',
                                        ('crossref', 'openalex'), 'month', True, '4')
        self.assertFalse(errors)
        self.assertEqual(len(list(HistoryRequest.model_validate(values).intervals())), 2)


class HistoryIntegrationTests(TkCase):
    def open_backend(self, directory, provider):
        settings = BackendSettings(data_dir=Path(directory), history_period_delay_seconds=0)
        self.app = Application(self.root, lambda: Backend(settings, provider_factory=lambda: provider))
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        self.app.tabs.select(self.app.history_tab)
        return settings

    def start_history(self, end='2024-01-01', split=False):
        self.app.history.new()
        form = self.app.history.form
        for key, value in {'topic': 'robotics', 'start': '2024-01-01', 'end': end, 'limit': '1', 'budget': '4'}.items():
            form.fields[key].delete(0, 'end')
            form.fields[key].insert(0, value)
        form.sources['openalex'].set(False)
        form.auto_split.set(split)
        form.period.set('По годам')
        form.submit()
        self.pump(lambda: form.closed and self.app.history.progress is not None)
        return self.app.history

    def finish(self):
        self.app.close()
        self.pump(lambda: self.controller.stopped)

    def test_retry_versions_attempts_and_offline_reopen(self):
        class Provider:
            calls = 0
            def iter_pages(self, request, cancel):
                self.calls += 1
                doc = DocumentRecord(source='crossref', source_id='one', doi='10.1234/one',
                                     title=f'robotics version {self.calls}', url='https://example.org/paper',
                                     raw_metadata={'revision': self.calls})
                yield SourcePage(documents=(doc,), scanned=1, total_available=1 if self.calls > 1 else 2,
                                 exhausted=self.calls > 1)
            def close(self):
                pass
        provider = Provider()
        with tempfile.TemporaryDirectory() as directory:
            settings = self.open_backend(directory, provider)
            panel = self.start_history()
            self.pump(lambda: panel.progress.state == 'partial', timeout=5)
            run_id = panel.selected_id
            self.assertEqual(panel.bar['value'], 100)
            self.assertIn('не подтверждена', panel.summary.get())
            first_job = panel.progress.periods[0].job_id
            panel.retry_incomplete.set(True)
            panel.resume()
            self.pump(lambda: panel.progress.coverage_complete and not panel.busy, timeout=5)
            self.assertEqual(panel.selected_id, run_id)
            self.assertEqual(provider.calls, 2)
            panel.documents()
            self.pump(lambda: not self.app.loading)
            self.assertEqual(self.app.history_filter, run_id)
            self.assertIsNone(self.app.job_filter)
            self.assertEqual(self.app.total, 1)
            key = next(iter(self.app.documents))
            self.app.document_tree.selection_set(key)
            self.app._select_document()
            self.assertIn('version 2', self.app.detail.get('1.0', 'end'))
            self.app.show_versions()
            versions = self.app.child_windows[-1]
            self.pump(lambda: not versions.loading)
            self.assertEqual(versions.total, 2)
            old = next(item for item in versions.items.values() if item.document.title.endswith('1'))
            versions.tree.selection_set(old.revision_id)
            versions.select()
            self.assertIn('version 1', versions.detail.get('1.0', 'end'))
            self.assertNotIn('"revision": 1', versions.raw.get('1.0', 'end'))
            versions.notebook.select(versions.raw)
            self.pump(lambda: '"revision": 1' in versions.raw.get('1.0', 'end'))
            self.assertIn('"revision": 1', versions.raw.get('1.0', 'end'))
            self.assertIn('version 2', self.app.detail.get('1.0', 'end'))
            versions.close()
            period = panel.progress.periods[0]
            panel.period_tree.selection_set(period.id)
            panel.select_period()
            from app.ui.history import AttemptsWindow

            previous_windows = tuple(self.app.child_windows)
            panel.attempts()
            self.pump(lambda: any(isinstance(dialog, AttemptsWindow) and dialog not in previous_windows
                                  for dialog in self.app.child_windows))
            attempts = next(dialog for dialog in self.app.child_windows
                            if isinstance(dialog, AttemptsWindow) and dialog not in previous_windows)
            self.assertEqual(len(attempts.tree.get_children()), 2)
            attempts.tree.selection_set(first_job)
            attempts.open()
            self.pump(lambda: not self.app.loading)
            self.assertEqual(self.app.job_filter, first_job)
            self.assertIsNone(self.app.history_filter)
            self.assertEqual(next(iter(self.app.documents.values())).document.title, 'robotics version 1')
            self.finish()
            def offline():
                raise AssertionError('Offline views must not construct a provider')
            with Backend(settings, provider_factory=offline) as backend:
                self.assertTrue(backend.get_history_progress(run_id).coverage_complete)
                self.assertEqual(backend.list_document_versions(key).total, 2)
                self.assertEqual(backend.list_documents(history_id=run_id).total, 1)

    def test_saved_history_and_cross_source_versions_open_without_network(self):
        class Provider:
            def iter_pages(self, request, cancel):
                yield SourcePage(documents=(DocumentRecord(source=request.source, source_id='same',
                                      doi='10.1234/same', title=f'robotics {request.source}',
                                      url='https://example.org/paper'),), scanned=1, exhausted=True)
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            settings = BackendSettings(data_dir=Path(directory), history_period_delay_seconds=0)
            with Backend(settings, provider_factory=Provider) as backend:
                report = backend.collect_history(HistoryRequest(topic='robotics', from_date='2024-01-01',
                                                  until_date='2024-01-01', sources=('crossref', 'openalex')))
            def offline():
                raise AssertionError('Network provider must not be created')
            self.app = Application(self.root, lambda: Backend(settings, provider_factory=offline))
            self.controller = self.app.controller
            self.pump(lambda: self.app.history.progress is not None)
            panel = self.app.history
            self.assertEqual(panel.selected_id, report.id)
            ml_panel = self.app.trends_panel
            self.pump(lambda: bool(ml_panel.histories))
            label = next(label for label, history in ml_panel.histories.items()
                         if history["id"] == report.id)
            ml_panel.history_box.set(label)
            ml_panel.select_history()
            self.assertEqual(ml_panel.history_id, report.id)
            self.assertEqual(ml_panel._corpus_sources, ('crossref', 'openalex'))
            self.assertIn('crossref, openalex', ml_panel.input_label.get())
            self.assertTrue(ml_panel.run_button.instate(['disabled']))
            self.assertTrue(ml_panel.source_box.instate(['disabled']))
            from unittest.mock import patch

            with patch.object(self.controller, 'call') as submit:
                ml_panel.analyze()
                submit.assert_not_called()
            self.assertIn('одного научного источника', ml_panel.status.get())
            self.assertTrue(panel.progress.coverage_complete)
            panel.parameters()
            self.pump(lambda: bool(self.app.child_windows))
            self.app.child_windows[-1].close()
            panel.documents()
            self.pump(lambda: not self.app.loading)
            self.assertEqual(self.app.total, 1)
            self.app.document_tree.selection_set(next(iter(self.app.documents)))
            self.app._select_document()
            self.app.show_versions()
            versions = self.app.child_windows[-1]
            self.pump(lambda: not versions.loading)
            self.assertEqual({item.document.source for item in versions.items.values()}, {'crossref', 'openalex'})
            self.finish()

    def test_split_periods_and_unique_history_documents(self):
        class Provider:
            def iter_pages(self, request, cancel):
                leaf = request.from_date == request.until_date
                doc = DocumentRecord(source='crossref', source_id=str(request.from_date) + str(leaf),
                                     title='robotics', url='https://example.org/paper')
                yield SourcePage(documents=(doc,), scanned=1, total_available=1 if leaf else 2, exhausted=leaf)
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            self.open_backend(directory, Provider())
            panel = self.start_history(end='2024-01-02', split=True)
            self.pump(lambda: panel.progress.coverage_complete, timeout=5)
            self.assertEqual(panel.progress.split_periods, 2)
            self.assertEqual(panel.progress.total_periods, 2)
            self.assertEqual(len(panel.period_tree.get_children()), 4)
            self.assertTrue(any('↳' in panel.period_tree.item(i, 'values')[0] for i in panel.period_tree.get_children()))
            panel.documents()
            self.pump(lambda: not self.app.loading)
            self.assertEqual(self.app.total, 2)
            self.finish()

    def test_cancel_keeps_partial_documents_and_resume(self):
        class Provider:
            calls = 0
            def iter_pages(self, request, cancel):
                self.calls += 1
                yield SourcePage(documents=(DocumentRecord(source='crossref', source_id='one',
                                      title='robotics', url='https://example.org/paper'),), scanned=1, exhausted=False)
                if self.calls == 1:
                    cancel.wait(5)
                else:
                    yield SourcePage(scanned=0, exhausted=True)
            def close(self):
                pass
        # Limit must exceed the first page so cancellation can occur during collection.
        with tempfile.TemporaryDirectory() as directory:
            self.open_backend(directory, Provider())
            self.app.history.new()
            form = self.app.history.form
            for key, value in {'topic': 'robotics', 'start': '2024-01-01', 'end': '2024-01-01'}.items():
                form.fields[key].insert(0, value)
            form.sources['openalex'].set(False)
            form.submit()
            panel = self.app.history
            self.pump(lambda: panel.progress and any(p.stored == 1 for p in panel.progress.periods), timeout=4)
            panel.cancel()
            self.pump(lambda: panel.progress.state == 'cancelled', timeout=4)
            panel.documents()
            self.pump(lambda: not self.app.loading)
            self.assertEqual(self.app.total, 1)
            self.app.tabs.select(self.app.history_tab)
            panel.resume()
            self.pump(lambda: panel.progress.coverage_complete, timeout=5)
            self.finish()


class VersionsTests(TkCase):
    def test_pagination_error_and_closing_during_read(self):
        backend = self.backend
        versions = [document(i) for i in range(65)]
        gate = threading.Event()
        started = threading.Event()
        state = {'error': False, 'wait': False}
        def list_versions(key, limit, offset):
            if state['wait']:
                started.set()
                gate.wait(3)
            if state['error']:
                raise RuntimeError('private details')
            return Record(items=versions[offset:offset + limit], total=len(versions))
        backend.list_document_versions = list_versions
        self.app = Application(self.root, lambda: backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        self.app.document_tree.selection_set(next(iter(self.app.documents)))
        self.app._select_document()
        self.app.show_versions()
        dialog = self.app.child_windows[-1]
        self.pump(lambda: not dialog.loading)
        self.assertEqual(len(dialog.items), 50)
        dialog.page(1)
        self.pump(lambda: not dialog.loading)
        self.assertEqual(len(dialog.items), 15)
        state['error'] = True
        dialog.load()
        self.pump(lambda: not dialog.loading)
        self.assertNotIn('private', dialog.status.get())
        self.assertEqual(len(dialog.items), 15)
        state['error'] = False
        state['wait'] = True
        dialog.load()
        self.pump(started.is_set)
        dialog.close()
        gate.set()
        self.pump(lambda: ('versions', dialog) not in self.controller.pending)
        self.assertTrue(dialog.closed)


class HistorySelectionTests(TkCase):
    def test_stale_progress_does_not_replace_new_selection(self):
        from app.backend.history_progress import HistoryProgress
        def progress(run_id):
            return HistoryProgress(id=run_id, topic=run_id, state='partial', total_periods=0,
                                   processed_periods=0, execution_percent=0, completed_periods=0,
                                   partial_periods=0, failed_periods=0, pending_periods=0, active_periods=0,
                                   cancelled_periods=0, interrupted_periods=0, split_periods=0,
                                   total_plan_periods=0, coverage_complete=False, partial_calendar_years=(),
                                   error_code=None, periods=())
        gate, started = threading.Event(), threading.Event()
        def get_progress(run_id):
            if run_id == 'first':
                started.set()
                gate.wait(3)
            return progress(run_id)
        self.backend.get_history_progress = get_progress
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        panel = self.app.history
        panel.choose('first')
        self.pump(started.is_set)
        panel.choose('second')
        gate.set()
        self.pump(lambda: panel.progress is not None)
        self.assertEqual(panel.progress.id, 'second')
        self.assertIn('second', panel.message.get())

    def check_history_actions_at_minimum_window_size(self, scale):
        self.app = Application(self.root, lambda: self.backend, ui_scale=scale)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        width, height = self.root.minsize()
        self.root.geometry(f'{width}x{height}')
        self.root.deiconify()
        self.app.tabs.select(self.app.history_tab)
        panel = self.app.history
        panel.tabs.select(panel.progress_tab)
        panel.summary.set('Обработано конечных периодов: 1200/1200 (100% выполнения); полных: 600, неполных: 600. Полнота всего запроса не подтверждена. Неполные календарные годы: 2023, 2024.')
        panel.period_detail.set('Просмотрено / сохранено / пропущено: 2000 / 1990 / 10. Полнота: неполная. Причина: достигнут лимит выдачи за день. Неполный календарный период. ' * 2)
        self.pump(lambda: (self.root.winfo_width(), self.root.winfo_height()) == (width, height))
        self.wait_mapped([panel.progress_tab, panel.period_tree, panel.period_documents, panel.attempts_button])
        bottom = self.app.history_tab.winfo_rooty() + self.app.history_tab.winfo_height()
        self.assertTrue(panel.attempts_button.winfo_ismapped(), str({
            'root': self.root.winfo_geometry(), 'main': self.app.history_tab.winfo_geometry(),
            'detail': panel.progress_tab.winfo_geometry(), 'tree': panel.period_tree.winfo_geometry(),
            'actions': panel.attempts_button.master.winfo_geometry(),
            'main_selected': self.app.tabs.select(), 'selected': panel.tabs.select(),
            'children': [(str(w), w.winfo_geometry(), w.winfo_ismapped()) for w in panel.progress_tab.winfo_children()]}))
        self.assertLessEqual(panel.attempts_button.winfo_rooty() + panel.attempts_button.winfo_height(), bottom,
                             str({'main': self.app.history_tab.winfo_geometry(),
                                  'notebook': panel.tabs.winfo_geometry(),
                                  'detail': panel.progress_tab.winfo_geometry(),
                                  'actions': panel.attempts_button.master.winfo_geometry(),
                                  'overview': panel.progress_overview.winfo_geometry(),
                                  'requests': [(str(widget), widget.winfo_reqheight())
                                               for widget in panel.progress_tab.winfo_children()]}))
        self.assertGreaterEqual(panel.period_tree.winfo_height(), 80, str({
            "root": self.root.winfo_geometry(), "progress": panel.progress_tab.winfo_geometry(),
            "children": [(str(widget), widget.winfo_geometry(), widget.winfo_reqheight())
                         for widget in panel.progress_tab.winfo_children()]}))
        for button in (panel.period_documents, panel.attempts_button):
            self.assertGreaterEqual(button.winfo_width(), button.winfo_reqwidth())
            self.assertGreaterEqual(button.winfo_height(), button.winfo_reqheight())
            self.assertLessEqual(button.winfo_rootx() + button.winfo_width(),
                                 panel.progress_tab.winfo_rootx() + panel.progress_tab.winfo_width())
        overview = panel.progress_overview
        overview.reveal(panel.period_tree)
        self.pump(lambda: overview.timer is None and overview.content.winfo_rooty()
                  == overview.canvas.winfo_rooty() - round(overview.canvas.canvasy(0)))
        visible_top = max(panel.period_tree.winfo_rooty(), overview.canvas.winfo_rooty())
        visible_bottom = min(panel.period_tree.winfo_rooty() + panel.period_tree.winfo_height(),
                             overview.canvas.winfo_rooty() + overview.canvas.winfo_height())
        self.assertGreaterEqual(visible_bottom - visible_top, 80)

    def test_history_actions_remain_visible_at_minimum_window_size(self):
        # 940x740 is the authored 100% minimum, independent of X server DPI.
        self.check_history_actions_at_minimum_window_size(1)
        panel = self.app.history
        panel.new()
        form = panel.form
        form.window.geometry('650x420')
        self.root.update()
        self.assertTrue(form.start.winfo_ismapped())
        self.assertLessEqual(form.start.winfo_rooty() + form.start.winfo_height(),
                             form.window.winfo_rooty() + form.window.winfo_height())

    def test_history_actions_remain_visible_at_150_percent(self):
        self.check_history_actions_at_minimum_window_size(1.5)

    def test_history_actions_remain_visible_at_200_percent(self):
        self.check_history_actions_at_minimum_window_size(2)

    def test_wrapped_history_actions_stay_inside_scaled_tab(self):
        self.check_history_actions_at_minimum_window_size(2)
        panel = self.app.history
        row = panel.attempts_button.master
        width = row.winfo_width()
        # Two individually fitting buttons must take two rows even on a host
        # whose native font normally keeps the original short labels together.
        for button in (panel.period_documents, panel.attempts_button):
            for _ in range(30):
                if button.winfo_reqwidth() >= width * .6:
                    break
                button.configure(text=button.cget('text') + ' документы')
            self.assertLess(button.winfo_reqwidth(), width)
        row._reflow()
        self.pump(lambda: row._wrapped and panel.attempts_button.winfo_y() > panel.period_documents.winfo_y())
        bottom = self.app.history_tab.winfo_rooty() + self.app.history_tab.winfo_height()
        self.assertLessEqual(panel.attempts_button.winfo_rooty() + panel.attempts_button.winfo_height(), bottom,
                             str({'main': self.app.history_tab.winfo_geometry(),
                                  'notebook': panel.tabs.winfo_geometry(),
                                  'detail': panel.progress_tab.winfo_geometry(),
                                  'actions': row.winfo_geometry(),
                                  'overview': panel.progress_overview.winfo_geometry()}))
