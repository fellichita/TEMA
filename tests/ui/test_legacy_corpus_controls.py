"""Saved-corpus provenance stays distinct from choosing a new collection."""

from pathlib import Path
import tempfile
from unittest.mock import patch

from app.ui.window import Application
from tests.ui.test_desktop import TkCase
from tests.ui.test_trend_result_sections import result_sections


class LegacyCorpusControlsTests(TkCase):
    def open_panel(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading and 'ml_histories' not in self.controller.pending)
        return self.app.trends_panel

    def test_other_source_clears_ready_corpus_and_result_but_preserves_query_and_years(self):
        panel = self.open_panel()
        panel.histories = {'stored': {'id': 'stored-history', 'request': {'topic': 'robotics',
                           'sources': ['crossref'], 'from_date': '2020-01-01', 'until_date': '2025-12-31'}}}
        panel.history_box.set('stored')
        panel.select_history()
        panel.render_result(result_sections())
        parameters = (panel.topic.get(), panel.start_year.get(), panel.end_year.get())
        self.assertTrue(panel.source_box.instate(['disabled']))
        self.assertIn('закреплён', panel.source_hint.get())
        self.assertTrue(panel.run_button.instate(['!disabled']))
        panel.source_reset.invoke()
        self.assertIsNone(panel.history_id)
        self.assertIsNone(panel.snapshot_path)
        self.assertIsNone(panel.result)
        self.assertEqual(panel._corpus_sources, ())
        self.assertEqual(parameters, (panel.topic.get(), panel.start_year.get(), panel.end_year.get()))
        self.assertEqual(str(panel.source_box['state']), 'readonly')
        self.assertTrue(panel.run_button.instate(['disabled']))
        self.assertTrue(panel.export_button.instate(['disabled']))
        self.assertTrue(panel.collect_button.instate(['!disabled']))

    def test_opened_snapshot_locks_its_actual_source_and_missing_demo_is_disabled(self):
        panel = self.open_panel()
        with patch('app.ml.service.inspect_snapshot', return_value={
                'source': 'crossref', 'topic': 'robotics', 'start_year': 2020, 'end_year': 2025, 'occurrences': 2}):
            panel.load_snapshot('synthetic.json')
            self.pump(lambda: not panel.busy)
        self.assertEqual(panel.source.get(), 'crossref')
        self.assertEqual(panel._corpus_sources, ('crossref',))
        self.assertTrue(panel.source_box.instate(['disabled']))
        self.assertIn('crossref', panel.input_label.get())
        with tempfile.TemporaryDirectory() as directory, patch.object(panel, 'load_snapshot') as inspect:
            panel.demo_path = Path(directory) / 'missing-demo.json'
            panel.demo()
            inspect.assert_not_called()
        self.assertFalse(panel.demo_available)
        self.assertTrue(panel.demo_button.instate(['disabled']))
        self.assertIn('Демо-корпус не установлен', panel.source_hint.get())
