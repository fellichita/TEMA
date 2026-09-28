"""A restored profile cannot inherit pending UI ownership from the old library."""

from app.ui.pilot_materials import _reset_library_views
from app.ui.window import Application
from tests.ui.test_desktop import TkCase


class PilotRestoreStateTests(TkCase):
    def test_reset_discards_display_snapshot_operation_and_old_cancellation_token(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.pump(lambda: self.app.ready and not self.app.loading)
        trends = self.app.trends_panel
        previous_token = trends.cancel_event
        previous_operation = object()
        trends._operation = previous_operation
        trends._cancel_pending = trends._collection_cancelled = True
        trends.source.set("crossref, openalex")
        trends._corpus_sources = ("crossref", "openalex")
        trends._corpus_error = "A previous source error"
        self.app._displayed_document_view = object()
        _reset_library_views(self.app.pilot_panel)
        self.assertIsNone(self.app._displayed_document_view)
        self.assertIsNone(trends._operation)
        self.assertFalse(trends._cancel_pending)
        self.assertFalse(trends._collection_cancelled)
        self.assertTrue(previous_token.is_set())
        self.assertIsNot(trends.cancel_event, previous_token)
        self.assertFalse(trends.cancel_event.is_set())
        self.assertEqual(trends.source.get(), "openalex")
        self.assertEqual(trends._corpus_sources, ())
        self.assertEqual(trends._corpus_error, "")
        self.assertEqual(self.callback_errors, [])
