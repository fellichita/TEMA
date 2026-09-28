"""Initial and changed job snapshots invalidate document reads only when needed."""

from types import SimpleNamespace
from threading import Event
from unittest.mock import patch

from app.ui.window import Application
from tests.ui.test_desktop import TkCase, document


def completed_job():
    return SimpleNamespace(id="finished", updated_at=1, state="succeeded", stored=1,
        request=SimpleNamespace(topic="saved topic", source="crossref"), scanned=1, skipped=0,
        total_available=1, coverage_complete=True, error_message=None, error_code=None)


class DocumentRefreshTests(TkCase):
    def open_application(self):
        self.app = Application(self.root, lambda: self.backend)
        self.controller = self.app.controller
        self.settle_documents()

    def settle_documents(self, *, total=None):
        self.pump(lambda: self.app.ready and not self.app.loading
                  and "jobs" not in self.controller.pending and "documents" not in self.controller.pending
                  and (total is None or self.app.total == total))

    def test_empty_startup_and_unchanged_job_polls_read_documents_once(self):
        self.open_application()
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.app.total, 65)
        self.app._request_jobs()
        self.settle_documents()
        self.assertEqual(len(self.backend.calls), 1)

    def test_unchanged_document_page_keeps_rows_and_selection(self):
        self.open_application()
        first = self.app.document_tree.get_children()[0]
        self.app.document_tree.selection_set(first)
        self.pump(lambda: self.app.selected_document is not None)
        with patch.object(self.app.document_tree, "delete", wraps=self.app.document_tree.delete) as deleted, \
                patch.object(self.app.document_tree, "insert", wraps=self.app.document_tree.insert) as inserted:
            self.app.refresh_documents()
            self.settle_documents()
            deleted.assert_not_called()
            inserted.assert_not_called()
        self.assertEqual(self.app.document_tree.selection(), (first,))

    def test_initial_completed_job_refreshes_snapshot_taken_before_its_document_arrived(self):
        original_documents = self.backend.list_documents
        snapshots = []
        snapshot_taken = Event()
        added = document(900)

        def read_documents(**options):
            page = original_documents(**options)
            snapshots.append(tuple(item.document_key for item in page.items))
            snapshot_taken.set()
            return page

        def read_jobs(**options):
            # Document reads have their own executor. Arrange the advertised
            # old-page/new-job ordering explicitly across the two workers.
            assert snapshot_taken.wait(2), "Initial document snapshot was not taken"
            if not self.backend.jobs:
                self.backend.records.insert(0, added)
                self.backend.jobs.append(completed_job())
            return self.backend.jobs

        with patch.object(self.backend, "list_documents", side_effect=read_documents), \
                patch.object(self.backend, "list_jobs", side_effect=read_jobs):
            self.open_application()
        self.assertNotIn(added.document_key, snapshots[0])
        self.assertIn(added.document_key, self.app.documents)
        self.assertEqual(self.app.total, 66)
        self.assertEqual(len(self.backend.calls), 2)

    def test_changed_and_then_removed_jobs_refresh_the_current_document_page(self):
        self.open_application()
        initial_reads = len(self.backend.calls)
        # A periodic poll may finish before a document arrives, but still be
        # waiting for delivery on Tk. Force that ordering instead of relying on
        # the native event loop to produce it occasionally.
        self.app._request_jobs()
        old_poll = self.controller.pending["jobs"][0]
        self.assertEqual(old_poll.result(timeout=2), [])
        added = document(901)
        self.backend.records.insert(0, added)
        self.backend.jobs.append(completed_job())
        self.app._request_jobs()
        # An older in-flight poll can be delivered before the next periodic
        # snapshot. Quiescence alone does not mean the changed page is visible.
        self.settle_documents(total=66)
        self.assertIn(added.document_key, self.app.documents)
        self.assertEqual(len(self.backend.calls), initial_reads + 1)
        self.backend.records.remove(added)
        self.backend.jobs.clear()
        self.app._request_jobs()
        self.settle_documents(total=65)
        self.assertNotIn(added.document_key, self.app.documents)
        self.assertEqual(self.app.total, 65)
        self.assertEqual(len(self.backend.calls), initial_reads + 2)

    def test_explicit_invalidation_still_refreshes_an_empty_job_snapshot(self):
        self.open_application()
        initial_reads = len(self.backend.calls)
        self.backend.records.clear()
        # Restore and cancellation explicitly invalidate this job snapshot.
        self.app.fingerprint = None
        self.app._request_jobs()
        self.settle_documents()
        self.assertEqual(self.app.total, 0)
        self.assertEqual(self.app.documents, {})
        self.assertEqual(len(self.backend.calls), initial_reads + 1)
