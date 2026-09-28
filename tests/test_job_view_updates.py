"""Progress-only job updates preserve rows and do not invalidate document data."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.ui.window import Application


class JobTree:
    def __init__(self):
        self.rows = {}
        self.deleted = []
        self.selected = ()

    def get_children(self):
        return tuple(self.rows)

    def selection(self):
        return self.selected

    def selection_set(self, item):
        self.selected = (item,)

    def delete(self, *items):
        self.deleted.extend(items)
        for item in items:
            self.rows.pop(item)

    def exists(self, item):
        return item in self.rows

    def insert(self, _parent, _position, *, iid, values):
        self.rows[iid] = tuple(map(str, values))

    def item(self, item, option=None, **options):
        if options:
            self.rows[item] = tuple(map(str, options["values"]))
        return self.rows[item]

    def move(self, item, _parent, index):
        rows = list(self.rows.items())
        pair = next(row for row in rows if row[0] == item)
        rows.remove(pair)
        rows.insert(index, pair)
        self.rows = dict(rows)


def test_progress_only_does_not_rebuild_rows_or_reread_documents():
    app = object.__new__(Application)
    app.fingerprint = app._document_jobs_fingerprint = ()
    app.cancel_requested, app.documents = set(), {"visible": object()}
    app.jobs = {}
    app.job_tree = JobTree()
    app._select_job, app.refresh_documents = Mock(), Mock()
    job = SimpleNamespace(id="job", updated_at=1, state="queued", stored=1, scanned=1, skipped=0,
                          request=SimpleNamespace(topic="test", source="crossref"))
    app._render_jobs([job])
    app.refresh_documents.reset_mock()
    job.updated_at, job.state = 2, "running"
    app._render_jobs([job])
    assert app.job_tree.deleted == []
    app.refresh_documents.assert_not_called()
    assert "Загрузка" in app.job_tree.rows["job"]
    # A repeated source record can change latest metadata without increasing stored.
    job.updated_at, job.scanned = 3, 2
    app._render_jobs([job])
    app.refresh_documents.assert_called_once_with()


def test_reordered_jobs_keep_server_order_without_losing_selection():
    app = object.__new__(Application)
    app.fingerprint = app._document_jobs_fingerprint = ()
    app.cancel_requested, app.documents = set(), {"visible": object()}
    app.jobs, app.job_tree = {}, JobTree()
    app._select_job, app.refresh_documents = Mock(), Mock()
    jobs = [SimpleNamespace(id=key, updated_at=1, state="queued", stored=1, scanned=1, skipped=0,
                            request=SimpleNamespace(topic=key, source="crossref")) for key in "abcd"]
    app._render_jobs(jobs)
    app.job_tree.selection_set("b")
    app.refresh_documents.reset_mock()
    app._render_jobs([jobs[index] for index in (2, 1, 3, 0)])
    assert app.job_tree.get_children() == ("c", "b", "d", "a")
    assert app.job_tree.selection() == ("b",)
    assert app.job_tree.deleted == []
    app.refresh_documents.assert_not_called()


def test_one_changed_job_updates_only_its_visible_row():
    app = object.__new__(Application)
    app.fingerprint = app._document_jobs_fingerprint = ()
    app.cancel_requested, app.documents = set(), {"visible": object()}
    app.jobs, app.job_tree = {}, JobTree()
    app._select_job, app.refresh_documents = Mock(), Mock()
    jobs = [SimpleNamespace(id=str(index), updated_at=1, state="queued", stored=0, scanned=0, skipped=0,
                            request=SimpleNamespace(topic=str(index), source="crossref")) for index in range(100)]
    app._render_jobs(jobs)
    with patch.object(app.job_tree, "item", wraps=app.job_tree.item) as redraw:
        jobs[0].updated_at = 2
        app._render_jobs(jobs)
        redraw.assert_not_called()
        jobs[-1].state = "running"
        app._render_jobs(jobs)
        assert redraw.call_count == 1
        assert redraw.call_args.args[0] == "99"
