"""The visible document page always owns its filters, offset and navigation."""

from types import SimpleNamespace
from unittest.mock import Mock

from app.ui.window import Application
from tests.test_pilot_operation_ownership import Control, DeferredController, Tree, Value
from tests.ui.test_desktop import document


def application():
    app = object.__new__(Application)
    app.ready, app.closing, app.loading = True, False, False
    app.offset = app.total = app.generation = 0
    app.query, app.document_scope = "", "Все документы"
    app.job_filter = app.history_filter = app.selected_document = None
    app.sort_by, app.descending = "default", False
    app._displayed_document_view = None
    app.documents, app.jobs = {}, {}
    app.document_tree = Tree()
    app.controller = DeferredController()
    app.sort_choice, app.scope, app.page_label = Value("По умолчанию"), Value(), Value()
    app.previous, app.next, app.open_link, app.versions_button = [Control() for _ in range(4)]
    app._operation_error = app._set_detail = Mock()
    app.refresh_documents()
    app.controller.complete("list_documents", SimpleNamespace(items=[document(1)], total=120))
    return app


def test_failed_new_query_restores_the_displayed_scope_and_pager():
    app = application()
    previous_scope = app.scope.get()
    previous_documents = dict(app.documents)
    app.query, app.offset, app.job_filter = "new query", 50, "another-job"
    app.document_scope, app.sort_by = "Документы другого процесса", "title"
    app.sort_choice.set("По названию")
    app.refresh_documents()
    assert app.scope.get() == previous_scope
    app.controller.complete("list_documents", error=OSError("test-only read error"))
    assert app.documents == previous_documents
    assert app.query == "" and app.offset == 0 and app.job_filter is None
    assert app.sort_by == "default" and app.sort_choice.get() == "По умолчанию"
    assert app.scope.get() == previous_scope
    assert "disabled" in app.previous.states
    assert "disabled" not in app.next.states
    app.change_page(1)
    _, _, request = app.controller.calls[-1]
    assert request["offset"] == 50 and request["query"] is None and request["job_id"] is None


def test_rejected_read_does_not_leave_a_loading_page():
    app = application()
    app.controller.call = lambda *_args, **_kwargs: False
    app.refresh_documents()
    assert not app.loading
    assert "disabled" not in app.next.states
