"""Closed version pages release snapshots even while a read is still pending."""

import gc
import weakref
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.ui.versions import VersionsWindow


class Payload:
    pass


def test_closed_page_releases_payloads_and_ignores_late_read():
    app = SimpleNamespace(closing=False, child_windows=[])
    destroyed = []
    page = VersionsWindow.__new__(VersionsWindow)
    page.app = app
    page.closed = False
    page.loading = True
    page.window = SimpleNamespace(destroy=lambda: destroyed.append(True))
    page.snapshot = Payload()
    page.items = {str(number): Payload() for number in range(50)}
    page.selected = page.items['0']
    page._raw_rendered = page.selected
    references = [weakref.ref(item) for item in [page.snapshot, *page.items.values()]]
    app.child_windows.append(page)
    # The future retains the bound callback until its read completes.
    pending_callback = page.loaded
    page.close()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert app.child_windows == []
    assert page.selected is page.snapshot is None
    assert page._raw_rendered is None
    assert not page.loading
    pending_callback(SimpleNamespace(items=[Payload()]))
    page.failed(RuntimeError('late failure'))
    page.close()
    assert page.items == {}
    assert destroyed == [True]
    reference = weakref.ref(page)
    del pending_callback, page
    gc.collect()
    assert reference() is None


def test_metadata_is_formatted_only_for_the_visible_tab_and_current_revision():
    page = VersionsWindow.__new__(VersionsWindow)
    page.app = SimpleNamespace(closing=False)
    page.closed = False
    page.selected = page._raw_rendered = None
    page.snapshot = SimpleNamespace(revision_id='current')
    page.items = {key: SimpleNamespace(revision_id=key, document=SimpleNamespace(raw_metadata={'revision': key}))
                  for key in ('first', 'second')}
    selection, selected_tab = ['first'], ['card']
    page.tree = SimpleNamespace(selection=lambda: selection)
    page.notebook = SimpleNamespace(select=lambda: selected_tab[0])
    page.open_link = SimpleNamespace(state=Mock())
    page.detail, page.raw = 'card', 'raw'
    text = {}
    page.set_text = lambda widget, value: text.__setitem__(widget, value)
    with patch('app.ui.versions.document_detail', return_value='card'), patch(
            'app.ui.versions.json.dumps', wraps=json.dumps) as serialize:
        page.select()
        selection[:] = ['second']
        page.select()
        serialize.assert_not_called()
        selected_tab[:] = ['raw']
        page.show_raw()
        assert json.loads(text['raw']) == {'revision': 'second'}
        page.show_raw()
        page.select()  # Duplicate native selection events must not serialize again.
        assert serialize.call_count == 1
        selection[:] = ['first']
        page.select()
        assert json.loads(text['raw']) == {'revision': 'first'}
        assert serialize.call_count == 2
        selected_tab[:] = ['card']
        selection[:] = ['second']
        page.select()
        assert page._raw_rendered is None
        assert 'first' not in text['raw']
        assert serialize.call_count == 2
        page.closed = True
        selected_tab[:] = ['raw']
        page.show_raw()
        assert serialize.call_count == 2
