"""An accepted history continues after dismissal without owning navigation."""

from types import SimpleNamespace
from unittest.mock import Mock

from app.ui.history import HistoryForm


def form_with_controller(accepted=True):
    form = HistoryForm.__new__(HistoryForm)
    form.closed = form.busy = False
    form.window = SimpleNamespace(destroy=Mock())
    form.panel = SimpleNamespace(enabled=lambda: True, choose=Mock(), poll=Mock(), error=Mock())
    pending = []

    def submit(key, method, success, failure, values):
        pending.append(success)
        return accepted

    form.app = SimpleNamespace(controller=SimpleNamespace(call=submit), tabs=SimpleNamespace(select=Mock()),
                               history_tab='history')
    values = {'topic': 'robotics', 'start': '2024-01-01', 'end': '2024-01-31', 'limit': '10', 'budget': '2'}
    form.fields = {key: SimpleNamespace(get=lambda value=value: value) for key, value in values.items()}
    form.sources = {'crossref': SimpleNamespace(get=lambda: True)}
    form.period = SimpleNamespace(get=lambda: 'По месяцам')
    form.auto_split = SimpleNamespace(get=lambda: False)
    form.start = SimpleNamespace(state=Mock())
    form.error_text = SimpleNamespace(set=Mock())
    return form, pending


def test_dismissed_submission_refreshes_background_list_without_navigation():
    form, pending = form_with_controller()
    form.submit()
    assert form.busy
    form.close()
    pending[0]('accepted-history')
    assert not form.busy
    form.panel.choose.assert_not_called()
    form.app.tabs.select.assert_not_called()
    form.panel.poll.assert_called_once_with()
    form.window.destroy.assert_called_once_with()


def test_open_form_navigates_to_its_accepted_history():
    form, pending = form_with_controller()
    form.submit()
    pending[0]('accepted-history')
    assert form.closed and not form.busy
    form.panel.choose.assert_called_once_with('accepted-history')
    form.app.tabs.select.assert_called_once_with('history')


def test_rejected_submission_restores_editable_form():
    form, _ = form_with_controller(accepted=False)
    form.submit()
    assert not form.busy and not form.closed
    form.start.state.assert_called_with(['!disabled'])
    assert 'не запущен' in form.error_text.set.call_args.args[0]
    form.panel.choose.assert_not_called()
