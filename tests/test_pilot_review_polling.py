"""A closed review's pending reads/cancellation cannot stall the next dialog."""

from types import SimpleNamespace

from app.ui import pilot_review
from app.ui.pilot_panel import PilotPanel
from tests.test_pilot_operation_ownership import Control, DeferredController, Value


def test_reopened_review_polls_and_cancels_independently_of_closed_dialog(monkeypatch):
    windows = []

    class Widget:
        def __init__(self, *_, **options):
            self.exists = True
            self.protocols = {}
            self.scheduled = []
            self.bindings = {}

        def pack(self, **_):
            pass

        def title(self, _):
            pass

        def geometry(self, _):
            pass

        def protocol(self, name, callback):
            self.protocols[name] = callback

        def winfo_exists(self):
            return self.exists

        def destroy(self):
            self.exists = False
            if '<Destroy>' in self.bindings:
                self.bindings['<Destroy>'](SimpleNamespace(widget=self))

        def bind(self, name, callback, **_):
            self.bindings[name] = callback

        def after(self, delay, callback):
            self.scheduled.append((delay, callback))
            return callback

        def after_cancel(self, callback):
            self.scheduled = [item for item in self.scheduled if item[1] is not callback]

    class Viewport(Widget):
        def __init__(self, *args, **options):
            super().__init__(*args, **options)
            self.content = Widget()

    def window(_):
        result = Widget()
        windows.append(result)
        return result

    monkeypatch.setattr(pilot_review.tk, 'Toplevel', window)
    monkeypatch.setattr(pilot_review.tk, 'StringVar', lambda value='': Value(value))
    monkeypatch.setattr(pilot_review, 'ScrollViewport', Viewport)
    monkeypatch.setattr(pilot_review.ttk, 'Label', Widget)
    monkeypatch.setattr(pilot_review.ttk, 'Button', Widget)
    panel = PilotPanel.__new__(PilotPanel)
    panel.app = SimpleNamespace(root=None, closing=False, controller=DeferredController(), child_windows=[])
    panel.active = panel.loading = False
    panel.loaded = True
    panel.generation = 0
    panel._loading_token = None
    panel.start_button = Control()
    panel.message = Value()
    controller = panel.app.controller

    pilot_review.start_review(panel, 'source-a', 'candidate-a')
    controller.complete('pilot_begin_review', 'review-a')
    old_key = next(key for key, call in controller.pending.items() if call[0] == 'pilot_review_progress')
    windows[0].protocols['WM_DELETE_WINDOW']()
    assert not panel.loading
    pilot_review.start_review(panel, 'source-b', 'candidate-b')
    controller.complete('pilot_begin_review', 'review-b')
    assert controller.arguments('pilot_review_progress') == [('review-a',), ('review-b',)]
    _, old_progress, _, _ = controller.pending.pop(old_key)
    old_progress({'state': 'running', 'message': 'old progress'})
    assert panel.loading
    controller.complete('pilot_review_progress', {'state': 'running', 'message': 'new progress'})
    assert len(windows[1].scheduled) == 1
    windows[1].protocols['WM_DELETE_WINDOW']()
    assert controller.arguments('pilot_cancel') == [('review-a',), ('review-b',)]
    assert not panel.loading
